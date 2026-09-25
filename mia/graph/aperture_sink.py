"""The capture-aperture drains' raw-file write path: persistent per-file sinks, O_DIRECT where the
alignment rules hold, and a small writer-thread pool.

WHY. In FULL-graph aperture mode the drain consumer thread IS the disk writer (the writer process
idles; docs/tp_support.md §8). The old path, per step and per layer, materialised the rows as a
fresh ``bytes`` object (``.numpy().tobytes()``, GIL held) and then ``open(path, "ab")`` / ``write`` /
``close`` -- two host copies and an open/close per layer per step, on ONE thread, with every
aperture row held until the last layer's append. Measured on the 70B TP4 full step (10.74 GB): about
1.84 GB/s, which saturates HS capture near 3 req/s. That sequence survives unchanged as
``MIA_APERTURE_WRITE_MODE=legacy``, for A/B validation only.

THE NEW PATH (``auto``, the default):

* **Zero-copy.** Each write hands ``os.pwrite`` a ``memoryview`` straight over the per-layer host
  buffer the D2H landed in. bfloat16 goes out as its raw 2-byte payload -- exactly the bytes the
  old ``.view(torch.uint16).numpy().tobytes()`` produced, which is what the reader reads back.
* **Files stay open.** One fd per raw file for the whole run, opened (``O_TRUNC``) when the drain
  is built and closed at ``close()``. The append offset is tracked here; no per-step open/close.
* **O_DIRECT where it is legal AND where it pays.** A file is opened ``O_DIRECT`` only when its
  row size is a multiple of the directory's direct-I/O block size, so every write length -- and
  therefore every file offset, a running sum of them -- is block-aligned by construction; the
  drain's host buffers are allocated at ``mem_align``. The block size is DETECTED
  (``statx(STATX_DIOALIGN)``, else the block device's ``logical_block_size`` from sysfs, then a
  probe write that must succeed), never assumed. A file whose rows are not a multiple (QK k rows
  at TP4 70B are 512 B, at TP8 256 B) is written through the zero-copy buffered path instead.
  ``auto`` then weighs the PREDICTED write size (:func:`predict_rows_per_write` x the row width)
  against :data:`DIRECT_MIN_BYTES`, the low end of the measured crossover band: below it an
  O_DIRECT write's device round-trip stops paying for itself (it does not shrink with the
  payload), and at the bottom of the ladder it measurably costs, so the file is written buffered.
  An explicit ``direct`` ignores the size rule -- it is a requirement, not a suggestion.
  The decision is made once per file, at open: a file is never written both ways.
* **The per-request disk staging shares this writer** (:class:`PerRequestSinks`), in buffered
  mode: same zero-copy ``pwrite``, same run-long fds, no ``bytes`` copy and no per-step
  open/close, for the same bytes.
* **Writer threads.** A pool of ``MIA_APERTURE_WRITE_THREADS`` (default 2) threads runs the writes
  while the drain thread keeps issuing per-layer D2H copies, so the copies overlap the writes.
  The drain joins every write of a step BEFORE ``advance_drain`` frees that step's rows.

LOUD FAILURES ONLY. A write that fails raises out of the task; the drain waits for the step's other
writes, then re-raises in its consumer thread, which dies without advancing the aperture cursor, so
the engine's backpressure check turns it into ``ApertureBackpressureError``. ``direct`` refuses at
construction when O_DIRECT cannot be honoured. Nothing here falls back after a file is open.
"""
from __future__ import annotations

import ctypes
import errno
import logging
import mmap
import os
import queue
import struct
import threading
import time
from concurrent.futures import Future, wait as _wait_futures
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from mia._profiler import PROF, is_enabled as is_prof_enabled
from .thread_device import bind_thread_to_device

logger = logging.getLogger(__name__)

WRITE_MODE_ENV = "MIA_APERTURE_WRITE_MODE"
WRITE_THREADS_ENV = "MIA_APERTURE_WRITE_THREADS"
DIRECT_MIN_BYTES_ENV = "MIA_APERTURE_DIRECT_MIN_BYTES"
WRITE_MODES = ("auto", "direct", "buffered", "legacy")
DEFAULT_WRITE_THREADS = 2
_MAX_WRITE_THREADS = 64

#: THE O_DIRECT CROSSOVER, in bytes per write (one raw file, one drained step). Below it a file
#: is opened buffered, at or above it O_DIRECT; `auto` weighs a file's PREDICTED write size
#: (``WriteShape.rows`` x that file's row width) against this. It governs ONE thing: the per-file
#: choice in the OFF-LOOP SHARED-FILE drains (``_ALLOW_DIRECT = True``), which always run a
#: writer pool of :func:`resolve_write_threads` threads -- default 2, and 2 on every engine run
#: recorded so far (the 1- and 4-thread install lines in the study's output are the drain bench's
#: own sweep). So the column to read it off is the per-STEP one at 2 threads, not the per-write
#: one at 1 -- but a run may set 1, so the threshold must not contradict that column either.
#:
#: MEASURED, job 1802981, H100 node p2-r14-n3, MIA pin 279ef1a, sink on the node-local NVMe (XFS
#: on an 8-way PM1733a LVM stripe, 512 B logical block): `tests/perf/drain_bench.py` driven over
#: two HS geometries x eight step sizes x {legacy, buffered, direct, auto} x {1, 2} writer
#: threads = 128 points, every one `--verify` ok (the bytes are identical in all four modes at
#: every size). Raw data `bench/small_sweep.1802981/small_sweep.json`.
#:
#: **Per drained STEP** -- what the drain thread actually spends, D2H included -- as O_DIRECT's
#: cost relative to the same step buffered (positive = O_DIRECT SLOWER):
#:
#:      per-layer write     8 KiB   16 KiB   32 KiB   64 KiB  128 KiB  256 KiB  512 KiB    1 MiB
#:      1 writer thread    +36.6%   +26.8%   +14.3%    +8.5%    -7.2%    -6.8%   -46.4%   -67.5%
#:      2 writer threads    +2.8%    -5.0%    -7.6%    +3.0%    -9.9%    -3.8%   -35.7%   -58.7%
#:
#: The ladder alternates geometries (8/32/128/512 KiB are the 8B shape, h4096 = 8 KiB rows;
#: 16/64/256 KiB and 1 MiB the 70B shard, h8192 = 16 KiB rows), so read it column by column.
#:
#: NOISE FLOOR, from the same job's 32 `auto`-vs-`direct` pairs -- identical path, identical
#: bytes, so their spread is pure noise: median 2.8 %, p90 11.9 %, max 36 %. Against it exactly
#: three points are DECISIVE for buffered -- 8, 16 and 32 KiB, and only at ONE writer thread --
#: and everything from 512 KiB up is decisive for direct at both thread counts. 64-256 KiB the
#: measurement cannot call either way, and at the shipped 2 threads nothing below 512 KiB is
#: decisive at all: buffered's best showing there is +3 %.
#:
#: **Hence 64 KiB: the low end of the undecidable band -- the smallest threshold no measured
#: column contradicts.** It keeps buffered exactly where buffered decisively wins, and inside the
#: band it errs towards direct because the error is asymmetric: wrong towards direct costs at
#: most +0.43 ms per step with a single writer and +0.04 ms with the shipped 2-thread pool (the
#: worst rows above, on steps of 1.18 and 1.43 ms), wrong towards buffered costs ~4x and grows
#: without limit (264 vs 67 ms per step at 64 MiB). Deliberately NOT thread-aware: one number
#: above every decisive 1-thread loss and below every decisive 2-thread win serves both, and the
#: pool size is not known where this is read. In rows: 8 at 8B, 4 at 70B. (Until 2026-09-20 this
#: read 128 KiB, taken from
#: the per-write column below; at 2 threads that column does not describe this decision, and the
#: rule it produced could only cost -- up to ~10 % of a step at 128 KiB.)
#:
#: **Per WRITE**, for completeness and because it IS the right column elsewhere -- direct as a
#: multiple of buffered: at 1 thread 1.74x (8 KiB), 1.65x, 1.33x, 1.17x, 0.91x, 0.95x, 0.49x
#: (512 KiB); at 2 threads 1.02x, 1.10x, 0.90x, 1.17x, 0.97x, 0.99x, 0.59x. The 1-thread row is
#: what the per-request staging faces -- it writes inline on the drain thread with no pool
#: (:class:`PerRequestSinks`) -- and a bare `write()` control in the same job (single-threaded,
#: no drain) puts parity at 256 KiB. Neither governs this constant.
#:
#: The mechanism under both: an O_DIRECT write carries an ~18-20 us device round-trip that does
#: not shrink with the payload, while a buffered write is a memcpy into the page cache at
#: ~3.2 GB/s with a ~2 us floor -- they meet where 3.2 GB/s spends 20 us, and a second writer
#: thread hides the round-trip but not the memcpy. Write-up: `results/model-scale/CAPTURE_IO.md`
#: section 6.1 in the profiling harness. Override: MIA_APERTURE_DIRECT_MIN_BYTES.
DIRECT_MIN_BYTES = 64 * 1024

# One pwrite never asks for more than this. Linux caps a single write at 0x7ffff000 bytes anyway;
# a power-of-two cap keeps every chunk boundary block-aligned for O_DIRECT.
_MAX_IO_BYTES = 1 << 30
# Host buffers for O_DIRECT are aligned to at least a page (cudaHostAlloc memory already is).
_PAGE = mmap.PAGESIZE
# Largest direct-I/O block size the probe will try.
_PROBE_MAX_BLOCK = 1 << 16


def _human_bytes(n: int) -> str:
    """``131072`` -> ``"128.0 KiB"``. Log-line sugar only; nothing parses it back."""
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if v < 1024 or unit == "GiB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} GiB"


class ApertureWriteConfigError(ValueError):
    """The requested aperture write path cannot be honoured (raised at drain construction)."""


class ApertureWriteError(RuntimeError):
    """A raw-file write failed or would have violated O_DIRECT's alignment rules."""


# --------------------------------------------------------------------------------------------
# Env
# --------------------------------------------------------------------------------------------

def resolve_write_mode() -> Tuple[str, bool]:
    """``(mode, explicit)`` from ``MIA_APERTURE_WRITE_MODE``. Unset or empty means ``auto``.

    Refuses anything but the four spellings (case-insensitive): a misspelt mode that silently ran
    another path would make an A/B measure the wrong thing."""
    raw = os.environ.get(WRITE_MODE_ENV)
    if raw is None or not raw.strip():
        return "auto", False
    mode = raw.strip().lower()
    if mode not in WRITE_MODES:
        raise ApertureWriteConfigError(
            f"{WRITE_MODE_ENV}={raw!r} is not one of {'|'.join(WRITE_MODES)} "
            f"(auto = O_DIRECT where aligned else buffered; legacy = the old tobytes + "
            f"open/append/close path, for A/B validation only)")
    return mode, True


def resolve_write_threads() -> int:
    """``MIA_APERTURE_WRITE_THREADS`` (default 2): writer threads per off-loop drain. Must be an
    integer in [1, 64]; anything else is refused rather than clamped."""
    raw = os.environ.get(WRITE_THREADS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_WRITE_THREADS
    try:
        n = int(raw.strip())
    except ValueError:
        n = None
    if n is None or not 1 <= n <= _MAX_WRITE_THREADS:
        raise ApertureWriteConfigError(
            f"{WRITE_THREADS_ENV}={raw!r} must be an integer in [1, {_MAX_WRITE_THREADS}]")
    return n


def resolve_per_request_write_mode(mode: str, explicit: bool) -> str:
    """The write path a per-request DISK staging takes, from the drain's resolved
    ``MIA_APERTURE_WRITE_MODE``: ``"legacy"`` for ``legacy``, else ``"buffered"``.

    Per-request delivery writes no shared raw file, so the per-file O_DIRECT decision has nothing
    to decide; what is left is whether the staging writes zero-copy through
    :class:`PerRequestSinks` (``auto``/``buffered``) or through the pre-2026-09-20 ``_raw_bytes`` +
    open/append/close sequence (``legacy``). Two configurations are REFUSED rather than quietly
    ignored, mirroring the shared-file drain:

    * an explicit ``direct`` -- there is no O_DIRECT path here, and the measurement says there
      should not be: at this staging's 8-16 KiB per write O_DIRECT is 1.65-1.74x slower
      (job 1802981, per write with a single writer -- which is exactly this path, since the
      staging writes inline on the drain thread with no pool), quite apart from a mid-buffer row
      slice having no alignment guarantee;
    * ``MIA_APERTURE_MMAP=1`` under any mode but ``legacy`` -- the mmap sink existed to remove the
      per-step ``open()``, which keeping the fd open already does."""
    if mode == "direct" and explicit:
        raise ApertureWriteConfigError(
            f"{WRITE_MODE_ENV}=direct refused: per-request delivery (MIA_APERTURE_PER_REQUEST=1) "
            f"writes no shared raw files, and its per-request disk staging has no O_DIRECT path "
            f"(its writes are one layer's rows for one request, 8-16 KiB, written inline on the "
            f"drain thread, where O_DIRECT measures 1.65-1.74x slower per write). Use auto (the "
            f"default) or buffered.")
    if mode == "legacy":
        return "legacy"
    if os.environ.get("MIA_APERTURE_MMAP", "0") != "0":
        raise ApertureWriteConfigError(
            f"MIA_APERTURE_MMAP={os.environ.get('MIA_APERTURE_MMAP')!r} selects the legacy mmap "
            f"sink, but {WRITE_MODE_ENV}={mode} ({'set' if explicit else 'the default'}). The "
            f"per-request disk staging now keeps one fd per layer open for the request, which is "
            f"what the mmap sink was for. Unset MIA_APERTURE_MMAP, or set {WRITE_MODE_ENV}=legacy "
            f"to use it.")
    return "buffered"


def resolve_direct_min_bytes() -> int:
    """``MIA_APERTURE_DIRECT_MIN_BYTES`` (default :data:`DIRECT_MIN_BYTES`, 64 KiB): the smallest
    PREDICTED write ``auto`` will open a file ``O_DIRECT`` for. Must be a non-negative integer;
    anything else is refused rather than clamped (a misread size would silently pick the slower
    path, which is exactly what this constant exists to stop). ``0`` restores the pre-2026-09-20
    alignment-only rule -- direct wherever the rows align, whatever the size."""
    raw = os.environ.get(DIRECT_MIN_BYTES_ENV)
    if raw is None or not raw.strip():
        return DIRECT_MIN_BYTES
    try:
        n = int(raw.strip())
    except ValueError:
        n = -1
    if n < 0:
        raise ApertureWriteConfigError(
            f"{DIRECT_MIN_BYTES_ENV}={raw!r} must be a non-negative integer (bytes per write; "
            f"default {DIRECT_MIN_BYTES}, the measured O_DIRECT crossover -- 0 disables the size "
            f"rule and decides on alignment alone)")
    return n


# --------------------------------------------------------------------------------------------
# What one step is predicted to write -- the size half of `auto`
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WriteShape:
    """How many rows one drained step is predicted to append to ONE raw file, and why.

    ``rows`` x a file's row width is the write size ``auto`` weighs against
    :data:`DIRECT_MIN_BYTES`. ``basis`` is the configuration that bounds it, printed in the
    install line so the choice can be argued with from the log alone."""
    rows: int
    basis: str

    def bytes_for(self, row_bytes: int) -> int:
        return int(self.rows) * int(row_bytes)


def predict_rows_per_write(kind: str, *, capture_mode: str,
                           max_batched_tokens: Optional[int], max_num_seqs: Optional[int],
                           aperture_rows: Optional[int]) -> Optional[WriteShape]:
    """An UPPER BOUND on the rows one drained step appends per raw file, from the CAPTURE
    CONFIGURATION alone (no traffic), or None when every bound it could use is missing -- in which
    case ``auto`` decides on alignment alone, as it did before 2026-09-20.

    The bound, per capture kind:

    * **QK**, either ``hookq_mode`` -- every token of the step has its k row appended (``k_full``;
      ``last_token`` only stops *q* being emitted, and q rides the same row span). One step's rows
      are therefore the step's token budget: ``min(max_num_batched_tokens, aperture rows)``.
    * **HS ``all_tokens``** (and any mode this function is not sure of) -- every token of the step
      is captured: the same token budget.
    * **HS ``last_token``** -- at most one row per IN-FLIGHT REQUEST per step, whatever the phase
      (a prefill-only capture writes the one row of each request whose prefill ends in that step;
      a decode-hooked one writes a row per active decoder). Bound: ``min(max_num_seqs, aperture
      rows)``. The measured `8b-hs-decode` shape is exactly this at 128 decoders -- 1.05 MB per
      layer write, 8x above the crossover.

    WHY ``last_token`` IS NOT BOUNDED AT ONE ROW, although a prefill-only last-token capture
    really does write one row per layer once. The mode this function is handed is the WORKER-WIDE
    FALLBACK (``worker.hs_mode`` / ``hookq_mode``); every request may override it in its
    ``extra_args``, and in serve most do. Predicting 1 row off a fallback would be a fabricated
    certainty, and it errs in the expensive direction: a file opened buffered that then takes
    64 MiB writes runs ~4x slower (264 vs 67 ms per step), while a file opened direct that takes
    8 KiB writes costs a bounded +0.12 ms (70B) to +0.29 ms (8B) per step -- on a step that costs
    ~1 ms in any mode and whose drain has >10x headroom (job 1802981, CAPTURE_IO.md 6.1/6.2). So
    every bound here is an upper bound and the bias is towards ``direct``.

    WHERE THE ``max_num_seqs`` BOUND BITES is arithmetic off :data:`DIRECT_MIN_BYTES`, not a
    second measurement: below 8 concurrent requests at 8B (8 KiB rows) and 4 at 70B (16 KiB rows)
    the predicted write is under the crossover and the file opens buffered. That window is
    narrower than the one where buffered DECISIVELY wins (writes of 8-32 KiB, and only with a
    single writer thread), because the threshold sits at the low end of the band the measurement
    cannot call either way -- see the constant for both columns. A run that knows it writes small
    and is not covered by the bound sets ``MIA_APERTURE_WRITE_MODE=buffered`` -- byte-identical,
    and measured 14-37 % of the drained step at 8-32 KiB with a single writer (inside the noise
    floor with the default 2-thread pool). ``ApertureWritePath.note_step_rows`` catches a
    prediction that was wrong the other way, once, in the log."""
    rows_cap = int(aperture_rows) if aperture_rows else 0
    mode = (capture_mode or "").strip().lower()

    def _bounded(n: Optional[int]) -> Optional[int]:
        if not n or int(n) <= 0:
            return None
        return min(int(n), rows_cap) if rows_cap > 0 else int(n)

    if kind == "hs" and mode == "last_token":
        n = _bounded(max_num_seqs)
        if n is not None:
            return WriteShape(n, f"hs last_token (the worker-wide default; a request may ask for "
                                 f"all_tokens): at most one row per in-flight request, bounded by "
                                 f"max_num_seqs={max_num_seqs}"
                                 + (f" and the {rows_cap}-row aperture" if rows_cap else ""))
    n = _bounded(max_batched_tokens)
    if n is None:
        return WriteShape(rows_cap, f"the {rows_cap}-row aperture is the only bound") \
            if rows_cap > 0 else None
    what = ("qk: every token's k row, both hookq_mode values" if kind == "qk"
            else f"hs {mode or 'all_tokens'}: every token of the step")
    return WriteShape(n, f"{what}, bounded by max_num_batched_tokens={max_batched_tokens}"
                         + (f" and the {rows_cap}-row aperture" if rows_cap else ""))


# --------------------------------------------------------------------------------------------
# Direct-I/O detection
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectIOInfo:
    """What ``probe_direct_io`` found for one directory. ``block_size`` is the offset/length
    alignment O_DIRECT needs there; ``mem_align`` the buffer-address alignment MIA uses (at least a
    page). ``source`` says where the block size came from; ``reason`` why direct I/O is off."""
    supported: bool
    block_size: int = 0
    mem_align: int = 0
    source: str = ""
    reason: str = ""


_STATX_DIOALIGN = 0x2000
_AT_FDCWD = -100


def _statx_dio_alignment(path: str) -> Optional[Tuple[int, int]]:
    """``(mem_align, offset_align)`` from ``statx(STATX_DIOALIGN)``, or None when the kernel or
    libc does not report it (Linux < 6.1, and this node's 5.14). ``offset_align == 0`` means the
    file system says direct I/O is unsupported."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = libc.statx
    except (OSError, AttributeError):
        return None
    buf = ctypes.create_string_buffer(256)
    try:
        rc = fn(ctypes.c_int(_AT_FDCWD), os.fsencode(path), ctypes.c_int(0),
                ctypes.c_uint(_STATX_DIOALIGN), buf)
    except Exception:  # noqa: BLE001 -- a foreign libc signature: "not reported"
        return None
    if rc != 0:
        return None
    mask = struct.unpack_from("I", buf.raw, 0)[0]
    if not mask & _STATX_DIOALIGN:
        return None
    mem_align, offset_align = struct.unpack_from("II", buf.raw, 152)
    return int(mem_align), int(offset_align)


def _sysfs_logical_block_size(path: str) -> Optional[int]:
    """``logical_block_size`` of the block device backing ``path`` (None for a file system with no
    block device -- tmpfs, NFS, GPFS -- or when sysfs is unreadable). Handles a partition by
    reading its parent disk's queue."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    if major == 0:
        return None
    base = f"/sys/dev/block/{major}:{minor}"
    for cand in (f"{base}/queue/logical_block_size", f"{base}/../queue/logical_block_size"):
        try:
            with open(cand) as f:
                v = int(f.read().strip())
            if v > 0:
                return v
        except (OSError, ValueError):
            continue
    return None


def probe_direct_io(directory: str) -> DirectIOInfo:
    """Can ``directory`` take O_DIRECT writes, and at what alignment? Detects, never assumes.

    The candidate block size comes from ``statx(STATX_DIOALIGN)`` when the kernel reports it, else
    from the backing block device's sysfs ``logical_block_size``. It is then CONFIRMED by a real
    probe: open a scratch file ``O_DIRECT`` (a file system that rejects the flag fails here, e.g.
    tmpfs before Linux 6.6) and ``pwrite`` one block from page-aligned memory at offset 0. If the
    candidate is refused with EINVAL, larger powers of two are tried up to 64 KiB; with no
    candidate the search starts at 512. The probe file is always removed."""
    if not hasattr(os, "O_DIRECT"):
        return DirectIOInfo(False, reason="this platform has no O_DIRECT")
    reported: Optional[int] = None
    mem_reported = 0
    source = "probe"
    sx = _statx_dio_alignment(directory)
    if sx is not None:
        mem_reported, off_align = sx
        if off_align == 0:
            return DirectIOInfo(False, source="statx",
                                reason=f"statx(STATX_DIOALIGN) reports no direct I/O on {directory}")
        reported, source = off_align, "statx"
    else:
        lbs = _sysfs_logical_block_size(directory)
        if lbs is not None:
            st = os.stat(directory)
            reported = lbs
            source = f"sysfs {os.major(st.st_dev)}:{os.minor(st.st_dev)} logical_block_size"
    path = os.path.join(directory, f".mia_odirect_probe.{os.getpid()}.{threading.get_ident()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as e:
        try:
            os.unlink(path)
        except OSError:
            pass
        return DirectIOInfo(False, source=source,
                            reason=f"open(O_DIRECT) under {directory} failed: {e}")
    try:
        buf = mmap.mmap(-1, _PROBE_MAX_BLOCK)          # anonymous -> page-aligned
        mv = memoryview(buf)
        try:
            candidates: List[int] = []
            bs = reported if reported else 512
            while bs <= _PROBE_MAX_BLOCK:
                candidates.append(bs)
                bs *= 2
            last_err = ""
            for bs in candidates:
                chunk = mv[:bs]
                err: Optional[OSError] = None
                try:
                    n = os.pwrite(fd, chunk, 0)
                except OSError as e:
                    err, n = e, -1
                finally:
                    chunk.release()     # an exception's traceback must not pin the mapping
                if err is not None:
                    if err.errno == errno.EINVAL:
                        last_err = str(err)
                        continue
                    return DirectIOInfo(False, source=source,
                                        reason=f"O_DIRECT probe write under {directory} failed: {err}")
                if n != bs:
                    return DirectIOInfo(False, source=source,
                                        reason=f"O_DIRECT probe write of {bs} B wrote {n} B")
                mem_align = max(_PAGE, int(mem_reported or 0), bs)
                if bs == reported:
                    src = source
                elif reported:
                    src = f"probe, {source} said {reported} B"
                else:
                    src = "probe"
                return DirectIOInfo(True, block_size=bs, mem_align=mem_align,
                                    source=f"{src}, probe write ok")
            return DirectIOInfo(False, source=source,
                                reason=f"no O_DIRECT block size <= {_PROBE_MAX_BLOCK} B accepted "
                                       f"under {directory} (last error: {last_err})")
        finally:
            mv.release()
            buf.close()
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


# --------------------------------------------------------------------------------------------
# Host buffers and zero-copy views
# --------------------------------------------------------------------------------------------

def alloc_host_rows(n_rows: int, width: int, dtype: torch.dtype, *, pinned: bool,
                    align: int) -> torch.Tensor:
    """An ``(n_rows, width)`` host tensor whose data pointer is a multiple of ``align``.

    Allocates exactly the bytes needed first: cudaHostAlloc (pinned) and large CPU allocations
    come back page-aligned, and torch's pinned allocator rounds a request up to a power of two, so
    padding every buffer by ``align`` would double pinned memory at the 70B step size. Only a
    misaligned result is re-allocated with slack and sliced to an aligned start."""
    elt = torch.empty((), dtype=dtype).element_size()
    nbytes = int(n_rows) * int(width) * elt
    raw = torch.empty(max(nbytes, 1), dtype=torch.uint8, pin_memory=pinned)
    if align > 1 and raw.data_ptr() % align:
        raw = torch.empty(nbytes + align, dtype=torch.uint8, pin_memory=pinned)
        off = (-raw.data_ptr()) % align
        raw = raw[off:off + nbytes]
    else:
        raw = raw[:nbytes]
    return raw.view(dtype).view(int(n_rows), int(width))


def tensor_bytes_view(t: torch.Tensor) -> memoryview:
    """A flat, zero-copy ``memoryview`` of a contiguous CPU tensor's bytes.

    Byte-for-byte the old ``_raw_bytes`` output: row-major, native (little-endian) layout, and a
    bfloat16 tensor's raw 2-byte payload -- the same bytes ``.view(torch.uint16).numpy().tobytes()``
    produced. The view keeps the tensor alive for as long as it exists."""
    if t.device.type != "cpu":
        raise ApertureWriteError(f"raw-file write needs a host tensor, got {t.device}")
    if not t.is_contiguous():
        raise ApertureWriteError("raw-file write needs a contiguous host tensor")
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return memoryview(b"")
    return memoryview(flat.view(torch.uint8).numpy())


# --------------------------------------------------------------------------------------------
# One raw file
# --------------------------------------------------------------------------------------------

class RawFileSink:
    """One raw file, open for the whole run. ``write`` appends at the tracked offset with
    ``os.pwrite`` straight from the tensor's memory (the GIL is released for the syscall).

    ``direct=True`` opens it ``O_DIRECT``; every write is then checked against the alignment rules
    (buffer address % ``mem_align``, length and offset % ``block_size``) and a violation RAISES --
    the drain guarantees them by construction, so a violation is a bug, never a reason to switch
    this file to buffered writes. One writer per file at a time (the drain joins a step's writes
    before the next step), so the offset needs no lock."""

    def __init__(self, path: str, *, direct: bool, block_size: int = 0, mem_align: int = 0):
        self.path = path
        self.direct = bool(direct)
        self.block_size = int(block_size) if direct else 0
        self.mem_align = int(mem_align) if direct else 0
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
        if self.direct:
            flags |= os.O_DIRECT
        # O_TRUNC: a re-run never appends onto stale bytes (the old path truncated up front too).
        # 0o666 & ~umask: the same permissions the old open(path, "wb") gave the file.
        self.fd: Optional[int] = os.open(path, flags, 0o666)
        self.offset = 0

    @property
    def mode(self) -> str:
        return "direct" if self.direct else "buffered"

    def write(self, t: torch.Tensor) -> int:
        """Append ``t``'s bytes; returns the byte count. Raises on any short/failed write."""
        fd = self.fd
        if fd is None:
            raise ApertureWriteError(f"write to {self.path} after its sink was closed")
        mv = tensor_bytes_view(t)
        n = len(mv)
        if n == 0:
            return 0
        off = self.offset
        if self.direct:
            bs, ma = self.block_size, self.mem_align
            ptr = t.data_ptr()
            if ptr % ma or n % bs or off % bs:
                raise ApertureWriteError(
                    f"O_DIRECT alignment violated on {self.path}: buffer 0x{ptr:x} % {ma} = "
                    f"{ptr % ma}, length {n} % {bs} = {n % bs}, offset {off} % {bs} = {off % bs} "
                    f"(the drain aligns these by construction -- this is a bug, not a fallback)")
        done = 0
        while done < n:
            chunk = mv[done:done + min(n - done, _MAX_IO_BYTES)]
            w = os.pwrite(fd, chunk, off + done)
            if w <= 0:
                raise ApertureWriteError(
                    f"pwrite to {self.path} wrote {w} of {n - done} bytes at offset {off + done}")
            done += w
        self.offset = off + n
        return n

    def close(self) -> None:
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


# --------------------------------------------------------------------------------------------
# Writer threads
# --------------------------------------------------------------------------------------------

_POOL_STOP = object()


class WriterPool:
    """``n`` daemon threads running write tasks; each returns a ``concurrent.futures.Future``.

    Deliberately NOT a ``ThreadPoolExecutor``: that one refuses new work once interpreter shutdown
    starts, which is exactly when the atexit backstop still needs the drain to finish its backlog.
    Every thread binds ``device`` first (docs/tp_support.md §6); a thread whose bind fails keeps
    serving its queue and fails every task with that error, so a submitted write always completes
    -- with a result or an exception -- and the drain can never wait forever on a dead writer."""

    def __init__(self, n_threads: int, device, name: str):
        self.n_threads = int(n_threads)
        self._device = device
        self._q: "queue.SimpleQueue" = queue.SimpleQueue()
        self._closed = False
        self._threads = [threading.Thread(target=self._run, name=f"{name}-{i}", daemon=True)
                         for i in range(self.n_threads)]
        for t in self._threads:
            t.start()

    def _run(self) -> None:
        bind_error: Optional[BaseException] = None
        try:
            bind_thread_to_device(self._device)
        except BaseException as e:  # noqa: BLE001 -- recorded; every task below fails with it
            bind_error = e
            logger.exception("aperture writer thread %s could not bind %s; every write it takes "
                             "will fail", threading.current_thread().name, self._device)
        while True:
            item = self._q.get()
            if item is _POOL_STOP:
                return
            fut, fn, args = item
            if not fut.set_running_or_notify_cancel():
                continue
            if bind_error is not None:
                fut.set_exception(ApertureWriteError(
                    f"aperture writer thread {threading.current_thread().name} could not bind "
                    f"its CUDA device {self._device}: {bind_error!r}"))
                continue
            try:
                fut.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001 -- delivered to the drain, never swallowed
                fut.set_exception(e)

    def submit(self, fn, *args) -> Future:
        if self._closed:
            raise ApertureWriteError("aperture writer pool is closed")
        fut: Future = Future()
        self._q.put((fut, fn, args))
        return fut

    def alive(self) -> bool:
        return all(t.is_alive() for t in self._threads)

    def close(self, timeout: float = 60.0) -> None:
        """Stop the threads after the queued tasks. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._q.put(_POOL_STOP)
        for t in self._threads:
            t.join(timeout=timeout)


def join_writes(futures: Iterable[Future]) -> List:
    """Wait for EVERY future, then raise the first failure (naming how many failed) or return the
    results in order. Waiting for all of them first means no write is still reading a host buffer
    when the drain gives up on the step."""
    futs = list(futures)
    if not futs:
        return []
    _wait_futures(futs)
    errors = [f.exception() for f in futs if f.exception() is not None]
    if errors:
        first = errors[0]
        if len(errors) > 1:
            raise ApertureWriteError(
                f"{len(errors)} of {len(futs)} aperture raw-file writes failed this step; "
                f"first: {first!r}") from first
        raise first
    return [f.result() for f in futs]


def join_writes_quietly(futures: Iterable[Future]) -> None:
    """Wait for every future without raising -- for a step that is already failing for another
    reason, so no write is left reading a host buffer while that error propagates."""
    futs = list(futures)
    if futs:
        _wait_futures(futs)


# --------------------------------------------------------------------------------------------
# A drain's whole write path
# --------------------------------------------------------------------------------------------

@dataclass
class _FileSpec:
    key: object
    path: str
    kind: str
    row_bytes: int


@dataclass
class WriteStats:
    """Cumulative per-drain write-path accounting (seconds and bytes). Kept regardless of
    ``MIA_PROFILE`` so the GPU micro-benchmark and a validation run can split a step's time."""
    steps: int = 0
    rows: int = 0
    step_s: float = 0.0
    d2h_s: float = 0.0
    write_s: float = 0.0
    write_tail_s: float = 0.0
    write_busy_s: float = 0.0
    bookkeeping_s: float = 0.0
    bytes_direct: int = 0
    bytes_buffered: int = 0
    bytes_legacy: int = 0
    last: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "steps", "rows", "step_s", "d2h_s", "write_s", "write_tail_s", "write_busy_s",
            "bookkeeping_s", "bytes_direct", "bytes_buffered", "bytes_legacy")}
        d["last"] = dict(self.last)
        return d


class ApertureWritePath:
    """The resolved write path of ONE drain: the per-file mode decision, the open sinks, and (for
    an off-loop drain) the writer pool.

    ``files`` maps a drain-side key (HS: the layer number; QK: ``("q", L)`` / ``("k", L)``) to
    ``(path, kind, row_bytes)``. ``allow_direct`` is False for the synchronous drain, whose host
    rows are pageable ``.to("cpu")`` tensors with no alignment guarantee.

    ``mode``: ``auto`` opens a file O_DIRECT iff the directory takes O_DIRECT, the file's rows are
    a multiple of the block size, AND its predicted write is at least
    :func:`resolve_direct_min_bytes` -- else buffered (the reason is kept for the install line);
    ``direct`` requires alignment of EVERY file and refuses at construction otherwise, ignoring
    size (an explicit mode is a requirement, never a suggestion); ``buffered`` never uses
    O_DIRECT. ``legacy`` is not handled here -- the drains keep that path verbatim.

    ``shape`` is what one drained step is predicted to write per file
    (:func:`predict_rows_per_write`); None means the caller could not predict, and ``auto`` then
    decides on alignment alone -- the rule before 2026-09-20."""

    def __init__(self, run_dir: str, files: Dict[object, Tuple[str, str, int]], mode: str, *,
                 allow_direct: bool, direct_refusal: str = "", label: str = "aperture",
                 shape: Optional[WriteShape] = None):
        if mode not in ("auto", "direct", "buffered"):
            raise ApertureWriteConfigError(f"ApertureWritePath: unsupported mode {mode!r}")
        self.run_dir = run_dir
        self.mode = mode
        self.label = label
        self.shape = shape
        self.direct_min_bytes = resolve_direct_min_bytes()
        self.specs: Dict[object, _FileSpec] = {
            k: _FileSpec(k, p, kind, int(rb)) for k, (p, kind, rb) in files.items()}
        self.dio = DirectIOInfo(False, reason="not probed (buffered mode)")
        if mode == "direct" and not allow_direct:
            raise ApertureWriteConfigError(
                f"{WRITE_MODE_ENV}=direct is not supported here: {direct_refusal or 'no aligned host buffers'}")
        if mode in ("auto", "direct") and allow_direct:
            self.dio = probe_direct_io(run_dir)
        elif mode == "auto":
            self.dio = DirectIOInfo(False, reason=direct_refusal or "direct I/O not offered here")
        if mode == "direct" and not self.dio.supported:
            raise ApertureWriteConfigError(
                f"{WRITE_MODE_ENV}=direct refused for {label}: {self.dio.reason}")
        # Per-file decision, made ONCE, before any byte is written: alignment first (O_DIRECT is
        # illegal without it), then the predicted write size (`auto` only -- an explicit `direct`
        # is a requirement, not a suggestion, and is never overruled by a prediction).
        self.kind_reason: Dict[str, str] = {}
        self.kind_predicted: Dict[str, int] = {}
        self._size_downgraded: Dict[str, int] = {}   # kind -> row bytes, watched by note_step_rows
        self._mispredicted = False
        decisions: Dict[object, bool] = {}
        for k, s in self.specs.items():
            direct = False
            if self.dio.supported and mode in ("auto", "direct"):
                predicted = self.shape.bytes_for(s.row_bytes) if self.shape is not None else None
                if predicted is not None:
                    self.kind_predicted.setdefault(s.kind, predicted)
                if s.row_bytes <= 0 or s.row_bytes % self.dio.block_size:
                    why = f"not a multiple of the {self.dio.block_size} B O_DIRECT block"
                    if mode == "direct":
                        raise ApertureWriteConfigError(
                            f"{WRITE_MODE_ENV}=direct refused for {label}: {s.kind} row "
                            f"{s.row_bytes} B is {why} ({s.path}); use auto to write such files "
                            f"buffered")
                    self.kind_reason.setdefault(s.kind, why)
                elif (mode == "auto" and predicted is not None
                        and predicted < self.direct_min_bytes):
                    self.kind_reason.setdefault(
                        s.kind,
                        f"below the {_human_bytes(self.direct_min_bytes)} O_DIRECT crossover, "
                        f"where a drained step measures 14-37 % slower direct with a single "
                        f"writer and inside the noise floor with the pool")
                    # Watched by note_step_rows: this one was decided on a PREDICTION, which real
                    # traffic can contradict (a request may override the worker-wide capture mode).
                    self._size_downgraded[s.kind] = s.row_bytes
                else:
                    direct = True
            decisions[k] = direct
        self.mem_align = self.dio.mem_align if any(decisions.values()) else 1
        self.sinks: Dict[object, RawFileSink] = {}
        try:
            for k, s in self.specs.items():
                direct = decisions[k]
                try:
                    self.sinks[k] = RawFileSink(s.path, direct=direct,
                                                block_size=self.dio.block_size,
                                                mem_align=self.dio.mem_align)
                except OSError as e:
                    if not direct or mode == "direct":
                        raise
                    # auto, before any write: this file takes the buffered path, stated below.
                    self.kind_reason.setdefault(s.kind, f"open(O_DIRECT) failed: {e}")
                    self.sinks[k] = RawFileSink(s.path, direct=False)
        except BaseException:
            for sk in self.sinks.values():
                try:
                    sk.close()
                except OSError:
                    pass
            self.sinks = {}
            raise
        self.pool: Optional[WriterPool] = None
        self.threads = 0
        self.closed = False
        self.stats = WriteStats()

    # ---- setup ----
    def start_pool(self, n_threads: int, device, name: str) -> None:
        if self.pool is None:
            self.pool = WriterPool(n_threads, device, name)
            self.threads = int(n_threads)

    # ---- per write ----
    def sink(self, key) -> RawFileSink:
        return self.sinks[key]

    def kind_modes(self) -> Dict[str, str]:
        """``{kind: "direct" | "buffered" | "direct+buffered"}`` over the open files."""
        seen: Dict[str, set] = {}
        for k, s in self.sinks.items():
            seen.setdefault(self.specs[k].kind, set()).add(s.mode)
        return {kind: "+".join(sorted(m)) for kind, m in seen.items()}

    def summary(self) -> str:
        """The one install line's body: mode per tensor kind with the predicted write size and the
        reason it went that way, then threads, block size and the configuration the prediction
        came from. Everything a reader needs to argue with the decision from the log alone."""
        parts = []
        rows = {}
        for s in self.specs.values():
            rows.setdefault(s.kind, s.row_bytes)
        for kind, m in self.kind_modes().items():
            bits = [f"row {rows.get(kind)} B"]
            pred = self.kind_predicted.get(kind)
            if pred is not None:
                bits.append(f"{_human_bytes(pred)}/write predicted")
            why = self.kind_reason.get(kind)
            if why and m != "direct":
                bits.append(why)
            elif pred is not None and m == "direct" and self.direct_min_bytes:
                bits.append(f">= the {_human_bytes(self.direct_min_bytes)} O_DIRECT crossover")
            parts.append(f"{kind}={m} (" + ", ".join(bits) + ")")
        threads = (f"{self.threads} writer thread(s)" if self.pool is not None
                   else "writes inline on the drain thread")
        if self.dio.supported:
            dio = (f"O_DIRECT block {self.dio.block_size} B, buffer align {self.dio.mem_align} B "
                   f"({self.dio.source})")
        else:
            dio = f"O_DIRECT off: {self.dio.reason}"
        basis = (f" | predicted per step: {self.shape.rows} row(s) -- {self.shape.basis}"
                 if self.shape is not None else
                 " | write size not predicted here: mode decided on alignment alone")
        return (f"write mode={self.mode} -> {', '.join(parts)} | {threads} | {dio} | "
                f"{len(self.sinks)} raw files kept open, zero-copy writes{basis}")

    def note_step_rows(self, rows: int) -> None:
        """A drained step spanned ``rows`` rows -- an upper bound on what any ONE of its files
        wrote (a selective drain compacts a file that no request asked for below it). WARNS ONCE,
        and only for a file ``auto`` opened buffered BECAUSE its predicted write was under the
        crossover, when this step's span reaches it.

        The prediction is made at install from the worker-wide capture mode, which every request
        may override (see :func:`predict_rows_per_write`), so it can be wrong -- and wrong this way
        is the expensive direction: buffered at 64 MiB per write is ~4x slower than O_DIRECT. The
        mode cannot change now (a file is never written both ways, by design), so the only useful
        thing to do is SAY SO rather than be quietly slow. Costs one comparison per step, and
        nothing at all once it has fired or when no file was downgraded by size."""
        if self._mispredicted or not self._size_downgraded:
            return
        for kind, row_bytes in self._size_downgraded.items():
            real = int(rows) * int(row_bytes)
            if real < self.direct_min_bytes:
                continue
            self._mispredicted = True
            msg = (f"{self.label}: {kind} raw files were opened BUFFERED because this capture's "
                   f"predicted write was {_human_bytes(self.kind_predicted.get(kind, 0))} "
                   f"(< the {_human_bytes(self.direct_min_bytes)} O_DIRECT crossover), but a step "
                   f"just spanned {rows} rows = up to {_human_bytes(real)} per file. The "
                   f"prediction came from the worker-wide capture mode and this traffic disagrees "
                   f"with it. The mode is fixed for the run (a file is never written both ways); "
                   f"re-run with {WRITE_MODE_ENV}=direct for up to 4x on this drain. Once only.")
            logger.warning(msg)
            print(f"[aperture-write] {msg}", flush=True)
            return

    def close(self) -> None:
        """Join the writer threads, then close every fd. Idempotent."""
        if self.closed:
            return
        self.closed = True
        if self.pool is not None:
            self.pool.close()
        errs = []
        for s in self.sinks.values():
            try:
                s.close()
            except OSError as e:
                errs.append(e)
        if errs:
            raise ApertureWriteError(f"closing {len(errs)} aperture raw file(s) failed: {errs[0]!r}")


class PerRequestSinks:
    """The per-request disk staging's raw files: opened on first use, kept open for the request,
    written zero-copy, always BUFFERED.

    The staging appends ONE request's rows for one layer per step, which is 8 KiB (8B, h4096) or
    16 KiB (70B, h8192) per write by construction -- permanently at the bottom of the write ladder
    (see :data:`DIRECT_MIN_BYTES`), and it writes them INLINE on the drain thread, with no writer
    pool. That is the one place the single-writer per-write column of job 1802981 is the column
    that applies, and there O_DIRECT measures 1.65-1.74x WORSE; the rows are also a mid-buffer
    slice with no alignment guarantee. So there is no mode to choose here: this is the buffered
    half of the same :class:`RawFileSink` the shared-file drain uses.

    Measured gain over the pre-2026-09-20 staging (``_raw_bytes`` copy + ``open(p,"ab")`` / write /
    close per layer per step): **1.39-1.59x** on that write -- legacy 0.0242 ms (8 KiB) / 0.0313 ms
    (16 KiB) -> buffered 0.0174 / 0.0197 ms, job 1802981.

    One writer at a time (the drain's consumer thread owns a request's staging), so no lock; the
    bytes are consumed synchronously inside :meth:`append`, before the caller's aperture rows are
    freed, exactly as the copy it replaces was."""

    def __init__(self, label: str = "per-request staging"):
        self.label = label
        self.sinks: Dict[object, RawFileSink] = {}

    def append(self, key, path: str, t: torch.Tensor) -> int:
        """Append ``t``'s bytes to ``key``'s file, opening it (``O_TRUNC``, so a re-run never
        appends onto stale bytes -- what the old ``open(p,"wb").close()`` did) on first use."""
        sink = self.sinks.get(key)
        if sink is None:
            sink = RawFileSink(path, direct=False)
            self.sinks[key] = sink
        return sink.write(t if t.is_contiguous() else t.contiguous())

    def close(self) -> None:
        """Close every fd. Idempotent, and best-effort per file: a staging dir can vanish under an
        aborted request, and a close error must never wedge the consumer thread."""
        sinks, self.sinks = self.sinks, {}
        for s in sinks.values():
            try:
                s.close()
            except OSError:
                logger.exception("%s: closing %s failed; continuing", self.label, s.path)


def timed_write(sink: RawFileSink, t: torch.Tensor) -> Tuple[int, float, str]:
    """A writer-pool task: ``(bytes, seconds, mode)`` for one file's append."""
    t0 = time.perf_counter()
    n = sink.write(t)
    return n, time.perf_counter() - t0, sink.mode


def record_step_stats(stats: WriteStats, kind: str, *, rows: int, step_s: float, d2h_s: float,
                      write_s: float, write_tail_s: float, busy_s: float, bookkeeping_s: float,
                      bytes_by_mode: Dict[str, int],
                      wp: "Optional[ApertureWritePath]" = None) -> None:
    """Account one drained step, in ``stats`` always and in PROF when ``MIA_PROFILE=1``.

    PROF names (timers in ms, one sample per step): ``aperture.<kind>.step`` (the whole drained
    step), ``.d2h`` (D2H issue until the last layer's rows landed on the host), ``.write`` (first
    write submitted until every write finished; overlaps ``.d2h`` in the pooled path),
    ``.write_tail`` (waiting on writes after D2H and bookkeeping were done: the part the overlap did
    not hide), ``.bookkeeping`` (copy plans + sidecar records). Counters:
    ``aperture.<kind>.bytes.{direct,buffered,legacy}`` and ``aperture.<kind>.write_busy_us`` (sum of
    per-file write durations across writer threads).

    ``wp`` (the drain's write path, when it has one) also gets this step's row count, so a
    size-predicted buffered decision that real traffic contradicts is reported once."""
    if wp is not None:
        wp.note_step_rows(rows)
    stats.last = {"rows": int(rows), "step_s": step_s, "d2h_s": d2h_s, "write_s": write_s,
                  "write_tail_s": write_tail_s, "write_busy_s": busy_s,
                  "bookkeeping_s": bookkeeping_s,
                  "bytes": int(sum(bytes_by_mode.values()))}
    stats.rows += int(rows)
    stats.step_s += step_s
    stats.d2h_s += d2h_s
    stats.write_s += write_s
    stats.write_tail_s += write_tail_s
    stats.write_busy_s += busy_s
    stats.bookkeeping_s += bookkeeping_s
    for m, n in bytes_by_mode.items():
        setattr(stats, f"bytes_{m}", getattr(stats, f"bytes_{m}") + int(n))
    # Last: a reader that sees `steps` reach N also sees step N's `last` (single writer).
    stats.steps += 1
    if not is_prof_enabled():
        return
    p = f"aperture.{kind}"
    PROF.record_ms(f"{p}.step", step_s * 1e3)
    PROF.record_ms(f"{p}.d2h", d2h_s * 1e3)
    PROF.record_ms(f"{p}.write", write_s * 1e3)
    PROF.record_ms(f"{p}.write_tail", write_tail_s * 1e3)
    PROF.record_ms(f"{p}.bookkeeping", bookkeeping_s * 1e3)
    for m, n in bytes_by_mode.items():
        PROF.incr(f"{p}.bytes.{m}", int(n))
    PROF.incr(f"{p}.write_busy_us", int(busy_s * 1e6))


__all__ = [
    "WRITE_MODE_ENV", "WRITE_THREADS_ENV", "DIRECT_MIN_BYTES_ENV", "WRITE_MODES",
    "DEFAULT_WRITE_THREADS", "DIRECT_MIN_BYTES",
    "ApertureWriteConfigError", "ApertureWriteError", "resolve_write_mode",
    "resolve_write_threads", "resolve_direct_min_bytes", "resolve_per_request_write_mode",
    "WriteShape", "predict_rows_per_write",
    "DirectIOInfo", "probe_direct_io", "alloc_host_rows",
    "tensor_bytes_view", "RawFileSink", "PerRequestSinks", "WriterPool", "join_writes",
    "WriteStats", "ApertureWritePath", "timed_write", "record_step_stats", "join_writes_quietly",
]
