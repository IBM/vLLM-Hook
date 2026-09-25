"""Multi-layer host drain for the HS capture aperture.

Decoder-layer apertures share ONE logical cursor (every layer scatters the same tokens each step).
Each pass reads every per-layer ``hs_buf``'s new ``[drain, write)`` segment, appends it to that
layer's raw file, and appends this step's ``LayerEntry`` records to a shared sidecar keyed by
``(req_id, layer)``. SELECTIVE DRAIN (``MIA_DRAIN_SELECTIVE``, default ON) copies only the
row ranges a request named, per layer — see ``build_copy_plans``. Runs synchronously, post-forward:
the ``hs_buf[s:e].cpu()`` D2H is stream-ordered after the in-graph ``capture_hs`` scatter (same
default stream), so it reads committed rows with no explicit event.

WRITE PATH (``MIA_APERTURE_WRITE_MODE``, default ``auto``; see ``aperture_sink``). The shared-file
drains keep every raw file open for the run and write zero-copy from the host rows -- O_DIRECT for
files whose rows are a multiple of the detected block size, buffered otherwise -- and the off-loop
drain overlaps per-layer D2H with a writer-thread pool, joining every write before
``advance_drain``. The sidecar is accumulated as per-step arrays (``HsSidecarLog``) instead of one
``LayerEntry`` per (request, layer, step). Raw files and sidecar are byte-identical to ``legacy``,
the pre-existing path kept verbatim below for A/B validation only.
"""
from __future__ import annotations

import bisect
import logging
import mmap
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

from mia._profiler import PROF
from .capture_aperture import CaptureAperture
from .per_request_delivery import PerRequestIndex
from .aperture_metadata import (
    HsSidecarLog, LayerEntry, StepMeta, expand_records, write_sidecar)
from .aperture_sink import (
    ApertureWriteConfigError, ApertureWriteError, ApertureWritePath, PerRequestSinks,
    WRITE_MODE_ENV, WriteShape, WriteStats, alloc_host_rows, join_writes, join_writes_quietly,
    record_step_stats, resolve_per_request_write_mode, resolve_write_mode, resolve_write_threads,
    timed_write)
from .thread_device import bind_thread_to_device

logger = logging.getLogger(__name__)

_GIB = 1024 ** 3


def _torch_dtype_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"`` — the string the aperture header/reader key on."""
    return str(dtype).rsplit(".", 1)[-1]


def _raw_bytes(t: torch.Tensor) -> bytes:
    """Row-major raw bytes of a CPU tensor. bfloat16 has no numpy dtype, so reinterpret its
    identical 2-byte payload as uint16 (the reader reads bf16 files back as uint16 then
    ``view(torch.bfloat16)``, so the bytes round-trip exactly)."""
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def _resolve_mmap_capacity_bytes(aperture: CaptureAperture) -> int:
    """Per-layer raw-file mmap pre-size, in bytes.

    ``MIA_APERTURE_MMAP_BYTES`` overrides outright. Default: one full aperture's worth of rows
    (``n_slots * row_bytes``), floored at 2 GiB — a STARTING size, not a cap: a run's total
    captured rows can exceed one aperture's worth many times over, and ``_MmapLayerWriter`` falls back
    to a plain append past the pre-sized capacity, so an under-estimate costs perf, not
    correctness.
    """
    override = os.environ.get("MIA_APERTURE_MMAP_BYTES")
    if override:
        return int(override)
    return max(2 * _GIB, int(aperture.n_slots) * int(aperture.row_bytes))


class _MmapLayerWriter:
    """Pre-sized ``MAP_SHARED`` write handle for one layer's raw file (``MIA_APERTURE_MMAP=1`` —
    intended for a LOCAL NVMe run dir).

    The memcpy into the mapping holds the GIL for its whole duration, so on a local run dir the
    plain ``write()`` (which releases it) wins end to end even though it moves the same bytes. Keep
    this path only when the sink is genuinely network-bound, where removing the per-step ``open()``
    cost outweighs the added GIL time.

    ``__init__`` ``ftruncate``s the file to ``capacity_bytes`` (sparse) and ``mmap``s it
    ``PROT_WRITE``/``MAP_SHARED``. ``append`` memmoves into the mapping; the FIRST write that would
    exceed ``capacity_bytes`` writes the part that still fits, then permanently falls back to a
    plain ``open(path, "ab")`` for the rest of this instance's life — never loses or corrupts data,
    logs once. ``close`` ``msync``s, truncates the file DOWN to the real written length (the reader
    ``np.memmap``s by byte length, so the file must end exactly at the real data), then ``munmap``s
    and closes the fd.
    """

    def __init__(self, path: str, capacity_bytes: int):
        self.path = path
        self.capacity = int(capacity_bytes)
        self.offset = 0
        self._overflowed = False
        self._warned = False
        # "w+b": truncate-to-0 first so a stale prior file never leaks bytes past this run.
        self._fh = open(path, "w+b")
        self._fh.truncate(self.capacity)
        self._mm = mmap.mmap(self._fh.fileno(), self.capacity, access=mmap.ACCESS_WRITE)

    def append(self, data: bytes) -> None:
        n = len(data)
        if n == 0:
            return
        if not self._overflowed:
            room = self.capacity - self.offset
            if n <= room:
                self._mm[self.offset:self.offset + n] = data
                self.offset += n
                return
            # Overflow: write what still fits, then fall back permanently. Release the mapping now
            # so the file doesn't sit at its zero-padded capacity while plain appends continue past it.
            if room > 0:
                self._mm[self.offset:self.offset + room] = data[:room]
                self.offset += room
                data = data[room:]
            self._overflowed = True
            if not self._warned:
                logger.warning(
                    "hs aperture mmap sink: %s exceeded its pre-sized mmap capacity (%d bytes); "
                    "falling back to plain append for the overflow (raise "
                    "MIA_APERTURE_MMAP_BYTES to size the mapping for this workload)",
                    self.path, self.capacity)
                self._warned = True
            self._mm.flush()
            self._mm.close()
            self._fh.truncate(self.offset)   # drop the still-mapped zero-padded tail
            self._fh.close()
            self._mm = None
            self._fh = None
        if data:
            with open(self.path, "ab") as f:
                f.write(data)
            self.offset += len(data)

    def close(self) -> None:
        if self._mm is not None:
            self._mm.flush()          # msync: durability, dirty pages -> the backing NVMe file
            self._mm.close()          # munmap
            self._mm = None
        if self._fh is not None:
            self._fh.truncate(self.offset)   # drop the zero-padded pre-sized tail
            self._fh.close()
            self._fh = None


import re as _re


def _aperture_debug() -> bool:
    """Gated disk-pipeline debug logging (MIA_APERTURE_DEBUG=1). Off by default -> no perf impact;
    read at call time so a per-worker env set before spawn (or a test monkeypatch) takes effect."""
    return os.environ.get("MIA_APERTURE_DEBUG") == "1"


def _dbg(msg: str) -> None:
    print(f"[hookplugin/aperture-disk] {msg}", flush=True)


def _stamp_file_row(entries: List[LayerEntry], cursor_before: Dict[int, int],
                     step_start_logical: int,
                     plans: Optional[Dict[int, "LayerCopyPlan"]] = None) -> None:
    """Stamp each entry's `file_row` = the real per-layer file cursor (`cursor_before[layer]`,
    captured before this step's append) plus the entry's offset WITHIN this step's append.

    FULL drain (`plans=None`): every layer got the whole row range, so the offset is simply
    `entry.logical_start - step_start_logical` (reduces to `file_row == entry.logical_start`).
    SELECTIVE drain (`plans` given): the layer got only the rows some request wanted, so the offset
    is that row's position in the COMPACTED copy (`plan.row_offset`) — without this split the entry
    would point at a row holding some OTHER request's data.

    A layer that is NOT INSTALLED, or an entry with `n_rows <= 0`, keeps the full-drain arithmetic
    even under a selective drain: neither names rows the plan could have copied, so `row_offset`
    would fail loud on it for no gain. Mutates each `LayerEntry` in place."""
    for e in entries:
        base = cursor_before.get(e.layer, 0)
        plan = plans.get(int(e.layer)) if plans is not None else None
        if plan is None or int(e.n_rows) <= 0:
            off = int(e.logical_start) - int(step_start_logical)
        else:
            off = plan.row_offset(e.logical_start)
        e.file_row = base + off


# SELECTIVE DRAIN (MIA_DRAIN_SELECTIVE, default ON): copy only the (layer, row-range) tiles
# some request named, not every installed layer's whole step span. The list is PER LAYER because
# requests in one batch want different layer sets. When everyone wants every layer the per-layer
# union IS the step span, so the result is identical to the flag being off -- is_degenerate_full_step
# detects that in one O(records) pass and takes the off path verbatim. Invariants this must not move:
# advance_drain still frees the whole span; _read_segments keeps its copy-stream ordering; the copy
# never reaches outside the step (checked in build_copy_plans, raises).

def _drain_selective_enabled() -> bool:
    """``MIA_DRAIN_SELECTIVE`` -- default ON; ``=0`` is the kill switch and the full-drain
    control. ``"1"`` stays the only ON value (a stray ``"true"`` reads as OFF, matching the other
    drain-related feature flags). Read once, at drain construction (install time), and pinned on
    the instance -- never re-read per step."""
    return os.environ.get("MIA_DRAIN_SELECTIVE", "1") == "1"


def _resolve_selective(armed: bool, *, off_loop: bool,
                       per_request: bool) -> Tuple[bool, Optional[str]]:
    """Decide whether THIS drain runs selectively, and (when armed but refused) why not.

    The synchronous drain and per-request delivery both FULL-DRAIN: neither has a selective path.
    Per-request delivery (``_demux_into_index``) slices each layer's rows by an offset into a DENSE
    host image of the step, which selective copying would compact and invalidate; the synchronous
    drain (``MIA_APERTURE_SYNC_DRAIN=1``) has its own copy loop with no selective branch at all.
    Both fall back rather than raise, so an unrelated perf default can never turn a working
    deployment into a hard install failure — the refusal is reported instead, via the returned
    reason string and ``row_counts()["rows_skipped"]``."""
    if not armed:
        return False, None
    if not off_loop:
        return False, ("the SYNCHRONOUS drain (MIA_APERTURE_SYNC_DRAIN=1) has no selective path; "
                       "every installed layer is drained")
    if per_request:
        return False, ("per-request delivery (MIA_APERTURE_PER_REQUEST=1) demuxes from a dense "
                       "host image of the step, so it drains every installed layer")
    return True, None


def _merge_ranges(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort ``(logical_start, n_rows)`` ranges and merge overlapping OR touching ones (``next_start
    <= cur_end`` -- touching matters: two requests packed back-to-back in one step must become ONE
    copy, not two). Empty ranges are dropped. Pure."""
    out: List[Tuple[int, int]] = []
    for s, n in sorted((int(s), int(n)) for s, n in ranges if int(n) > 0):
        if out and s <= out[-1][0] + out[-1][1]:
            end = max(out[-1][0] + out[-1][1], s + n)
            out[-1] = (out[-1][0], end - out[-1][0])
        else:
            out.append((s, n))
    return out


@dataclass(frozen=True)
class LayerCopyPlan:
    """What ONE layer's drain copies out of the aperture for ONE step.

    ``starts``/``lengths`` are the merged LOGICAL row ranges wanted for this layer, ascending;
    ``segments`` are those ranges mapped through ``aperture.segments_at`` (so a range straddling the
    physical wrap contributes its 2 segments), in the SAME logical order the copies are issued and
    concatenated in. ``offsets[i]`` is where range ``i``'s first row lands in the COMPACTED copy --
    the quantity that stops being ``logical_start - step_start`` the moment a layer skips rows, and
    the reason ``row_offset`` exists. ``total_rows`` sizes that layer's pinned host buffer.

    Immutable and shareable: the non-selective path hands the SAME whole-span plan object to every
    installed layer."""
    starts: Tuple[int, ...]
    lengths: Tuple[int, ...]
    offsets: Tuple[int, ...]
    segments: Tuple[Tuple[int, int], ...]
    total_rows: int

    @classmethod
    def from_ranges(cls, ranges: Iterable[Tuple[int, int]], aperture: CaptureAperture) -> "LayerCopyPlan":
        merged = _merge_ranges(ranges)
        starts: List[int] = []
        lengths: List[int] = []
        offsets: List[int] = []
        segments: List[Tuple[int, int]] = []
        off = 0
        for s, n in merged:
            starts.append(s)
            lengths.append(n)
            offsets.append(off)
            segments.extend(aperture.segments_at(s, n))
            off += n
        return cls(tuple(starts), tuple(lengths), tuple(offsets), tuple(segments), off)

    @property
    def ranges(self) -> Tuple[Tuple[int, int], ...]:
        """The merged logical ``(start, n_rows)`` ranges — derived, never stored twice."""
        return tuple(zip(self.starts, self.lengths))

    def row_offset(self, logical_start: int) -> int:
        """Where the aperture's logical row ``logical_start`` lands in this layer's COMPACTED copy.

        Fails LOUD (``KeyError``) on a row this plan never copied: under selective drain a wrong
        offset is not a crash, it is plausible-looking wrong data in the artifact (right shape, right
        layer set, stale contents) — the defect class this plan has already shipped once. A drain
        error is fatal by design (it never advances the cursor, so backpressure surfaces it), which
        is the correct blast radius for "the plan and the entries disagree"."""
        ls = int(logical_start)
        i = bisect.bisect_right(self.starts, ls) - 1
        if i < 0 or ls >= self.starts[i] + self.lengths[i]:
            raise KeyError(
                f"logical row {ls} is not covered by this layer's copy plan {self.ranges}")
        return self.offsets[i] + (ls - self.starts[i])


def is_degenerate_full_step(records, installed: Set[int], start_logical: int,
                            n_rows: int) -> bool:
    """Does THIS step want every installed layer over its whole span — i.e. is the selective copy
    list provably the same one the ``selective=False`` branch produces? Pure; one pass over
    ``records``, and it BAILS on the first subset record, so the common subset case is O(1).

    THE TEST, exact rather than heuristic:

      1. ``len(rec.layers) == len(installed)`` — the fast bail; fewer layers than installed means
         the subset case by definition.
      2. the FIRST record's layer set is checked exactly (``set(ls) == installed``); every later
         record is compared to it with ``list ==`` (early-exit). Same set in a different ORDER reads
         as "not degenerate" and takes the slow path — a conservative MISS, never a wrong answer.
      3. the records must TILE the span: a cursor starts at ``start_logical``, each record must
         begin exactly at it, and the walk must end exactly at ``start_logical + n_rows`` — the exact
         form of "the union of the wanted ranges IS the whole span" (min/max/sum would admit an
         overlap-plus-gap arrangement; this cannot).

    THE SPAN BOUND IS STRENGTHENED, NOT SKIPPED: the monotonic cursor walk means every record it
    accepts is contained in ``[start_logical, start_logical + n_rows)`` by construction, so an
    out-of-span record cannot reach the fast path — it fails the walk and falls into
    ``build_copy_plans``' record-driven branch, which raises. The fast-path plan is then built from
    the SPAN, not from the records, restoring the pre-selective-drain structural bound.

    A zero-row record, or flat ``LayerEntry`` input (no ``layers`` attribute), makes the step
    non-degenerate — a conservative miss, not a wrong answer."""
    n_inst = len(installed)
    if n_inst == 0:
        return False
    cursor = int(start_logical)
    hi = cursor + int(n_rows)
    first_layers = None
    for rec in records:
        ls = getattr(rec, "layers", None)
        if ls is None or len(ls) != n_inst:
            return False
        n = int(rec.n_rows)
        if n <= 0 or int(rec.logical_start) != cursor:
            return False
        cursor += n
        if first_layers is None:
            if set(ls) != installed:
                return False
            first_layers = ls
        elif ls != first_layers:
            return False
    return first_layers is not None and cursor == hi


def _whole_span_plans(layers: Iterable[int], aperture: CaptureAperture, start_logical: int,
                      n_rows: int) -> Dict[int, "LayerCopyPlan"]:
    """Full-drain copy list: every installed layer copies the whole step span, via ONE shared plan
    object. Both the flag-off branch and the degenerate fast path return exactly this — same
    construction, same key order — so "the flag is off" and "everyone wants everything" are one
    code path, not two."""
    whole = LayerCopyPlan.from_ranges([(int(start_logical), int(n_rows))], aperture)
    return {int(ln): whole for ln in layers}


def build_copy_plans(records, layers: Iterable[int], aperture: CaptureAperture,
                     start_logical: int, n_rows: int,
                     selective: bool) -> Dict[int, LayerCopyPlan]:
    """The copy list: ``{layer -> LayerCopyPlan}`` for ONE step. Pure (reads only ``aperture.n_slots``
    via ``segments_at``); no I/O, no drain state.

    ``selective=False`` reproduces full-drain behaviour exactly — every installed layer copies the
    whole step span ``[start_logical, start_logical + n_rows)`` — via ONE shared plan object.

    ``selective=True`` takes, per layer, the union of ``[logical_start, logical_start + n_rows)``
    over the records naming that layer, merged, and maps it through ``segments_at``. A layer no
    record names gets NO entry and is not copied. Except in the degenerate case — every request
    wanting every installed layer over the whole span — where it short-circuits to the same
    ``_whole_span_plans`` the ``selective=False`` branch returns (``is_degenerate_full_step``).

    ``records`` accepts either per-request ``ReqCaptureRecord``s (each carrying its OWN ``layers``
    list) or already-flat ``LayerEntry``s, mirroring ``expand_records``' contract; reading them
    directly is equivalent to unioning over ``expand_records(records)`` while avoiding a second
    O(reqs x layers) fan-out on the consumer thread.

    Layers a record names but that are not installed are ignored (no buffer, no file).

    EVERY merged range is checked against the step span and a violation RAISES: the copy used to be
    STRUCTURALLY bounded (literally ``segments_at(item.start_logical, item.n_rows)``), and building
    the list from the records turns that into an unchecked assumption about the routing builders —
    an out-of-bound record would have the consumer read rows the step's event does not fence and the
    engine may be actively scattering into. The raise restores the bound as checked; a drain error
    is fatal by design and never advances the aperture cursor, so it surfaces as backpressure rather
    than a plausible-looking artifact (symmetric with ``LayerCopyPlan.row_offset``'s ``KeyError``).
    """
    if not selective:
        return _whole_span_plans(layers, aperture, start_logical, n_rows)
    installed = {int(ln) for ln in layers}
    # Degenerate fast path: when every request wants every installed layer over the whole span,
    # the loop below would rebuild `_whole_span_plans` the expensive way. `is_degenerate_full_step`
    # decides that first, and its cursor walk enforces the step-span bound at least as strictly as
    # the check below. `_drain_item` already tests this itself and passes the result in as
    # `selective`, so this re-check never fires from production; it stays reachable here so
    # `test_drain_selective.py` (which calls this with a raw `selective=True`) exercises it too.
    if is_degenerate_full_step(records, installed, start_logical, n_rows):
        return _whole_span_plans(layers, aperture, start_logical, n_rows)
    wanted: Dict[int, List[Tuple[int, int]]] = {}
    for rec in records:
        rec_layers = getattr(rec, "layers", None)
        if rec_layers is None:                 # already a flat LayerEntry -> one layer
            rec_layers = (rec.layer,)
        n = int(rec.n_rows)
        if n <= 0:
            continue
        s = int(rec.logical_start)
        for ln in rec_layers:
            ln = int(ln)
            if ln in installed:
                wanted.setdefault(ln, []).append((s, n))
    lo = int(start_logical)
    hi = lo + int(n_rows)
    plans: Dict[int, LayerCopyPlan] = {}
    for ln, rs in wanted.items():
        plan = LayerCopyPlan.from_ranges(rs, aperture)
        # Merged ranges are sorted and disjoint, so the first start and the last end bound them all
        # -- an O(1) check per wanted layer, not per record.
        if plan.total_rows and (plan.starts[0] < lo
                                or plan.starts[-1] + plan.lengths[-1] > hi):
            raise ValueError(
                f"selective drain: layer {ln} was asked for rows {plan.ranges}, which reach "
                f"outside this step's span [{lo}, {hi}). The consumer would read aperture rows this "
                f"step does not own -- unfenced by the step's scatter event and possibly being "
                f"written right now. Refusing to copy (the aperture cursor is not advanced).")
        plans[ln] = plan
    return plans


def record_captured_cells(records) -> int:
    """Total (layer, row) cells the per-request ``records`` NAME -- what a SELECTIVE drain actually
    copies this step, without paying `build_copy_plans`' O(records x layers) merge/sort. Sums each
    record's ``n_rows * len(rec.layers)`` (one cell per entry for an already-flat ``LayerEntry``
    with no ``.layers``) -- O(records), so an ENGINE-LOOP caller isn't repaying, on the loop the
    selective drain exists to relieve, the cost it was built to move off.

    Equal to the per-layer union `build_copy_plans` computes: two records from the same step never
    share a aperture row, so summing per-record cells can never double-count a (layer, row) pair.

    ONLY VALID when the drain is actually running selectively (`_selective_active()` True); a
    non-selective drain copies every installed layer regardless of what any record names, so a
    caller there must use ``n_rows_total * n_installed`` instead -- getting that gate right is the
    caller's job (`install_hs.py`'s gauge is the one production caller)."""
    total = 0
    for rec in records:
        n = int(getattr(rec, "n_rows", 0) or 0)
        if n <= 0:
            continue
        layers = getattr(rec, "layers", None)
        total += n if layers is None else n * len(layers)
    return total


def _match_disk_route(rid: str, route_keys) -> Optional[str]:
    """Map a drain-seen id to the EXTERNAL key it was disk-routed under, or None.

    Serve rewrites the external request_id to an INTERNAL ``{external}-{random8}``, so the id the
    drain sees on rows and on the finish signal is INTERNAL while the router registered the disk
    route under the EXTERNAL id. Mirror ``workers/_common.iter_matching_req_ids``' exact-or-``{ext}-``
    rule so every disk hop keys consistently on the external id -- without it a serve disk-routed
    request's rows fall to the host index instead of its staging. ``route_keys`` is a snapshot of
    ``self._disk_routed`` keys (taken under ``_index_lock``); this match itself is pure (no lock)."""
    rid = str(rid)
    if rid in route_keys:
        return rid
    for ext in route_keys:
        if rid.startswith(f"{ext}-"):
            return ext
    return None


def _sanitize_req_id(req_id: str) -> str:
    """Filesystem-safe per-request run-dir name. Internal vLLM req_ids are already nearly path-safe;
    collapse any stray unsafe char to ``_`` (the true req_id is stored verbatim in each
    ``LayerEntry.req_id`` in the sidecar, so the reconstruction keys on the exact id regardless of
    the dir name)."""
    s = _re.sub(r"[^A-Za-z0-9._-]", "_", str(req_id))
    return s or "req"


class _PerRequestDiskStaging:
    """The DISK route's per-request staging: stream ONE request's demuxed rows to its OWN
    per-request run_dir laid out exactly like the shared-file run — per-layer ``hs_layer_<L>.raw``
    (via a reused :class:`_MmapLayerWriter`) + a per-request ``hs_aperture_meta.jsonl`` sidecar — so
    ``aperture_reader.load_multilayer_aperture_artifact(run_dir)`` reconstructs that single request
    byte-identically.

    KEY LAYOUT INVARIANT: because only THIS request writes into these files, each block's
    ``LayerEntry.logical_start`` is the running per-``(req, layer)`` row count (0, then n_rows, ...)
    == the row offset into this layer's file — the exact invariant the reader keys on, now scoped to
    one request. (The shared-file drain instead uses the global aperture cursor as ``logical_start``;
    here we RELABEL to a per-request-local offset so the whole-run reader reconstructs one request
    from a run_dir that contains only it.)

    Written entirely on the drain's CONSUMER thread (one writer per request), so the row appends need
    no lock; the drain guards only the ``_disk_staging`` dict membership (create/pop) it lives in.

    WRITE PATH (``write_mode``, resolved ONCE by the drain from ``MIA_APERTURE_WRITE_MODE``).
    ``buffered`` -- what ``auto``, the default, resolves to here -- writes through
    :class:`PerRequestSinks`: the shared-file drain's zero-copy writer, one run-long fd per layer,
    no ``bytes`` copy and no open/close per step. ``legacy`` keeps the pre-2026-09-20 sequence
    (``_raw_bytes`` then ``open(p,"ab")``/write/close per layer per step, or the mmap writer under
    ``MIA_APERTURE_MMAP=1``) verbatim, for A/B validation. Bytes, file names, ``LayerEntry``
    offsets and the sidecar are IDENTICAL either way (``tests/test_capture_data_path.py`` compares
    them byte for byte). O_DIRECT is never used here: these writes are one layer's rows for one
    request -- 8 KiB at 8B, 16 KiB at 70B, issued inline on the drain thread with no writer pool,
    which is the configuration job 1802981 measures direct 1.65-1.74x WORSE per write in -- and
    the rows are a mid-buffer slice with no alignment guarantee."""

    def __init__(self, req_id: str, run_dir: str, header: dict, capacity_bytes: int,
                 use_mmap: bool, write_mode: str = "legacy"):
        self.req_id = str(req_id)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.header = dict(header)
        self.meta_path = os.path.join(run_dir, "hs_aperture_meta.jsonl")
        self._capacity = int(capacity_bytes)
        self.write_mode = str(write_mode)
        self._legacy = self.write_mode == "legacy"
        self._use_mmap = bool(use_mmap) and self._legacy
        self._writers: Dict[int, _MmapLayerWriter] = {}   # layer -> mmap writer (legacy mmap path)
        self._plain: Dict[int, str] = {}                  # layer -> raw path (legacy plain path)
        self._sinks = None if self._legacy else PerRequestSinks(f"hs staging req {req_id}")
        self._rows: Dict[int, int] = {}                   # layer -> cumulative rows == next start
        self._entries: List[LayerEntry] = []
        self._closed = False

    def _raw_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"hs_layer_{layer}.raw")

    def append(self, layer: int, rows_cpu: torch.Tensor, n_rows: int, mode: str) -> None:
        """Append one layer's already-on-host rows (LOGICAL/step order) to this request's per-layer
        raw file and record the matching ``LayerEntry`` at the per-request-local row offset. The
        write is SYNCHRONOUS on this thread — a zero-copy ``pwrite`` straight from the caller's
        rows, or the legacy ``_raw_bytes`` copy — so the caller's source view is fully consumed
        before the aperture frees it on ``advance_drain``; no clone needed either way."""
        start = self._rows.get(layer, 0)
        if self._sinks is not None:
            self._sinks.append(layer, self._raw_path(layer), rows_cpu)
        elif self._use_mmap:
            w = self._writers.get(layer)
            if w is None:
                w = _MmapLayerWriter(self._raw_path(layer), self._capacity)
                self._writers[layer] = w
            w.append(_raw_bytes(rows_cpu))
        else:
            p = self._plain.get(layer)
            if p is None:
                p = self._raw_path(layer)
                open(p, "wb").close()   # truncate up front: never append onto stale bytes
                self._plain[layer] = p
            with open(p, "ab") as f:
                f.write(_raw_bytes(rows_cpu))
        self._entries.append(LayerEntry(self.req_id, int(layer), int(start), int(n_rows), mode))
        self._rows[layer] = start + int(n_rows)

    def close(self) -> None:
        """Finalize on the request's FINISH: msync+truncate every per-layer mmap writer (durability;
        drop the zero-padded pre-sized tail), then write this request's sidecar. Idempotent.

        TOLERATES A PARTIAL / ABORTED STAGING. A request aborted mid-generation staged only SOME of
        its per-layer files, and its run_dir may even have been rmtree'd by the abort-discard path
        racing this finish. So:
          * the sidecar references ONLY the layers actually ``append``ed (``self._entries`` — never a
            never-created ``hs_layer_<L>.raw``), so a partial layer set closes without a missing-file
            error; and
          * each writer close AND the sidecar write are BEST-EFFORT (guarded): a vanished run_dir or a
            partial mmap must NOT raise out of ``_handle_finish`` and wedge the off-loop consumer (the
            observed serve failure: ``FileNotFoundError`` in ``close()`` propagating into ``_run``).
        The HEALTHY path is byte-identical: the run_dir is present, every appended file exists, and
        the sidecar is written exactly as before."""
        if self._closed:
            return
        self._closed = True
        if self._sinks is not None:
            self._sinks.close()          # close this request's run-long fds (best-effort, logged)
        for w in self._writers.values():
            try:
                w.close()                # msync + truncate this layer's mmap (durability)
            except Exception:            # noqa: BLE001 -- best-effort msync of a partial/aborted writer
                logger.exception(
                    "hs per-request staging: writer close failed for req %r (partial staging); "
                    "continuing", self.req_id)
        try:
            if os.path.isdir(self.run_dir):
                write_sidecar(self.meta_path, [StepMeta(list(self._entries))], self.header)
            # else: the run_dir vanished (aborted + discarded under us) -> nothing to deliver; skip.
        except Exception:                # noqa: BLE001 -- a partial/vanished dir must never wedge finish
            logger.exception(
                "hs per-request staging: sidecar write failed for req %r under %r (partial/aborted "
                "staging); delivery skipped", self.req_id, self.run_dir)

    def discard(self) -> None:
        """ABORT cleanup: release this request's open writers (fds/mmap) WITHOUT writing a sidecar
        (an aborted request is never delivered/read), then remove its staging dir to reclaim NVMe.
        Idempotent; best-effort (never raises)."""
        if self._sinks is not None:
            self._sinks.close()
        for w in self._writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001 -- best-effort release of a partial mmap
                pass
        self._writers = {}
        self._closed = True
        import shutil
        shutil.rmtree(self.run_dir, ignore_errors=True)


class MultiLayerApertureDrain:
    """Drains a shared-cursor ``CaptureAperture`` across N per-layer ``hs_buf`` buffers.

    ``layers`` is ``[(layer_1based, hs_buf), ...]`` in layer order. Each drain appends the SAME
    ``[drain, write)`` rows from every layer's buffer to that layer's raw file, so today the shared
    aperture-wide logical row offset (`LayerEntry.logical_start`) happens to equal the row offset into
    EVERY per-layer file — but the reader keys on the SEPARATE per-layer `file_row` field, stamped
    here from a real per-layer running cursor (`self._file_rows`) rather than assumed equal.
    ``record_entries`` queues this step's ``LayerEntry`` records; ``drain_once`` copies the pending
    region, appends per-layer, stamps `file_row`, advances the shared drain cursor, and returns rows
    moved. ``close`` writes the accumulated shared sidecar.
    """

    # The synchronous drain writes pageable ``.to("cpu")`` tensors with no alignment guarantee, so
    # it never opens a file O_DIRECT; the off-loop drain D2Hs into its own aligned host buffers.
    _ALLOW_DIRECT = False
    _DIRECT_REFUSAL = ("the synchronous drain (MIA_APERTURE_SYNC_DRAIN=1) writes pageable host "
                       "copies with no alignment guarantee; it writes zero-copy buffered")

    def __init__(self, aperture: CaptureAperture, layers: List[Tuple[int, torch.Tensor]],
                 run_dir: str, header: dict, setup_sink: bool = True,
                 shape: Optional[WriteShape] = None):
        self.aperture = aperture
        self.layers = list(layers)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.header = dict(header)
        self.meta_path = os.path.join(run_dir, "hs_aperture_meta.jsonl")
        self.raw_paths = {ln: os.path.join(run_dir, f"hs_layer_{ln}.raw")
                          for ln, _ in self.layers}
        # What one drained step is predicted to write per layer file, from the capture CONFIG
        # (install_hs computes it; None = not predicted, and `auto` then decides on alignment
        # alone, as it did before 2026-09-20). Only `auto` consults it.
        self.shape = shape
        # WRITE PATH (MIA_APERTURE_WRITE_MODE, default auto). `legacy` keeps the pre-existing sink
        # below byte for byte; every other mode opens the raw files ONCE through ApertureWritePath
        # (O_DIRECT where aligned, else zero-copy buffered) and keeps the sidecar as per-step
        # arrays. Resolved once here and pinned on the instance.
        self.write_mode, self._write_mode_explicit = resolve_write_mode()
        self._wp: Optional[ApertureWritePath] = None
        self._sidecar: Optional[HsSidecarLog] = None
        self._io_lock = threading.Lock()
        self._wstats = WriteStats()
        self._write_note = ""
        # mmap-NVMe raw sink, default OFF (MIA_APERTURE_MMAP=1 opts in). Default is the plain
        # open(ab)+write path: write() releases the GIL, the mmap memcpy holds it, and holding it on
        # the drain consumer thread taxes the engine loop -- both paths write the same bytes, this
        # is a scheduling choice, not a format one. `run_dir` is caller-supplied; this class never
        # hardcodes a path.
        # `setup_sink=False` (per-request delivery mode) skips the shared per-layer raw files
        # entirely -- those rows are demuxed by req_id into a PerRequestIndex instead.
        self._mmap_writers: Dict[int, _MmapLayerWriter] = {}
        self._mmap_enabled = False
        # The write path a DISK-routed request's per-request staging takes (per-request delivery
        # only; `legacy` for every shared-file drain, which never builds one).
        self._perreq_write_mode = "legacy"
        if not setup_sink:
            # Per-request delivery writes no shared raw file, so the per-file O_DIRECT decision
            # does not apply. Its DISK sub-route's per-request staging does write, and it takes
            # the same zero-copy writer in buffered mode (`_perreq_write_mode`).
            self._perreq_write_mode = resolve_per_request_write_mode(
                self.write_mode, self._write_mode_explicit)
            self._write_note = (
                f"write mode={self.write_mode} does not apply to shared raw files: per-request "
                f"delivery (MIA_APERTURE_PER_REQUEST=1) writes none. Its disk staging writes "
                + ("zero-copy buffered, one fd per layer kept open for the request"
                   if self._perreq_write_mode != "legacy"
                   else "through the legacy per-step tobytes + open/append/close path"))
        elif self.write_mode != "legacy":
            if os.environ.get("MIA_APERTURE_MMAP", "0") != "0":
                raise ApertureWriteConfigError(
                    f"MIA_APERTURE_MMAP={os.environ.get('MIA_APERTURE_MMAP')!r} selects the legacy "
                    f"mmap sink, but {WRITE_MODE_ENV}={self.write_mode} "
                    f"({'set' if self._write_mode_explicit else 'the default'}). Unset "
                    f"MIA_APERTURE_MMAP (the files now stay open for the run, which is what the "
                    f"mmap sink was for), or set {WRITE_MODE_ENV}=legacy to use it.")
            files = {ln: (self.raw_paths[ln], "hs", int(buf.shape[1]) * int(buf.element_size()))
                     for ln, buf in self.layers}
            self._wp = ApertureWritePath(
                run_dir, files, self.write_mode, allow_direct=self._ALLOW_DIRECT,
                direct_refusal=self._DIRECT_REFUSAL, label=f"hs aperture drain ({run_dir})",
                shape=self.shape)
            self._sidecar = HsSidecarLog([ln for ln, _ in self.layers], _stamp_file_row)
        else:
            self._mmap_enabled = os.environ.get("MIA_APERTURE_MMAP", "0") != "0"
            if self._mmap_enabled:
                cap = _resolve_mmap_capacity_bytes(aperture)
                try:
                    for ln, path in self.raw_paths.items():
                        self._mmap_writers[ln] = _MmapLayerWriter(path, cap)
                except OSError as e:
                    logger.warning(
                        "hs aperture mmap sink: failed to mmap raw file(s) under %s (%s); falling back "
                        "to the plain append path for the whole run (unset MIA_APERTURE_MMAP to "
                        "silence)", run_dir, e)
                    for w in self._mmap_writers.values():
                        try:
                            w.close()
                        except Exception:  # noqa: BLE001 — best-effort cleanup of a partial mmap set
                            pass
                    self._mmap_writers = {}
                    self._mmap_enabled = False
            if not self._mmap_enabled:
                # Truncate raw files up front so a re-run never appends onto stale bytes.
                for p in self.raw_paths.values():
                    open(p, "wb").close()
        self._steps: List[StepMeta] = []
        self._pending_entries: List[LayerEntry] = []
        self._closed = False
        # SELECTIVE DRAIN (MIA_DRAIN_SELECTIVE, default ON) -- resolved ONCE here, never
        # re-read per step. The SYNCHRONOUS drain has no selective path, so this base class always
        # resolves to a full drain (with the refusal reason recorded when the flag IS armed);
        # OffLoopApertureDrain re-resolves for itself below.
        self.selective, self.selective_disabled_reason = _resolve_selective(
            _drain_selective_enabled(), off_loop=False, per_request=False)
        # Rows this drain ACTUALLY copied / did not copy, counted at the copy site (never derived
        # from a request's layer list) -- the witness that tells "copied everything anyway" apart
        # from "copied only the wanted tiles". Single writer (this drain's thread); read by the RPC.
        self._rows_copied = 0
        self._rows_skipped = 0
        # Steps on which an ARMED selective drain found nothing to skip and took the degenerate
        # fast path. "0 skipped" alone can't distinguish that from an armed-but-idle or refused
        # drain, so this is counted separately at the decision site, one per drained step.
        self._degenerate_steps = 0
        # The installed layer numbers, in `self.layers` order, plus their set -- hoisted out of the
        # per-step drain so neither the copy-list call nor the degenerate test rebuilds them.
        self._layer_nums: List[int] = [int(ln) for ln, _ in self.layers]
        self._layer_set = set(self._layer_nums)
        # Per-layer running file cursor: rows appended so far to layer `ln`'s raw file. Bumped in
        # `_append_layer_rows`; read (pre-bump) by `drain_once` to stamp each step's `file_row`.
        # Pre-populated for every installed layer so `.get(ln, 0)` never depends on insertion order.
        self._file_rows: Dict[int, int] = {ln: 0 for ln, _ in self.layers}
        # Non-legacy modes: the same cursor as an array over `self.layers` order, plus whether all
        # layers stand at one value (then a step's cursor is one int, not an array, in the sidecar
        # log). `_pending_records` holds the sync drain's un-expanded records until `drain_once`.
        self._fr = np.zeros(len(self.layers), dtype=np.int64)
        self._fr_uniform = True
        self._pending_records: list = []

    # ---- non-legacy file-row cursor ----
    def _cursor_advance_all(self, n_rows: int):
        """Every installed layer is appended ``n_rows`` this step. Returns the pre-step cursor for
        the sidecar log: one int while all layers agree, else a copy of the per-layer array."""
        if self._fr_uniform:
            cur = int(self._fr[0]) if len(self._fr) else 0
        else:
            cur = self._fr.copy()
        self._fr += int(n_rows)
        return cur

    def _cursor_advance(self, appended: Dict[int, int]):
        """Only the layers in ``appended`` (index -> rows) are appended this step (a compacting
        selective step). Returns the pre-step cursor with -1 for every layer not appended."""
        cur = np.full(len(self._fr), -1, dtype=np.int64)
        for i, rows in appended.items():
            cur[i] = self._fr[i]
            self._fr[i] += int(rows)
        self._fr_uniform = bool(len(self._fr) == 0 or self._fr.min() == self._fr.max())
        return cur

    def has_sidecar_entries(self) -> bool:
        """Whether any drained step produced a sidecar entry (``tp_shard.drain_holds_data``)."""
        if self._sidecar is not None:
            return self._sidecar.has_entries()
        return bool(self._steps)

    def write_stats(self) -> dict:
        """Cumulative write-path accounting of this drain (seconds, bytes per mode, last step)."""
        return self._wstats.as_dict()

    def write_path_summary(self) -> str:
        """The install line's body: write mode per tensor kind, writer threads, O_DIRECT block."""
        if self._wp is not None:
            return self._wp.summary()
        if self._write_note:
            return self._write_note
        sink = ("pre-sized MAP_SHARED mmap (MIA_APERTURE_MMAP=1)" if self._mmap_enabled
                else "tobytes + open/append/close per step")
        return (f"write mode=legacy -> hs=legacy ({sink}, on the drain thread) | for A/B "
                f"validation only ({WRITE_MODE_ENV}=legacy)")

    def _selective_active(self) -> bool:
        """Whether THIS drain copies selectively, re-checked at the USE SITE.

        `self.selective` is read live here, not cached — the constructor sets it once, but nothing
        stops a caller (`test_per_request_full_drain_is_enforced_at_the_use_site` deliberately does)
        from force-setting it afterwards, and reading it fresh means that is picked up correctly
        rather than silently ignored. `per_request` is set once, in `__init__`, and nothing in this
        file mutates it later — so re-testing it here is not undoing a possible later change the way
        the `.selective` read is; it is what keeps "selective AND not per_request" a single,
        self-contained rule at the one place every use site calls, instead of splitting it across a
        constructor-time decision and a use-site read. Cheap enough to be unconditional — two
        attribute reads per step."""
        return bool(self.selective) and not bool(getattr(self, "per_request", False))

    def row_counts(self) -> dict:
        """Read-only drain census, reachable across the worker boundary via
        `collective_rpc("get_drain_row_counts")`. `hs.drain.rows_copied` is accumulated where the
        copies are ISSUED, so a selective run that quietly copied everything reads as a full drain
        rather than as a pass; `hs.drain.rows_skipped` is what an unconditional full drain of the
        same steps WOULD have copied, minus that. `selective_disabled_reason` is non-None only when
        the flag was armed and refused (see `_resolve_selective`)."""
        return {
            "hs.drain.rows_copied": int(self._rows_copied),
            "hs.drain.rows_skipped": int(self._rows_skipped),
            "hs.drain.degenerate_steps": int(self._degenerate_steps),
            "selective": bool(self._selective_active()),
            "selective_disabled_reason": self.selective_disabled_reason,
        }

    def record_entries(self, entries: List) -> None:
        # `entries` is per-request ReqCaptureRecord (or already-flat LayerEntry for a direct-drain
        # caller); expand_records fans each record into the flat per-(req, layer) LayerEntry list.
        # Runs on-loop for the sync drain; the off-loop path expands in the consumer thread instead.
        # Non-legacy modes keep the records as they are: the sidecar log derives the entries.
        if self._sidecar is not None:
            self._pending_records.extend(entries)
            return
        self._pending_entries.extend(expand_records(entries))

    def _append_layer_rows(self, ln: int, rows_cpu: torch.Tensor) -> None:
        """Append one layer's already-on-host rows (LOGICAL order) to its raw file — via the
        pre-sized mmap (default) or the plain ``open(ab)+write`` fallback/control. Bumps this
        layer's running file-row cursor (`self._file_rows`) by the rows just appended — the same
        count this call physically wrote, so the cursor always equals this layer's real file length
        in rows."""
        data = _raw_bytes(rows_cpu)
        writer = self._mmap_writers.get(ln) if self._mmap_enabled else None
        if writer is not None:
            writer.append(data)
        else:
            with open(self.raw_paths[ln], "ab") as f:
                f.write(data)
        self._file_rows[ln] = self._file_rows.get(ln, 0) + int(rows_cpu.shape[0])

    def drain_once(self) -> int:
        """Copy the aperture's pending ``[drain, write)`` rows out of every per-layer buffer, append
        them per layer, stamp this step's sidecar entries' `file_row`, queue them, and advance the
        shared drain cursor. Returns rows moved (0 if nothing pending)."""
        if self._wp is not None:
            return self._drain_once_fast()
        moved = self.aperture.pending_rows()
        if moved == 0:
            return 0
        # Physical [s,e) segments in LOGICAL order (segment 0 at the drain cursor's slot; a wrap
        # adds [0, ...)), identical for every layer since they share the cursor. `step_start_logical`
        # is this step's own base -- the drain cursor BEFORE this step's rows are consumed below.
        segments = self.aperture.drained_segments()
        step_start_logical = self.aperture._drain
        cursor_before: Dict[int, int] = {}
        for ln, hs_buf in self.layers:
            pieces = [hs_buf[s:e].detach().to("cpu") for s, e in segments]
            rows_cpu = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
            cursor_before[ln] = self._file_rows.get(ln, 0)
            self._append_layer_rows(ln, rows_cpu)
            # Full drain by construction on this path (see _resolve_selective): every installed
            # layer copies every row, so nothing is ever skipped here.
            self._rows_copied += int(rows_cpu.shape[0])
        if self._pending_entries:
            _stamp_file_row(self._pending_entries, cursor_before, step_start_logical)
            self._steps.append(StepMeta(list(self._pending_entries)))
            self._pending_entries = []
        self.aperture.advance_drain(moved)
        return moved

    def _drain_once_fast(self) -> int:
        """``drain_once`` for the non-legacy modes: the same copies, written zero-copy through the
        run-long sinks (inline, buffered), and the step's records kept in the sidecar log."""
        moved = self.aperture.pending_rows()
        if moved == 0:
            return 0
        t0 = time.perf_counter()
        segments = self.aperture.drained_segments()
        step_start_logical = self.aperture._drain
        wp = self._wp
        by_mode: Dict[str, int] = {}
        with self._io_lock:
            if wp.closed:
                raise ApertureWriteError(
                    f"hs aperture drain ({self.run_dir}): drain_once after close() closed the raw "
                    f"files -- flush_aperture must run after the last step")
            cursor = self._cursor_advance_all(moved)
            t_w = 0.0
            for ln, hs_buf in self.layers:
                pieces = [hs_buf[s:e].detach().to("cpu") for s, e in segments]
                rows_cpu = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
                tw = time.perf_counter()
                sink = wp.sinks[ln]
                n = sink.write(rows_cpu.contiguous())
                t_w += time.perf_counter() - tw
                by_mode[sink.mode] = by_mode.get(sink.mode, 0) + n
                self._rows_copied += int(rows_cpu.shape[0])
            tb = time.perf_counter()
            block = self._sidecar.prepare(self._pending_records, step_start_logical, cursor, None)
            self._pending_records = []
            self._sidecar.commit(block)
            t_b = time.perf_counter() - tb
        self.aperture.advance_drain(moved)
        step_s = time.perf_counter() - t0
        record_step_stats(self._wstats, "hs", rows=moved, step_s=step_s,
                          d2h_s=max(step_s - t_w - t_b, 0.0), write_s=t_w, write_tail_s=0.0,
                          busy_s=t_w, bookkeeping_s=t_b, bytes_by_mode=by_mode, wp=self._wp)
        return moved

    def close(self) -> None:
        """Flush+truncate+release every per-layer mmap writer (msync durability, drop the
        zero-padded pre-sized tail, munmap), then write the shared sidecar (idempotent; safe to
        call from ``flush_aperture`` + atexit)."""
        if self._wp is not None:
            self._close_fast()
            return
        if self._closed:
            return
        for w in self._mmap_writers.values():
            w.close()
        write_sidecar(self.meta_path, self._steps, self.header)
        self._closed = True

    def _close_fast(self) -> None:
        """Non-legacy close: wait for an in-flight step (never close an fd under a write), join the
        writer threads, close every raw file, then write the sidecar from the log. Idempotent. A
        step wedged past ``MIA_APERTURE_DRAIN_JOIN_S`` leaves its fds open (closing them under a
        live write could let the fd number be reused by another file) and is logged; the sidecar
        still records every COMMITTED step."""
        if self._closed:
            return
        _close_write_path(self, "hs")


def _close_write_path(drain, kind: str) -> None:
    """Shared by the HS and QK drains' non-legacy ``close``: take the drain's io lock (so no fd is
    closed under a step's writes), close the write path (joins the writer threads, closes the fds),
    and write the sidecar from the log even if closing an fd failed; that failure is re-raised."""
    join_s = float(os.environ.get("MIA_APERTURE_DRAIN_JOIN_S", "60") or "60")
    got = drain._io_lock.acquire(timeout=join_s)
    err: Optional[BaseException] = None
    try:
        if got:
            try:
                drain._wp.close()
            except BaseException as e:  # noqa: BLE001 -- re-raised after the sidecar is written
                err = e
        else:
            logger.error(
                "%s aperture drain close (%s): a drain step still holds the raw files after %.0f s; "
                "leaving them open and writing the sidecar of the committed steps", kind,
                drain.run_dir, join_s)
        drain._sidecar.write(drain.meta_path, drain.header)
        drain._closed = True
    finally:
        if got:
            drain._io_lock.release()
    if err is not None:
        raise err


_STOP = object()   # queue sentinel: stop the off-loop consumer thread


@dataclass
class _DrainItem:
    """One enqueued step: the sidecar entries, the step's shared-aperture start slot + row count, and a
    CUDA event recorded (on the engine's forward stream) AFTER this step's ``capture_hs`` scatter, so
    the consumer's D2H is ordered after the scatter committed those rows. ``event`` is None on CPU."""
    entries: list
    start_logical: int
    n_rows: int
    event: object = None


@dataclass
class _Finish:
    """A per-request FINISH signal, enqueued onto the SAME consumer queue AFTER that request's
    row-entries (FIFO). The consumer, on dequeuing it, marks the request finished in the
    PerRequestIndex — by then all its rows are drained/noted (the FIFO invariant)."""
    req_id: str


class OffLoopApertureDrain(MultiLayerApertureDrain):
    """Off-loop (consumer-thread) sibling of the SYNCHRONOUS ``MultiLayerApertureDrain``.

    The engine loop, per active step, does an O(1) ``enqueue(entries, start_logical, n_rows, event)``
    and does NOT drain. A dedicated CONSUMER THREAD (per worker process) owns the drain: it waits the
    step's scatter event, reads each per-layer ``hs_buf``'s ``[start_logical, start_logical+n_rows)``
    region on a DEDICATED COPY STREAM (``record_stream`` guards the source rows), writes per-layer raw
    + sidecar, then ``advance_drain(n_rows)`` — which FREES aperture rows and so RELEASES the engine's
    reserve backpressure.

    The consumer reads only COMMITTED ``[drain, write)`` rows (behind the write cursor, disjoint from
    the next step's writes), fenced by the event — never a torn/stale read. The FIFO invariant
    (consumer processes steps in enqueue order, each step's rows begin exactly at the drain cursor)
    is what lets ``_drain_item`` stamp the reader-facing ``file_row`` from each layer's own running
    append cursor, same mechanism as the sync drain.

    SELECTIVE DRAIN (``MIA_DRAIN_SELECTIVE``, default ON): the consumer copies only the
    ``(layer, row-range)`` tiles some request in that step actually asked for, instead of every
    installed layer's whole step span. See ``build_copy_plans`` for the copy list and the
    degenerate-case contract, and ``_resolve_selective`` for the two consumers that FULL-drain
    regardless (per-request delivery and the synchronous drain).

    TWO consumer modes (``per_request``, default OFF, additive):
      * shared-file (default): appends each layer's drained rows to its raw file + sidecar (above).
      * per-request (``per_request=True``): demuxes each step's drained rows BY req_id into a
        ``PerRequestIndex`` (``_demux_into_index``) and consumes ``_Finish`` items
        (``enqueue_finish`` -> ``_handle_finish`` -> ``mark_finished``) to drive per-request
        assembly, writing NO shared file. The same FIFO queue carries both ``_DrainItem`` and
        ``_Finish``, so a request's finish is processed only after all its rows are noted.

    DISK SUB-ROUTE (within per-request mode; INACTIVE unless ``route_to_disk`` is called): a request
    the router marks via ``route_to_disk(req_id, dest)`` at request-start streams its demuxed rows to
    its OWN per-request run_dir (``_PerRequestDiskStaging``) instead of the host-buffer index, and on
    ``_Finish`` the file is msync'd and handed to an ``OffloadProcess`` for transfer to ``dest`` —
    then its staging state is freed (``disk_residency`` -> 0). A per-request run with no disk routes
    is byte-identical to the host-buffer path above.
    """

    # The off-loop drain D2Hs into its own host buffers, allocated at the O_DIRECT alignment.
    _ALLOW_DIRECT = True

    def __init__(self, aperture: CaptureAperture, layers, run_dir: str, header: dict,
                 per_request: bool = False, index: Optional[PerRequestIndex] = None,
                 offload=None, disk_base: Optional[str] = None,
                 shape: Optional[WriteShape] = None):
        # Writer-thread count, validated BEFORE the base class opens any raw file (a bad value must
        # not leave fds behind). Only the shared-file, non-legacy drain has a writer pool.
        _threads = (resolve_write_threads()
                    if not per_request and resolve_write_mode()[0] != "legacy" else 0)
        # per_request (default OFF): when ON the consumer demuxes each step's contiguous drained
        # rows BY req_id into a PerRequestIndex (below) and enqueue_finish() drives assembly, INSTEAD
        # of writing the shared per-layer files (setup_sink follows per_request).
        super().__init__(aperture, layers, run_dir, header, setup_sink=not per_request,
                         shape=shape)
        self.per_request = bool(per_request)
        # Re-resolve selective drain now that per_request is known — this is the ONE drain that has
        # a selective path (the base class resolved for the synchronous drain). per_request
        # FULL-DRAINS with the reason recorded; `install_hs` warns it and `row_counts()` reports it.
        self.selective, self.selective_disabled_reason = _resolve_selective(
            _drain_selective_enabled(), off_loop=True, per_request=self.per_request)
        self.index: Optional[PerRequestIndex] = (
            index if index is not None
            else (PerRequestIndex() if self.per_request else None))
        # DISK ROUTE (a per-request sub-mode, default INACTIVE): a request the router marks via
        # route_to_disk() streams its demuxed rows to its OWN per-request run_dir
        # (_PerRequestDiskStaging) instead of the host-buffer PerRequestIndex, and on finish the file
        # is msync'd + handed to the OffloadProcess for transfer to the client dest. Both maps stay
        # empty until route_to_disk() is called, so a per_request run with no disk routes is
        # byte-identical to the host-buffer path.
        #   * _disk_routed (req_id -> dest): SHARED state — the router thread writes it, the consumer
        #     reads it — so every access is under self._index_lock (below), same discipline as index.
        #   * _disk_staging (req_id -> _PerRequestDiskStaging): consumer-thread-owned; the dict
        #     membership (create on first row / pop on finish) is guarded by _index_lock so the
        #     residency reader can size it, but the row appends run lock-free on the single writer.
        self._offload = offload
        self._disk_base = disk_base or os.path.join(run_dir, "perreq")
        self._disk_routed: Dict[str, str] = {}
        self._disk_staging: Dict[str, _PerRequestDiskStaging] = {}
        # SERVER-side staging source dir per DELIVERED disk-routed request, remembered on finish so
        # the confirm path (or an abort) can unlink it after the client copy lands. The durable
        # CLIENT dest copy is never touched. Guarded by _index_lock like the two maps above.
        self._disk_delivered_src: Dict[str, str] = {}
        # DEFERRED settled-reclaim: delivered-source dirs that clear_request_disk found the offload
        # STILL READING (copytree in flight) when a confirm TIMEOUT dropped the confirm-path unlink.
        # rmtree'ing then would corrupt the client dest and break the offload's retry (source gone),
        # so the source is parked here and reclaimed once the offload SETTLES
        # (_reclaim_settled_pending). EMPTY on the happy path -> a strict no-op. Guarded by
        # _index_lock.
        self._disk_reclaim_pending: Dict[str, str] = {}
        # ABORTED disk requests: EXTERNAL keys the engine-thread abort (clear_request_disk) MARKED
        # but the CONSUMER has not yet reclaimed. SINGLE-OWNER staging-dir lifecycle: the abort MUST
        # NOT rmtree a staging dir the consumer might still be demuxing into -- it only marks here;
        # the CONSUMER thread alone creates, writes, AND deletes a per-request staging dir, so it
        # also owns the discard (on this request's _Finish, or at finalize_all). _disk_write skips a
        # marked req and _handle_finish/finalize_all discard + drop the mark. Guarded by _index_lock
        # like the maps above.
        self._disk_aborted: set = set()
        # HOST-route abort marks: the RPC-path sibling of _disk_aborted. Once a mid-flight HOST
        # request is cleared (clear_aperture_request -> mark_host_aborted), the consumer must NEVER
        # (re-)stage its drained rows into the PerRequestIndex -- a backlogged _DrainItem (rows
        # enqueued before the abort, consumed after the free) would re-note_rows the entry
        # clear_aperture_request just freed, and nothing frees it again; its _Finish would then
        # mark_finished the phantom into _deliverable. A DISK-route abort that popped _disk_routed is
        # skipped directly off _disk_aborted in _demux_into_index instead. Both marks are dropped
        # when the request's _Finish is processed (or at finalize_all). Guarded by _index_lock.
        self._host_aborted: set = set()
        # TP layer shard (set by install_hs at a sharded TP > 1): a disk-routed request that
        # FINISHED on this rank with nothing staged -- it wanted only layers other ranks own -- is
        # remembered here (EXTERNAL key) so confirm_aperture_delivery can answer "not this rank's"
        # (None) instead of "not landed yet" (False) until the deliver timeout. Read (not popped) by
        # every confirm poll -- the driver's poll must see the same answer each time -- and dropped
        # by clear_request_disk, which the serve path runs for every aperture request when it ends.
        # Guarded by _index_lock. Off (and empty) at TP = 1, so that path is unchanged.
        self._note_unstaged_finish = False
        self._disk_unstaged: set = set()
        self._perreq_cap = int(os.environ.get(
            "MIA_APERTURE_PERREQ_MMAP_BYTES", str(64 * 1024 * 1024)) or (64 * 1024 * 1024))
        self._perreq_mmap = os.environ.get("MIA_APERTURE_MMAP", "0") != "0"
        # Sole lock serializing every access to the shared PerRequestIndex: the retrieval thread
        # (get_aperture_per_request, mid-serving) and this consumer thread mutate the SAME index
        # concurrently, and PerRequestIndex has no lock of its own. A plain Lock (not RLock) is
        # correct: no guarded body re-acquires it (each bottoms out in a non-locking self.index.*
        # call, and stop() calls finalize_all() without holding it), and re-entry should deadlock
        # loudly rather than be silently allowed.
        self._index_lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        dev = self.layers[0][1].device if self.layers else torch.device("cpu")
        self._is_cuda = (dev.type == "cuda") and torch.cuda.is_available()
        self._stream = torch.cuda.Stream(dev) if self._is_cuda else None
        self._aperture_depth = max(1, int(os.environ.get("MIA_CAPTURE_DRAIN_APERTURE", "3") or "3"))
        self._copy_events = ([torch.cuda.Event() for _ in range(self._aperture_depth)]
                             if self._stream is not None else [])
        self._aperture_idx = 0
        # Per-layer PERSISTENT pinned host buffer, reused each step (grown on demand). Reuse is safe
        # because the consumer is single-threaded and syncs the copy before writing the file bytes,
        # so a step's bytes are consumed before the next step reuses the buffer — no per-step
        # cudaHostAlloc on the hot path.
        self._pinned: dict = {ln: None for ln, _ in self.layers}
        # Non-legacy write path: per-layer host buffers at the write path's alignment (pinned on
        # CUDA), one completion event per layer so a layer's write starts the moment ITS rows land
        # (the K-deep `_copy_events` set above serves the legacy path only), and the writer pool,
        # whose threads bind this drain's device first.
        self._host: Dict[int, torch.Tensor] = {}
        self._layer_events: list = []
        if self._wp is not None:
            if self._stream is not None:
                self._layer_events = [torch.cuda.Event() for _ in self.layers]
            try:
                self._wp.start_pool(_threads, self._stream.device if self._stream is not None
                                    else None, "mia-hs-aperture-write")
            except BaseException:
                self._wp.close()
                raise
        self._thread = threading.Thread(
            target=self._run, name="mia-hs-aperture-drain", daemon=True)
        self._started = False
        self._error: Optional[BaseException] = None

    # ---- engine side (O(1)) ----
    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def is_alive(self) -> bool:
        return bool(self._started and self._thread.is_alive())

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def enqueue(self, entries: list, start_logical: int, n_rows: int, event=None) -> None:
        """O(1) hand-off. ``entries`` ownership TRANSFERS to the queue item — the caller must not
        mutate it afterward (the routing wrapper reassigns ``registry._hs_step_entries = []``, so the
        old list is owned solely here)."""
        with PROF.timed("graph.enqueue"):   # on-loop enqueue cost (expected O(1)); no-op if PROF off
            self._q.put(_DrainItem(entries, int(start_logical), int(n_rows), event))

    def enqueue_finish(self, req_id) -> None:
        """O(1) hand-off of a per-request FINISH. NO-OP unless per_request mode is on (so the
        shared-file default path never sees a _Finish item). Must be enqueued AFTER the request's
        last row-entries (the caller does this post-``enqueue`` in the execute_model wrapper) so the
        consumer marks it finished only once every row is drained/noted (FIFO invariant)."""
        if not self.per_request:
            return
        self._q.put(_Finish(str(req_id)))

    def route_to_disk(self, req_id, dest, offload=None) -> None:
        """SEAM for the Task-10 router: mark ``req_id`` for the per-request DISK route — its rows
        stream to its own NVMe run_dir and, on finish, the file is offloaded to ``dest`` — INSTEAD
        of the host-buffer PerRequestIndex. NO-OP unless per_request mode is on.

        MUST be called BEFORE the request's rows reach the consumer (the router runs at
        request-start, which precedes the request's first forward/enqueue, so the consumer always
        sees the route before it demuxes the request's rows). Registered under ``_index_lock`` — the
        consumer reads ``_disk_routed`` under the same lock. Lazily starts a thread-backed
        OffloadProcess on first use unless one is injected (constructor or the ``offload`` arg).

        The lazy OffloadProcess is CONSTRUCTED OFF ``_index_lock`` (the opt-in process backend spawns
        a child + threads -- doing it under the lock would stall the consumer, which takes the same
        lock on its hot path), then adopted under the lock only if still absent. A second
        route_to_disk racing the build finds ``self._offload`` already set and DISCARDS its loser
        (closes it), so exactly one is ever adopted -- no double-construct leak."""
        if not self.per_request:
            return
        req_id = str(req_id)
        # Build any lazily-created OffloadProcess OFF the lock (peek is racy-but-safe: a lost race
        # just builds a spare that is closed below). An injected `offload` never needs a build.
        new_offload = None
        if offload is None and self._offload is None:
            from mia.graph.offload_process import OffloadProcess
            use_proc = os.environ.get("MIA_OFFLOAD_PROCESS", "0") == "1"
            new_offload = OffloadProcess(use_process=use_proc)
        with self._index_lock:
            if offload is not None:
                self._offload = offload
            elif self._offload is None and new_offload is not None:
                self._offload = new_offload
                new_offload = None   # adopted -> don't close it below
                # Before multiprocessing terminates an mp-backend child, then at atexit
                # (mia/graph/child_process.py rule 2).
                from mia.graph.child_process import register_shutdown
                register_shutdown(self._offload.close)
            self._disk_routed[req_id] = str(dest)
        if new_offload is not None:
            new_offload.close()      # lost the construct race (another route adopted one) -> discard
        if _aperture_debug():
            _dbg(f"route_to_disk: req_id={req_id!r} (EXTERNAL) dest={dest!r} "
                 f"offload={type(self._offload).__name__}")

    def finished_unstaged(self, req_id) -> bool:
        """True when disk-routed ``req_id`` (EXTERNAL id) finished on this rank with nothing
        staged: under the TP layer shard it asked only for layers other ranks capture. A pure read
        (every confirm poll must get the same answer); ``clear_request_disk`` drops the mark.
        Always False at TP = 1 (nothing is ever marked there)."""
        with self._index_lock:
            return str(req_id) in self._disk_unstaged

    def disk_residency(self) -> int:
        """Number of disk-routed requests still holding per-request staging state (files open /
        pre-finalize). Drops to 0 once every routed request has finished (close+offload+free)."""
        with self._index_lock:
            return len(self._disk_staging)

    def unlink_delivered_source(self, req_id) -> bool:
        """Remove the SERVER-side per-request staging SOURCE dir for a DELIVERED disk-routed request.
        The durable CLIENT dest copy is untouched. Called from the worker's confirm path AFTER the
        offload confirms the file landed at ``dest``. Pops the
        recorded source under ``_index_lock``; the rmtree runs off-lock. Idempotent -> False when
        there is nothing recorded (already unlinked, or the request was never disk-routed)."""
        req_id = str(req_id)
        with self._index_lock:
            src = self._disk_delivered_src.pop(req_id, None)
        if src is None:
            return False
        import shutil
        shutil.rmtree(src, ignore_errors=True)
        return True

    def _reclaim_settled_pending(self) -> None:
        """DEFERRED settled-reclaim: rmtree each parked delivered-source whose offload has now
        SETTLED (done / gave up -> no longer reading the dir). Populated ONLY by
        ``clear_request_disk`` when it found the offload still in flight; EMPTY on the happy path
        -> a strict no-op then. Called opportunistically from the consumer loop and from
        ``finalize_all`` (shutdown). Snapshot under ``_index_lock``, check ``settled()`` + rmtree OFF
        the lock. A source whose offload never settles is LEFT rather than rmtree'd mid-copy -- it
        cannot be safely reclaimed without corrupting the client dest."""
        off = self._offload
        if off is None:
            return
        with self._index_lock:
            pending = list(self._disk_reclaim_pending.items())
        if not pending:
            return
        settled_ids = [ext for ext, _ in pending if off.settled(ext)]
        if not settled_ids:
            return
        to_rm = []
        with self._index_lock:
            for ext in settled_ids:
                src = self._disk_reclaim_pending.pop(ext, None)
                if src is not None:
                    to_rm.append((ext, src))
        import shutil
        for ext, src in to_rm:
            shutil.rmtree(src, ignore_errors=True)
            if _aperture_debug():
                _dbg(f"reclaim settled delivered-src: req={ext!r} src={src!r}")

    def mark_host_aborted(self, req_id) -> None:
        """ABORT cleanup for the HOST (RPC) route: mark ``req_id`` so the consumer never (re-)stages
        its drained rows into the ``PerRequestIndex`` after ``clear_aperture_request`` freed its entry --
        the RPC-path sibling of ``clear_request_disk``'s ``_disk_aborted`` mark. NO-OP unless
        per_request mode is on.

        MARK ONLY A REQUEST WITH LIVE HOST STATE. ``clear_aperture_request`` runs in the serve generate
        ``finally`` for BOTH a genuine mid-flight abort AND normal completion; a completed request
        was already delivered+freed (no live entry), so marking it would leave a STALE mark no
        ``_Finish`` prunes -> unbounded growth over a long serve run. A genuinely-aborting request is
        still generating, so its entry IS live here. The index is keyed by the INTERNAL drain-seen
        id; this abort id is EXTERNAL -> match with the same exact-or-``{ext}-`` rule the disk maps
        use. Set under ``_index_lock``, BEFORE ``clear_aperture_request`` frees the entry, so any note
        racing the free is already suppressed."""
        if not self.per_request or self.index is None:
            return
        req_id = str(req_id)
        with self._index_lock:
            live = any(_match_disk_route(rid, (req_id,)) is not None
                       for rid in self.index.live_req_ids())
            if live:
                self._host_aborted.add(req_id)
        if _aperture_debug():
            _dbg(f"mark_host_aborted: req={req_id!r} live={live} "
                 f"host_aborted={list(self._host_aborted)}")

    def clear_request_disk(self, req_id) -> None:
        """ABORT cleanup for the DISK route: MARK the request aborted; do NOT destroy its live
        staging. SINGLE-OWNER staging-dir lifecycle -- only the CONSUMER thread ever creates,
        writes, or deletes a per-request staging dir, so the engine-thread abort must never rmtree a
        dir the consumer might still be demuxing this request's remaining rows into (rmtree'ing it
        out from under the consumer's ``_disk_write`` raises ``FileNotFoundError`` there and kills
        the consumer, stranding every subsequent per-request delivery).

        Under ``_index_lock``: pop ``_disk_routed`` (so any NEW demux entry for this id resolves to
        host/skip, not disk) and, iff there is live staging or a still-live route, add the EXTERNAL id
        to ``_disk_aborted`` -- the consumer will DISCARD its staging on this request's ``_Finish`` (or
        at ``finalize_all``), dropping ``disk_residency`` then. A recorded ``_disk_delivered_src`` means
        a finish already finalized + SUBMITTED this dir (FIFO: the consumer is provably DONE WRITING it)
        -- but the OFFLOAD thread may still be READING it (copytree) if a confirm TIMEOUT skipped the
        confirm-path unlink, so it is rmtree'd here ONLY once the offload has SETTLED (done/gave-up ->
        no longer reading); if still in flight it is parked in ``_disk_reclaim_pending`` and reclaimed
        by ``_reclaim_settled_pending`` once the offload settles -- never rmtree'd mid-copy, never
        leaked. That is NOT the live staging dir. No-op when it was never disk-routed / already fully
        reclaimed. Strict no-op when per_request is off.

        Pairs with ``_handle_finish``: whichever the finish sees first wins consistently -- if the
        finish already claimed the staging (delivered), this marks nothing and only reclaims the
        recorded source; if not, the mark makes the finish take the DISCARD (abort-reclaim) branch."""
        if not self.per_request:
            return
        req_id = str(req_id)
        marked = False
        with self._index_lock:
            routed = self._disk_routed.pop(req_id, None)
            src = self._disk_delivered_src.pop(req_id, None)
            self._disk_unstaged.discard(req_id)
            if req_id in self._disk_staging or routed is not None:
                # Live staging OR still-routed (rows may be in flight): mark for the consumer to
                # reclaim. Do NOT pop _disk_staging and do NOT rmtree the live dir here.
                self._disk_aborted.add(req_id)
                marked = True
        # Decide the delivered-source's fate OFF the lock (settled() takes the offload's OWN lock; keep
        # _index_lock unheld across it and across rmtree). Only rmtree once the offload has SETTLED, so
        # it is never removed out from under an in-flight copytree; otherwise defer (never leak).
        reclaimed = False
        deferred = False
        if src is not None:
            if self._offload is None or self._offload.settled(req_id):
                import shutil
                shutil.rmtree(src, ignore_errors=True)   # settled offload -> safe to reclaim now
                reclaimed = True
            else:
                with self._index_lock:
                    self._disk_reclaim_pending[req_id] = src   # rmtree once the offload settles
                deferred = True
        if _aperture_debug():
            _dbg(f"clear_request_disk MARK-abort: req={req_id!r} marked={marked} "
                 f"delivered_src_reclaimed={reclaimed} deferred_reclaim={deferred} "
                 f"(consumer owns the staging-dir discard)")

    # ---- consumer thread ----
    def _finalize_finish_isolated(self, req_id) -> None:
        """Run ``_handle_finish`` under PER-REQUEST FINALIZE ISOLATION.

        A finalize error (a partially-staged aborted disk request whose ``close()``/offload raises,
        an ``OffloadProcess.submit`` double-submit, a marshal error) must fail for THAT request ONLY:
        caught, logged loud with the req_id, and the caller CONTINUES to the next item. The off-loop
        consumer thread is NEVER killed by one request's finalize; every other request still drains
        + finalizes + delivers.

        CONTRAST the DRAIN path (``_drain_item``): a drain error never advances the aperture cursor, so
        the never-drop / backpressure guarantee must fail LOUD, not be swallowed — drain errors stay
        FATAL and propagate to ``_run``'s outer handler. Only per-request FINALIZE is isolated here."""
        try:
            self._handle_finish(req_id)
        except Exception:  # noqa: BLE001 -- isolate ONE request's finalize; never wedge the consumer
            logger.exception(
                "hs off-loop aperture drain: per-request FINALIZE failed for req_id=%r; that request's "
                "delivery is dropped, the consumer continues", req_id)
            if _aperture_debug():
                _dbg(f"finish FAILED (isolated, consumer continues): req_id={req_id!r}")

    def _run(self) -> None:
        try:
            # FIRST, before any CUDA call: select the device the copy stream lives on. A new
            # thread's current device is cuda:0; on any TP rank r >= 1 that GPU belongs to another
            # process, and the stream context's exit restored a device-0 stream -- the crash G1 hit
            # in the QK drain (mia/graph/thread_device.py). Live for HS too since the TP layer
            # shard (MIA_HS_TP_SHARD): every HS rank runs this drain on its own GPU. A failure is
            # recorded like any consumer death (loud, below).
            bind_thread_to_device(self._stream.device if self._stream is not None else None)
            while True:
                item = self._q.get()
                if item is _STOP:
                    self._q.task_done()
                    break
                try:
                    if isinstance(item, _Finish):
                        # PER-REQUEST FINALIZE ISOLATION (never wedges the consumer) — see
                        # _finalize_finish_isolated. A finalize raise fails for THAT request only.
                        self._finalize_finish_isolated(item.req_id)
                    else:
                        # DRAIN stays fatal: a failure here never advances the aperture cursor, so the
                        # never-drop guarantee must fail LOUD via the outer handler / backpressure.
                        self._drain_item(item)
                    # Deferred settled-reclaim of any delivered-source parked by a confirm-timeout
                    # abort (rmtree-vs-offload-read race fix). No-op when nothing is pending (the
                    # happy path). Never raises (rmtree ignore_errors); does not gate the drain.
                    self._reclaim_settled_pending()
                finally:
                    self._q.task_done()
        except BaseException as e:  # noqa: BLE001 — surface + let backpressure fail loud
            self._error = e
            logger.exception("hs off-loop aperture drain consumer thread died")
            # The thread exits; the engine's reserve-block liveness check (is_alive()) trips and
            # raises ApertureBackpressureError, so a dead consumer fails LOUD (never a silent drop).

    def _read_segments(self, plans: Dict[int, LayerCopyPlan],
                       event) -> List[Tuple[int, torch.Tensor]]:
        """D2H each layer's WANTED ``hs_buf`` segments into a contiguous (LOGICAL-order) host buffer.

        PER LAYER, not per step: ``plans[ln]`` says which physical segments that layer needs and how
        many rows they total, so the pinned buffer is sized to THAT layer's wanted rows and a layer
        absent from ``plans`` is not copied at all. ``build_copy_plans(selective=False)`` hands every
        installed layer the same whole-step plan — "the flag is off" and "everyone wants everything"
        are the same code path, not two.

        The caller owns the plans (it built them) and therefore owns the per-layer row mapping the
        append path + ``_stamp_file_row`` need to place rows — that is why the mapping is passed IN
        rather than returned: one object, one owner, no chance of the copy and the bookkeeping
        disagreeing about which rows moved.

        cuda: on the dedicated copy stream — wait the K-stale event, wait the step's scatter
        ``event`` (order the copies AFTER the scatter), copy each layer's segments into its reused
        pinned buffer with ``record_stream`` on the source, then wait THIS step's completion event
        before returning (so the file write reads materialized bytes). Copying FEWER segments must
        never mean copying them UNORDERED: every wait/record below is unchanged by the lever, only
        the segment list narrowed. ``event.synchronize`` / the copy wait RELEASE the GIL, so the
        engine thread runs the next forward meanwhile — the off-loop overlap.
        cpu (tests): plain ``.to('cpu')`` (event is a no-op stub)."""
        if self._stream is not None:
            slot = self._aperture_idx
            stale = self._copy_events[slot] if slot < len(self._copy_events) else None
            if stale is not None:
                stale.synchronize()                     # K-stale copy done → its pinned buf reusable
            if event is not None:
                self._stream.wait_event(event)          # order copies AFTER this step's scatter
            pieces: List[Tuple[int, torch.Tensor]] = []
            with torch.cuda.stream(self._stream):
                for ln, hs_buf in self.layers:
                    plan = plans.get(ln)
                    if plan is None or plan.total_rows == 0:
                        continue                        # nobody wants this layer this step
                    buf = self._pinned_buf(ln, plan.total_rows, hs_buf.shape[1], hs_buf.dtype)
                    off = 0
                    for s, e in plan.segments:
                        src = hs_buf[s:e]
                        buf[off:off + (e - s)].copy_(src, non_blocking=True)
                        src.record_stream(self._stream)  # allocator can't recycle the source mid-copy
                        off += (e - s)
                        self._rows_copied += (e - s)     # counted where the copy is ISSUED
                    pieces.append((ln, buf))
            done = self._copy_events[slot] if slot < len(self._copy_events) else None
            if done is not None:
                done.record(self._stream)
                done.synchronize()                       # copies landed → safe to read on the host
            else:
                self._stream.synchronize()
            self._aperture_idx = (slot + 1) % self._aperture_depth
            return pieces
        # CPU path
        if event is not None:
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001 — CPU stub events
                pass
        out: List[Tuple[int, torch.Tensor]] = []
        for ln, hs_buf in self.layers:
            plan = plans.get(ln)
            if plan is None or plan.total_rows == 0:
                continue
            parts = []
            for s, e in plan.segments:
                parts.append(hs_buf[s:e].detach().to("cpu"))
                self._rows_copied += (e - s)
            rows = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            out.append((ln, rows))
        return out

    def _pinned_buf(self, ln: int, n_rows: int, hidden: int, dtype) -> torch.Tensor:
        buf = self._pinned[ln]
        if buf is None or buf.shape[0] < n_rows:
            buf = torch.empty(n_rows, hidden, dtype=dtype, pin_memory=self._is_cuda)
            self._pinned[ln] = buf
        return buf[:n_rows]

    def _drain_item(self, item: _DrainItem) -> None:
        if self._wp is not None:
            self._drain_item_fast(item)
        else:
            self._drain_item_legacy(item)

    def _host_buf(self, ln: int, n_rows: int, width: int, dtype) -> torch.Tensor:
        """This layer's reused host buffer (grown on demand), aligned for O_DIRECT. Reuse is safe:
        a step joins every write that reads it before the next step's D2H lands in it."""
        buf = self._host.get(ln)
        if buf is None or buf.shape[0] < n_rows:
            buf = alloc_host_rows(n_rows, width, dtype, pinned=self._is_cuda,
                                  align=self._wp.mem_align)
            self._host[ln] = buf
        return buf[:n_rows]

    def _issue_d2h(self, plans: Dict[int, LayerCopyPlan], event) -> list:
        """Issue every wanted layer's D2H into its host buffer, one completion event per layer.
        Returns ``[(ln, layer index, host rows, event or None)]`` in layer order.

        cuda: the copy stream waits the step's scatter ``event`` once, then each layer's segments
        are copied (``record_stream`` on the source, as the legacy path does) and that layer's event
        recorded -- the caller waits them one at a time, handing each layer to a writer as soon as
        its rows are on the host, while later layers are still copying. cpu (tests): synchronous
        copies into the same host buffers, no events."""
        jobs = []
        if self._stream is not None:
            if event is not None:
                self._stream.wait_event(event)          # order copies AFTER this step's scatter
            with torch.cuda.stream(self._stream):
                for i, (ln, hs_buf) in enumerate(self.layers):
                    plan = plans.get(ln)
                    if plan is None or plan.total_rows == 0:
                        continue                        # nobody wants this layer this step
                    buf = self._host_buf(ln, plan.total_rows, hs_buf.shape[1], hs_buf.dtype)
                    off = 0
                    for s, e in plan.segments:
                        src = hs_buf[s:e]
                        buf[off:off + (e - s)].copy_(src, non_blocking=True)
                        src.record_stream(self._stream)
                        off += (e - s)
                        self._rows_copied += (e - s)     # counted where the copy is ISSUED
                    ev = self._layer_events[i]
                    ev.record(self._stream)
                    jobs.append((ln, i, buf, ev))
            return jobs
        if event is not None:
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001 — CPU stub events
                pass
        for i, (ln, hs_buf) in enumerate(self.layers):
            plan = plans.get(ln)
            if plan is None or plan.total_rows == 0:
                continue
            buf = self._host_buf(ln, plan.total_rows, hs_buf.shape[1], hs_buf.dtype)
            off = 0
            for s, e in plan.segments:
                buf[off:off + (e - s)].copy_(hs_buf[s:e])
                off += (e - s)
                self._rows_copied += (e - s)
            jobs.append((ln, i, buf, None))
        return jobs

    def _drain_item_fast(self, item: _DrainItem) -> None:
        """One step on the non-legacy write path. Same copy list, same accounting, same FIFO and
        release rules as ``_drain_item_legacy``; what differs is HOW the rows reach the files:

          1. D2H every wanted layer on the copy stream, one event per layer;
          2. as each layer's event completes, submit its write (zero-copy, to its run-long sink) to
             the writer pool -- later layers keep copying meanwhile;
          3. with the writes in flight, build the step's sidecar block (arrays, no LayerEntry);
          4. JOIN every write (a failure re-raises here, after all of them finished), commit the
             sidecar block, and only then ``advance_drain`` -- rows are freed after their bytes are
             written, exactly as before.

        Holds ``_io_lock`` for the whole write phase so ``close()`` can never close an fd under a
        write."""
        t0 = time.perf_counter()
        aperture = self.aperture
        assert item.start_logical == aperture._drain, (
            f"FIFO drain violation: item.start_logical={item.start_logical} != "
            f"aperture._drain={aperture._drain}")
        selective = self._selective_active()
        compacting = selective and not is_degenerate_full_step(
            item.entries, self._layer_set, item.start_logical, item.n_rows)
        if selective and not compacting:
            self._degenerate_steps += 1
        plans = build_copy_plans(item.entries, self._layer_nums, aperture,
                                 item.start_logical, item.n_rows, selective=compacting)
        t_plans = time.perf_counter() - t0
        wp = self._wp
        with self._io_lock:
            if wp.closed:
                raise ApertureWriteError(
                    f"hs aperture drain ({self.run_dir}): a step reached the consumer after close() "
                    f"closed the raw files -- flush_aperture (stop, then close) must follow the last "
                    f"step; this step's rows are NOT written and the aperture is not advanced")
            t1 = time.perf_counter()
            copied_before = self._rows_copied
            jobs = self._issue_d2h(plans, item.event)
            copied_here = self._rows_copied - copied_before
            full_here = len(self.layers) * int(item.n_rows)
            assert copied_here <= full_here, (
                f"selective drain copied {copied_here} rows for a step that owns at most "
                f"{full_here} ({len(self.layers)} layers x {item.n_rows} rows) -- the copy list "
                f"escaped the step")
            self._rows_skipped += full_here - copied_here
            # File-row cursor BEFORE this step (what the reader's `file_row` is stamped from). A
            # compacting step appends only the layers it copied; otherwise every installed layer
            # gets the whole span.
            if compacting:
                cursor = self._cursor_advance({i: int(buf.shape[0]) for _, i, buf, _ in jobs})
            else:
                cursor = self._cursor_advance_all(int(item.n_rows))
            futs = []
            t_first = None
            try:
                for ln, _i, buf, ev in jobs:
                    if ev is not None:
                        ev.synchronize()                # this layer's rows landed (GIL released)
                    if t_first is None:
                        t_first = time.perf_counter()
                    futs.append(wp.pool.submit(timed_write, wp.sinks[ln], buf))
                t2 = time.perf_counter()                # every layer's D2H has landed
                block = self._sidecar.prepare(item.entries, item.start_logical, cursor,
                                              plans if compacting else None)
            except BaseException:
                join_writes_quietly(futs)               # no write may outlive the failing step
                raise
            t3 = time.perf_counter()
            results = join_writes(futs)                 # raises the first write failure
            t4 = time.perf_counter()
            self._sidecar.commit(block)
        # Free the rows LAST -- only after every byte of this step is written.
        aperture.advance_drain(item.n_rows)
        by_mode: Dict[str, int] = {}
        busy = 0.0
        for n, secs, mode in results:
            by_mode[mode] = by_mode.get(mode, 0) + n
            busy += secs
        record_step_stats(
            self._wstats, "hs", rows=item.n_rows, step_s=time.perf_counter() - t0,
            d2h_s=t2 - t1, write_s=(t4 - t_first) if t_first is not None else 0.0,
            write_tail_s=t4 - t3, busy_s=busy, bookkeeping_s=t_plans + (t3 - t2),
            bytes_by_mode=by_mode, wp=self._wp)

    def _drain_item_legacy(self, item: _DrainItem) -> None:
        """The pre-existing drain step, kept verbatim as ``MIA_APERTURE_WRITE_MODE=legacy`` (A/B
        validation only) and for per-request delivery; only the step accounting was added."""
        t0 = time.perf_counter()
        aperture = self.aperture
        # FIFO invariant: the consumer drains steps in enqueue order, so this step's rows begin
        # EXACTLY at the current drain cursor. Assert it — the `file_row` the reader keys on is
        # stamped as "this layer's append cursor + this row's offset WITHIN this step", which is only
        # the true file position if the step really starts where the drain cursor sits.
        assert item.start_logical == aperture._drain, (
            f"FIFO drain violation: item.start_logical={item.start_logical} != "
            f"aperture._drain={aperture._drain}")
        # THE COPY LIST. Selective: per layer, only the row ranges some request in THIS step named.
        # Full (the default, and always when per_request is on): the whole step span for every
        # installed layer -- the identical `aperture.segments_at(start_logical, n_rows)` list this method
        # used to compute inline, now expressed as one shared plan.
        selective = self._selective_active()
        # THE DEGENERATE FAST PATH: "everyone wants everything this step" is decided ONCE here, in
        # one O(records) walk, so both `build_copy_plans` and `_stamp_file_row` below take their
        # cheap flag-off branch instead of the general per-layer/per-entry one. `compacting` (not
        # `selective`) is what the two consumers below key on: False both when the lever is off and
        # when it is armed with nothing to skip -- the same statement about the copy list either way.
        # `row_counts()["selective"]` still reads the ARMED state regardless.
        compacting = selective and not is_degenerate_full_step(
            item.entries, self._layer_set, item.start_logical, item.n_rows)
        if selective and not compacting:
            self._degenerate_steps += 1
        plans = build_copy_plans(item.entries, self._layer_nums, aperture,
                                 item.start_logical, item.n_rows, selective=compacting)
        copied_before = self._rows_copied
        t1 = time.perf_counter()
        with PROF.timed("bank.consumer.d2h"):
            pieces = self._read_segments(plans, item.event)
        t2 = time.perf_counter()
        t3 = t2
        nbytes = 0
        # Rows an unconditional FULL drain of this step would have copied, minus what was actually
        # copied (measured above, at the copy site). 0 on the full-drain path by construction. The
        # real enforcement is `build_copy_plans`' `ValueError`, which rejects any per-layer range
        # reaching outside the step BEFORE a single row is copied, so `pieces` can never in fact hold
        # more than `full_here` rows; this assert documents that invariant at the point it is
        # consumed and is stripped under `python -O`, same as every other assert in this file.
        copied_here = self._rows_copied - copied_before
        full_here = len(self.layers) * int(item.n_rows)
        assert copied_here <= full_here, (
            f"selective drain copied {copied_here} rows for a step that owns at most {full_here} "
            f"({len(self.layers)} layers x {item.n_rows} rows) -- the copy list escaped the step")
        self._rows_skipped += full_here - copied_here
        if self.per_request:
            # Per-request delivery: split this step's rows by req_id into the PerRequestIndex.
            self._demux_into_index(item, pieces)
        else:
            # Shared-file path (default): append each COPIED layer's rows to its raw file, tracking
            # that layer's pre-append file-row cursor for the stamping below. Under a selective drain
            # `pieces` omits the layers nobody wanted, so their cursors (and files) simply do not
            # move this step — which is precisely the divergence `file_row` exists for.
            cursor_before: Dict[int, int] = {}
            for ln, rows_cpu in pieces:
                cursor_before[ln] = self._file_rows.get(ln, 0)
                self._append_layer_rows(ln, rows_cpu)
                nbytes += int(rows_cpu.numel()) * int(rows_cpu.element_size())
            t3 = time.perf_counter()
            # Expand this step's per-request records into the flat per-(req, layer) LayerEntry list
            # off the engine loop (here, on the consumer thread) — same fields/order the on-loop
            # fan-out produces, so StepMeta / sidecar bytes stay byte-identical.
            entries = expand_records(item.entries)
            if entries:
                # Stamp file_row from the real per-layer cursor -- item.start_logical is this step's
                # own base, identical to the FIFO-asserted aperture._drain above. When the copy
                # COMPACTED, the within-step offset is the row's position in that layer's compacted
                # copy, so the plans go along; otherwise `plans=None` keeps the full-drain arithmetic,
                # which on a whole-span plan is the same number `row_offset` would return.
                _stamp_file_row(entries, cursor_before, item.start_logical,
                                plans=plans if compacting else None)
                self._steps.append(StepMeta(entries))
        # Free the rows LAST — only after the D2H landed AND the rows were consumed (written to the
        # file / cloned into the index), so the engine can never scatter into a physical slot the
        # consumer is still reading (never-drop + no torn read; backpressure holds until this advance).
        aperture.advance_drain(item.n_rows)
        t4 = time.perf_counter()
        record_step_stats(
            self._wstats, "hs", rows=item.n_rows, step_s=t4 - t0, d2h_s=t2 - t1,
            write_s=t3 - t2, write_tail_s=t3 - t2, busy_s=t3 - t2,
            bookkeeping_s=(t1 - t0) + (t4 - t3), bytes_by_mode={"legacy": nbytes} if nbytes else {})

    def _demux_into_index(self, item: _DrainItem, pieces: List[Tuple[int, torch.Tensor]]) -> None:
        """Slice each layer's contiguous drained host rows by each ``LayerEntry``'s req_id range and
        stage them in the ``PerRequestIndex``.

        ``pieces`` are ``(layer, rows)`` where ``rows`` holds this step's ``[start_logical,
        start_logical+n_rows)`` region in LOGICAL order (segments already concatenated by
        ``_read_segments``), so row ``k`` == logical row ``item.start_logical + k``. THAT DENSE STEP
        IMAGE IS WHY per-request mode always FULL-drains: selective drain compacts each layer's rows
        to only what some request wanted, which would invalidate the ``off`` arithmetic below --
        ``_resolve_selective`` refuses the combination at construction and ``_selective_active``
        re-checks it at the use site, so ``pieces`` here is always dense. An entry for ``(req,
        layer)`` occupies logical rows ``[entry.logical_start, entry.logical_start+n_rows)`` -> host
        offset ``entry.logical_start - item.start_logical``. Host-buffer slices are CLONED: the
        source is a reused pinned buffer (CUDA) or a aperture view (CPU) that ``advance_drain`` lets the
        engine overwrite, so a stored view would later be corrupted. DISK-routed slices are written
        to the request's per-request file synchronously here (a zero-copy ``pwrite``, or the
        ``_raw_bytes`` copy under ``legacy``), consuming the view before ``advance_drain`` — so
        they need no clone and never enter the host index."""
        by_layer = {ln: rows for ln, rows in pieces}
        base = int(item.start_logical)
        # Expand this step's per-request records into the flat per-(req, layer) LayerEntry list off
        # the engine loop (here, on the consumer thread) — same fields/order the on-loop fan-out
        # produces. Heterogeneous per-request layer sets are preserved: each record carries its own
        # `layers`, so each entry's (req_id, layer) range is that request's own.
        entries = expand_records(item.entries)
        with self._index_lock:
            routed_keys = tuple(self._disk_routed) if self._disk_routed else ()
        any_disk = bool(routed_keys)
        # Clone each host-buffer slice OFF the lock (the torch copy), then note under the lock. The
        # lock guards ONLY the direct self.index mutations — never the tensor work / D2H / disk
        # writes — so the critical section stays short and never overlaps the retrieval thread.
        staged = []
        for e in entries:
            layer_rows = by_layer.get(e.layer)
            if layer_rows is None:
                continue          # entry's layer not among the drained layers (should not happen)
            off = int(e.logical_start) - base
            sl = layer_rows[off:off + int(e.n_rows)]
            # e.req_id is the INTERNAL '{external}-{rand}' under serve; the disk routes are keyed by
            # the EXTERNAL id -> match with the same exact-or-'{ext}-' rule the RPC path uses, then
            # STAGE keyed by the resolved external id so finish/confirm/abort/unlink all agree.
            ext = _match_disk_route(e.req_id, routed_keys) if any_disk else None
            if ext is not None:
                if _aperture_debug():
                    _dbg(f"demux DISK hit: entry.req_id={e.req_id!r} -> route={ext!r} "
                         f"layer={e.layer} n_rows={int(e.n_rows)}")
                # PER-ENTRY DISK ISOLATION: a single disk request's staging write must NEVER wedge
                # the whole consumer -- catch it here (not around the whole _drain_item), log LOUD,
                # mark the request aborted so its remaining rows are skipped and its dir reclaimed,
                # and CONTINUE so every other request in this step is unaffected. HOST rows keep
                # their never-drop guarantee (cloned/noted below regardless); only a per-request
                # STAGING error is isolated here, a host clone/note error stays fatal.
                try:
                    self._disk_write(ext, e.layer, sl, int(e.n_rows), e.hs_mode)
                except Exception:  # noqa: BLE001 -- isolate ONE disk request; never wedge the consumer
                    logger.exception(
                        "hs off-loop aperture drain: per-request DISK demux write failed for req=%r "
                        "layer=%s; that request's disk delivery is dropped + its staging reclaimed, "
                        "the consumer continues", ext, e.layer)
                    with self._index_lock:
                        if ext in self._disk_staging or ext in self._disk_routed:
                            self._disk_aborted.add(ext)
                    if _aperture_debug():
                        _dbg(f"demux DISK write FAILED (isolated): req={ext!r} layer={e.layer}")
            else:
                if any_disk and _aperture_debug():
                    _dbg(f"demux DISK miss: entry.req_id={e.req_id!r} not in routes "
                         f"{list(routed_keys)} -> host index")
                staged.append((e.req_id, e.layer, sl.clone()))
        with self._index_lock:
            # ABORT SKIP, re-checked HERE so it is atomic with the note: a request aborted after its
            # rows were drained -- HOST route (clear_aperture_request -> mark_host_aborted) or DISK route
            # whose _disk_routed was popped (clear_request_disk -> _disk_aborted, so its post-pop
            # rows fell through to `staged`) -- must NOT (re-)create a host entry slot that nothing
            # will ever free. The abort marks are added under this same lock, so either this note
            # sees the mark (skip) or it precedes the abort's free (which then removes the just-noted
            # entry) -- never a stranded slot.
            ab_host = tuple(self._host_aborted) if self._host_aborted else ()
            ab_disk = tuple(self._disk_aborted) if self._disk_aborted else ()
            for req_id, layer, rows_slice in staged:
                if ((ab_host and _match_disk_route(req_id, ab_host) is not None)
                        or (ab_disk and _match_disk_route(req_id, ab_disk) is not None)):
                    if _aperture_debug():
                        _dbg(f"demux HOST-SKIP aborted: req={req_id!r} layer={layer}")
                    continue
                self.index.note_rows(req_id, layer, rows_slice)

    def _disk_write(self, req_id, layer, rows_cpu, n_rows: int, mode: str) -> None:
        """Append a disk-routed request's step rows to its per-request file (creating its staging on
        the first row). The dict membership is guarded by ``_index_lock`` (so ``disk_residency`` can
        size it); the file write runs lock-free on the single consumer-thread writer.

        SKIP GUARD: re-check under the lock, BEFORE creating or appending staging, that this request
        is neither ABORTED nor un-routed. ``_demux_into_index`` matched it against a ``routed_keys``
        snapshot taken BEFORE it took the lock, so a concurrent abort (``clear_request_disk`` pops
        ``_disk_routed`` + marks ``_disk_aborted``) could land in that window. Skipping here means the
        consumer NEVER creates/opens a layer file inside a dir the abort slated for discard (which
        would otherwise raise ``FileNotFoundError`` and kill the consumer) and never re-creates the
        staging of a discarded request."""
        with self._index_lock:
            if req_id in self._disk_aborted or req_id not in self._disk_routed:
                if _aperture_debug():
                    _dbg(f"disk_write SKIP (aborted/unrouted): req={req_id!r} layer={layer} "
                         f"n_rows={n_rows}")
                return
            stg = self._disk_staging.get(req_id)
            if stg is None:
                stg = _PerRequestDiskStaging(
                    req_id, os.path.join(self._disk_base, _sanitize_req_id(req_id)),
                    self.header, self._perreq_cap, self._perreq_mmap,
                    write_mode=self._perreq_write_mode)
                self._disk_staging[req_id] = stg
        stg.append(layer, rows_cpu, n_rows, mode)

    def _handle_finish(self, req_id) -> None:
        """Finish a request: for a DISK-routed request finalize its per-request file (msync + write
        its sidecar) and hand it to the OffloadProcess for transfer to the client dest, then free
        its staging state (residency -> 0); for a host-buffer request mark it finished in the
        PerRequestIndex. FIFO: this ``_Finish`` trails all of the request's ``_DrainItem``s, so its
        rows are already drained/noted. Only touch a request the drain actually saw —
        ``finished_req_ids`` also lists non-capturing requests, and acting on one would create a
        spurious empty deliverable.

        SINGLE-OWNER ABORT RECLAIM: the CONSUMER thread owns the discard of an aborted disk request's
        staging dir -- never the engine/abort thread (which only MARKS in ``_disk_aborted``). If this
        finish id resolves to an aborted external key, CLAIM + DISCARD its staging here (close fds +
        rmtree the dir, on THIS consumer thread) and drop all its maps; an aborted request is never
        delivered. FIFO guarantees no ``_disk_write`` for the request runs after this ``_Finish``
        (its rows all trailed it), so the discard can never be resurrected."""
        req_id = str(req_id)
        # Claim ownership atomically under the FIRST lock. The ABORT-RECLAIM check runs first: an
        # aborted id is claimed + its maps dropped here, its staging discarded OFF-lock below. For a
        # normal finish: pop BOTH route maps AND record the delivered-source in this SAME section --
        # not get-then-later-pop. A concurrent abort must always see EITHER this live staging OR the
        # recorded delivered_src, never neither.
        with self._index_lock:
            # The finish id is the INTERNAL '{external}-{rand}' (from finished_req_ids) under serve;
            # the disk maps are keyed by the EXTERNAL id (route_to_disk). Resolve to the route/abort
            # key so a serve request's finish reaches its staging (finalize_all passes the external
            # staging key, which resolves to itself). ext=None -> not disk-routed -> host path below.
            aborted_ext = (_match_disk_route(req_id, tuple(self._disk_aborted))
                           if self._disk_aborted else None)
            if aborted_ext is not None:
                # ABORTED: this consumer owns the discard. Drop every map entry now (residency falls
                # once staging is popped), rmtree the dir OFF-lock below. Never deliver.
                self._disk_aborted.discard(aborted_ext)
                self._host_aborted.discard(aborted_ext)   # defensive: keep the host mark bounded too
                self._disk_routed.pop(aborted_ext, None)
                self._disk_delivered_src.pop(aborted_ext, None)
                stg_abort = self._disk_staging.pop(aborted_ext, None)
                ext = None
                disk_dest = None
                stg = None
            else:
                stg_abort = None
                ext = _match_disk_route(req_id, tuple(self._disk_routed))
                disk_dest = self._disk_routed.pop(ext, None) if ext is not None else None
                stg = self._disk_staging.pop(ext, None) if disk_dest is not None else None
                if stg is not None and self._offload is not None:
                    self._disk_delivered_src[ext] = stg.run_dir
                if disk_dest is not None and stg is None and self._note_unstaged_finish:
                    self._disk_unstaged.add(ext)   # routed here, but none of its layers are ours
        if aborted_ext is not None:
            # OFF-lock (never hold _index_lock across close/rmtree): discard the aborted staging dir.
            if stg_abort is not None:
                stg_abort.discard()          # close fds + rmtree the source (no offload, no sidecar)
            if _aperture_debug():
                _dbg(f"finish ABORT-reclaim: id={req_id!r} route={aborted_ext!r} "
                     f"discarded={stg_abort is not None} (single-owner consumer discard)")
            return
        if disk_dest is not None:
            if _aperture_debug():
                _dbg(f"finish DISK: id={req_id!r} route={ext!r} "
                     f"run_dir={(stg.run_dir if stg else None)!r} dest={disk_dest!r} "
                     f"submit={stg is not None and self._offload is not None}")
            # DISK route. close()/submit() run under the consumer's PER-REQUEST FINALIZE ISOLATION
            # (_finalize_finish_isolated / finalize_all): a raise here fails THIS request's delivery
            # only, never the pipeline. close() tolerates a partial/aborted dir (see
            # _PerRequestDiskStaging.close). The staging + any recorded delivered_src are already
            # claimed here, so a failure never leaves a half-owned entry for the abort path; and
            # close()/submit() run exactly once, since only this method popped stg.
            if stg is not None:
                stg.close()                     # msync + per-request sidecar (single writer, off-lock)
                if self._offload is not None:
                    # Submit + confirm both key on the EXTERNAL id (ext): confirm_aperture_delivery ->
                    # offload.wait(external), so the offload job MUST be submitted under external too.
                    self._offload.submit(ext, stg.run_dir, disk_dest)  # non-blocking, never-drop
            return
        if self.index is None:
            return
        with self._index_lock:
            # ABORTED HOST request: its _Finish (finished_req_ids includes aborts) resolves to a
            # mark_host_aborted id -> NEVER deliver it. Drop any entry the abort left (defensive --
            # the demux skip usually prevented one) and discard the mark so _host_aborted stays
            # bounded (this finish is the request's terminal event). Only then does a normal finish
            # mark_finished.
            host_ab = (_match_disk_route(req_id, tuple(self._host_aborted))
                       if self._host_aborted else None)
            if host_ab is not None:
                self.index.free(req_id)
                self._host_aborted.discard(host_ab)
                if _aperture_debug():
                    _dbg(f"finish HOST-abort drop: id={req_id!r} mark={host_ab!r}")
                return
            if req_id in self.index.live_req_ids():
                self.index.mark_finished(req_id)

    # ---- shutdown / flush ----
    def finalize_all(self) -> None:
        """END-OF-RUN ONLY: mark every still-live per-request request finished so it becomes
        deliverable via ``pop_deliverable``.

        Closes the last-step straggler gap: a request that finishes on the FINAL executed step
        never gets its ``_Finish`` enqueued (``finished_req_ids`` is reported the step AFTER a
        request's last rows — a step vLLM may never run when the last request, or a whole batch,
        finishes on the same step), so the streaming path would silently drop it even though all of
        its rows are already drained/noted in the index.

        MUST run only at genuine end-of-run, NEVER mid-serving: an in-flight request is legitimately
        live and marking it finished early would deliver PARTIAL rows. The sole caller is ``stop()``
        (reached only via the worker's ``flush_aperture`` — the once-after-all-requests-finish
        durable-flush RPC; aperture-path serve retrieval reads durable files and never touches the
        drain), so this can never fire while a request is still generating.

        Marking all live requests finished also delivers any request ABORTED mid-capture (its
        partial rows) — a conscious choice, acceptable at genuine shutdown. STRICT NO-OP when
        per_request is off (index is None) → the shared-file default path is byte-identical.

        DISK-route stragglers: a disk-routed request that finished on the final step likewise never
        gets its ``_Finish``, so it is still holding open per-request staging. Finalize
        each (close + offload + free) via the SAME ``_handle_finish`` path — so its file is delivered,
        not orphaned. Runs first, before the host-index sweep. An ABORTED disk request still staged at
        shutdown (its ``_Finish`` never arrived) is reclaimed the same way: ``_handle_finish`` takes
        the single-owner discard branch on the ``_disk_aborted`` mark. A routed-but-un-staged aborted
        id (marked, no rows) is swept by the explicit ``_disk_aborted`` pass below so its mark never
        leaks."""
        # Disk stragglers: finalize (close+submit+free) any still-staged disk-routed request. Runs
        # on the collector thread only after the consumer is joined (see stop()), so it cannot race
        # the consumer's staging writes. _handle_finish acquires _index_lock itself, so iterate over
        # a snapshot WITHOUT holding the lock (no re-entrant acquire).
        with self._index_lock:
            disk_pending = list(self._disk_staging.keys())
        for req_id in disk_pending:
            # Per-request finalize isolation here too: one straggler whose close()/offload raises
            # (e.g. an aborted-mid-capture request finalized at shutdown) must not abort the sweep of
            # the rest, nor the host-index sweep below. An aborted staging here takes _handle_finish's
            # single-owner DISCARD branch (never delivered).
            self._finalize_finish_isolated(req_id)
        # Sweep any still-marked aborted ids whose staging is already gone (routed-but-un-staged, or
        # already reclaimed): _handle_finish discards the mark. Idempotent vs the loop above.
        with self._index_lock:
            aborted_pending = list(self._disk_aborted)
        for ab_id in aborted_pending:
            self._finalize_finish_isolated(ab_id)
        # End-of-run settled-reclaim: rmtree any delivered-source a confirm-timeout abort parked whose
        # offload has since settled (no-op when nothing is pending). A source whose offload is still
        # in flight at genuine end-of-run is left for the offload to finish -- never rmtree'd mid-copy.
        self._reclaim_settled_pending()
        if self.index is None:
            return
        # Runs on the collector thread only after the consumer is provably joined (see stop()), so
        # it can never race the consumer's index writes; the lock is for uniformity + the (harmless)
        # retrieval-thread overlap. stop() does NOT hold the lock, so acquiring it here is safe.
        with self._index_lock:
            # Free any still-live HOST-aborted request (never deliver a partial aborted request) and
            # clear the mark set so it never leaks past shutdown; then mark the genuine stragglers
            # finished.
            if self._host_aborted:
                ab_host = tuple(self._host_aborted)
                for rid in [r for r in self.index.live_req_ids()
                            if _match_disk_route(r, ab_host) is not None]:
                    self.index.free(rid)
                self._host_aborted.clear()
            for req_id in self.index.live_req_ids():
                self.index.mark_finished(req_id)

    def close(self) -> None:
        """Non-legacy shared-file drain: run ``stop()`` first when the consumer is still alive, so
        the atexit backstop (which calls ``close()`` without ``flush_aperture``'s ``stop()``) writes
        every step already enqueued before the raw files close, instead of closing them under the
        backlog. The files are closed and the sidecar written even if the consumer failed; that
        failure is then re-raised. Legacy and per-request drains close exactly as before."""
        if self._wp is None or self._closed:
            return super().close()
        err: Optional[BaseException] = None
        if self._started:                   # alive: drain the backlog; dead: surface its error
            try:
                self.stop()
            except BaseException as e:  # noqa: BLE001 -- re-raised once the files are closed
                err = e
        try:
            super().close()
        except BaseException:
            if err is None:
                raise
            logger.exception("hs aperture drain close failed after a consumer failure")
        if err is not None:
            raise err

    def stop(self) -> None:
        """Drain the queue, join the consumer, finalize any end-of-run stragglers, and surface a
        consumer-thread error. Idempotent.

        Ordering (the crux): ``_STOP`` is enqueued AFTER every row ``_DrainItem`` / ``_Finish``, so
        the joined consumer has drained/noted every row into the index; only THEN does
        ``finalize_all()`` mark the still-live stragglers finished → ``pop_deliverable`` sees them.
        The finalize runs on this (collector) thread, but only once the consumer is provably not
        running (joined, or never started), so it can never race the consumer's index writes. A join
        TIMEOUT (wedged consumer) leaves ``is_alive()`` True → the finalize is skipped rather than
        racing the still-running consumer (the wedge itself surfaces via backpressure/error)."""
        if self._started and self._thread.is_alive():
            self._q.put(_STOP)
            join_s = float(os.environ.get("MIA_APERTURE_DRAIN_JOIN_S", "60") or "60")
            self._thread.join(timeout=join_s)
        self._started = False
        if not self._thread.is_alive():
            self.finalize_all()
        if self._error is not None:
            raise RuntimeError(
                "hs off-loop aperture drain consumer thread failed; captured HS may be incomplete"
            ) from self._error
