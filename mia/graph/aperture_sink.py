"""Raw-file write path for the aperture drains: persistent sinks, O_DIRECT, writer threads."""
from __future__ import annotations

import ctypes
import errno
import fcntl
import functools
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
from mia.errors import MiaRefusal
from .thread_device import bind_thread_to_device

logger = logging.getLogger(__name__)

WRITE_MODE_ENV = "MIA_APERTURE_WRITE_MODE"
WRITE_THREADS_ENV = "MIA_APERTURE_WRITE_THREADS"
DIRECT_MIN_BYTES_ENV = "MIA_APERTURE_DIRECT_MIN_BYTES"
WRITE_MODES = ("auto", "direct", "buffered", "legacy")
DEFAULT_WRITE_THREADS = 2
_MAX_WRITE_THREADS = 64

DIRECT_MIN_BYTES = 64 * 1024

_MAX_IO_BYTES = 1 << 30
_PAGE = mmap.PAGESIZE
_PROBE_MAX_BLOCK = 1 << 16


def _human_bytes(n: int) -> str:
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


class RunDirInUseError(ApertureWriteConfigError, MiaRefusal):
    """Another live engine holds this aperture run dir (a refusal: the engine must not start)."""


class RunDirLockError(ApertureWriteConfigError, MiaRefusal):
    """The run-dir lock failed for a reason other than another engine holding it."""


def lock_run_dir(run_dir: str, kind: str):
    """Exclusive per-kind lock on ``run_dir`` for one live drain."""
    path = os.path.join(run_dir, f".{kind}_aperture.lock")
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise RunDirInUseError(
                    f"{run_dir} is in use by another live engine, whose {kind.upper()} capture "
                    f"files this one would truncate. Give each engine its own MIA_APERTURE_DIR."
                ) from None
            if e.errno in (errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS):
                logger.warning("%s: this filesystem takes no flock (%s); a second engine on "
                               "this dir would not be refused", run_dir, e)
                print(f"[mia/aperture] {run_dir}: no run-dir lock on this filesystem ({e}); "
                      f"give each engine its own MIA_APERTURE_DIR", flush=True)
                return None
            raise RunDirLockError(f"{path}: cannot take the run-dir lock: {e!r}") from e
        try:
            same = os.fstat(fd).st_ino == os.stat(path).st_ino
        except FileNotFoundError:
            same = False
        if same:
            return (fd, path)
        os.close(fd)


def release_run_lock(drain) -> None:
    lock = getattr(drain, "_run_lock", None)
    if lock is not None:
        drain._run_lock = None
        fd, path = lock
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        os.close(fd)


def releases_run_lock_on_failure(init):
    """Decorate a drain ``__init__``: a construction that raises gives its run-dir lock back."""
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        try:
            init(self, *args, **kwargs)
        except BaseException:
            release_run_lock(self)
            raise
    return wrapper


def resolve_write_mode() -> Tuple[str, bool]:
    """``(mode, explicit)`` from ``MIA_APERTURE_WRITE_MODE``."""
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
    """``MIA_APERTURE_WRITE_THREADS`` (default 2): writer threads per off-loop drain."""
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
    """Write path for per-request disk staging: legacy if the drain is legacy, else buffered."""
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
    """Smallest predicted write (bytes) that ``auto`` opens with O_DIRECT; default 64 KiB."""
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


@dataclass(frozen=True)
class WriteShape:
    """How many rows one drained step is predicted to append to ONE raw file, and why."""
    rows: int
    basis: str

    def bytes_for(self, row_bytes: int) -> int:
        return int(self.rows) * int(row_bytes)


def predict_rows_per_write(kind: str, *, capture_mode: str,
                           max_batched_tokens: Optional[int], max_num_seqs: Optional[int],
                           aperture_rows: Optional[int]) -> Optional[WriteShape]:
    """Upper bound on rows one drained step appends per raw file, or None if unknown."""
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


@dataclass(frozen=True)
class DirectIOInfo:
    """What ``probe_direct_io`` found for one directory."""
    supported: bool
    block_size: int = 0
    mem_align: int = 0
    source: str = ""
    reason: str = ""


_STATX_DIOALIGN = 0x2000
_AT_FDCWD = -100


def _statx_dio_alignment(path: str) -> Optional[Tuple[int, int]]:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = libc.statx
    except (OSError, AttributeError):
        return None
    buf = ctypes.create_string_buffer(256)
    try:
        rc = fn(ctypes.c_int(_AT_FDCWD), os.fsencode(path), ctypes.c_int(0),
                ctypes.c_uint(_STATX_DIOALIGN), buf)
    except Exception:  # noqa: BLE001
        return None
    if rc != 0:
        return None
    mask = struct.unpack_from("I", buf.raw, 0)[0]
    if not mask & _STATX_DIOALIGN:
        return None
    mem_align, offset_align = struct.unpack_from("II", buf.raw, 152)
    return int(mem_align), int(offset_align)


def _sysfs_logical_block_size(path: str) -> Optional[int]:
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
    """Can ``directory`` take O_DIRECT writes, and at what alignment?"""
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
        buf = mmap.mmap(-1, _PROBE_MAX_BLOCK)
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
                    chunk.release()
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


def alloc_host_rows(n_rows: int, width: int, dtype: torch.dtype, *, pinned: bool,
                    align: int) -> torch.Tensor:
    """An ``(n_rows, width)`` host tensor whose data pointer is a multiple of ``align``."""
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
    """A flat, zero-copy ``memoryview`` of a contiguous CPU tensor's bytes."""
    if t.device.type != "cpu":
        raise ApertureWriteError(f"raw-file write needs a host tensor, got {t.device}")
    if not t.is_contiguous():
        raise ApertureWriteError("raw-file write needs a contiguous host tensor")
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return memoryview(b"")
    return memoryview(flat.view(torch.uint8).numpy())


class RawFileSink:
    """One raw file, open for the whole run."""

    def __init__(self, path: str, *, direct: bool, block_size: int = 0, mem_align: int = 0):
        self.path = path
        self.direct = bool(direct)
        self.block_size = int(block_size) if direct else 0
        self.mem_align = int(mem_align) if direct else 0
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
        if self.direct:
            flags |= os.O_DIRECT
        self.fd: Optional[int] = os.open(path, flags, 0o666)
        self.offset = 0

    @property
    def mode(self) -> str:
        return "direct" if self.direct else "buffered"

    def write(self, t: torch.Tensor) -> int:
        """Append ``t``'s bytes; returns the byte count."""
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


_POOL_STOP = object()


class WriterPool:
    """``n`` daemon threads running write tasks; each returns a ``concurrent.futures.Future``."""

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
        except BaseException as e:  # noqa: BLE001
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
            except BaseException as e:  # noqa: BLE001
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
        """Stop the threads after the queued tasks."""
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._q.put(_POOL_STOP)
        for t in self._threads:
            t.join(timeout=timeout)


def join_writes(futures: Iterable[Future]) -> List:
    """Wait for every future, then raise the first failure or return the results in order."""
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
    """Wait for every future without raising, for a step already failing for another reason."""
    futs = list(futures)
    if futs:
        _wait_futures(futs)


@dataclass
class _FileSpec:
    key: object
    path: str
    kind: str
    row_bytes: int


@dataclass
class WriteStats:
    """Cumulative per-drain write-path accounting (seconds and bytes)."""
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
    """The resolved write path of one drain: per-file modes, open sinks and writer pool."""

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
        self.kind_reason: Dict[str, str] = {}
        self.kind_predicted: Dict[str, int] = {}
        self._size_downgraded: Dict[str, int] = {}
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

    def start_pool(self, n_threads: int, device, name: str) -> None:
        if self.pool is None:
            self.pool = WriterPool(n_threads, device, name)
            self.threads = int(n_threads)

    def sink(self, key) -> RawFileSink:
        return self.sinks[key]

    def kind_modes(self) -> Dict[str, str]:
        """``{kind: "direct" | "buffered" | "direct+buffered"}`` over the open files."""
        seen: Dict[str, set] = {}
        for k, s in self.sinks.items():
            seen.setdefault(self.specs[k].kind, set()).add(s.mode)
        return {kind: "+".join(sorted(m)) for kind, m in seen.items()}

    def summary(self) -> str:
        """Install-line summary: mode per tensor kind with predicted write size, threads, block size."""
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
        """Record that a drained step spanned ``rows`` rows."""
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
        """Join the writer threads, then close every fd."""
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
    """A per-request staging's raw files: opened on first use, written zero-copy, buffered."""

    def __init__(self, label: str = "per-request staging"):
        self.label = label
        self.sinks: Dict[object, RawFileSink] = {}

    def append(self, key, path: str, t: torch.Tensor) -> int:
        """Append ``t``'s bytes to ``key``'s file, truncating it on first use."""
        sink = self.sinks.get(key)
        if sink is None:
            sink = RawFileSink(path, direct=False)
            self.sinks[key] = sink
        return sink.write(t if t.is_contiguous() else t.contiguous())

    def close(self) -> None:
        """Close every fd."""
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
    """Account one drained step, in ``stats`` always and in PROF when ``MIA_PROFILE=1``."""
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

