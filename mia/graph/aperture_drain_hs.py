"""Multi-layer host drain for the HS capture aperture."""
from __future__ import annotations

import bisect
import logging
import mmap
import os
import queue
import re
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
    return str(dtype).rsplit(".", 1)[-1]


def _raw_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def _resolve_mmap_capacity_bytes(aperture: CaptureAperture) -> int:
    override = os.environ.get("MIA_APERTURE_MMAP_BYTES")
    if override:
        return int(override)
    return max(2 * _GIB, int(aperture.n_slots) * int(aperture.row_bytes))


class _MmapLayerWriter:
    """Pre-sized MAP_SHARED write handle for one layer's raw file (MIA_APERTURE_MMAP=1)."""

    def __init__(self, path: str, capacity_bytes: int):
        self.path = path
        self.capacity = int(capacity_bytes)
        self.offset = 0
        self._overflowed = False
        self._warned = False
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
            self._fh.truncate(self.offset)
            self._fh.close()
            self._mm = None
            self._fh = None
        if data:
            with open(self.path, "ab") as f:
                f.write(data)
            self.offset += len(data)

    def close(self) -> None:
        if self._mm is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None
        if self._fh is not None:
            self._fh.truncate(self.offset)
            self._fh.close()
            self._fh = None




def _aperture_debug() -> bool:
    return os.environ.get("MIA_APERTURE_DEBUG") == "1"


def _dbg(msg: str) -> None:
    print(f"[mia/aperture-disk] {msg}", flush=True)


def _stamp_file_row(entries: List[LayerEntry], cursor_before: Dict[int, int],
                     step_start_logical: int,
                     plans: Optional[Dict[int, "LayerCopyPlan"]] = None) -> None:
    for e in entries:
        base = cursor_before.get(e.layer, 0)
        plan = plans.get(int(e.layer)) if plans is not None else None
        if plan is None or int(e.n_rows) <= 0:
            off = int(e.logical_start) - int(step_start_logical)
        else:
            off = plan.row_offset(e.logical_start)
        e.file_row = base + off


def _drain_selective_enabled() -> bool:
    return os.environ.get("MIA_DRAIN_SELECTIVE", "1") == "1"


def _resolve_selective(armed: bool, *, off_loop: bool,
                       per_request: bool) -> Tuple[bool, Optional[str]]:
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
    """What ONE layer's drain copies out of the aperture for ONE step."""
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
        """Where logical row ``logical_start`` lands in this layer's compacted copy."""
        ls = int(logical_start)
        i = bisect.bisect_right(self.starts, ls) - 1
        if i < 0 or ls >= self.starts[i] + self.lengths[i]:
            raise KeyError(
                f"logical row {ls} is not covered by this layer's copy plan {self.ranges}")
        return self.offsets[i] + (ls - self.starts[i])


def is_degenerate_full_step(records, installed: Set[int], start_logical: int,
                            n_rows: int) -> bool:
    """True when this step wants every installed layer over its whole span."""
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
    whole = LayerCopyPlan.from_ranges([(int(start_logical), int(n_rows))], aperture)
    return {int(ln): whole for ln in layers}


def build_copy_plans(records, layers: Iterable[int], aperture: CaptureAperture,
                     start_logical: int, n_rows: int,
                     selective: bool) -> Dict[int, LayerCopyPlan]:
    """The copy list: ``{layer -> LayerCopyPlan}`` for ONE step."""
    if not selective:
        return _whole_span_plans(layers, aperture, start_logical, n_rows)
    installed = {int(ln) for ln in layers}
    if is_degenerate_full_step(records, installed, start_logical, n_rows):
        return _whole_span_plans(layers, aperture, start_logical, n_rows)
    wanted: Dict[int, List[Tuple[int, int]]] = {}
    for rec in records:
        rec_layers = getattr(rec, "layers", None)
        if rec_layers is None:
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
    """Total (layer, row) cells the per-request records name, i.e. what a selective drain copies."""
    total = 0
    for rec in records:
        n = int(getattr(rec, "n_rows", 0) or 0)
        if n <= 0:
            continue
        layers = getattr(rec, "layers", None)
        total += n if layers is None else n * len(layers)
    return total


def _match_disk_route(rid: str, route_keys) -> Optional[str]:
    rid = str(rid)
    if rid in route_keys:
        return rid
    for ext in route_keys:
        if rid.startswith(f"{ext}-"):
            return ext
    return None


def _sanitize_req_id(req_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]", "_", str(req_id))
    return s or "req"


class _PerRequestDiskStaging:
    """Disk-route staging: streams one request's rows to its own run dir, laid out like a shared run."""

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
        self._writers: Dict[int, _MmapLayerWriter] = {}
        self._plain: Dict[int, str] = {}
        self._sinks = None if self._legacy else PerRequestSinks(f"hs staging req {req_id}")
        self._rows: Dict[int, int] = {}
        self._entries: List[LayerEntry] = []
        self._closed = False

    def _raw_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"hs_layer_{layer}.raw")

    def append(self, layer: int, rows_cpu: torch.Tensor, n_rows: int, mode: str) -> None:
        """Append one layer's host rows to this request's raw file and record the matching LayerEntry."""
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
                open(p, "wb").close()
                self._plain[layer] = p
            with open(p, "ab") as f:
                f.write(_raw_bytes(rows_cpu))
        self._entries.append(LayerEntry(self.req_id, int(layer), int(start), int(n_rows), mode))
        self._rows[layer] = start + int(n_rows)

    def close(self) -> None:
        """Finalize on finish: sync and truncate every layer writer, then write this request's sidecar."""
        if self._closed:
            return
        self._closed = True
        if self._sinks is not None:
            self._sinks.close()
        for w in self._writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "hs per-request staging: writer close failed for req %r (partial staging); "
                    "continuing", self.req_id)
        try:
            if os.path.isdir(self.run_dir):
                write_sidecar(self.meta_path, [StepMeta(list(self._entries))], self.header)
        except Exception:  # noqa: BLE001
            logger.exception(
                "hs per-request staging: sidecar write failed for req %r under %r (partial/aborted "
                "staging); delivery skipped", self.req_id, self.run_dir)

    def discard(self) -> None:
        """Abort cleanup: release open writers without a sidecar and remove the staging dir."""
        if self._sinks is not None:
            self._sinks.close()
        for w in self._writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        self._writers = {}
        self._closed = True
        import shutil
        shutil.rmtree(self.run_dir, ignore_errors=True)


class MultiLayerApertureDrain:
    """Drains a shared-cursor ``CaptureAperture`` across N per-layer ``hs_buf`` buffers."""

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
        self.shape = shape
        self.write_mode, self._write_mode_explicit = resolve_write_mode()
        self._wp: Optional[ApertureWritePath] = None
        self._sidecar: Optional[HsSidecarLog] = None
        self._io_lock = threading.Lock()
        self._wstats = WriteStats()
        self._write_note = ""
        self._mmap_writers: Dict[int, _MmapLayerWriter] = {}
        self._mmap_enabled = False
        self._perreq_write_mode = "legacy"
        if not setup_sink:
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
                        except Exception:  # noqa: BLE001
                            pass
                    self._mmap_writers = {}
                    self._mmap_enabled = False
            if not self._mmap_enabled:
                for p in self.raw_paths.values():
                    open(p, "wb").close()
        self._steps: List[StepMeta] = []
        self._pending_entries: List[LayerEntry] = []
        self._closed = False
        self.selective, self.selective_disabled_reason = _resolve_selective(
            _drain_selective_enabled(), off_loop=False, per_request=False)
        self._rows_copied = 0
        self._rows_skipped = 0
        self._degenerate_steps = 0
        self._layer_nums: List[int] = [int(ln) for ln, _ in self.layers]
        self._layer_set = set(self._layer_nums)
        self._file_rows: Dict[int, int] = {ln: 0 for ln, _ in self.layers}
        self._fr = np.zeros(len(self.layers), dtype=np.int64)
        self._fr_uniform = True
        self._pending_records: list = []

    def _cursor_advance_all(self, n_rows: int):
        if self._fr_uniform:
            cur = int(self._fr[0]) if len(self._fr) else 0
        else:
            cur = self._fr.copy()
        self._fr += int(n_rows)
        return cur

    def _cursor_advance(self, appended: Dict[int, int]):
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
        """Install-line summary: write mode per tensor kind, writer threads, O_DIRECT block."""
        if self._wp is not None:
            return self._wp.summary()
        if self._write_note:
            return self._write_note
        sink = ("pre-sized MAP_SHARED mmap (MIA_APERTURE_MMAP=1)" if self._mmap_enabled
                else "tobytes + open/append/close per step")
        return (f"write mode=legacy -> hs=legacy ({sink}, on the drain thread) | for A/B "
                f"validation only ({WRITE_MODE_ENV}=legacy)")

    def _selective_active(self) -> bool:
        return bool(self.selective) and not bool(getattr(self, "per_request", False))

    def row_counts(self) -> dict:
        """Read-only drain census, reachable via collective_rpc('get_drain_row_counts')."""
        return {
            "hs.drain.rows_copied": int(self._rows_copied),
            "hs.drain.rows_skipped": int(self._rows_skipped),
            "hs.drain.degenerate_steps": int(self._degenerate_steps),
            "selective": bool(self._selective_active()),
            "selective_disabled_reason": self.selective_disabled_reason,
        }

    def record_entries(self, entries: List) -> None:
        if self._sidecar is not None:
            self._pending_records.extend(entries)
            return
        self._pending_entries.extend(expand_records(entries))

    def _append_layer_rows(self, ln: int, rows_cpu: torch.Tensor) -> None:
        data = _raw_bytes(rows_cpu)
        writer = self._mmap_writers.get(ln) if self._mmap_enabled else None
        if writer is not None:
            writer.append(data)
        else:
            with open(self.raw_paths[ln], "ab") as f:
                f.write(data)
        self._file_rows[ln] = self._file_rows.get(ln, 0) + int(rows_cpu.shape[0])

    def drain_once(self) -> int:
        """Copy pending aperture rows out of every layer buffer, queue sidecar entries, advance the cursor."""
        if self._wp is not None:
            return self._drain_once_fast()
        moved = self.aperture.pending_rows()
        if moved == 0:
            return 0
        segments = self.aperture.drained_segments()
        step_start_logical = self.aperture._drain
        cursor_before: Dict[int, int] = {}
        for ln, hs_buf in self.layers:
            pieces = [hs_buf[s:e].detach().to("cpu") for s, e in segments]
            rows_cpu = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
            cursor_before[ln] = self._file_rows.get(ln, 0)
            self._append_layer_rows(ln, rows_cpu)
            self._rows_copied += int(rows_cpu.shape[0])
        if self._pending_entries:
            _stamp_file_row(self._pending_entries, cursor_before, step_start_logical)
            self._steps.append(StepMeta(list(self._pending_entries)))
            self._pending_entries = []
        self.aperture.advance_drain(moved)
        return moved

    def _drain_once_fast(self) -> int:
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
        """Flush, truncate and release every layer writer, then write the shared sidecar (idempotent)."""
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
        if self._closed:
            return
        _close_write_path(self, "hs")


def _close_write_path(drain, kind: str) -> None:
    join_s = float(os.environ.get("MIA_APERTURE_DRAIN_JOIN_S", "60") or "60")
    got = drain._io_lock.acquire(timeout=join_s)
    err: Optional[BaseException] = None
    try:
        if got:
            try:
                drain._wp.close()
            except BaseException as e:  # noqa: BLE001
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


_STOP = object()


@dataclass
class _DrainItem:
    """One enqueued step: sidecar entries, aperture start slot, row count and a post-scatter event."""
    entries: list
    start_logical: int
    n_rows: int
    event: object = None


@dataclass
class _Finish:
    """Per-request finish signal, queued after that request's row entries."""
    req_id: str


class OffLoopApertureDrain(MultiLayerApertureDrain):
    """Off-loop (consumer-thread) sibling of the SYNCHRONOUS ``MultiLayerApertureDrain``."""

    _ALLOW_DIRECT = True

    def __init__(self, aperture: CaptureAperture, layers, run_dir: str, header: dict,
                 per_request: bool = False, index: Optional[PerRequestIndex] = None,
                 offload=None, disk_base: Optional[str] = None,
                 shape: Optional[WriteShape] = None):
        _threads = (resolve_write_threads()
                    if not per_request and resolve_write_mode()[0] != "legacy" else 0)
        super().__init__(aperture, layers, run_dir, header, setup_sink=not per_request,
                         shape=shape)
        self.per_request = bool(per_request)
        self.selective, self.selective_disabled_reason = _resolve_selective(
            _drain_selective_enabled(), off_loop=True, per_request=self.per_request)
        self.index: Optional[PerRequestIndex] = (
            index if index is not None
            else (PerRequestIndex() if self.per_request else None))
        self._offload = offload
        self._disk_base = disk_base or os.path.join(run_dir, "perreq")
        self._disk_routed: Dict[str, str] = {}
        self._disk_staging: Dict[str, _PerRequestDiskStaging] = {}
        self._disk_delivered_src: Dict[str, str] = {}
        self._disk_reclaim_pending: Dict[str, str] = {}
        self._disk_aborted: set = set()
        self._host_aborted: set = set()
        self._note_unstaged_finish = False
        self._disk_unstaged: set = set()
        self._perreq_cap = int(os.environ.get(
            "MIA_APERTURE_PERREQ_MMAP_BYTES", str(64 * 1024 * 1024)) or (64 * 1024 * 1024))
        self._perreq_mmap = os.environ.get("MIA_APERTURE_MMAP", "0") != "0"
        self._index_lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        dev = self.layers[0][1].device if self.layers else torch.device("cpu")
        self._is_cuda = (dev.type == "cuda") and torch.cuda.is_available()
        self._stream = torch.cuda.Stream(dev) if self._is_cuda else None
        self._aperture_depth = max(1, int(os.environ.get("MIA_CAPTURE_DRAIN_APERTURE", "3") or "3"))
        self._copy_events = ([torch.cuda.Event() for _ in range(self._aperture_depth)]
                             if self._stream is not None else [])
        self._aperture_idx = 0
        self._pinned: dict = {ln: None for ln, _ in self.layers}
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
        """O(1) hand-off."""
        with PROF.timed("graph.enqueue"):
            self._q.put(_DrainItem(entries, int(start_logical), int(n_rows), event))

    def enqueue_finish(self, req_id) -> None:
        """O(1) hand-off of a per-request FINISH."""
        if not self.per_request:
            return
        self._q.put(_Finish(str(req_id)))

    def route_to_disk(self, req_id, dest, offload=None) -> None:
        """Route ``req_id`` to per-request disk staging, offloaded to ``dest`` when it finishes."""
        if not self.per_request:
            return
        req_id = str(req_id)
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
                new_offload = None
                from mia.graph.child_process import register_shutdown
                register_shutdown(self._offload.close)
            self._disk_routed[req_id] = str(dest)
        if new_offload is not None:
            new_offload.close()
        if _aperture_debug():
            _dbg(f"route_to_disk: req_id={req_id!r} (EXTERNAL) dest={dest!r} "
                 f"offload={type(self._offload).__name__}")

    def finished_unstaged(self, req_id) -> bool:
        """True when disk-routed ``req_id`` finished on this rank with nothing staged."""
        with self._index_lock:
            return str(req_id) in self._disk_unstaged

    def disk_residency(self) -> int:
        """Number of disk-routed requests still holding staging state."""
        with self._index_lock:
            return len(self._disk_staging)

    def unlink_delivered_source(self, req_id) -> bool:
        """Remove the server-side staging dir of a delivered disk-routed request."""
        req_id = str(req_id)
        with self._index_lock:
            src = self._disk_delivered_src.pop(req_id, None)
        if src is None:
            return False
        import shutil
        shutil.rmtree(src, ignore_errors=True)
        return True

    def _reclaim_settled_pending(self) -> None:
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
        """Abort cleanup for the RPC route: never re-stage this request's drained rows."""
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
        """Abort cleanup for the disk route: mark the request aborted, keeping its live staging."""
        if not self.per_request:
            return
        req_id = str(req_id)
        marked = False
        with self._index_lock:
            routed = self._disk_routed.pop(req_id, None)
            src = self._disk_delivered_src.pop(req_id, None)
            self._disk_unstaged.discard(req_id)
            if req_id in self._disk_staging or routed is not None:
                self._disk_aborted.add(req_id)
                marked = True
        reclaimed = False
        deferred = False
        if src is not None:
            if self._offload is None or self._offload.settled(req_id):
                import shutil
                shutil.rmtree(src, ignore_errors=True)
                reclaimed = True
            else:
                with self._index_lock:
                    self._disk_reclaim_pending[req_id] = src
                deferred = True
        if _aperture_debug():
            _dbg(f"clear_request_disk MARK-abort: req={req_id!r} marked={marked} "
                 f"delivered_src_reclaimed={reclaimed} deferred_reclaim={deferred} "
                 f"(consumer owns the staging-dir discard)")

    def _finalize_finish_isolated(self, req_id) -> None:
        try:
            self._handle_finish(req_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "hs off-loop aperture drain: per-request FINALIZE failed for req_id=%r; that request's "
                "delivery is dropped, the consumer continues", req_id)
            if _aperture_debug():
                _dbg(f"finish FAILED (isolated, consumer continues): req_id={req_id!r}")

    def _run(self) -> None:
        try:
            bind_thread_to_device(self._stream.device if self._stream is not None else None)
            while True:
                item = self._q.get()
                if item is _STOP:
                    self._q.task_done()
                    break
                try:
                    if isinstance(item, _Finish):
                        self._finalize_finish_isolated(item.req_id)
                    else:
                        self._drain_item(item)
                    self._reclaim_settled_pending()
                finally:
                    self._q.task_done()
        except BaseException as e:  # noqa: BLE001
            self._error = e
            logger.exception("hs off-loop aperture drain consumer thread died")

    def _read_segments(self, plans: Dict[int, LayerCopyPlan],
                       event) -> List[Tuple[int, torch.Tensor]]:
        if self._stream is not None:
            slot = self._aperture_idx
            stale = self._copy_events[slot] if slot < len(self._copy_events) else None
            if stale is not None:
                stale.synchronize()
            if event is not None:
                self._stream.wait_event(event)
            pieces: List[Tuple[int, torch.Tensor]] = []
            with torch.cuda.stream(self._stream):
                for ln, hs_buf in self.layers:
                    plan = plans.get(ln)
                    if plan is None or plan.total_rows == 0:
                        continue
                    buf = self._pinned_buf(ln, plan.total_rows, hs_buf.shape[1], hs_buf.dtype)
                    off = 0
                    for s, e in plan.segments:
                        src = hs_buf[s:e]
                        buf[off:off + (e - s)].copy_(src, non_blocking=True)
                        src.record_stream(self._stream)
                        off += (e - s)
                        self._rows_copied += (e - s)
                    pieces.append((ln, buf))
            done = self._copy_events[slot] if slot < len(self._copy_events) else None
            if done is not None:
                done.record(self._stream)
                done.synchronize()
            else:
                self._stream.synchronize()
            self._aperture_idx = (slot + 1) % self._aperture_depth
            return pieces
        if event is not None:
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001
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
        buf = self._host.get(ln)
        if buf is None or buf.shape[0] < n_rows:
            buf = alloc_host_rows(n_rows, width, dtype, pinned=self._is_cuda,
                                  align=self._wp.mem_align)
            self._host[ln] = buf
        return buf[:n_rows]

    def _issue_d2h(self, plans: Dict[int, LayerCopyPlan], event) -> list:
        jobs = []
        if self._stream is not None:
            if event is not None:
                self._stream.wait_event(event)
            with torch.cuda.stream(self._stream):
                for i, (ln, hs_buf) in enumerate(self.layers):
                    plan = plans.get(ln)
                    if plan is None or plan.total_rows == 0:
                        continue
                    buf = self._host_buf(ln, plan.total_rows, hs_buf.shape[1], hs_buf.dtype)
                    off = 0
                    for s, e in plan.segments:
                        src = hs_buf[s:e]
                        buf[off:off + (e - s)].copy_(src, non_blocking=True)
                        src.record_stream(self._stream)
                        off += (e - s)
                        self._rows_copied += (e - s)
                    ev = self._layer_events[i]
                    ev.record(self._stream)
                    jobs.append((ln, i, buf, ev))
            return jobs
        if event is not None:
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001
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
            if compacting:
                cursor = self._cursor_advance({i: int(buf.shape[0]) for _, i, buf, _ in jobs})
            else:
                cursor = self._cursor_advance_all(int(item.n_rows))
            futs = []
            t_first = None
            try:
                for ln, _i, buf, ev in jobs:
                    if ev is not None:
                        ev.synchronize()
                    if t_first is None:
                        t_first = time.perf_counter()
                    futs.append(wp.pool.submit(timed_write, wp.sinks[ln], buf))
                t2 = time.perf_counter()
                block = self._sidecar.prepare(item.entries, item.start_logical, cursor,
                                              plans if compacting else None)
            except BaseException:
                join_writes_quietly(futs)
                raise
            t3 = time.perf_counter()
            results = join_writes(futs)
            t4 = time.perf_counter()
            self._sidecar.commit(block)
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
        copied_before = self._rows_copied
        t1 = time.perf_counter()
        with PROF.timed("bank.consumer.d2h"):
            pieces = self._read_segments(plans, item.event)
        t2 = time.perf_counter()
        t3 = t2
        nbytes = 0
        copied_here = self._rows_copied - copied_before
        full_here = len(self.layers) * int(item.n_rows)
        assert copied_here <= full_here, (
            f"selective drain copied {copied_here} rows for a step that owns at most {full_here} "
            f"({len(self.layers)} layers x {item.n_rows} rows) -- the copy list escaped the step")
        self._rows_skipped += full_here - copied_here
        if self.per_request:
            self._demux_into_index(item, pieces)
        else:
            cursor_before: Dict[int, int] = {}
            for ln, rows_cpu in pieces:
                cursor_before[ln] = self._file_rows.get(ln, 0)
                self._append_layer_rows(ln, rows_cpu)
                nbytes += int(rows_cpu.numel()) * int(rows_cpu.element_size())
            t3 = time.perf_counter()
            entries = expand_records(item.entries)
            if entries:
                _stamp_file_row(entries, cursor_before, item.start_logical,
                                plans=plans if compacting else None)
                self._steps.append(StepMeta(entries))
        aperture.advance_drain(item.n_rows)
        t4 = time.perf_counter()
        record_step_stats(
            self._wstats, "hs", rows=item.n_rows, step_s=t4 - t0, d2h_s=t2 - t1,
            write_s=t3 - t2, write_tail_s=t3 - t2, busy_s=t3 - t2,
            bookkeeping_s=(t1 - t0) + (t4 - t3), bytes_by_mode={"legacy": nbytes} if nbytes else {})

    def _demux_into_index(self, item: _DrainItem, pieces: List[Tuple[int, torch.Tensor]]) -> None:
        by_layer = {ln: rows for ln, rows in pieces}
        base = int(item.start_logical)
        entries = expand_records(item.entries)
        with self._index_lock:
            routed_keys = tuple(self._disk_routed) if self._disk_routed else ()
        any_disk = bool(routed_keys)
        staged = []
        for e in entries:
            layer_rows = by_layer.get(e.layer)
            if layer_rows is None:
                continue
            off = int(e.logical_start) - base
            sl = layer_rows[off:off + int(e.n_rows)]
            ext = _match_disk_route(e.req_id, routed_keys) if any_disk else None
            if ext is not None:
                if _aperture_debug():
                    _dbg(f"demux DISK hit: entry.req_id={e.req_id!r} -> route={ext!r} "
                         f"layer={e.layer} n_rows={int(e.n_rows)}")
                try:
                    self._disk_write(ext, e.layer, sl, int(e.n_rows), e.hs_mode)
                except Exception:  # noqa: BLE001
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
        req_id = str(req_id)
        with self._index_lock:
            aborted_ext = (_match_disk_route(req_id, tuple(self._disk_aborted))
                           if self._disk_aborted else None)
            if aborted_ext is not None:
                self._disk_aborted.discard(aborted_ext)
                self._host_aborted.discard(aborted_ext)
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
                    self._disk_unstaged.add(ext)
        if aborted_ext is not None:
            if stg_abort is not None:
                stg_abort.discard()
            if _aperture_debug():
                _dbg(f"finish ABORT-reclaim: id={req_id!r} route={aborted_ext!r} "
                     f"discarded={stg_abort is not None} (single-owner consumer discard)")
            return
        if disk_dest is not None:
            if _aperture_debug():
                _dbg(f"finish DISK: id={req_id!r} route={ext!r} "
                     f"run_dir={(stg.run_dir if stg else None)!r} dest={disk_dest!r} "
                     f"submit={stg is not None and self._offload is not None}")
            if stg is not None:
                stg.close()
                if self._offload is not None:
                    self._offload.submit(ext, stg.run_dir, disk_dest)
            return
        if self.index is None:
            return
        with self._index_lock:
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

    def finalize_all(self) -> None:
        """End of run only: mark every live request finished so pop_deliverable can return it."""
        with self._index_lock:
            disk_pending = list(self._disk_staging.keys())
        for req_id in disk_pending:
            self._finalize_finish_isolated(req_id)
        with self._index_lock:
            aborted_pending = list(self._disk_aborted)
        for ab_id in aborted_pending:
            self._finalize_finish_isolated(ab_id)
        self._reclaim_settled_pending()
        if self.index is None:
            return
        with self._index_lock:
            if self._host_aborted:
                ab_host = tuple(self._host_aborted)
                for rid in [r for r in self.index.live_req_ids()
                            if _match_disk_route(r, ab_host) is not None]:
                    self.index.free(rid)
                self._host_aborted.clear()
            for req_id in self.index.live_req_ids():
                self.index.mark_finished(req_id)

    def close(self) -> None:
        """Stop a live consumer, then close the raw files so every enqueued step is written."""
        if self._wp is None or self._closed:
            return super().close()
        err: Optional[BaseException] = None
        if self._started:
            try:
                self.stop()
            except BaseException as e:  # noqa: BLE001
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
        """Drain the queue, join the consumer, finalize stragglers and re-raise any consumer error."""
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

