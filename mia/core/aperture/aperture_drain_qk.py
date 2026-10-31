"""Multi-layer host drain for the QK capture aperture."""
from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from mia._profiler import PROF
from .capture_aperture import CaptureAperture
from mia.core.delivery.per_request_delivery import PerRequestIndex
from .aperture_drain_hs import (
    _STOP,
    _DrainItem,
    _Finish,
    _close_write_path,
    _dbg,
    _aperture_debug,
    _sanitize_req_id,
)
from .aperture_gather import (
    DELIVER_ENV as GATHER_DELIVER_ENV, ApertureGatherProcess, delivery_enabled)
from .aperture_layout import QK_LAYOUT
from .aperture_run_index import (
    FLUSH_MS_ENV, GATHER_ENV, QkRunIndex, RunIndexFlusher, clear_chain, flush_interval_ms,
    gather_enabled, new_run_id)
from .aperture_trim import (
    TRIM_ENV as GATHER_TRIM_ENV, trim_chunk_bytes, trim_enabled, trim_explicit, trim_lag_bytes)
from .aperture_metadata import (
    QKStepEntry, QkSidecarLog, StepMeta, expand_qk_records, write_qk_sidecar)
from .aperture_sink import (
    ApertureWriteConfigError, ApertureWriteError, ApertureWritePath, PerRequestSinks,
    WriteShape, WriteStats, alloc_host_rows, join_writes, join_writes_quietly,
    lock_run_dir, record_step_stats, release_run_lock, releases_run_lock_on_failure,
    resolve_per_request_write_mode, resolve_write_mode, resolve_write_threads, timed_write_at)
from mia.core.runtime.thread_device import bind_thread_to_device
from mia.core.delivery.offload_process import OffloadProcess
from mia.workers._common import request_id_base

logger = logging.getLogger(__name__)


@dataclass
class _InFlight:
    """A step with d2h issued and writes submitted, not yet joined or published."""
    item: _DrainItem
    futs: list
    block: object
    parity: int
    t_first: Optional[float]
    d2h_s: float
    bookkeeping_s: float
    issue_s: float

def _match_disk_route(rid: str, route_keys) -> Optional[str]:
    rid = str(rid)
    if rid in route_keys:
        return rid
    base = request_id_base(rid)
    return base if base is not None and base in route_keys else None


class _PerRequestQKDiskStaging:
    """QK disk-route staging: streams one request's q/k rows to its own run dir."""

    def __init__(self, req_id: str, run_dir: str, header: dict):
        self.req_id = str(req_id)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.header = dict(header)
        self.meta_path = os.path.join(run_dir, "qk_aperture_meta.jsonl")
        self._sinks = PerRequestSinks(f"qk staging req {req_id}")
        self._q_rows: Dict[int, int] = {}
        self._k_rows: Dict[int, int] = {}
        self._entries: List[QKStepEntry] = []
        self._closed = False

    def _q_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"qk_q_layer_{layer}.raw")

    def _k_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"qk_k_layer_{layer}.raw")

    def append(self, layer: int, q_rows_cpu, k_rows_cpu: torch.Tensor,
               prefix_end: int, num_computed: int) -> None:
        """Append one step's q/k rows for this (req, layer) and record the matching QKStepEntry."""
        k_start = self._k_rows.get(layer, 0)
        k_n = int(k_rows_cpu.shape[0])
        self._sinks.append(("k", layer), self._k_path(layer), k_rows_cpu)
        self._k_rows[layer] = k_start + k_n
        if q_rows_cpu is not None and int(q_rows_cpu.shape[0]) > 0:
            q_start = self._q_rows.get(layer, 0)
            q_n = int(q_rows_cpu.shape[0])
            self._sinks.append(("q", layer), self._q_path(layer), q_rows_cpu)
            self._q_rows[layer] = q_start + q_n
        else:
            q_start, q_n = -1, 0
        self._entries.append(QKStepEntry(
            req_id=self.req_id, layer=int(layer),
            k_start=int(k_start), k_rows=int(k_n),
            q_start=int(q_start), q_rows=int(q_n),
            prefix_end=int(prefix_end), num_computed=int(num_computed)))

    def close(self) -> None:
        """Finalize on finish: close every q/k sink, then write this request's sidecar."""
        if self._closed:
            return
        self._closed = True
        self._sinks.close()
        try:
            if os.path.isdir(self.run_dir):
                write_qk_sidecar(self.meta_path, [StepMeta(list(self._entries))], self.header)
        except Exception:  # noqa: BLE001
            logger.exception(
                "qk per-request staging: sidecar write failed for req %r under %r (partial/aborted "
                "staging); delivery skipped", self.req_id, self.run_dir)

    def discard(self) -> None:
        """Abort cleanup: release open q/k sinks without a sidecar and remove the staging dir."""
        self._sinks.close()
        self._closed = True
        shutil.rmtree(self.run_dir, ignore_errors=True)


class MultiLayerQKApertureDrain:
    """Drains a shared-cursor ``CaptureAperture`` across N per-layer ``(q_buf, k_buf)`` pairs."""

    _ALLOW_DIRECT = False
    _DIRECT_REFUSAL = ("the synchronous drain (MIA_APERTURE_SYNC_DRAIN=1) writes pageable host "
                       "copies with no alignment guarantee; it writes zero-copy buffered")

    @releases_run_lock_on_failure
    def __init__(self, aperture: CaptureAperture, layers: List[Tuple[int, torch.Tensor, torch.Tensor]],
                 run_dir: str, header: dict, setup_sink: bool = True,
                 shape: Optional[WriteShape] = None):
        self.aperture = aperture
        self.layers = list(layers)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.header = dict(header)
        self.meta_path = os.path.join(run_dir, "qk_aperture_meta.jsonl")
        self.q_raw_paths = {ln: os.path.join(run_dir, f"qk_q_layer_{ln}.raw")
                            for ln, _, _ in self.layers}
        self.k_raw_paths = {ln: os.path.join(run_dir, f"qk_k_layer_{ln}.raw")
                            for ln, _, _ in self.layers}
        self.write_mode, self._write_mode_explicit = resolve_write_mode()
        self.shape = shape
        self._wp: Optional[ApertureWritePath] = None
        self._sidecar: Optional[QkSidecarLog] = None
        self._io_lock = threading.Lock()
        self._wstats = WriteStats()
        self._write_note = ""
        self._pending_records: list = []
        self.gather = gather_enabled()
        self._run_index: Optional[QkRunIndex] = None
        self._run_flusher: Optional[RunIndexFlusher] = None
        self.run_id: Optional[str] = None
        self._gather_proc: Optional[ApertureGatherProcess] = None
        self.gather_stamp = False
        self._flush_ms = 0
        self._deferred_finishes: list = []
        if self.gather:
            if not setup_sink:
                raise ApertureWriteConfigError(
                    f"{GATHER_ENV}=1 reads the SHARED q/k layer files, but this drain writes none: "
                    f"per-request delivery (MIA_APERTURE_PER_REQUEST=1) demuxes every row into its "
                    f"request's staging instead. "
                    f"Unset one of the two (MIA_APERTURE_DELIVERY=hybrid sets them for you).")
            if int(os.environ.get("MIA_DP_SIZE", "1") or "1") > 1:
                raise ApertureWriteConfigError(
                    f"{GATHER_ENV}=1 on a Q/K capture at "
                    f"MIA_DP_SIZE={os.environ.get('MIA_DP_SIZE')}: the Q/K "
                    f"hybrid is not collected across DP engines (DP>1 refused).")
            self._flush_ms = flush_interval_ms()
        if delivery_enabled():
            if not self.gather:
                raise ApertureWriteConfigError(
                    f"{GATHER_DELIVER_ENV}=1 needs the run index ({GATHER_ENV}=1); set it, or "
                    f"unset {GATHER_DELIVER_ENV}.")
            if self._flush_ms <= 0:
                raise ApertureWriteConfigError(
                    f"{GATHER_DELIVER_ENV}=1 needs the index PUBLISHED while the run is going; "
                    f"set {FLUSH_MS_ENV}.")
        if trim_explicit():
            armed = trim_enabled()
            trim_chunk_bytes()
            trim_lag_bytes()
            if armed and not delivery_enabled():
                raise ApertureWriteConfigError(
                    f"{GATHER_TRIM_ENV}=1 frees the shared files behind the gather's cursor, but "
                    f"{GATHER_DELIVER_ENV} is off: nothing would be trimmed. Unset "
                    f"{GATHER_TRIM_ENV}.")
        self._run_lock = lock_run_dir(run_dir, "qk") if setup_sink else None
        if setup_sink:
            clear_chain(run_dir, layout=QK_LAYOUT)
        if not setup_sink:
            resolve_per_request_write_mode(self.write_mode, self._write_mode_explicit)
            self._write_note = (
                f"write mode={self.write_mode} does not apply to shared raw files: per-request "
                f"delivery (MIA_APERTURE_PER_REQUEST=1) writes none. Its disk staging writes "
                f"zero-copy buffered, one fd per q/k file kept open for the request")
        else:
            files = {}
            for ln, q_buf, k_buf in self.layers:
                files[("q", ln)] = (self.q_raw_paths[ln], "q",
                                    int(q_buf.shape[1]) * int(q_buf.element_size()))
                files[("k", ln)] = (self.k_raw_paths[ln], "k",
                                    int(k_buf.shape[1]) * int(k_buf.element_size()))
            self._wp = ApertureWritePath(
                run_dir, files, self.write_mode, allow_direct=self._ALLOW_DIRECT,
                direct_refusal=self._DIRECT_REFUSAL, label=f"qk aperture drain ({run_dir})",
                shape=self.shape)
            self._sidecar = QkSidecarLog()
            if self.gather:
                self._run_index = QkRunIndex(self._sidecar)
                if self._flush_ms > 0:
                    self.gather_stamp = True
                    self.run_id = new_run_id()
                    self._run_flusher = RunIndexFlusher(
                        self._run_index, run_dir, self.header, self._flush_ms,
                        name=f"qk run index ({run_dir})", run_id=self.run_id)
                    self._run_flusher.start()
                    self._gather_proc = ApertureGatherProcess.from_env(
                        run_dir, header=self.header, run_id=self.run_id, layout="qk")
        self._steps: List[StepMeta] = []
        self._pending_entries: List[QKStepEntry] = []
        self._closed = False

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
            note = self._wp.summary()
            if self.gather_stamp:
                note += (f" | qk hybrid delivery ON: run index qk_run_index.seg.* every "
                         f"{self._flush_ms} ms, run id {self.run_id}")
            return note
        return self._write_note

    def record_entries(self, entries: List) -> None:
        if self._sidecar is not None:
            self._pending_records.extend(entries)
            return
        self._pending_entries.extend(expand_qk_records(entries))

    def drain_once(self) -> int:
        """Copy pending aperture rows out of every q/k buffer, queue sidecar entries, advance the cursor."""
        moved = self.aperture.pending_rows()
        if moved == 0:
            return 0
        t0 = time.perf_counter()
        start = self.aperture._drain
        segments = self.aperture.drained_segments()
        wp = self._wp
        by_mode: Dict[str, int] = {}
        t_w = 0.0
        with self._io_lock:
            if wp.closed:
                raise ApertureWriteError(
                    f"qk aperture drain ({self.run_dir}): drain_once after close() closed the raw "
                    f"files -- flush_aperture must run after the last step")
            for ln, q_buf, k_buf in self.layers:
                for tag, buf in (("q", q_buf), ("k", k_buf)):
                    parts = [buf[s:e].detach().to("cpu") for s, e in segments]
                    rows = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
                    tw = time.perf_counter()
                    sink = wp.sinks[(tag, ln)]
                    n = sink.write(rows.contiguous())
                    t_w += time.perf_counter() - tw
                    by_mode[sink.mode] = by_mode.get(sink.mode, 0) + n
            tb = time.perf_counter()
            block = self._sidecar.prepare(self._pending_records)
            self._pending_records = []
            self._sidecar.commit(block)
            if self._run_index is not None:
                self._run_index.note_step(start, moved, block)
                self._release_deferred_finishes()
            t_b = time.perf_counter() - tb
        self.aperture.advance_drain(moved)
        step_s = time.perf_counter() - t0
        record_step_stats(self._wstats, "qk", rows=moved, step_s=step_s,
                          d2h_s=max(step_s - t_w - t_b, 0.0), write_s=t_w, write_tail_s=0.0,
                          busy_s=t_w, bookkeeping_s=t_b, bytes_by_mode=by_mode, wp=self._wp)
        return moved

    def note_gather_finish(self, req_id, asked: bool = False, meta: Optional[dict] = None) -> None:
        """Record a finished request for the gather's completion stamp; held until it is indexed."""
        if self._run_index is None or not self.gather_stamp:
            return
        with self._io_lock:
            if self._pending_records:
                self._deferred_finishes.append((str(req_id), bool(asked), meta))
            else:
                self._run_index.note_finish(req_id, asked=asked, meta=meta)

    def _release_deferred_finishes(self) -> None:
        """Note the held finishes (caller holds ``_io_lock``)."""
        if self._deferred_finishes:
            held, self._deferred_finishes = self._deferred_finishes, []
            for rid, asked, meta in held:
                self._run_index.note_finish(rid, asked=asked, meta=meta)

    def flush_run_index(self, *, up_to=None) -> Optional[str]:
        """Publish the next index segment now; return its path or None."""
        f = self._run_flusher
        return None if f is None else f.flush_once(up_to=up_to)

    def close(self) -> None:
        """Flush, truncate and release every q/k writer, then write the shared QK sidecar (idempotent)."""
        if self._wp is not None:
            if not self._closed:
                if self._deferred_finishes and self._run_index is not None:
                    with self._io_lock:
                        self._release_deferred_finishes()
                _close_write_path(self, "qk")
            return
        if self._closed:
            return
        write_qk_sidecar(self.meta_path, self._steps, self.header)
        self._closed = True
        release_run_lock(self)


class OffLoopQKApertureDrain(MultiLayerQKApertureDrain):
    """Off-loop (consumer-thread) sibling of ``MultiLayerQKApertureDrain``."""

    _ALLOW_DIRECT = True

    @releases_run_lock_on_failure
    def __init__(self, aperture, layers, run_dir: str, header: dict,
                 per_request: bool = False, index: Optional[PerRequestIndex] = None,
                 offload=None, disk_base: Optional[str] = None,
                 shape: Optional[WriteShape] = None):
        _threads = resolve_write_threads() if not per_request else 0
        super().__init__(aperture, layers, run_dir, header, setup_sink=not per_request,
                         shape=shape)
        self.per_request = bool(per_request)
        self.index: Optional[PerRequestIndex] = (
            index if index is not None
            else (PerRequestIndex() if self.per_request else None))
        self._offload = offload
        self._disk_base = disk_base or os.path.join(run_dir, "perreq")
        self._disk_routed: Dict[str, str] = {}
        self._disk_staging: Dict[str, _PerRequestQKDiskStaging] = {}
        self._disk_delivered_src: Dict[str, str] = {}
        self._disk_reclaim_pending: Dict[str, str] = {}
        self._disk_aborted: set = set()
        self._host_aborted: set = set()
        self._qk_kmeta: Dict[str, Dict[int, list]] = {}
        self._index_lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        dev = self.layers[0][1].device if self.layers else torch.device("cpu")
        self._is_cuda = (dev.type == "cuda") and torch.cuda.is_available()
        self._stream = torch.cuda.Stream(dev) if self._is_cuda else None
        self._aperture_depth = max(1, int(os.environ.get("MIA_CAPTURE_DRAIN_APERTURE", "3") or "3"))
        self._copy_events = ([torch.cuda.Event() for _ in range(self._aperture_depth)]
                             if self._stream is not None else [])
        self._aperture_idx = 0
        self._q_pinned: dict = {ln: None for ln, _, _ in self.layers}
        self._k_pinned: dict = {ln: None for ln, _, _ in self.layers}
        self._host: Dict[Tuple[str, int, int], torch.Tensor] = {}   # (q|k, layer, parity)
        self._parity = 0
        self._inflight: Optional["_InFlight"] = None
        self._held_finishes: list = []
        self._layer_events: list = []
        if self._wp is not None:
            if self._stream is not None:
                self._layer_events = [torch.cuda.Event() for _ in self.layers]
            try:
                self._wp.start_pool(_threads, self._stream.device if self._stream is not None
                                    else None, "mia-qk-aperture-write")
            except BaseException:
                self._wp.close()
                raise
        self._thread = threading.Thread(
            target=self._run, name="mia-qk-aperture-drain", daemon=True)
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

    def note_gather_finish(self, req_id, asked: bool = False, meta: Optional[dict] = None) -> None:
        """Queue the gather's completion stamp behind this request's rows."""
        if self.gather_stamp:
            self._q.put(_Finish(str(req_id), bool(asked), meta))

    def route_to_disk(self, req_id, dest, offload=None) -> None:
        """Route ``req_id`` to per-request disk staging, offloaded to ``dest`` when it finishes."""
        if not self.per_request:
            return
        req_id = str(req_id)
        new_offload = None
        if offload is None and self._offload is None:
            use_proc = os.environ.get("MIA_OFFLOAD_PROCESS", "0") == "1"
            new_offload = OffloadProcess(use_process=use_proc)
        with self._index_lock:
            if offload is not None:
                self._offload = offload
            elif self._offload is None and new_offload is not None:
                self._offload = new_offload
                new_offload = None
                # lazy: child_process reads env at import; keep it out of plugin load
                from mia.core.runtime.child_process import register_shutdown
                register_shutdown(self._offload.close)
            self._disk_routed[req_id] = str(dest)
        if new_offload is not None:
            new_offload.close()
        if _aperture_debug():
            _dbg(f"qk route_to_disk: req_id={req_id!r} (EXTERNAL) dest={dest!r} "
                 f"offload={type(self._offload).__name__}")

    def disk_residency(self) -> int:
        """Number of disk-routed requests still holding per-request staging state."""
        with self._index_lock:
            return len(self._disk_staging)

    def unlink_delivered_source(self, req_id) -> bool:
        """Remove the server-side staging dir of a delivered disk-routed request."""
        req_id = str(req_id)
        with self._index_lock:
            src = self._disk_delivered_src.pop(req_id, None)
        if src is None:
            return False
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
        for ext, src in to_rm:
            shutil.rmtree(src, ignore_errors=True)
            if _aperture_debug():
                _dbg(f"qk reclaim settled delivered-src: req={ext!r} src={src!r}")

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
            _dbg(f"qk mark_host_aborted: req={req_id!r} live={live} "
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
            if req_id in self._disk_staging or routed is not None:
                self._disk_aborted.add(req_id)
                marked = True
        reclaimed = False
        deferred = False
        if src is not None:
            if self._offload is None or self._offload.settled(req_id):
                shutil.rmtree(src, ignore_errors=True)
                reclaimed = True
            else:
                with self._index_lock:
                    self._disk_reclaim_pending[req_id] = src
                deferred = True
        if _aperture_debug():
            _dbg(f"qk clear_request_disk MARK-abort: req={req_id!r} marked={marked} "
                 f"delivered_src_reclaimed={reclaimed} deferred_reclaim={deferred} "
                 f"(consumer owns the staging-dir discard)")

    def _finalize_finish_isolated(self, req_id) -> None:
        try:
            self._handle_finish(req_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "qk off-loop aperture drain: per-request FINALIZE failed for req_id=%r; that request's "
                "delivery is dropped, the consumer continues", req_id)
            if _aperture_debug():
                _dbg(f"qk finish FAILED (isolated, consumer continues): req_id={req_id!r}")

    _POLL_S = 0.001

    def _next_item(self):
        """The next queue item; polls while a step is in flight so it can complete."""
        while True:
            if self._inflight is None:
                return self._q.get()
            try:
                return self._q.get(timeout=self._POLL_S)
            except queue.Empty:
                if all(f.done() for f in self._inflight.futs):
                    self._complete_inflight()

    def _run(self) -> None:
        try:
            bind_thread_to_device(self._stream.device if self._stream is not None else None)
            while True:
                item = self._next_item()
                if item is _STOP:
                    try:
                        self._complete_inflight()
                    finally:
                        self._q.task_done()
                    break
                owns_done = False                    # True: in-flight bookkeeping calls task_done
                try:
                    if isinstance(item, _Finish):
                        if self.gather_stamp:
                            if self._inflight is not None:
                                # stamp after the in-flight step is published
                                self._held_finishes.append(item)
                                owns_done = True
                            else:
                                self._run_index.note_finish(item.req_id, asked=item.asked,
                                                            meta=item.meta)
                        else:
                            self._complete_inflight()
                            self._finalize_finish_isolated(item.req_id)
                    else:
                        owns_done = self._wp is not None
                        self._drain_item(item)
                    self._reclaim_settled_pending()
                finally:
                    if not owns_done:
                        self._q.task_done()
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._abandon_inflight()
            logger.exception("qk off-loop aperture drain consumer thread died")

    def _pinned_buf(self, cache: dict, ln: int, n_rows: int, width: int, dtype) -> torch.Tensor:
        buf = cache[ln]
        if buf is None or buf.shape[0] < n_rows:
            buf = torch.empty(n_rows, width, dtype=dtype, pin_memory=self._is_cuda)
            cache[ln] = buf
        return buf[:n_rows]

    def _read_segments(self, segments: List[Tuple[int, int]], event):
        total = sum(e - s for s, e in segments)
        if self._stream is not None:
            slot = self._aperture_idx
            stale = self._copy_events[slot] if slot < len(self._copy_events) else None
            if stale is not None:
                stale.synchronize()
            if event is not None:
                self._stream.wait_event(event)
            pieces = []
            with torch.cuda.stream(self._stream):
                for ln, q_buf, k_buf in self.layers:
                    qbuf = self._pinned_buf(self._q_pinned, ln, total, q_buf.shape[1], q_buf.dtype)
                    kbuf = self._pinned_buf(self._k_pinned, ln, total, k_buf.shape[1], k_buf.dtype)
                    off = 0
                    for s, e in segments:
                        n = e - s
                        qsrc, ksrc = q_buf[s:e], k_buf[s:e]
                        qbuf[off:off + n].copy_(qsrc, non_blocking=True)
                        kbuf[off:off + n].copy_(ksrc, non_blocking=True)
                        qsrc.record_stream(self._stream)
                        ksrc.record_stream(self._stream)
                        off += n
                    pieces.append((ln, qbuf, kbuf))
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
        out = []
        for ln, q_buf, k_buf in self.layers:
            qp = [q_buf[s:e].detach().to("cpu") for s, e in segments]
            kp = [k_buf[s:e].detach().to("cpu") for s, e in segments]
            out.append((ln,
                        qp[0] if len(qp) == 1 else torch.cat(qp, dim=0),
                        kp[0] if len(kp) == 1 else torch.cat(kp, dim=0)))
        return out

    def _drain_item(self, item: _DrainItem) -> None:
        if self._wp is not None:
            self._drain_item_fast(item)
        else:
            self._drain_item_per_request(item)

    def _host_buf(self, tag: str, ln: int, n_rows: int, width: int, dtype,
                  parity: int = 0) -> torch.Tensor:
        key = (tag, ln, parity)
        buf = self._host.get(key)
        if buf is None or buf.shape[0] < n_rows:
            buf = alloc_host_rows(n_rows, width, dtype, pinned=self._is_cuda,
                                  align=self._wp.mem_align)
            self._host[key] = buf
        return buf[:n_rows]

    def _issue_d2h(self, segments: List[Tuple[int, int]], event, parity: int = 0) -> list:
        total = sum(e - s for s, e in segments)
        jobs = []
        if self._stream is not None:
            if event is not None:
                self._stream.wait_event(event)
            with torch.cuda.stream(self._stream):
                for i, (ln, q_buf, k_buf) in enumerate(self.layers):
                    qh = self._host_buf("q", ln, total, q_buf.shape[1], q_buf.dtype, parity)
                    kh = self._host_buf("k", ln, total, k_buf.shape[1], k_buf.dtype, parity)
                    off = 0
                    for s, e in segments:
                        n = e - s
                        qsrc, ksrc = q_buf[s:e], k_buf[s:e]
                        qh[off:off + n].copy_(qsrc, non_blocking=True)
                        kh[off:off + n].copy_(ksrc, non_blocking=True)
                        qsrc.record_stream(self._stream)
                        ksrc.record_stream(self._stream)
                        off += n
                    ev = self._layer_events[i]
                    ev.record(self._stream)
                    jobs.append((ln, qh, kh, ev))
            return jobs
        if event is not None:
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001
                pass
        for ln, q_buf, k_buf in self.layers:
            qh = self._host_buf("q", ln, total, q_buf.shape[1], q_buf.dtype, parity)
            kh = self._host_buf("k", ln, total, k_buf.shape[1], k_buf.dtype, parity)
            off = 0
            for s, e in segments:
                n = e - s
                qh[off:off + n].copy_(q_buf[s:e])
                kh[off:off + n].copy_(k_buf[s:e])
                off += n
            jobs.append((ln, qh, kh, None))
        return jobs

    @property
    def _pipelined(self) -> bool:
        """Only the hybrid-gather configuration pipelines."""
        return self._run_index is not None and self.gather_stamp

    def _drain_item_fast(self, item: _DrainItem) -> None:
        """Depth-1 pipeline: issue step N+1's d2h, complete step N, then submit N+1's writes."""
        prev = self._inflight
        try:
            ctx = self._begin_step(item, prev)
            if prev is not None:
                # held finishes were queued before this step, so their last step is <= prev
                self._inflight = None
                held, self._held_finishes = self._held_finishes, []
                self._finish_step(prev, held)
            new = self._submit_step(item, ctx)
        except BaseException:
            self._q.task_done()                       # never registered: nothing else releases it
            raise
        self._inflight = new
        if not self._pipelined:
            self._complete_inflight()

    def _begin_step(self, item: _DrainItem, prev: "Optional[_InFlight]"):
        t0 = time.perf_counter()
        aperture = self.aperture
        expect = aperture._drain + (prev.item.n_rows if prev is not None else 0)
        assert item.start_logical == expect, (
            f"FIFO drain violation: item.start_logical={item.start_logical} != "
            f"aperture._drain + in-flight rows = {expect}")
        parity = self._parity
        assert prev is None or prev.parity != parity, \
            "host buffer set still owned by an outstanding write"
        segments = aperture.segments_at(item.start_logical, item.n_rows)
        with self._io_lock:
            if self._wp.closed:
                raise ApertureWriteError(
                    f"qk aperture drain ({self.run_dir}): a step reached the consumer after close() "
                    f"closed the raw files -- flush_aperture (stop, then close) must follow the last "
                    f"step; this step's rows are NOT written and the aperture is not advanced")
            t1 = time.perf_counter()
            jobs = self._issue_d2h(segments, item.event, parity)
        return jobs, parity, t0, t1, time.perf_counter()

    def _submit_step(self, item: _DrainItem, ctx) -> "_InFlight":
        jobs, parity, t0, t1, t_a = ctx
        wp = self._wp
        tb = time.perf_counter()
        with self._io_lock:
            futs = []
            t_first = None
            try:
                for ln, qh, kh, ev in jobs:
                    if ev is not None:
                        ev.synchronize()
                    if t_first is None:
                        t_first = time.perf_counter()
                    for tag, h in (("q", qh), ("k", kh)):
                        sink = wp.sinks[(tag, ln)]
                        # offset claimed at submit, in step order
                        off = sink.claim(int(h.numel()) * int(h.element_size()))
                        futs.append(wp.pool.submit(timed_write_at, sink, h, off))
                t2 = time.perf_counter()
                block = self._sidecar.prepare(item.entries)
            except BaseException:
                join_writes_quietly(futs)
                raise
            t3 = time.perf_counter()
        self._parity ^= 1
        return _InFlight(item=item, futs=futs, block=block, parity=parity, t_first=t_first,
                         d2h_s=(t_a - t1) + (t2 - tb), bookkeeping_s=(t1 - t0) + (t3 - t2),
                         issue_s=(t_a - t0) + (time.perf_counter() - tb))

    def _complete_inflight(self) -> None:
        """Join, commit, publish and release the in-flight step, then stamp held finishes."""
        inf, self._inflight = self._inflight, None
        if inf is None:
            return
        held, self._held_finishes = self._held_finishes, []
        self._finish_step(inf, held)

    def _finish_step(self, inf: "_InFlight", held: list) -> None:
        try:
            tj = time.perf_counter()
            with self._io_lock:
                results = join_writes(inf.futs)        # rows durable ...
                t4 = time.perf_counter()
                self._sidecar.commit(inf.block)
                if self._run_index is not None:
                    self._run_index.note_step(inf.item.start_logical, inf.item.n_rows,
                                              inf.block)   # ... then published
            self.aperture.advance_drain(inf.item.n_rows)   # ... then released
            if held:
                for fin in held:
                    self._run_index.note_finish(fin.req_id, asked=fin.asked, meta=fin.meta)
            t5 = time.perf_counter()
            by_mode: Dict[str, int] = {}
            busy = 0.0
            for n, secs, mode in results:
                by_mode[mode] = by_mode.get(mode, 0) + n
                busy += secs
            record_step_stats(
                self._wstats, "qk", rows=inf.item.n_rows, step_s=inf.issue_s + (t5 - tj),
                d2h_s=inf.d2h_s, write_s=(t4 - inf.t_first) if inf.t_first is not None else 0.0,
                write_tail_s=t4 - tj, busy_s=busy, bookkeeping_s=inf.bookkeeping_s + (t5 - t4),
                bytes_by_mode=by_mode, wp=self._wp)
        finally:
            for _ in range(1 + len(held)):
                self._q.task_done()

    def _abandon_inflight(self) -> None:
        """Wait out the in-flight step's writes without publishing it; release its queue slot."""
        inf, self._inflight = self._inflight, None
        held, self._held_finishes = self._held_finishes, []
        if inf is not None:
            try:
                join_writes_quietly(inf.futs)
            finally:
                for _ in range(1 + len(held)):
                    self._q.task_done()

    def _drain_item_per_request(self, item: _DrainItem) -> None:
        t0 = time.perf_counter()
        aperture = self.aperture
        assert item.start_logical == aperture._drain, (
            f"FIFO drain violation: item.start_logical={item.start_logical} != "
            f"aperture._drain={aperture._drain}")
        segments = aperture.segments_at(item.start_logical, item.n_rows)
        t1 = time.perf_counter()
        with PROF.timed("bank.consumer.d2h"):
            pieces = self._read_segments(segments, item.event)
        t2 = time.perf_counter()
        t3 = t2
        self._demux_into_index(item, pieces)
        aperture.advance_drain(item.n_rows)
        t4 = time.perf_counter()
        record_step_stats(
            self._wstats, "qk", rows=item.n_rows, step_s=t4 - t0, d2h_s=t2 - t1,
            write_s=t3 - t2, write_tail_s=t3 - t2, busy_s=t3 - t2,
            bookkeeping_s=(t1 - t0) + (t4 - t3), bytes_by_mode={})

    def _demux_into_index(self, item: _DrainItem, pieces) -> None:
        by_layer = {ln: (q, k) for ln, q, k in pieces}
        base = int(item.start_logical)
        entries = expand_qk_records(item.entries)
        with self._index_lock:
            routed_keys = tuple(self._disk_routed) if self._disk_routed else ()
        any_disk = bool(routed_keys)
        staged = []
        for e in entries:
            qk = by_layer.get(e.layer)
            if qk is None:
                continue
            q_layer_rows, k_layer_rows = qk
            k_off = int(e.k_start) - base
            k_slice = k_layer_rows[k_off:k_off + int(e.k_rows)]
            if int(e.q_rows) > 0 and int(e.q_start) >= 0:
                q_off = int(e.q_start) - base
                q_slice = q_layer_rows[q_off:q_off + int(e.q_rows)]
            else:
                q_slice = None
            ext = _match_disk_route(e.req_id, routed_keys) if any_disk else None
            if ext is not None:
                if _aperture_debug():
                    _dbg(f"qk demux DISK hit: entry.req_id={e.req_id!r} -> route={ext!r} "
                         f"layer={e.layer} k_rows={int(e.k_rows)} q_rows={int(e.q_rows)}")
                try:
                    self._disk_write(ext, e.layer, q_slice, k_slice,
                                     int(e.prefix_end), int(e.num_computed))
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "qk off-loop aperture drain: per-request DISK demux write failed for req=%r "
                        "layer=%s; that request's disk delivery is dropped + its staging reclaimed, "
                        "the consumer continues", ext, e.layer)
                    with self._index_lock:
                        if ext in self._disk_staging or ext in self._disk_routed:
                            self._disk_aborted.add(ext)
                    if _aperture_debug():
                        _dbg(f"qk demux DISK write FAILED (isolated): req={ext!r} layer={e.layer}")
            else:
                if any_disk and _aperture_debug():
                    _dbg(f"qk demux DISK miss: entry.req_id={e.req_id!r} not in routes "
                         f"{list(routed_keys)} -> host index")
                staged.append((e.req_id, int(e.layer),
                               None if q_slice is None else q_slice.clone(),
                               k_slice.clone(), int(e.prefix_end), int(e.num_computed)))
        with self._index_lock:
            ab_host = tuple(self._host_aborted) if self._host_aborted else ()
            ab_disk = tuple(self._disk_aborted) if self._disk_aborted else ()
            for req_id, layer, q_clone, k_clone, prefix_end, num_computed in staged:
                if ((ab_host and _match_disk_route(req_id, ab_host) is not None)
                        or (ab_disk and _match_disk_route(req_id, ab_disk) is not None)):
                    if _aperture_debug():
                        _dbg(f"qk demux HOST-SKIP aborted: req={req_id!r} layer={layer}")
                    continue
                kmeta = None
                if prefix_end >= 0:
                    lst = self._qk_kmeta.setdefault(req_id, {}).setdefault(layer, [])
                    lst.append(prefix_end)
                    kmeta = {"prefix_ends": list(lst)}
                self.index.note_rows(req_id, ("k", layer), k_clone, kmeta=kmeta,
                                     num_computed=num_computed)
                if q_clone is not None:
                    self.index.note_rows(req_id, ("q", layer), q_clone)

    def _disk_write(self, req_id, layer, q_slice, k_slice, prefix_end: int, num_computed: int) -> None:
        with self._index_lock:
            if req_id in self._disk_aborted or req_id not in self._disk_routed:
                if _aperture_debug():
                    _dbg(f"qk disk_write SKIP (aborted/unrouted): req={req_id!r} layer={layer}")
                return
            stg = self._disk_staging.get(req_id)
            if stg is None:
                stg = _PerRequestQKDiskStaging(
                    req_id, os.path.join(self._disk_base, _sanitize_req_id(req_id)),
                    self.header)
                self._disk_staging[req_id] = stg
        stg.append(layer, q_slice, k_slice, prefix_end, num_computed)

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
                self._qk_kmeta.pop(aborted_ext, None)
                stg_abort = self._disk_staging.pop(aborted_ext, None)
                ext = None
                disk_dest = None
                stg = None
            else:
                stg_abort = None
                ext = _match_disk_route(req_id, tuple(self._disk_routed))
                disk_dest = self._disk_routed.pop(ext, None) if ext is not None else None
                stg = self._disk_staging.pop(ext, None) if disk_dest is not None else None
                if disk_dest is not None and stg is None:
                    # a routed request that captured no rows: deliver an empty sidecar (zero rows)
                    stg = _PerRequestQKDiskStaging(
                        req_id, os.path.join(self._disk_base, _sanitize_req_id(req_id)),
                        self.header)
                if stg is not None and self._offload is not None:
                    self._disk_delivered_src[ext] = stg.run_dir
        if aborted_ext is not None:
            if stg_abort is not None:
                stg_abort.discard()
            if _aperture_debug():
                _dbg(f"qk finish ABORT-reclaim: id={req_id!r} route={aborted_ext!r} "
                     f"discarded={stg_abort is not None} (single-owner consumer discard)")
            return
        if disk_dest is not None:
            if _aperture_debug():
                _dbg(f"qk finish DISK: id={req_id!r} route={ext!r} "
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
                self._qk_kmeta.pop(req_id, None)
                if _aperture_debug():
                    _dbg(f"qk finish HOST-abort drop: id={req_id!r} mark={host_ab!r}")
                return
            if req_id in self.index.live_req_ids():
                self.index.mark_finished(req_id)
                self._qk_kmeta.pop(req_id, None)

    def finalize_all(self) -> None:
        """End of run only: mark every live request finished so pop_deliverable_qk can return it."""
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
            self._qk_kmeta.clear()

    def aperture_residency(self) -> "Tuple[int, int]":
        """Non-destructive (host_live_count, disk_residency) for this drain."""
        with self._index_lock:
            host_live = len(self.index.live_req_ids()) if self.index is not None else 0
        disk = int(self.disk_residency())
        return (int(host_live), disk)

    def close(self) -> None:
        """Stop a live consumer, then close the files and write the sidecar; re-raise consumer errors."""
        if self._closed:
            return super().close()
        err: Optional[BaseException] = None
        if self._started:
            try:
                self.stop()
            except BaseException as e:  # noqa: BLE001
                err = e
        inf = self._inflight
        if inf is not None:
            # wait out the in-flight pwrites before the files close
            join_writes_quietly(inf.futs)
            if not self._thread.is_alive():
                self._abandon_inflight()
        try:
            super().close()
        except BaseException:
            if err is None:
                raise
            logger.exception("qk aperture drain close failed after a consumer failure")
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
                "qk off-loop aperture drain consumer thread failed; captured QK may be incomplete"
            ) from self._error

