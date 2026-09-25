"""Multi-layer host drain for the QK capture aperture (QK port of aperture_drain_hs).

QK captures TWO tensors per token: post-RoPE ``q`` and ``k``, scattered by ``capture_qk`` into
per-layer static buffers at the SAME routed index — TWO parallel per-layer apertures (``q_buf``,
``k_buf``) sharing ONE logical cursor. Each step reserves ``qlen`` rows (K needs every key); this
drain reads the new ``[drain, write)`` region from BOTH buffers per layer and appends to that
layer's two raw files plus one shared sidecar (``qk_aperture_meta.jsonl``). The ``q`` slots on a
non-emit (``last_token`` mid-prefill) step are written but never referenced by the sidecar — dead,
harmless; only q_start/q_rows/prefix_end distinguish an emit_q step from a keep-K-only step.
Mirrors ``aperture_drain_hs`` (sync + off-loop drains) and reuses its byte-sink helpers and off-loop
cross-stream discipline verbatim; only the per-layer buffer count (2, not 1) differs.

WRITE PATH (``MIA_APERTURE_WRITE_MODE``, default ``auto``): as in ``aperture_drain_hs`` -- run-long
sinks, zero-copy writes, O_DIRECT per file where the rows are a multiple of the detected block size
(the q files usually are; k rows are narrow -- 512 B at 70B TP4, 256 B at TP8 -- and take the
buffered path when they are not), per-layer D2H overlapped with a writer-thread pool, and the
sidecar kept as per-step arrays (``QkSidecarLog``). ``legacy`` is the old path, kept verbatim for
A/B validation only.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch

from mia._profiler import PROF
from .capture_aperture import CaptureAperture
from .per_request_delivery import PerRequestIndex
from .aperture_drain_hs import (
    _MmapLayerWriter,
    _STOP,
    _DrainItem,
    _Finish,
    _close_write_path,
    _dbg,
    _match_disk_route,
    _raw_bytes,
    _aperture_debug,
    _sanitize_req_id,
    _torch_dtype_name,
)
from .aperture_metadata import (
    QKStepEntry, QkSidecarLog, StepMeta, expand_qk_records, write_qk_sidecar)
from .aperture_sink import (
    ApertureWriteConfigError, ApertureWriteError, ApertureWritePath, PerRequestSinks,
    WRITE_MODE_ENV, WriteShape, WriteStats, alloc_host_rows, join_writes, join_writes_quietly,
    record_step_stats, resolve_per_request_write_mode, resolve_write_mode, resolve_write_threads,
    timed_write)
from .thread_device import bind_thread_to_device

logger = logging.getLogger(__name__)

_GIB = 1024 ** 3


def _resolve_qk_mmap_capacity_bytes(n_slots: int, row_bytes: int) -> int:
    """Per-layer raw-file mmap pre-size for a q OR k file (see aperture_drain_hs._resolve_mmap_capacity_bytes):
    ``MIA_APERTURE_MMAP_BYTES`` overrides outright, else ``max(2 GiB, n_slots * row_bytes)``. A
    STARTING size, not a hard cap — ``_MmapLayerWriter`` falls back to a plain append past it."""
    override = os.environ.get("MIA_APERTURE_MMAP_BYTES")
    if override:
        return int(override)
    return max(2 * _GIB, int(n_slots) * int(row_bytes))


class _PerRequestQKDiskStaging:
    """QK DISK route's per-request staging (QK port of ``_PerRequestDiskStaging``): stream ONE
    request's demuxed q + k rows to its OWN per-request run_dir laid out exactly like the shared QK
    run — per-layer ``qk_q_layer_<L>.raw`` + ``qk_k_layer_<L>.raw`` (via reused
    :class:`_MmapLayerWriter`s) + a per-request ``qk_aperture_meta.jsonl`` sidecar — so
    ``aperture_reader.load_multilayer_qk_aperture_artifact(run_dir)`` reconstructs that single request
    byte-identically.

    RELABEL INVARIANT: because only THIS request writes these files, each ``QKStepEntry``'s
    ``k_start`` / ``q_start`` is the RUNNING per-``(req, layer)`` row count into its own file (0, then
    n_rows, ...) — the exact offset the QK reader keys on, now scoped to one request. (The shared-file
    drain instead uses the global aperture cursor as the slot; here we RELABEL to a per-request-local
    offset.) ``prefix_end`` (the request's cumulative key count) and ``num_computed`` (its cached-prefix
    length) are ALREADY per-request, so they pass through UNCHANGED — and the reader's first-step
    ``num_computed > 0`` deferral guard fires identically.

    Q is written COMPACTLY (only the emitted q rows, matching the host ``assemble_qk`` path), so a
    ``last_token`` non-emit step appends k only; its entry carries ``q_start=-1, q_rows=0``. K is
    appended EVERY step (its rows concatenate to ``k_full``).

    Written entirely on the drain's CONSUMER thread (one writer per request), so the row appends need
    no lock; the drain guards only the ``_disk_staging`` dict membership it lives in.

    WRITE PATH (``write_mode``, resolved ONCE by the drain from ``MIA_APERTURE_WRITE_MODE``), the
    QK twin of :class:`~mia.graph.aperture_drain_hs._PerRequestDiskStaging`'s: ``buffered`` -- what
    ``auto``, the default, resolves to here -- writes through :class:`PerRequestSinks` (zero-copy
    ``pwrite``, one run-long fd per q/k file, no ``bytes`` copy, no open/close per step);
    ``legacy`` keeps the pre-2026-09-20 sequence verbatim for A/B validation. Same bytes, same
    files, same ``QKStepEntry`` offsets, same sidecar either way. No O_DIRECT: these writes are one
    layer's rows for one request, issued inline on the drain thread with no writer pool, where job
    1802981 measures direct 1.65-1.74x slower per write."""

    def __init__(self, req_id: str, run_dir: str, header: dict, capacity_bytes: int,
                 use_mmap: bool, write_mode: str = "legacy"):
        self.req_id = str(req_id)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.header = dict(header)
        self.meta_path = os.path.join(run_dir, "qk_aperture_meta.jsonl")
        self._capacity = int(capacity_bytes)
        self.write_mode = str(write_mode)
        self._legacy = self.write_mode == "legacy"
        self._use_mmap = bool(use_mmap) and self._legacy
        self._sinks = None if self._legacy else PerRequestSinks(f"qk staging req {req_id}")
        self._q_writers: Dict[int, _MmapLayerWriter] = {}   # layer -> q mmap writer (legacy)
        self._k_writers: Dict[int, _MmapLayerWriter] = {}   # layer -> k mmap writer (legacy)
        self._q_plain: Dict[int, str] = {}                  # layer -> q raw path (legacy plain)
        self._k_plain: Dict[int, str] = {}                  # layer -> k raw path (legacy plain)
        self._q_rows: Dict[int, int] = {}                   # layer -> cumulative q rows == next q_start
        self._k_rows: Dict[int, int] = {}                   # layer -> cumulative k rows == next k_start
        self._entries: List[QKStepEntry] = []
        self._closed = False

    def _q_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"qk_q_layer_{layer}.raw")

    def _k_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"qk_k_layer_{layer}.raw")

    def _append_one(self, writers: dict, plain: dict, path_fn, layer: int,
                    rows_cpu: torch.Tensor, which: str) -> None:
        if self._sinks is not None:
            self._sinks.append((which, layer), path_fn(layer), rows_cpu)
        elif self._use_mmap:
            w = writers.get(layer)
            if w is None:
                w = _MmapLayerWriter(path_fn(layer), self._capacity)
                writers[layer] = w
            w.append(_raw_bytes(rows_cpu))
        else:
            p = plain.get(layer)
            if p is None:
                p = path_fn(layer)
                open(p, "wb").close()   # truncate up front: never append onto stale bytes
                plain[layer] = p
            with open(p, "ab") as f:
                f.write(_raw_bytes(rows_cpu))

    def append(self, layer: int, q_rows_cpu, k_rows_cpu: torch.Tensor,
               prefix_end: int, num_computed: int) -> None:
        """Append ONE step's already-on-host rows for this ``(req, layer)`` to its per-request q/k
        files at the per-request-local relabeled offset, recording the matching ``QKStepEntry``. The
        write is SYNCHRONOUS on this thread (a zero-copy ``pwrite``, or the ``_raw_bytes`` copy
        under ``legacy``), so the caller's source view is fully consumed before the aperture frees
        it — no clone needed. ``q_rows_cpu`` is None on a non-emit
        (``last_token`` mid-prefill) step: append k only, record ``q_start=-1, q_rows=0``."""
        k_start = self._k_rows.get(layer, 0)
        k_n = int(k_rows_cpu.shape[0])
        self._append_one(self._k_writers, self._k_plain, self._k_path, layer, k_rows_cpu, "k")
        self._k_rows[layer] = k_start + k_n
        if q_rows_cpu is not None and int(q_rows_cpu.shape[0]) > 0:
            q_start = self._q_rows.get(layer, 0)
            q_n = int(q_rows_cpu.shape[0])
            self._append_one(self._q_writers, self._q_plain, self._q_path, layer, q_rows_cpu, "q")
            self._q_rows[layer] = q_start + q_n
        else:
            q_start, q_n = -1, 0
        self._entries.append(QKStepEntry(
            req_id=self.req_id, layer=int(layer),
            k_start=int(k_start), k_rows=int(k_n),
            q_start=int(q_start), q_rows=int(q_n),
            prefix_end=int(prefix_end), num_computed=int(num_computed)))

    def close(self) -> None:
        """Finalize on the request's FINISH: msync+truncate every per-layer q/k mmap writer, then
        write this request's QK sidecar. Idempotent. TOLERATES A PARTIAL / ABORTED STAGING: the
        sidecar references ONLY the layers actually appended, and each writer
        close AND the sidecar write are BEST-EFFORT (a vanished run_dir / partial mmap must never raise
        out of ``_handle_finish`` and wedge the off-loop consumer)."""
        if self._closed:
            return
        self._closed = True
        if self._sinks is not None:
            self._sinks.close()          # close this request's run-long fds (best-effort, logged)
        for w in (*self._q_writers.values(), *self._k_writers.values()):
            try:
                w.close()
            except Exception:            # noqa: BLE001 -- best-effort msync of a partial/aborted writer
                logger.exception(
                    "qk per-request staging: writer close failed for req %r (partial staging); "
                    "continuing", self.req_id)
        try:
            if os.path.isdir(self.run_dir):
                write_qk_sidecar(self.meta_path, [StepMeta(list(self._entries))], self.header)
        except Exception:                # noqa: BLE001 -- a partial/vanished dir must never wedge finish
            logger.exception(
                "qk per-request staging: sidecar write failed for req %r under %r (partial/aborted "
                "staging); delivery skipped", self.req_id, self.run_dir)

    def discard(self) -> None:
        """ABORT cleanup: release this request's open q/k writers WITHOUT writing a sidecar (an aborted
        request is never delivered/read), then remove its staging dir. Idempotent; best-effort."""
        if self._sinks is not None:
            self._sinks.close()
        for w in (*self._q_writers.values(), *self._k_writers.values()):
            try:
                w.close()
            except Exception:  # noqa: BLE001 -- best-effort release of a partial mmap
                pass
        self._q_writers = {}
        self._k_writers = {}
        self._closed = True
        import shutil
        shutil.rmtree(self.run_dir, ignore_errors=True)


class MultiLayerQKApertureDrain:
    """Drains a shared-cursor ``CaptureAperture`` across N per-layer ``(q_buf, k_buf)`` pairs.

    ``layers`` is ``[(layer_num, q_buf, k_buf), ...]`` in layer order (``layer_num`` is 0-based, ==
    the eager qkv_hook's ``match_attn`` layer number). Each drain appends the SAME ``[drain, write)``
    rows from every layer's q_buf and k_buf to that layer's two raw files (the shared logical row
    offset is the row offset into EVERY per-layer file — the invariant the reader keys on).
    ``record_entries`` queues this step's ``QKStepEntry`` records; ``drain_once`` copies the pending
    region, appends per layer (q + k), advances the shared drain cursor, and returns rows moved.
    ``close`` writes the accumulated shared sidecar.
    """

    # See MultiLayerApertureDrain: the synchronous drain writes pageable host copies (no O_DIRECT).
    _ALLOW_DIRECT = False
    _DIRECT_REFUSAL = ("the synchronous drain (MIA_APERTURE_SYNC_DRAIN=1) writes pageable host "
                       "copies with no alignment guarantee; it writes zero-copy buffered")

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
        # WRITE PATH (MIA_APERTURE_WRITE_MODE, default auto) -- same contract as the HS drain:
        # `legacy` keeps the sink below byte for byte; every other mode opens each q/k raw file ONCE
        # (O_DIRECT per file where aligned, else zero-copy buffered) and keeps the sidecar as arrays.
        self.write_mode, self._write_mode_explicit = resolve_write_mode()
        # What one drained step is predicted to write per raw file, from the capture CONFIG
        # (install_qk computes it; None = not predicted, and `auto` then decides on alignment
        # alone, as it did before 2026-09-20). Only `auto` consults it.
        self.shape = shape
        # The write path a DISK-routed request's per-request staging takes (per-request delivery
        # only; `legacy` for every shared-file drain, which never builds one).
        self._perreq_write_mode = "legacy"
        self._wp: Optional[ApertureWritePath] = None
        self._sidecar: Optional[QkSidecarLog] = None
        self._io_lock = threading.Lock()
        self._wstats = WriteStats()
        self._write_note = ""
        self._pending_records: list = []
        # mmap-NVMe raw sink, default OFF (MIA_APERTURE_MMAP=1 opts in). Same semantics + fallback
        # as the HS drain (plain GIL-releasing write() is the default); one writer per q AND k file.
        # ``setup_sink=False`` (the per-request delivery mode, OffLoopQKApertureDrain(per_request=True))
        # skips the shared per-layer raw files entirely — those rows are demuxed by req_id into a
        # PerRequestIndex instead, so opening/pre-sizing the shared sink would be pure waste. The
        # default (True) path is byte-for-byte identical to before this param existed.
        self._mmap_enabled = False
        self._q_writers: Dict[int, _MmapLayerWriter] = {}
        self._k_writers: Dict[int, _MmapLayerWriter] = {}
        if not setup_sink:
            # Per-request delivery writes no shared raw file, so the per-file O_DIRECT decision has
            # nothing to decide. Its DISK sub-route's per-request staging does write, and takes the
            # same zero-copy writer in buffered mode (`_perreq_write_mode`).
            self._perreq_write_mode = resolve_per_request_write_mode(
                self.write_mode, self._write_mode_explicit)
            self._write_note = (
                f"write mode={self.write_mode} does not apply to shared raw files: per-request "
                f"delivery (MIA_APERTURE_PER_REQUEST=1) writes none. Its disk staging writes "
                + ("zero-copy buffered, one fd per q/k file kept open for the request"
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
        else:
            self._mmap_enabled = os.environ.get("MIA_APERTURE_MMAP", "0") != "0"
            if self._mmap_enabled:
                try:
                    for ln, q_buf, k_buf in self.layers:
                        q_cap = _resolve_qk_mmap_capacity_bytes(
                            aperture.n_slots, q_buf.shape[1] * q_buf.element_size())
                        k_cap = _resolve_qk_mmap_capacity_bytes(
                            aperture.n_slots, k_buf.shape[1] * k_buf.element_size())
                        self._q_writers[ln] = _MmapLayerWriter(self.q_raw_paths[ln], q_cap)
                        self._k_writers[ln] = _MmapLayerWriter(self.k_raw_paths[ln], k_cap)
                except OSError as e:
                    logger.warning(
                        "qk aperture mmap sink: failed to mmap raw file(s) under %s (%s); falling back to "
                        "the plain append path for the whole run (unset MIA_APERTURE_MMAP to silence)",
                        run_dir, e)
                    for w in (*self._q_writers.values(), *self._k_writers.values()):
                        try:
                            w.close()
                        except Exception:  # noqa: BLE001
                            pass
                    self._q_writers = {}
                    self._k_writers = {}
                    self._mmap_enabled = False
            if not self._mmap_enabled:
                for p in (*self.q_raw_paths.values(), *self.k_raw_paths.values()):
                    open(p, "wb").close()
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
        """The install line's body: write mode per tensor kind, writer threads, O_DIRECT block."""
        if self._wp is not None:
            return self._wp.summary()
        if self._write_note:
            return self._write_note
        sink = ("pre-sized MAP_SHARED mmap (MIA_APERTURE_MMAP=1)" if self._mmap_enabled
                else "tobytes + open/append/close per step")
        return (f"write mode=legacy -> q=legacy, k=legacy ({sink}, on the drain thread) | for A/B "
                f"validation only ({WRITE_MODE_ENV}=legacy)")

    def record_entries(self, entries: List) -> None:
        # `entries` is per-request QKReqCaptureRecord (or already-flat QKStepEntry for a direct-drain
        # caller); expand_qk_records fans each record into the flat per-(req, layer) QKStepEntry
        # list. Runs on-loop for the sync drain; the off-loop path expands in the consumer thread.
        # Non-legacy modes keep the records as they are: the sidecar log derives the entries.
        if self._sidecar is not None:
            self._pending_records.extend(entries)
            return
        self._pending_entries.extend(expand_qk_records(entries))

    def _append(self, writers: dict, raw_paths: dict, ln: int, rows_cpu: torch.Tensor) -> None:
        data = _raw_bytes(rows_cpu)
        writer = writers.get(ln) if self._mmap_enabled else None
        if writer is not None:
            writer.append(data)
        else:
            with open(raw_paths[ln], "ab") as f:
                f.write(data)

    def drain_once(self) -> int:
        """Copy the aperture's pending ``[drain, write)`` rows out of every per-layer q_buf AND k_buf,
        append them per layer, queue this step's sidecar entries, and advance the shared drain
        cursor. Returns rows moved (0 if nothing pending)."""
        if self._wp is not None:
            return self._drain_once_fast()
        moved = self.aperture.pending_rows()
        if moved == 0:
            return 0
        segments = self.aperture.drained_segments()
        for ln, q_buf, k_buf in self.layers:
            qp = [q_buf[s:e].detach().to("cpu") for s, e in segments]
            kp = [k_buf[s:e].detach().to("cpu") for s, e in segments]
            self._append(self._q_writers, self.q_raw_paths, ln,
                         qp[0] if len(qp) == 1 else torch.cat(qp, dim=0))
            self._append(self._k_writers, self.k_raw_paths, ln,
                         kp[0] if len(kp) == 1 else torch.cat(kp, dim=0))
        if self._pending_entries:
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
            t_b = time.perf_counter() - tb
        self.aperture.advance_drain(moved)
        step_s = time.perf_counter() - t0
        record_step_stats(self._wstats, "qk", rows=moved, step_s=step_s,
                          d2h_s=max(step_s - t_w - t_b, 0.0), write_s=t_w, write_tail_s=0.0,
                          busy_s=t_w, bookkeeping_s=t_b, bytes_by_mode=by_mode, wp=self._wp)
        return moved

    def close(self) -> None:
        """Flush+truncate+release every mmap writer (q + k), then write the shared QK sidecar
        (idempotent; safe to call from ``flush_aperture`` + atexit)."""
        if self._wp is not None:
            if not self._closed:
                _close_write_path(self, "qk")
            return
        if self._closed:
            return
        for w in (*self._q_writers.values(), *self._k_writers.values()):
            w.close()
        write_qk_sidecar(self.meta_path, self._steps, self.header)
        self._closed = True


class OffLoopQKApertureDrain(MultiLayerQKApertureDrain):
    """Off-loop (consumer-thread) sibling of ``MultiLayerQKApertureDrain`` — the QK analogue of
    ``OffLoopApertureDrain``.

    The engine loop, per active step, does an O(1) ``enqueue(entries, start_logical, n_rows, event)``
    and does NOT drain. A dedicated CONSUMER THREAD waits the step's scatter event, reads each
    per-layer q_buf AND k_buf ``[start_logical, start_logical+n_rows)`` region on a DEDICATED COPY
    STREAM (``record_stream`` guards the source rows), writes per-layer q + k raw + sidecar, then
    ``advance_drain(n_rows)`` — which FREES aperture rows and so RELEASES the engine's reserve
    backpressure. Byte-identical to the sync path (reads only committed ``[drain, write)`` rows,
    fenced by the event; FIFO invariant keeps every file's row offset == the logical cursor).

    TWO consumer modes (``per_request``, default OFF — additive, the default path is unchanged; the
    QK port of ``OffLoopApertureDrain``'s per-request delivery):
      * shared-file (default): appends each layer's drained q + k rows to its two raw files + sidecar.
      * per-request (``per_request=True``): demuxes each step's q + k rows BY req_id into a
        ``PerRequestIndex`` via the ``("q", layer)`` / ``("k", layer)`` staging convention
        (``_demux_into_index``) and consumes ``_Finish`` items (``enqueue_finish`` -> ``_handle_finish``
        -> ``mark_finished``) to drive per-request ``assemble_qk`` delivery, writing NO shared file.

    DISK SUB-ROUTE (WITHIN per-request mode; INACTIVE unless ``route_to_disk`` is called): a request the
    router marks streams its demuxed q + k rows to its OWN per-request run_dir
    (``_PerRequestQKDiskStaging``: per-layer ``qk_q_layer_<L>.raw`` + ``qk_k_layer_<L>.raw`` + a
    per-request QK sidecar) instead of the host-buffer index, and on ``_Finish`` the file is msync'd and
    handed to an ``OffloadProcess`` for transfer to ``dest``, then freed. All the concurrency fixes from
    ``OffLoopApertureDrain`` are ported: id-divergence match (``_match_disk_route``), per-request finalize
    isolation, partial-staging tolerance, single-owner staging-dir lifecycle, host-residency abort marks.
    A per-request run with NO disk routes is byte-identical to the host-buffer path.
    """

    # The off-loop drain D2Hs into its own host buffers, allocated at the O_DIRECT alignment.
    _ALLOW_DIRECT = True

    def __init__(self, aperture, layers, run_dir: str, header: dict,
                 per_request: bool = False, index: Optional[PerRequestIndex] = None,
                 offload=None, disk_base: Optional[str] = None,
                 shape: Optional[WriteShape] = None):
        # Writer-thread count, validated BEFORE the base class opens any raw file.
        _threads = (resolve_write_threads()
                    if not per_request and resolve_write_mode()[0] != "legacy" else 0)
        # per_request (GATED, default OFF): when ON the consumer demuxes each step's q + k rows BY
        # req_id into a PerRequestIndex (the ("q", layer) / ("k", layer) staging convention) and
        # enqueue_finish() drives QK assembly, INSTEAD of writing the shared per-layer files. Default
        # OFF keeps the shared-file drain byte-for-byte unchanged (MultiLayerQKApertureDrain's mmap sink
        # follows per_request via setup_sink=not per_request below).
        super().__init__(aperture, layers, run_dir, header, setup_sink=not per_request,
                         shape=shape)
        self.per_request = bool(per_request)
        self.index: Optional[PerRequestIndex] = (
            index if index is not None
            else (PerRequestIndex() if self.per_request else None))
        # DISK ROUTE (a per-request SUB-mode, default INACTIVE): identical maps + single-owner
        # staging-dir lifecycle as OffLoopApertureDrain (HS). Both maps stay empty until route_to_disk()
        # is called, so a per_request run with NO disk routes is byte-identical to the host path.
        self._offload = offload
        self._disk_base = disk_base or os.path.join(run_dir, "perreq")
        self._disk_routed: Dict[str, str] = {}
        self._disk_staging: Dict[str, _PerRequestQKDiskStaging] = {}
        self._disk_delivered_src: Dict[str, str] = {}
        # DEFERRED settled-reclaim (rmtree-vs-offload-read race fix, mirror of OffLoopApertureDrain):
        # delivered-source dirs clear_request_disk found the offload STILL READING (copytree in
        # flight) when a confirm TIMEOUT dropped the confirm-path unlink; reclaimed once the offload
        # SETTLES (_reclaim_settled_pending). EMPTY on the happy path -> a strict no-op. Guarded by
        # _index_lock.
        self._disk_reclaim_pending: Dict[str, str] = {}
        self._disk_aborted: set = set()
        self._host_aborted: set = set()
        # Per-(req_id, layer) RUNNING cumulative prefix_ends list, used to build the LAST-WRITE-WINS
        # kmeta the ("k", layer) note_rows stores (assemble_qk reads it at finish). Consumer-thread-
        # owned like the disk maps; guarded by _index_lock for uniformity, cleared per-req on finish.
        self._qk_kmeta: Dict[str, Dict[int, list]] = {}
        self._perreq_cap = int(os.environ.get(
            "MIA_APERTURE_PERREQ_MMAP_BYTES", str(64 * 1024 * 1024)) or (64 * 1024 * 1024))
        self._perreq_mmap = os.environ.get("MIA_APERTURE_MMAP", "0") != "0"
        # Drain-OWNED lock serializing EVERY access to the shared PerRequestIndex + disk/abort maps
        # (same discipline + rationale as OffLoopApertureDrain: plain Lock, never nested, never held
        # across close/rmtree/copytree/D2H).
        self._index_lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        dev = self.layers[0][1].device if self.layers else torch.device("cpu")
        self._is_cuda = (dev.type == "cuda") and torch.cuda.is_available()
        self._stream = torch.cuda.Stream(dev) if self._is_cuda else None
        self._aperture_depth = max(1, int(os.environ.get("MIA_CAPTURE_DRAIN_APERTURE", "3") or "3"))
        self._copy_events = ([torch.cuda.Event() for _ in range(self._aperture_depth)]
                             if self._stream is not None else [])
        self._aperture_idx = 0
        # Per-layer PERSISTENT pinned host buffers (q + k), reused each step (grown on demand).
        self._q_pinned: dict = {ln: None for ln, _, _ in self.layers}
        self._k_pinned: dict = {ln: None for ln, _, _ in self.layers}
        # Non-legacy write path (see OffLoopApertureDrain): aligned per-layer q/k host buffers, one
        # completion event per layer (after its q AND k copies), and the writer pool.
        self._host: Dict[Tuple[str, int], torch.Tensor] = {}
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
        """O(1) hand-off. ``entries`` ownership TRANSFERS to the queue item (the caller reassigns
        ``registry._qk_step_entries = []``, so the old list is owned solely here)."""
        with PROF.timed("graph.enqueue"):
            self._q.put(_DrainItem(entries, int(start_logical), int(n_rows), event))

    def enqueue_finish(self, req_id) -> None:
        """O(1) hand-off of a per-request FINISH (reuses the HS ``_Finish`` item). NO-OP unless
        per_request mode is on. Must be enqueued AFTER the request's last row-entries so the consumer
        marks it finished only once every row is drained/noted (FIFO invariant)."""
        if not self.per_request:
            return
        self._q.put(_Finish(str(req_id)))

    def route_to_disk(self, req_id, dest, offload=None) -> None:
        """SEAM for the router: mark ``req_id`` for the per-request DISK route — its q + k rows stream
        to its own NVMe run_dir and, on finish, the file is offloaded to ``dest`` — INSTEAD of the
        host-buffer PerRequestIndex. NO-OP unless per_request mode is on. MUST be called BEFORE the
        request's rows reach the consumer (the router runs at request-start). Registered under
        ``_index_lock``; lazily starts an OffloadProcess on first use unless one is injected.

        The lazy OffloadProcess is CONSTRUCTED OFF ``_index_lock`` (the process backend spawns a
        child + threads; building under the lock would stall the consumer) then adopted under the
        lock only if still absent -- a racing route finds one set and DISCARDS its loser (closes it),
        so exactly one is ever adopted (no double-construct leak)."""
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
            _dbg(f"qk route_to_disk: req_id={req_id!r} (EXTERNAL) dest={dest!r} "
                 f"offload={type(self._offload).__name__}")

    def disk_residency(self) -> int:
        """Number of disk-routed requests still holding per-request staging state. Drops to 0 once
        every routed request has finished (close+offload+free)."""
        with self._index_lock:
            return len(self._disk_staging)

    def unlink_delivered_source(self, req_id) -> bool:
        """Remove the SERVER-side per-request staging SOURCE dir for a DELIVERED disk-routed request,
        reclaiming live NVMe. The durable CLIENT dest copy is untouched. Idempotent -> False when
        there is nothing recorded."""
        req_id = str(req_id)
        with self._index_lock:
            src = self._disk_delivered_src.pop(req_id, None)
        if src is None:
            return False
        import shutil
        shutil.rmtree(src, ignore_errors=True)
        return True

    def _reclaim_settled_pending(self) -> None:
        """DEFERRED settled-reclaim (rmtree-vs-offload-read race fix; mirror of OffLoopApertureDrain):
        rmtree each parked delivered-source whose offload has now SETTLED. Populated ONLY by
        ``clear_request_disk`` when it found the offload in flight; EMPTY on the happy path -> a
        strict no-op. Called from the consumer loop (per item) and ``finalize_all`` (shutdown).
        Snapshot under ``_index_lock``, ``settled()`` + rmtree OFF the lock. A never-settling offload's
        source is LEFT rather than rmtree'd mid-copy."""
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
                _dbg(f"qk reclaim settled delivered-src: req={ext!r} src={src!r}")

    def mark_host_aborted(self, req_id) -> None:
        """ABORT cleanup for the HOST (RPC) route: mark ``req_id`` so the consumer never (re-)stages
        its drained rows into the ``PerRequestIndex`` after ``clear_aperture_request`` freed its entry.
        MARK ONLY A REQUEST WITH LIVE HOST STATE (else a completed request would leave a stale mark no
        ``_Finish`` prunes). The index is keyed by the INTERNAL id, this abort id is EXTERNAL -> match
        with the exact-or-``{ext}-`` rule. NO-OP unless per_request mode is on. The index tuple layer
        keys (``("q"/"k", layer)``) do not affect the req_id match."""
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
        """ABORT cleanup for the DISK route (single-owner staging-dir lifecycle): MARK the request
        aborted; do NOT destroy its live staging. Only the CONSUMER thread ever creates, writes, or
        deletes a per-request staging dir, so the engine-thread abort must never rmtree a dir the
        consumer might still be demuxing this request's remaining rows into. Under ``_index_lock``:
        pop ``_disk_routed`` and, iff there is live staging or a still-live route, add the EXTERNAL id
        to ``_disk_aborted`` — the consumer DISCARDs its staging on this request's ``_Finish`` (or at
        ``finalize_all``). A recorded ``_disk_delivered_src`` means a finish already finalized +
        SUBMITTED this dir (FIFO: consumer done WRITING) -- but the OFFLOAD thread may still be READING
        it (copytree) if a confirm TIMEOUT skipped the confirm-path unlink, so it is rmtree'd here ONLY
        once the offload has SETTLED; if still in flight it is parked in ``_disk_reclaim_pending`` and
        reclaimed by ``_reclaim_settled_pending`` once the offload settles -- never mid-copy, never
        leaked. Strict no-op when per_request is off."""
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
        # Only rmtree the delivered-source once the offload has SETTLED (no longer reading it);
        # otherwise defer to _reclaim_settled_pending (never rmtree mid-copytree, never leak). Decided
        # OFF the lock: settled() takes the offload's OWN lock -- keep _index_lock unheld across it and
        # rmtree. Happy path: src is None here (confirm success already unlinked) -> inert.
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
            _dbg(f"qk clear_request_disk MARK-abort: req={req_id!r} marked={marked} "
                 f"delivered_src_reclaimed={reclaimed} deferred_reclaim={deferred} "
                 f"(consumer owns the staging-dir discard)")

    # ---- consumer thread ----
    def _finalize_finish_isolated(self, req_id) -> None:
        """Run ``_handle_finish`` under PER-REQUEST FINALIZE ISOLATION: a finalize error (a partially
        staged aborted disk request whose ``close()``/offload raises, a double-submit, a marshal error)
        must fail for THAT request ONLY — caught, logged LOUD with the req_id, the consumer CONTINUES.
        CONTRAST the DRAIN path (``_drain_item``): a drain error never advances the aperture cursor, so the
        never-drop guarantee must fail LOUD and stay FATAL to ``_run``'s outer handler."""
        try:
            self._handle_finish(req_id)
        except Exception:  # noqa: BLE001 -- isolate ONE request's finalize; never wedge the consumer
            logger.exception(
                "qk off-loop aperture drain: per-request FINALIZE failed for req_id=%r; that request's "
                "delivery is dropped, the consumer continues", req_id)
            if _aperture_debug():
                _dbg(f"qk finish FAILED (isolated, consumer continues): req_id={req_id!r}")

    def _run(self) -> None:
        try:
            # FIRST, before any CUDA call: select the device the copy stream lives on. A new
            # thread's current device is cuda:0; on TP rank r >= 1 that GPU belongs to another
            # process, and the stream context's exit restored a device-0 stream -- the G1 crash
            # (mia/graph/thread_device.py). A failure here is recorded like any consumer death.
            bind_thread_to_device(self._stream.device if self._stream is not None else None)
            while True:
                item = self._q.get()
                if item is _STOP:
                    self._q.task_done()
                    break
                try:
                    if isinstance(item, _Finish):
                        # PER-REQUEST FINALIZE ISOLATION (never wedges the consumer).
                        self._finalize_finish_isolated(item.req_id)
                    else:
                        # DRAIN stays fatal: a failure here never advances the aperture cursor.
                        self._drain_item(item)
                    # Deferred settled-reclaim of any delivered-source parked by a confirm-timeout
                    # abort. No-op when nothing is pending (the happy path); never raises.
                    self._reclaim_settled_pending()
                finally:
                    self._q.task_done()
        except BaseException as e:  # noqa: BLE001 — surface + let backpressure fail loud
            self._error = e
            logger.exception("qk off-loop aperture drain consumer thread died")

    def _pinned_buf(self, cache: dict, ln: int, n_rows: int, width: int, dtype) -> torch.Tensor:
        buf = cache[ln]
        if buf is None or buf.shape[0] < n_rows:
            buf = torch.empty(n_rows, width, dtype=dtype, pin_memory=self._is_cuda)
            cache[ln] = buf
        return buf[:n_rows]

    def _read_segments(self, segments: List[Tuple[int, int]], event):
        """D2H each per-layer q_buf AND k_buf ``[segments]`` into contiguous (LOGICAL-order) host
        buffers. cuda: on the dedicated copy stream (wait the scatter event, ``record_stream`` the
        source, K-deep event aperture). cpu (tests): plain ``.to('cpu')``. Returns
        ``[(ln, q_rows_cpu, k_rows_cpu), ...]``."""
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
        # CPU path (tests)
        if event is not None:
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001 — CPU stub events
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
            self._drain_item_legacy(item)

    def _host_buf(self, tag: str, ln: int, n_rows: int, width: int, dtype) -> torch.Tensor:
        """This (q|k, layer)'s reused host buffer (grown on demand), aligned for O_DIRECT."""
        key = (tag, ln)
        buf = self._host.get(key)
        if buf is None or buf.shape[0] < n_rows:
            buf = alloc_host_rows(n_rows, width, dtype, pinned=self._is_cuda,
                                  align=self._wp.mem_align)
            self._host[key] = buf
        return buf[:n_rows]

    def _issue_d2h(self, segments: List[Tuple[int, int]], event) -> list:
        """Issue every layer's q AND k D2H into its host buffers, one completion event per layer.
        Returns ``[(ln, q rows, k rows, event or None)]`` in layer order (cpu: synchronous copies,
        no events)."""
        total = sum(e - s for s, e in segments)
        jobs = []
        if self._stream is not None:
            if event is not None:
                self._stream.wait_event(event)
            with torch.cuda.stream(self._stream):
                for i, (ln, q_buf, k_buf) in enumerate(self.layers):
                    qh = self._host_buf("q", ln, total, q_buf.shape[1], q_buf.dtype)
                    kh = self._host_buf("k", ln, total, k_buf.shape[1], k_buf.dtype)
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
            except Exception:  # noqa: BLE001 — CPU stub events
                pass
        for ln, q_buf, k_buf in self.layers:
            qh = self._host_buf("q", ln, total, q_buf.shape[1], q_buf.dtype)
            kh = self._host_buf("k", ln, total, k_buf.shape[1], k_buf.dtype)
            off = 0
            for s, e in segments:
                n = e - s
                qh[off:off + n].copy_(q_buf[s:e])
                kh[off:off + n].copy_(k_buf[s:e])
                off += n
            jobs.append((ln, qh, kh, None))
        return jobs

    def _drain_item_fast(self, item: _DrainItem) -> None:
        """One step on the non-legacy write path (see ``OffLoopApertureDrain._drain_item_fast``):
        per-layer D2H, each layer's q and k writes submitted as soon as its rows land, the sidecar
        block built while the writes run, every write joined, then ``advance_drain``."""
        t0 = time.perf_counter()
        aperture = self.aperture
        assert item.start_logical == aperture._drain, (
            f"FIFO drain violation: item.start_logical={item.start_logical} != "
            f"aperture._drain={aperture._drain}")
        segments = aperture.segments_at(item.start_logical, item.n_rows)
        wp = self._wp
        with self._io_lock:
            if wp.closed:
                raise ApertureWriteError(
                    f"qk aperture drain ({self.run_dir}): a step reached the consumer after close() "
                    f"closed the raw files -- flush_aperture (stop, then close) must follow the last "
                    f"step; this step's rows are NOT written and the aperture is not advanced")
            t1 = time.perf_counter()
            jobs = self._issue_d2h(segments, item.event)
            futs = []
            t_first = None
            try:
                for ln, qh, kh, ev in jobs:
                    if ev is not None:
                        ev.synchronize()                # this layer's q and k landed
                    if t_first is None:
                        t_first = time.perf_counter()
                    futs.append(wp.pool.submit(timed_write, wp.sinks[("q", ln)], qh))
                    futs.append(wp.pool.submit(timed_write, wp.sinks[("k", ln)], kh))
                t2 = time.perf_counter()
                block = self._sidecar.prepare(item.entries)
            except BaseException:
                join_writes_quietly(futs)
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
            self._wstats, "qk", rows=item.n_rows, step_s=time.perf_counter() - t0,
            d2h_s=t2 - t1, write_s=(t4 - t_first) if t_first is not None else 0.0,
            write_tail_s=t4 - t3, busy_s=busy, bookkeeping_s=(t1 - t0) + (t3 - t2),
            bytes_by_mode=by_mode, wp=self._wp)

    def _drain_item_legacy(self, item: _DrainItem) -> None:
        """The pre-existing drain step, kept verbatim as ``MIA_APERTURE_WRITE_MODE=legacy`` (A/B
        validation only) and for per-request delivery; only the step accounting was added."""
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
        nbytes = 0
        if self.per_request:
            # Per-request delivery: split this step's q + k rows by req_id into the PerRequestIndex.
            # (_demux_into_index expands item.entries -> flat QKStepEntry itself, off-loop.)
            self._demux_into_index(item, pieces)
        else:
            # Shared-file path (default): append every layer's q + k rows to its two raw files.
            for ln, q_rows, k_rows in pieces:
                self._append(self._q_writers, self.q_raw_paths, ln, q_rows)
                self._append(self._k_writers, self.k_raw_paths, ln, k_rows)
                nbytes += (int(q_rows.numel()) * int(q_rows.element_size())
                           + int(k_rows.numel()) * int(k_rows.element_size()))
            t3 = time.perf_counter()
            # LayerEntry COLLAPSE: expand this step's per-request records into the flat per-(req,
            # layer) QKStepEntry list OFF the engine loop (here, on the consumer thread) — same fields,
            # same order the on-loop fan-out produced. StepMeta / sidecar bytes stay byte-identical.
            entries = expand_qk_records(item.entries)
            if entries:
                self._steps.append(StepMeta(entries))
        # Free the rows LAST — only after the D2H landed AND the rows were consumed, so the engine can
        # never scatter into a physical slot the consumer is still reading (never-drop + no torn read).
        aperture.advance_drain(item.n_rows)
        t4 = time.perf_counter()
        record_step_stats(
            self._wstats, "qk", rows=item.n_rows, step_s=t4 - t0, d2h_s=t2 - t1,
            write_s=t3 - t2, write_tail_s=t3 - t2, busy_s=t3 - t2,
            bookkeeping_s=(t1 - t0) + (t4 - t3), bytes_by_mode={"legacy": nbytes} if nbytes else {})

    def _demux_into_index(self, item: _DrainItem, pieces) -> None:
        """Slice each layer's contiguous drained q + k host rows by each ``QKStepEntry``'s req_id range
        and stage them in the ``PerRequestIndex`` under the two-stream convention: k rows under
        ``("k", layer)`` EVERY step (with a LAST-WRITE-WINS cumulative ``prefix_ends`` kmeta), q rows
        under ``("q", layer)`` only on emit steps. ``assemble_qk`` rebuilds ``k_all`` from those.

        ``pieces`` are ``(layer, q_rows, k_rows)`` holding this step's ``[start_logical,
        start_logical+n_rows)`` region in LOGICAL order, so an entry for ``(req, layer)`` occupies host
        offset ``entry.k_start - item.start_logical`` in the k buffer (and ``entry.q_start -
        item.start_logical`` in the q buffer for the emitted rows). Host-buffer slices are CLONED (the
        source is a reused pinned buffer / aperture view the engine may overwrite). DISK-routed slices are
        written to the request's per-request q/k files synchronously here (a zero-copy ``pwrite``,
        or the ``_raw_bytes`` copy under ``legacy``), consuming the view before ``advance_drain`` —
        no clone, never entering the host index.

        When no request is disk-routed (the default per_request path), every entry takes the
        clone+note branch, byte-identical to the shared-file reconstruction."""
        by_layer = {ln: (q, k) for ln, q, k in pieces}
        base = int(item.start_logical)
        # LayerEntry COLLAPSE: expand this step's per-request records into the flat per-(req, layer)
        # QKStepEntry list OFF the engine loop (here, on the consumer thread) — same fields + order the
        # on-loop fan-out produced, so the demux slices exactly the rows it did before. Heterogeneous
        # per-request layer sets are preserved: each record carries its own `layers`, so each entry's
        # (req_id, layer) range is that request's own.
        entries = expand_qk_records(item.entries)
        with self._index_lock:
            routed_keys = tuple(self._disk_routed) if self._disk_routed else ()
        any_disk = bool(routed_keys)
        # (req_id, layer, q_clone_or_None, k_clone, prefix_end) — cloned OFF the lock; noted UNDER it.
        staged = []
        for e in entries:
            qk = by_layer.get(e.layer)
            if qk is None:
                continue          # entry's layer not among the drained layers (should not happen)
            q_layer_rows, k_layer_rows = qk
            k_off = int(e.k_start) - base
            k_slice = k_layer_rows[k_off:k_off + int(e.k_rows)]
            if int(e.q_rows) > 0 and int(e.q_start) >= 0:
                q_off = int(e.q_start) - base
                q_slice = q_layer_rows[q_off:q_off + int(e.q_rows)]
            else:
                q_slice = None
            # e.req_id is the INTERNAL '{external}-{rand}' under serve; the disk routes are keyed by
            # the EXTERNAL id -> match with the exact-or-'{ext}-' rule, then STAGE keyed by the resolved
            # external id so finish/confirm/abort/unlink all agree.
            ext = _match_disk_route(e.req_id, routed_keys) if any_disk else None
            if ext is not None:
                if _aperture_debug():
                    _dbg(f"qk demux DISK hit: entry.req_id={e.req_id!r} -> route={ext!r} "
                         f"layer={e.layer} k_rows={int(e.k_rows)} q_rows={int(e.q_rows)}")
                # PER-ENTRY DISK ISOLATION (defense-in-depth): a single disk request's staging write
                # must NEVER wedge the whole consumer -- catch it here, log LOUD, mark the request
                # aborted (its remaining rows skipped + dir reclaimed), and CONTINUE. Host rows keep
                # their never-drop guarantee (cloned/noted below regardless; cursor advances anyway).
                try:
                    self._disk_write(ext, e.layer, q_slice, k_slice,
                                     int(e.prefix_end), int(e.num_computed))
                except Exception:  # noqa: BLE001 -- isolate ONE disk request; never wedge the consumer
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
                               k_slice.clone(), int(e.prefix_end)))
        with self._index_lock:
            # ABORT SKIP re-checked HERE so it is atomic with the note: a request aborted after its
            # rows were drained (HOST -> mark_host_aborted, or DISK whose _disk_routed was popped ->
            # _disk_aborted so its post-pop rows fell through to `staged`) must NOT (re-)create a host
            # slot nothing frees.
            ab_host = tuple(self._host_aborted) if self._host_aborted else ()
            ab_disk = tuple(self._disk_aborted) if self._disk_aborted else ()
            for req_id, layer, q_clone, k_clone, prefix_end in staged:
                if ((ab_host and _match_disk_route(req_id, ab_host) is not None)
                        or (ab_disk and _match_disk_route(req_id, ab_disk) is not None)):
                    if _aperture_debug():
                        _dbg(f"qk demux HOST-SKIP aborted: req={req_id!r} layer={layer}")
                    continue
                # k stream EVERY step, carrying the cumulative prefix_ends (LAST-WRITE-WINS on the
                # ("k", layer) key -- assemble_qk reads the final list at finish). prefix_end < 0 is a
                # non-emit (last_token mid-prefill) step -> no new boundary -> kmeta=None.
                kmeta = None
                if prefix_end >= 0:
                    lst = self._qk_kmeta.setdefault(req_id, {}).setdefault(layer, [])
                    lst.append(prefix_end)
                    kmeta = {"prefix_ends": list(lst)}
                self.index.note_rows(req_id, ("k", layer), k_clone, kmeta=kmeta)
                if q_clone is not None:
                    self.index.note_rows(req_id, ("q", layer), q_clone)

    def _disk_write(self, req_id, layer, q_slice, k_slice, prefix_end: int, num_computed: int) -> None:
        """Append a disk-routed request's step q + k rows to its per-request files (creating its
        staging on the first row). The dict membership is guarded by ``_index_lock``; the file write
        runs lock-free on the single consumer-thread writer.

        SKIP GUARD (single-owner dir lifecycle): re-check under the lock, BEFORE creating/appending,
        that this request is neither ABORTED nor un-routed — ``_demux_into_index`` matched it against a
        ``routed_keys`` snapshot taken BEFORE the lock, so a concurrent abort could land in that window.
        Skipping here means the consumer NEVER opens a layer file inside a dir the abort slated for
        discard (closing the ``rmtree``-vs-``open`` race)."""
        with self._index_lock:
            if req_id in self._disk_aborted or req_id not in self._disk_routed:
                if _aperture_debug():
                    _dbg(f"qk disk_write SKIP (aborted/unrouted): req={req_id!r} layer={layer}")
                return
            stg = self._disk_staging.get(req_id)
            if stg is None:
                stg = _PerRequestQKDiskStaging(
                    req_id, os.path.join(self._disk_base, _sanitize_req_id(req_id)),
                    self.header, self._perreq_cap, self._perreq_mmap,
                    write_mode=self._perreq_write_mode)
                self._disk_staging[req_id] = stg
        stg.append(layer, q_slice, k_slice, prefix_end, num_computed)

    def _handle_finish(self, req_id) -> None:
        """Finish a request: for a DISK-routed request finalize its per-request q/k files (msync +
        sidecar) and hand it to the OffloadProcess for transfer to the client dest, then free its
        staging (residency -> 0); for a host-buffer request mark it finished in the PerRequestIndex.
        FIFO: this ``_Finish`` trails all of the request's ``_DrainItem``s. SINGLE-OWNER ABORT RECLAIM:
        the CONSUMER thread owns the discard of an aborted disk request's staging dir."""
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
                if stg is not None and self._offload is not None:
                    self._disk_delivered_src[ext] = stg.run_dir
        if aborted_ext is not None:
            if stg_abort is not None:
                stg_abort.discard()          # close fds + rmtree the source (no offload, no sidecar)
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
                stg.close()                     # msync + per-request QK sidecar (single writer, off-lock)
                if self._offload is not None:
                    self._offload.submit(ext, stg.run_dir, disk_dest)  # non-blocking, never-drop
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
                self._qk_kmeta.pop(req_id, None)   # cumulative list is stored on the entry now

    # ---- shutdown / flush ----
    def finalize_all(self) -> None:
        """END-OF-RUN ONLY: mark every still-live per-request request finished so it becomes
        deliverable via ``pop_deliverable_qk``. Closes the last-step straggler gap (a request finishing
        on the FINAL executed step never gets its ``_Finish``). MUST run only at genuine end-of-run.
        STRICT NO-OP when per_request is off (index is None + empty disk maps) -> the shared-file
        default path is byte-identical. Mirrors ``OffLoopApertureDrain.finalize_all``."""
        with self._index_lock:
            disk_pending = list(self._disk_staging.keys())
        for req_id in disk_pending:
            self._finalize_finish_isolated(req_id)
        with self._index_lock:
            aborted_pending = list(self._disk_aborted)
        for ab_id in aborted_pending:
            self._finalize_finish_isolated(ab_id)
        # End-of-run settled-reclaim of any confirm-timeout-parked delivered-source (no-op when none).
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
        """NON-DESTRUCTIVE ``(host_live_count, disk_residency)`` — number of requests still holding a
        host-buffer ``PerRequestIndex`` entry and the number still holding per-request DISK staging.
        Read WITHOUT stopping the drain / popping / freeing (the residency gate polls it mid-serving).
        ``disk_residency`` takes ``_index_lock`` itself (non-reentrant), so it runs OUTSIDE the host
        read's hold."""
        with self._index_lock:
            host_live = len(self.index.live_req_ids()) if self.index is not None else 0
        disk = int(self.disk_residency())
        return (int(host_live), disk)

    def close(self) -> None:
        """Non-legacy shared-file drain: ``stop()`` a started consumer first so every enqueued step is
        written before the files close (see ``OffLoopApertureDrain.close``); the files are closed and
        the sidecar written even if the consumer failed, which is then re-raised. Legacy and
        per-request drains close exactly as before."""
        if self._wp is None or self._closed:
            return super().close()
        err: Optional[BaseException] = None
        if self._started:
            try:
                self.stop()
            except BaseException as e:  # noqa: BLE001 -- re-raised once the files are closed
                err = e
        try:
            super().close()
        except BaseException:
            if err is None:
                raise
            logger.exception("qk aperture drain close failed after a consumer failure")
        if err is not None:
            raise err

    def stop(self) -> None:
        """Drain the queue, join the consumer, finalize end-of-run stragglers, surface a consumer-thread
        error. Idempotent. ``_STOP`` is enqueued AFTER every row/finish item, so the joined consumer has
        noted every row into the index; only THEN does ``finalize_all()`` mark still-live stragglers.
        The finalize runs on this (collector) thread once the consumer is provably not running."""
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
