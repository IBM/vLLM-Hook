"""CUDA-graph hidden-state capture install on the capture-aperture path."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from mia._profiler import PROF
from mia.graph import register_graph_ops
from mia.graph.capture_aperture import CaptureAperture, ApertureBackpressureError
from mia.graph.hosts import HSCaptureHost
from mia.graph.registry import HostRegistry, get_registry, set_registry
from mia.graph.aperture_metadata import ReqCaptureRecord
from mia.graph.aperture_sizing import aperture_bytes_is_explicit, resolve_aperture_bytes_auto
from mia.graph.tp_shard import (
    HS_ALL_RANKS_ENV, HS_MODE_RANK0, HS_MODE_ROUND_ROBIN, HS_MODE_SINGLE, HSShard,
    hs_rows_for_mode, rank_dir_name, refuse_pipeline_parallel, resolve_hs_shard_mode,
    resolve_tp_coords)
from mia.graph.install import (
    _capture_idle_key,
    _resolve_max_num_batched_tokens,
    _run_dummy_pass,
    install_prepare_inputs_routing,
)
from mia.errors import MiaConfigurationError, MiaSizingError
from mia.runner import StepView
from mia.workers._common import iter_matched_modules
from mia.workers.hs_capture_worker import match_layer

logger = logging.getLogger(__name__)


_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
_HS_HOST_ATTR = "_mia_hs_host"
_capture_dbg = {"n": 0, "cap": 0}


def _require_buffer_mode_hs() -> None:
    mode = os.environ.get("MIA_HS_CAPTURE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise MiaConfigurationError(
            f"MIA_HS_CAPTURE={mode!r} is no longer supported: the PIECEWISE hs_probe "
            "capture mode was removed. Buffer mode is the only FULL-cudagraph HS path; "
            "unset MIA_HS_CAPTURE or set it to 'buffer'.")


def get_drain_row_counts(worker) -> dict:
    """Rows the HS aperture drain copied and skipped in this worker process."""
    drain = getattr(worker, "_hs_drain", None)
    counts = getattr(drain, "row_counts", None)
    if drain is None or not callable(counts):
        return {"hs.drain.rows_copied": 0, "hs.drain.rows_skipped": 0,
                "hs.drain.degenerate_steps": 0,
                "selective": False, "selective_disabled_reason": None}
    return counts()


def _route_vectorized_enabled() -> bool:
    return os.environ.get("MIA_ROUTE_VECTORIZED", "0") == "1"


def _route_decode_cache_enabled() -> bool:
    return os.environ.get("MIA_ROUTE_DECODE_CACHE", "1") != "0"


@dataclass
class _DecodeEntry:
    """Routing fields that stay constant for one capturing request's lifetime."""
    __slots__ = ("layer_rows", "mode", "layers")
    layer_rows: object
    mode: str
    layers: list


def _wrap_layer_class(cls: type) -> None:
    if cls in _WRAPPED_LAYER_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_LAYER_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            out = orig_fwd(self, *args, **kwargs)
            host = getattr(self, _HS_HOST_ATTR, None)
            if host is not None and host.do_capture:
                if (isinstance(out, tuple) and len(out) >= 2
                        and isinstance(out[0], torch.Tensor)
                        and isinstance(out[1], torch.Tensor)):
                    torch.ops.mia.capture_hs(
                        out[0], out[1], host.hs_buf, host.capture_index, 1)
                else:
                    h = out[0] if isinstance(out, tuple) else out
                    if isinstance(h, torch.Tensor):
                        torch.ops.mia.capture_hs(
                            h, h, host.hs_buf, host.capture_index, 0)
            return out
        return wrapped

    cls.forward = make_wrapped(orig_forward)


def install_hs_hosts(worker) -> Optional[HostRegistry]:
    """Install the CUDA-graph HS capture path via decoder-layer wrap."""
    _require_buffer_mode_hs()

    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install_hs] no model on model_runner; skip HS host install")
        return None

    refuse_pipeline_parallel(getattr(worker.parallel_config, "pipeline_parallel_size", 1),
                             "HS graph install")
    register_graph_ops()

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden_size = int(getattr(text_cfg, "hidden_size"))
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))

    tp_rank, tp_size = resolve_tp_coords(worker)
    shard_mode = resolve_hs_shard_mode(tp_size)
    owned_rows = hs_rows_for_mode(shard_mode, num_layers, tp_size, tp_rank)
    should_capture = (tp_rank == 0) if shard_mode == HS_MODE_SINGLE else bool(owned_rows)
    capture_all = os.environ.get(HS_ALL_RANKS_ENV) == "1"
    symmetric = tp_size > 1 and os.environ.get("MIA_HS_TP_SYMMETRIC", "1") != "0"
    bake_op = should_capture or symmetric

    worker._conf = {"hidden_size": hidden_size, "num_layers": num_layers}
    worker._should_capture = should_capture
    worker._tp_rank = tp_rank
    worker._hs_capture_all_ranks = capture_all
    worker._hs_shard_mode = shard_mode
    worker._hs_owned_rows = list(owned_rows)
    worker._hs_tp_size = int(tp_size)
    if shard_mode == HS_MODE_ROUND_ROBIN:
        owned1 = [r + 1 for r in owned_rows]
        print(f"[graph/install_hs] HS TP layer shard (tp_rank {tp_rank}/{tp_size}): round-robin "
              f"(MIA_HS_TP_SHARD, default 1) -> this rank captures {len(owned1)} of {num_layers} "
              f"layer(s) {owned1}; the other {num_layers - len(owned1)} bake capture_hs into "
              f"1-row sinks (symmetric graph)", flush=True)
    elif shard_mode == HS_MODE_RANK0:
        print(f"[graph/install_hs] HS TP layer shard OFF (MIA_HS_TP_SHARD=0, A/B only): tp_rank "
              f"{tp_rank}/{tp_size} captures {len(owned_rows)} of {num_layers} layer(s)",
              flush=True)
    if not hasattr(worker, "hs_mode"):
        worker.hs_mode = "last_token"

    if not getattr(worker, "_captured_states", None):
        worker._captured_states = {}
    if not getattr(worker, "_disk_states", None):
        worker._disk_states = {}

    from mia.graph.writer_process import init_writer_process, mark_no_writer
    if should_capture:
        init_writer_process(worker)
    else:
        mark_no_writer(worker, "HS sink rank: captures nothing")

    matched = list(iter_matched_modules(model, match_layer))
    if not matched:
        print("[graph/install_hs] no decoder layers matched LAYER_PATTERNS; "
              "HS graph capture inactive")
        set_registry(worker, "hs", None)
        return None

    device_t = next(model.parameters()).device

    registry = _install_hs_buffer(
        worker, model, matched, num_layers, hidden_size,
        should_capture, device_t, bake_op, owned_rows=owned_rows, symmetric=symmetric)
    set_registry(worker, "hs", registry)
    return registry


def _resolve_aperture_rows(worker, num_layers, hidden_size, buf_dtype, device,
                           rows_needed=None) -> tuple:
    elem_size = torch.empty(0, dtype=buf_dtype).element_size()
    if str(device).startswith("cuda"):
        total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        total_gpu = 1 << 30
    try:
        gpu_util = float(getattr(worker.vllm_config.cache_config,
                                 "gpu_memory_utilization", 0.9))
    except Exception:  # noqa: BLE001
        gpu_util = 0.9
    per_layer_row_bytes = hidden_size * elem_size
    aperture_bytes = resolve_aperture_bytes_auto(
        total_gpu, gpu_util, rows_needed=rows_needed,
        row_bytes=int(num_layers) * per_layer_row_bytes, what="HS capture")
    R = int(aperture_bytes // (num_layers * per_layer_row_bytes))
    if R < 1:
        raise MiaSizingError(
            f"HS capture aperture too small: aperture_bytes={aperture_bytes} num_layers={num_layers} "
            f"hidden={hidden_size} dtype={buf_dtype} -> R={R} rows/layer (<1). Raise "
            f"MIA_APERTURE_GPU_BYTES or reduce the model.")
    return R, aperture_bytes


def _install_hs_buffer(worker, model, matched, num_layers, hidden_size,
                       should_capture, device, bake_op=None, owned_rows=None,
                       symmetric=None) -> Optional[HostRegistry]:
    if bake_op is None:
        bake_op = should_capture
    if symmetric is None:
        symmetric = bool(bake_op)
    owned = (list(range(num_layers)) if owned_rows is None
             else sorted({int(r) for r in owned_rows if 0 <= int(r) < num_layers}))
    owned_set = set(owned)
    cap = _resolve_max_num_batched_tokens(worker)
    buf_dtype = model.dtype if hasattr(model, "dtype") \
        else next(model.parameters()).dtype

    if not should_capture or not owned:
        if bake_op:
            return _install_hs_sink(worker, matched, num_layers, hidden_size, cap, buf_dtype,
                                    device)
        for _name, module, _ln in matched:
            _wrap_layer_class(type(module))
        worker._capture_aperture = None
        return None

    n_owned = len(owned)
    R, aperture_bytes = _resolve_aperture_rows(worker, n_owned, hidden_size, buf_dtype, device,
                                               rows_needed=cap)
    if R < cap:
        _src = ("explicit MIA_APERTURE_GPU_BYTES" if aperture_bytes_is_explicit()
                else "default aperture")
        print(f"[graph/install_hs] WARNING: aperture rows/layer R={R} < token cap={cap} ({_src}); "
              f"a single max-token step may exceed the aperture -> backpressure "
              f"(ApertureBackpressureError after MIA_APERTURE_BACKPRESSURE_TIMEOUT_S). "
              f"Steady decode still fits.")

    registry: Optional[HostRegistry] = None
    aperture: Optional[CaptureAperture] = None
    if bake_op:
        registry = HostRegistry(
            num_layers=num_layers, cap=cap, device=device,
            should_capture=should_capture,
        )
        aperture = CaptureAperture(row_bytes=hidden_size * torch.empty(0, dtype=buf_dtype).element_size(),
                              n_slots=R, device=device, dtype=buf_dtype, row_shape=(hidden_size,))
        registry._hs_aperture = aperture
        registry._hs_step_entries = []
        registry._hs_owned_rows = None if n_owned == num_layers else list(owned)
        registry._hs_owned_set = None if n_owned == num_layers else frozenset(owned)
        registry.sentinel_row = aperture.SENTINEL
        registry.incremental_enabled = False
        registry.gpu_routing = False
        registry.capture_index_all.fill_(aperture.SENTINEL)
        for _slot in registry._aperture.slots:
            _slot["capture_index"].fill_(aperture.SENTINEL)
        worker._capture_aperture = aperture

    sink_index = sink_active = None
    if symmetric and n_owned < num_layers:
        sink_index = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        sink_active = torch.zeros(num_layers, dtype=torch.int32, device=device)
        worker._hs_sink_index = sink_index
        worker._hs_sink_active = sink_active

    n_hosts = n_sinks = 0
    for name, module, layer_num0 in matched:
        if bake_op and 0 <= layer_num0 < num_layers and layer_num0 in owned_set:
            hs_buf = torch.zeros(R + 1, hidden_size, dtype=buf_dtype, device=device)
            host = HSCaptureHost(
                module_name=name,
                layer_num=layer_num0,
                egress_layer_num=layer_num0 + 1,
                cap=cap,
                hidden=hidden_size,
                dtype=buf_dtype,
                device=device,
                has_residual=1,
                do_capture=True,
                hs_buf=hs_buf,
            )
            setattr(module, _HS_HOST_ATTR, host)
            registry.register_host(host)
            n_hosts += 1
        elif sink_index is not None and 0 <= layer_num0 < num_layers:
            _attach_sink_host(module, name, layer_num0, cap, hidden_size, buf_dtype, device,
                              sink_index, sink_active)
            n_sinks += 1
        _wrap_layer_class(type(module))

    if registry is not None:
        registry.assign_views()
        buf_bytes = sum(
            h.hs_buf.numel() * h.hs_buf.element_size()
            for _, h in registry.iter_hosts()
        )
        _layers_desc = (f"{num_layers} layers" if n_owned == num_layers
                        else f"{n_owned} of {num_layers} layers owned")
        print(f"[graph/install_hs] HS capture aperture: {buf_bytes / (1024**2):.1f} MiB on {device} "
              f"(R={R} rows/layer, {_layers_desc}, aperture_bytes={aperture_bytes / (1024**2):.0f} "
              f"MiB budget, token cap={cap}, sentinel_row={registry.sentinel_row})")

    print(f"[graph/install_hs] buffer-mode HS capture wired: {n_hosts} host(s) over "
          f"{num_layers} layer slot(s); should_capture={should_capture}; "
          f"hidden_size={hidden_size}; NO splitting op (rides decode cudagraph)"
          + (f"; {n_sinks} unowned layer(s) baked into 1-row sinks" if n_sinks else ""))
    return registry


def _attach_sink_host(module, name, layer_num0, cap, hidden_size, buf_dtype, device,
                      sink_index, sink_active) -> None:
    host = HSCaptureHost(
        module_name=name,
        layer_num=layer_num0,
        egress_layer_num=layer_num0 + 1,
        cap=cap,
        hidden=hidden_size,
        dtype=buf_dtype,
        device=device,
        has_residual=1,
        do_capture=True,
        hs_buf=torch.zeros(1, hidden_size, dtype=buf_dtype, device=device),
    )
    host.bind_views(sink_index[layer_num0], sink_active[layer_num0])
    setattr(module, _HS_HOST_ATTR, host)


def _install_hs_sink(worker, matched, num_layers, hidden_size, cap, buf_dtype, device) -> None:
    sink_index = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
    sink_active = torch.zeros(num_layers, dtype=torch.int32, device=device)
    worker._hs_sink_index = sink_index
    worker._hs_sink_active = sink_active
    worker._capture_aperture = None
    n_hosts = 0
    for name, module, layer_num0 in matched:
        if 0 <= layer_num0 < num_layers:
            _attach_sink_host(module, name, layer_num0, cap, hidden_size, buf_dtype, device,
                              sink_index, sink_active)
            n_hosts += 1
        _wrap_layer_class(type(module))
    sink_bytes = (sink_index.numel() * sink_index.element_size()
                  + n_hosts * hidden_size * torch.empty(0, dtype=buf_dtype).element_size())
    print(f"[graph/install_hs] TP-symmetric SINK on tp_rank {getattr(worker, '_tp_rank', '?')}: "
          f"capture_hs baked on {n_hosts} layer(s) into 1-row sink buffers "
          f"({sink_bytes / (1024**2):.1f} MiB); NO aperture, NO drain, NO writer process, NO "
          f"tp_rank dir -- this rank captures no HS layer", flush=True)
    return None


def _aperture_reserve_or_block(aperture: CaptureAperture, n: int, consumer=None) -> int:
    start = aperture.reserve(n)
    if start is not None:
        return start
    timeout = float(os.environ.get("MIA_APERTURE_BACKPRESSURE_TIMEOUT_S", "10") or "10")
    poll = float(os.environ.get("MIA_APERTURE_BACKPRESSURE_POLL_S", "0.001") or "0.001")
    PROF.incr("hs.aperture.backpressure")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if consumer is not None and not consumer.is_alive():
            err = getattr(consumer, "error", None)
            raise ApertureBackpressureError(
                f"HS capture aperture full and the off-loop drain consumer is DEAD: need {n} rows, "
                f"free={aperture.free_rows()} of {aperture.n_slots} rows/layer. Consumer error: {err!r}")
        time.sleep(poll)
        start = aperture.reserve(n)
        if start is not None:
            return start
    raise ApertureBackpressureError(
        f"HS capture aperture full: need {n} rows, free={aperture.free_rows()} of {aperture.n_slots} "
        f"rows/layer; reserve blocked past {timeout}s. The aperture cannot hold this step — raise "
        f"MIA_APERTURE_GPU_BYTES or the off-loop drain is not keeping up.")


def _build_routing_hs(step: StepView, registry: HostRegistry,
                      vectorized: Optional[bool] = None,
                      decode_cache: Optional[bool] = None) -> list:
    registry._hs_step_entries = []
    registry._hs_step_start = None
    registry._hs_step_rows = 0
    if not registry.should_capture:
        return []
    aperture: Optional[CaptureAperture] = getattr(registry, "_hs_aperture", None)
    if aperture is None:
        return []
    consumer = getattr(registry, "_hs_consumer", None)
    req_ids = step.req_ids

    bs = step.num_reqs
    capture_index_pinned = registry.capture_index_pinned
    cap = registry.cap
    default_hooks_on = getattr(registry, "_default_hooks_on", "prefill")
    default_hs_mode = getattr(registry, "_worker_hs_mode", "last_token")
    owned = getattr(registry, "_hs_owned_rows", None)
    owned_set = getattr(registry, "_hs_owned_set", None)

    if decode_cache is None:
        decode_cache = _route_decode_cache_enabled()
    if decode_cache:
        return _build_routing_hs_decode_cache(step, registry)

    if vectorized is None:
        vectorized = _route_vectorized_enabled()
    if vectorized:
        return _build_routing_hs_vectorized(
            step, registry, aperture, consumer, capture_index_pinned, cap)

    plans: list = []
    records: list = []
    qsl = step.query_start_loc_np
    for i in range(bs):
        req_id = req_ids[i]
        extra = step.extra_args_for(i)
        if not extra or extra.get("output_hidden_states") is None:
            continue

        output_spec = extra.get("output_hidden_states")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            is_prefill = bool(step.is_prefilling_np[i])
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hs_mode", default_hs_mode)
        start = int(qsl[i])
        end = int(qsl[i + 1])
        if end <= start:
            continue
        end = min(end, cap)
        if end <= start:
            continue

        if layer_filter is None:
            rows_layers = list(range(registry.num_layers)) if owned is None else list(owned)
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
            if owned_set is not None:
                rows_layers = [L for L in rows_layers if L in owned_set]
        if not rows_layers:
            continue

        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _aperture_reserve_or_block(aperture, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot
        registry._hs_step_rows += n
        phys = aperture.physical_slots(start_slot, n)
        layer_idx_t = torch.tensor(rows_layers, dtype=torch.long)
        if req_mode == "last_token":
            capture_index_pinned[layer_idx_t, end - 1] = int(phys[0])
        else:
            phys_t = torch.tensor(phys, dtype=torch.int64)
            capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        records.append(ReqCaptureRecord(
            req_id=str(req_id), logical_start=start_slot, n_rows=n, hs_mode=req_mode,
            layers=[L + 1 for L in rows_layers]))
        plans.append({
            "req_id": req_id,
            "n_rows": n,
            "layers": rows_layers,
            "hs_mode": req_mode,
        })

    registry._hs_step_entries = records
    return plans


def _build_routing_hs_vectorized(step: StepView, registry: HostRegistry,
                                 aperture: CaptureAperture, consumer,
                                 capture_index_pinned, cap: int) -> list:
    plans: list = []
    records: list = []
    rows_acc: list = []
    cols_acc: list = []
    slots_acc: list = []
    req_ids = step.req_ids
    bs = step.num_reqs
    qsl = step.query_start_loc_np
    default_hooks_on = getattr(registry, "_default_hooks_on", "prefill")
    default_hs_mode = getattr(registry, "_worker_hs_mode", "last_token")
    owned = getattr(registry, "_hs_owned_rows", None)
    owned_set = getattr(registry, "_hs_owned_set", None)
    for i in range(bs):
        req_id = req_ids[i]
        extra = step.extra_args_for(i)
        if not extra or extra.get("output_hidden_states") is None:
            continue

        output_spec = extra.get("output_hidden_states")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            is_prefill = bool(step.is_prefilling_np[i])
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hs_mode", default_hs_mode)
        start = int(qsl[i])
        end = int(qsl[i + 1])
        if end <= start:
            continue
        end = min(end, cap)
        if end <= start:
            continue

        if layer_filter is None:
            rows_layers = list(range(registry.num_layers)) if owned is None else list(owned)
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
            if owned_set is not None:
                rows_layers = [L for L in rows_layers if L in owned_set]
        if not rows_layers:
            continue

        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _aperture_reserve_or_block(aperture, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot
        registry._hs_step_rows += n
        phys = aperture.physical_slots(start_slot, n)

        rows_np = np.asarray(rows_layers, dtype=np.int64)
        nl = int(rows_np.shape[0])
        if req_mode == "last_token":
            rows_acc.append(rows_np)
            cols_acc.append(np.full(nl, end - 1, dtype=np.int64))
            slots_acc.append(np.full(nl, int(phys[0]), dtype=np.int64))
        else:
            phys_np = np.asarray(phys, dtype=np.int64)
            cols_span = np.arange(start, end, dtype=np.int64)
            rows_acc.append(np.repeat(rows_np, n))
            cols_acc.append(np.tile(cols_span, nl))
            slots_acc.append(np.tile(phys_np, nl))

        records.append(ReqCaptureRecord(
            req_id=str(req_id), logical_start=start_slot, n_rows=n, hs_mode=req_mode,
            layers=[L + 1 for L in rows_layers]))
        plans.append({
            "req_id": req_id,
            "n_rows": n,
            "layers": rows_layers,
            "hs_mode": req_mode,
        })

    if rows_acc:
        rows_t = torch.from_numpy(np.concatenate(rows_acc))
        cols_t = torch.from_numpy(np.concatenate(cols_acc))
        slots_t = torch.from_numpy(np.concatenate(slots_acc))
        capture_index_pinned[rows_t, cols_t] = slots_t

    registry._hs_step_entries = records
    return plans


def _build_routing_hs_decode_cache(step: StepView, registry: HostRegistry) -> list:
    registry._hs_step_entries = []
    registry._hs_step_start = None
    registry._hs_step_rows = 0
    if not registry.should_capture:
        return []
    aperture: Optional[CaptureAperture] = getattr(registry, "_hs_aperture", None)
    if aperture is None:
        return []
    consumer = getattr(registry, "_hs_consumer", None)
    req_ids = step.req_ids
    bs = step.num_reqs
    qsl = step.query_start_loc_np
    cap = registry.cap
    capture_index_pinned = registry.capture_index_pinned
    default_hooks_on = getattr(registry, "_default_hooks_on", "prefill")
    default_hs_mode = getattr(registry, "_worker_hs_mode", "last_token")
    owned = getattr(registry, "_hs_owned_rows", None)
    owned_set = getattr(registry, "_hs_owned_set", None)
    if not hasattr(registry, "_dc_entries"):
        registry._dc_entries = {}
    cache = registry._dc_entries

    live: set = set()
    plans: list = []
    records: list = []
    rows_acc: list = []
    cols_acc: list = []
    slots_acc: list = []
    for i in range(bs):
        req_id = req_ids[i]
        key = str(req_id)
        live.add(key)
        start = int(qsl[i])
        end = int(qsl[i + 1])
        if end <= start:
            continue
        end = min(end, cap)
        if end <= start:
            continue
        n_tokens = end - start
        is_prefill = bool(step.is_prefilling_np[i])
        entry = cache.get(key)

        if entry is not None and n_tokens == 1 and not is_prefill:
            n = 1
            start_slot = _aperture_reserve_or_block(aperture, n, consumer)
            if registry._hs_step_start is None:
                registry._hs_step_start = start_slot
            registry._hs_step_rows += n
            col = end - 1
            phys0 = start_slot % aperture.n_slots
            nl = int(entry.layer_rows.shape[0])
            rows_acc.append(entry.layer_rows)
            cols_acc.append(np.full(nl, col, dtype=np.int64))
            slots_acc.append(np.full(nl, phys0, dtype=np.int64))
            records.append(ReqCaptureRecord(
                req_id=key, logical_start=start_slot, n_rows=n, hs_mode=entry.mode,
                layers=entry.layers))
            plans.append({"req_id": req_id, "n_rows": n,
                          "layers": [L - 1 for L in entry.layers], "hs_mode": entry.mode})
            continue

        extra = step.extra_args_for(i)
        if not extra or extra.get("output_hidden_states") is None:
            cache.pop(key, None)
            continue
        output_spec = extra.get("output_hidden_states")
        layer_filter = ({int(x) for x in output_spec}
                        if isinstance(output_spec, list) else None)
        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            if hooks_on == "prefill" and not is_prefill:
                cache.pop(key, None)
                continue
            if hooks_on == "decode" and is_prefill:
                continue
        req_mode = extra.get("hs_mode", default_hs_mode)
        if layer_filter is None:
            rows_layers = list(range(registry.num_layers)) if owned is None else list(owned)
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
            if owned_set is not None:
                rows_layers = [L for L in rows_layers if L in owned_set]
        if not rows_layers:
            continue
        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _aperture_reserve_or_block(aperture, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot
        registry._hs_step_rows += n
        phys = aperture.physical_slots(start_slot, n)
        rows_np = np.asarray(rows_layers, dtype=np.int64)
        nl = int(rows_np.shape[0])
        if req_mode == "last_token":
            rows_acc.append(rows_np)
            cols_acc.append(np.full(nl, end - 1, dtype=np.int64))
            slots_acc.append(np.full(nl, int(phys[0]), dtype=np.int64))
        else:
            phys_np = np.asarray(phys, dtype=np.int64)
            cols_span = np.arange(start, end, dtype=np.int64)
            rows_acc.append(np.repeat(rows_np, n))
            cols_acc.append(np.tile(cols_span, nl))
            slots_acc.append(np.tile(phys_np, nl))
        layers_tmpl = [L + 1 for L in rows_layers]
        records.append(ReqCaptureRecord(
            req_id=key, logical_start=start_slot, n_rows=n, hs_mode=req_mode,
            layers=layers_tmpl))
        plans.append({"req_id": req_id, "n_rows": n, "layers": rows_layers,
                      "hs_mode": req_mode})
        if hooks_on != "prefill":
            cache[key] = _DecodeEntry(layer_rows=rows_np, mode=req_mode, layers=layers_tmpl)
        else:
            cache.pop(key, None)

    if len(cache) > len(live):
        for k in list(cache):
            if k not in live:
                del cache[k]

    if rows_acc:
        rows_t = torch.from_numpy(np.concatenate(rows_acc))
        cols_t = torch.from_numpy(np.concatenate(cols_acc))
        slots_t = torch.from_numpy(np.concatenate(slots_acc))
        capture_index_pinned[rows_t, cols_t] = slots_t

    registry._hs_step_entries = records
    return plans


def install_execute_model_wrapper_hs(model_runner, worker) -> None:
    """Install HS aperture routing (``prepare_inputs`` wrap) and the per-step drain (``execute_model``)."""
    if getattr(model_runner, "_mia_hs_wrapped", False):
        return

    model_runner._mia_hs_wrapped = True

    _route_vec = _route_vectorized_enabled()
    _route_dc = _route_decode_cache_enabled()

    def _hs_routing_key(step, registry):
        return _capture_idle_key(step, registry, "output_hidden_states")

    if _route_dc:
        print("[graph/install_hs] HS routing decode-cache ENABLED "
              "(default ON; MIA_ROUTE_DECODE_CACHE=0 to disable)", flush=True)
    if _route_vec and _route_dc:
        print("[graph/install_hs] MIA_ROUTE_VECTORIZED=1 has NO EFFECT while "
              "MIA_ROUTE_DECODE_CACHE is on: the decode-cache fast path returns before the "
              "vectorized dispatch is reached. Set MIA_ROUTE_DECODE_CACHE=0 to exercise it.",
              flush=True)

    def _hs_build_routing(step, registry):
        return _build_routing_hs(step, registry,
                                  vectorized=_route_vec, decode_cache=_route_dc)

    install_prepare_inputs_routing(model_runner, worker, _hs_build_routing, label="hs",
                                   routing_key_fn=_hs_routing_key)

    registry: Optional[HostRegistry] = get_registry(worker, "hs")
    if registry is not None:
        registry._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

    aperture = getattr(registry, "_hs_aperture", None) if registry is not None else None
    drain = None
    _sync_drain = os.environ.get("MIA_APERTURE_SYNC_DRAIN", "0") == "1"
    if registry is not None and aperture is not None:
        from mia.graph.aperture_drain_hs import (
            MultiLayerApertureDrain, OffLoopApertureDrain, _torch_dtype_name, record_captured_cells)
        hidden = int(worker._conf["hidden_size"])
        layers = [(host.egress_layer_num, host.hs_buf) for _, host in registry.iter_hosts()]
        buf_dtype = layers[0][1].dtype if layers else torch.float32
        base = os.environ.get("MIA_APERTURE_DIR", "./hs_aperture_dump")
        tp_rank = getattr(worker, "_tp_rank", None)
        if tp_rank is None:
            tp_rank = resolve_tp_coords(worker)[0]
        run_dir = os.path.join(base, rank_dir_name(tp_rank))
        header = {"dtype": _torch_dtype_name(buf_dtype),
                  "row_shape": [hidden], "hidden": hidden}
        _tp_size = int(resolve_tp_coords(worker)[1])
        _num_layers = int(worker._conf.get("num_layers", len(layers)))
        header.update({
            "tp_rank": int(tp_rank),
            "tp_size": _tp_size,
            "num_layers": _num_layers,
            "capture_all_ranks": bool(getattr(worker, "_hs_capture_all_ranks", False)),
        })
        if getattr(worker, "_hs_shard_mode", None) == HS_MODE_ROUND_ROBIN:
            header.update(HSShard.of(int(tp_rank), _tp_size, _num_layers).as_header())
        from mia.graph.install import predict_capture_write_shape
        _shape = predict_capture_write_shape(
            worker, "hs", str(getattr(worker, "hs_mode", "last_token") or "last_token"),
            int(aperture.n_slots))
        if _sync_drain:
            drain = MultiLayerApertureDrain(aperture, layers, run_dir, header, shape=_shape)
            registry._hs_consumer = None
            _mode = "sync per-step"
        else:
            _per_request = os.environ.get("MIA_APERTURE_PER_REQUEST", "0") == "1"
            drain = OffLoopApertureDrain(aperture, layers, run_dir, header,
                                         per_request=_per_request, shape=_shape)
            drain._note_unstaged_finish = (
                getattr(worker, "_hs_shard_mode", None) == HS_MODE_ROUND_ROBIN)
            drain.start()
            registry._hs_consumer = drain
            _pr = " + per-request delivery" if _per_request else ""
            _mode = f"OFF-LOOP consumer thread (drain_aperture={drain._aperture_depth}){_pr}"
        if getattr(drain, "selective", False):
            _mode += " + SELECTIVE drain (default ON; MIA_DRAIN_SELECTIVE=0 to disable)"
        elif getattr(drain, "selective_disabled_reason", None):
            _mode += " + selective drain has no effect here"
            _explicit = os.environ.get("MIA_DRAIN_SELECTIVE") is not None
            if _explicit:
                logger.warning(
                    "selective drain (MIA_DRAIN_SELECTIVE=%s, set explicitly) has no effect "
                    "for this drain: %s. The full drain runs instead (every installed layer, every "
                    "row) -- byte-identical, but the Lever C saving is NOT in effect.",
                    os.environ.get("MIA_DRAIN_SELECTIVE"), drain.selective_disabled_reason)
                print("[graph/install_hs] *** selective drain requested but IGNORED: "
                      f"{drain.selective_disabled_reason} -> FULL drain ***", flush=True)
            else:
                logger.info(
                    "selective drain (MIA_DRAIN_SELECTIVE, default ON) has no effect for this "
                    "drain: %s. The full drain runs -- byte-identical, no action needed.",
                    drain.selective_disabled_reason)
                print("[graph/install_hs] selective drain (default) has no effect here: "
                      f"{drain.selective_disabled_reason} -> FULL drain", flush=True)
        else:
            _mode += " + selective drain OFF (MIA_DRAIN_SELECTIVE=0)"
        worker._hs_drain = drain
        worker._hs_run_dir = run_dir
        _wline = (f"[graph/install_hs] HS aperture write path (tp_rank {int(tp_rank)}): "
                  f"{drain.write_path_summary()}")
        print(_wline, flush=True)
        logger.info(_wline)
        import atexit
        atexit.register(lambda d=drain: d.close())
        print(f"[graph/install_hs] HS aperture drain ON -> {run_dir} "
              f"(R={aperture.n_slots} rows/layer, {len(layers)} layers, {_mode})", flush=True)
    else:
        print("[graph/install_hs] no capture aperture; HS drain NOT wired", flush=True)

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run") or kwargs.get("is_profile"):
            return _run_dummy_pass(orig_execute_model, get_registry(worker, "hs"),
                                   scheduler_output, args, kwargs)

        registry: Optional[HostRegistry] = get_registry(worker, "hs")
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        registry._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

        with PROF.timed("graph.forward"):
            result = orig_execute_model(scheduler_output, *args, **kwargs)

        plans = getattr(registry, "_pending_plans", None) or []
        drain = getattr(worker, "_hs_drain", None)
        records = getattr(registry, "_hs_step_entries", None) or []

        if aperture is not None:
            _cap_rows = int(getattr(registry, "_hs_step_rows", 0) or 0)
            if _cap_rows > 0:
                if drain is not None and drain._selective_active():
                    _cells = record_captured_cells(records)
                else:
                    _cells = _cap_rows * len(layers)
                PROF.gauge("captured.bytes.hs", float(_cells) * float(aperture.row_bytes))
        if plans and drain is not None:
            if _sync_drain:
                with PROF.timed("graph.drain"):
                    drain.record_entries(records)
                    drain.drain_once()
            else:
                event = None
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record()
                start_logical = getattr(registry, "_hs_step_start", None)
                n_rows = int(getattr(registry, "_hs_step_rows", 0) or 0)
                if n_rows > 0 and start_logical is not None:
                    drain.enqueue(records, start_logical, n_rows, event)

        if drain is not None and getattr(drain, "per_request", False):
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if finished:
                for _rid in finished:
                    drain.enqueue_finish(_rid)

        _fin_evidence = getattr(scheduler_output, "finished_req_ids", None)
        if _fin_evidence:
            PROF.incr("hook.fire.hs", len(_fin_evidence) * len(layers))

        registry._pending_plans = []
        registry._hs_step_entries = []
        registry._hs_step_start = None
        registry._hs_step_rows = 0

        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install_hs] execute_model wrapper installed (HS aperture drain)")


__all__ = ["install_hs_hosts", "install_execute_model_wrapper_hs"]

