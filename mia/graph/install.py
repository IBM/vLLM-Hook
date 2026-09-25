"""CUDA-graph QK capture: install, per-step routing and egress."""
from __future__ import annotations

import contextlib
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from vllm.forward_context import get_forward_context

from mia._profiler import PROF
from mia.graph import register_graph_ops
from mia.graph.capture_aperture import CaptureAperture, ApertureBackpressureError
from mia.graph.hosts import QKCaptureHost
from mia.graph.registry import HostRegistry, get_registry, set_registry
from mia.graph.aperture_metadata import QKReqCaptureRecord
from mia.graph.aperture_sizing import aperture_bytes_is_explicit, resolve_aperture_bytes_auto
from mia.graph.tp_shard import (
    check_attn_modules_match_shard,
    qk_conf_head_dim,
    qk_shard,
    rank_dir_name,
    refuse_pipeline_parallel,
    resolve_tp_coords,
)
from mia.errors import MiaConfigurationError, MiaRefusal, MiaSizingError
from mia.runner import StepView, install_request_arg_stash, require_v2_runner, step_view
from mia.workers._common import iter_matched_modules
from mia.workers.qk_capture_worker import match_attn


_GRAPH_MODE_ENV = "MIA_GRAPH_MODE"
_graph_mode_enabled = False


def set_graph_mode(enabled: bool) -> None:
    """Arm/disarm the graph path."""
    global _graph_mode_enabled
    _graph_mode_enabled = bool(enabled)
    os.environ[_GRAPH_MODE_ENV] = "1" if enabled else "0"


def graph_mode_enabled() -> bool:
    """True if the graph path should install (module flag in the driver, env var in workers)."""
    return _graph_mode_enabled or os.environ.get(_GRAPH_MODE_ENV) == "1"


_WRAPPED_ATTN_CLASSES: Dict[type, Any] = {}
_HOST_ATTR = "_mia_qk_host"
_REG_ATTR = "_mia_qk_registry"

_PREFIXK_STASH = os.environ.get("MIA_QK_PREFIXK_STASH") == "1"


def _require_buffer_mode() -> None:
    mode = os.environ.get("MIA_QK_CAPTURE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise MiaConfigurationError(
            f"MIA_QK_CAPTURE={mode!r} is no longer supported: the PIECEWISE op/seam "
            "capture mode was removed. Buffer mode is the only FULL-cudagraph capture path; "
            "unset MIA_QK_CAPTURE or set it to 'buffer'.")

_BATCHED_EGRESS = os.environ.get("MIA_BATCHED_EGRESS", "1") == "1"

_capture_dbg = {"n": 0}

_NO_PREFIXK = os.environ.get("MIA_QK_NO_PREFIXK") == "1"


def _aperture_reserve_or_block(aperture: CaptureAperture, n: int, consumer=None) -> int:
    start = aperture.reserve(n)
    if start is not None:
        return start
    timeout = float(os.environ.get("MIA_APERTURE_BACKPRESSURE_TIMEOUT_S", "10") or "10")
    poll = float(os.environ.get("MIA_APERTURE_BACKPRESSURE_POLL_S", "0.001") or "0.001")
    PROF.incr("qk.aperture.backpressure")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if consumer is not None and not consumer.is_alive():
            err = getattr(consumer, "error", None)
            raise ApertureBackpressureError(
                f"QK capture aperture full and the off-loop drain consumer is DEAD: need {n} rows, "
                f"free={aperture.free_rows()} of {aperture.n_slots} rows/layer. Consumer error: {err!r}")
        time.sleep(poll)
        start = aperture.reserve(n)
        if start is not None:
            return start
    raise ApertureBackpressureError(
        f"QK capture aperture full: need {n} rows, free={aperture.free_rows()} of {aperture.n_slots} "
        f"rows/layer; reserve blocked past {timeout}s. Raise MIA_APERTURE_GPU_BYTES or "
        f"the off-loop drain is not keeping up.")


def _resolve_qk_aperture_rows(worker, num_layers, q_dim, k_dim, buf_dtype, device,
                              rows_needed=None) -> tuple:
    elem = torch.empty(0, dtype=buf_dtype).element_size()
    if str(device).startswith("cuda"):
        total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        total_gpu = 1 << 30
    try:
        gpu_util = float(getattr(worker.vllm_config.cache_config,
                                 "gpu_memory_utilization", 0.9))
    except Exception:  # noqa: BLE001
        gpu_util = 0.9
    per_slot_bytes = (int(q_dim) + int(k_dim)) * elem
    aperture_bytes = resolve_aperture_bytes_auto(
        total_gpu, gpu_util, rows_needed=rows_needed,
        row_bytes=int(num_layers) * per_slot_bytes, what="QK capture")
    R = int(aperture_bytes // (num_layers * per_slot_bytes))
    if R < 1:
        raise MiaSizingError(
            f"QK capture aperture too small: aperture_bytes={aperture_bytes} num_layers={num_layers} "
            f"q_dim={q_dim} k_dim={k_dim} dtype={buf_dtype} -> R={R} rows/layer (<1). Raise "
            f"MIA_APERTURE_GPU_BYTES or reduce the model.")
    return R, aperture_bytes


def _cpu_1d(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    for attr in ("cpu", "np", "gpu"):
        v = getattr(x, attr, None)
        if v is None or callable(v):
            continue
        if isinstance(v, torch.Tensor):
            return v.detach().to("cpu")
        try:
            return torch.as_tensor(v)
        except Exception:  # noqa: BLE001
            continue
    return None


def _wrap_attn_class(cls: type) -> None:
    if cls in _WRAPPED_ATTN_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_ATTN_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            host = getattr(self, _HOST_ATTR, None)
            if host is not None and host.do_capture:
                if _PREFIXK_STASH:
                    reg = getattr(self, _REG_ATTR, None)
                    if reg is not None and reg.fwd_ctx is None:
                        try:
                            ctx = get_forward_context()
                            reg.stash_forward_context(
                                ctx, getattr(ctx, "attn_metadata", None)
                            )
                        except Exception:  # noqa: BLE001
                            pass
                if len(args) >= 2:
                    host.capture(args[0], args[1])
            return orig_fwd(self, *args, **kwargs)

        return wrapped

    cls.forward = make_wrapped(orig_forward)


def _resolve_max_num_batched_tokens(worker) -> int:
    for owner in (worker.model_runner, worker):
        sched = getattr(owner, "scheduler_config", None)
        if sched is not None:
            mnbt = getattr(sched, "max_num_batched_tokens", None)
            if mnbt is not None:
                return int(mnbt)
    vc = getattr(worker, "vllm_config", None)
    if vc is not None:
        sched = getattr(vc, "scheduler_config", None)
        if sched is not None:
            mnbt = getattr(sched, "max_num_batched_tokens", None)
            if mnbt is not None:
                return int(mnbt)
    raise MiaConfigurationError(
        "could not resolve max_num_batched_tokens for QK buffer CAP"
    )


def _resolve_max_num_seqs(worker) -> Optional[int]:
    for owner in (getattr(worker, "model_runner", None), worker,
                  getattr(worker, "vllm_config", None)):
        sched = getattr(owner, "scheduler_config", None) if owner is not None else None
        n = getattr(sched, "max_num_seqs", None) if sched is not None else None
        if n:
            return int(n)
    return None


def predict_capture_write_shape(worker, kind: str, capture_mode: str, aperture_rows: int):
    """Upper bound on one step's per-file write for this worker's capture, or None if unknown."""
    from .aperture_sink import predict_rows_per_write
    try:
        return predict_rows_per_write(
            kind, capture_mode=capture_mode,
            max_batched_tokens=_resolve_max_num_batched_tokens(worker),
            max_num_seqs=_resolve_max_num_seqs(worker),
            aperture_rows=aperture_rows)
    except Exception as e:  # noqa: BLE001
        print(f"[graph/install] could not predict the {kind} aperture write size ({e!r}); the "
              f"write mode will be decided on alignment alone", flush=True)
        return None


def install_qk_hosts(worker) -> Optional[HostRegistry]:
    """Class-wrap Attention.forward (and, in buffer mode, build per-layer hosts)."""
    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install] no model on model_runner; skip QK host install")
        return None

    _require_buffer_mode()
    refuse_pipeline_parallel(getattr(worker.parallel_config, "pipeline_parallel_size", 1),
                             "QK graph install")

    register_graph_ops()

    tp_rank, tp_size = resolve_tp_coords(worker)
    should_capture = True

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    num_h = int(getattr(text_cfg, "num_attention_heads"))
    num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
    hidden = int(getattr(text_cfg, "hidden_size"))
    head_dim = qk_conf_head_dim(text_cfg)
    buf_head_dim = head_dim
    attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
    shard = qk_shard(tp_rank, tp_size, num_h, num_kv, buf_head_dim)
    q_dim = shard.q_width
    k_dim = shard.k_width

    worker._conf = dict(
        num_attention_heads=num_h,
        num_key_value_heads=num_kv,
        hidden_size=hidden,
        head_dim=head_dim,
        attention_multiplier=attn_mult,
    )
    worker._should_capture = should_capture
    worker._qk_shard = shard
    worker._tp_rank = tp_rank
    if not hasattr(worker, "hookq_mode"):
        worker.hookq_mode = "all_tokens"
    worker._score_mode_default = os.environ.get("MIA_QK_SCORE", "0") == "1"
    worker._score_head_default = int(os.environ.get("MIA_QK_SCORE_HEAD", "0"))
    worker._score_dtype = torch.float16
    if worker._score_mode_default:
        raise MiaConfigurationError(
            "MIA_QK_SCORE=1 (attention-score capture) is not supported on the QK capture-aperture "
            "path (v1 = raw q/k only). Unset it, or use the eager/bank path for score capture.")

    if not hasattr(worker, "_captured_states") or worker._captured_states is None:
        worker._captured_states = {}
    if not hasattr(worker, "_disk_states") or worker._disk_states is None:
        worker._disk_states = {}

    from mia.graph.writer_process import init_writer_process
    init_writer_process(worker)

    cap = _resolve_max_num_batched_tokens(worker)

    device = next(model.parameters()).device

    matched = list(iter_matched_modules(model, match_attn))
    if not matched:
        print("[graph/install] no attention modules matched ATTN_PATTERNS; "
              "QK graph capture inactive")
        set_registry(worker, "qk", None)
        return None
    num_layers = max(layer_num for _, _, layer_num in matched) + 1
    check_attn_modules_match_shard(matched, shard)
    worker._qk_num_layers = num_layers

    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    registry: Optional[HostRegistry] = None
    aperture: Optional[CaptureAperture] = None
    n_hosts = 0
    if should_capture:
        R, aperture_bytes = _resolve_qk_aperture_rows(worker, num_layers, q_dim, k_dim, buf_dtype,
                                                      device, rows_needed=cap)
        if R < cap:
            _src = ("explicit MIA_APERTURE_GPU_BYTES" if aperture_bytes_is_explicit()
                    else "default aperture")
            print(f"[graph/install] WARNING: QK aperture rows/layer R={R} < token cap={cap} "
                  f"({_src}); a single max-token step may exceed the aperture -> backpressure "
                  f"(ApertureBackpressureError after MIA_APERTURE_BACKPRESSURE_TIMEOUT_S). "
                  f"Steady decode still fits.")
        registry = HostRegistry(
            num_layers=num_layers, cap=cap, device=device,
            should_capture=should_capture,
        )
        elem = torch.empty(0, dtype=buf_dtype).element_size()
        aperture = CaptureAperture(row_bytes=k_dim * elem, n_slots=R, device=device,
                              dtype=buf_dtype, row_shape=(k_dim,))
        registry._qk_aperture = aperture
        registry._qk_step_entries = []
        registry.sentinel_row = aperture.SENTINEL
        registry.incremental_enabled = False
        registry.gpu_routing = False
        registry.capture_index_all.fill_(aperture.SENTINEL)
        for _slot in registry._aperture.slots:
            _slot["capture_index"].fill_(aperture.SENTINEL)
        worker._capture_aperture = aperture

    for name, module, layer_num in matched:
        if should_capture:
            q_buf = torch.zeros(R + 1, q_dim, dtype=buf_dtype, device=device)
            k_buf = torch.zeros(R + 1, k_dim, dtype=buf_dtype, device=device)
            host = QKCaptureHost(
                module_name=name,
                layer_num=layer_num,
                cap=cap,
                q_dim=q_dim,
                k_dim=k_dim,
                dtype=buf_dtype,
                device=device,
                do_capture=True,
                q_buf=q_buf,
                k_buf=k_buf,
            )
            setattr(module, _HOST_ATTR, host)
            setattr(module, _REG_ATTR, registry)
            registry.register_host(host)
            n_hosts += 1
        _wrap_attn_class(type(module))

    if registry is not None:
        registry.assign_views()
        buf_bytes = sum(
            h.q_buf.numel() * h.q_buf.element_size()
            + h.k_buf.numel() * h.k_buf.element_size()
            for _, h in registry.iter_hosts()
        )
        print(f"[graph/install] QK capture aperture: {buf_bytes / (1024**2):.1f} MiB on {device} "
              f"(R={aperture.n_slots} rows/layer, {num_layers} layers, "
              f"aperture_bytes={aperture_bytes / (1024**2):.0f} MiB budget, token cap={cap}, "
              f"sentinel_row={registry.sentinel_row}; q_dim={q_dim} k_dim={k_dim})")

    set_registry(worker, "qk", registry)
    print(f"[graph/install] QK aperture hosts installed: {n_hosts} host(s) over "
          f"{num_layers} layer slot(s); should_capture={should_capture}; cap={cap}; "
          f"tp_rank={tp_rank}/{tp_size} q_heads=[{shard.q_head_start}, "
          f"{shard.q_head_start + shard.num_local_q_heads}) kv_heads=[{shard.kv_head_start}, "
          f"{shard.kv_head_start + shard.num_local_kv_heads}) x{shard.num_kv_head_replicas} "
          f"replica(s); NO splitting op (rides decode cudagraph)")
    return registry


def _build_routing(step: StepView, registry: HostRegistry) -> list:
    registry._qk_step_entries = []
    registry._qk_step_start = None
    registry._qk_step_rows = 0
    if not registry.should_capture:
        return []
    aperture: Optional[CaptureAperture] = getattr(registry, "_qk_aperture", None)
    if aperture is None:
        return []
    consumer = getattr(registry, "_qk_consumer", None)
    req_ids = step.req_ids

    bs = step.num_reqs
    capture_index_pinned = registry.capture_index_pinned
    cap = registry.cap
    default_hooks_on = getattr(registry, "_default_hooks_on", "prefill")
    default_hookq_mode = getattr(registry, "_worker_hookq_mode", "all_tokens")
    score_mode_default = getattr(registry, "_worker_score_mode", False)

    plans: list = []
    records: list = []
    for i in range(bs):
        req_id = req_ids[i]
        extra = step.extra_args_for(i)
        if not extra or extra.get("output_qk") is None:
            continue

        output_spec = extra.get("output_qk")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, dict):
            layer_filter = {int(k) for k in output_spec.keys()}
        elif isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        is_prefill = bool(step.is_prefilling_np[i])
        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hookq_mode", default_hookq_mode)
        cap_mode = extra.get("qk_capture", "score" if score_mode_default else "qk")
        if cap_mode == "score":
            PROF.incr("qk.aperture.score_unsupported")
            if _capture_dbg.get("score_warn", 0) < 1:
                _capture_dbg["score_warn"] = 1
                print("[graph/install] WARNING: per-request qk_capture='score' is not supported on "
                      "the QK capture-aperture path (v1 raw q/k only); this request is NOT captured.",
                      flush=True)
            continue

        start = int(step.query_start_loc_np[i])
        end = int(step.query_start_loc_np[i + 1])
        if end > cap:
            end = cap
        if end <= start:
            continue
        qlen = end - start

        num_computed = int(step.num_computed_tokens_np[i])
        abs_end = num_computed + qlen

        if layer_filter is None:
            req_layers = list(range(registry.num_layers))
        else:
            req_layers = [L for L in layer_filter if 0 <= L < registry.num_layers]
        if not req_layers:
            continue

        emit_q = True
        if req_mode == "last_token" and is_prefill:
            num_prompt = int(step.prompt_len_np[i])
            emit_q = abs_end >= num_prompt

        n = qlen
        start_slot = _aperture_reserve_or_block(aperture, n, consumer)
        if registry._qk_step_start is None:
            registry._qk_step_start = start_slot
        registry._qk_step_rows += n
        phys = aperture.physical_slots(start_slot, n)
        phys_t = torch.tensor(phys, dtype=torch.int64)
        layer_idx_t = torch.tensor(req_layers, dtype=torch.long)
        capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        if emit_q:
            if req_mode == "all_tokens":
                q_start, q_rows = start_slot, n
            else:
                q_start, q_rows = start_slot + n - 1, 1
            prefix_end = int(abs_end)
        else:
            q_start, q_rows, prefix_end = -1, 0, -1

        records.append(QKReqCaptureRecord(
            req_id=str(req_id),
            k_start=int(start_slot), k_rows=int(n),
            q_start=int(q_start), q_rows=int(q_rows),
            prefix_end=int(prefix_end), num_computed=int(num_computed),
            layers=[int(L) for L in req_layers]))

        plans.append({"req_id": req_id, "n_rows": n, "layers": req_layers,
                      "hookq_mode": req_mode, "emit_q": emit_q})

    registry._qk_step_entries = records
    return plans


def prefix_block_ids(step: StepView, req_index: int, num_blocks: int,
                      group: int = 0) -> Optional[torch.Tensor]:
    """Block ids holding batch row ``req_index``'s cached prefix keys."""
    if not step.block_tables:
        return None
    return step.block_tables[group][req_index, :num_blocks]


def _upload_width(model_runner, step: StepView, cap: int) -> int:
    qsl_np = step.query_start_loc_np
    real_n = int(qsl_np[-1]) if qsl_np.size else 0
    try:
        sizes = getattr(model_runner, "cudagraph_batch_sizes", None)
        maxbs = max(sizes) if sizes else cap
    except Exception:  # noqa: BLE001
        maxbs = cap
    return max(1, min(int(cap), max(real_n, int(maxbs))))


_IDLE_ROUTE_KEY = ("__idle__",)


def _capture_idle_key(step: StepView, registry, output_attr: str) -> Optional[tuple]:
    req_ids = step.req_ids
    if not req_ids:
        return None
    default_hooks_on = getattr(registry, "_default_hooks_on", "prefill")
    for i in range(step.num_reqs):
        extra = step.extra_args_for(i)
        if not extra or extra.get(output_attr) is None:
            continue
        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            is_prefill = bool(step.is_prefilling_np[i])
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue
        return None
    return _IDLE_ROUTE_KEY


def install_prepare_inputs_routing(model_runner, worker, build_routing_fn,
                                   label: str = "qk", routing_key_fn=None) -> None:
    """Wrap V2's ``prepare_inputs`` so routing lands after input prep and before the forward."""
    flag = f"_mia_{label}_prep_wrapped"
    if getattr(model_runner, flag, False):
        return
    require_v2_runner(model_runner)
    stash = install_request_arg_stash(model_runner)
    setattr(model_runner, flag, True)

    orig_prepare = model_runner.prepare_inputs
    _no_skip = os.environ.get("MIA_ROUTE_NO_SKIP") == "1"

    def wrapped_prepare_inputs(*args, **kwargs):
        input_batch = orig_prepare(*args, **kwargs)
        registry: Optional[HostRegistry] = get_registry(worker, label)
        if registry is None or not registry.should_capture:
            return input_batch
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return input_batch
        except Exception:  # noqa: BLE001
            pass
        try:
            registry.begin_step()
            step = step_view(model_runner, input_batch, stash)
            width = _upload_width(model_runner, step, registry.cap)

            key = (routing_key_fn(step, registry) if routing_key_fn is not None
                   else registry.routing_key(step))
            if (not _no_skip
                    and key is not None
                    and key == getattr(registry, "_last_route_key", None)
                    and width == getattr(registry, "_last_route_width", None)):
                registry._pending_plans = getattr(registry, "_last_plans", [])
                PROF.incr("graph.route.skip")
                return input_batch

            with PROF.timed("graph.route"):
                if getattr(registry, "gpu_routing", False):
                    registry._pending_assignments = []
                    plans = registry.build_and_upload_gpu(step, width, build_routing_fn)
                    PROF.incr("graph.route.gpu")
                elif getattr(registry, "incremental_enabled", False):
                    registry._pending_assignments = []
                    plans = build_routing_fn(step, registry)
                    uploaded = registry.apply_incremental_routing(
                        registry._pending_assignments, width)
                    PROF.incr("graph.route.upload" if uploaded
                              else "graph.route.noupload")
                else:
                    with PROF.timed("graph.route.reset", tier=2):
                        registry.reset_pinned(width)
                    with PROF.timed("graph.route.buildfn", tier=2):
                        plans = build_routing_fn(step, registry)
                    with PROF.timed("graph.route.upload_h2d", tier=2):
                        registry.upload(width)
            registry._pending_plans = plans
            registry._last_route_key = key
            registry._last_route_width = width
            registry._last_plans = plans
            PROF.incr("graph.route.build")
        except ApertureBackpressureError:
            raise
        except Exception as e:  # noqa: BLE001
            registry._pending_plans = []
            registry._last_route_key = None
            if hasattr(registry, "force_full_routing"):
                registry.force_full_routing()
            PROF.incr("graph.route.errors")
            print(f"[graph/install] FATAL: prepare_inputs routing failed ({label}: {e}). "
                  f"MIA raises rather than continuing without capture/steering: a run that "
                  f"silently stops capturing is indistinguishable from a complete one.",
                  flush=True)
            raise
        return input_batch

    model_runner.prepare_inputs = wrapped_prepare_inputs
    print(f"[graph/install] prepare_inputs routing wrapper installed ({label})")


@contextlib.contextmanager
def _stale_view_guard(registry):
    if registry is None:
        yield
        return
    capture_index_all = getattr(registry, "capture_index_all", None)
    saved = capture_index_all.clone() if capture_index_all is not None else None
    if capture_index_all is not None:
        capture_index_all.zero_()
    try:
        yield
    finally:
        if capture_index_all is not None:
            capture_index_all.copy_(saved)


def _run_dummy_pass(orig_execute_model, registry, scheduler_output, args, kwargs):
    with _stale_view_guard(registry):
        return orig_execute_model(scheduler_output, *args, **kwargs)


def install_execute_model_wrapper(model_runner, worker) -> None:
    """Install QK aperture routing (``prepare_inputs`` wrap) and the per-step drain (``execute_model``)."""
    if getattr(model_runner, "_mia_qk_wrapped", False):
        return

    model_runner._mia_qk_wrapped = True

    def _qk_routing_key(step, registry):
        return _capture_idle_key(step, registry, "output_qk")

    install_prepare_inputs_routing(model_runner, worker, _build_routing, label="qk",
                                   routing_key_fn=_qk_routing_key)

    registry: Optional[HostRegistry] = get_registry(worker, "qk")
    if registry is not None:
        registry._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")
        registry._worker_score_mode = getattr(worker, "_score_mode_default", False)
        registry._worker_score_head = getattr(worker, "_score_head_default", 0)

    aperture = getattr(registry, "_qk_aperture", None) if registry is not None else None
    drain = None
    _sync_drain = os.environ.get("MIA_APERTURE_SYNC_DRAIN", "0") == "1"
    if registry is not None and aperture is not None:
        from mia.graph.aperture_drain_hs import _torch_dtype_name
        from mia.graph.aperture_drain_qk import (
            MultiLayerQKApertureDrain, OffLoopQKApertureDrain)
        layers = [(layer_num, host.q_buf, host.k_buf)
                  for layer_num, host in registry.iter_hosts()]
        buf_dtype = layers[0][1].dtype if layers else torch.float32
        q_dim = int(layers[0][1].shape[1]) if layers else 0
        k_dim = int(layers[0][2].shape[1]) if layers else 0
        base = os.environ.get("MIA_APERTURE_DIR", "./qk_aperture_dump")
        shard = getattr(worker, "_qk_shard", None)
        tp_rank = shard.tp_rank if shard is not None else resolve_tp_coords(worker)[0]
        run_dir = os.path.join(base, rank_dir_name(tp_rank))
        header = {"dtype": _torch_dtype_name(buf_dtype),
                  "q_row_shape": [q_dim], "k_row_shape": [k_dim],
                  "q_dim": q_dim, "k_dim": k_dim,
                  "hookq_mode": getattr(worker, "hookq_mode", "all_tokens")}
        if shard is not None:
            header.update(shard.as_header())
        header["num_layers"] = int(getattr(worker, "_qk_num_layers", len(layers)))
        _shape = predict_capture_write_shape(
            worker, "qk", str(header.get("hookq_mode") or "all_tokens"), int(aperture.n_slots))
        if _sync_drain:
            drain = MultiLayerQKApertureDrain(aperture, layers, run_dir, header, shape=_shape)
            registry._qk_consumer = None
            _mode = "sync per-step"
        else:
            _per_request = os.environ.get("MIA_APERTURE_PER_REQUEST", "0") == "1"
            drain = OffLoopQKApertureDrain(aperture, layers, run_dir, header,
                                           per_request=_per_request, shape=_shape)
            drain.start()
            registry._qk_consumer = drain
            _pr = " + per-request delivery" if _per_request else ""
            _mode = f"OFF-LOOP consumer thread (drain_aperture={drain._aperture_depth}){_pr}"
        worker._qk_drain = drain
        worker._qk_run_dir = run_dir
        print(f"[graph/install] QK aperture write path (tp_rank {int(tp_rank)}): "
              f"{drain.write_path_summary()}", flush=True)
        import atexit
        atexit.register(lambda d=drain: d.close())
        print(f"[graph/install] QK aperture drain ON -> {run_dir} "
              f"(R={aperture.n_slots} rows/layer, {len(layers)} layers, {_mode})", flush=True)
    else:
        print("[graph/install] no capture aperture; QK drain NOT wired", flush=True)

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run") or kwargs.get("is_profile"):
            return _run_dummy_pass(orig_execute_model, get_registry(worker, "qk"),
                                   scheduler_output, args, kwargs)

        registry: Optional[HostRegistry] = get_registry(worker, "qk")
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        registry._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")
        registry._worker_score_mode = getattr(worker, "_score_mode_default", False)
        registry._worker_score_head = getattr(worker, "_score_head_default", 0)

        with PROF.timed("graph.forward"):
            result = orig_execute_model(scheduler_output, *args, **kwargs)

        plans = getattr(registry, "_pending_plans", None) or []
        drain = getattr(worker, "_qk_drain", None)

        if aperture is not None and layers:
            _cap_rows = int(getattr(registry, "_qk_step_rows", 0) or 0)
            if _cap_rows > 0:
                _elt = layers[0][1].element_size()
                PROF.gauge("captured.bytes.qk",
                           float(_cap_rows) * len(layers) * (q_dim + k_dim) * _elt)
        if plans and drain is not None:
            entries = getattr(registry, "_qk_step_entries", None) or []
            if _sync_drain:
                with PROF.timed("graph.drain"):
                    drain.record_entries(entries)
                    drain.drain_once()
            else:
                event = None
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record()
                start_logical = getattr(registry, "_qk_step_start", None)
                n_rows = int(getattr(registry, "_qk_step_rows", 0) or 0)
                if n_rows > 0 and start_logical is not None:
                    drain.enqueue(entries, start_logical, n_rows, event)

        if drain is not None and getattr(drain, "per_request", False):
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if finished:
                for _rid in finished:
                    drain.enqueue_finish(_rid)

        _fin_evidence = getattr(scheduler_output, "finished_req_ids", None)
        if _fin_evidence:
            PROF.incr("hook.fire.qk", len(_fin_evidence) * len(layers))

        registry._pending_plans = []
        registry._qk_step_entries = []
        registry._qk_step_start = None
        registry._qk_step_rows = 0

        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install] execute_model wrapper installed (QK aperture drain)")


_LOAD_MODEL_PATCHED = False


def patch_worker_load_model() -> None:
    """Patch ``Worker.load_model`` to install the graph path after the model is built."""
    global _LOAD_MODEL_PATCHED
    if _LOAD_MODEL_PATCHED:
        return

    from vllm.v1.worker.gpu_worker import Worker

    orig_load_model = Worker.load_model

    def patched_load_model(self, *args, **kwargs):
        result = orig_load_model(self, *args, **kwargs)

        if not graph_mode_enabled():
            return result

        graph_install = getattr(self, "graph_install", None)
        if not callable(graph_install):
            return result

        try:
            graph_install()
        except MiaRefusal:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[graph/install] graph install FAILED ({e}); continuing "
                  f"without graph capture.")
            PROF.incr("graph.install.errors")

        return result

    Worker.load_model = patched_load_model
    _LOAD_MODEL_PATCHED = True


__all__ = [
    "set_graph_mode",
    "graph_mode_enabled",
    "patch_worker_load_model",
    "install_qk_hosts",
    "install_execute_model_wrapper",
    "install_prepare_inputs_routing",
    "_capture_idle_key",
    "_stale_view_guard",
    "_run_dummy_pass",
]

