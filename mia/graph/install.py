"""CUDA-graph QK capture — install, per-step routing, and egress."""
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


# ---------------------------------------------------------------------------
# Process-wide graph-mode flag
# ---------------------------------------------------------------------------
# Mirrored in an env var because create_engine_config runs in the driver but
# load_model runs in the (possibly spawned) worker, where the module global would
# not survive — the env var, set before workers launch, does.
_GRAPH_MODE_ENV = "MIA_GRAPH_MODE"
_graph_mode_enabled = False


def set_graph_mode(enabled: bool) -> None:
    """Arm/disarm the graph path. Sets both the module global and the env var so
    spawned workers (where load_model runs) inherit the decision."""
    global _graph_mode_enabled
    _graph_mode_enabled = bool(enabled)
    os.environ[_GRAPH_MODE_ENV] = "1" if enabled else "0"


def graph_mode_enabled() -> bool:
    """True if the graph path should install: module global (driver) or inherited
    env var (worker subprocess)."""
    return _graph_mode_enabled or os.environ.get(_GRAPH_MODE_ENV) == "1"


# ---------------------------------------------------------------------------
# Class-level Attention.forward wrap (survives the Dynamo instance-hook bypass)
# ---------------------------------------------------------------------------
# Per-class originals → wrap is idempotent and reversible. One wrapped class
# serves every layer; an instance with no host/layer attr falls straight through.
_WRAPPED_ATTN_CLASSES: Dict[type, Any] = {}
_HOST_ATTR = "_mia_qk_host"          # per-instance QKCaptureHost
_REG_ATTR = "_mia_qk_registry"       # per-instance HostRegistry back-ref

# Prefix-K forward-context stash inside the traced forward. Off by default: the
# get_forward_context() call there is novel and may graph-break, and it's only
# needed for prefix-K (num_cached > 0). Enable once core capture is confirmed.
_PREFIXK_STASH = os.environ.get("MIA_QK_PREFIXK_STASH") == "1"


def _require_buffer_mode() -> None:
    """Buffer mode is the only FULL-cudagraph capture path.

    The PIECEWISE op/seam capture mechanism (v0.3.0) was removed; ``MIA_QK_CAPTURE``
    is retained only so an explicit ``op``/``seam`` request fails loud instead of silently
    running buffer. Unset (or ``=buffer``) is the normal case.
    """
    mode = os.environ.get("MIA_QK_CAPTURE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise MiaConfigurationError(
            f"MIA_QK_CAPTURE={mode!r} is no longer supported: the PIECEWISE op/seam "
            "capture mode was removed. Buffer mode is the only FULL-cudagraph capture path; "
            "unset MIA_QK_CAPTURE or set it to 'buffer'.")

# Batched egress (MIA_BATCHED_EGRESS, default ON; =0 is the per-request fallback): ONE
# index_select per layer gathers only the rows being saved into a compact own-storage tensor;
# each request takes a VIEW into it, cutting launches from O(layers x requests) to O(layers).
# Byte-identical: index_select copies into fresh storage, so views survive buffer reuse exactly
# like the old per-request .clone() did.
_BATCHED_EGRESS = os.environ.get("MIA_BATCHED_EGRESS", "1") == "1"

_capture_dbg = {"n": 0}  # one-shot capture confirmation counter

# _NO_PREFIXK skips prefix-K reconstruction (bisection / no prefix caching).
_NO_PREFIXK = os.environ.get("MIA_QK_NO_PREFIXK") == "1"

# ---------------------------------------------------------------------------
# Capture-aperture helpers. The QK path scatters q + k into TWO parallel per-layer apertures
# sharing ONE CaptureAperture cursor, then drains them off-loop to durable per-layer
# raw files (graph/aperture_drain_qk.py). Mirrors the HS aperture (graph/install_hs.py) —
# duplicated here (not imported) so install_hs's import of this module stays acyclic.
# ---------------------------------------------------------------------------

def _aperture_reserve_or_block(aperture: CaptureAperture, n: int, consumer=None) -> int:
    """Reserve ``n`` contiguous aperture rows for this step's q/k capture, BLOCKING (polling) on
    aperture-full rather than dropping (the never-drop contract). Runs on the ENGINE thread inside the
    ``prepare_inputs`` routing wrapper.

    With the OFF-LOOP consumer drain the block is genuine: ``time.sleep`` releases the GIL, so the
    consumer thread can drain earlier (already-forwarded) steps and ``advance_drain`` to free rows —
    no deadlock, since the consumer only drains PAST steps whose forwards already completed. With
    the SYNCHRONOUS drain (``consumer is None``) the previous step already fully drained, so the
    reserve succeeds at once and the block only engages on a mis-sized aperture. Fails loud
    (``ApertureBackpressureError``, re-raised by the routing wrapper) on a DEAD consumer or a mis-sized
    aperture. QK twin of ``install_hs._aperture_reserve_or_block``."""
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
    """Size the shared QK capture aperture, resolved LAZILY at install (after vLLM carves KV).

    ``aperture_bytes`` comes from ``resolve_aperture_bytes_auto``: an explicit
    ``MIA_APERTURE_GPU_BYTES`` always wins; the default is 4 GiB grown to one max-token step
    (``rows_needed`` = ``max_num_batched_tokens`` rows of ``num_layers * (q_dim + k_dim)``) when that
    is larger, refusing at install when the grown default does not fit the free margin.
    ``q_dim`` / ``k_dim`` are THIS RANK's sharded widths (``graph/tp_shard.qk_shard``) -- each TP rank
    owns its own aperture for its own heads.
    q and k share ONE cursor (the ``capture_qk`` op scatters both at the SAME index), so a slot costs
    ``(q_dim + k_dim)`` elements; with ``num_layers`` parallel per-layer aperture pairs, ``R`` = per-layer
    aperture rows = ``aperture_bytes // (num_layers * (q_dim + k_dim) * dtype_size)``. Both ``q_buf`` and
    ``k_buf`` get ``R`` usable rows (+1 sentinel). Fails loud if ``R < 1`` (spec §8: no silent degrade).
    Returns ``(R, aperture_bytes)``."""
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
    """Coerce a query_start_loc/seq_lens carrier (CpuGpuBuffer-like or plain
    tensor) to a 1-D CPU tensor, preferring the CPU mirror to avoid a device sync.
    None if nothing usable."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    # CpuGpuBuffer: .cpu is a CPU tensor attribute, NOT the Tensor.cpu method.
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
    """Class-wrap ``cls.forward``, idempotent per class. Calls ``host.capture(q,k)``
    (the buffer-mode static scatter). args[0]=post-RoPE q, args[1]=k (the eager hook's
    input[0]/input[1]); capture runs before the original forward.

    Also snapshots the live forward context onto the registry on its first fire each
    step — the only place it's provably live — so egress can read kv_cache/attn_metadata
    for prefix-K post-forward.
    """
    if cls in _WRAPPED_ATTN_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_ATTN_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            # Static-buffer scatter. do_capture is an install-time constant, so this
            # branch adds no data-dependent control flow to the traced region.
            host = getattr(self, _HOST_ATTR, None)
            if host is not None and host.do_capture:
                # Stash the live forward context for post-forward prefix-K egress (only the first
                # attn layer of the step takes effect; bare-except so a torn-down context never
                # breaks the forward). Gated off by default since get_forward_context() in traced
                # Python may graph-break.
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
                # args[0]=post-RoPE q, args[1]=k (positional in every Attention
                # signature; guard length defensively).
                if len(args) >= 2:
                    host.capture(args[0], args[1])
            return orig_fwd(self, *args, **kwargs)

        return wrapped

    cls.forward = make_wrapped(orig_forward)


def _resolve_max_num_batched_tokens(worker) -> int:
    """Return max_num_batched_tokens for the buffer cap, version-robustly: probe
    model_runner.scheduler_config then worker.vllm_config.scheduler_config. Raises
    if neither carrier has it."""
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
    """``max_num_seqs`` (the scheduler's in-flight request bound), version-robustly, or None.

    Unlike its ``max_num_batched_tokens`` sibling this NEVER raises: it feeds only the aperture
    write path's size PREDICTION (``aperture_sink.predict_rows_per_write``), which decides on
    alignment alone when a bound is missing. Nothing correctness-bearing reads it."""
    for owner in (getattr(worker, "model_runner", None), worker,
                  getattr(worker, "vllm_config", None)):
        sched = getattr(owner, "scheduler_config", None) if owner is not None else None
        n = getattr(sched, "max_num_seqs", None) if sched is not None else None
        if n:
            return int(n)
    return None


def predict_capture_write_shape(worker, kind: str, capture_mode: str, aperture_rows: int):
    """``aperture_sink.WriteShape`` -- an UPPER BOUND on one step's per-file write -- for this
    worker's capture, or None when no bound can be read (``auto`` then decides on alignment alone,
    the rule before 2026-09-20).

    This is where the write path learns how big a write will be WITHOUT any traffic: the
    worker-wide capture mode (``hs_mode`` / ``hookq_mode``) and the scheduler's two bounds are all
    resolved at install. The capture mode is only a FALLBACK a request can override, which is why
    the bound it feeds is an upper one -- see ``predict_rows_per_write``. NEVER raises: a
    write-mode hint must not be able to fail an engine boot."""
    from .aperture_sink import predict_rows_per_write
    try:
        return predict_rows_per_write(
            kind, capture_mode=capture_mode,
            max_batched_tokens=_resolve_max_num_batched_tokens(worker),
            max_num_seqs=_resolve_max_num_seqs(worker),
            aperture_rows=aperture_rows)
    except Exception as e:  # noqa: BLE001 -- a size HINT; alignment-only is the safe fallback
        print(f"[graph/install] could not predict the {kind} aperture write size ({e!r}); the "
              f"write mode will be decided on alignment alone", flush=True)
        return None


# ---------------------------------------------------------------------------
# Host install (EVERY TP rank, each for its own heads) — runs at load_model, before compile/capture
# ---------------------------------------------------------------------------


def install_qk_hosts(worker) -> Optional[HostRegistry]:
    """Class-wrap Attention.forward (and, in buffer mode, build per-layer hosts).

    Runs at load_model, after the model is built but BEFORE compile/capture, so
    buffers land before the cudagraph pool and their data_ptrs stay fixed across
    replays. Runs identically on every TP rank: each rank's hosts are sized to, and
    capture, that rank's own shard of the heads (``graph/tp_shard.qk_shard``), and each
    rank drains to its own ``tp_rank_<r>/``. Returns the HostRegistry (buffer mode) or None.
    """
    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install] no model on model_runner; skip QK host install")
        return None

    _require_buffer_mode()  # op/seam removed; fail loud on an explicit request
    refuse_pipeline_parallel(getattr(worker.parallel_config, "pipeline_parallel_size", 1),
                             "QK graph install")

    # Register the capture op(s) once per process, BEFORE any wrap can fire.
    register_graph_ops()

    # EVERY TP rank captures. Q and K are NOT replicated: they are the output of the
    # column-parallel qkv_proj, read before attention, so rank r holds only its own slice of the
    # heads (graph/tp_shard.py). A rank-0-only capture recorded H_q/tp of the query heads and
    # labelled it the layer. Each rank writes its own shard to tp_rank_<r>/ with the geometry in
    # the sidecar header; aperture_reader.merge_qk_aperture_ranks rebuilds the global layer.
    # All ranks now bake the same op against their own buffers, so the compiled graph is
    # structurally identical on every rank (the NCCL graph-capture lockstep hazard documented in
    # graph/install_hs.py never arises for QK).
    tp_rank, tp_size = resolve_tp_coords(worker)
    should_capture = True

    # Model dims pulled EXACTLY like the eager worker so buffer widths and the
    # analyzer config match.
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    num_h = int(getattr(text_cfg, "num_attention_heads"))
    num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
    hidden = int(getattr(text_cfg, "hidden_size"))
    # ONE head_dim for buffers, the sidecar header and _conf (the eager worker uses the same
    # helper): the real post-RoPE per-head width. The analyzers view a merged row as
    # (num_attention_heads, head_dim), so _conf must describe the width the merge produces.
    head_dim = qk_conf_head_dim(text_cfg)            # for _conf (== the eager worker's)
    buf_head_dim = head_dim                           # the real per-head width q/k carry
    attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
    # Under TP each rank's Attention op produces only its SHARD of heads (vLLM shards attention
    # heads by tp_size, replicating KV heads when there are fewer than tp_size); head_dim is not
    # sharded, only the head COUNT is. The static capture buffers are sized to THIS rank's
    # sharded width, or the capture_qk scatter's index_copy_ mismatches q_src and crashes at
    # cudagraph warmup. Byte-identical at tp_size=1. _conf deliberately keeps the FULL
    # counts: it describes the MERGED artifact the analyzers consume, not one rank's shard.
    shard = qk_shard(tp_rank, tp_size, num_h, num_kv, buf_head_dim)
    q_dim = shard.q_width
    k_dim = shard.k_width

    # _conf feeds get_captured_states / flush_disk payload["config"] — populate it
    # identically to the eager worker.
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
    # Worker-wide fallback for hookq_mode when a request omits it (matches eager).
    if not hasattr(worker, "hookq_mode"):
        worker.hookq_mode = "all_tokens"
    # v0.6.0 score-capture defaults (graph mode skips install_hooks, so set them here too).
    # SCORE mode is OUT OF SCOPE for the QK capture-aperture path (v1 = raw q/k only): the score is an
    # O(S^2) [S_q,S_k] matrix recomputed at retrieval from staged Q/K, which the aperture's per-step
    # scatter → durable q/k files does not stage. A worker-wide score default is a fail-loud install
    # error (a per-request qk_capture="score" is refused in _build_routing).
    worker._score_mode_default = os.environ.get("MIA_QK_SCORE", "0") == "1"
    worker._score_head_default = int(os.environ.get("MIA_QK_SCORE_HEAD", "0"))
    worker._score_dtype = torch.float16
    if worker._score_mode_default:
        raise MiaConfigurationError(
            "MIA_QK_SCORE=1 (attention-score capture) is not supported on the QK capture-aperture "
            "path (v1 = raw q/k only). Unset it, or use the eager/bank path for score capture.")

    # Egress buckets — same dicts the eager path / RPC retrieval consume.
    if not hasattr(worker, "_captured_states") or worker._captured_states is None:
        worker._captured_states = {}
    if not hasattr(worker, "_disk_states") or worker._disk_states is None:
        worker._disk_states = {}

    # Rank-1c: writer PROCESS for off-GIL serialize+write (no-op unless MIA_WRITER_PROCESS=1).
    from mia.graph.writer_process import init_writer_process
    init_writer_process(worker)

    # CAP = max_num_batched_tokens: every token in the largest possible all-prefill
    # batch can land in a distinct buffer row.
    cap = _resolve_max_num_batched_tokens(worker)

    device = next(model.parameters()).device

    # Enumerate matched attn modules first so we know num_layers for the registry.
    matched = list(iter_matched_modules(model, match_attn))
    if not matched:
        print("[graph/install] no attention modules matched ATTN_PATTERNS; "
              "QK graph capture inactive")
        set_registry(worker, "qk", None)
        return None
    num_layers = max(layer_num for _, _, layer_num in matched) + 1
    # The shard is DERIVED from the config (mirroring vLLM's QKVParallelLinear); refuse if the live
    # Attention modules disagree, so a model that shards differently can never write shards whose
    # header lies about which heads they hold.
    check_attn_modules_match_shard(matched, shard)
    worker._qk_num_layers = num_layers

    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    # ---- Capture-aperture geometry. Each captured layer holds TWO persistent aperture buffers — q_buf
    # (R+1, q_dim) and k_buf (R+1, k_dim) — with row R the shared SENTINEL (pad / no-capture
    # discard). ALL layers advance in lockstep off ONE shared CaptureAperture cursor (q + k scatter
    # to the SAME index via capture_qk), so one reserve serves every layer; the HostRegistry's
    # per-(layer, token) routing slabs now carry ADVANCING aperture slots, not batch positions. Buffers
    # are built here (load_model, before the cudagraph pool) so their data_ptrs stay fixed across
    # replays. Every TP rank captures its own heads, so every rank bakes the same op (symmetric).
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
        # ONE shared logical cursor across the parallel per-layer q/k apertures. aperture.buf is NOT
        # allocated (storage is the per-layer q_buf/k_buf); we use only
        # reserve/physical_slots/segments/advance_drain/free_rows/SENTINEL. row_bytes/row_shape
        # describe a K row for the mmap default sizing (the drain sizes q + k files independently).
        elem = torch.empty(0, dtype=buf_dtype).element_size()
        aperture = CaptureAperture(row_bytes=k_dim * elem, n_slots=R, device=device,
                              dtype=buf_dtype, row_shape=(k_dim,))
        registry._qk_aperture = aperture
        registry._qk_step_entries = []
        # Pad / no-capture lanes route to the aperture SENTINEL row (R), NOT 0 — aperture slot 0 is a REAL
        # storage row now. reset_pinned fills this; the incremental / GPU routers use column-diff
        # (not advancing positions), so disable them → the wrapper takes the legacy
        # reset -> build -> upload branch (identical to HS).
        registry.sentinel_row = aperture.SENTINEL
        registry.incremental_enabled = False
        registry.gpu_routing = False
        registry.capture_index_all.fill_(aperture.SENTINEL)
        for _slot in registry._aperture.slots:
            _slot["capture_index"].fill_(aperture.SENTINEL)
        worker._capture_aperture = aperture

    # Build a host per matched module (with its (R+1, dim) aperture buffers), attach, wrap its class.
    for name, module, layer_num in matched:
        if should_capture:
            q_buf = torch.zeros(R + 1, q_dim, dtype=buf_dtype, device=device)
            k_buf = torch.zeros(R + 1, k_dim, dtype=buf_dtype, device=device)
            host = QKCaptureHost(
                module_name=name,
                layer_num=layer_num,
                cap=cap,                 # token cap: register_host check + routing slab width
                q_dim=q_dim,
                k_dim=k_dim,
                dtype=buf_dtype,
                device=device,
                do_capture=True,
                q_buf=q_buf,             # aperture storage: (R+1, dim), NOT (cap+1, dim)
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


# ---------------------------------------------------------------------------
# Routing: build the pinned per-(layer, token) destination index for one step
# ---------------------------------------------------------------------------


def _build_routing(step: StepView, registry: HostRegistry) -> list:
    """Capture-aperture routing: map each captured token's batch column to an ADVANCING shared-aperture
    slot (persists until the drain reads it), NOT the old batch-position row ``p+1`` overwritten
    each step.

    q and k scatter to the SAME index (the ``capture_qk`` op writes q_buf[idx] AND k_buf[idx]), so
    ONE ``aperture.reserve(qlen)`` per request advances both apertures and serves every requested layer. K is
    kept EVERY step (the per-step k rows concatenate to the full ``k_full``), so the reserve is the
    WHOLE span ``qlen = end - start`` every step (unlike the HS aperture's last_token 1-row reserve). q is
    kept only on ``emit_q`` (all_tokens: every step; last_token: the final prefill chunk + each decode
    step) — recorded in the METADATA, not the routing: the op scatters q into all the reserved slots
    regardless, and a non-emit step's q rows are simply never referenced by the sidecar (dead,
    harmless, like the HS sentinel lanes).

    LayerEntry COLLAPSE: stashes ONE :class:`QKReqCaptureRecord` per capturing request on
    ``registry._qk_step_entries`` — carrying that request's OWN 0-based layer list — instead of
    fanning out ``num_layers`` ``QKStepEntry`` objects on the engine loop (the O(reqs x layers)
    per-fire allocation; ``QKStepEntry`` has 8 fields, all shared across a request's layers except
    ``layer``). The drain expands each record into the identical flat ``QKStepEntry`` list OFF the
    loop, so the sidecar / demux are byte-for-byte unchanged. Records this step's shared-aperture start
    slot + total reserved rows (``_qk_step_start`` / ``_qk_step_rows``) for the off-loop consumer.
    Gating (output_qk filter, hooks_on, hookq_mode, chunked-last_token emit_q) mirrors the eager
    qkv_hook. Runs pre-forward on the ENGINE thread; the reserve BLOCKS on a full aperture (never drops),
    and ApertureBackpressureError propagates (the wrapper re-raises it).

    Reads only ``step`` (the immutable snapshot of `prepare_inputs`'s RETURNED InputBatch) and
    ``registry`` — never the model runner. Worker-wide fallbacks (``_default_hooks_on``,
    ``_worker_hookq_mode``, ``_worker_score_mode``) live on ``registry`` (refreshed each step by
    ``install_execute_model_wrapper``'s wrapper), since they are per-worker config, not per-step data.
    """
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
    capture_index_pinned = registry.capture_index_pinned  # (num_layers, cap)
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

        # output_qk: True (all) | [layer_ids] | {layer: [heads]} — only keys filter layers.
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
        # SCORE mode is out of scope on the aperture path (v1 raw q/k only). A per-request request is
        # refused (skip + a capped warn) rather than silently captured as q/k; the worker-wide
        # default already fails loud at install.
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
        if end > cap:  # invariant: a step's tokens <= max_num_batched_tokens == cap
            end = cap
        if end <= start:
            continue
        qlen = end - start  # this step's scheduled tokens for req i

        # abs_end = absolute key count through this step; num_computed_tokens_np[i] is the pre-step
        # processed count (the cached-prefix length on the request's FIRST capture step — 0 for a
        # fresh prefill; the reader FAILS LOUD on a >0 first step, since v1 does not prepend the
        # trimmed prefix keys from paged KV).
        num_computed = int(step.num_computed_tokens_np[i])
        abs_end = num_computed + qlen

        if layer_filter is None:
            req_layers = list(range(registry.num_layers))
        else:
            req_layers = [L for L in layer_filter if 0 <= L < registry.num_layers]
        if not req_layers:
            continue

        # last_token + chunked prefill: K still ACCUMULATES on every prefill chunk (so k_full is
        # complete), but Q + prefix marker are emitted only on the FINAL chunk. all_tokens emits Q
        # every step; decode is always "final" (emit Q).
        emit_q = True
        if req_mode == "last_token" and is_prefill:
            # PROMPT length (prompt_len_np), not prefill_len_np -- prefill_len can exceed the
            # prompt after preemption-resume (see mia.runner.step_view).
            num_prompt = int(step.prompt_len_np[i])
            emit_q = abs_end >= num_prompt

        # ---- aperture reserve (shared cursor) + advancing slots ----
        # Reserve the WHOLE span (K needs every token every step) and route ALL qlen columns to the
        # reserved slots. q rides the same slots; emit_q only gates the METADATA q record.
        n = qlen
        start_slot = _aperture_reserve_or_block(aperture, n, consumer)
        if registry._qk_step_start is None:
            registry._qk_step_start = start_slot
        registry._qk_step_rows += n
        phys = aperture.physical_slots(start_slot, n)                 # n ints in [0, R)
        phys_t = torch.tensor(phys, dtype=torch.int64)
        layer_idx_t = torch.tensor(req_layers, dtype=torch.long)
        capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        # q metadata (LOGICAL slots, wrap-agnostic — the drain writes files in logical order):
        #   all_tokens : the whole span -> q_start = start_slot, q_rows = qlen
        #   last_token : only the span's last token -> q_start = start_slot + qlen - 1, q_rows = 1
        #   non-emit   : q_start = -1, q_rows = 0 (dead q rows, unreferenced)
        if emit_q:
            if req_mode == "all_tokens":
                q_start, q_rows = start_slot, n
            else:
                q_start, q_rows = start_slot + n - 1, 1
            prefix_end = int(abs_end)
        else:
            q_start, q_rows, prefix_end = -1, 0, -1

        # LayerEntry COLLAPSE: ONE per-request record (this request's OWN 0-based layers in req_layers
        # order) instead of num_layers QKStepEntry objects here; the drain expands it off-loop into the
        # identical flat per-(req, layer) QKStepEntry list. Every field but `layer` is shared across
        # this request's layers, so the record is exact.
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


# ---------------------------------------------------------------------------
# Prefix-K block ids (V2). `prefix_block_ids` is the batch-ordered replacement for the
# V1 `input_batch.block_table.block_tables[0].get_device_tensor(num_reqs)[req_idx]`
# read a prior version of this module used inside a buffer-mode cached-prefix-key
# reader (`_read_cached_keys_buffer`). That reader had ZERO call sites at the time of
# the V2 port -- the per-step egress that fed it cached keys was deleted with the QK
# aperture port, and the v1 aperture path FAILS LOUD on a cached prefix instead
# (`aperture_reader.py` raises when a request's first capture step has
# `num_computed > 0`) -- so it was deleted rather than mechanically ported. This
# helper is what remains: the V2-shaped block-id lookup, ready for whichever future
# reader reconstructs cached prefix keys from the persistent paged KV cache (still
# reachable, unchanged, via `compilation_config.static_forward_context[name].kv_cache`
# -- see `key_cache_from_layer_kv` in `mia.workers.qk_capture_worker`).
# ---------------------------------------------------------------------------


def prefix_block_ids(step: StepView, req_index: int, num_blocks: int,
                      group: int = 0) -> Optional[torch.Tensor]:
    """Block ids holding batch row ``req_index``'s cached prefix keys.

    V1: ``input_batch.block_table.block_tables[g].get_device_tensor(num_reqs)[i]`` --
    ``i`` there is an input-batch SLOT, a different numbering than the batch row.
    V2: ``step.block_tables[g][i]`` -- ``mia.runner.step_view`` already sliced
    ``runner.block_tables.input_block_tables`` (itself already gathered into BATCH
    order by V2's ``gather_block_tables()``) to ``num_reqs``, so ``req_index`` is a
    batch row in both worlds and no slot-index indirection is needed here.

    Returns None if this step has no KV-cache groups at all (e.g. an
    attention-free model), else a ``(num_blocks,)`` slice of group ``group``'s
    block-table row.
    """
    if not step.block_tables:
        return None
    return step.block_tables[group][req_index, :num_blocks]


# ---------------------------------------------------------------------------
# prepare_inputs routing — the load-bearing integration point (plan §5)
# ---------------------------------------------------------------------------
#
# V2's InputBatch is TRANSIENT (built and returned by `prepare_inputs`, never stored
# on the runner), so the pre-D1 helpers that reconstructed a host-side query_start_loc
# by reaching into `model_runner.input_batch` / `scheduler_output` post hoc
# (`_preforward_qsl`, `_runner_qsl`) are gone: `prepare_inputs`'s RETURN VALUE already
# carries `query_start_loc_np`, and `step_view()` snapshots it directly. Wrapping the
# higher, return-value-bearing method made that whole reconstruction unnecessary.


def _upload_width(model_runner, step: StepView, cap: int) -> int:
    """Smallest contiguous column prefix the routing upload must cover this step.

    The capture/steer ops read ``index[:n]`` where ``n = q.shape[0]`` is the PADDED
    token count — a cudagraph decode batch is padded up to a captured size — so the
    upload must cover ``[0, padded_n)`` with ``[real_tokens, padded_n)`` left at the
    sentinel (0). We bound ``padded_n`` above by ``max(cudagraph_batch_sizes)`` (the
    largest captured decode graph) and the real token count (eager prefill is not
    padded). Uploading only ``[:, :real_tokens]`` would be WRONG: the cudagraph
    padding rows would read stale routing and scatter dummy tokens into real buffer
    rows. ``max(real_n, max_graph_batch)`` is a safe upper bound on ``padded_n`` in
    every case: decode pads up to a captured size (≤ max_graph_batch); a prefill that
    fits a captured size pads up to it (≤ max_graph_batch); a prefill too big to
    cudagraph runs eager (``padded_n == real_n``). Returns a width in ``[1, cap]``.
    ``model_runner`` is read directly here (not via ``step``): ``cudagraph_batch_sizes``
    is static engine config set once at init, not per-step transient state.
    """
    qsl_np = step.query_start_loc_np
    real_n = int(qsl_np[-1]) if qsl_np.size else 0
    try:
        sizes = getattr(model_runner, "cudagraph_batch_sizes", None)
        maxbs = max(sizes) if sizes else cap
    except Exception:  # noqa: BLE001
        maxbs = cap
    return max(1, min(int(cap), max(real_n, int(maxbs))))


# Sentinel routing key meaning "no request captures anything this step" (W1′
# idle-skip). A constant tuple, so it is identical across consecutive idle decode
# steps → the routing wrapper skips reset/build/upload after ONE transition
# zero-upload, and the resident (zeroed) slab replays as a no-op scatter. Distinct
# from any real per-request signature and from None (None forces a rebuild).
_IDLE_ROUTE_KEY = ("__idle__",)


def _capture_idle_key(step: StepView, registry, output_attr: str) -> Optional[tuple]:
    """W1′ idle-skip key for buffer-mode capture (QK + HS).

    Returns ``_IDLE_ROUTE_KEY`` only when NO request in the batch would emit a
    capture plan this step — a pure non-capturing step, e.g. every decode step of a
    ``hooks_on=prefill`` workload (the profiled qk-lasttok / hs-lasttok case). The
    routing wrapper then skips ``reset_pinned`` + ``_build_routing`` + ``upload``;
    the device slabs (zeroed on the active→idle transition) make the baked-in
    scatter a no-op, so the whole per-step routing tax disappears on idle steps.

    Returns ``None`` (force a rebuild — today's behaviour, byte-identical) the
    moment any request WOULD capture, or when the batch is unreadable. Cheap:
    scalar gating only (no tensor writes, no f-strings), early-exits to ``None`` on
    the first capturing request. Mirrors the active-request gate of
    ``_build_routing`` / ``_build_routing_hs`` up to the plan decision; a request
    that passes this gate but ultimately produces no plan (empty layer set / empty
    range) only costs a missed skip, never wrong data.
    """
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
        # This request would capture this step → not idle → force a rebuild.
        return None
    return _IDLE_ROUTE_KEY


def install_prepare_inputs_routing(model_runner, worker, build_routing_fn,
                                   label: str = "qk", routing_key_fn=None) -> None:
    """Wrap V2's ``prepare_inputs`` so routing lands after input prep and before the
    (possibly cudagraph-replayed) forward.

    V2's ``InputBatch`` is TRANSIENT — built and returned by ``prepare_inputs``,
    never stored on the runner — so the routing tables are built from the RETURN
    VALUE, snapshotted once into an immutable ``StepView`` (``mia.runner.step_view``).
    This is still the legal off-graph home for host writes: the forward has not
    started, and the buffers the in-graph op reads are ours and static. The
    resulting plans are stashed on the registry for the execute_model wrapper's
    post-forward egress.

    ``build_routing_fn(step, registry) -> list`` is every subsystem's routing
    builder (``_build_routing`` / ``_build_routing_hs`` / ``_build_routing_steer``).
    ``routing_key_fn(step, registry)`` is the W1 invalidation-key source; when given
    (capture passes a ``_capture_idle_key``-based function) it overrides
    ``registry.routing_key(step)`` so capture gets the idle-skip without the
    registry needing the worker's gating.

    The per-subsystem registry (``mia.graph.registry.get_registry(worker, label)``)
    is looked up EVERY call, not cached in the closure — ``worker._mia_registries``
    can gain a new subsystem after this wrapper installs.

    Fails loud at install time: ``require_v2_runner`` raises if the live runner is
    not V2 (V1 never degrades silently), and ``model_runner.prepare_inputs`` is
    accessed directly — V2 always defines it, so there is nothing to probe for.
    Idempotent per ``label``. FAIL-LOUD thereafter: a per-step routing failure PROPAGATES
    rather than degrading to "no capture this step". It used to degrade and warn once, which
    left a run that stopped capturing at step 3 looking exactly like a complete one. The
    never-drop contract outranks availability -- ``ApertureBackpressureError`` was already
    exempted for that reason, and every other routing failure drops a capturing request just
    as silently. See the handler for the full reasoning.
    """
    flag = f"_mia_{label}_prep_wrapped"
    if getattr(model_runner, flag, False):
        return
    require_v2_runner(model_runner)
    stash = install_request_arg_stash(model_runner)
    setattr(model_runner, flag, True)

    orig_prepare = model_runner.prepare_inputs
    # DIAGNOSTIC (default off, byte-identical): force every step to re-route (disable the W1
    # idle-skip) so a profiling run gets a clean per-step routing sample at a fixed batch.
    # Re-uploading an identical routing changes no captured value — it only re-does redundant work.
    _no_skip = os.environ.get("MIA_ROUTE_NO_SKIP") == "1"

    def wrapped_prepare_inputs(*args, **kwargs):
        input_batch = orig_prepare(*args, **kwargs)
        registry: Optional[HostRegistry] = get_registry(worker, label)
        if registry is None or not registry.should_capture:
            return input_batch
        # Skip during vLLM's cudagraph capture pass — a host sync / pinned write
        # while capturing is illegal. Buffers are populated at replay, not capture.
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return input_batch
        except Exception:  # noqa: BLE001
            pass
        try:
            registry.begin_step()
            step = step_view(model_runner, input_batch, stash)
            width = _upload_width(model_runner, step, registry.cap)

            # W1 invalidation: when the key + upload width match last step, the device slabs
            # already hold an identical routing, so reset/build/upload is skipped and the step is
            # a pure graph replay. Steering's key is real on a stable batch (bit-identical every
            # decode step). Capture's W1' key returns _IDLE_ROUTE_KEY on a fully non-capturing step
            # (consecutive idle steps skip after one transition zero-upload) and None the moment
            # any request captures. A composition change or prefill<->decode flip moves the key and
            # forces a re-route.
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
                    # GPU-side routing: O(reqs) host work + a GPU scatter fills the slabs,
                    # replacing the O(num_layers x cap) host build. Byte-identical device slabs.
                    # STEER (MIA_STEER_GPU_ROUTING): SteerRegistry resolves configs itself
                    # (ignores build_routing_fn). CAPTURE (MIA_CAPTURE_GPU_ROUTING):
                    # HostRegistry runs build_routing_fn for the gating/plans, then scatters the
                    # capture-index. Each registry sets its own gpu_routing, so they don't cross.
                    registry._pending_assignments = []
                    plans = registry.build_and_upload_gpu(step, width, build_routing_fn)
                    PROF.incr("graph.route.gpu")
                elif getattr(registry, "incremental_enabled", False):
                    # Build plans every step (cheap O(reqs) scalar work) but write+upload ONLY the
                    # capture-index columns that changed since last step. Steady-state decode /
                    # idle steps change 0 columns -> 0 upload (a pure replay); a prefill / finish /
                    # condense touches O(changed).
                    registry._pending_assignments = []
                    plans = build_routing_fn(step, registry)
                    uploaded = registry.apply_incremental_routing(
                        registry._pending_assignments, width)
                    PROF.incr("graph.route.upload" if uploaded
                              else "graph.route.noupload")
                else:
                    # Legacy: reset+upload the full column prefix the (cudagraph-padded)
                    # forward reads — for decode ~max_graph_batch, not the full cap slab.
                    # tier-2 sub-timers (FINE only) decompose the per-fire routing cost into
                    # reset / host-build / H2D-upload so a profiling run can attribute the
                    # O(reqs x layers) scaling; strict no-op unless MIA_PROFILE_FINE=1.
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
            # NEVER-DROP: the capture aperture is full and could not be relieved within the block
            # timeout (a mis-sized aperture or a dead off-loop drain consumer). Do NOT clear the plans
            # and continue — that would silently drop a capturing request. PROPAGATE so the engine
            # step fails LOUD. The reserve genuinely BLOCKED on the engine thread inside
            # build_routing_fn (the off-loop consumer frees rows there); only an unrelievable full
            # aperture reaches here.
            raise
        except Exception as e:  # noqa: BLE001
            registry._pending_plans = []
            registry._last_route_key = None  # force a re-route after an error
            if hasattr(registry, "force_full_routing"):
                registry.force_full_routing()  # re-establish the whole slab next step
            # FAIL LOUD. This handler serves QK, HS and steer alike, and it used to degrade
            # the step to no capture / no steering, warn ONCE, and carry on. That made a run
            # which stopped capturing at step 3 indistinguishable from a complete one: the
            # generate() call succeeds, the artifacts are simply short, and the single warning
            # has long since scrolled away. The PROF counter that carried the true rate is a
            # no-op unless MIA_PROFILE=1, so under the default the one-shot line WAS the
            # entire signal.
            #
            # Why raising, and why it is not the same judgement call as the load_model
            # handler's blanket fallback:
            #   * MIA's never-drop contract says a capturing request is never silently
            #     skipped. ApertureBackpressureError is already re-raised just above for
            #     exactly that reason; every OTHER routing failure drops the same request just
            #     as silently, so the carve-out was covering one anticipated cause, not the
            #     contract.
            #   * The realistic failure here is DETERMINISTIC, not transient: this wrapper
            #     reads V2-internal fields, so the shape that breaks it (vLLM moves or renames
            #     one) breaks every subsequent step too. Degrading does not buy a degraded
            #     run; it buys a complete-looking run with no artifacts.
            #   * It costs nothing on any path that works. If this handler fired on a
            #     validated path, capture would already be missing and the parity suite would
            #     be red -- it is green 12/12. So raising changes behaviour only on paths that
            #     are already broken and currently lying about it.
            # The original exception propagates unchanged (no wrapping) so the traceback still
            # points at the real failure; the line below only says what it means.
            PROF.incr("graph.route.errors")
            print(f"[graph/install] FATAL: prepare_inputs routing failed ({label}: {e}). "
                  f"MIA raises rather than continuing without capture/steering: a run that "
                  f"silently stops capturing is indistinguishable from a complete one.",
                  flush=True)
            raise
        return input_batch

    model_runner.prepare_inputs = wrapped_prepare_inputs
    print(f"[graph/install] prepare_inputs routing wrapper installed ({label})")


# ---------------------------------------------------------------------------
# The stale-STEPVIEW hazard: V2 drives warmup, cudagraph capture and memory
# profiling through the SAME `execute_model` entry point real steps use
# (`vllm/v1/worker/gpu/model_runner.py::execute_model`, kwargs `dummy_run`,
# `skip_attn_for_dummy_run`, `is_profile`, `context_len`). For those passes V2
# builds the batch with `InputBatch.make_dummy()` and — per that same
# `execute_model` — calls `self.prepare_inputs(...)` ONLY `if not dummy_run`; a
# dummy/profile pass never reaches it. `prepare_inputs` is the ONLY place that
# refreshes a registry's per-step routing (`mia.runner.step_view`, the routing
# wrapper above), so a dummy pass's registry state is whatever the PREVIOUS real
# step left there.
#
# Skipping our own post-forward drain on such a pass (below) is necessary but
# not sufficient: under FULL cudagraph, `execute_model` still runs the forward
# unconditionally (`cudagraph_manager.run_fullgraph(batch_desc)`, gated only on
# `batch_desc.cg_mode`, never on `dummy_run`) — a real graph REPLAY that
# re-executes the baked in-graph scatter op. That op reads
# `registry.capture_index_all` (device-resident; `c+1` = "column c is active
# this step", `0` = "discard", per `HostRegistry.__init__`) with no per-call
# Python gate of its own. Left alone, a dummy pass would replay the op against
# LAST REAL STEP's still-armed columns, scattering this pass's garbage hidden
# states into real — possibly not-yet-drained — aperture rows.
#
# `_stale_view_guard` blinds that table (zeroes it — the registry's own
# documented "discard everything" state) for exactly the duration of the wrapped
# call and restores the ORIGINAL values afterwards, even on exception, so the
# next REAL step's W1 idle-skip still sees the correct last-real-routing state.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _stale_view_guard(registry):
    """Make a V2 dummy/profile `execute_model` call structurally unable to be
    served (or to leave behind) a stale routing view. See the module comment
    above. A no-op when `registry` is None (subsystem never installed)."""
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
    """Run one V2 dummy/profile `execute_model` call with the stale routing view
    blinded for its duration. Shared by the HS and QK drain wrappers (identical
    hazard, identical `registry.capture_index_all` shape) — never drains, never
    touches any per-step registry field, forwards every kwarg untouched."""
    with _stale_view_guard(registry):
        return orig_execute_model(scheduler_output, *args, **kwargs)


# ---------------------------------------------------------------------------
# execute_model wrapper (never compiled — all per-request Python lives here)
# ---------------------------------------------------------------------------


def install_execute_model_wrapper(model_runner, worker) -> None:
    """Install the QK capture-APERTURE routing (``prepare_inputs`` wrapper) + the per-step drain
    (``execute_model`` wrapper). Idempotent.

    Routing runs after vLLM's input prep (so a NEW request's prefill routes correctly), reserving
    advancing shared-aperture slots and scattering q + k into the per-layer q/k apertures; the drain reads
    THIS step's newly-scattered aperture region post-forward and writes it durably to disk (per-layer q +
    k raw files + shared sidecar) — no RPC/bank/egress copy-out (storage-only). Two modes: OFF-LOOP
    consumer thread (default) or SYNCHRONOUS on-loop drain (``MIA_APERTURE_SYNC_DRAIN=1``, the
    fallback used for validation).
    """
    if getattr(model_runner, "_mia_qk_wrapped", False):
        return

    model_runner._mia_qk_wrapped = True

    # Routing — runs after prepare_inputs's input prep, so prefill routes correctly.
    # W1′: idle-skip key gates on output_qk (skips the per-step tax on non-capturing steps).
    def _qk_routing_key(step, registry):
        return _capture_idle_key(step, registry, "output_qk")

    install_prepare_inputs_routing(model_runner, worker, _build_routing, label="qk",
                                   routing_key_fn=_qk_routing_key)

    # Worker fallback mode + default phase live on the REGISTRY (not the model runner):
    # the routing builder now reads only (step, registry), never the runner, so per-worker
    # config that isn't part of the per-step snapshot travels via the registry instead.
    # Score is unsupported on the aperture path; kept here so the routing gate reads it (a
    # request asking for score is refused there).
    registry: Optional[HostRegistry] = get_registry(worker, "qk")
    if registry is not None:
        registry._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")
        registry._worker_score_mode = getattr(worker, "_score_mode_default", False)
        registry._worker_score_head = getattr(worker, "_score_head_default", 0)

    # --- Build the QK aperture drain (capture-aperture path, no bank/RPC): scatter -> aperture -> drain -> disk.
    #   * OFF-LOOP (default): a dedicated CONSUMER THREAD owns the drain; the execute_model wrapper
    #     does an O(1) enqueue (this step's entries + a CUDA event recorded AFTER the scatter) and the
    #     thread does the D2H + write off the engine loop. advance_drain frees aperture rows -> genuine
    #     reserve backpressure (never-drop).
    #   * SYNCHRONOUS (MIA_APERTURE_SYNC_DRAIN=1): the per-step on-loop drain (fallback path);
    #     the .cpu() D2H is stream-ordered after the in-graph capture_qk scatter.
    aperture = getattr(registry, "_qk_aperture", None) if registry is not None else None
    drain = None
    _sync_drain = os.environ.get("MIA_APERTURE_SYNC_DRAIN", "0") == "1"
    if registry is not None and aperture is not None:
        from mia.graph.aperture_drain_hs import _torch_dtype_name
        from mia.graph.aperture_drain_qk import (
            MultiLayerQKApertureDrain, OffLoopQKApertureDrain)
        # layers = [(layer_num, q_buf, k_buf), ...]; layer_num is 0-based (== eager match_attn).
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
        # TP geometry (every TP size, TP=1 included; purely additive keys): which global heads this
        # rank's q/k rows hold. aperture_reader.merge_qk_aperture_ranks keys the merge on these.
        if shard is not None:
            header.update(shard.as_header())
        header["num_layers"] = int(getattr(worker, "_qk_num_layers", len(layers)))
        # How big one step's per-file writes will be, from the capture config alone -- the size
        # half of MIA_APERTURE_WRITE_MODE=auto (aperture_sink.DIRECT_MIN_BYTES). None = not
        # predictable here, and auto then decides on alignment alone.
        _shape = predict_capture_write_shape(
            worker, "qk", str(header.get("hookq_mode") or "all_tokens"), int(aperture.n_slots))
        if _sync_drain:
            drain = MultiLayerQKApertureDrain(aperture, layers, run_dir, header, shape=_shape)
            registry._qk_consumer = None
            _mode = "sync per-step"
        else:
            # Per-request delivery (GATED, default OFF): the consumer demuxes each step's q + k rows by
            # req_id into a PerRequestIndex + a FINISH signal drives assemble_qk delivery, INSTEAD of
            # writing the shared per-layer files. Default OFF = the shared-file QK drain, unchanged.
            _per_request = os.environ.get("MIA_APERTURE_PER_REQUEST", "0") == "1"
            drain = OffLoopQKApertureDrain(aperture, layers, run_dir, header,
                                           per_request=_per_request, shape=_shape)
            drain.start()                        # spin up the consumer BEFORE the first enqueue
            registry._qk_consumer = drain         # reserve-block reads is_alive() for the dead backstop
            _pr = " + per-request delivery" if _per_request else ""
            _mode = f"OFF-LOOP consumer thread (drain_aperture={drain._aperture_depth}){_pr}"
        worker._qk_drain = drain
        worker._qk_run_dir = run_dir
        # ONE line per rank: how the q/k raw files are written (MIA_APERTURE_WRITE_MODE), per tensor
        # kind, with the writer-thread count and the detected O_DIRECT block size.
        print(f"[graph/install] QK aperture write path (tp_rank {int(tp_rank)}): "
              f"{drain.write_path_summary()}", flush=True)
        import atexit
        atexit.register(lambda d=drain: d.close())   # best-effort backstop; flush_aperture is the contract
        print(f"[graph/install] QK aperture drain ON -> {run_dir} "
              f"(R={aperture.n_slots} rows/layer, {len(layers)} layers, {_mode})", flush=True)
    else:
        print("[graph/install] no capture aperture; QK drain NOT wired", flush=True)

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run") or kwargs.get("is_profile"):
            # Warmup / cudagraph-capture / memory-profiling pass: no real requests,
            # registry routing not refreshed (prepare_inputs never ran for it). See
            # the stale-STEPVIEW hazard comment above _stale_view_guard.
            return _run_dummy_pass(orig_execute_model, get_registry(worker, "qk"),
                                   scheduler_output, args, kwargs)

        registry: Optional[HostRegistry] = get_registry(worker, "qk")
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        # Refresh mode/phase each step so a later worker mutation can't desync routing.
        registry._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")
        registry._worker_score_mode = getattr(worker, "_score_mode_default", False)
        registry._worker_score_head = getattr(worker, "_score_head_default", 0)

        # Forward: prepare_inputs (wrapped) builds+uploads routing (reserving aperture slots), then the
        # graph replays and capture_qk scatters q + k into the per-layer apertures.
        with PROF.timed("graph.forward"):
            result = orig_execute_model(scheduler_output, *args, **kwargs)

        # Post-forward: hand THIS step's newly-scattered aperture region to the drain. plans is non-empty
        # iff this step reserved aperture rows (active step); idle steps skip via the W1' routing key.
        plans = getattr(registry, "_pending_plans", None) or []
        drain = getattr(worker, "_qk_drain", None)

        # Component-1 capture evidence (VHP prof_harvest): the off-loop aperture path never runs the
        # eager register_forward_hook, so hook.fire.qk / captured.bytes.qk (emitted only there)
        # read 0 even though the aperture captured + persisted this step. captured.bytes.qk is the
        # q + k_full bytes the drain writes to NVMe this step, sampled once (never per-append).
        if aperture is not None and layers:
            _cap_rows = int(getattr(registry, "_qk_step_rows", 0) or 0)
            if _cap_rows > 0:
                _elt = layers[0][1].element_size()
                PROF.gauge("captured.bytes.qk",
                           float(_cap_rows) * len(layers) * (q_dim + k_dim) * _elt)
        if plans and drain is not None:
            # LayerEntry COLLAPSE: `_qk_step_entries` holds ONE QKReqCaptureRecord per request now; the
            # drain expands them into the flat QKStepEntry list off-loop (record_entries / _drain_item /
            # _demux_into_index).
            entries = getattr(registry, "_qk_step_entries", None) or []
            if _sync_drain:
                with PROF.timed("graph.drain"):
                    drain.record_entries(entries)
                    drain.drain_once()
            else:
                event = None
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record()               # current (forward) stream, after the scatter ops
                start_logical = getattr(registry, "_qk_step_start", None)
                n_rows = int(getattr(registry, "_qk_step_rows", 0) or 0)
                if n_rows > 0 and start_logical is not None:
                    drain.enqueue(entries, start_logical, n_rows, event)

        # Per-request delivery: enqueue a FINISH for each request finished since the previous step
        # (finished_req_ids lists requests dropped from input_batch BEFORE this step, so their last
        # rows were already enqueued -> FIFO holds). OUTSIDE the `if plans` gate so a finish is never
        # lost on an idle step. No-op unless the drain is per-request.
        if drain is not None and getattr(drain, "per_request", False):
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if finished:
                for _rid in finished:
                    drain.enqueue_finish(_rid)

        # hook.fire.qk: once per captured layer per FINISHED request -> harvest recovers the
        # capturing-request count (hook_fire_count / n_layers) as the per-request-KB denominator.
        _fin_evidence = getattr(scheduler_output, "finished_req_ids", None)
        if _fin_evidence:
            PROF.incr("hook.fire.qk", len(_fin_evidence) * len(layers))

        registry._pending_plans = []        # consume
        registry._qk_step_entries = []      # consume (the list is now owned by the queue item)
        registry._qk_step_start = None
        registry._qk_step_rows = 0

        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install] execute_model wrapper installed (QK aperture drain)")


# ---------------------------------------------------------------------------
# load_model monkey-patch — the single install entry point for register()
# ---------------------------------------------------------------------------

_LOAD_MODEL_PATCHED = False


def patch_worker_load_model() -> None:
    """Monkey-patch ``Worker.load_model`` to install the graph path after the model
    is built (before compile/capture). Idempotent. Runs the worker's own
    ``graph_install`` only when graph mode is armed AND the worker defines one —
    so the steer worker (no graph path) is untouched and graph-off is the original
    behaviour byte-for-byte.
    """
    global _LOAD_MODEL_PATCHED
    if _LOAD_MODEL_PATCHED:
        return

    from vllm.v1.worker.gpu_worker import Worker

    orig_load_model = Worker.load_model

    def patched_load_model(self, *args, **kwargs):
        result = orig_load_model(self, *args, **kwargs)

        # Gate 1: graph mode armed, else leave the eager path untouched.
        if not graph_mode_enabled():
            return result

        # Gate 2: dispatch by the worker's own graph_install (QK/HS define it; steer
        # doesn't, so it stays eager).
        graph_install = getattr(self, "graph_install", None)
        if not callable(graph_install):
            return result

        try:
            graph_install()
        except MiaRefusal:
            # A DELIBERATE refusal — MIA saying "this configuration cannot capture". The
            # blanket handler below exists for genuinely unexpected install failures, and a
            # documented refusal is not one of those: swallowing it boots the engine with
            # MIA loaded, nothing installed, generate() succeeding and NOTHING CAPTURED,
            # which is the exact failure the refusal was written to prevent.
            #
            # This was live, not hypothetical. The handler below predates this branch; this
            # branch then added require_v2_runner() into graph_install()'s call chain, so the
            # V2-only guarantee — which the plan requires to raise, never degrade — was
            # silently absorbed in graph mode. The audit that catches this class looks at
            # added RAISES under pre-existing handlers, not at added handlers.
            #
            # Raising is also the mildest option available: it happens during load_model,
            # before a single token is produced.
            raise
        except Exception as e:  # noqa: BLE001
            # Never let an UNEXPECTED install failure take down model loading — fall back to
            # no capture rather than crashing. Deliberate refusals took the branch above.
            print(f"[graph/install] graph install FAILED ({e}); continuing "
                  f"without graph capture.")
            PROF.incr("graph.install.errors")

        return result

    Worker.load_model = patched_load_model
    _LOAD_MODEL_PATCHED = True


# ---------------------------------------------------------------------------
# Public entry points (for _plugin.register())
# ---------------------------------------------------------------------------

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
