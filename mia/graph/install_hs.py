"""CUDA-graph hidden-state capture install — capture-aperture path.

The in-graph ``capture_hs`` scatter writes each captured token's residual straight into a
persistent, fixed-size GPU APERTURE (one parallel aperture per layer, sharing ONE logical cursor) at an
advancing slot; a per-step drain reads the newly-written contiguous region and writes it durably
to local disk. No per-step egress copy-out, no clone bank, no RPC — the path is
scatter -> aperture -> drain -> disk.
"""
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
from mia.graph.install import (  # shared helpers (no op-mode coupling)
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


# ApertureBackpressureError is defined in capture_aperture (imported above) so graph/install.py can
# re-raise it out of the shared routing wrapper without a circular import; re-exported here for the
# existing tests + the never-drop contract. See its docstring.

# Class-wrap bookkeeping, idempotent and reversible.
_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
_HS_HOST_ATTR = "_mia_hs_host"     # per-instance HSCaptureHost
_capture_dbg = {"n": 0, "cap": 0}


def _require_buffer_mode_hs() -> None:
    """Buffer mode is the only FULL-cudagraph HS capture path. The PIECEWISE hs_probe
    op mechanism was removed; ``MIA_HS_CAPTURE`` is retained only so an explicit
    ``op`` request fails loud instead of silently running buffer."""
    mode = os.environ.get("MIA_HS_CAPTURE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise MiaConfigurationError(
            f"MIA_HS_CAPTURE={mode!r} is no longer supported: the PIECEWISE hs_probe "
            "capture mode was removed. Buffer mode is the only FULL-cudagraph HS path; "
            "unset MIA_HS_CAPTURE or set it to 'buffer'.")

# Aperture sizing is resolved by graph/aperture_sizing.py::resolve_aperture_bytes_auto — a FIXED 4 GiB aperture by
# default (MIA_APERTURE_GPU_BYTES), with the legacy MIA_CAPTURE_GPU_RESERVE_FRAC ratio kept
# for back-compat. See that module for the precedence + fit-check gate.


# Selective-drain census. The counts belong to the drain OBJECT (one per worker process, created
# at install), not a module-level int, so reading them from live state needs no global that
# survives a re-install or leaks between tests. Observation channel: a plain getter here plus a
# `collective_rpc("get_drain_row_counts")` on the worker.
def get_drain_row_counts(worker) -> dict:
    """Rows the HS aperture drain ACTUALLY copied out of the aperture / did not copy, this worker process.

    `hs.drain.rows_copied` is accumulated at the copy site inside `_read_segments`, never derived
    from what the requests asked for — without it, a selective-drain validation leg cannot tell
    "copied only the wanted tiles" from "silently copied everything" (both reconstruct
    byte-identically). `hs.drain.rows_skipped` is what an unconditional full drain of the same
    steps would have copied, minus that; it reads 0 whenever the full drain runs instead (flag off,
    per-request delivery, synchronous drain) — `selective` / `selective_disabled_reason` say which.
    `hs.drain.degenerate_steps` counts steps where an ARMED selective drain found every installed
    layer wanted over the whole span and took the flag-off fast path — the only witness separating
    "fast path fired every step" from "never fired" on an all-layers workload (both report
    `rows_skipped == 0`). All zeros when no aperture drain is installed.
    """
    drain = getattr(worker, "_hs_drain", None)
    counts = getattr(drain, "row_counts", None)
    if drain is None or not callable(counts):
        return {"hs.drain.rows_copied": 0, "hs.drain.rows_skipped": 0,
                "hs.drain.degenerate_steps": 0,
                "selective": False, "selective_disabled_reason": None}
    return counts()



def _route_vectorized_enabled() -> bool:
    """Whether ``_build_routing_hs`` builds the routing plane in ONE shot (vectorized) rather than
    the legacy per-request torch scatter. Read ONCE at install and captured in the routing-wrapper
    closure, so it never costs a per-step env read. Byte-identical either way: only HOW the plane /
    entries / plans are built changes, never the values.

    ``MIA_ROUTE_VECTORIZED`` — DEFAULT OFF, kept opt-in: an A/B against the legacy loop found
    no collapse in the linear-in-N routing-build cost. MEASURED (Task E2, hermetic CPU bench —
    ``tests/perf/host_build_bench_results.md`` §3, width=64, num_layers=32): the dominant O(N)
    term is the WHOLE shared per-request Python loop, not any one piece of it. The scatter this
    flag amortizes is 93% eliminable in isolation (0.4644 -> 0.0325 ms), yet the real vectorized
    builder is only 38% cheaper than the real legacy one (0.4064 vs 0.6575 ms), because it still
    pays per request: the ``extra_args_for`` dict lookup, the
    ``output_hidden_states``/``hooks_on``/``hs_mode`` gating branches, a fresh
    ``list(range(num_layers))``, the numpy conversions, the ``plans`` dict literal and the
    ``ReqCaptureRecord`` append. ``ReqCaptureRecord`` construction alone is just 5.9% of the
    vectorized builder's total — an earlier version of this docstring named it "the dominant
    O(N) term", which the measurement refutes. Do NOT flip the default without a winning A/B.

    NO EFFECT UNDER THE PRODUCTION DEFAULT. ``_build_routing_hs`` checks the decode cache FIRST
    and returns from ``_build_routing_hs_decode_cache`` unconditionally, so while
    ``MIA_ROUTE_DECODE_CACHE`` is on (its default) the vectorized dispatch below it is never
    reached and this flag is dead. Measured in the same run (§4): 0.2914 ms with the cache alone
    vs 0.2861 ms with both — equal within noise. Setting both warns once at install; to actually
    exercise this path, set ``MIA_ROUTE_DECODE_CACHE=0`` as well."""
    return os.environ.get("MIA_ROUTE_VECTORIZED", "0") == "1"


def _route_decode_cache_enabled() -> bool:
    """Whether the HS routing build reuses a cached per-request gating across steady-decode steps
    (MIA_ROUTE_DECODE_CACHE, default ON; kill switch =0). Read ONCE at install and threaded
    into ``_build_routing_hs`` via the ``decode_cache`` kwarg (mirrors ``_route_vectorized_enabled``),
    so the per-step routing build never re-reads the env. Byte-identical: the cache only skips
    RE-DERIVING routing that is constant for a request's lifetime — every emitted value (plane,
    records, plans, aperture cursor) matches the slow build, so this is a pure host-cost reduction.
    """
    return os.environ.get("MIA_ROUTE_DECODE_CACHE", "1") != "0"



@dataclass
class _DecodeEntry:
    """Config-INVARIANT routing fields for one capturing request (constant for its lifetime). The
    COLUMN and the aperture SLOT are deliberately NOT stored — both are recomputed each step (column live
    from qsl, slot fresh from the reserve), because condensation moves the column and the aperture cursor
    advances every step."""
    __slots__ = ("layer_rows", "mode", "layers")
    layer_rows: object   # np.ndarray int64, 0-based registry rows (slow path's rows_layers order)
    mode: str            # "last_token" | "all_tokens"
    layers: list         # [L+1 for L in layer_rows] — the ReqCaptureRecord.layers template


def _wrap_layer_class(cls: type) -> None:
    """Class-wrap ``cls.forward`` to scatter the layer's residual stream into the
    static host buffer via the ``capture_hs`` op (NO splitting op → absorbed into the
    decode cudagraph).

    Idempotent per class. The scatter runs AFTER the original forward (we need its
    output). ``do_capture`` is an install-time constant, so this adds no
    data-dependent control flow to the traced region.
    """
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
    """Install the CUDA-graph HS capture path via decoder-layer wrap.

    Runs from the ``load_model`` patch, BEFORE compile/capture. Builds a per-layer
    ``HSCaptureHost`` (static ``hs_buf``) + a ``HostRegistry`` for routing and class-wraps
    the layer to emit ``capture_hs`` (NO splitting op — absorbed into the decode
    cudagraph under FULL). Returns the registry; the worker installs the execute_model
    wrapper.
    """
    _require_buffer_mode_hs()  # op mode removed; fail loud on an explicit request

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

    # The residual stream is REPLICATED across TP ranks (every row-parallel output is
    # all-reduced before it is added back), so any one rank's copy of a layer IS the layer. At
    # TP > 1 the layers are therefore SHARDED across the ranks (MIA_HS_TP_SHARD, default 1):
    # rank r captures the 0-based layers i with i % tp_size == r (graph/tp_shard.py has the rule
    # and why), so each rank's aperture, drain thread and disk writes cover ~L/tp layers instead
    # of rank 0 carrying all L. MIA_HS_TP_SHARD=0 restores the old layout (tp_rank 0 captures
    # every layer, the other ranks nothing) for A/B. Diagnostic MIA_HS_CAPTURE_ALL_RANKS=1
    # captures EVERY layer on EVERY rank, each into its own tp_rank_<r>/ (replicas of one
    # residual; used to prove bitwise replication across ranks).
    tp_rank, tp_size = resolve_tp_coords(worker)
    shard_mode = resolve_hs_shard_mode(tp_size)
    owned_rows = hs_rows_for_mode(shard_mode, num_layers, tp_size, tp_rank)
    # TP = 1 keeps its old rule verbatim (tp_rank 0 always captures).
    should_capture = (tp_rank == 0) if shard_mode == HS_MODE_SINGLE else bool(owned_rows)
    # The header flag stays the ENV's (it is what the diagnostic was set to), so a TP = 1 run with
    # it set writes the same header it always did; the LAYOUT is shard_mode's business.
    capture_all = os.environ.get(HS_ALL_RANKS_ENV) == "1"
    # Under TP the FULL decode cudagraph must be IDENTICAL across ranks. The capture_hs op reads
    # the decoder layer's POST-all_reduce residual and is baked INSIDE the compiled layer that
    # holds the TP collective, so baking it on some ranks (or some layers of some ranks) only
    # breaks the NCCL graph-capture lockstep -> engine-init hang at torch.cuda.synchronize() in
    # the cudagraph-capture __enter__. (The likely mechanism: the op is an extra consumer of the
    # all-reduce output, which changes whether vLLM's fuse_allreduce_rms pass fuses that
    # all-reduce with the next RMSNorm; fused and unfused all-reduces use different communication
    # kernels, so an asymmetric bake leaves ranks waiting in different collectives.) Fix: BAKE the
    # op on EVERY layer of EVERY rank with identical call sites and data dependencies. A layer
    # this rank does not own bakes it against a SINK -- a 1-row buffer that its all-zero
    # capture_index routes every token into; a rank that owns no layer (MIA_HS_TP_SHARD=0 ranks
    # >= 1, or fewer layers than ranks) has NO aperture, NO drain thread, NO writer process and NO
    # tp_rank_* dir (_install_hs_buffer / _install_hs_sink). Kill switch MIA_HS_TP_SYMMETRIC=0
    # (only owned layers bake -- expected to hang at TP > 1; kept to reproduce the failure).
    symmetric = tp_size > 1 and os.environ.get("MIA_HS_TP_SYMMETRIC", "1") != "0"
    bake_op = should_capture or symmetric

    # _conf must match what the eager worker writes (get_captured_states/flush_disk).
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

    # Rank-1c: writer PROCESS for off-GIL serialize+write. A non-capturing rank never writes an
    # artifact, so it never starts one (init_writer_process is idempotent on the attribute).
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

    # Static buffers + routing; the capture_hs scatter rides the decode cudagraph.
    registry = _install_hs_buffer(
        worker, model, matched, num_layers, hidden_size,
        should_capture, device_t, bake_op, owned_rows=owned_rows, symmetric=symmetric)
    set_registry(worker, "hs", registry)
    return registry


def _resolve_aperture_rows(worker, num_layers, hidden_size, buf_dtype, device,
                           rows_needed=None) -> tuple:
    """Size the shared GPU capture aperture, resolved LAZILY at install (after vLLM carves KV).

    ``aperture_bytes`` comes from ``resolve_aperture_bytes_auto``: an explicit
    ``MIA_APERTURE_GPU_BYTES`` always wins; the default is 4 GiB grown to one max-token step
    (``rows_needed`` = ``max_num_batched_tokens`` rows of ``num_layers x hidden x dtype``) when that is
    larger -- 10 GiB for an all-layer Llama-3.1-70B capture at 8192 tokens, where 4 GiB held only
    3276 rows and the first long prefill died in ``ApertureBackpressureError`` -- refusing at install
    when the grown default does not fit the free margin.
    ``R`` = per-layer aperture rows = ``aperture_bytes // (num_layers x hidden x dtype_size)`` — because the
    ``num_layers`` parallel per-layer apertures share the budget and one shared cursor. Fails loud if
    ``R < 1`` (spec §8: no silent degrade). Returns ``(R, aperture_bytes)``.

    ``num_layers`` is the number of layers THIS RANK captures: every layer at TP = 1 (and in the
    rank-0-only / all-ranks layouts), its round-robin share under the TP layer shard (20 of 80 for
    Llama-3.1-70B at TP4), so the default grows to one step of this rank's OWN layers (2.5 GiB
    there, not 10) and an explicit ``MIA_APERTURE_GPU_BYTES`` -- always a PER-RANK budget -- buys
    ``tp_size`` times the rows it would for all layers.
    """
    elem_size = torch.empty(0, dtype=buf_dtype).element_size()
    if str(device).startswith("cuda"):
        total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        total_gpu = 1 << 30  # CPU (tests): a nominal 1 GiB budget
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
    """Build the HS capture-aperture hosts + routing registry (no splitting op).

    Each OWNED decoder layer holds a persistent ``hs_buf`` ``(R+1, hidden)``: rows ``[0, R)``
    are the layer's aperture slots and row ``R`` is the shared SENTINEL (pad / no-capture discard).
    ALL owned layers' apertures advance in lockstep off ONE shared ``CaptureAperture`` logical
    cursor (``registry._hs_aperture``) — every captured layer scatters the SAME tokens each step,
    so one reserve serves every layer. The ``HostRegistry`` still owns the per-(layer, token)
    routing slabs (``capture_index`` width = ``cap`` tokens) the ``capture_hs`` scatter reads on
    replay; those now carry ADVANCING aperture slots, not batch positions. Buffers are built here —
    at load_model, before the cudagraph pool — so their data_ptrs stay fixed across replays.

    ``owned_rows`` (0-based; default every layer) are the layers THIS rank captures: all of them at
    TP = 1, its round-robin share under the TP layer shard (``graph/tp_shard.py``). The aperture is
    sized over the owned layers only (``_resolve_aperture_rows(num_layers=len(owned_rows))``), and
    only owned layers get an aperture buffer, a registry host, a raw file and a drain slot.

    ``bake_op`` (default = ``should_capture``) installs the wrap+host+op so the scatter is baked
    into the graph; ``symmetric`` (default = ``bool(bake_op)`` — whenever the op is baked at all,
    the layers this rank does NOT own bake it too) says every layer this rank does not own still
    bakes it. Under TP both are True on every rank (symmetric graph —
    see ``install_hs_hosts``). An unowned layer bakes the SAME op at the SAME call site against a
    SINK: a ``(1, hidden)`` buffer and a row of an all-zero ``(num_layers, cap)`` capture_index, so
    every token lands on the sink's only row (the fused kernel clamps into ``[0, rows-1]`` and the
    aten path's index is 0). A rank that owns NO layer (``_install_hs_sink``) allocates no
    aperture, resolves no aperture budget, builds no ``HostRegistry`` / pinned mirrors, and returns
    ``None`` -- so ``install_execute_model_wrapper_hs`` wires no drain and creates no ``tp_rank_*``
    dir.
    """
    if bake_op is None:
        bake_op = should_capture
    if symmetric is None:
        symmetric = bool(bake_op)      # default: every layer this rank does not own bakes a sink
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
        # MIA_HS_TP_SYMMETRIC=0 on a non-capturing rank: no op baked, nothing sized.
        for _name, module, _ln in matched:
            _wrap_layer_class(type(module))
        worker._capture_aperture = None
        return None

    # Shared capture-aperture geometry: R rows/layer, sentinel row == R (== CaptureAperture.SENTINEL),
    # sized over the layers THIS rank captures.
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
        # ONE shared logical cursor across the parallel per-layer apertures. row_bytes/dtype/row_shape
        # describe a layer's row; aperture.buf is NOT allocated (storage is the per-layer hs_bufs) —
        # we use only reserve/physical_slots/drained_segments/advance_drain/free_rows/SENTINEL.
        aperture = CaptureAperture(row_bytes=hidden_size * torch.empty(0, dtype=buf_dtype).element_size(),
                              n_slots=R, device=device, dtype=buf_dtype, row_shape=(hidden_size,))
        registry._hs_aperture = aperture
        registry._hs_step_entries = []
        # The routing builders keep only the rows this rank owns. None = every layer (TP = 1, the
        # rank-0-only and all-ranks layouts): the builders then run exactly as before.
        registry._hs_owned_rows = None if n_owned == num_layers else list(owned)
        registry._hs_owned_set = None if n_owned == num_layers else frozenset(owned)
        # Pad / no-capture lanes must route to the aperture SENTINEL row (R), NOT 0: aperture slot 0 is a
        # REAL storage row now, so a zero-filled lane would corrupt it. reset_pinned fills this.
        registry.sentinel_row = aperture.SENTINEL
        # The capture-aperture uses advancing positions, not column-diff, so the incremental / GPU
        # routers are disabled: the wrapper takes the legacy reset -> build -> upload branch.
        registry.incremental_enabled = False
        registry.gpu_routing = False
        # Prime the device slab + every pinned mirror to the sentinel so any pre-first-upload read
        # (e.g. the cudagraph capture pass, which skips routing) scatters to the discard row, never
        # a real slot 0.
        registry.capture_index_all.fill_(aperture.SENTINEL)
        for _slot in registry._aperture.slots:
            _slot["capture_index"].fill_(aperture.SENTINEL)
        worker._capture_aperture = aperture

    # Unowned layers of a capturing rank bake into sinks (symmetric TP graph); their all-zero
    # routing slab is separate from the registry's (whose unowned rows hold the aperture SENTINEL,
    # out of range for a 1-row buffer on the aten path).
    sink_index = sink_active = None
    if symmetric and n_owned < num_layers:
        sink_index = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        sink_active = torch.zeros(num_layers, dtype=torch.int32, device=device)
        worker._hs_sink_index = sink_index
        worker._hs_sink_active = sink_active

    n_hosts = n_sinks = 0
    for name, module, layer_num0 in matched:
        if bake_op and 0 <= layer_num0 < num_layers and layer_num0 in owned_set:
            # Pre-build the (R+1, hidden) aperture buffer and hand it to the host (host.cap stays the
            # token cap so register_host's cap check + the routing slab width still match).
            hs_buf = torch.zeros(R + 1, hidden_size, dtype=buf_dtype, device=device)
            host = HSCaptureHost(
                module_name=name,
                layer_num=layer_num0,            # 0-based registry-slab row
                egress_layer_num=layer_num0 + 1,  # 1-based artifact layer_num
                cap=cap,
                hidden=hidden_size,
                dtype=buf_dtype,
                device=device,
                has_residual=1,                   # refined per-call in the wrap
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
    """Give ``module`` a SINK host: the same ``HSCaptureHost`` (and so the same baked
    ``capture_hs`` call) a capturing layer gets, over a ``(1, hidden)`` buffer and row
    ``layer_num0`` of an all-zero routing slab nothing ever uploads to."""
    host = HSCaptureHost(
        module_name=name,
        layer_num=layer_num0,
        egress_layer_num=layer_num0 + 1,
        cap=cap,
        hidden=hidden_size,
        dtype=buf_dtype,
        device=device,
        has_residual=1,
        do_capture=True,                 # the op IS baked; it writes the sink row
        hs_buf=torch.zeros(1, hidden_size, dtype=buf_dtype, device=device),
    )
    host.bind_views(sink_index[layer_num0], sink_active[layer_num0])
    setattr(module, _HS_HOST_ATTR, host)


def _install_hs_sink(worker, matched, num_layers, hidden_size, cap, buf_dtype, device) -> None:
    """A rank that captures NO layer: its half of the TP-symmetric bake (see ``_install_hs_buffer``).

    Every matched layer gets the same ``HSCaptureHost`` + class wrap a capturing layer gets, so
    the compiled graph has the op at the same call sites with the same inputs -- only the
    buffers behind it are tiny: ``hs_buf`` is ``(1, hidden)`` (row 0 is the only, and the
    sentinel, row) and each host's ``capture_index`` is a row of one all-zero
    ``(num_layers, cap)`` int64 slab that nothing ever uploads to. Per-layer buffers (not one
    shared one) keep the mutation graph identical to a capturing rank's, where each layer mutates
    its own buffer. Total footprint: ``num_layers x (cap x 8 + hidden x dtype)`` bytes (~6.3 MiB
    for Llama-3.1-70B at cap 8192) instead of the aperture's gigabytes. Returns None: no registry,
    so no routing, no drain, no run dir. Reached by ranks >= 1 under ``MIA_HS_TP_SHARD=0`` and,
    under the layer shard, only by a rank with no layer (a model with fewer layers than ranks)."""
    sink_index = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
    sink_active = torch.zeros(num_layers, dtype=torch.int32, device=device)
    worker._hs_sink_index = sink_index          # keep the slab alive for the hosts' views
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


# ---------------------------------------------------------------------------
# Buffer-mode routing + egress + execute_model wrapper (HS analogue of the QK
# path in graph/install.py). All per-request Python lives in the wrapper, OUTSIDE
# the compiled region; the graph only READS the routing buffers we upload here.
# ---------------------------------------------------------------------------


def _aperture_reserve_or_block(aperture: CaptureAperture, n: int, consumer=None) -> int:
    """Reserve ``n`` contiguous aperture rows for this step's capture, BLOCKING (polling) on aperture-full
    rather than dropping (the never-drop contract). This runs on the ENGINE thread inside the
    ``prepare_inputs`` routing wrapper.

    With the OFF-LOOP consumer drain, the block is GENUINE: while this polls, ``time.sleep`` releases
    the GIL, the consumer thread drains earlier (already-forwarded) steps and calls ``advance_drain``,
    which frees rows and lets the reserve succeed. No deadlock — the consumer only drains PAST steps
    whose forwards already completed, independent of this blocked engine step. With the SYNCHRONOUS
    drain (``consumer is None``) the previous step fully drains, so ``free_rows == R`` and a reserve
    of ``n <= cap <= R`` succeeds immediately; the block then only engages on a mis-sized aperture.

    Fails loud (``ApertureBackpressureError``, which the routing wrapper RE-RAISES — never swallows) when
    the block cannot be relieved: a DEAD consumer (``is_alive()`` False → fail fast, don't wait the
    whole timeout) or a mis-sized aperture (a single step needs more than the whole aperture holds → after
    the bounded poll). ``MIA_APERTURE_BACKPRESSURE_TIMEOUT_S`` (default 10) / ``..._POLL_S`` (0.001)."""
    start = aperture.reserve(n)
    if start is not None:
        return start
    timeout = float(os.environ.get("MIA_APERTURE_BACKPRESSURE_TIMEOUT_S", "10") or "10")
    poll = float(os.environ.get("MIA_APERTURE_BACKPRESSURE_POLL_S", "0.001") or "0.001")
    PROF.incr("hs.aperture.backpressure")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Dead-consumer backstop: if the off-loop drain died, no one will ever free rows — fail
        # fast rather than block the whole timeout. (Surfaces the consumer's own error too.)
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
    """Capture-aperture routing: map each captured token's batch column to an ADVANCING shared-aperture
    slot (persists until the drain reads it), NOT the old batch-position row ``p+1``.

    All requested layers of a request scatter the SAME tokens to the SAME aperture slots (parallel
    per-layer apertures + ONE shared logical cursor), so ONE ``aperture.reserve(n)`` per request serves
    every layer. ``all_tokens`` reserves the whole span (``n = end - start``); ``last_token``
    reserves ``n = 1`` and routes ONLY the span's last column to that slot (the rest stay at the
    SENTINEL, so no aperture space is spent on tokens we won't keep).

    LayerEntry COLLAPSE: stashes ONE ``ReqCaptureRecord`` per capturing request on
    ``registry._hs_step_entries`` — carrying that request's OWN 1-based layer list — instead of
    fanning out ``num_layers`` ``LayerEntry`` objects on the engine loop (the O(reqs x layers)
    per-fire allocation a profile flagged as a co-dominant routing binder). The drain expands each
    record into the identical flat ``LayerEntry`` list OFF the loop, so the sidecar / demux are
    byte-for-byte unchanged. Returns a lightweight plan per active request (the wrapper's active/idle
    gate + W1' cache read it as truthy/empty).

    Gating (output_hidden_states filter, hooks_on prefill/decode/both) mirrors the eager hs_hook.
    Layer filter is 1-based (config layers) -> 0-based registry rows via ``ln-1``; the record's
    ``layers`` store the 1-based artifact numbers (``L+1``), matching the eager path's ``layer_num``.

    Reads only ``step`` and ``registry`` — never the model runner. Worker-wide fallbacks
    (``_default_hooks_on``, ``_worker_hs_mode``) live on ``registry`` (refreshed each step by
    ``install_execute_model_wrapper_hs``'s wrapper).
    """
    registry._hs_step_entries = []
    # The OFF-LOOP consumer needs this step's shared-aperture start slot + total reserved rows to read
    # exactly [start, start+rows) (the engine may reserve later steps ahead of the drain cursor, so
    # the whole pending region would over-read). Reset here; accumulated over the reserving requests.
    registry._hs_step_start = None
    registry._hs_step_rows = 0
    # TP symmetry: a non-capture rank bakes the op but never routes — leave capture_index at the
    # sentinel so the scatter is a pure no-op discard (residual is replicated; rank 0 holds the
    # real data). The registry's upload is should_capture-gated too; this is explicit.
    if not registry.should_capture:
        return []
    aperture: Optional[CaptureAperture] = getattr(registry, "_hs_aperture", None)
    if aperture is None:
        return []
    # The off-loop drain consumer (None on the synchronous path) — passed to the reserve-block so a
    # dead consumer fails fast/loud instead of blocking the whole timeout.
    consumer = getattr(registry, "_hs_consumer", None)
    req_ids = step.req_ids

    bs = step.num_reqs
    capture_index_pinned = registry.capture_index_pinned  # (num_layers, cap)
    cap = registry.cap
    default_hooks_on = getattr(registry, "_default_hooks_on", "prefill")
    default_hs_mode = getattr(registry, "_worker_hs_mode", "last_token")
    # TP layer shard: the 0-based rows this rank captures (None = every row).
    owned = getattr(registry, "_hs_owned_rows", None)
    owned_set = getattr(registry, "_hs_owned_set", None)

    # MIA_ROUTE_DECODE_CACHE: cached-gating decode fast-path wins even when
    # MIA_ROUTE_VECTORIZED is also set (checked first, before the vectorized dispatch).
    if decode_cache is None:
        decode_cache = _route_decode_cache_enabled()
    if decode_cache:
        return _build_routing_hs_decode_cache(step, registry)

    # MIA_ROUTE_VECTORIZED: build the whole step's plane / entries / plans in ONE shot,
    # killing the O(reqs) per-request torch.tensor + scatter the legacy loop below pays. Same
    # gating, same reserve order (shared cursor), same values -> byte-identical device slab,
    # _hs_step_entries, _hs_step_start/_rows, and plans (test_route_vectorized_parity).
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

        # output_hidden_states: True (all layers) | [1-based layer list].
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
        end = min(end, cap)  # over-cap tokens stay in the sentinel
        if end <= start:
            continue

        # Which 0-based registry rows this request wants -- of the rows THIS rank captures (the
        # TP layer shard; every row when `owned` is None, i.e. TP = 1: unchanged).
        if layer_filter is None:
            rows_layers = list(range(registry.num_layers)) if owned is None else list(owned)
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
            if owned_set is not None:
                rows_layers = [L for L in rows_layers if L in owned_set]
        if not rows_layers:
            continue

        # ---- aperture reserve (shared cursor) + advancing slots ----
        # last_token keeps only the span's last token, so it needs a single aperture row; all_tokens
        # keeps the whole span. Reserve ONCE (all layers share these slots) and BLOCK on full.
        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _aperture_reserve_or_block(aperture, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot            # step's rows begin at the first reserve
        registry._hs_step_rows += n                          # total rows this step (all reserves)
        phys = aperture.physical_slots(start_slot, n)           # n ints in [0, R)
        layer_idx_t = torch.tensor(rows_layers, dtype=torch.long)
        if req_mode == "last_token":
            # Only the span's last column -> the reserved slot; [start, end-1) stay SENTINEL.
            capture_index_pinned[layer_idx_t, end - 1] = int(phys[0])
        else:
            phys_t = torch.tensor(phys, dtype=torch.int64)
            capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        # LayerEntry COLLAPSE: ONE per-request record (this request's OWN 1-based layers in fan-out
        # order) instead of num_layers LayerEntry objects here; the drain expands it off-loop.
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
    """Vectorized twin of ``_build_routing_hs``'s legacy loop — byte-identical, cheaper build.

    The legacy loop pays, PER capturing request, a ``torch.tensor(rows_layers)`` (+ a
    ``torch.tensor(phys)`` for all_tokens) and a torch advanced-index scatter into the pinned
    plane — an O(reqs) torch-dispatch cost that dominates routing, linear in concurrent capturing
    requests. Here a cheap Python gating pass (dict/attr reads only) reserves aperture slots in batch
    order (IDENTICAL cursor / start_slot / ``_hs_step_start`` / ``_hs_step_rows`` to the legacy
    path — the never-drop reserve is unchanged)
    and accumulates flat ``(layer_row, col, slot)`` numpy triples; the plane is then written with
    ONE advanced-index assign. Requests occupy DISJOINT columns (qsl is cumulative) and a request's
    layer rows are distinct, so the flat triples carry NO duplicate ``(row, col)`` pairs — the
    single assign writes exactly the cells the per-request scatters would, to the same int64 values,
    leaving every other cell at the sentinel ``reset_pinned`` set. The per-request ``ReqCaptureRecord``
    order (request order) and each record's ``layers`` order (``[L+1 for L in rows_layers]``) — and
    ``plans`` — match the legacy path exactly, so the drain's off-loop expansion is byte-identical.
    """
    plans: list = []
    records: list = []
    # Flat plane triples across ALL requests -> ONE torch advanced-index assign at the end.
    rows_acc: list = []
    cols_acc: list = []
    slots_acc: list = []
    req_ids = step.req_ids
    bs = step.num_reqs
    qsl = step.query_start_loc_np
    default_hooks_on = getattr(registry, "_default_hooks_on", "prefill")
    default_hs_mode = getattr(registry, "_worker_hs_mode", "last_token")
    owned = getattr(registry, "_hs_owned_rows", None)       # TP layer shard (None = every row)
    owned_set = getattr(registry, "_hs_owned_set", None)
    for i in range(bs):
        req_id = req_ids[i]
        extra = step.extra_args_for(i)
        if not extra or extra.get("output_hidden_states") is None:
            continue

        # output_hidden_states: True (all layers) | [1-based layer list].
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
        end = min(end, cap)  # over-cap tokens stay in the sentinel
        if end <= start:
            continue

        # Which 0-based registry rows this request wants (SAME expression as the legacy path, so
        # rows_layers order — and thus the LayerEntry order — is identical, set-iteration and all),
        # restricted to the rows THIS rank captures (the TP layer shard; None = every row).
        if layer_filter is None:
            rows_layers = list(range(registry.num_layers)) if owned is None else list(owned)
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
            if owned_set is not None:
                rows_layers = [L for L in rows_layers if L in owned_set]
        if not rows_layers:
            continue

        # ---- aperture reserve (shared cursor) + advancing slots — IDENTICAL to the legacy path ----
        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _aperture_reserve_or_block(aperture, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot
        registry._hs_step_rows += n
        phys = aperture.physical_slots(start_slot, n)            # n ints in [0, R)

        # ---- collect flat plane triples (numpy, no per-request torch dispatch) ----
        rows_np = np.asarray(rows_layers, dtype=np.int64)
        nl = int(rows_np.shape[0])
        if req_mode == "last_token":
            # Legacy: capture_index_pinned[layer_idx_t, end-1] = int(phys[0]).
            rows_acc.append(rows_np)
            cols_acc.append(np.full(nl, end - 1, dtype=np.int64))
            slots_acc.append(np.full(nl, int(phys[0]), dtype=np.int64))
        else:
            # Legacy: capture_index_pinned[layer_idx_t[:,None], start:end] = phys_t[None,:], i.e.
            # cell (L, start+j) = phys[j] for every L in rows_layers, j in [0, n).
            phys_np = np.asarray(phys, dtype=np.int64)       # length n
            cols_span = np.arange(start, end, dtype=np.int64)  # length n
            rows_acc.append(np.repeat(rows_np, n))
            cols_acc.append(np.tile(cols_span, nl))
            slots_acc.append(np.tile(phys_np, nl))

        # LayerEntry COLLAPSE: ONE per-request record, request order then rows_layers order (matches
        # legacy); the drain expands it into the flat LayerEntry list off-loop.
        records.append(ReqCaptureRecord(
            req_id=str(req_id), logical_start=start_slot, n_rows=n, hs_mode=req_mode,
            layers=[L + 1 for L in rows_layers]))
        plans.append({
            "req_id": req_id,
            "n_rows": n,
            "layers": rows_layers,
            "hs_mode": req_mode,
        })

    # ONE advanced-index assign fills the whole step's plane (no cross-request cell collisions).
    if rows_acc:
        rows_t = torch.from_numpy(np.concatenate(rows_acc))
        cols_t = torch.from_numpy(np.concatenate(cols_acc))
        slots_t = torch.from_numpy(np.concatenate(slots_acc))
        capture_index_pinned[rows_t, cols_t] = slots_t

    registry._hs_step_entries = records
    return plans


def _build_routing_hs_decode_cache(step: StepView, registry: HostRegistry) -> list:
    """Cached-gating decode fast-path (MIA_ROUTE_DECODE_CACHE). Byte-identical to
    ``_build_routing_hs`` (vectorized=False): a STABLE-DECODE request (cache hit, this step contributes
    exactly one new token, not a prefill) skips the gating/layer-build and contributes its CACHED
    ``layer_rows`` + the LIVE column + a FRESH aperture slot to one batched plane write; a NEW/CHANGED
    request runs the full slow body and (re)populates the cache. Requests are walked in ``step.req_ids``
    order and the aperture is reserved in that order, so the shared cursor assigns the identical slots the
    slow path would — the byte-identity invariant.
    """
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
    owned = getattr(registry, "_hs_owned_rows", None)       # TP layer shard (None = every row)
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

        # ---- FAST PATH: cached config + one new token + decoding -> only the slot is new. ----
        if entry is not None and n_tokens == 1 and not is_prefill:
            n = 1
            start_slot = _aperture_reserve_or_block(aperture, n, consumer)
            if registry._hs_step_start is None:
                registry._hs_step_start = start_slot
            registry._hs_step_rows += n
            col = end - 1                       # n_tokens==1 -> start == end-1 (both modes)
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

        # ---- SLOW PATH: full gating build (identical to the legacy/vectorized body) + cache it. ----
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
        # Cache the config-invariant fields for the request's next (decode) steps — but ONLY for
        # requests that capture on decode. hooks_on="prefill" (the capture DEFAULT) captures on prefill
        # and must capture NOTHING on decode; caching it would make the fast-path fire on decode. Leave
        # it uncached so its decode steps fall through to this slow body, which correctly skips via the
        # `hooks_on=="prefill" and not is_prefill` gate above. hooks_on in {both, decode} always caches.
        if hooks_on != "prefill":
            cache[key] = _DecodeEntry(layer_rows=rows_np, mode=req_mode, layers=layers_tmpl)
        else:
            cache.pop(key, None)

    # Evict entries whose request left input_batch (finish/abort) — no cross-step leak.
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
    """Install the HS capture-aperture routing (``prepare_inputs`` wrapper) + the per-step drain
    (``execute_model`` wrapper). Idempotent.

    Routing runs after vLLM's input prep (so a NEW request's prefill routes correctly), reserving
    advancing aperture slots; the drain reads THIS step's newly-scattered aperture region post-forward and
    writes it durably to disk (per-layer raw files + shared sidecar) — no RPC/bank/egress copy-out.
    """
    if getattr(model_runner, "_mia_hs_wrapped", False):
        return

    model_runner._mia_hs_wrapped = True

    # Read MIA_ROUTE_VECTORIZED / MIA_ROUTE_DECODE_CACHE ONCE at install and pin them
    # in the closures below (mirrors ROUTE_NO_SKIP), so the per-step routing build never re-reads
    # the env.
    _route_vec = _route_vectorized_enabled()
    _route_dc = _route_decode_cache_enabled()

    # Routing — runs after prepare_inputs's input prep, so prefill routes correctly.
    # W1′: idle-skip key gates on output_hidden_states (skips the per-step tax on
    # non-capturing steps; rebuilds the moment any request captures).
    def _hs_routing_key(step, registry):
        return _capture_idle_key(step, registry, "output_hidden_states")

    if _route_dc:
        print("[graph/install_hs] HS routing decode-cache ENABLED "
              "(default ON; MIA_ROUTE_DECODE_CACHE=0 to disable)", flush=True)
    # A knob that cannot bite is worse than no knob: someone tuning MIA_ROUTE_VECTORIZED
    # under the default config would A/B two identical code paths and conclude the flag does
    # nothing useful, when in fact it was never reached. _build_routing_hs returns from the
    # decode-cache branch before the vectorized dispatch. Measured equal within noise
    # (0.2914 vs 0.2861 ms) — tests/perf/host_build_bench_results.md §4.
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

    # Worker fallback mode + default phase live on the REGISTRY (not the model runner):
    # the routing builder now reads only (step, registry), never the runner.
    registry: Optional[HostRegistry] = get_registry(worker, "hs")
    if registry is not None:
        registry._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

    # --- Build the multi-layer host drain (capture-aperture path, no bank/RPC): scatter -> aperture ->
    # drain -> disk. Two modes:
    #   * OFF-LOOP (default): a dedicated CONSUMER THREAD owns the drain; the execute_model wrapper
    #     does an O(1) enqueue (this step's entries + a CUDA event recorded AFTER the scatter) and the
    #     thread does the D2H + write off the engine loop, overlapping decode. advance_drain frees
    #     aperture rows -> genuine reserve backpressure (never-drop).
    #   * SYNCHRONOUS (MIA_APERTURE_SYNC_DRAIN=1): the per-step on-loop drain (fallback path). The
    #     .cpu() D2H is stream-ordered after the in-graph capture_hs scatter.
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
        # TP provenance (purely additive keys): which rank wrote this dir. capture_all_ranks marks
        # the diagnostic where every rank writes a (replica) dir of every layer.
        _tp_size = int(resolve_tp_coords(worker)[1])
        _num_layers = int(worker._conf.get("num_layers", len(layers)))
        header.update({
            "tp_rank": int(tp_rank),
            "tp_size": _tp_size,
            "num_layers": _num_layers,
            "capture_all_ranks": bool(getattr(worker, "_hs_capture_all_ranks", False)),
        })
        # The TP layer shard: this dir holds ONLY this rank's round-robin share of the layers, so
        # the header says which rule and which layers (1-based, the hs_layer_<L>.raw numbers).
        # Readers recompute the rule and refuse a header that disagrees; they union every rank's
        # dir (aperture_reader.merge_hs_aperture_ranks). Absent at TP = 1 and in the rank-0-only /
        # all-ranks layouts (each dir there holds every layer), so those headers are unchanged.
        if getattr(worker, "_hs_shard_mode", None) == HS_MODE_ROUND_ROBIN:
            header.update(HSShard.of(int(tp_rank), _tp_size, _num_layers).as_header())
        # How big one step's per-layer writes will be, from the capture config alone -- the size
        # half of MIA_APERTURE_WRITE_MODE=auto (aperture_sink.DIRECT_MIN_BYTES). None = not
        # predictable here, and auto then decides on alignment alone.
        from mia.graph.install import predict_capture_write_shape
        _shape = predict_capture_write_shape(
            worker, "hs", str(getattr(worker, "hs_mode", "last_token") or "last_token"),
            int(aperture.n_slots))
        if _sync_drain:
            drain = MultiLayerApertureDrain(aperture, layers, run_dir, header, shape=_shape)
            registry._hs_consumer = None
            _mode = "sync per-step"
        else:
            # Per-request delivery (GATED, default OFF): the consumer demuxes each step's rows by
            # req_id into a PerRequestIndex + a FINISH signal drives assembly, INSTEAD of writing the
            # shared per-layer files. Default OFF = the shared-file drain, unchanged.
            _per_request = os.environ.get("MIA_APERTURE_PER_REQUEST", "0") == "1"
            drain = OffLoopApertureDrain(aperture, layers, run_dir, header,
                                         per_request=_per_request, shape=_shape)
            # Per-request DISK route under the TP layer shard: a request whose layers this rank does
            # not own is routed here too (the route RPC is collective) but stages nothing; the
            # drain notes such a finish so confirm_aperture_delivery answers "not mine" (None)
            # instead of "not landed" (False) forever.
            drain._note_unstaged_finish = (
                getattr(worker, "_hs_shard_mode", None) == HS_MODE_ROUND_ROBIN)
            drain.start()                       # spin up the consumer thread BEFORE the first enqueue
            registry._hs_consumer = drain        # reserve-block reads is_alive() for the dead backstop
            _pr = " + per-request delivery" if _per_request else ""
            _mode = f"OFF-LOOP consumer thread (drain_aperture={drain._aperture_depth}){_pr}"
        # Say what the drain DECIDED, not what was asked for: it resolves selectivity at
        # construction (graph/aperture_drain_hs._resolve_selective), and two configurations legitimately
        # refuse an armed flag and FULL-drain instead (per-request delivery, the synchronous drain).
        if getattr(drain, "selective", False):
            _mode += " + SELECTIVE drain (default ON; MIA_DRAIN_SELECTIVE=0 to disable)"
        elif getattr(drain, "selective_disabled_reason", None):
            _mode += " + selective drain has no effect here"
            # The flag defaults ON, so most drains that hit this branch were never "armed" by
            # anyone -- they simply run a config (per-request delivery, the sync drain) selective
            # drain does not reach, and there is nothing for an operator to act on. Only escalate to
            # a warning when the env was set EXPLICITLY; the default-driven case gets an info line.
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
        # ONE line per capturing rank: how the raw files are written (MIA_APERTURE_WRITE_MODE),
        # per tensor kind, with the writer-thread count and the detected O_DIRECT block size.
        _wline = (f"[graph/install_hs] HS aperture write path (tp_rank {int(tp_rank)}): "
                  f"{drain.write_path_summary()}")
        print(_wline, flush=True)
        logger.info(_wline)
        # atexit is a best-effort backstop only (the worker process is often killed, not joined —
        # so the parity oracle / caller MUST call flush_aperture() to persist the sidecar).
        import atexit
        atexit.register(lambda d=drain: d.close())
        print(f"[graph/install_hs] HS aperture drain ON -> {run_dir} "
              f"(R={aperture.n_slots} rows/layer, {len(layers)} layers, {_mode})", flush=True)
    else:
        print("[graph/install_hs] no capture aperture; HS drain NOT wired", flush=True)

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run") or kwargs.get("is_profile"):
            # Warmup / cudagraph-capture / memory-profiling pass: no real requests,
            # registry routing not refreshed (prepare_inputs never ran for it). See
            # the stale-STEPVIEW hazard comment in mia/graph/install.py, above
            # _stale_view_guard.
            return _run_dummy_pass(orig_execute_model, get_registry(worker, "hs"),
                                   scheduler_output, args, kwargs)

        registry: Optional[HostRegistry] = get_registry(worker, "hs")
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        registry._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
        registry._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

        # Forward: prepare_inputs (wrapped) builds+uploads routing (reserving aperture slots), then
        # the graph replays and capture_hs scatters into the per-layer apertures. Timed
        # (measurement-only): graph.forward includes routing (graph.route, nested) + the replay.
        with PROF.timed("graph.forward"):
            result = orig_execute_model(scheduler_output, *args, **kwargs)

        # Post-forward: hand THIS step's newly-scattered aperture region to the drain. plans is non-empty
        # iff this step reserved aperture rows (active step); idle steps skip via the W1' routing key.
        plans = getattr(registry, "_pending_plans", None) or []
        drain = getattr(worker, "_hs_drain", None)
        # LayerEntry COLLAPSE: `_hs_step_entries` holds ONE ReqCaptureRecord per request now; the
        # drain expands them into the flat LayerEntry list off-loop (record_entries / _drain_item).
        # Fetched here (ahead of the `if plans` block below), not only inside it, because the
        # captured-bytes gauge right below also needs it under selective drain.
        records = getattr(registry, "_hs_step_entries", None) or []

        # Component-1 capture evidence (VHP prof_harvest): the off-loop aperture path never runs the
        # eager register_forward_hook, so hook.fire.hs / captured.bytes.hs (emitted only there)
        # read 0 even though the aperture captured + persisted this step; report it here instead.
        # captured.bytes.hs is the bytes THIS drain actually writes to NVMe this step, which is NOT
        # `_cap_rows * len(layers)` (every installed layer) once selective drain is active: a subset
        # request only costs its OWN `n_rows * len(rec.layers)`, and `record_captured_cells` sums
        # exactly that in O(records), not the O(records x layers) `build_copy_plans` pays. The
        # degenerate all-layers case collapses to the old formula (every record tiles the whole
        # span), and when selective drain is not active the drain copies every layer regardless of
        # what any record names, so the old formula is correct there too. Sampled once per step.
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
                # SYNCHRONOUS: read + write on the engine loop (the fallback path). The .cpu() D2H is
                # stream-ordered after the in-graph scatter (same/default stream), so no event needed.
                with PROF.timed("graph.drain"):
                    drain.record_entries(records)
                    drain.drain_once()
            else:
                # OFF-LOOP: record a CUDA event on the forward stream AFTER the scatter, then O(1)
                # enqueue (records + this step's start slot / row count + event). The consumer thread
                # waits the event, D2Hs, expands + writes, and advance_drains — all off the engine loop.
                event = None
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record()             # current (forward) stream, after the scatter ops
                start_logical = getattr(registry, "_hs_step_start", None)
                n_rows = int(getattr(registry, "_hs_step_rows", 0) or 0)
                if n_rows > 0 and start_logical is not None:
                    drain.enqueue(records, start_logical, n_rows, event)

        # Per-request delivery: enqueue a FINISH for each request finished since the previous step.
        # In vLLM v1 `scheduler_output.finished_req_ids` lists requests finished BETWEEN the prior and
        # current step (`_update_states` drops them from input_batch BEFORE this step's forward), so a
        # request here finished at the PREVIOUS step — its last rows were already enqueued then, so
        # the FIFO invariant (rows before finish) holds. Runs OUTSIDE the `if plans` gate so a finish
        # is never lost on an idle (non-capturing) step. No-op unless the drain is per-request.
        if drain is not None and getattr(drain, "per_request", False):
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if finished:
                for _rid in finished:
                    drain.enqueue_finish(_rid)

        # hook.fire.hs: once per captured layer per FINISHED request, so the harvest's
        # hook_fire_count / n_layers recovers the capturing-request count (its per-request-KB
        # denominator, warm-up-inclusive to match the cumulative captured.bytes numerator). Runs
        # every step, independent of per_request mode (a finish can land on an idle step).
        _fin_evidence = getattr(scheduler_output, "finished_req_ids", None)
        if _fin_evidence:
            PROF.incr("hook.fire.hs", len(_fin_evidence) * len(layers))

        registry._pending_plans = []       # consume
        registry._hs_step_entries = []     # consume (the list is now owned by the queue item)
        registry._hs_step_start = None
        registry._hs_step_rows = 0

        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install_hs] execute_model wrapper installed (HS aperture drain)")


__all__ = ["install_hs_hosts", "install_execute_model_wrapper_hs"]
