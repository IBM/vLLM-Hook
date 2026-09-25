"""CUDA-graph activation-steering install (buffer mode).

Steering is the MUTATING counterpart of the read-only QK/HS capture paths: it
adds a direction to the residual stream at one decoder layer so the change
propagates through every downstream layer and shifts the model's output. It is
expressed as the ``mia::steer_buffer`` op — a masked, in-place
``residual += coeff * vec`` that is NOT a splitting op, so it is absorbed into the
decode cudagraph and replays every step (no Python on replay). The in-place
mutation under the ``mutates_args=["residual"]`` contract threads the steered
residual into the next graph segment. Works across decode under FULL cudagraph.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from vllm.forward_context import get_forward_context

from mia._profiler import PROF
from mia.graph import register_graph_ops
from mia.graph.hosts import SteerHost
from mia.graph.install import (  # shared helpers
    _IDLE_ROUTE_KEY,
    _resolve_max_num_batched_tokens,
    install_prepare_inputs_routing,
)
from mia.graph.registry import PinnedMirror, set_registry
from mia.runner import StepView
from mia.workers._common import iter_matched_modules
from mia.workers.hs_capture_worker import match_layer
from mia.workers.steer_worker import (
    _load_steering_vector,
    _resolve_steer_config,
    _parse_steer_layers,
    STEER_COL_ALL,
    STEER_COL_NONE,
    is_default_steer_modes,
    resolve_steer_modes,
    steer_col_for,
    steer_span,
)

# Class-wrap bookkeeping, idempotent and reversible.
_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
_STEER_HOST_ATTR = "_mia_steer_host"   # per-instance SteerHost

# Set at install so the routing helpers can reach the worker (vector cache, env config).
_ACTIVE_WORKER_STEER = None


def _require_buffer_mode_steer() -> None:
    """Buffer mode is the only FULL-cudagraph steering path. The PIECEWISE steer_residual
    op mechanism was removed; ``MIA_STEER_MODE`` is retained only so an explicit
    ``op`` request fails loud instead of silently running buffer."""
    mode = os.environ.get("MIA_STEER_MODE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise RuntimeError(
            f"MIA_STEER_MODE={mode!r} is no longer supported: the PIECEWISE steer_residual "
            "mode was removed. Buffer mode is the only FULL-cudagraph steering path; "
            "unset MIA_STEER_MODE or set it to 'buffer'.")
# Opt-in idle-skip for the steer routing wrapper (Phase 4). When ON, the routing
# wrapper's W1 invalidation key becomes _steer_idle_key: a constant _IDLE_ROUTE_KEY
# on any step where NO request would steer, else the real registry.routing_key
# signature. This lets fully-non-steering churn steps skip reset/build/upload even
# though req_ids/qsl change every step (which the default registry.routing_key
# cannot skip, because its key changes with the batch). Default OFF →
# routing_key_fn stays None → behaviour byte-identical to today.
_STEER_ROUTE_FASTPATH = os.environ.get("MIA_STEER_ROUTE_FASTPATH", "0") == "1"


def _wrap_layer_class(cls: type) -> None:
    """Class-wrap ``cls.forward`` to steer the layer's residual output.

    Idempotent per class. Runs AFTER the original forward (we steer its output).
    The op mutates the residual IN PLACE, so the output object is returned
    unchanged — the next layer reads the same (now-steered) residual tensor. For
    the fused-residual pattern the layer returns ``(hidden, residual)`` and we pass
    index 1 (the residual) as the mutated arg, matching the eager hook which
    modifies ``output[1]`` and leaves ``hidden`` untouched. For a single-tensor
    output the whole tensor is the residual stream.
    """
    if cls in _WRAPPED_LAYER_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_LAYER_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            out = orig_fwd(self, *args, **kwargs)
            # Masked in-place steer_buffer add (NO splitting op → absorbed into the
            # decode cudagraph). do_steer is an install-time constant, so this adds
            # no data-dependent control flow.
            host = getattr(self, _STEER_HOST_ATTR, None)
            if host is not None and host.do_steer:
                if (isinstance(out, tuple) and len(out) >= 2
                        and isinstance(out[1], torch.Tensor)):
                    host.steer(out[1])      # mutate the residual in place
                elif isinstance(out, torch.Tensor):
                    host.steer(out)
            return out
        return wrapped

    cls.forward = make_wrapped(orig_forward)


def install_steer_hosts(worker) -> None:
    """Install the buffer-mode steering path via decoder-layer wrap.

    Runs from the ``load_model`` patch, BEFORE compile/capture. Builds a per-layer
    ``SteerHost`` + a ``SteerRegistry`` for routing and class-wraps the layer
    ``forward`` to emit the ``steer_buffer`` op (masked in-place residual add, NO
    splitting op — absorbed into the decode cudagraph under FULL).

    No ``should_capture`` rank gating: steering MUST be applied identically on
    every TP rank (the residual is replicated across ranks; steering only rank 0
    would desync the ranks). HS capture reads that same replicated residual on
    rank 0 only; QK capture runs on every rank, each for its own heads
    (``graph/tp_shard.py``). The vector table is full-width (``hidden``) on every
    rank because the residual is never sharded.
    """
    global _ACTIVE_WORKER_STEER

    _require_buffer_mode_steer()  # op mode removed; fail loud on an explicit request
    from mia.graph.tp_shard import refuse_pipeline_parallel
    refuse_pipeline_parallel(
        getattr(getattr(worker, "parallel_config", None), "pipeline_parallel_size", 1),
        "steer graph install")

    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install_steer] no model on model_runner; skip steer install")
        return

    register_graph_ops()
    _ACTIVE_WORKER_STEER = worker

    # State the steering body reads (the eager worker builds these in
    # _install_hooks, which graph mode skips). Keep per-worker so a config-less
    # request resolves nothing rather than crashing.
    if not getattr(worker, "_vector_cache", None):
        worker._vector_cache = {}
    if not hasattr(worker, "_env_config_path"):
        worker._env_config_path = os.environ.get("MIA_STEER_CONFIG")

    matched = list(iter_matched_modules(model, match_layer))
    if not matched:
        print("[graph/install_steer] no decoder layers matched LAYER_PATTERNS; "
              "steering inactive")
        return

    device_t = next(model.parameters()).device

    # steer_buffer masked in-place add, absorbed into the decode cudagraph.
    _install_steer_buffer(worker, model, matched, device_t)


# ---------------------------------------------------------------------------
# Buffer mode (FULL decode): SteerRegistry + masked steer_buffer add
# ---------------------------------------------------------------------------


class SteerRegistry:
    """Per-worker owner of the steering routing slabs + the resident vector table.

    Mirrors ``HostRegistry`` (so it plugs into ``install_prepare_inputs_routing``),
    but its slabs are the steering data plane: per-(layer, token) ``coeff_all`` float,
    ``vec_id_all`` int (which vector), and ``mode_all`` int (0=add_vector, 1=adjust_rs).
    ``vec_table (V_max, hidden)`` is the LoRA-style vector library and ``avg_proj_table
    (V_max,)`` holds each adjust_rs vector's target projection; both are pre-allocated so
    their ``data_ptr`` is fixed across replays (rows written in place). All ranks steer,
    so ``should_capture`` is always True.
    """

    def __init__(self, num_layers, cap, hidden, v_max, device, dtype):
        self.num_layers = int(num_layers)
        self.cap = int(cap)
        self.hidden = int(hidden)
        self.v_max = int(v_max)
        self.device = torch.device(device)
        self.should_capture = True  # steering is applied on EVERY TP rank

        self.coeff_all = torch.zeros(num_layers, cap, dtype=torch.float32, device=device)
        self.vec_id_all = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        self.mode_all = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        # The resident vector library + per-vector adjust_rs target (fixed data_ptr).
        self.vec_table = torch.zeros(v_max, hidden, dtype=dtype, device=device)
        self.avg_proj_table = torch.zeros(v_max, dtype=torch.float32, device=device)

        # Aperture-buffered pinned mirrors (W2/W6a): coeff/vec_id/mode_pinned are
        # properties onto the current slot, so reset_pinned never waits on the
        # in-flight H2D copy. See PinnedMirror.
        pin = self.device.type == "cuda"
        self._aperture = PinnedMirror(
            [
                ("coeff", (num_layers, cap), torch.float32),
                ("vec_id", (num_layers, cap), torch.int64),
                ("mode", (num_layers, cap), torch.int64),
            ],
            pin=pin,
        )

        self.hosts: Dict[int, SteerHost] = {}
        self.vec_paths: Dict[str, int] = {}  # vector_path -> row in vec_table
        self.vec_has_avgproj: set = set()    # vids whose avg_proj_table row is loaded
        self._next_vec_id = 0
        self._pending_plans: list = []

        # Incremental/persistent routing (dynamic, arbitrary-layer safe). The legacy
        # reset+build+upload rewrote [num_layers, width]x3 EVERY non-skipped step even
        # though only the currently-active layer-rows are non-zero; under open-loop churn
        # the W1 whole-step skip (routing_key) fails every step, so that full O(num_layers
        # x width) cost fired on the critical path each step and drove the concurrency
        # slope. Incremental keeps a per-column shadow of the interned (layer, coeff, vid,
        # mode) state and rewrites/uploads ONLY the columns that changed since last step:
        # steady decode -> 0 upload (pure replay), churn/finish -> O(changed). The active
        # layer-rows are discovered from the requests actually in flight, so ANY per-request
        # layer works with no install-time layer knowledge (unlike wiring only target
        # layers, which the FULL graph freezes at capture). Kill switch
        # MIA_INCREMENTAL_ROUTING=0 -> legacy full reset+upload.
        self.incremental_enabled = (
            os.environ.get("MIA_INCREMENTAL_ROUTING", "1") != "0"
            and self.device.type == "cuda"
            and self.should_capture)
        self._col_state: Optional[np.ndarray] = None     # (num_layers, cap) int64 shadow; 0=inactive
        self._inc_coeff: Optional[torch.Tensor] = None   # (num_layers, cap) f32 pinned
        self._inc_vecid: Optional[torch.Tensor] = None   # (num_layers, cap) i64 pinned
        self._inc_mode: Optional[torch.Tensor] = None    # (num_layers, cap) i64 pinned
        self._inc_event: Optional[torch.cuda.Event] = None
        self._inc_force_full = False   # next apply rewrites [0,width) in full (post-error)
        # Intern (layer, coeff, vid, mode) -> id (>=1; 0 reserved for "inactive"). A steer
        # deployment fixes the vector/method/coefficient, so this stays tiny (adjust_rs is
        # always coeff=0; add_vector a fixed per-request coefficient).
        self._st_intern: Dict[tuple, int] = {}
        self._st_val: Dict[int, tuple] = {}
        self._st_next = 1
        self._pending_assignments: list = []

        # GPU-side routing (MIA_STEER_GPU_ROUTING, default ON): a per-slot config table —
        # each request's (vid, mode, coeff, layer_mask) resolved ONCE, indexed by batch POSITION,
        # refreshed host-side O(reqs) only on a req_ids change — that a GPU scatter expands into
        # the coeff/vec_id/mode slabs. This removes the O(num_layers x cap) host build
        # (apply_incremental_routing's numpy new_state + diff) that scales with N x B. The GPU
        # scatter is byte-identical to the host path and wins under churn; it scopes to the STEER
        # worker only (SteerRegistry is built solely by the steer install), so it triggers ONLY on
        # steering workloads. Capture's separate MIA_CAPTURE_GPU_ROUTING stays OFF — capture
        # routing is already O(cap) position-deterministic, so the scatter has nothing to win there.
        # `gpu_routing` (the wrapper gate) additionally requires cuda; arrays are built whenever the
        # env resolves truthy so CPU unit tests can drive refresh_slot_config.
        # phase x positions gate state. `_any_gated` LATCHES the first time a request
        # with non-default modes is resolved and is never cleared: it only ever adds
        # precision to routing_key, so a stale True costs a missed whole-step skip,
        # never a wrong one. Default deployments never latch and pay nothing.
        self._any_gated = False
        self._step_cols: Optional[list] = None   # this step's per-slot columns

        self._gpu_routing_env = os.environ.get("MIA_STEER_GPU_ROUTING", "1") != "0"
        self.gpu_routing = self._gpu_routing_env and self.device.type == "cuda"
        self.slot_vid = self.slot_mode = self.slot_coeff = self.slot_layer_mask = None
        self._slot_req_key: Optional[tuple] = None   # last tuple(req_ids) the table was built for
        if self._gpu_routing_env:
            pin = self.device.type == "cuda"
            self.slot_vid = torch.zeros(cap, dtype=torch.int64, device=device)
            self.slot_mode = torch.zeros(cap, dtype=torch.int64, device=device)
            self.slot_coeff = torch.zeros(cap, dtype=torch.float32, device=device)
            self.slot_layer_mask = torch.zeros(cap, num_layers, dtype=torch.bool, device=device)
            self._slot_vid_h = torch.zeros(cap, dtype=torch.int64, pin_memory=pin)
            self._slot_mode_h = torch.zeros(cap, dtype=torch.int64, pin_memory=pin)
            self._slot_coeff_h = torch.zeros(cap, dtype=torch.float32, pin_memory=pin)
            self._slot_mask_h = torch.zeros(cap, num_layers, dtype=torch.bool, pin_memory=pin)
            # ZERO-COPY numpy views of the host mirrors. Every slot write goes through
            # these: a fancy-index assignment is ONE operation, where the equivalent
            # tensor write costs one ATen dispatch PER ELEMENT (~3 us) — at
            # optimal_layer="all" that is num_layers dispatches per request, i.e. the
            # host cost scaled with reqs x layers. The device copies below read these
            # exact buffers, so nothing about the upload changes.
            self._slot_vid_np = self._slot_vid_h.numpy()
            self._slot_mode_np = self._slot_mode_h.numpy()
            self._slot_coeff_np = self._slot_coeff_h.numpy()
            self._slot_mask_np = self._slot_mask_h.numpy()
            # phase x positions gate plane (allocated always; only uploaded when gated).
            self.slot_col = torch.full((cap,), STEER_COL_ALL, dtype=torch.int64, device=device)
            self._slot_col_h = torch.full((cap,), STEER_COL_ALL, dtype=torch.int64,
                                          pin_memory=pin)

    def register_host(self, host: SteerHost) -> None:
        self.hosts[host.layer_num] = host

    def assign_views(self) -> None:
        for layer_num, host in self.hosts.items():
            host.bind_views(self.coeff_all[layer_num], self.vec_id_all[layer_num],
                            self.mode_all[layer_num], self.vec_table,
                            self.avg_proj_table)

    def begin_step(self) -> None:
        pass  # steering keeps no forward-context stash

    @property
    def coeff_pinned(self) -> torch.Tensor:
        return self._aperture.cur("coeff")

    @property
    def vec_id_pinned(self) -> torch.Tensor:
        return self._aperture.cur("vec_id")

    @property
    def mode_pinned(self) -> torch.Tensor:
        return self._aperture.cur("mode")

    def _resolve_step_cols(self, step: StepView) -> list:
        """Per-slot steer column for THIS step — the graph side's single evaluation of
        the phase x positions gate.

        Returns one entry per batch slot: ``STEER_COL_ALL`` (whole span, the default),
        ``STEER_COL_NONE`` (nothing this step), or an absolute flat column. Latches
        ``_any_gated`` when any request carries non-default modes.

        Reads only ``step`` — never the model runner. Defensive throughout: an
        unresolvable config resolves that slot to ``STEER_COL_NONE`` (never steer
        wrongly) rather than raising into the forward.
        """
        req_ids = step.req_ids
        if not req_ids:
            return []
        worker = _ACTIVE_WORKER_STEER
        env_path = getattr(worker, "_env_config_path", None) if worker else None
        qsl = step.query_start_loc_np
        bs = step.num_reqs
        cols = []
        for i in range(bs):
            extra = step.extra_args_for(i)
            steer_arg = (extra or {}).get("steer")
            cfg = _resolve_steer_config(steer_arg, env_path)
            if cfg is None:
                cols.append(STEER_COL_NONE)
                continue
            try:
                phase, positions = resolve_steer_modes(cfg)
            except ValueError:
                cols.append(STEER_COL_NONE)
                continue
            if not is_default_steer_modes(phase, positions):
                self._any_gated = True
            start = int(qsl[i])
            end = min(int(qsl[i + 1]), self.cap)
            is_prefill = bool(step.is_prefilling_np[i])
            is_final = True
            if is_prefill and positions == "last_token":
                is_final = (int(step.num_computed_tokens_np[i]) + (end - start)) \
                    >= int(step.prompt_len_np[i])
            cols.append(steer_col_for(phase, positions, is_prefill, is_final, start, end))
        return cols

    def routing_key(self, step: StepView) -> Optional[tuple]:
        """Invalidation signature for the routing wrapper (W1).

        Steering routes by FLAT position (``query_start_loc_np[i]``), and a request's
        resolved steer config is immutable for its lifetime (fixed at admission), so on a
        stable batch the coeff/vec_id/mode mirrors — and the device slabs already
        holding them — are bit-identical every decode step. The signature is thus
        ``(req_ids, qsl)``: when it (and the upload width) are unchanged, the wrapper
        SKIPS reset/build/upload and the step is a pure graph replay with the resident
        steering buffer. Returns ``None`` only when the batch is empty (force a
        re-route). adjust_rs recomputes its coefficient in-kernel from the live
        residual every replay, so skipping the host upload does not freeze its math.

        When any request uses a non-default ``phase``/``positions`` (``_any_gated``), the
        signature is extended with the per-slot steer columns — see the inline comment for
        the chunked-prefill hazard this closes.
        """
        req_ids = step.req_ids
        if not req_ids:
            return None
        base = (tuple(req_ids), tuple(int(x) for x in step.query_start_loc_np))
        if not self._any_gated:
            self._step_cols = None
            return base
        # Gated deployments: (req_ids, qsl) alone is NOT a sufficient signature. Two
        # consecutive equal-length prefill chunks of a stable batch produce an IDENTICAL
        # (req_ids, qsl), so the wrapper would skip and the FINAL chunk would never route
        # -> the request steers nothing at all. Append the resolved gate state.
        # Deliberately NOT num_computed_tokens: that moves every decode step and would
        # kill the steady-state W1 skip. step_cols is stable through steady decode.
        cols = self._resolve_step_cols(step)
        self._step_cols = cols
        return base + (tuple(cols),)

    def reset_pinned(self, width: Optional[int] = None) -> None:
        self._aperture.wait_current()
        coeff = self._aperture.cur("coeff")
        vec_id = self._aperture.cur("vec_id")
        mode = self._aperture.cur("mode")
        if width is None:
            coeff.zero_()
            vec_id.zero_()
            mode.zero_()
        else:
            w = max(1, min(int(width), self.cap))
            coeff[:, :w].zero_()
            vec_id[:, :w].zero_()
            mode[:, :w].zero_()

    def upload(self, width: Optional[int] = None) -> None:
        # Only the [:, :width] column prefix the cudagraph-padded forward reads (see
        # install._upload_width); the stale tail beyond width is never read by the op.
        coeff = self._aperture.cur("coeff")
        vec_id = self._aperture.cur("vec_id")
        mode = self._aperture.cur("mode")
        if width is None:
            self.coeff_all.copy_(coeff, non_blocking=True)
            self.vec_id_all.copy_(vec_id, non_blocking=True)
            self.mode_all.copy_(mode, non_blocking=True)
        else:
            w = max(1, min(int(width), self.cap))
            self.coeff_all[:, :w].copy_(coeff[:, :w], non_blocking=True)
            self.vec_id_all[:, :w].copy_(vec_id[:, :w], non_blocking=True)
            self.mode_all[:, :w].copy_(mode[:, :w], non_blocking=True)
        self._aperture.record_advance()

    def force_full_routing(self) -> None:
        """Force the next ``apply_incremental_routing`` to rewrite [0,width) in full.

        Used by the routing wrapper after an error left the device slabs in an unknown
        state, so recovery re-establishes the whole prefix rather than trusting the
        stale shadow.
        """
        self._inc_force_full = True
        self._slot_req_key = None   # GPU routing: force a slot-config rebuild + re-scatter too

    def _intern_state(self, layer: int, coeff: float, vid: int, mode: int) -> int:
        """Map a (layer, coeff, vid, mode) steer state to a stable id (>=1)."""
        key = (int(layer), float(coeff), int(vid), int(mode))
        sid = self._st_intern.get(key)
        if sid is None:
            sid = self._st_next
            self._st_next += 1
            self._st_intern[key] = sid
            self._st_val[sid] = key
        return sid

    def apply_incremental_routing(self, assignments: list, width: int) -> bool:
        """Write + upload ONLY the (layer, column) routing cells whose steer state changed.

        ``assignments`` is one ``(start, end, layer, coeff, vid, mode)`` per (request,
        target layer): a request steering N layers emits N assignments sharing the same
        ``[start, end)`` column range but distinct layers. Each (layer, column) cell holds
        at most one active steer state (requests own disjoint flat column ranges within a
        forward, and a request lists each target layer once), interned to a stable id; a
        cell needs a rewrite iff that id changed vs the ``_col_state`` shadow.

        The shadow is 2-D ``(num_layers, cap)`` — one interned id per (layer, column). A 1-D
        per-column shadow collapsed the N same-column assignments onto one cell, keeping only
        the LAST layer, so a multi-layer request silently dropped all but one layer. Tracking
        (layer, column) writes every target row. Steady-state stable decode (and every idle
        step) changes 0 cells -> no upload, a pure replay; a +1 prefill / finish / condense
        changes only the affected cells -> O(changed) work across the three (coeff/vec_id/
        mode) planes. Returns True iff an upload was issued (diagnostic).

        Clearing is load-bearing: a (layer, column) that WAS steered and is now inactive (its
        request finished / moved, or a target layer dropped) flips to id 0, is detected as
        changed, and its three planes are zeroed — so stale steering never persists into a
        replay. A cell whose state is stable keeps its id and is skipped (0 upload).
        """
        cap = self.cap
        nL = self.num_layers
        w = max(1, min(int(width), cap))
        if self._col_state is None:
            self._col_state = np.zeros((nL, cap), dtype=np.int64)
            pin = self.device.type == "cuda"
            self._inc_coeff = torch.zeros(nL, cap, dtype=torch.float32, pin_memory=pin)
            self._inc_vecid = torch.zeros(nL, cap, dtype=torch.int64, pin_memory=pin)
            self._inc_mode = torch.zeros(nL, cap, dtype=torch.int64, pin_memory=pin)
            self._inc_event = torch.cuda.Event() if pin else None
            # Device slabs (constructor zeros) + shadow + mirrors all start zeroed.

        # Desired per-(layer, column) interned state over [:, :cap] (only [:, :w) is
        # read/diffed; active columns are always < real_n <= w). Disjoint flat column ranges
        # AND distinct rows per request -> no cell is written twice.
        new_state = np.zeros((nL, cap), dtype=np.int64)
        for (start, end, layer, coeff, vid, mode) in assignments:
            if end <= start:
                continue
            new_state[layer, start:end] = self._intern_state(layer, coeff, vid, mode)

        old = self._col_state
        if self._inc_force_full:
            changed_mask = np.ones((nL, w), dtype=bool)  # device state unknown -> rewrite all
            self._inc_force_full = False
        else:
            changed_mask = new_state[:, :w] != old[:, :w]
        rows_idx, cols_idx = np.nonzero(changed_mask)
        if rows_idx.size == 0:
            return False

        # Row-narrowing: upload only the layer-ROWS holding a changed cell, over the changed
        # column span. Every other row is identically 0 on both device and mirror (never
        # written), so copying it would be wasted H2D — far fewer bytes on the common
        # single-target-layer deployment, and it fires every active-upload step regardless of
        # churn. Multi-layer / multi-request-different-layer batches are fully preserved: every
        # in-flight target (layer, column) is a changed cell, so its row is uploaded. Op
        # placement is unchanged (the op stays baked on ALL layers).
        rows = np.unique(rows_idx)
        lo = int(cols_idx.min())
        hi = int(cols_idx.max()) + 1

        if self._inc_event is not None:
            self._inc_event.synchronize()  # last upload of these mirrors is done
        rr = torch.from_numpy(rows_idx)
        cc = torch.from_numpy(cols_idx)
        # Clear every changed cell in all three planes, then set the active ones per distinct
        # new state (a cell going to id 0 = deactivated stays cleared).
        self._inc_coeff[rr, cc] = 0
        self._inc_vecid[rr, cc] = 0
        self._inc_mode[rr, cc] = 0
        new_changed = new_state[rows_idx, cols_idx]
        for sid in np.unique(new_changed):
            if sid == 0:
                continue
            m = new_changed == sid
            _layer, coeff, vid, mode = self._st_val[int(sid)]
            rr_s = torch.from_numpy(rows_idx[m])
            cc_s = torch.from_numpy(cols_idx[m])
            self._inc_coeff[rr_s, cc_s] = coeff
            self._inc_vecid[rr_s, cc_s] = vid
            self._inc_mode[rr_s, cc_s] = mode

        for L in rows.tolist():
            self.coeff_all[L, lo:hi].copy_(self._inc_coeff[L, lo:hi], non_blocking=True)
            self.vec_id_all[L, lo:hi].copy_(self._inc_vecid[L, lo:hi], non_blocking=True)
            self.mode_all[L, lo:hi].copy_(self._inc_mode[L, lo:hi], non_blocking=True)
        if self._inc_event is not None:
            self._inc_event.record()
        old[:, :w] = new_state[:, :w]
        return True

    def refresh_slot_config(self, step: StepView) -> bool:
        """Rebuild the positional per-slot config table IFF the batch composition/order
        changed. Returns True iff it re-resolved + re-uploaded.

        Each request's steer config is immutable for its lifetime, so an unchanged
        ``tuple(req_ids)`` means the slot table is already valid -> skip (no host work, no
        H2D). Reuses the EXACT per-request resolution of ``_build_routing_steer``
        (``_resolve_steer_config`` / ``_parse_steer_layers`` / ``vec_id_for_path``) but
        writes O(reqs) per-slot entries indexed by batch POSITION i, not the
        O(num_layers x cap) plane. The ``vec_id_for_path`` side effect (interning a vector
        into ``vec_table`` / ``avg_proj_table``) is preserved — bypassing it would leave the
        resident vector library unloaded. Positions >= bs are left stale on purpose: the
        scatter's ``req_of_col`` only indexes ``[0, bs)``, so they are never read.

        Reads only ``step`` — never the model runner.
        """
        req_ids = step.req_ids
        key = tuple(req_ids)
        if key == self._slot_req_key:
            return False
        worker = _ACTIVE_WORKER_STEER
        env_path = getattr(worker, "_env_config_path", None) if worker else None
        bs = len(req_ids)
        prev = self._slot_req_key
        # Re-resolve ONLY the positions whose occupant changed. A request's steer config is
        # immutable for its lifetime — the SAME invariant the W1 whole-step ``routing_key`` skip
        # already relies on — so an unchanged req_id at position i means slot i is still valid.
        # Positions at/after ``len(prev)`` had no prior occupant and are always re-resolved, which
        # keeps a batch that GROWS past its previous high-water mark from reading a slot left over
        # from an older, larger batch. Re-resolving only the changed positions keeps admitting one
        # request O(1) instead of O(batch): re-resolving every in-flight request on each arrival
        # would make admission cost scale with how many requests are already running.
        if prev is None:
            changed = range(bs)
        else:
            n_prev = len(prev)
            changed = [i for i in range(bs) if i >= n_prev or prev[i] != req_ids[i]]
        vid_np, mode_np = self._slot_vid_np, self._slot_mode_np
        coeff_np, mask_np = self._slot_coeff_np, self._slot_mask_np
        for i in changed:
            # Clear FIRST: a reused slot must never inherit the previous occupant's
            # vector/method/coeff/layers when the newcomer resolves to nothing.
            vid_np[i] = 0
            mode_np[i] = 0
            coeff_np[i] = 0.0
            mask_np[i] = False
            extra = step.extra_args_for(i)
            steer_arg = (extra or {}).get("steer")
            cfg = _resolve_steer_config(steer_arg, env_path)
            if cfg is None:
                continue
            layers = _parse_steer_layers(cfg.get("optimal_layer", -1), self.num_layers)
            method = cfg.get("method", "adjust_rs")
            vector_path = cfg.get("vector_path")
            if not layers or method not in ("add_vector", "adjust_rs") or not vector_path:
                continue
            vid = self.vec_id_for_path(vector_path, worker)
            if vid is None:
                continue
            if method == "adjust_rs" and vid not in self.vec_has_avgproj:
                # adjust_rs needs avg_proj for the in-kernel coefficient; skip (never steer wrong)
                continue
            vid_np[i] = vid
            mode_np[i] = 1 if method == "adjust_rs" else 0
            coeff_np[i] = 0.0 if method == "adjust_rs" else float(
                cfg.get("coefficient", 0.0))
            # ONE vectorized write for this request's whole layer set. Per-request layer
            # sets stay arbitrary/heterogeneous — this is a positional scatter, NOT a
            # shared-layer-set fast path (which would silently flatten a mixed batch).
            mask_np[i, np.asarray(layers, dtype=np.intp)] = True
            # Capture evidence (worker-side): this slot just resolved to an ACTIVE steer. Counts
            # ~steered requests over the run (a composition change admits requests one at a time),
            # so the harvest's hook_fire_count > 0 and the capture verdict confirms steering FIRED.
            # Harvested OFFLINE via dump_profiler (serve uses the driver-side steer.fire — the worker
            # dump is lost at serve teardown). PROF-only; no behavioural change.
            PROF.incr("steer.fire")
        self.slot_vid[:bs].copy_(self._slot_vid_h[:bs], non_blocking=True)
        self.slot_mode[:bs].copy_(self._slot_mode_h[:bs], non_blocking=True)
        self.slot_coeff[:bs].copy_(self._slot_coeff_h[:bs], non_blocking=True)
        self.slot_layer_mask[:bs].copy_(self._slot_mask_h[:bs], non_blocking=True)
        self._slot_req_key = key
        return True

    def _qsl_device(self, step: StepView):
        """This step's cumulative query_start_loc on the DEVICE, ``[:num_reqs+1]``.

        ``step.query_start_loc`` IS the tensor `prepare_inputs` returned (V2 keeps it
        device-resident for attention to read directly), already sliced to
        ``[:num_reqs+1]`` by ``step_view`` — no resident fallback buffer or extra H2D
        copy needed here now that routing reads the RETURNED batch directly.
        """
        return step.query_start_loc

    def build_and_upload_gpu(self, step: StepView, width, build_routing_fn=None) -> list:
        """Lever A per-step routing: O(reqs) host slot refresh (only on a req_ids change) +
        a GPU scatter into the coeff/vec_id/mode slabs — replacing the host build. Returns []
        (steering has no egress -> plans are unused). ``build_routing_fn`` is ignored (steer
        does its own per-request resolution; the arg keeps the wrapper worker-agnostic —
        HostRegistry's capture variant DOES use it). Byte-identical device slabs vs the host
        path (proven by the slab-equal oracle). Any error propagates to the wrapper's except,
        which forces a full re-route next step.
        """
        from mia.graph.steer_routing_gpu import scatter_routing
        # True iff the batch composition changed -> the only steps on which a gated
        # request can have ENTERED (see the pre-latch probe below).
        composition_changed = self.refresh_slot_config(step)
        qsl_np = step.query_start_loc_np
        real_n = int(qsl_np[-1]) if qsl_np.size else 0
        qsl_dev = self._qsl_device(step)
        # phase x positions gate. routing_key already resolved this step's columns when
        # _any_gated was set; resolve here PRE-LATCH, i.e. before any gated request has
        # been seen. When nothing is gated we pass slot_col=None -> HAS_COL=False -> the
        # kernel is byte-identical to the pre-gate one and no H2D is issued.
        #
        # Gate the probe on a composition change. A request's steer config is immutable
        # for its lifetime — the invariant the W1 skip and the incremental slot refresh
        # already rest on — so a gated request can only enter on a step where the batch
        # composition changed, and it latches on that same step (no lost step). Probing
        # unconditionally would re-resolve EVERY slot on every non-skipped step just to
        # notice a latch that never fires on the default path — paying back exactly the
        # O(batch) host cost the incremental refresh exists to remove.
        slot_col = None
        cols = self._step_cols
        if cols is None and composition_changed:
            cols = self._resolve_step_cols(step)
        if self._any_gated and cols:
            bs = min(len(cols), self.cap)
            self._slot_col_h[:bs].copy_(torch.as_tensor(cols[:bs], dtype=torch.int64))
            self.slot_col[:bs].copy_(self._slot_col_h[:bs], non_blocking=True)
            slot_col = self.slot_col
        self._step_cols = None
        scatter_routing(qsl_dev, self.slot_vid, self.slot_mode, self.slot_coeff,
                        self.slot_layer_mask, self.coeff_all, self.vec_id_all, self.mode_all,
                        real_n, width, slot_col=slot_col)
        return []

    def vec_id_for_path(self, path: str, worker) -> Optional[int]:
        """Resolve ``vector_path`` to a vec_table row, loading + caching on first use.

        The vector's ``dir`` is written into a PRE-ALLOCATED ``vec_table`` row in place
        (data_ptr fixed), and its ``avg_proj`` (adjust_rs vectors carry it) into the
        matching ``avg_proj_table`` row, so a never-before-seen vector is admitted
        without moving either table — graph safe. Returns None if the path can't be
        loaded or the table is full.
        """
        vid = self.vec_paths.get(path)
        if vid is not None:
            return vid
        cache = getattr(worker, "_vector_cache", None) if worker is not None else None
        if cache is None and worker is not None:
            worker._vector_cache = cache = {}
        data = cache.get(path) if cache is not None else None
        if data is None:
            try:
                raw = _load_steering_vector(path)
            except Exception:  # noqa: BLE001
                return None
            d = raw["dir"]
            data = {"dir": d if torch.is_tensor(d) else torch.tensor(d)}
            if "avg_proj" in raw:  # adjust_rs vectors carry the target projection
                ap = raw["avg_proj"]
                data["avg_proj"] = float(ap.item()) if torch.is_tensor(ap) else float(ap)
            if cache is not None:
                cache[path] = data
        if self._next_vec_id >= self.v_max:
            return None  # table full
        vid = self._next_vec_id
        self._next_vec_id += 1
        self.vec_table[vid].copy_(
            data["dir"].to(self.vec_table.device, dtype=self.vec_table.dtype).view(-1))
        if "avg_proj" in data:
            self.avg_proj_table[vid] = float(data["avg_proj"])
            self.vec_has_avgproj.add(vid)
        self.vec_paths[path] = vid
        return vid


def _install_steer_buffer(worker, model, matched, device) -> SteerRegistry:
    """Build the buffer-mode steering hosts + routing registry (no splitting op).

    One ``SteerHost`` per matched decoder layer; a ``SteerRegistry`` owns the
    per-(layer, token) coeff/vec_id routing slabs and the resident vector table.
    Routing runs in the ``prepare_inputs`` wrapper (per-request config → coeff/vec
    buffers); the layer wrap applies ``steer_buffer`` (masked in-place add). There is
    NO egress — steering produces no artifacts, only the residual mutation.
    """
    cap = _resolve_max_num_batched_tokens(worker)
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = int(getattr(text_cfg, "hidden_size"))
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0)) or (
        max(ln for _, _, ln in matched) + 1)
    v_max = int(os.environ.get("MIA_STEER_VMAX", "16"))
    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    registry = SteerRegistry(num_layers, cap, hidden, v_max, device, buf_dtype)

    n_wired = 0
    for name, module, layer_num0 in matched:
        if 0 <= layer_num0 < num_layers:
            host = SteerHost(module_name=name, layer_num=layer_num0, cap=cap,
                             do_steer=True)
            setattr(module, _STEER_HOST_ATTR, host)
            registry.register_host(host)
            n_wired += 1
        _wrap_layer_class(type(module))
    registry.assign_views()

    set_registry(worker, "steer", registry)
    # Routing only — steering has no egress, so no execute_model wrapper is needed.
    # Phase 4 idle routing fast-path (opt-in, MIA_STEER_ROUTE_FASTPATH=1):
    # override the W1 invalidation key with _steer_idle_key so fully-non-steering
    # churn steps skip reset/build/upload. Default OFF → routing_key_fn=None →
    # install.py falls back to registry.routing_key → byte-identical to today.
    install_prepare_inputs_routing(
        worker.model_runner, worker, _build_routing_steer, label="steer",
        routing_key_fn=_steer_idle_key if _STEER_ROUTE_FASTPATH else None)
    print(f"[graph/install_steer] buffer-mode steering wired: {n_wired} layer(s) over "
          f"{num_layers} slots; cap={cap} hidden={hidden} V_max={v_max}; "
          f"NO splitting op (rides decode cudagraph)")
    return registry


def _steer_idle_key(step: StepView, registry):
    """W1 idle-skip key for buffer-mode steering (mirrors ``_capture_idle_key``).

    Returns ``_IDLE_ROUTE_KEY`` only when NO request in the batch would steer — a
    constant across consecutive idle steps, so the routing wrapper skips
    reset/build/upload even under churn (where ``req_ids`` / ``qsl`` change every
    step and the default ``registry.routing_key`` would rebuild every step because
    its signature moves with the batch). Returns the *real* ``registry.routing_key``
    signature the moment any request MIGHT steer (or when env-configured steering is
    active, which targets EVERY request) — so active steps keep today's whole-step
    skip/rebuild behaviour byte-identical, and the idle key never leaks into an
    active step.

    Cheap scalar gate only (no tensor writes, no f-strings), early-exits on the first
    steering request. A request with a falsy ``extra_args["steer"]`` cannot steer
    (matches ``_resolve_steer_config``'s ``if not steer_arg: return None``); a truthy
    one is treated conservatively as steering — a false positive (it resolves to a
    no-op via a bad layer/vector) only costs a missed skip, never wrong data.

    Correctness of the active↔idle boundary: the last active step stored a real
    ``routing_key`` as ``_last_route_key``; the first fully-idle step returns
    ``_IDLE_ROUTE_KEY`` which DIFFERS, so the wrapper does NOT skip → the incremental
    router rebuilds with 0 assignments → ``apply_incremental_routing`` zeroes the
    vacated steer columns exactly once. Only *subsequent* all-idle steps skip
    (replaying the now-zeroed slabs = a genuine no-op steer). Symmetric on
    idle→active: prior ``_last_route_key`` is ``_IDLE_ROUTE_KEY``, the new real key
    differs → rebuild writes the new steer columns. No stale steer column can leak.
    """
    req_ids = step.req_ids
    if not req_ids:
        return None
    worker = _ACTIVE_WORKER_STEER
    # Env-configured steering targets EVERY request unconditionally (no per-request
    # `steer` arg), so the batch can never be idle → defer to the real signature.
    if worker is not None and getattr(worker, "_env_config_path", None):
        return registry.routing_key(step)
    for i in range(step.num_reqs):
        extra = step.extra_args_for(i)
        if extra and extra.get("steer"):
            # This request MIGHT steer → not idle. Return the real per-request
            # signature so active steps keep today's whole-step skip/rebuild
            # behaviour (byte-identical); the transition to fully-idle still forces a
            # rebuild because _IDLE_ROUTE_KEY differs from this key.
            return registry.routing_key(step)
    return _IDLE_ROUTE_KEY


def _build_routing_steer(step: StepView, registry: SteerRegistry) -> list:
    """Fill the pinned coeff/vec_id/mode mirrors from each request's steer config.

    For every request whose resolved config targets layer L, write the vector's
    ``vec_id`` and the per-token ``mode`` across the request's token span at row L:
      * ``add_vector`` → ``mode=0`` + ``coeff=coefficient`` (host-supplied delta).
      * ``adjust_rs``  → ``mode=1`` + ``coeff=0`` (the op computes the coefficient in
        kernel from the live residual; the vector must carry ``avg_proj``).
    Unsteered (layer, token) entries stay 0 (an exact no-op). Uses flat positions
    (steering modifies the current step's residual; no accumulation).
    """
    worker = _ACTIVE_WORKER_STEER
    req_ids = step.req_ids
    bs = step.num_reqs
    qsl = step.query_start_loc_np
    inc = registry.incremental_enabled
    # Legacy aperture mirrors are only written when incremental routing is off.
    if not inc:
        coeff_pinned = registry.coeff_pinned
        vecid_pinned = registry.vec_id_pinned
        mode_pinned = registry.mode_pinned
    cap = registry.cap
    env_path = getattr(worker, "_env_config_path", None) if worker else None

    plans: list = []
    assignments: list = []  # (start, end, layer, coeff, vid, mode) for the incremental router
    for i in range(bs):
        req_id = req_ids[i]
        extra = step.extra_args_for(i)
        steer_arg = (extra or {}).get("steer")
        cfg = _resolve_steer_config(steer_arg, env_path)
        if cfg is None:
            continue
        layers = _parse_steer_layers(cfg.get("optimal_layer", -1), registry.num_layers)
        if not layers:
            continue
        method = cfg.get("method", "adjust_rs")
        if method not in ("add_vector", "adjust_rs"):
            continue
        vector_path = cfg.get("vector_path")
        if not vector_path:
            continue
        vid = registry.vec_id_for_path(vector_path, worker)
        if vid is None:
            continue
        if method == "adjust_rs" and vid not in registry.vec_has_avgproj:
            # adjust_rs needs the vector's avg_proj target; without it the in-kernel
            # coefficient would be wrong, so skip rather than steer incorrectly.
            continue
        mode_val = 1 if method == "adjust_rs" else 0
        coefficient = 0.0 if method == "adjust_rs" else float(cfg.get("coefficient", 0.0))

        start = int(qsl[i])
        end = min(int(qsl[i + 1]), cap)
        if end <= start:
            continue

        # phase x positions gate — the same steer_span() the eager hook and the GPU
        # router use, so the three paths cannot disagree. Default modes return the whole
        # span, keeping this byte-identical to the pre-gate behaviour.
        try:
            phase, positions = resolve_steer_modes(cfg)
        except ValueError:
            continue
        if not is_default_steer_modes(phase, positions):
            registry._any_gated = True
        is_prefill = bool(step.is_prefilling_np[i])
        is_final = True
        if is_prefill and positions == "last_token":
            # PROMPT length (prompt_len_np), not prefill_len_np -- prefill_len can exceed
            # the prompt after preemption-resume (see mia.runner.step_view).
            n_computed = int(step.num_computed_tokens_np[i])
            n_prompt = int(step.prompt_len_np[i])
            is_final = (n_computed + (end - start)) >= n_prompt
        span = steer_span(phase, positions, is_prefill, is_final, start, end)
        if span is None:
            continue
        row_lo, row_hi = span

        for layer in layers:
            if inc:
                # Record (column-range, layer, coeff, vid, mode); the registry diffs vs last
                # step and writes/uploads only the columns that changed — no per-request tensor
                # ops on the hot path (the cost that scaled with batch width x num_layers).
                assignments.append((row_lo, row_hi, layer, coefficient, vid, mode_val))
            else:
                coeff_pinned[layer, row_lo:row_hi] = coefficient
                vecid_pinned[layer, row_lo:row_hi] = vid
                mode_pinned[layer, row_lo:row_hi] = mode_val
            plans.append({"req_idx": i, "layer": layer, "method": method, "vid": vid})

    if inc:
        registry._pending_assignments = assignments
    return plans


__all__ = ["install_steer_hosts"]
