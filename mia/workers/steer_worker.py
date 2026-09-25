import os
import json
import numpy as np
import torch
from typing import TYPE_CHECKING, Any, Dict, Optional

from mia._profiler import PROF
from mia.runner import StepView, install_request_arg_stash, require_v2_runner, step_view
from mia.workers._common import iter_matched_modules, match_layer

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


def _load_steering_vector(vector_path: str) -> Dict:
    """Load and parse a steering vector .pt file. Returns the raw dict."""
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found at: {vector_path}")
    return torch.load(vector_path, weights_only=False)


def _resolve_steer_config(steer_arg, env_default_path: Optional[str]) -> Optional[Dict]:
    """Normalize ``extra_args["steer"]`` into a config dict, or None if disabled.

    Accepts:
        True            -> read from env_default_path (legacy)
        {dict}          -> use directly (per-request override)
        False / None    -> steering disabled for this request
    """
    if not steer_arg:
        return None
    if isinstance(steer_arg, dict):
        return steer_arg
    # Legacy True: fall back to the file pointed to by MIA_STEER_CONFIG
    if env_default_path and os.path.exists(env_default_path):
        with open(env_default_path) as f:
            return json.load(f).get("steering", {})
    return None


def _parse_steer_layers(optimal_layer, num_layers: int) -> list:
    """``optimal_layer`` -> sorted unique list of valid layer indices.

    Accepts ``int`` (single layer, legacy), ``list[int]`` (multi-layer steer), or ``"all"``.
    Out-of-range / unparseable entries are dropped. Backward-compatible: an int ``L`` in range
    returns ``[L]``, so the single-layer path is unchanged.
    """
    if optimal_layer == "all":
        return list(range(int(num_layers)))
    raw = optimal_layer if isinstance(optimal_layer, (list, tuple)) else [optimal_layer]
    out = set()
    for x in raw:
        try:
            L = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= L < int(num_layers):
            out.add(L)
    return sorted(out)


def _steer_targets_layer(optimal_layer, this_layer: int) -> bool:
    """True if a request whose config has ``optimal_layer`` should steer ``this_layer``.

    Eager-path counterpart of ``_parse_steer_layers`` that needs no ``num_layers`` (``"all"``
    always matches). Accepts int | list[int] | ``"all"``.
    """
    if optimal_layer == "all":
        return True
    if isinstance(optimal_layer, (list, tuple)):
        for x in optimal_layer:
            try:
                if int(x) == this_layer:
                    return True
            except (TypeError, ValueError):
                continue
        return False
    try:
        return int(optimal_layer) == this_layer
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Position + phase modes (the phase x positions gate)
# ---------------------------------------------------------------------------
# Two ORTHOGONAL per-request axes, mirroring the capture path's hooks_on x mode:
#   phase     : "prefill" | "decode" | "both"        (default "both")
#   positions : "all_tokens" | "last_token"          (default "all_tokens")
# Both defaults reproduce the steer-everything behaviour that predates them.
#
# NOTE the deliberate divergence from capture: capture's hooks_on defaults to
# "prefill", steering's phase defaults to "both". Steering today applies everywhere and
# no existing config may change meaning.
_STEER_PHASES = ("prefill", "decode", "both")
_STEER_POSITIONS = ("all_tokens", "last_token")

# Sentinels for the resolved per-step steer column consumed by the graph GPU router.
STEER_COL_ALL = -1    # steer the request's whole span (today's behaviour)
STEER_COL_NONE = -2   # steer nothing this step


def resolve_steer_modes(cfg: Optional[Dict]) -> tuple:
    """``(phase, positions)`` for a resolved steer config dict.

    Legacy ``apply_at_all_positions: false`` maps to ``positions="last_token"`` when no
    explicit ``positions`` key is present; an explicit ``positions`` always wins.

    Raises ``ValueError`` on an unknown value. This is deliberate: a silently-ignored
    ``positions`` typo would steer EVERY token while the caller believes they are
    steering one — the same failure mode ``optimizations.PUBLIC_LEVERS`` fails loud on.
    """
    if not cfg:
        return ("both", "all_tokens")
    phase = cfg.get("phase", "both")
    if phase not in _STEER_PHASES:
        raise ValueError(
            f"steering.phase={phase!r} is invalid; expected one of {_STEER_PHASES}")
    positions = cfg.get("positions")
    if positions is None:
        positions = ("all_tokens" if cfg.get("apply_at_all_positions", True)
                     else "last_token")
    if positions not in _STEER_POSITIONS:
        raise ValueError(
            f"steering.positions={positions!r} is invalid; "
            f"expected one of {_STEER_POSITIONS}")
    return (phase, positions)


def is_default_steer_modes(phase: str, positions: str) -> bool:
    """True iff these modes reproduce the steer-every-token-of-every-pass behaviour."""
    return phase == "both" and positions == "all_tokens"


def steer_span(phase: str, positions: str, is_prefill: bool, is_final_chunk: bool,
               start: int, end: int):
    """The ``(lo, hi)`` half-open column range this request steers this pass, or None.

    THE single source of truth for the phase x positions gate — the eager hook, the graph
    host router and the graph GPU router all bottom out here, so the three paths cannot
    disagree.

    ``is_final_chunk`` is capture's ``emit_q`` gate (``graph/install.py`` ~line 526):
    ``num_computed + qlen >= num_prompt``. It makes ``last_token`` CHUNK-INVARIANT — the
    last *prompt* token, never a scheduler chunk boundary. Without it the steered set
    would depend on ``max_num_batched_tokens`` and stop being reproducible.

    A decode pass has exactly one query token, so ``last_token`` and ``all_tokens``
    coincide there; no special case is needed.
    """
    if end <= start:
        return None
    if phase == "prefill" and not is_prefill:
        return None
    if phase == "decode" and is_prefill:
        return None
    if positions == "last_token":
        if is_prefill and not is_final_chunk:
            return None
        return (end - 1, end)
    return (start, end)


def steer_col_for(phase: str, positions: str, is_prefill: bool, is_final_chunk: bool,
                  start: int, end: int) -> int:
    """``steer_span`` expressed as the graph router's per-slot ``slot_col`` sentinel.

    ``STEER_COL_ALL`` whenever the whole span is steered (so the default path keeps a
    constant, upload-free routing key), ``STEER_COL_NONE`` when nothing is, else the
    absolute flat column to steer.
    """
    span = steer_span(phase, positions, is_prefill, is_final_chunk, start, end)
    if span is None:
        return STEER_COL_NONE
    lo, hi = span
    if lo == start and hi == end:
        return STEER_COL_ALL
    return lo


def _steer_rows(rows: "torch.Tensor", cfg: Dict, data: Dict) -> "torch.Tensor":
    """Apply one steer config to EVERY row of ``rows``; return the result.

    The exact op sequence the eager path has always used. It knows nothing about requests:
    ``SteerWorker._steer_per_request`` passes it the whole residual and keeps only the rows
    each config owns. Keep it that way -- both methods are row-wise in exact arithmetic but
    not bitwise: ``matmul`` blocking depends on the row count, so ``adjust_rs`` on a row
    slice can differ from the whole-tensor op in the last ulp.
    """
    method = cfg.get("method", "adjust_rs")
    steering_vec = data["dir"].to(rows.device, dtype=rows.dtype)
    if method == "add_vector":
        coefficient = float(cfg.get("coefficient", 0))
        return rows + coefficient * steering_vec.view(1, -1)
    if method == "adjust_rs":
        unit_vec = steering_vec  # use dir as unit vector (matches old behavior)
        avg_proj = data["avg_proj"].to(rows.device, dtype=rows.dtype)
        current_projections = torch.matmul(rows, unit_vec)
        coeff = (avg_proj - current_projections).unsqueeze(-1)
        return rows + coeff * unit_vec.view(1, -1)
    raise ValueError(f"Unknown steering method: {method}")


def _effective_key(cfg: Dict) -> tuple:
    """Everything ``_steer_rows`` reads from ``cfg``: equal keys steer a row identically.

    ``adjust_rs`` derives its coefficient from the live residual and ignores
    ``coefficient`` -- as the graph path does, which forces it to 0 there -- so adjust_rs
    configs on one vector share a key whatever coefficient they carry.
    """
    method = cfg.get("method", "adjust_rs")
    coeff = float(cfg.get("coefficient", 0)) if method == "add_vector" else None
    return (cfg.get("vector_path"), method, coeff)


class SteerWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    Per-request steering: each request can pass its own steering config in
    extra_args["steer"] (dict) — different requests in the same batch can use
    different vectors / methods / coefficients / optimal layers.
    """

    if TYPE_CHECKING:
        model_runner: Any

    _hooks_installed: bool = False
    _bad_cfg_warned: int = 0   # cap the invalid-config warning (hot path)
    _unmappable_warned: bool = False   # warn once: rows the step cannot attribute
    _step: "StepView | None" = None

    def install_hooks(self):
        """Install steering hooks on every transformer layer. Idempotent.

        Each hook checks per-request ``extra_args["steer"]`` and applies
        steering only when the request targets this hook's layer.

        Fails loud (UnsupportedRunnerError) if the live runner is not vLLM's V2
        GPUModelRunner. This check -- and the StepView wiring below it -- MUST run
        OUTSIDE the try/except around ``_install_hooks()``: that guard used to swallow
        everything into a log line, which would silently defeat the fail-loud contract
        (a V1 runner would again steer nothing while reporting success).
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        runner = self.model_runner
        require_v2_runner(runner)
        # Steering is applied on EVERY TP rank (the residual it edits is replicated), but under
        # PIPELINE parallelism a rank owns only its stage's layers -- refuse, like capture does.
        # Outside the try/except below, which would otherwise swallow the refusal.
        from mia.graph.tp_shard import refuse_pipeline_parallel
        refuse_pipeline_parallel(
            getattr(getattr(self, "parallel_config", None), "pipeline_parallel_size", 1),
            "steer install_hooks")
        stash = install_request_arg_stash(runner)
        self._step = None

        # Snapshot the transient per-step InputBatch into an immutable StepView the
        # instant it's produced. Forward hooks read self._step; they never touch the
        # runner directly (V2's InputBatch is not stored on the runner).
        original_prepare = runner.prepare_inputs

        def prepare_inputs(*args, **kwargs):
            input_batch = original_prepare(*args, **kwargs)
            self._step = step_view(runner, input_batch, stash)
            return input_batch

        runner.prepare_inputs = prepare_inputs

        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")

    def dump_profiler(self) -> "str | None":
        """collective_rpc-callable: dump this WORKER process's PROF snapshot to
        MIA_PROFILE_DIR and return the path (None if profiling is off). The
        steer.fire evidence counter lives in the worker (refresh_slot_config), so the
        offline driver reads it only via this dump. Mirrors the HS/QK workers."""
        from mia._profiler import PROF
        return PROF.dump(role="worker-rpc")

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # Cache for steering vectors loaded from disk, keyed by vector_path.
        # Loading per-request would be too slow.
        self._vector_cache: Dict[str, Dict] = {}
        # Legacy fallback: MIA_STEER_CONFIG points to a JSON file
        # whose "steering" key has the per-worker default config. Used only when
        # extra_args["steer"] is True (boolean) instead of a dict.
        self._env_config_path = os.environ.get("MIA_STEER_CONFIG")

        def steering_hook(input, output, this_layer: int):
            step = self._step
            if step is None:
                return output
            # Every request whose resolved config targets THIS layer, with its modes.
            entries = []      # (i, cfg, phase, positions)
            for i in range(step.num_reqs):
                steer_arg = (step.extra_args_for(i) or {}).get("steer")
                resolved = _resolve_steer_config(steer_arg, self._env_config_path)
                if resolved is None:
                    continue
                _ol = resolved.get("optimal_layer", -1)
                if not _steer_targets_layer(_ol, this_layer):
                    continue
                try:
                    phase, positions = resolve_steer_modes(resolved)
                except ValueError as exc:
                    # Match the graph routers: an invalid config makes THAT request
                    # inert rather than crashing the forward. Loud but capped — the
                    # offline path already fails hard at MiaLLM.load_config, so this
                    # only fires for a serve request that hand-rolled bad extra_args.
                    if self._bad_cfg_warned < 4:
                        self._bad_cfg_warned += 1
                        print(f"[steer] ignoring request with invalid steer config: {exc}",
                              flush=True)
                    continue
                entries.append((i, resolved, phase, positions))
            if not entries:
                return output

            is_tuple = isinstance(output, tuple)
            if is_tuple:
                hidden_states, residuals = output
            else:
                hidden_states = None
                residuals = output

            PROF.incr("steer.fire")

            # Every mode, the defaults included, steers only each request's own rows with its
            # own config. Never shortcut to one config over the whole tensor: that steers
            # every co-scheduled request -- unsteered ones and padding rows included -- with
            # whichever config sits in the lowest batch row.
            with PROF.timed("steer.apply", tier=2):
                residuals = self._steer_per_request(residuals, entries, step)

            if is_tuple:
                return (hidden_states, residuals)
            else:
                return residuals

        # Hook every transformer layer; the closure decides per-request whether
        # to actually steer based on extra_args["steer"].
        self._hooks = []
        matched = []
        for name, module, layer_num in iter_matched_modules(model, match_layer):
            hook = module.register_forward_hook(
                lambda m, i, o, ln=layer_num: steering_hook(i, o, ln)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} steering hooks on layers: {matched}")

    def _vector_cache_for(self, cfg: Dict) -> Dict:
        """Load + cache a config's steering vector (keyed by ``vector_path``)."""
        vector_path = cfg["vector_path"]
        data = self._vector_cache.get(vector_path)
        if data is None:
            raw = _load_steering_vector(vector_path)
            data = {"dir": torch.tensor(raw["dir"])}
            if "avg_proj" in raw:
                # Whatever method loads the file first, as the graph loader does: the cache
                # is keyed by path alone, so loading avg_proj only for an adjust_rs first
                # requester left a later adjust_rs on the same file without it (KeyError
                # mid-forward). An adjust_rs vector without avg_proj still raises, at use.
                # as_tensor, because the use site calls .to(device, dtype): a plain-float
                # avg_proj would run under CUDA graphs (the loader coerces either) but raise
                # here. A no-op for every shipped vector, which already stores a 0-d tensor.
                data["avg_proj"] = torch.as_tensor(raw["avg_proj"])
            self._vector_cache[vector_path] = data
        return data

    def _steer_per_request(self, residuals, entries, step: StepView):
        """Steer each request's OWN rows with its OWN config; leave every other row exact.

        Request ``i`` owns rows ``[qsl[i], qsl[i+1])`` of the flat residual, read from the
        StepView's HOST ``query_start_loc_np`` -- the copy the graph routers use, so there
        is no device sync and no forward-context dependency. ``steer_span`` narrows that
        to the phase x positions gate; for the default modes it is the whole span.

        Requests are grouped by ``_effective_key``. Each group runs ``_steer_rows`` ONCE,
        on the whole tensor (so every steered row is bit-identical to the op the default
        path always ran), and a host-built row mask keeps only the group's rows. That is
        O(#distinct configs) ops per layer, independent of batch width. When one group
        owns every row the mask is skipped and the result IS the whole-tensor op.

        If the step claims more rows than the tensor has (a stale StepView on a dummy run,
        a sharded residual), no row can be attributed to a request, so none is steered.
        """
        # UNVERIFIED under speculative decoding: with (opt-in) adaptive draft verification
        # the host query_start_loc_np is an even split of the draft budget while the device
        # qsl is re-allocated per request (vllm/v1/worker/gpu/spec_decode/
        # adaptive_verification.py, ~369-377 and reallocate_drafts), so these spans can be
        # wrong there. The graph routers read the same host copy; MIA + spec decode is untested.
        qsl = step.query_start_loc_np
        n_rows = int(residuals.shape[0])
        mapped = int(qsl[step.num_reqs]) if len(qsl) > step.num_reqs else None
        if mapped is None or mapped > n_rows:
            if not self._unmappable_warned:
                self._unmappable_warned = True
                print(f"[steer] the step maps {mapped} rows but the residual has {n_rows}; "
                      "steering nothing for this forward (logged once)", flush=True)
            return residuals

        groups: Dict[tuple, tuple] = {}   # effective key -> (cfg, [(lo, hi), ...])
        for (i, cfg, phase, positions) in entries:
            start = int(qsl[i])
            end = int(qsl[i + 1])
            is_prefill = bool(step.is_prefilling_np[i])
            is_final = True
            if is_prefill and positions == "last_token":
                # PROMPT length (prompt_len_np), not prefill_len -- prefill_len can
                # exceed the prompt after preemption-resume.
                is_final = (int(step.num_computed_tokens_np[i]) + (end - start)) \
                    >= int(step.prompt_len_np[i])
            span = steer_span(phase, positions, is_prefill, is_final, start, end)
            if span is None:
                continue
            groups.setdefault(_effective_key(cfg), (cfg, []))[1].append(span)

        # Spans of different requests are disjoint, so each group writes only rows no other
        # group owns and the order of the groups does not matter.
        out = residuals
        for cfg, spans in groups.values():
            steered = _steer_rows(residuals, cfg, self._vector_cache_for(cfg))
            mask = np.zeros(n_rows, dtype=bool)
            for lo, hi in spans:
                mask[lo:hi] = True
            if mask.all():
                out = steered
                continue
            # non_blocking: a pageable source is staged before .to() returns, so the
            # host mask may die with this frame, and the H2D copy costs no stream sync.
            keep = torch.from_numpy(mask).to(residuals.device, non_blocking=True)
            out = torch.where(keep.unsqueeze(-1), steered, out)
        return out

    # ------------------------------------------------------------------
    # v0.3.0 CUDA-graph steering install (graph mode only)
    # ------------------------------------------------------------------

    def graph_install(self):
        """Install the CUDA-graph steering path (buffer mode).

        Thin delegating entry called by the Worker.load_model monkey-patch
        (graph/install.py:patch_worker_load_model) AFTER the model is built but
        BEFORE warm-up/compile/capture. Only reached when graph mode is armed;
        the eager v0.2.0 register_forward_hook path is untouched.

        Unlike the QK/HS capture workers (which seed egress buckets here), steering
        produces no artifacts — it only mutates the residual — so this just wires
        the steer_buffer op via graph.install_steer.
        """
        from mia.graph.install_steer import install_steer_hosts
        install_steer_hosts(self)
