"""Activation-steering worker (steer): eager hooks and the CUDA-graph buffer path."""
import os
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

from mia._profiler import PROF
from mia.runner import StepView, install_request_arg_stash, require_v2_runner, step_view
from mia.workers._common import iter_matched_modules, match_layer


def _load_steering_vector(vector_path: str) -> Dict:
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found at: {vector_path}")
    return torch.load(vector_path, weights_only=False)


def _resolve_steer_config(steer_arg, env_default_path: Optional[str]) -> Optional[Dict]:
    if not steer_arg:
        return None
    if isinstance(steer_arg, dict):
        return steer_arg
    if env_default_path and os.path.exists(env_default_path):
        with open(env_default_path) as f:
            return json.load(f).get("steering", {})
    return None


def _parse_steer_layers(optimal_layer, num_layers: int) -> list:
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


_STEER_PHASES = ("prefill", "decode", "both")
_STEER_POSITIONS = ("all_tokens", "last_token")

STEER_COL_ALL = -1
STEER_COL_NONE = -2


def resolve_steer_modes(cfg: Optional[Dict]) -> tuple:
    """``(phase, positions)`` for a resolved steer config dict."""
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
    """True iff these modes steer every token of every pass."""
    return phase == "both" and positions == "all_tokens"


def steer_span(phase: str, positions: str, is_prefill: bool, is_final_chunk: bool,
               start: int, end: int):
    """The ``(lo, hi)`` half-open column range this request steers this pass, or None."""
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
    """``steer_span`` expressed as the graph router's per-slot ``slot_col`` sentinel."""
    span = steer_span(phase, positions, is_prefill, is_final_chunk, start, end)
    if span is None:
        return STEER_COL_NONE
    lo, hi = span
    if lo == start and hi == end:
        return STEER_COL_ALL
    return lo


def _steer_rows(rows: "torch.Tensor", cfg: Dict, data: Dict) -> "torch.Tensor":
    method = cfg.get("method", "adjust_rs")
    steering_vec = data["dir"].to(rows.device, dtype=rows.dtype)
    if method == "add_vector":
        coefficient = float(cfg.get("coefficient", 0))
        return rows + coefficient * steering_vec.view(1, -1)
    if method == "adjust_rs":
        unit_vec = steering_vec
        avg_proj = data["avg_proj"].to(rows.device, dtype=rows.dtype)
        current_projections = torch.matmul(rows, unit_vec)
        coeff = (avg_proj - current_projections).unsqueeze(-1)
        return rows + coeff * unit_vec.view(1, -1)
    raise ValueError(f"Unknown steering method: {method}")


def _effective_key(cfg: Dict) -> tuple:
    method = cfg.get("method", "adjust_rs")
    coeff = float(cfg.get("coefficient", 0)) if method == "add_vector" else None
    return (cfg.get("vector_path"), method, coeff)


class SteerWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls."""

    if TYPE_CHECKING:
        model_runner: Any

    _hooks_installed: bool = False
    _bad_cfg_warned: int = 0
    _unmappable_warned: bool = False
    _step: "StepView | None" = None

    def install_hooks(self):
        """Install steering hooks on every transformer layer."""
        if self._hooks_installed:
            return
        self._hooks_installed = True
        runner = self.model_runner
        require_v2_runner(runner)
        # lazy: keep mia.graph (reads env at import) out of import mia
        from mia.graph.tp_shard import refuse_pipeline_parallel
        refuse_pipeline_parallel(
            getattr(getattr(self, "parallel_config", None), "pipeline_parallel_size", 1),
            "steer install_hooks")
        stash = install_request_arg_stash(runner)
        self._step = None

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
        """Dump this worker's profiler snapshot to MIA_PROFILE_DIR; return the path or None."""
        return PROF.dump(role="worker-rpc")

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self._vector_cache: Dict[str, Dict] = {}
        self._env_config_path = os.environ.get("MIA_STEER_CONFIG")

        def steering_hook(input, output, this_layer: int):
            step = self._step
            if step is None:
                return output
            entries = []
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

            with PROF.timed("steer.apply", tier=2):
                residuals = self._steer_per_request(residuals, entries, step)

            if is_tuple:
                return (hidden_states, residuals)
            else:
                return residuals

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
        vector_path = cfg["vector_path"]
        data = self._vector_cache.get(vector_path)
        if data is None:
            raw = _load_steering_vector(vector_path)
            data = {"dir": torch.tensor(raw["dir"])}
            if "avg_proj" in raw:
                data["avg_proj"] = torch.as_tensor(raw["avg_proj"])
            self._vector_cache[vector_path] = data
        return data

    def _steer_per_request(self, residuals, entries, step: StepView):
        qsl = step.query_start_loc_np
        n_rows = int(residuals.shape[0])
        mapped = int(qsl[step.num_reqs]) if len(qsl) > step.num_reqs else None
        if mapped is None or mapped > n_rows:
            if not self._unmappable_warned:
                self._unmappable_warned = True
                print(f"[steer] the step maps {mapped} rows but the residual has {n_rows}; "
                      "steering nothing for this forward (logged once)", flush=True)
            return residuals

        groups: Dict[tuple, tuple] = {}
        for (i, cfg, phase, positions) in entries:
            start = int(qsl[i])
            end = int(qsl[i + 1])
            is_prefill = bool(step.is_prefilling_np[i])
            is_final = True
            if is_prefill and positions == "last_token":
                is_final = (int(step.num_computed_tokens_np[i]) + (end - start)) \
                    >= int(step.prompt_len_np[i])
            span = steer_span(phase, positions, is_prefill, is_final, start, end)
            if span is None:
                continue
            groups.setdefault(_effective_key(cfg), (cfg, []))[1].append(span)

        out = residuals
        for cfg, spans in groups.values():
            steered = _steer_rows(residuals, cfg, self._vector_cache_for(cfg))
            mask = np.zeros(n_rows, dtype=bool)
            for lo, hi in spans:
                mask[lo:hi] = True
            if mask.all():
                out = steered
                continue
            keep = torch.from_numpy(mask).to(residuals.device, non_blocking=True)
            out = torch.where(keep.unsqueeze(-1), steered, out)
        return out


    def graph_install(self):
        """Install the CUDA-graph steering path (buffer mode)."""
        # lazy: keep mia.graph (reads env at import) out of import mia
        from mia.graph.install_steer import install_steer_hosts
        install_steer_hosts(self)

