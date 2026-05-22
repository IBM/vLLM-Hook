import os
import json
import torch
from typing import TYPE_CHECKING, Any, Dict, Optional

from vllm_hook_plugins.workers._common import iter_matched_modules
from vllm_hook_plugins.workers.probe_hidden_states_worker import match_layer

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
    # Legacy True: fall back to the file pointed to by VLLM_ACTSTEER_CONFIG
    if env_default_path and os.path.exists(env_default_path):
        with open(env_default_path) as f:
            return json.load(f).get("steering", {})
    return None


class SteerHookActWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    Per-request steering: each request can pass its own steering config in
    extra_args["steer"] (dict) — different requests in the same batch can use
    different vectors / methods / coefficients / optimal layers.
    """

    if TYPE_CHECKING:
        model_runner: Any

    _hooks_installed: bool = False

    def install_hooks(self):
        """Install steering hooks on every transformer layer. Idempotent.

        Each hook checks per-request ``extra_args["steer"]`` and applies
        steering only when the request targets this hook's layer.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # Cache for steering vectors loaded from disk, keyed by vector_path.
        # Loading per-request would be too slow.
        self._vector_cache: Dict[str, Dict] = {}
        # Legacy fallback: VLLM_ACTSTEER_CONFIG points to a JSON file
        # whose "steering" key has the per-worker default config. Used only when
        # extra_args["steer"] is True (boolean) instead of a dict.
        self._env_config_path = os.environ.get("VLLM_ACTSTEER_CONFIG")

        def steering_hook(input, output, this_layer: int):
            req_ids = getattr(self.model_runner.input_batch, "req_ids", [])
            # Find the first active steering config that targets this layer.
            cfg: Optional[Dict] = None
            for r in req_ids:
                req_state = self.model_runner.requests.get(r)
                if not req_state or req_state.sampling_params is None:
                    continue
                steer_arg = (req_state.sampling_params.extra_args or {}).get("steer")
                resolved = _resolve_steer_config(steer_arg, self._env_config_path)
                if resolved is None:
                    continue
                if int(resolved.get("optimal_layer", -1)) != this_layer:
                    continue
                cfg = resolved
                break
            if cfg is None:
                return output

            method = cfg.get("method", "adjust_rs")
            coefficient = float(cfg.get("coefficient", 0))
            apply_at_all_positions = bool(cfg.get("apply_at_all_positions", True))
            vector_path = cfg["vector_path"]

            # Cache the loaded vector data per path.
            data = self._vector_cache.get(vector_path)
            if data is None:
                raw = _load_steering_vector(vector_path)
                data = {"dir": torch.tensor(raw["dir"])}
                if method == "adjust_rs":
                    data["avg_proj"] = raw["avg_proj"]
                self._vector_cache[vector_path] = data

            is_tuple = isinstance(output, tuple)
            if is_tuple:
                hidden_states, residuals = output
            else:
                hidden_states = None
                residuals = output

            steering_vec = data["dir"].to(residuals.device, dtype=residuals.dtype)

            if method == "add_vector":
                if not apply_at_all_positions:
                    raise NotImplementedError("Only supports apply_at_all_positions=True for now.")
                steering_vec = steering_vec.view(1, -1)
                residuals = residuals + coefficient * steering_vec

            elif method == "adjust_rs":
                unit_vec = steering_vec  # use dir as unit vector (matches old behavior)
                avg_proj = data["avg_proj"].to(residuals.device, dtype=residuals.dtype)

                current_projections = torch.matmul(residuals, unit_vec)
                coeff = (avg_proj - current_projections).unsqueeze(-1)
                unit_vec = unit_vec.view(1, -1)

                residuals = residuals + coeff * unit_vec

            else:
                raise ValueError(f"Unknown steering method: {method}")

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
