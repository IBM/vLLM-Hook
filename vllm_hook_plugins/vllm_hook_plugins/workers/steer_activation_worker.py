import os
import torch
from typing import TYPE_CHECKING, Any, Dict

from vllm.forward_context import get_forward_context
from vllm_hook_plugins.workers._common import (
    get_query_metadata,
    iter_matched_modules,
    match_layer,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


def _load_steering_vector(vector_path: str) -> Dict:
    """Load and parse a steering vector .pt file. Returns the raw dict."""
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found at: {vector_path}")
    return torch.load(vector_path, weights_only=False)


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

        def _apply_steer(slice_view: torch.Tensor, cfg: Dict) -> torch.Tensor:
            """Return the steered residual slice for one request."""
            method = cfg.get("method", "adjust_rs")
            coefficient = float(cfg.get("coefficient", 0))

            # Cache the loaded vector data per path.
            vector_path = cfg["vector_path"]
            data = self._vector_cache.get(vector_path)
            if data is None:
                raw = _load_steering_vector(vector_path)
                data = {"dir": torch.tensor(raw["dir"])}
                if method == "adjust_rs":
                    data["avg_proj"] = raw["avg_proj"]
                self._vector_cache[vector_path] = data

            steering_vec = data["dir"].to(slice_view.device, dtype=slice_view.dtype)
            target = slice_view[-1:]  # (1, hidden)

            if method == "add_vector":
                steered = target + coefficient * steering_vec.view(1, -1)
            elif method == "adjust_rs":
                unit_vec = steering_vec  # use dir as unit vector (matches old behavior)
                avg_proj = data["avg_proj"].to(target.device, dtype=target.dtype)
                current_projections = torch.matmul(target, unit_vec)
                coeff = (avg_proj - current_projections).unsqueeze(-1)
                steered = target + coeff * unit_vec.view(1, -1)
            else:
                raise ValueError(f"Unknown steering method: {method}")
            # stitch the steered last row back
            out = slice_view.clone()
            out[-1:] = steered
            return out

        def steering_hook(input, output, layer_num: int):
            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)

            # Warmup or non-attention passes: nothing to do
            if metadata is None:
                return output
            if torch.cuda.is_current_stream_capturing():
                return output

            query_start_loc, seq_lens = get_query_metadata(metadata)
            if query_start_loc is None:
                return output

            bs = len(query_start_loc) - 1
            last_indices = query_start_loc

            try:
                req_ids = self.model_runner.input_batch.req_ids
            except Exception:
                return output

            is_tuple = isinstance(output, tuple)
            if is_tuple:
                hidden_states, residuals = output
            else:
                hidden_states = None
                residuals = output

            # Per-request slicing — non-steered rows keep their original values
            # because we only write to [start:end] for matching requests.
            steered_any = False
            for i in range(bs):
                req_id = req_ids[i]
                req_state = self.model_runner.requests.get(req_id)
                if req_state is None or req_state.sampling_params is None:
                    continue
                extra = req_state.sampling_params.extra_args
                if not extra:
                    continue

                cfg = extra.get("steer")
                if not isinstance(cfg, dict):
                    continue
                if int(cfg.get("optimal_layer", -1)) != layer_num:
                    continue

                # if no apply_at_all_positions, then no-ops for decode phase
                if not bool(cfg.get("apply_at_all_positions", True)):
                    is_prefill = len(req_state.output_token_ids) == 0
                    if not is_prefill:
                        continue

                start = int(last_indices[i].item())
                end = int(last_indices[i + 1].item())
                residuals[start:end] = _apply_steer(residuals[start:end], cfg)
                steered_any = True

            if not steered_any:
                return output
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
