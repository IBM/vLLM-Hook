"""GPU-side metrics for recurrent-depth adaptive exit.

Model-agnostic: large tensors stay on the worker; the analyzer only sees
``ConvergenceState`` scalars (``[B]`` / ``[B, S]``). Adapters direct how latents
are produced (``iterate_forward``, KV cache, native baselines).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from vllm_hook_plugins.protocols.exit_controller import AnalyzerDecision, ConvergenceState
from vllm_hook_plugins.protocols.recurrent_config import RecurrentDepthConfig


def rmsnorm_vjp(g: Tensor, x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Exact VJP of RMSNorm ``y = weight * x / rms(x)`` (float32 upcast).

    Used only for Stage-2 margin ascent; unused by exit. Adapters whose final
    norm differs should supply their own VJP.
    """
    xf, gf = x.float(), g.float()
    n = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    x_hat = xf * n
    gw = gf * weight.float()
    out = (gw - x_hat * (gw * x_hat).mean(-1, keepdim=True)) * n
    return out.type_as(x)


class RecurrentDepthWorker:
    """Computes per-iteration convergence metrics on GPU."""

    def __init__(self, model: Any, cfg: Optional[RecurrentDepthConfig] = None):
        self.model = model
        self.cfg = cfg or RecurrentDepthConfig()
        # Final norm + eps: Raven uses transformer.ln_f / norm_eps; other
        # families may expose equivalents — adapters can monkey-patch if needed.
        self.ln_f = getattr(getattr(model, "transformer", model), "ln_f", None)
        if self.ln_f is None:
            self.ln_f = getattr(getattr(model, "model", model), "norm", None)
        self.eps = float(
            getattr(getattr(model, "config", None), "norm_eps", None)
            or getattr(getattr(model, "config", None), "rms_norm_eps", 1e-6)
        )

        self.w_margin: Optional[Tensor] = None
        if self.cfg.enable_steering:
            self.w_margin = self._build_w_margin(model, self.cfg)

        self._prev_logprobs: Optional[Tensor] = None
        self._prev_entropy: Optional[Tensor] = None
        self._prev_top1: Optional[Tensor] = None
        self._top1_stability: Optional[Tensor] = None
        self._prev_delta_vector: Optional[Tensor] = None
        self._cached_direction: Optional[Tensor] = None
        self._forward_index = -1
        self._trajectory_samples: list[dict[str, Any]] = []

    @staticmethod
    def _build_w_margin(model, cfg: RecurrentDepthConfig) -> Tensor:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            raise ValueError("enable_steering requires model.lm_head")
        W_U = lm_head.weight  # [V, D]
        shared = set(cfg.refusal_ids) & set(cfg.harmful_ids)
        r_ids = [t for t in cfg.refusal_ids if t not in shared]
        a_ids = [t for t in cfg.harmful_ids if t not in shared]
        if not r_ids or not a_ids:
            raise ValueError(
                "enable_steering requires non-overlapping refusal_ids and harmful_ids"
            )
        return (W_U[r_ids].mean(0) - W_U[a_ids].mean(0)).detach().clone()

    def reset(self):
        self._prev_logprobs = None
        self._prev_entropy = None
        self._prev_top1 = None
        self._top1_stability = None
        self._prev_delta_vector = None
        self._cached_direction = None
        self._forward_index += 1

    def reset_trajectory_samples(self) -> None:
        self._trajectory_samples.clear()

    def pop_trajectory_samples(self) -> list[dict[str, Any]]:
        samples = list(self._trajectory_samples)
        self._trajectory_samples.clear()
        return samples

    # ---------------- Stage-2 scaffolding, in progress (not used for exit) ----------------

    def margin_and_direction(self, x: Tensor):
        """``x: [B,S,D] → ([B,S] margin, [B,S,D] unit ascent direction)``."""
        assert self.w_margin is not None and self.ln_f is not None
        x_norm = self.ln_f(x)
        margin = x_norm @ self.w_margin
        g = self.w_margin.expand_as(x)
        direction = rmsnorm_vjp(g, x, self.ln_f.weight, self.eps)
        return margin, F.normalize(direction, dim=-1)

    # ---------------- prediction metrics (adapter supplies latent projection) ----------------

    def compute_prediction_metrics(
        self,
        latents: Tensor,
        aux_inputs: dict,
    ) -> dict[str, Tensor]:
        """Compute output-space summaries from one shared coda/head projection."""
        predict_from_latents = getattr(self.model, "predict_from_latents", None)
        if predict_from_latents is None:
            raise AttributeError(
                "prediction metrics require model.predict_from_latents "
                "(adapter responsibility)"
            )
        out = predict_from_latents(latents, **aux_inputs)
        logits = out.logits.float()
        if logits.ndim == 2 and latents.ndim == 3 and latents.shape[1] == 1:
            logits = logits.unsqueeze(1)
        if logits.shape[:-1] != latents.shape[:-1]:
            raise ValueError(
                "predict_from_latents logits must align with latent rows: "
                f"got logits {tuple(logits.shape)} and latents "
                f"{tuple(latents.shape)}"
            )
        if logits.shape[-1] < 2:
            raise ValueError("prediction metrics require at least two logits")

        logprobs = F.log_softmax(logits, dim=-1)
        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(dim=-1)
        top_values, top_tokens = logits.topk(2, dim=-1)
        top1 = top_tokens[..., 0]
        confidence = probs.gather(-1, top1.unsqueeze(-1)).squeeze(-1)
        margin = top_values[..., 0] - top_values[..., 1]

        if self._prev_logprobs is None:
            kl = torch.zeros_like(entropy)
            entropy_delta = torch.zeros_like(entropy)
            stability = torch.zeros_like(top1)
        else:
            # torch.kl_div(input=current, target=previous) gives KL(prev || current).
            kl = F.kl_div(
                logprobs,
                self._prev_logprobs,
                log_target=True,
                reduction="none",
            ).sum(-1)
            assert self._prev_entropy is not None
            assert self._prev_top1 is not None
            assert self._top1_stability is not None
            entropy_delta = (entropy - self._prev_entropy).abs()
            stability = torch.where(
                top1 == self._prev_top1,
                self._top1_stability + 1,
                torch.zeros_like(self._top1_stability),
            )

        self._prev_logprobs = logprobs.detach()
        self._prev_entropy = entropy.detach()
        self._prev_top1 = top1.detach()
        self._top1_stability = stability.detach()
        return {
            "entropy": entropy,
            "entropy_delta": entropy_delta,
            "top1_confidence": confidence,
            "logit_margin": margin,
            "top1_token": top1,
            "top1_stability": stability,
            "predictive_kl": kl,
        }

    def compute_kl(self, latents: Tensor, aux_inputs: dict) -> Tensor:
        """Compatibility wrapper returning ``KL(previous || current)``."""
        return self.compute_prediction_metrics(latents, aux_inputs)["predictive_kl"]

    @staticmethod
    def _flat_metric_values(value: Tensor) -> Tensor:
        """Collapse ``[T]`` / ``[T, 1]`` / ``[B, S]`` metrics to a 1-D row vector."""
        flat = value.detach().reshape(-1)
        return flat

    def _record_trajectory(
        self,
        step: int,
        metrics: dict[str, Tensor],
        valid_metrics: dict[str, Tensor],
        phase_mask: Optional[Tensor],
    ) -> None:
        """Append flat per-token rows for offline parquet logging.

        Decode rows are the useful signal for adaptive-exit calibration, so when
        a phase mask is present we only keep decode tokens. Invalid metrics are
        stored as NaN (or -1 for integer token ids) instead of parallel bool maps.
        """
        flat_metrics = {
            name: self._flat_metric_values(value).float().cpu()
            for name, value in metrics.items()
            if name != "top1_token"
        }
        if "top1_token" in metrics:
            flat_metrics["top1_token"] = self._flat_metric_values(
                metrics["top1_token"]
            ).long().cpu()

        n_tokens = next(iter(flat_metrics.values())).numel()
        if phase_mask is None:
            keep = torch.ones(n_tokens, dtype=torch.bool)
            is_decode = keep
        else:
            is_decode = self._flat_metric_values(phase_mask).bool().cpu()
            if is_decode.numel() != n_tokens:
                raise ValueError(
                    "phase_mask length must match metric rows: "
                    f"{is_decode.numel()} vs {n_tokens}"
                )
            keep = is_decode
            if not bool(keep.any()):
                return

        flat_valid = {
            name: self._flat_metric_values(mask).bool().cpu()
            for name, mask in valid_metrics.items()
        }
        token_indices = keep.nonzero(as_tuple=False).view(-1).tolist()
        has_prediction = "entropy" in flat_metrics
        for token_index in token_indices:
            row: dict[str, Any] = {
                "forward_index": int(self._forward_index),
                "iteration": int(step),
                "token_index": int(token_index),
                "is_decode": bool(is_decode[token_index].item()),
                "has_prediction_metrics": has_prediction,
            }
            for name, values in flat_metrics.items():
                valid = flat_valid.get(name)
                ok = True if valid is None else bool(valid[token_index].item())
                if not ok:
                    row[name] = None if name != "top1_token" else -1
                    continue
                item = values[token_index].item()
                row[name] = int(item) if name == "top1_token" else float(item)
            self._trajectory_samples.append(row)

    # ---------------- assemble ConvergenceState for analyzer ----------------

    def build_state(
        self,
        x: Tensor,
        prev_x: Tensor,
        step: int,
        aux_inputs: Optional[dict] = None,
        phase_mask: Optional[Tensor] = None,
    ) -> ConvergenceState:
        """Build analyzer inputs. ``x`` / ``prev_x`` are pre-steering latents."""
        delta_vector = (x - prev_x).float()
        delta = delta_vector.norm(dim=-1)  # [B, S]
        h_norm = x.float().norm(dim=-1).clamp(min=1e-9)
        rel_delta = delta / h_norm
        cosine_distance = 1.0 - F.cosine_similarity(
            x.float(),
            prev_x.float(),
            dim=-1,
            eps=1e-9,
        )
        previous_delta_vector = self._prev_delta_vector
        acceleration_valid = previous_delta_vector is not None
        if previous_delta_vector is None:
            acceleration = torch.zeros_like(delta)
        else:
            previous_speed = previous_delta_vector.norm(dim=-1)
            acceleration = (
                delta_vector - previous_delta_vector
            ).norm(dim=-1) / (delta + previous_speed).clamp(min=1e-9)
        self._prev_delta_vector = delta_vector.detach()

        metrics = {
            "hidden_delta": delta,
            "hidden_norm": h_norm,
            "normalized_displacement": rel_delta,
            "cosine_distance": cosine_distance,
            "normalized_acceleration": acceleration,
        }
        valid = torch.ones_like(delta, dtype=torch.bool)
        valid_metrics = {name: valid for name in metrics}
        valid_metrics["normalized_acceleration"] = torch.full_like(
            valid,
            acceleration_valid,
        )

        margin = None
        if self.cfg.enable_steering and self.w_margin is not None:
            margin, direction = self.margin_and_direction(x)
            self._cached_direction = direction
            metrics["safety_margin"] = margin
            valid_metrics["safety_margin"] = valid

        prediction_requested = self.cfg.needs_prediction_metrics()
        # vLLM's logits processor may select only sampled rows during prefill.
        # Until adapters provide an explicit logits-to-latent row map, compute
        # output-space metrics only when every flattened row is decode.
        prediction_rows_aligned = (
            phase_mask is None or bool(phase_mask.all())
        )
        if prediction_requested and prediction_rows_aligned:
            if aux_inputs is None:
                raise ValueError(
                    "prediction metrics require adapter prediction inputs"
                )
            had_previous_prediction = self._prev_logprobs is not None
            prediction_metrics = self.compute_prediction_metrics(x, aux_inputs)
            metrics.update(prediction_metrics)
            for name in prediction_metrics:
                valid_metrics[name] = valid
            for name in ("entropy_delta", "predictive_kl"):
                valid_metrics[name] = torch.full_like(
                    valid,
                    had_previous_prediction,
                )

        if self.cfg.capture_trajectory:
            self._record_trajectory(
                step,
                metrics,
                valid_metrics,
                phase_mask,
            )

        return ConvergenceState(
            iteration=step,
            metrics=metrics,
            valid_metrics=valid_metrics,
            eligible_mask=phase_mask,
        )

    def apply_steering(self, x: Tensor, decision: AnalyzerDecision) -> Tensor:
        """Sparse: ``steer_gate`` is 0 for nearly all positions in Stage 1."""
        if (
            not self.cfg.enable_steering
            or self._cached_direction is None
            or decision.steer_gate is None
        ):
            return x
        if not torch.any(decision.steer_gate != 0):
            return x
        return x + self._cached_direction * decision.steer_gate.unsqueeze(-1)
