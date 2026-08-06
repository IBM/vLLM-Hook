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
        self._cached_direction: Optional[Tensor] = None

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
        self._cached_direction = None

    # ---------------- Stage-2 scaffolding, in progress (not used for exit) ----------------

    def margin_and_direction(self, x: Tensor):
        """``x: [B,S,D] → ([B,S] margin, [B,S,D] unit ascent direction)``."""
        assert self.w_margin is not None and self.ln_f is not None
        x_norm = self.ln_f(x)
        margin = x_norm @ self.w_margin
        g = self.w_margin.expand_as(x)
        direction = rmsnorm_vjp(g, x, self.ln_f.weight, self.eps)
        return margin, F.normalize(direction, dim=-1)

    # ---------------- optional metric (adapter must expose predict_from_latents) ----------------

    # NOTE: KL divergence requires coda + head and is thus more expensive to compute
    # than other metrics.This is adapter responsibility to expose
    # a predict_from_latents method (e.g. Raven).
    def compute_kl(self, latents: Tensor, aux_inputs: dict) -> Tensor:
        """Runs model-specific coda+head if available. Returns ``[B,S]``."""
        predict = getattr(self.model, "predict_from_latents", None)
        if predict is None:
            raise AttributeError(
                "compute_kl requires model.predict_from_latents (adapter responsibility)"
            )
        out = predict(latents, **aux_inputs)
        logprobs = F.log_softmax(out.logits.float(), dim=-1)
        if self._prev_logprobs is None:
            kl = torch.zeros(logprobs.shape[:2], device=logprobs.device, dtype=logprobs.dtype)
        else:
            kl = F.kl_div(
                self._prev_logprobs, logprobs, log_target=True, reduction="none"
            ).sum(-1)
        self._prev_logprobs = logprobs.detach()
        return kl

    # ---------------- assemble ConvergenceState for analyzer ----------------

    def build_state(
        self,
        x: Tensor,
        prev_x: Tensor,
        step: int,
        aux_inputs: Optional[dict] = None,
    ) -> ConvergenceState:
        """Build analyzer inputs. ``x`` / ``prev_x`` are pre-steering latents."""
        delta = (x - prev_x).norm(dim=-1)  # [B, S]
        h_norm = x.norm(dim=-1).clamp(min=1e-9)
        rel_delta = delta / h_norm

        margin = None
        if self.cfg.enable_steering and self.w_margin is not None:
            margin, direction = self.margin_and_direction(x)
            self._cached_direction = direction

        kl = None
        if self.cfg.compute_kl and aux_inputs is not None:
            kl = self.compute_kl(x, aux_inputs)

        return ConvergenceState(
            iteration=step,
            hidden_delta=delta,
            h_norm=h_norm,
            rel_delta=rel_delta,
            kl_divergence=kl,
            colsum_concentration=None,
            safety_margin=margin,
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
