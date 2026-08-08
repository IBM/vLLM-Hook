"""Huginn / Raven native exit baselines for A/B vs contraction.

These reproduce ``generate_with_adaptive_compute`` criteria for the Raven models
at per-position ``[B, S]`` granularity. They are Raven-family specific and
are therefore not in the shared vLLM-Hook analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from vllm_hook_plugins.protocols.exit_controller import (
    AnalyzerDecision,
    ConvergenceState,
    ExitController,
)

HUGINN_CRITERIA = (
    "latent-diff",
    "kl",
    "minp-kl",
    "entropy-diff",
    "argmax-stability",
    "none",
)

_AUTO_THRESHOLD = {
    "latent-diff": 0.03,
    "entropy-diff": 1e-3,
    "kl": 5e-4,
    "minp-kl": 1e-6,
    "argmax-stability": 5.0,
    "none": 1.0,
}


@dataclass
class RavenBaselineConfig:
    """Comparison-arm knobs mirroring Huginn ``generate_with_adaptive_compute``."""

    criterion: str = "latent-diff"
    exit_threshold: Optional[float] = None  # None → Huginn "auto"
    min_steps: int = 1

    def __post_init__(self):
        if self.criterion not in HUGINN_CRITERIA:
            raise ValueError(
                f"Unknown Huginn criterion {self.criterion!r}; expected {HUGINN_CRITERIA}"
            )

    @property
    def threshold(self) -> float:
        if self.exit_threshold is not None:
            return float(self.exit_threshold)
        return float(_AUTO_THRESHOLD[self.criterion])

    @property
    def needs_logits(self) -> bool:
        return self.criterion in ("kl", "minp-kl", "entropy-diff", "argmax-stability")


class RavenBaselineExit:
    """Per-position Huginn baseline exit evaluator."""

    def __init__(self, cfg: Optional[RavenBaselineConfig] = None):
        self.cfg = cfg or RavenBaselineConfig()
        self.prev_entropy: Optional[Tensor] = None
        self.prev_argmax: Optional[Tensor] = None
        self.stable_steps: Optional[Tensor] = None
        self._prev_logprobs: Optional[Tensor] = None

    def reset(self):
        self.prev_entropy = None
        self.prev_argmax = None
        self.stable_steps = None
        self._prev_logprobs = None

    def analyze(
        self,
        state: ConvergenceState,
        ctrl: ExitController,
        *,
        logits: Optional[Tensor] = None,
    ) -> AnalyzerDecision:
        device = state.rel_delta.device
        B, S = state.rel_delta.shape
        exit_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        thr = self.cfg.threshold

        if state.iteration >= self.cfg.min_steps:
            c = self.cfg.criterion
            if c == "none":
                pass
            elif c == "latent-diff":
                exit_mask = (state.rel_delta < thr) & ctrl.active
            elif c in ("kl", "minp-kl"):
                exit_mask = self._kl_exit(state, ctrl, logits, thr, minp=(c == "minp-kl"))
            elif c == "entropy-diff":
                exit_mask = self._entropy_exit(ctrl, logits, thr)
            elif c == "argmax-stability":
                exit_mask = self._argmax_exit(ctrl, logits, thr)

        # Keep contraction's prev_delta warm if both arms share a controller.
        ctrl.prev_delta = state.hidden_delta.detach()
        gate = torch.zeros(B, S, dtype=state.rel_delta.dtype, device=device)
        return AnalyzerDecision(exit_mask=exit_mask, steer_gate=gate)

    def _kl_exit(
        self,
        state: ConvergenceState,
        ctrl: ExitController,
        logits: Optional[Tensor],
        thr: float,
        *,
        minp: bool,
    ) -> Tensor:
        if state.kl_divergence is not None:
            kl = state.kl_divergence
        elif logits is not None:
            kl = self._kl_from_logits(logits, minp=minp)
        else:
            return torch.zeros_like(ctrl.active)
        if not (kl > 0).any() and state.iteration <= self.cfg.min_steps:
            return torch.zeros_like(ctrl.active)
        return (kl < thr) & ctrl.active

    def _kl_from_logits(self, logits: Tensor, *, minp: bool) -> Tensor:
        # logits: [B, S, V]
        if minp:
            probs = F.softmax(logits.float(), dim=-1)
            max_p = probs.max(dim=-1, keepdim=True).values
            masked = probs.clone()
            masked[probs < 0.1 * max_p] = 1.0 / probs.shape[-1]
            logprobs = (masked / masked.sum(-1, keepdim=True)).log()
        else:
            logprobs = F.log_softmax(logits.float(), dim=-1)
        if self._prev_logprobs is None:
            kl = torch.zeros(logprobs.shape[:2], device=logprobs.device, dtype=logprobs.dtype)
        else:
            kl = F.kl_div(
                logprobs, self._prev_logprobs, log_target=True, reduction="none"
            ).sum(-1)
        self._prev_logprobs = logprobs.detach()
        return kl

    def _entropy_exit(
        self, ctrl: ExitController, logits: Optional[Tensor], thr: float
    ) -> Tensor:
        if logits is None:
            return torch.zeros_like(ctrl.active)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(logprobs.exp() * logprobs).sum(-1)
        if self.prev_entropy is None:
            self.prev_entropy = entropy.detach()
            return torch.zeros_like(ctrl.active)
        diff = (entropy - self.prev_entropy).abs()
        self.prev_entropy = entropy.detach()
        return (diff < thr) & ctrl.active

    def _argmax_exit(
        self, ctrl: ExitController, logits: Optional[Tensor], thr: float
    ) -> Tensor:
        if logits is None:
            return torch.zeros_like(ctrl.active)
        argmax = logits.argmax(dim=-1)
        if self.prev_argmax is None:
            self.prev_argmax = argmax.detach()
            self.stable_steps = torch.zeros_like(ctrl.exit_iteration)
            return torch.zeros_like(ctrl.active)
        same = argmax == self.prev_argmax
        self.stable_steps = torch.where(
            same, self.stable_steps + 1, torch.zeros_like(self.stable_steps)
        )
        self.prev_argmax = argmax.detach()
        return (self.stable_steps >= int(thr)) & ctrl.active
