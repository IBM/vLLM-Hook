"""Convergence analyzer for recurrent-depth adaptive exit.

Model-agnostic: contraction-rate exit only. Family-specific baselines
(Huginn latent-diff / KL / …) live in the corresponding adapter.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from vllm_hook_plugins.protocols.exit_controller import (
    AnalyzerDecision,
    ConvergenceState,
    ExitController,
)
from vllm_hook_plugins.protocols.recurrent_config import RecurrentDepthConfig


class RecurrentConvergenceAnalyzer:
    """Per-position exit via predictive contraction rate::

        r̂ = ‖Δx_t‖ / ‖Δx_{t-1}‖          # consecutive steps, not vs first
        remaining ≈ ‖Δx_t‖ · r̂ / (1 − r̂)
        exit when remaining / ‖x_t‖ < ρ  AND  r̂ < 1

    ``r̂ ≥ 1`` marks non-converging orbit/slider tokens (discussed in papers
    like Blayney et al. 2024, Han et al. 2025) for logging.
    Native Huginn criteria are available for A/B comparison at per-position grain.
    Stage 1 (exit) only reads convergence signals (not margin, which is used only for steering).
    """

    def __init__(self, cfg: Optional[RecurrentDepthConfig] = None):
        self.cfg = cfg or RecurrentDepthConfig()
        self.rho = float(self.cfg.rho)
        self.min_steps = int(self.cfg.min_steps)

    def analyze(self, state: ConvergenceState, ctrl: ExitController) -> AnalyzerDecision:
        device = state.hidden_delta.device
        B, S = state.hidden_delta.shape
        exit_mask = torch.zeros(B, S, dtype=torch.bool, device=device)

        if state.iteration >= self.min_steps:
            exit_mask = self._contraction_exit(state, ctrl)

        # Advance prev_delta after the decision so the next r̂ uses ‖Δx_{t-1}‖.
        ctrl.prev_delta = state.hidden_delta.detach()

        steer_gate = torch.zeros(B, S, dtype=state.hidden_delta.dtype, device=device)
        return AnalyzerDecision(exit_mask=exit_mask, steer_gate=steer_gate)

    def _contraction_exit(self, state: ConvergenceState, ctrl: ExitController) -> Tensor:
        if ctrl.prev_delta is None:
            return torch.zeros_like(ctrl.active)

        r_hat = state.hidden_delta / ctrl.prev_delta.clamp(min=1e-9)
        converging = r_hat < 1.0
        ctrl.nonconverging |= (~converging) & ctrl.active

        r_safe = r_hat.clamp(max=0.999)
        remaining = state.hidden_delta * r_safe / (1.0 - r_safe)
        rel_remaining = remaining / state.h_norm.clamp(min=1e-9)

        # ρ = 0 → never exit (exact-match validation).
        if self.rho <= 0.0:
            return torch.zeros_like(ctrl.active)

        return (rel_remaining < self.rho) & converging & ctrl.active
