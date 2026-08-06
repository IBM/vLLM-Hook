"""Per-position exit control and shared convergence dataclasses.

Shapes are ``[B, S]`` throughout so prefill can exit per position and decode
(``S == 1``) can exit per batch row. Large tensors (logits, attention, steer
directions) are only GPU-side (only scalars the analyzer needs are here).

Slicing for active tokens to avoid redundant forward passes isenabled for decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class ConvergenceState:
    """Cheap per-iteration signals produced by the worker.

    Absolute ``hidden_delta`` and ``h_norm`` drive the contraction-rate exit.
    ``rel_delta`` is logged / available to adapters for baseline comparisons.
    Optional fields stay ``None`` until enabled.
    """

    iteration: int
    hidden_delta: Tensor  # [B, S] ||x_t - x_{t-1}||
    h_norm: Tensor  # [B, S] ||x_t||
    rel_delta: Tensor  # [B, S] ||Δx|| / ||x||
    kl_divergence: Optional[Tensor] = None  # [B, S]
    colsum_concentration: Optional[Tensor] = None  # [B, S] ∈ [0, 1]
    safety_margin: Optional[Tensor] = None  # [B, S] Stage 2 only


@dataclass
class AnalyzerDecision:
    exit_mask: Tensor  # [B, S] bool — True → freeze this position
    steer_gate: Tensor  # [B, S] float — Stage 2 magnitude; 0 = no steer


class ExitController:
    """Per-position loop state shared across worker ↔ analyzer calls."""

    def __init__(self, B: int, S: int, device: torch.device):
        self.B = B
        self.S = S
        self.active = torch.ones(B, S, dtype=torch.bool, device=device)
        self.prev_delta: Optional[Tensor] = None  # [B, S] absolute ||Δx||
        self.M_initial: Optional[Tensor] = None  # Stage 2 drift reference
        self.nonconverging = torch.zeros(B, S, dtype=torch.bool, device=device)
        self.exit_iteration = torch.full((B, S), -1, dtype=torch.long, device=device)

    def apply(self, decision: AnalyzerDecision, iteration: int) -> Tensor:
        newly = decision.exit_mask & self.active
        self.exit_iteration[newly] = iteration
        self.active &= ~decision.exit_mask
        return self.active
