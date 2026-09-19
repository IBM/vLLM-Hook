"""Per-position exit control and shared convergence dataclasses.

Shapes are ``[B, S]`` throughout so prefill can exit per position and decode
(``S == 1``) can exit per batch row. Large tensors (logits, attention, steer
directions) are only GPU-side (only scalars the analyzer needs are here).

Slicing for active tokens to avoid redundant forward passes isenabled for decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class ConvergenceState:
    """Single-source per-iteration metric values produced by the worker."""

    iteration: int
    metrics: Dict[str, Tensor]
    valid_metrics: Dict[str, Tensor] = field(default_factory=dict)
    eligible_mask: Optional[Tensor] = None

    def require_metric(self, name: str) -> Tensor:
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise ValueError(f"required recurrent metric {name!r} is unavailable") from exc

    def metric_valid(self, name: str) -> Tensor:
        value = self.require_metric(name)
        return self.valid_metrics.get(name, torch.ones_like(value, dtype=torch.bool))

    @property
    def hidden_delta(self) -> Tensor:
        return self.require_metric("hidden_delta")

    @property
    def h_norm(self) -> Tensor:
        return self.require_metric("hidden_norm")

    @property
    def rel_delta(self) -> Tensor:
        return self.require_metric("normalized_displacement")

    @property
    def kl_divergence(self) -> Optional[Tensor]:
        return self.metrics.get("predictive_kl")


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
        self.hit_count = torch.zeros(B, S, dtype=torch.long, device=device)

    def apply(self, decision: AnalyzerDecision, iteration: int) -> Tensor:
        newly = decision.exit_mask & self.active
        self.exit_iteration[newly] = iteration
        self.active &= ~decision.exit_mask
        return self.active
