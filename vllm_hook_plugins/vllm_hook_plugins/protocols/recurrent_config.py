"""Shared knobs for the model-agnostic recurrent-depth stack.

Model-specific options (Huginn baseline criteria, decode row-slicing, cache
lookup strategy) live on the adapter, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecurrentDepthConfig:
    """Knobs for :class:`RecurrentDepthWorker` / :class:`RecurrentConvergenceAnalyzer`.

    Exit uses the training-free contraction criterion only. Safety steering is
    independent (Stage 2) and optional.
    """

    # Contraction-rate relative remaining threshold (RECUR_DEPTH ρ).
    # ρ = 0 → nothing exits (exact-match for baseline looped Raven).
    rho: float = 0.0
    min_steps: int = 1  # never exit before this many recurrence steps
    compute_kl: bool = False  # opt-in metric; adapter supplies predict_from_latents
    compute_colsum: bool = False  # Stage 2 (steering); needs Q/K capture

    # ---- Stage 2 scaffolding (optional; unused for exit) ----
    enable_steering: bool = False
    refusal_ids: List[int] = field(default_factory=list)
    harmful_ids: List[int] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "RecurrentDepthConfig":
        if not d:
            return RecurrentDepthConfig()
        known = {f.name for f in RecurrentDepthConfig.__dataclass_fields__.values()}
        return RecurrentDepthConfig(**{k: v for k, v in d.items() if k in known})
