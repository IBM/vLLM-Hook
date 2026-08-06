"""In-process recurrent-depth protocol (model-agnostic).

Wires a GPU worker + contraction analyzer onto any adapter that exposes
``attach_hook_stack(worker, analyzer, cfg, ...)``. Model-specific forward /
cache / baseline-exit details stay in the adapter. This protocol allows
for flexible model-specific wiring with general-purpose scaffolding for recurrence
within vLLM-Hook.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from vllm_hook_plugins.analyzers.recurrent_conv_analyzer import RecurrentConvergenceAnalyzer
from vllm_hook_plugins.protocols.exit_controller import ConvergenceState
from vllm_hook_plugins.protocols.recurrent_config import RecurrentDepthConfig
from vllm_hook_plugins.workers.recurrent_depth_worker import RecurrentDepthWorker


class RecurrentDepthProtocol:
    """Attach / query the adaptive-exit stack on a recurrent-depth adapter."""

    def __init__(
        self,
        model: Any,
        cfg: Optional[RecurrentDepthConfig] = None,
        *,
        worker: Optional[RecurrentDepthWorker] = None,
        analyzer: Optional[RecurrentConvergenceAnalyzer] = None,
        hook: Optional[Callable] = None,
        attach_kwargs: Optional[dict] = None,
    ):
        self.cfg = cfg or RecurrentDepthConfig()
        self.model = model
        self.worker = worker or RecurrentDepthWorker(model, self.cfg)
        self.analyzer = analyzer or RecurrentConvergenceAnalyzer(self.cfg)
        self._iteration_hook = hook

        if not hasattr(model, "attach_hook_stack"):
            raise TypeError(
                f"{type(model).__name__} has no attach_hook_stack; "
                "use a recurrent-depth model adapter"
            )
        model.attach_hook_stack(
            self.worker, self.analyzer, self.cfg, **(attach_kwargs or {})
        )

    def register_iteration_hook(
        self,
        hook: Callable[[int, torch.Tensor, torch.Tensor], Optional[torch.Tensor]],
    ):
        """Optional callback ``(iteration, hidden_states, active_mask) → maybe new x``."""
        self._iteration_hook = hook

    def set_aux_inputs(self, aux: Optional[dict]):
        """Adapter-specific auxiliaries (e.g. Raven predict_from_latents kwargs)."""
        self.model._aux_inputs = aux

    def get_convergence_state(self) -> Optional[ConvergenceState]:
        return getattr(self.model, "last_convergence_state", None)

    @property
    def last_exit_iteration(self):
        return getattr(self.model, "last_exit_iteration", None)

    @property
    def last_nonconverging(self):
        return getattr(self.model, "last_nonconverging", None)


def attach_recurrent_depth(
    model,
    cfg: Optional[RecurrentDepthConfig] = None,
    *,
    raven_cfg=None,
    **cfg_kwargs,
) -> RecurrentDepthProtocol:
    """Attach the shared stack. Pass ``raven_cfg=RavenAdapterConfig(...)`` for Raven."""
    if cfg is None and cfg_kwargs:
        cfg = RecurrentDepthConfig(**cfg_kwargs)
    attach_kwargs = {}
    if raven_cfg is not None:
        attach_kwargs["raven_cfg"] = raven_cfg
    return RecurrentDepthProtocol(model, cfg=cfg, attach_kwargs=attach_kwargs)
