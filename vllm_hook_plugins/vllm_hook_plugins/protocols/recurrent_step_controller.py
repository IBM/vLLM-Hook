"""Per-step exit/steer controller for recurrent-depth executors.

This is the single object a model's recurrence loop talks to. It wraps the
existing Hook stack unchanged:

    RecurrentDepthWorker.build_state  ->  RecurrentConvergenceAnalyzer.analyze
                                      ->  ExitController.apply

and exposes one ``step`` call to run at the end of every recurrence iteration.

Shape bridging (the only new concern vs. the HF adapter):
    vLLM executors operate on **flattened** hidden states ``[T, D]`` (all
    batched tokens collapsed into one dim). The shared worker/analyzer/
    ExitController are written for ``[B, S]`` metrics / ``[B, S, D]`` latents.
    We treat each flattened token as an independent row (``B = T``, ``S = 1``)
    with a single ``unsqueeze(1)`` / ``squeeze(1)``. The HF path keeps passing
    real ``[B, S, D]`` and is unaffected.

Stage 1 uses the training-free contraction criterion only. Steering
(``steer``) is a no-op until Stage 2 enables it via the config.

vLLM executors should construct this via :meth:`from_config` so a
``recurrent_depth`` dict (including import-string ``worker`` / ``analyzer``)
can be forwarded from ``HookLLM`` / ``hf_overrides`` into the GPU worker.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from vllm_hook_plugins.analyzers.recurrent_conv_analyzer import RecurrentConvergenceAnalyzer
from vllm_hook_plugins.protocols.exit_controller import AnalyzerDecision, ExitController
from vllm_hook_plugins.protocols.recurrent_config import (
    RecurrentDepthConfig,
    build_recurrent_stack,
)
from vllm_hook_plugins.workers.recurrent_depth_worker import RecurrentDepthWorker


class RecurrentStepController:
    """Drive per-token adaptive exit for one flattened recurrence loop.

    Typical use inside a vLLM executor ``forward``::

        active = self.controller.reset(hidden_states.shape[0], hidden_states.device)
        for step in range(max_recurrence):
            prev = hidden_states
            h = run_core_block(hidden_states)          # [T, D]
            decision = self.controller.step(h, prev, step)
            hidden_states = self.controller.steer(h, decision)  # no-op Stage 1
            active = self.controller.apply(decision, step)      # [T] bool
            if not active.any():
                break
    """

    def __init__(
        self,
        model: Any,
        cfg: Optional[RecurrentDepthConfig] = None,
        *,
        worker: Optional[RecurrentDepthWorker] = None,
        analyzer: Optional[RecurrentConvergenceAnalyzer] = None,
    ) -> None:
        self.cfg = cfg or RecurrentDepthConfig()
        self.worker = worker or RecurrentDepthWorker(model, self.cfg)
        self.analyzer = analyzer or RecurrentConvergenceAnalyzer(self.cfg)
        self.ctrl: Optional[ExitController] = None

    @classmethod
    def from_config(
        cls,
        model: Any,
        recur_dict: Optional[dict] = None,
        *,
        worker: Any = None,
        analyzer: Any = None,
    ) -> "RecurrentStepController":
        """Build a controller from a ``recurrent_depth`` dict.

        Typically arrives via ``HookLLM(..., hf_overrides={"recurrent_depth": ...})``.

        ``worker`` / ``analyzer`` keys may be import strings (``pkg.mod:Class``),
        classes, or instances. Omitted → stock :class:`RecurrentDepthWorker` /
        :class:`RecurrentConvergenceAnalyzer`.
        """
        cfg, worker_obj, analyzer_obj = build_recurrent_stack(
            model, recur_dict, worker=worker, analyzer=analyzer
        )
        return cls(model, cfg, worker=worker_obj, analyzer=analyzer_obj)

    # ------------------------------------------------------------------ #
    # Per-forward lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, num_tokens: int, device: torch.device) -> Tensor:
        """Begin a new forward pass; return the initial ``[T]`` active mask."""
        self.worker.reset()
        self.ctrl = ExitController(num_tokens, 1, device)
        return self.ctrl.active.squeeze(1)

    # ------------------------------------------------------------------ #
    # End-of-step protocol call
    # ------------------------------------------------------------------ #

    def step(self, hidden_states: Tensor, prev_hidden_states: Tensor, recurrent_step: int) -> AnalyzerDecision:
        """Run worker metrics + analyzer for one step.

        ``hidden_states`` / ``prev_hidden_states`` are pre-steering flattened latents ``[T, D]``.
        Returns an :class:`AnalyzerDecision` whose masks are ``[T, 1]``.
        """
        if self.ctrl is None:
            raise RuntimeError("RecurrentStepController.reset() must precede step().")
        state = self.worker.build_state(
            hidden_states.unsqueeze(1), prev_hidden_states.unsqueeze(1), recurrent_step
        )
        return self.analyzer.analyze(state, self.ctrl)

    def apply(self, decision: AnalyzerDecision, recurrent_step: int) -> Tensor:
        """Record exits and return the updated ``[T]`` active mask."""
        assert self.ctrl is not None
        return self.ctrl.apply(decision, recurrent_step).squeeze(1)

    def steer(self, x: Tensor, decision: AnalyzerDecision) -> Tensor:
        """Stage-2 steering. No-op when steering disabled (returns ``x``).

        ``x``: flattened ``[T, D]``. ``decision.steer_gate``: ``[T, 1]``. The
        worker caches the unit direction as ``[T, 1, D]`` during ``build_state``.
        """
        direction = getattr(self.worker, "_cached_direction", None)
        gate = decision.steer_gate
        if not self.cfg.enable_steering or direction is None or gate is None:
            return x
        if not torch.any(gate != 0):
            return x
        return x + gate * direction.squeeze(1)

    # ------------------------------------------------------------------ #
    # Diagnostics (flattened [T] views)
    # ------------------------------------------------------------------ #

    @property
    def active(self) -> Optional[Tensor]:
        return None if self.ctrl is None else self.ctrl.active.squeeze(1)

    @property
    def exit_iteration(self) -> Optional[Tensor]:
        return None if self.ctrl is None else self.ctrl.exit_iteration.squeeze(1)

    @property
    def nonconverging(self) -> Optional[Tensor]:
        return None if self.ctrl is None else self.ctrl.nonconverging.squeeze(1)
