"""Model-neutral metric policies for recurrent-depth adaptive exit."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from torch import Tensor

from vllm_hook_plugins.protocols.exit_controller import (
    AnalyzerDecision,
    ConvergenceState,
    ExitController,
)
from vllm_hook_plugins.protocols.recurrent_config import RecurrentDepthConfig
from vllm_hook_plugins.protocols.recurrent_policy import (
    READOUT_METRICS,
    MetricCondition,
    Readout,
)


class RecurrentConvergenceAnalyzer:
    """Evaluate calibrated single or composite per-position exit policies."""

    def __init__(self, cfg: Optional[RecurrentDepthConfig] = None):
        self.cfg = cfg or RecurrentDepthConfig()
        self.policy = self.cfg.resolved_policy()

    def analyze(self, state: ConvergenceState, ctrl: ExitController) -> AnalyzerDecision:
        eligible = ctrl.active
        if state.eligible_mask is not None:
            eligible = eligible & state.eligible_mask

        hit = torch.zeros_like(ctrl.active)
        if (
            self.policy.enabled
            and state.iteration >= self.policy.min_steps
            and bool(eligible.any())
        ):
            hit = self._evaluate_group(
                state,
                ctrl,
                self.policy.conditions,
                self.policy.combine,
            )
            if self.policy.confirmation_conditions:
                hit &= self._evaluate_group(
                    state,
                    ctrl,
                    self.policy.confirmation_conditions,
                    self.policy.confirmation_combine,
                )
            hit &= eligible

        ctrl.hit_count = torch.where(
            hit,
            ctrl.hit_count + 1,
            torch.zeros_like(ctrl.hit_count),
        )
        exit_mask = hit & (ctrl.hit_count >= self.policy.patience)

        # Advance after the decision so contraction sees the preceding delta.
        ctrl.prev_delta = state.hidden_delta.detach()
        steer_gate = torch.zeros_like(state.hidden_delta)
        return AnalyzerDecision(exit_mask=exit_mask, steer_gate=steer_gate)

    def _evaluate_group(
        self,
        state: ConvergenceState,
        ctrl: ExitController,
        conditions: Sequence[MetricCondition],
        combine: Literal["all", "any"],
    ) -> Tensor:
        result = (
            torch.ones_like(ctrl.active)
            if combine == "all"
            else torch.zeros_like(ctrl.active)
        )
        for condition in conditions:
            condition_hit = self._evaluate_condition(state, ctrl, condition)
            if combine == "all":
                result &= condition_hit
            else:
                result |= condition_hit
        return result

    def _evaluate_condition(
        self,
        state: ConvergenceState,
        ctrl: ExitController,
        condition: MetricCondition,
    ) -> Tensor:
        if condition.readout is Readout.CONTRACTION:
            return self._contraction_hit(state, ctrl, condition.threshold)

        try:
            metric_name, comparison = READOUT_METRICS[condition.readout]
        except KeyError as exc:
            raise ValueError(
                f"unsupported recurrent readout {condition.readout.value!r}"
            ) from exc
        value = state.require_metric(metric_name)
        valid = state.metric_valid(metric_name)
        if comparison == "lt":
            return (value < condition.threshold) & valid & ctrl.active
        return (value >= condition.threshold) & valid & ctrl.active

    def _contraction_hit(
        self,
        state: ConvergenceState,
        ctrl: ExitController,
        threshold: float,
    ) -> Tensor:
        if ctrl.prev_delta is None:
            return torch.zeros_like(ctrl.active)

        r_hat = state.hidden_delta / ctrl.prev_delta.clamp(min=1e-9)
        converging = r_hat < 1.0
        ctrl.nonconverging |= (~converging) & ctrl.active

        r_safe = r_hat.clamp(max=0.999)
        remaining = state.hidden_delta * r_safe / (1.0 - r_safe)
        rel_remaining = remaining / state.h_norm.clamp(min=1e-9)
        return (rel_remaining < threshold) & converging & ctrl.active
