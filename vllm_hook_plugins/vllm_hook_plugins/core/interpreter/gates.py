"""Per-request gate state: pooled condition-layer readings, a readout that
turns each pooled row into one value, and a rule that decides over them.

Evidence is per-position hidden rows kept in float32 on CPU and pooled at
freeze, so nonlinear readouts see the same pooled vector any external
implementation would. Rows are keyed by absolute position and overwrite on
replay, so re-executed passes (preemption recompute, the post-stop extra
pass) are idempotent. The decision freezes once, at the first pass covering
the final prompt position with that position's readings complete, and holds
for the request; the stored rows are dropped at freeze. A gate whose request
ends before the prompt does never freezes and frees its rows with the
request. Instances hold no engine state: the worker feeds them
condition-layer rows and pass coverage and asks for a decision.
"""
from __future__ import annotations

import torch

from vllm_hook_plugins.core.kinds import ARTIFACT_TENSORS

_EPS = 1e-8


def affine(pooled: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Signed linear score ``weights . pooled``."""
    return weights @ pooled


def cosine(pooled: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Signed cosine similarity between the pooled row and ``direction``."""
    denominator = pooled.norm() * direction.norm() + _EPS
    return (pooled @ direction) / denominator


def projected_cosine(pooled: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Cosine between the pooled row and its rank-one projection through
    ``direction``, squashed by tanh: ``cos(pooled, tanh(pooled @ P))`` with
    ``P = direction direction^T / (direction . direction + eps)``.
    """
    projector = torch.outer(direction, direction) / (direction @ direction + _EPS)
    projected = torch.tanh(pooled @ projector)
    denominator = pooled.norm() * projected.norm() + _EPS
    return (pooled @ projected) / denominator


READOUTS = {
    "affine": affine,
    "cosine": cosine,
    "projected_cosine": projected_cosine,
}


def sum_threshold(values: dict, *, bias: float) -> bool:
    """Open where the summed per-layer values plus ``bias`` are >= 0 (ties
    open).
    """
    return sum(values.values()) + bias >= 0


def per_key_threshold(values: dict, *, threshold: float, comparator: str, aggregate: str) -> bool:
    """Compare each per-layer value against ``threshold`` (``"ge"``:
    value >= threshold; ``"le"``: value <= threshold), then combine across
    layers with ``"any"`` or ``"all"``.
    """
    if comparator == "ge":
        passed = [value >= threshold for value in values.values()]
    else:
        passed = [value <= threshold for value in values.values()]
    return all(passed) if aggregate == "all" else any(passed)


RULES = {
    "sum_threshold": sum_threshold,
    "per_key_threshold": per_key_threshold,
}


class GateState:
    """One instance per (request, gated op).

    ``observe(layer, positions, stream_rows)`` ingests condition-layer
    readings; ``note_pass(positions, prompt_len)`` reports each pass's
    coverage and freezes the decision when the trigger is met; ``decision()``
    returns True/False/None (None = undecided); ``reset()`` clears rows and
    the held decision after a request restart.
    """

    def __init__(self, *, layers, pooling: str, readout_kind: str, readout_tensor: torch.Tensor,
                 rule_kind: str, rule_params: dict):
        self.layers = [int(layer) for layer in layers]
        self._row_for = {layer: index for index, layer in enumerate(self.layers)}
        self.pooling = pooling
        self.readout_kind = readout_kind
        self.readout_tensor = readout_tensor.detach().float()
        self.rule_kind = rule_kind
        self.rule_params = dict(rule_params)
        self._rows: dict = {}  # (layer, position) -> cpu float32 row
        self._held: bool | None = None

    def observe(self, layer: int, positions: range, stream_rows: torch.Tensor) -> None:
        if layer not in self._row_for or self._held is not None:
            return
        rows = stream_rows.detach().to("cpu", copy=True).float()
        for index, position in enumerate(positions):
            self._rows[(layer, position)] = rows[index]

    def evidence_complete_at(self, position: int) -> bool:
        """Whether every condition layer has been read at ``position``. The
        freeze trigger uses this to avoid deciding before the trigger pass's
        own readings have arrived (a condition layer above the gated op's
        layer is only read later in the pass).
        """
        return all((layer, position) in self._rows for layer in self.layers)

    def note_pass(self, positions: range, prompt_len: int) -> None:
        if self._held is not None:
            return
        covers_final_prompt = positions.start <= prompt_len - 1 < positions.stop
        if covers_final_prompt and self.evidence_complete_at(prompt_len - 1):
            self._freeze()
        elif positions.start >= prompt_len:
            self._freeze()

    def _pool(self, layer: int) -> torch.Tensor | None:
        observed = [
            (position, row)
            for (row_layer, position), row in self._rows.items()
            if row_layer == layer
        ]
        if not observed:
            return None
        if self.pooling == "last":
            return max(observed, key=lambda item: item[0])[1]
        return torch.stack([row for _, row in observed]).mean(dim=0)

    def _freeze(self) -> None:
        readout = READOUTS[self.readout_kind]
        values: dict = {}
        for layer in self.layers:
            pooled = self._pool(layer)
            if pooled is None:
                continue
            values[layer] = float(readout(pooled, self.readout_tensor[self._row_for[layer]]))
        if not values:
            self._held = False
        else:
            self._held = bool(RULES[self.rule_kind](values, **self.rule_params))
        self._rows.clear()

    def decision(self) -> bool | None:
        return self._held

    def reset(self) -> None:
        self._rows.clear()
        self._held = None


def build_gate(gate_spec, artifacts: dict) -> GateState:
    """Construct the gate state machine for a parsed ``GateSpec``.

    ``artifacts`` maps artifact id -> tensor dict, as resolved by the
    registry. The readout tensor's rows align with ``gate_spec.layers``.
    """
    tensors = artifacts[gate_spec.readout.artifact]
    tensor_name = ARTIFACT_TENSORS[gate_spec.readout.kind][0]
    return GateState(
        layers=gate_spec.layers,
        pooling=gate_spec.pooling,
        readout_kind=gate_spec.readout.kind,
        readout_tensor=tensors[tensor_name],
        rule_kind=gate_spec.rule.kind,
        rule_params=gate_spec.rule.params,
    )
