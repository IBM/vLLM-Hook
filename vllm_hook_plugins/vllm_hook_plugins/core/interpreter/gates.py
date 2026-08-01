"""Per-request gate state machines, one per gate kind.

All evidence is keyed by absolute position and overwrites on replay, so
re-executed passes (preemption recompute, the post-stop extra pass) are
idempotent. Instances hold no engine state: the worker feeds them
condition-layer rows and pass coverage, and asks for a decision.
"""
from __future__ import annotations

import torch


class GateState:
    """One instance per (request, gated op).

    ``observe(layer, positions, stream_rows)`` ingests condition-layer
    readings; ``decision()`` returns True/False/None (None = undecided).
    ``reset()`` clears evidence after a request restart.
    ``note_pass(positions, prompt_len)`` reports each pass's coverage
    before the decision is read; only ``cache_once`` acts on it.
    """

    def observe(self, layer: int, positions: range, stream_rows: torch.Tensor) -> None:
        raise NotImplementedError

    def note_pass(self, positions: range, prompt_len: int) -> None:
        pass

    def evidence_complete_at(self, position: int) -> bool:
        """Whether every condition layer has been read at ``position``.
        ``cache_once`` uses this to avoid freezing a decision before the
        trigger pass's own readings have arrived (a condition layer above
        the gated op's layer is only read later in the pass).
        """
        return True

    def decision(self) -> bool | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class ProbeSumGate(GateState):
    """Calibrated linear probe over pooled per-layer readings.

    Each condition layer contributes ``weights[i] @ pool(x_layer)``, where
    ``weights`` rows align with the ``condition_layers`` order and pooling
    runs over that layer's observed positions (``"mean"``) or takes the
    reading at the highest observed position (``"last"``). The decision is
    ``sum of observed-layer contributions + bias >= 0`` (ties open),
    undecided before any evidence. Scores are computed in float32
    regardless of stream dtype.
    """

    def __init__(self, *, condition_layers: list, pooling: str, weights: torch.Tensor, bias: float = 0.0):
        self.condition_layers = [int(layer) for layer in condition_layers]
        self._weight_row = {layer: index for index, layer in enumerate(self.condition_layers)}
        self.pooling = pooling
        self.weights = weights.detach().float()
        self.bias = float(bias)
        self._scores: dict = {}  # (layer, position) -> float

    def observe(self, layer: int, positions: range, stream_rows: torch.Tensor) -> None:
        row = self._weight_row.get(layer)
        if row is None:
            return
        weight = self.weights[row].to(stream_rows.device)
        # one device sync per pass, not one per token
        scores = (stream_rows.detach().float() @ weight).tolist()
        for idx, position in enumerate(positions):
            self._scores[(layer, position)] = scores[idx]

    def evidence_complete_at(self, position: int) -> bool:
        return all((layer, position) in self._scores for layer in self.condition_layers)

    def decision(self) -> bool | None:
        if not self._scores:
            return None
        total = 0.0
        for layer in self.condition_layers:
            observed = [
                (position, score)
                for (score_layer, position), score in self._scores.items()
                if score_layer == layer
            ]
            if not observed:
                continue
            if self.pooling == "last":
                total += max(observed)[1]
            else:
                total += sum(score for _, score in observed) / len(observed)
        return total + self.bias >= 0

    def reset(self) -> None:
        self._scores.clear()


class MultiKeyThresholdGate(GateState):
    """Voting gate over multiple probe keys.

    Key ``k`` fires when any observed (layer, position) row has
    ``row @ weights[k] + biases[k] > 0``. The gate opens while the
    fraction of fired keys is >= ``threshold``. Scores are computed in
    float32 regardless of stream dtype.
    """

    def __init__(self, *, threshold: float, condition_layers: list, weights: torch.Tensor, biases: torch.Tensor | None = None):
        self.threshold = float(threshold)
        self.condition_layers = frozenset(condition_layers)
        self.weights = weights.detach().float()
        if biases is None:
            biases = torch.zeros(self.weights.shape[0])
        self.biases = biases.detach().float()
        self._fired: dict = {}  # (layer, position) -> tuple[bool, ...] per key

    def observe(self, layer: int, positions: range, stream_rows: torch.Tensor) -> None:
        if layer not in self.condition_layers:
            return
        weights = self.weights.to(stream_rows.device)
        biases = self.biases.to(stream_rows.device)
        fired = (stream_rows.detach().float() @ weights.T + biases) > 0  # (rows, num_keys)
        # one device sync per pass, not one per (token, key)
        fired = fired.tolist()
        for idx, position in enumerate(positions):
            self._fired[(layer, position)] = tuple(fired[idx])

    def evidence_complete_at(self, position: int) -> bool:
        return all((layer, position) in self._fired for layer in self.condition_layers)

    def decision(self) -> bool | None:
        if not self._fired:
            return None
        num_keys = self.weights.shape[0]
        if num_keys == 0:
            return False
        fired_any = [False] * num_keys
        for per_key in self._fired.values():
            fired_any = [a or b for a, b in zip(fired_any, per_key)]
        return sum(fired_any) / num_keys >= self.threshold

    def reset(self) -> None:
        self._fired.clear()


class CacheOnceGate(GateState):
    """Freeze the inner gate's decision at the end of the prompt.

    The single decision is made at the first pass covering the final
    prompt position, once that pass's condition readings for the final
    position have arrived (undecided counts as closed), and holds for the
    request. When a condition layer sits above the gated op's layer, its
    reading lands after the op fires in the trigger pass, so the freeze
    defers to the first post-prompt pass — the decision then reflects the
    full prompt. Before the decision the gate is closed.
    """

    def __init__(self, inner: GateState):
        self.inner = inner
        self._held: bool | None = None

    def observe(self, layer: int, positions: range, stream_rows: torch.Tensor) -> None:
        if self._held is None:
            self.inner.observe(layer, positions, stream_rows)

    def note_pass(self, positions: range, prompt_len: int) -> None:
        if self._held is not None:
            return
        covers_final_prompt = positions.start <= prompt_len - 1 < positions.stop
        if covers_final_prompt and self.inner.evidence_complete_at(prompt_len - 1):
            self._held = bool(self.inner.decision())
        elif positions.start >= prompt_len:
            self._held = bool(self.inner.decision())

    def decision(self) -> bool | None:
        return self._held

    def reset(self) -> None:
        self._held = None
        self.inner.reset()


GATES = {
    "cache_once": CacheOnceGate,
    "probe_sum": ProbeSumGate,
    "multi_key_threshold": MultiKeyThresholdGate,
}


def build_gate(gate_spec, artifacts: dict) -> GateState:
    """Construct the gate state machine for a parsed ``GateSpec``.

    ``artifacts`` maps artifact id -> tensor dict, as resolved by the
    registry. Probe gates read their tensors here; optional tensors
    (``bias`` / ``biases``) default when absent from the artifact.
    """
    if gate_spec.kind == "cache_once":
        return CacheOnceGate(build_gate(gate_spec.inner, artifacts))
    tensors = artifacts[gate_spec.artifact]
    if gate_spec.kind == "probe_sum":
        return ProbeSumGate(
            condition_layers=gate_spec.params["condition_layers"],
            pooling=gate_spec.params["pooling"],
            weights=tensors["weights"],
            bias=float(tensors["bias"]) if "bias" in tensors else 0.0,
        )
    if gate_spec.kind == "multi_key_threshold":
        return MultiKeyThresholdGate(
            threshold=gate_spec.params["threshold"],
            condition_layers=gate_spec.params["condition_layers"],
            weights=tensors["weights"],
            biases=tensors.get("biases"),
        )
    raise KeyError(f"no gate state machine for kind {gate_spec.kind!r}")
