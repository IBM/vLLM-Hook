# tests/core/test_gate_state.py
"""Behavioral matrix for the per-request gate state machine, driven with
plain CPU tensors through observe/note_pass/decision/reset. GateState is
deliberately vLLM-free, so its whole freeze-and-hold contract is exercised
here rather than on a GPU host.
"""
import torch

from vllm_hook_plugins.core.interpreter.gates import GateState, build_gate
from vllm_hook_plugins.core.schema import GateSpec, ReadoutSpec, RuleSpec

PROBE = "sha256:" + "cd" * 32


def _affine_sum_gate(*, layers, pooling, weights, bias):
    """An affine readout with one weight row per condition layer, decided by
    a summed threshold over the pooled per-layer scores.
    """
    return GateState(
        layers=layers,
        pooling=pooling,
        readout_kind="affine",
        readout_tensor=torch.tensor(weights),
        rule_kind="sum_threshold",
        rule_params={"bias": bias},
    )


# ---------------------------------------------------------------------------
# freeze timing over prefill shapes
# ---------------------------------------------------------------------------


def test_single_pass_prefill_freezes_at_end_of_prompt():
    gate = _affine_sum_gate(layers=[0], pooling="mean", weights=[[1.0]], bias=-0.4)
    gate.observe(0, range(0, 8), torch.full((8, 1), 0.5))
    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate.decision() is True  # mean = 0.5; 0.5 - 0.4 >= 0


def test_chunked_prefill_defers_decision_until_final_position_arrives():
    gate = _affine_sum_gate(layers=[0], pooling="mean", weights=[[1.0]], bias=-0.4)

    # mid-prompt chunk: evidence lands but the final prompt position has not
    gate.observe(0, range(0, 4), torch.full((4, 1), 0.5))
    gate.note_pass(range(0, 4), prompt_len=8)
    assert gate.decision() is None

    # the chunk covering the final prompt position triggers the one decision
    gate.observe(0, range(4, 8), torch.full((4, 1), 0.5))
    gate.note_pass(range(4, 8), prompt_len=8)
    assert gate.decision() is True


def test_freeze_defers_when_condition_layer_sits_above_gated_op():
    # The op-layer coverage check runs before a higher condition layer has
    # been read in the covering pass; the freeze defers to the first
    # post-prompt pass so the decision reflects the whole prompt.
    gate = _affine_sum_gate(layers=[5], pooling="mean", weights=[[1.0]], bias=-0.4)

    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate.decision() is None  # layer 5 not read yet at op time

    gate.observe(5, range(0, 8), torch.full((8, 1), 0.5))  # reading lands later in the pass
    gate.note_pass(range(8, 9), prompt_len=8)
    assert gate.decision() is True


# ---------------------------------------------------------------------------
# undecided-is-closed and freeze-hold
# ---------------------------------------------------------------------------


def test_undecided_before_freeze():
    gate = _affine_sum_gate(layers=[0], pooling="mean", weights=[[1.0]], bias=-0.4)
    assert gate.decision() is None
    gate.observe(0, range(0, 4), torch.full((4, 1), 0.5))
    assert gate.decision() is None  # evidence without a covering pass stays undecided


def test_freeze_closed_when_no_evidence_ever_arrives():
    gate = _affine_sum_gate(layers=[5], pooling="mean", weights=[[1.0]], bias=-0.4)
    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate.decision() is None  # undecided, closed at op time
    gate.note_pass(range(8, 9), prompt_len=8)
    assert gate.decision() is False  # frozen closed at the post-prompt pass
    gate.observe(5, range(9, 10), torch.tensor([[100.0]]))
    assert gate.decision() is False


def test_frozen_decision_ignores_later_evidence():
    gate = _affine_sum_gate(layers=[0], pooling="mean", weights=[[1.0]], bias=-0.4)
    gate.observe(0, range(0, 8), torch.full((8, 1), 0.5))
    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate.decision() is True

    gate.observe(0, range(8, 9), torch.tensor([[-100.0]]))
    gate.note_pass(range(8, 9), prompt_len=8)
    assert gate.decision() is True


# ---------------------------------------------------------------------------
# replay idempotence and reset
# ---------------------------------------------------------------------------


def test_replayed_pass_overwrites_position_rows_same_decision():
    gate = _affine_sum_gate(layers=[0], pooling="mean", weights=[[1.0]], bias=-3.0)
    gate.observe(0, range(0, 2), torch.tensor([[2.0], [6.0]]))  # mean = 4.0
    # a re-executed prefill pass writes the same rows again
    gate.observe(0, range(0, 2), torch.tensor([[2.0], [6.0]]))
    gate.note_pass(range(0, 2), prompt_len=2)
    assert gate.decision() is True  # 4.0 - 3.0 >= 0, not doubled


def test_reset_clears_rows_and_held_decision():
    gate = _affine_sum_gate(layers=[0], pooling="mean", weights=[[1.0]], bias=-0.4)
    gate.observe(0, range(0, 8), torch.full((8, 1), 0.5))
    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate.decision() is True

    gate.reset()
    assert gate.decision() is None
    assert gate._rows == {}


def test_rows_freed_after_freeze():
    gate = _affine_sum_gate(layers=[0], pooling="mean", weights=[[1.0]], bias=-0.4)
    gate.observe(0, range(0, 8), torch.full((8, 1), 0.5))
    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate._rows == {}  # buffers released once the decision is held


# ---------------------------------------------------------------------------
# pooling and multi-layer aggregation
# ---------------------------------------------------------------------------


def test_mean_and_last_pooling_select_different_rows():
    weights = [[1.0]]
    mean_gate = _affine_sum_gate(layers=[0], pooling="mean", weights=weights, bias=0.0)
    mean_gate.observe(0, range(0, 3), torch.tensor([[5.0], [5.0], [-1.0]]))
    mean_gate.note_pass(range(0, 3), prompt_len=3)
    assert mean_gate.decision() is True  # mean = 3.0

    last_gate = _affine_sum_gate(layers=[0], pooling="last", weights=weights, bias=0.0)
    last_gate.observe(0, range(0, 3), torch.tensor([[5.0], [5.0], [-1.0]]))
    last_gate.note_pass(range(0, 3), prompt_len=3)
    assert last_gate.decision() is False  # last position scores -1.0


def test_non_condition_layer_readings_are_ignored():
    gate = _affine_sum_gate(layers=[2], pooling="mean", weights=[[1.0, 0.0]], bias=-3.0)
    gate.observe(2, range(0, 1), torch.tensor([[2.0, 5.0]]))
    gate.observe(7, range(0, 1), torch.tensor([[100.0, 0.0]]))  # not a condition layer
    gate.note_pass(range(0, 1), prompt_len=1)
    assert gate.decision() is False  # only layer 2's 2.0 counts; 2.0 - 3.0 < 0


def test_affine_sum_threshold_across_two_layers():
    # Two condition layers, one weight row each; sum_threshold opens the gate
    # when the summed pooled per-layer scores plus bias reach zero.
    gate = _affine_sum_gate(
        layers=[1, 3],
        pooling="mean",
        weights=[[1.0, 0.0], [0.0, 1.0]],
        bias=-5.0,
    )
    gate.observe(1, range(0, 2), torch.tensor([[2.0, 0.0], [4.0, 0.0]]))  # mean = 3.0
    gate.observe(3, range(0, 2), torch.tensor([[0.0, 2.0], [0.0, 4.0]]))  # mean = 3.0
    gate.note_pass(range(0, 2), prompt_len=2)
    assert gate.decision() is True  # 3.0 + 3.0 - 5.0 >= 0


# ---------------------------------------------------------------------------
# build_gate wiring
# ---------------------------------------------------------------------------


def test_build_gate_wires_readout_tensor_and_rule_params():
    artifacts = {PROBE: {"directions": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}}
    spec = GateSpec(
        layers=(2, 4),
        pooling="last",
        readout=ReadoutSpec(kind="cosine", artifact=PROBE),
        rule=RuleSpec(kind="per_key_threshold",
                      params={"threshold": 0.5, "comparator": "ge", "aggregate": "any"}),
    )
    gate = build_gate(spec, artifacts)
    assert isinstance(gate, GateState)
    assert gate.layers == [2, 4]
    assert gate.pooling == "last"
    assert gate.readout_kind == "cosine"
    assert gate.rule_kind == "per_key_threshold"
    assert gate.rule_params == {"threshold": 0.5, "comparator": "ge", "aggregate": "any"}
    assert torch.equal(gate.readout_tensor, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_build_gate_cosine_decision_over_two_layers():
    # directions pick out axis 0 at layer 2 and axis 1 at layer 4; a signed
    # cosine >= 0.5 on either layer opens the gate (aggregate any).
    artifacts = {PROBE: {"directions": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}}
    spec = GateSpec(
        layers=(2, 4),
        pooling="mean",
        readout=ReadoutSpec(kind="cosine", artifact=PROBE),
        rule=RuleSpec(kind="per_key_threshold",
                      params={"threshold": 0.5, "comparator": "ge", "aggregate": "any"}),
    )
    gate = build_gate(spec, artifacts)
    gate.observe(2, range(0, 1), torch.tensor([[1.0, 0.0]]))  # cos = 1.0 with axis 0
    gate.observe(4, range(0, 1), torch.tensor([[1.0, 0.0]]))  # cos = 0.0 with axis 1
    gate.note_pass(range(0, 1), prompt_len=1)
    assert gate.decision() is True  # layer 2 clears the threshold
