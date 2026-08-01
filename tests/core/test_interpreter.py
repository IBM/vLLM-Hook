# tests/core/test_interpreter.py
"""Golden-tensor tests per transform/modifier/gate kind. These functions
are the reference semantics, so the expectations here are the spec.
"""
import math

import pytest
import torch

from vllm_hook_plugins.core.interpreter import apply_op, build_gate, scope_rows
from vllm_hook_plugins.core.interpreter.gates import (
    CacheOnceGate,
    MultiKeyThresholdGate,
    ProbeSumGate,
)
from vllm_hook_plugins.core.interpreter.modifiers import alignment_adaptive, norm_preserving
from vllm_hook_plugins.core.interpreter.transforms import (
    additive,
    directional_ablation,
    head_additive,
    rotation,
)
from vllm_hook_plugins.core.schema import GateSpec, ModifierSpec, OpSpec, ScopeSpec

VEC = "sha256:" + "ab" * 32

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def test_additive_golden():
    stream = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    vector = torch.tensor([10.0, -10.0])
    out = additive(stream, vector=vector, strength=0.5)
    assert torch.equal(out, torch.tensor([[6.0, -3.0], [8.0, -1.0]]))
    # input untouched
    assert torch.equal(stream, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_directional_ablation_removes_component_and_is_idempotent():
    stream = torch.randn(6, 32)
    vector = torch.randn(32) * 3
    out = directional_ablation(stream, vector=vector)
    unit = vector / vector.norm()
    assert torch.allclose(out @ unit, torch.zeros(6), atol=1e-5)
    again = directional_ablation(out, vector=vector)
    assert torch.allclose(out, again, atol=1e-6)


def test_rotation_offset_quarter_turn_maps_b1_to_b2():
    hidden = 16
    b1 = torch.zeros(hidden)
    b1[0] = 1.0
    b2 = torch.zeros(hidden)
    b2[1] = 1.0
    basis = torch.stack([b1, b2])
    out = rotation(b1.unsqueeze(0), basis=basis, angle=math.pi / 2, mode="offset")
    assert torch.allclose(out[0], b2, atol=1e-6)


def test_rotation_target_sets_absolute_in_plane_angle():
    hidden = 16
    b1 = torch.zeros(hidden)
    b1[0] = 1.0
    b2 = torch.zeros(hidden)
    b2[1] = 1.0
    basis = torch.stack([b1, b2])
    # rows at different starting angles all land at the target angle
    stream = torch.stack([2.0 * b1, 3.0 * b2, -1.5 * b1])
    out = rotation(stream, basis=basis, angle=math.pi / 4, mode="target")
    angles = torch.atan2(out @ b2, out @ b1)
    assert torch.allclose(angles, torch.full((3,), math.pi / 4), atol=1e-5)
    # in-plane magnitude preserved per row
    assert torch.allclose(out.norm(dim=-1), stream.norm(dim=-1), atol=1e-5)


@pytest.mark.parametrize("mode", ["target", "offset"])
def test_rotation_preserves_norm_and_orthogonal_complement(mode):
    stream = torch.randn(5, 32)
    basis = torch.randn(2, 32)
    out = rotation(stream, basis=basis, angle=0.7, mode=mode)
    assert torch.allclose(out.norm(dim=-1), stream.norm(dim=-1), atol=1e-5)
    # components orthogonal to the (orthonormalized) plane are untouched
    b1 = basis[0] / (basis[0].norm() + 1e-8)
    b2 = basis[1] - (basis[1] @ b1) * b1
    b2 = b2 / (b2.norm() + 1e-8)
    residual_in = stream - (stream @ b1).unsqueeze(-1) * b1 - (stream @ b2).unsqueeze(-1) * b2
    residual_out = out - (out @ b1).unsqueeze(-1) * b1 - (out @ b2).unsqueeze(-1) * b2
    assert torch.allclose(residual_in, residual_out, atol=1e-5)


def test_head_additive_broadcasts_per_head_and_shared_vectors():
    heads = torch.zeros(3, 4, 8)
    per_head = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    out = head_additive(heads, vector=per_head, strength=2.0)
    assert torch.equal(out[1], 2.0 * per_head)
    shared = torch.ones(8)
    out = head_additive(heads, vector=shared, strength=3.0)
    assert torch.equal(out, torch.full((3, 4, 8), 3.0))


# ---------------------------------------------------------------------------
# modifiers
# ---------------------------------------------------------------------------


def test_norm_preserving_restores_row_norms():
    stream = torch.randn(4, 16)
    vector = torch.randn(16) * 5

    def inner(rows):
        return additive(rows, vector=vector, strength=2.0)

    out = norm_preserving(inner)(stream)
    assert torch.allclose(out.norm(dim=-1), stream.norm(dim=-1), atol=1e-5)
    # direction matches the unmodified transform
    raw = inner(stream)
    cos = (out * raw).sum(-1) / (out.norm(dim=-1) * raw.norm(dim=-1))
    assert torch.all(cos > 0.999)


def test_alignment_adaptive_masks_rows_by_projection_threshold():
    vector = torch.zeros(8)
    vector[0] = 1.0
    aligned = torch.zeros(1, 8)
    aligned[0, 0] = 2.0  # projection = 2 > 0 -> transformed
    opposed = torch.zeros(1, 8)
    opposed[0, 0] = -2.0  # projection = -2 <= 0 -> untouched

    def inner(rows):
        return rows + 1.0

    wrapped = alignment_adaptive(inner, vector=vector, threshold=0.0, use_cosine=False)
    assert torch.allclose(wrapped(aligned), aligned + 1.0)
    assert torch.allclose(wrapped(opposed), opposed)

    # threshold is strict: alignment exactly at the threshold stays untouched
    at_threshold = torch.zeros(1, 8)
    at_threshold[0, 0] = 1.0
    strict = alignment_adaptive(inner, vector=vector, threshold=1.0 - 1e-7, use_cosine=False)
    assert torch.allclose(strict(at_threshold), at_threshold + 1.0)
    strict = alignment_adaptive(inner, vector=vector, threshold=1.0, use_cosine=False)
    assert torch.allclose(strict(at_threshold), at_threshold)


def test_alignment_adaptive_cosine_mode_normalizes_rows():
    vector = torch.zeros(8)
    vector[0] = 1.0
    weakly_aligned = torch.ones(1, 8)  # cos = 1/sqrt(8) ~ 0.35

    def inner(rows):
        return rows + 1.0

    below = alignment_adaptive(inner, vector=vector, threshold=0.5, use_cosine=True)
    assert torch.allclose(below(weakly_aligned), weakly_aligned)
    above = alignment_adaptive(inner, vector=vector, threshold=0.3, use_cosine=True)
    assert torch.allclose(above(weakly_aligned), weakly_aligned + 1.0)


# ---------------------------------------------------------------------------
# apply_op composition
# ---------------------------------------------------------------------------


def _op(modifiers=()):
    return OpSpec(
        layers=(0,),
        transform_kind="additive",
        transform_params={"strength": 2.0},
        artifact=VEC,
        modifiers=tuple(modifiers),
        scope=ScopeSpec(kind="all", params={}),
        gate=None,
    )


def test_apply_op_matches_direct_call():
    stream = torch.randn(5, 16)
    vector = torch.randn(16)
    artifacts = {VEC: {"vector": vector}}
    out = apply_op(_op(), stream, artifacts)
    assert torch.allclose(out, additive(stream, vector=vector, strength=2.0))


def test_apply_op_composes_modifiers_innermost_first():
    stream = torch.randn(5, 16)
    vector = torch.randn(16)
    artifacts = {VEC: {"vector": vector}}
    op = _op(modifiers=[
        ModifierSpec(kind="alignment_adaptive", params={"threshold": 0.0, "use_cosine": False}, artifact=VEC),
        ModifierSpec(kind="norm_preserving", params={}, artifact=None),
    ])
    out = apply_op(op, stream, artifacts)

    def inner(rows):
        return additive(rows, vector=vector, strength=2.0)

    reference = norm_preserving(
        alignment_adaptive(inner, vector=vector, threshold=0.0, use_cosine=False)
    )(stream)
    assert torch.allclose(out, reference)


def test_apply_op_casts_artifacts_to_stream_dtype():
    stream = torch.randn(3, 16, dtype=torch.float64)
    artifacts = {VEC: {"vector": torch.randn(16, dtype=torch.float32)}}
    out = apply_op(_op(), stream, artifacts)
    assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def test_probe_sum_pools_per_layer_and_overwrites_on_replay():
    weights = torch.tensor([[1.0, 0.0]])
    gate = ProbeSumGate(condition_layers=[2], pooling="mean", weights=weights, bias=-3.0)
    assert gate.decision() is None

    gate.observe(2, range(0, 1), torch.tensor([[2.0, 5.0]]))
    assert gate.decision() is False  # mean = 2.0; 2.0 - 3.0 < 0

    gate.observe(2, range(1, 2), torch.tensor([[6.0, 5.0]]))
    assert gate.decision() is True  # mean = 4.0; 4.0 - 3.0 >= 0

    # replaying position 1 overwrites, not accumulates
    gate.observe(2, range(1, 2), torch.tensor([[6.0, 5.0]]))
    assert gate.decision() is True
    gate.observe(2, range(1, 2), torch.tensor([[0.5, 0.0]]))
    assert gate.decision() is False  # mean = 1.25

    # readings from non-condition layers are ignored
    gate.observe(7, range(2, 3), torch.tensor([[100.0, 0.0]]))
    assert gate.decision() is False

    gate.reset()
    assert gate.decision() is None


def test_probe_sum_sums_pooled_contributions_across_layers():
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gate = ProbeSumGate(condition_layers=[1, 3], pooling="mean", weights=weights, bias=-5.0)

    gate.observe(1, range(0, 2), torch.tensor([[2.0, 0.0], [4.0, 0.0]]))  # mean = 3.0
    assert gate.decision() is False  # unobserved layer contributes nothing yet
    gate.observe(3, range(0, 2), torch.tensor([[0.0, 2.0], [0.0, 4.0]]))  # mean = 3.0
    assert gate.decision() is True  # 3.0 + 3.0 - 5.0 >= 0


def test_probe_sum_last_pooling_takes_highest_observed_position():
    weights = torch.tensor([[1.0]])
    gate = ProbeSumGate(condition_layers=[0], pooling="last", weights=weights, bias=0.0)
    gate.observe(0, range(0, 3), torch.tensor([[5.0], [5.0], [-1.0]]))
    assert gate.decision() is False  # position 2 scores -1.0
    gate.observe(0, range(2, 3), torch.tensor([[2.0]]))
    assert gate.decision() is True  # replay overwrote position 2


def test_multi_key_threshold_counts_fired_keys():
    # keys fire on x>0 / y>0 / x+y<0 respectively
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    gate = MultiKeyThresholdGate(threshold=0.5, condition_layers=[1], weights=weights)
    assert gate.decision() is None

    gate.observe(1, range(0, 1), torch.tensor([[1.0, -1.0]]))
    assert gate.decision() is False  # 1/3 fired

    gate.observe(1, range(1, 2), torch.tensor([[-1.0, 1.0]]))
    assert gate.decision() is True  # 2/3 fired across positions

    gate.reset()
    assert gate.decision() is None


def test_cache_once_freezes_decision_at_final_prompt_position():
    weights = torch.tensor([[1.0]])
    inner = ProbeSumGate(condition_layers=[0], pooling="mean", weights=weights, bias=-0.4)
    gate = CacheOnceGate(inner)

    # mid-prompt pass: evidence lands but no decision yet
    gate.observe(0, range(0, 4), torch.full((4, 1), 0.5))
    gate.note_pass(range(0, 4), prompt_len=8)
    assert gate.decision() is None

    # pass covering the final prompt position triggers the single decision
    gate.observe(0, range(4, 8), torch.full((4, 1), 0.5))
    gate.note_pass(range(4, 8), prompt_len=8)
    assert gate.decision() is True  # mean = 0.5; 0.5 - 0.4 >= 0

    # later evidence cannot flip the held decision
    gate.observe(0, range(8, 9), torch.tensor([[-100.0]]))
    gate.note_pass(range(8, 9), prompt_len=8)
    assert gate.decision() is True

    gate.reset()
    assert gate.decision() is None
    assert inner.decision() is None


def test_cache_once_defers_freeze_until_trigger_pass_evidence_arrives():
    """A condition layer above the gated op's layer feeds after the op
    fires in the trigger pass; the freeze then defers to the first
    post-prompt pass so the decision reflects the full prompt.
    """
    inner = ProbeSumGate(condition_layers=[5], pooling="mean", weights=torch.tensor([[1.0]]), bias=-0.4)
    gate = CacheOnceGate(inner)

    # op-layer check in the covering pass: layer 5 has not been read yet
    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate.decision() is None

    # the reading lands later in the same pass
    gate.observe(5, range(0, 8), torch.full((8, 1), 0.5))

    # first post-prompt pass freezes with the full prompt evidence
    gate.note_pass(range(8, 9), prompt_len=8)
    assert gate.decision() is True

    # frozen: later evidence is ignored
    gate.observe(5, range(8, 9), torch.tensor([[-100.0]]))
    gate.note_pass(range(9, 10), prompt_len=8)
    assert gate.decision() is True


def test_cache_once_freezes_closed_when_no_evidence_ever_arrives():
    inner = ProbeSumGate(condition_layers=[5], pooling="mean", weights=torch.tensor([[1.0]]), bias=-0.4)
    gate = CacheOnceGate(inner)
    gate.note_pass(range(0, 8), prompt_len=8)
    assert gate.decision() is None  # undecided, closed at op time
    gate.note_pass(range(8, 9), prompt_len=8)
    assert gate.decision() is False  # frozen closed at the post-prompt pass
    gate.observe(5, range(9, 10), torch.tensor([[100.0]]))
    assert gate.decision() is False


def test_build_gate_resolves_artifacts_and_nesting():
    probe = "sha256:" + "cd" * 32
    artifacts = {probe: {"weights": torch.tensor([[1.0]]), "bias": torch.tensor(0.5)}}
    spec = GateSpec(
        kind="cache_once",
        params={},
        artifact=None,
        inner=GateSpec(kind="probe_sum", params={"condition_layers": [3], "pooling": "mean"},
                       artifact=probe, inner=None),
    )
    gate = build_gate(spec, artifacts)
    assert isinstance(gate, CacheOnceGate)
    assert isinstance(gate.inner, ProbeSumGate)
    assert gate.inner.bias == 0.5
    assert gate.inner.pooling == "mean"


# ---------------------------------------------------------------------------
# scopes (basic; pass-shape truth tables live in tests/workers)
# ---------------------------------------------------------------------------


class _View:
    def __init__(self, positions, prompt_len):
        self.positions = positions
        self.prompt_len = prompt_len


@pytest.mark.parametrize("scope,positions,prompt_len,expected", [
    (ScopeSpec("all", {}), range(0, 8), 8, slice(0, 8)),
    (ScopeSpec("after_prompt", {}), range(0, 8), 8, None),
    (ScopeSpec("after_prompt", {}), range(8, 9), 8, slice(0, 1)),
    (ScopeSpec("after_prompt", {}), range(6, 10), 8, slice(2, 4)),
    (ScopeSpec("from_position", {"position": 3}), range(0, 8), 8, slice(3, 8)),
    (ScopeSpec("from_position", {"position": 12}), range(0, 8), 8, None),
    (ScopeSpec("last_k", {"k": 2}), range(0, 8), 8, slice(6, 8)),
    (ScopeSpec("last_k", {"k": 100}), range(0, 8), 8, slice(0, 8)),
    (ScopeSpec("last_k", {"k": 1}), range(8, 9), 8, slice(0, 1)),
])
def test_scope_rows(scope, positions, prompt_len, expected):
    assert scope_rows(_View(positions, prompt_len), scope) == expected
