# tests/core/test_interpreter.py
"""Golden-tensor tests per transform/modifier/readout/rule kind. These
functions are the reference semantics, so the expectations here are the spec.
"""
import math

import pytest
import torch

from vllm_hook_plugins.core.interpreter import apply_op, scope_rows
from vllm_hook_plugins.core.interpreter.gates import (
    affine,
    cosine,
    per_key_threshold,
    projected_cosine,
    sum_threshold,
)
from vllm_hook_plugins.core.interpreter.modifiers import alignment_adaptive, norm_preserving
from vllm_hook_plugins.core.interpreter.transforms import (
    additive,
    head_additive,
    projection,
    rotation,
)
from vllm_hook_plugins.core.schema import ModifierSpec, OpSpec, ScopeSpec

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


def test_projection_removes_component_and_is_idempotent():
    stream = torch.randn(6, 32)
    vector = torch.randn(32) * 3
    out = projection(stream, vector=vector)
    unit = vector / vector.norm()
    assert torch.allclose(out @ unit, torch.zeros(6), atol=1e-5)
    again = projection(out, vector=vector)
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
# readouts (pooled row -> one value per condition layer)
# ---------------------------------------------------------------------------


def test_affine_is_signed_dot_with_weight_row():
    pooled = torch.tensor([2.0, 5.0])
    weights = torch.tensor([1.0, 0.0])
    assert float(affine(pooled, weights)) == pytest.approx(2.0)
    assert float(affine(pooled, torch.tensor([0.0, -1.0]))) == pytest.approx(-5.0)


def test_cosine_sign_tracks_alignment():
    direction = torch.tensor([1.0, 0.0, 0.0])
    aligned = torch.tensor([3.0, 0.0, 0.0])
    opposed = torch.tensor([-2.0, 0.0, 0.0])
    orthogonal = torch.tensor([0.0, 4.0, 0.0])
    assert float(cosine(aligned, direction)) == pytest.approx(1.0, abs=1e-6)
    assert float(cosine(opposed, direction)) == pytest.approx(-1.0, abs=1e-6)
    assert float(cosine(orthogonal, direction)) == pytest.approx(0.0, abs=1e-6)


def test_cosine_zero_pooled_is_finite():
    # The epsilon floor keeps a zero row from dividing by zero.
    value = float(cosine(torch.zeros(4), torch.tensor([1.0, 0.0, 0.0, 0.0])))
    assert math.isfinite(value)
    assert value == pytest.approx(0.0)


def test_projected_cosine_matches_rank_one_projector_formula():
    pooled = torch.tensor([0.7, -1.3, 2.0])
    direction = torch.tensor([1.0, 2.0, -0.5])
    eps = 1e-8
    projector = torch.outer(direction, direction) / (direction @ direction + eps)
    projected = torch.tanh(pooled @ projector)
    expected = (pooled @ projected) / (pooled.norm() * projected.norm() + eps)
    assert float(projected_cosine(pooled, direction)) == pytest.approx(float(expected), abs=1e-6)


def test_projected_cosine_zero_direction_is_finite():
    # A zero direction collapses the projector to zeros; the epsilon floor
    # keeps the result finite rather than NaN.
    value = float(projected_cosine(torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3)))
    assert math.isfinite(value)


# ---------------------------------------------------------------------------
# rules (per-layer values -> boolean decision)
# ---------------------------------------------------------------------------


def test_sum_threshold_ties_open():
    # sum(values) + bias >= 0, with exact-zero opening the gate.
    assert sum_threshold({0: 1.5, 1: 0.5}, bias=-2.0) is True  # exactly 0
    assert sum_threshold({0: 1.0}, bias=-1.5) is False
    assert sum_threshold({0: 4.0}, bias=1.0) is True


def test_per_key_threshold_ge_and_le_with_any_all():
    values = {0: 0.8, 1: 0.2}
    assert per_key_threshold(values, threshold=0.5, comparator="ge", aggregate="any") is True
    assert per_key_threshold(values, threshold=0.5, comparator="ge", aggregate="all") is False
    assert per_key_threshold(values, threshold=0.5, comparator="le", aggregate="any") is True
    assert per_key_threshold(values, threshold=0.5, comparator="le", aggregate="all") is False
    # comparison is inclusive at the threshold
    assert per_key_threshold({0: 0.5}, threshold=0.5, comparator="ge", aggregate="all") is True
    assert per_key_threshold({0: 0.5}, threshold=0.5, comparator="le", aggregate="all") is True


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
