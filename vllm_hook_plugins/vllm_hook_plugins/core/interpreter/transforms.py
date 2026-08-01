"""Pure tensor math, one function per transform kind.

``stream`` is (rows, hidden) for residual transforms and
(rows, num_heads, head_dim) for ``head_additive``. Functions never mutate
their inputs and always return a new tensor in the input's dtype/device.
These functions ARE the reference semantics: the worker and any external
implementation call the same code, so equivalence is a property of one
function, not two.
"""
from __future__ import annotations

import math

import torch


def additive(stream: torch.Tensor, *, vector: torch.Tensor, strength: float) -> torch.Tensor:
    """``stream + strength * vector``, the vector broadcast over rows."""
    return stream + strength * vector


def directional_ablation(stream: torch.Tensor, *, vector: torch.Tensor) -> torch.Tensor:
    """Remove each row's component along ``vector``:
    ``out = stream - (stream @ v̂) v̂`` with ``v̂ = vector / ||vector||``.
    """
    unit = vector / vector.norm()
    coefficients = stream @ unit
    return stream - coefficients.unsqueeze(-1) * unit


def rotation(stream: torch.Tensor, *, basis: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate each row by ``angle`` radians within the plane spanned by
    ``basis[0]`` and ``basis[1]``.

    The basis is orthonormalized here (Gram-Schmidt: normalize ``basis[0]``,
    project it out of ``basis[1]``, normalize the remainder) so callers may
    pass any two linearly independent plane-spanning vectors. Components
    orthogonal to the plane are untouched.
    """
    b1 = basis[0] / basis[0].norm()
    b2 = basis[1] - (basis[1] @ b1) * b1
    b2 = b2 / b2.norm()
    c1 = stream @ b1
    c2 = stream @ b2
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    c1_rot = c1 * cos_a - c2 * sin_a
    c2_rot = c1 * sin_a + c2 * cos_a
    return stream + (c1_rot - c1).unsqueeze(-1) * b1 + (c2_rot - c2).unsqueeze(-1) * b2


def head_additive(heads: torch.Tensor, *, vector: torch.Tensor, strength: float) -> torch.Tensor:
    """``heads + strength * vector`` on a (rows, num_heads, head_dim) stream.

    ``vector`` is (num_heads, head_dim) for per-head offsets or (head_dim,)
    to offset every head identically; both broadcast over rows.
    """
    return heads + strength * vector


TRANSFORMS = {
    "additive": additive,
    "directional_ablation": directional_ablation,
    "rotation": rotation,
    "head_additive": head_additive,
}
