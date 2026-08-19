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


def projection(stream: torch.Tensor, *, vector: torch.Tensor) -> torch.Tensor:
    """Remove each row's component along ``vector``:
    ``out = stream - (stream @ v̂) v̂`` with ``v̂ = vector / ||vector||``.

    Computed via einsum contractions over a ``[1, H]`` basis so the kernel dispatch matches
    the reference in-process implementation bit-for-bit.
    """
    basis = (vector / vector.norm()).unsqueeze(0)
    coefficients = torch.einsum("rh,kh->rk", stream, basis)
    component = torch.einsum("rk,kh->rh", coefficients, basis)
    return stream - component


def rotation(stream: torch.Tensor, *, basis: torch.Tensor, angle: float, mode: str) -> torch.Tensor:
    """Rotate each row's in-plane component within the plane spanned by
    ``basis[0]`` and ``basis[1]``.

    ``mode="target"`` rotates each row's in-plane component TO the absolute
    angle ``angle`` measured from ``basis[0]`` (a per-row rotation by
    ``angle - atan2(c2, c1)``); ``mode="offset"`` rotates every row BY
    ``angle``. The basis is orthonormalized here (Gram-Schmidt with a 1e-8
    norm floor: normalize ``basis[0]``, project it out of ``basis[1]``,
    normalize the remainder) so callers may pass any two linearly independent
    plane-spanning vectors. Components orthogonal to the plane are untouched.
    """
    b1 = basis[0] / (basis[0].norm() + 1e-8)
    b2 = basis[1] - (basis[1] @ b1) * b1
    b2 = b2 / (b2.norm() + 1e-8)
    c1 = stream @ b1
    c2 = stream @ b2
    if mode == "target":
        delta = angle - torch.atan2(c2, c1)
        cos_d, sin_d = torch.cos(delta), torch.sin(delta)
    else:
        cos_d = stream.new_tensor(math.cos(angle))
        sin_d = stream.new_tensor(math.sin(angle))
    c1_new = cos_d * c1 - sin_d * c2
    c2_new = sin_d * c1 + cos_d * c2
    delta_c1 = (c1_new - c1).unsqueeze(-1)
    delta_c2 = (c2_new - c2).unsqueeze(-1)
    return stream + delta_c1 * b1 + delta_c2 * b2


def head_additive(heads: torch.Tensor, *, vector: torch.Tensor, strength: float) -> torch.Tensor:
    """``heads + strength * vector`` on a (rows, num_heads, head_dim) stream.

    ``vector`` is (num_heads, head_dim) for per-head offsets or (head_dim,)
    to offset every head identically; both broadcast over rows.
    """
    return heads + strength * vector


TRANSFORMS = {
    "additive": additive,
    "projection": projection,
    "rotation": rotation,
    "head_additive": head_additive,
}
