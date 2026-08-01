"""Wrapper chain per modifier kind, composed innermost-first around a transform.

Each entry in ``MODIFIERS`` is a factory: it takes the inner callable
(``stream -> transformed stream``) plus the modifier's own params/tensors
and returns a new callable with the same signature. ``apply_op`` composes
``modifiers[0]`` closest to the transform, then ``modifiers[1]`` around
that, and so on.
"""
from __future__ import annotations

import torch

# Norm floor guarding divisions; small enough to be inert at model scales.
_EPS = 1e-6


def norm_preserving(inner):
    """Rescale each transformed row back to its input L2 norm, so the
    wrapped transform only redirects the row, never changes its length.
    """

    def wrapped(stream: torch.Tensor) -> torch.Tensor:
        out = inner(stream)
        in_norm = stream.norm(dim=-1, keepdim=True)
        out_norm = out.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        return out * (in_norm / out_norm)

    return wrapped


def alignment_adaptive(inner, *, vector: torch.Tensor):
    """Scale the wrapped transform's effect per row by that row's cosine
    alignment with ``vector``, clamped to [0, 1]:
    ``out = stream + a * (inner(stream) - stream)`` with
    ``a = clamp(cos(stream_row, vector), 0, 1)``. Rows pointing away from
    the direction are left untouched; aligned rows get the full transform.
    """

    def wrapped(stream: torch.Tensor) -> torch.Tensor:
        out = inner(stream)
        unit = vector / vector.norm()
        alignment = (stream @ unit) / stream.norm(dim=-1).clamp_min(_EPS)
        alignment = alignment.clamp(0.0, 1.0).unsqueeze(-1)
        return stream + alignment * (out - stream)

    return wrapped


MODIFIERS = {
    "norm_preserving": norm_preserving,
    "alignment_adaptive": alignment_adaptive,
}
