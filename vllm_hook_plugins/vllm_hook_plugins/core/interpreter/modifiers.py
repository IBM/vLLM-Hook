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
_EPS = 1e-8


def norm_preserving(inner):
    """Rescale transformed rows whose L2 norm increased back to their input
    norm; rows whose norm decreased or held are returned as transformed.
    """

    def wrapped(stream: torch.Tensor) -> torch.Tensor:
        original_norm = stream.norm(dim=-1, keepdim=True)
        out = inner(stream)
        new_norm = out.norm(dim=-1, keepdim=True)
        needs_rescale = new_norm > original_norm
        if needs_rescale.any():
            scale = torch.where(needs_rescale, original_norm / (new_norm + _EPS), torch.ones_like(new_norm))
            out = out * scale
        return out

    return wrapped


def alignment_adaptive(inner, *, vector: torch.Tensor, threshold: float, use_cosine: bool):
    """Apply the wrapped transform only at rows aligned with ``vector``.

    Each row's alignment is its cosine similarity with ``vector`` when
    ``use_cosine`` is true, else its projection onto the unit-normalized
    vector. Rows with alignment strictly above ``threshold`` receive the
    transformed value; every other row is returned unchanged.
    """

    def wrapped(stream: torch.Tensor) -> torch.Tensor:
        if use_cosine:
            alignment = torch.nn.functional.cosine_similarity(stream, vector.view(1, -1), dim=-1)
        else:
            unit = vector / (vector.norm() + _EPS)
            alignment = stream @ unit
        keep = (alignment > threshold).unsqueeze(-1)
        out = inner(stream)
        return torch.where(keep, out, stream)

    return wrapped


MODIFIERS = {
    "norm_preserving": norm_preserving,
    "alignment_adaptive": alignment_adaptive,
}
