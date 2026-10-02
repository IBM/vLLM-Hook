"""Resolve a model config or steering vector: MIA's copy first, then the upstream one.

MIA ships its own configs and vectors under ``model_configs/mia_v0/`` and
``steering_vectors/mia_v0/`` so that upstream's files stay exactly as they are. A demo
asks for a name, not a directory, and gets MIA's version when one exists and upstream's
otherwise -- which is what lets a single demo run against both sets.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(top: str, rel: str) -> str:
    rel = rel.lstrip("/")
    ours = REPO_ROOT / top / "mia_v0" / rel
    theirs = REPO_ROOT / top / rel
    if ours.is_file():
        return str(ours)
    if theirs.is_file():
        return str(theirs)
    raise FileNotFoundError(
        f"{rel!r} is in neither {top}/mia_v0/ nor {top}/. MIA keeps its own copies under "
        f"{top}/mia_v0/ and falls back to the upstream file of the same name; add it to "
        f"one of the two, or pass an explicit path.")


def config_path(rel: str) -> str:
    """Absolute path for ``model_configs/<rel>``, preferring MIA's copy."""
    return _resolve("model_configs", rel)


def vector_path(rel: str) -> str:
    """Absolute path for ``steering_vectors/<rel>``, preferring MIA's copy."""
    return _resolve("steering_vectors", rel)


__all__ = ["REPO_ROOT", "config_path", "vector_path"]
