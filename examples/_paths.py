"""Resolve a name under ``model_configs/`` or ``steering_vectors/`` to an absolute path.

Configs resolve from anywhere. A config's ``vector_path`` and the demos' ``./cache`` are relative
to the working directory, so run the demos (and ``vllm serve``) from the repo root.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(top: str, rel: str) -> str:
    path = REPO_ROOT / top / rel.lstrip("/")
    if path.is_file():
        return str(path)
    raise FileNotFoundError(
        f"{rel!r} is not under {top}/. Add it there, or pass an explicit path.")


def config_path(rel: str) -> str:
    """Absolute path for ``model_configs/<rel>``."""
    return _resolve("model_configs", rel)


def vector_path(rel: str) -> str:
    """Absolute path for ``steering_vectors/<rel>``."""
    return _resolve("steering_vectors", rel)


__all__ = ["REPO_ROOT", "config_path", "vector_path"]
