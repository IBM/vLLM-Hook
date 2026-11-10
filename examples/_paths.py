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
    try:  # not in a checkout: fall back to the files packaged in the wheel
        from mia import resources
        return resources._resolve(top, rel)
    except (ImportError, FileNotFoundError):
        raise FileNotFoundError(
            f"{rel!r} is not under {top}/. Add it there, or pass an explicit path.") from None


def config_path(rel: str) -> str:
    """Absolute path for ``model_configs/<rel>`` (repo checkout or installed package)."""
    return _resolve("model_configs", rel)


def vector_path(rel: str) -> str:
    """Absolute path for ``steering_vectors/<rel>`` (repo checkout or installed package)."""
    return _resolve("steering_vectors", rel)


def config_dir(rel: str) -> Path:
    """Directory ``model_configs/<rel>`` (repo checkout or installed package)."""
    path = REPO_ROOT / "model_configs" / rel
    if path.is_dir():
        return path
    try:
        from mia import resources
        packaged = Path(resources.__file__).resolve().parent / "data" / "model_configs" / rel
        if packaged.is_dir():
            return packaged
    except ImportError:
        pass
    return path
