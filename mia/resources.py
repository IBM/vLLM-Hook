"""Locate the example model configs and steering vectors shipped with the package.

An installed wheel carries them under ``mia/data/``; a source checkout has them at the repo root.
"""
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def _resolve(top: str, rel: str) -> str:
    for base in (_PKG / "data", _PKG.parent):
        path = base / top / rel.lstrip("/")
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"{rel!r} not found under {top}/ (package data or repo root)")


def config_path(rel: str) -> str:
    """Absolute path of ``model_configs/<rel>``, e.g. ``hidden_states/Qwen2-1.5B-Instruct.json``."""
    return _resolve("model_configs", rel)


def vector_path(rel: str) -> str:
    """Absolute path of ``steering_vectors/<rel>``, e.g. ``phi3_format.pt``."""
    return _resolve("steering_vectors", rel)
