"""Recurrent-depth model adapters, split by runtime.

- ``model_adapters.hf``   — HuggingFace adapters + upstream HF Raven sources
- ``model_adapters.vllm`` — out-of-tree vLLM executors (requires vLLM)

Public names are re-exported **lazily** so ``from model_adapters.vllm import ...``
does not import the HF stack (or ``hf.raven_config_minimal``).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AdaptiveRavenForCausalLM",
    "RavenAdapterConfig",
    "RavenBaselineConfig",
    "RavenBaselineExit",
    "RavenForCausalLM",
    "HuginnDynamicCache",
    "RowSliceCacheProxy",
    "block_geometry_from_config",
    "ADAPTIVE_RAVEN_ARCH",
    "register_adaptive_raven",
    "AdaptiveRavenForvLLM",
    "AdaptiveRavenModel",
]

_HF_ATTRS = frozenset(
    {
        "AdaptiveRavenForCausalLM",
        "RavenAdapterConfig",
        "RavenBaselineConfig",
        "RavenBaselineExit",
        "RavenForCausalLM",
        "HuginnDynamicCache",
        "RowSliceCacheProxy",
        "block_geometry_from_config",
    }
)
_VLLM_ATTRS = frozenset(
    {
        "ADAPTIVE_RAVEN_ARCH",
        "register_adaptive_raven",
        "AdaptiveRavenForvLLM",
        "AdaptiveRavenModel",
    }
)


def __getattr__(name: str) -> Any:
    if name in _HF_ATTRS:
        from . import hf as _hf

        return getattr(_hf, name)
    if name in _VLLM_ATTRS:
        from . import vllm as _vllm

        return getattr(_vllm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(__all__))
