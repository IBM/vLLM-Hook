"""Recurrent-depth model adapters, split by runtime.

- ``model_adapters.hf``   — HuggingFace adapters + upstream HF Raven sources
- ``model_adapters.vllm`` — out-of-tree vLLM executors (requires vLLM)

Public names are re-exported here so ``from model_adapters import ...`` still
works. Prefer the subpackages when importing a specific file.
"""

from .hf import (
    AdaptiveRavenForCausalLM,
    HuginnDynamicCache,
    RavenAdapterConfig,
    RavenBaselineConfig,
    RavenBaselineExit,
    RavenForCausalLM,
    RowSliceCacheProxy,
    block_geometry_from_config,
)

__all__ = [
    "AdaptiveRavenForCausalLM",
    "RavenAdapterConfig",
    "RavenBaselineConfig",
    "RavenBaselineExit",
    "RavenForCausalLM",
    "HuginnDynamicCache",
    "RowSliceCacheProxy",
    "block_geometry_from_config",
]

# vLLM executor path is optional so the HF oracle works without vLLM installed.
try:
    from .vllm import AdaptiveRavenForvLLM, AdaptiveRavenModel

    __all__ += ["AdaptiveRavenForvLLM", "AdaptiveRavenModel"]
except Exception:  # pragma: no cover - vLLM optional for HF-only usage
    pass
