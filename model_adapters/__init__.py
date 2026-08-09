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

from .vllm import ADAPTIVE_RAVEN_ARCH, register_adaptive_raven

__all__ += ["ADAPTIVE_RAVEN_ARCH", "register_adaptive_raven"]

# Executor classes pull in vLLM; optional so the HF oracle still imports.
try:
    from .vllm import AdaptiveRavenForvLLM, AdaptiveRavenModel

    __all__ += ["AdaptiveRavenForvLLM", "AdaptiveRavenModel"]
except Exception:  # pragma: no cover - vLLM optional for HF-only usage
    pass
