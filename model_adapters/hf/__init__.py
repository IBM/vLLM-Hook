"""HuggingFace adapters and upstream HF Raven / retrofitted-Llama sources."""

from .looped_llama_adapter import (
    AdaptiveRavenForCausalLM,
    RavenAdapterConfig,
    RowSliceCacheProxy,
    block_geometry_from_config,
)
from .raven_baseline_exit import RavenBaselineConfig, RavenBaselineExit
from .raven_modeling_minimal_llama import HuginnDynamicCache, RavenForCausalLM

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
