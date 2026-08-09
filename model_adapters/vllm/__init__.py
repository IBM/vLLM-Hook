"""Out-of-tree vLLM executors for recurrent-depth models.

``register_adaptive_raven`` is cheap (no CUDA). The executor module is imported
only when vLLM instantiates the registered architecture, or when a caller
explicitly imports ``AdaptiveRavenForvLLM``.
"""

from __future__ import annotations

from typing import Any

# Distinct from seal-rg's ``RavenForCausalLM``. Huginn checkpoints still
# advertise that name, so callers must also set
# ``hf_overrides={"architectures": [ADAPTIVE_RAVEN_ARCH], ...}``.
ADAPTIVE_RAVEN_ARCH = "AdaptiveRavenForvLLM"
_ADAPTIVE_RAVEN_PATH = "model_adapters.vllm.adaptive_raven_vllm:AdaptiveRavenForvLLM"


def register_adaptive_raven() -> None:
    """Register only the adaptive Raven vLLM executor (not the Hook worker/analyzer set)."""
    from vllm import ModelRegistry

    if ADAPTIVE_RAVEN_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ADAPTIVE_RAVEN_ARCH, _ADAPTIVE_RAVEN_PATH)


def __getattr__(name: str) -> Any:
    if name in {"AdaptiveRavenDecoderLayer", "AdaptiveRavenForvLLM", "AdaptiveRavenModel"}:
        from . import adaptive_raven_vllm as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ADAPTIVE_RAVEN_ARCH",
    "register_adaptive_raven",
]
