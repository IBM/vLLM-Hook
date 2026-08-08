"""Out-of-tree vLLM executors for recurrent-depth models.

Importing this subpackage requires vLLM. The plugin registry uses the lazy
string ``model_adapters.vllm.adaptive_raven_vllm:AdaptiveRavenForvLLM`` so
CUDA layers are not initialized until a Raven checkpoint is actually loaded.
"""

from .adaptive_raven_vllm import (
    AdaptiveRavenDecoderLayer,
    AdaptiveRavenForvLLM,
    AdaptiveRavenModel,
)

__all__ = [
    "AdaptiveRavenDecoderLayer",
    "AdaptiveRavenForvLLM",
    "AdaptiveRavenModel",
]
