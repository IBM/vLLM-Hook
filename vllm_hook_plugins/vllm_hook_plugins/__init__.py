from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.hook_client import HookClient
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker
from vllm_hook_plugins.workers.spotlight_worker import SpotlightWorker
from vllm_hook_plugins.workers.highlighter_worker import HighlighterWorker
from vllm_hook_plugins.workers.recurrent_depth_worker import RecurrentDepthWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from vllm_hook_plugins.analyzers.science_hallucination_analyzer import ScienceHallucinationAnalyzer
from vllm_hook_plugins.analyzers.recurrent_conv_analyzer import RecurrentConvergenceAnalyzer
from vllm_hook_plugins.utils.spotlight.utils import generate_with_spotlight
from vllm_hook_plugins.utils.TokenHighlighter.utils import (
    analyze_with_highlighter,
    generate_with_highlighter,
    load_highlighter_config,
)
from vllm_hook_plugins.analyzers.highlighter_analyzer import HighlighterAnalyzer
from vllm_hook_plugins.protocols.recurrent_depth import (
    RecurrentDepthProtocol,
    attach_recurrent_depth,
)
from vllm_hook_plugins.protocols.recurrent_config import (
    RecurrentDepthConfig,
    build_recurrent_stack,
    load_cls,
)
from vllm_hook_plugins.protocols.recurrent_step_controller import RecurrentStepController


def register_plugins():

    # Register workers
    PluginRegistry.register_worker("probe_hook_qk",       ProbeHookQKWorker)
    PluginRegistry.register_worker("steer_hook_act",      SteerHookActWorker)
    PluginRegistry.register_worker("probe_hidden_states", ProbeHiddenStatesWorker)
    PluginRegistry.register_worker("probe_spotlight",     SpotlightWorker)
    PluginRegistry.register_worker("token_highlighter",   HighlighterWorker)

    # Register analyzers
    PluginRegistry.register_analyzer("attn_tracker",          AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker",         CorerAnalyzer)
    PluginRegistry.register_analyzer("hidden_states",         HiddenStatesAnalyzer)
    PluginRegistry.register_analyzer("science_hallucination", ScienceHallucinationAnalyzer)
    PluginRegistry.register_analyzer("token_highlighter", HighlighterAnalyzer)
    # Recurrent depth runs in-process inside the model's recurrence loop (not a
    # WorkerExtension mixin). Analyzer registered for discovery / HookLLM naming.
    PluginRegistry.register_analyzer("recurrent_depth",       RecurrentConvergenceAnalyzer)

    # Register the adaptive Raven executor with vLLM's model registry so
    # HookLLM / vllm serve can host "RavenForCausalLM" checkpoints with the
    # adaptive-exit protocol. Lazy string path avoids importing vLLM layers
    # (and CUDA-initialising) in forked worker processes until a Raven model
    # is actually loaded. Guarded so environments without vLLM still register
    # the in-process workers/analyzers above.
    try:
        from vllm import ModelRegistry

        if "RavenForCausalLM" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "RavenForCausalLM",
                "model_adapters.vllm.adaptive_raven_vllm:AdaptiveRavenForvLLM",
            )
    except Exception:
        pass

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "HookClient",
    "ProbeHookQKWorker",
    "SteerHookActWorker",
    "ProbeHiddenStatesWorker",
    "SpotlightWorker",
    "HighlighterWorker",
    "RecurrentDepthWorker",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "HiddenStatesAnalyzer",
    "ScienceHallucinationAnalyzer",
    "RecurrentConvergenceAnalyzer",
    "RecurrentDepthProtocol",
    "RecurrentDepthConfig",
    "build_recurrent_stack",
    "load_cls",
    "RecurrentStepController",
    "attach_recurrent_depth",
    "generate_with_spotlight",
    "generate_with_highlighter",
    "analyze_with_highlighter",
    "load_highlighter_config",
    "HighlighterAnalyzer",
    "register_plugins",
]
