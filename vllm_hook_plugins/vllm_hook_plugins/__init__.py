from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.hook_client import HookClient
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker
from vllm_hook_plugins.workers.spotlight_worker import SpotlightWorker
from vllm_hook_plugins.workers.highlighter_worker import HighlighterWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from vllm_hook_plugins.analyzers.science_hallucination_analyzer import ScienceHallucinationAnalyzer
from vllm_hook_plugins.utils.spotlight.utils import generate_with_spotlight
from vllm_hook_plugins.utils.TokenHighlighter.utils import (
    analyze_with_highlighter,
    generate_with_highlighter,
    load_highlighter_config,
)
from vllm_hook_plugins.analyzers.highlighter_analyzer import HighlighterAnalyzer


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

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "HookClient",
    "ProbeHookQKWorker",
    "SteerHookActWorker",
    "ProbeHiddenStatesWorker",
    "SpotlightWorker",
    "HighlighterWorker",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "HiddenStatesAnalyzer",
    "ScienceHallucinationAnalyzer",
    "generate_with_spotlight",
    "generate_with_highlighter",
    "analyze_with_highlighter",
    "load_highlighter_config",
    "HighlighterAnalyzer",
    "register_plugins"
]
