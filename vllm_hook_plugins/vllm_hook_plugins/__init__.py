"""vllm-hook-plugins package root.

Public names are re-exported lazily (PEP 562): the workers, analyzers,
and wrappers import vLLM at module level, so resolving them eagerly here
would make ``import vllm_hook_plugins.core`` — the engine-free surface —
impossible on installs without vLLM. Attribute access is unchanged for
callers: ``from vllm_hook_plugins import HookLLM`` still works.
"""
import importlib

_LAZY_ATTRS = {
    "PluginRegistry": "vllm_hook_plugins.registry",
    "HookLLM": "vllm_hook_plugins.hook_llm",
    "HookClient": "vllm_hook_plugins.hook_client",
    "ProbeHookQKWorker": "vllm_hook_plugins.workers.probe_hookqk_worker",
    "SteerHookActWorker": "vllm_hook_plugins.workers.steer_activation_worker",
    "ProbeHiddenStatesWorker": "vllm_hook_plugins.workers.probe_hidden_states_worker",
    "SpotlightWorker": "vllm_hook_plugins.workers.spotlight_worker",
    "HighlighterWorker": "vllm_hook_plugins.workers.highlighter_worker",
    "UnifiedHookWorker": "vllm_hook_plugins.workers.unified_worker",
    "AttntrackerAnalyzer": "vllm_hook_plugins.analyzers.attention_tracker_analyzer",
    "CorerAnalyzer": "vllm_hook_plugins.analyzers.core_reranker_analyzer",
    "HiddenStatesAnalyzer": "vllm_hook_plugins.analyzers.hidden_states_analyzer",
    "ScienceHallucinationAnalyzer": "vllm_hook_plugins.analyzers.science_hallucination_analyzer",
    "HighlighterAnalyzer": "vllm_hook_plugins.analyzers.highlighter_analyzer",
    "HNodeHallucinationAnalyzer": "vllm_hook_plugins.analyzers.hnode_hallucination_analyzer",
    "generate_with_spotlight": "vllm_hook_plugins.utils.spotlight.utils",
    "analyze_with_highlighter": "vllm_hook_plugins.utils.TokenHighlighter.utils",
    "generate_with_highlighter": "vllm_hook_plugins.utils.TokenHighlighter.utils",
    "load_highlighter_config": "vllm_hook_plugins.utils.TokenHighlighter.utils",
}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(importlib.import_module(module_name), name)
    globals()[name] = attr
    return attr


def register_plugins():
    from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
    from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
    from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
    from vllm_hook_plugins.analyzers.highlighter_analyzer import HighlighterAnalyzer
    from vllm_hook_plugins.analyzers.hnode_hallucination_analyzer import HNodeHallucinationAnalyzer
    from vllm_hook_plugins.analyzers.science_hallucination_analyzer import ScienceHallucinationAnalyzer
    from vllm_hook_plugins.registry import PluginRegistry
    from vllm_hook_plugins.workers.highlighter_worker import HighlighterWorker
    from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker
    from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
    from vllm_hook_plugins.workers.spotlight_worker import SpotlightWorker
    from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
    from vllm_hook_plugins.workers.unified_worker import UnifiedHookWorker

    # Register workers
    PluginRegistry.register_worker("probe_hook_qk", ProbeHookQKWorker)
    PluginRegistry.register_worker("steer_hook_act", SteerHookActWorker)
    PluginRegistry.register_worker("probe_hidden_states", ProbeHiddenStatesWorker)
    PluginRegistry.register_worker("probe_spotlight", SpotlightWorker)
    PluginRegistry.register_worker("token_highlighter", HighlighterWorker)
    PluginRegistry.register_worker("unified", UnifiedHookWorker)

    # Register analyzers
    PluginRegistry.register_analyzer("attn_tracker", AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker", CorerAnalyzer)
    PluginRegistry.register_analyzer("hidden_states", HiddenStatesAnalyzer)
    PluginRegistry.register_analyzer("science_hallucination", ScienceHallucinationAnalyzer)
    PluginRegistry.register_analyzer("token_highlighter",     HighlighterAnalyzer)
    PluginRegistry.register_analyzer("hnode_hallucination",   HNodeHallucinationAnalyzer)

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "HookClient",
    "ProbeHookQKWorker",
    "SteerHookActWorker",
    "ProbeHiddenStatesWorker",
    "SpotlightWorker",
    "HighlighterWorker",
    "UnifiedHookWorker",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "HiddenStatesAnalyzer",
    "ScienceHallucinationAnalyzer",
    "generate_with_spotlight",
    "generate_with_highlighter",
    "analyze_with_highlighter",
    "load_highlighter_config",
    "HighlighterAnalyzer",
    "HNodeHallucinationAnalyzer",
    "register_plugins"
]
