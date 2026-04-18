from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker
from vllm_hook_plugins.workers.highlighter_worker import HighlighterWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from vllm_hook_plugins.analyzers.highlighter_analyzer import HighlighterAnalyzer


def register_plugins():

    # Register workers
    PluginRegistry.register_worker("probe_hook_qk",       ProbeHookQKWorker,       hooks_on=(True,  False))
    PluginRegistry.register_worker("steer_hook_act",      SteerHookActWorker,      hooks_on=(False, True))
    PluginRegistry.register_worker("probe_hidden_states", ProbeHiddenStatesWorker, hooks_on=(True,  False))
    PluginRegistry.register_worker("token_highlighter",   HighlighterWorker,       hooks_on=(True,  True))

    # Register analyzers
    PluginRegistry.register_analyzer("attn_tracker",   AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker",  CorerAnalyzer)
    PluginRegistry.register_analyzer("hidden_states",  HiddenStatesAnalyzer)
    PluginRegistry.register_analyzer("token_highlighter", HighlighterAnalyzer)

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "ProbeHookQKWorker",
    "SteerHookActWorker",
    "ProbeHiddenStatesWorker",
    "HighlighterWorker",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "HiddenStatesAnalyzer",
    "HighlighterAnalyzer",
    "register_plugins"
]