from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.workers.spotlight_worker import SpotlightWorker
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from vllm_hook_plugins.utils.spotlight.utils import generate_with_spotlight


def register_plugins():

    # Register workers
    PluginRegistry.register_worker("probe_hook_qk",       ProbeHookQKWorker,       hooks_on=(True,  False))
    PluginRegistry.register_worker("steer_hook_act",      SteerHookActWorker,      hooks_on=(False, True))
    PluginRegistry.register_worker("probe_spotlight",     SpotlightWorker,         hooks_on=(True,  True))
    PluginRegistry.register_worker("probe_hidden_states", ProbeHiddenStatesWorker, hooks_on=(True,  False))

    # Register analyzers
    PluginRegistry.register_analyzer("attn_tracker",   AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker",  CorerAnalyzer)
    PluginRegistry.register_analyzer("hidden_states",  HiddenStatesAnalyzer)

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "ProbeHookQKWorker",
    "SteerHookActWorker",
    "SpotlightWorker",
    "ProbeHiddenStatesWorker",
    "generate_with_spotlight",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "HiddenStatesAnalyzer",
    "register_plugins"
]