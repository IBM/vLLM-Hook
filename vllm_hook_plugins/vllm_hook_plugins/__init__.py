import platform
from importlib.util import find_spec

from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer


def _can_register_metal_worker() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and find_spec("vllm_metal") is not None
        and find_spec("mlx") is not None
    )


def register_plugins():
    # Register workers
    PluginRegistry.register_worker("probe_hook_qk", ProbeHookQKWorker)
    if _can_register_metal_worker():
        try:
            from vllm_hook_plugins.workers.metal import ProbeHookQKWorkerMetal
        except Exception:
            ProbeHookQKWorkerMetal = None
        if ProbeHookQKWorkerMetal is not None:
            PluginRegistry.register_worker("probe_hook_qk_metal", ProbeHookQKWorkerMetal)
    PluginRegistry.register_worker("steer_hook_act", SteerHookActWorker)

    # Register analyzers
    PluginRegistry.register_analyzer("attn_tracker", AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker", CorerAnalyzer)

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "ProbeHookQKWorker", 
    "SteerHookActWorker",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "register_plugins"
]
