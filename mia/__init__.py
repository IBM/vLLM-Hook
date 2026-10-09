"""MIA: capture and steer vLLM model internals; public API and plugin registration."""
from mia.registry import PluginRegistry
from mia.llm import MiaLLM
from mia.client import MiaClient
from mia.workers.qk_capture_worker import QKCaptureWorker
from mia.workers.steer_worker import SteerWorker
from mia.workers.hs_capture_worker import HSCaptureWorker
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.analyzers.core_reranker_analyzer import CorerAnalyzer
from mia.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from mia.analyzers.science_hallucination_analyzer import ScienceHallucinationAnalyzer
from mia.analyzers.hnode_hallucination_analyzer import HNodeHallucinationAnalyzer
from mia.analyzers.attnlink_analyzer import AttnLinkAnalyzer
from mia.analyzers.tool_call_risk_analyzer import ToolCallRiskAnalyzer


def register_plugins():
    """Register MIA's workers and analyzers in ``PluginRegistry`` (vLLM plugin entry point)."""
    PluginRegistry.register_worker("capture_qk",       QKCaptureWorker)
    PluginRegistry.register_worker("steer",      SteerWorker)
    PluginRegistry.register_worker("capture_hs", HSCaptureWorker)

    PluginRegistry.register_analyzer("attnlink",              AttnLinkAnalyzer)
    PluginRegistry.register_analyzer("attn_tracker",          AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker",         CorerAnalyzer)
    PluginRegistry.register_analyzer("hidden_states",         HiddenStatesAnalyzer)
    PluginRegistry.register_analyzer("science_hallucination", ScienceHallucinationAnalyzer)
    PluginRegistry.register_analyzer("hnode_hallucination",   HNodeHallucinationAnalyzer)
    PluginRegistry.register_analyzer("tool_call_risk",        ToolCallRiskAnalyzer)

__all__ = [
    "PluginRegistry",
    "MiaLLM",
    "MiaClient",
    "QKCaptureWorker",
    "SteerWorker",
    "HSCaptureWorker",
    "AttntrackerAnalyzer",
    "AttnLinkAnalyzer",
    "CorerAnalyzer",
    "HiddenStatesAnalyzer",
    "ScienceHallucinationAnalyzer",
    "HNodeHallucinationAnalyzer",
    "ToolCallRiskAnalyzer",
    "register_plugins"
]
