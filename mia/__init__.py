"""MIA: capture and steer vLLM model internals; public API and plugin registration."""
from mia.registry import PluginRegistry
from mia.llm import MiaLLM
from mia.client import MiaClient
from mia.workers.qk_capture_worker import QKCaptureWorker
from mia.workers.steer_worker import SteerWorker
from mia.workers.hs_capture_worker import HSCaptureWorker
from mia.workers.spotlight_worker import SpotlightWorker
from mia.workers.highlighter_worker import HighlighterWorker
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.analyzers.core_reranker_analyzer import CorerAnalyzer
from mia.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from mia.analyzers.science_hallucination_analyzer import ScienceHallucinationAnalyzer
from mia.utils.spotlight.utils import generate_with_spotlight
from mia.utils.TokenHighlighter.utils import (
    analyze_with_highlighter,
    generate_with_highlighter,
    load_highlighter_config,
)
from mia.analyzers.highlighter_analyzer import HighlighterAnalyzer
from mia.analyzers.hnode_hallucination_analyzer import HNodeHallucinationAnalyzer


def register_plugins():
    PluginRegistry.register_worker("capture_qk",       QKCaptureWorker)
    PluginRegistry.register_worker("steer",      SteerWorker)
    PluginRegistry.register_worker("capture_hs", HSCaptureWorker)
    PluginRegistry.register_worker("spotlight",     SpotlightWorker)
    PluginRegistry.register_worker("token_highlighter",   HighlighterWorker)

    PluginRegistry.register_analyzer("attn_tracker",          AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker",         CorerAnalyzer)
    PluginRegistry.register_analyzer("hidden_states",         HiddenStatesAnalyzer)
    PluginRegistry.register_analyzer("science_hallucination", ScienceHallucinationAnalyzer)
    PluginRegistry.register_analyzer("token_highlighter",     HighlighterAnalyzer)
    PluginRegistry.register_analyzer("hnode_hallucination",   HNodeHallucinationAnalyzer)

__all__ = [
    "PluginRegistry",
    "MiaLLM",
    "MiaClient",
    "QKCaptureWorker",
    "SteerWorker",
    "HSCaptureWorker",
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
    "HNodeHallucinationAnalyzer",
    "register_plugins"
]

