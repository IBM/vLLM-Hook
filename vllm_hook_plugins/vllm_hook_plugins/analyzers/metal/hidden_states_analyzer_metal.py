from vllm_hook_plugins.analyzers import hidden_states_analyzer as hidden_module
from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from vllm_hook_plugins.metal.run_utils_metal import load_and_merge_hs_cache


class HiddenStatesAnalyzerMetal(HiddenStatesAnalyzer):
    def analyze(self, analyzer_spec=None, run_id=None, probes=None):
        if probes is not None:
            return super().analyze(
                analyzer_spec=analyzer_spec,
                run_id=run_id,
                probes=probes,
            )

        original_loader = hidden_module.load_and_merge_hs_cache
        hidden_module.load_and_merge_hs_cache = load_and_merge_hs_cache
        try:
            return super().analyze(analyzer_spec=analyzer_spec, run_id=run_id)
        finally:
            hidden_module.load_and_merge_hs_cache = original_loader
