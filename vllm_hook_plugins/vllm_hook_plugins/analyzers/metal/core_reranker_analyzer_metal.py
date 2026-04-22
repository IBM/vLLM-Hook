from vllm_hook_plugins.analyzers import core_reranker_analyzer as core_module
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
from vllm_hook_plugins.metal.run_utils_metal import load_and_merge_qk_cache


class CorerAnalyzerMetal(CorerAnalyzer):
    def score_documents(self, *args, **kwargs):
        original_loader = core_module.load_and_merge_qk_cache
        core_module.load_and_merge_qk_cache = load_and_merge_qk_cache
        try:
            return super().score_documents(*args, **kwargs)
        finally:
            core_module.load_and_merge_qk_cache = original_loader
