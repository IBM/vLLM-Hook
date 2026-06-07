import numpy as np

from vllm_hook_plugins.analyzers import attention_tracker_analyzer as attn_module
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.metal.run_utils_metal import load_and_merge_qk_cache


class AttntrackerAnalyzerMetal(AttntrackerAnalyzer):
    def compute_attention_from_qk(self, run_id: str = None, probes=None):
        if probes is not None:
            return super().compute_attention_from_qk(run_id, probes=probes)

        original_loader = attn_module.load_and_merge_qk_cache
        attn_module.load_and_merge_qk_cache = load_and_merge_qk_cache
        try:
            return super().compute_attention_from_qk(run_id)
        finally:
            attn_module.load_and_merge_qk_cache = original_loader

    def attn2score(self, batch_attention, batch_input_range, attn_func="sum_normalize"):
        if not isinstance(batch_input_range, list):
            batch_input_range = [batch_input_range]

        batch_scores = []
        for attention, input_range in zip(batch_attention, batch_input_range):
            scores = []
            for _, layer_data in attention.items():
                head_indices = layer_data["head_indices"]
                attention_tensor = layer_data["attention"]

                for i, _ in enumerate(head_indices):
                    head_attention = attention_tensor[i, :].float().numpy()

                    inst_attn = head_attention[input_range[0][0]:input_range[0][1]]
                    data_attn = head_attention[input_range[1][0]:input_range[1][1]]

                    if "sum" in attn_func:
                        score = np.sum(inst_attn)
                    elif "max" in attn_func:
                        score = np.max(inst_attn)
                    else:
                        raise NotImplementedError

                    if "normalize" in attn_func:
                        total = np.sum(inst_attn) + np.sum(data_attn) + 1e-8
                        score = score / total

                    scores.append(score)
            batch_scores.append(np.mean(scores))
        return batch_scores
