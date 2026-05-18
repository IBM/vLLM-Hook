import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List

from vllm_hook_plugins.run_utils import load_and_merge_qk_cache, unpack_qk


class AttntrackerAnalyzer:
    
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        self.layer_to_heads = layer_to_heads
    
    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        probes: Optional[Dict] = None,
    ) -> Optional[Dict]:

        attention_weights = self.compute_attention_from_qk(run_id, probes=probes)
        score = self.attn2score(attention_weights, analyzer_spec['input_range'], analyzer_spec['attn_func'])

        return {
            "score": score
        }


    def compute_attention_from_qk(self, run_id: str = None, probes: Optional[Dict] = None) -> Dict[str, Dict]:

        if probes is not None:
            config = probes["config"]
            qk_cache = probes["qk_cache"]
        else:
            if run_id is None:
                raise ValueError("compute_attention_from_qk: pass either probes= or run_id=.")
            cache = load_and_merge_qk_cache(self.hook_dir, run_id)
            config = cache["config"]
            qk_cache = cache["qk_cache"]
        batch_attention_weights = None

        for layer_name, qk_data in qk_cache.items():
            layer_num = qk_data['layer_num']
            important_head_indices = self.layer_to_heads[layer_num]
            q_list, k_list = unpack_qk(qk_data)

            if batch_attention_weights is None:
                batch_attention_weights = [dict() for _ in range(len(q_list))]

            for i, (q_last, k_all) in enumerate(zip(q_list, k_list)):
                
                seq_len = k_all.shape[0]
                
                # Reshape to heads
                q_heads = q_last.view(config["num_attention_heads"], config["head_dim"])
                q_heads = q_heads.unsqueeze(0).unsqueeze(2)  # [1, 32, 1, 128]
                
                k_heads = k_all.view(seq_len, config["num_key_value_heads"], config["head_dim"])
                k_heads = k_heads.permute(1, 0, 2).unsqueeze(0)  # [1, 8, seq_len, 128]
                
                if config["num_key_value_heads"] < config["num_attention_heads"]:
                    num_repeat = config["num_attention_heads"] // config["num_key_value_heads"]
                    k_heads = k_heads.repeat_interleave(num_repeat, dim=1)  # [1, 32, seq_len, 128]
                
                scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * config["attention_multiplier"]
                
                full_attention = F.softmax(scores, dim=-1).squeeze(2).squeeze(0)  # [32, seq_len]
                
                # filter to only important heads
                filtered_attention = full_attention[important_head_indices, :]  # [num_important_heads, seq_len]
                
                batch_attention_weights[i][layer_name] = {
                    'attention': filtered_attention,
                    'head_indices': important_head_indices,
                    'layer_index': layer_num
                }
        
        return batch_attention_weights
    
    def attn2score(self, batch_attention: List[Dict[str, Dict]], batch_input_range: List[Tuple[Tuple[int, int], Tuple[int, int]]], attn_func: str = "sum_normalize") -> float:
        """Following https://github.com/khhung-906/Attention-Tracker/blob/main/detector/utils.py"""
        if not isinstance(batch_input_range, list):
            batch_input_range = [batch_input_range]

        batch_scores = []
        for attention, input_range in zip(batch_attention, batch_input_range):
            scores = []
            for _, layer_data in attention.items():
                attn_np = layer_data['attention'].numpy()  # [num_heads, seq_len] — single transfer

                inst_attn = attn_np[:, input_range[0][0]:input_range[0][1]]  # [num_heads, inst_len]
                data_attn = attn_np[:, input_range[1][0]:input_range[1][1]]  # [num_heads, data_len]

                if "sum" in attn_func:
                    head_scores = inst_attn.sum(axis=1)
                elif "max" in attn_func:
                    head_scores = inst_attn.max(axis=1)
                else:
                    raise NotImplementedError

                if "normalize" in attn_func:
                    total = inst_attn.sum(axis=1) + data_attn.sum(axis=1) + 1e-8
                    head_scores = head_scores / total

                scores.extend(head_scores.tolist())
            batch_scores.append(np.mean(scores))
        return batch_scores
    