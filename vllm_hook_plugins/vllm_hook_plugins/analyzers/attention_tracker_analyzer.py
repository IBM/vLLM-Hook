import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List

from vllm_hook_plugins.run_utils import latest_run_id, load_and_merge_qk_cache


class AttntrackerAnalyzer:
    
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        self.layer_to_heads = layer_to_heads
    
    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None
    ) -> Optional[Dict]:
        
        run_id_file = os.environ.get("VLLM_RUN_ID")

        attention_weights = self.compute_attention_from_qk(run_id_file)
        score = self.attn2score(attention_weights, analyzer_spec['input_range'], analyzer_spec['attn_func'])

        return {
            "score": score
        }
        
            
    def compute_attention_from_qk(self, run_id_file: str) -> Dict[str, Dict]:

        run_id = latest_run_id(run_id_file)
        cache = load_and_merge_qk_cache(self.hook_dir, run_id)
        config = cache["config"]
        qk_cache = cache["qk_cache"]
        bs = len(next(iter(qk_cache.values()))['q'])
        batch_attention_weights = [dict() for _ in range(bs)]

        for layer_name, qk_data in qk_cache.items():
            layer_num = qk_data['layer_num']
                
            important_head_indices = self.layer_to_heads[layer_num]
            
            for i in range(bs):
                q_last = qk_data['q'][i]  # [4096]
                k_all = qk_data['k_all'][i]    # [seq_len, 1024]
                
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
                head_indices = layer_data['head_indices']
                attention_tensor = layer_data['attention']  # [num_heads, seq_len]
                
                for i, _ in enumerate(head_indices):
                    head_attention = attention_tensor[i, :].numpy()  # [seq_len]
                    
                    # Get instruction and data attention
                    inst_attn = head_attention[input_range[0][0]:input_range[0][1]]
                    data_attn = head_attention[input_range[1][0]:input_range[1][1]]
                    
                    # Calculate score based on function
                    if "sum" in attn_func:
                        score = np.sum(inst_attn)
                    elif "max" in attn_func:
                        score = np.max(inst_attn)
                    else: raise NotImplementedError
                    
                    if "normalize" in attn_func:
                        total = np.sum(inst_attn) + np.sum(data_attn) + 1e-8
                        score = score / total
                    
                    scores.append(score)
            batch_scores.append(np.mean(scores))
        return batch_scores
    