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
        analyzer_spec = analyzer_spec or {}
        debug = self._debug_enabled(analyzer_spec)
        run_id_file = os.environ.get("VLLM_RUN_ID")
        if debug:
            print(f"[attn analyzer] input_range={analyzer_spec['input_range']} attn_func={analyzer_spec['attn_func']}")

        attention_weights = self.compute_attention_from_qk(run_id_file, debug=debug)
        score, per_head_scores = self.attn2score(
            attention_weights,
            analyzer_spec['input_range'],
            analyzer_spec['attn_func'],
            debug=debug,
        )
        top_k = int(analyzer_spec.get("debug_top_k", 5))
        if debug:
            self._print_head_rankings(per_head_scores, top_k=top_k)

        return {
            "score": score,
            "per_head_scores": per_head_scores,
        }
        
            
    def _debug_enabled(self, analyzer_spec: Dict) -> bool:
        if "debug" in analyzer_spec:
            return bool(analyzer_spec["debug"])
        return os.environ.get("VLLM_HOOK_DEBUG", "") == "1"

    def compute_attention_from_qk(self, run_id_file: str, debug: bool = False) -> Dict[str, Dict]:

        run_id = latest_run_id(run_id_file)
        cache = load_and_merge_qk_cache(self.hook_dir, run_id)
        config = cache["config"]
        qk_cache = cache["qk_cache"]
        if debug:
            print(f"[attn analyzer] cached modules={list(qk_cache.keys())}")
        bs = len(next(iter(qk_cache.values()))['q'])
        if debug:
            print(f"[attn analyzer] inferred batch size from cache={bs}")
        batch_attention_weights = [dict() for _ in range(bs)]

        for layer_name, qk_data in qk_cache.items():
            layer_num = qk_data['layer_num']
                
            important_head_indices = self.layer_to_heads[layer_num]
            if debug:
                print(
                    f"[attn analyzer] layer={layer_num} module={layer_name} "
                    f"important_heads={important_head_indices}"
                )
            
            for i in range(bs):
                q_last = qk_data['q'][i]
                if q_last.ndim == 2:
                    q_last = q_last[-1]
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
    
    def attn2score(self, batch_attention: List[Dict[str, Dict]], batch_input_range: List[Tuple[Tuple[int, int], Tuple[int, int]]], attn_func: str = "sum_normalize", debug: bool = False) -> float:
        """Following https://github.com/khhung-906/Attention-Tracker/blob/main/detector/utils.py"""
        if not isinstance(batch_input_range, list):
            batch_input_range = [batch_input_range]

        batch_scores = []
        batch_per_head_scores = []
        for attention, input_range in zip(batch_attention, batch_input_range):
            scores = []
            sample_per_head_scores = []
            for layer_name, layer_data in attention.items():
                head_indices = layer_data['head_indices']
                attention_tensor = layer_data['attention']  # [num_heads, seq_len]
                
                for i, _ in enumerate(head_indices):
                    head_attention = attention_tensor[i, :].float().numpy()  # [seq_len]
                    
                    # Get instruction and data attention
                    inst_attn = head_attention[input_range[0][0]:input_range[0][1]]
                    data_attn = head_attention[input_range[1][0]:input_range[1][1]]
                    if debug:
                        print(
                            f"[attn analyzer] layer={layer_name} head={head_indices[i]} "
                            f"seq_len={len(head_attention)} "
                            f"inst_slice={input_range[0]} sum={np.sum(inst_attn):.6f} "
                            f"data_slice={input_range[1]} sum={np.sum(data_attn):.6f}"
                        )
                        print(
                            f"[attn analyzer] layer={layer_name} head={head_indices[i]} "
                            f"first10={head_attention[:10]}"
                        )
                    
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
                    sample_per_head_scores.append(
                        {
                            "layer_name": layer_name,
                            "layer_index": layer_data["layer_index"],
                            "head_index": head_indices[i],
                            "seq_len": len(head_attention),
                            "instruction_sum": float(np.sum(inst_attn)),
                            "data_sum": float(np.sum(data_attn)),
                            "score": float(score),
                            "first10": head_attention[:10].tolist(),
                        }
                    )
            batch_scores.append(np.mean(scores))
            batch_per_head_scores.append(sample_per_head_scores)
        return batch_scores, batch_per_head_scores

    def _print_head_rankings(self, batch_per_head_scores: List[List[Dict]], top_k: int = 5) -> None:
        for sample_idx, sample_scores in enumerate(batch_per_head_scores):
            if not sample_scores:
                print(f"[attn analyzer] sample={sample_idx} no captured head scores")
                continue

            ranked_by_score = sorted(
                sample_scores,
                key=lambda item: (item["score"], item["instruction_sum"]),
                reverse=True,
            )
            print(
                f"[attn analyzer] sample={sample_idx} top_{top_k}_heads_by_score="
                f"{[(item['layer_index'], item['head_index'], round(item['score'], 6), round(item['instruction_sum'], 6), round(item['data_sum'], 6)) for item in ranked_by_score[:top_k]]}"
            )

            ranked_by_inst = sorted(
                sample_scores,
                key=lambda item: item["instruction_sum"],
                reverse=True,
            )
            print(
                f"[attn analyzer] sample={sample_idx} top_{top_k}_heads_by_instruction_mass="
                f"{[(item['layer_index'], item['head_index'], round(item['instruction_sum'], 6), round(item['data_sum'], 6)) for item in ranked_by_inst[:top_k]]}"
            )
    
