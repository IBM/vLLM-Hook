import os
import torch
from typing import Dict, List, Optional

from vllm_hook_plugins.run_utils import latest_run_id, load_and_merge_hs_cache
from vllm_hook_plugins.shm_utils import load_from_shm


class HiddenStatesAnalyzer:

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir

    def analyze(self, analyzer_spec: Optional[Dict] = None) -> Dict:
        if os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            hs_cache = load_from_shm(self.hook_dir, os.environ.get("VLLM_RUN_ID"))
        else:
            run_id_file = os.environ.get("VLLM_RUN_ID")
            run_id = latest_run_id(run_id_file)
            cache = load_and_merge_hs_cache(self.hook_dir, run_id)
            hs_cache = cache["hs_cache"]

        reduce = (analyzer_spec or {}).get("reduce", "none")

        result = {}
        for layer_name, data in hs_cache.items():
            tensors: List[torch.Tensor] = data["hidden_states"]
            if reduce == "none":
                result[layer_name] = tensors
            elif reduce == "mean":
                # meaningful for all_tokens mode: average over sequence dim
                result[layer_name] = [
                    t.mean(dim=0) if t.dim() > 1 else t for t in tensors
                ]
            elif reduce == "norm":
                result[layer_name] = [
                    torch.norm(t.float()).item() for t in tensors
                ]
            else:
                raise NotImplementedError(f"Unknown reduce: {reduce}")

        return {"hidden_states": result}
