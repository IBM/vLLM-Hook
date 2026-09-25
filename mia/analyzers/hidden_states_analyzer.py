"""Hidden-states analyzer: loads captured hidden states and applies a reduction."""
import os
import torch
from typing import Dict, List, Optional

from mia._profiler import PROF
from mia.run_utils import load_and_merge_hs_cache
from mia.shm_utils import load_from_shm


class HiddenStatesAnalyzer:
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir

    def analyze(self, analyzer_spec: Optional[Dict] = None, run_id: Optional[str] = None, probes: Optional[Dict] = None) -> Dict:
        peak_gpu_mb = None
        if probes is not None:
            hs_cache = probes["hs_cache"]
        elif os.environ.get("MIA_USE_SHM", "0") == "1":
            with PROF.timed("io.shm_load"):
                hs_cache, peak_gpu_mb = load_from_shm(self.hook_dir, run_id)
        else:
            if run_id is None:
                raise ValueError("HiddenStatesAnalyzer.analyze: pass either probes= or run_id=.")
            cache = load_and_merge_hs_cache(self.hook_dir, run_id)
            hs_cache = cache["hs_cache"]

        reduce = (analyzer_spec or {}).get("reduce", "none")

        with PROF.timed("analyzer.kernel"):
            result = {}
            for layer_name, data in hs_cache.items():
                tensors: List[torch.Tensor] = data["hidden_states"]
                if reduce == "none":
                    result[layer_name] = tensors
                elif reduce == "mean":
                    result[layer_name] = [
                        t.mean(dim=0) if t.dim() > 1 else t for t in tensors
                    ]
                elif reduce == "norm":
                    result[layer_name] = [
                        torch.norm(t.float()).item() for t in tensors
                    ]
                else:
                    raise NotImplementedError(f"Unknown reduce: {reduce}")

        out = {"hidden_states": result}
        if peak_gpu_mb is not None:
            out["peak_gpu_mb"] = peak_gpu_mb
        return out

