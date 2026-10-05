"""Hidden-states analyzer: loads captured hidden states and applies a reduction."""
from typing import Dict, List, Optional

import torch

from mia._profiler import PROF
from mia.run_utils import load_and_merge_hs_cache


class HiddenStatesAnalyzer:
    """Captured hidden states per layer, optionally reduced per prompt."""
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        """``hook_dir`` holds the disk runs; ``layer_to_heads`` is unused."""
        self.hook_dir = hook_dir

    def analyze(self, analyzer_spec: Optional[Dict] = None, run_id: Optional[str] = None, probes: Optional[Dict] = None) -> Dict:
        """``{"hidden_states": {layer: [...]}}``; spec ``reduce`` is none, mean or norm."""
        if probes is not None:
            hs_cache = probes["hs_cache"]
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
                        None if t is None else t.mean(dim=0) if t.dim() > 1 else t
                        for t in tensors
                    ]
                elif reduce == "norm":
                    result[layer_name] = [
                        None if t is None else torch.norm(t.float()).item() for t in tensors
                    ]
                else:
                    raise NotImplementedError(f"Unknown reduce: {reduce}")

        return {"hidden_states": result}

