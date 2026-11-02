"""Hidden-states analyzer: loads captured hidden states and applies a reduction."""
from typing import Dict, List, Optional

import torch

from mia._profiler import PROF
from mia.artifacts import load_and_merge_hs_cache


def _real_rows(data: Dict, tensors) -> List[Optional[int]]:
    """Real (unpadded) row count of each pass in ``tensors``.

    A list holds unpadded passes. A padded ``(passes, rows, hidden)`` tensor carries its counts in
    ``pass_lens``; artifacts written before ``pass_lens`` existed fall back to the last non-zero row.
    """
    if isinstance(tensors, (list, tuple)):
        return [None if t is None else (int(t.shape[0]) if t.dim() > 1 else 1) for t in tensors]
    lens = data.get("pass_lens")
    if lens is not None:
        return [int(n) for n in lens]
    if tensors.dim() < 3:
        return [1] * int(tensors.shape[0])
    nz = tensors.reshape(tensors.shape[0], tensors.shape[1], -1).abs().sum(-1) != 0
    idx = torch.arange(1, tensors.shape[1] + 1)
    return [int((nz[i] * idx).max()) for i in range(tensors.shape[0])]


class HiddenStatesAnalyzer:
    """Captured hidden states per layer, optionally reduced per prompt."""
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        """``hook_dir`` holds the disk runs; ``layer_to_heads`` is unused."""
        self.hook_dir = hook_dir

    def analyze(self, analyzer_spec: Optional[Dict] = None, run_id: Optional[str] = None, probes: Optional[Dict] = None) -> Dict:
        """``{"hidden_states": {layer: [...]}}``; spec ``reduce`` is none, mean, mean_tokens or norm.

        ``mean`` is one vector per forward pass over that pass's real rows (padding excluded).
        ``mean_tokens`` is one vector per request over every real row of every pass, so it does not
        depend on how chunked prefill split the prompt (all_tokens mode only).
        """
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
                    lens = _real_rows(data, tensors)
                    result[layer_name] = [
                        None if t is None else t[:n or None].mean(dim=0) if t.dim() > 1 else t
                        for t, n in zip(tensors, lens)
                    ]
                elif reduce == "mean_tokens":
                    if data.get("hs_mode", "all_tokens") == "last_token":
                        raise ValueError("reduce='mean_tokens' needs hs_mode='all_tokens'")
                    lens = _real_rows(data, tensors)
                    rows = [t[:n].reshape(n, -1) for t, n in zip(tensors, lens) if t is not None and n]
                    result[layer_name] = ([torch.cat(rows).float().mean(dim=0).to(rows[0].dtype)]
                                          if rows else [None])
                elif reduce == "norm":
                    result[layer_name] = [
                        None if t is None else torch.norm(t.float()).item() for t in tensors
                    ]
                else:
                    raise NotImplementedError(f"Unknown reduce: {reduce}")

        return {"hidden_states": result}
