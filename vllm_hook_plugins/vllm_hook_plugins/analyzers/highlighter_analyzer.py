import os
import glob
from typing import Dict, List, Optional

import torch

from vllm_hook_plugins.run_utils import latest_run_id


class HighlighterAnalyzer:
    """Read token gradient norms and return compliance-driving tokens
    from the HighlighterWorker (those with large affirmation gradients)."""

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        self.layer_to_heads = layer_to_heads

    def analyze(self, analyzer_spec: Optional[Dict] = None) -> Optional[Dict]:

        # Retrieve the run id file from the environment variable
        run_id_file = os.environ.get("VLLM_RUN_ID")
        if not run_id_file:
            return None

        run_id = latest_run_id(run_id_file)
        trace = self._load_gradient_trace(run_id)
        if trace is None:
            return {"run_id": run_id, "results": []}

        cfg = analyzer_spec or {}
        top_k = int(cfg.get("top_k", 5))

        # Potential feature/update: Optional analyzer-side override should re-rank token_scores only
        # (for report flexibility) while keeping worker-applied soft_indices intact.
        # Reporting highlighting results per prompt.
        results = [] 
        for seq in trace.get("sequences", []):
            scores = seq.get("token_scores", [])
            if not scores:
                continue
            drivers = seq.get("soft_indices", [])
            analysis_drivers = self._select_analysis_drivers(scores, cfg, drivers)
            token_ids = seq.get("token_ids", [])
            tokens = seq.get("tokens", [])
            if not tokens:
                tokens = [str(tid) for tid in token_ids]
            soft_beta = seq.get("soft_beta")
            soft_set = set(drivers)
            analysis_set = set(analysis_drivers)
            token_view = [
                {
                    "idx": i,
                    "token_id": token_ids[i] if i < len(token_ids) else None,
                    "token": tokens[i],
                    "score": scores[i],
                    "scaled": i in soft_set,
                    "analysis_selected": i in analysis_set,
                    "scale_factor": soft_beta if i in soft_set else 1.0,
                }
                for i in range(len(scores))
            ]
            softened_tokens = [
                f"{tok} [x{soft_beta}]" if i in soft_set else tok
                for i, tok in enumerate(tokens)
            ]
            top = self._top_tokens(token_ids, tokens, scores, top_k=top_k)
            results.append({
                "token_ids": token_ids,
                "tokens": tokens,
                "softened_tokens_view": softened_tokens,
                "token_view": token_view,
                "token_scores": scores,
                "drivers": drivers,
                "analysis_drivers": analysis_drivers,
                "soft_beta": soft_beta,
                "top_tokens": top,
            })

        return {
            "run_id": run_id,
            "top_k": top_k,
            "results": results,
        }


    def _select_analysis_drivers(self, scores: List[float], cfg: Dict, default_drivers: List[int]) -> List[int]:
        mode = cfg.get("mode")
        if mode is None:
            return default_drivers
        if not scores:
            return []
        s = torch.tensor(scores, dtype=torch.float32)
        if mode == "top_percentage":
            alpha = float(cfg.get("alpha", 0.25))
            k = max(1, int(len(scores) * alpha))
            return sorted(torch.argsort(s, descending=True)[:k].tolist())
        threshold_k = float(cfg.get("threshold_k", 2.0))
        threshold = s.mean() + threshold_k * s.std(unbiased=False)
        return [i for i, v in enumerate(scores) if float(v) > float(threshold)]

    def _load_gradient_trace(self, run_id: str) -> Optional[Dict]:
        patt = os.path.join(self.hook_dir, run_id, "**", "highlighter.pt")
        paths = glob.glob(patt, recursive=True)
        if not paths:
            return None
        # Load latest gradient trace file
        latest = max(paths, key=os.path.getmtime)
        return torch.load(latest, map_location="cpu")

    def _top_tokens(
        self,
        token_ids: List[int],
        tokens: List[str],
        scores: List[float],
        top_k: int,
    ) -> List[Dict]:
        if top_k <= 0 or not scores:
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top = []
        for i in order:
            token_text = tokens[i] if i < len(tokens) else str(token_ids[i] if i < len(token_ids) else "<na>")
            top.append({"idx": i, "token": token_text, "score": scores[i]})
        return top
