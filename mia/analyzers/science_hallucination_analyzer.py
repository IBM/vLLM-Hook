"""Science hallucination analyzer: classifies captured hidden states with a trained probe."""
import json
import os
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from mia.artifacts import load_and_merge_hs_cache, unpack_hidden_states


class ScienceHallucinationAnalyzer:
    """Classify captured hidden states, after the final RMSNorm, with a trained classifier."""
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        """``hook_dir`` holds the disk runs; ``layer_to_heads`` is unused."""
        self.hook_dir = hook_dir
        self._clf = None
        self.default_clf_path: Optional[str] = None
        self.default_model_id: Optional[str] = None
        self._norm_cache: Dict[str, Tuple[torch.Tensor, float]] = {}

    def _fetch_final_norm(self, model_id: str) -> Tuple[torch.Tensor, float]:
        if model_id in self._norm_cache:
            return self._norm_cache[model_id]
        # lazy: optional dependency (huggingface_hub)
        from huggingface_hub import snapshot_download

        snap = snapshot_download(model_id, allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            "*.safetensors",
        ])
        with open(os.path.join(snap, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        key = "model.norm.weight"
        shard = idx["weight_map"][key]
        with safe_open(os.path.join(snap, shard), framework="pt") as f:
            weight = f.get_tensor(key).to(torch.float32)
        with open(os.path.join(snap, "config.json")) as f:
            cfg = json.load(f)
        eps = float(cfg.get("rms_norm_eps", 1e-5))
        self._norm_cache[model_id] = (weight, eps)
        return weight, eps

    @staticmethod
    def _apply_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        return (x * weight.to(torch.float32)).to(orig_dtype)

    def analyze(self, analyzer_spec: Optional[Dict] = None, run_id: Optional[str] = None, probes: Optional[Dict] = None) -> Dict:
        """``{"predictions": [...]}`` from classifier ``clf_path``; ``model_id`` gives the norm."""
        if probes is not None:
            hs_cache = probes["hs_cache"]
        else:
            if run_id is None:
                raise ValueError("ScienceHallucinationAnalyzer.analyze: pass either probes= or run_id=.")
            cache = load_and_merge_hs_cache(self.hook_dir, run_id)
            hs_cache = cache["hs_cache"]

        spec = analyzer_spec or {}

        clf_path = spec.get("clf_path") or self.default_clf_path
        if clf_path is None:
            raise ValueError("ScienceHallucinationAnalyzer requires analyzer_spec['clf_path'] or analyzer.default_clf_path to be set.")
        if self._clf is None:
            # lazy: optional dependency (joblib)
            import joblib  
            self._clf = joblib.load(clf_path)
        clf = self._clf

        model_id = spec.get("model_id") or self.default_model_id
        if model_id is None:
            raise ValueError("ScienceHallucinationAnalyzer requires analyzer_spec['model_id'] or analyzer.default_model_id to be set so the final RMSNorm can be loaded from the HF cache for replay on captured residuals.")
        norm_weight, norm_eps = self._fetch_final_norm(model_id)

        layer_name = next(iter(hs_cache))
        feats = unpack_hidden_states(hs_cache[layer_name])
        feats = torch.stack([feat.float().cpu() for feat in feats], dim=0)
        feats = self._apply_rmsnorm(feats, norm_weight.cpu().float(), float(norm_eps))

        feats_np = feats.numpy()
        preds = clf.predict(feats_np)

        out: Dict = {"predictions": preds.tolist()}
        label_names = spec.get("label_names")
        if label_names is not None:
            out["prediction_labels"] = [label_names[int(p)] for p in preds]
        return out

