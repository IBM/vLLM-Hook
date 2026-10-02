"""Hallucination-detection analyzer (H-Node probe)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from mia.run_utils import load_and_merge_hs_cache, unpack_hidden_states
from mia.shm_utils import load_from_shm


class HNodeHallucinationAnalyzer:
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        self._probe = None
        self._probe_path: Optional[str] = None

    def _ensure_probe(self, probe_path: str):
        from mia.utils.hnode.score import HNodeProbe

        if self._probe is None or self._probe_path != probe_path:
            self._probe = HNodeProbe.load(probe_path)
            self._probe_path = probe_path
        return self._probe

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        probes: Optional[Dict] = None,
    ) -> Dict:
        spec = analyzer_spec or {}
        probe_path = spec.get("probe_path")
        if not probe_path:
            raise ValueError(
                "HNodeHallucinationAnalyzer requires analyzer_spec={'probe_path': ...}"
            )
        threshold = float(spec.get("threshold", 0.5))

        probe = self._ensure_probe(probe_path)

        peak_gpu_mb = None
        if probes is not None:
            hs_cache = probes["hs_cache"]
        elif os.environ.get("MIA_USE_SHM", "0") == "1":
            hs_cache, peak_gpu_mb = load_from_shm(self.hook_dir, run_id)
        else:
            if run_id is None:
                raise ValueError(
                    "HNodeHallucinationAnalyzer.analyze: pass either probes= or run_id=."
                )
            cache = load_and_merge_hs_cache(self.hook_dir, run_id)
            hs_cache = cache["hs_cache"]

        target_module = None
        for module_name, entry in hs_cache.items():
            if int(entry["layer_num"]) == probe.best_layer:
                target_module = module_name
                break

        if target_module is None:
            available = sorted(int(e["layer_num"]) for e in hs_cache.values())
            raise RuntimeError(
                f"Probe expects activations from layer {probe.best_layer}, but the "
                f"current config captured layers {available}. Update the model config "
                f"to include layer {probe.best_layer} in 'hidden_states.layers'."
            )

        tensors: List[torch.Tensor] = unpack_hidden_states(hs_cache[target_module])
        rows = [t if t.dim() == 1 else t[-1] for t in tensors]
        batch = torch.stack(rows).float().cpu().numpy()

        scores = probe.score(batch)

        out = {
            "probabilities": [s.probability for s in scores],
            "h_node_excess": [s.h_node_excess for s in scores],
            "margins": [s.margin for s in scores],
            "verdicts": [
                "hallucinated" if s.probability >= threshold else "grounded"
                for s in scores
            ],
            "best_layer": probe.best_layer,
            "threshold": threshold,
            "n_h_nodes": probe.artifact.n_h_nodes,
        }
        if peak_gpu_mb is not None:
            out["peak_gpu_mb"] = peak_gpu_mb
        return out

