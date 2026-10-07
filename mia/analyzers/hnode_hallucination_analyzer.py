"""Hallucination-detection analyzer (H-Node probe)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from mia.artifacts import load_and_merge_hs_cache, unpack_hidden_states
from mia.probes.hnode.score import HNodeProbe


class HNodeHallucinationAnalyzer:
    """Hallucination verdicts from an H-Node probe applied to captured hidden states."""
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        """``hook_dir`` holds the disk runs; ``layer_to_heads`` is unused."""
        self.hook_dir = hook_dir
        self._probe = None
        self._probe_path: Optional[str] = None

    def _ensure_probe(self, probe_path: str):
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
        """Score each prompt with the probe at spec ``probe_path`` (default ``threshold`` 0.5)."""
        spec = analyzer_spec or {}
        probe_path = spec.get("probe_path")
        if not probe_path:
            raise ValueError(
                "HNodeHallucinationAnalyzer requires analyzer_spec={'probe_path': ...}"
            )
        threshold = float(spec.get("threshold", 0.5))

        probe = self._ensure_probe(probe_path)

        if probes is not None:
            hs_cache = probes["hs_cache"]
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
        return out

