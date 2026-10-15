"""Tool-call risk analyzer for agent steps (logistic probe on the last prompt token)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from mia.artifacts import load_and_merge_hs_cache, unpack_hidden_states
from mia.utils.tool_call_risk.score import ToolCallRiskProbe


class ToolCallRiskAnalyzer:
    """P(next action follows a prompt injection) per agent step, scored before generation."""
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        """``hook_dir`` holds the disk runs; ``layer_to_heads`` is unused."""
        self.hook_dir = hook_dir
        self._probe = None
        self._probe_path: Optional[str] = None

    def _ensure_probe(self, probe_path: str):
        if self._probe is None or self._probe_path != probe_path:
            self._probe = ToolCallRiskProbe.load(probe_path)
            self._probe_path = probe_path
        return self._probe

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        probes: Optional[Dict] = None,
    ) -> Dict:
        """Score each step with the probe at spec ``probe_path``.

        ``threshold`` defaults to the one stored in probe.json.
        """
        spec = analyzer_spec or {}
        probe_path = spec.get("probe_path")
        if not probe_path:
            raise ValueError(
                "ToolCallRiskAnalyzer requires analyzer_spec={'probe_path': ...}"
            )
        probe = self._ensure_probe(probe_path)
        threshold = float(spec.get("threshold", probe.artifact.threshold))

        if probes is not None:
            hs_cache = probes["hs_cache"]
        else:
            if run_id is None:
                raise ValueError(
                    "ToolCallRiskAnalyzer.analyze: pass either probes= or run_id=."
                )
            cache = load_and_merge_hs_cache(self.hook_dir, run_id)
            hs_cache = cache["hs_cache"]

        target_module = None
        for module_name, entry in hs_cache.items():
            if int(entry["layer_num"]) == probe.layer:
                target_module = module_name
                break

        if target_module is None:
            available = sorted(int(e["layer_num"]) for e in hs_cache.values())
            raise RuntimeError(
                f"Probe expects activations from layer {probe.layer}, but the "
                f"current config captured layers {available}. Update the model config "
                f"to include layer {probe.layer} in 'hidden_states.layers'."
            )

        tensors: List[torch.Tensor] = unpack_hidden_states(hs_cache[target_module])
        # The probe reads the last prompt token. In all_tokens mode keep only that row.
        rows = [t if t.dim() == 1 else t[-1] for t in tensors]
        batch = torch.stack(rows).float().cpu().numpy()

        scores = probe.score(batch)

        return {
            "probabilities": [s.probability for s in scores],
            "margins": [s.margin for s in scores],
            "verdicts": ["flag" if s.probability >= threshold else "pass" for s in scores],
            "layer": probe.layer,
            "threshold": threshold,
        }
