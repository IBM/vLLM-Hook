"""Two-stage probe analyzer: a stage-1 probe gates, a stage-2 probe confirms.

Both stages are linear probes over hidden states at the last prompt token, in
the ``tool_call_risk`` artifact format (``probe.npz`` + ``probe.json``). A step
is flagged only when both stages fire, and stage 2 is scored only on the steps
stage 1 flags. Nothing here depends on the task the probes were trained for.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from mia.artifacts import load_and_merge_hs_cache, unpack_hidden_states
from mia.utils.tool_call_risk.score import ToolCallRiskProbe
from mia.utils.two_stage.cascade import TwoStageDetector

STAGES = ("stage1", "stage2")


def last_token_features(hs_cache: Dict, layer: int) -> np.ndarray:
    """(steps, hidden) array of last-prompt-token states at ``layer`` (1-based)."""
    for entry in hs_cache.values():
        if int(entry["layer_num"]) == layer:
            tensors: List[torch.Tensor] = unpack_hidden_states(entry)
            rows = [t if t.dim() == 1 else t[-1] for t in tensors]
            return torch.stack(rows).float().cpu().numpy()
    available = sorted(int(e["layer_num"]) for e in hs_cache.values())
    raise RuntimeError(
        f"A probe expects activations from layer {layer}, but the current config captured "
        f"layers {available}. Add layer {layer} to 'hidden_states.layers'."
    )


def resolve_spec(spec: Dict) -> Dict[str, Dict]:
    """Merge ``cascade_path`` (if given) with per-stage overrides from the spec.

    ``cascade.json`` holds {"stage1": {"probe_path", "threshold"}, "stage2": {...}}.
    Relative probe paths are resolved against the directory of cascade.json.
    """
    stages: Dict[str, Dict] = {s: {} for s in STAGES}
    path = spec.get("cascade_path")
    if path:
        with open(path) as f:
            cfg = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
        for s in STAGES:
            stages[s].update(cfg.get(s, {}))
            p = stages[s].get("probe_path")
            if p and not os.path.isabs(p):
                stages[s]["probe_path"] = os.path.join(base, p)
    for s in STAGES:
        stages[s].update(spec.get(s, {}))
        if not stages[s].get("probe_path"):
            raise ValueError(
                "TwoStageProbeAnalyzer needs analyzer_spec={'cascade_path': ...} or "
                f"{{'stage1': {{'probe_path': ...}}, 'stage2': {{'probe_path': ...}}}} (missing {s})"
            )
    return stages


class TwoStageProbeAnalyzer:
    """Flags a step when the stage-1 and stage-2 probes both fire."""

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        """``hook_dir`` holds the disk runs; ``layer_to_heads`` is unused."""
        self.hook_dir = hook_dir
        self._probes: Dict[str, ToolCallRiskProbe] = {}

    def _probe(self, path: str) -> ToolCallRiskProbe:
        if path not in self._probes:
            self._probes[path] = ToolCallRiskProbe.load(path)
        return self._probes[path]

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        probes: Optional[Dict] = None,
    ) -> Dict:
        """Score each step with both stages.

        Both thresholds must come from ``cascade.json`` or the spec. Each
        probe.json threshold is ignored, because it was calibrated for that
        probe alone. Set ``"score_all": true`` to score stage 2 on every step,
        which is what recalibrating the thresholds needs.
        """
        spec = analyzer_spec or {}
        stages = resolve_spec(spec)
        p1 = self._probe(stages["stage1"]["probe_path"])
        p2 = self._probe(stages["stage2"]["probe_path"])
        missing = [s for s in STAGES if stages[s].get("threshold") is None]
        if missing:
            raise ValueError(
                f"TwoStageProbeAnalyzer needs a calibrated threshold for {', '.join(missing)}. "
                "A probe.json threshold was set for that probe alone, so it is not used here. "
                "Calibrate both with mia.utils.two_stage.calibrate_cascade and write them with "
                "save_cascade_config, or pass them in the spec."
            )
        t1 = float(stages["stage1"]["threshold"])
        t2 = float(stages["stage2"]["threshold"])

        if probes is not None:
            hs_cache = probes["hs_cache"]
        else:
            if run_id is None:
                raise ValueError("TwoStageProbeAnalyzer.analyze: pass either probes= or run_id=.")
            hs_cache = load_and_merge_hs_cache(self.hook_dir, run_id)["hs_cache"]

        x1 = last_token_features(hs_cache, p1.layer)
        x2 = x1 if p2.layer == p1.layer else last_token_features(hs_cache, p2.layer)

        def stage1(idx):
            return [s.probability for s in p1.score(x1[idx])]

        def stage2(idx):
            return [s.probability for s in p2.score(x2[idx])]

        result = TwoStageDetector(stage1, stage2, t1, t2)(
            np.arange(len(x1)), score_all=bool(spec.get("score_all", False)))

        return {
            "verdicts": ["flag" if f else "pass" for f in result.flags],
            "reached_stage2": [bool(r) for r in result.reached_stage2],
            "stage1": {
                "probabilities": result.stage1_scores.tolist(),
                "layer": p1.layer,
                "threshold": t1,
            },
            "stage2": {
                "probabilities": [None if np.isnan(p) else float(p) for p in result.stage2_scores],
                "layer": p2.layer,
                "threshold": t2,
            },
        }
