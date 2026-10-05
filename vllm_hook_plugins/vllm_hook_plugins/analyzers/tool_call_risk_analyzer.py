"""Tool-call risk analyzer for agent steps.

Consumes the last-token hidden states captured by ``ProbeHiddenStatesWorker``
during prefill of an agent step (system prompt, tool schemas, conversation and
tool results so far) and applies a logistic probe that predicts whether the
model's next action is an unsafe tool call. Scoring happens before any output
token is sampled, so a caller can hold or reroute the step.

Same analyzer contract as ``HNodeHallucinationAnalyzer``:

    analyze(analyzer_spec=None, run_id=None, probes=None)

Usage::

    llm = HookLLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        worker_name="probe_hidden_states",
        analyzer_name="tool_call_risk",
        config_file="model_configs/tool_call_risk/Qwen2.5-7B-Instruct.infer.json",
        ...,
    )
    llm.generate(agent_prompts, SamplingParams(max_tokens=256),
                 save_to_disk=True, run_id="step_0")
    result = llm.analyze(
        analyzer_spec={"probe_path": "cache/tool_call_risk/probe.npz"},
        run_id="step_0",
    )
    # -> {"probabilities": [...], "margins": [...], "verdicts": ["flag"|"pass", ...],
    #     "layer": int, "threshold": float}
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from vllm_hook_plugins.run_utils import load_and_merge_hs_cache, unpack_hidden_states
from vllm_hook_plugins.shm_utils import load_from_shm


class ToolCallRiskAnalyzer:

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        self._probe = None
        self._probe_path: Optional[str] = None

    def _ensure_probe(self, probe_path: str):
        from vllm_hook_plugins.utils.tool_call_risk.score import ToolCallRiskProbe

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
        spec = analyzer_spec or {}
        probe_path = spec.get("probe_path")
        if not probe_path:
            raise ValueError(
                "ToolCallRiskAnalyzer requires analyzer_spec={'probe_path': ...}"
            )
        probe = self._ensure_probe(probe_path)
        # The artifact carries the threshold it was calibrated at; the spec can
        # override it for a different operating point.
        threshold = float(spec.get("threshold", probe.artifact.threshold))

        peak_gpu_mb = None
        if probes is not None:
            hs_cache = probes["hs_cache"]
        elif os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            hs_cache, peak_gpu_mb = load_from_shm(self.hook_dir, run_id)
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

        out = {
            "probabilities": [s.probability for s in scores],
            "margins": [s.margin for s in scores],
            "verdicts": ["flag" if s.probability >= threshold else "pass" for s in scores],
            "layer": probe.layer,
            "threshold": threshold,
        }
        if peak_gpu_mb is not None:
            out["peak_gpu_mb"] = peak_gpu_mb
        return out
