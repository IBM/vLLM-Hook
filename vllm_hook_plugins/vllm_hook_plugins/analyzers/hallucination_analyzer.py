"""Hallucination-detection analyzer (H-Node probe).

Sits next to ``HiddenStatesAnalyzer`` / ``ScienceHallucinationAnalyzer`` in the
plugin registry: consumes the last-token hidden states captured by
``ProbeHiddenStatesWorker`` and applies a pre-trained H-Node probe to produce a
hallucination probability per prompt.

Follows the post-refactor analyzer contract (same shape as
``ScienceHallucinationAnalyzer``):

    analyze(analyzer_spec=None, run_id=None, probes=None)

Hidden states are sourced from one of three places:
  - ``probes`` dict (in-memory RPC path, ``output[0].probes``)
  - shared memory (``VLLM_HOOK_USE_SHM=1``)
  - disk artifacts merged by ``load_and_merge_hs_cache`` (``run_id``)

Usage (in-memory / RPC path)::

    llm = HookLLM(
        model=...,
        worker_name="probe_hidden_states",
        analyzer_name="hallucination",
        config_file="model_configs/hallucination_detection/<model>.infer.json",
        ...,
    )
    output = llm.generate(prompts, SamplingParams(max_tokens=1))
    result = llm.analyze(
        probes=output[0].probes,
        analyzer_spec={"probe_path": "artifacts/probe.npz", "threshold": 0.5},
    )
    # -> {"probabilities": [...], "h_node_excess": [...], "margins": [...],
    #     "best_layer": int, "verdicts": ["grounded"|"hallucinated", ...]}
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from vllm_hook_plugins.run_utils import load_and_merge_hs_cache, unpack_hidden_states
from vllm_hook_plugins.shm_utils import load_from_shm


class HallucinationAnalyzer:

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        # The probe is loaded lazily on first analyze() call. Hot-reloaded if
        # analyzer_spec carries a different probe_path.
        self._probe = None
        self._probe_path: Optional[str] = None

    def _ensure_probe(self, probe_path: str):
        from vllm_hook_plugins.utils.hnode.score import HNodeProbe

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
                "HallucinationAnalyzer requires analyzer_spec={'probe_path': ...}"
            )
        threshold = float(spec.get("threshold", 0.5))

        probe = self._ensure_probe(probe_path)

        peak_gpu_mb = None
        if probes is not None:
            hs_cache = probes["hs_cache"]
        elif os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            hs_cache, peak_gpu_mb = load_from_shm(self.hook_dir, run_id)
        else:
            if run_id is None:
                raise ValueError(
                    "HallucinationAnalyzer.analyze: pass either probes= or run_id=."
                )
            cache = load_and_merge_hs_cache(self.hook_dir, run_id)
            hs_cache = cache["hs_cache"]

        # Find the module whose 1-based layer_num matches the probe's best layer.
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

        # unpack_hidden_states normalizes both the RPC (single stacked tensor)
        # and disk (list of tensors) formats into a list of per-pass tensors.
        tensors: List[torch.Tensor] = unpack_hidden_states(hs_cache[target_module])
        # last_token mode: each tensor is (hidden,). all_tokens mode: (seq, hidden)
        # — keep last token only. Stack to (batch, hidden).
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
