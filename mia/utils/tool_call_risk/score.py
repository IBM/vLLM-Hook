"""Numpy-only scorer for a tool-call risk probe.

Used by ``ToolCallRiskAnalyzer`` at inference. The probe is an L2 logistic
regression over standardized hidden states taken at the last prompt token of
an agent step, i.e. before the model generates its next action. A high score
means the model is likely to emit an unsafe tool call at this step.

Artifact layout (two files side by side):

    probe.npz   mean, scale, weights (hidden_size,), bias (), layer ()
    probe.json  {"model_name": ..., "layer": int, "threshold": float, ...}

``layer`` follows the vLLM-Hook convention (1-based, output of the Nth decoder
block, same as HuggingFace ``hidden_states[N]``).

Probe training lives outside this repo:
        https://github.com/rishabhsinha17/latent-state-auditing
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ToolCallProbeArtifact:
    """Loaded tool-call risk probe (inference-only view)."""
    model_name: str
    layer: int
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float = 0.5
    meta: Dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "ToolCallProbeArtifact":
        data = np.load(path)
        with open(path.replace(".npz", ".json")) as f:
            meta = json.load(f)
        layer = int(data["layer"])
        if int(meta.get("layer", layer)) != layer:
            raise ValueError(
                f"{path}: layer {layer} in probe.npz does not match "
                f"layer {meta['layer']} in probe.json"
            )
        weights = data["weights"]
        for key in ("mean", "scale"):
            if data[key].shape != weights.shape:
                raise ValueError(
                    f"{path}: '{key}' has shape {data[key].shape}, "
                    f"expected {weights.shape}"
                )
        return cls(
            model_name=meta["model_name"],
            layer=layer,
            mean=data["mean"],
            scale=data["scale"],
            weights=weights,
            bias=float(data["bias"]),
            threshold=float(meta.get("threshold", 0.5)),
            meta=meta,
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass
class ToolCallRiskScore:
    probability: float   # P(unsafe tool call at this step)
    margin: float        # raw logit


class ToolCallRiskProbe:

    def __init__(self, artifact: ToolCallProbeArtifact):
        self.artifact = artifact
        self._w = artifact.weights.astype(np.float32)
        self._b = float(artifact.bias)
        self._mean = artifact.mean.astype(np.float32)
        self._scale = np.where(artifact.scale == 0, 1.0, artifact.scale).astype(np.float32)

    @classmethod
    def load(cls, path: str) -> "ToolCallRiskProbe":
        return cls(ToolCallProbeArtifact.load(path))

    @property
    def layer(self) -> int:
        return self.artifact.layer

    @property
    def hidden_size(self) -> int:
        return int(self._w.shape[0])

    def score(self, activations: np.ndarray) -> List[ToolCallRiskScore]:
        """Score a batch. ``activations`` has shape (batch, hidden_size)."""
        h = activations.astype(np.float32, copy=False)
        if h.ndim == 1:
            h = h[None, :]
        if h.shape[1] != self.hidden_size:
            raise ValueError(
                f"Probe expects hidden size {self.hidden_size}, got {h.shape[1]}. "
                f"Was it trained on {self.artifact.model_name}?"
            )
        logits = ((h - self._mean) / self._scale) @ self._w + self._b
        probs = _sigmoid(logits)
        return [
            ToolCallRiskScore(probability=float(p), margin=float(l))
            for p, l in zip(probs, logits)
        ]
