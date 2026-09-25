"""Numpy-only scorer for a trained H-Node probe."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np


@dataclass
class ProbeArtifact:
    """Loaded H-Node probe artifact (inference-only view)."""
    model_name: str
    best_layer: int
    auc_per_layer: Dict[int, float]
    hidden_size: int
    weights: np.ndarray
    bias: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    h_node_indices: np.ndarray
    h_node_baselines: np.ndarray
    baseline_percentile: int
    n_h_nodes: int
    train_size: int
    eval_size: int

    @classmethod
    def load(cls, path: str) -> "ProbeArtifact":
        data = np.load(path)
        meta_path = path.replace(".npz", ".json")
        with open(meta_path) as f:
            meta = json.load(f)
        return cls(
            model_name=meta["model_name"],
            best_layer=int(data["best_layer"]),
            auc_per_layer={int(k): float(v) for k, v in meta["auc_per_layer"].items()},
            hidden_size=int(data["hidden_size"]),
            weights=data["weights"],
            bias=float(data["bias"]),
            scaler_mean=data["scaler_mean"],
            scaler_scale=data["scaler_scale"],
            h_node_indices=data["h_node_indices"],
            h_node_baselines=data["h_node_baselines"],
            baseline_percentile=int(data["baseline_percentile"]),
            n_h_nodes=int(data["n_h_nodes"]),
            train_size=int(data["train_size"]),
            eval_size=int(data["eval_size"]),
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass
class HallucinationScore:
    probability: float
    h_node_excess: float
    margin: float


class HNodeProbe:
    """In-memory probe — lightweight wrapper around a ``ProbeArtifact``."""

    def __init__(self, artifact: ProbeArtifact):
        self.artifact = artifact
        self._w = artifact.weights.astype(np.float32)
        self._b = float(artifact.bias)
        self._mean = artifact.scaler_mean.astype(np.float32)
        self._scale = np.where(artifact.scaler_scale == 0, 1.0, artifact.scaler_scale).astype(np.float32)
        self._h_idx = artifact.h_node_indices.astype(np.int64)
        self._h_base = artifact.h_node_baselines.astype(np.float32)

    @classmethod
    def load(cls, path: str) -> "HNodeProbe":
        return cls(ProbeArtifact.load(path))

    @property
    def best_layer(self) -> int:
        return self.artifact.best_layer

    def score(self, activations: np.ndarray) -> List[HallucinationScore]:
        """Score a batch."""
        h = activations.astype(np.float32, copy=False)
        if h.ndim == 1:
            h = h[None, :]

        std = (h - self._mean) / self._scale
        logits = std @ self._w + self._b
        probs = _sigmoid(logits)

        excess = np.maximum(0.0, h[:, self._h_idx] - self._h_base)
        mean_excess = excess.mean(axis=1)

        return [
            HallucinationScore(
                probability=float(p),
                h_node_excess=float(e),
                margin=float(l),
            )
            for p, e, l in zip(probs, mean_excess, logits)
        ]


def score_activations(
    probe_path: str,
    activations: Union[np.ndarray, Sequence[np.ndarray]],
) -> List[HallucinationScore]:
    """Convenience: load probe + score in one call."""
    probe = HNodeProbe.load(probe_path)
    if isinstance(activations, np.ndarray):
        return probe.score(activations)
    return probe.score(np.stack([np.asarray(a) for a in activations]))

