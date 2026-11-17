"""
CPU-Only: the science-hallucination analyzer reads the final norm from any checkpoint layout and uses
the classifier it is given
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from mia.analyzers.science_hallucination_analyzer import ScienceHallucinationAnalyzer

joblib = pytest.importorskip("joblib")

HIDDEN = 4
NORM = torch.tensor([1.0, 2.0, 3.0, 4.0])


def _write_config(snap, eps=1e-6):
    (snap / "config.json").write_text(json.dumps({"rms_norm_eps": eps}))


def test_final_norm_from_single_file_checkpoint(tmp_path):
    # small models (Qwen2-1.5B-Instruct, Llama-3.2-1B) ship one model.safetensors and no index
    save_file({"model.norm.weight": NORM, "model.embed_tokens.weight": torch.zeros(2, HIDDEN)},
              str(tmp_path / "model.safetensors"))
    _write_config(tmp_path)
    weight, eps = ScienceHallucinationAnalyzer._read_final_norm(str(tmp_path))
    torch.testing.assert_close(weight, NORM)
    assert eps == 1e-6


def test_final_norm_from_sharded_checkpoint(tmp_path):
    save_file({"model.embed_tokens.weight": torch.zeros(2, HIDDEN)},
              str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"model.norm.weight": NORM}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.norm.weight": "model-00002-of-00002.safetensors"}}))
    _write_config(tmp_path, eps=1e-5)
    weight, eps = ScienceHallucinationAnalyzer._read_final_norm(str(tmp_path))
    torch.testing.assert_close(weight, NORM)
    assert eps == 1e-5


class _Always:
    """A classifier that predicts one label for every row."""
    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return torch.full((len(x),), self.label).numpy()


def test_classifier_follows_clf_path(tmp_path, monkeypatch):
    joblib.dump(_Always(0), tmp_path / "zero.joblib")
    joblib.dump(_Always(2), tmp_path / "two.joblib")
    monkeypatch.setattr(ScienceHallucinationAnalyzer, "_fetch_final_norm",
                        lambda self, model_id: (torch.ones(HIDDEN), 1e-6))
    probes = {"hs_cache": {"model.layers.1": {"layer_num": 1,
                                              "hidden_states": torch.randn(3, HIDDEN)}},
              "config": {}}
    analyzer = ScienceHallucinationAnalyzer(str(tmp_path), {})

    def predict(name):
        spec = {"clf_path": str(tmp_path / name), "model_id": "m"}
        return analyzer.analyze(spec, probes=probes)["predictions"]

    assert predict("zero.joblib") == [0, 0, 0]
    assert predict("two.joblib") == [2, 2, 2]
