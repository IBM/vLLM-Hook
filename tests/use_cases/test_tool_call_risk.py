# tests/use_cases/test_tool_call_risk.py
# CPU only, no model download. Checks the numpy scorer and the analyzer's
# layer lookup against a synthetic probe.
import json

import numpy as np
import pytest
import torch

from mia.analyzers.tool_call_risk_analyzer import ToolCallRiskAnalyzer
from mia.utils.tool_call_risk.score import ToolCallRiskProbe

HIDDEN = 16
LAYER = 21


def _write_probe(tmp_path, layer=LAYER, threshold=0.5, hidden=HIDDEN, json_layer=None):
    rng = np.random.default_rng(0)
    mean = rng.normal(size=hidden).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, size=hidden).astype(np.float32)
    weights = rng.normal(size=hidden).astype(np.float32)
    bias = np.float32(-0.25)
    path = tmp_path / "probe.npz"
    np.savez(path, mean=mean, scale=scale, weights=weights, bias=bias, layer=np.int64(layer))
    meta = {"model_name": "test/model", "layer": layer if json_layer is None else json_layer,
            "threshold": threshold}
    (tmp_path / "probe.json").write_text(json.dumps(meta))
    return str(path), (mean, scale, weights, float(bias))


def _reference(x, mean, scale, weights, bias):
    z = ((x - mean) / scale) @ weights + bias
    return 1.0 / (1.0 + np.exp(-z)), z


def test_scores_match_reference_logistic(tmp_path):
    path, params = _write_probe(tmp_path)
    probe = ToolCallRiskProbe.load(path)
    x = np.random.default_rng(1).normal(size=(5, HIDDEN)).astype(np.float32)
    want_p, want_z = _reference(x, *params)
    got = probe.score(x)
    np.testing.assert_allclose([s.probability for s in got], want_p, rtol=1e-5)
    np.testing.assert_allclose([s.margin for s in got], want_z, rtol=1e-5)


def test_extreme_logits_do_not_overflow(tmp_path):
    path, _ = _write_probe(tmp_path)
    probe = ToolCallRiskProbe.load(path)
    x = np.full((2, HIDDEN), 1e4, dtype=np.float32)
    x[1] *= -1
    probs = [s.probability for s in probe.score(x)]
    assert all(np.isfinite(probs)) and all(0.0 <= p <= 1.0 for p in probs)


def test_hidden_size_mismatch_is_reported(tmp_path):
    path, _ = _write_probe(tmp_path)
    probe = ToolCallRiskProbe.load(path)
    with pytest.raises(ValueError, match="hidden size"):
        probe.score(np.zeros((1, HIDDEN + 1), dtype=np.float32))


def test_layer_disagreement_between_npz_and_json_is_rejected(tmp_path):
    path, _ = _write_probe(tmp_path, json_layer=LAYER + 1)
    with pytest.raises(ValueError, match="does not match"):
        ToolCallRiskProbe.load(path)


@pytest.mark.parametrize("stacked", [True, False])
def test_analyzer_scores_probe_layer(tmp_path, stacked):
    path, params = _write_probe(tmp_path, threshold=0.5)
    x = np.random.default_rng(2).normal(size=(3, HIDDEN)).astype(np.float32)
    rows = [torch.from_numpy(r) for r in x]
    other = [torch.zeros(HIDDEN) for _ in rows]
    # RPC path stacks per-request tensors; disk path keeps a list.
    hs = (lambda ts: torch.stack(ts)) if stacked else (lambda ts: ts)
    hs_cache = {
        "model.layers.3": {"hidden_states": hs(other), "layer_num": 4},
        "model.layers.20": {"hidden_states": hs(rows), "layer_num": LAYER},
    }
    out = ToolCallRiskAnalyzer(str(tmp_path), {}).analyze(
        analyzer_spec={"probe_path": path}, probes={"hs_cache": hs_cache}
    )
    want_p, _ = _reference(x, *params)
    np.testing.assert_allclose(out["probabilities"], want_p, rtol=1e-5)
    assert out["layer"] == LAYER and out["threshold"] == 0.5
    assert out["verdicts"] == ["flag" if p >= 0.5 else "pass" for p in want_p]


def test_spec_threshold_overrides_artifact(tmp_path):
    path, _ = _write_probe(tmp_path, threshold=0.5)
    hs_cache = {"model.layers.20": {"hidden_states": [torch.zeros(HIDDEN)], "layer_num": LAYER}}
    out = ToolCallRiskAnalyzer(str(tmp_path), {}).analyze(
        analyzer_spec={"probe_path": path, "threshold": 1.0}, probes={"hs_cache": hs_cache}
    )
    assert out["threshold"] == 1.0 and out["verdicts"] == ["pass"]


def test_missing_layer_names_the_fix(tmp_path):
    path, _ = _write_probe(tmp_path)
    hs_cache = {"model.layers.3": {"hidden_states": [torch.zeros(HIDDEN)], "layer_num": 4}}
    with pytest.raises(RuntimeError, match="include layer 21"):
        ToolCallRiskAnalyzer(str(tmp_path), {}).analyze(
            analyzer_spec={"probe_path": path}, probes={"hs_cache": hs_cache}
        )
