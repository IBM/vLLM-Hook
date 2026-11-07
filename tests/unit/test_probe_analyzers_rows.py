"""
CPU-Only: the tool-call risk and H-Node analyzers score each request's last prompt token in every
hidden-state layout (padded passes, chunked prefill, disk runs, batches merged by generate)
"""

import json

import numpy as np
import pytest
import torch

from mia.analyzers.hnode_hallucination_analyzer import HNodeHallucinationAnalyzer
from mia.analyzers.tool_call_risk_analyzer import ToolCallRiskAnalyzer

H = 4
LAYER = 3


def _prompt(n, first=1.0):
    """``n`` prompt rows; row i is all ``first + i``, so a probe with unit weights reads it back."""
    return torch.arange(first, first + n).repeat_interleave(H).view(n, H)


def _padded(passes):
    return torch.nn.utils.rnn.pad_sequence(passes, batch_first=True)


def _tool_call_probe(d):
    np.savez(d / "probe.npz", mean=np.zeros(H), scale=np.ones(H), weights=np.ones(H), bias=0.0,
             layer=LAYER)
    (d / "probe.json").write_text(json.dumps({"model_name": "m", "layer": LAYER}))
    return str(d / "probe.npz")


def _hnode_probe(d):
    np.savez(d / "hnode.npz", best_layer=LAYER, hidden_size=H, weights=np.ones(H), bias=0.0,
             scaler_mean=np.zeros(H), scaler_scale=np.ones(H), h_node_indices=np.array([0]),
             h_node_baselines=np.zeros(1), baseline_percentile=95, n_h_nodes=1, train_size=1,
             eval_size=1)
    (d / "hnode.json").write_text(json.dumps({"model_name": "m", "auc_per_layer": {}}))
    return str(d / "hnode.npz")


def _margins(kind, tmp_path, entry):
    probes = {"hs_cache": {f"model.layers.{LAYER}": {"layer_num": LAYER, **entry}}, "config": {}}
    if kind == "tool_call":
        analyzer, path = ToolCallRiskAnalyzer(str(tmp_path), {}), _tool_call_probe(tmp_path)
    else:
        analyzer, path = HNodeHallucinationAnalyzer(str(tmp_path), {}), _hnode_probe(tmp_path)
    return analyzer.analyze({"probe_path": path}, probes=probes)["margins"]


KINDS = ["tool_call", "hnode"]


@pytest.mark.parametrize("kind", KINDS)
def test_eager_chunked_prefill_scores_the_last_prompt_token_once(tmp_path, kind):
    # a 5-token prompt prefilled in chunks of 3 and 2, as one output's probes: padded to (2, 3, H)
    x = _prompt(5)
    entry = {"hidden_states": _padded([x[:3], x[3:]]), "hs_mode": "all_tokens",
             "pass_lens": [3, 2], "prefill_passes": 2}
    assert _margins(kind, tmp_path, entry) == [5.0 * H]


@pytest.mark.parametrize("kind", KINDS)
def test_decode_passes_after_the_prompt_are_not_scored(tmp_path, kind):
    # graph delivery, hooks_on="both": the prompt pass, then one padded row per decode step
    x, gen = _prompt(5), _prompt(2, first=50.0)
    entry = {"hidden_states": _padded([x, gen[:1], gen[1:]]), "hs_mode": "all_tokens",
             "pass_lens": [5, 1, 1]}
    assert _margins(kind, tmp_path, entry) == [5.0 * H]


@pytest.mark.parametrize("kind", KINDS)
def test_disk_run_gives_one_score_per_request(tmp_path, kind):
    # disk layout: unpadded passes; request 0 was prefilled in two chunks, request 1 in one
    a, b = _prompt(5), _prompt(4, first=10.0)
    entry = {"hidden_states": [a[:3], a[3:], b], "hs_mode": "all_tokens", "req_passes": [2, 1]}
    assert _margins(kind, tmp_path, entry) == [5.0 * H, 13.0 * H]


@pytest.mark.parametrize("kind", KINDS)
def test_batch_merged_by_generate(tmp_path, kind):
    # generate() joins each request's prompt chunks into one unpadded pass
    a, b = _prompt(5), _prompt(4, first=10.0)
    entry = {"hidden_states": [a, b], "hs_mode": "all_tokens", "req_passes": [1, 1]}
    assert _margins(kind, tmp_path, entry) == [5.0 * H, 13.0 * H]


@pytest.mark.parametrize("kind", KINDS)
def test_last_token_mode_is_unchanged(tmp_path, kind):
    entry = {"hidden_states": torch.stack([_prompt(1, 7.0)[0], _prompt(1, 9.0)[0]]),
             "hs_mode": "last_token"}
    assert _margins(kind, tmp_path, entry) == [7.0 * H, 9.0 * H]
