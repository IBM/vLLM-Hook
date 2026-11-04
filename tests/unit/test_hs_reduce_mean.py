"""CPU-only checks of the hidden_states analyzer's length-aware reductions on padded layouts.

Chunked prefill (eager) and decode passes (graph delivery) both leave a request's hidden states as a
zero-padded ``(passes, rows, hidden)`` tensor. ``reduce="mean"`` must ignore the padding, and
``reduce="mean_tokens"`` must give the same vector however the prompt was split into passes.
"""
import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from mia.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer
from mia.core.delivery.delivered_probes import (hs_probes, padded_offline_probes, padded_reader,
                                                trim_probes)

H = 8
LAYER = "model.layers.1"


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, H, generator=g) + 1.0     # +1: no real row is all zeros


def _probes(passes, *, padded=True, lens=True, mode="all_tokens"):
    hs = pad_sequence(passes, batch_first=True) if padded else list(passes)
    entry = {"hidden_states": hs, "layer_num": 1, "hs_mode": mode}
    if padded and lens:
        entry["pass_lens"] = [p.shape[0] for p in passes]
    return {"hs_cache": {LAYER: entry}, "config": {}}


def _analyze(probes, reduce):
    return HiddenStatesAnalyzer("", {}).analyze({"reduce": reduce}, probes=probes)["hidden_states"][LAYER]


@pytest.mark.parametrize("padded,lens", [(True, True), (True, False), (False, False)])
def test_mean_excludes_padding(padded, lens):
    X = _rows(10)
    passes = list(torch.split(X, [4, 4, 2]))           # last chunk is short -> 2 padded rows
    got = _analyze(_probes(passes, padded=padded, lens=lens), "mean")
    assert len(got) == 3
    for g, p in zip(got, passes):
        torch.testing.assert_close(g, p.mean(0))


@pytest.mark.parametrize("split", [[10], [4, 4, 2], [3, 3, 3, 1], [1] * 10])
@pytest.mark.parametrize("padded,lens", [(True, True), (True, False), (False, False)])
def test_mean_tokens_does_not_depend_on_chunking(split, padded, lens):
    X = _rows(10, seed=1)
    got = _analyze(_probes(list(torch.split(X, split)), padded=padded, lens=lens), "mean_tokens")
    assert len(got) == 1 and got[0].shape == (H,)
    torch.testing.assert_close(got[0], X.mean(0))


def test_mean_tokens_refuses_last_token_mode():
    probes = {"hs_cache": {LAYER: {"hidden_states": _rows(3), "layer_num": 1,
                                   "hs_mode": "last_token"}}, "config": {}}
    with pytest.raises(ValueError, match="all_tokens"):
        _analyze(probes, "mean_tokens")


def test_last_token_mean_unchanged():
    X = _rows(3)
    probes = {"hs_cache": {LAYER: {"hidden_states": X, "layer_num": 1, "hs_mode": "last_token"}},
              "config": {}}
    for g, x in zip(_analyze(probes, "mean"), X):
        torch.testing.assert_close(g, x)


def test_trim_clips_pass_lens():
    probes = _probes([_rows(6), _rows(1, 2)])
    trim_probes(probes, "hs_cache", 5)
    e = probes["hs_cache"][LAYER]
    assert e["hidden_states"].shape[1] == 5 and e["pass_lens"] == [5, 1]


META = {"hs_mode": "all_tokens", "hooks_on": "both", "n_prompt": 6, "n_gen": 3, "config": {}}


@pytest.mark.parametrize("layout", ["rpc", "disk"])
def test_delivered_prompt_and_decode_passes(layout):
    """Graph-mode delivery: one 6-row prompt pass, then 1-row decode passes padded to 6."""
    X = _rows(8, seed=3)                                  # n_prompt + n_gen - 1 rows
    probes = hs_probes({1: X}, dict(META), {1: LAYER}, layout=layout)
    assert probes["hs_cache"][LAYER]["pass_lens"] == [6, 1, 1]
    means = _analyze(probes, "mean")
    torch.testing.assert_close(means[0], X[:6].mean(0))
    torch.testing.assert_close(means[1], X[6])            # was X[6] / 6 before the fix
    torch.testing.assert_close(_analyze(probes, "mean_tokens")[0], X.mean(0))


def test_padded_reader_records_pass_lens():
    X = _rows(8, seed=4)
    meta = dict(META)
    out = padded_reader(meta)([1], 8, (H,), "float32")
    t, views = out[1]
    off = 0
    for v in views:
        n = len(v) // (H * 4)
        v[:] = X[off:off + n].numpy().tobytes()
        off += n
    probes = padded_offline_probes({1: t}, meta, {1: LAYER})
    assert probes["hs_cache"][LAYER]["pass_lens"] == [6, 1, 1]
    torch.testing.assert_close(_analyze(probes, "mean_tokens")[0], X.mean(0))
