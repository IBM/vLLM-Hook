"""
CPU-Only: CoRe scores documents with the configured retrieval heads, read from real on-disk Q/K runs
"""

import math

import pytest
import torch
from mia.analyzers.core_reranker_analyzer import CorerAnalyzer

HEADS, HEAD_DIM, SEQ = 4, 8, 30
LAYERS = (2, 5)
DOC_A, DOC_B = (2, 8), (10, 16)
Q_START, AFTER_INSTRUCT, Q_END = 18, 21, SEQ - 1   # the query ends the prompt


def _save_run(hook_dir, run_id, layers, kv_heads=HEADS):
    """A one-request run of ``{layer: (q, k_all)}`` as the Q/K writer leaves it on disk."""
    cache = {
        "config": {"num_attention_heads": HEADS, "num_key_value_heads": kv_heads,
                   "head_dim": HEAD_DIM},
        "qk_cache": {f"model.layers.{L}.self_attn": {"layer_num": L, "q": [q], "k_all": [k]}
                     for L, (q, k) in layers.items()},
    }
    run_dir = hook_dir / run_id
    run_dir.mkdir()
    torch.save(cache, run_dir / "qk.pt")


def _random_layers(seed, kv_heads=HEADS):
    g = torch.Generator().manual_seed(seed)
    return {L: (torch.randn(SEQ, HEADS * HEAD_DIM, generator=g),
                torch.randn(SEQ, kv_heads * HEAD_DIM, generator=g)) for L in LAYERS}


def _token_scores(hook_dir, layer_to_heads, run_id="doc"):
    per_doc, _ = CorerAnalyzer(str(hook_dir), layer_to_heads).score_documents(
        run_id, [[DOC_A, DOC_B]], [Q_START], [AFTER_INSTRUCT], [Q_END])
    return per_doc[0]


def _expected_token_scores(layers, layer_to_heads, kv_heads=HEADS):
    """The same scores written out one layer, head, query row and key at a time."""
    groups = HEADS // kv_heads
    total = torch.zeros(SEQ)
    for L, heads in layer_to_heads.items():
        q, k = layers[L]
        rows = q[Q_START:Q_END + 1].view(-1, HEADS, HEAD_DIM)
        keys = k.view(SEQ, kv_heads, HEAD_DIM)
        n_rows = rows.shape[0]
        for h in heads:
            for r in range(n_rows):
                seen = Q_START + r + 1      # query row r is prompt position Q_START + r
                logits = keys[:seen, h // groups] @ rows[r, h] / math.sqrt(HEAD_DIM)
                total[:seen] += torch.softmax(logits, dim=0) / n_rows
    return [total[s:e + 1] for s, e in (DOC_A, DOC_B)]


def _assert_same(got, want):
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("layer_to_heads", [
    {2: [0], 5: [1]},
    {2: [3], 5: [2]},
    {2: [0, 3], 5: [1]},    # a different number of heads per layer
])
def test_scores_come_from_the_configured_heads_only(tmp_path, layer_to_heads):
    layers = _random_layers(0)
    _save_run(tmp_path, "doc", layers)
    got = _token_scores(tmp_path, layer_to_heads)
    _assert_same(got, _expected_token_scores(layers, layer_to_heads))


def test_query_head_reads_its_shared_key_head(tmp_path):
    # GQA: 4 query heads over 2 key heads, so query head 3 reads key head 1
    layers = _random_layers(0, kv_heads=2)
    _save_run(tmp_path, "doc", layers, kv_heads=2)
    layer_to_heads = {2: [3], 5: [0]}
    got = _token_scores(tmp_path, layer_to_heads)
    _assert_same(got, _expected_token_scores(layers, layer_to_heads, kv_heads=2))


def test_config_order_of_layers_does_not_matter(tmp_path):
    # shipped configs list heads by importance, not by layer: [[15, 21], [16, 12], [9, 26], ...]
    layers = _random_layers(0)
    _save_run(tmp_path, "doc", layers)
    _assert_same(_token_scores(tmp_path, {5: [1], 2: [0]}), _token_scores(tmp_path, {2: [0], 5: [1]}))


@pytest.mark.parametrize("unset", [None, {}])
def test_no_configured_heads_scores_with_all_heads(tmp_path, unset):
    layers = _random_layers(0)
    _save_run(tmp_path, "doc", layers)
    every_head = {L: list(range(HEADS)) for L in LAYERS}
    _assert_same(_token_scores(tmp_path, unset), _expected_token_scores(layers, every_head))


def test_captured_layer_without_configured_heads_is_refused(tmp_path):
    _save_run(tmp_path, "doc", _random_layers(0))
    with pytest.raises(ValueError, match="layer 5"):
        _token_scores(tmp_path, {2: [0]})


def test_ranking_follows_the_retrieval_head(tmp_path):
    """Head 0 attends to document B and heads 1-3 to document A: the config decides the ranking."""
    q, k = torch.zeros(SEQ, HEADS, HEAD_DIM), torch.zeros(SEQ, HEADS, HEAD_DIM)
    q[..., 0] = 1.0
    k[DOC_A[0]:DOC_A[1] + 1, 1:, 0] = 6.0
    k[DOC_B[0]:DOC_B[1] + 1, 0, 0] = 6.0
    layer = (q.flatten(1), k.flatten(1))
    _save_run(tmp_path, "doc", {2: layer})
    no_doc_signal = (q.flatten(1), torch.zeros(SEQ, HEADS * HEAD_DIM))
    _save_run(tmp_path, "na", {2: no_doc_signal})

    def ranking(layer_to_heads):
        spec = {"query_spec": ([DOC_A, DOC_B], Q_START, AFTER_INSTRUCT, Q_END),
                "na_spec": ([DOC_A, DOC_B], Q_START, AFTER_INSTRUCT, Q_END)}
        return CorerAnalyzer(str(tmp_path), layer_to_heads).analyze(
            spec, run_ids=["doc", "na"])["ranking"][0]

    assert ranking({2: [0]}) == [1, 0]
    assert ranking({2: [1, 2, 3]}) == [0, 1]


def test_per_layer_scale_and_sliding_window_apply_to_the_configured_heads(tmp_path):
    """Query row r at position 18 + r sees only keys p with r + 18 - p < 3, scaled by the model's 0.5."""
    layers = _random_layers(0)
    _save_run(tmp_path, "doc", layers)
    path = tmp_path / "doc" / "qk.pt"
    cache = torch.load(path)
    cache["config"]["layers"] = {"2": {"attention_multiplier": 0.5, "sliding_window": 3}}
    cache["qk_cache"] = {k: v for k, v in cache["qk_cache"].items() if v["layer_num"] == 2}
    torch.save(cache, path)

    q, k = layers[2]
    rows = q[Q_START:Q_END + 1].view(-1, HEADS, HEAD_DIM)
    keys = k.view(SEQ, HEADS, HEAD_DIM)
    want = torch.zeros(SEQ)
    for r in range(rows.shape[0]):
        pos = Q_START + r
        seen = list(range(pos - 2, pos + 1))
        logits = keys[seen, 1] @ rows[r, 1] * 0.5
        want[seen] += torch.softmax(logits, dim=0) / rows.shape[0]

    per_doc, _ = CorerAnalyzer(str(tmp_path), {2: [1]}).score_documents(
        "doc", [[(16, 20), DOC_A]], [Q_START], [AFTER_INSTRUCT], [Q_END])
    torch.testing.assert_close(per_doc[0][0], want[16:21], atol=1e-5, rtol=1e-4)
    assert per_doc[0][1].sum() == pytest.approx(0.0, abs=1e-6)     # outside every window
