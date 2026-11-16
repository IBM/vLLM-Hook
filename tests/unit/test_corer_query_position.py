"""
CPU-Only: CoRe's query rows attend only to keys at or before their own prompt position, also when
the chat template puts tokens after the query (Mistral's [/INST], Llama's assistant header)
"""

import math

import pytest
import torch
from mia.analyzers.core_reranker_analyzer import CorerAnalyzer

HEADS, HEAD_DIM, SEQ = 4, 8, 30
LAYER = 2
DOC_A, DOC_B = (2, 8), (10, 16)
Q_START, AFTER_INSTRUCT, Q_END = 18, 21, 25      # positions 26..29 are the template suffix
ALL_HEADS = {LAYER: list(range(HEADS))}


def _save_run(hook_dir, run_id, q, k, config=None):
    cache = {"config": {"num_attention_heads": HEADS, "num_key_value_heads": HEADS,
                        "head_dim": HEAD_DIM, **(config or {})},
             "qk_cache": {f"model.layers.{LAYER}.self_attn": {"layer_num": LAYER, "q": [q],
                                                              "k_all": [k]}}}
    (hook_dir / run_id).mkdir()
    torch.save(cache, hook_dir / run_id / "qk.pt")


def _random_qk(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(SEQ, HEADS * HEAD_DIM, generator=g),
            torch.randn(SEQ, HEADS * HEAD_DIM, generator=g))


def _token_scores(hook_dir, spans=(DOC_A, DOC_B)):
    per_doc, _ = CorerAnalyzer(str(hook_dir), ALL_HEADS).score_documents(
        "doc", [list(spans)], [Q_START], [AFTER_INSTRUCT], [Q_END])
    return per_doc[0]


def _expected(q, k, scale, spans=(DOC_A, DOC_B)):
    """Causal attention written out: query row r is prompt position Q_START + r."""
    rows = q[Q_START:Q_END + 1].view(-1, HEADS, HEAD_DIM)
    keys = k.view(SEQ, HEADS, HEAD_DIM)
    total = torch.zeros(SEQ)
    for h in range(HEADS):
        for r in range(rows.shape[0]):
            seen = Q_START + r + 1
            total[:seen] += torch.softmax(keys[:seen, h] @ rows[r, h] * scale, 0) / rows.shape[0]
    return [total[s:e + 1] for s, e in spans]


def test_query_does_not_attend_to_the_template_suffix(tmp_path):
    q, k = _random_qk(0)
    _save_run(tmp_path, "doc", q, k)
    got = _token_scores(tmp_path)
    for g, w in zip(got, _expected(q, k, 1 / math.sqrt(HEAD_DIM))):
        torch.testing.assert_close(g, w, atol=1e-5, rtol=1e-4)


def test_suffix_keys_take_no_attention_from_the_documents(tmp_path):
    # keys after the query score high; every other key scores 0, so a query row that sees only
    # the prompt up to itself spreads its attention evenly: row r gives 7 / (19 + r) to DOC_A
    q, k = torch.zeros(SEQ, HEADS, HEAD_DIM), torch.zeros(SEQ, HEADS, HEAD_DIM)
    q[..., 0] = 1.0
    k[Q_END + 1:, :, 0] = 20.0
    _save_run(tmp_path, "doc", q.flatten(1), k.flatten(1))
    doc_a = _token_scores(tmp_path, spans=(DOC_A,))[0]
    n_rows = Q_END - Q_START + 1
    want = HEADS * sum(7 / (Q_START + r + 1) for r in range(n_rows)) / n_rows
    assert doc_a.sum().item() == pytest.approx(want, rel=1e-5)


def test_per_layer_config_matches_uniform_config(tmp_path):
    # the per-layer path (Gemma 4 style configs) already masks by true position: both must agree
    q, k = _random_qk(1)
    _save_run(tmp_path, "doc", q, k)
    uniform = _token_scores(tmp_path)
    (tmp_path / "doc" / "qk.pt").unlink()
    (tmp_path / "doc").rmdir()
    _save_run(tmp_path, "doc", q, k, config={
        "layers": {str(LAYER): {"attention_multiplier": 1 / math.sqrt(HEAD_DIM)}}})
    for u, p in zip(uniform, _token_scores(tmp_path)):
        torch.testing.assert_close(u, p, atol=1e-5, rtol=1e-4)
