"""Tensor-parallel QK geometry and the per-rank merge (mia/graph/tp_shard.py, aperture_reader).

The defect this suite pins: at tensor_parallel_size > 1 MIA captured QK on tp_rank 0 only,
with buffers sized to that rank's SHARD -- so the artifact held H_q/tp of the query heads and
every reader (``_conf``, the analyzers, the RPC ``parts[0]``) treated it as the whole layer.
Now every rank captures its own heads, stamps its geometry, and a merge rebuilds the global
layout. These tests are the oracle for that merge: global Q/K tensors are sliced into per-rank
shards by an INDEPENDENT re-statement of vLLM's sharding (``_vllm_slices`` below -- literal
arithmetic, not a call into the code under test), then merged back and required to be
BITWISE equal to the original. They also require the merge to REFUSE every partial or
inconsistent rank set, a rank-0-only capture first among them.

Hermetic: CPU tensors, temp dirs, no engine.
"""
from __future__ import annotations

import inspect
import json
import os
import random

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

import torch

from mia.graph.tp_shard import (
    QK_SHARD_FIELDS,
    TP_SHARD_KEY,
    TPShardError,
    check_complete_shard_set,
    merge_head_tensors,
    merge_qk_payloads,
    parse_rank_dir,
    qk_shard,
    qk_shard_from_header,
    rank_dir_name,
)
from mia.errors import MiaConfigurationError

D = 8          # small head_dim keeps the fixtures fast; geometry is head-count arithmetic


def _vllm_slices(h_q, h_kv, tp, r):
    """INDEPENDENT oracle: which global heads vLLM 0.29 puts on rank r.

    Written from vllm/model_executor/layers/linear.py (QKVParallelLinear.__init__ and its
    weight_loader: ``shard_rank = tp_rank // num_kv_head_replicas``), NOT from the module
    under test. ``test_vllm_still_shards_the_way_the_oracle_says`` pins that source."""
    nq = h_q // tp
    if tp >= h_kv:
        nkv, rep = 1, tp // h_kv
    else:
        nkv, rep = h_kv // tp, 1
    q_heads = list(range(r * nq, (r + 1) * nq))
    kv_heads = list(range((r // rep) * nkv, (r // rep) * nkv + nkv))
    return q_heads, kv_heads


def _cols(heads):
    return [h * D + j for h in heads for j in range(D)]


def _global(h_q, h_kv, rows=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, h_q * D, generator=g).to(torch.bfloat16),
            torch.randn(rows, h_kv * D, generator=g).to(torch.bfloat16))


def _shards(h_q, h_kv, tp, q, k):
    out = []
    for r in range(tp):
        qh, kh = _vllm_slices(h_q, h_kv, tp, r)
        out.append((qk_shard(r, tp, h_q, h_kv, D), q[:, _cols(qh)], k[:, _cols(kh)]))
    return out


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

GEOMETRIES = [
    # (H_q, H_kv, tp) -- the study's two models at every TP it runs, plus KV replication.
    (32, 8, 1), (32, 8, 2), (32, 8, 4), (32, 8, 8),        # Llama-3.1-8B
    (64, 8, 4), (64, 8, 8),                                  # Llama-3.1-70B
    (12, 2, 4), (8, 2, 8), (16, 1, 4),                       # H_kv < tp: KV heads replicated
]


@pytest.mark.parametrize("h_q,h_kv,tp", GEOMETRIES)
def test_shard_matches_vllm_head_assignment(h_q, h_kv, tp):
    for r in range(tp):
        s = qk_shard(r, tp, h_q, h_kv, D)
        qh, kh = _vllm_slices(h_q, h_kv, tp, r)
        assert (s.q_head_start, s.num_local_q_heads) == (qh[0], len(qh))
        assert (s.kv_head_start, s.num_local_kv_heads) == (kh[0], len(kh))
        assert s.q_width == len(qh) * D and s.k_width == len(kh) * D
        assert s.num_kv_head_replicas == (tp // h_kv if tp >= h_kv else 1)


def test_the_study_shapes_literally():
    """External literal anchors (the study's configs), so a formula edit cannot move both the
    oracle and the implementation together unnoticed."""
    s = qk_shard(3, 4, 64, 8, 128)          # 70B, TP4, rank 3
    assert (s.q_head_start, s.num_local_q_heads, s.kv_head_start, s.num_local_kv_heads) \
        == (48, 16, 6, 2)
    assert (s.q_width, s.k_width) == (2048, 256)
    s = qk_shard(7, 8, 64, 8, 128)          # 70B, TP8, rank 7
    assert (s.q_head_start, s.num_local_q_heads, s.kv_head_start, s.num_local_kv_heads) \
        == (56, 8, 7, 1)
    s = qk_shard(5, 8, 32, 8, 128)          # 8B, TP8, rank 5
    assert (s.q_head_start, s.num_local_q_heads, s.kv_head_start, s.num_local_kv_heads) \
        == (20, 4, 5, 1)
    s = qk_shard(3, 4, 12, 2, 128)          # Qwen2.5-1.5B at TP4: KV replicated x2
    assert (s.q_head_start, s.num_local_q_heads, s.kv_head_start, s.num_local_kv_heads,
            s.num_kv_head_replicas) == (9, 3, 1, 1, 2)


def test_vllm_still_shards_the_way_the_oracle_says():
    """Drift detector: the oracle above restates vLLM's arithmetic. If vLLM changes it, this
    fails before a GPU run writes mislabelled shards."""
    from vllm.model_executor.layers import linear
    init = inspect.getsource(linear.QKVParallelLinear.__init__)
    assert "self.num_heads = divide(self.total_num_heads, tp_size)" in init
    assert "if tp_size >= self.total_num_kv_heads:" in init
    assert "self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)" in init
    assert "self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)" in init
    loader = inspect.getsource(linear.QKVParallelLinear.weight_loader)
    assert "shard_rank = self.tp_rank // self.num_kv_head_replicas" in loader


@pytest.mark.parametrize("args", [(0, 3, 32, 8, D), (0, 4, 30, 8, D), (0, 3, 12, 2, D),
                                  (4, 4, 32, 8, D), (0, 0, 32, 8, D)])
def test_impossible_geometry_is_refused(args):
    with pytest.raises(MiaConfigurationError):
        qk_shard(*args)


def test_header_round_trip_and_tampering():
    s = qk_shard(2, 4, 64, 8, 128)
    h = s.as_header()
    assert tuple(h) == QK_SHARD_FIELDS
    assert json.loads(json.dumps(h)) == h                  # JSON-native
    assert qk_shard_from_header(dict(h, dtype="bfloat16", q_row_shape=[2048])) == s
    assert qk_shard_from_header({"dtype": "bfloat16"}) is None      # pre-TP header
    with pytest.raises(TPShardError, match="incomplete"):
        qk_shard_from_header({k: v for k, v in h.items() if k != "kv_head_start"})
    with pytest.raises(TPShardError, match="disagrees"):
        qk_shard_from_header(dict(h, q_head_start=0))       # lies about its heads
    with pytest.raises(TPShardError, match="impossible"):
        qk_shard_from_header(dict(h, tp_size=3))


def test_rank_dir_naming():
    assert rank_dir_name(3) == "tp_rank_3"
    assert parse_rank_dir("/a/b/tp_rank_12/") == 12
    assert parse_rank_dir("tp_rank_x") is None and parse_rank_dir("rank_1") is None


# --------------------------------------------------------------------------------------
# The merge
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("h_q,h_kv,tp", GEOMETRIES)
def test_merge_rebuilds_the_global_layer_bitwise(h_q, h_kv, tp):
    q, k = _global(h_q, h_kv)
    shards = _shards(h_q, h_kv, tp, q, k)
    random.Random(tp).shuffle(shards)                       # input order must not matter
    mq = merge_head_tensors("q", [(s, sq) for s, sq, _ in shards])
    mk = merge_head_tensors("k", [(s, sk) for s, _, sk in shards], check_replicas=True)
    assert torch.equal(mq, q) and torch.equal(mk, k)


@pytest.mark.parametrize("h_q,h_kv,tp", [(12, 2, 4), (8, 2, 8), (16, 1, 4)])
def test_replicated_kv_heads_are_deduplicated_not_concatenated(h_q, h_kv, tp):
    """NEGATIVE CONTROL for the de-dup: naively concatenating every rank's K (what the eager
    disk loader did) is tp/H_kv times too wide. The merge must produce exactly H_kv heads."""
    q, k = _global(h_q, h_kv)
    shards = _shards(h_q, h_kv, tp, q, k)
    naive = torch.cat([sk for _, _, sk in shards], dim=-1)
    assert naive.shape[-1] == tp * D != h_kv * D
    mk = merge_head_tensors("k", [(s, sk) for s, _, sk in shards])
    assert mk.shape[-1] == h_kv * D and torch.equal(mk, k)


def test_rank0_only_capture_is_refused():
    """THE REGRESSION: the old QK path captured rank 0 alone. Its lone shard says tp_size=4;
    the merge must refuse rather than return a quarter of the heads."""
    q, k = _global(32, 8)
    s0, q0, k0 = _shards(32, 8, 4, q, k)[0]
    with pytest.raises(TPShardError, match=r"missing tp_rank\(s\) \[1, 2, 3\]"):
        merge_head_tensors("q", [(s0, q0)])
    with pytest.raises(TPShardError, match="missing"):
        merge_head_tensors("k", [(s0, k0)])


def test_every_other_inconsistent_rank_set_is_refused():
    q, k = _global(32, 8)
    sh = _shards(32, 8, 4, q, k)
    items = [(s, sq) for s, sq, _ in sh]
    with pytest.raises(TPShardError, match="duplicate"):
        merge_head_tensors("q", items + [items[1]])
    with pytest.raises(TPShardError, match="width"):
        merge_head_tensors("q", [(items[0][0], items[0][1][:, :-1])] + items[1:])
    with pytest.raises(TPShardError, match="leading shape"):
        merge_head_tensors("q", [(items[0][0], items[0][1][:-1])] + items[1:])
    other = qk_shard(1, 4, 64, 8, D)
    with pytest.raises(TPShardError, match="geometry"):
        check_complete_shard_set([items[0][0], other])


def test_divergent_kv_replicas_are_caught_only_when_asked():
    q, k = _global(12, 2)
    sh = _shards(12, 2, 4, q, k)
    items = [(s, sk.clone()) for s, _, sk in sh]
    items[1][1].add_(1)                           # rank 1 is rank 0's replica; perturb it
    merged = merge_head_tensors("k", items)       # lowest rank of each group wins
    assert torch.equal(merged, k)
    with pytest.raises(TPShardError, match="replicated KV"):
        merge_head_tensors("k", items, check_replicas=True)


def _payload(shard, q_list, kall_list, tp_shard=True):
    p = {"qk_cache": {"model.layers.0.self_attn.attn": {
        "q": q_list, "k_all": kall_list, "layer_num": 0, "hookq_mode": "all_tokens"}},
        "config": {"num_attention_heads": shard.num_attention_heads}}
    if tp_shard:
        p[TP_SHARD_KEY] = shard.as_header()
    return p


def test_payload_merge_handles_padded_tensors_and_per_pass_lists():
    """RPC payloads are padded tensors (N_passes, max_seq, width); disk payloads are per-pass
    lists. Both merge on the last dim; the tp_shard key is dropped."""
    q, k = _global(32, 8, rows=6)
    sh = _shards(32, 8, 4, q, k)
    padded = [_payload(s, sq.view(2, 3, -1), sk.view(2, 3, -1)) for s, sq, sk in sh]
    out = merge_qk_payloads(padded[::-1])
    e = out["qk_cache"]["model.layers.0.self_attn.attn"]
    assert TP_SHARD_KEY not in out
    assert torch.equal(e["q"], q.view(2, 3, -1)) and torch.equal(e["k_all"], k.view(2, 3, -1))
    lists = [_payload(s, [sq[:2], sq[2:]], [sk[:2], sk]) for s, sq, sk in sh]
    e = merge_qk_payloads(lists)["qk_cache"]["model.layers.0.self_attn.attn"]
    assert torch.equal(e["q"][0], q[:2]) and torch.equal(e["q"][1], q[2:])
    assert torch.equal(e["k_all"][1], k)


def test_a_tp1_payload_is_returned_untouched():
    q, k = _global(32, 8)
    s = qk_shard(0, 1, 32, 8, D)
    p = _payload(s, q, k, tp_shard=False)
    assert merge_qk_payloads([p]) is p            # byte-identical TP=1 path: same object


def test_payload_merge_refuses_partial_and_unlabelled_sets():
    q, k = _global(32, 8)
    sh = _shards(32, 8, 4, q, k)
    with pytest.raises(TPShardError, match="missing"):
        merge_qk_payloads([_payload(s, sq, sk) for s, sq, sk in sh[:3]])
    with pytest.raises(TPShardError, match="carry no"):
        merge_qk_payloads([_payload(s, sq, sk, tp_shard=False) for s, sq, sk in sh])
    bad = [_payload(s, sq, sk) for s, sq, sk in sh]
    bad[2]["qk_cache"]["model.layers.0.self_attn.attn"]["k_prefix_ends"] = [1]
    bad[0]["qk_cache"]["model.layers.0.self_attn.attn"]["k_prefix_ends"] = [2]
    with pytest.raises(TPShardError, match="disagree"):
        merge_qk_payloads(bad)
    score = [_payload(s, sq, sk) for s, sq, sk in sh]
    for p in score:
        p["qk_cache"]["model.layers.0.self_attn.attn"]["capture"] = "score"
    with pytest.raises(TPShardError, match="SCORE"):
        merge_qk_payloads(score)


# --------------------------------------------------------------------------------------
# The on-disk contract: per-rank aperture dirs written by the REAL drain, merged by the reader
# --------------------------------------------------------------------------------------

def _write_rank_dir(base, shard, per_req, n_layers, header_extra=None, with_tp=True,
                    prefix_end_shift=0):
    """Write one rank's QK aperture dump with the production drain + sidecar writer.

    ``per_req`` is ``[(req_id, q_rank (S, qw), k_rank (S, kw))]``; each request is ONE
    all_tokens prefill step whose rows go to consecutive aperture slots, every layer holding
    ``layer_scale * data`` so a layer mix-up is visible."""
    from mia.graph.aperture_drain_qk import MultiLayerQKApertureDrain
    from mia.graph.aperture_metadata import QKReqCaptureRecord
    from mia.graph.capture_aperture import CaptureAperture

    total = sum(q.shape[0] for _, q, _ in per_req)
    qw, kw = per_req[0][1].shape[1], per_req[0][2].shape[1]
    ap = CaptureAperture(row_bytes=kw * 2, n_slots=total, device="cpu",
                         dtype=torch.bfloat16, row_shape=(kw,))
    layers = [(L, torch.zeros(total + 1, qw, dtype=torch.bfloat16),
               torch.zeros(total + 1, kw, dtype=torch.bfloat16)) for L in range(n_layers)]
    header = {"dtype": "bfloat16", "q_row_shape": [qw], "k_row_shape": [kw],
              "q_dim": qw, "k_dim": kw, "hookq_mode": "all_tokens"}
    if with_tp:
        header.update(shard.as_header())
    header.update(header_extra or {})
    run_dir = os.path.join(base, rank_dir_name(shard.tp_rank))
    drain = MultiLayerQKApertureDrain(ap, layers, run_dir, header)
    for req_id, qr, kr in per_req:
        n = qr.shape[0]
        start = ap.reserve(n)
        for L, qb, kb in layers:
            qb[start:start + n] = qr * (L + 1)
            kb[start:start + n] = kr * (L + 1)
        drain.record_entries([QKReqCaptureRecord(
            req_id=req_id, k_start=start, k_rows=n, q_start=start, q_rows=n,
            prefix_end=n + prefix_end_shift, num_computed=0, layers=list(range(n_layers)))])
        drain.drain_once()
    drain.close()
    return run_dir


def _per_rank_requests(h_q, h_kv, tp, n_req=2):
    reqs = [(f"req{i}", *_global(h_q, h_kv, rows=3 + i, seed=10 + i)) for i in range(n_req)]
    per_rank = {r: [] for r in range(tp)}
    for req_id, q, k in reqs:
        for s, sq, sk in _shards(h_q, h_kv, tp, q, k):
            per_rank[s.tp_rank].append((req_id, sq, sk))
    return reqs, per_rank


@pytest.mark.parametrize("h_q,h_kv,tp", [(32, 8, 4), (64, 8, 8), (12, 2, 4)])
def test_reader_merges_real_rank_dirs_to_the_global_layer(tmp_path, h_q, h_kv, tp):
    from mia.graph.aperture_reader import load_qk_aperture_tp, merge_qk_aperture_ranks

    n_layers = 3
    reqs, per_rank = _per_rank_requests(h_q, h_kv, tp)
    dirs = [_write_rank_dir(str(tmp_path), qk_shard(r, tp, h_q, h_kv, D), per_rank[r], n_layers)
            for r in range(tp)]
    for merged in (merge_qk_aperture_ranks(dirs[::-1]), load_qk_aperture_tp(str(tmp_path))):
        assert set(merged) == {r for r, _, _ in reqs}
        for req_id, q, k in reqs:
            assert set(merged[req_id]) == set(range(n_layers))
            for L in range(n_layers):
                e = merged[req_id][L]
                assert e["q"].shape == (q.shape[0], h_q * D)
                assert torch.equal(e["q"], q * (L + 1))
                assert torch.equal(e["k_full"], k * (L + 1))
                assert torch.equal(e["k_all"][0], k * (L + 1))
                assert e["k_prefix_ends"] == [q.shape[0]]


def test_reader_refuses_a_rank0_only_dump(tmp_path):
    """The on-disk form of the regression: one rank dir whose header says tp_size=4."""
    from mia.graph.aperture_reader import load_qk_aperture_tp
    from mia.graph.aperture_reader import load_multilayer_qk_aperture_artifact

    _, per_rank = _per_rank_requests(32, 8, 4)
    d0 = _write_rank_dir(str(tmp_path), qk_shard(0, 4, 32, 8, D), per_rank[0], 2)
    with pytest.raises(TPShardError, match="missing"):
        load_qk_aperture_tp(str(tmp_path))
    # ...while the single-dir reader would happily return the narrow shard -- which is why
    # nothing downstream may read one rank dir and call it the layer.
    art = load_multilayer_qk_aperture_artifact(d0)
    assert art["req0"][0]["q"].shape[-1] == 8 * D


def test_reader_refuses_ranks_that_captured_different_requests(tmp_path):
    from mia.graph.aperture_reader import load_qk_aperture_tp

    _, per_rank = _per_rank_requests(32, 8, 2)
    _write_rank_dir(str(tmp_path), qk_shard(0, 2, 32, 8, D), per_rank[0], 2)
    _write_rank_dir(str(tmp_path), qk_shard(1, 2, 32, 8, D), per_rank[1][:1], 2)
    with pytest.raises(TPShardError, match="different"):
        load_qk_aperture_tp(str(tmp_path))


def test_reader_refuses_a_header_that_contradicts_its_files(tmp_path):
    from mia.graph.aperture_reader import merge_qk_aperture_ranks

    _, per_rank = _per_rank_requests(32, 8, 2)
    d0 = _write_rank_dir(str(tmp_path), qk_shard(0, 2, 32, 8, D), per_rank[0], 1,
                         header_extra={"q_row_shape": [5]})
    d1 = _write_rank_dir(str(tmp_path), qk_shard(1, 2, 32, 8, D), per_rank[1], 1)
    with pytest.raises(TPShardError, match="contradict"):
        merge_qk_aperture_ranks([d0, d1])


def test_reader_accepts_a_pre_tp_single_dir(tmp_path):
    """Back-compat: a TP=1 dump written before the header carried TP fields reads as-is."""
    from mia.graph.aperture_reader import load_qk_aperture_tp

    reqs, per_rank = _per_rank_requests(32, 8, 1)
    _write_rank_dir(str(tmp_path), qk_shard(0, 1, 32, 8, D), per_rank[0], 2, with_tp=False)
    merged = load_qk_aperture_tp(str(tmp_path))
    assert torch.equal(merged["req1"][1]["q"], reqs[1][1] * 2)


# --------------------------------------------------------------------------------------
# Consumers: the driver's RPC merge, the analyzer, the disk loaders
# --------------------------------------------------------------------------------------

def test_driver_merge_replaces_parts0():
    """`_plugin.merge_probe_parts` is what the RPC paths call instead of `parts[0]`."""
    from mia._plugin import merge_probe_parts

    q, k = _global(32, 8, rows=4)
    sh = _shards(32, 8, 4, q, k)
    parts = [_payload(s, sq.view(1, 4, -1), sk.view(1, 4, -1)) for s, sq, sk in sh]
    merged = merge_probe_parts(parts)
    assert torch.equal(merged["qk_cache"]["model.layers.0.self_attn.attn"]["q"],
                       q.view(1, 4, -1))
    with pytest.raises(TPShardError):
        merge_probe_parts(parts[:1])             # rank 0 alone at TP4: refused, not returned
    single = {"qk_cache": {}, "config": {}}
    assert merge_probe_parts([single]) is single  # TP=1: unchanged object
    hs = [{"hs_cache": {"a": 1}}, {"hs_cache": {"a": 2}}]
    assert merge_probe_parts(hs) is hs[0]         # replicated HS: rank 0's copy


def test_block_until_held_waits_for_every_rank_across_polls():
    """Per-request delivery pops each rank's stash EXACTLY once, on that rank's own schedule.
    The poller must keep what it already received and merge only when all ranks arrived."""
    import asyncio
    import pickle

    from mia import _plugin

    q, k = _global(32, 8, rows=2)
    sh = _shards(32, 8, 2, q, k)
    blobs = [pickle.dumps(_payload(s, sq.view(1, 2, -1), sk.view(1, 2, -1)))
             for s, sq, sk in sh]
    polls = [[blobs[0], None], [None, None], [None, blobs[1]]]   # each part delivered once

    class Engine:
        async def collective_rpc(self, method, args=()):
            return polls.pop(0) if polls else [None, None]

    out = asyncio.run(_plugin._await_aperture_per_request(Engine(), "r0"))
    assert torch.equal(out["qk_cache"]["model.layers.0.self_attn.attn"]["q"], q.view(1, 2, -1))


def test_analyzer_on_merged_shards_equals_analyzer_on_the_global_layer():
    """`_conf` keeps the GLOBAL head counts and the analyzer views q as (H_q, head_dim): that is
    only correct on the MERGED payload. Merged-from-4-ranks must score identically to the
    never-sharded layer."""
    from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer

    h_q, h_kv, seq = 32, 8, 6
    g = torch.Generator().manual_seed(3)
    q_last = torch.randn(1, h_q * D, generator=g)
    k_all = torch.randn(1, seq, h_kv * D, generator=g)
    conf = {"num_attention_heads": h_q, "num_key_value_heads": h_kv, "head_dim": D,
            "attention_multiplier": D ** -0.5, "hidden_size": h_q * D}
    name = "model.layers.0.self_attn.attn"
    parts = []
    for r in range(4):
        qh, kh = _vllm_slices(h_q, h_kv, 4, r)
        s = qk_shard(r, 4, h_q, h_kv, D)
        parts.append({"qk_cache": {name: {"q": q_last[:, _cols(qh)], "k_all": k_all[..., _cols(kh)],
                                          "layer_num": 0}},
                      "config": conf, TP_SHARD_KEY: s.as_header()})
    merged = merge_qk_payloads(parts)
    ref = {"qk_cache": {name: {"q": q_last, "k_all": k_all, "layer_num": 0}}, "config": conf}
    an = AttntrackerAnalyzer(hook_dir="/nonexistent", layer_to_heads={0: [1, 17, 30]})
    a = an.compute_attention_from_qk(probes=merged)[0][name]["attention"]
    b = an.compute_attention_from_qk(probes=ref)[0][name]["attention"]
    assert torch.equal(a, b)


def test_disk_loader_merges_eager_rank_shards(tmp_path):
    """The eager flush_disk writes hook_dir/run_id/tp_rank_<r>/qk.pt with a tp_shard at TP>1;
    load_and_merge_qk_cache must merge them in head order (not glob order, not rank 0)."""
    from mia.run_utils import load_and_merge_qk_cache

    h_q, h_kv, tp = 12, 2, 4
    q, k = _global(h_q, h_kv, rows=3)
    name = "model.layers.0.self_attn.attn"
    for s, sq, sk in _shards(h_q, h_kv, tp, q, k)[::-1]:
        d = tmp_path / "run" / rank_dir_name(s.tp_rank)
        d.mkdir(parents=True)
        torch.save({"config": {"num_attention_heads": h_q}, TP_SHARD_KEY: s.as_header(),
                    "qk_cache": {name: {"q": [sq], "k_all": [sk], "layer_num": 0,
                                        "hookq_mode": "all_tokens"}}}, d / "qk.pt")
    merged = load_and_merge_qk_cache(str(tmp_path), "run")
    e = merged["qk_cache"][name]
    assert torch.equal(e["q"][0], q) and torch.equal(e["k_all"][0], k)
    assert merged["meta"]["tp_ranks"] == [0, 1, 2, 3]


def test_disk_loader_returns_one_hs_replica_instead_of_concatenating(tmp_path):
    """HS shards from several ranks are replicas of ONE residual stream; the loader used to
    concatenate them along hidden (tp x hidden wide)."""
    from mia.run_utils import load_and_merge_hs_cache

    hs = torch.randn(4, 16)
    for r in range(2):
        d = tmp_path / "run" / rank_dir_name(r)
        d.mkdir(parents=True)
        torch.save({"config": {"hidden_size": 16},
                    "hs_cache": {"model.layers.0": {"hidden_states": [hs], "layer_num": 1}}},
                   d / "hidden_states.pt")
    merged = load_and_merge_hs_cache(str(tmp_path), "run")
    assert merged["hs_cache"]["model.layers.0"]["hidden_states"][0].shape == (4, 16)


def test_eager_payload_carries_its_shard_only_above_tp1():
    from types import SimpleNamespace

    from mia.workers.qk_capture_worker import _attach_tp_shard

    p = _attach_tp_shard({}, SimpleNamespace(_qk_shard=qk_shard(0, 1, 32, 8, D)))
    assert p == {}                                  # TP=1 payload byte-identical
    p = _attach_tp_shard({}, SimpleNamespace(_qk_shard=qk_shard(2, 4, 32, 8, D)))
    assert p[TP_SHARD_KEY]["tp_rank"] == 2 and p[TP_SHARD_KEY]["q_head_start"] == 16


# --------------------------------------------------------------------------------------
# The four mutants that survived W1's suite (each test fails with its mutant applied)
# --------------------------------------------------------------------------------------

def test_reader_refuses_ranks_that_disagree_on_k_prefix_ends(tmp_path):
    """Same (request, layer) set on both ranks, but rank 1's prefix boundaries differ: its K rows
    belong to a different step structure, and merging them head-wise would silently mix them.
    (Mutant: the merge skipping its k_prefix_ends comparison.)"""
    from mia.graph.aperture_reader import merge_qk_aperture_ranks

    _, per_rank = _per_rank_requests(32, 8, 2)
    d0 = _write_rank_dir(str(tmp_path), qk_shard(0, 2, 32, 8, D), per_rank[0], 2)
    d1 = _write_rank_dir(str(tmp_path), qk_shard(1, 2, 32, 8, D), per_rank[1], 2,
                         prefix_end_shift=-1)
    with pytest.raises(TPShardError, match="k_prefix_ends"):
        merge_qk_aperture_ranks([d0, d1])


def test_disk_confirm_waits_for_every_staging_rank(monkeypatch):
    """Per-request DISK delivery at TP>1: every QK rank offloads its own head shard. The confirm
    returns only once EVERY rank that staged the request has landed it; a rank answering None
    (an HS sink) never blocks. (Mutant: confirming as soon as ANY rank landed.)"""
    import asyncio

    from mia import _plugin

    monkeypatch.setenv("MIA_APERTURE_DELIVER_TIMEOUT_S", "0.3")

    class Engine:
        def __init__(self, polls):
            self.polls, self.calls = list(polls), 0

        async def collective_rpc(self, method, args=()):
            assert method == "confirm_aperture_delivery"
            self.calls += 1
            return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]

    def confirm(polls):
        e = Engine(polls)
        return asyncio.run(_plugin._await_aperture_disk_confirm(e, "r0")), e.calls

    ok, calls = confirm([[True, False]])            # rank 1's shard never lands
    assert ok is False and calls > 1
    ok, calls = confirm([[True, False], [False, True], [True, True]])
    assert ok is True and calls == 3                # returns on the poll the LAST rank lands
    ok, calls = confirm([[True, None]])             # a non-staging rank never blocks
    assert ok is True and calls == 1
    ok, _ = confirm([[None, None]])                 # nobody staged: never "confirmed"
    assert ok is False


@pytest.mark.parametrize("rank,tp,want", [(2, 4, "/dest/tp_rank_2"), (0, 4, "/dest/tp_rank_0"),
                                          (0, 1, "/dest")])
def test_qk_per_request_disk_route_lands_each_rank_in_its_own_dir(rank, tp, want):
    """Every QK rank offloads its own shard of a disk-routed request; at TP>1 each must land in
    ``dest/tp_rank_<r>/`` or the shards overwrite each other (TP=1 keeps ``dest``). (Mutant: the
    per-rank join dropped.)"""
    from types import SimpleNamespace as NS

    from mia.workers.qk_capture_worker import QKCaptureWorker

    got = []
    drain = NS(per_request=True, route_to_disk=lambda rid, dest: got.append((rid, dest)))
    w = NS(_qk_drain=drain, _qk_shard=qk_shard(rank, tp, 32, 8, D))
    assert QKCaptureWorker.route_aperture_to_disk(w, "r0", "/dest") is True
    assert got == [("r0", want)]


def test_eager_rpc_payload_carries_its_shard_and_the_driver_rebuilds_the_layer():
    """The eager/bank RPC (get_captured_states) at TP=4: each rank's payload must carry its
    ``tp_shard``, so ``merge_probe_parts`` can rebuild the global heads; at TP=1 the payload has
    no such key. (Mutant: get_captured_states not stamping the shard.)"""
    import pickle
    from types import SimpleNamespace as NS

    import zstandard

    from mia._plugin import merge_probe_parts
    from mia.workers.qk_capture_worker import QKCaptureWorker

    h_q, h_kv, tp, seq = 32, 8, 4, 3
    name = "model.layers.0.self_attn.attn"
    q, _ = _global(h_q, h_kv, rows=1, seed=5)
    _, k = _global(h_q, h_kv, rows=seq, seed=6)

    def rpc(shard, qs, ks):
        w = NS(_captured_states={"r0-abc": {name: {"q": [qs], "k_all": [ks], "layer_num": 0,
                                                   "hookq_mode": "last_token"}}},
               _conf={"num_attention_heads": h_q, "num_key_value_heads": h_kv, "head_dim": D},
               hookq_mode="last_token", _qk_shard=shard)
        blob = QKCaptureWorker.get_captured_states(w, "r0")
        return pickle.loads(zstandard.ZstdDecompressor().decompress(blob))

    parts = []
    for r in range(tp):
        qh, kh = _vllm_slices(h_q, h_kv, tp, r)
        p = rpc(qk_shard(r, tp, h_q, h_kv, D), q[:, _cols(qh)], k[:, _cols(kh)])
        assert p[TP_SHARD_KEY]["tp_rank"] == r and p[TP_SHARD_KEY]["tp_size"] == tp
        parts.append(p)
    merged = merge_probe_parts(parts)
    assert torch.equal(merged["qk_cache"][name]["q"], q.unsqueeze(0))
    assert torch.equal(merged["qk_cache"][name]["k_all"], k.unsqueeze(0))
    tp1 = rpc(qk_shard(0, 1, h_q, h_kv, D), q, k)
    assert TP_SHARD_KEY not in tp1                  # TP=1 payload unchanged
