"""HS capture SHARDED BY LAYER across the TP ranks (FULL-graph aperture mode), GPU-free.

The residual stream is replicated on every TP rank, so any rank's copy of a layer IS the layer.
Before this change tp_rank 0 captured every layer while the other ranks baked 1-row sinks: one
drain thread carried the whole capture (70B TP4 HS: 10.74 GB per full step through one rank at
1.84 GB/s, and a 10 GiB aperture on rank 0 only). Now, at TP > 1 (``MIA_HS_TP_SHARD``, default 1),
rank ``r`` captures the 0-based layers ``i % tp_size == r``; ``MIA_HS_TP_SHARD=0`` restores the old
layout for A/B. Pinned here, each through MIA's real install / routing / op / drain / reader code
on CPU with a fake V2 worker:

  * the owned-layer assignment (80 x TP4/8, 32 x TP4/8, and the uneven / fewer-layers edges);
  * the CUDA-graph bake is SYMMETRIC: the same op, at the same call sites, with the same inputs,
    on every rank -- only the buffer behind an unowned layer is a 1-row sink;
  * each rank's aperture is sized over its OWN layers (70B TP4: 2.5 GiB one step, not 10);
  * the per-rank sidecar header records tp_rank / tp_size / num_layers / the owned layers;
  * the routing keeps only owned layers, for all three builders;
  * a sharded capture reads back (union of the rank dirs) equal to the TP=1 capture, and the
    reader REFUSES a gap, a duplicate, a lying header and cross-rank row disagreement;
  * TP = 1 is byte-identical to the pre-shard code (golden digests taken at fd73aa6);
  * the per-request RPC / disk paths merge like QK's ``merge_probe_parts``;
  * the opt-in autocap bounds the per-rank HS row.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py)

import torch
import torch.nn as nn

from mia.errors import MiaConfigurationError
from mia.graph.registry import get_registry

HIDDEN, CAP = 16, 64
GIB = 1 << 30


# --------------------------------------------------------------------------------------
# Fakes: a V2-shaped runner whose prepare_inputs returns a ready StepView and whose
# execute_model runs the (class-wrapped) decoder layers on deterministic per-step data.
# --------------------------------------------------------------------------------------

def _model(n_layers, hidden=HIDDEN):
    """Fresh layer classes per call (MIA wraps CLASSES), named like vLLM's Llama."""

    class Layer(nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, h, r):
            # A different, deterministic update per layer, bf16 like the real residual.
            return (h * (1 + 0.125 * self.i)).to(h.dtype), (r + 0.25 * self.i).to(r.dtype)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=n_layers,
                                          num_attention_heads=4, num_key_value_heads=2)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer(i) for i in range(n_layers)])
            self.w = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    return Model()


class _Runner:
    __module__ = "vllm.v1.worker.gpu.model_runner"

    def __init__(self, model, hidden=HIDDEN):
        self.model = model
        self.hidden = hidden
        self.next_step = None
        self.step_idx = 0
        self.block_tables = types.SimpleNamespace(
            input_block_tables=(torch.zeros((4, 3), dtype=torch.int32),))

    def add_requests(self, so):
        pass

    def finish_requests(self, so):
        pass

    def prepare_inputs(self, *a, **k):
        return self.next_step

    def execute_model(self, scheduler_output, *a, **k):
        step = self.next_step
        n = int(step.query_start_loc_np[-1])
        g = torch.Generator().manual_seed(1000 + self.step_idx)
        h = torch.randn(n, self.hidden, generator=g).to(torch.bfloat16)
        r = torch.randn(n, self.hidden, generator=g).to(torch.bfloat16)
        for layer in self.model.model.layers:
            h, r = layer(h, r)
        self.step_idx += 1
        return "executed"


def _worker(tp, rank, n_layers, hidden=HIDDEN):
    model = _model(n_layers, hidden)
    return SimpleNamespace(
        model_runner=_Runner(model, hidden),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1),
        rank=rank,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(gpu_memory_utilization=0.5),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=CAP)),
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A small explicit aperture (CPU 'devices' report 1 GiB), step_view bypassed (the fake
    runner hands the StepView straight through), a COUNTING writer-process stub, and every
    HS-layout knob unset."""
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(1 << 20))
    for k in ("MIA_HS_TP_SHARD", "MIA_HS_CAPTURE_ALL_RANKS", "MIA_HS_TP_SYMMETRIC",
              "MIA_APERTURE_SYNC_DRAIN", "MIA_APERTURE_PER_REQUEST", "MIA_ROUTE_DECODE_CACHE",
              "MIA_ROUTE_VECTORIZED", "MIA_APERTURE_WRITE_MODE", "MIA_DRAIN_SELECTIVE",
              "MIA_APERTURE_MMAP"):
        monkeypatch.delenv(k, raising=False)
    import mia.graph.install as inst
    monkeypatch.setattr(inst, "step_view", lambda runner, ib, stash: ib)
    # The registered op has only a CUDA kernel; on CPU run its implementation (the aten body the
    # CUDA op falls back to) under the same name the class wrap calls.
    from mia.graph import register_graph_ops
    from mia.graph.ops import _capture_hs_impl
    register_graph_ops()
    monkeypatch.setattr(torch.ops.mia, "capture_hs", _capture_hs_impl, raising=False)
    started = []
    import mia.graph.writer_process as wp
    monkeypatch.setattr(wp.WriterProcess, "from_env",
                        classmethod(lambda cls: started.append(1) or None))
    return SimpleNamespace(tmp=tmp_path, writers=started)


def _install(worker, base):
    from mia.graph.install_hs import install_execute_model_wrapper_hs, install_hs_hosts
    prev = os.environ.get("MIA_APERTURE_DIR")
    os.environ["MIA_APERTURE_DIR"] = str(base)
    try:
        install_hs_hosts(worker)
        install_execute_model_wrapper_hs(worker.model_runner, worker)
    finally:
        if prev is None:
            os.environ.pop("MIA_APERTURE_DIR", None)
        else:
            os.environ["MIA_APERTURE_DIR"] = prev


def _step(reqs, extra):
    """``reqs`` = [(req_id, n_tokens, is_prefill), ...]; ``extra`` = {req_id: extra_args}."""
    from mia.runner import StepView
    n = len(reqs)
    counts = np.array([t for _, t, _ in reqs], dtype=np.int32)
    qsl = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return StepView(
        req_ids=[r for r, _, _ in reqs], num_reqs=n, num_scheduled_tokens=counts,
        query_start_loc=torch.from_numpy(qsl.copy()), query_start_loc_np=qsl,
        num_computed_tokens_np=np.zeros(n, dtype=np.int32),
        prefill_len_np=counts.copy(), prompt_len_np=counts.copy(),
        is_prefilling_np=np.array([p for _, _, p in reqs], dtype=bool),
        seq_lens=torch.from_numpy(counts.copy()), block_tables=(), extra_args=extra)


def _requests(n_layers):
    """Three requests covering the builders' branches: all layers / all tokens / both phases,
    a 1-based subset with last_token, and a subset whose layers all belong to tp_rank 0 at TP4
    (layers 1 and 5) captured on prefill only."""
    return {
        "A": {"output_hidden_states": True, "hs_mode": "all_tokens", "hooks_on": "both"},
        "B": {"output_hidden_states": [2, 3, n_layers - 1], "hs_mode": "last_token",
              "hooks_on": "both"},
        "C": {"output_hidden_states": [1, 5], "hs_mode": "all_tokens", "hooks_on": "prefill"},
        "D": {"output_hidden_states": True, "hs_mode": "last_token"},
    }


def _schedule():
    return [
        [("A", 5, True), ("B", 4, True), ("C", 3, True)],
        [("A", 1, False), ("B", 1, False), ("C", 1, False)],
        [("A", 1, False), ("B", 1, False), ("C", 1, False), ("D", 6, True)],
        [("A", 1, False), ("B", 1, False), ("D", 1, False)],
    ]


def _run(worker, base, n_layers, schedule=None, extra=None):
    """Install, run every scheduled step through the REAL routing + execute_model wrappers (the
    class-wrapped layers call the real ``mia::capture_hs`` op), then flush_aperture."""
    from mia.workers.hs_capture_worker import HSCaptureWorker
    _install(worker, base)
    extra = extra or _requests(n_layers)
    runner = worker.model_runner
    for reqs in (schedule or _schedule()):
        runner.next_step = _step(reqs, extra)
        runner.prepare_inputs()
        runner.execute_model(SimpleNamespace(finished_req_ids=set()))
    return HSCaptureWorker.flush_aperture(worker)


def _digests(run_dir):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(run_dir).iterdir()) if p.is_file()}


def _art_equal(a, b):
    assert set(a) == set(b), (sorted(a), sorted(b))
    for q in a:
        assert sorted(a[q]) == sorted(b[q]), (q, sorted(a[q]), sorted(b[q]))
        for L in a[q]:
            assert a[q][L].shape == b[q][L].shape and torch.equal(a[q][L], b[q][L]), (q, L)


# --------------------------------------------------------------------------------------
# The assignment rule
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_layers,tp", [(80, 4), (80, 8), (32, 4), (32, 8)])
def test_round_robin_assignment_is_disjoint_complete_and_balanced(n_layers, tp):
    from mia.graph.tp_shard import (
        hs_layer_owner, hs_max_owned_layers, hs_owned_layers, hs_owned_rows)
    per = [hs_owned_rows(n_layers, tp, r) for r in range(tp)]
    flat = [i for rows in per for i in rows]
    assert sorted(flat) == list(range(n_layers))                 # complete
    assert len(flat) == len(set(flat))                          # disjoint
    assert {len(rows) for rows in per} == {n_layers // tp}      # balanced: 20/10/8/4 each
    for r, rows in enumerate(per):
        assert rows == [i for i in range(n_layers) if i % tp == r]   # the documented rule
        assert all(hs_layer_owner(i, tp) == r for i in rows)
        assert hs_owned_layers(n_layers, tp, r) == [i + 1 for i in rows]
    assert hs_max_owned_layers(n_layers, tp, "round_robin") == n_layers // tp
    # A contiguous subset (the last 8 layers) is spread over every rank, not dumped on one.
    last8 = list(range(n_layers - 8 + 1, n_layers + 1))
    from mia.graph.tp_shard import hs_expected_ranks
    assert hs_expected_ranks(last8, tp) == list(range(min(tp, 8)))


def test_uneven_and_fewer_layers_than_ranks():
    from mia.graph.tp_shard import hs_max_owned_layers, hs_owned_rows
    assert [len(hs_owned_rows(30, 4, r)) for r in range(4)] == [8, 8, 7, 7]
    assert hs_max_owned_layers(30, 4, "round_robin") == 8
    assert [hs_owned_rows(3, 4, r) for r in range(4)] == [[0], [1], [2], []]
    assert hs_max_owned_layers(80, 4, "rank0") == 80


def test_requested_layers_and_expected_ranks():
    from mia.graph.tp_shard import hs_expected_ranks, hs_requested_layers
    assert hs_requested_layers(True, 8) == list(range(1, 9))
    assert hs_requested_layers([8, 1, 1, 0, 99], 8) == [1, 8]     # 1-based, deduped, in range
    assert hs_expected_ranks([1, 5, 9, 13], 4) == [0]              # stride tp -> one rank
    assert hs_expected_ranks(range(1, 81), 8) == list(range(8))


def test_a_bad_shard_value_is_refused_at_engine_construction(monkeypatch):
    """Before any model is built, in the driver -- not at the HS install in a worker subprocess
    (the same seam MIA_WORKER is parsed at)."""
    import mia._plugin as plugin

    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode="NONE"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, tensor_parallel_size=4),
        use_v2_model_runner=True)
    monkeypatch.setattr(plugin, "_original_create_engine_config",
                        lambda self, *a, **k: cfg, raising=False)
    for k in ("MIA_WORKER", "MIA_ALLOW_CUDAGRAPH", "VLLM_USE_V2_MODEL_RUNNER"):
        monkeypatch.delenv(k, raising=False)
    args = SimpleNamespace(worker_extension_cls=None, enforce_eager=False,
                           compilation_config=None, pipeline_parallel_size=1,
                           tensor_parallel_size=4)
    monkeypatch.setenv("MIA_HS_TP_SHARD", "yes")
    with pytest.raises(MiaConfigurationError, match="MIA_HS_TP_SHARD"):
        plugin._patched_create_engine_config(args)
    monkeypatch.setenv("MIA_HS_TP_SHARD", "0")
    assert plugin._patched_create_engine_config(args) is cfg
    # The OTHER layout flag goes through the same seam: it WINS over MIA_HS_TP_SHARD, so a
    # spelling it does not read ("true") would silently run the sharded layout for an operator
    # who asked for replicas -- and the replication leg is the check that proves the residual
    # really is replicated across ranks.
    monkeypatch.setenv("MIA_HS_CAPTURE_ALL_RANKS", "true")
    with pytest.raises(MiaConfigurationError, match="MIA_HS_CAPTURE_ALL_RANKS"):
        plugin._patched_create_engine_config(args)
    monkeypatch.setenv("MIA_HS_CAPTURE_ALL_RANKS", "1")
    assert plugin._patched_create_engine_config(args) is cfg


def test_shard_mode_resolution(monkeypatch):
    from mia.graph.tp_shard import resolve_hs_shard_mode as mode
    assert mode(1, {}) == "single" and mode(1, {"MIA_HS_TP_SHARD": "0"}) == "single"
    assert mode(1, {"MIA_HS_CAPTURE_ALL_RANKS": "1"}) == "single"
    assert mode(4, {}) == "round_robin" and mode(4, {"MIA_HS_TP_SHARD": ""}) == "round_robin"
    assert mode(4, {"MIA_HS_TP_SHARD": "1"}) == "round_robin"
    assert mode(4, {"MIA_HS_TP_SHARD": "0"}) == "rank0"
    assert mode(4, {"MIA_HS_TP_SHARD": "0", "MIA_HS_CAPTURE_ALL_RANKS": "1"}) == "all_ranks"
    assert mode(4, {"MIA_HS_CAPTURE_ALL_RANKS": "0"}) == "round_robin"
    assert mode(4, {"MIA_HS_CAPTURE_ALL_RANKS": ""}) == "round_robin"
    # Both flags pick the LAYOUT, so neither reads a near-miss as "off" -- at EVERY tp size, so a
    # typo cannot sit unnoticed in a TP = 1 env until the day it runs at TP > 1.
    for tp in (1, 4):
        for name in ("MIA_HS_TP_SHARD", "MIA_HS_CAPTURE_ALL_RANKS"):
            for bad in ("yes", "true", "on", "TRUE", "2"):
                with pytest.raises(MiaConfigurationError, match=name):
                    mode(tp, {name: bad})


# --------------------------------------------------------------------------------------
# Install: owned layers, symmetric bake, per-rank sizing, header
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("tp", [2, 4])
def test_each_rank_installs_only_its_owned_layers(env, tp):
    from mia.graph.aperture_reader import read_sidecar_header
    n_layers = 8
    for rank in range(tp):
        w = _worker(tp, rank, n_layers)
        base = env.tmp / f"tp{tp}"
        _install(w, base)
        owned = [i for i in range(n_layers) if i % tp == rank]
        reg = get_registry(w, "hs")
        assert w._should_capture is True and reg is not None
        assert sorted(reg.hosts) == owned                            # only owned hosts register
        assert reg._hs_owned_rows == owned
        R = w._capture_aperture.n_slots
        assert R == (1 << 20) // (len(owned) * HIDDEN * 2)           # sized over OWNED layers
        for i, layer in enumerate(w.model_runner.model.model.layers):
            host = layer._mia_hs_host
            assert host.do_capture
            if i in owned:
                assert host.hs_buf.shape == (R + 1, HIDDEN)
            else:
                assert host.hs_buf.shape == (1, HIDDEN)              # sink
                assert int(host.capture_index.abs().sum()) == 0
        drain = w._hs_drain
        assert [ln for ln, _ in drain.layers] == [i + 1 for i in owned]
        run_dir = base / f"tp_rank_{rank}"
        assert sorted(p.name for p in run_dir.glob("*.raw")) == sorted(
            f"hs_layer_{i + 1}.raw" for i in owned)
        drain.close()
        hdr = read_sidecar_header(str(run_dir / "hs_aperture_meta.jsonl"))
        assert hdr == {"dtype": "bfloat16", "row_shape": [HIDDEN], "hidden": HIDDEN,
                       "tp_rank": rank, "tp_size": tp, "num_layers": n_layers,
                       "capture_all_ranks": False, "layer_shard": "round_robin",
                       "owned_layers": [i + 1 for i in owned]}
    assert len(env.writers) == tp                   # every capturing rank starts its writer


def _record_op_calls(monkeypatch):
    calls = []

    def rec(h, r, hs_buf, index, has_residual):
        calls.append((hs_buf.shape[1], hs_buf.dtype, tuple(index.shape), index.dtype,
                      int(has_residual), tuple(h.shape)))
    monkeypatch.setattr(torch.ops.mia, "capture_hs", rec, raising=False)
    return calls


def _op_list(worker, monkeypatch):
    """The op calls one forward of every decoder layer issues, in order -- what the compiled
    graph bakes on this rank."""
    calls = _record_op_calls(monkeypatch)
    h = torch.zeros(3, HIDDEN, dtype=torch.bfloat16)
    for layer in worker.model_runner.model.model.layers:
        h, r = layer(h, h)
    return list(calls)


@pytest.mark.parametrize("tp", [2, 4])
def test_symmetric_bake_same_op_list_on_every_rank(env, monkeypatch, tp):
    """The NCCL graph-capture lockstep needs the SAME ops at the SAME call sites on every rank:
    every layer of every rank bakes ``capture_hs`` with the same input shapes / dtypes and a
    routing index the width of the token cap -- owned or not. The TP=1 graph is the reference."""
    from mia.graph.install_hs import install_hs_hosts
    n_layers = 8
    w1 = _worker(1, 0, n_layers)
    install_hs_hosts(w1)
    ref = _op_list(w1, monkeypatch)
    assert len(ref) == n_layers
    for rank in range(tp):
        w = _worker(tp, rank, n_layers)
        install_hs_hosts(w)
        assert _op_list(w, monkeypatch) == ref, rank
    # The check can see an asymmetric bake: with the kill switch only owned layers bake.
    monkeypatch.setenv("MIA_HS_TP_SYMMETRIC", "0")
    w = _worker(tp, 1, n_layers)
    install_hs_hosts(w)
    assert len(_op_list(w, monkeypatch)) == n_layers // tp


def test_per_rank_aperture_sizing_at_the_70b_shape(monkeypatch):
    """Rank 0 of 70B TP4 captures 20 of 80 layers: one 8192-token step is 2.5 GiB (it was 10 GiB
    for all 80 on rank 0). The default stays max(4 GiB, one step); an explicit per-rank value
    buys tp x the rows an all-layer rank got."""
    from mia.graph.install_hs import _resolve_aperture_rows
    from mia.graph.tp_shard import hs_owned_rows
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda d: SimpleNamespace(total_memory=80 * GIB))
    w = SimpleNamespace(vllm_config=SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.85)))
    n_owned = len(hs_owned_rows(80, 4, 0))
    assert n_owned == 20
    step = 8192 * n_owned * 8192 * 2
    assert step == int(2.5 * GIB)
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(step))          # the harness' one-step value
    R, nbytes = _resolve_aperture_rows(w, n_owned, 8192, torch.bfloat16, "cuda:0",
                                       rows_needed=8192)
    assert (R, nbytes) == (8192, step)
    old_R, old_bytes = _resolve_aperture_rows(w, 80, 8192, torch.bfloat16, "cuda:0",
                                              rows_needed=8192)
    assert old_R == 2048 and 10 * GIB - step == int(7.5 * GIB)       # what rank 0 frees
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES")
    R, nbytes = _resolve_aperture_rows(w, n_owned, 8192, torch.bfloat16, "cuda:0",
                                       rows_needed=8192)
    assert nbytes == 4 * GIB and R == (4 * GIB) // (20 * 8192 * 2)  # 4 GiB floor, 1.6 steps


def test_tp1_install_is_the_old_install(env):
    from mia.graph.aperture_reader import read_sidecar_header
    w = _worker(1, 0, 8)
    _install(w, env.tmp / "tp1")
    reg = get_registry(w, "hs")
    assert reg._hs_owned_rows is None and reg._hs_owned_set is None   # routing unfiltered
    assert sorted(reg.hosts) == list(range(8))
    w._hs_drain.close()
    hdr = read_sidecar_header(str(env.tmp / "tp1" / "tp_rank_0" / "hs_aperture_meta.jsonl"))
    assert set(hdr) == {"dtype", "row_shape", "hidden", "tp_rank", "tp_size", "num_layers",
                        "capture_all_ranks"}


def test_rank0_ab_layout_keeps_the_old_header_and_sinks(env, monkeypatch):
    from mia.graph.aperture_reader import read_sidecar_header
    monkeypatch.setenv("MIA_HS_TP_SHARD", "0")
    w0, w1 = _worker(2, 0, 8), _worker(2, 1, 8)
    _install(w0, env.tmp / "ab")
    _install(w1, env.tmp / "ab")
    assert sorted(get_registry(w0, "hs").hosts) == list(range(8))
    assert get_registry(w1, "hs") is None and getattr(w1, "_hs_drain", None) is None
    assert not (env.tmp / "ab" / "tp_rank_1").exists()
    w0._hs_drain.close()
    hdr = read_sidecar_header(str(env.tmp / "ab" / "tp_rank_0" / "hs_aperture_meta.jsonl"))
    assert "owned_layers" not in hdr and "layer_shard" not in hdr and hdr["tp_size"] == 2


def test_all_ranks_diagnostic_captures_every_layer_on_every_rank(env, monkeypatch):
    from mia.graph.aperture_reader import read_sidecar_header
    monkeypatch.setenv("MIA_HS_CAPTURE_ALL_RANKS", "1")
    for rank in range(2):
        w = _worker(2, rank, 8)
        _install(w, env.tmp / "all")
        assert sorted(get_registry(w, "hs").hosts) == list(range(8))
        w._hs_drain.close()
        hdr = read_sidecar_header(
            str(env.tmp / "all" / f"tp_rank_{rank}" / "hs_aperture_meta.jsonl"))
        assert hdr["capture_all_ranks"] is True and "owned_layers" not in hdr


def test_fewer_layers_than_ranks_leaves_a_pure_sink_rank(env):
    w = _worker(4, 3, 3)
    _install(w, env.tmp / "few")
    assert w._should_capture is False and get_registry(w, "hs") is None
    assert w._writer_mode == "none (HS sink rank: captures nothing)"
    assert not (env.tmp / "few" / "tp_rank_3").exists()


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("builder", ["legacy", "vectorized", "decode_cache"])
def test_routing_keeps_only_owned_layers(env, builder):
    from mia.graph.install_hs import _build_routing_hs, install_hs_hosts
    n_layers, tp = 8, 4
    kw = {"legacy": dict(vectorized=False, decode_cache=False),
          "vectorized": dict(vectorized=True, decode_cache=False),
          "decode_cache": dict(decode_cache=True)}[builder]
    extra = _requests(n_layers)
    step = _step([("A", 5, True), ("B", 4, True), ("C", 3, True)], extra)

    def route(tp_, rank):
        w = _worker(tp_, rank, n_layers)
        install_hs_hosts(w)
        reg = get_registry(w, "hs")
        reg.reset_pinned(CAP)
        _build_routing_hs(step, reg, **kw)
        return reg, {rec.req_id: list(rec.layers) for rec in reg._hs_step_entries}

    _, ref = route(1, 0)
    union: dict = {}
    for rank in range(tp):
        reg, got = route(tp, rank)
        owned1 = {i + 1 for i in range(n_layers) if i % tp == rank}
        for q, layers in got.items():
            assert set(layers) <= owned1
            assert layers == [L for L in ref[q] if L in owned1]      # order kept
            union.setdefault(q, []).extend(layers)
        # A request with none of this rank's layers reserves nothing and records nothing.
        if not (set(ref["C"]) & owned1):
            assert "C" not in got
        # Unowned layers' routing rows stay at the sentinel (nothing routed there).
        pinned = reg.capture_index_pinned
        for i in range(n_layers):
            if i % tp != rank:
                assert bool((pinned[i] == reg.sentinel_row).all())
        rows = sum(1 if extra[q]["hs_mode"] == "last_token" else n
                   for q, n, _ in [("A", 5, 1), ("B", 4, 1), ("C", 3, 1)] if q in got)
        assert reg._hs_step_rows == rows
    assert {q: sorted(v) for q, v in union.items()} == {q: sorted(v) for q, v in ref.items()}


# --------------------------------------------------------------------------------------
# End to end: capture -> per-rank drains -> reader union
# --------------------------------------------------------------------------------------

def _run_tp(env, tp, n_layers, name, **kw):
    base = env.tmp / name
    dirs = []
    for rank in range(tp):
        d = _run(_worker(tp, rank, n_layers), base, n_layers, **kw)
        dirs.append(d)
    return base, dirs


@pytest.mark.parametrize("drain", ["sync", "offloop"])
@pytest.mark.parametrize("tp", [2, 4])
def test_sharded_capture_reads_back_equal_to_tp1(env, monkeypatch, drain, tp):
    from mia.graph.aperture_reader import load_hs_aperture_tp, merge_hs_aperture_ranks
    if drain == "sync":
        monkeypatch.setenv("MIA_APERTURE_SYNC_DRAIN", "1")
    n_layers = 8
    base1, dirs1 = _run_tp(env, 1, n_layers, "tp1")
    base, dirs = _run_tp(env, tp, n_layers, f"tp{tp}")
    assert [os.path.basename(d) for d in dirs] == [f"tp_rank_{r}" for r in range(tp)]
    ref = load_hs_aperture_tp(str(base1))
    got = load_hs_aperture_tp(str(base))
    _art_equal(got, ref)
    _art_equal(merge_hs_aperture_ranks(dirs), ref)          # the flush_aperture dirs, merged
    # The all-layer request's per-layer raw rows equal the TP=1 file's rows of that layer.
    assert set(ref["A"]) == set(range(1, n_layers + 1))


def test_all_layer_workload_raw_files_are_byte_identical_to_tp1(env):
    """With every request on every layer, each rank reserves exactly what TP=1 reserves, so each
    owned layer's raw file is byte-for-byte the TP=1 file of that layer."""
    n_layers, tp = 8, 4
    extra = {"A": {"output_hidden_states": True, "hs_mode": "all_tokens", "hooks_on": "both"},
             "D": {"output_hidden_states": True, "hs_mode": "last_token"}}
    sched = [[("A", 5, True), ("D", 3, True)], [("A", 1, False), ("D", 1, False)]]
    base1, _ = _run_tp(env, 1, n_layers, "tp1", schedule=sched, extra=extra)
    base, _ = _run_tp(env, tp, n_layers, "tp4", schedule=sched, extra=extra)
    d1 = _digests(base1 / "tp_rank_0")
    for rank in range(tp):
        dr = _digests(base / f"tp_rank_{rank}")
        raws = {k: v for k, v in dr.items() if k.endswith(".raw")}
        assert len(raws) == n_layers // tp
        for k, v in raws.items():
            assert v == d1[k], (rank, k)


def test_a_subset_capture_reads_back_from_the_empty_rank_dirs(env):
    """A run whose only request wants layers 1 and 5 captures on tp_rank 0 alone. The other ranks
    still install their drain, so their header-only sidecars are on disk and the union reads the
    capture with no gap: "captured nothing" is distinguishable from "lost"."""
    from mia.graph.aperture_reader import load_hs_aperture_tp
    extra = {"C": {"output_hidden_states": [1, 5], "hs_mode": "all_tokens", "hooks_on": "both"}}
    sched = [[("C", 3, True)], [("C", 1, False)]]
    base1, _ = _run_tp(env, 1, 8, "tp1", schedule=sched, extra=extra)
    base, dirs = _run_tp(env, 4, 8, "tp4", schedule=sched, extra=extra)
    assert dirs == [str(base / "tp_rank_0"), None, None, None]
    for r in range(4):
        assert (base / f"tp_rank_{r}" / "hs_aperture_meta.jsonl").is_file()
    got = load_hs_aperture_tp(str(base))
    _art_equal(got, load_hs_aperture_tp(str(base1)))
    assert sorted(got["C"]) == [1, 5]


def test_reader_refuses_a_gap(env):
    from mia.graph.aperture_reader import load_hs_aperture_tp, merge_hs_aperture_ranks
    from mia.graph.tp_shard import TPShardError
    base, dirs = _run_tp(env, 4, 8, "tp4")
    with pytest.raises(TPShardError, match=r"missing tp_rank\(s\) \[2\]"):
        merge_hs_aperture_ranks([d for d in dirs if not d.endswith("tp_rank_2")])
    shutil.rmtree(dirs[2])
    with pytest.raises(TPShardError, match=r"missing tp_rank\(s\) \[2\]"):
        load_hs_aperture_tp(str(base))
    # A capture that only ever wanted rank 0 / 1 / 3 layers may omit rank 2 -- if the caller says so.
    got = load_hs_aperture_tp(str(base), expected_layers=[1, 2, 4])
    assert set(got["A"]) == {1, 2, 4, 5, 6, 8}


def test_reader_refuses_duplicates(env, tmp_path):
    from mia.graph.aperture_reader import merge_hs_aperture_ranks
    from mia.graph.tp_shard import TPShardError
    base, dirs = _run_tp(env, 4, 8, "tp4")
    twin = tmp_path / "elsewhere" / "tp_rank_1"
    shutil.copytree(dirs[1], twin)
    with pytest.raises(TPShardError, match=r"duplicate tp_rank\(s\) \[1\]"):
        merge_hs_aperture_ranks(dirs + [str(twin)])


def _rewrite_header(run_dir, **changes):
    meta = Path(run_dir) / "hs_aperture_meta.jsonl"
    lines = meta.read_text().splitlines(keepends=True)
    hdr = json.loads(lines[0])
    hdr["__header__"].update(changes)
    lines[0] = json.dumps(hdr) + "\n"
    meta.write_text("".join(lines))


def test_reader_refuses_a_header_that_lies(env):
    from mia.graph.aperture_reader import load_hs_aperture_tp
    from mia.graph.tp_shard import TPShardError
    base, dirs = _run_tp(env, 4, 8, "tp4")
    _rewrite_header(dirs[1], owned_layers=[2, 6, 7])
    with pytest.raises(TPShardError, match="disagrees with the round-robin rule"):
        load_hs_aperture_tp(str(base))


def test_reader_refuses_a_layer_a_rank_does_not_own(env):
    """A sidecar entry naming another rank's layer: the rows would be double-sourced."""
    from mia.graph.aperture_reader import load_hs_aperture_tp
    from mia.graph.tp_shard import TPShardError
    base, dirs = _run_tp(env, 4, 8, "tp4")
    meta = Path(dirs[1]) / "hs_aperture_meta.jsonl"
    lines = meta.read_text().splitlines(keepends=True)
    e = json.loads(lines[1])
    e["l"] = 1                                           # rank 0's layer
    lines.append(json.dumps(e) + "\n")
    shutil.copy(Path(dirs[1]) / f"hs_layer_{json.loads(lines[1])['l']}.raw",
                Path(dirs[1]) / "hs_layer_1.raw")
    meta.write_text("".join(lines))
    with pytest.raises(TPShardError, match="does not own"):
        load_hs_aperture_tp(str(base))


def test_reader_refuses_ranks_that_captured_different_tokens(env):
    from mia.graph.aperture_reader import load_hs_aperture_tp
    from mia.graph.tp_shard import TPShardError
    base, dirs = _run_tp(env, 2, 8, "tp2")
    meta = Path(dirs[1]) / "hs_aperture_meta.jsonl"
    lines = meta.read_text().splitlines(keepends=True)
    kept = [lines[0]] + [ln for ln in lines[1:] if not (json.loads(ln)["r"] == "A"
                                                          and json.loads(ln)["s"] == 1)]
    meta.write_text("".join(kept))
    with pytest.raises(TPShardError, match="different row counts"):
        load_hs_aperture_tp(str(base))


def test_reader_refuses_mixed_sharded_and_unsharded_dirs(env, tmp_path):
    from mia.graph.aperture_reader import merge_hs_aperture_ranks
    from mia.graph.tp_shard import TPShardError
    _, dirs1 = _run_tp(env, 1, 8, "tp1")
    _, dirs = _run_tp(env, 2, 8, "tp2")
    with pytest.raises(TPShardError, match="some HS dirs carry a layer-shard header"):
        merge_hs_aperture_ranks([dirs[0], dirs1[0]])


def test_rank0_ab_run_reads_back_equal_and_unsharded(env, monkeypatch):
    from mia.graph.aperture_reader import load_hs_aperture_tp
    monkeypatch.setenv("MIA_HS_TP_SHARD", "0")
    base1, _ = _run_tp(env, 1, 8, "tp1")
    base, dirs = _run_tp(env, 2, 8, "ab")
    assert dirs == [str(base / "tp_rank_0"), None]
    _art_equal(load_hs_aperture_tp(str(base)), load_hs_aperture_tp(str(base1)))


def test_all_ranks_replicas_are_checked_bitwise(env, monkeypatch):
    from mia.graph.aperture_reader import load_hs_aperture_tp
    from mia.graph.tp_shard import TPShardError
    monkeypatch.setenv("MIA_HS_CAPTURE_ALL_RANKS", "1")
    base1, _ = _run_tp(env, 1, 8, "tp1")
    monkeypatch.setenv("MIA_HS_CAPTURE_ALL_RANKS", "1")
    base, dirs = _run_tp(env, 2, 8, "all")
    _art_equal(load_hs_aperture_tp(str(base), check_replicas=True),
               load_hs_aperture_tp(str(base1)))
    raw = Path(dirs[1]) / "hs_layer_3.raw"
    b = bytearray(raw.read_bytes())
    b[0] ^= 1
    raw.write_bytes(bytes(b))
    load_hs_aperture_tp(str(base))                      # the plain read takes rank 0's copy
    with pytest.raises(TPShardError, match="not bitwise equal"):
        load_hs_aperture_tp(str(base), check_replicas=True)


# --------------------------------------------------------------------------------------
# TP = 1 byte identity against the pre-shard code (fd73aa6)
# --------------------------------------------------------------------------------------

#: sha256 of every file the TP=1 pipeline below writes, taken by running this exact pipeline
#: against MIA fd73aa6 (the parent of the layer shard; its three routing builders and its
#: auto / legacy write paths all produced these). Two sets because the synchronous drain is a
#: FULL drain while the off-loop drain is SELECTIVE (MIA_DRAIN_SELECTIVE, default on), so the
#: subset request's layers differ in length. Every builder x drain combination must reproduce its
#: set: the layer shard is a strict no-op at TP = 1.
TP1_GOLDEN = {
    "sync": {
        "hs_aperture_meta.jsonl":
            "3988f2354f9472d59ba02ca4e8497c9b41301f76e8796703a7483179a2df16e2",
        "hs_layer_1.raw":
            "26523d4403d0a81d6016160e61553eace5aaeba583e06632626589a64d5d5886",
        "hs_layer_2.raw":
            "73ef32e3655dd9c4e813a8c31c067284b5f16f51fe9f6e20aed551037864884a",
        "hs_layer_3.raw":
            "1349c2cfb22fe23edb8889c93a01fc722aff965ee4d31162c61cf00f66fd8901",
        "hs_layer_4.raw":
            "cf652a220eb4c65c9387f39c9a6631738c32dfcad75f1da714250d228af902d7",
        "hs_layer_5.raw":
            "b5a6bfd2feb8e4fe8dab41a415f443b59ccebbf65aa79c010019af0799f8b44a",
        "hs_layer_6.raw":
            "7c44a2db9f4bf21cf165645f72524925b9e2b84a8856ef2ddefe5077c7418dc3",
        "hs_layer_7.raw":
            "f87b796357c9212cc5e4f2aac6700e363f4a8af908074966783a3f5add1c1f75",
        "hs_layer_8.raw":
            "c517b4bbf56e2470aa9ec97c7460731c61712ee0afa8c045401050a9cc4ea67e",
    },
    "offloop": {
        "hs_aperture_meta.jsonl":
            "be7b9e91ed14c48a0b6389b467fe2a06e807a3f46fbb4334d5566fb988af17b0",
        "hs_layer_1.raw":
            "d4e4341be549772b6fadf1002fd795776ea844d408ea9597ff358bc839035904",
        "hs_layer_2.raw":
            "ee2796521edb9e2b979c209a77b9e9964232d6ce74903185063abe43105a6533",
        "hs_layer_3.raw":
            "08499e2dc2c6ba704d851b9805ea4a88ad5abdf41a2b34312e5e7a0535597c5c",
        "hs_layer_4.raw":
            "acb3639ccd21414e827b79a5e4afd9be527deffd3ddd6dfeb8f761d4f6347410",
        "hs_layer_5.raw":
            "b3f401f286df0d8993c599cce6d15afb18eb1229d90b16c465f5260fc2e69a71",
        "hs_layer_6.raw":
            "c3b72307117af32d7a619ecdc9d69309f790f2471ddb900dcde2d65d508eddab",
        "hs_layer_7.raw":
            "5b5d42756442f967ad606e9eb4436de721af89347a3d508c2099988d0539d7eb",
        "hs_layer_8.raw":
            "86da33877743da2d63cc220d5934fdd5e99c907c983d396d11be91b59b777279",
    },
}


@pytest.mark.parametrize("builder", ["decode_cache", "vectorized", "legacy"])
@pytest.mark.parametrize("drain", ["sync", "offloop-auto", "offloop-legacy"])
def test_tp1_artifacts_are_byte_identical_to_the_pre_shard_code(env, monkeypatch, builder,
                                                                 drain):
    if builder != "decode_cache":
        monkeypatch.setenv("MIA_ROUTE_DECODE_CACHE", "0")
    if builder == "vectorized":
        monkeypatch.setenv("MIA_ROUTE_VECTORIZED", "1")
    if drain == "sync":
        monkeypatch.setenv("MIA_APERTURE_SYNC_DRAIN", "1")
    elif drain == "offloop-legacy":
        monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "legacy")
    d = _run(_worker(1, 0, 8), env.tmp / "tp1", 8)
    assert _digests(d) == TP1_GOLDEN["sync" if drain == "sync" else "offloop"]


# --------------------------------------------------------------------------------------
# Per-request RPC / disk delivery, merged like QK's merge_probe_parts
# --------------------------------------------------------------------------------------

def _hs_parts(tp, n_layers, layers, rows=3, drop_rank=None):
    """What each rank's get_aperture_per_request returns for one request (decompressed)."""
    import pickle
    import zstandard as zstd
    from mia.graph.tp_shard import HSShard
    from mia.workers.hs_capture_worker import _marshal_perreq_hs
    parts = []
    for r in range(tp):
        shard = HSShard.of(r, tp, n_layers)
        mine = {L: torch.full((rows, HIDDEN), float(L)) for L in layers
                if L in shard.owned_layers}
        if not mine or r == drop_rank:
            continue
        blob = _marshal_perreq_hs(mine, {"num_layers": n_layers}, shard)
        parts.append(pickle.loads(zstd.ZstdDecompressor().decompress(blob)))
    return parts


def test_merge_probe_parts_unions_hs_layer_shards():
    from mia._plugin import _expected_probe_parts, merge_probe_parts
    from mia.graph.tp_shard import HS_SHARD_KEY
    parts = _hs_parts(4, 8, range(1, 9))
    assert _expected_probe_parts(parts, True) == 4
    merged = merge_probe_parts(parts, True)
    assert HS_SHARD_KEY not in merged and list(merged["hs_cache"]) == list(range(1, 9))
    for L, e in merged["hs_cache"].items():
        assert e["layer_num"] == L and torch.equal(e["hidden_states"],
                                                   torch.full((3, HIDDEN), float(L)))
    # A subset request: only its layers' owners deliver, and that is the whole answer.
    sub = _hs_parts(4, 8, [1, 5])
    assert len(sub) == 1 and _expected_probe_parts(sub, [1, 5]) == 1
    assert list(merge_probe_parts(sub, [1, 5])["hs_cache"]) == [1, 5]


def test_merge_probe_parts_refuses_hs_gaps_and_duplicates():
    from mia._plugin import merge_probe_parts
    from mia.graph.tp_shard import TPShardError
    with pytest.raises(TPShardError, match=r"missing tp_rank\(s\) \[2\]"):
        merge_probe_parts(_hs_parts(4, 8, range(1, 9), drop_rank=2), True)
    parts = _hs_parts(4, 8, range(1, 9))
    with pytest.raises(TPShardError, match="duplicate"):
        merge_probe_parts(parts + [parts[1]], True)
    bad = _hs_parts(4, 8, range(1, 9))
    bad[1]["hs_cache"][1] = bad[0]["hs_cache"][1]           # rank 1 delivers rank 0's layer 1
    with pytest.raises(TPShardError, match="does not own"):
        merge_probe_parts(bad, True)
    short = _hs_parts(4, 8, range(1, 9))
    short[3]["hs_cache"][4]["hidden_states"] = torch.zeros(2, HIDDEN)
    with pytest.raises(TPShardError, match="different row counts"):
        merge_probe_parts(short, True)


def test_tp1_and_unsharded_hs_payloads_are_unchanged():
    import pickle
    import zstandard as zstd
    from mia._plugin import merge_probe_parts
    from mia.graph.tp_shard import HS_SHARD_KEY
    from mia.workers.hs_capture_worker import _marshal_perreq_hs
    p = pickle.loads(zstd.ZstdDecompressor().decompress(
        _marshal_perreq_hs({1: torch.ones(2, HIDDEN)}, {"num_layers": 8})))
    assert HS_SHARD_KEY not in p and set(p) == {"hs_cache", "config"}
    assert merge_probe_parts([p]) is p


def _perreq_worker(tp, rank, tmp_path, monkeypatch):
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.capture_aperture import CaptureAperture
    ap = CaptureAperture(row_bytes=HIDDEN * 2, n_slots=16, device="cpu", dtype=torch.bfloat16,
                         row_shape=(HIDDEN,))
    layers = [(L, torch.zeros(17, HIDDEN, dtype=torch.bfloat16)) for L in (rank + 1,)]
    drain = OffLoopApertureDrain(ap, layers, str(tmp_path / f"tp_rank_{rank}"),
                                 {"dtype": "bfloat16", "row_shape": [HIDDEN]}, per_request=True)
    w = SimpleNamespace(_hs_drain=drain, _tp_rank=rank, _hs_tp_size=tp,
                        _hs_shard_mode="round_robin" if tp > 1 else "single",
                        _conf={"num_layers": 8, "hidden_size": HIDDEN})
    drain._note_unstaged_finish = tp > 1
    return w, drain


def test_disk_route_goes_to_a_per_rank_dest_and_unstaged_ranks_answer_none(tmp_path,
                                                                           monkeypatch):
    from mia.workers.hs_capture_worker import HSCaptureWorker
    w, drain = _perreq_worker(4, 1, tmp_path, monkeypatch)
    try:
        assert HSCaptureWorker.route_aperture_to_disk(w, "req", str(tmp_path / "dest")) is True
        assert drain._disk_routed["req"] == str(tmp_path / "dest" / "tp_rank_1")
        drain._handle_finish("req")              # it finished; none of its layers are rank 1's
        for _ in range(3):                       # every poll gets the same answer
            assert HSCaptureWorker.confirm_aperture_delivery(w, "req", 0.0) is None
        drain.clear_request_disk("req")          # the serve path's end-of-request cleanup
        assert not drain.finished_unstaged("req")
    finally:
        if drain._offload is not None:
            drain._offload.close()


def test_tp1_disk_route_and_confirm_are_unchanged(tmp_path, monkeypatch):
    from mia.workers.hs_capture_worker import HSCaptureWorker
    w, drain = _perreq_worker(1, 0, tmp_path, monkeypatch)
    try:
        assert HSCaptureWorker.route_aperture_to_disk(w, "req", str(tmp_path / "dest")) is True
        assert drain._disk_routed["req"] == str(tmp_path / "dest")
        drain._handle_finish("req")
        assert HSCaptureWorker.confirm_aperture_delivery(w, "req", 0.0) is False
    finally:
        if drain._offload is not None:
            drain._offload.close()


# --------------------------------------------------------------------------------------
# flush_aperture, autocap
# --------------------------------------------------------------------------------------

def test_flush_aperture_returns_every_sharded_rank_dir_that_holds_rows(env):
    base, dirs = _run_tp(env, 4, 8, "tp4")
    assert dirs == [str(base / f"tp_rank_{r}") for r in range(4)]
    extra = {"C": {"output_hidden_states": [1, 5], "hs_mode": "all_tokens", "hooks_on": "both"}}
    base, dirs = _run_tp(env, 4, 8, "only0", schedule=[[("C", 3, True)]], extra=extra)
    assert dirs == [str(base / "tp_rank_0"), None, None, None]


def _cfg70b(tp):
    text = SimpleNamespace(num_hidden_layers=80, hidden_size=8192, num_attention_heads=64,
                           num_key_value_heads=8, head_dim=128)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text, dtype="torch.bfloat16"),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.85),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp))


@pytest.mark.parametrize("tp,shard,want", [(1, None, 80), (4, None, 20), (8, None, 10),
                                           (4, "0", 80), (4, "1", 20)])
def test_autocap_bounds_the_per_rank_hs_layers(monkeypatch, tp, shard, want):
    from mia._plugin import _hs_layers_per_rank
    monkeypatch.delenv("MIA_HS_CAPTURE_ALL_RANKS", raising=False)
    if shard is None:
        monkeypatch.delenv("MIA_HS_TP_SHARD", raising=False)
    else:
        monkeypatch.setenv("MIA_HS_TP_SHARD", shard)
    assert _hs_layers_per_rank(_cfg70b(tp), 80) == want


# --------------------------------------------------------------------------------------
# The OFFLINE per-request route (LLM.generate): merge a COMPLETE rank set, never a partial one
# --------------------------------------------------------------------------------------

def _hs_blobs(tp, n_layers, layers, rows=3, silent=()):
    """What each rank's ``get_aperture_per_request`` RPC returns for one request: the marshaled
    ZSTD-pickle bytes, or None for a rank that owns none of the request's layers (or that is
    listed in ``silent`` -- a rank whose consumer never finishes it)."""
    from mia.graph.tp_shard import HSShard
    from mia.workers.hs_capture_worker import _marshal_perreq_hs
    blobs = []
    for r in range(tp):
        shard = HSShard.of(r, tp, n_layers) if tp > 1 else None
        owned = shard.owned_layers if shard is not None else list(layers)
        mine = {L: torch.full((rows, HIDDEN), float(L)) for L in layers if L in owned}
        blobs.append(None if not mine or r in silent
                     else _marshal_perreq_hs(mine, {"num_layers": n_layers}, shard))
    return blobs


class _ApertureLLM:
    """The driver side of the offline aperture route. Each rank delivers its part EXACTLY ONCE,
    on the RPC round named by ``ready`` -- mirroring ``get_aperture_per_request``, which POPS the
    request out of that rank's stash, and the off-loop consumers, which process a request's
    ``_Finish`` on their own schedules."""

    def __init__(self, blobs, ready=None):
        self.blobs = blobs
        self.ready = ready or [0] * len(blobs)
        self.rounds = 0
        self.delivered: set = set()
        self._mia_installed = True

    def collective_rpc(self, method, args=()):
        if method != "get_aperture_per_request":
            return [None] * len(self.blobs)
        rnd, self.rounds = self.rounds, self.rounds + 1
        out = []
        for r, blob in enumerate(self.blobs):
            ok = blob is not None and r not in self.delivered and rnd >= self.ready[r]
            if ok:
                self.delivered.add(r)
            out.append(blob if ok else None)
        return out


@pytest.fixture
def offline_aperture(monkeypatch):
    """The offline HS aperture per-request route armed, with ``generate`` itself stubbed out."""
    from mia import _plugin
    monkeypatch.setenv("MIA_APERTURE_PER_REQUEST", "1")
    monkeypatch.setenv("MIA_ALLOW_CUDAGRAPH", "1")
    monkeypatch.setenv("MIA_APERTURE_DELIVER_TIMEOUT_S", "5")
    monkeypatch.delenv("MIA_STORAGE_ROUTER", raising=False)
    monkeypatch.setattr(_plugin, "_graph_mode", lambda: True)

    def generate(llm, hs_layers=True, req="req-0"):
        out = SimpleNamespace(request_id=req, probes=None)
        monkeypatch.setattr(_plugin, "_original_llm_generate",
                            lambda self, p, sp, **kw: [out])
        sp = SimpleNamespace(extra_args={"output_hidden_states": hs_layers,
                                         "hs_mode": "all_tokens"})
        assert _plugin._patched_llm_generate(llm, ["p"], sp) == [out]
        return out

    return generate


def test_offline_per_request_waits_for_every_owning_rank_before_merging(offline_aperture):
    """THE REGRESSION. At TP=4 an all-layer request is 4 parts, and each rank's off-loop consumer
    delivers on its own schedule -- so the first rounds hold a PARTIAL set. Merging that raises
    TPShardError ("missing tp_rank(s)") straight out of generate(); dropping it silently would
    lose the parts the round already popped. It must accumulate and merge the complete set."""
    from mia.graph.tp_shard import HS_SHARD_KEY
    llm = _ApertureLLM(_hs_blobs(4, 8, range(1, 9)), ready=[0, 1, 2, 3])
    out = offline_aperture(llm)
    assert llm.rounds == 4 and llm.delivered == {0, 1, 2, 3}
    assert HS_SHARD_KEY not in out.probes
    assert list(out.probes["hs_cache"]) == list(range(1, 9))
    for L, e in out.probes["hs_cache"].items():
        assert torch.equal(e["hidden_states"], torch.full((3, HIDDEN), float(L)))


def test_offline_per_request_gives_up_loudly_when_a_rank_never_delivers(offline_aperture,
                                                                        monkeypatch, capsys):
    """A rank that never finishes the request: bounded, LOUD, probes left unset -- never a
    TPShardError out of generate(), and never a half-width answer."""
    monkeypatch.setenv("MIA_APERTURE_DELIVER_TIMEOUT_S", "0.1")
    llm = _ApertureLLM(_hs_blobs(4, 8, range(1, 9), silent=(2,)))
    out = offline_aperture(llm)
    assert out.probes is None and llm.rounds > 1 and llm.delivered == {0, 1, 3}
    msg = capsys.readouterr().out
    assert "PER-REQUEST DELIVERY INCOMPLETE" in msg and "'req-0'" in msg
    assert "3 of 4 rank part(s) held" in msg and "DISCARDED" in msg


def test_offline_per_request_answers_on_the_first_round_when_one_part_is_the_whole_answer(
        offline_aperture):
    """Unchanged behaviour, and no new waiting, wherever ONE part is the complete answer:
    TP=1, a subset request whose layers one rank owns, and the MIA_HS_TP_SHARD=0 / all-ranks
    layouts (shard-less payloads). "Nothing delivered yet" still answers None at once."""
    llm = _ApertureLLM(_hs_blobs(1, 8, range(1, 9)))
    out = offline_aperture(llm)
    assert llm.rounds == 1 and list(out.probes["hs_cache"]) == list(range(1, 9))

    llm = _ApertureLLM([None])                      # its off-loop finish has not been processed
    assert offline_aperture(llm).probes is None and llm.rounds == 1

    sub = _ApertureLLM(_hs_blobs(4, 8, [1, 5]))     # layers 1 and 5 are both tp_rank 0's
    out = offline_aperture(sub, hs_layers=[1, 5])
    assert sub.rounds == 1 and sub.delivered == {0}
    assert list(out.probes["hs_cache"]) == [1, 5]

    unsharded = _ApertureLLM([_hs_blobs(1, 8, range(1, 9))[0], None, None, None])
    out = offline_aperture(unsharded)
    assert unsharded.rounds == 1 and list(out.probes["hs_cache"]) == list(range(1, 9))
