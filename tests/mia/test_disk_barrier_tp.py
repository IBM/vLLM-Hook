"""The save_to_disk durability barrier and the eager disk loaders at TP > 1 (GPU-free).

Since the writer process runs on every TP rank, the eager QK ``flush_disk`` is ASYNCHRONOUS at
TP > 1: each rank hands its shard to its own writer child, and the shards land on their own
schedules. The barrier used to return as soon as the run dir's file set was non-empty and stable
for one poll -- i.e. when the FIRST rank landed -- and the loader then returned that one rank's
heads as the whole artifact. Reproduced here with two rank writers landing at different times.

The fixes pinned here:
  * ``flush_disk`` returns the rank dir it wrote (or handed to its writer), False when it
    captured nothing;
  * both barriers (offline ``_wait_disk_artifact``, serve ``durable_wait``) wait for EVERY rank
    dir the flush_disk collective named;
  * the QK disk loaders refuse a partial TP > 1 shard set (a lone shard included) instead of
    returning 1/tp of the heads.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")  # `import mia` reaches vllm through mia/llm.py

import torch

from mia import _plugin
from mia.graph.tp_shard import TP_SHARD_KEY, TPShardError, qk_shard, rank_dir_name

H_Q, H_KV, D, TP = 4, 2, 8, 2
LAYER = "model.layers.0.self_attn.attn"
#: Rank 0 lands almost at once; rank 1 lands well after the barrier's 5 ms stability window.
LAND_DELAY_S = {0: 0.01, 1: 0.30}


def _global_qk(seq=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(seq, H_Q * D, generator=g, dtype=torch.float32),
            torch.randn(seq, H_KV * D, generator=g, dtype=torch.float32))


def _rank_slice(t, r, n_heads):
    per = n_heads // TP
    return t[..., r * per * D:(r + 1) * per * D].contiguous()


class _DelayedWriter:
    """Stand-in for one rank's writer child: submit() returns at once (as the real queue put
    does) and the artifact lands ``delay`` seconds later through the real ``write_artifact``."""

    def __init__(self, delay: float):
        self.delay = delay
        self.threads: list = []

    def submit(self, worker_kind, cpu_cache, run_dir, default_mode, tp_rank, use_st, force_pt,
               pt_filename, *, req_ids=None, block=False):
        from mia.graph.artifact_writer import write_artifact

        def land():
            time.sleep(self.delay)
            write_artifact(worker_kind, cpu_cache, run_dir, default_mode, tp_rank, use_st,
                           force_pt, pt_filename)

        th = threading.Thread(target=land, daemon=True)
        th.start()
        self.threads.append(th)
        return True


def _qk_rank_worker(r: int, req_id: str, q, k, writer=None, tp=TP):
    """A fake TP worker carrying just what QKCaptureWorker.flush_disk reads."""
    return SimpleNamespace(
        _disk_states={req_id: {LAYER: {"q": [_rank_slice(q, r, H_Q)],
                                       "k_all": [_rank_slice(k, r, H_KV)],
                                       "layer_num": 0, "hookq_mode": "all_tokens"}}},
        _conf={"num_attention_heads": H_Q, "num_key_value_heads": H_KV, "head_dim": D},
        hookq_mode="all_tokens", _tp_rank=r, _qk_shard=qk_shard(r, tp, H_Q, H_KV, D),
        _capture_consumer=None, _egress_stream=None, _writer_process=writer)


class _FakeLLM:
    """The driver side: collective_rpc runs each fake rank's REAL flush_disk, in rank order,
    exactly as vLLM's executor returns one result per worker."""

    def __init__(self, ranks):
        self.ranks = ranks
        self._mia_installed = True
        self.calls: list = []

    def _rpc(self, method, args=()):
        self.calls.append(method)
        if method == "flush_disk":
            from mia.workers.qk_capture_worker import QKCaptureWorker
            return [QKCaptureWorker.flush_disk(w, *args) for w in self.ranks]
        return [None for _ in self.ranks]

    def collective_rpc(self, method, args=()):
        return self._rpc(method, args)


class _FakeAsyncLLM(_FakeLLM):
    async def collective_rpc(self, method, args=()):
        return self._rpc(method, args)


@pytest.fixture
def eager_plugin(monkeypatch):
    for k in ("MIA_SINK", "MIA_PROFILE_MODE", "MIA_APERTURE_PER_REQUEST", "MIA_USE_SAFETENSORS",
              "MIA_STORAGE_ROUTER", "MIA_QK_AUTO_SELECT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(_plugin, "_graph_mode", lambda: False)


def _two_ranks_landing_apart(req_id="req-0"):
    q, k = _global_qk()
    writers = {r: _DelayedWriter(LAND_DELAY_S[r]) for r in range(TP)}
    ranks = [_qk_rank_worker(r, req_id, q, k, writer=writers[r]) for r in range(TP)]
    return q, k, ranks, writers


def _join(writers):
    for w in writers.values():
        for th in w.threads:
            th.join(timeout=5)


# --------------------------------------------------------------------------------------------
# The reproduction, through both callers of the barrier
# --------------------------------------------------------------------------------------------

def test_offline_save_to_disk_returns_only_after_every_rank_landed(tmp_path, eager_plugin,
                                                                   monkeypatch):
    """LLM.generate(save_to_disk=True) at TP=2 with the writers landing 290 ms apart: generate()
    must not return until BOTH ranks' shards are on disk, and the loader must then read the full
    width. (Old code: returned after rank 0, and the loader handed back half the heads.)"""
    from mia.run_utils import load_and_merge_qk_cache

    q, k, ranks, writers = _two_ranks_landing_apart()
    out = SimpleNamespace(request_id="req-0")
    monkeypatch.setattr(_plugin, "_original_llm_generate", lambda self, p, sp, **kw: [out])
    sp = SimpleNamespace(extra_args={"output_qk": {0: [0]}, "hookq_mode": "all_tokens",
                                     "save_to_disk": True, "run_id": "run",
                                     "hook_dir": str(tmp_path)})
    llm = _FakeLLM(ranks)

    _plugin._patched_llm_generate(llm, ["p"], sp)

    for r in range(TP):
        assert (tmp_path / "run" / rank_dir_name(r) / "qk.pt").is_file(), \
            f"generate() returned before tp_rank {r}'s shard landed"
    merged = load_and_merge_qk_cache(str(tmp_path), "run")
    e = merged["qk_cache"][LAYER]
    assert e["q"][0].shape[-1] == H_Q * D and e["k_all"][0].shape[-1] == H_KV * D
    assert torch.equal(e["q"][0], q) and torch.equal(e["k_all"][0], k)
    _join(writers)


def test_serve_durable_wait_returns_only_after_every_rank_landed(tmp_path, eager_plugin,
                                                                 monkeypatch):
    """The serve path's opt-in durable_wait is the same barrier (async): the finished output
    must not be yielded until every rank's shard is on disk."""
    q, k, ranks, writers = _two_ranks_landing_apart()
    finished = SimpleNamespace(request_id="req-0", finished=True, prompt_token_ids=[1, 2],
                               outputs=[SimpleNamespace(token_ids=[3])])

    async def orig(self, prompt, sampling_params, request_id, **kw):
        yield finished

    monkeypatch.setattr(_plugin, "_original_generate", orig)
    sp = SimpleNamespace(extra_args={"output_qk": {0: [0]}, "hookq_mode": "all_tokens",
                                     "save_to_disk": True, "durable_wait": True,
                                     "run_id": "run", "hook_dir": str(tmp_path)},
                         max_tokens=1)
    llm = _FakeAsyncLLM(ranks)
    landed_at_yield: dict = {}

    async def drive():
        async for _o in _plugin._patched_generate(llm, "p", sp, "req-0"):
            landed_at_yield.update({r: (tmp_path / "run" / rank_dir_name(r) / "qk.pt").is_file()
                                    for r in range(TP)})

    asyncio.run(drive())
    assert "flush_disk" in llm.calls
    assert landed_at_yield == {0: True, 1: True}, landed_at_yield
    _join(writers)


def test_offline_barrier_at_tp1_returns_as_soon_as_the_one_rank_lands(tmp_path, eager_plugin,
                                                                      monkeypatch):
    """Control: TP=1 (one rank, tp_rank_0) behaves as before -- it waits for the shard and returns
    promptly once it lands, and the loader reads it unchanged (no tp_shard key)."""
    from mia.run_utils import load_and_merge_qk_cache

    q, k = _global_qk()
    w = _DelayedWriter(0.05)
    rank = _qk_rank_worker(0, "req-0", q, k, writer=w, tp=1)
    rank._disk_states = {"req-0": {LAYER: {"q": [q], "k_all": [k], "layer_num": 0,
                                           "hookq_mode": "all_tokens"}}}
    out = SimpleNamespace(request_id="req-0")
    monkeypatch.setattr(_plugin, "_original_llm_generate", lambda self, p, sp, **kw: [out])
    sp = SimpleNamespace(extra_args={"output_qk": {0: [0]}, "save_to_disk": True,
                                     "run_id": "run", "hook_dir": str(tmp_path)})
    t0 = time.monotonic()
    _plugin._patched_llm_generate(_FakeLLM([rank]), ["p"], sp)
    assert time.monotonic() - t0 < 5.0
    assert (tmp_path / "run" / "tp_rank_0" / "qk.pt").is_file()
    merged = load_and_merge_qk_cache(str(tmp_path), "run")
    assert TP_SHARD_KEY not in merged and torch.equal(merged["qk_cache"][LAYER]["q"][0], q)
    for th in w.threads:
        th.join(timeout=5)


# --------------------------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------------------------

def test_flush_disk_returns_the_rank_dir_it_wrote(tmp_path):
    """QK: each rank returns its own <hook_dir>/<run_id>/tp_rank_<r> (inline save here), so the
    driver knows which dirs to wait for. False when the rank captured nothing."""
    from mia.workers.qk_capture_worker import QKCaptureWorker

    q, k = _global_qk()
    for r in range(TP):
        w = _qk_rank_worker(r, "req-0", q, k, writer=None)
        got = QKCaptureWorker.flush_disk(w, ["req-0"], "run", str(tmp_path))
        assert got == os.path.join(str(tmp_path), "run", rank_dir_name(r))
        assert (tmp_path / "run" / rank_dir_name(r) / "qk.pt").is_file()
    empty = _qk_rank_worker(0, "other", q, k, writer=None)
    assert QKCaptureWorker.flush_disk(empty, ["req-0"], "run2", str(tmp_path)) is False


def test_hs_flush_disk_returns_rank0_dir_and_false_on_a_sink_rank(tmp_path):
    """HS captures on tp_rank 0 only (the residual is replicated): rank 0 returns its dir, a
    non-capturing rank (empty buckets) returns False -- so the barrier waits for rank 0 alone."""
    from mia.workers.hs_capture_worker import HSCaptureWorker

    hs = torch.randn(3, 16)
    base = dict(_conf={"hidden_size": 16}, hs_mode="all_tokens", _capture_consumer=None,
                _egress_stream=None, _writer_process=None)
    r0 = SimpleNamespace(_disk_states={"req-0": {"model.layers.0": {
        "hidden_states": [hs], "layer_num": 1, "hs_mode": "all_tokens"}}}, _tp_rank=0, **base)
    r1 = SimpleNamespace(_disk_states={}, _tp_rank=1, **base)
    got = [HSCaptureWorker.flush_disk(w, ["req-0"], "run", str(tmp_path)) for w in (r0, r1)]
    assert got == [os.path.join(str(tmp_path), "run", "tp_rank_0"), False]
    assert _plugin._flushed_rank_dirs(got, "run", str(tmp_path)) == [
        os.path.join(str(tmp_path), "run", "tp_rank_0")]


def test_flushed_rank_dirs_reads_every_result_shape(tmp_path):
    run = os.path.join(str(tmp_path), "run")
    # dirs as returned by the workers (re-rooted under the DRIVER's hook_dir by rank name)
    got = _plugin._flushed_rank_dirs(["/elsewhere/run/tp_rank_1", False, "x/run/tp_rank_0"],
                                     "run", str(tmp_path))
    assert got == [os.path.join(run, "tp_rank_0"), os.path.join(run, "tp_rank_1")]
    # a bare True (the Token Highlighter: it writes tp_rank_0 itself, whatever its rank) names no
    # dir -> None, i.e. the old whole-run-dir wait, never a wait for tp_rank_<position>
    assert _plugin._flushed_rank_dirs([True, False, True], "run", str(tmp_path)) is None
    assert _plugin._flushed_rank_dirs([True], "run", str(tmp_path)) is None
    # nothing named -> None (the caller keeps the legacy whole-run-dir wait)
    assert _plugin._flushed_rank_dirs([False, None], "run", str(tmp_path)) is None
    assert _plugin._flushed_rank_dirs(None, "run", str(tmp_path)) is None


def _land(path, delay, payload=b"x" * 64):
    def go():
        time.sleep(delay)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            f.write(payload)
        os.rename(path + ".tmp", path)
    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th


def test_wait_disk_artifact_waits_for_every_named_rank(tmp_path):
    run = tmp_path / "run"
    dirs = [str(run / rank_dir_name(r)) for r in range(2)]
    ths = [_land(os.path.join(dirs[0], "qk.pt"), 0.01), _land(os.path.join(dirs[1], "qk.pt"), 0.25)]
    assert _plugin._wait_disk_artifact("run", str(tmp_path), dirs) is True
    assert all(os.path.isfile(os.path.join(d, "qk.pt")) for d in dirs)
    for th in ths:
        th.join()


def test_await_disk_artifact_waits_for_every_named_rank(tmp_path):
    run = tmp_path / "run"
    dirs = [str(run / rank_dir_name(r)) for r in range(2)]
    ths = [_land(os.path.join(dirs[0], "qk.pt"), 0.01), _land(os.path.join(dirs[1], "qk.pt"), 0.25)]
    assert asyncio.run(_plugin._await_disk_artifact("run", str(tmp_path), dirs)) is True
    assert all(os.path.isfile(os.path.join(d, "qk.pt")) for d in dirs)
    for th in ths:
        th.join()


def test_wait_disk_artifact_times_out_loud_naming_the_missing_rank(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_plugin, "_ARTIFACT_WAIT_S", 0.1)
    run = tmp_path / "run"
    dirs = [str(run / rank_dir_name(r)) for r in range(2)]
    _land(os.path.join(dirs[0], "qk.pt"), 0.0).join()
    assert _plugin._wait_disk_artifact("run", str(tmp_path), dirs) is False
    assert "tp_rank_1" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# The loaders refuse a partial TP > 1 shard set
# --------------------------------------------------------------------------------------------

def _write_pt_shard(tmp_path, r, tp=TP):
    q, k = _global_qk()
    s = qk_shard(r, tp, H_Q, H_KV, D)
    d = tmp_path / "run" / rank_dir_name(r)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"num_attention_heads": H_Q}, TP_SHARD_KEY: s.as_header(),
                "qk_cache": {LAYER: {"q": [_rank_slice(q, r, H_Q)],
                                     "k_all": [_rank_slice(k, r, H_KV)],
                                     "layer_num": 0, "hookq_mode": "all_tokens"}}},
               d / "qk.pt")


@pytest.mark.parametrize("present", [0, 1])
def test_pt_loader_refuses_a_lone_tp_shard(tmp_path, monkeypatch, present):
    from mia.run_utils import load_and_merge_qk_cache

    monkeypatch.delenv("MIA_USE_SAFETENSORS", raising=False)
    _write_pt_shard(tmp_path, present)
    with pytest.raises(TPShardError, match=rf"missing tp_rank\(s\) \[{1 - present}\]"):
        load_and_merge_qk_cache(str(tmp_path), "run")


def test_safetensors_loader_refuses_a_lone_tp_shard(tmp_path, monkeypatch):
    from mia.graph.artifact_writer import save_qk_cache_safetensors
    from mia.run_utils import load_and_merge_qk_cache

    monkeypatch.setenv("MIA_USE_SAFETENSORS", "1")
    q, k = _global_qk()
    s = qk_shard(0, TP, H_Q, H_KV, D)
    cache = {"config": {"num_attention_heads": H_Q}, TP_SHARD_KEY: s.as_header(),
             "qk_cache": {LAYER: {"q": [_rank_slice(q, 0, H_Q)], "k_all": [_rank_slice(k, 0, H_KV)],
                                  "layer_num": 0, "hookq_mode": "all_tokens"}}}
    d = tmp_path / "run" / rank_dir_name(0)
    d.mkdir(parents=True)
    save_qk_cache_safetensors(cache, str(d), "all_tokens", 0)
    assert json.loads((d / "qk.json").read_text())[TP_SHARD_KEY]["tp_size"] == TP
    with pytest.raises(TPShardError, match=r"missing tp_rank\(s\) \[1\]"):
        load_and_merge_qk_cache(str(tmp_path), "run")


def test_loader_still_merges_a_complete_set_and_reads_a_tp1_artifact(tmp_path, monkeypatch):
    """Controls: the full TP set merges to full width; a TP=1 artifact (no tp_shard) loads."""
    from mia.run_utils import load_and_merge_qk_cache

    monkeypatch.delenv("MIA_USE_SAFETENSORS", raising=False)
    for r in range(TP):
        _write_pt_shard(tmp_path, r)
    merged = load_and_merge_qk_cache(str(tmp_path), "run")
    assert merged["qk_cache"][LAYER]["q"][0].shape[-1] == H_Q * D
    q, k = _global_qk()
    d = tmp_path / "tp1" / rank_dir_name(0)
    d.mkdir(parents=True)
    torch.save({"config": {}, "qk_cache": {LAYER: {"q": [q], "k_all": [k], "layer_num": 0}}},
               d / "qk.pt")
    assert torch.equal(load_and_merge_qk_cache(str(tmp_path), "tp1")["qk_cache"][LAYER]["q"][0], q)


def test_artifact_glob_with_timeout_waits_for_the_expected_ranks(tmp_path):
    from mia.run_utils import _artifact_glob

    run = tmp_path / "run"
    ths = [_land(str(run / rank_dir_name(0) / "qk.pt"), 0.0),
           _land(str(run / rank_dir_name(1) / "qk.pt"), 0.25)]
    ths[0].join()
    got = _artifact_glob(str(tmp_path), "run", "qk.pt", timeout=5.0, expected_ranks=2)
    assert sorted(os.path.basename(os.path.dirname(p)) for p in got) == ["tp_rank_0", "tp_rank_1"]
    for th in ths:
        th.join()


def test_a_worker_that_answers_true_keeps_the_whole_run_dir_wait(tmp_path, eager_plugin,
                                                                  monkeypatch):
    """The Token Highlighter's flush_disk answers True and has already written
    <run>/tp_rank_0/highlighter_activations.pt on EVERY rank's call. At TP=2 the barrier must not
    wait for a tp_rank_1 that will never exist (a 10 s stall per generate)."""
    class _Highlighter(_FakeLLM):
        def _rpc(self, method, args=()):
            self.calls.append(method)
            if method == "flush_disk":
                _req_ids, run_id, hook_dir = args
                d = os.path.join(hook_dir, run_id, "tp_rank_0")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "highlighter_activations.pt"), "wb") as f:
                    f.write(b"x" * 32)
                return [True, True]
            return [None, None]

    out = SimpleNamespace(request_id="req-0")
    monkeypatch.setattr(_plugin, "_original_llm_generate", lambda self, p, sp, **kw: [out])
    sp = SimpleNamespace(extra_args={"output_qk": {0: [0]}, "save_to_disk": True,
                                     "run_id": "run", "hook_dir": str(tmp_path)})
    t0 = time.monotonic()
    _plugin._patched_llm_generate(_Highlighter([None, None]), ["p"], sp)
    assert time.monotonic() - t0 < 2.0, "waited for a tp_rank_1 the highlighter never writes"
