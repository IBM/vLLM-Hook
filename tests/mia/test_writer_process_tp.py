"""The out-of-process writer runs on EVERY TP rank, the same way, and never outlives its worker.

The defect (gate G1, LSF 1777562): vLLM runs each TP worker as a DAEMONIC multiprocessing child, and
multiprocessing refuses to start a child from one. Every capturing rank at TP>1 logged
"[writer-process] failed to start, falling back to in-process save: AssertionError('daemonic
processes are not allowed to have children')", while TP=1 (writer started by the non-daemonic
EngineCore) logged "async serialize+write process ON". The TP=1 and TP>1 save paths were not like
for like.

These tests drive ``tests/_writer_lifecycle_probe.py``, which runs the writer inside a REAL spawn
child of a main process: daemonic (a TP worker) and non-daemonic (EngineCore at TP=1). They pin:

* the writer starts from a daemonic parent, by the same mechanism, and its artifact is byte-identical
  to the one a non-daemonic parent's writer writes;
* a worker that exits without calling close() still gets its pending writes drained, because the
  close runs before multiprocessing's exit path terminates daemonic children;
* no orphans: when the parent is SIGKILLed (vLLM's last resort after its shutdown timeout), the writer
  child exits by itself. At TP=1 the old child blocked on ``q.get()`` forever.

GPU-free, but real processes: each mode spawns a parent and a writer child, which both import
``mia`` (and so vLLM). The six modes run concurrently.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("vllm")  # the writer child imports `mia`, which pulls in vLLM

import torch

PROBE = Path(__file__).resolve().parent / "_writer_lifecycle_probe.py"
MODES = ("write-daemonic", "write-plain", "exit-daemonic", "exit-plain",
         "kill-daemonic", "kill-plain")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("writer_tp")
    env = dict(os.environ, MIA_WRITER_PROCESS="1", MIA_CHILD_PARENT_POLL_S="0.2",
               PYTHONPATH=os.pathsep.join([str(PROBE.parents[2]), os.environ.get("PYTHONPATH", "")]))
    procs = {m: subprocess.Popen([sys.executable, str(PROBE), m, str(root / m)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for m in MODES}
    out = {}
    for m, p in procs.items():
        try:
            log, _ = p.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
            log, _ = p.communicate()
        res = root / m / "result.json"
        out[m] = (json.loads(res.read_text()) if res.exists() else {"missing": True}, log)
    return out


def _ok(runs, mode):
    res, log = runs[mode]
    assert not res.get("missing"), f"{mode}: the probe wrote no result\n{log[-3000:]}"
    return res, log


@pytest.mark.parametrize("flavor", ["daemonic", "plain"])
def test_the_writer_process_runs_whether_or_not_the_worker_is_daemonic(runs, flavor):
    res, log = _ok(runs, f"write-{flavor}")
    assert res["writer_mode"] == "process", (res, log[-3000:])
    assert res["warmup_s"] is not None, f"the child never served its warm-up\n{log[-3000:]}"
    assert res["parent_daemonic"] is (flavor == "daemonic")
    assert res["submitted"] is True
    assert res["artifact"], "the writer child wrote nothing"
    assert res["children_alive_after"] == []
    # one log line per rank saying which mode it runs, keeping the TP=1 text
    assert "async serialize+write process ON: tp_rank 0/1" in log
    assert ("daemonic TP worker" if flavor == "daemonic" else "non-daemonic process") in log
    assert "failed to start" not in log


def test_the_daemonic_writer_writes_the_same_bytes_as_the_tp1_writer(runs):
    """Same mechanism at every TP: the TP>1 (daemonic) writer's artifact equals the TP=1 one's."""
    a = Path(_ok(runs, "write-daemonic")[0]["artifact"]).read_bytes()
    b = Path(_ok(runs, "write-plain")[0]["artifact"]).read_bytes()
    assert a == b
    got = torch.load(_ok(runs, "write-daemonic")[0]["artifact"], weights_only=False)
    t = got["hs_cache"]["model.layers.0"]["hidden_states"][0]
    assert torch.equal(t, torch.arange(64, dtype=torch.float32).reshape(8, 8))


@pytest.mark.parametrize("flavor", ["daemonic", "plain"])
def test_a_worker_that_exits_without_close_still_drains_its_writer(runs, flavor):
    """multiprocessing's exit path terminates daemonic children BEFORE atexit runs, so the close
    must be a multiprocessing finalizer, not only an atexit handler."""
    res, log = _ok(runs, f"exit-{flavor}")
    assert res["writer_mode"] == "process", (res, log[-3000:])
    assert res["warmup_s"] is not None, f"the child never served its warm-up\n{log[-3000:]}"
    assert res["parent_exitcode"] == 0
    assert res["artifact"], f"pending write lost at worker exit\n{log[-3000:]}"
    assert res["children_alive_after"] == []


@pytest.mark.parametrize("flavor", ["daemonic", "plain"])
def test_a_killed_worker_leaves_no_orphan_writer(runs, flavor):
    """vLLM SIGKILLs a worker that outlives its shutdown timeout. The writer child must notice and
    exit by itself (it holds both ends of its queue's pipe, so it never sees EOF)."""
    res, log = _ok(runs, f"kill-{flavor}")
    assert res["writer_mode"] == "process", (res, log[-3000:])
    assert res["parent_exitcode"] == -9
    # The probe waits 30 s; the old child never exits (measured ~3.4 s here, mostly its own
    # interpreter teardown after the 0.2 s parent check).
    assert res["children_alive_after"] == [], (
        f"writer child {res['children_alive_after']} outlived its SIGKILLed parent by 30 s")
    assert res["seconds_to_child_exit"] is not None


# --------------------------------------------------------------------------------------------
# Review follow-ups: a malformed poll env, a writer that dies after "ON", HS sink ranks
# --------------------------------------------------------------------------------------------

def _load_child_process_module(monkeypatch, value):
    """Execute mia/graph/child_process.py afresh (stdlib-only imports) under ``value``."""
    import importlib.util

    if value is None:
        monkeypatch.delenv("MIA_CHILD_PARENT_POLL_S", raising=False)
    else:
        monkeypatch.setenv("MIA_CHILD_PARENT_POLL_S", value)
    spec = importlib.util.spec_from_file_location(
        "_cp_under_test", PROBE.parents[2] / "mia" / "graph" / "child_process.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("value,want", [(None, 1.0), ("", 1.0), ("0.2", 0.2), ("abc", 1.0),
                                        ("0", 1.0), ("-3", 1.0), ("nan", 1.0), ("inf", 1.0)])
def test_a_malformed_parent_poll_env_falls_back_instead_of_failing_install(monkeypatch, capsys,
                                                                           value, want):
    """writer_process imports child_process at its top, OUTSIDE init_writer_process's try, so a
    bare float() of MIA_CHILD_PARENT_POLL_S crashed engine install on a typo."""
    assert _load_child_process_module(monkeypatch, value).PARENT_POLL_S == want
    if value not in (None, "", "0.2"):
        assert "MIA_CHILD_PARENT_POLL_S" in capsys.readouterr().out


class _GoneWriter:
    """A writer whose child died after its "ON" line: alive() False, submit() refuses."""
    _procs = [type("P", (), {"pid": 4242, "exitcode": -9})()]
    _feeder_error = None

    def alive(self):
        return False

    def submit(self, *a, **k):
        return False


def test_a_refused_submit_saves_inline_loudly_once_and_records_the_mode(tmp_path, capsys):
    from types import SimpleNamespace

    from mia.graph.tp_shard import qk_shard
    from mia.workers.qk_capture_worker import QKCaptureWorker

    def rank(req):
        return SimpleNamespace(
            _disk_states={req: {"model.layers.0.self_attn.attn": {
                "q": [torch.ones(3, 16)], "k_all": [torch.ones(3, 8)], "layer_num": 0,
                "hookq_mode": "all_tokens"}}},
            _conf={}, hookq_mode="all_tokens", _tp_rank=1, _qk_shard=qk_shard(1, 2, 4, 2, 8),
            _capture_consumer=None, _egress_stream=None, _writer_process=_GoneWriter(),
            _writer_mode="process")

    w = rank("a")
    assert QKCaptureWorker.flush_disk(w, ["a"], "r1", str(tmp_path))
    w._disk_states = rank("b")._disk_states
    assert QKCaptureWorker.flush_disk(w, ["b"], "r2", str(tmp_path))
    for run in ("r1", "r2"):                              # the data is still saved (inline)
        assert (tmp_path / run / "tp_rank_1" / "qk.pt").is_file()
    out = capsys.readouterr().out
    assert out.count("[writer-process] submit refused (writer gone)") == 1, out
    assert w._writer_mode.startswith("in-process (writer gone after start"), w._writer_mode
    assert "4242" in w._writer_mode
