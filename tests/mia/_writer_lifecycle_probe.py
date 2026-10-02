"""Drive MIA's writer process from a REAL multiprocessing parent, the way vLLM does, and report.

Run as a script (``python tests/_writer_lifecycle_probe.py <mode> <out_dir>``); pytest drives it
from ``tests/test_writer_process_tp.py``. It is a script, not a test module, because the parent
must be a genuine ``multiprocessing`` child of a main process (vLLM's ``WorkerProc`` at TP>1 is a
DAEMONIC one; EngineCore at TP=1 is a non-daemonic one), and a ``__main__``-defined target is what
``spawn`` pickles cleanly.

Modes (the parent is always a spawn child of this script):

  write-daemonic   daemonic parent (a TP worker): init_writer_process on a fake worker, wait for the
                   child to serve a warm-up artifact, submit one artifact, close(), return.
  write-plain      the same from a NON-daemonic parent (EngineCore at TP=1).
  exit-daemonic    daemonic parent (after the same warm-up) submits and RETURNS without close():
                   multiprocessing's exit path must still drain the writer before it terminates
                   the child.
  exit-plain       the same from a non-daemonic parent.
  kill-daemonic    daemonic parent starts the writer, waits until it has written one artifact (it
                   is serving), reports, blocks; this script SIGKILLs it and waits for the writer
                   child to go (no orphan).
  kill-plain       the same with a non-daemonic parent (the old TP=1 orphan).

Writes ``<out_dir>/result.json``::

  {"mode", "writer_mode", "parent_daemonic", "parent_pid", "child_pids", "submitted",
   "parent_exitcode", "children_alive_after": [pid, ...], "artifact": path | null,
   "seconds_to_child_exit": float | null, "warmup_s": float | null (write/exit modes)}

Children that outlive the check are SIGKILLed before exit, so a failing run never leaks processes.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line.split(":", 1)[1]
    except OSError:
        return False
    return True


def _cache():
    import torch
    # 8x8 float32 = 256 B: a multiple of the packer's 8-byte alignment, so the packed buffer holds
    # no uninitialized pad bytes and two writers' .pt files can be compared byte for byte.
    return {"hs_cache": {"model.layers.0": {
                "hidden_states": [torch.arange(64, dtype=torch.float32).reshape(8, 8)],
                "layer_num": 1, "hs_mode": "all_tokens"}},
            "config": {"hidden_size": 8, "num_layers": 1}}


def _serve_one(wp, run_dir: str) -> float | None:
    """Submit one artifact and wait (<= 300 s) until the writer child has written it. Returns
    the seconds it took, None if it never landed."""
    t0 = time.monotonic()
    wp.submit("hs", _cache(), run_dir, "all_tokens", 0, False, False, "hidden_states.pt",
              req_ids=["warmup"], block=True)
    art = os.path.join(run_dir, "hidden_states.pt")
    while time.monotonic() - t0 < 300:
        if os.path.exists(art):
            return round(time.monotonic() - t0, 2)
        time.sleep(0.05)
    return None


def _parent(out_dir: str, close: bool, block: bool, report) -> None:
    import types

    sys.path.insert(0, REPO)
    from mia.graph.writer_process import init_writer_process

    w = types.SimpleNamespace()                    # a fake worker: tp_rank 0 of 1
    info = {"parent_pid": os.getpid(), "child_pids": [], "submitted": None}
    wp = None
    try:                                           # ALWAYS report: the script must never hang
        init_writer_process(w)
        wp = w._writer_process
        info["writer_mode"] = getattr(w, "_writer_mode",
                                      "process" if wp is not None else "in-process (unlabelled)")
        info["parent_daemonic"] = getattr(wp, "parent_daemonic", None)
        info["child_pids"] = [p.pid for p in wp._procs] if wp is not None else []
        if wp is not None:
            if not block:
                # Test the drain, not the child's import time: a spawned child imports mia (and
                # so vLLM), which took 12.5 s on a loaded login node -- longer than close()'s
                # bounded 10 s join, so a write submitted before the child serves was lost to
                # startup, not to the exit ordering these modes pin. Wait until the child has
                # written a warm-up artifact, then submit the measured one.
                info["warmup_s"] = _serve_one(wp, os.path.join(out_dir, "warmup", "tp_rank_0"))
            run_dir = os.path.join(out_dir, "run", "tp_rank_0")
            info["submitted"] = wp.submit("hs", _cache(), run_dir, "all_tokens", 0, False, False,
                                          "hidden_states.pt", req_ids=["r0"], block=True)
            if block:
                # Kill only a writer that is SERVING: wait for its first artifact, so the
                # measured time is parent-death -> child-exit, not the child's own startup.
                art = os.path.join(run_dir, "hidden_states.pt")
                deadline = time.monotonic() + 300
                while not os.path.exists(art) and time.monotonic() < deadline:
                    time.sleep(0.05)
    except BaseException as e:  # noqa: BLE001
        info["error"] = repr(e)
    report.put(info)
    if block:
        time.sleep(600)                            # until this script SIGKILLs us
    if close and wp is not None:
        wp.close()
    # else: return with the writer still open -- multiprocessing's exit path must drain it.


def main(mode: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("MIA_CHILD_PARENT_POLL_S", "0.2")
    kind, flavor = mode.split("-", 1)
    ctx = mp.get_context("spawn")
    report = ctx.Queue()
    p = ctx.Process(target=_parent, name=f"fake-{flavor}-worker",
                    args=(out_dir, kind == "write", kind == "kill", report),
                    daemon=(flavor == "daemonic"))
    p.start()
    try:
        info = report.get(timeout=300)             # the parent reports once its writer is up
    except Exception as e:  # noqa: BLE001 -- a parent that died before reporting
        info = {"error": f"no report from the parent: {e!r}", "child_pids": []}
    t_kill = None
    if kind == "kill":
        os.kill(p.pid, signal.SIGKILL)
        t_kill = time.monotonic()
    p.join(timeout=120)
    info["parent_exitcode"] = p.exitcode
    deadline = time.monotonic() + 30
    first_gone = None
    while time.monotonic() < deadline:
        alive = [c for c in info["child_pids"] if _alive(c)]
        if not alive:
            first_gone = time.monotonic()
            break
        time.sleep(0.05)
    info["children_alive_after"] = [c for c in info["child_pids"] if _alive(c)]
    info["seconds_to_child_exit"] = (round(first_gone - t_kill, 2)
                                     if (t_kill is not None and first_gone is not None) else None)
    art = os.path.join(out_dir, "run", "tp_rank_0", "hidden_states.pt")
    info["artifact"] = art if os.path.exists(art) else None
    info["mode"] = mode
    for c in info["children_alive_after"]:        # never leak an orphan from a failing run
        try:
            os.kill(c, signal.SIGKILL)
        except OSError:
            pass
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(info, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
