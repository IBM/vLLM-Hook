"""Start MIA helper child processes from any process, daemonic TP workers included."""
from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.process as mpp
import os
import queue as _queue
import threading
from multiprocessing import util

SHUTDOWN_EXIT_PRIORITY = 100

_DEFAULT_PARENT_POLL_S = 1.0


def _parent_poll_s_from_env() -> float:
    raw = os.environ.get("MIA_CHILD_PARENT_POLL_S", "")
    if not raw.strip():
        return _DEFAULT_PARENT_POLL_S
    try:
        val = float(raw)
    except ValueError:
        val = None
    if val is None or not val > 0 or val != val or val == float("inf"):
        print(f"[mia/child-process] MIA_CHILD_PARENT_POLL_S={raw!r} is not a positive number of "
              f"seconds; using {_DEFAULT_PARENT_POLL_S} s", flush=True)
        return _DEFAULT_PARENT_POLL_S
    return val


PARENT_POLL_S = _parent_poll_s_from_env()

_START_LOCK = threading.Lock()


def start_child(proc) -> bool:
    """``proc.start()``, also from a DAEMONIC process."""
    cur = mpp.current_process()
    with _START_LOCK:
        daemonic = bool(cur._config.get("daemon"))
        if daemonic:
            cur._config["daemon"] = False
        try:
            proc.start()
        finally:
            if daemonic:
                cur._config["daemon"] = True
    return daemonic


def register_shutdown(close) -> None:
    """Run ``close`` at interpreter or process exit, in shutdown order."""
    util.Finalize(None, close, exitpriority=SHUTDOWN_EXIT_PRIORITY)
    atexit.register(close)


def get_until_parent_exits(q, poll_s: float | None = None):
    """``q.get()`` for a helper child: next item, or None once the queue is empty and the parent exited."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return q.get()
    wait_s = PARENT_POLL_S if poll_s is None else float(poll_s)
    while True:
        try:
            return q.get(timeout=wait_s)
        except _queue.Empty:
            if not parent.is_alive():
                return None


__all__ = ["PARENT_POLL_S", "SHUTDOWN_EXIT_PRIORITY", "get_until_parent_exits",
           "register_shutdown", "start_child"]

