"""Start MIA's spawned helper children from ANY process, a daemonic vLLM TP worker included, and
never leave one behind.

WHY THIS EXISTS. vLLM runs every tensor-parallel worker as a DAEMONIC multiprocessing child
(``multiproc_executor.WorkerProc.make_worker_process``: ``daemon=True``). ``multiprocessing`` refuses
to start a child from a daemonic process: ``AssertionError('daemonic processes are not allowed to
have children')``. So at TP>1 every capturing rank logged "[writer-process] failed to start, falling
back to in-process save" (gate G1), while at TP=1 the writer is started by the non-daemonic
EngineCore and runs. The two save paths were not like for like.

The refusal protects two things. This module provides both itself, and then starts the child the
same way at every TP (same ``spawn`` context, same ``torch.multiprocessing`` queue IPC, same child
entry point):

1. NO ORPHANS. A daemonic process is terminated with its parent, and multiprocessing fears its
   children would outlive it. Every MIA helper child reads its work queue through
   :func:`get_until_parent_exits`. That watches ``multiprocessing.parent_process().sentinel``, a pipe
   that the parent's death closes, however it dies (SIGKILL included). When the parent is gone,
   the child finishes the items already queued and exits. This also removes a latent TP=1 orphan:
   a SIGKILLed EngineCore used to leave its writer child blocked on ``q.get()`` forever, because the
   child holds both ends of the queue's pipe and never sees EOF.

2. SHUTDOWN ORDER. In a multiprocessing child (every vLLM TP worker, and EngineCore at TP=1),
   ``Process._bootstrap`` calls ``multiprocessing.util._exit_function()`` when the target returns.
   That function TERMINATES the process's daemonic children, and it runs BEFORE Python's ``atexit``
   handlers. So a ``close()`` registered only with ``atexit`` ran after its child was already dead,
   and any write still in flight was lost. :func:`register_shutdown` also registers the close as a
   ``multiprocessing.util.Finalize`` at ``exitpriority`` 100. ``_exit_function`` runs it FIRST:
   before the queues' own finalizers (priority 10) and before it terminates daemonic children. The
   owner's close is bounded by timeouts, so it cannot hold up vLLM's worker shutdown
   (``VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`` then SIGTERM, then SIGKILL). If vLLM kills the worker
   anyway, rule 1 still ends the child.

:func:`start_child` lifts the daemonic-parent assertion for exactly one ``Process.start()`` and
restores it, under a lock. It returns whether the parent was daemonic, so the owner can log which
case each rank ran.
"""
from __future__ import annotations

import os
import queue as _queue
import threading

#: ``multiprocessing.util.Finalize`` priority for an owner's close: above the mp queue finalizers
#: (``Queue._finalize_close`` = 10), so the owner can still push its stop sentinels through its
#: queues, and ``_exit_function`` runs every priority >= 0 before it terminates daemonic children.
SHUTDOWN_EXIT_PRIORITY = 100

_DEFAULT_PARENT_POLL_S = 1.0


def _parent_poll_s_from_env() -> float:
    """``MIA_CHILD_PARENT_POLL_S`` as a positive float, else the 1.0 s default (logged). Parsed at
    import, and ``writer_process`` imports this module at ITS top, outside ``init_writer_process``'s
    try: a bare ``float()`` turned a malformed value into an engine-install crash instead of the
    default. Zero or a negative value is refused too (an idle child would spin on ``get``)."""
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


#: How often an idle child checks that its parent is still alive, in seconds. The check only runs
#: when the queue is EMPTY, so it adds no latency to an item. Env-overridable for tests.
PARENT_POLL_S = _parent_poll_s_from_env()

_START_LOCK = threading.Lock()


def start_child(proc) -> bool:
    """``proc.start()``, also from a DAEMONIC process. Returns True when the calling process is
    daemonic (a vLLM TP worker), False otherwise (EngineCore at TP=1, a script).

    multiprocessing checks ``current_process()._config['daemon']`` in ``start()`` and nowhere else.
    The flag is cleared for this one call and restored in ``finally``, under a module lock, so
    concurrent MIA starts cannot interleave. The child's own ``daemon=True`` is unaffected: it is
    still terminated by its parent's ``_exit_function`` (after the owner's registered close). The
    two protections the assertion stood for are this module's rules 1 and 2."""
    import multiprocessing.process as mpp

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
    """Run ``close`` at interpreter / process exit in the order rule 2 needs.

    It is registered twice: as a multiprocessing Finalize (``exitpriority`` 100, which runs inside
    ``_exit_function`` before daemonic children are terminated, in multiprocessing children AND in
    a main process, where ``_exit_function`` is itself an atexit handler) and with ``atexit`` (a
    backstop for an interpreter whose multiprocessing exit hook already ran). ``close`` must
    therefore be idempotent and bounded."""
    import atexit
    from multiprocessing import util

    util.Finalize(None, close, exitpriority=SHUTDOWN_EXIT_PRIORITY)
    atexit.register(close)


def get_until_parent_exits(q, poll_s: float | None = None):
    """``q.get()`` for a helper CHILD: returns the next item, or ``None`` (every MIA child's stop
    sentinel) once the queue is empty AND the parent process has exited.

    Items already queued when the parent dies are still returned first, so a backlog handed over
    before a SIGKILL is written rather than dropped. ``multiprocessing.parent_process()`` is None
    only in a main process (a direct call, e.g. a unit test); then this is a plain blocking get."""
    import multiprocessing

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
