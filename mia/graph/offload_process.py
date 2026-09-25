"""A separate CPU worker that transfers a finished request's per-request file to the client
destination -- the delivery half of the per-request pipeline (capture writes the file, this
ships it out).

Off-loop-worker shape: an UNBOUNDED work queue (holds only tiny ``(req_id, src_path, dest)``
path tuples, since the artifact bytes already live on NVMe, so holding an arbitrarily long
backlog is memory-safe and IS the "hold, never drop" contract), a NON-BLOCKING ``submit`` (the
engine loop never stalls or drops, even with a stalled worker and a deep backlog), a worker
loop, and never-drop hold-and-retry semantics for a transfer that raises.

``wait(req_id, timeout)`` is the client-blocks contract: a caller that needs to know a specific
request's file has actually landed at ``dest`` blocks on a per-req ``threading.Event`` rather than
polling. It returns True only for an actual completed transfer -- a job that gave up after
exhausting its retries still wakes a blocked ``wait()`` promptly (rather than making the caller
sit out the whole timeout) but reports False, same as a real timeout. ``poll_done()`` /
``poll_failed()`` are the complementary non-blocking "what settled since I last asked" views --
done vs. permanently-gave-up -- for a caller that wants to drive its own accounting loop instead
of blocking per request. A req_id observed via ``poll_done()``/``poll_failed()`` is safe to
``submit()`` again immediately: the settle (what makes it visible to poll) and a resubmit's
re-arm of the completion Event are serialized under one lock, so a resubmit can never observe a
stale signal left over from the job it is replacing (see ``OffloadProcess.__init__``).

ONE-SUBMIT-PER-REQ_ID: ``submit()`` raises ``ValueError`` if `req_id` is already in flight (queued
or being retried) -- a shared per-req Event means a second submit while the first is in flight
could let a stale completion satisfy the new caller's ``wait()``. In the real pipeline a req_id is
delivered exactly once, so resubmitting an in-flight id is a contract violation, not a supported
flow (resubmitting a req_id that has already settled is fine and re-arms its Event).

Default ``transfer_fn`` is a shared-filesystem copy (``shutil.copy`` for a FILE,
``shutil.copytree`` for a per-request run DIRECTORY -- the disk route ships a whole per-request
run_dir of per-layer raw files + a sidecar); injectable for tests and for a future network sender
(any ``callable(src_path, dest) -> None`` that raises on failure).

BACKEND: THREAD by default (``use_process=False``). The transfer is a file/dir COPY (or a
network send) -- pure I/O that releases the GIL during the copy syscalls, so a worker THREAD never
stalls the engine's GPU forward. A real child PROCESS (``use_process=True``,
``torch.multiprocessing`` spawn, mirroring ``writer_process`` / ``server_analyze_process``) is
AVAILABLE for a future network sender that could hang uninterruptibly (a thread cannot be
force-killed; a stuck child can be ``terminate()``d) -- but for a pure copy it buys no real
isolation, so it is opt-in, not the default. An injected ``transfer_fn`` forces the thread backend
(a test closure / non-default sender is not guaranteed picklable across an mp spawn boundary; the
mp child always runs the module-level ``_default_transfer``).
"""
from __future__ import annotations

import os
import queue as _queue
import shutil
import threading
import time


def _aperture_debug() -> bool:
    """Gated disk-pipeline debug logging (MIA_APERTURE_DEBUG=1). Off by default -> no perf impact.
    Read at call time so the mp child (which snapshots the parent env at spawn) also honors it."""
    return os.environ.get("MIA_APERTURE_DEBUG") == "1"


def _default_transfer(src_path: str, dest: str) -> None:
    """Shared-FS transfer. A per-request run DIRECTORY (the disk route's artifact: per-layer raw
    files + ``hs_aperture_meta.jsonl``) is copied whole via ``shutil.copytree`` (``dirs_exist_ok`` so a
    retry over a partial dest never raises); a single FILE via ``shutil.copy`` (bytes + perm bits;
    not an atomic rename since dest may be a different mount). ``dest`` is a run_dir the read side
    reconstructs with ``aperture_reader.load_multilayer_aperture_artifact(dest)`` (dir case) -- so the
    delivered layout must match what the staging wrote."""
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dest, dirs_exist_ok=True)
        return
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy(src_path, dest)


def _run_transfer_with_retries(transfer_fn, req_id: str, src_path: str, dest: str,
                               max_retries: int, retry_delay: float) -> bool:
    """Run ONE job's transfer with bounded hold-and-retry (never-drop up to the bound). Returns
    True on success, False after ``max_retries`` consecutive failures (a doomed job is given up on,
    never reported as done). Shared verbatim by the thread worker (``_run``) and the mp child
    (``_offload_child``) so both backends have identical never-drop semantics."""
    attempts = 0
    while True:
        attempts += 1
        try:
            if _aperture_debug():
                print(f"[hookplugin/aperture-disk] offload START req_id={req_id!r} src={src_path!r} "
                      f"dest={dest!r} attempt={attempts}", flush=True)
            transfer_fn(src_path, dest)
            if _aperture_debug():
                print(f"[hookplugin/aperture-disk] offload DONE  req_id={req_id!r} dest={dest!r}",
                      flush=True)
            return True
        except Exception as e:  # noqa: BLE001 -- never crash the worker/child on one bad item
            if _aperture_debug():
                print(f"[hookplugin/aperture-disk] offload ERROR req_id={req_id!r} src={src_path!r} "
                      f"dest={dest!r} attempt={attempts}: {e!r}", flush=True)
            if attempts >= max_retries:
                print(f"[offload-process] transfer FAILED for {req_id!r} after "
                      f"{attempts} attempts, giving up: {e!r}", flush=True)
                return False
            print(f"[offload-process] transfer failed for {req_id!r} "
                  f"(attempt {attempts}/{max_retries}): {e!r}; retrying", flush=True)
            time.sleep(retry_delay)


def _offload_child(q_in, q_out, max_retries: int, retry_delay: float) -> None:
    """mp child entry (``use_process=True``): drain the job queue, transfer each with the SAME
    bounded-retry helper the thread path uses, and report ``(req_id, ok)`` back on ``q_out`` for the
    parent's collector thread to settle. Pure filesystem I/O -- no torch, no CUDA (blank
    ``CUDA_VISIBLE_DEVICES`` belt-and-braces). Always runs the module-level ``_default_transfer``
    (an injected transfer_fn forces the thread backend, so it never reaches here)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mia.graph.child_process import get_until_parent_exits
    while True:
        # None = close()'s sentinel, or the parent (e.g. a SIGKILLed TP worker) is gone and the
        # queue is empty: never an orphan (mia/graph/child_process.py).
        item = get_until_parent_exits(q_in)
        if item is None:
            break
        req_id, src_path, dest = item
        ok = _run_transfer_with_retries(_default_transfer, req_id, src_path, dest,
                                        max_retries, retry_delay)
        q_out.put((req_id, ok))


class OffloadProcess:
    """Lazily-nothing (eager-started) daemon thread + an UNBOUNDED queue. One worker, one consumer
    -> jobs for the same req_id run in submit order.

    NEVER DROPS: a ``transfer_fn`` that raises is HELD and RETRIED (not dropped) up to
    ``max_retries`` attempts, with ``retry_delay`` seconds between attempts (mirrors
    ``WriterProcess``'s retry-until-it-lands + never-crash-the-worker-on-one-bad-item posture in
    ``_writer_child``). Only after ``max_retries`` consecutive failures is a job given up on --
    it is dropped from the in-flight queue and recorded into ``poll_failed()`` instead;
    ``wait``/``poll_done`` never report a gave-up job as done (a doomed job is never silently
    marked complete). See the module docstring for the queue-sizing and one-submit-per-req_id
    rationale."""

    def __init__(self, transfer_fn=None, max_retries: int = 8, retry_delay: float = 0.05,
                 *, use_process: bool = False) -> None:
        self._transfer_fn = transfer_fn or _default_transfer
        self._max_retries = max(1, int(max_retries))
        self._retry_delay = max(0.0, float(retry_delay))
        # THREAD default; the mp child is opt-in and only when the caller did NOT inject a
        # transfer_fn (a closure / non-default sender may not pickle across the spawn boundary; the
        # mp child always runs the module-level _default_transfer). See the module docstring's
        # BACKEND note for the thread-is-enough-for-a-copy justification.
        self._use_process = bool(use_process) and transfer_fn is None

        # ONE lock guards all job-bookkeeping state below (events, result, inflight, done, failed),
        # so that submit()'s resubmit check (in-flight test + Event.clear() + inflight.add()) and
        # _mark_done()/_mark_failed()'s settle (inflight.discard() + result write + done/failed
        # append + Event.set()) run in the SAME critical section. With separate locks, a poll-based
        # resubmit could clear the (shared, reused-by-req_id) Event and then have the OLD job's
        # still-pending set() overwrite it, letting a stale completion satisfy the NEW job's wait().
        # Only bookkeeping is covered here -- transfer_fn and the blocking half of
        # Event.wait()/put_nowait() always run OUTSIDE this lock, so holding it is cheap and never
        # blocks on I/O; the worker thread never calls transfer_fn while holding it, so there is no
        # lock-ordering hazard.
        self._lock = threading.Lock()
        # Per-req completion signal for the client-blocks contract (wait()).
        self._events: dict[str, threading.Event] = {}
        # Per-req terminal outcome (True=done, False=gave-up), set alongside the Event so wait()
        # can tell a give-up wakeup apart from a real completion.
        self._result: dict[str, bool] = {}
        # req_ids currently queued or mid-retry -- guards resubmission of an in-flight req_id.
        self._inflight: set[str] = set()
        # Req ids transferred since the last poll_done() call.
        self._done: list[str] = []
        # Req ids that gave up (retry-exhausted) since the last poll_failed() call.
        self._failed: list[str] = []

        if self._use_process:
            # Real spawned child (mirrors writer_process / server_analyze_process): the child does
            # the copy in its own interpreter; a collector thread bridges its (req_id, ok) results
            # into the SAME per-req Event/dict bookkeeping the thread backend writes to directly, so
            # wait()/poll_done()/poll_failed() are identical code on either backend.
            import torch.multiprocessing as tmp  # lazy: keep the thread path torch-free
            self._ctx = tmp.get_context("spawn")
            # Unbounded (tiny path tuples) -> submit's put_nowait can never block/drop (never-drop).
            self._q = self._ctx.Queue()
            self._q.cancel_join_thread()
            self._q_out = self._ctx.Queue()
            self._q_out.cancel_join_thread()
            # Blank CUDA_VISIBLE_DEVICES in the PARENT before start() so spawn snapshots it into the
            # child's env (the child is a pure-fs copier -- no CUDA needed; this just keeps it from
            # touching the GPU). Restore right after (the parent's CUDA is already initialized).
            _saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self._proc = self._ctx.Process(
                    target=_offload_child,
                    args=(self._q, self._q_out, self._max_retries, self._retry_delay),
                    daemon=True, name="mia-offload")
                # start_child: also from a daemonic vLLM TP worker, where Process.start() refuses
                # (mia/graph/child_process.py).
                from mia.graph.child_process import start_child
                start_child(self._proc)
            finally:
                if _saved_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cvd
            self._collector = threading.Thread(target=self._collect_mp, daemon=True,
                                               name="mia-offload-collector")
            self._collector.start()
            self._worker = None
        else:
            # Unbounded: jobs are tiny (req_id, src_path, dest) path tuples, not tensors/SHM handles
            # -- holding an arbitrarily long backlog is memory-safe and IS the "hold, never drop"
            # contract. submit() uses put_nowait() below, so it can never block or raise Full.
            self._q = _queue.Queue()
            self._proc = None
            self._collector = None
            self._worker = threading.Thread(target=self._run, name="mia-offload-worker",
                                            daemon=True)
            self._worker.start()

    # ------------------------------------------------------------------
    # producer side (called from the engine loop / driver)
    # ------------------------------------------------------------------

    def submit(self, req_id: str, src_path: str, dest: str) -> None:
        """Non-blocking enqueue of a transfer job (unbounded queue -- see the class/module
        docstrings; a backlog is held, never dropped, and never stalls the caller). Creates (or
        reuses) this req's completion Event BEFORE enqueuing, so a `wait()` call racing right
        after `submit()` can never miss the signal.

        Raises ValueError if `req_id` is already in flight (queued or being retried) -- one
        submit per req_id until it settles; see the module docstring's ONE-SUBMIT-PER-REQ_ID
        contract."""
        with self._lock:
            if req_id in self._inflight:
                raise ValueError(
                    f"OffloadProcess.submit: req_id {req_id!r} is already in flight "
                    f"(queued or retrying) -- a req_id may only be submitted once until it "
                    f"completes or gives up")
            self._inflight.add(req_id)
            ev = self._events.get(req_id)
            if ev is None:
                ev = threading.Event()
                self._events[req_id] = ev
            else:
                ev.clear()  # re-submission of a settled req_id: wait() must block again
            # mp: refuse early (roll back the inflight add) if the child died -- mirrors
            # WriterProcess.submit / ServerAnalyzeProcess.submit. The THREAD backend (the default,
            # never-drop) has no such precheck: its worker is a daemon that only stops on the
            # sentinel, so alive() is always True short of a bug.
            if self._use_process and not self.alive():
                self._inflight.discard(req_id)
                return
        self._q.put_nowait((req_id, src_path, dest))  # unbounded (thread or mp) -> non-blocking

    def wait(self, req_id: str, timeout: float = None) -> bool:
        """Block until `req_id`'s transfer settles. Returns True only if it actually COMPLETED
        within `timeout`; returns False on timeout (including "never submitted") AND for a job
        that gave up after exhausting its retries -- a give-up still sets the Event so a blocked
        wait() wakes promptly (see poll_failed()), but it is never reported as success."""
        with self._lock:
            ev = self._events.get(req_id)
            if ev is None:
                ev = threading.Event()
                self._events[req_id] = ev
        if not ev.wait(timeout=timeout):
            return False
        with self._lock:
            return bool(self._result.get(req_id, False))

    def settled(self, req_id: str) -> bool:
        """NON-DESTRUCTIVE per-req test: True iff `req_id` was submitted and has since SETTLED --
        its transfer completed OR permanently gave up (retry-exhausted) -- and is not currently in
        flight. False while queued/retrying, or if never submitted.

        Distinct from poll_done()/poll_failed(), which DRAIN their lists ("since last poll"): this
        is a read-only membership test a caller can poll repeatedly and independently. Its use is
        "is the offload still READING the source dir?": a settled job (done or gave-up) is no longer
        touching src_path, so the source is safe to remove -- an in-flight one is not (rmtree mid-
        copytree corrupts the client dest and breaks the retry, whose source would vanish). A
        resubmitted req_id re-adds itself to `_inflight`, so this correctly reports False again for
        the duration of the new job even though `_result` still holds the prior outcome."""
        with self._lock:
            return req_id in self._result and req_id not in self._inflight

    def poll_done(self) -> list:
        """Drain + return the req_ids transferred since the last poll_done() call (non-blocking)."""
        with self._lock:
            out, self._done = self._done, []
        return out

    def poll_failed(self) -> list:
        """Drain + return the req_ids that permanently gave up (retry-exhausted) since the last
        poll_failed() call (non-blocking). Mirrors poll_done() for the failure case -- lets a
        caller learn a job will NEVER complete and report an error instead of waiting on it."""
        with self._lock:
            out, self._failed = self._failed, []
        return out

    def close(self, timeout: float = 15.0) -> None:
        """Best-effort, TIME-BOUNDED shutdown: signal the worker to stop and join it, but always
        return within `timeout` seconds regardless of whether the worker actually exits.

        Does NOT drain-then-join via an untimed `self._q.join()` -- that would hang forever if the
        job the worker is CURRENTLY on has a `transfer_fn` that blocks without ever raising
        (plausible for a future network sender hitting an unreachable host), and the bounded
        `worker.join()` below would then never even run. The sentinel put is itself non-blocking
        (unbounded queue), so it's safe to enqueue even while the worker is stuck on an earlier
        item -- once unstuck it will still drain everything queued before the sentinel.

        RESIDUAL LIMITATION (thread backend): a `threading.Thread` cannot be force-killed, so a
        truly stuck worker keeps running after close() returns -- the opt-in `use_process=True`
        backend fixes this for real (a stuck child can be `terminate()`d)."""
        if self._use_process:
            try:
                self._q.put(None, timeout=10)     # stop the child AFTER it drains its backlog
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._proc is not None and self._proc.is_alive():
                    self._proc.join(timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
            try:
                # Unblock the collector's blocking get() with a parent-injected sentinel -- anything
                # the child already put is ahead of it in FIFO order, so no real result is dropped.
                self._q_out.put(None, timeout=5)
            except Exception:  # noqa: BLE001
                pass
            if self._collector is not None:
                self._collector.join(timeout=5)
            return
        self._q.put(None)  # sentinel; non-blocking (unbounded queue) even if the worker is stuck
        self._worker.join(timeout=timeout)

    def alive(self) -> bool:
        """Is the backend worker running? (mp: the child process; thread: the worker thread.)
        Used by submit()'s mp precheck and by a caller that wants to know before submitting."""
        if self._use_process:
            return self._proc is not None and self._proc.is_alive()
        return self._worker is not None and self._worker.is_alive()

    # ------------------------------------------------------------------
    # worker side
    # ------------------------------------------------------------------

    def _mark_done(self, req_id: str) -> None:
        """Settle a completed transfer. The whole transition -- inflight.discard(), the result
        write, the done-list append (what makes this visible to poll_done()), and the completion
        Event.set() -- runs under ONE lock, atomic with submit()'s resubmit check (see the lock's
        docstring in __init__). That ordering is load-bearing: append-then-set with SEPARATE locks
        let a poll-based resubmit's Event.clear() land between them and then get overwritten by
        this same set(), leaking a stale completion into the resubmitted job's wait()."""
        with self._lock:
            self._inflight.discard(req_id)
            self._result[req_id] = True
            self._done.append(req_id)
            ev = self._events.get(req_id)
            if ev is not None:
                ev.set()

    def _mark_failed(self, req_id: str) -> None:
        """Retry-exhausted give-up: record into poll_failed() and wake a blocked wait() -- but
        wait() must still report False for it (see wait()'s docstring). Same one-lock atomicity
        with submit() as _mark_done() -- see its docstring."""
        with self._lock:
            self._inflight.discard(req_id)
            self._result[req_id] = False
            self._failed.append(req_id)
            ev = self._events.get(req_id)
            if ev is not None:
                ev.set()

    def _run(self) -> None:
        """Thread backend: drain the queue, transfer each job with the shared bounded-retry helper,
        and settle it (done vs gave-up) under the bookkeeping lock."""
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            req_id, src_path, dest = item
            ok = _run_transfer_with_retries(self._transfer_fn, req_id, src_path, dest,
                                            self._max_retries, self._retry_delay)
            if ok:
                self._mark_done(req_id)
            else:
                self._mark_failed(req_id)
            self._q.task_done()

    def _collect_mp(self) -> None:
        """mp backend: bridge the child's (req_id, ok) results into the same per-req Event/dict
        settle the thread path uses -- so wait()/poll_done()/poll_failed() are backend-agnostic."""
        while True:
            item = self._q_out.get()
            if item is None:
                break
            req_id, ok = item
            if ok:
                self._mark_done(req_id)
            else:
                self._mark_failed(req_id)
