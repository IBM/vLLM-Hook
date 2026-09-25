"""A separate CPU process that runs a REDUCIBLE analyzer's server-side reduce off the GPU and
off the drain consumer thread.

When the router decides a request's chosen analyzer is REDUCIBLE server-side (e.g.
``hidden_states`` with ``analyzer_spec={"reduce": "mean"|"norm"}``), it ships the small RESULT
instead of the whole per-request artifact. ``analyze_where`` (``graph/delivery_router.py``) is
``"inflight"`` (analyze from the host-buffer artifact the drain assembled -- see
``per_request_delivery.PerRequestIndex.pop_deliverable``) or ``"from_disk"`` (the artifact was too
big to hold -- stream it to a per-request file, then analyze that file). This module runs BOTH
cases entirely off the GPU forward and off the drain consumer thread, in a dedicated CPU worker.

Mirrors ``writer_process.py``'s off-loop-child shape (``torch.multiprocessing`` spawn,
``CUDA_VISIBLE_DEVICES=""`` + OMP/BLAS caps set in the PARENT before ``start()`` -- too late inside
the child, see that module's own docstring for why) and ``offload_process.py``'s API shape
(``submit`` non-blocking, ``wait(req_id, timeout)``, ``poll_done``/``poll_failed``, ``close``, an
injectable seam for testing).

ANALYZE CONTRACT (matches ``llm.py`` / ``client.py`` / ``run_utils.dispatch_disk_analyze``):
``PluginRegistry.get_analyzer(name).analyzer`` is the analyzer CLASS; instantiate it
``analyzer_cls(hook_dir, layer_to_heads)`` then call ``.analyze(analyzer_spec=..., probes=...)`` for
the in-flight/host-buffer source, or route through ``dispatch_disk_analyze(analyzer, analyzer_spec,
run_id=..., run_ids=...)`` for the on-disk source -- NOT a raw ``.analyze(run_id=...)`` call, so a
future two-pass analyzer (CoRer-style, ``run_ids=[doc, na]``) is not silently unsupported here.

TESTABILITY SEAM (mirrors ``OffloadProcess``'s injectable ``transfer_fn``): the actual invocation
runs through ``analyze_fn(source_kind, source, analyzer_name, analyzer_spec) -> result``, default =
``_default_analyze_fn`` (the registry-based real invocation above). An injected ``analyze_fn`` forces
the THREAD backend (see PROCESS-VS-THREAD below) so a no-GPU test can hand it an arbitrary Python
closure / fake analyzer without that closure needing to survive an mp spawn pickle.

``source`` SHAPE -- a plain dict carrying everything ``_default_analyze_fn`` needs (kept flat so the
4-arg ``analyze_fn`` signature above never has to grow):
  * ``source_kind="inflight"``: ``{"probes": <the artifact dict, e.g. {"hs_cache": {...}}>,
    "hook_dir": <str, optional>, "layer_to_heads": <dict, optional>}``.
  * ``source_kind="from_disk"``: ``{"run_id": <str>, "hook_dir": <str>,
    "layer_to_heads": <dict, optional>, "run_ids": <list[str], optional -- two-pass analyzers>}``.

PROCESS-VS-THREAD: the same core consume loop (``_consume_loop``) runs on either backend --
``use_process=True`` (default, production) spawns a real ``torch.multiprocessing`` child running the
DEFAULT registry-based ``analyze_fn`` (a module-level function, so it pickles by reference cleanly
for the spawn target); ``use_process=False`` (forced whenever a caller injects a custom
``analyze_fn``) runs the IDENTICAL loop on a ``threading.Thread`` so tests stay fast/deterministic and
can inject an arbitrary non-picklable fake. This mirrors ``offload_process.py``'s thread/process
split, except here BOTH backends are built now rather than a deferred hardening step.

NOT WIRED INTO ANY WORKER (deliberate): this is a new, self-contained module only.
``init_server_analyze_process(worker)`` below is the ready lazy-start hook -- mirrors
``writer_process.init_writer_process`` exactly (idempotent, default-OFF here pending
calibration) -- for a future request-start router to call once an ``analyze_where`` exists to
route to. This module does not touch ``PerRequestIndex``, the drain, the QK path,
``get_captured_states``, ``flush_aperture*``, ``get_aperture_per_request``, or the router.
"""
from __future__ import annotations

import os
import queue as _queue
import threading


# The child does pure CPU reduce (mean/norm over already-CPU tensors, no CUDA); cap its thread
# pools like writer_process's child so importing torch in a spawned child of the (already forked)
# worker never trips the process/thread rlimit.
_CHILD_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}

_registry_ready = False


def _ensure_registry() -> None:
    """Populate ``PluginRegistry`` once per process. Load-bearing for the mp backend: a spawned
    child is a FRESH interpreter (unlike fork, it does not inherit the parent's already-populated
    registry), so ``PluginRegistry.get_analyzer`` would otherwise always return ``None`` there.
    Mirrors ``client.py``'s own ``from mia import register_plugins;
    register_plugins()`` at construction. Idempotent (re-registering is just dict writes) -- safe
    to call on every ``_default_analyze_fn`` invocation; guarded so the (heavy, one-time) import
    only runs once per process."""
    global _registry_ready
    if not _registry_ready:
        from mia import register_plugins
        register_plugins()
        _registry_ready = True


def _default_analyze_fn(source_kind: str, source: dict, analyzer_name: str, analyzer_spec):
    """Registry-based real invocation -- module level so it pickles by reference as the mp spawn
    target. See the module docstring for the exact ``source`` shape per ``source_kind``."""
    _ensure_registry()
    from mia.registry import PluginRegistry
    from mia.run_utils import dispatch_disk_analyze

    entry = PluginRegistry.get_analyzer(analyzer_name)
    if entry is None:
        raise ValueError(f"ServerAnalyzeProcess: unknown analyzer {analyzer_name!r}")
    analyzer_cls = entry.analyzer
    hook_dir = (source or {}).get("hook_dir") or ""
    layer_to_heads = (source or {}).get("layer_to_heads") or {}
    analyzer = analyzer_cls(hook_dir, layer_to_heads)

    if source_kind == "inflight":
        probes = source["probes"]
        return analyzer.analyze(analyzer_spec=analyzer_spec, probes=probes)
    if source_kind == "from_disk":
        run_id = source.get("run_id")
        run_ids = source.get("run_ids")
        return dispatch_disk_analyze(analyzer, analyzer_spec, run_id=run_id, run_ids=run_ids)
    raise ValueError(f"ServerAnalyzeProcess: unknown source_kind {source_kind!r} "
                      f"(expected 'inflight' or 'from_disk')")


def _consume_loop(get_item, put_result, analyze_fn) -> None:
    """Core loop shared by the thread backend (plain ``queue.Queue`` callables) and the mp child
    entry point (``mp.Queue`` callables) -- see the module docstring. ``get_item()`` blocks for the
    next ``(req_id, source_kind, source, analyzer_name, analyzer_spec)`` 5-tuple, or ``None`` (the
    stop sentinel). Every analyze error is caught and reported as ``(req_id, False, repr(exc))`` --
    ONE bad request never kills the loop (failure isolation), mirroring ``_writer_child``'s
    per-item try/except in ``writer_process.py``."""
    while True:
        item = get_item()
        if item is None:
            break
        req_id, source_kind, source, analyzer_name, analyzer_spec = item
        try:
            result = analyze_fn(source_kind, source, analyzer_name, analyzer_spec)
            put_result((req_id, True, result))
        except Exception as e:  # noqa: BLE001 -- never crash the loop on one bad item
            put_result((req_id, False, repr(e)))


def _analyze_child(q_in, q_out) -> None:
    """mp child entry point. The thread caps + blank ``CUDA_VISIBLE_DEVICES`` here are
    belt-and-braces only -- the LOAD-BEARING set is in the parent before ``start()`` (see
    ``ServerAnalyzeProcess.__init__``), which is already in the child's inherited environment by
    the time spawn's bootstrap imports this module (and therefore torch). Runs the shared loop
    against the DEFAULT registry-based ``analyze_fn`` -- an injected ``analyze_fn`` never reaches
    this function (it forces the thread backend instead, see the module docstring)."""
    for k, v in _CHILD_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mia.graph.child_process import get_until_parent_exits
    # A None from the getter (close()'s sentinel, or: parent gone and queue empty) stops the loop,
    # so the child never outlives its worker (mia/graph/child_process.py).
    _consume_loop(lambda: get_until_parent_exits(q_in), q_out.put, _default_analyze_fn)


class ServerAnalyzeProcess:
    """A CPU-only analyze worker (real child process by default; thread backend for tests / an
    injected ``analyze_fn``). One consumer -> jobs run in submit order. See the module docstring
    for the PROCESS-VS-THREAD split and the ``source`` shape contract.

    Never touches the GPU forward: this class is not on the ``execute_model`` critical path at
    all -- a caller (the router / worker finalize) submits a finished request's artifact here
    from an off-loop thread and later collects the small result."""

    def __init__(self, analyze_fn=None, use_process: bool = True,
                 maxsize: int = 64, put_timeout: float = 30.0) -> None:
        self._put_timeout = float(put_timeout)
        # An injected analyze_fn forces the thread backend -- see the module docstring: a
        # closure/fake analyzer is not guaranteed picklable across an mp spawn boundary, and tests
        # want a fast, deterministic backend regardless.
        self._use_process = bool(use_process) and analyze_fn is None
        self._analyze_fn = analyze_fn or _default_analyze_fn
        # The THREAD backend runs arbitrary analyzer code, so its thread binds the constructing
        # thread's CUDA device first (None without CUDA) -- mia/graph/thread_device.py.
        from mia.graph.thread_device import creator_cuda_device
        self._device = creator_cuda_device()

        # Bookkeeping mirrors OffloadProcess: ONE lock over all per-req state so a resubmit can
        # never race a settle (see offload_process.py's __init__ docstring for the atomicity
        # rationale this reproduces verbatim).
        self._lock = threading.Lock()
        self._events: dict = {}          # req_id -> threading.Event
        self._result: dict = {}          # req_id -> analyze() return value (success only)
        self._ok: dict = {}              # req_id -> True success / False failed
        self._error: dict = {}           # req_id -> repr(exc) for a failed req
        self._inflight: set = set()
        self._done: list = []
        self._failed: list = []

        if self._use_process:
            import torch.multiprocessing as tmp
            self._ctx = tmp.get_context("spawn")
            self._q_in = self._ctx.Queue(maxsize=max(1, int(maxsize)))
            self._q_in.cancel_join_thread()
            self._q_out = self._ctx.Queue()
            self._q_out.cancel_join_thread()
            # Load-bearing: put CUDA_VISIBLE_DEVICES="" + the thread caps into os.environ BEFORE
            # start() so spawn snapshots them into the child's environment -- env vars are read at
            # torch/libgomp init, on the child's FIRST import of torch, triggered by unpickling the
            # target-BY-REFERENCE (_analyze_child) -- too late inside _analyze_child itself.
            # Restore the parent's own values right after start() (the parent's CUDA is already
            # initialized, so its live state is unaffected). Mirrors writer_process.py exactly.
            saved = {k: os.environ.get(k) for k in (*_CHILD_THREAD_ENV, "CUDA_VISIBLE_DEVICES")}
            try:
                for k, v in _CHILD_THREAD_ENV.items():
                    os.environ.setdefault(k, v)
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self._proc = self._ctx.Process(target=_analyze_child, args=(self._q_in, self._q_out),
                                               daemon=True, name="mia-server-analyze")
                # start_child: also from a daemonic vLLM TP worker (mia/graph/child_process.py).
                from mia.graph.child_process import start_child
                start_child(self._proc)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            # Collector thread: bridges the mp result queue into the SAME per-req Event/dict
            # bookkeeping the thread backend writes to directly -- wait()/poll_done()/poll_failed()
            # are then identical code on either backend.
            self._collector = threading.Thread(target=self._collect_mp, daemon=True,
                                               name="mia-server-analyze-collector")
            self._collector.start()
            self._worker = None
        else:
            # Unbounded, like OffloadProcess -- jobs here are tiny tuples (the artifact tensors
            # live in `source`, but nothing here is copied again before handoff), so holding an
            # arbitrarily deep backlog is the "hold, never drop" contract, not a memory risk beyond
            # what the caller already retained.
            self._q_in = _queue.Queue()
            self._proc = None
            self._collector = None
            self._worker = threading.Thread(target=self._run_thread, daemon=True,
                                            name="mia-server-analyze-worker")
            self._worker.start()

    # ------------------------------------------------------------------
    # producer side (called from the router / worker finalize, off-loop)
    # ------------------------------------------------------------------

    def submit(self, req_id: str, source_kind: str, source: dict,
               analyzer_name: str, analyzer_spec=None) -> bool:
        """Non-blocking enqueue of an analyze job. Returns True once queued; on the mp backend a
        saturated/dead child returns False (data-safety fall-through -- the caller decides, e.g.
        fall back to a raw-delivery route); the thread backend's unbounded queue never refuses.

        Creates (or re-arms) this req's completion Event BEFORE enqueuing, so a `wait()` call
        racing right after `submit()` can never miss the signal (mirrors OffloadProcess.submit).
        Raises ValueError if `req_id` is already in flight -- one submit per req_id until it
        settles, same one-submit-per-req_id contract as OffloadProcess."""
        with self._lock:
            if req_id in self._inflight:
                raise ValueError(
                    f"ServerAnalyzeProcess.submit: req_id {req_id!r} is already in flight "
                    f"-- a req_id may only be submitted once until it completes or fails")
            self._inflight.add(req_id)
            ev = self._events.get(req_id)
            if ev is None:
                ev = threading.Event()
                self._events[req_id] = ev
            else:
                ev.clear()  # re-submission of a settled req_id: wait() must block again
            # Mirror WriterProcess.submit(): if the backend child is not alive, refuse early
            if self._use_process and not self.alive():
                self._inflight.discard(req_id)
                return False
        item = (req_id, source_kind, source, analyzer_name, analyzer_spec)
        if self._use_process:
            try:
                self._q_in.put(item, timeout=self._put_timeout)
                return True
            except Exception:  # noqa: BLE001 -- full/dead child -> caller decides
                with self._lock:
                    self._inflight.discard(req_id)
                return False
        self._q_in.put_nowait(item)  # unbounded -- never raises
        return True

    def wait(self, req_id: str, timeout: float = None):
        """Block until `req_id`'s analyze settles. Returns the analyze RESULT on success, or
        `None` on timeout (including "never submitted") AND for a job whose analyze raised (a
        failure still sets the Event so a blocked wait() wakes promptly -- see poll_failed() for
        the error detail)."""
        with self._lock:
            ev = self._events.get(req_id)
            if ev is None:
                ev = threading.Event()
                self._events[req_id] = ev
        if not ev.wait(timeout=timeout):
            return None
        with self._lock:
            if self._ok.get(req_id):
                return self._result.get(req_id)
            return None

    def poll_done(self) -> list:
        """Drain + return the req_ids that analyzed successfully since the last poll_done() call
        (non-blocking). Use `wait()` or a future `result(req_id)` accessor to read the value."""
        with self._lock:
            out, self._done = self._done, []
        return out

    def poll_failed(self) -> list:
        """Drain + return `(req_id, error_repr)` pairs for analyze calls that RAISED since the
        last poll_failed() call (non-blocking). Deliberately richer than OffloadProcess's
        plain-req_id poll_failed() -- the error message IS the useful signal here (a caller wants
        to know *why* the reduce failed, not just that it did), and it is unlike the offload path,
        which has no comparable per-job diagnostic beyond "ran out of retries"."""
        with self._lock:
            out = [(rid, self._error.get(rid, "")) for rid in self._failed]
            self._failed = []
        return out

    def close(self, timeout: float = 15.0) -> None:
        """Best-effort, TIME-BOUNDED shutdown (never hangs on a dead/stuck child or worker --
        mirrors OffloadProcess.close / WriterProcess.close)."""
        try:
            self._q_in.put(None, timeout=(10 if self._use_process else None))
        except Exception:  # noqa: BLE001
            pass
        if self._use_process:
            try:
                if self._proc is not None and self._proc.is_alive():
                    self._proc.join(timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
            try:
                # Unblock the collector's blocking get() with a PARENT-injected sentinel --
                # anything the child already put is ahead of it in FIFO order, so this never
                # drops a real result even if the child hadn't fully drained by the join above.
                self._q_out.put(None, timeout=5)
            except Exception:  # noqa: BLE001
                pass
            if self._collector is not None:
                self._collector.join(timeout=5)
        else:
            if self._worker is not None:
                self._worker.join(timeout=timeout)

    def alive(self) -> bool:
        if self._use_process:
            return self._proc is not None and self._proc.is_alive()
        return self._worker is not None and self._worker.is_alive()

    # ------------------------------------------------------------------
    # worker side
    # ------------------------------------------------------------------

    def _run_thread(self) -> None:
        # FIRST: this rank's device, before an analyzer can touch CUDA (a new thread starts on
        # cuda:0, another process's GPU on TP rank >= 1). A failed bind kills this thread loud
        # (its traceback) and alive() reports False.
        from mia.graph.thread_device import bind_thread_to_device
        bind_thread_to_device(self._device)
        _consume_loop(self._q_in.get, self._handle_result, self._analyze_fn)

    def _collect_mp(self) -> None:
        while True:
            item = self._q_out.get()
            if item is None:
                break
            self._handle_result(item)

    def _handle_result(self, item) -> None:
        req_id, ok, payload = item
        with self._lock:
            self._inflight.discard(req_id)
            self._ok[req_id] = ok
            if ok:
                self._result[req_id] = payload
                self._done.append(req_id)
            else:
                self._error[req_id] = payload
                self._failed.append(req_id)
            ev = self._events.get(req_id)
            if ev is not None:
                ev.set()


def init_server_analyze_process(worker) -> None:
    """Start ``worker._server_analyze_process`` once, unless ``MIA_SERVER_ANALYZE_PROCESS``
    is unset/``"0"``. Mirrors ``writer_process.init_writer_process`` exactly (idempotent, cheap
    idle child when nothing is submitted).

    DEFAULT OFF (unlike the writer process): no caller submits to this process yet -- a future
    router decides ``analyze_where in {inflight, from_disk}``, and calibration decides when a
    reducible analyze is actually worth it. Starting an idle CPU child by default on every
    graph/eager install, ahead of that decision, would be pure downside (extra child-process
    startup cost -- the same heavy one-time ``mia`` package import
    writer_process's child already pays -- with no upside until something calls ``submit()``).
    Flip this default once a router is wired, or call ``init_server_analyze_process(worker)``
    directly at that point (this function is idempotent either way -- a second call is a no-op
    via the ``hasattr`` guard).

    NOT CALLED from any worker/install file yet (deliberate -- see the module docstring's "NOT
    WIRED INTO ANY WORKER" note)."""
    if hasattr(worker, "_server_analyze_process"):
        return
    if os.environ.get("MIA_SERVER_ANALYZE_PROCESS", "0") != "1":
        worker._server_analyze_process = None
        return
    try:
        worker._server_analyze_process = ServerAnalyzeProcess()
        # Before multiprocessing terminates the child, then at atexit (child_process rule 2).
        from mia.graph.child_process import register_shutdown
        register_shutdown(worker._server_analyze_process.close)
        print("[server-analyze] CPU analyze process ON", flush=True)
    except Exception as e:  # noqa: BLE001 -- never fail worker init on this
        worker._server_analyze_process = None
        print(f"[server-analyze] failed to start, falling back to inline analyze: {e!r}",
              flush=True)
