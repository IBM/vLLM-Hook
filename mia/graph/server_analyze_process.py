"""CPU process that runs a reducible analyzer's server-side reduce off the GPU path."""
from __future__ import annotations

import os
import queue as _queue
import threading


_CHILD_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}

_registry_ready = False


def _ensure_registry() -> None:
    global _registry_ready
    if not _registry_ready:
        from mia import register_plugins
        register_plugins()
        _registry_ready = True


def _default_analyze_fn(source_kind: str, source: dict, analyzer_name: str, analyzer_spec):
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
    while True:
        item = get_item()
        if item is None:
            break
        req_id, source_kind, source, analyzer_name, analyzer_spec = item
        try:
            result = analyze_fn(source_kind, source, analyzer_name, analyzer_spec)
            put_result((req_id, True, result))
        except Exception as e:  # noqa: BLE001
            put_result((req_id, False, repr(e)))


def _analyze_child(q_in, q_out) -> None:
    for k, v in _CHILD_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mia.graph.child_process import get_until_parent_exits
    _consume_loop(lambda: get_until_parent_exits(q_in), q_out.put, _default_analyze_fn)


class ServerAnalyzeProcess:
    """CPU-only analyze worker: a child process by default, a thread for injected analyze_fn."""

    def __init__(self, analyze_fn=None, use_process: bool = True,
                 maxsize: int = 64, put_timeout: float = 30.0) -> None:
        self._put_timeout = float(put_timeout)
        self._use_process = bool(use_process) and analyze_fn is None
        self._analyze_fn = analyze_fn or _default_analyze_fn
        from mia.graph.thread_device import creator_cuda_device
        self._device = creator_cuda_device()

        self._lock = threading.Lock()
        self._events: dict = {}
        self._result: dict = {}
        self._ok: dict = {}
        self._error: dict = {}
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
            saved = {k: os.environ.get(k) for k in (*_CHILD_THREAD_ENV, "CUDA_VISIBLE_DEVICES")}
            try:
                for k, v in _CHILD_THREAD_ENV.items():
                    os.environ.setdefault(k, v)
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self._proc = self._ctx.Process(target=_analyze_child, args=(self._q_in, self._q_out),
                                               daemon=True, name="mia-server-analyze")
                from mia.graph.child_process import start_child
                start_child(self._proc)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self._collector = threading.Thread(target=self._collect_mp, daemon=True,
                                               name="mia-server-analyze-collector")
            self._collector.start()
            self._worker = None
        else:
            self._q_in = _queue.Queue()
            self._proc = None
            self._collector = None
            self._worker = threading.Thread(target=self._run_thread, daemon=True,
                                            name="mia-server-analyze-worker")
            self._worker.start()


    def submit(self, req_id: str, source_kind: str, source: dict,
               analyzer_name: str, analyzer_spec=None) -> bool:
        """Non-blocking enqueue of an analyze job."""
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
                ev.clear()
            if self._use_process and not self.alive():
                self._inflight.discard(req_id)
                return False
        item = (req_id, source_kind, source, analyzer_name, analyzer_spec)
        if self._use_process:
            try:
                self._q_in.put(item, timeout=self._put_timeout)
                return True
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._inflight.discard(req_id)
                return False
        self._q_in.put_nowait(item)
        return True

    def wait(self, req_id: str, timeout: float = None):
        """Block until `req_id`'s analyze settles."""
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
        """Drain and return req_ids analyzed successfully since the last call (non-blocking)."""
        with self._lock:
            out, self._done = self._done, []
        return out

    def poll_failed(self) -> list:
        """Drain and return (req_id, error_repr) for analyze calls that raised since the last call."""
        with self._lock:
            out = [(rid, self._error.get(rid, "")) for rid in self._failed]
            self._failed = []
        return out

    def close(self, timeout: float = 15.0) -> None:
        """Best-effort, time-bounded shutdown that never hangs on a stuck child."""
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


    def _run_thread(self) -> None:
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
    """Start ``worker._server_analyze_process`` once, unless MIA_SERVER_ANALYZE_PROCESS is off."""
    if hasattr(worker, "_server_analyze_process"):
        return
    if os.environ.get("MIA_SERVER_ANALYZE_PROCESS", "0") != "1":
        worker._server_analyze_process = None
        return
    try:
        worker._server_analyze_process = ServerAnalyzeProcess()
        from mia.graph.child_process import register_shutdown
        register_shutdown(worker._server_analyze_process.close)
        print("[server-analyze] CPU analyze process ON", flush=True)
    except Exception as e:  # noqa: BLE001
        worker._server_analyze_process = None
        print(f"[server-analyze] failed to start, falling back to inline analyze: {e!r}",
              flush=True)

