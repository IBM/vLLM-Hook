"""Background worker that ships a finished request's per-request file to the client destination."""
from __future__ import annotations

import os
import queue as _queue
import shutil
import threading
import time


def _aperture_debug() -> bool:
    return os.environ.get("MIA_APERTURE_DEBUG") == "1"


def _default_transfer(src_path: str, dest: str) -> None:
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dest, dirs_exist_ok=True)
        return
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy(src_path, dest)


def _run_transfer_with_retries(transfer_fn, req_id: str, src_path: str, dest: str,
                               max_retries: int, retry_delay: float) -> bool:
    attempts = 0
    while True:
        attempts += 1
        try:
            if _aperture_debug():
                print(f"[mia/aperture-disk] offload START req_id={req_id!r} src={src_path!r} "
                      f"dest={dest!r} attempt={attempts}", flush=True)
            transfer_fn(src_path, dest)
            if _aperture_debug():
                print(f"[mia/aperture-disk] offload DONE  req_id={req_id!r} dest={dest!r}",
                      flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            if _aperture_debug():
                print(f"[mia/aperture-disk] offload ERROR req_id={req_id!r} src={src_path!r} "
                      f"dest={dest!r} attempt={attempts}: {e!r}", flush=True)
            if attempts >= max_retries:
                print(f"[offload-process] transfer FAILED for {req_id!r} after "
                      f"{attempts} attempts, giving up: {e!r}", flush=True)
                return False
            print(f"[offload-process] transfer failed for {req_id!r} "
                  f"(attempt {attempts}/{max_retries}): {e!r}; retrying", flush=True)
            time.sleep(retry_delay)


def _offload_child(q_in, q_out, max_retries: int, retry_delay: float) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mia.graph.child_process import get_until_parent_exits
    while True:
        item = get_until_parent_exits(q_in)
        if item is None:
            break
        req_id, src_path, dest = item
        ok = _run_transfer_with_retries(_default_transfer, req_id, src_path, dest,
                                        max_retries, retry_delay)
        q_out.put((req_id, ok))


class OffloadProcess:
    """Eagerly started daemon thread with an unbounded transfer queue."""

    def __init__(self, transfer_fn=None, max_retries: int = 8, retry_delay: float = 0.05,
                 *, use_process: bool = False) -> None:
        self._transfer_fn = transfer_fn or _default_transfer
        self._max_retries = max(1, int(max_retries))
        self._retry_delay = max(0.0, float(retry_delay))
        self._use_process = bool(use_process) and transfer_fn is None

        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._result: dict[str, bool] = {}
        self._inflight: set[str] = set()
        self._done: list[str] = []
        self._failed: list[str] = []

        if self._use_process:
            import torch.multiprocessing as tmp
            self._ctx = tmp.get_context("spawn")
            self._q = self._ctx.Queue()
            self._q.cancel_join_thread()
            self._q_out = self._ctx.Queue()
            self._q_out.cancel_join_thread()
            _saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self._proc = self._ctx.Process(
                    target=_offload_child,
                    args=(self._q, self._q_out, self._max_retries, self._retry_delay),
                    daemon=True, name="mia-offload")
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
            self._q = _queue.Queue()
            self._proc = None
            self._collector = None
            self._worker = threading.Thread(target=self._run, name="mia-offload-worker",
                                            daemon=True)
            self._worker.start()


    def submit(self, req_id: str, src_path: str, dest: str) -> None:
        """Non-blocking enqueue of a transfer job; backlog is held, never dropped."""
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
                ev.clear()
            if self._use_process and not self.alive():
                self._inflight.discard(req_id)
                return
        self._q.put_nowait((req_id, src_path, dest))

    def wait(self, req_id: str, timeout: float = None) -> bool:
        """Block until `req_id`'s transfer settles."""
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
        """True iff ``req_id`` was submitted and its transfer completed or gave up."""
        with self._lock:
            return req_id in self._result and req_id not in self._inflight

    def poll_done(self) -> list:
        """Drain and return req_ids transferred since the last call (non-blocking)."""
        with self._lock:
            out, self._done = self._done, []
        return out

    def poll_failed(self) -> list:
        """Drain and return req_ids that permanently failed since the last call (non-blocking)."""
        with self._lock:
            out, self._failed = self._failed, []
        return out

    def close(self, timeout: float = 15.0) -> None:
        """Best-effort shutdown that always returns within ``timeout`` seconds."""
        if self._use_process:
            try:
                self._q.put(None, timeout=10)
            except Exception:  # noqa: BLE001
                pass
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
            return
        self._q.put(None)
        self._worker.join(timeout=timeout)

    def alive(self) -> bool:
        """Is the backend worker running?"""
        if self._use_process:
            return self._proc is not None and self._proc.is_alive()
        return self._worker is not None and self._worker.is_alive()


    def _mark_done(self, req_id: str) -> None:
        with self._lock:
            self._inflight.discard(req_id)
            self._result[req_id] = True
            self._done.append(req_id)
            ev = self._events.get(req_id)
            if ev is not None:
                ev.set()

    def _mark_failed(self, req_id: str) -> None:
        with self._lock:
            self._inflight.discard(req_id)
            self._result[req_id] = False
            self._failed.append(req_id)
            ev = self._events.get(req_id)
            if ev is not None:
                ev.set()

    def _run(self) -> None:
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
        while True:
            item = self._q_out.get()
            if item is None:
                break
            req_id, ok = item
            if ok:
                self._mark_done(req_id)
            else:
                self._mark_failed(req_id)

