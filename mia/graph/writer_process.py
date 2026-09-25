"""Separate process that serializes and writes captured artifacts off the engine GIL."""
from __future__ import annotations

import os
import queue as _queue

from mia.graph.child_process import get_until_parent_exits, register_shutdown, start_child
from mia.graph.thread_device import bind_thread_to_device, creator_cuda_device

_CHILD_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def _writer_child(q) -> None:
    for k, v in _CHILD_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mia.graph.artifact_writer import write_artifact
    from mia.graph.tensor_pack import unpack_tensor_tree
    while True:
        item = get_until_parent_exits(q)
        if item is None:
            break
        try:
            tag = item[0]
            if tag == "P":
                (_, wk, buffer, manifest, run_dir, mode, tp, use_st, fpt, ptn) = item
                cpu_cache = unpack_tensor_tree(buffer, manifest)
                write_artifact(wk, cpu_cache, run_dir, mode, tp, use_st, fpt, ptn)
            else:
                write_artifact(*item[1:])
        except Exception as e:  # noqa: BLE001
            print(f"[writer-process] save failed: {e!r}", flush=True)


class WriterProcess:
    """Lazily-started spawned daemon + a bounded torch.multiprocessing.Queue."""

    def __init__(self, maxsize: int = 4, n_writers: int = 1) -> None:
        self._device = creator_cuda_device()
        import torch.multiprocessing as tmp
        self._n_writers = max(1, int(n_writers))
        _want = os.environ.get("MIA_WRITER_SHARING", "file_system")
        try:
            if _want in tmp.get_all_sharing_strategies():
                tmp.set_sharing_strategy(_want)
        except Exception:  # noqa: BLE001
            pass
        self._ctx = tmp.get_context("spawn")
        self._q = self._ctx.Queue(maxsize=max(1, int(maxsize)))
        self._q.cancel_join_thread()
        saved = {k: os.environ.get(k) for k in (*_CHILD_THREAD_ENV, "CUDA_VISIBLE_DEVICES")}
        try:
            for k, v in _CHILD_THREAD_ENV.items():
                os.environ.setdefault(k, v)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self._procs = []
            self.parent_daemonic = False
            for _i in range(self._n_writers):
                _p = self._ctx.Process(target=_writer_child, args=(self._q,),
                                       daemon=True, name=f"mia-writer-{_i}")
                self.parent_daemonic = start_child(_p)
                self._procs.append(_p)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        import threading
        from mia.graph.tensor_pack import pack_tensor_tree
        self._pack = os.environ.get("MIA_WRITER_PACK", "1") != "0"
        self._pack_fn = pack_tensor_tree
        self._put_timeout = float(os.environ.get("MIA_WRITER_PUT_TIMEOUT", "30") or "30")
        _inq_max = int(os.environ.get("MIA_WRITER_INQ_MAX", "64") or "64")
        self._inq = _queue.Queue(maxsize=max(1, _inq_max))
        self.pages_done = _queue.Queue()
        self._feeder = threading.Thread(target=self._feed, name="mia-writer-feeder",
                                        daemon=True)
        self._feeder.start()

    def _feed(self) -> None:
        try:
            bind_thread_to_device(getattr(self, "_device", None))
        except Exception as e:  # noqa: BLE001
            self._feeder_error = e
            print(f"[writer-process] feeder thread could not select {self._device}: {e!r}; the "
                  f"writer refuses new work (submit -> False -> inline save)", flush=True)
            return
        from mia.graph.artifact_writer import write_artifact
        while True:
            raw = self._inq.get()
            if raw is None:
                break
            (wk, cpu_cache, run_dir, mode, tp, use_st, fpt, ptn, req_ids) = raw
            try:
                if not any(p.is_alive() for p in self._procs):
                    raise RuntimeError("no writer child is alive")
                if self._pack:
                    buffer, manifest = self._pack_fn(cpu_cache)
                    buffer.share_memory_()
                    self._q.put(("P", wk, buffer, manifest, run_dir, mode, tp,
                                 bool(use_st), bool(fpt), ptn), timeout=self._put_timeout)
                    for _rid in req_ids:
                        self.pages_done.put(_rid)
                else:
                    self._q.put(("R", wk, cpu_cache, run_dir, mode, tp,
                                 bool(use_st), bool(fpt), ptn), timeout=self._put_timeout)
            except Exception as e:  # noqa: BLE001
                try:
                    write_artifact(wk, cpu_cache, run_dir, mode, int(tp),
                                   bool(use_st), bool(fpt), ptn)
                    for _rid in req_ids:
                        self.pages_done.put(_rid)
                    print(f"[writer-process] feeder handoff failed ({e!r}); wrote {run_dir} "
                          f"inline on the feeder thread", flush=True)
                except Exception as e2:  # noqa: BLE001
                    print(f"[writer-process] feeder FALLBACK write FAILED for {run_dir}: {e2!r} "
                          f"(after handoff error {e!r})", flush=True)

    def poll_pages_done(self) -> list:
        """Drain req_ids whose disk-path bytes the feeder has copied out (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self.pages_done.get_nowait())
            except _queue.Empty:
                break
        return out

    @classmethod
    def from_env(cls):
        """Return a started WriterProcess UNLESS MIA_WRITER_PROCESS=0 forces it off."""
        if os.environ.get("MIA_WRITER_PROCESS", "1") == "0":
            return None
        maxsize = int(os.environ.get("MIA_WRITER_PROCESS_QSIZE", "4") or "4")
        n_writers = int(os.environ.get("MIA_WRITER_PROCESS_N", "1") or "1")
        return cls(maxsize, n_writers)

    def alive(self) -> bool:
        feeder = getattr(self, "_feeder", None)
        return (bool(getattr(self, "_procs", None)) and any(p.is_alive() for p in self._procs)
                and (feeder is None or feeder.is_alive()))

    def submit(self, worker_kind: str, cpu_cache: dict, run_dir: str, default_mode: str,
               tp_rank: int, use_safetensors: bool, force_pt: bool, pt_filename: str,
               *, req_ids: list = None, block: bool = False) -> bool:
        """Hand off to the off-engine feeder."""
        if not self.alive():
            return False
        item = (worker_kind, cpu_cache, run_dir, default_mode, int(tp_rank),
                bool(use_safetensors), bool(force_pt), pt_filename, list(req_ids or []))
        try:
            if block:
                self._inq.put(item, timeout=self._put_timeout)
            else:
                self._inq.put_nowait(item)
            return True
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        """Drain and stop, with bounded waits for the feeder and each child."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._inq.put(None, timeout=10)
            self._feeder.join(timeout=15)
        except Exception:  # noqa: BLE001
            pass
        try:
            for _p in self._procs:
                if _p.is_alive():
                    self._q.put(None, timeout=10)
            for _p in self._procs:
                _p.join(timeout=10)
        except Exception:  # noqa: BLE001
            pass


def _tree_bytes(obj) -> int:
    total = 0
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        else:
            try:
                total += int(x.element_size()) * int(x.numel())
            except Exception:  # noqa: BLE001
                pass
    return total


def _mem_limit_bytes() -> int:
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit():
                n = int(v)
                if 0 < n < (1 << 62):
                    return n
        except Exception:  # noqa: BLE001
            pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return 8 * (1 << 30)


def _resolve_flush_budget() -> int:
    b = os.environ.get("MIA_FLUSH_RAM_BUDGET_BYTES")
    if b:
        return max(1 << 20, int(b))
    frac = float(os.environ.get("MIA_FLUSH_RAM_BUDGET_FRAC", "0.25") or "0.25")
    return max(1 << 20, int(frac * _mem_limit_bytes()))


def _tp_label(worker) -> str:
    try:
        from mia.graph.tp_shard import resolve_tp_coords
        tp_rank, tp_size = resolve_tp_coords(worker)
        return f"tp_rank {tp_rank}/{tp_size}"
    except Exception:  # noqa: BLE001
        return "tp_rank ?"


def mark_no_writer(worker, reason: str) -> None:
    """Mark this rank as never writing artifacts, so no writer is started."""
    worker._writer_process = None
    worker._writer_mode = f"none ({reason})"
    print(f"[writer-process] NONE: {_tp_label(worker)} starts no writer ({reason})", flush=True)


def note_submit_refused(worker, wp) -> None:
    """Record that ``submit`` was refused, so this flush saves inline on the engine thread."""
    try:
        alive = bool(wp.alive())
    except Exception:  # noqa: BLE001
        alive = False
    reason = ("queue full past MIA_WRITER_PUT_TIMEOUT" if alive else "writer gone")
    try:
        from mia._profiler import PROF
        PROF.incr("writer.submit_refused")
    except Exception:  # noqa: BLE001
        pass
    if not alive:
        err = getattr(wp, "_feeder_error", None)
        procs = [(p.pid, p.exitcode) for p in (getattr(wp, "_procs", None) or [])]
        worker._writer_mode = (f"in-process (writer gone after start: children (pid, exitcode) "
                               f"{procs}" + (f", feeder error {err!r}" if err else "") + ")")
    seen = getattr(worker, "_writer_refusals_logged", None)
    if seen is None:
        seen = set()
        worker._writer_refusals_logged = seen
    if reason in seen:
        return
    seen.add(reason)
    print(f"[writer-process] submit refused ({reason}): {_tp_label(worker)} saves this flush "
          f"INLINE on the engine thread" + ("" if alive else
          f" and every later one; {worker._writer_mode}") + " (logged once per reason)",
          flush=True)


def init_writer_process(worker) -> None:
    """Start ``worker._writer_process`` once, unless MIA_WRITER_PROCESS=0."""
    if hasattr(worker, "_writer_process"):
        return
    try:
        from mia.graph.tp_shard import resolve_tp_coords
        tp_rank, tp_size = resolve_tp_coords(worker)
        where = f"tp_rank {tp_rank}/{tp_size}"
    except Exception:  # noqa: BLE001
        where = "tp_rank ?"
    try:
        worker._writer_process = WriterProcess.from_env()
        if worker._writer_process is not None:
            wp = worker._writer_process
            register_shutdown(wp.close)
            worker._writer_mode = "process"
            print(f"[writer-process] async serialize+write process ON: {where}, child pid(s) "
                  f"{[p.pid for p in wp._procs]}, started from a "
                  f"{'daemonic TP worker' if wp.parent_daemonic else 'non-daemonic process'} "
                  f"(pid {os.getpid()})", flush=True)
        else:
            worker._writer_mode = "in-process (MIA_WRITER_PROCESS=0)"
            print(f"[writer-process] OFF (MIA_WRITER_PROCESS=0): {where} serializes+writes "
                  f"in-process", flush=True)
    except Exception as e:  # noqa: BLE001
        worker._writer_process = None
        worker._writer_mode = f"in-process (start failed: {e!r})"
        print(f"[writer-process] failed to start, falling back to in-process save: {e!r} "
              f"({where})", flush=True)

