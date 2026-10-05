"""Process-local profiler for MIA."""
from __future__ import annotations

import atexit
import json
import os
import statistics
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, ContextManager, Dict, Iterator, List, Optional


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


class _NullCtx:
    """No-op timer context used when profiling is disabled."""
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NULL_CTX = _NullCtx()


_ENABLED  = _env_bool("MIA_PROFILE")
_FINE     = _env_bool("MIA_PROFILE_FINE")
_CUDA_EVT = _env_bool("MIA_PROFILE_CUDA") or _FINE
_MEM_SAMP = _env_bool("MIA_PROFILE_MEM")
_DUMP_DIR = os.environ.get("MIA_PROFILE_DIR", "/tmp/mia_profile")


def is_enabled() -> bool:
    return _ENABLED


class Profiler:
    """Thread-safe, process-local profiler."""

    def __init__(self) -> None:
        self.timers:   Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int]         = defaultdict(int)
        self.gauges:   Dict[str, List[float]] = defaultdict(list)
        self.events:   List[Dict[str, Any]]   = []
        self._lock = threading.Lock()
        self._dump_seq = 0
        self._start_wall = time.time()


    def timed(self, name: str, *, tier: int = 1) -> ContextManager[None]:
        """Wall-clock timer."""
        if not _ENABLED or (tier == 2 and not _FINE):
            return _NULL_CTX
        return self._timed_active(name)

    @contextmanager
    def _timed_active(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self.timers[name].append(dt_ms)

    def timed_cuda(self, name: str, *, tier: int = 1) -> ContextManager[None]:
        """GPU timer via CUDA events."""
        if not _ENABLED or not _CUDA_EVT or (tier == 2 and not _FINE):
            return _NULL_CTX
        return self._timed_cuda_active(name)

    @contextmanager
    def _timed_cuda_active(self, name: str) -> Iterator[None]:
        try:
            # lazy: the profiler also runs without torch (ImportError path below)
            import torch
        except ImportError:
            yield
            return
        if not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            with self._lock:
                self.timers[name].append(ms)

    def record_ms(self, name: str, ms: float) -> None:
        """Add a caller-measured timer sample for a span that is not one with-block."""
        if not _ENABLED:
            return
        with self._lock:
            self.timers[name].append(float(ms))


    def incr(self, name: str, n: int = 1) -> None:
        if not _ENABLED:
            return
        with self._lock:
            self.counters[name] += n

    def gauge(self, name: str, value: float) -> None:
        if not _ENABLED:
            return
        with self._lock:
            self.gauges[name].append(float(value))

    def event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Record a one-shot tagged event with a timestamp."""
        if not _ENABLED:
            return
        rec = {"t": time.time() - self._start_wall, "name": name}
        if payload:
            rec["payload"] = payload
        with self._lock:
            self.events.append(rec)
            if len(self.events) > 10_000:
                del self.events[: len(self.events) - 10_000]


    def reset(self) -> None:
        with self._lock:
            self.timers.clear()
            self.counters.clear()
            self.gauges.clear()
            self.events.clear()
            self._start_wall = time.time()

    def _summarize(self, samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"count": 0}
        n = len(samples)
        out = {
            "count": n,
            "sum":   sum(samples),
            "mean":  sum(samples) / n,
            "min":   min(samples),
            "max":   max(samples),
        }
        if n > 1:
            out["std"] = statistics.stdev(samples)
            sorted_s = sorted(samples)
            out["p50"] = sorted_s[n // 2]
            out["p90"] = sorted_s[min(n - 1, int(n * 0.90))]
            out["p99"] = sorted_s[min(n - 1, int(n * 0.99))]
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Return summary statistics for every recorded metric."""
        with self._lock:
            return {
                "enabled":  _ENABLED,
                "tier":     2 if _FINE else 1,
                "cuda":     _CUDA_EVT,
                "pid":      os.getpid(),
                "wall_s":   time.time() - self._start_wall,
                "timers":   {k: {**self._summarize(v), "samples_ms": list(v)}
                             for k, v in self.timers.items()},
                "counters": dict(self.counters),
                "gauges":   {k: {**self._summarize(v), "samples": list(v)}
                             for k, v in self.gauges.items()},
                "events":   list(self.events),
            }

    def summary_only(self) -> Dict[str, Any]:
        """Snapshot without per-sample arrays."""
        with self._lock:
            return {
                "enabled":  _ENABLED,
                "tier":     2 if _FINE else 1,
                "pid":      os.getpid(),
                "wall_s":   time.time() - self._start_wall,
                "timers":   {k: self._summarize(v) for k, v in self.timers.items()},
                "counters": dict(self.counters),
                "gauges":   {k: self._summarize(v) for k, v in self.gauges.items()},
            }

    def dump(self, path: Optional[str] = None, *, role: str = "proc") -> Optional[str]:
        """Write the full snapshot as JSON."""
        if not _ENABLED:
            return None
        if path is None:
            os.makedirs(_DUMP_DIR, exist_ok=True)
            self._dump_seq += 1
            path = os.path.join(
                _DUMP_DIR,
                f"profile-{role}-{os.getpid()}-{self._dump_seq}.json",
            )
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2, default=str)
        return path


PROF = Profiler()


def _detect_role() -> str:
    for key in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK", "PMI_RANK"):
        v = os.environ.get(key)
        if v is not None and v != "":
            return f"worker-r{v}"
    main = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    if "vllm" in main.lower() and "engine" in main.lower():
        return "worker"
    return "driver"


_ROLE = _detect_role()


def _atexit_dump() -> None:
    try:
        path = PROF.dump(role=_ROLE)
        if path is not None:
            print(f"[mia profiler] wrote {path}", file=sys.stderr, flush=True)
    except Exception:
        try:
            print("[mia profiler] atexit dump FAILED:", file=sys.stderr)
            traceback.print_exc()
        except Exception:
            pass


if _ENABLED:
    atexit.register(_atexit_dump)


class MemorySampler:
    """Background thread sampling NVML GPU memory and process RSS."""

    def __init__(self, interval_s: float = 0.05, gpu_index: int = 0) -> None:
        self.interval = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_handle = None
        self._psutil_proc = None
        self._torch = None
        self._cuda_dev: Optional[int] = None

    def _try_init(self) -> bool:
        try:
            # lazy: optional deps (pynvml, psutil); the profiler also runs without torch
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            self._nvml_handle = None
        try:
            import psutil
            self._psutil_proc = psutil.Process(os.getpid())
        except Exception:
            self._psutil_proc = None
        try:
            import torch
            self._torch = torch
        except Exception:
            self._torch = None
        return (self._nvml_handle is not None
                or self._psutil_proc is not None
                or self._torch is not None)

    def _own_cuda_device(self) -> Optional[int]:
        if self._cuda_dev is not None:
            return self._cuda_dev
        t = self._torch
        try:
            if t is None or not (t.cuda.is_available() and t.cuda.is_initialized()):
                return None
            used = [d for d in range(t.cuda.device_count()) if t.cuda.memory_reserved(d) > 0]
            if len(used) != 1:
                return None
            # lazy: keep mia.graph out of import mia; optional dependency (pynvml)
            from mia.graph.thread_device import bind_thread_to_device
            bind_thread_to_device(t.device("cuda", used[0]))
        except Exception:
            return None
        self._cuda_dev = used[0]
        if used[0] != self.gpu_index:
            try:
                import pynvml
                uuid = str(t.cuda.get_device_properties(used[0]).uuid)
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByUUID(
                    uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
            except Exception:
                self._nvml_handle = None
        return self._cuda_dev

    def _loop(self) -> None:
        # lazy: optional dependency (pynvml)
        import pynvml
        while not self._stop.is_set():
            dev = self._own_cuda_device()
            if self._nvml_handle is not None:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    PROF.gauge("mem.gpu_mb", info.used / 1024 ** 2)
                except Exception:
                    pass
            if self._psutil_proc is not None:
                try:
                    rss = self._psutil_proc.memory_info().rss
                    PROF.gauge("mem.host_rss_mb", rss / 1024 ** 2)
                except Exception:
                    pass
            if (self._torch is not None
                    and self._torch.cuda.is_available()
                    and self._torch.cuda.is_initialized()):
                try:
                    d = self.gpu_index if dev is None else dev
                    alloc    = self._torch.cuda.memory_allocated(d)
                    reserved = self._torch.cuda.memory_reserved(d)
                    PROF.gauge("mem.cuda_alloc_mb",    alloc    / 1024 ** 2)
                    PROF.gauge("mem.cuda_reserved_mb", reserved / 1024 ** 2)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        if not (_ENABLED and _MEM_SAMP):
            return
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._try_init():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="mia-mem-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


MEM_SAMPLER = MemorySampler()


if _ENABLED and _MEM_SAMP:
    MEM_SAMPLER.start()

