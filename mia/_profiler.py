"""Process-local profiler for MIA.

Activated by ``MIA_PROFILE=1``. When disabled, every entry point is a
no-op so the production hot path is unchanged.

Env vars
--------
MIA_PROFILE         "1" to enable. Default off.
MIA_PROFILE_FINE    "1" to also record tier-2 timers (more events,
                          more overhead — adds CUDA event syncs).
MIA_PROFILE_CUDA    "1" to record timed_cuda(...) GPU events.
                          Always implied when PROFILE_FINE=1.
MIA_PROFILE_DIR     Where dump() writes JSON. Default
                          /tmp/mia_profile.
MIA_PROFILE_MEM     "1" to start the background NVML+RSS sampler.

Usage
-----
    from mia._profiler import PROF

    with PROF.timed("hookllm.generate"):
        outputs = llm.generate(...)

    PROF.incr("hook.fire")
    PROF.gauge("rpc.payload_bytes", len(payload))

    PROF.dump()   # writes <PROFILE_DIR>/profile-<pid>-<seq>.json
"""
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
    """The disabled-timer context manager: a stateless, reusable, allocation-free singleton.

    A ``@contextmanager`` generator function is not actually free when disabled -- it still
    builds a generator object and a wrapper, then drives them through __enter__/__exit__. This
    singleton makes the disabled path a genuine no-op instead of an approximate one, which
    matters on the default per-step routing path where the measured quantity is sub-millisecond.

    Stateless and therefore safe to share across threads and to nest arbitrarily: it holds no
    per-entry data, so re-entering the same object is indistinguishable from entering a fresh one.
    """
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
    """Thread-safe, process-local profiler.

    Three collection primitives:
    - ``timed(name)`` — CPU wall-clock context manager, records ms.
    - ``timed_cuda(name)`` — GPU time via CUDA events, records ms.
                              No-op unless MIA_PROFILE_CUDA=1.
    - ``incr(name, n)`` — monotonic counter.
    - ``gauge(name, value)`` — append a numeric sample (bytes, queue depth).

    ``snapshot()`` returns summary stats. ``dump(path)`` writes JSON.
    """

    def __init__(self) -> None:
        self.timers:   Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int]         = defaultdict(int)
        self.gauges:   Dict[str, List[float]] = defaultdict(list)
        self.events:   List[Dict[str, Any]]   = []   # bounded; used for traces
        self._lock = threading.Lock()
        self._dump_seq = 0
        self._start_wall = time.time()

    # ------------------------------------------------------------------
    # Timer primitives
    # ------------------------------------------------------------------

    def timed(self, name: str, *, tier: int = 1) -> ContextManager[None]:
        """Wall-clock timer. A GENUINE no-op when disabled — see ``_NullCtx``."""
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
        """GPU timer via CUDA events. A GENUINE no-op when disabled — see ``_NullCtx``."""
        # Tier-2 by default — CUDA events require a sync that distorts timing.
        if not _ENABLED or not _CUDA_EVT or (tier == 2 and not _FINE):
            return _NULL_CTX
        return self._timed_cuda_active(name)

    @contextmanager
    def _timed_cuda_active(self, name: str) -> Iterator[None]:
        try:
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
        """Add one timer sample measured by the caller -- for a span that is not one ``with``
        block (the aperture drain splits one step into overlapping d2h / write phases)."""
        if not _ENABLED:
            return
        with self._lock:
            self.timers[name].append(float(ms))

    # ------------------------------------------------------------------
    # Counter / gauge primitives
    # ------------------------------------------------------------------

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
        """Record a one-shot tagged event with a timestamp.

        Bounded to 10k events to avoid runaway memory in long-running
        servers; older events are dropped.
        """
        if not _ENABLED:
            return
        rec = {"t": time.time() - self._start_wall, "name": name}
        if payload:
            rec["payload"] = payload
        with self._lock:
            self.events.append(rec)
            if len(self.events) > 10_000:
                del self.events[: len(self.events) - 10_000]

    # ------------------------------------------------------------------
    # Reset / snapshot / dump
    # ------------------------------------------------------------------

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
        """Return summary statistics for every recorded metric.

        Timer units are ms. Gauge units depend on the call site.
        Includes the raw samples too — JSON is small enough and lets
        callers compute new percentiles later.
        """
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
        """Snapshot without per-sample arrays — for sidecar embedding."""
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
        """Write the full snapshot as JSON. Returns the path written, or None
        if profiling is disabled.

        ``role`` is a short tag baked into the filename (``driver`` / ``worker``
        / ``proc``) so multi-process runs produce identifiable dumps.
        """
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


# Singleton — every wrap call sites the same instance.
PROF = Profiler()


# ---------------------------------------------------------------------------
# atexit dump — runs while the import system is still alive, unlike __del__
# ---------------------------------------------------------------------------

# Best-effort identification of "is this the driver or a worker subprocess?"
# Workers spawn with VLLM_DP_RANK / RANK / LOCAL_RANK set; the driver doesn't.
def _detect_role() -> str:
    for key in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK", "PMI_RANK"):
        v = os.environ.get(key)
        if v is not None and v != "":
            return f"worker-r{v}"
    # vLLM v1 sometimes uses VLLM_WORKER_MULTIPROC_METHOD without populating
    # a rank var; fall back to checking the main module name.
    main = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    if "vllm" in main.lower() and "engine" in main.lower():
        return "worker"
    return "driver"


_ROLE = _detect_role()


def _atexit_dump() -> None:
    """Called once per process during normal interpreter exit.

    Safe to invoke even when profiling is disabled — dump() short-circuits.
    Writes a sentinel line to stderr on failure so debugging doesn't depend
    on a silent JSON-missing symptom.
    """
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


# ---------------------------------------------------------------------------
# Background memory sampler (NVML GPU + driver-process RSS)
# ---------------------------------------------------------------------------


class MemorySampler:
    """Background daemon thread that samples NVML GPU memory + process RSS.

    Activated by MIA_PROFILE_MEM=1 (and PROFILE=1). Updates two
    gauges on PROF: ``mem.gpu_mb`` and ``mem.host_rss_mb``.

    DEVICE: under tensor parallelism a worker process sees every GPU of the node but works on
    ``cuda:r``. The sampler follows the device THIS process allocates on (``_own_cuda_device``) --
    it used to read ``cuda:0``'s allocator stats (zeros for rank r >= 1) and NVML index 0 (another
    rank's GPU). At TP=1 that device is ``gpu_index`` (0), so TP=1 readings are unchanged.
    """

    def __init__(self, interval_s: float = 0.05, gpu_index: int = 0) -> None:
        self.interval = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_handle = None
        self._psutil_proc = None
        self._torch = None  # lazy import; None until first successful sample
        self._cuda_dev: Optional[int] = None  # this process's own CUDA device, once known

    def _try_init(self) -> bool:
        try:
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
        """The CUDA device this PROCESS works on: the only one its caching allocator has reserved
        memory on (a vLLM worker allocates on its own ``cuda:r`` only). None until then, or when the
        answer is ambiguous; sticky once found. Read from allocator stats (no CUDA runtime call).

        Once found, this sampler thread selects that device before its first CUDA call (the rule in
        ``mia/graph/thread_device.py``), and re-homes the NVML handle to that GPU by UUID when it is
        not ``gpu_index`` -- or drops it, because a reading of another rank's GPU is worse than none."""
        if self._cuda_dev is not None:
            return self._cuda_dev
        t = self._torch
        try:
            if t is None or not (t.cuda.is_available() and t.cuda.is_initialized()):
                return None
            used = [d for d in range(t.cuda.device_count()) if t.cuda.memory_reserved(d) > 0]
            if len(used) != 1:
                return None
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
        import pynvml  # may be imported lazily
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
            # Per-process CUDA caching-allocator view. Skip until torch's CUDA
            # context exists (driver process imports torch eagerly; workers
            # only after they construct the model).
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


# Auto-start the memory sampler when the module is imported with PROFILE_MEM=1.
# Driver-side import will catch the driver process; the worker process imports
# happen at install_hooks time, which also covers worker-side sampling.
if _ENABLED and _MEM_SAMP:
    MEM_SAMPLER.start()
