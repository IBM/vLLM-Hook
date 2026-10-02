"""Every thread MIA starts that can reach CUDA selects its device before its first CUDA call.

The defect (gate G1, LSF 1777562): at tensor_parallel_size > 1 the QK aperture drain consumer
thread died on EVERY rank >= 1 with "CUDA-capable device(s) is/are busy or unavailable". A new
thread's current CUDA device is 0, which on rank r >= 1 belongs to another process (Worker_TP0)
under LSF's exclusive_process mode. The drain's ``with torch.cuda.stream(copy_stream)`` recorded
the thread's device-0 stream on enter and restored it on exit (``cudaSetDevice(0)``). Rank 0 never
failed, so no TP=1 run could catch it. The HS drain had the same pattern, latent only because rank 0
is the only HS drain.

These tests are GPU-free. ``_CudaEmu`` stands in for the CUDA runtime's PER-THREAD current device:
a new thread starts on device 0, and touching a device this process does not own raises the error
G1 logged. It reproduces torch's ``StreamContext`` restore-on-exit, so the pre-fix drains fail HERE
exactly as they failed on the cluster, and the fixed ones must bind first and never touch cuda:0.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import types

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py)

import torch

G1_ERROR = "CUDA error: CUDA-capable device(s) is/are busy or unavailable"


class _CudaEmu:
    """The per-thread current device of the CUDA runtime, and exclusive_process ownership.

    ``owned`` is the set of devices THIS process may touch; the rest belong to other TP ranks. Every
    touch is logged as ``(thread name, what, device)`` so a test can read the order of CUDA calls."""

    def __init__(self, owned):
        self.owned = set(owned)
        self.tls = threading.local()
        self.log: list = []
        self._lock = threading.Lock()

    def current(self) -> int:
        return getattr(self.tls, "dev", 0)          # a new thread starts on device 0

    def touch(self, what: str, dev: int) -> None:
        with self._lock:
            self.log.append((threading.current_thread().name, what, int(dev)))
        if int(dev) not in self.owned:
            raise RuntimeError(G1_ERROR)

    def set_device(self, device) -> None:
        idx = device if isinstance(device, int) else torch.device(device).index
        self.touch("set_device", idx)
        self.tls.dev = int(idx)

    def stream(self, s):
        """torch.cuda.StreamContext: remember the thread's device, switch to the stream's, and on
        exit RESTORE the remembered one (``set_stream(src_prev_stream)`` -> ``cudaSetDevice``)."""
        emu = self

        class _Ctx:
            def __enter__(self):
                self.prev = emu.current()
                emu.touch("stream_enter", s.device.index)
                emu.tls.dev = s.device.index
                return s

            def __exit__(self, *exc):
                emu.touch("stream_exit_restore", self.prev)
                emu.tls.dev = self.prev
                return False

        return _Ctx()

    def calls_on(self, thread_name: str) -> list:
        return [(what, dev) for name, what, dev in self.log if name == thread_name]


class _FakeStream:
    def __init__(self, emu, dev):
        self.emu, self.device = emu, torch.device("cuda", dev)

    def wait_event(self, ev):
        self.emu.touch("stream.wait_event", self.device.index)

    def synchronize(self):
        self.emu.touch("stream.synchronize", self.device.index)


class _FakeEvent:
    def __init__(self, emu, dev):
        self.emu, self.dev = emu, dev

    def record(self, stream=None):
        self.emu.touch("event.record", self.dev)

    def synchronize(self):
        self.emu.touch("event.synchronize", self.dev)


@pytest.fixture
def rank3(monkeypatch):
    """This process is TP rank 3 of 4 under exclusive_process: it owns cuda:3 and nothing else."""
    emu = _CudaEmu(owned={3})
    monkeypatch.setattr(torch.cuda, "set_device", emu.set_device)
    monkeypatch.setattr(torch.cuda, "stream", emu.stream)
    monkeypatch.setattr(torch.Tensor, "record_stream",
                        lambda self, s: emu.touch("record_stream", s.device.index), raising=False)
    return emu


def _arm_cuda_handles(drain, emu, dev=3):
    """Give a CPU-built drain the CUDA handles its constructor makes on a GPU: a copy stream on
    cuda:<dev> and the K-deep copy events. ``_is_cuda`` stays False only so ``_pinned_buf`` does not
    ask a GPU-less box for pinned memory; every other line of the CUDA copy path runs.

    The default write path (``MIA_APERTURE_WRITE_MODE=auto``) also has one completion event per
    layer and a writer-thread pool whose threads bind the copy stream's device: the pool built on
    the CPU box bound nothing, so it is rebuilt here on cuda:<dev>, as the constructor builds it on
    a GPU."""
    drain._stream = _FakeStream(emu, dev)
    drain._copy_events = [_FakeEvent(emu, dev) for _ in range(drain._aperture_depth)]
    wp = getattr(drain, "_wp", None)
    if wp is not None:
        drain._layer_events = [_FakeEvent(emu, dev) for _ in drain.layers]
        name = wp.pool._threads[0].name.rsplit("-", 1)[0]
        n = wp.pool.n_threads
        wp.pool.close()
        wp.pool = None
        wp.start_pool(n, torch.device("cuda", dev), name)


def _assert_writers_bound(drain, emu, prefix, dev):
    """Every writer thread of the default write path selected ``dev`` before anything else and
    never touched another device."""
    n = drain._wp.threads
    assert n >= 1
    for i in range(n):
        calls = emu.calls_on(f"{prefix}-{i}")
        assert calls and calls[0] == ("set_device", dev), (prefix, i, calls)
        assert all(d == dev for _, d in calls), calls


# --------------------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------------------

def test_bind_is_a_noop_off_cuda_and_refuses_an_ambiguous_device(rank3):
    from mia.graph.thread_device import bind_thread_to_device

    assert bind_thread_to_device(None) is None
    assert bind_thread_to_device("cpu") is None
    with pytest.raises(ValueError, match="without an index"):
        bind_thread_to_device("cuda")
    assert rank3.log == []                         # none of the above touched CUDA
    assert bind_thread_to_device(torch.device("cuda", 3)) == torch.device("cuda", 3)
    assert rank3.log[-1][1:] == ("set_device", 3)
    with pytest.raises(RuntimeError, match="busy or unavailable"):
        bind_thread_to_device("cuda:0")            # another rank's GPU: loud, never silent


def test_creator_device_never_initializes_cuda(monkeypatch):
    from mia.graph import thread_device

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device",
                        lambda: pytest.fail("current_device() would initialize CUDA"))
    assert thread_device.creator_cuda_device() is None
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert thread_device.creator_cuda_device() == torch.device("cuda", 2)


# --------------------------------------------------------------------------------------
# The two aperture drain consumer threads
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("write_mode", ["auto", "legacy"])
def test_qk_drain_thread_binds_its_stream_device_before_any_cuda_call(tmp_path, rank3, monkeypatch,
                                                                      write_mode):
    """The G1 failure, reproduced: rank 3 of TP4. Before the fix this drain's consumer died on the
    stream-context exit restoring cuda:0, and stop() raised 'consumer thread failed'. Both write
    paths: the default one's writer threads must bind cuda:3 first too."""
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", write_mode)
    from mia.graph.aperture_drain_qk import OffLoopQKApertureDrain
    from mia.graph.aperture_metadata import QKReqCaptureRecord
    from mia.graph.aperture_reader import load_multilayer_qk_aperture_artifact
    from mia.graph.capture_aperture import CaptureAperture

    n, qw, kw = 5, 16, 8
    ap = CaptureAperture(row_bytes=kw * 2, n_slots=16, device="cpu", dtype=torch.bfloat16,
                         row_shape=(kw,))
    layers = [(L, torch.zeros(17, qw, dtype=torch.bfloat16), torch.zeros(17, kw, dtype=torch.bfloat16))
              for L in range(2)]
    header = {"dtype": "bfloat16", "q_row_shape": [qw], "k_row_shape": [kw], "q_dim": qw,
              "k_dim": kw, "hookq_mode": "all_tokens"}
    run_dir = str(tmp_path / "tp_rank_3")
    drain = OffLoopQKApertureDrain(ap, layers, run_dir, header)
    _arm_cuda_handles(drain, rank3)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(n, qw, generator=g).to(torch.bfloat16)
    k = torch.randn(n, kw, generator=g).to(torch.bfloat16)
    start = ap.reserve(n)
    for L, qb, kb in layers:
        qb[start:start + n] = q * (L + 1)
        kb[start:start + n] = k * (L + 1)
    drain.start()
    drain.enqueue([QKReqCaptureRecord(req_id="r0", k_start=start, k_rows=n, q_start=start,
                                      q_rows=n, prefix_end=n, num_computed=0, layers=[0, 1])],
                  start, n, event=_FakeEvent(rank3, 3))
    drain.stop()                                   # re-raises a consumer-thread death
    drain.close()

    assert drain.error is None
    calls = rank3.calls_on("mia-qk-aperture-drain")
    assert calls[0] == ("set_device", 3), calls[:3]
    assert all(dev == 3 for _, dev in calls), calls   # never touched another rank's GPU
    assert ("stream_exit_restore", 3) in calls
    if write_mode != "legacy":
        _assert_writers_bound(drain, rank3, "mia-qk-aperture-write", 3)
    art = load_multilayer_qk_aperture_artifact(run_dir)
    assert torch.equal(art["r0"][1]["q"], q * 2) and torch.equal(art["r0"][1]["k_full"], k * 2)


@pytest.mark.parametrize("write_mode", ["auto", "legacy"])
def test_hs_drain_thread_binds_its_stream_device_before_any_cuda_call(tmp_path, rank3, monkeypatch,
                                                                      write_mode):
    """Same pattern in the HS drain: latent at TP>1 only because rank 0 is its only consumer, live
    under MIA_HS_CAPTURE_ALL_RANKS=1 and on any rank a future change makes capture."""
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", write_mode)
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.aperture_metadata import ReqCaptureRecord
    from mia.graph.aperture_reader import load_multilayer_aperture_artifact
    from mia.graph.capture_aperture import CaptureAperture

    n, hidden = 4, 16
    ap = CaptureAperture(row_bytes=hidden * 2, n_slots=16, device="cpu", dtype=torch.bfloat16,
                         row_shape=(hidden,))
    layers = [(L, torch.zeros(17, hidden, dtype=torch.bfloat16)) for L in (1, 2)]
    header = {"dtype": "bfloat16", "row_shape": [hidden], "hidden": hidden}
    run_dir = str(tmp_path / "tp_rank_3")
    drain = OffLoopApertureDrain(ap, layers, run_dir, header)
    _arm_cuda_handles(drain, rank3)
    h = torch.randn(n, hidden, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    start = ap.reserve(n)
    for L, buf in layers:
        buf[start:start + n] = h * L
    drain.start()
    drain.enqueue([ReqCaptureRecord(req_id="r0", logical_start=start, n_rows=n,
                                    hs_mode="all_tokens", layers=[1, 2])],
                  start, n, event=_FakeEvent(rank3, 3))
    drain.stop()
    drain.close()

    assert drain.error is None
    calls = rank3.calls_on("mia-hs-aperture-drain")
    assert calls[0] == ("set_device", 3), calls[:3]
    assert all(dev == 3 for _, dev in calls), calls
    if write_mode != "legacy":
        _assert_writers_bound(drain, rank3, "mia-hs-aperture-write", 3)
    art = load_multilayer_aperture_artifact(run_dir)
    assert torch.equal(art["r0"][2], h * 2)


@pytest.mark.parametrize("which", ["hs", "qk"])
def test_a_drain_that_cannot_bind_its_device_fails_loud(tmp_path, rank3, which):
    """If the bind itself fails, the consumer must die LOUD (error recorded, stop() raises) -- the
    engine's reserve-block liveness check then raises ApertureBackpressureError instead of the
    capture silently stalling."""
    from mia.graph.capture_aperture import CaptureAperture

    ap = CaptureAperture(row_bytes=16, n_slots=8, device="cpu", dtype=torch.bfloat16, row_shape=(8,))
    if which == "hs":
        from mia.graph.aperture_drain_hs import OffLoopApertureDrain as D
        drain = D(ap, [(1, torch.zeros(9, 8, dtype=torch.bfloat16))], str(tmp_path),
                  {"dtype": "bfloat16", "row_shape": [8], "hidden": 8})
    else:
        from mia.graph.aperture_drain_qk import OffLoopQKApertureDrain as D
        drain = D(ap, [(0, torch.zeros(9, 8, dtype=torch.bfloat16),
                        torch.zeros(9, 8, dtype=torch.bfloat16))], str(tmp_path),
                  {"dtype": "bfloat16", "q_row_shape": [8], "k_row_shape": [8]})
    _arm_cuda_handles(drain, rank3, dev=0)          # a stream on a device this rank does not own
    drain.start()
    drain._thread.join(timeout=10)
    assert not drain.is_alive()
    assert drain.error is not None and G1_ERROR in str(drain.error)
    with pytest.raises(RuntimeError, match="consumer thread failed"):
        drain.stop()


# --------------------------------------------------------------------------------------
# The writer process's feeder thread
# --------------------------------------------------------------------------------------

def test_writer_feeder_thread_binds_its_creators_device_first(rank3):
    """The feeder packs host tensors and may drop the last reference to a pinned (page-backed)
    buffer, which runs the CUDA host allocator's release path on the feeder thread. It therefore
    binds the device of the thread that built the WriterProcess, before anything else."""
    from mia.graph.writer_process import WriterProcess

    wp = object.__new__(WriterProcess)             # the thread body alone -- no child spawned
    wp._device = torch.device("cuda", 3)
    wp._inq = queue.Queue()
    wp._inq.put(None)                              # stop at once
    t = threading.Thread(target=wp._feed, name="mia-writer-feeder")
    t.start()
    t.join(timeout=10)
    assert rank3.calls_on("mia-writer-feeder") == [("set_device", 3)]


def test_writer_captures_the_constructing_threads_device(monkeypatch):
    from mia.graph import writer_process

    monkeypatch.setattr(writer_process, "creator_cuda_device", lambda: torch.device("cuda", 2))
    seen = {}

    class _Stop(Exception):
        pass

    def fake_get_context(method):                  # stop right after the device is captured
        seen["device"] = getattr(wp, "_device", "unset")
        raise _Stop

    import torch.multiprocessing as tmp
    monkeypatch.setattr(tmp, "set_sharing_strategy", lambda *_: None)   # no global side effect
    monkeypatch.setattr(tmp, "get_context", fake_get_context)
    wp = object.__new__(writer_process.WriterProcess)
    with pytest.raises(_Stop):
        writer_process.WriterProcess.__init__(wp)
    assert seen["device"] == torch.device("cuda", 2)


# --------------------------------------------------------------------------------------
# The profiler's memory sampler thread
# --------------------------------------------------------------------------------------

def _fake_pynvml(handles_by_uuid):
    m = types.ModuleType("pynvml")
    m.nvmlInit = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: f"nvml-index-{i}"
    m.nvmlDeviceGetHandleByUUID = lambda u: handles_by_uuid[u]
    used = {"nvml-index-0": 100 * 2 ** 20, "nvml-gpu-2": 222 * 2 ** 20}
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=used[h])
    return m


def _run_sampler_once(monkeypatch, reserved: dict, allocated: dict, emu):
    from mia import _profiler

    monkeypatch.setattr(_profiler, "_ENABLED", True)
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml({"GPU-2222": "nvml-gpu-2"}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: reserved.get(d, 0))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: allocated.get(d, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda d: types.SimpleNamespace(uuid="2222" if d == 2 else f"{d}{d}{d}{d}"))
    for k in ("mem.cuda_alloc_mb", "mem.gpu_mb"):
        _profiler.PROF.gauges.pop(k, None)
    s = _profiler.MemorySampler(interval_s=0.005)
    assert s._try_init()
    t = threading.Thread(target=s._loop, name="mia-mem-sampler")
    t.start()
    deadline = time.monotonic() + 10
    while not _profiler.PROF.gauges.get("mem.cuda_alloc_mb") and time.monotonic() < deadline:
        time.sleep(0.005)
    s._stop.set()
    t.join(timeout=10)
    out = {k: _profiler.PROF.gauges.pop(k, [None])[-1] for k in ("mem.cuda_alloc_mb", "mem.gpu_mb")}
    return out, emu.calls_on("mia-mem-sampler")


def test_memory_sampler_reads_its_own_rank_not_cuda0(monkeypatch):
    """Rank 2 of TP4: the sampler used to read cuda:0's allocator stats (zeros for this process)
    and NVML index 0 (another rank's GPU). It must follow the device this process allocates on."""
    emu = _CudaEmu(owned={2})
    monkeypatch.setattr(torch.cuda, "set_device", emu.set_device)
    got, calls = _run_sampler_once(monkeypatch, reserved={2: 8 * 2 ** 30},
                                   allocated={2: 5 * 2 ** 20}, emu=emu)
    assert got["mem.cuda_alloc_mb"] == 5.0
    assert got["mem.gpu_mb"] == 222.0              # NVML handle re-homed to cuda:2's UUID
    assert calls and calls[0] == ("set_device", 2)
    assert all(dev == 2 for _, dev in calls)


def test_memory_sampler_at_tp1_is_unchanged(monkeypatch):
    """TP=1 (EngineCore on cuda:0): the same device and the same NVML index as before."""
    emu = _CudaEmu(owned={0})
    monkeypatch.setattr(torch.cuda, "set_device", emu.set_device)
    got, _ = _run_sampler_once(monkeypatch, reserved={0: 8 * 2 ** 30},
                               allocated={0: 7 * 2 ** 20}, emu=emu)
    assert got["mem.cuda_alloc_mb"] == 7.0
    assert got["mem.gpu_mb"] == 100.0              # NVML index 0, as before


def test_server_analyze_thread_backend_binds_its_creators_device_first(monkeypatch, rank3):
    """Not wired into any worker yet, but its thread backend runs arbitrary analyzer code: same
    rule as every other MIA thread."""
    from mia.graph import server_analyze_process as sap

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    p = sap.ServerAnalyzeProcess(analyze_fn=lambda *a: "ok")      # injected fn -> thread backend
    try:
        p.submit("r0", "inflight", {"probes": {}}, "any")
        assert p.wait("r0", timeout=10)
    finally:
        p.close()
    assert rank3.calls_on("mia-server-analyze-worker")[0] == ("set_device", 3)
