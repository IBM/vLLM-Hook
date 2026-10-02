"""The capture-aperture drains' write path (``MIA_APERTURE_WRITE_MODE``): fast, and byte-identical.

The 70B HS drain topped out at 1.84 GB/s because, per step and per layer, ONE thread materialised
the rows as a ``bytes`` copy (``_raw_bytes``) and then ``open``/``write``/``close``d the layer's file,
and the sidecar grew one ``LayerEntry`` object per (request, layer, step). The default (``auto``)
path keeps every raw file open, writes zero-copy (O_DIRECT where the rows align with the detected
block size), overlaps per-layer D2H with a writer pool, and keeps the sidecar as arrays.

Everything here is GPU-free: the drains run their CPU copy path, which fills the same aligned host
buffers and drives the same sinks, writer pool and sidecar log as the CUDA path. The central check
drives identical synthetic multi-step captures -- several requests per step, per-request layer
subsets (compacting selective drains), a layer that is not installed, zero-row records, aperture
wrap-around, a backlog of queued steps and a one-row final step -- through ``legacy`` (the old path,
verbatim) and through the new modes, and requires every raw file and the sidecar to be the same
bytes.
"""
from __future__ import annotations

import errno
import fcntl
import filecmp
import gc
import os
import random
import threading
import time

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py)

import torch

HS_META = "hs_aperture_meta.jsonl"
QK_META = "qk_aperture_meta.jsonl"


# --------------------------------------------------------------------------------------------
# Synthetic captures
# --------------------------------------------------------------------------------------------

def _hs_schedule(seed: int, n_steps: int, layers, max_rows: int = 5, uninstalled: bool = True):
    """Per step: ``[(req_id, n_rows, layers, hs_mode)]``. Mixes all-layer requests (degenerate full
    steps), per-request subsets (compacting steps), a record naming an uninstalled layer (the
    sidecar keeps such entries; the reader cannot load them, having no file), and zero-row
    records; ends with a one-row step. Rows per step stay <= 3 * max_rows."""
    rng = random.Random(seed)
    steps = []
    for k in range(n_steps):
        reqs = []
        for r in range(rng.randint(1, 3)):
            n = rng.randint(1, max_rows)
            u = rng.random()
            if u < 0.4:
                lays = list(layers)
            elif u < 0.85:
                lays = sorted(rng.sample(list(layers), rng.randint(1, len(layers))))
            elif uninstalled:
                lays = [layers[0], 99]              # 99 is not installed
            else:
                lays = [layers[-1]]
            reqs.append((f"req{r}-{k // 4}", n, lays, rng.choice(["all_tokens", "last_token"])))
        if rng.random() < 0.15:
            reqs.insert(rng.randint(0, len(reqs)), ("zero", 0, [layers[-1]], "all_tokens"))
        steps.append(reqs)
    steps.append([("tail", 1, list(layers), "last_token")])
    return steps


def _reserve(ap, n, drain=None, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        s = ap.reserve(n)
        if s is not None:
            return s
        if drain is not None and getattr(drain, "error", None) is not None:
            raise RuntimeError("drain died") from drain.error
        if time.monotonic() > deadline:
            raise TimeoutError("aperture never freed rows")
        time.sleep(0.0005)


def _fill(bufs, ap, start, n, g):
    for buf in bufs:
        rows = torch.randn(n, buf.shape[1], generator=g).to(buf.dtype)
        for j in range(n):
            buf[(start + j) % ap.n_slots] = rows[j]


def _run_hs(run_dir, mode, monkeypatch, *, hidden=16, layers=(1, 2, 3), n_slots=16, steps=24,
            seed=7, sync=False, close_without_stop=False, selective=None, extra_env=None,
            uninstalled=True):
    """Drive one synthetic HS capture through a drain in ``mode``; return the drain (closed)."""
    from mia.graph.aperture_drain_hs import MultiLayerApertureDrain, OffLoopApertureDrain
    from mia.graph.aperture_metadata import ReqCaptureRecord
    from mia.graph.capture_aperture import CaptureAperture

    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", mode)
    if selective is not None:
        monkeypatch.setenv("MIA_DRAIN_SELECTIVE", selective)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    ap = CaptureAperture(row_bytes=hidden * 2, n_slots=n_slots, device="cpu",
                         dtype=torch.bfloat16, row_shape=(hidden,))
    lay = [(L, torch.zeros(n_slots + 1, hidden, dtype=torch.bfloat16)) for L in layers]
    header = {"dtype": "bfloat16", "row_shape": [hidden], "hidden": hidden, "tp_rank": 0,
              "tp_size": 1, "num_layers": len(layers), "capture_all_ranks": False}
    cls = MultiLayerApertureDrain if sync else OffLoopApertureDrain
    drain = cls(ap, lay, str(run_dir), header)
    if mode != "legacy":
        assert drain.write_mode == mode and drain._wp is not None      # the new path is in use
    if not sync:
        drain.start()
    g = torch.Generator().manual_seed(seed)
    for reqs in _hs_schedule(seed, steps, list(layers), uninstalled=uninstalled):
        recs, start, total = [], None, 0
        for rid, n, lays, hs_mode in reqs:
            s = _reserve(ap, n, None if sync else drain)
            if start is None:
                start = s
            total += n
            _fill([b for _, b in lay], ap, s, n, g)
            recs.append(ReqCaptureRecord(req_id=rid, logical_start=s, n_rows=n, hs_mode=hs_mode,
                                         layers=list(lays)))
        if sync:
            drain.record_entries(recs)
            drain.drain_once()
        elif total > 0:
            drain.enqueue(recs, start, total)
    if not sync and not close_without_stop:
        drain.stop()
    drain.close()
    return drain


def _qk_schedule(seed: int, n_steps: int, layers, max_rows: int = 5):
    """Per step: ``[(req_id, n_rows, emit_q, layers, num_computed)]``; each request keeps one layer
    set (all layers or a subset) for its life, as a real request does."""
    rng = random.Random(seed)
    steps = []
    req_layers = {}
    for k in range(n_steps):
        reqs = []
        for r in range(rng.randint(1, 3)):
            n = rng.randint(1, max_rows)
            emit = rng.random() < 0.7
            rid = f"q{r}-{k // 4}"
            if rid not in req_layers:
                req_layers[rid] = (list(layers) if rng.random() < 0.5 else
                                   sorted(rng.sample(list(layers), rng.randint(1, len(layers)))))
            reqs.append((rid, n, emit, req_layers[rid], rng.randint(0, 3)))
        steps.append(reqs)
    steps.append([("tail", 1, True, list(layers), 0)])
    return steps


def _run_qk(run_dir, mode, monkeypatch, *, qw=256, kw=64, layers=(0, 1, 2), n_slots=16,
            steps=24, seed=11, sync=False, extra_env=None):
    from mia.graph.aperture_drain_qk import MultiLayerQKApertureDrain, OffLoopQKApertureDrain
    from mia.graph.aperture_metadata import QKReqCaptureRecord
    from mia.graph.capture_aperture import CaptureAperture

    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", mode)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    ap = CaptureAperture(row_bytes=kw * 2, n_slots=n_slots, device="cpu", dtype=torch.bfloat16,
                         row_shape=(kw,))
    lay = [(L, torch.zeros(n_slots + 1, qw, dtype=torch.bfloat16),
            torch.zeros(n_slots + 1, kw, dtype=torch.bfloat16)) for L in layers]
    header = {"dtype": "bfloat16", "q_row_shape": [qw], "k_row_shape": [kw], "q_dim": qw,
              "k_dim": kw, "hookq_mode": "all_tokens", "tp_rank": 0, "tp_size": 1,
              "num_layers": len(layers)}
    cls = MultiLayerQKApertureDrain if sync else OffLoopQKApertureDrain
    drain = cls(ap, lay, str(run_dir), header)
    if mode != "legacy":
        assert drain.write_mode == mode and drain._wp is not None
    if not sync:
        drain.start()
    g = torch.Generator().manual_seed(seed)
    prefix = {}
    for reqs in _qk_schedule(seed, steps, list(layers)):
        recs, start, total = [], None, 0
        for rid, n, emit, lays, nc in reqs:
            s = _reserve(ap, n, None if sync else drain)
            if start is None:
                start = s
            total += n
            _fill([b for _, q, k in lay for b in (q, k)], ap, s, n, g)
            first = rid not in prefix                 # the reader refuses a cached first step
            prefix[rid] = prefix.get(rid, 0) + n
            recs.append(QKReqCaptureRecord(
                req_id=rid, k_start=s, k_rows=n, q_start=s if emit else -1,
                q_rows=n if emit else 0, prefix_end=prefix[rid] if emit else -1,
                num_computed=0 if first else nc, layers=list(lays)))
        if sync:
            drain.record_entries(recs)
            drain.drain_once()
        else:
            drain.enqueue(recs, start, total)
    if not sync:
        drain.stop()
    drain.close()
    return drain


def _assert_same_tree(a, b):
    fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
    assert fa == fb, (fa, fb)
    for f in fa:
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        assert filecmp.cmp(pa, pb, shallow=False), (
            f"{f} differs: {os.path.getsize(pa)} vs {os.path.getsize(pb)} bytes")
        assert os.stat(pa).st_mode == os.stat(pb).st_mode, f  # same permissions as the old open()


def _direct_block(path) -> int:
    from mia.graph.aperture_sink import probe_direct_io
    os.makedirs(path, exist_ok=True)
    info = probe_direct_io(str(path))
    if not info.supported:
        pytest.skip(f"no O_DIRECT under {path}: {info.reason}")
    return info.block_size


def _is_o_direct(fd) -> bool:
    return bool(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)


def _open_fds() -> set:
    return set(os.listdir("/proc/self/fd"))


# --------------------------------------------------------------------------------------------
# Byte identity against the legacy path
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["buffered", "auto"])
@pytest.mark.parametrize("selective", ["1", "0"])
def test_hs_offloop_matches_legacy_byte_for_byte(tmp_path, monkeypatch, mode, selective):
    """TP=1 layout (tp_rank_0), selective on and off. `auto` here writes buffered: 32 B rows are not
    a multiple of any O_DIRECT block."""
    legacy = _run_hs(tmp_path / "legacy" / "tp_rank_0", "legacy", monkeypatch, selective=selective)
    new = _run_hs(tmp_path / mode / "tp_rank_0", mode, monkeypatch, selective=selective)
    _assert_same_tree(tmp_path / "legacy" / "tp_rank_0", tmp_path / mode / "tp_rank_0")
    assert legacy.row_counts() == new.row_counts()
    if selective == "1":
        assert new.row_counts()["hs.drain.rows_skipped"] > 0        # compacting steps happened


@pytest.mark.parametrize("mode", ["buffered", "auto"])
def test_hs_tp1_reader_sees_the_same_capture(tmp_path, monkeypatch, mode):
    """The TP=1 reader (load_hs_aperture_tp) over both dumps: the same requests, layers, tensors."""
    from mia.graph.aperture_reader import load_hs_aperture_tp

    _run_hs(tmp_path / "legacy" / "tp_rank_0", "legacy", monkeypatch, uninstalled=False)
    _run_hs(tmp_path / mode / "tp_rank_0", mode, monkeypatch, uninstalled=False)
    _assert_same_tree(tmp_path / "legacy" / "tp_rank_0", tmp_path / mode / "tp_rank_0")
    a = load_hs_aperture_tp(str(tmp_path / "legacy"))
    b = load_hs_aperture_tp(str(tmp_path / mode))
    assert a.keys() == b.keys() and all(
        a[r].keys() == b[r].keys() and all(torch.equal(a[r][L], b[r][L]) for L in a[r]) for r in a)


def test_hs_offloop_direct_matches_legacy_and_really_uses_o_direct(tmp_path, monkeypatch):
    """Rows sized to the detected O_DIRECT block: every raw file is opened O_DIRECT and written from
    aligned host buffers, and the bytes are still the legacy bytes."""
    bs = _direct_block(tmp_path / "probe")
    hidden = max(256, bs // 2)
    seen = {}
    from mia.graph import aperture_sink

    real_write = aperture_sink.RawFileSink.write

    def spy(self, t):
        seen.setdefault(self.path, set()).add(
            (self.direct, _is_o_direct(self.fd), t.data_ptr() % self.mem_align if self.direct else 0))
        return real_write(self, t)

    monkeypatch.setattr(aperture_sink.RawFileSink, "write", spy)
    _run_hs(tmp_path / "legacy", "legacy", monkeypatch, hidden=hidden)
    d = _run_hs(tmp_path / "direct", "direct", monkeypatch, hidden=hidden)
    _assert_same_tree(tmp_path / "legacy", tmp_path / "direct")
    assert seen and all(v == {(True, True, 0)} for v in seen.values()), seen
    assert d.write_stats()["bytes_direct"] > 0 and d.write_stats()["bytes_buffered"] == 0
    assert "hs=direct" in d.write_path_summary() and f"block {bs} B" in d.write_path_summary()


@pytest.mark.parametrize("mode", ["buffered", "auto"])
def test_hs_sync_drain_matches_legacy(tmp_path, monkeypatch, mode):
    _run_hs(tmp_path / "legacy", "legacy", monkeypatch, sync=True, hidden=256)
    d = _run_hs(tmp_path / mode, mode, monkeypatch, sync=True, hidden=256)
    _assert_same_tree(tmp_path / "legacy", tmp_path / mode)
    assert "hs=buffered" in d.write_path_summary()          # the sync drain never goes O_DIRECT


@pytest.mark.parametrize("mode", ["buffered", "auto"])
@pytest.mark.parametrize("sync", [False, True])
def test_qk_matches_legacy_byte_for_byte(tmp_path, monkeypatch, mode, sync):
    from mia.graph.aperture_reader import load_multilayer_qk_aperture_artifact

    _run_qk(tmp_path / "legacy", "legacy", monkeypatch, sync=sync)
    _run_qk(tmp_path / mode, mode, monkeypatch, sync=sync)
    _assert_same_tree(tmp_path / "legacy", tmp_path / mode)
    a = load_multilayer_qk_aperture_artifact(str(tmp_path / "legacy"))
    b = load_multilayer_qk_aperture_artifact(str(tmp_path / mode))
    for r in a:
        for L in a[r]:
            assert torch.equal(a[r][L]["k_full"], b[r][L]["k_full"])
            assert torch.equal(a[r][L]["q"], b[r][L]["q"])


def test_qk_auto_decides_per_file_and_never_mixes(tmp_path, monkeypatch):
    """q rows a multiple of the block go O_DIRECT, narrow k rows (the 70B TP4 / TP8 shape) go
    buffered -- each file one way for its whole life -- and the bytes match legacy."""
    bs = _direct_block(tmp_path / "probe")
    qw, kw = max(256, bs // 2), 64                        # q row = bs-aligned, k row = 128 B
    from mia.graph import aperture_sink

    modes = {}
    real_write = aperture_sink.RawFileSink.write

    def spy(self, t):
        modes.setdefault(os.path.basename(self.path), set()).add(_is_o_direct(self.fd))
        return real_write(self, t)

    monkeypatch.setattr(aperture_sink.RawFileSink, "write", spy)
    _run_qk(tmp_path / "legacy", "legacy", monkeypatch, qw=qw, kw=kw)
    d = _run_qk(tmp_path / "auto", "auto", monkeypatch, qw=qw, kw=kw)
    _assert_same_tree(tmp_path / "legacy", tmp_path / "auto")
    assert modes and all(len(v) == 1 for v in modes.values()), modes
    assert all(v == {True} for f, v in modes.items() if f.startswith("qk_q_")), modes
    assert all(v == {False} for f, v in modes.items() if f.startswith("qk_k_")), modes
    s = d.write_path_summary()
    assert "q=direct" in s and "k=buffered" in s and f"not a multiple of the {bs} B" in s


# --------------------------------------------------------------------------------------------
# O_DIRECT detection, refusal and fallback
# --------------------------------------------------------------------------------------------

def test_probe_detects_the_block_size_rather_than_assuming_it(tmp_path, monkeypatch):
    """A device whose sysfs says 512 but which rejects anything below 4096 (a 4Kn device behind a
    stale report) is detected as 4096; a statx report of offset_align 0 means unsupported."""
    from mia.graph import aperture_sink

    monkeypatch.setattr(aperture_sink, "_statx_dio_alignment", lambda p: None)
    monkeypatch.setattr(aperture_sink, "_sysfs_logical_block_size", lambda p: 512)
    real = os.pwrite

    def pwrite_4kn(fd, data, off):
        if len(data) % 4096 or off % 4096:
            raise OSError(errno.EINVAL, "Invalid argument")
        return real(fd, data, off)

    monkeypatch.setattr(aperture_sink.os, "pwrite", pwrite_4kn)
    info = aperture_sink.probe_direct_io(str(tmp_path))
    assert info.supported and info.block_size == 4096 and info.mem_align >= 4096, info
    assert "sysfs" in info.source
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".mia_odirect_probe")]

    monkeypatch.setattr(aperture_sink, "_statx_dio_alignment", lambda p: (4, 0))
    info = aperture_sink.probe_direct_io(str(tmp_path))
    assert not info.supported and "statx" in info.reason


def _reject_o_direct(monkeypatch):
    """Emulate a file system that refuses O_DIRECT at open() (tmpfs before Linux 6.6)."""
    from mia.graph import aperture_sink
    real_open = os.open

    def fake_open(path, flags, *a, **k):
        if flags & os.O_DIRECT:
            raise OSError(errno.EINVAL, "Invalid argument", path)
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(aperture_sink.os, "open", fake_open)


def test_direct_is_refused_cleanly_where_o_direct_is_rejected(tmp_path, monkeypatch):
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.aperture_sink import ApertureWriteConfigError
    from mia.graph.capture_aperture import CaptureAperture

    _reject_o_direct(monkeypatch)
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "direct")
    ap = CaptureAperture(row_bytes=512, n_slots=8, device="cpu", dtype=torch.bfloat16,
                         row_shape=(256,))
    before = _open_fds()
    with pytest.raises(ApertureWriteConfigError, match=r"direct refused.*open\(O_DIRECT\)"):
        OffLoopApertureDrain(ap, [(1, torch.zeros(9, 256, dtype=torch.bfloat16))], str(tmp_path),
                             {"dtype": "bfloat16", "row_shape": [256], "hidden": 256})
    assert _open_fds() == before                     # nothing left open
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".mia_odirect_probe")]


def test_auto_falls_back_to_buffered_where_o_direct_is_rejected(tmp_path, monkeypatch):
    _run_hs(tmp_path / "legacy", "legacy", monkeypatch, hidden=256)
    _reject_o_direct(monkeypatch)
    d = _run_hs(tmp_path / "auto", "auto", monkeypatch, hidden=256)
    _assert_same_tree(tmp_path / "legacy", tmp_path / "auto")
    s = d.write_path_summary()
    assert "hs=buffered" in s and "O_DIRECT off" in s and "open(O_DIRECT)" in s, s
    assert d.write_stats()["bytes_direct"] == 0 and d.write_stats()["bytes_buffered"] > 0


def test_real_tmpfs_if_it_rejects_o_direct(tmp_path, monkeypatch):
    """The same refusal against a real tmpfs, where this kernel still rejects O_DIRECT (Linux < 6.6
    without the backport; RHEL 9's 5.14 accepts it, so this skips there)."""
    from mia.graph.aperture_sink import ApertureWriteConfigError, probe_direct_io

    shm = "/dev/shm"
    if not os.path.isdir(shm) or not os.access(shm, os.W_OK):
        pytest.skip("no writable /dev/shm")
    d = os.path.join(shm, f"mia_wp_test_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    try:
        if probe_direct_io(d).supported:
            pytest.skip("this kernel's tmpfs accepts O_DIRECT")
        with pytest.raises(ApertureWriteConfigError):
            _run_hs(os.path.join(d, "direct"), "direct", monkeypatch, hidden=256)
        _run_hs(os.path.join(d, "auto"), "auto", monkeypatch, hidden=256)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_direct_is_refused_when_a_file_cannot_be_aligned(tmp_path, monkeypatch):
    """QK k rows of 128 B cannot be O_DIRECT: `direct` refuses at construction, leaving nothing
    open, instead of quietly writing those files buffered."""
    from mia.graph.aperture_sink import ApertureWriteConfigError

    _direct_block(tmp_path / "probe")
    before = _open_fds()
    with pytest.raises(ApertureWriteConfigError, match=r"direct refused.*k row 128 B"):
        _run_qk(tmp_path / "d", "direct", monkeypatch, qw=256, kw=64)
    assert _open_fds() == before


def test_the_sync_drain_refuses_direct(tmp_path, monkeypatch):
    from mia.graph.aperture_sink import ApertureWriteConfigError

    with pytest.raises(ApertureWriteConfigError, match="synchronous drain"):
        _run_hs(tmp_path / "d", "direct", monkeypatch, sync=True, hidden=256)


def test_a_misaligned_write_to_an_o_direct_file_raises_instead_of_falling_back(tmp_path):
    from mia.graph.aperture_sink import ApertureWriteError, RawFileSink, alloc_host_rows

    bs = _direct_block(tmp_path)
    sink = RawFileSink(str(tmp_path / "x.raw"), direct=True, block_size=bs, mem_align=4096)
    try:
        good = alloc_host_rows(2, bs, torch.uint8, pinned=False, align=4096)
        assert sink.write(good) == 2 * bs and _is_o_direct(sink.fd)
        with pytest.raises(ApertureWriteError, match="alignment violated"):
            sink.write(good.view(-1)[1:bs + 1])      # misaligned address and length
        assert sink.offset == 2 * bs
    finally:
        sink.close()


def test_aligned_host_rows(tmp_path):
    from mia.graph.aperture_sink import alloc_host_rows, tensor_bytes_view

    for n, w in ((1, 16), (3, 256), (7, 4096)):
        t = alloc_host_rows(n, w, torch.bfloat16, pinned=False, align=4096)
        assert t.shape == (n, w) and t.dtype == torch.bfloat16 and t.data_ptr() % 4096 == 0
        t.copy_(torch.randn(n, w).to(torch.bfloat16))
        assert bytes(tensor_bytes_view(t)) == t.view(torch.uint16).numpy().tobytes()


# --------------------------------------------------------------------------------------------
# Loud failures and release semantics
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["hs", "qk"])
def test_a_writer_thread_error_kills_the_drain_loudly_and_frees_nothing(tmp_path, monkeypatch, kind):
    """ENOSPC on one file's write, on the 3rd step: the consumer dies with that error, the aperture
    cursor stays at the failing step's start (its rows are never released as if written), stop()
    raises, and the sidecar names only the steps that were fully written."""
    from mia.graph import aperture_sink
    from mia.graph.aperture_metadata import ReqCaptureRecord, QKReqCaptureRecord
    from mia.graph.capture_aperture import CaptureAperture

    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "buffered")
    calls = {"n": 0}
    real_write = aperture_sink.RawFileSink.write

    def failing(self, t):
        if os.path.basename(self.path).startswith(("hs_layer_2", "qk_k_layer_1")):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError(errno.ENOSPC, "No space left on device", self.path)
        return real_write(self, t)

    monkeypatch.setattr(aperture_sink.RawFileSink, "write", failing)
    ap = CaptureAperture(row_bytes=32, n_slots=64, device="cpu", dtype=torch.bfloat16,
                         row_shape=(16,))
    if kind == "hs":
        from mia.graph.aperture_drain_hs import OffLoopApertureDrain as D
        layers = [(L, torch.zeros(65, 16, dtype=torch.bfloat16)) for L in (1, 2, 3)]
        drain = D(ap, layers, str(tmp_path), {"dtype": "bfloat16", "row_shape": [16], "hidden": 16})
        rec = lambda s, n: ReqCaptureRecord("r", s, n, "all_tokens", [1, 2, 3])  # noqa: E731
    else:
        from mia.graph.aperture_drain_qk import OffLoopQKApertureDrain as D
        layers = [(L, torch.zeros(65, 16, dtype=torch.bfloat16),
                   torch.zeros(65, 16, dtype=torch.bfloat16)) for L in (0, 1)]
        drain = D(ap, layers, str(tmp_path), {"dtype": "bfloat16", "q_row_shape": [16],
                                             "k_row_shape": [16], "q_dim": 16, "k_dim": 16,
                                             "hookq_mode": "all_tokens"})
        rec = lambda s, n: QKReqCaptureRecord("r", s, n, s, n, s + n, 0, [0, 1])  # noqa: E731
    drain.start()
    starts = []
    for _ in range(5):
        s = ap.reserve(4)
        starts.append(s)
        drain.enqueue([rec(s, 4)], s, 4)
    drain._thread.join(timeout=30)
    assert not drain.is_alive()
    assert isinstance(drain.error, OSError) and drain.error.errno == errno.ENOSPC
    assert ap._drain == starts[2]                    # steps 0, 1 released; step 2 NOT
    with pytest.raises(RuntimeError, match="consumer thread failed") as ei:
        drain.stop()
    assert isinstance(ei.value.__cause__, OSError)
    drain.close()
    meta = (tmp_path / (HS_META if kind == "hs" else QK_META)).read_text().splitlines()
    assert {int(__import__("json").loads(l)["s"]) for l in meta[1:]} == {0, 1}


def test_join_writes_waits_for_every_write_then_names_the_failures():
    from concurrent.futures import Future

    from mia.graph.aperture_sink import ApertureWriteError, WriterPool, join_writes

    pool = WriterPool(2, None, "t-writer")
    gate = threading.Event()
    done = []

    def slow():
        gate.wait(10)
        done.append(1)
        return 1

    def boom():
        raise OSError(errno.EIO, "I/O error")

    futs = [pool.submit(boom), pool.submit(slow), pool.submit(boom)]
    threading.Timer(0.2, gate.set).start()
    with pytest.raises(ApertureWriteError, match="2 of 3") as ei:
        join_writes(futs)
    assert done == [1]                                # the slow write finished before the raise
    assert isinstance(ei.value.__cause__, OSError)
    pool.close()
    assert isinstance(join_writes([]), list)
    f = Future()
    f.set_result((1, 0.0, "buffered"))
    assert join_writes([f]) == [(1, 0.0, "buffered")]


def test_a_writer_that_cannot_bind_its_device_fails_every_write(monkeypatch):
    from mia.graph import aperture_sink

    def bad_bind(dev):
        raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")

    monkeypatch.setattr(aperture_sink, "bind_thread_to_device", bad_bind)
    pool = aperture_sink.WriterPool(2, torch.device("cuda", 3), "t-bind")
    futs = [pool.submit(lambda: 1) for _ in range(4)]
    with pytest.raises(aperture_sink.ApertureWriteError, match="could not bind"):
        aperture_sink.join_writes(futs)
    assert all(f.done() for f in futs)                # never a write left hanging
    pool.close()


def test_rows_are_released_only_after_their_writes_finish(tmp_path, monkeypatch):
    from mia.graph import aperture_sink
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.aperture_metadata import ReqCaptureRecord
    from mia.graph.capture_aperture import CaptureAperture

    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "buffered")
    started, release = threading.Event(), threading.Event()
    real_write = aperture_sink.RawFileSink.write

    def gated(self, t):
        if self.path.endswith("hs_layer_3.raw"):
            started.set()
            assert release.wait(20)
        return real_write(self, t)

    monkeypatch.setattr(aperture_sink.RawFileSink, "write", gated)
    ap = CaptureAperture(row_bytes=32, n_slots=32, device="cpu", dtype=torch.bfloat16,
                         row_shape=(16,))
    layers = [(L, torch.zeros(33, 16, dtype=torch.bfloat16)) for L in (1, 2, 3)]
    drain = OffLoopApertureDrain(ap, layers, str(tmp_path),
                                 {"dtype": "bfloat16", "row_shape": [16], "hidden": 16})
    drain.start()
    s = ap.reserve(6)
    drain.enqueue([ReqCaptureRecord("r", s, 6, "all_tokens", [1, 2, 3])], s, 6)
    assert started.wait(20)
    time.sleep(0.2)
    assert ap._drain == s and ap.free_rows() == 26   # layer 3 still writing: nothing freed
    release.set()
    drain.stop()
    assert ap._drain == s + 6
    drain.close()


def test_close_without_stop_drains_the_backlog_first(tmp_path, monkeypatch):
    """The atexit backstop calls close() alone. On the new path that writes every enqueued step
    before the files close -- the artifact equals a proper stop()+close() one."""
    _run_hs(tmp_path / "legacy", "legacy", monkeypatch)
    d = _run_hs(tmp_path / "auto", "auto", monkeypatch, close_without_stop=True)
    _assert_same_tree(tmp_path / "legacy", tmp_path / "auto")
    assert d.error is None and not d.is_alive()


def test_files_stay_open_for_the_run_and_close_once(tmp_path, monkeypatch):
    """No raw file is opened after construction (the old path opened each one per layer per step),
    no bytes copy is made (`_raw_bytes` is never called), every fd is closed by close(), and a
    second close() is a no-op."""
    import builtins

    from mia.graph import aperture_drain_hs, aperture_sink

    before = _open_fds()
    real_open, real_os_open = builtins.open, os.open
    armed = {"on": False}

    def guard_open(path, *a, **k):
        if armed["on"] and str(path).endswith(".raw"):
            raise AssertionError(f"raw file re-opened during the run: {path}")
        return real_open(path, *a, **k)

    def guard_os_open(path, *a, **k):
        if armed["on"] and str(path).endswith(".raw"):
            raise AssertionError(f"raw file re-opened during the run: {path}")
        return real_os_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", guard_open)
    monkeypatch.setattr(aperture_sink.os, "open", guard_os_open)
    monkeypatch.setattr(aperture_drain_hs, "_raw_bytes",
                        lambda t: pytest.fail("the new path must not materialise bytes"))
    real_init = aperture_drain_hs.OffLoopApertureDrain.start

    def arm_on_start(self):
        armed["on"] = True
        return real_init(self)

    monkeypatch.setattr(aperture_drain_hs.OffLoopApertureDrain, "start", arm_on_start)
    d = _run_hs(tmp_path / "auto", "auto", monkeypatch)
    armed["on"] = False
    assert all(s.fd is None for s in d._wp.sinks.values())
    assert not any(t.is_alive() for t in d._wp.pool._threads)
    d.close()
    assert _open_fds() == before


def test_a_step_after_close_is_a_loud_error(tmp_path, monkeypatch):
    from mia.graph.aperture_drain_hs import MultiLayerApertureDrain
    from mia.graph.aperture_metadata import ReqCaptureRecord
    from mia.graph.aperture_sink import ApertureWriteError
    from mia.graph.capture_aperture import CaptureAperture

    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "buffered")
    ap = CaptureAperture(row_bytes=32, n_slots=8, device="cpu", dtype=torch.bfloat16,
                         row_shape=(16,))
    d = MultiLayerApertureDrain(ap, [(1, torch.zeros(9, 16, dtype=torch.bfloat16))],
                                str(tmp_path), {"dtype": "bfloat16", "row_shape": [16]})
    d.close()
    s = ap.reserve(2)
    d.record_entries([ReqCaptureRecord("r", s, 2, "all_tokens", [1])])
    with pytest.raises(ApertureWriteError, match="after close"):
        d.drain_once()
    assert ap._drain == 0


# --------------------------------------------------------------------------------------------
# Bookkeeping: arrays, not one object per (request, layer, step)
# --------------------------------------------------------------------------------------------

def test_the_new_path_builds_no_layer_entries(tmp_path, monkeypatch):
    """The sidecar is kept as per-step arrays: no expand_records fan-out and no LayerEntry objects
    accumulate over the run, yet the sidecar is the legacy file."""
    from mia.graph import aperture_drain_hs, aperture_metadata

    _run_hs(tmp_path / "legacy", "legacy", monkeypatch, steps=40)
    monkeypatch.setattr(aperture_drain_hs, "expand_records",
                        lambda r: pytest.fail("expand_records on the new drain path"))
    gc.collect()
    n0 = sum(type(o) is aperture_metadata.LayerEntry for o in gc.get_objects())
    d = _run_hs(tmp_path / "auto", "auto", monkeypatch, steps=40)
    gc.collect()
    n1 = sum(type(o) is aperture_metadata.LayerEntry for o in gc.get_objects())
    assert n1 == n0, (n0, n1)
    assert d._sidecar.has_entries() and all(b[0] == "rows" for b in d._sidecar.blocks)
    _assert_same_tree(tmp_path / "legacy", tmp_path / "auto")


def test_sidecar_logs_equal_the_legacy_writers_on_awkward_records(tmp_path):
    """Direct unit check of the two logs against write_sidecar / write_qk_sidecar, including the
    records the array path must not guess at: flat already-expanded entries, a float field, a
    req_id that is an int next to its string twin, empty layer lists and uninstalled layers."""
    from mia.graph.aperture_drain_hs import LayerCopyPlan, _stamp_file_row
    from mia.graph.aperture_metadata import (
        HsSidecarLog, LayerEntry, QkSidecarLog, QKReqCaptureRecord, QKStepEntry,
        ReqCaptureRecord, StepMeta, expand_qk_records, expand_records, write_qk_sidecar,
        write_sidecar)
    from mia.graph.capture_aperture import CaptureAperture
    import numpy as np

    header = {"dtype": "bfloat16", "row_shape": (4,), "hidden": 4}
    ap = CaptureAperture(row_bytes=8, n_slots=64, device="cpu")
    installed = [1, 2, 3]
    steps = [
        ([ReqCaptureRecord("a", 0, 3, "all_tokens", [1, 2, 3]),
          ReqCaptureRecord(1, 3, 2, "last_token", [2, 99])], 0, 3, None),
        ([ReqCaptureRecord("1", 5, 1, "all_tokens", [3])], 5, np.array([4, 5, -1]),
         {3: LayerCopyPlan.from_ranges([(5, 1)], ap)}),
        ([ReqCaptureRecord("e", 6, 2, "all_tokens", [])], 6, 9, None),        # no entries
        ([LayerEntry("flat", 2, 7, 2, "all_tokens")], 7, 9, None),            # flat entry
        ([ReqCaptureRecord("f", 9.0, 1, "all_tokens", [1])], 9, 11, None),    # float field
        ([ReqCaptureRecord("z", 10, 0, "all_tokens", [1, 2])], 10, 12, None),
    ]
    log = HsSidecarLog(installed, _stamp_file_row)
    legacy = []
    for recs, start, cur, plans in steps:
        log.commit(log.prepare(recs, start, cur, plans))
        entries = expand_records([LayerEntry(e.req_id, e.layer, e.logical_start, e.n_rows,
                                             e.hs_mode) if isinstance(e, LayerEntry) else e
                                  for e in recs])
        if entries:
            if isinstance(cur, int):
                cd = {ln: cur for ln in installed}
            else:
                cd = {ln: int(cur[i]) for i, ln in enumerate(installed) if cur[i] >= 0}
            _stamp_file_row(entries, cd, start, plans)
            legacy.append(StepMeta(entries))
    write_sidecar(str(tmp_path / "legacy.jsonl"), legacy, header)
    log.write(str(tmp_path / "log.jsonl"), header)
    assert (tmp_path / "legacy.jsonl").read_bytes() == (tmp_path / "log.jsonl").read_bytes()
    assert [b[0] for b in log.blocks] == ["rows", "rows", "entries", "entries", "rows"]

    qsteps = [
        [QKReqCaptureRecord("a", 0, 3, 0, 3, 3, 0, [0, 1]),
         QKReqCaptureRecord(7, 3, 2, -1, 0, -1, 1, [1])],
        [QKStepEntry("flat", 2, 5, 1, 5, 1, 6, 0)],
        [QKReqCaptureRecord("n", 6, 1, 6, 1, 4, 0, [])],
        [QKReqCaptureRecord("b", 7, 2, 7, 2, 9, 2, [2, 0, 1])],
    ]
    qlog = QkSidecarLog()
    qlegacy = []
    for recs in qsteps:
        qlog.commit(qlog.prepare(recs))
        e = expand_qk_records(recs)
        if e:
            qlegacy.append(StepMeta(e))
    qh = {"dtype": "bfloat16", "q_row_shape": [8], "k_row_shape": (2,)}
    write_qk_sidecar(str(tmp_path / "ql.jsonl"), qlegacy, qh)
    qlog.write(str(tmp_path / "qn.jsonl"), qh)
    assert (tmp_path / "ql.jsonl").read_bytes() == (tmp_path / "qn.jsonl").read_bytes()


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("env,value,match", [
    ("MIA_APERTURE_WRITE_MODE", "fast", "is not one of"),
    ("MIA_APERTURE_WRITE_MODE", "o_direct", "is not one of"),
    ("MIA_APERTURE_WRITE_THREADS", "0", r"integer in \[1, 64\]"),
    ("MIA_APERTURE_WRITE_THREADS", "two", r"integer in \[1, 64\]"),
    ("MIA_APERTURE_WRITE_THREADS", "65", r"integer in \[1, 64\]"),
])
def test_bad_write_env_is_refused_before_any_file_opens(tmp_path, monkeypatch, env, value, match):
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.aperture_sink import ApertureWriteConfigError
    from mia.graph.capture_aperture import CaptureAperture

    monkeypatch.setenv(env, value)
    ap = CaptureAperture(row_bytes=32, n_slots=8, device="cpu", dtype=torch.bfloat16,
                         row_shape=(16,))
    before = _open_fds()
    with pytest.raises(ApertureWriteConfigError, match=match):
        OffLoopApertureDrain(ap, [(1, torch.zeros(9, 16, dtype=torch.bfloat16))],
                             str(tmp_path / "d"), {"dtype": "bfloat16", "row_shape": [16]})
    assert _open_fds() == before
    assert not (tmp_path / "d" / "hs_layer_1.raw").exists()


@pytest.mark.parametrize("threads", ["1", "3"])
def test_writer_thread_count_is_honoured(tmp_path, monkeypatch, threads):
    d = _run_hs(tmp_path / "t", "buffered", monkeypatch,
                extra_env={"MIA_APERTURE_WRITE_THREADS": threads})
    assert d._wp.threads == int(threads) and len(d._wp.pool._threads) == int(threads)
    assert f"{threads} writer thread(s)" in d.write_path_summary()
    assert [t.name for t in d._wp.pool._threads] == [
        f"mia-hs-aperture-write-{i}" for i in range(int(threads))]


def test_mmap_sink_belongs_to_legacy_only(tmp_path, monkeypatch):
    from mia.graph.aperture_sink import ApertureWriteConfigError

    with pytest.raises(ApertureWriteConfigError, match="MIA_APERTURE_MMAP"):
        _run_hs(tmp_path / "a", "auto", monkeypatch, extra_env={"MIA_APERTURE_MMAP": "1"})
    monkeypatch.delenv("MIA_APERTURE_MMAP")
    _run_hs(tmp_path / "plain", "legacy", monkeypatch)
    d = _run_hs(tmp_path / "mmap", "legacy", monkeypatch, extra_env={"MIA_APERTURE_MMAP": "1"})
    assert "mmap" in d.write_path_summary()
    _assert_same_tree(tmp_path / "plain", tmp_path / "mmap")


def test_per_request_delivery_refuses_explicit_direct_and_reports_the_rest(tmp_path, monkeypatch):
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.aperture_sink import ApertureWriteConfigError
    from mia.graph.capture_aperture import CaptureAperture

    ap = CaptureAperture(row_bytes=32, n_slots=8, device="cpu", dtype=torch.bfloat16,
                         row_shape=(16,))
    args = (ap, [(1, torch.zeros(9, 16, dtype=torch.bfloat16))], str(tmp_path),
            {"dtype": "bfloat16", "row_shape": [16]})
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "direct")
    with pytest.raises(ApertureWriteConfigError, match="per-request delivery"):
        OffLoopApertureDrain(*args, per_request=True)
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "auto")
    d = OffLoopApertureDrain(*args, per_request=True)
    assert d._wp is None and "does not apply" in d.write_path_summary()
    d.close()


def test_legacy_summary_and_drain_holds_data(tmp_path, monkeypatch):
    """flush_aperture returns a rank dir only when its drain wrote rows: still true on the new
    path, whose sidecar is no longer a list of StepMeta."""
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.capture_aperture import CaptureAperture
    from mia.graph.tp_shard import drain_holds_data

    d = _run_hs(tmp_path / "legacy", "legacy", monkeypatch)
    assert d.write_path_summary().startswith("write mode=legacy -> hs=legacy")
    assert drain_holds_data(d)
    d2 = _run_hs(tmp_path / "auto", "auto", monkeypatch)
    assert drain_holds_data(d2) and d2.has_sidecar_entries()
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "auto")
    ap = CaptureAperture(row_bytes=32, n_slots=8, device="cpu", dtype=torch.bfloat16,
                         row_shape=(16,))
    empty = OffLoopApertureDrain(ap, [(1, torch.zeros(9, 16, dtype=torch.bfloat16))],
                                 str(tmp_path / "empty"), {"dtype": "bfloat16", "row_shape": [16]})
    empty.start()
    empty.stop()
    empty.close()
    assert os.path.getsize(tmp_path / "empty" / "hs_layer_1.raw") == 0
    assert not empty.has_sidecar_entries() and not drain_holds_data(empty)


def test_prof_splits_the_step(tmp_path, monkeypatch):
    from mia import _profiler

    monkeypatch.setattr(_profiler, "_ENABLED", True)
    _profiler.PROF.reset()
    try:
        d = _run_hs(tmp_path / "p", "buffered", monkeypatch)
        c, t = dict(_profiler.PROF.counters), dict(_profiler.PROF.timers)
    finally:
        _profiler.PROF.reset()
    stats = d.write_stats()
    raw = sum(os.path.getsize(tmp_path / "p" / f) for f in os.listdir(tmp_path / "p")
              if f.endswith(".raw"))
    assert c["aperture.hs.bytes.buffered"] == raw == stats["bytes_buffered"]
    assert c.get("aperture.hs.write_busy_us", 0) >= 0
    for name in ("step", "d2h", "write", "write_tail", "bookkeeping"):
        assert len(t[f"aperture.hs.{name}"]) == stats["steps"] > 0
