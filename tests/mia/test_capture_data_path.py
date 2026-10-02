"""The capture data path decides itself: which WRITER a request's bytes take, and which ROUTE.

Three decisions, one contract: a user sets nothing.

1. **Per-request disk staging on the fix-A writer.** The DISK route's per-request staging
   (`_PerRequestDiskStaging`, `_PerRequestQKDiskStaging`) used to do `_raw_bytes` (a `tobytes`
   copy) then `open(p,"ab")`/write/close per layer per step. It now writes zero-copy through the
   same `RawFileSink` the shared-file drain uses, with one fd per file kept open for the request,
   in BUFFERED mode. Measured 1.39-1.59x on that write (job 1802981). The bytes, file names,
   `LayerEntry`/`QKStepEntry` offsets and the sidecar must be IDENTICAL to the legacy path -- that
   is what the first section checks, byte for byte, on synthetic requests.

2. **Size-aware `auto`.** Direct-vs-buffered used to be decided by alignment alone. `auto` now also
   weighs the write size MIA predicts from the capture configuration against the crossover
   (`aperture_sink.DIRECT_MIN_BYTES`, 64 KiB -- the low end of the band job 1802981 cannot call
   either way, so the rule bites only where buffered decisively wins): a low-concurrency
   last_token capture writes 8-32 KiB per file per step and takes the buffered path; an
   all_tokens-shaped one writes megabytes and takes O_DIRECT. Alignment still gates direct, and an
   explicit mode still wins.

3. **A derived route threshold.** `route_to_disk` / `decide_route` compare two cost MODELS
   (`intercept + slope*KB` per side, per worker kind) and the RPC/disk threshold is SOLVED from
   them, per kind -- not a hardcoded 512 KiB shared by both.

GPU-free: CPU tensors, the real staging/drain/sink classes, and pure functions.
"""
from __future__ import annotations

import fcntl
import filecmp
import os

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py)

import torch

HIDDEN = 16          # 32 B rows -- deliberately NOT block-aligned in most places
HS_META = "hs_aperture_meta.jsonl"
QK_META = "qk_aperture_meta.jsonl"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _rows(n: int, fill: float, width: int = HIDDEN) -> torch.Tensor:
    """One step's already-on-host rows for one layer, distinct per (step, layer)."""
    t = torch.arange(n * width, dtype=torch.float32).reshape(n, width) + fill
    return t.to(torch.bfloat16)


def _same_tree(a, b):
    fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
    assert fa == fb, (fa, fb)
    for f in fa:
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        assert filecmp.cmp(pa, pb, shallow=False), (
            f"{f} differs: {os.path.getsize(pa)} vs {os.path.getsize(pb)} bytes")
        assert os.stat(pa).st_mode == os.stat(pb).st_mode, f   # same perms as the old open()


#: (step, layer, n_rows, hs_mode) -- several steps, a layer that joins late, a 1-row step, and a
#: layer whose row counts differ from its neighbours' (so a relabeled offset that drifted shows).
_HS_STEPS = [
    (0, 1, 3, "all_tokens"), (0, 2, 3, "all_tokens"),
    (1, 1, 5, "all_tokens"), (1, 2, 5, "all_tokens"), (1, 7, 2, "last_token"),
    (2, 1, 1, "all_tokens"), (2, 7, 4, "last_token"),
    (3, 2, 1, "all_tokens"),
]


def _drive_hs_staging(run_dir, write_mode, use_mmap=False):
    from mia.graph.aperture_drain_hs import _PerRequestDiskStaging

    stg = _PerRequestDiskStaging("req-A", str(run_dir), {"dtype": "bfloat16",
                                                         "row_shape": [HIDDEN],
                                                         "hidden": HIDDEN},
                                 capacity_bytes=1 << 16, use_mmap=use_mmap,
                                 write_mode=write_mode)
    for step, layer, n, mode in _HS_STEPS:
        stg.append(layer, _rows(n, 100 * step + layer), n, mode)
    stg.close()
    return stg


def _drive_qk_staging(run_dir, write_mode, use_mmap=False):
    from mia.graph.aperture_drain_qk import _PerRequestQKDiskStaging

    stg = _PerRequestQKDiskStaging("req-A", str(run_dir),
                                   {"dtype": "bfloat16", "q_row_shape": [HIDDEN],
                                    "k_row_shape": [HIDDEN], "q_dim": HIDDEN, "k_dim": HIDDEN,
                                    "hookq_mode": "last_token"},
                                   capacity_bytes=1 << 16, use_mmap=use_mmap,
                                   write_mode=write_mode)
    prefix = 0
    for i, (step, layer, n, _mode) in enumerate(_HS_STEPS):
        emit = i % 3 != 1                                  # some steps append k only
        prefix += n
        stg.append(layer, _rows(n, 100 * step + layer) if emit else None,
                   _rows(n, 500 + 100 * step + layer), prefix if emit else -1,
                   0 if step == 0 else 2)
    stg.close()
    return stg


def _direct_block(path) -> int:
    from mia.graph.aperture_sink import probe_direct_io

    os.makedirs(path, exist_ok=True)
    info = probe_direct_io(str(path))
    if not info.supported:
        pytest.skip(f"no O_DIRECT under {path}: {info.reason}")
    return info.block_size


def _write_path(run_dir, *, row_bytes, mode="auto", shape=None, kind="hs", n_files=2):
    from mia.graph.aperture_sink import ApertureWritePath

    os.makedirs(run_dir, exist_ok=True)
    files = {i: (os.path.join(str(run_dir), f"{kind}_layer_{i}.raw"), kind, row_bytes)
             for i in range(n_files)}
    return ApertureWritePath(str(run_dir), files, mode, allow_direct=True,
                             label=f"{kind} test", shape=shape)


# --------------------------------------------------------------------------------------------
# 1. The per-request disk staging, byte for byte
# --------------------------------------------------------------------------------------------

def test_hs_per_request_staging_on_the_new_writer_is_byte_identical(tmp_path):
    """THE FIX, and its only hard requirement. The staging writes the same raw files and the same
    sidecar through the zero-copy buffered sinks as it did through tobytes + open/append/close."""
    _drive_hs_staging(tmp_path / "legacy", "legacy")
    stg = _drive_hs_staging(tmp_path / "buffered", "buffered")
    assert stg._sinks is not None and not stg._legacy          # the new writer really ran
    _same_tree(tmp_path / "legacy", tmp_path / "buffered")
    assert (tmp_path / "buffered" / HS_META).exists()
    assert sorted(os.listdir(tmp_path / "buffered")) == [
        HS_META, "hs_layer_1.raw", "hs_layer_2.raw", "hs_layer_7.raw"]


def test_qk_per_request_staging_on_the_new_writer_is_byte_identical(tmp_path):
    """The QK twin: q written compactly (non-emit steps append k only), k every step."""
    _drive_qk_staging(tmp_path / "legacy", "legacy")
    stg = _drive_qk_staging(tmp_path / "buffered", "buffered")
    assert stg._sinks is not None
    _same_tree(tmp_path / "legacy", tmp_path / "buffered")
    assert (tmp_path / "buffered" / QK_META).exists()


def test_hs_staging_matches_the_legacy_mmap_writer_too(tmp_path):
    """MIA_APERTURE_MMAP=1 is the other legacy sink. Same bytes, so an operator moving off it
    (which `auto` now does for them) loses nothing."""
    _drive_hs_staging(tmp_path / "mmap", "legacy", use_mmap=True)
    _drive_hs_staging(tmp_path / "buffered", "buffered")
    _same_tree(tmp_path / "mmap", tmp_path / "buffered")


def test_the_new_staging_reads_back_as_one_request(tmp_path):
    """Byte identity is checked above; this checks the thing byte identity is FOR -- MIA's own
    reader reconstructs the request from a run_dir that contains only it."""
    from mia.graph.aperture_reader import load_multilayer_aperture_artifact

    _drive_hs_staging(tmp_path / "legacy", "legacy")
    _drive_hs_staging(tmp_path / "new", "buffered")
    old = load_multilayer_aperture_artifact(str(tmp_path / "legacy"))["req-A"]
    new = load_multilayer_aperture_artifact(str(tmp_path / "new"))["req-A"]
    assert sorted(old) == sorted(new) == [1, 2, 7]
    for layer in old:
        a, b = old[layer], new[layer]
        a = a["hidden_states"] if isinstance(a, dict) else a
        b = b["hidden_states"] if isinstance(b, dict) else b
        assert torch.equal(a, b) and a.shape[0] > 0


def test_staging_opens_each_file_once_copies_nothing_and_closes_every_fd(tmp_path, monkeypatch):
    """The three things the rewrite is FOR, asserted directly: after the first append of a layer
    the file is never re-opened, no `bytes` object is materialised, and close() leaves no fd."""
    import builtins

    from mia.graph import aperture_drain_hs, aperture_sink

    before = set(os.listdir("/proc/self/fd"))
    stg = aperture_drain_hs._PerRequestDiskStaging(
        "req-fd", str(tmp_path / "fd"), {"dtype": "bfloat16", "row_shape": [HIDDEN]},
        capacity_bytes=1 << 16, use_mmap=False, write_mode="buffered")
    for layer in (1, 2):                                  # open each layer's file once, up front
        stg.append(layer, _rows(2, layer), 2, "all_tokens")
    assert len(stg._sinks.sinks) == 2 and all(s.fd is not None
                                              for s in stg._sinks.sinks.values())

    real_open, real_os_open = builtins.open, os.open

    def guard(path, *a, **k):
        if str(path).endswith(".raw"):
            raise AssertionError(f"raw file re-opened during the run: {path}")
        return real_open(path, *a, **k)

    def guard_os(path, *a, **k):
        if str(path).endswith(".raw"):
            raise AssertionError(f"raw file re-opened during the run: {path}")
        return real_os_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", guard)
    monkeypatch.setattr(aperture_sink.os, "open", guard_os)
    monkeypatch.setattr(aperture_drain_hs, "_raw_bytes",
                        lambda t: pytest.fail("the new staging must not materialise bytes"))
    for step in range(1, 4):
        for layer in (1, 2):
            stg.append(layer, _rows(2, step * 10 + layer), 2, "all_tokens")
    monkeypatch.undo()
    stg.close()
    assert not stg._sinks.sinks
    assert set(os.listdir("/proc/self/fd")) == before
    stg.close()                                           # idempotent


def test_an_aborted_staging_releases_its_fds_and_writes_no_sidecar(tmp_path):
    """`discard()` is the abort path: fds released, dir gone, no sidecar (never delivered)."""
    from mia.graph.aperture_drain_hs import _PerRequestDiskStaging

    d = tmp_path / "aborted"
    stg = _PerRequestDiskStaging("req-X", str(d), {"dtype": "bfloat16", "row_shape": [HIDDEN]},
                                 capacity_bytes=1 << 16, use_mmap=False, write_mode="buffered")
    stg.append(1, _rows(2, 1.0), 2, "all_tokens")
    stg.discard()
    assert not d.exists() and not stg._sinks.sinks
    stg.discard()                                         # idempotent


# --------------------------------------------------------------------------------------------
# 1b. The drain resolves that staging mode from MIA_APERTURE_WRITE_MODE
# --------------------------------------------------------------------------------------------

def _per_request_drain(tmp_path):
    from mia.graph.aperture_drain_hs import OffLoopApertureDrain
    from mia.graph.capture_aperture import CaptureAperture

    ap = CaptureAperture(row_bytes=HIDDEN * 2, n_slots=8, device="cpu", dtype=torch.bfloat16,
                         row_shape=(HIDDEN,))
    return OffLoopApertureDrain(ap, [(1, torch.zeros(9, HIDDEN, dtype=torch.bfloat16))],
                                str(tmp_path), {"dtype": "bfloat16", "row_shape": [HIDDEN]},
                                per_request=True)


@pytest.mark.parametrize("env,want", [(None, "buffered"), ("auto", "buffered"),
                                      ("buffered", "buffered"), ("legacy", "legacy")])
def test_the_drain_resolves_the_staging_write_mode(tmp_path, monkeypatch, env, want):
    """DEFAULT (env unset) = the new writer. `legacy` still reaches the old one for an A/B."""
    if env is None:
        monkeypatch.delenv("MIA_APERTURE_WRITE_MODE", raising=False)
    else:
        monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", env)
    d = _per_request_drain(tmp_path / (env or "default"))
    try:
        assert d._perreq_write_mode == want
        note = d.write_path_summary()
        assert "does not apply to shared raw files" in note
        assert ("zero-copy buffered" in note) == (want == "buffered")
    finally:
        d.close()


def test_the_drains_disk_route_stages_through_the_resolved_writer(tmp_path, monkeypatch):
    """End to end through the drain's own `_disk_write`: same staged bytes under the default
    (auto -> buffered) as under legacy, and the staging really used the new sinks."""
    trees = {}
    for mode in ("legacy", "auto"):
        monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", mode)
        d = _per_request_drain(tmp_path / mode)
        try:
            d._disk_routed["req-R"] = str(tmp_path / mode / "dest")
            for step in range(3):
                d._disk_write("req-R", 1, _rows(2, step), 2, "all_tokens")
            stg = d._disk_staging["req-R"]
            assert (stg._sinks is not None) == (mode == "auto")
            stg.close()
            trees[mode] = os.path.join(str(tmp_path / mode), "perreq", "req-R")
        finally:
            d.close()
    _same_tree(trees["legacy"], trees["auto"])


def test_per_request_delivery_refuses_direct_and_the_mmap_sink(tmp_path, monkeypatch):
    """Both refusals are LOUD at construction, and both name what to do instead: there is no
    O_DIRECT path here (and at 8-16 KiB per write there should not be), and the mmap sink existed
    to remove a per-step open() that the run-long fd already removes."""
    from mia.graph.aperture_sink import ApertureWriteConfigError

    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "direct")
    with pytest.raises(ApertureWriteConfigError, match="per-request delivery"):
        _per_request_drain(tmp_path / "direct")
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "auto")
    monkeypatch.setenv("MIA_APERTURE_MMAP", "1")
    with pytest.raises(ApertureWriteConfigError, match="MIA_APERTURE_MMAP"):
        _per_request_drain(tmp_path / "mmap")
    monkeypatch.setenv("MIA_APERTURE_WRITE_MODE", "legacy")
    d = _per_request_drain(tmp_path / "legacy")           # legacy + mmap is still allowed
    try:
        assert d._perreq_write_mode == "legacy"
    finally:
        d.close()


# --------------------------------------------------------------------------------------------
# 2. Size-aware `auto`
# --------------------------------------------------------------------------------------------

def test_predict_rows_per_write_is_an_upper_bound_on_the_configuration():
    """Every rule here is an UPPER bound, so the bias is towards direct -- being wrong that way
    costs at most +0.43 ms/step with a single writer and +0.04 ms with the shipped 2-thread pool,
    being wrong the other way costs ~4x (job 1802981)."""
    from mia.graph.aperture_sink import predict_rows_per_write as p

    kw = dict(max_batched_tokens=8192, max_num_seqs=128, aperture_rows=8192)
    # last_token HS: at most one row per in-flight request per step, whatever the phase.
    assert p("hs", capture_mode="last_token", **kw).rows == 128
    assert p("hs", capture_mode="all_tokens", **kw).rows == 8192
    # QK captures every token's k row in BOTH modes -- last_token only stops q being emitted.
    assert p("qk", capture_mode="last_token", **kw).rows == 8192
    assert p("qk", capture_mode="all_tokens", **kw).rows == 8192
    # The aperture is the tighter bound when it is smaller than the token budget.
    assert p("hs", capture_mode="all_tokens", max_batched_tokens=8192, max_num_seqs=128,
             aperture_rows=512).rows == 512
    # An unknown mode is treated as the largest thing it could be.
    assert p("hs", capture_mode="", **kw).rows == 8192
    # The aperture alone still bounds it; with nothing at all, no prediction (alignment only).
    assert p("hs", capture_mode="all_tokens", max_batched_tokens=None, max_num_seqs=None,
             aperture_rows=64).rows == 64
    assert p("hs", capture_mode="all_tokens", max_batched_tokens=None, max_num_seqs=None,
             aperture_rows=None) is None


def test_a_last_token_capture_picks_buffered_only_at_low_concurrency():
    """The `max_num_seqs` bound bites below 8 concurrent requests at 8B (8 KiB rows) and 4 at 70B
    (16 KiB rows) -- arithmetic off the threshold, and NARROWER than the window where buffered
    decisively wins (8-32 KiB per write, and only with a single writer), because the threshold
    sits at the low end of the undecidable band. Above it a last_token step can write 1 MiB+ (the
    measured `8b-hs-decode` shape) and must take O_DIRECT."""
    from mia.graph.aperture_sink import DIRECT_MIN_BYTES, predict_rows_per_write as p

    def bytes_at(seqs, row_bytes):
        return p("hs", capture_mode="last_token", max_batched_tokens=8192, max_num_seqs=seqs,
                 aperture_rows=8192).bytes_for(row_bytes)

    assert bytes_at(4, 8192) < DIRECT_MIN_BYTES                  # 8B, 4 decoders -> buffered
    assert bytes_at(8, 8192) >= DIRECT_MIN_BYTES                 # 8B, 8 -> the crossover exactly
    assert bytes_at(2, 16384) < DIRECT_MIN_BYTES                 # 70B, 2 decoders -> buffered
    assert bytes_at(4, 16384) >= DIRECT_MIN_BYTES                # 70B, 4
    assert bytes_at(128, 8192) == 1024 * 1024                    # the measured 8b-hs-decode shape


def test_auto_picks_buffered_for_a_last_token_shape_and_direct_for_all_tokens(tmp_path):
    """The headline of the size rule, at the study's 8B row width (4096 x bf16 = 8 KiB), which is
    block-aligned -- so alignment alone would have opened BOTH of these O_DIRECT. The small one is
    a last_token capture at 4 concurrent requests (4 x 8 KiB = 32 KiB per write), inside the band
    where the measurement decisively favours buffered."""
    from mia.graph.aperture_sink import WriteShape

    block = _direct_block(tmp_path)
    assert 8192 % block == 0, block
    small = _write_path(tmp_path / "last", row_bytes=8192,
                        shape=WriteShape(4, "hs last_token, max_num_seqs=4"))
    big = _write_path(tmp_path / "all", row_bytes=8192,
                      shape=WriteShape(8192, "hs all_tokens"))
    try:
        assert small.kind_modes() == {"hs": "buffered"}
        assert "32.0 KiB/write predicted" in small.summary()
        assert "below the 64.0 KiB O_DIRECT crossover" in small.summary()
        assert "hs last_token" in small.summary()          # the basis, for the log reader
        assert big.kind_modes() == {"hs": "direct"}
        assert ">= the 64.0 KiB O_DIRECT crossover" in big.summary()
    finally:
        small.close()
        big.close()


def test_the_size_reason_names_the_measurement_it_was_decided_on(tmp_path):
    """The reason reaches the install line, which the harness and humans read back as evidence, so
    it must describe the column the threshold came off -- the drained STEP, per writer count --
    and not restate the isolated per-write ratio at ONE writer thread, which is a different
    configuration from the pooled drain this decision governs (job 1802981: below the threshold a
    step measures +14..+37 % direct at 1 writer thread, and inside the job's own noise floor,
    median 2.8 % / p90 11.9 %, at the shipped 2)."""
    from mia.graph.aperture_sink import WriteShape

    _direct_block(tmp_path)
    wp = _write_path(tmp_path / "why", row_bytes=8192,
                     shape=WriteShape(4, "hs last_token, max_num_seqs=4"))
    try:
        why = wp.summary()
    finally:
        wp.close()
    assert "drained step" in why and "single writer" in why and "noise floor" in why
    # The per-WRITE single-writer ladder is the per-request staging's column, not this one.
    assert "1.74x" not in why and "1.65x" not in why


def test_an_unaligned_row_is_buffered_whatever_the_size(tmp_path):
    """Alignment still gates O_DIRECT: a row width that is not a multiple of the block can never
    take it, however big the predicted write is (QK k rows at TP4/TP8 are exactly this case)."""
    from mia.graph.aperture_sink import WriteShape

    block = _direct_block(tmp_path)
    wp = _write_path(tmp_path / "odd", row_bytes=block + 1, kind="k",
                     shape=WriteShape(8192, "qk all_tokens"))
    try:
        assert wp.kind_modes() == {"k": "buffered"}
        assert "not a multiple of the" in wp.summary()
    finally:
        wp.close()


def test_an_explicit_mode_always_wins(tmp_path):
    """`direct` and `buffered` are requirements, not suggestions: neither consults the size."""
    from mia.graph.aperture_sink import WriteShape

    _direct_block(tmp_path)
    tiny = WriteShape(1, "hs last_token, max_num_seqs=1")
    forced = _write_path(tmp_path / "forced", row_bytes=8192, mode="direct", shape=tiny)
    buffered = _write_path(tmp_path / "buf", row_bytes=8192, mode="buffered",
                           shape=WriteShape(8192, "hs all_tokens"))
    try:
        assert forced.kind_modes() == {"hs": "direct"}
        assert fcntl.fcntl(forced.sink(0).fd, fcntl.F_GETFL) & os.O_DIRECT
        assert buffered.kind_modes() == {"hs": "buffered"}
    finally:
        forced.close()
        buffered.close()


def test_the_crossover_is_one_named_constant_and_env_overridable(tmp_path, monkeypatch):
    from mia.graph.aperture_sink import (
        ApertureWriteConfigError, DIRECT_MIN_BYTES, WriteShape, resolve_direct_min_bytes)

    _direct_block(tmp_path)
    assert DIRECT_MIN_BYTES == 64 * 1024 == resolve_direct_min_bytes()
    # The measured bracket (job 1802981, per drained step against that job's own noise floor):
    # buffered wins decisively only at <= 32 KiB and only with a single writer; direct wins
    # decisively from 512 KiB up at both thread counts. A default outside this bracket would
    # contradict a measured column -- 64 KiB is its low end, where the error is cheapest.
    assert 32 * 1024 < DIRECT_MIN_BYTES <= 512 * 1024
    shape = WriteShape(4, "4 x 8 KiB = 32 KiB per write")          # below the default crossover
    wp = _write_path(tmp_path / "default", row_bytes=8192, shape=shape)
    try:
        assert wp.kind_modes() == {"hs": "buffered"}
    finally:
        wp.close()
    monkeypatch.setenv("MIA_APERTURE_DIRECT_MIN_BYTES", "16384")    # now 32 KiB clears it
    wp = _write_path(tmp_path / "lowered", row_bytes=8192, shape=shape)
    try:
        assert wp.kind_modes() == {"hs": "direct"}
    finally:
        wp.close()
    monkeypatch.setenv("MIA_APERTURE_DIRECT_MIN_BYTES", "0")        # 0 = the alignment-only rule
    wp = _write_path(tmp_path / "off", row_bytes=8192, shape=WriteShape(1, "one row"))
    try:
        assert wp.kind_modes() == {"hs": "direct"}
    finally:
        wp.close()
    monkeypatch.setenv("MIA_APERTURE_DIRECT_MIN_BYTES", "lots")
    with pytest.raises(ApertureWriteConfigError, match="MIA_APERTURE_DIRECT_MIN_BYTES"):
        _write_path(tmp_path / "bad", row_bytes=8192, shape=shape)


def test_a_prediction_that_traffic_contradicts_is_reported_once(tmp_path, capsys):
    """The prediction comes from the worker-wide capture mode, which a request may override. When
    that happens the file is already open buffered (a file is never written both ways), so the only
    useful thing left is to SAY SO -- once, loudly, with the fix."""
    from mia.graph.aperture_sink import WriteShape, record_step_stats, WriteStats

    _direct_block(tmp_path)
    wp = _write_path(tmp_path / "wrong", row_bytes=8192, shape=WriteShape(4, "hs last_token"))
    try:
        assert wp.kind_modes() == {"hs": "buffered"}
        stats = WriteStats()
        kw = dict(step_s=0.0, d2h_s=0.0, write_s=0.0, write_tail_s=0.0, busy_s=0.0,
                  bookkeeping_s=0.0, bytes_by_mode={}, wp=wp)
        record_step_stats(stats, "hs", rows=4, **kw)          # as predicted: silent
        assert "aperture-write" not in capsys.readouterr().out
        record_step_stats(stats, "hs", rows=4096, **kw)       # 32 MiB: the prediction was wrong
        out = capsys.readouterr().out
        assert "opened BUFFERED" in out and "32.0 MiB per file" in out
        assert "WRITE_MODE=direct" in out
        record_step_stats(stats, "hs", rows=4096, **kw)       # reported ONCE
        assert "aperture-write" not in capsys.readouterr().out
    finally:
        wp.close()


def test_a_direct_file_is_never_second_guessed(tmp_path, capsys):
    """The watch is only on files downgraded BY SIZE: a direct file writing small rows is the
    bounded sub-millisecond case, not worth a line, and an alignment downgrade cannot be
    reconsidered."""
    from mia.graph.aperture_sink import WriteShape, record_step_stats, WriteStats

    block = _direct_block(tmp_path)
    big = _write_path(tmp_path / "big", row_bytes=8192, shape=WriteShape(8192, "hs all_tokens"))
    odd = _write_path(tmp_path / "odd", row_bytes=block + 1, kind="k",
                      shape=WriteShape(1, "one row"))
    try:
        kw = dict(step_s=0.0, d2h_s=0.0, write_s=0.0, write_tail_s=0.0, busy_s=0.0,
                  bookkeeping_s=0.0, bytes_by_mode={})
        record_step_stats(WriteStats(), "hs", rows=1, wp=big, **kw)
        record_step_stats(WriteStats(), "qk", rows=99999, wp=odd, **kw)
        assert "aperture-write" not in capsys.readouterr().out
    finally:
        big.close()
        odd.close()


def test_no_prediction_keeps_the_pre_2026_09_20_rule(tmp_path):
    """A caller that cannot predict (a bench, a direct construction) is not silently downgraded:
    with no shape, `auto` decides on alignment alone, exactly as it did before."""
    _direct_block(tmp_path)
    wp = _write_path(tmp_path / "unknown", row_bytes=8192, shape=None)
    try:
        assert wp.kind_modes() == {"hs": "direct"}
        assert "decided on alignment alone" in wp.summary()
    finally:
        wp.close()


def test_the_install_line_still_parses_as_the_harness_reads_it(tmp_path):
    """The profiling harness judges every capture run on this line (`evidence.check_write_path`):
    `write mode=<mode>`, `<kind>=<path>`, `N writer thread(s)`, `O_DIRECT block N B`, and the
    absence of `does not apply` on a shared-file drain. The added size text must not disturb it."""
    import re

    from mia.graph.aperture_sink import WriteShape

    _direct_block(tmp_path)
    wp = _write_path(tmp_path / "line", row_bytes=8192,
                     shape=WriteShape(4, "hs last_token, max_num_seqs=4"))
    try:
        wp.start_pool(2, torch.device("cpu"), "t")
        line = wp.summary()
    finally:
        wp.close()
    assert re.search(r"write mode=([a-z]+)", line).group(1) == "auto"
    assert re.findall(r"\b(hs|q|k)=(direct\+buffered|direct|buffered|legacy)\b",
                      line) == [("hs", "buffered")]
    assert re.search(r"(\d+) writer thread\(s\)", line).group(1) == "2"
    assert re.search(r"O_DIRECT block (\d+) B", line)
    assert not re.search(r"open\(O_DIRECT\) failed", line)
    assert "does not apply" not in line


# --------------------------------------------------------------------------------------------
# 3. The route: a derived threshold, per worker kind
# --------------------------------------------------------------------------------------------

def test_the_crossover_is_solved_from_the_two_cost_models():
    """Not a written-down constant: at the crossover the two models are equal, and `route_to_disk`
    flips exactly there. HS and QK get their own, because their RPC slopes differ 5x."""
    from mia.run_utils import predicted_disk_ms, predicted_rpc_ms, rpc_disk_crossover_kb, route_to_disk

    for kind in ("hs", "qk"):
        kb = rpc_disk_crossover_kb(kind)
        assert predicted_rpc_ms(kind, kb) == pytest.approx(predicted_disk_ms(kind, kb), rel=1e-9)
        assert not route_to_disk(kind, kb * 0.99)
        assert route_to_disk(kind, kb * 1.01)
    hs, qk = rpc_disk_crossover_kb("hs"), rpc_disk_crossover_kb("qk")
    assert hs == pytest.approx(539.6, abs=0.5)     # 5.0 + 0.03 KB  vs  20.0 + 0.0022 KB
    assert qk == pytest.approx(100.5, abs=0.5)     # 5.0 + 0.157 KB vs  20.0 + 0.0078 KB
    assert qk < hs / 5


def test_every_router_coefficient_has_an_env_override(monkeypatch):
    """Each side of each model is retunable without a restart -- and retuning one MOVES the
    threshold, because the threshold is solved from them."""
    from mia.run_utils import rpc_disk_crossover_kb

    base = rpc_disk_crossover_kb("hs")
    monkeypatch.setenv("MIA_ROUTER_DISK_HANDOFF_MS", "1.0")
    assert rpc_disk_crossover_kb("hs") < base           # a cheap handoff -> disk much earlier
    monkeypatch.delenv("MIA_ROUTER_DISK_HANDOFF_MS")
    monkeypatch.setenv("MIA_ROUTER_DISK_SLOPE_MS_PER_KB_HS", "0.02")
    assert rpc_disk_crossover_kb("hs") > base           # a dearer disk -> RPC for longer
    monkeypatch.delenv("MIA_ROUTER_DISK_SLOPE_MS_PER_KB_HS")
    monkeypatch.setenv("MIA_ROUTER_RPC_SLOPE_MS_PER_KB_HS", "0.157")
    assert rpc_disk_crossover_kb("hs") == pytest.approx(rpc_disk_crossover_kb("qk"), rel=0.2)


def test_no_crossover_when_rpc_never_costs_more(monkeypatch):
    """A degenerate retune must not divide by zero or route everything to disk by accident."""
    from mia.run_utils import NO_CROSSOVER_KB, rpc_disk_crossover_kb, route_to_disk

    monkeypatch.setenv("MIA_ROUTER_RPC_SLOPE_MS_PER_KB_HS", "0.0")
    assert rpc_disk_crossover_kb("hs") == NO_CROSSOVER_KB
    assert not route_to_disk("hs", 1e6)
    monkeypatch.setenv("MIA_ROUTER_DISK_HANDOFF_MS", "1.0")     # disk cheaper from byte zero
    assert rpc_disk_crossover_kb("hs") == 0.0


def test_the_aperture_thresholds_are_the_derived_ones_per_kind(monkeypatch):
    from mia import _plugin
    from mia.run_utils import rpc_disk_crossover_kb

    for kind in ("hs", "qk"):
        t_rpc, t_analyze = _plugin._aperture_route_thresholds(kind)
        assert t_rpc == t_analyze == int(rpc_disk_crossover_kb(kind) * 1024)
    assert _plugin._aperture_route_thresholds("qk")[0] < _plugin._aperture_route_thresholds("hs")[0]
    monkeypatch.setenv("MIA_ROUTER_T_RPC", "4096")
    monkeypatch.setenv("MIA_ROUTER_T_ANALYZE", "8192")
    assert _plugin._aperture_route_thresholds("hs") == (4096, 8192)


# --------------------------------------------------------------------------------------------
# 4. The default needs no user setting
# --------------------------------------------------------------------------------------------

class _FakeEngine:
    """Just enough of a vLLM engine for the size model: Llama-3.1-8B's shape."""

    class model_config:                                   # noqa: N801 -- mirrors vLLM's attribute
        class hf_text_config:                             # noqa: N801
            num_attention_heads = 32
            num_key_value_heads = 8
            hidden_size = 4096
            num_hidden_layers = 32


def _route(extra, prompt_len=512, max_tokens=300):
    from mia import _plugin

    return _plugin._maybe_storage_route(
        _FakeEngine(), {"prompt_token_ids": list(range(prompt_len))}, dict(extra), max_tokens)


def test_the_storage_router_is_on_by_default_and_needs_no_setting(monkeypatch):
    """The owner's ask: a user sets nothing. A tiny last-token HS capture comes back over RPC; a
    big all-tokens one goes to disk; QK goes to disk at a size HS would still ship."""
    from mia.optimizations import env_is_on

    monkeypatch.delenv("MIA_STORAGE_ROUTER", raising=False)
    for var in ("MIA_ROUTER_T_RPC", "MIA_ROUTER_T_ANALYZE", "MIA_ROUTER_DISK_HANDOFF_MS"):
        monkeypatch.delenv(var, raising=False)
    assert env_is_on("storage_router")
    assert _route({"output_hidden_states": list(range(1, 33)), "hs_mode": "last_token",
                   "hooks_on": "prefill"}) is False                       # 256 KB -> RPC
    assert _route({"output_hidden_states": list(range(1, 33)), "hs_mode": "all_tokens",
                   "hooks_on": "prefill"}) is True                        # 128 MB -> disk
    assert _route({"output_qk": {"0": [0, 1]}, "hookq_mode": "last_token",
                   "hooks_on": "prefill"}) is True                        # QK crosses far earlier
    assert _route({"steer": {"layer": 1}}) is None                        # steer stores nothing


def test_an_explicit_save_to_disk_is_never_overridden():
    """`save_to_disk` is how a caller asks for a durable FILE, so the router must only fire when
    the caller expressed no preference. The rule lives at the ONE call site, in the serve patch:
    `if "save_to_disk" not in extra:`."""
    import inspect

    from mia import _plugin

    src = inspect.getsource(_plugin._patched_generate)
    call = src.index("_maybe_storage_route(")
    guard = src.rindex('if "save_to_disk" not in extra:', 0, call)
    assert guard < call and src.count("_maybe_storage_route(") == 1


# --------------------------------------------------------------------------------------------
# 5. The size prediction prices every CAPTURED STEP, and estimates an unknown gen_len
#
# `last_token` names which token of a step is kept, NOT how often the hook fires. The hook fires
# on every forward step the request is in, so a decode-hooked last-token capture keeps one vector
# per layer PER STEP. `predict_artifact_kb` priced it at one step's worth (`L * hidden`), which is
# right for `hooks_on: prefill` and ~300x small for the `hooks_on: both` serve default.
# --------------------------------------------------------------------------------------------

L32, HIDDEN_8B, HEAD_DIM_8B = 32, 4096, 128
ONE_HS_STEP_KB = L32 * HIDDEN_8B * 2 / 1024            # 256.0 KB -- all 32 layers, one step


def _hs_kb(gran, hooks_on, gen_len, prompt_len=512):
    from mia.run_utils import predict_artifact_kb

    return predict_artifact_kb("hs", gran, prompt_len, L32, 1, HEAD_DIM_8B,
                               HIDDEN_8B, 2, gen_len, hooks_on)


def test_hs_last_token_prices_one_row_per_layer_per_captured_step():
    """The bug, stated as arithmetic. `hooks_on: prefill` captures ONE step (unchanged, and it is
    the profiled hs-lasttok case); `both` captures the prefill plus every decode step."""
    assert _hs_kb("last_token", "prefill", 300) == pytest.approx(ONE_HS_STEP_KB)
    assert _hs_kb("last_token", "both", 300) == pytest.approx(ONE_HS_STEP_KB * 301)
    assert _hs_kb("last_token", "decode", 300) == pytest.approx(ONE_HS_STEP_KB * 300)
    # the prompt length does not enter a last_token size, at any hooks_on -- one row, not a span
    assert _hs_kb("last_token", "both", 300, prompt_len=8192) == \
        pytest.approx(_hs_kb("last_token", "both", 300, prompt_len=16))


def test_a_decode_hooked_last_token_capture_is_not_priced_as_a_single_step():
    """The regression this section exists for: at the serve default the artifact is ~78 MB, not
    256 KB, and 256 KB is what the old model returned."""
    assert _hs_kb("last_token", "both", 300) / 1024 == pytest.approx(75.25, abs=0.1)   # MB
    assert _hs_kb("last_token", "both", 300) > 250 * _hs_kb("last_token", "prefill", 300)


def test_qk_needed_no_matching_fix():
    """QK's model already grows with the context and with the per-step q row, so it was never
    blind to gen_len. Pinned so a future edit to the HS branch does not 'fix' QK by symmetry."""
    from mia.run_utils import predict_artifact_kb

    def qk(hooks_on, gen_len):
        return predict_artifact_kb("qk", "last_token", 512, L32, 1, HEAD_DIM_8B,
                                   L32 * HEAD_DIM_8B, 2, gen_len, hooks_on)

    assert qk("both", 300) > qk("prefill", 300)
    assert qk("both", 600) > qk("both", 300)


def test_the_route_does_not_need_an_accurate_gen_len():
    """Why a rough estimate is enough: the router asks which SIDE of the crossover the artifact
    falls on. For this shape the answer stops changing at two decode tokens, so every plausible
    estimate agrees -- the fix does not depend on guessing the generation length well."""
    from mia.run_utils import route_to_disk, rpc_disk_crossover_kb

    assert rpc_disk_crossover_kb("hs") == pytest.approx(539.6, abs=0.5)
    assert not route_to_disk("hs", _hs_kb("last_token", "both", 1))     # 2 steps = 512 KB -> RPC
    for g in (2, 16, 256, 4096):                                        # >= 3 steps -> disk, all
        assert route_to_disk("hs", _hs_kb("last_token", "both", g)), g
    # and the prefill-only optimum the cb campaigns proved is untouched
    assert not route_to_disk("hs", _hs_kb("last_token", "prefill", 300))


def test_an_unknown_max_tokens_gets_an_estimate_not_a_zero(monkeypatch):
    """A serve request need not pin max_tokens. `0` is not a decode length any request has; it is
    what made the prediction collapse to one step."""
    from mia.run_utils import DEFAULT_GEN_LEN, estimate_gen_len

    monkeypatch.delenv("MIA_ROUTER_DEFAULT_GEN_LEN", raising=False)
    assert estimate_gen_len(300) == 300              # a pinned max_tokens always wins
    assert estimate_gen_len(None) == DEFAULT_GEN_LEN
    assert estimate_gen_len(0) == DEFAULT_GEN_LEN
    assert estimate_gen_len("not a number") == DEFAULT_GEN_LEN
    monkeypatch.setenv("MIA_ROUTER_DEFAULT_GEN_LEN", "1024")
    assert estimate_gen_len(None) == 1024
    monkeypatch.setenv("MIA_ROUTER_DEFAULT_GEN_LEN", "junk")
    assert estimate_gen_len(None) == DEFAULT_GEN_LEN  # a bad override must not crash a request


def test_both_routers_price_an_unpinned_request_the_same_way():
    """The storage router and the aperture delivery router must agree, so a request cannot have
    its sink chosen from one size and its transport from another."""
    from mia import _plugin

    extra = {"output_hidden_states": list(range(1, 33)), "hs_mode": "last_token",
             "hooks_on": "both"}
    prompt = {"prompt_token_ids": list(range(512))}
    assert _route(extra, max_tokens=None) is True          # no max_tokens -> still sized -> disk
    assert _route(extra, max_tokens=300) is True
    decision = _plugin._decide_aperture_route(_FakeEngine(), prompt, dict(extra), None)
    assert decision is not None and decision.transport == "disk"


def test_no_call_site_prices_an_unpinned_request_at_zero():
    """The four routers all read max_tokens off the request. Each must go through the estimate --
    a new one that writes `int(max_tokens) if max_tokens else 0` reintroduces the bug."""
    import inspect

    from mia import _plugin

    for fn in (_plugin._maybe_storage_route, _plugin._decide_aperture_route,
               _plugin._decide_aperture_route_qk):
        src = inspect.getsource(fn)
        if "gen_len" not in src:
            continue
        assert "estimate_gen_len(" in src, fn.__name__
        assert "if max_tokens else 0" not in src, fn.__name__


# --------------------------------------------------------------------------------------------
# 6. A disk-routed request under per-request delivery must actually be WRITTEN
#
# W13 (2026-09-21, H100) ran the per-request delivery path with a disk route for the first time
# and captured 47 GB while writing 0 bytes, silently. Two sites encoded "does this request take
# the aperture per-request path?" separately and disagreed on `disk`: the request-start guard
# refused to arm the route, so nothing was staged; finalize's `sink == "disk"` branch then called
# flush_disk, which flushes the EAGER worker's buffers -- empty, because the capture went to the
# aperture. The drain builds no shared-file sink in per-request mode, so there was no third place
# for the bytes to land. Every evidence check passed, including one that read direct=0 buffered=0
# legacy=0 and passed vacuously.
#
# It hit BOTH ways of reaching disk: an explicit save_to_disk, and MIA's OWN storage router, which
# writes extra["save_to_disk"]=True when it picks disk and so tripped the same guard -- i.e. the
# automatic data path's disk half delivered nothing.
# --------------------------------------------------------------------------------------------

@pytest.fixture
def per_request_armed(monkeypatch):
    monkeypatch.setenv("MIA_APERTURE_PER_REQUEST", "1")
    monkeypatch.setenv("MIA_ALLOW_CUDAGRAPH", "1")
    monkeypatch.delenv("MIA_PROFILE_MODE", raising=False)
    monkeypatch.delenv("MIA_SINK", raising=False)


def test_a_disk_sink_takes_the_aperture_path_under_per_request_delivery(per_request_armed):
    """The regression, as the predicate. `disk` is what this path exists to serve -- it must not
    be an exclusion. Both ways of asking are covered: the caller's own key, and the value MIA's
    storage router writes when it picks disk (they are the same key by the time finalize runs)."""
    from mia import _plugin

    for sink_extra in ({"save_to_disk": True}, {"save_to_disk": False}, {}):
        extra = {"output_hidden_states": [1, 2], **sink_extra}
        assert _plugin._takes_aperture_per_request(extra, True, False, False) is True, sink_extra


def test_only_drop_is_excluded_and_the_path_stays_hs_only(per_request_armed, monkeypatch):
    """`drop` stores nothing by definition, so staging it would orphan NVMe files. QK and steer
    are excluded because this delivery path is HS-only -- a QK request must not be swallowed by a
    branch that returns hs_cache."""
    from mia import _plugin

    hs = {"output_hidden_states": [1]}
    monkeypatch.setenv("MIA_SINK", "drop")
    assert _plugin._takes_aperture_per_request(hs, True, False, False) is False
    monkeypatch.delenv("MIA_SINK")
    assert _plugin._takes_aperture_per_request(hs, True, True, False) is False   # also QK
    assert _plugin._takes_aperture_per_request(hs, True, False, True) is False   # steering
    assert _plugin._takes_aperture_per_request(hs, False, False, False) is False  # no HS


def test_the_path_is_inert_unless_armed(monkeypatch):
    """Additive gate: with per-request mode off, every request keeps its existing response path,
    so the finalize reordering below cannot change a run that does not arm this."""
    from mia import _plugin

    monkeypatch.delenv("MIA_APERTURE_PER_REQUEST", raising=False)
    monkeypatch.setenv("MIA_ALLOW_CUDAGRAPH", "1")
    assert _plugin._takes_aperture_per_request({"output_hidden_states": [1]},
                                               True, False, False) is False


def test_profile_mode_still_excludes_it(per_request_armed, monkeypatch):
    """Profile mode measures Component 1 only and runs no delivery; the predicate must keep
    agreeing with that, or a profile run would stage files it never reads."""
    from mia import _plugin

    monkeypatch.setenv("MIA_PROFILE_MODE", "1")
    assert _plugin._takes_aperture_per_request({"output_hidden_states": [1]},
                                               True, False, False) is False


def test_the_two_sites_use_the_one_predicate_and_the_disk_branch_declines():
    """The ROOT CAUSE, pinned at the source. The request-start guard and the finalize gate must
    both ask `_takes_aperture_per_request`, and the `sink == "disk"` branch must decline to it --
    otherwise flush_disk wins again and the bytes are dropped. A source check because the
    alternative is booting an engine."""
    import inspect

    from mia import _plugin

    gen = inspect.getsource(_plugin._patched_generate)
    # the HS arming guard and the HS finalize gate, both on the one KIND predicate (section 7
    # generalised it to tell HS from QK; the shared boolean is what the disk branch reads)
    assert gen.count('_aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "hs"') == 2
    # the disk branch is conditional on NOT taking any aperture path
    assert ('elif (sink == "disk"\n                      and not _takes_aperture_per_request('
            in gen), "the disk branch no longer declines to the aperture path"
    # and the old, separately-spelled conditions are gone from BOTH workers
    assert "elif _aperture_per_request_mode() and wants_hs and not wants_qk:" not in gen
    assert "elif _aperture_per_request_mode() and wants_qk and not wants_hs:" not in gen


def test_an_explicit_disk_ask_forces_the_disk_transport(per_request_armed):
    """CAPTURE_IO.md 8.1: an explicit ask is a requirement, not a hint. A small artifact whose
    sink is already `disk` must not be shipped over RPC -- the caller asked for a FILE. Uses the
    W13 8B last-token shape, which the size model prices at 256 KB, well under the 539.6 KB
    crossover, so the size model on its own would say RPC."""
    from mia import _plugin

    prompt = {"prompt_token_ids": list(range(512))}
    small = {"output_hidden_states": list(range(1, 33)), "hs_mode": "last_token",
             "hooks_on": "prefill"}

    no_pref = _plugin._decide_aperture_route(_FakeEngine(), prompt, dict(small), 300)
    assert no_pref is not None and no_pref.transport == "rpc"      # the size model's own answer

    asked = _plugin._decide_aperture_route(_FakeEngine(), prompt,
                                           dict(small, save_to_disk=True), 300)
    assert asked is not None and asked.transport == "disk"
    assert asked.analyze_where == no_pref.analyze_where            # only transport is forced


# --------------------------------------------------------------------------------------------
# 7. QK loses the bytes the same way HS did
#
# W13 is HS-only, so the fix in section 6 was applied to HS and the QK sibling guard deliberately
# left excluding `disk` -- recorded at the time as "whether QK loses bytes the same way is NOT
# established". It does, and by exactly the same three-way coincidence:
#   * `aperture_drain_qk.py`: `setup_sink=not per_request` -- no shared files, same as HS;
#   * `_takes_aperture_per_request` requires `wants_hs and not wants_qk`, so it is False for a QK
#     request, so finalize's `sink == "disk"` branch does NOT decline and flush_disk wins;
#   * flush_disk flushes the eager worker's `_disk_states`, empty because capture went to the
#     aperture.
# Same silent drop, one workload over.
# --------------------------------------------------------------------------------------------

def test_the_qk_drain_also_builds_no_shared_sink_under_per_request(per_request_armed):
    """The first ingredient, read off the QK drain rather than assumed from the HS one."""
    import inspect

    from mia.graph import aperture_drain_qk

    src = inspect.getsource(aperture_drain_qk)
    assert "setup_sink=not per_request" in src


def test_a_qk_disk_request_takes_the_aperture_path_not_flush_disk(per_request_armed):
    """The regression. A QK-only capture with a disk sink must be owned by the per-request
    aperture path, exactly as HS is -- otherwise flush_disk runs against empty eager buffers and
    the capture is dropped."""
    from mia import _plugin

    for sink_extra in ({"save_to_disk": True}, {"save_to_disk": False}, {}):
        extra = {"output_qk": {"0": [0, 1]}, **sink_extra}
        assert _plugin._takes_aperture_per_request(extra, False, True, False) is True, sink_extra


def test_the_two_paths_are_told_apart_so_qk_is_not_delivered_as_hs(per_request_armed):
    """Generalising the predicate must NOT let a QK request fall into the HS finalize branch,
    which returns hs_cache and would silently drop qk_cache. The KIND is what each branch gates
    on; the boolean only says 'some aperture path owns this request'."""
    from mia import _plugin

    hs = {"output_hidden_states": [1, 2]}
    qk = {"output_qk": {"0": [0]}}
    assert _plugin._aperture_per_request_kind(hs, True, False, False) == "hs"
    assert _plugin._aperture_per_request_kind(qk, False, True, False) == "qk"
    # a request that wants BOTH is owned by neither: it falls through to get_captured_states
    assert _plugin._aperture_per_request_kind({**hs, **qk}, True, True, False) is None
    assert _plugin._aperture_per_request_kind(hs, True, False, True) is None   # steering


def test_an_explicit_disk_ask_forces_the_disk_transport_for_qk_too(per_request_armed):
    """CAPTURE_IO.md 8.1 applies to both workers: an explicit ask is a requirement. QK crosses its
    (much lower) 100.5 KB threshold at almost any real size, so this mostly matters for a tiny
    capture -- which is exactly when the size model would say RPC and the caller would get no
    file."""
    from mia import _plugin

    prompt = {"prompt_token_ids": list(range(8))}      # tiny: one head, 8 tokens
    tiny = {"output_qk": {"0": [0]}, "hookq_mode": "last_token", "hooks_on": "prefill"}

    no_pref = _plugin._decide_aperture_route_qk(_FakeEngine(), prompt, dict(tiny), 1)
    assert no_pref is not None and no_pref.transport == "rpc"
    asked = _plugin._decide_aperture_route_qk(_FakeEngine(), prompt,
                                              dict(tiny, save_to_disk=True), 1)
    assert asked is not None and asked.transport == "disk"


def test_both_qk_sites_use_the_kind_predicate():
    """Same source check as HS: the QK arming guard and the QK finalize gate must both read the
    one predicate, so they cannot drift apart on `disk` the way the HS pair did."""
    import inspect

    from mia import _plugin

    gen = inspect.getsource(_plugin._patched_generate)
    assert gen.count('_aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "qk"') == 2
    assert gen.count('_aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "hs"') == 2
    assert 'and _resolve_sink(extra) not in ("drop", "disk")' not in gen


def test_no_module_level_guard_excludes_a_sink_behind_the_predicate_s_back():
    """The bug CLASS, closed rather than the two instances patched.

    Both instances had the same shape: a site that decides DELIVERY excluding a sink that a
    separate site had already decided. The sweep that closed it: `setup_sink=not per_request`
    appears in exactly two drains (HS `aperture_drain_hs.py`, QK `aperture_drain_qk.py`), so
    there are exactly two capture paths whose shared sink per-request mode disables, and both are
    now owned by `_aperture_per_request_kind`.

    What this test prevents is a THIRD: any new `_resolve_sink(...) not in (...)` exclusion in
    the plugin, which is how both instances were spelled. Sink exclusions belong in the predicate
    (where `drop` lives, with its reason) and nowhere else — a guard elsewhere can disagree with
    finalize, and when it does the bytes are dropped silently, which is how 47 GB went missing.
    """
    import inspect
    import re

    from mia import _plugin

    src = inspect.getsource(_plugin)
    # the only legitimate sink test outside the predicate is an EQUALITY (== "disk"/"drop"),
    # never a set exclusion -- an exclusion is a second opinion about who owns the request.
    offenders = re.findall(r"_resolve_sink\([^)]*\)\s+not\s+in\s*\([^)]*\)", src)
    assert not offenders, offenders


# --------------------------------------------------------------------------------------------
# 8. Two defects the QK GPU leg exposed, neither of them the delivery bug
#
# The QK cells confirmed the delivery fix (580-642 artifacts delivered where the bug would show
# zero) and then failed for two OTHER reasons, both silent:
#   A. the harness sends `output_qk` as a PYTHON REPR (`{'0': [0, 8]}`), which `json.loads`
#      rejects; the decoder did `continue` and the string survived. The capture WORKER parses it
#      anyway, so capture looked fine -- but the routers type-check and bail, so the request took
#      the RPC default whatever its size. HS escaped only because `[1, 2, 3]` is valid JSON.
#   B. the explicit-disk force sat AFTER the size prediction, so a request whose size could not
#      be predicted returned None early and fell back to RPC -- silently overriding an explicit
#      `save_to_disk: true`. The forced-DISK QK cell delivered 580 artifacts over RPC, and PASSED,
#      because delivery happened -- just not by the route the caller required.
# --------------------------------------------------------------------------------------------

def test_a_python_repr_dict_decodes_instead_of_surviving_as_a_string():
    """Defect A. `str(dict)` is single-quoted and is not JSON; the literal fallback catches it."""
    import ast
    import json

    # what the harness actually emits, versus what the JSON-only decoder could accept
    repr_form = "{'0': [0, 8], '1': [0, 8]}"
    with pytest.raises(ValueError):
        json.loads(repr_form)
    assert ast.literal_eval(repr_form) == {"0": [0, 8], "1": [0, 8]}
    # HS's form is valid JSON, which is why only QK was hit
    assert json.loads("[1, 2, 3]") == [1, 2, 3]


def test_the_decoder_tries_json_then_a_literal_then_passes_through():
    """The order matters and the pass-through must survive: a bare word like an hs_mode value is
    not a structure and must reach the worker unchanged."""
    import inspect

    from mia import _plugin

    src = inspect.getsource(_plugin._patched_generate)
    assert "_ast.literal_eval(extra[_k])" in src
    assert "isinstance(_decoded, (dict, list, tuple, bool, int, float))" in src


def test_an_explicit_disk_ask_does_not_depend_on_predicting_a_size(per_request_armed):
    """Defect B. Both routers must honour a settled disk sink BEFORE they try to price the
    request -- otherwise an unpredictable size silently downgrades a requirement to RPC."""
    from mia import _plugin

    # a QK request the size model CANNOT price (output_qk not a mapping -> head_layers 0)
    unpriceable = {"output_qk": "whole-model", "save_to_disk": True}
    d = _plugin._decide_aperture_route_qk(_FakeEngine(), {"prompt_token_ids": [1, 2, 3]},
                                          dict(unpriceable), 8)
    assert d is not None and d.transport == "disk", "an explicit disk ask was downgraded to RPC"

    # and an HS request with no prompt length, which also bails before the model
    d2 = _plugin._decide_aperture_route(_FakeEngine(), {"prompt_token_ids": []},
                                        {"output_hidden_states": [1], "save_to_disk": True}, 8)
    assert d2 is not None and d2.transport == "disk"


def test_every_bail_out_falls_back_to_the_force_rather_than_to_rpc():
    """Same anti-drift rule as the delivery predicate: one helper, both routers. The shape that
    matters is FALLBACK, not short-circuit -- every path that cannot price a request returns the
    helper (which is None unless a disk sink was pinned), and no path returns a bare None except
    the "this is not my worker" gate at the top.

    An earlier revision consulted the helper FIRST. That honoured the disk ask but quietly threw
    away `analyze_from_disk` for every reducible disk-sink capture, because it never ran the size
    model at all. Inert today (ServerAnalyzeProcess is deferred) and still wrong, so the ordering
    is pinned here rather than left to be rediscovered."""
    import inspect

    from mia import _plugin

    for fn in (_plugin._decide_aperture_route, _plugin._decide_aperture_route_qk):
        src = inspect.getsource(fn)
        body = src[src.index("_prompt_token_len(prompt)"):]
        assert "_explicit_disk_route(extra)" in body, fn.__name__
        # after the ownership gate, nothing may bail to a bare None -- that means "RPC default"
        assert "return None" not in body, f"{fn.__name__}: a bail-out still downgrades to RPC"
        # and the size-model path forces only the TRANSPORT, keeping analyze_where
        assert 'RouteDecision("disk", decision.analyze_where)' in src, fn.__name__


def test_a_priceable_disk_request_keeps_the_analyze_decision(per_request_armed):
    """The regression the rewrite above prevents: when the size IS predictable, forcing disk must
    not discard what the model decided about reducing the artifact."""
    from mia.graph.delivery_router import decide_route

    # a reducible capture large enough to stream: the model says disk + analyze from_disk
    d = decide_route(50 << 20, True, 1 << 19, 1 << 19)
    assert (d.transport, d.analyze_where) == ("disk", "from_disk")
    # forcing the transport must preserve that, which is what the call sites do
    from mia.graph.delivery_router import RouteDecision
    assert RouteDecision("disk", d.analyze_where).analyze_where == "from_disk"
