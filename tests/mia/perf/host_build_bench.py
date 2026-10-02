"""Hermetic (CPU-only) bench for MIA's on-loop routing-table builders (Task E2).

MIA's only on-loop (critical-path) host cost is the per-step routing build that maps
each capturing request's batch columns to capture-aperture slots: HS's
``_build_routing_hs`` (``mia/graph/install_hs.py``, with a vectorized twin
``_build_routing_hs_vectorized`` and a cached decode fast-path
``_build_routing_hs_decode_cache``), QK's ``_build_routing`` (``mia/graph/install.py``),
and steer's ``_build_routing_steer`` (``mia/graph/install_steer.py``). This module
measures all three against running-batch width and layer count, and decomposes HS's
build into its two suspected O(N) components (``ReqCaptureRecord`` construction vs. the
plane-fill torch scatter), on the CLAIM under test:

    "build time grows linearly with concurrent capturing requests, flat in layers,
     and the vectorized (MIA_ROUTE_VECTORIZED=1) plane-fill scatter collapses it."

``HostRegistry`` (``mia/graph/registry.py:114``) and ``CaptureAperture``
(``mia/graph/capture_aperture.py``) are both plain CPU-constructible (no CUDA calls
unless ``.alloc_gpu()`` is called, which nothing here calls), so this whole bench runs
on the login node with no GPU, no vLLM engine, and no ``tests/mia/parity`` fixture. Per
Ruling E-5, ``make_fake_step`` lives HERE, not in ``tests/mia/parity/capture_workload.py``
(that module is parity-only and must not grow a performance helper).

Run:
    python tests/mia/perf/host_build_bench.py

See ``tests/mia/perf/host_build_bench_results.md`` (committed alongside this file) for the
last recorded run and the verdict it produced.
"""
from __future__ import annotations

import os

# MUST be set before numpy/torch import: this login node has 128 cores, and without
# these OpenBLAS/MKL spins up a full thread pool the first time an op's array crosses
# some internal size threshold -- measured directly here as a ~20x cliff in HS
# "vectorized" and "decode_cache" timings jumping from width=64 to width=128 that
# vanished the moment these were set (same cliff this repo's memory already names:
# "OpenBLAS thread storm" -- see the Env-hygiene global constraint). `setdefault` so a
# caller's own explicit env still wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gc
import time
from typing import Callable, Optional

import numpy as np
import torch

torch.set_num_threads(1)  # belt-and-suspenders: torch's own pool, separate from the env vars above

from mia.graph.aperture_metadata import ReqCaptureRecord
from mia.graph.capture_aperture import CaptureAperture
from mia.graph.install import _build_routing as build_routing_qk
from mia.graph.install_hs import (
    _build_routing_hs as build_routing_hs,
    _build_routing_hs_decode_cache as build_routing_hs_decode_cache,
    _build_routing_hs_vectorized as build_routing_hs_vectorized,
)
from mia.graph.install_steer import SteerRegistry
from mia.graph.install_steer import _build_routing_steer as build_routing_steer
from mia.graph.registry import HostRegistry
from mia.runner import StepView

WIDTHS = (1, 8, 32, 64, 128, 200)
LAYER_COUNTS = (1, 8, 16, 32)
DEFAULT_LAYERS = 32
DEFAULT_WIDTH_FOR_LAYER_SWEEP = 64
ITERS = 200
CAP = 4096            # generous column width; no fixture request ever nears it
N_SLOTS = 16384        # aperture rows; drained every iteration so it never fills

# For the all_tokens / large-token-span regime (E2 follow-up round 2): a capturing
# request now reserves/writes `tokens_per_req` rows/columns instead of 1, so the widest
# sweep point (200 reqs x 64 tokens = 12800) must fit both the aperture (N_SLOTS, already
# 16384) and the routing plane's column width. CAP stays 4096 for every OTHER section
# (unchanged, so previously-recorded numbers there stay reproducible) -- only the
# all_tokens sweep uses this larger cap.
TOKENS_PER_REQ_ALL_TOKENS = 64
CAP_ALL_TOKENS = 16384


# ---------------------------------------------------------------------------
# Fixture builders (hermetic: no vLLM engine, no CUDA).
# ---------------------------------------------------------------------------


def make_fake_step(width: int, *, num_layers: int = DEFAULT_LAYERS,
                    phase: str = "prefill", tokens_per_req: int = 8,
                    req_id_base: int = 0, subsystem: str = "hs",
                    hs_mode: str = "last_token") -> StepView:
    """Build a ``StepView`` of ``width`` concurrently-capturing requests.

    ``phase="prefill"`` gives each call FRESH request ids -- a churn workload (every
    request is new, as during a burst of short prompts). ``phase="decode"`` gives
    DETERMINISTIC ids (``req_id_base .. req_id_base+width``) with ``is_prefilling=False``
    and exactly one new token per request, so repeated calls at the SAME ``req_id_base``
    model steady-state decode of the SAME batch -- the shape the HS decode-cache fast
    path targets (a cache hit needs the same request, not a prefill, exactly one new
    token). Each request is a capturing request (the builders' idle-skip is not what is
    being measured here).

    ``hs_mode`` (HS only) selects the reserve/scatter shape ``_build_routing_hs``
    computes: ``"last_token"`` reserves ONE aperture row per request regardless of
    span (the registry default, and what the original E2 sweep used exclusively);
    ``"all_tokens"`` reserves the WHOLE ``tokens_per_req``-token span per request, so
    both the aperture reserve and the routing-plane scatter scale with tokens, not
    just request count -- a materially different regime for the vectorized flag's A/B
    (E2 follow-up round 2), since ``_build_routing_hs_vectorized``'s win is a
    plane-fill scatter it amortizes across the WHOLE step's rows. During
    ``phase="decode"`` a request always contributes exactly one new token, so
    ``last_token`` and ``all_tokens`` coincide there (same as production: see
    ``steer_span``'s docstring) -- the regime only diverges during ``phase="prefill"``.
    """
    req_ids = [f"bench{req_id_base + i}" for i in range(width)]
    if phase == "prefill":
        n_tok = np.full(width, tokens_per_req, dtype=np.int32)
        is_prefill = np.ones(width, dtype=bool)
        num_computed = np.zeros(width, dtype=np.int32)
    elif phase == "decode":
        n_tok = np.ones(width, dtype=np.int32)
        is_prefill = np.zeros(width, dtype=bool)
        num_computed = np.full(width, tokens_per_req, dtype=np.int32)
    else:
        raise ValueError(f"unknown phase {phase!r}")

    qsl = np.zeros(width + 1, dtype=np.int64)
    qsl[1:] = np.cumsum(n_tok)

    if subsystem == "hs":
        extra = {rid: {"output_hidden_states": True, "hooks_on": "both",
                       "hs_mode": hs_mode} for rid in req_ids}
    elif subsystem == "qk":
        extra = {rid: {"output_qk": True, "hooks_on": "both",
                       "hookq_mode": "all_tokens"} for rid in req_ids}
    elif subsystem == "steer":
        extra = {rid: {"steer": {"method": "add_vector", "optimal_layer": "all",
                                 "vector_path": "bench-vec", "coefficient": 1.0}}
                for rid in req_ids}
    else:
        raise ValueError(f"unknown subsystem {subsystem!r}")

    return StepView(
        req_ids=req_ids,
        num_reqs=width,
        num_scheduled_tokens=n_tok,
        query_start_loc=torch.from_numpy(qsl),
        query_start_loc_np=qsl,
        num_computed_tokens_np=num_computed,
        prefill_len_np=n_tok.copy(),
        prompt_len_np=n_tok.copy(),
        is_prefilling_np=is_prefill,
        seq_lens=torch.from_numpy((num_computed + n_tok).astype(np.int64)),
        block_tables=(),
        extra_args=extra,
    )


def make_hs_registry(num_layers: int, cap: int = CAP, n_slots: int = N_SLOTS) -> HostRegistry:
    reg = HostRegistry(num_layers=num_layers, cap=cap, device="cpu", should_capture=True)
    aperture = CaptureAperture(row_bytes=8, n_slots=n_slots, device="cpu")
    reg._hs_aperture = aperture
    reg._hs_consumer = None
    reg._hs_step_entries = []
    reg._hs_step_start = None
    reg._hs_step_rows = 0
    reg._default_hooks_on = "prefill"
    reg._worker_hs_mode = "last_token"
    return reg


def make_qk_registry(num_layers: int, cap: int = CAP, n_slots: int = N_SLOTS) -> HostRegistry:
    reg = HostRegistry(num_layers=num_layers, cap=cap, device="cpu", should_capture=True)
    aperture = CaptureAperture(row_bytes=8, n_slots=n_slots, device="cpu")
    reg._qk_aperture = aperture
    reg._qk_consumer = None
    reg._qk_step_entries = []
    reg._qk_step_start = None
    reg._qk_step_rows = 0
    reg._default_hooks_on = "prefill"
    reg._worker_hookq_mode = "all_tokens"
    reg._worker_score_mode = False
    return reg


def make_steer_registry(num_layers: int, cap: int = CAP) -> SteerRegistry:
    reg = SteerRegistry(num_layers=num_layers, cap=cap, hidden=8, v_max=4,
                        device="cpu", dtype=torch.float32)
    # Pre-seed the vector table so `vec_id_for_path` short-circuits on its first call
    # (this bench times ROUTING, not vector-file loading -- see `vec_id_for_path`).
    reg.vec_paths["bench-vec"] = 0
    return reg


def _timed_ms(fn: Callable[[], None], iters: int, repeats: int = 5) -> float:
    """Median-of-``repeats`` timing, ``iters`` calls per repeat block, GC disabled
    during each block.

    This is a shared, multi-tenant login node -- a single wall-clock block can be
    blown out by an unrelated process's scheduling burst or (with many small-object
    allocations per iteration, as these builders do) a GC pause. Taking the MEDIAN
    across independent repeat blocks discards a one-off bad block; disabling the
    cyclic GC during each block removes its jitter without changing what is measured
    (nothing here creates reference cycles the collector would need to break).
    """
    samples = []
    for _ in range(repeats):
        was_enabled = gc.isenabled()
        gc.disable()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = time.perf_counter() - t0
        if was_enabled:
            gc.enable()
        samples.append(dt / iters * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def _drain(aperture: CaptureAperture) -> None:
    """Advance the drain cursor to the write cursor so the aperture never fills across
    a long timing loop -- the off-loop consumer's job in production, done synchronously
    here since this bench never touches the device buffer the aperture guards."""
    aperture.advance_drain(aperture._write - aperture._drain)


# ---------------------------------------------------------------------------
# Timed builder calls.
# ---------------------------------------------------------------------------


def time_hs(width: int, num_layers: int = DEFAULT_LAYERS, iters: int = ITERS,
            phase: str = "prefill", vectorized: Optional[bool] = None,
            decode_cache: Optional[bool] = None, warm: bool = False,
            repeats: int = 5, hs_mode: str = "last_token",
            tokens_per_req: int = 8, cap: int = CAP) -> float:
    """ms/step for ``_build_routing_hs`` at a fixed ``(vectorized, decode_cache)``.

    ``warm=True`` runs ONE untimed call first to populate the decode cache for these
    exact request ids before the timed loop -- required to measure the decode-cache
    FAST path rather than its cold first-touch (== slow-path) cost. The SAME registry
    (and its warmed cache) is reused across every repeat block: only fresh-registry
    setup is excluded from the timing, not cache warmth.

    ``hs_mode``/``tokens_per_req``/``cap`` select the regime (see ``make_fake_step``);
    ``cap`` must be >= ``width * tokens_per_req`` or requests past the column cap get
    silently truncated by the builder itself (the same ``end = min(end, cap)`` a real
    ``max_num_batched_tokens`` budget applies), which would understate the width.
    """
    reg = make_hs_registry(num_layers, cap=cap)
    step = make_fake_step(width, num_layers=num_layers, phase=phase, subsystem="hs",
                          hs_mode=hs_mode, tokens_per_req=tokens_per_req)
    if warm:
        build_routing_hs(step, reg, vectorized=vectorized, decode_cache=decode_cache)
        _drain(reg._hs_aperture)

    def _call() -> None:
        build_routing_hs(step, reg, vectorized=vectorized, decode_cache=decode_cache)
        _drain(reg._hs_aperture)

    return _timed_ms(_call, iters, repeats)


def time_qk(width: int, num_layers: int = DEFAULT_LAYERS, iters: int = ITERS,
            phase: str = "prefill", repeats: int = 5,
            tokens_per_req: int = 8, cap: int = CAP) -> float:
    """QK has no ``hs_mode``-equivalent regime switch -- ``_build_routing`` always
    reserves/scatters the WHOLE span every step regardless of ``hookq_mode`` (which
    only gates Q-emission metadata, not the K scatter every step pays); ``tokens_per_req``
    is QK's one lever on token-span size. See ``make_fake_step`` re: ``cap`` sizing."""
    reg = make_qk_registry(num_layers, cap=cap)
    step = make_fake_step(width, num_layers=num_layers, phase=phase, subsystem="qk",
                          tokens_per_req=tokens_per_req)

    def _call() -> None:
        build_routing_qk(step, reg)
        _drain(reg._qk_aperture)

    return _timed_ms(_call, iters, repeats)


def time_steer(width: int, num_layers: int = DEFAULT_LAYERS, iters: int = ITERS,
               phase: str = "prefill", repeats: int = 5) -> float:
    reg = make_steer_registry(num_layers)
    step = make_fake_step(width, num_layers=num_layers, phase=phase, subsystem="steer")

    def _call() -> None:
        build_routing_steer(step, reg)

    return _timed_ms(_call, iters, repeats)


# ---------------------------------------------------------------------------
# Decomposition: isolate the two suspected O(N) components of the HS build.
# ---------------------------------------------------------------------------


def time_record_construction(width: int, num_layers: int = DEFAULT_LAYERS,
                              iters: int = ITERS, repeats: int = 5) -> float:
    """Isolates ONLY the ``ReqCaptureRecord`` allocation: ``width`` records/iter, each
    carrying a ``num_layers``-length ``layers`` list. Both the legacy AND the vectorized
    HS builder append exactly one of these per capturing request (the "LayerEntry
    collapse" already folded the old per-(request, layer) fan-out into this one object
    per request) -- this is the term ``_route_vectorized_enabled``'s docstring names as
    the dominant O(N) cost shared by both paths.
    """
    layers_tmpl = [L + 1 for L in range(num_layers)]

    def _call() -> None:
        records = [ReqCaptureRecord(req_id=f"bench{i}", logical_start=i, n_rows=1,
                                    hs_mode="last_token", layers=layers_tmpl)
                  for i in range(width)]
        assert len(records) == width

    return _timed_ms(_call, iters, repeats)


def time_scatter_per_request(width: int, num_layers: int = DEFAULT_LAYERS,
                             iters: int = ITERS, repeats: int = 5) -> float:
    """Isolates ONLY the legacy per-request torch scatter: ``width`` individual
    ``torch.tensor(...)`` + advanced-index writes into a ``(num_layers, cap)`` plane --
    what the vectorized builder amortizes into ONE assign at the end of the step."""
    plane = torch.zeros(num_layers, CAP, dtype=torch.int64)
    rows_layers = list(range(num_layers))

    def _call() -> None:
        for i in range(width):
            layer_idx_t = torch.tensor(rows_layers, dtype=torch.long)
            plane[layer_idx_t, i % CAP] = i

    return _timed_ms(_call, iters, repeats)


def time_scatter_batched(width: int, num_layers: int = DEFAULT_LAYERS,
                         iters: int = ITERS, repeats: int = 5) -> float:
    """Isolates ONLY the vectorized ONE-shot scatter: flatten ``width`` requests'
    (layer, col, slot) triples with numpy, then ONE advanced-index write -- the
    plane-fill scatter ``_build_routing_hs_vectorized`` amortizes."""
    plane = torch.zeros(num_layers, CAP, dtype=torch.int64)
    rows_layers = np.arange(num_layers, dtype=np.int64)

    def _call() -> None:
        cols = np.arange(width, dtype=np.int64) % CAP
        rows_acc = np.tile(rows_layers, width)
        cols_acc = np.repeat(cols, num_layers)
        slots_acc = np.repeat(cols, num_layers)
        plane[torch.from_numpy(rows_acc), torch.from_numpy(cols_acc)] = torch.from_numpy(slots_acc)

    return _timed_ms(_call, iters, repeats)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _slope(widths, timings) -> tuple:
    """Least-squares ``ms = a + b*width`` fit; returns ``(b, r2)``. ``b`` is ms/step
    per ADDITIONAL concurrent capturing request -- 0 (flat) at one extreme, and (for a
    per-request torch-dispatch cost) a small positive constant at the other."""
    x = np.asarray(widths, dtype=np.float64)
    y = np.asarray(timings, dtype=np.float64)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return b, r2


def _print_series(title: str, widths, timings) -> None:
    b, r2 = _slope(widths, timings)
    print(f"\n{title}")
    for w, t in zip(widths, timings):
        print(f"  width={w:4d}  {t:9.4f} ms/step")
    print(f"  slope (least-squares ms/step per +1 req): {b*1000:8.4f} us/req   R^2={r2:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--quick", action="store_true", help="fewer iters, for a fast smoke run")
    args = p.parse_args()
    iters = 20 if args.quick else args.iters

    print("=" * 78)
    print("Task E2 host-build bench -- hermetic CPU run")
    print(f"iters={iters}  widths={WIDTHS}  layer_counts={LAYER_COUNTS}"
         f"  default_layers={DEFAULT_LAYERS}")
    print("=" * 78)

    # ---- 1. Slope: width sweep at DEFAULT_LAYERS, one series per builder/config. ----
    print("\n### 1. SLOPE vs concurrent capturing requests (num_layers=%d) ###" % DEFAULT_LAYERS)

    legacy_prefill = [time_hs(w, iters=iters, phase="prefill", vectorized=False,
                              decode_cache=False) for w in WIDTHS]
    _print_series("HS legacy loop (churn/prefill, decode_cache=OFF)", WIDTHS, legacy_prefill)

    vectorized_prefill = [time_hs(w, iters=iters, phase="prefill", vectorized=True,
                                  decode_cache=False) for w in WIDTHS]
    _print_series("HS vectorized (churn/prefill, decode_cache=OFF)", WIDTHS, vectorized_prefill)

    default_decode = [time_hs(w, iters=iters, phase="decode", vectorized=False,
                              decode_cache=True, warm=True) for w in WIDTHS]
    _print_series("HS PRODUCTION DEFAULT (steady decode, decode_cache=ON, warm)",
                  WIDTHS, default_decode)

    qk_prefill = [time_qk(w, iters=iters, phase="prefill") for w in WIDTHS]
    _print_series("QK legacy loop (churn/prefill) -- only builder QK has", WIDTHS, qk_prefill)

    steer_prefill = [time_steer(w, iters=iters, phase="prefill") for w in WIDTHS]
    _print_series("Steer legacy loop (churn/prefill) -- only builder steer has",
                  WIDTHS, steer_prefill)

    # ---- 2. Layers sweep at a fixed width: is it really flat in layers? ----
    print("\n### 2. LAYERS sweep at width=%d (churn/prefill) ###" % DEFAULT_WIDTH_FOR_LAYER_SWEEP)
    w = DEFAULT_WIDTH_FOR_LAYER_SWEEP

    hs_by_layers = [time_hs(w, num_layers=L, iters=iters, phase="prefill",
                            vectorized=False, decode_cache=False) for L in LAYER_COUNTS]
    _print_series("HS legacy loop vs num_layers", LAYER_COUNTS, hs_by_layers)

    hs_vec_by_layers = [time_hs(w, num_layers=L, iters=iters, phase="prefill",
                                vectorized=True, decode_cache=False) for L in LAYER_COUNTS]
    _print_series("HS vectorized vs num_layers", LAYER_COUNTS, hs_vec_by_layers)

    qk_by_layers = [time_qk(w, num_layers=L, iters=iters, phase="prefill") for L in LAYER_COUNTS]
    _print_series("QK legacy loop vs num_layers", LAYER_COUNTS, qk_by_layers)

    steer_by_layers = [time_steer(w, num_layers=L, iters=iters, phase="prefill")
                       for L in LAYER_COUNTS]
    _print_series("Steer legacy loop vs num_layers", LAYER_COUNTS, steer_by_layers)

    # ---- 3. Decomposition: which O(N) term dominates the HS build? ----
    print("\n### 3. DECOMPOSITION at width=%d, num_layers=%d ###"
         % (DEFAULT_WIDTH_FOR_LAYER_SWEEP, DEFAULT_LAYERS))
    w, L = DEFAULT_WIDTH_FOR_LAYER_SWEEP, DEFAULT_LAYERS
    rec_ms = time_record_construction(w, L, iters)
    scat_per_req_ms = time_scatter_per_request(w, L, iters)
    scat_batched_ms = time_scatter_batched(w, L, iters)
    full_legacy_ms = time_hs(w, num_layers=L, iters=iters, phase="prefill",
                             vectorized=False, decode_cache=False)
    full_vec_ms = time_hs(w, num_layers=L, iters=iters, phase="prefill",
                          vectorized=True, decode_cache=False)
    print(f"  ReqCaptureRecord construction only:      {rec_ms:9.4f} ms/step")
    print(f"  Per-request torch scatter only (legacy): {scat_per_req_ms:9.4f} ms/step")
    print(f"  Batched numpy+torch scatter only (vec):  {scat_batched_ms:9.4f} ms/step")
    print(f"  Full legacy builder:                     {full_legacy_ms:9.4f} ms/step")
    print(f"  Full vectorized builder:                 {full_vec_ms:9.4f} ms/step")
    print(f"  legacy - vectorized (measured scatter win):  {full_legacy_ms - full_vec_ms:9.4f} ms/step")
    print(f"  isolated scatter win (per-req - batched):    {scat_per_req_ms - scat_batched_ms:9.4f} ms/step")
    rec_share = rec_ms / full_vec_ms * 100 if full_vec_ms else float("nan")
    print(f"  record construction as %% of the vectorized total: {rec_share:5.1f}%%")

    # ---- 4. The A/B: legacy vs vectorized x decode_cache ON vs OFF, matched width. ----
    print("\n### 4. A/B at width=%d, num_layers=%d, steady DECODE (warm) ###"
         % (DEFAULT_WIDTH_FOR_LAYER_SWEEP, DEFAULT_LAYERS))
    w, L = DEFAULT_WIDTH_FOR_LAYER_SWEEP, DEFAULT_LAYERS
    combos = [
        ("decode_cache=OFF, vectorized=OFF (legacy)", False, False),
        ("decode_cache=OFF, vectorized=ON",           True,  False),
        ("decode_cache=ON,  vectorized=OFF (PRODUCTION DEFAULT)", False, True),
        ("decode_cache=ON,  vectorized=ON  (vectorized is ignored -- decode_cache checked first)",
         True, True),
    ]
    for label, vec, dc in combos:
        ms = time_hs(w, num_layers=L, iters=iters, phase="decode",
                    vectorized=vec, decode_cache=dc, warm=True)
        print(f"  {label:75s} {ms:9.4f} ms/step")

    # ---- 5. ALL_TOKENS regime (E2 follow-up round 2): does the verdict change when ----
    # the plane-fill scatter scales with TOKENS, not just requests? Only meaningful during
    # phase="prefill" -- a decode step always contributes exactly one new token, so
    # last_token and all_tokens coincide there (see make_fake_step's docstring).
    print("\n### 5. ALL_TOKENS regime: width sweep, num_layers=%d, tokens_per_req=%d, churn/prefill ###"
         % (DEFAULT_LAYERS, TOKENS_PER_REQ_ALL_TOKENS))

    all_tok_legacy = [time_hs(w, iters=iters, phase="prefill", vectorized=False,
                              decode_cache=False, hs_mode="all_tokens",
                              tokens_per_req=TOKENS_PER_REQ_ALL_TOKENS, cap=CAP_ALL_TOKENS)
                     for w in WIDTHS]
    _print_series("HS legacy loop, hs_mode=all_tokens (decode_cache=OFF)", WIDTHS, all_tok_legacy)

    all_tok_vectorized = [time_hs(w, iters=iters, phase="prefill", vectorized=True,
                                  decode_cache=False, hs_mode="all_tokens",
                                  tokens_per_req=TOKENS_PER_REQ_ALL_TOKENS, cap=CAP_ALL_TOKENS)
                         for w in WIDTHS]
    _print_series("HS vectorized, hs_mode=all_tokens (decode_cache=OFF)", WIDTHS, all_tok_vectorized)

    all_tok_default = [time_hs(w, iters=iters, phase="prefill", vectorized=False,
                               decode_cache=True, hs_mode="all_tokens",
                               tokens_per_req=TOKENS_PER_REQ_ALL_TOKENS, cap=CAP_ALL_TOKENS)
                      for w in WIDTHS]
    _print_series("HS PRODUCTION DEFAULT, hs_mode=all_tokens, churn/prefill "
                 "(decode_cache's SLOW-PATH body runs every call -- prefill never hits its "
                 "fast path)", WIDTHS, all_tok_default)

    qk_all_tok = [time_qk(w, iters=iters, phase="prefill",
                          tokens_per_req=TOKENS_PER_REQ_ALL_TOKENS, cap=CAP_ALL_TOKENS)
                 for w in WIDTHS]
    _print_series("QK legacy loop, tokens_per_req=%d (QK has no hs_mode-equivalent "
                 "switch -- always full-span; this is its token-span sensitivity check)"
                 % TOKENS_PER_REQ_ALL_TOKENS, WIDTHS, qk_all_tok)

    b_legacy, _ = _slope(WIDTHS, all_tok_legacy)
    b_vec, _ = _slope(WIDTHS, all_tok_vectorized)
    print(f"\n  all_tokens legacy slope:     {b_legacy*1000:8.4f} us/req")
    print(f"  all_tokens vectorized slope: {b_vec*1000:8.4f} us/req")
    print(f"  vectorized/legacy slope ratio: {b_vec/b_legacy:.3f}"
         f"  (last_token regime ratio was 6.41/10.72 = 0.598)")

    print("\nDone. See tests/mia/perf/host_build_bench_results.md for the recorded verdict.")


if __name__ == "__main__":
    main()
