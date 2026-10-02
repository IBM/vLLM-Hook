"""MIA's own invariants — the ones no external reference can check.

(a) **eager == FULL**: the in-graph aperture scatter op and its replay-time routing must
    reproduce what the plain forward-hook path captures. Bit-exact.
(b) **alone == in-batch**: a request captured by itself must be captured identically when it
    shares the step with 31 others. This is the indexing killer — V2's padded rows and
    ``idx_mapping`` are exactly where a batched mis-route hides, and both the T1 reference
    (one request resident at a time) and any frozen golden would miss it.

Three things this file has to get right that the plan's sketch could not know yet.

**1. One engine per process.** Every arm boots its own engine, and an LSF
``exclusive_process`` GPU rejects a second engine in the same process. So the module is
BOTH an orchestrator (``main``) and a single arm (``arm``): the orchestrator holds no
engine, spawns one subprocess per arm, and does every comparison afterwards from files.
That is also what lets each arm carry its own environment — the routing levers below are
read ONCE at install, so a lever A/B is only expressible as separate processes.

**2. The two arms do not retrieve capture the same way, and cannot.** Eager capture lands in
the worker's host buckets and comes back as ``output.probes`` (``get_captured_states``).
FULL-graph capture never touches those buckets: the baked op scatters into the GPU capture
aperture, an off-loop drain streams it to per-layer raw files, and the artifact is read back
with ``flush_aperture`` + ``mia.graph.aperture_reader``. Comparing ``probes`` to ``probes``
would silently compare an empty tree against an empty tree — the exact false PASS
``compare_artifacts`` exists to stop. So each arm is normalized to ONE canonical form:

    HS  ->  hidden_states  [n_captured_rows, hidden]
    QK  ->  q [n_q_rows, H_q*d], k_full [n_k_rows, H_kv*d], k_prefix_ends [n_steps]

flat, unpadded, in capture order. The eager side gets there by UN-padding the worker's
``pad_sequence`` stack; the graph side is already flat. Normalization is lossless on both
sides — no reduction, no tolerance — and the row count is asserted against the analytic
expectation (``n_prompt + n_gen - 1``), which is also the concrete proof that D3's
stale-routing guard works: a warmup/dummy pass that leaked into the aperture shows up here
as extra rows.

**3. ``cudagraph_mode=FULL`` is not the default.** vLLM 0.29 resolves to
``FULL_AND_PIECEWISE`` and MIA refuses it; ``capture_workload.run_workload`` names FULL
explicitly for every graph arm.

Run:  ``python tests/mia/parity/t2_invariants.py <workload> <out_root>``
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.capture_workload import (  # noqa: E402
    WORKLOADS,
    _save,
    generation_tensors,
    run_workload,
)
from tests.mia.parity.compare_artifacts import compare, steer_logprob_delta  # noqa: E402


# ---------------------------------------------------------------------------
# BAND REGISTRY — a bound that can be widened from the environment, silently, is not a
# bound at all (task D5 item 4).
# ---------------------------------------------------------------------------
# The structural review showed this run passing:
#
#   MIA_T2_T0_GRAPH_BAND=10 MIA_T2_STEER_FUSED_BAND=10 MIA_T2_GPU_ROUTING_BAND=10 pytest
#     -> 88 passed
#
# and at `STEER_FUSED_BAND=10` a totally broken fused steer -- one applying a whole 7.7e-01
# intervention to the wrong rows -- passes too. `run_parity.sh` neither pinned nor echoed
# any of them, so a log recorded nothing about what was actually enforced.
#
# Every bound in this file is therefore declared through `_band()`, which:
#   * keeps the DOCUMENTED value in the source, next to the derivation, as the only default;
#   * REFUSES to run when an override WIDENS it (`MIA_T2_ALLOW_BAND_OVERRIDE=1` is the
#     deliberate, loud escape hatch, and it is printed on every band line);
#   * allows a TIGHTENING override, because a stricter gate cannot manufacture a pass;
#   * records what it resolved to, so `--print-bands` can put the whole enforced set in the
#     job log before any engine boots.
_BANDS: dict[str, dict] = {}


def _band(env_var: str, documented: str, what: str) -> float:
    """One band constant: documented default, refuse-to-widen override, recorded for the log."""
    doc = float(documented)
    raw = os.environ.get(env_var)
    value = doc if raw is None else float(raw)
    if raw is None:
        source = "documented default"
    elif value > doc:
        if os.environ.get("MIA_T2_ALLOW_BAND_OVERRIDE") != "1":
            raise RuntimeError(
                f"REFUSING TO RUN: {env_var}={raw} WIDENS the documented band {doc:.3e} "
                f"for {what}. That bound is derived from measurement and its margins are "
                f"written beside it in tests/mia/parity/t2_invariants.py; widening it from the "
                f"environment makes the suite's PASS mean something other than what the "
                f"derivation says. Tighten it freely, or set MIA_T2_ALLOW_BAND_OVERRIDE=1 "
                f"to override deliberately -- every band line then says so in the log.")
        source = f"WIDENED from {doc:.3e} by {env_var} (MIA_T2_ALLOW_BAND_OVERRIDE=1)"
    elif value < doc:
        source = f"TIGHTENED from {doc:.3e} by {env_var}"
    else:
        source = f"{env_var} set to the documented value"
    _BANDS[env_var] = {"documented": doc, "value": value, "source": source, "what": what}
    return value


def print_bands() -> None:
    """Echo every enforced bound. Called by run_parity.sh BEFORE any arm boots.

    A green run whose log does not say what it enforced is not evidence of anything.
    """
    print(f"[T2] band registry: {len(_BANDS)} enforced bounds", flush=True)
    for env_var in sorted(_BANDS):
        rec = _BANDS[env_var]
        print(f"[T2] BAND {env_var} = {rec['value']:.6e} "
              f"(documented {rec['documented']:.6e}; {rec['source']}) -- {rec['what']}",
              flush=True)


# ---------------------------------------------------------------------------
# The leg matrix. One entry == one engine process.
# ---------------------------------------------------------------------------
# `width` is the RUNNING BATCH: `batch` repeats the 3-prompt set and `max_num_seqs` caps
# residency, so batch=11 + max_num_seqs=32 puts req0 in a step with 31 other requests and
# leaves one request over to run after them (the scheduler admits in arrival order, so req0
# is always in the first, full-width wave).
#
# `env` is per-leg because every routing lever below is read ONCE at install and captured in
# the wrapper closure — it cannot be flipped inside a process.
_BATCH_REPEATS = 11
_BATCH_SEQS = 32


def leg(graph, batch=1, seqs=1, cudagraph=None, env=None, arm="mia"):
    """One engine process.

    ``graph``     — MIA arms its BAKED-OP + capture-aperture path (MIA_ALLOW_CUDAGRAPH).
    ``cudagraph`` — vLLM compiles the model and replays CUDA graphs (defaults to ``graph``).

    Separating the two is what makes this suite able to ANSWER its own invariants rather
    than only measure them. `enforce_eager=True` makes vLLM skip torch.compile entirely
    ("Enforce eager set, disabling torch.compile and CUDAGraphs"), so an eager arm and a
    FULL arm do not run the same kernels — their hidden states differ before MIA touches
    anything, and a bit-exact eager-vs-FULL capture comparison is measuring vLLM's
    compiled-vs-eager numerics, not MIA's capture path. `graph=True, cudagraph=False` runs
    MIA's in-graph op against the SAME uncompiled kernels the forward-hook path runs
    against, which isolates the capture mechanism exactly.

    ``arm="off"`` is the capture-OFF control: no plugin at all (tests/mia/parity/t1_reference).
    """
    return {"graph": graph, "batch": batch, "seqs": seqs, "arm": arm,
            "cudagraph": graph if cudagraph is None else cudagraph,
            "env": env or {}}


# Carried item 2 of the task brief: D1 converted FIVE routing builders to read `StepView`
# and unit-tested only the wrapper plumbing with lambda stand-ins. The `full_batch_*` legs
# put each one on a real GPU under real prefill+decode at width 32 and check them against
# each other -- which is the builders' own contract (they differ in HOW the plane is built,
# never in what it contains). Those comparisons are clean: both sides are FULL-graph runs of
# the same batch, so the kernels and the composition match and any difference is the
# builder. BANDED as of task D4f -- see `_BUILDER_EQUIV_REL_BAND`'s comment: a bit-exact
# requirement here was never achievable, because every builder disagrees with ITSELF
# boot-to-boot at the same tail requests (LSF 1722805).
_CAPTURE_LEGS = {
    "eager_alone":         leg(False),
    "eager_batch":         leg(False, _BATCH_REPEATS, _BATCH_SEQS),
    "full_alone":          leg(True),
    "full_batch":          leg(True, _BATCH_REPEATS, _BATCH_SEQS),
    # MATCHED-KERNEL arms: MIA's baked op + aperture, vLLM uncompiled. The only pair in
    # this suite in which the two capture MECHANISMS can be compared bit-exactly.
    "aperture_alone":      leg(True, cudagraph=False),
    "aperture_batch":      leg(True, _BATCH_REPEATS, _BATCH_SEQS, cudagraph=False),
    # T0 under FULL graphs: capture must still be a pure observer when MIA's work is a
    # baked op replayed by vLLM rather than a Python forward hook. T0 was eager-only.
    # TORCHDYNAMO_DISABLE=0 explicitly: t1_reference setdefault()s it to "1" at import
    # (it is an eager-only module by origin), which would leave this "FULL graph" control
    # running uncompiled and make it the wrong control.
    "t0_off_full":         leg(False, cudagraph=True, arm="off",
                               env={"TORCHDYNAMO_DISABLE": "0"}),
    # Carried item 1: step.query_start_loc consumed as a DEVICE tensor by the GPU scatter,
    # across steps, under real graph replay. Default-off lever, so it needs its own leg.
    "full_batch_gpuroute": leg(True, _BATCH_REPEATS, _BATCH_SEQS,
                               env={"MIA_CAPTURE_GPU_ROUTING": "1"}),
}

_HS_LEGS = dict(_CAPTURE_LEGS)
# _build_routing_hs (legacy per-request scatter) and _build_routing_hs_vectorized: HS-only.
_HS_LEGS["full_batch_legacy"] = leg(True, _BATCH_REPEATS, _BATCH_SEQS,
                                    env={"MIA_ROUTE_DECODE_CACHE": "0",
                                         "MIA_ROUTE_VECTORIZED": "0"})
_HS_LEGS["full_batch_vec"] = leg(True, _BATCH_REPEATS, _BATCH_SEQS,
                                 env={"MIA_ROUTE_DECODE_CACHE": "0",
                                      "MIA_ROUTE_VECTORIZED": "1"})
# NO-TAIL pair (task D4 fix round 1). With 33 prompts in 32 slots the last request forms a
# second wave, so the PREFILL wave composition -- and with it the padded graph size the tail
# requests run under -- can differ between two boots whose per-step host work differs. That
# is what made `gpu-routing vs host-routing` diverge on req30-32 in LSF 1702981 (divergence
# starts at ROW 0, the prefill, and the GENERATION differs there too). 33 prompts in 33 slots
# removes the second wave entirely and re-asks the question. req0 then shares its steps with
# 32 others rather than 31, which satisfies the invariant a fortiori.
_HS_LEGS["full_batch_notail"] = leg(True, _BATCH_REPEATS, 33)
_HS_LEGS["full_batch_notail_gpuroute"] = leg(True, _BATCH_REPEATS, 33,
                                             env={"MIA_CAPTURE_GPU_ROUTING": "1"})

_QK_LEGS = dict(_CAPTURE_LEGS)

# DISTINCT-LENGTH pair (task D4 fix round 2, F4). 33 requests, 33 different prompt lengths,
# one eager and one matched-kernel aperture arm. This is the leg that closes the same-shape
# blind spot BY CONSTRUCTION instead of documenting it: with no two requests sharing a row
# count, a whole-request mis-route cannot hide inside a magnitude band -- it changes the
# shape, and `_compare_replay_band` fails on shape with no tolerance involved.
_DISTINCT_ENV = {"MIA_PARITY_PROMPTS": "distinct"}
for _wl_legs in (_HS_LEGS, _QK_LEGS):
    _wl_legs["eager_batch_distinct"] = leg(False, 1, 33, env=dict(_DISTINCT_ENV))
    _wl_legs["aperture_batch_distinct"] = leg(True, 1, 33, cudagraph=False,
                                              env=dict(_DISTINCT_ENV))
    # REAL REPLAY, DISTINCT LENGTHS (task D5 item 6). The pair above runs
    # `cudagraph=False`, which is what buys it bit-exactness -- and which also means it
    # captures and replays NO CUDA graph and, because
    # `model_runner.py:1166` computes `num_tokens_after_padding = max(num_tokens,
    # batch_desc.num_tokens)`, pads to nothing. So until now the same-shape hole was closed
    # EXACTLY WHERE PADDED ROWS DO NOT EXIST and left open exactly where they do: the only
    # leg touching replay-time routing (`replay-band-w32`) runs the 3-prompt set, whose only
    # same-shape pairs are same-prompt replicas -- and a whole-request mis-route between two
    # replicas measures 2.23e-03 (HS) / 1.11e-03 (QK), UNDER every band in this file.
    #
    # This leg is the two conditions AT ONCE: 33 distinct prompt lengths under real capture
    # + replay. It cannot be bit-exact (it inherits the compiled-vs-uncompiled confound), so
    # it is judged on the existing relative + per-row band -- but the band is not what does
    # the work here. With no two requests sharing a row count, a whole-request mis-route
    # changes the captured SHAPE, and `_compare_replay_band` reports a shape mismatch as a
    # problem with no tolerance involved, however wide the band is.
    _wl_legs["full_batch_distinct"] = leg(True, 1, 33, env=dict(_DISTINCT_ENV))


# Steering writes no artifact, so its legs carry only generation + their own unsteered
# control (capture_workload._drive produces both and asserts liveness in-process).
# MIA_STEER_GPU_ROUTING is DEFAULT ON, so the FULL legs already exercise
# _build_routing_steer and the device-resident query_start_loc on the steer side.
_STEER_LEGS = {
    "eager_alone":          leg(False),
    "aperture_alone":       leg(True, cudagraph=False),   # steer op, matched kernels
    # GPU-measured (LSF 1702981): with the kernels matched (the UNSTEERED control between
    # these two arms is exactly 0.000e+00), the baked steer op still moves the steered
    # logprobs 2.342e-02 away from the forward-hook path. The prime suspect is MIA_STEER_FUSED
    # (Triton, default ON), whose dot-product reduction order differs from the aten reference
    # the eager worker runs. This leg turns the fused kernel off and re-asks.
    "aperture_alone_nofuse": leg(True, cudagraph=False, env={"MIA_STEER_FUSED": "0"}),
    "full_alone":           leg(True),
    "full_batch":           leg(True, _BATCH_REPEATS, _BATCH_SEQS),
    # The host-routing control for the steer side of carried item 1.
    "full_alone_hostroute": leg(True, env={"MIA_STEER_GPU_ROUTING": "0"}),
    # MIXED-ARM AT WIDTH 33 (task D5 item 7). Every gate above runs at max_num_seqs=1, so
    # the steer routing plane -- per-token-column, therefore per-request -- was never
    # exercised where it can put one request's steering on its neighbour. This leg arms
    # ALTERNATE requests of a 33-request, 33-distinct-length batch under real capture +
    # replay, which is the only configuration in which "each request received ITS OWN
    # steering" is distinguishable from "each received its neighbour's": an unarmed request
    # that moves has been steered by someone else's config. See
    # `capture_workload.steer_arm_mask` and `_drive_steer_mixed`.
    #
    # ITS CONTROL IS AN A/B PAIR, not a fully-unarmed baseline (task D6/D7). LSF 1725139's
    # control ran with NOBODY armed -- a different batch composition from the pass being
    # judged -- and 5/17 unarmed requests moved by up to 2.575421e-02 against the 1e-3 floor.
    # The decisive experiment (`tests/mia/parity/run_steer_ab_leak_control.py`, LSF 1725670) held
    # composition IDENTICAL instead and found all 17 unarmed requests bit-exact
    # (0.000000e+00) with the weakest ARMED delta at 2.757723e-01: NO leakage: the old
    # control was the defect. `_drive_steer_mixed` DRIVES this leg by calling that script's
    # `run_steer_ab_pass` directly (task D10) -- not by re-deriving its mechanics from
    # `_make_zero_vector` alone, which is what it did when LSF 1731353 still found 3/33
    # unarmed requests moving (see `run_steer_ab_pass`'s docstring for the numbers).
    "full_batch_mixed_arm": leg(True, 1, 33,
                                env={"MIA_PARITY_PROMPTS": "distinct",
                                     "MIA_PARITY_STEER_ARM": "alternate"}),
}

LEGS = {"hs_small": _HS_LEGS, "qk_small": _QK_LEGS, "steer_small": _STEER_LEGS}

# ---------------------------------------------------------------------------
# The REAL-REPLAY band (task D4 fix round 1).
#
# `op-identity` and `routing-identity-w32` are bit-exact, but they buy that exactness by
# running MIA's op against an UNCOMPILED model (`graph=True, cudagraph=False`) -- which
# captures and replays no CUDA graph at all, and, because
# `model_runner.py:1166` computes `num_tokens_after_padding = max(num_tokens,
# batch_desc.num_tokens)`, pads to nothing when the cudagraph mode is NONE. So they prove
# "baked op == forward-hook path" and "routing at width 32 == forward-hook routing", but they
# never exercise REPLAY-TIME routing against PADDED rows.
#
# That matters concretely. `reset_pinned` fills only `[:width]` with the sentinel, on the
# stated assumption that the op never reads past `width`; if `_upload_width` ever
# under-bounds the padded token count, the padding rows read STALE routing under a real
# replay and scatter into LIVE aperture slots. The §2 row audit cannot see it (its counts
# come from the host-side reservation, not from what the op wrote) and
# `routing-identity-w32` cannot see it (there is no padding). The same hole exists for a
# routing slab that is re-allocated rather than written in place under a baked graph.
#
# The leg that DOES touch real replay is hooks-vs-FULL at width 32. It cannot be bit-exact
# -- it inherits the compiled-vs-uncompiled confound -- so it is a MAGNITUDE band. TWO
# metrics are applied, because one of them alone has a hole.
#
# (1) TENSOR-GLOBAL relative: max|a-b| / max|a| over the whole tensor. Measured, LSF 1705469:
#
#       observed, real replay      HS 6.12e-03 (w1) / 5.49e-03 (w32); QK 2.12e-03
#       ------------------------------ band: 2e-2 -----------------------------------
#       off-by-one row within the request, FLOOR over tensors     4.51e-01   (23x)
#       a zeroed / sentinel row, FLOOR over WHICH row             1.11e-02 (HS) <-- BELOW
#                                                                 2.70e-01 (QK)
#
#     That HS figure is the hole, and it is why metric (2) exists. This metric normalises by
#     the tensor-global maximum, so zeroing a row whose own values are small relative to the
#     tensor's largest row produces a small number -- 1.11e-02 for HS, UNDER the band. The
#     "50x sentinel margin" quoted in an earlier revision of this comment was the best case
#     (zeroing the largest row), not the floor.
#
# (2) PER-ROW relative: max over rows r of max|a[r]-b[r]| / max|a[r]|. Normalising inside the
#     row makes a destroyed row cost the same whatever its magnitude. Measured, LSF 1705469:
#
#       observed, real replay      HS 2.74e-02, QK 2.70e-03
#       ----------------------------- band: 1.5e-1 ----------------------------------
#       off-by-one row, FLOOR                       5.96e-01 (HS) / 7.47e-01 (QK)  (4.0x)
#       a zeroed / sentinel row                     1.0000 exactly, ANY row        (6.7x)
#
#     A zeroed row is now exactly 1.0 by construction rather than magnitude-dependent, so the
#     hazard this suite exists for -- padding rows reading stale routing and scattering into
#     live aperture slots, or a re-allocated slab leaving the sentinel behind -- is caught
#     with a real margin instead of an advertised one.
#
# Integer metadata (`layer_num`, `k_prefix_ends`) is held BIT-EXACT inside the same gate, and
# the key sets must match, so an artifact present on only one side is a failure rather than
# something the row-wise loop never visits.
#
# WHAT NEITHER METRIC EXCLUDES, measured rather than asserted: a whole-request mis-route
# between two requests whose captured tensors have the SAME SHAPE. In this workload the
# same-shape pairs are same-prompt replicas, and their mutual difference -- which IS the
# signal such a mis-route would produce -- measures only 2.23e-03 (HS) / 1.11e-03 (QK)
# tensor-global, 1.11e-02 / 1.35e-03 per-row (LSF 1705469). All four are UNDER both bands, so
# no value-based gate on this workload can see that swap. (An earlier revision of this
# comment cited HS >= 7.06e-03 / QK >= 3.86e-01 here and called the margin discrimination;
# those figures were for DIFFERENT-PROMPT requests truncated to a common row count, which is
# not a same-shape pair that can actually arise -- the QK line in particular advertised ~19x
# discrimination where the true number is ~18x the WRONG way. Corrected.) Nor are replicas
# "byte-equal by construction", as that revision claimed: the suite's own `replica-identity`
# prints a non-zero worst max|d|; they are equal to within 2.23e-03 relative, i.e. under the
# band.
#
# That blind spot is CLOSED BY CONSTRUCTION rather than left documented -- see the
# `*_batch_distinct` legs, which run 33 requests with 33 DISTINCT prompt lengths so that any
# whole-request mis-route changes the row count and lands as a SHAPE mismatch, which is fatal
# with no tolerance involved.
_REPLAY_REL_BAND = _band("MIA_T2_REPLAY_REL_BAND", "2e-2",
                        "real-replay hooks-vs-FULL capture, tensor-global relative")
_REPLAY_ROW_BAND = _band("MIA_T2_REPLAY_ROW_BAND", "1.5e-1",
                        "real-replay hooks-vs-FULL capture, per-row relative")

# ---------------------------------------------------------------------------
# BANDED gates (task D4 fix round 1, second pass).
#
# Three legs measure a LEVER or a PUBLISHED LIMITATION rather than the port. Each keeps a
# gate -- an unbounded "expected fail" catches nothing -- but the gate is a magnitude bound
# taken from the measurements in hand, so the known delta passes and a REGRESSION still
# trips it. Every bound below says what it measures, the measured value, why it is not a
# port regression, and where the limitation is published for users.
#
# 1. T0 under FULL graphs (`T0-graph`). MEASURES: whether arming capture perturbs
#    generation once vLLM compiles. MEASURED: HS worst |d(logprob)| = 1.815024e-02, the SAME
#    value in LSF 1702981 and 1704450 (reproducible, not noise); QK 5.364418e-07 and
#    0.000000e+00 in the two jobs. Token ids identical on every request, both workloads,
#    both jobs. NOT A PORT REGRESSION: in eager mode T0 is bit-exact (task C4) with the same
#    process topology and the same control, so compilation is the only variable -- MIA's
#    baked op changes what inductor is given to fuse. PUBLISHED: README "Known Limitations"
#    and tests/mia/parity/TOLERANCES.md. The band is 5e-2: 2.8x above the measured HS value and
#    ~15x BELOW the scale of a real intervention on these logprobs (a steer moves them
#    7.72e-01), so a capture path that started genuinely perturbing generation still fails.
#    Token ids are held BIT-EXACT inside the gate -- that part must never move.
_T0_GRAPH_LOGPROB_BAND = _band("MIA_T2_T0_GRAPH_BAND", "5e-2",
                              "T0 under FULL graphs, |d(logprob)| capture-off vs on")

# 2. The fused steer kernel (`steer op-identity`). MEASURES: `MIA_STEER_FUSED`, a V1-era
#    lever that ships ON. MEASURED: 2.342296e-02 against the eager forward-hook steer, with
#    the UNSTEERED control between the same two arms at exactly 0.000000e+00 -- so it is the
#    kernel, not numerics. NOT A PORT REGRESSION: the same arm with `MIA_STEER_FUSED=0`
#    reproduces the eager steer BIT-EXACTLY (`op-identity-nofuse`, which stays a FATAL
#    bit-exact gate at STEER_ATOL) -- MIA's baked steer op, its V2 routing and the graph
#    install are exact; only the Triton projection reduction differs. PUBLISHED:
#    mia/optimizations.py PUBLIC_LEVERS and README. Band 5e-2: 2.1x above the measurement,
#    ~15x below the 7.72e-01 steering effect it would have to approach to be a real defect.
_STEER_FUSED_LOGPROB_BAND = _band("MIA_T2_STEER_FUSED_BAND", "5e-2",
                                 "MIA_STEER_FUSED lever, |d(logprob)| vs the eager steer")

# 3. GPU capture routing (`gpu-routing vs host-routing`). MEASURES: `MIA_CAPTURE_GPU_ROUTING`,
#    a default-OFF lever, against host routing under FULL graph. MEASURED (relative,
#    max|a-b| / max|a|): HS 2.228e-03 at width 32 in LSF 1702981 and 0.000000e+00 in 1704450
#    -- boot-dependent -- and 1.876e-03 at width 33, where it differs on all 33 requests.
#    NOT A PORT REGRESSION, on evidence rather than assertion: GENERATION ITSELF differs
#    between the two legs (33/33 requests, worst |d(logprob)| = 1.977980e-02, token ids
#    identical on all 33), and capture cannot alter generation -- so the two arms did not
#    run the same forward, and the capture delta is downstream of an input-side difference.
#    The magnitudes confirm it: the delta GROWS with depth (rel 1.33e-05 / 1.65e-05 /
#    1.88e-03 at layers 1 / 16 / 32), which is accumulation, whereas a mis-route is
#    full-scale -- an off-by-one row measures 4.51e-01 tensor-global / 5.96e-01 per-row (the
#    FLOOR over tensors, LSF 1705469) and a zeroed row 1.0 per-row, 240-530x larger. (An
#    earlier revision quoted 6.02e-01 here and 4.58e-01 in the replay-band comment for the
#    same quantity; both were single best-case tensors, not the floor.) The earlier "tail
#    wave" hypothesis is DEAD: raising max_num_seqs to 33 removed the tail and made the
#    difference global, which fits a padding/bucketing change, not a tail. PUBLISHED: README
#    "Known Limitations" and TOLERANCES.md. Band 1e-2: 4.5x above the worst measurement,
#    45-100x below the two mis-route signatures. Same blind spot as `_REPLAY_REL_BAND`, with
#    the corrected figure: it does not by itself exclude a same-SHAPE cross-request
#    mis-route, whose real signal here is 2.23e-03 (HS) / 1.11e-03 (QK) -- UNDER the band.
#    The `*_batch_distinct` legs close that by construction.
_GPU_ROUTING_REL_BAND = _band("MIA_T2_GPU_ROUTING_BAND", "1e-2",
                             "MIA_CAPTURE_GPU_ROUTING lever vs host routing, relative")

# 4. QK's `routing-identity-w32` (task D4c). MEASURES: whether the two capture MECHANISMS
#    (forward-hook probes vs the baked aperture op) still agree once boot-to-boot vLLM
#    kernel nondeterminism is accounted for. This gate was bit-exact and PASSED in LSF
#    1705469 and 1705794, then FAILED in 1710144 with no code change in between.
#
#    DIAGNOSIS (LSF 1710884, 3 boots per arm, same config, same code, same GPU model):
#    neither arm reproduces itself boot-to-boot, and the cross-arm delta is the SAME
#    magnitude as the within-arm delta -- the noise floor exceeds the signal this gate was
#    built to measure. Bit-exact (`compare_artifacts.compare`, absolute max|d|) counts, from
#    the raw artifacts under `that job’s artifacts (A1,A2,A3,B1,B2,B3)
#    (A = eager forward-hook, B = aperture; 33-request width-32 batch, layer0/15/31):
#
#      same-arm:  A1-vs-A2 24 problems max|d|=3.281e-02 | A1-vs-A3 48, 3.281e-02 |
#                 A2-vs-A3 24, 3.281e-02 | B1-vs-B2 192, 1.986e-02 | B1-vs-B3 12, 3.507e-02 |
#                 B2-vs-B3 192, 1.921e-02
#      cross-arm: A1-vs-B1 12, 3.507e-02 | A2-vs-B2 168, 1.562e-02 | A3-vs-B3 48, 3.281e-02
#
#    Token ids identical on every request, both arms, all 3 boots (0/33 differ) -- generation
#    is stable, so this is sub-argmax noise, not a routing defect. `hs_small`'s
#    `routing-identity-w32` stayed bit-exact across these SAME runs -- the asymmetry is real,
#    not a suite-wide flake, and is consistent with attention-internal reduction order
#    (split-K / atomics / per-boot kernel autotuning): arm A only READS what vLLM's attention
#    computed, so if it varies boot-to-boot the variation is vLLM's, not MIA's. This is also
#    NOT "QK is always noisy": the `w33-distinct` legs passed BIT-EXACT for both `hs` and
#    `qk` in LSF 1710144 -- the same job that failed this gate -- so the effect is
#    width/composition-dependent, not a property of QK capture in general.
#
#    An absolute max|d| of ~3.3e-02 does not by itself say whether that is noise or a
#    mis-route -- q/k_full run to a few tens in magnitude, so 3.3e-02 absolute is only
#    "roughly 1e-2" relative by rough estimate. Measuring the ACTUAL relative + per-row
#    metric (`_compare_replay_band`'s own metric, reused rather than reimplemented) directly
#    on the LSF 1710884 artifacts, over all 3 same-arm-A, 3 same-arm-B and 3 matched-boot
#    cross-arm (A1/B1, A2/B2, A3/B3) pairs -- 198 float tensors per pair -- gives a much
#    tighter cluster than that estimate:
#
#      worst tensor-global relative (max|a-b|/max|a|): 1.446e-03 (A1-vs-B1 and B1-vs-B3,
#          req0/layer031::k_full)
#      worst per-row relative (max_r max|a[r]-b[r]|/max|a[r]|): 1.880e-03 (B1-vs-B3,
#          req1/layer031::k_full)
#      all 9 pairs land in [1.216e-03, 1.880e-03] on both metrics -- same magnitude
#          same-arm or cross-arm, matching the "noise floor, not a mis-route" diagnosis.
#
#    BAND: reuses the `_REPLAY_REL_BAND` / `_REPLAY_ROW_BAND` magnitudes (2e-2 / 1.5e-1) --
#    the same category of comparison (QK's own two mechanisms, matched/uncompiled kernels)
#    that those bands already govern for the real-replay leg, kept as a SEPARATE constant so
#    the two gates can be retuned independently.
#
#      rel band 2e-2: 13.8x ABOVE the worst measured boot spread (1.446e-03); 22.6x BELOW
#          the off-by-one-row FLOOR measured elsewhere in this suite (4.512e-01
#          tensor-global, LSF 1705469).
#      row band 1.5e-1: 79.8x ABOVE the worst measured boot per-row spread (1.880e-03);
#          4.0x BELOW the off-by-one-row per-row FLOOR (5.959e-01, LSF 1705469) and 6.7x
#          BELOW a zeroed/sentinel row (1.0 exactly). The 4.0x is the thinnest margin of the
#          four -- it is the SAME margin `_REPLAY_ROW_BAND` already runs on for its own gate
#          (documented above), not a new risk, but it is the number to watch if a future
#          off-by-one measurement on this workload comes in lower than 5.959e-01.
#
#    Integer metadata (`layer_num`, `k_prefix_ends`) and shape stay BIT-EXACT inside this same
#    gate (`_compare_replay_band` never bands them) -- a mis-route that changes row counts
#    still fails as a SHAPE mismatch, with no tolerance involved. HS's `routing-identity-w32`
#    is UNCHANGED (still fatal, bit-exact). PUBLISHED: README "Known Limitations" and
#    TOLERANCES.md.
#
# 4b. QK's `routing-identity-w33-distinct` (task D9, task-D8-report.md's separate finding).
#    SAME diagnosis as 4. above, re-measured directly on `1726855/t2_qk_small/
#    {eager,aperture}_batch_distinct`: generations IDENTICAL on all 33 requests (no
#    trajectory divergence -- unlike the HS finding in the SAME job, see the trajectory-
#    bifurcation comment above `_BIFURCATION_LOGPROB_FLOOR`), shapes and integer metadata
#    (`layer_num`, `k_prefix_ends`) bit-exact, 156 of 396 float tensors differ, worst
#    tensor-global relative **1.870e-03**, concentrated on layer031 q/k_full across ALL 33
#    requests -- inside the SAME [1.216e-03, 1.880e-03] boot-spread cluster item 4 measures,
#    10.7x under this band, 241x under the off-by-one floor (4.512e-01). The "~1e-2..2e-2 at
#    req11" figures in a raw log for this leg are `_compare`'s ABSOLUTE max|d|: q/k values run
#    to tens in magnitude, so the absolute number is not the relative one this band reads --
#    the same conversion item 4's own comment already calls out. Arm A (the forward hook)
#    only READS what vLLM's attention computed, so boot-varying split-K / atomics /
#    per-boot autotuning is vLLM's variance here too, not MIA's. BANDED with the IDENTICAL
#    `_ROUTING_IDENTITY_QK_REL_BAND` / `_ROUTING_IDENTITY_QK_ROW_BAND` constants as the w32
#    gate above, not a new bound -- this is the same noise source at a different width, and a
#    separate margin would only obscure that. Integer metadata and shape remain BIT-EXACT and
#    FATAL inside this gate, exactly as for w32. HS's `routing-identity-w33-distinct` is
#    UNCHANGED (still fatal, bit-exact) -- HS is bit-reproducible on this leg; do NOT band it.
#    PUBLISHED: README "Known Limitations" and TOLERANCES.md.
_ROUTING_IDENTITY_QK_REL_BAND = _band("MIA_T2_ROUTING_QK_REL_BAND", "2e-2",
                                     "QK routing-identity-w32 / w33-distinct, "
                                     "tensor-global relative")
_ROUTING_IDENTITY_QK_ROW_BAND = _band("MIA_T2_ROUTING_QK_ROW_BAND", "1.5e-1",
                                     "QK routing-identity-w32 / w33-distinct, per-row relative")

# 5. Routing-builder equivalence (`hs_small builder legacy vs default`, `hs_small builder
#    vectorized vs default`, task D4f). MEASURES: whether the three interchangeable HS
#    routing-table builders -- legacy (`_build_routing_hs`, per-request host scatter),
#    vectorized (`_build_routing_hs_vectorized`), and decode-cache (`_build_routing_hs_
#    decode_cache`, the DEFAULT) -- compute the same routing plane. These two gates were
#    bit-exact; that is what "carried item 2" originally asked for.
#
#    DIAGNOSIS (LSF 1722805, 3 boots per builder, 40 comparison pairs; do not re-run this --
#    the diagnosis is complete): EVERY builder fails against ITSELF across boots, at the SAME
#    tail requests (reqs 30-32, the second prefill wave at this batch width) -- SELF default
#    FAIL 3/9/9 problems across the 3 boot-pairs, SELF legacy FAIL 3/6/3 (reqs 28-29), SELF
#    vectorized PASS 0 then FAIL 9/9 (reqs 30-32), SELF w33 default FAIL 6 (reqs 30-31). A
#    bit-exact gate between two DIFFERENT builders was never achievable: the noise floor
#    already exceeds it WITHIN a single builder, boot to boot. This is the same vLLM
#    kernel/attention nondeterminism `_ROUTING_IDENTITY_QK_REL_BAND` documents for QK, now
#    measured on HS's own builders.
#
#    THE BUILDERS ARE GENUINELY EQUIVALENT -- this is confirmed, not merely undisprovable.
#    Several CROSS-builder pairs agree BIT-EXACTLY: `defaultboot3-vs-vectorizedboot1`,
#    `defaultboot3-vs-vectorizedboot2`, `legacyboot3-vs-vectorizedboot1`,
#    `legacyboot3-vs-vectorizedboot2`, and `defaultboot3-vs-legacyboot3`. If legacy and
#    vectorized computed different routing they could never land on identical bits on ANY
#    pair. That is positive evidence FOR carried item 2 (the builders are interchangeable),
#    not a weakening of the gate. `token_ids_differ=no` on all 40 pairs -- generation is
#    unaffected; the noise is sub-argmax.
#
#    BAND: reuses the `_REPLAY_REL_BAND` / `_REPLAY_ROW_BAND` magnitudes (2e-2 / 1.5e-1) --
#    the existing relative + per-row band, applied to all 40 LSF 1722805 pairs, already
#    PASSES: worst tensor-global relative 2.230e-03, worst per-row relative 8.348e-03. Kept
#    as SEPARATE, independently-named constants so this gate and the replay-band gate can be
#    retuned without touching each other.
#
#      rel band 2e-2: ~9.0x ABOVE the worst measured pair (2.230e-03, 40 pairs); ~22.5x
#          BELOW the off-by-one-row floor measured elsewhere in this suite (4.5e-01
#          tensor-global, LSF 1705469).
#      row band 1.5e-1: ~18.0x ABOVE the worst measured per-row pair (8.348e-03, 40 pairs);
#          ~4.0x BELOW the off-by-one-row per-row floor (~6.0e-01, LSF 1705469) and ~6.7x
#          BELOW a zeroed/sentinel row (1.0 exactly, LSF 1705469). The 4.0x is the thinnest
#          margin of the four -- the same margin `_REPLAY_ROW_BAND` and
#          `_ROUTING_IDENTITY_QK_ROW_BAND` already run on, not a new risk.
#
#    Integer metadata (`layer_num`) and shape stay BIT-EXACT and FATAL inside this same gate
#    (`_compare_replay_band` never bands them, and a shape mismatch is always a problem) -- a
#    mis-route that changes row counts still fails with no tolerance involved.
#
#    WHAT THIS GATE NOW PROVES: the three builders agree to within run-to-run noise -- the
#    same standard every other value-based gate in this file already holds vLLM's own
#    kernels to. WHAT IT NO LONGER PROVES: bit-identity. Nothing at this batch width IS
#    bit-identical across boots, including a builder against ITSELF, so a bit-exact gate here
#    was never actually measuring the builders -- it was measuring whichever boot happened to
#    land closest. PUBLISHED: README "Known Limitations" and tests/mia/parity/TOLERANCES.md.
_BUILDER_EQUIV_REL_BAND = _band("MIA_T2_BUILDER_REL_BAND", "2e-2",
                               "HS routing-builder equivalence, tensor-global relative")
_BUILDER_EQUIV_ROW_BAND = _band("MIA_T2_BUILDER_ROW_BAND", "1.5e-1",
                               "HS routing-builder equivalence, per-row relative")

# 6. THE CEILING on every steer budget that is computed AT RUNTIME (task D5 items 2 and 5).
#
#    Two steer gates size themselves from a measured floor -- `steer-gap` from the UNSTEERED
#    eager-vs-FULL delta, `width` from the UNSTEERED alone-vs-batch delta -- because the
#    confound they have to tolerate (vLLM's compiled-vs-eager numerics; the model's own
#    batch-variance) is not a constant we can pin. `max(floor, atol)` with no upper bound is
#    a gate that WIDENS TO MATCH ITS OWN FAILURE: if the floor blew up, so did the budget,
#    and the gate passes whatever it is given. That is the defect this constant closes.
#
#    MEASURED, the two floors this caps, over 7 GPU jobs (LSF 1704450, 1705469, 1705794,
#    1710144, 1719790, 1720902, 1723978):
#
#      steer-gap floor  UNSTEERED eager-vs-FULL     3.635375e-02 in ALL 7 (reproducible)
#      width control    UNSTEERED alone-vs-batch    2.933863e-02 .. 5.865807e-02
#
#    THE THING A BLOWN BUDGET WOULD LET THROUGH: a stale or mis-routed steer config applies
#    (or drops) a whole intervention, and an adjust_rs steer on this workload moves the
#    logprobs 7.632304e-01 .. 8.145643e-01 -- measured on every arm of every one of those
#    jobs, so the scale is not an estimate.
#
#    CEILING 1.5e-1:
#      2.6x ABOVE the worst floor ever measured (5.866e-02), so no observed boot trips it;
#      8.4x above the worst STEERED width delta (1.779e-02) and 4.1x above the worst
#      STEERED gap delta (1.801e-02), so neither gate sits near its own cap;
#      5.1x BELOW the smallest steering intervention ever measured (7.632e-01), so a budget
#      pinned AT the ceiling still fails a real mis-steer by a factor of five.
#    A floor that exceeds the ceiling is not a licence to widen -- it is a FAILURE, reported
#    as one by both gates.
_STEER_BUDGET_CEILING = _band("MIA_T2_STEER_BUDGET_CEILING", "1.5e-1",
                             "ceiling on every runtime-computed steer budget")

# 7. The suite's ONE tolerance, registered here so it is echoed and cannot be widened from
#    the environment either. `run_parity.sh` exports STEER_ATOL=1e-5 and the rationale,
#    date, versions, model and dtype are in tests/mia/parity/TOLERANCES.md. Steering is compared
#    through its effect on float logprobs rather than as a bit-exact artifact, and the
#    bit-exact steer gate (`op-identity-nofuse`) is the one that runs AT this value --
#    measured 0.000000e+00 in every GPU job to date, i.e. five orders of margin.
_STEER_ATOL = _band("STEER_ATOL", "1e-5",
                    "the suite's only tolerance: |d(logprob)| for the bit-exact steer gates")

# 8. THE BATCHED ORACLE (`T1-batched capture_hs`, `T1-batched capture_qk`, task D5 item 8 /
#    task D7). MEASURES: whether MIA's batched capture agrees with an INDEPENDENT reference
#    that never imports MIA -- a plain vanilla-vLLM forward hook, run outside `mia/runner.py`
#    entirely -- at width 33. This is the ONE comparison in the whole suite that is not
#    MIA-vs-MIA: every T2 gate above imports StepView/step_view on BOTH arms, so a bug there
#    moves both arms identically and every T2 gate still passes; T1's original oracle ran
#    ONE request at a time and never exercised batching, the one thing the V1->V2 port
#    actually changed. See run_parity.sh's own comment on this section and
#    tests/test_parity_batched_oracle.py for the token-id row segmentation that keeps it
#    independent of MIA's routing. This gate lives in `run_parity.sh`, not `LEGS`/
#    `EXPECTED_GATES` -- it has no workload entry, so `_compare_replay_band` is invoked
#    directly (via `--compare-t1-batched` below) rather than through `judge_capture`.
#
#    LSF 1725139 failed BOTH capture kinds' `layer*.safetensors` payload bit-exact
#    (at that time `generation.safetensors` -- prompts, token ids, logprobs -- stayed
#    bit-exact; task D10 below revisits that claim, which did NOT hold in a later job):
#      capture_qk worst rel = 7.457e-04  (req0/layer031::q; absolute max|d|=7.812e-03 looks
#          alarming only because q/k_full tensors have one dominant outlier channel)
#      capture_hs worst rel = 1.272e-03  (req0, growing with depth: 1.221e-04 -> 3.125e-02
#          -> 3.750e-01 ABSOLUTE at layers 1/16/32 -- relative is the number that matters;
#          HS activations grow with depth, so the absolute figure alone overstates it)
#
#    NOT A NEW REGIME. Re-measured with `_compare_replay_band`'s OWN metric (reused, not
#    reimplemented) over ALL 15 pairwise boots of LSF 1710884's d4b/{A1,A2,A3,B1,B2,B3} --
#    supersedes the 9-pair figure quoted for `_ROUTING_IDENTITY_QK_REL_BAND` above and
#    confirms it as the true max, not a cherry-pick: worst tensor-global rel 1.446e-03, worst
#    per-row rel 1.880e-03. BOTH T1-batched failures sit AT OR BELOW that boot-to-boot floor
#    -- i.e. this is the SAME vLLM kernel nondeterminism (split-K / atomics / per-boot
#    autotuning) `_ROUTING_IDENTITY_QK_REL_BAND` already documents, now visible (a) in a
#    comparison with NO MIA code on either side, and (b) in HS's batched capture, where it
#    had not previously been measured. (README's `3.5e-02` is a DIFFERENT, ABSOLUTE number --
#    the same-arm QK boot spread in LSF 1710884's own units -- not this relative + per-row
#    metric; it is not the floor being compared against here. Do not cite it as one.)
#
#    BAND: reuses the IDENTICAL magnitude as `_ROUTING_IDENTITY_QK_REL_BAND` /
#    `_ROUTING_IDENTITY_QK_ROW_BAND` above (2e-2 / 1.5e-1) rather than a new, looser bound --
#    the same CATEGORY of comparison (matched/uncompiled-kernel capture vs an independent
#    reader, gated on vLLM's own boot-to-boot float noise). Kept as separate, clearly-named
#    constants -- one pair covers BOTH capture_hs and capture_qk here, since the magnitude is
#    the same and neither measurement is close to it -- so this gate and the T2 QK gate can
#    be retuned independently later.
#
#      rel band 2e-2: 26.8x above the worst measured value (7.457e-04, QK), 15.7x above the
#          worst measured value (1.272e-03, HS); 22.6x below the off-by-one-row FLOOR
#          (4.512e-01 tensor-global, LSF 1705469) -- the same floor every other band in this
#          file is checked against.
#      row band 1.5e-1: see tests/test_parity_t1_batched_band.py for the per-row figures
#          pinned as constants and the margin assertions, mirroring
#          test_qk_routing_identity_band.py.
#
#    8b. THE SIBLING GATE (`T1-batched capture_hs/capture_qk GENERATION`, task D10). This
#    gate's own `generation.safetensors` comparison (`run_parity.sh`, `compare_artifacts.py
#    --require-bit-exact --name generation.safetensors`) was believed bit-exact above --
#    true in LSF 1725139, FALSE in LSF 1731353: BOTH capture kinds failed on the SAME two
#    channels, at the SAME magnitude:
#      req0/generation.safetensors::cumulative_logprob  max|d| = 1.854e-02
#      req0/generation.safetensors::token_logprobs      max|d| = 1.269e-02
#    `token_ids` and `prompt_token_ids` were NOT among the mismatches, on EITHER capture
#    kind -- the model produced the identical sequence in both arms on all 33 requests; only
#    the emitted logprobs moved. This is not the bifurcation `_bifurcation_report` exists to
#    catch (that is a token-id DIVERGENCE) and not a batched-routing defect (agreeing token
#    ids on 33/33 requests rules out a row mis-route) -- it is the SAME cross-boot logprob
#    noise `_T0_GRAPH_LOGPROB_BAND` was built for (worst measured HS 1.815024e-02, the SAME
#    order as this 1.854e-02), now observed between the vanilla-vLLM T1 reference boot and
#    the MIA boot rather than between two MIA boots.
#
#    BAND: `_T1_BATCHED_LOGPROB_BAND` reuses `_T0_GRAPH_LOGPROB_BAND`'s value (5e-2) rather
#    than inventing a new one: 2.7x above the worst measured value here (1.854e-02),
#    consistent with the 2.8x margin `_T0_GRAPH_LOGPROB_BAND` already carries for the
#    identical quantity, and ~15x below the 7.72e-01 scale of a real steering intervention.
#    Kept as its own named constant (not a second call site for `_T0_GRAPH_LOGPROB_BAND`)
#    because it gates a DIFFERENT comparison -- reference-vs-MIA generation at width 33, not
#    capture-off-vs-on at width 1 -- and the two must stay independently retunable.
#    `token_ids` and `prompt_token_ids` stay BIT-EXACT and FATAL inside this same comparison
#    (`_compare_generation_band`, reused via `--compare-t1-batched-generation` below, not
#    reimplemented) -- that is the load-bearing claim this gate exists to make: MIA does not
#    change what the model generates. PUBLISHED: README "Known Limitations" and
#    tests/mia/parity/TOLERANCES.md.
#
#    Integer metadata (`layer_num`, `k_prefix_ends`) and SHAPE stay BIT-EXACT and FATAL
#    inside this same comparison (`_compare_replay_band` never bands them) -- the batched
#    oracle's 33 DISTINCT prompt lengths (task D5 item 8) mean a whole-request mis-route
#    changes row COUNT, not just magnitude, so it still fails as a SHAPE mismatch with no
#    tolerance involved, exactly as for every other banded gate in this file.
#
#    WHAT THIS GATE STILL PROVES, band or not: multi-row batched ROUTING -- 33 distinct-length
#    requests in ONE step, row assignment read from token ids alone by a reference that
#    imports NOTHING from mia/runner.py -- agrees with that independent oracle to within
#    run-to-run noise vLLM itself already exhibits. That independence is the entire reason
#    this gate exists (it is the one gate in this suite that is not MIA-vs-MIA), and banding
#    its VALUES does not remove it -- it only concedes the bit-for-bit requirement that a
#    genuinely non-associative batched kernel replayed across two different process boots can
#    never promise. Width-1 `T1 capture_hs` / `T1 capture_qk` are UNCHANGED: still bit-exact,
#    still fatal, and -- being one request per pass, with no batched reduction to reorder --
#    immune to this effect entirely, so they remain the strongest evidence in this suite. Do
#    NOT band those. PUBLISHED: README "Known Limitations" and tests/mia/parity/TOLERANCES.md.
_T1_BATCHED_REL_BAND = _band(
    "MIA_T1_BATCHED_REL_BAND", "2e-2",
    "T1-batched vs the independent width-33 oracle, tensor-global relative")
_T1_BATCHED_ROW_BAND = _band(
    "MIA_T1_BATCHED_ROW_BAND", "1.5e-1",
    "T1-batched vs the independent width-33 oracle, per-row relative")
# Item 8b above: the SIBLING gate over `generation.safetensors` (task D10). Reuses
# `_T0_GRAPH_LOGPROB_BAND`'s value, not its call site -- see the derivation in the comment
# block above for the margin (2.7x over the measured 1.854e-02).
_T1_BATCHED_LOGPROB_BAND = _band(
    "MIA_T1_BATCHED_LOGPROB_BAND", "5e-2",
    "T1-batched generation vs the independent width-33 oracle, |d(logprob)| "
    "(token ids and prompt token ids stay bit-exact)")

# Wall-clock ceiling for ONE arm (engine boot + compile + cudagraph capture + 16 steps).
_ARM_TIMEOUT_S = int(os.environ.get("MIA_T2_ARM_TIMEOUT_S", "2700"))


# ---------------------------------------------------------------------------
# THE EXPECTED GATE SET — a gate that can VANISH is not a gate (task D5 item 1).
# ---------------------------------------------------------------------------
# The structural review of this suite found that `judge_steer` gated its optional legs on
# `if <leg> is not None:`, so a leg that FAILED TO RUN made its gate DISAPPEAR rather than
# fail. Demonstrated on the real LSF 1723978 artifacts: dropping 4 of the 6 steer legs still
# printed `VERDICT T2 steer_small: PASS (0 failing invariants: [])` and exited 0 — and the
# first gate lost was `op-identity-nofuse`, the ONLY bit-exact gate on the steer path.
# Nothing counted the verdicts against anything.
#
# `judge_capture` never had that hole, because its comparisons are driven by the static
# `LEGS` matrix and a missing arm becomes `FAIL (an arm did not run)`. This is the same
# discipline, made explicit and checked at the END OF THE RUN rather than left as a property
# of how each judge happens to be written:
#
#   * every label below MUST appear in `verdicts` when the workload is judged — a missing
#     one is a FAILURE, not a silently shorter list;
#   * every label emitted MUST appear below — a new gate has to be declared here, so the
#     expected set cannot quietly fall behind the code.
#
# Both directions are enforced by `gate_set_problems()` and by the hermetic
# `tests/test_parity_expected_gates.py`, which runs each judge over a fully-populated
# synthetic leg tree and asserts SET EQUALITY (so a typo in this table fails without a GPU).
#
# INFO (non-fatal) verdicts are listed too: they are collected evidence, and evidence that
# can silently stop being collected is the same defect one rung down.
def _capture_gates(workload: str) -> tuple[str, ...]:
    """The gates `judge_capture` must emit for ``workload``, derived from the LEG matrix.

    Written as one function rather than three literal tuples because the leg matrix already
    decides which comparisons exist (`full_batch_legacy` is HS-only, the QK
    routing-identity gate is banded, ...). Deriving the EXPECTATION from the same `LEGS`
    table the judge reads keeps the two from drifting, and the hermetic set-equality test
    is what catches a mistake in either.
    """
    w = workload
    gates = [
        f"{w} RECORD eager-vs-full (bit-exact)",
        f"{w} op-identity",
        f"{w} RECORD alone-vs-batch (bit-exact)",
    ]
    if w == "qk_small":
        gates += [f"{w} RECORD routing-identity-w32 (bit-exact)",
                  f"{w} routing-identity-w32 [BANDED]"]
    else:
        gates += [f"{w} routing-identity-w32"]
    if w == "qk_small":
        gates += [f"{w} RECORD routing-identity-w33-distinct (bit-exact)",
                  f"{w} routing-identity-w33-distinct [BANDED]"]
    else:
        gates += [f"{w} routing-identity-w33-distinct"]
    gates += [
        f"{w} routing-identity-w33-distinct-replay [BANDED]",
        f"{w} replica-identity FULL",
        f"{w} replica-identity eager",
        f"{w} CONTROL eager alone-vs-batch",
        f"{w} replay-band-w32",
        f"{w} replay-band-w1",
        f"{w} CONTROL generation eager-vs-full",
        f"{w} T0-graph capture-off-vs-on",
    ]
    for leg_name, label in (("full_batch_legacy", "builder legacy"),
                            ("full_batch_vec", "builder vectorized")):
        if leg_name in LEGS[w]:
            gates += [f"{w} RECORD {label} vs default (bit-exact)",
                      f"{w} {label} vs default [BANDED]"]
    if "full_batch_gpuroute" in LEGS[w]:
        gates += [f"{w} RECORD gpu-routing vs host-routing (bit-exact)",
                  f"{w} gpu-routing vs host-routing [BANDED]",
                  f"{w} gpu-routing generation [BANDED]"]
    if "full_batch_notail" in LEGS[w]:
        gates += [f"{w} RECORD gpu-routing no-tail (bit-exact)",
                  f"{w} gpu-routing vs host-routing, no tail wave [BANDED]",
                  f"{w} gpu-routing no-tail generation [BANDED]"]
    return tuple(gates)


_STEER_GATES: tuple[str, ...] = (
    "steer_small liveness",
    "steer_small steer-gap",
    "steer_small op-identity-nofuse",      # the one the vanishing bug lost FIRST
    "steer_small op-identity [BANDED]",
    "steer_small gpu-routing",
    "steer_small width [BANDED]",
    "steer_small per-request-arming",
    "steer_small INFO per-request-arming bit-exact",
)

EXPECTED_GATES: dict[str, tuple[str, ...]] = {
    "hs_small": _capture_gates("hs_small"),
    "qk_small": _capture_gates("qk_small"),
    "steer_small": _STEER_GATES,
}


def gate_problems(expected, verdicts: list) -> tuple[list[str], list[str]]:
    """``(missing, undeclared)`` for ANY declared gate set and the verdicts it emitted.

    The declared-vs-emitted machinery itself, with no dependency on this file's workload
    table, so another tier can reuse it rather than re-implement it. `gate_set_problems`
    below is this function bound to T2's own `EXPECTED_GATES`; T3
    (`tests/mia/parity/t3_crossbranch.py`) imports THIS one and binds its own table.
    """
    emitted = [label for label, _ok, _fatal in verdicts]
    missing = [label for label in expected if label not in emitted]
    undeclared = sorted({label for label in emitted if label not in expected})
    return missing, undeclared


def gate_set_problems(workload: str, verdicts: list) -> tuple[list[str], list[str]]:
    """``(missing, undeclared)`` for one judged workload.

    ``missing``    — a declared gate that never emitted a verdict. THE defect this exists
                     for: a leg that did not run must FAIL, never disappear.
    ``undeclared`` — a verdict whose label is not in the table above. A new gate that is
                     not declared cannot be missed later, so it is refused now.
    """
    return gate_problems(EXPECTED_GATES[workload], verdicts)


# ---------------------------------------------------------------------------
# Canonical extraction — eager (probes) and FULL-graph (capture aperture)
# ---------------------------------------------------------------------------

def _step_lengths(n_prompt: int, n_steps: int) -> list[int]:
    """Rows captured per forward pass for an ``all_tokens`` / ``hooks_on=both`` request.

    One unchunked prefill pass holding the whole prompt, then one row per decode pass. The
    caller asserts the total against the artifact, so a chunked prefill (which would break
    this shape) fails loudly instead of silently mis-slicing.
    """
    return [n_prompt] + [1] * (n_steps - 1)


def _unpad(padded, lengths):
    """``pad_sequence`` stack -> the flat rows it was built from. Lossless."""
    import torch

    if padded.dim() != 3:
        raise RuntimeError(
            f"expected a [n_steps, max_len, D] all_tokens stack, got {tuple(padded.shape)}; "
            f"this normalizer is only valid for all_tokens capture")
    if padded.shape[0] != len(lengths):
        raise RuntimeError(f"{padded.shape[0]} passes but {len(lengths)} step lengths")
    if padded.shape[1] < max(lengths):
        raise RuntimeError(
            f"padded width {padded.shape[1]} < longest step {max(lengths)}: the step-length "
            f"model does not describe this artifact (chunked prefill?)")
    return torch.cat([padded[i, :n] for i, n in enumerate(lengths)], dim=0)


def _expected_rows(output) -> tuple[int, int, int]:
    """``(n_prompt, n_gen, n_captured_rows)`` for one request.

    The last sampled token never gets a forward pass, so an ``all_tokens`` capture over
    ``hooks_on=both`` holds exactly ``n_prompt + n_gen - 1`` rows. Any OTHER number is the
    finding: extra rows mean a warmup/dummy pass leaked into the aperture (D3's guard), and
    missing rows mean a step was routed away.
    """
    n_prompt = len(output.prompt_token_ids)
    n_gen = len(output.outputs[0].token_ids)
    return n_prompt, n_gen, n_prompt + n_gen - 1


def _canonical_eager(subsystem: str, output) -> dict:
    """``{layer: {key: tensor}}`` for one request, from the forward-hook probes."""
    import torch

    probes = getattr(output, "probes", None) or {}
    n_prompt, n_gen, want = _expected_rows(output)
    out: dict = {}

    if subsystem == "hs":
        for entry in (probes.get("hs_cache") or {}).values():
            hs = entry["hidden_states"]
            flat = _unpad(hs, _step_lengths(n_prompt, hs.shape[0]))
            out[int(entry["layer_num"])] = {"hidden_states": flat}
    elif subsystem == "qk":
        for entry in (probes.get("qk_cache") or {}).values():
            q, k_all = entry["q"], entry["k_all"]
            n_steps = q.shape[0]
            ends = [n_prompt + j for j in range(n_steps)]
            # k_all is the padded growing-prefix stack; its LAST row, trimmed to that step's
            # prefix end, is the full key history — the same tensor the aperture writes flat.
            k_full = k_all[n_steps - 1][: ends[-1]]
            out[int(entry["layer_num"])] = {
                "q": _unpad(q, _step_lengths(n_prompt, n_steps)),
                "k_full": k_full,
                "k_prefix_ends": torch.tensor(ends, dtype=torch.int64),
            }
    else:
        raise KeyError(subsystem)

    _check_rows("eager", subsystem, output.request_id, out, n_prompt, n_gen, want)
    return out


def _match_req(art: dict, request_id: str):
    """The aperture's key for ``request_id`` (vLLM may append a suffix internally)."""
    if request_id in art:
        return art[request_id]
    for key in art:
        if key.startswith(f"{request_id}-"):
            return art[key]
    raise RuntimeError(
        f"request {request_id!r} has NO rows in the capture aperture (keys present: "
        f"{sorted(art)[:8]}{'...' if len(art) > 8 else ''}). Zero artifacts on one side is a "
        f"FAILURE, not a pass.")


def _canonical_graph(subsystem: str, output, art: dict) -> dict:
    """``{layer: {key: tensor}}`` for one request, from the reconstructed aperture dump."""
    import torch

    n_prompt, n_gen, want = _expected_rows(output)
    per_layer = _match_req(art, output.request_id)
    out: dict = {}

    if subsystem == "hs":
        # aperture_metadata.LayerEntry.layer is the 1-based artifact layer == eager layer_num.
        for layer, tensor in per_layer.items():
            out[int(layer)] = {"hidden_states": tensor}
    elif subsystem == "qk":
        # QKStepEntry.layer is 0-based == the eager qkv_hook's layer_num.
        for layer, entry in per_layer.items():
            out[int(layer)] = {
                "q": entry["q"],
                "k_full": entry["k_full"],
                "k_prefix_ends": torch.tensor([int(x) for x in entry["k_prefix_ends"]],
                                              dtype=torch.int64),
            }
    else:
        raise KeyError(subsystem)

    _check_rows("graph", subsystem, output.request_id, out, n_prompt, n_gen, want)
    return out


def _check_rows(arm, subsystem, request_id, per_layer, n_prompt, n_gen, want) -> None:
    """Row-count audit. Printed for every (request, layer); a mismatch RAISES.

    This is the warmup-leak proof: the aperture's row count for a request is decided by the
    routing the baked op replayed against, so a dummy/profile pass that scattered into live
    rows changes it. It is also the cheapest possible detector for a mis-slice that happens
    to land on another request of the same prompt length.
    """
    for layer in sorted(per_layer):
        vals = per_layer[layer]
        rows = vals.get("hidden_states", vals.get("q")).shape[0]
        detail = f"{arm} req={request_id} layer={layer} rows={rows} want={want}"
        if rows != want:
            raise RuntimeError(
                f"[T2] ROW-COUNT MISMATCH: {detail} (n_prompt={n_prompt} n_gen={n_gen}). "
                f"More rows than the request generated means non-request rows (a warmup / "
                f"dummy / cudagraph-capture pass) reached the artifact; fewer means a step "
                f"was routed away.")
        if subsystem == "qk":
            ends = [int(x) for x in vals["k_prefix_ends"]]
            expect = [n_prompt + j for j in range(n_gen)]
            if ends != expect:
                raise RuntimeError(
                    f"[T2] K-PREFIX MISMATCH: {detail} prefix_ends={ends[:6]}... "
                    f"expected {expect[:6]}...")
            if vals["k_full"].shape[0] != expect[-1]:
                raise RuntimeError(
                    f"[T2] K-FULL MISMATCH: {detail} k_full rows="
                    f"{vals['k_full'].shape[0]} expected {expect[-1]}")
        print(f"[T2] rows {detail}", flush=True)


# ---------------------------------------------------------------------------
# One arm (runs in its own process, holds the only engine)
# ---------------------------------------------------------------------------

def _drive_t2(engine, wl, out_dir: Path, batch: int) -> Path:
    """Generate, then normalize BOTH retrieval paths to the canonical artifact layout."""
    import torch

    from tests.mia.parity.capture_workload import (
        extra_args_for,
        prompts_for,
        sampling_params_for,
    )

    graph = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"
    prompts = prompts_for(wl, batch)
    if os.environ.get("MIA_PARITY_PROMPTS") == "distinct":
        # The whole point of this leg is that no two requests share a row count. Enforce it
        # against the REAL tokenizer rather than trusting the word counts.
        from tests.mia.parity.capture_workload import assert_distinct_prompt_lengths

        lengths = assert_distinct_prompt_lengths(engine.get_tokenizer(), prompts)
        print(f"[T2] distinct-length workload: {len(prompts)} prompts, token counts "
              f"{min(lengths)}..{max(lengths)}, all different", flush=True)
    outputs = engine.generate(prompts, sampling_params_for(wl, extra_args_for(wl)),
                              use_tqdm=False)

    art = None
    if graph:
        # FULL-graph retrieval: the drain streams to per-layer raw files DURING generate;
        # flush_aperture drains the tail and writes the sidecar the reader needs. The worker
        # is usually killed rather than joined, so the atexit backstop is not the contract —
        # this RPC is.
        from mia.graph.aperture_reader import (
            load_multilayer_aperture_artifact,
            load_multilayer_qk_aperture_artifact,
        )

        run_dirs = [d for d in engine.collective_rpc("flush_aperture") if d]
        print(f"[T2] flush_aperture -> {run_dirs}", flush=True)
        if not run_dirs:
            raise RuntimeError(
                "flush_aperture returned no run_dir: the FULL-graph capture aperture was "
                "never installed, so this arm captured NOTHING.")
        loader = (load_multilayer_aperture_artifact if wl.subsystem == "hs"
                  else load_multilayer_qk_aperture_artifact)
        art = loader(run_dirs[0])
        print(f"[T2] aperture holds {len(art)} requests", flush=True)

    written = 0
    for index, output in enumerate(outputs):
        req_dir = Path(out_dir) / f"req{index}"
        _save(req_dir / "generation.safetensors", generation_tensors(output))
        written += 1
        per_layer = (_canonical_graph(wl.subsystem, output, art) if graph
                     else _canonical_eager(wl.subsystem, output))
        if not per_layer:
            raise RuntimeError(f"request {output.request_id} captured NO layers")
        for layer, tensors in sorted(per_layer.items()):
            payload = dict(tensors)
            payload["layer_num"] = torch.tensor(int(layer), dtype=torch.int64)
            _save(req_dir / f"layer{int(layer):03d}.safetensors", payload)
            written += 1

    print(f"[T2] wrote {written} canonical artifact files under {out_dir}", flush=True)
    if written == 0:
        raise RuntimeError(f"capture produced NOTHING under {out_dir}")
    return Path(out_dir)


def _drive_steer_mixed(engine, wl, out_dir: Path, batch: int) -> Path:
    """A batch in which only SOME requests are armed to steer (task D5 item 7), judged
    against an A/B control rather than a differently-composed baseline (task D6/D7).

    THE GAP THIS CLOSES (item 7). The steer legs all ran at ``max_num_seqs=1``, so the
    per-request steer routing plane was never exercised at a width where it can put one
    request's steering on another. Widening the batch alone does not fix that either -- with
    every request carrying the SAME config, "each request got its own steering" and "each
    request got its neighbour's" predict identical logprobs. Arming ALTERNATE requests
    separates them (see ``capture_workload.steer_arm_mask``), and the prompts are the 33
    DISTINCT-length ones so each request is individually identifiable rather than a replica
    of its neighbour.

    THE CONTROL BUG THIS FIXES (task D6/D7). LSF 1725139 ran this leg's control as a SECOND
    pass with NOBODY armed at all -- a batch composition that diverges from the pass being
    judged the moment any armed request generates differently. 5 of 17 unarmed requests then
    moved by up to 2.575421e-02 against the 1e-3 floor, indistinguishable by construction
    from a real per-row leak. The decisive experiment
    (``tests/mia/parity/run_steer_ab_leak_control.py``, LSF 1725670) held composition IDENTICAL
    instead -- same 33 prompts, same order, same alternating mask, the SAME 16 rows armed in
    BOTH passes -- and found all 17 unarmed requests bit-exact (0.000000e+00) while the
    weakest ARMED delta was 2.757723e-01: there is NO leakage, and the original gate's
    failure was an artifact of comparing two differently-composed batches.

    So this leg's ``control`` (``write_request_artifacts(..., label="control")``) is now
    Arm B of that experiment, not a fully-unarmed run: the SAME 16 rows armed with the SAME
    ``adjust_rs`` config, except ``vector_path`` points at an ALL-ZERO-``dir`` clone of the
    real vector (same ``avg_proj``) -- built by
    ``run_steer_ab_leak_control._make_zero_vector``, reused here rather than reimplemented.
    ``coefficient: 0.0`` is NOT usable as that zero-effect control:
    ``install_steer.py``/``steer_worker.py`` force ``coefficient=0.0`` for method
    ``adjust_rs`` unconditionally (the magnitude is computed IN-KERNEL as
    ``avg_proj - residual . unit``), so that literal ask is already the status quo and would
    change nothing. An all-zero ``dir`` still runs the identical ``steer_mode=1``, 2-pass
    kernel and the identical per-row routing/masking as the real config -- it computes a
    real, nonzero in-kernel coefficient and multiplies it by a zero ``unit``, landing on an
    EXACT zero addition without skipping the op or touching which rows get an assignment.

    Because Arm A and Arm B arm the SAME rows with the SAME config shape (only the vector's
    VALUES differ), the UNARMED rows' inputs, shapes and kernel invocations are identical in
    both passes -- nothing about their processing can differ, so their honest delta is
    EXACTLY 0.0, not merely small. ``_judge_steer_per_request_arming`` now gates on that
    bit-exactly; see its docstring.

    Writes the same ``generation`` / ``control`` pair every steer leg writes, plus an
    ``arming.safetensors`` marker per request so the judge reads what this driver actually
    did instead of re-deriving it and agreeing with itself.

    TASK D10: the drive itself (prompts, mask, zero-vector config, the two ``generate()``
    calls) is no longer duplicated here -- it is
    ``tests.mia.parity.run_steer_ab_leak_control.run_steer_ab_pass``, called directly. Before
    D10 this function was a hand-maintained TWIN of that module's ``_drive_ab_leak_control``
    (it already reused ``_make_zero_vector`` alone), and LSF 1731353 found the twin's gate
    (`steer_small per-request-arming`) moving 3 of 33 unarmed requests by
    1.613855e-02 / 2.467152e-02 / 2.032498e-02 -- the SAME single-boot, single-engine, same-
    prompts, same-mask configuration in which `run_steer_ab_leak_control.py`'s OWN dedicated
    GPU run (LSF 1725670) measured bit-exact 0.0 on all 17. Being "structurally the same
    code" was not the same as being the SAME code; it now is. See
    ``run_steer_ab_pass``'s docstring for the full account.
    """
    from tests.mia.parity.run_steer_ab_leak_control import run_steer_ab_pass

    run_steer_ab_pass(engine, wl, Path(out_dir), batch=batch,
                      label_a="generation", label_b="control")
    return Path(out_dir)


def run_arm(workload: str, out_dir: Path, *, graph: bool, batch: int,
            cudagraph: bool, arm: str = "mia") -> Path:
    if arm == "off":
        # Capture-OFF control: no plugin anywhere. Writes generation only, by definition.
        from tests.mia.parity.t1_reference import reference_generate
        return reference_generate(Path(out_dir), workload_name=workload, graph=cudagraph)

    wl = WORKLOADS[workload]
    # Steering has no captured payload: its observable is the effect on the logprobs, and
    # capture_workload._drive already produces the steered run, its own unsteered control,
    # and the in-process liveness assertion. Nothing to normalize. The one exception is the
    # MIXED-ARM leg, whose whole point is that the requests do NOT share one SamplingParams.
    if wl.subsystem == "steer":
        drive = (_drive_steer_mixed
                 if os.environ.get("MIA_PARITY_STEER_ARM") == "alternate" else None)
    else:
        drive = _drive_t2
    return run_workload(workload, out_dir, graph=graph, batch=batch, drive=drive,
                        cudagraph=cudagraph)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _leg_env(spec: dict, scratch: Path) -> dict:
    env = dict(os.environ)
    env["MIA_PARITY_MAX_NUM_SEQS"] = str(spec["seqs"])
    # Per-leg scratch: the aperture raw files APPEND, so two graph legs sharing a directory
    # would read each other's rows back. Kept OUT of the artifact tree so the comparator's
    # recursive glob never sees it.
    env["MIA_PARITY_SCRATCH"] = str(scratch)
    if spec["cudagraph"]:
        # The graph path bakes its work into a custom op and WANTS compile; a leaked
        # TORCHDYNAMO_DISABLE=1 from the T0/T1 legs above would leave the cudagraph replaying
        # unfused kernels. Eager needs the opposite (torch.compile replaces the forward, so
        # register_forward_hook never fires) — mia/llm.py setdefault()s it there.
        env.pop("TORCHDYNAMO_DISABLE", None)
    else:
        env["TORCHDYNAMO_DISABLE"] = "1"
    if spec["arm"] == "off":
        # The control must load NO plugin. t1_reference also sets this at import, but an
        # arm that silently loaded MIA would make the T0 comparison vacuous, so it is set
        # here too rather than trusted.
        env["VLLM_PLUGINS"] = ""
    env.update(spec["env"])
    return env


def run_legs(workload: str, root: Path, only: set[str] | None = None) -> dict:
    """Run every leg of ``workload`` in its own process. Returns ``{leg: out_dir | None}``."""
    root = Path(root)
    scratch_root = Path(os.environ.get("MIA_PARITY_SCRATCH") or (root / "_scratch")) / "t2"
    done: dict = {}
    for name, spec in LEGS[workload].items():
        if only and name not in only:
            continue
        out_dir = root / name
        env = _leg_env(spec, scratch_root / workload / name)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", workload, str(out_dir),
               "--batch", str(spec["batch"]), "--control-arm", spec["arm"]]
        cmd += ["--graph"] if spec["graph"] else []
        cmd += ["--cudagraph"] if spec["cudagraph"] else []
        banner = (f"workload={workload} leg={name} arm={spec['arm']} graph={spec['graph']} "
                  f"cudagraph={spec['cudagraph']} batch={spec['batch']} "
                  f"max_num_seqs={spec['seqs']} env={spec['env'] or '{}'}")
        print(f"\n=== [T2] ARM {banner} ===", flush=True)
        t0 = time.time()
        try:
            # Bounded: FULL-graph CAPTURE is the one step in this suite that can HANG rather
            # than fail (D3's guard clones/zeroes a device tensor around vLLM's dummy passes,
            # and that work must stay outside torch.cuda.graph()'s capture region). A hang
            # would otherwise consume the whole job and leave every later leg unrun, so an arm
            # that overruns is killed and reported as a failed leg.
            rc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), timeout=_ARM_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            print(f"[T2] ARM TIMED OUT after {_ARM_TIMEOUT_S}s: {banner}", flush=True)
            rc = -1
        print(f"[T2] ARM {name} finished rc={rc} in {time.time() - t0:.1f}s", flush=True)
        if rc != 0:
            print(f"[T2] ARM FAILED (rc={rc}): {banner}", flush=True)
            done[name] = None
        else:
            done[name] = out_dir
    return done


def _compare(label: str, a: Path | None, b: Path | None, *, name: str,
             verdicts: list, fatal: bool = True, tag: str = "T2") -> None:
    """One bit-exact comparison + its VERDICT line. Never loosens to a tolerance.

    ``tag`` only changes the printed ``VERDICT {tag} ...`` / ``INFO {tag} ...`` prefix
    (default ``"T2"``, this file's own tier), mirroring ``_compare_replay_band``'s: T3
    (`tests/mia/parity/t3_crossbranch.py`) reuses this comparator for its bit-exact gates and its
    log must be greppable as T3's.
    """
    tag = f"VERDICT {tag}" if fatal else f"INFO {tag}"
    if a is None or b is None:
        print(f"{tag} {label}: FAIL (an arm did not run)", flush=True)
        verdicts.append((label, False, fatal))
        return
    problems = compare(Path(a), Path(b), atol=None, name=name)
    n_a = len(list(Path(a).rglob(name)))
    n_b = len(list(Path(b).rglob(name)))
    print(f"COUNT {label}: A={n_a} files ({a})  B={n_b} files ({b})", flush=True)
    for problem in problems[:20]:
        print("MISMATCH", problem, flush=True)
    if len(problems) > 20:
        print(f"MISMATCH ... and {len(problems) - 20} more", flush=True)
    ok = not problems
    print(f"{tag} {label}: {'PASS' if ok else 'FAIL'} ({len(problems)} problems)", flush=True)
    verdicts.append((label, ok, fatal))


def replica_identity(tree: Path | None, label: str, verdicts: list, *, fatal: bool) -> None:
    """Within ONE run, requests carrying the SAME prompt at DIFFERENT batch rows.

    ``prompts_for`` repeats the 3-prompt set, so req0, req3, req6 ... are the same prompt
    scheduled into different rows of the same steps. This is the one alone-vs-batch-shaped
    check with NO confound at all: one engine, one batch composition, one set of kernels, so
    a difference between two replicas can only come from which rows each request was given.
    Run on BOTH the eager and the FULL tree, because the answer is only interpretable next
    to the other arm's.
    """
    import torch
    from safetensors.torch import load_file

    if tree is None:
        print(f"INFO T2 {label}: SKIPPED (arm did not run)", flush=True)
        verdicts.append((label, False, fatal))
        return
    reqs = sorted(int(d.name[3:]) for d in Path(tree).glob("req*"))
    n_prompts = 3
    pairs = worst = 0
    worst_key = ""
    groups = 0
    for base in range(n_prompts):
        group = [r for r in reqs if r % n_prompts == base]
        if len(group) < 2:
            continue
        groups += 1
        ref = group[0]
        for path in sorted((Path(tree) / f"req{ref}").glob("layer*.safetensors")):
            want = load_file(str(path))
            for other in group[1:]:
                got = load_file(str(Path(tree) / f"req{other}" / path.name))
                for key, a in want.items():
                    b = got[key]
                    pairs += 1
                    if a.shape != b.shape:
                        print(f"MISMATCH {label} req{ref} vs req{other} {path.name}::{key}: "
                              f"shape {tuple(a.shape)} != {tuple(b.shape)}", flush=True)
                        worst = float("inf"); worst_key = f"{path.name}::{key}"
                    elif not torch.equal(a, b):
                        d = (a.float() - b.float()).abs().max().item()
                        if d > worst:
                            worst, worst_key = d, f"req{ref}/req{other} {path.name}::{key}"
    tag = "VERDICT T2" if fatal else "INFO T2"
    # AN EMPTY COMPARISON IS NOT AGREEMENT. With fewer than `n_prompts * 2` requests no
    # prompt has two replicas, every loop above is skipped, and the line this used to print
    # was `PASS (0 replica tensor pairs, worst max|d|=0.000000e+00)` -- the exact string
    # shape this branch removed everywhere else, reading as "perfect agreement" while
    # meaning "nothing was compared". It is reachable: the suite's own 3-request hermetic
    # fixture produces it, and a `full_batch` leg that ever stopped repeating its prompt set
    # would produce it on the real run, silently, with 180 pairs becoming 0 and the verdict
    # staying green.
    if pairs == 0:
        print(f"MISMATCH {label}: no replica pair exists to compare -- {len(reqs)} request(s) "
              f"across {n_prompts} prompts gave {groups} group(s) with 2+ members. This "
              f"check needs the prompt set to REPEAT; with nothing compared, a PASS here "
              f"would mean 'no disagreement was observed' only in the sense that no "
              f"observation was made.", flush=True)
        print(f"{tag} {label}: FAIL (0 replica tensor pairs -- VACUOUS, see MISMATCH above)",
              flush=True)
        verdicts.append((label, False, fatal))
        return
    ok = worst == 0
    print(f"{tag} {label}: {'PASS' if ok else 'FAIL'} "
          f"({pairs} replica tensor pairs from {groups} replica group(s), "
          f"worst max|d|={worst:.6e} at {worst_key or 'n/a'})", flush=True)
    verdicts.append((label, ok, fatal))


def _compare_replay_band(label: str, a: Path | None, b: Path | None, *, name: str,
                         verdicts: list, rel: float = _REPLAY_REL_BAND,
                         row: float = _REPLAY_ROW_BAND, tag: str = "T2") -> None:
    """Magnitude gate for the arms that actually replay a CUDA graph.

    Float tensors must satisfy BOTH bands: tensor-global `max|a-b| / max|a| <= rel`, and
    per-row `max_r (max|a[r]-b[r]| / max|a[r]|) <= row`. The per-row one is what catches a
    destroyed row whose own magnitude is small. Integer tensors are bit-exact, and the two
    key sets must match. See the `_REPLAY_REL_BAND` comment for where the bounds come from
    and for the one thing neither excludes.

    ``tag`` only changes the printed ``VERDICT {tag} ...`` prefix (default ``"T2"``, this
    file's own suite) -- task D7 reuses this function, unmodified otherwise, for the
    `T1-batched` gate in `run_parity.sh`, which is not part of T2's leg/workload machinery.
    """
    import torch
    from safetensors.torch import load_file

    if a is None or b is None:
        print(f"VERDICT {tag} {label}: FAIL (an arm did not run)", flush=True)
        verdicts.append((label, False, True))
        return

    a_files = {p.relative_to(Path(a)).as_posix() for p in Path(a).rglob(name)}
    b_files = {p.relative_to(Path(b)).as_posix() for p in Path(b).rglob(name)}
    print(f"COUNT {label}: A={len(a_files)} files ({a})  B={len(b_files)} files ({b})",
          flush=True)

    problems: list[str] = []
    if not a_files:
        problems.append(f"side A has no artifacts matching {name!r} -- this arm produced NOTHING")
    if not b_files:
        problems.append(f"side B has no artifacts matching {name!r} -- this arm produced NOTHING")
    # Both directions: an artifact present only on B is a LEAKED extra request, and a
    # side-A-only walk would never visit it.
    for extra in sorted(b_files - a_files)[:5]:
        problems.append(f"{extra}: present on side B only (a leaked extra request?)")
    for missing in sorted(a_files - b_files)[:5]:
        problems.append(f"{missing}: present on side A only")

    worst_rel, worst_rel_key, worst_row, worst_row_key, n = 0.0, "", 0.0, "", 0
    for rel_path in sorted(a_files & b_files):
        x_all = load_file(str(Path(a) / rel_path))
        y_all = load_file(str(Path(b) / rel_path))
        if x_all.keys() != y_all.keys():
            problems.append(f"{rel_path}: tensor key sets differ "
                            f"({sorted(x_all)} vs {sorted(y_all)})")
            continue
        for key, x in x_all.items():
            y = y_all[key]
            if x.shape != y.shape:
                problems.append(f"{rel_path}::{key}: shape {tuple(x.shape)} != "
                                f"{tuple(y.shape)} -- a whole-request mis-route between "
                                f"requests of different length lands here, with no "
                                f"tolerance involved")
                continue
            n += 1
            if not x.is_floating_point():
                if not torch.equal(x, y):
                    problems.append(f"{rel_path}::{key}: integer metadata NOT bit-exact")
                continue
            xf, yf = x.float(), y.float()
            scale = xf.abs().max().item()
            if scale == 0.0:
                if not torch.equal(x, y):
                    problems.append(f"{rel_path}::{key}: all-zero on A but not on B")
                continue
            r = (xf - yf).abs().max().item() / scale
            if r > worst_rel:
                worst_rel, worst_rel_key = r, f"{rel_path}::{key}"
            if r > rel:
                problems.append(f"{rel_path}::{key}: rel={r:.3e} > band={rel:.3e}")
            # Per-row: normalise inside each row so a destroyed row costs the same whatever
            # its magnitude relative to the rest of the tensor.
            dims = tuple(range(1, xf.dim())) or (0,)
            if xf.dim() >= 2:
                num = (xf - yf).abs().amax(dim=dims)
                den = xf.abs().amax(dim=dims).clamp_min(1e-30)
                rr = (num / den).max().item()
                if rr > worst_row:
                    worst_row, worst_row_key = rr, f"{rel_path}::{key}"
                if rr > row:
                    problems.append(f"{rel_path}::{key}: per-row={rr:.3e} > band={row:.3e}")

    for problem in problems[:20]:
        print("MISMATCH", problem, flush=True)
    if len(problems) > 20:
        print(f"MISMATCH ... and {len(problems) - 20} more", flush=True)
    ok = not problems
    print(f"VERDICT {tag} {label}: {'PASS' if ok else 'FAIL'} ({n} tensors, worst rel="
          f"{worst_rel:.3e} at {worst_rel_key or 'n/a'} [band {rel:.3e}], worst per-row="
          f"{worst_row:.3e} at {worst_row_key or 'n/a'} [band {row:.3e}])", flush=True)
    verdicts.append((label, ok, True))


# ---------------------------------------------------------------------------
# Trajectory bifurcation (task D9, diagnosed in task-D8-report.md) — the blind spot in
# `_compare_replay_band` for every eager<->FULL replay leg.
# ---------------------------------------------------------------------------
# task D8's investigation of `hs_small routing-identity-w33-distinct-replay` (req25, LSF
# 1726855) found NOT a MIA defect: the FULL-cudagraph arm's compiled kernels moved a
# borderline greedy step's logprob by 1.15e-02 nats -- inside the already-published
# FULL-graph compile-fusion delta this file bands elsewhere (`_T0_GRAPH_LOGPROB_BAND`, worst
# measured 1.815024e-02; the gpu-routing precedent at 1.977980e-02) -- and that was enough to
# flip the argmax at a step whose top probability was only ~9%. From decode index 3 onward
# the two arms therefore generated, and captured, DIFFERENT TOKENS: eager token 14448 vs
# FULL token 1068. `prompt_len=36` predicts the first affected row at `36 + 3 = 39`; the
# measured first bad row WAS 39, and every row before it sat at the ~1-ULP floor (8.1e-04).
#
# THE GAP THIS CLOSES. `_compare_replay_band` is invoked with `name="layer*.safetensors"`
# only -- it never looks at `generation.safetensors`. Every leg here fixes `max_tokens`, so a
# bifurcation keeps ROW COUNTS EQUAL and cannot surface as the SHAPE mismatch the suite's
# `*_batch_distinct` design otherwise relies on to catch a whole-request problem with no
# tolerance involved; it surfaces instead as an unabsorbable magnitude blowup on the rows
# that fed the divergent token, indistinguishable at a glance from real tensor corruption --
# exactly what cost task D8 a full GPU-free investigation. All three eager<->FULL replay legs
# (`routing-identity-w33-distinct-replay`, `replay-band-w32`, `replay-band-w1`) carry this
# exposure; `_compare_replay_band_checked` below is the fix, used ONLY by those three call
# sites in `judge_capture` -- every OTHER caller of `_compare_replay_band` (builder
# equivalence, gpu-routing, T1-batched) is a FULL-vs-FULL or independent-oracle comparison
# outside this diagnosis's scope and is left untouched.
#
# WHAT A BIFURCATION MEANS, AND WHY IT IS NOT FATAL BY ITSELF. Capture cannot alter
# generation -- MIA's baked op and the forward hook are both pure observers -- so a
# trajectory split here is not evidence of a capture defect; it is the SAME published
# FULL-graph logprob perturbation (README "Known Limitations", `_T0_GRAPH_LOGPROB_BAND`)
# landing, on an unlucky boot, close enough to an existing near-tie to move the argmax. That
# is a known, already-accepted consequence of running FULL graphs, not a new failure mode, so
# making the suite FAIL every time it happens would fail it for a limitation it already
# tolerates elsewhere.
#
# BUT A BIFURCATION IS NOT A BLANKET EXCUSE. `_compare_replay_band_checked` still fails the
# leg when:
#   (1) the rows BEFORE the divergence disagree beyond the existing band. The shared prefix
#       consumed the SAME tokens on both arms (that is the entire premise that lets rows
#       0..38 sit at 8.1e-04 in the diagnosis), so it stays a live, banded comparison exactly
#       as if no divergence had ever happened -- a bifurcation only EXCUSES the rows it
#       cannot explain, the ones that consumed a token the other arm never generated.
#   (2) the bifurcation carries NO explanatory logprob delta. Greedy sampling from
#       bit-identical logits cannot produce two different argmaxes, so a "bifurcation" whose
#       |d(logprob)| does not clear the suite's own bit-exact floor is not a numerical
#       tie-flip -- it is a different bug (a token-stream misalignment, a stale index read)
#       wearing the tie-flip's clothes, and must fail loudly rather than being waved through.
#
# `_BIFURCATION_LOGPROB_FLOOR` reuses `_STEER_ATOL` (the suite's own bit-exact logprob
# tolerance) rather than inventing a new number: it is already the value this file treats as
# "no numeric difference worth the name" everywhere else. It is deliberately NOT run through
# `_band()` -- that helper's "refuse to widen" direction protects a MAXIMUM tolerance (bigger
# = weaker); this constant is a MINIMUM (bigger = STRICTER, since fewer deltas would clear
# it), so reusing `_band()`'s widen-refusal here would guard the wrong direction and could be
# read as protection it does not actually provide. There is no environment override.
_BIFURCATION_LOGPROB_FLOOR = _STEER_ATOL


def _load_generation_tensors(root: Path, req: str,
                             name: str = "generation.safetensors") -> dict | None:
    """``generation.safetensors`` for one request, or ``None`` if it is not there.

    ``name`` selects which per-request file to read (default the steered/plain generation).
    T3 passes ``"control.safetensors"`` so the unsteered control arm of ``steer_small`` gets
    the same bifurcation handling as the steered one instead of none at all.

    Missing is not this module's problem to raise on -- every OTHER gate in this file
    already asserts `generation.safetensors` exists for the workloads that need it; treating
    "absent" as "no divergence information available" keeps this detector a pure ADDITION
    that never invents a new failure mode of its own.
    """
    from safetensors.torch import load_file

    path = Path(root) / req / name
    if not path.exists():
        return None
    return load_file(str(path))


def _bifurcation_report(a: Path, b: Path, reqs: set,
                        name: str = "generation.safetensors") -> tuple[dict, list, list]:
    """Per-request trajectory-bifurcation detector for an eager<->FULL replay pair.

    Returns ``(cutoffs, problems, notes)``:

    * ``cutoffs[req] = (divergence_row, total_rows)`` for every request whose generated
      tokens diverged WITH an explanatory logprob delta. Rows ``[0, divergence_row)`` are
      still live for the tensor comparison; rows ``>= divergence_row`` fed a token the other
      arm never generated and are excluded from ANY band, not merely widened for.
    * ``problems`` -- pre-formatted FATAL findings: a divergence with no explanatory delta,
      or one with no logprob channel to judge it by. These are folded into the caller's
      `problems` list, so they fail the leg the same way a tensor mismatch would.
    * ``notes`` -- ``BIFURCATION`` log lines, printed before any ``MISMATCH`` line so a
      reader can never mistake this for capture corruption.
    """
    import torch

    cutoffs: dict = {}
    problems: list = []
    notes: list = []
    for req in sorted(reqs):
        ga = _load_generation_tensors(a, req, name)
        gb = _load_generation_tensors(b, req, name)
        if ga is None or gb is None:
            continue
        pa, pb = ga.get("prompt_token_ids"), gb.get("prompt_token_ids")
        if pa is None or pb is None:
            continue
        if pa.shape != pb.shape or not torch.equal(pa, pb):
            problems.append(
                f"{req}: prompt_token_ids differ between the two arms -- a bifurcation "
                f"check is meaningless when the two sides did not run the same request")
            continue
        prompt_len = int(pa.shape[0])
        ta, tb = ga.get("token_ids"), gb.get("token_ids")
        if ta is None or tb is None:
            continue
        n = min(int(ta.shape[0]), int(tb.shape[0]))
        idx = next((i for i in range(n) if int(ta[i]) != int(tb[i])), None)
        if idx is None:
            continue  # identical trajectories (or a length mismatch, which is a SHAPE
                      # problem the tensor-level row-count audit elsewhere already catches)
        row = prompt_len + idx
        total_rows = prompt_len + n - 1
        lp_a, lp_b = ga.get("token_logprobs"), gb.get("token_logprobs")
        delta = None
        if (torch.is_tensor(lp_a) and torch.is_tensor(lp_b)
                and idx < lp_a.shape[0] and idx < lp_b.shape[0]):
            delta = abs(float(lp_a[idx]) - float(lp_b[idx]))
        header = (f"{req}: TRAJECTORY BIFURCATION at decode index {idx} (row {row}) -- "
                 f"token {int(ta[idx])} (side A) vs {int(tb[idx])} (side B)")
        if delta is None:
            problems.append(
                f"{header}: no token_logprobs channel exists to explain it. A bifurcation "
                f"must carry a logprob delta that accounts for the flip; treating an "
                f"unexplained one as a DIFFERENT bug (not a tie-flip). FATAL.")
            continue
        if delta <= _BIFURCATION_LOGPROB_FLOOR:
            problems.append(
                f"{header}: |d(logprob)|={delta:.3e} does not clear the bit-exact floor "
                f"({_BIFURCATION_LOGPROB_FLOOR:.3e}) -- greedy sampling from bit-identical "
                f"logits cannot produce two different argmaxes, so this is a DIFFERENT bug "
                f"wearing the tie-flip's clothes, not a numeric tie. FATAL.")
            continue
        notes.append(
            f"BIFURCATION {header}, |d(logprob)|={delta:.3e}. This is a GREEDY TIE-FLIP "
            f"under a FULL-graph compile -- a published limitation (README 'Known "
            f"Limitations', _T0_GRAPH_LOGPROB_BAND), NOT capture corruption. Tensors are "
            f"compared only for rows 0..{row - 1} ({row} of {total_rows} rows); rows "
            f"{row}..{total_rows - 1} ({total_rows - row} rows) are SKIPPED -- they hold the "
            f"activations of a token the other arm never generated and are not comparable "
            f"under any band.")
        cutoffs[req] = (row, total_rows)
    return cutoffs, problems, notes


def _compare_replay_band_checked(label: str, a: Path | None, b: Path | None, *, name: str,
                                 verdicts: list, rel: float = _REPLAY_REL_BAND,
                                 row: float = _REPLAY_ROW_BAND, tag: str = "T2") -> None:
    """``_compare_replay_band``, made trajectory-bifurcation-aware (task D9).

    Reads ``generation.safetensors`` for every request FIRST and reports any divergence
    explicitly (see the module comment above `_BIFURCATION_LOGPROB_FLOOR`), then runs
    EXACTLY the same shape / key-set / integer-metadata / rel+per-row-band logic
    `_compare_replay_band` runs, except that a request with an EXPLAINED bifurcation has its
    floating tensors truncated to the rows before the divergence -- the rows after it are
    excluded from comparison entirely, not merely subjected to a wider band. Integer
    metadata (``layer_num``, ``k_prefix_ends``) is never truncated: it does not depend on
    which token was sampled, only on counts, so it stays bit-exact over the WHOLE tensor.

    A request whose bifurcation has NO explanatory logprob delta fails the leg outright
    (folded into ``problems`` by `_bifurcation_report`) -- a bifurcation is not a blanket
    excuse. A request with no divergence at all is judged identically to
    `_compare_replay_band`.
    """
    import torch
    from safetensors.torch import load_file

    if a is None or b is None:
        print(f"VERDICT {tag} {label}: FAIL (an arm did not run)", flush=True)
        verdicts.append((label, False, True))
        return

    a_files = {p.relative_to(Path(a)).as_posix() for p in Path(a).rglob(name)}
    b_files = {p.relative_to(Path(b)).as_posix() for p in Path(b).rglob(name)}
    print(f"COUNT {label}: A={len(a_files)} files ({a})  B={len(b_files)} files ({b})",
          flush=True)

    reqs = {rel_path.split("/", 1)[0] for rel_path in (a_files | b_files) if "/" in rel_path}
    cutoffs, bifurcation_problems, notes = _bifurcation_report(a, b, reqs)
    for note in notes:
        print(note, flush=True)

    problems: list = list(bifurcation_problems)
    if not a_files:
        problems.append(f"side A has no artifacts matching {name!r} -- this arm produced NOTHING")
    if not b_files:
        problems.append(f"side B has no artifacts matching {name!r} -- this arm produced NOTHING")
    for extra in sorted(b_files - a_files)[:5]:
        problems.append(f"{extra}: present on side B only (a leaked extra request?)")
    for missing in sorted(a_files - b_files)[:5]:
        problems.append(f"{missing}: present on side A only")

    worst_rel, worst_rel_key, worst_row, worst_row_key, n = 0.0, "", 0.0, "", 0
    for rel_path in sorted(a_files & b_files):
        req = rel_path.split("/", 1)[0]
        cutoff = cutoffs.get(req, (None, None))[0]
        x_all = load_file(str(Path(a) / rel_path))
        y_all = load_file(str(Path(b) / rel_path))
        if x_all.keys() != y_all.keys():
            problems.append(f"{rel_path}: tensor key sets differ "
                            f"({sorted(x_all)} vs {sorted(y_all)})")
            continue
        for key, x in x_all.items():
            y = y_all[key]
            if x.shape != y.shape:
                problems.append(f"{rel_path}::{key}: shape {tuple(x.shape)} != "
                                f"{tuple(y.shape)} -- a whole-request mis-route between "
                                f"requests of different length lands here, with no "
                                f"tolerance involved")
                continue
            n += 1
            if not x.is_floating_point():
                # Integer metadata (layer_num, k_prefix_ends) never depends on WHICH token
                # was sampled -- only on counts and config -- so it is checked bit-exact over
                # the WHOLE tensor, past any divergence: truncating it would only weaken a
                # check a bifurcation cannot affect either way.
                if not torch.equal(x, y):
                    problems.append(f"{rel_path}::{key}: integer metadata NOT bit-exact")
                continue
            xf, yf = x.float(), y.float()
            suffix = ""
            if cutoff is not None:
                if cutoff <= 0 or xf.dim() < 1:
                    continue  # nothing before the divergence to compare
                if xf.shape[0] > cutoff:
                    xf, yf = xf[:cutoff], yf[:cutoff]
                    suffix = f" (rows 0..{cutoff - 1}, pre-divergence only)"
            if xf.numel() == 0:
                continue
            scale = xf.abs().max().item()
            if scale == 0.0:
                if not torch.equal(xf, yf):
                    problems.append(f"{rel_path}::{key}: all-zero on A but not on B{suffix}")
                continue
            r = (xf - yf).abs().max().item() / scale
            if r > worst_rel:
                worst_rel, worst_rel_key = r, f"{rel_path}::{key}"
            if r > rel:
                problems.append(f"{rel_path}::{key}: rel={r:.3e} > band={rel:.3e}{suffix}")
            dims = tuple(range(1, xf.dim())) or (0,)
            if xf.dim() >= 2:
                num = (xf - yf).abs().amax(dim=dims)
                den = xf.abs().amax(dim=dims).clamp_min(1e-30)
                rr = (num / den).max().item()
                if rr > worst_row:
                    worst_row, worst_row_key = rr, f"{rel_path}::{key}"
                if rr > row:
                    problems.append(f"{rel_path}::{key}: per-row={rr:.3e} > band={row:.3e}"
                                    f"{suffix}")

    for problem in problems[:20]:
        print("MISMATCH", problem, flush=True)
    if len(problems) > 20:
        print(f"MISMATCH ... and {len(problems) - 20} more", flush=True)
    ok = not problems
    if cutoffs:
        total_compared = sum(c[0] for c in cutoffs.values())
        total_skipped = sum(c[1] - c[0] for c in cutoffs.values())
        bif_summary = (f"; {len(cutoffs)} request(s) bifurcated: {total_compared} "
                       f"pre-divergence rows compared, {total_skipped} post-divergence rows "
                       f"skipped (see BIFURCATION lines above)")
    else:
        bif_summary = ""
    print(f"VERDICT {tag} {label}: {'PASS' if ok else 'FAIL'} ({n} tensors, worst rel="
          f"{worst_rel:.3e} at {worst_rel_key or 'n/a'} [band {rel:.3e}], worst per-row="
          f"{worst_row:.3e} at {worst_row_key or 'n/a'} [band {row:.3e}]{bif_summary})",
          flush=True)
    verdicts.append((label, ok, True))


def _compare_generation_band(label: str, a: Path | None, b: Path | None, *, verdicts: list,
                            bound: float, note: str, tag: str = "T2",
                            name: str = "generation.safetensors") -> None:
    """Banded gate over generation: PROMPTS and TOKEN IDS bit-exact, logprobs within `bound`.

    The token ids are the part that must never move -- they are what the model actually
    produced. The logprobs carry a measured, published, non-port delta, so they get a band
    rather than a tolerance-free equality that would pin the suite to FAIL.

    ``tag`` only changes the printed ``VERDICT <tag> ...`` prefix (default ``"T2"``, this
    file's own tier); ``--compare-t1-batched-generation`` (task D10) passes ``tag="T1"``,
    mirroring ``_compare_replay_band``'s own ``tag`` parameter, since that CLI path judges a
    leg with no T2 workload/LEGS entry of its own.

    ``name`` selects the per-request file (default ``generation.safetensors``), mirroring
    ``_compare_replay_band``'s ``name``. ``steer_small`` writes a SECOND logprob file --
    ``control.safetensors``, the unsteered arm -- and until T3 passed ``name`` here, that
    file was bit-exact within a version and structurally checked across versions but never
    VALUE-banded across them, so a drift confined to the control would have failed nothing
    (task E1 review, finding 9).

    TWO SILENT-VACUITY HOLES, closed (task D5 item 3).

    1. This function used to `continue` past a logprob channel that was MISSING on one side
       or SHAPE-MISMATCHED, and then print `worst |d(logprob)|=0.000000e+00` -- which reads
       as perfect agreement and is in fact "nothing was compared". An arm that stopped
       emitting `token_logprobs` at all scored a clean PASS. The key sets must now MATCH
       (the check `compare()` and `_compare_replay_band` already had), a shape mismatch is a
       FAILURE, and a file pair that yielded NO comparable float channel is a failure too --
       so the printed worst is always over something.
    2. It never compared `prompt_token_ids`, so the T0-graph gate did not verify that the
       two arms ran the SAME PROMPTS. Two trees generated from different inputs could agree
       on nothing whatsoever and still be adjudicated on their logprob distance alone. The
       prompts are held bit-exact, like the token ids.
    """
    import torch
    from safetensors.torch import load_file

    if a is None or b is None:
        print(f"VERDICT {tag} {label}: FAIL (an arm did not run)", flush=True)
        verdicts.append((label, False, True))
        return
    files = sorted(Path(a).rglob(name))
    if not files:
        print(f"VERDICT {tag} {label}: FAIL (side A produced no {name} artifacts)",
              flush=True)
        verdicts.append((label, False, True))
        return
    worst, problems, compared = 0.0, [], 0
    for path in files:
        rel = path.relative_to(Path(a))
        other = Path(b) / rel
        if not other.exists():
            problems.append(f"{rel}: missing on side B")
            continue
        x, y = load_file(str(path)), load_file(str(other))
        # KEY SETS FIRST. Everything below is vacuous when a channel exists on one side
        # only, and skipping it is what made an empty comparison print 0.000000e+00.
        if x.keys() != y.keys():
            problems.append(f"{rel}: tensor key sets differ ({sorted(x)} vs {sorted(y)}) -- "
                            f"a channel present on one side only is NOT agreement")
            continue
        # BIT-EXACT, both of them. `prompt_token_ids` is what says the two arms ran the
        # same request at all; `token_ids` is what says the model produced the same thing.
        for key in ("prompt_token_ids", "token_ids"):
            if key not in x:
                problems.append(f"{rel}::{key}: absent from BOTH sides -- this gate cannot "
                                f"tell whether the two arms ran the same generation")
            elif x[key].shape != y[key].shape:
                problems.append(f"{rel}::{key}: shape {tuple(x[key].shape)} != "
                                f"{tuple(y[key].shape)}")
            elif not torch.equal(x[key], y[key]):
                problems.append(
                    f"{rel}::{key}: NOT bit-exact -- "
                    + ("the two arms did not run the same prompts"
                       if key == "prompt_token_ids" else "generation itself moved"))
        seen = 0
        for key in ("token_logprobs", "cumulative_logprob"):
            if key not in x:
                problems.append(f"{rel}::{key}: absent from BOTH sides -- there is no float "
                                f"channel left for this band to bound")
                continue
            if x[key].shape != y[key].shape:
                problems.append(f"{rel}::{key}: shape {tuple(x[key].shape)} != "
                                f"{tuple(y[key].shape)} -- skipping it would print a worst "
                                f"of 0.000000e+00 for a comparison that never happened")
                continue
            seen += 1
            compared += 1
            d = (x[key].double() - y[key].double()).abs().max().item()
            worst = max(worst, d)
            if d > bound:
                problems.append(f"{rel}::{key}: |d|={d:.3e} > band={bound:.3e}")
        if not seen:
            problems.append(f"{rel}: NO comparable logprob channel -- a 'worst |d|' of 0 "
                            f"here would mean nothing was compared, not that nothing moved")
    for problem in problems[:20]:
        print("MISMATCH", problem, flush=True)
    if len(problems) > 20:
        print(f"MISMATCH ... and {len(problems) - 20} more", flush=True)
    ok = not problems
    print(f"VERDICT {tag} {label}: {'PASS' if ok else 'FAIL'} (BANDED: {note}; worst "
          f"|d(logprob)|={worst:.6e} over {compared} channels, band={bound:.3e}; prompt + "
          f"token ids bit-exact)", flush=True)
    verdicts.append((label, ok, True))


def judge_capture(workload: str, legs: dict, verdicts: list) -> None:
    """The two invariants, decomposed so a failure is attributable.

    The literal comparisons the plan names come first and stay bit-exact. Each is then
    paired with the control that says what a non-zero delta MEANS, and with the
    matched-kernel arm that answers the same question with the confound removed:

      * eager-vs-full CONFOUNDS the capture mechanism with vLLM's compiled-vs-eager
        numerics (an eager arm has torch.compile off entirely), so `op-identity` re-asks it
        with MIA's baked op running against the SAME uncompiled kernels the hooks run
        against.
      * alone-vs-batch CONFOUNDS routing with the model's own batch-variance, so
        `routing-identity-w32` re-asks it between the two capture mechanisms at the SAME
        width, and `replica-identity` asks it inside a SINGLE run.
    """
    L = "layer*.safetensors"      # the captured payload
    G = "generation.safetensors"  # what the model produced

    # ---- (a) eager == FULL --------------------------------------------------------
    # RECORDED, not gated. Bit-exactness across this boundary is not achievable by ANY
    # correct implementation (an eager arm has torch.compile off entirely), so gating on it
    # would pin the suite to FAIL forever and hide real regressions. The numbers are printed
    # and the adjudication is in the report; the gates are op-identity + replay-band-w32.
    _compare(f"{workload} RECORD eager-vs-full (bit-exact)", legs.get("eager_alone"),
             legs.get("full_alone"), name=L, verdicts=verdicts, fatal=False)
    # THE ANSWER to (a): forward hooks vs baked op + aperture, identical kernels.
    _compare(f"{workload} op-identity", legs.get("eager_alone"), legs.get("aperture_alone"),
             name=L, verdicts=verdicts)

    # ---- (b) alone == in-batch ----------------------------------------------------
    alone, batched = legs.get("full_alone"), legs.get("full_batch")
    # RECORDED, not gated, for the same reason: the model is not batch-invariant, and the
    # eager control below fails this comparison by MORE than the graph arm does.
    _compare(f"{workload} RECORD alone-vs-batch (bit-exact)",
             alone / "req0" if alone else None,
             batched / "req0" if batched else None,
             name=L, verdicts=verdicts, fatal=False)
    # THE ANSWER to (b): the same 33-request batch captured two ways. Both arms see the
    # identical forward, so every row either lands where the hook path put it or it does not.
    # QK is BANDED (task D4c) -- see `_ROUTING_IDENTITY_QK_REL_BAND`'s comment: this leg is
    # not self-reproducible boot-to-boot (LSF 1710884), and the noise is vLLM's attention
    # kernel, not MIA's. HS stays bit-exact and fatal; UNTOUCHED.
    if workload == "qk_small":
        _compare(f"{workload} RECORD routing-identity-w32 (bit-exact)",
                 legs.get("eager_batch"), legs.get("aperture_batch"),
                 name=L, verdicts=verdicts, fatal=False)
        _compare_replay_band(f"{workload} routing-identity-w32 [BANDED]",
                             legs.get("eager_batch"), legs.get("aperture_batch"),
                             name=L, verdicts=verdicts,
                             rel=_ROUTING_IDENTITY_QK_REL_BAND,
                             row=_ROUTING_IDENTITY_QK_ROW_BAND)
    else:
        _compare(f"{workload} routing-identity-w32", legs.get("eager_batch"),
                 legs.get("aperture_batch"), name=L, verdicts=verdicts)
    # Same comparison with 33 DISTINCT prompt lengths, so a whole-request mis-route is a
    # shape mismatch rather than something a magnitude band has to catch. See the
    # `*_batch_distinct` leg comment. QK is BANDED (task D9): the SAME width-33 boot
    # nondeterminism as `routing-identity-w32` above, not a mis-route -- see
    # `_ROUTING_IDENTITY_QK_REL_BAND`'s comment for the measurement. HS stays bit-exact and
    # fatal; UNTOUCHED (it is bit-reproducible here).
    if workload == "qk_small":
        _compare(f"{workload} RECORD routing-identity-w33-distinct (bit-exact)",
                 legs.get("eager_batch_distinct"), legs.get("aperture_batch_distinct"),
                 name=L, verdicts=verdicts, fatal=False)
        _compare_replay_band(f"{workload} routing-identity-w33-distinct [BANDED]",
                             legs.get("eager_batch_distinct"), legs.get("aperture_batch_distinct"),
                             name=L, verdicts=verdicts,
                             rel=_ROUTING_IDENTITY_QK_REL_BAND,
                             row=_ROUTING_IDENTITY_QK_ROW_BAND)
    else:
        _compare(f"{workload} routing-identity-w33-distinct", legs.get("eager_batch_distinct"),
                 legs.get("aperture_batch_distinct"), name=L, verdicts=verdicts)
    # ... and the same 33 DISTINCT lengths under REAL capture + replay (task D5 item 6), the
    # one place padded rows and distinct shapes exist together. BANDED by necessity, but the
    # band is not what does the work: a whole-request mis-route here changes the row count,
    # and a SHAPE mismatch is fatal inside `_compare_replay_band_checked` at any band width.
    # See the `full_batch_distinct` leg comment. GENERATION-AWARE (task D9): this is an
    # eager<->FULL boundary, so a greedy tie-flip can bifurcate the trajectory without
    # changing row counts -- see `_compare_replay_band_checked`'s comment.
    _compare_replay_band_checked(f"{workload} routing-identity-w33-distinct-replay [BANDED]",
                                 legs.get("eager_batch_distinct"), legs.get("full_batch_distinct"),
                                 name=L, verdicts=verdicts)
    replica_identity(batched, f"{workload} replica-identity FULL", verdicts, fatal=False)
    replica_identity(legs.get("eager_batch"), f"{workload} replica-identity eager",
                     verdicts, fatal=False)

    # ---- controls (informational): what a non-zero delta above MEANS ---------------
    e_alone, e_batch = legs.get("eager_alone"), legs.get("eager_batch")
    _compare(f"{workload} CONTROL eager alone-vs-batch",
             e_alone / "req0" if e_alone else None,
             e_batch / "req0" if e_batch else None,
             name=L, verdicts=verdicts, fatal=False)
    # THE REPLAY GATE. Hooks vs a REAL FULL-cudagraph replay, 33 requests, padded rows and
    # all -- the only leg in this suite that exercises replay-time routing. Magnitude band by
    # necessity; see _REPLAY_REL_BAND for the bound, its margins, and its blind spots.
    # GENERATION-AWARE (task D9): both are eager<->FULL comparisons, so a greedy tie-flip can
    # bifurcate the trajectory without changing row counts -- see
    # `_compare_replay_band_checked`'s comment.
    _compare_replay_band_checked(f"{workload} replay-band-w32", e_batch, batched, name=L,
                                 verdicts=verdicts)
    _compare_replay_band_checked(f"{workload} replay-band-w1", e_alone, legs.get("full_alone"),
                                 name=L, verdicts=verdicts)
    _compare(f"{workload} CONTROL generation eager-vs-full", e_alone, legs.get("full_alone"),
             name=G, verdicts=verdicts, fatal=False)

    # ---- T0 under FULL graphs: BANDED (see _T0_GRAPH_LOGPROB_BAND) -----------------
    # Token ids bit-exact -- that is the observer contract, and it holds. The logprobs carry
    # a measured, reproducible, published compile-fusion delta, so they get a bound.
    _compare_generation_band(f"{workload} T0-graph capture-off-vs-on",
                             legs.get("t0_off_full"), legs.get("full_alone"),
                             verdicts=verdicts, bound=_T0_GRAPH_LOGPROB_BAND,
                             note="FULL-graph non-interference limitation, published in "
                                  "README + TOLERANCES.md")

    # ---- carried item 2: the converted routing builders [BANDED, task D4f] ---------
    # See `_BUILDER_EQUIV_REL_BAND`'s comment: EVERY builder disagrees with ITSELF
    # boot-to-boot at this width (LSF 1722805), so a bit-exact cross-builder requirement was
    # never achievable, and several cross-builder pairs land BIT-EXACT anyway -- the positive
    # evidence that the builders compute the same routing. The bit-exact comparison is still
    # RECORDED so a run that achieves 0 stays visible.
    for name, label in (("full_batch_legacy", "builder legacy"),
                        ("full_batch_vec", "builder vectorized")):
        if name in LEGS[workload]:
            _compare(f"{workload} RECORD {label} vs default (bit-exact)", batched,
                     legs.get(name), name=L, verdicts=verdicts, fatal=False)
            _compare_replay_band(f"{workload} {label} vs default [BANDED]", batched,
                                 legs.get(name), name=L, verdicts=verdicts,
                                 rel=_BUILDER_EQUIV_REL_BAND, row=_BUILDER_EQUIV_ROW_BAND)

    # ---- carried item 1: device-resident query_start_loc under real replay ---------
    # BANDED, both widths. `MIA_CAPTURE_GPU_ROUTING` is not bit-reproducible against host
    # routing under FULL graph -- measured HS 2.228e-03 relative at width 32 in LSF 1702981
    # and 0.000000e+00 in 1704450 (boot-dependent), 1.876e-03 at width 33. The bit-exact
    # comparison is still RECORDED so a run that achieves 0 is visible; the gate is the band.
    # See _GPU_ROUTING_REL_BAND for why this is a lever property and not a mis-route.
    if "full_batch_gpuroute" in LEGS[workload]:
        gpuroute = legs.get("full_batch_gpuroute")
        _compare(f"{workload} RECORD gpu-routing vs host-routing (bit-exact)", batched,
                 gpuroute, name=L, verdicts=verdicts, fatal=False)
        # CHECKED, not the bare band: this leg's own companion gate records generation
        # differing on 33/33 requests (1.98e-02 measured here, 2.14e-02 in LSF 1738651) --
        # LARGER than the 1.15e-02 delta that produced the one tie-flip this suite has ever
        # observed. It cannot cause a false PASS (the generation gate below holds token ids
        # fatal), but with the bare comparator a tie-flip here would present as unexplained
        # tensor corruption and cost the D8-style bug hunt that D9 closed. The checked
        # comparator names it a bifurcation instead.
        _compare_replay_band_checked(f"{workload} gpu-routing vs host-routing [BANDED]",
                                     batched, gpuroute, name=L, verdicts=verdicts,
                                     rel=_GPU_ROUTING_REL_BAND)
        _compare_generation_band(f"{workload} gpu-routing generation [BANDED]", batched,
                                 gpuroute, verdicts=verdicts,
                                 bound=_T0_GRAPH_LOGPROB_BAND,
                                 note="the two legs do not run the same forward; token ids "
                                      "must still match")
    if "full_batch_notail" in LEGS[workload]:
        # The same question with the second prefill wave removed (see the leg comment).
        notail, notail_gpu = legs.get("full_batch_notail"), legs.get("full_batch_notail_gpuroute")
        _compare(f"{workload} RECORD gpu-routing no-tail (bit-exact)", notail, notail_gpu,
                 name=L, verdicts=verdicts, fatal=False)
        _compare_replay_band_checked(
            f"{workload} gpu-routing vs host-routing, no tail wave [BANDED]",
            notail, notail_gpu, name=L, verdicts=verdicts, rel=_GPU_ROUTING_REL_BAND)
        _compare_generation_band(f"{workload} gpu-routing no-tail generation [BANDED]",
                                 notail, notail_gpu, verdicts=verdicts,
                                 bound=_T0_GRAPH_LOGPROB_BAND,
                                 note="generation differs between these legs (33/33 requests) "
                                      "-- the evidence that capture is not the cause")


def judge_steer(legs: dict, verdicts: list) -> None:
    """Carried item 5: adjudicate the deferred steer stale-config gap BY MEASUREMENT.

    D3 closed the stale-routing hazard for HS/QK with a guard on the drain wrapper, and left
    the identical exposure on steer's ``coeff_all``/``vec_id_all``/``mode_all`` open: there
    is no ``execute_model`` wrapper on the steer path to hang a guard on, and steer captures
    nothing, so there is no never-drop contract at stake. Only a FULL-cudagraph run can
    exercise it at all — that is the mode in which vLLM replays the compiled graph
    unconditionally on a dummy pass, so the baked steer op fires against whatever routing the
    previous REAL step left behind.

    Steering writes no artifact, so the observable is its effect on the logprobs, and three
    numbers decide it — the first two only mean something together:

      * ``d_floor``   — UNSTEERED eager vs UNSTEERED FULL. vLLM's compiled-vs-eager
        numerical floor for this model, with steering out of the picture entirely.
      * ``d_steered`` — STEERED eager vs STEERED FULL.
      * ``d_op``      — STEERED eager vs STEERED matched-kernel aperture arm. No compile
        difference at all, so this one is a clean bit-exactness question about the steer op
        and its routing.

    A stale steer config surviving into a real step would move the steered logprobs by the
    size of a steering intervention — measured at ~7.7e-01 on this workload (see
    tests/mia/parity/TOLERANCES.md), five orders of magnitude above the floor. So
    ``d_steered <= min(max(d_floor, STEER_ATOL), _STEER_BUDGET_CEILING)`` is the gap being
    benign; anything near the intervention scale is the gap being real. The CEILING is not
    decoration: without it the budget is `max(floor, atol)` computed at runtime, so a floor
    that blew up would widen the gate to match -- see `_STEER_BUDGET_CEILING`.
    """
    from tests.mia.parity.capture_workload import STEER_LIVENESS_ATOL

    eager, full = legs.get("eager_alone"), legs.get("full_alone")
    atol = _STEER_ATOL
    if eager is None:
        # Every steer gate is measured AGAINST the eager forward-hook arm, so without it
        # none of them can be answered -- and a gate that cannot be answered FAILS. Driven
        # off EXPECTED_GATES so this bail-out can never fall behind the declared set.
        print("VERDICT T2 steer_small: the eager reference arm did not run -- every steer "
              "gate below fails for want of a reference, none of them disappears",
              flush=True)
        # Everything is marked FATAL on this path, INFO gates included: nothing ran, and a
        # run in that state must not be readable as a partial pass.
        for label in EXPECTED_GATES["steer_small"]:
            print(f"VERDICT T2 {label}: FAIL (the eager reference arm did not run)",
                  flush=True)
            verdicts.append((label, False, True))
        return
    if full is None:
        # A `--only` probe of the matched-kernel arms. The stale-config gap needs a FULL
        # arm (it is the only mode that replays the graph on a dummy pass), but the op
        # comparisons below do not, so fall through instead of skipping them.
        print("VERDICT T2 steer_small steer-gap: FAIL (the FULL-cudagraph arm did not run)",
              flush=True)
        verdicts.append(("steer_small steer-gap", False, True))
        _judge_steer_op(legs, eager, atol, verdicts)
        return

    d_floor = steer_logprob_delta(full, eager, label_a="control", label_b="control")
    d_steered = steer_logprob_delta(full, eager)
    live_full = steer_logprob_delta(full, full, label_b="control")
    live_eager = steer_logprob_delta(eager, eager, label_b="control")

    print(f"[T2] steer floor   : max|d(logprob)| UNSTEERED eager-vs-FULL = {d_floor:.6e}",
          flush=True)
    print(f"[T2] steer effect  : max|d(logprob)| STEERED   eager-vs-FULL = {d_steered:.6e}",
          flush=True)
    print(f"[T2] steer liveness: FULL={live_full:.6e} eager={live_eager:.6e} "
          f"(floor {STEER_LIVENESS_ATOL:.1e})", flush=True)

    alive = live_full > STEER_LIVENESS_ATOL and live_eager > STEER_LIVENESS_ATOL
    # THE CEILING (task D5 item 5). `max(d_floor, atol)` with no upper bound is a gate that
    # WIDENS TO MATCH ITS OWN FAILURE: the floor is the UNSTEERED eager-vs-FULL delta, so a
    # run in which that blew up -- which is itself a finding -- would hand the steered
    # comparison a budget big enough to hide a complete mis-steer. Measured, the floor is
    # 3.635375e-02 in ALL SEVEN GPU jobs to date; `_STEER_BUDGET_CEILING` (1.5e-1) sits 4.1x
    # above it and 5.1x below the smallest real steering intervention (7.632304e-01). A
    # floor past the ceiling is a FAILURE, not a licence.
    floor_ok = d_floor <= _STEER_BUDGET_CEILING
    budget = min(max(d_floor, atol), _STEER_BUDGET_CEILING)
    ok = alive and floor_ok and d_steered <= budget
    print(f"VERDICT T2 steer_small liveness: {'PASS' if alive else 'FAIL'}", flush=True)
    if not floor_ok:
        print(f"MISMATCH steer-gap: the UNSTEERED compiled-vs-eager floor {d_floor:.6e} "
              f"itself exceeds the ceiling {_STEER_BUDGET_CEILING:.3e} (measured "
              f"3.635375e-02 in all 7 GPU jobs) -- the budget is capped rather than widened "
              f"to match, and this run is a FAIL on the floor", flush=True)
    print(f"VERDICT T2 steer_small steer-gap: {'PASS' if ok else 'FAIL'} "
          f"(d_steered={d_steered:.6e} vs budget {budget:.6e} = min(max(floor="
          f"{d_floor:.6e}, atol={atol:.1e}), ceiling={_STEER_BUDGET_CEILING:.3e}))",
          flush=True)
    verdicts.append(("steer_small liveness", alive, True))
    verdicts.append(("steer_small steer-gap", ok, True))

    _judge_steer_op(legs, eager, atol, verdicts)

    # Steer routing at width, and the GPU-vs-host routing control for carried item 1.
    _judge_steer_width(legs, full, atol, verdicts)
    _judge_steer_per_request_arming(legs, atol, verdicts)
    host = legs.get("full_alone_hostroute")
    if host is None:
        # NEVER `if host is not None:` alone. A leg that failed to run must FAIL its gate,
        # not delete it — see EXPECTED_GATES for the run that exited 0 with four gates gone.
        print("VERDICT T2 steer_small gpu-routing vs host-routing: FAIL "
              "(the host-routing arm did not run)", flush=True)
        verdicts.append(("steer_small gpu-routing", False, True))
    else:
        d_route = steer_logprob_delta(full, host)
        route_ok = d_route <= atol
        print(f"[T2] steer routing : max|d(logprob)| GPU-routing vs host-routing = "
              f"{d_route:.6e} (atol {atol:.1e})", flush=True)
        print(f"VERDICT T2 steer_small gpu-routing vs host-routing: "
              f"{'PASS' if route_ok else 'FAIL'}", flush=True)
        verdicts.append(("steer_small gpu-routing", route_ok, True))


def _judge_steer_per_request_arming(legs: dict, atol: float, verdicts: list) -> None:
    """Did each request receive ITS OWN steering, at width 33? (task D5 item 7; A/B fix D7)

    THE GAP THIS CLOSES: every other steer gate runs at ``max_num_seqs=1``, so a steer
    mis-route at width > 1 was unmeasured -- and simply widening the batch would not have
    measured it either, because with every request carrying the SAME config, "each request
    got its own steering" and "each request got its neighbour's" predict identical logprobs.

    The ``full_batch_mixed_arm`` leg arms ALTERNATE requests of a 33-request,
    33-distinct-length batch under real capture + replay. ITS CONTROL IS AN A/B PAIR (task
    D7; see ``_drive_steer_mixed`` and ``run_steer_ab_leak_control.py``), not a fully-unarmed
    baseline: the SAME 16 rows are armed in BOTH passes, and the second pass ("control")
    swaps the vector for an all-zero-``dir`` clone (same ``avg_proj``) instead of running
    with nobody armed. That is what makes the UNARMED rows a bit-exact question rather than
    a magnitude one -- their inputs, shapes and kernel invocations are LITERALLY IDENTICAL
    between the two passes (same rows armed, same config shape; only the ARMED rows' vector
    VALUES differ), so an honest unarmed delta is EXACTLY 0.0, not merely small.

    WHY THIS REPLACED THE OLD CONTROL. LSF 1725139's control ran a differently-composed
    batch (nobody armed anywhere), and 5 of 17 unarmed requests moved by up to 2.575421e-02
    against the 1e-3 floor -- indistinguishable, by construction, from a real per-row leak.
    The decisive A/B experiment (``tests/mia/parity/run_steer_ab_leak_control.py``, LSF 1725670)
    held composition identical and found all 17 unarmed requests bit-exact (0.000000e+00)
    while the weakest ARMED delta was 2.757723e-01: there is NO leakage; the old gate's
    failure was an artifact of comparing two differently-composed batches, not a routing
    defect. See README "Known Limitations" and tests/mia/parity/TOLERANCES.md.

    TASK D10: adopting the A/B design was necessary but not, on its own, sufficient. LSF
    1731353 re-ran this exact gate and still found 3 of 33 unarmed requests moving
    (1.613855e-02 / 2.467152e-02 / 2.032498e-02, all ~12x below the weakest armed effect) --
    because ``_drive_steer_mixed`` was a hand-maintained TWIN of
    ``run_steer_ab_leak_control.py``'s driver rather than a call into it, and the two had
    drifted. It now calls that script's ``run_steer_ab_pass`` directly (one driver, two
    callers).

    TASK D12 -- THE BIT-EXACT UNARMED-LOGPROB PREMISE ITSELF WAS FALSE. D9/D10 kept reading
    LSF 1731353's 3-request finding as boot-to-boot variance across *separate* engine
    processes -- a parameter difference to hunt for. ``tests/mia/parity/run_steer_ab_null_control.py``
    (LSF 1733997) rules that out: it runs the SAME A-vs-B comparison TWICE inside ONE engine
    boot, with nothing else different --

        GATE 1 (P1 real vs P2 zero):  unarmed worst=3.265401e-02  armed worst=1.216572e+01
        GATE 2 (P3 real vs P4 zero):  unarmed worst=0.000000e+00  armed worst=1.216572e+01

    -- and the two draws disagree on the unarmed rows as badly as two separate jobs did.
    P1/P3 are the identical config run twice in the same boot; so are P2/P4. Two passes that
    differ in NOTHING still land on different unarmed-row logprobs, so "an honest unarmed
    delta is EXACTLY 0.0" was never a property of this system to gate on -- no rewording of
    the harness can make a channel bit-exact that is not reproducible pass-to-pass. The SAME
    job's PATH pass (P1 vs P5, identical vector BYTES at a different ``vector_path``) also
    moved ARMED rows by 5.830579e-02 -- pure path noise, no steering difference at all --
    which is why the old positive-control floor (``STEER_LIVENESS_ATOL = 1e-3``) was too
    weak to trust either.

    What DID hold across every pair in that job, both NULLs and the PATH pass included:
    unarmed TOKEN IDS never flipped (0 flips). That is the deterministic channel, so this
    gate now adjudicates on it instead of on logprobs:

      UNARMED rows -- TOKEN IDS must be bit-exact between Arm A and Arm B (FATAL). Same
          spirit as the pre-D12 check ("any difference is a leak"), moved to the channel
          that is actually reproducible rather than the one that only looked like it was.
      ARMED rows -- the positive control is unchanged in kind but the floor is
          ``STEER_ARM_POSITIVE_CONTROL_ATOL`` (``tests/mia/parity/capture_workload.py``), not
          ``STEER_LIVENESS_ATOL``: derived from LSF 1733997's measured PATH-noise ceiling
          (5.830579e-02) and live-steering signal (1.216572e+01) -- see that constant's
          comment for the margins. A future change that silently disabled steering
          everywhere still shows ~0 armed movement and FAILS here.
      UNARMED logprob movement is now INFO, not fatal: printed against the measured
          pass-to-pass envelope (``STEER_UNARMED_LOGPROB_ENVELOPE``, LSF 1733997) so the
          number stays visible without ever failing a run on a channel proven irreproducible
          pass-to-pass within a single boot.

    See ``tests/mia/parity/run_steer_ab_null_control.py`` (kept in the tree as the evidence for
    this decision) and ``tests/mia/parity/TOLERANCES.md`` / README "Known Limitations".
    """
    import torch
    from safetensors.torch import load_file

    from tests.mia.parity.capture_workload import (
        LOGPROB_KEYS,
        STEER_ARM_POSITIVE_CONTROL_ATOL,
        STEER_UNARMED_LOGPROB_ENVELOPE,
    )

    tree = legs.get("full_batch_mixed_arm")
    if tree is None:
        for label in ("steer_small per-request-arming",
                      "steer_small INFO per-request-arming bit-exact"):
            print(f"VERDICT T2 {label}: FAIL (the mixed-arm width-33 steer arm did not run)",
                  flush=True)
            verdicts.append((label, False, True))
        return

    problems: list[str] = []
    n_armed = n_unarmed = 0
    worst_unarmed_logprob, weakest_armed = 0.0, float("inf")
    for req_dir in sorted(Path(tree).glob("req*"), key=lambda p: int(p.name[3:])):
        marker = req_dir / "arming.safetensors"
        gen, ctl = req_dir / "generation.safetensors", req_dir / "control.safetensors"
        if not (marker.exists() and gen.exists() and ctl.exists()):
            problems.append(f"{req_dir.name}: incomplete ({marker.name}/{gen.name}/"
                            f"{ctl.name} must all exist) -- nothing to adjudicate")
            continue
        is_armed = bool(int(load_file(str(marker))["armed"].item()))
        a, b = load_file(str(gen)), load_file(str(ctl))
        shared = [k for k in LOGPROB_KEYS if k in a and k in b and a[k].shape == b[k].shape]
        delta = (max(float((a[k].double() - b[k].double()).abs().max().item())
                     for k in shared) if shared else None)
        if is_armed:
            # THE POSITIVE CONTROL (task D7, floor re-derived task D12; see
            # STEER_ARM_POSITIVE_CONTROL_ATOL). Counted toward n_armed off the marker alone,
            # not off whether a channel to judge it exists -- an armed row that produced no
            # comparable logprob at all is still an armed row, and a batch of nothing-but-
            # unjudgeable armed rows must not silently read as "not mixed".
            n_armed += 1
            if delta is None:
                problems.append(
                    f"{req_dir.name}: ARMED, no comparable logprob channel between the "
                    f"steered run and its control -- the positive control cannot be "
                    f"evaluated, so this request is unjudged")
                continue
            weakest_armed = min(weakest_armed, delta)
            if delta <= STEER_ARM_POSITIVE_CONTROL_ATOL:
                problems.append(
                    f"{req_dir.name}: ARMED but did not clear the positive-control floor "
                    f"({delta:.6e} <= {STEER_ARM_POSITIVE_CONTROL_ATOL:.6e}) -- its "
                    f"steering was routed away, OR Arm B failed to be a real zero-effect "
                    f"control (task D12; floor derived from LSF 1733997's path-noise "
                    f"ceiling 5.830579e-02 and live-steering signal 1.216572e+01)")
        else:
            n_unarmed += 1
            # THE DETERMINISTIC CHANNEL (task D12). Arm A and Arm B arm the SAME rows with
            # the SAME config shape -- only the ARMED rows' vector VALUES differ -- so an
            # unarmed row's generated TOKEN IDS are identical in both passes or one of them
            # took a neighbour's steering. Unlike the logprob channel this replaced, LSF
            # 1733997 measured 0 token-id flips on unarmed rows across every pair it ran,
            # NULLs and PATH pass included: this is the channel that is actually
            # reproducible pass-to-pass, not merely the one that used to look like it was.
            if "token_ids" not in a or "token_ids" not in b:
                problems.append(
                    f"{req_dir.name}: UNARMED, no token_ids channel in one of the two "
                    f"passes -- the deterministic check cannot be evaluated, so this "
                    f"request is unjudged")
                continue
            tok_a, tok_b = a["token_ids"], b["token_ids"]
            tok_exact = tok_a.shape == tok_b.shape and bool(torch.equal(tok_a, tok_b))
            if delta is not None:
                worst_unarmed_logprob = max(worst_unarmed_logprob, delta)
            if not tok_exact:
                problems.append(
                    f"{req_dir.name}: UNARMED but its TOKEN IDS differ between Arm A and "
                    f"Arm B -- it received a NEIGHBOUR's steering (task D12's deterministic "
                    f"channel; logprob delta on this request was "
                    f"{'n/a' if delta is None else f'{delta:.6e}'})")
    # Both classes must be non-empty, or the comparison is vacuous in the direction that
    # matters: 33 armed requests prove nothing about routing, and 33 unarmed ones prove
    # nothing about liveness.
    if not n_armed or not n_unarmed:
        problems.append(f"the mixed-arm batch is not mixed: {n_armed} armed, {n_unarmed} "
                        f"unarmed -- this gate can only separate a correct steer from a "
                        f"mis-routed one when both classes are present")

    for problem in problems[:20]:
        print("MISMATCH", problem, flush=True)
    if len(problems) > 20:
        print(f"MISMATCH ... and {len(problems) - 20} more", flush=True)
    ok = not problems
    print(f"[T2] steer arming : {n_armed} armed (weakest effect "
          f"{0.0 if weakest_armed == float('inf') else weakest_armed:.6e}, floor "
          f"{STEER_ARM_POSITIVE_CONTROL_ATOL:.3e}), {n_unarmed} unarmed (token-id bit-exact "
          f"is the gate; worst logprob movement {worst_unarmed_logprob:.6e} is INFO only)",
          flush=True)
    print(f"VERDICT T2 steer_small per-request-arming: {'PASS' if ok else 'FAIL'} "
          f"({len(problems)} problems)", flush=True)
    verdicts.append(("steer_small per-request-arming", ok, True))

    # RECORDED, not gated (task D12). Pre-D12 this line re-checked bit-exactness on the same
    # logprob channel the FATAL gate used to gate on, and was "redundant by construction"
    # with it -- that reasoning is gone along with the premise: LSF 1733997 (see the
    # docstring) proved the unarmed-logprob bit-exact premise false inside a single boot, so
    # the FATAL gate above no longer reads logprobs at all. This line is what remains of
    # that measurement: worth printing, never worth failing on.
    # STEER_UNARMED_LOGPROB_ENVELOPE (3.265401e-02) is the worst pass-to-pass SPREAD LSF
    # 1733997 measured on this exact quantity -- a description of noise, not a tolerance --
    # so "exceeds it" means only "this boot's noise was bigger than the one sampled so far",
    # never "something is wrong". The gate's no-leakage property rests on the FATAL token-id
    # check and the armed positive control above, not on this number.
    within_envelope = worst_unarmed_logprob <= STEER_UNARMED_LOGPROB_ENVELOPE
    print(f"INFO T2 steer_small INFO per-request-arming bit-exact: "
          f"{'WITHIN' if within_envelope else 'ABOVE'} the measured pass-to-pass envelope "
          f"(worst unarmed |d(logprob)| {worst_unarmed_logprob:.6e} vs envelope "
          f"{STEER_UNARMED_LOGPROB_ENVELOPE:.3e}, LSF 1733997) -- NOT a gate: unarmed "
          f"logprobs are not reproducible pass-to-pass even within one boot; see "
          f"tests/mia/parity/run_steer_ab_null_control.py and TOLERANCES.md", flush=True)
    verdicts.append(("steer_small INFO per-request-arming bit-exact", within_envelope, False))


def _judge_steer_width(legs: dict, full: Path, atol: float, verdicts: list) -> None:
    """req0 steered ALONE vs the same request steered inside a 32-wide batch.

    Task D5 item 2. This leg booted a whole 32-wide steer engine and printed

        [T2] steer width: alone-vs-batch = 1.656e-02

    with NO verdict line and NO `verdicts.append` -- collected evidence, enforced by
    nothing. It is enforced here, and the bound is DERIVED rather than chosen.

    WHAT THE NUMBER CONFOUNDS: the model is not batch-invariant, so req0's logprobs move
    between width 1 and width 32 whether or not steering is involved. The control for that
    is printed on the same line and is the SAME comparison with steering out of the picture
    entirely (each arm's own unsteered `control` tree). MEASURED across 7 GPU jobs
    (1704450, 1705469, 1705794, 1710144, 1719790, 1720902, 1723978):

        STEERED   alone-vs-batch   1.656437e-02 (x6), 1.778936e-02 (x1)  -> worst 1.78e-02
        UNSTEERED alone-vs-batch   2.933863e-02 .. 5.865807e-02          -> worst 5.87e-02

    The steered delta came in BELOW the unsteered batch-variance in all 7 -- i.e. widening
    the batch moves req0 by the same amount with or without a steer, which is the invariant
    ("the steer req0 receives at width 32 is the steer it receives alone").

    SO THE BUDGET IS THE CONTROL, WITH A CEILING. `max(control, atol)` alone would be a
    self-widening gate -- exactly the defect `steer-gap` had -- so it is capped by
    `_STEER_BUDGET_CEILING`, and a control that blows PAST the ceiling is itself a failure
    rather than a licence. Discrimination: a steer mis-route at width would apply the wrong
    request's (or no) steering, which is a full intervention, measured at 7.63e-01..8.15e-01
    on this workload -- 5.1x above the ceiling, 43x above the worst steered measurement.
    """
    batched = legs.get("full_batch")
    if batched is None:
        print("VERDICT T2 steer_small width [BANDED]: FAIL (the 32-wide steer arm did not "
              "run)", flush=True)
        verdicts.append(("steer_small width [BANDED]", False, True))
        return
    d_width = steer_logprob_delta(full, batched)
    d_ctrl = steer_logprob_delta(full, batched, label_a="control", label_b="control")
    budget = min(max(d_ctrl, atol), _STEER_BUDGET_CEILING)
    floor_ok = d_ctrl <= _STEER_BUDGET_CEILING
    ok = floor_ok and d_width <= budget
    print(f"[T2] steer width   : max|d(logprob)| STEERED alone-vs-batch = {d_width:.6e} "
          f"(unsteered control {d_ctrl:.6e})", flush=True)
    if not floor_ok:
        print(f"MISMATCH steer width: the UNSTEERED batch-variance control {d_ctrl:.6e} "
              f"itself exceeds the ceiling {_STEER_BUDGET_CEILING:.3e} -- the budget is "
              f"capped rather than widened to match, and this run is a FAIL on the floor",
              flush=True)
    print(f"VERDICT T2 steer_small width [BANDED]: {'PASS' if ok else 'FAIL'} "
          f"(d_width={d_width:.6e} vs budget {budget:.6e} = "
          f"min(max(control={d_ctrl:.6e}, atol={atol:.1e}), ceiling="
          f"{_STEER_BUDGET_CEILING:.3e}))", flush=True)
    verdicts.append(("steer_small width [BANDED]", ok, True))


def _judge_steer_op(legs: dict, eager: Path, atol: float, verdicts: list) -> None:
    """The steer OP itself, with vLLM's compiled-vs-eager numerics removed.

    GPU-measured (LSF 1703642): with the kernels matched, `MIA_STEER_FUSED=0` reproduces the
    eager forward-hook steer BIT-EXACTLY (0.000000e+00), while the shipped default
    `MIA_STEER_FUSED=1` sits 2.342296e-02 away on a steering effect of 7.72e-01. So the whole
    gap is the fused Triton kernel's reduction, not the baked op, not the V2 routing.
    """
    # Is the op-identity gap the FUSED Triton kernel's reduction order? Same arm with
    # MIA_STEER_FUSED=0 (the aten reference path the eager worker also runs).
    nofuse = legs.get("aperture_alone_nofuse")
    if nofuse is None:
        print("VERDICT T2 steer_small op-identity-nofuse: FAIL "
              "(the MIA_STEER_FUSED=0 arm did not run)", flush=True)
        verdicts.append(("steer_small op-identity-nofuse", False, True))
    else:
        d_nf = steer_logprob_delta(nofuse, eager)
        d_nf_floor = steer_logprob_delta(nofuse, eager, label_a="control", label_b="control")
        print(f"[T2] steer op nofuse: max|d(logprob)| STEERED eager-vs-aperture(FUSED=0) = "
              f"{d_nf:.6e} (unsteered control {d_nf_floor:.6e}, atol {atol:.1e})", flush=True)
        # FATAL, and it has to be: this is the ONLY bit-exact gate on the steer path. The
        # fused arm below is BANDED (it measures a lever), so if this one were advisory a
        # regression in the aten-reference steer under a 5e-2 logprob delta would pass the
        # build silently -- `main()` fails only on `fatal and not ok`, and run_parity.sh
        # keys off that exit code. It was `fatal=False` from fix round 1 until round 2.
        print(f"VERDICT T2 steer_small op-identity-nofuse: "
              f"{'PASS' if d_nf <= atol else 'FAIL'} ({d_nf:.6e} vs atol {atol:.1e})",
              flush=True)
        verdicts.append(("steer_small op-identity-nofuse", d_nf <= atol, True))

    # The steer op itself, with the compile difference removed.
    aperture = legs.get("aperture_alone")
    if aperture is None:
        print("VERDICT T2 steer_small op-identity [BANDED]: FAIL "
              "(the matched-kernel aperture arm did not run)", flush=True)
        verdicts.append(("steer_small op-identity [BANDED]", False, True))
    else:
        d_op = steer_logprob_delta(aperture, eager)
        d_op_floor = steer_logprob_delta(aperture, eager, label_a="control",
                                         label_b="control")
        # BANDED: this arm runs the DEFAULT fused Triton kernel, a V1-era lever. The
        # port-proving gate is `op-identity-nofuse` above, which is bit-exact at `atol`.
        op_ok = d_op <= _STEER_FUSED_LOGPROB_BAND
        print(f"[T2] steer op      : max|d(logprob)| STEERED eager-vs-aperture = {d_op:.6e} "
              f"(unsteered control {d_op_floor:.6e}, atol {atol:.1e})", flush=True)
        print(f"VERDICT T2 steer_small op-identity [BANDED]: {'PASS' if op_ok else 'FAIL'} "
              f"(MIA_STEER_FUSED lever, published in PUBLIC_LEVERS + README; "
              f"{d_op:.6e} vs band {_STEER_FUSED_LOGPROB_BAND:.3e})", flush=True)
        verdicts.append(("steer_small op-identity [BANDED]", op_ok, True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="store_true",
                    help="run ONE arm in this process (internal; one engine per process)")
    ap.add_argument("--print-bands", action="store_true",
                    help="echo every enforced bound and exit (run_parity.sh calls this "
                         "before any arm boots, so the log records what was enforced)")
    ap.add_argument("workload", nargs="?", choices=sorted(WORKLOADS))
    ap.add_argument("out", nargs="?")
    ap.add_argument("--graph", action="store_true", help="arm MIA's baked-op/aperture path")
    ap.add_argument("--cudagraph", action="store_true", help="let vLLM compile + replay graphs")
    ap.add_argument("--control-arm", default="mia", choices=("mia", "off"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma-separated leg names")
    ap.add_argument("--compare-t1-batched", nargs=2, metavar=("REF_DIR", "MIA_DIR"),
                    default=None,
                    help="band-compare a T1-batched artifact pair with _compare_replay_band "
                         "and exit (task D7). T1-batched has no LEGS/workload entry of its "
                         "own -- run_parity.sh calls this directly instead of going through "
                         "judge_capture/adjudicate.")
    ap.add_argument("--compare-t1-batched-generation", nargs=2,
                    metavar=("REF_DIR", "MIA_DIR"), default=None,
                    help="band-compare a T1-batched generation.safetensors pair with "
                         "_compare_generation_band and exit (task D10 item 1): token ids "
                         "and prompt token ids stay bit-exact and fatal, the logprob "
                         "channels are banded at _T1_BATCHED_LOGPROB_BAND. Mirrors "
                         "--compare-t1-batched -- run_parity.sh calls this directly for the "
                         "same reason (no LEGS/workload entry).")
    ap.add_argument("--name", default="*.safetensors",
                    help="glob passed to --compare-t1-batched (e.g. layer*.safetensors to "
                         "band only the captured payload, leaving generation.safetensors to "
                         "a separate bit-exact compare_artifacts.py call)")
    ap.add_argument("--label", default=None,
                    help="verdict label for --compare-t1-batched[-generation]")
    args = ap.parse_args(argv)

    if args.compare_t1_batched:
        ref_dir, mia_dir = (Path(p) for p in args.compare_t1_batched)
        verdicts: list = []
        _compare_replay_band(args.label or "T1-batched", ref_dir, mia_dir,
                             name=args.name, verdicts=verdicts,
                             rel=_T1_BATCHED_REL_BAND, row=_T1_BATCHED_ROW_BAND, tag="T1")
        return 0 if verdicts[0][1] else 1

    if args.compare_t1_batched_generation:
        ref_dir, mia_dir = (Path(p) for p in args.compare_t1_batched_generation)
        verdicts = []
        _compare_generation_band(
            args.label or "T1-batched generation", ref_dir, mia_dir, verdicts=verdicts,
            bound=_T1_BATCHED_LOGPROB_BAND,
            note="T1-batched vs the independent width-33 oracle, cross-boot logprob noise "
                 "(task D10 item 1, LSF 1731353)", tag="T1")
        return 0 if verdicts[0][1] else 1

    if args.print_bands:
        # Importing this module is what resolves and validates every band, so reaching this
        # line at all already means no override widened one.
        print_bands()
        return 0
    if not args.workload or not args.out:
        ap.error("workload and out are required unless --print-bands is given")
    if args.workload not in LEGS:
        ap.error(f"{args.workload} has no leg matrix in LEGS; T2 judges "
                 f"{sorted(LEGS)}")

    if args.arm:
        run_arm(args.workload, Path(args.out), graph=args.graph, batch=args.batch,
                cudagraph=args.cudagraph, arm=args.control_arm)
        return 0

    only = set(args.only.split(",")) if args.only else None
    legs = run_legs(args.workload, Path(args.out), only)
    print(f"\n=== [T2] {args.workload}: legs "
          f"{json.dumps({k: (v is not None) for k, v in legs.items()})} ===", flush=True)

    verdicts: list = []
    if WORKLOADS[args.workload].subsystem == "steer":
        judge_steer(legs, verdicts)
    else:
        judge_capture(args.workload, legs, verdicts)

    return adjudicate(args.workload, verdicts)


def adjudicate(workload: str, verdicts: list) -> int:
    """The gate-set assertion + the exit code. Separated so it is hermetically testable.

    THE GATE-SET ASSERTION (task D5 item 1). Counting only the verdicts that were emitted
    is how a suite reports PASS with four of its gates deleted: the list simply gets
    shorter and nothing notices. Assert against the DECLARED set instead, in both
    directions, and treat either kind of drift as a fatal invariant of its own.
    """
    missing, undeclared = gate_set_problems(workload, verdicts)
    for label in missing:
        print(f"MISSING GATE {label}: declared in EXPECTED_GATES but NO verdict was emitted",
              flush=True)
    for label in undeclared:
        print(f"UNDECLARED GATE {label}: emitted a verdict but is not in EXPECTED_GATES -- "
              f"declare it, or it can vanish later without anyone noticing", flush=True)
    gates_ok = not missing and not undeclared
    print(f"VERDICT T2 {workload} expected-gates: {'PASS' if gates_ok else 'FAIL'} "
          f"({len(EXPECTED_GATES[workload])} declared, {len(verdicts)} emitted, "
          f"{len(missing)} missing, {len(undeclared)} undeclared)", flush=True)
    verdicts.append((f"{workload} expected-gates", gates_ok, True))

    failed = [label for label, ok, fatal in verdicts if fatal and not ok]
    print(f"VERDICT T2 {workload}: {'PASS' if not failed else 'FAIL'} "
          f"({len(failed)} failing invariants: {failed})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
