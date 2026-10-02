"""The steer gates must accept the measured deltas and reject an intervention-scale one.

Task D5 items 2, 4 and 5. Three of this suite's steer bounds were round numbers with NO
discrimination test and an env override, so

    MIA_T2_STEER_FUSED_BAND=10 pytest

passed a totally broken fused steer -- a band ten times the size of the effect it is
supposed to bound. And two budgets were computed AT RUNTIME from a measured floor with no
upper bound, so a floor that blew up widened the gate to match.

Every bound exercised here is pinned from BOTH sides on the same synthetic trees the real
judges read:

  ACCEPT  the largest value ever measured on GPU for that comparison;
  REJECT  a corruption at the scale the gate exists to catch -- for steer that is the
          intervention itself, measured at 7.632304e-01 .. 8.145643e-01 on this workload
          across LSF 1704450 / 1705469 / 1705794 / 1710144 / 1719790 / 1720902 / 1723978.

A band with no reject test is a number, not a gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    _STEER_BUDGET_CEILING,
    _STEER_FUSED_LOGPROB_BAND,
    _judge_steer_op,
    _judge_steer_width,
)

# The measured magnitudes this file discriminates between. See the module docstring for the
# jobs they come from; each also appears beside the constant it justifies in t2_invariants.
STEER_EFFECT = 7.632304e-01        # the SMALLEST steering intervention ever measured here
FUSED_MEASURED = 2.342296e-02      # MIA_STEER_FUSED=1 vs the eager forward-hook steer
WIDTH_STEERED_MEASURED = 1.778936e-02   # STEERED   alone-vs-batch, worst of 7 jobs
WIDTH_CONTROL_MEASURED = 5.865807e-02   # UNSTEERED alone-vs-batch, worst of 7 jobs
ATOL = 1e-5


def _arm(root: Path, *, steered: float, control: float) -> Path:
    """One steer arm: a `generation` tree and its own unsteered `control` tree.

    Only the logprob channels matter -- they are the whole observable of a steer -- so the
    two offsets are what the gate under test actually reads.
    """
    for i in range(3):
        req = root / f"req{i}"
        req.mkdir(parents=True, exist_ok=True)
        for name, shift in (("generation", steered), ("control", control)):
            save_file({"prompt_token_ids": torch.tensor([5, 6, 7], dtype=torch.int64),
                       "token_ids": torch.tensor([11, 12, 13], dtype=torch.int64),
                       "token_logprobs": torch.tensor([-1.0, -2.0, -3.0],
                                                      dtype=torch.float64) + shift,
                       "cumulative_logprob": torch.tensor(-6.0 + shift, dtype=torch.float64)},
                      str(req / f"{name}.safetensors"))
    return root


def _width_verdict(tmp_path: Path, *, steered_delta: float, control_delta: float) -> bool:
    alone = _arm(tmp_path / "alone", steered=0.0, control=0.0)
    batch = _arm(tmp_path / "batch", steered=steered_delta, control=control_delta)
    verdicts: list = []
    _judge_steer_width({"full_batch": batch}, alone, ATOL, verdicts)
    return verdicts[0][1]


def _op_verdict(tmp_path: Path, *, fused_delta: float, nofuse_delta: float = 0.0) -> dict:
    eager = _arm(tmp_path / "eager", steered=0.0, control=0.0)
    legs = {"aperture_alone": _arm(tmp_path / "fused", steered=fused_delta, control=0.0),
            "aperture_alone_nofuse": _arm(tmp_path / "nofuse", steered=nofuse_delta,
                                          control=0.0)}
    verdicts: list = []
    _judge_steer_op(legs, eager, ATOL, verdicts)
    return {label: ok for label, ok, _fatal in verdicts}


# ---------------------------------------------------------------------------
# item 2: the width gate, which used to print a number and gate nothing
# ---------------------------------------------------------------------------

def test_width_gate_accepts_the_measured_steered_delta(tmp_path):
    """1.778936e-02 steered against a 5.865807e-02 unsteered control: the worst pair the 7
    GPU jobs produced. The invariant holds there -- widening the batch moves req0 by the
    same amount with or without a steer -- so the gate must pass it."""
    assert _width_verdict(tmp_path, steered_delta=WIDTH_STEERED_MEASURED,
                          control_delta=WIDTH_CONTROL_MEASURED)


def test_width_gate_rejects_an_intervention_scale_delta(tmp_path):
    """req0 receiving the WRONG steering (or none) at width 32 moves its logprobs by a whole
    intervention. 7.632304e-01 against the same measured control must FAIL."""
    assert not _width_verdict(tmp_path, steered_delta=STEER_EFFECT,
                              control_delta=WIDTH_CONTROL_MEASURED)


def test_width_gate_does_not_widen_to_match_a_blown_control(tmp_path):
    """THE SELF-WIDENING DEFECT. If the unsteered control itself blew up to intervention
    scale, `max(control, atol)` would hand the steered delta a budget big enough to hide a
    complete mis-steer. The ceiling caps the budget AND the blown control is itself a
    failure."""
    assert not _width_verdict(tmp_path, steered_delta=STEER_EFFECT,
                              control_delta=STEER_EFFECT)


def test_width_gate_fails_on_a_blown_control_even_when_the_steered_delta_is_small(tmp_path):
    """A control past the ceiling is a finding in its own right: the batch-variance floor
    this gate rests on is no longer the thing that was measured."""
    assert not _width_verdict(tmp_path, steered_delta=1e-6,
                              control_delta=_STEER_BUDGET_CEILING * 2)


def test_width_gate_fails_when_the_batched_arm_did_not_run(tmp_path):
    verdicts: list = []
    _judge_steer_width({}, _arm(tmp_path / "alone", steered=0.0, control=0.0), ATOL, verdicts)
    assert verdicts == [("steer_small width [BANDED]", False, True)]


# ---------------------------------------------------------------------------
# item 4: the fused-steer band, which passed a broken kernel at BAND=10
# ---------------------------------------------------------------------------

def test_fused_band_accepts_the_measured_lever_delta(tmp_path):
    """`MIA_STEER_FUSED=1` sits 2.342296e-02 from the eager forward-hook steer -- a
    reduction-order difference in a shipped lever, reproduced in every job. The band exists
    to tolerate exactly that."""
    assert _op_verdict(tmp_path, fused_delta=FUSED_MEASURED)["steer_small op-identity [BANDED]"]


def test_fused_band_rejects_an_intervention_scale_delta(tmp_path):
    """The reject side the band never had. A fused kernel that steers the wrong rows, or
    not at all, sits at the intervention scale (7.632304e-01); at the documented band
    (5e-2) that fails by 15x, and this test is what stops the band drifting up to meet
    it."""
    assert not _op_verdict(tmp_path, fused_delta=STEER_EFFECT)[
        "steer_small op-identity [BANDED]"]


def test_fused_band_sits_between_the_measurement_and_the_corruption():
    """The documented margins, asserted rather than claimed in a comment: 2.1x above the
    measured lever delta, 15x below a real intervention."""
    assert FUSED_MEASURED < _STEER_FUSED_LOGPROB_BAND < STEER_EFFECT
    assert _STEER_FUSED_LOGPROB_BAND / FUSED_MEASURED > 2.0
    assert STEER_EFFECT / _STEER_FUSED_LOGPROB_BAND > 10.0


def test_nofuse_gate_stays_bit_exact_whatever_the_fused_band_is(tmp_path):
    """`op-identity-nofuse` is the ONLY bit-exact gate on the steer path and is NOT banded:
    it must fail on a delta the fused band would happily accept."""
    verdicts = _op_verdict(tmp_path, fused_delta=0.0, nofuse_delta=FUSED_MEASURED)
    assert not verdicts["steer_small op-identity-nofuse"]
    assert verdicts["steer_small op-identity [BANDED]"]


# ---------------------------------------------------------------------------
# item 5: `steer-gap` must not widen to match a floor that blew up
# ---------------------------------------------------------------------------
# Its budget was `max(d_floor, atol)`, computed at runtime with NO ceiling, where `d_floor`
# is the UNSTEERED eager-vs-FULL delta. If that floor blew up -- itself a finding -- the
# gate widened to match and passed whatever it was given. Measured, the floor is
# 3.635375e-02 in all 7 GPU jobs and the steered delta 1.801313e-02 in all 7.

GAP_FLOOR_MEASURED = 3.635375e-02       # UNSTEERED eager-vs-FULL, all 7 jobs
GAP_STEERED_MEASURED = 1.801313e-02     # STEERED   eager-vs-FULL, all 7 jobs


def _gap_verdicts(tmp_path: Path, *, floor: float, steered: float) -> dict:
    """Run the real `judge_steer` over two arms whose floor and steered deltas are set.

    The eager arm sits at zero; the FULL arm's `control` tree carries the floor and its
    `generation` tree the steered delta, which is exactly what the two
    `steer_logprob_delta` calls in `judge_steer` measure. Liveness needs each arm to differ
    from its OWN control by more than 1e-3, so the eager arm is given a 1.0 separation and
    the FULL arm keeps one too.
    """
    from tests.mia.parity.t2_invariants import judge_steer

    eager = _arm(tmp_path / "eager", steered=0.0, control=1.0)
    full = _arm(tmp_path / "full", steered=steered, control=1.0 + floor)
    legs = {"eager_alone": eager, "full_alone": full,
            "aperture_alone": _arm(tmp_path / "ap", steered=0.0, control=1.0),
            "aperture_alone_nofuse": _arm(tmp_path / "nf", steered=0.0, control=1.0),
            "full_batch": _arm(tmp_path / "fb", steered=0.0, control=1.0),
            "full_alone_hostroute": _arm(tmp_path / "hr", steered=0.0, control=1.0)}
    verdicts: list = []
    judge_steer(legs, verdicts)
    return {label: ok for label, ok, _fatal in verdicts}


def test_steer_gap_accepts_the_measured_floor_and_delta(tmp_path):
    """The measured pair, in all 7 GPU jobs: a steered delta of 1.801313e-02 under an
    unsteered compiled-vs-eager floor of 3.635375e-02. The gap is benign there."""
    assert _gap_verdicts(tmp_path, floor=GAP_FLOOR_MEASURED,
                         steered=GAP_STEERED_MEASURED)["steer_small steer-gap"]


def test_steer_gap_rejects_an_intervention_scale_delta(tmp_path):
    """A stale steer config surviving into a real step moves the steered logprobs by a whole
    intervention (7.632304e-01). Under the measured floor, that must FAIL."""
    assert not _gap_verdicts(tmp_path, floor=GAP_FLOOR_MEASURED,
                             steered=STEER_EFFECT)["steer_small steer-gap"]


def test_steer_gap_does_not_widen_to_match_a_blown_floor(tmp_path):
    """THE SELF-WIDENING DEFECT, reproduced. With no ceiling, a floor at intervention scale
    makes `max(floor, atol)` big enough to pass an intervention-scale steered delta -- the
    gate accepting the very thing it exists to catch."""
    assert not _gap_verdicts(tmp_path, floor=STEER_EFFECT,
                             steered=STEER_EFFECT)["steer_small steer-gap"]


def test_steer_gap_fails_on_a_blown_floor_even_when_the_steered_delta_is_tiny(tmp_path):
    """A floor past the ceiling is a finding in its own right: the compiled-vs-eager
    numerics this gate rests on are no longer what was measured, so the gate's premise is
    gone whatever the steered delta happens to be."""
    assert not _gap_verdicts(tmp_path, floor=_STEER_BUDGET_CEILING * 2,
                             steered=1e-9)["steer_small steer-gap"]


def test_steer_budget_ceiling_sits_between_the_measured_floors_and_the_intervention():
    """The documented margins, asserted: 2.6x above the worst floor ever measured (the
    width control, 5.865807e-02), 4.1x above the steer-gap floor, 5.1x below the smallest
    steering intervention."""
    assert WIDTH_CONTROL_MEASURED < _STEER_BUDGET_CEILING < STEER_EFFECT
    assert GAP_FLOOR_MEASURED < _STEER_BUDGET_CEILING
    assert _STEER_BUDGET_CEILING / WIDTH_CONTROL_MEASURED > 2.0
    assert STEER_EFFECT / _STEER_BUDGET_CEILING > 5.0


# ---------------------------------------------------------------------------
# item 7: a steer gate above width 1 -- did each request get ITS OWN steering?
# ---------------------------------------------------------------------------
# Every steer gate ran at max_num_seqs=1, so a steer mis-route at width > 1 was unmeasured.
# Widening the batch alone does not measure it either: with every request carrying the SAME
# config, "each request got its own steering" and "each got its neighbour's" predict
# identical logprobs. Arming ALTERNATE requests separates them, and these tests pin both
# failure signatures against the measured intervention scale.

def _arming_tree(root: Path, *, armed_delta: float, unarmed_delta: float,
                 n: int = 9, leak_at: int | None = None,
                 leak_delta: float = STEER_EFFECT,
                 tok_flip_at: int | None = None,
                 dead_at: int | None = None) -> Path:
    """A mixed-arm leg's tree, with optional single-request corruptions.

    `leak_at` makes one UNARMED request's LOGPROBS move (task D12: no longer the gate's
    fatal signature, since LSF 1733997 proved that channel is not reproducible pass-to-pass
    -- see the tests below) by `leak_delta`. `tok_flip_at` makes one UNARMED request's
    generated TOKEN IDS differ between the two passes -- the deterministic leak signature
    the gate now actually gates on. `dead_at` makes one ARMED request not move (its steering
    was routed away). One request is enough -- that is the point of a per-request gate.
    """
    from tests.mia.parity.capture_workload import steer_arm_mask

    for i, armed in enumerate(steer_arm_mask(n)):
        delta = armed_delta if armed else unarmed_delta
        if leak_at == i and not armed:
            delta = leak_delta
        if dead_at == i and armed:
            delta = 0.0
        req = root / f"req{i}"
        req.mkdir(parents=True, exist_ok=True)
        for name, shift in (("control", 0.0), ("generation", delta)):
            flip = tok_flip_at == i and not armed and name == "generation"
            tok = [11, 12, 999] if flip else [11, 12, 13]
            save_file({"prompt_token_ids": torch.tensor([5, 6, 7], dtype=torch.int64),
                       "token_ids": torch.tensor(tok, dtype=torch.int64),
                       "token_logprobs": torch.tensor([-1.0, -2.0, -3.0],
                                                      dtype=torch.float64) + shift,
                       "cumulative_logprob": torch.tensor(-6.0 + shift, dtype=torch.float64)},
                      str(req / f"{name}.safetensors"))
        save_file({"armed": torch.tensor(int(armed), dtype=torch.int64)},
                  str(req / "arming.safetensors"))
    return root


def _arming_verdicts(tree: Path | None) -> dict:
    from tests.mia.parity.t2_invariants import _judge_steer_per_request_arming

    verdicts: list = []
    _judge_steer_per_request_arming(
        {} if tree is None else {"full_batch_mixed_arm": tree}, ATOL, verdicts)
    return {label: ok for label, ok, _fatal in verdicts}


def test_arming_gate_passes_when_each_request_got_its_own_steering(tmp_path):
    """The correct outcome: every armed request moves by a whole intervention, every unarmed
    one does not move at all. Unarmed rows run the same kernels on the same shapes in both
    passes, so their honest delta is 0.0 rather than merely small."""
    tree = _arming_tree(tmp_path / "ok", armed_delta=STEER_EFFECT, unarmed_delta=0.0)
    assert _arming_verdicts(tree)["steer_small per-request-arming"]


def test_arming_gate_catches_one_unarmed_request_receiving_a_neighbours_steer(tmp_path):
    """THE MIS-ROUTE this gate exists for, at width > 1, on the channel task D12 moved it to.
    LSF 1733997 (`run_steer_ab_null_control.py`) proved the unarmed LOGPROB channel is not
    reproducible pass-to-pass even within one boot -- the SAME comparison sampled twice in
    one engine boot disagreed (3.265401e-02 vs exactly 0.0) -- but unarmed TOKEN IDS never
    flipped across every pair in that job (0/0/0/.../0). A single unarmed request whose
    generated token ids differ between Arm A and Arm B is therefore the mis-route signature
    now, independent of any logprob movement. One request is enough."""
    tree = _arming_tree(tmp_path / "leak", armed_delta=STEER_EFFECT, unarmed_delta=0.0,
                        tok_flip_at=2)
    assert not _arming_verdicts(tree)["steer_small per-request-arming"]


def test_arming_gate_no_longer_fails_on_unarmed_logprob_movement_alone(tmp_path):
    """Task D12, supersedes the pre-D12 test of the same shape. The bit-exact-logprob gate
    this replaces (task D7) treated ANY unarmed logprob movement as a leak, on the premise
    that the honest delta is EXACTLY 0.0. LSF 1733997 falsified that premise IN ONE BOOT: the
    SAME A-vs-B comparison, sampled twice with nothing else different, measured 3.265401e-02
    on one draw and 0.000000e+00 on the other -- so a gate that still failed on this channel
    would fail roughly half its own honest passes for no defect at all. With unarmed TOKEN
    IDS held bit-exact (the deterministic channel), the worst pass-to-pass unarmed logprob
    movement measured so far (`STEER_UNARMED_LOGPROB_ENVELOPE`, 3.265401e-02) must NOT fail
    the FATAL gate -- it is recorded as INFO instead."""
    from tests.mia.parity.capture_workload import STEER_UNARMED_LOGPROB_ENVELOPE

    tree = _arming_tree(tmp_path / "logprob-only", armed_delta=STEER_EFFECT,
                        unarmed_delta=0.0, leak_at=2,
                        leak_delta=STEER_UNARMED_LOGPROB_ENVELOPE)
    assert _arming_verdicts(tree)["steer_small per-request-arming"], (
        "an unarmed logprob movement with token ids intact must not fail the fatal gate "
        "post-D12 -- that channel is known to be irreproducible pass-to-pass")


def test_arming_gate_catches_one_armed_request_losing_its_steer(tmp_path):
    """The other direction: an armed request whose steering was routed away does not move at
    all -- a whole-batch liveness check (the only kind the suite had) would still pass,
    because the other armed requests moved. Task D12 raised the floor this trips on from
    STEER_LIVENESS_ATOL (1e-3) to STEER_ARM_POSITIVE_CONTROL_ATOL; task D13 re-anchored that
    floor's upper bound after D12 picked a whole-batch worst-case (1.216572e+01) instead of
    the weakest PER-REQUEST armed effect, which left the floor ABOVE every real signal and
    unable to pass on genuine steering at all. The floor now sits strictly between the
    measured PATH-noise ceiling (5.830579e-02 -- the same real vector at a different
    `vector_path`, i.e. no steering difference at all) and the weakest measured per-request
    armed effect (2.757719e-01 / 2.757723e-01, two jobs) -- see
    test_arming_gate_atol_sits_between_noise_and_signal for the hermetic version of this
    check. Here we only need it to still catch a dead armed request."""
    from tests.mia.parity.capture_workload import STEER_ARM_POSITIVE_CONTROL_ATOL

    tree = _arming_tree(tmp_path / "dead", armed_delta=STEER_EFFECT, unarmed_delta=0.0,
                        dead_at=3)
    assert not _arming_verdicts(tree)["steer_small per-request-arming"]

    # A dead armed request (effect 0.0) must never clear a floor that sits strictly above
    # the noise ceiling -- sanity check on the fixture, not a re-derivation (see
    # test_arming_gate_atol_sits_between_noise_and_signal for that).
    assert STEER_ARM_POSITIVE_CONTROL_ATOL > 5.830579e-02


def test_arming_gate_atol_sits_between_noise_and_signal(tmp_path):
    """Task D13. THE BUG (D12): the positive-control floor was anchored on 1.216572e+01, LSF
    1733997's worst effect across a whole real-vs-zero BATCH comparison -- not a bound on any
    single request. That put the floor (10x the noise ceiling = 5.830579e-01) ABOVE the
    weakest genuine PER-REQUEST armed effect (2.757719e-01), so no real steering could ever
    clear it: `MISMATCH req29: ARMED but did not clear the positive-control floor
    (2.757719e-01 <= 5.830579e-01)` (LSF 1737784). A per-request floor must be re-derived
    against per-request anchors on BOTH sides, and pinned hermetically so a future edit
    cannot repeat this mistake on a GPU an hour later instead of in this test.

    LOWER anchor -- path noise the floor must exceed: LSF 1733997's PATH pass (the SAME real
    vector bytes, only `vector_path` renamed, i.e. no steering difference at all) moved
    ARMED rows by 5.830579e-02; the same job's worst UNARMED pass-to-pass logprob envelope
    (STEER_UNARMED_LOGPROB_ENVELOPE) was smaller, 3.265401e-02 -- so 5.830579e-02 is the
    binding noise ceiling.

    UPPER anchor -- the weakest genuine per-request armed effect the floor must stay under:
    2.757719e-01, measured on THIS gate's own full mixed-arm batch (LSF 1737784), agreeing to
    6 significant figures with 2.757723e-01 from the original decisive A/B experiment (LSF
    1725670, a different job on a different node) -- the signal is stable across boots and
    nodes, so the floor has real room to sit below it.
    """
    from tests.mia.parity.capture_workload import (
        STEER_ARM_POSITIVE_CONTROL_ATOL,
        STEER_UNARMED_LOGPROB_ENVELOPE,
    )

    noise_ceiling = 5.830579e-02
    weakest_armed_this_job = 2.757719e-01     # LSF 1737784
    weakest_armed_other_job = 2.757723e-01    # LSF 1725670

    # The noise ceiling really is the larger of the two recorded noise measurements, and the
    # floor must clear it.
    assert noise_ceiling > STEER_UNARMED_LOGPROB_ENVELOPE
    assert STEER_ARM_POSITIVE_CONTROL_ATOL > noise_ceiling

    # The floor must sit strictly below EVERY measured weakest-armed-effect anchor, not just
    # the smaller of the two -- this is what D12's whole-batch anchor got backwards.
    assert STEER_ARM_POSITIVE_CONTROL_ATOL < weakest_armed_this_job
    assert STEER_ARM_POSITIVE_CONTROL_ATOL < weakest_armed_other_job

    # Sensible margin on both sides (task D13: ~1.7-2.2x above noise, ~2.1-2.8x below
    # signal), not merely "somewhere in the gap".
    assert STEER_ARM_POSITIVE_CONTROL_ATOL / noise_ceiling >= 1.7
    assert weakest_armed_this_job / STEER_ARM_POSITIVE_CONTROL_ATOL >= 2.1
    assert weakest_armed_other_job / STEER_ARM_POSITIVE_CONTROL_ATOL >= 2.1


def test_arming_gate_fails_a_batch_that_is_not_actually_mixed(tmp_path):
    """33 armed requests prove nothing about routing and 33 unarmed ones prove nothing about
    liveness, so a batch with only one class is vacuous and says so."""
    from tests.mia.parity.capture_workload import steer_arm_mask

    root = tmp_path / "allarmed"
    for i in range(len(steer_arm_mask(9))):
        req = root / f"req{i}"
        req.mkdir(parents=True, exist_ok=True)
        for name, shift in (("control", 0.0), ("generation", STEER_EFFECT)):
            save_file({"token_logprobs": torch.tensor([-1.0, -2.0], dtype=torch.float64)
                       + shift,
                       "cumulative_logprob": torch.tensor(-3.0 + shift, dtype=torch.float64)},
                      str(req / f"{name}.safetensors"))
        save_file({"armed": torch.tensor(1, dtype=torch.int64)},
                  str(req / "arming.safetensors"))
    assert not _arming_verdicts(root)["steer_small per-request-arming"]


def test_arming_gate_fails_when_the_leg_did_not_run(tmp_path):
    verdicts = _arming_verdicts(None)
    assert not verdicts["steer_small per-request-arming"]
    assert not verdicts["steer_small INFO per-request-arming bit-exact"]


def test_arming_mask_alternates_and_covers_both_classes():
    """Alternating puts an unarmed request on both sides of almost every armed one, so an
    off-by-one in the column mapping lands on an unarmed row in either direction."""
    from tests.mia.parity.capture_workload import steer_arm_mask

    mask = steer_arm_mask(33)
    assert len(mask) == 33 and 0 < sum(mask) < 33
    assert all(mask[i] != mask[i + 1] for i in range(len(mask) - 1))
