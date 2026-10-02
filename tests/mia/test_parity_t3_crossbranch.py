"""T3's gates, proved without a GPU — including the one proof a band is worth nothing without.

A band with no discrimination test is a defect in this suite, not a feature: at
`MIA_T2_STEER_FUSED_BAND=10` a totally broken fused steer once passed 88 tests. So the
questions here are the ones that make T3's PASS mean something:

  * does the DERIVED band reject a mis-route, and accept the control it was derived from?
  * does the ceiling still bite when the control itself goes wild?
  * does a gate whose arm never ran FAIL, rather than quietly vanishing from the tally?
  * does the declared artifact table catch `steer_small`'s shape (control + generation and
    NO layer tensors), an empty tree, and a tree with the wrong tensor keys?
  * is a cross-version trajectory bifurcation handled the way Phase D decided — excused only
    when a logprob delta explains it, and never for the rows before it?
  * does the MEASUREMENT the band is derived from agree with the GATE it is enforced by?

The synthetic payloads are float32 rather than the float16 a real run produces: the deltas
injected below (1e-3 control-sized, 4.5e-01 mis-route-sized) are exact in float32, while in
float16 a 1e-3 relative delta is about one ULP and the achieved delta would not be the
injected one. Nothing under test depends on the dtype -- the comparators call `.float()` --
and a test whose numbers are not the numbers it thinks it injected proves nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.capture_workload import WORKLOADS, _ascii  # noqa: E402
from tests.mia.parity.t2_invariants import _compare_replay_band_checked  # noqa: E402
from tests.mia.parity.t3_crossbranch import (  # noqa: E402
    _LOGPROB_CEILING,
    _MARGIN,
    _REL_CEILING,
    _ROW_CEILING,
    ARMS,
    EXPECTED_ARTIFACTS,
    EXPECTED_GATES,
    adjudicate,
    assert_layer_conventions,
    compare_generation_band_checked,
    derive_band,
    derive_logprob_band,
    inventory_problems,
    judge,
    main,
    measure,
)

WORKLOAD_NAMES = ("hs_small", "qk_small", "steer_small")

_STEPS = 4          # decode steps == generated tokens
_PROMPT = 5         # prompt tokens
_HIDDEN = 8

# The two signatures every band in this suite is measured against, from
# tests/test_parity_band_discrimination.py (LSF 1705469 / task D4).
CONTROL_SIZED = 1.0e-3          # plausible cross-version kernel drift
MIS_ROUTE_SIZED = 4.512e-01     # tensor-global floor of a one-row shift


# ---------------------------------------------------------------------------
# Synthetic artifact trees
# ---------------------------------------------------------------------------

def _payload(seed: int, steps: int, rows: int) -> torch.Tensor:
    """A [steps, rows, hidden] tensor whose max |value| is exactly 1.0 in EVERY step.

    That normalisation is what makes the injected deltas readable: with every row's scale
    pinned to 1.0, an added epsilon shows up as a tensor-global relative of exactly epsilon
    AND a per-row relative of exactly epsilon, so a test can assert against the number it
    injected instead of against whatever the data happened to make of it.
    """
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.rand((steps, rows, _HIDDEN), generator=generator) * 2 - 1
    scale = tensor.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (tensor / scale).float()


def build_tree(root: Path, workload: str) -> Path:
    """The canonical artifact tree for one workload — exactly what EXPECTED_ARTIFACTS declares."""
    wl = WORKLOADS[workload]
    root = Path(root)
    for index in range(len(wl.prompts)):
        req = root / f"req{index}"
        req.mkdir(parents=True, exist_ok=True)
        logprobs = torch.linspace(-1.0, -4.0, _STEPS).double() - index
        generation = {
            "prompt_token_ids": torch.arange(100 + index, 100 + index + _PROMPT,
                                             dtype=torch.int64),
            "token_ids": torch.arange(500 + index, 500 + index + _STEPS, dtype=torch.int64),
            "token_logprobs": logprobs,
            "cumulative_logprob": torch.tensor(float(logprobs.sum()), dtype=torch.float64),
        }
        save_file(generation, str(req / "generation.safetensors"))
        if wl.subsystem == "steer":
            control = dict(generation)
            control["token_logprobs"] = logprobs + 0.5      # a live steer moves the logprobs
            control["cumulative_logprob"] = torch.tensor(
                float((logprobs + 0.5).sum()), dtype=torch.float64)
            save_file(control, str(req / "control.safetensors"))
            continue
        for order, layer in enumerate(wl.layers):
            seed = 1000 * index + order
            if wl.subsystem == "hs":
                tensors = {"hidden_states": _payload(seed, _STEPS, _PROMPT),
                           "layer_num": torch.tensor(int(layer), dtype=torch.int64),
                           "hs_mode": _ascii("all_tokens")}
            else:
                tensors = {"q": _payload(seed, _STEPS, _PROMPT),
                           "k_all": _payload(seed + 7, _STEPS, _PROMPT + _STEPS - 1),
                           "layer_num": torch.tensor(int(layer), dtype=torch.int64),
                           "hookq_mode": _ascii("all_tokens")}
            save_file(tensors, str(req / f"layer{int(layer):03d}.safetensors"))
    return root


def mutate(src: Path, dst: Path, *, add: float = 0.0, scale_step: tuple | None = None,
           logprob_add: float = 0.0, logprob_file: str | None = None,
           token_at: tuple | None = None,
           drop: str | None = None, rekey: str | None = None) -> Path:
    """Copy a tree, applying exactly one kind of change. Returns `dst`."""
    dst = Path(dst)
    for path in sorted(Path(src).rglob("*.safetensors")):
        rel = path.relative_to(src)
        if drop is not None and rel.as_posix() == drop:
            continue
        tensors = dict(load_file(str(path)))
        if rel.name.startswith("layer"):
            for key, value in list(tensors.items()):
                if not value.is_floating_point():
                    continue
                if add:
                    value = value.clone()
                    value[0, 0, 0] += add
                    tensors[key] = value
                if scale_step is not None:
                    step, factor = scale_step
                    value = value.clone()
                    value[step] = value[step] * factor
                    tensors[key] = value
        if rel.name in ("generation.safetensors", "control.safetensors"):
            if logprob_add and logprob_file in (None, rel.name):
                tensors["token_logprobs"] = tensors["token_logprobs"] + logprob_add
                tensors["cumulative_logprob"] = tensors["cumulative_logprob"] + logprob_add
            if token_at is not None:
                req, index, value = token_at
                if rel.parent.name == req and rel.name == "generation.safetensors":
                    ids = tensors["token_ids"].clone()
                    ids[index] = value
                    tensors["token_ids"] = ids
        if rekey is not None and rel.as_posix() == rekey:
            key = sorted(tensors)[0]
            tensors[f"{key}_renamed"] = tensors.pop(key)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(target))
    return dst


def build_arms(tmp_path: Path, workload: str, *, control: float = 0.0,
               new_world_extra: float = 0.0,
               new_world_step: tuple | None = None) -> dict:
    """Six arm trees laid out the way a healthy job produces them.

    `a21 == c21` and (absent `new_world_extra` / `new_world_step`) `a29 == c29`, so G1 and
    G2 are bit-exact and the band is exactly `MARGIN x control` — the case the derivation
    was written for.
    """
    base = build_tree(tmp_path / "base", workload)
    c21 = mutate(base, tmp_path / "c21")
    a21 = mutate(base, tmp_path / "a21")
    a21r = mutate(base, tmp_path / "a21r")
    c29 = mutate(base, tmp_path / "c29", add=control)
    a29 = mutate(c29, tmp_path / "a29", add=new_world_extra, scale_step=new_world_step)
    ref = mutate(base, tmp_path / "ref")
    return {"ref": ref, "c21": c21, "a21": a21, "a21r": a21r, "c29": c29, "a29": a29}


def run_judge(workload: str, arms: dict) -> dict:
    verdicts: list = []
    judge(workload, arms, verdicts)
    return {label: (ok, fatal) for label, ok, fatal in verdicts}


# ---------------------------------------------------------------------------
# The declared tables
# ---------------------------------------------------------------------------

def test_layer_conventions_hold():
    """hs_small (1,16,32) and qk_small (0,15,31) must name the SAME physical blocks."""
    text = assert_layer_conventions()
    assert "blocks (0, 15, 31)" in text
    assert text.count("blocks (0, 15, 31)") == 2      # HS and QK agree
    assert "steer_small (15,) -> blocks (15,)" in text


def test_steer_declares_control_and_generation_and_no_layer_tensors():
    """The vacuity ruling E-3 exists for: a naive compare over two files passes trivially."""
    declared = EXPECTED_ARTIFACTS["steer_small"]
    assert set(declared) == {f"req{i}/{name}.safetensors"
                             for i in range(3) for name in ("generation", "control")}
    assert not any("layer" in path for path in declared)


@pytest.mark.parametrize("workload", ("hs_small", "qk_small"))
def test_capture_declares_three_requests_of_generation_plus_three_layers(workload):
    declared = EXPECTED_ARTIFACTS[workload]
    assert len(declared) == 3 * (1 + 3)
    layers = WORKLOADS[workload].layers
    for index in range(3):
        for layer in layers:
            assert f"req{index}/layer{int(layer):03d}.safetensors" in declared


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_inventory_accepts_the_canonical_tree(tmp_path, workload):
    assert inventory_problems(build_tree(tmp_path / "t", workload), workload) == []


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_inventory_reports_an_empty_tree_as_producing_nothing(tmp_path, workload):
    empty = tmp_path / "empty"
    empty.mkdir()
    problems = inventory_problems(empty, workload)
    assert len(problems) == 1 and "produced NOTHING" in problems[0]


def test_inventory_catches_a_missing_layer_file(tmp_path):
    base = build_tree(tmp_path / "base", "hs_small")
    trimmed = mutate(base, tmp_path / "trimmed", drop="req1/layer016.safetensors")
    problems = inventory_problems(trimmed, "hs_small")
    assert any("req1/layer016.safetensors" in p and "DECLARED" in p for p in problems)


def test_inventory_catches_a_renamed_tensor_key(tmp_path):
    base = build_tree(tmp_path / "base", "qk_small")
    bad = mutate(base, tmp_path / "bad", rekey="req0/layer015.safetensors")
    problems = inventory_problems(bad, "qk_small")
    assert any("tensor keys" in p for p in problems)


def test_inventory_treats_an_arm_that_never_ran_as_a_problem():
    problems = inventory_problems(None, "hs_small")
    assert len(problems) == 1 and "did not run" in problems[0]


# ---------------------------------------------------------------------------
# The gate set: declared vs emitted, in both directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_a_full_judge_run_emits_exactly_the_declared_gates(tmp_path, workload):
    emitted = run_judge(workload, build_arms(tmp_path, workload, control=CONTROL_SIZED))
    assert set(emitted) == set(EXPECTED_GATES[workload]), (
        f"declared-but-missing={sorted(set(EXPECTED_GATES[workload]) - set(emitted))}, "
        f"emitted-but-undeclared={sorted(set(emitted) - set(EXPECTED_GATES[workload]))}")


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_a_healthy_job_passes_every_fatal_gate(tmp_path, workload):
    arms = build_arms(tmp_path, workload, control=CONTROL_SIZED)
    emitted = run_judge(workload, arms)
    failed = [label for label, (ok, fatal) in emitted.items() if fatal and not ok]
    assert failed == []


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_a_missing_arm_fails_its_gates_instead_of_deleting_them(tmp_path, workload):
    """THE defect the EXPECTED_GATES machinery exists for: a shorter list, not a redder one."""
    arms = build_arms(tmp_path, workload, control=CONTROL_SIZED)
    arms["a29"] = None
    verdicts: list = []
    judge(workload, arms, verdicts)
    emitted = {label: ok for label, ok, _fatal in verdicts}
    assert set(emitted) == set(EXPECTED_GATES[workload])
    assert emitted[f"{workload} G2 MIA-inert-0.29 a29-vs-c29 (bit-exact)"] is False
    assert emitted[f"{workload} G5 T3 structure a21-vs-a29"] is False
    assert adjudicate(workload, verdicts) == 1


def test_adjudicate_fails_on_a_gate_that_was_never_declared():
    verdicts = [(label, True, True) for label in EXPECTED_GATES["hs_small"]]
    verdicts.append(("hs_small something new", True, True))
    assert adjudicate("hs_small", verdicts) == 1


def test_adjudicate_fails_on_a_declared_gate_that_never_reported():
    verdicts = [(label, True, True) for label in EXPECTED_GATES["hs_small"][1:]]
    assert adjudicate("hs_small", verdicts) == 1


# ---------------------------------------------------------------------------
# THE BAND — the derivation, and the discrimination it is worthless without
# ---------------------------------------------------------------------------

def test_band_is_the_triangle_sum_with_the_documented_margin():
    g1 = {"rel": 1e-4, "row": 2e-4}
    g4 = {"rel": 1e-3, "row": 3e-3}
    g2 = {"rel": 2e-4, "row": 1e-4}
    band = derive_band(g4, g1, g2, key="rel", ceiling=_REL_CEILING, what="unit")
    assert band == pytest.approx(_MARGIN * (1e-4 + 1e-3 + 2e-4))
    row = derive_band(g4, g1, g2, key="row", ceiling=_ROW_CEILING, what="unit")
    assert row == pytest.approx(_MARGIN * (2e-4 + 3e-3 + 1e-4))


def test_band_is_capped_at_the_ceiling_when_the_control_runs_away():
    wild = {"rel": 0.5, "row": 0.9}
    zero = {"rel": 0.0, "row": 0.0}
    assert derive_band(wild, zero, zero, key="rel", ceiling=_REL_CEILING,
                       what="unit") == _REL_CEILING
    assert derive_band(wild, zero, zero, key="row", ceiling=_ROW_CEILING,
                       what="unit") == _ROW_CEILING
    assert derive_logprob_band(0.5, 0.0, 0.0) == _LOGPROB_CEILING


def test_band_is_not_derivable_when_a_leg_did_not_run():
    zero = {"rel": 0.0, "row": 0.0}
    assert derive_band(None, zero, zero, key="rel", ceiling=_REL_CEILING, what="unit") is None
    assert derive_band(zero, None, zero, key="rel", ceiling=_REL_CEILING, what="unit") is None
    assert derive_logprob_band(None, 0.0, 0.0) is None


def _capture_gate(tmp_path: Path, delta: float, band: float, tag: str = "g") -> bool:
    """One banded capture comparison, exactly as `judge` runs it.

    `tag` keeps each invocation's trees apart, so a test may call this more than once
    without one call's mutation leaking into the next one's comparison.
    """
    root = tmp_path / tag
    base = build_tree(root / "base", "hs_small")
    a = mutate(base, root / "a")
    b = mutate(base, root / "b", add=delta)
    verdicts: list = []
    _compare_replay_band_checked("unit", a, b, name="layer*.safetensors",
                                 verdicts=verdicts, rel=band, row=band, tag="T3")
    return verdicts[0][1]


def test_band_accepts_a_control_sized_delta(tmp_path):
    """The case the derivation is written for: with G1 and G2 bit-exact, the band is
    `MARGIN x control` and the new world differs from the old by exactly the control."""
    band = _MARGIN * CONTROL_SIZED
    assert _capture_gate(tmp_path, CONTROL_SIZED, band)


def test_band_rejects_a_mis_route_sized_delta(tmp_path):
    """The reject side. A whole-request off-by-one row measures ~4.5e-01; a band derived
    from a 1e-3 control is 2e-3, and 4.5e-01 is 225x that."""
    band = _MARGIN * CONTROL_SIZED
    assert not _capture_gate(tmp_path, MIS_ROUTE_SIZED, band)


def test_band_still_rejects_a_mis_route_at_the_ceiling(tmp_path):
    """Even when a runaway control caps the band, the gate must keep discriminating —
    that is the whole reason the ceiling is a ceiling and not a licence."""
    assert not _capture_gate(tmp_path, MIS_ROUTE_SIZED, _REL_CEILING)


def test_the_gate_trips_end_to_end_when_the_new_world_carries_a_mis_route(tmp_path):
    """The discrimination proof through `judge`, not just through one comparator.

    `a29` gets a whole step scaled by 0.55 — the off-by-one-row signature, 4.5e-01 relative
    — on top of the control drift. Both the FATAL claims that should notice do: G2 (MIA is
    no longer inert on 0.29) and G5's banded capture gate.
    """
    arms = build_arms(tmp_path, "hs_small", control=CONTROL_SIZED,
                      new_world_step=(1, 1.0 - MIS_ROUTE_SIZED))
    emitted = run_judge("hs_small", arms)
    assert emitted["hs_small G2 MIA-inert-0.29 a29-vs-c29 (bit-exact)"][0] is False
    assert emitted["hs_small G5 T3 capture a21-vs-a29 [BANDED from G4]"][0] is False


def test_a_healthy_job_is_not_tripped_by_the_control_alone(tmp_path):
    """The other half of a discrimination test: the gate that rejects must also accept."""
    arms = build_arms(tmp_path, "hs_small", control=CONTROL_SIZED)
    emitted = run_judge("hs_small", arms)
    assert emitted["hs_small G5 T3 capture a21-vs-a29 [BANDED from G4]"][0] is True


def test_measure_agrees_with_the_gate_it_derives_from(tmp_path):
    """The band is only "derived from the control" if the two are the SAME measurement.

    `measure` runs the gate with infinite bands and reads its verdict line, so this asserts
    the loop closes: a band just above what `measure` reported passes, and one just below it
    fails. "Just" is 0.2%, because the comparator renders its worst value with `%.3e` — four
    significant figures, so a reading can understate the truth by at most 0.05%. That
    quantisation is documented beside `measure` and is four orders of magnitude under the
    `_MARGIN` the band multiplies it by; asserting at 1e-6 would be asserting that a printf
    format is exact, not that the measurement and the gate agree.
    """
    base = build_tree(tmp_path / "base", "qk_small")
    a = mutate(base, tmp_path / "a")
    b = mutate(base, tmp_path / "b", add=CONTROL_SIZED)
    result = measure("unit", a, b, name="layer*.safetensors", verdicts=None,
                     structure_fatal=False)
    assert result is not None
    assert result["rel"] == pytest.approx(CONTROL_SIZED, rel=1e-3)

    def gate(band: float) -> bool:
        verdicts: list = []
        _compare_replay_band_checked("unit", a, b, name="layer*.safetensors",
                                     verdicts=verdicts, rel=band, row=1e9, tag="T3")
        return verdicts[0][1]

    assert gate(result["rel"] * 1.002)
    assert not gate(result["rel"] * 0.998)


def test_measure_reports_structure_problems_and_fails_its_gate(tmp_path):
    """Infinite bands leave structure — and only structure — able to fail."""
    base = build_tree(tmp_path / "base", "hs_small")
    a = mutate(base, tmp_path / "a")
    b = mutate(base, tmp_path / "b", drop="req2/layer032.safetensors")
    verdicts: list = []
    measure("unit", a, b, name="layer*.safetensors", verdicts=verdicts, structure_fatal=True)
    assert verdicts[0][1] is False and verdicts[0][2] is True


def test_measure_on_a_missing_arm_fails_rather_than_returning_zero(tmp_path):
    verdicts: list = []
    assert measure("unit", None, tmp_path, name="*.safetensors", verdicts=verdicts,
                   structure_fatal=True) is None
    assert verdicts[0][1] is False


# ---------------------------------------------------------------------------
# Token ids first: the cross-version tie-flip
# ---------------------------------------------------------------------------

def test_an_unexplained_token_divergence_fails_the_generation_gate(tmp_path):
    """Greedy sampling from bit-identical logits cannot produce two different argmaxes, so a
    divergence with no logprob delta is a DIFFERENT bug wearing the tie-flip's clothes."""
    base = build_tree(tmp_path / "base", "hs_small")
    a = mutate(base, tmp_path / "a")
    b = mutate(base, tmp_path / "b", token_at=("req1", 2, 999))
    verdicts: list = []
    compare_generation_band_checked("unit", a, b, verdicts=verdicts, bound=1e-2,
                                    note="unit")
    assert verdicts[0][1] is False


def test_an_explained_tie_flip_is_judged_on_the_rows_before_it(tmp_path):
    """A bifurcation carrying a real logprob delta is a tie-flip: the rows after it hold a
    different token's activations and are not comparable under any band, and the rows before
    it stay a live, banded comparison."""
    base = build_tree(tmp_path / "base", "hs_small")
    a = mutate(base, tmp_path / "a")
    b = mutate(base, tmp_path / "b", token_at=("req1", 2, 999), logprob_add=1e-2)
    verdicts: list = []
    compare_generation_band_checked("unit", a, b, verdicts=verdicts, bound=5e-2, note="unit")
    assert verdicts[0][1] is True

    # ... and the same flip fails when the band cannot absorb the PRE-divergence rows.
    verdicts = []
    compare_generation_band_checked("unit", a, b, verdicts=verdicts, bound=1e-3, note="unit")
    assert verdicts[0][1] is False


def test_the_generation_gate_fails_when_no_band_could_be_derived(tmp_path):
    base = build_tree(tmp_path / "base", "hs_small")
    a = mutate(base, tmp_path / "a")
    verdicts: list = []
    compare_generation_band_checked("unit", a, a, verdicts=verdicts, bound=None, note="unit")
    assert verdicts[0][1] is False


# ---------------------------------------------------------------------------
# Steer: no layer tensors, so liveness is the only thing standing between the gate
# and a vacuous pass
# ---------------------------------------------------------------------------

def test_an_inert_steer_arm_fails_liveness(tmp_path):
    """Two arms whose steering silently became a no-op agree perfectly. Without liveness
    that reads as a clean T3 PASS over an intervention that never happened."""
    arms = build_arms(tmp_path, "steer_small")
    dead = tmp_path / "dead"
    for path in sorted(Path(arms["a29"]).rglob("*.safetensors")):
        target = dead / path.relative_to(arms["a29"])
        target.parent.mkdir(parents=True, exist_ok=True)
        tensors = dict(load_file(str(path)))
        if path.name == "control.safetensors":       # control == generation -> inert steer
            tensors = dict(load_file(str(path.parent / "generation.safetensors")))
        save_file(tensors, str(target))
    arms["a29"] = dead
    emitted = run_judge("steer_small", arms)
    assert emitted["steer_small liveness a29"][0] is False
    assert emitted["steer_small liveness a21"][0] is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_print_policy_records_what_is_enforced(capsys):
    assert main(["--print-policy"]) == 0
    text = capsys.readouterr().out
    assert "band policy" in text and "NO environment override" in text
    assert "layer conventions" in text
    for workload in WORKLOAD_NAMES:
        assert f"declared artifacts {workload}" in text


def test_cli_runs_a_workload_end_to_end(tmp_path):
    arms = build_arms(tmp_path, "hs_small", control=CONTROL_SIZED)
    argv = ["hs_small"]
    for name in ARMS:
        argv += [f"--{name}", str(arms[name])]
    assert main(argv) == 0


# ---------------------------------------------------------------------------
# The band the job ACTUALLY enforced was zero (task E1 review, finding 3)
# ---------------------------------------------------------------------------

def test_the_zero_band_is_a_real_gate_not_a_vacuous_one(tmp_path):
    """Job 1749270 measured `G4 = 0.000000e+00` on hs/qk, so the enforced band was `0.0`.

    Every discrimination test above pins a NONZERO band, which leaves the live one untested.
    Zero is not a special case in the comparator -- the check is `r > rel`, so a zero band
    degrades into bit-exactness on the float channel -- but "is not a special case" is a
    claim about code that can change, and this is the band a green T3 currently rests on.
    """
    assert _capture_gate(tmp_path, 0.0, 0.0, tag="zero_pass") is True
    assert _capture_gate(tmp_path, 1e-6, 0.0, tag="zero_fail") is False


def test_a_zero_band_derives_from_a_zero_control():
    """...and the derivation produces exactly that band from job 1749270's own numbers."""
    zero = {"rel": 0.0, "row": 0.0}
    assert derive_band(zero, zero, zero, key="rel", ceiling=_REL_CEILING, what="unit") == 0.0
    assert derive_logprob_band(0.0, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# A non-finite band would pass everything (task E1 review, finding 4)
# ---------------------------------------------------------------------------

def test_a_nan_band_would_pass_a_mis_route(tmp_path):
    """The defect the guard exists for, demonstrated rather than asserted in prose.

    `min(nan, ceiling)` is `nan` and the gate's comparison is `r > rel`, which is False for
    every `r` against NaN. So a single NaN band term turns the strictest gate in the tier
    into an unconditional PASS -- here, on the 4.512e-01 mis-route signature it is supposed
    to reject. Nothing may produce one.
    """
    assert _capture_gate(tmp_path, MIS_ROUTE_SIZED, float("nan"), tag="nan") is True


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_a_non_finite_band_term_fails_closed(bad):
    zero = {"rel": 0.0, "row": 0.0}
    poison = {"rel": bad, "row": bad}
    for control, g1, g2 in ((poison, zero, zero), (zero, poison, zero), (zero, zero, poison)):
        assert derive_band(control, g1, g2, key="rel", ceiling=_REL_CEILING,
                           what="unit") is None
        assert derive_band(control, g1, g2, key="row", ceiling=_ROW_CEILING,
                           what="unit") is None
    for control, g1, g2 in ((bad, 0.0, 0.0), (0.0, bad, 0.0), (0.0, 0.0, bad)):
        assert derive_logprob_band(control, g1, g2) is None


def test_a_non_derivable_band_fails_the_gate_it_feeds(tmp_path):
    """Fail CLOSED: `None` must reach the gate as a FAIL, never as "skip the check"."""
    arms = build_arms(tmp_path, "hs_small", control=CONTROL_SIZED)
    verdicts: list = []
    compare_generation_band_checked("unit", arms["a21"], arms["a29"], verdicts=verdicts,
                                    bound=None, note="unit")
    assert verdicts[0] == ("unit", False, True)


# ---------------------------------------------------------------------------
# The `fatal` FLAG, not just `ok` (task E1 review, finding 5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_exactly_the_info_gates_are_non_fatal(tmp_path, workload):
    """The Phase-D defect this closes: a gate described as FATAL in prose, emitted with
    `fatal=False`, silently disarming the only bit-exact steer gate while 88 tests stayed
    green. Every assertion elsewhere in this file filters on `fatal and not ok`, which
    passes vacuously if every gate became non-fatal -- so assert the flag itself.

    The invariant is exact and readable from the label: a gate is non-fatal **iff** its
    label says `[INFO]`.
    """
    emitted = run_judge(workload, build_arms(tmp_path, workload, control=CONTROL_SIZED))
    for label, (_ok, fatal) in sorted(emitted.items()):
        assert fatal is ("[INFO]" not in label), (
            f"{label}: emitted fatal={fatal}, but its label says "
            f"{'INFO' if '[INFO]' in label else 'FATAL'}")


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_the_fatal_gates_are_the_ones_the_design_names(tmp_path, workload):
    """And the FATAL set is the intended one, not merely self-consistent with the labels."""
    emitted = run_judge(workload, build_arms(tmp_path, workload, control=CONTROL_SIZED))
    fatal = {label for label, (_ok, is_fatal) in emitted.items() if is_fatal}
    for stem in ("inventory (all arms)", "G1 MIA-inert-0.21", "G2 MIA-inert-0.29",
                 "G4 CONTROL", "G5 T3 structure", "G5 T3 generation"):
        assert any(stem in label for label in fatal), f"{stem} is not gated FATAL"
    assert not any("[INFO]" in label for label in fatal)


def test_a_mis_route_makes_the_job_exit_nonzero(tmp_path):
    """`ok is False` on a verdict is not the same claim as "the job goes red"."""
    arms = build_arms(tmp_path, "hs_small", control=CONTROL_SIZED,
                      new_world_step=(1, 1.0 - MIS_ROUTE_SIZED))
    verdicts: list = []
    judge("hs_small", arms, verdicts)
    assert adjudicate("hs_small", verdicts) == 1


def test_a_healthy_job_exits_zero(tmp_path):
    """The other half: the red must be caused by the defect, not by the harness."""
    arms = build_arms(tmp_path, "hs_small", control=CONTROL_SIZED)
    verdicts: list = []
    judge("hs_small", arms, verdicts)
    assert adjudicate("hs_small", verdicts) == 0


# ---------------------------------------------------------------------------
# The 0.21 shadow depends on one absent line (task E1 review, finding 7)
# ---------------------------------------------------------------------------

def test_capture_workload_has_no_sys_path_insert():
    """T3's 0.21 arms import a PINNED `mia` worktree ahead of the editable install, and that
    works only because this one driver does not insert REPO_ROOT at `sys.path[0]` the way
    every other driver in `tests/mia/parity/` does. Adding it "for consistency" would point the
    old-world arm at HEAD's `mia`. The failure would be loud (HEAD refuses V1), but the arm
    is the only definition of "the old world" T3 has, so pin the property.
    """
    path = REPO_ROOT / "tests" / "mia" / "parity" / "capture_workload.py"
    code = [line for line in path.read_text().splitlines()
            if not line.lstrip().startswith("#")]
    offenders = [line for line in code if "sys.path.insert" in line]
    assert offenders == [], (
        f"{path} now inserts into sys.path: {offenders}. See the comment beside REPO_ROOT "
        f"in that file -- this breaks tests/mia/parity/run_crossbranch.sh's 0.21 arms.")


def test_mia_provenance_is_enforced_when_demanded(tmp_path, monkeypatch):
    """`MIA_PARITY_REQUIRE_MIA_UNDER` makes the arm re-check its own imported package.

    The job script's preflight probe runs in a different process with a different
    `sys.path[0]`, so it proves the resolution for the probe, not for the engine.
    """
    import types

    from tests.mia.parity.capture_workload import REQUIRE_MIA_UNDER, _assert_mia_provenance

    shadow = tmp_path / "shadow"
    (shadow / "mia").mkdir(parents=True)
    good = types.SimpleNamespace(__file__=str(shadow / "mia" / "__init__.py"))
    other = tmp_path / "live" / "mia"
    other.mkdir(parents=True)
    bad = types.SimpleNamespace(__file__=str(other / "__init__.py"))

    monkeypatch.delenv(REQUIRE_MIA_UNDER, raising=False)
    assert _assert_mia_provenance(bad)          # unset -> reports, enforces nothing

    monkeypatch.setenv(REQUIRE_MIA_UNDER, str(shadow))
    assert _assert_mia_provenance(good)
    with pytest.raises(RuntimeError, match="refusing to run"):
        _assert_mia_provenance(bad)


def test_mia_provenance_accepts_the_symlinked_shadow(tmp_path, monkeypatch):
    """A symlink is not a mismatch -- and getting this wrong breaks the arm it protects.

    The real arrangement reaches the pinned package through a directory holding a single
    symlink to a git worktree's `mia/`, so `__file__` and its `resolve()`d target are two
    different real paths and BOTH are correct. A first cut of this check resolved the
    candidate but not the root and rejected exactly that layout, which would have crashed
    both 0.21 arms on all three workloads. Pin the shape, not just the intent.
    """
    import types

    from tests.mia.parity.capture_workload import REQUIRE_MIA_UNDER, _assert_mia_provenance

    worktree = tmp_path / "worktree" / "mia"
    worktree.mkdir(parents=True)
    (worktree / "__init__.py").write_text("")
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "mia").symlink_to(worktree, target_is_directory=True)

    through_symlink = types.SimpleNamespace(__file__=str(shadow / "mia" / "__init__.py"))
    monkeypatch.setenv(REQUIRE_MIA_UNDER, str(shadow))
    assert _assert_mia_provenance(through_symlink)

    # ... and naming the symlink TARGET as the requirement works too, since that is the same
    # package by a different name.
    monkeypatch.setenv(REQUIRE_MIA_UNDER, str(tmp_path / "worktree"))
    assert _assert_mia_provenance(through_symlink)

    # ... while an unrelated package is still refused with either spelling.
    elsewhere = tmp_path / "live" / "mia"
    elsewhere.mkdir(parents=True)
    other = types.SimpleNamespace(__file__=str(elsewhere / "__init__.py"))
    with pytest.raises(RuntimeError, match="refusing to run"):
        _assert_mia_provenance(other)


# ---------------------------------------------------------------------------
# steer_small's UNSTEERED control channel (task E1 review, finding 9)
# ---------------------------------------------------------------------------

def _control_gate(tmp_path: Path, delta: float, band: float, *, tag: str,
                  channel: str = "control.safetensors") -> bool:
    """One banded comparison of `steer_small`'s `<channel>`, as `judge` runs it."""
    root = tmp_path / tag
    base = build_tree(root / "base", "steer_small")
    a = mutate(base, root / "a")
    b = mutate(base, root / "b", logprob_add=delta, logprob_file="control.safetensors")
    verdicts: list = []
    compare_generation_band_checked("unit", a, b, verdicts=verdicts, bound=band,
                                    note="unit", name=channel)
    return verdicts[0][1]


def test_the_control_channel_gate_sees_what_the_generation_gate_cannot(tmp_path):
    """THE hole, closed. `steer_small` writes no layer tensors, so `control.safetensors` was
    bit-exact within a version (G1/G2) and structurally checked across versions (G5
    structure, bands `inf`) but never VALUE-banded across them, because
    `_compare_generation_band` walked `generation.safetensors` only. The `*.safetensors`
    payload made it look covered. It does drift in practice: all six control files differ
    between 0.21 and 0.29 in job 1749270.

    The proof is the contrast, not the failure alone: the SAME drift that the new gate
    rejects is invisible to the generation gate even at a band of zero.
    """
    assert _control_gate(tmp_path, 1e-2, 1e-3, tag="ctl_reject") is False
    assert _control_gate(tmp_path, 1e-2, 0.0, tag="ctl_blind",
                         channel="generation.safetensors") is True


def test_the_control_channel_gate_accepts_a_band_sized_drift(tmp_path):
    """The accept side, so the gate is discriminating rather than merely strict."""
    assert _control_gate(tmp_path, 1e-2, 2e-2, tag="ctl_accept") is True


def test_a_control_only_drift_cannot_silently_widen_its_own_band(tmp_path):
    """The finding-1 invariant, on the channel finding 9 added.

    A MIA-side drift confined to the control channel feeds `g2_ctl` and so would widen the
    very band that judges it. That is survivable only because G1/G2 now gate EVERY channel
    bit-exact: the widening cannot happen without the FATAL inertness gate going red first,
    and the job exiting non-zero.
    """
    arms = build_arms(tmp_path, "steer_small")
    arms["a29"] = mutate(arms["a29"], tmp_path / "a29_drift", logprob_add=1e-2,
                         logprob_file="control.safetensors")
    verdicts: list = []
    judge("steer_small", arms, verdicts)
    emitted = {label: ok for label, ok, _fatal in verdicts}
    assert emitted["steer_small G2 MIA-inert-0.29 a29-vs-c29 (bit-exact)"] is False
    assert adjudicate("steer_small", verdicts) == 1


def test_the_steer_control_gate_accepts_an_undrifted_control(tmp_path):
    arms = build_arms(tmp_path, "steer_small")
    emitted = run_judge("steer_small", arms)
    assert emitted["steer_small G5 T3 control a21-vs-a29 [BANDED]"][0] is True


def test_the_steer_control_gate_is_declared(tmp_path):
    assert "steer_small G5 T3 control a21-vs-a29 [BANDED]" in EXPECTED_GATES["steer_small"]
    for workload in ("hs_small", "qk_small"):
        assert not any("G5 T3 control" in label for label in EXPECTED_GATES[workload])
