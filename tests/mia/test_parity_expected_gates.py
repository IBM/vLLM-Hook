"""A gate that FAILED TO RUN must fail the suite, not disappear from it.

Task D5 item 1. The structural review of `tests/mia/parity/` demonstrated, on the real LSF
1723978 artifacts, that `judge_steer` gated its optional legs on `if <leg> is not None:` --
so dropping 4 of the 6 steer legs still printed

    VERDICT T2 steer_small: PASS (0 failing invariants: [])

and exited 0. The first gate lost was `op-identity-nofuse`, the ONLY bit-exact gate on the
steer path. Nothing counted the verdicts against anything: the list of failures was empty
because the list of gates was short.

These tests pin the fix from both ends:

* `EXPECTED_GATES` declares, per workload, the gate names that MUST be present, and
  `adjudicate()` fails the run when one is absent (or when a gate nobody declared appears).
* The judges themselves emit an explicit `FAIL (an arm did not run)` instead of skipping,
  which is the discipline `judge_capture` already had.

The set-equality test is also what keeps `EXPECTED_GATES` honest without a GPU: a typo in
that table, or a gate added to a judge and not declared, fails here at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    EXPECTED_GATES,
    LEGS,
    adjudicate,
    gate_set_problems,
    judge_capture,
    judge_steer,
)

_N_REQS = 3


def _generation(dir_: Path, name: str, *, shift: float = 0.0) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    save_file({"prompt_token_ids": torch.tensor([5, 6, 7], dtype=torch.int64),
               "token_ids": torch.tensor([11, 12, 13, 14], dtype=torch.int64),
               "token_logprobs": torch.tensor([-1.0, -2.0, -3.0, -4.0],
                                              dtype=torch.float64) + shift,
               "cumulative_logprob": torch.tensor(-10.0 + shift, dtype=torch.float64)},
              str(dir_ / f"{name}.safetensors"))


def _capture_tree(root: Path) -> Path:
    """One leg's artifact tree: identical values everywhere, so every gate PASSES.

    The point of these tests is the SHAPE of the verdict list, not the numbers, so the
    trees agree exactly and the judge runs its clean path end to end.
    """
    g = torch.Generator().manual_seed(0)
    for i in range(_N_REQS):
        req = root / f"req{i}"
        _generation(req, "generation")
        for layer in (1, 16, 32):
            save_file({"hidden_states": torch.randn(7, 8, generator=g).to(torch.float16),
                       "layer_num": torch.tensor(layer, dtype=torch.int64)},
                      str(req / f"layer{layer:03d}.safetensors"))
    return root


def _steer_tree(root: Path) -> Path:
    """A steer leg: `generation` plus its own unsteered `control`, one clear step apart.

    The 1.0 separation is far above `STEER_LIVENESS_ATOL` (1e-3), so the liveness gates
    pass; every arm carries the SAME steered values, so the cross-arm deltas are 0.
    """
    for i in range(_N_REQS):
        req = root / f"req{i}"
        _generation(req, "generation")
        _generation(req, "control", shift=1.0)
    return root


def _mixed_arm_tree(root: Path, n: int = 33) -> Path:
    """The width-33 mixed-arm leg: alternate requests armed, each with its own control.

    Armed requests sit a clear 1.0 from their control (far above the 1e-3 liveness floor);
    unarmed ones sit exactly on theirs, which is what a correctly routed per-request steer
    produces. `arming.safetensors` is the marker the driver writes and the judge reads.
    """
    from tests.mia.parity.capture_workload import steer_arm_mask

    for i, armed in enumerate(steer_arm_mask(n)):
        req = root / f"req{i}"
        _generation(req, "control")
        _generation(req, "generation", shift=1.0 if armed else 0.0)
        save_file({"armed": torch.tensor(int(armed), dtype=torch.int64)},
                  str(req / "arming.safetensors"))
    return root


def _all_legs(workload: str, tmp_path: Path) -> dict:
    build = _steer_tree if workload == "steer_small" else _capture_tree
    legs = {name: build(tmp_path / workload / name) for name in LEGS[workload]}
    if "full_batch_mixed_arm" in legs:
        legs["full_batch_mixed_arm"] = _mixed_arm_tree(
            tmp_path / workload / "full_batch_mixed_arm_mixed")
    return legs


def _judge(workload: str, legs: dict) -> list:
    verdicts: list = []
    if workload == "steer_small":
        judge_steer(legs, verdicts)
    else:
        judge_capture(workload, legs, verdicts)
    return verdicts


@pytest.mark.parametrize("workload", sorted(EXPECTED_GATES))
def test_expected_gate_set_is_exactly_what_the_judge_emits(workload, tmp_path):
    """SET EQUALITY, both directions, with every leg present.

    Missing means a declared gate never ran; undeclared means a gate exists that the table
    does not know about and could therefore vanish later unnoticed. Either one is a typo or
    a drift, and either one fails here rather than on a GPU three hours in.
    """
    verdicts = _judge(workload, _all_legs(workload, tmp_path))
    missing, undeclared = gate_set_problems(workload, verdicts)
    assert not missing, f"{workload}: declared gates that emitted no verdict: {missing}"
    assert not undeclared, (
        f"{workload}: gates emitted that EXPECTED_GATES does not declare: {undeclared}")


@pytest.mark.parametrize("workload", sorted(EXPECTED_GATES))
def test_a_full_run_of_identical_trees_passes(workload, tmp_path):
    """The control for the tests below: with every leg present and every tree identical,
    `adjudicate` returns 0. A test that only ever sees failure cannot tell a broken gate
    from a broken fixture."""
    verdicts = _judge(workload, _all_legs(workload, tmp_path))
    assert adjudicate(workload, verdicts) == 0, [v for v in verdicts if v[2] and not v[1]]


def test_dropping_four_steer_legs_fails_the_run(tmp_path, capsys):
    """THE REPRODUCTION. Exactly the four legs the review dropped, and the run must FAIL.

    Before the fix this printed `PASS (0 failing invariants: [])` and exited 0.
    """
    legs = _all_legs("steer_small", tmp_path)
    for dropped in ("aperture_alone_nofuse", "aperture_alone", "full_batch",
                    "full_alone_hostroute"):
        legs.pop(dropped)

    verdicts = _judge("steer_small", legs)
    rc = adjudicate("steer_small", verdicts)
    assert rc != 0, "dropping 4 of the 6 steer legs still exited 0"

    failed = {label for label, ok, fatal in verdicts if fatal and not ok}
    assert "steer_small op-identity-nofuse" in failed, (
        "the ONLY bit-exact steer gate was lost rather than failed; failures were "
        f"{sorted(failed)}")
    assert "steer_small op-identity [BANDED]" in failed


def test_a_gate_that_is_never_emitted_fails_the_run(capsys):
    """The backstop itself, with the judges out of the picture: a verdict list that is
    simply SHORT -- which is what a silently deleted gate produces -- must not pass."""
    verdicts = [(label, True, True) for label in EXPECTED_GATES["steer_small"][:-1]]
    assert adjudicate("steer_small", verdicts) != 0
    assert "MISSING GATE" in capsys.readouterr().out


def test_an_undeclared_gate_fails_the_run(tmp_path):
    """A gate added to a judge but not declared is refused, so the table cannot fall behind
    the code and quietly stop protecting a new gate."""
    verdicts = [(label, True, True) for label in EXPECTED_GATES["steer_small"]]
    verdicts.append(("steer_small some-new-gate", True, True))
    assert adjudicate("steer_small", verdicts) != 0


def test_the_eager_reference_arm_missing_fails_every_steer_gate(tmp_path):
    """`judge_steer` used to return after ONE verdict when the eager arm was absent, which
    deleted every other steer gate at once. All of them must fail instead."""
    legs = _all_legs("steer_small", tmp_path)
    legs.pop("eager_alone")
    verdicts = _judge("steer_small", legs)
    missing, _ = gate_set_problems("steer_small", verdicts)
    assert not missing, f"gates vanished with the reference arm gone: {missing}"
    failed = {label for label, ok, fatal in verdicts if fatal and not ok}
    assert failed == set(EXPECTED_GATES["steer_small"])
