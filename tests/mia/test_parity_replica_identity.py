"""`replica_identity` must not report agreement it never observed.

The check compares requests carrying the SAME prompt at DIFFERENT batch rows within one run
-- the one alone-vs-batch-shaped comparison with no confound at all, since it is one engine,
one batch composition, one set of kernels. `prompts_for` repeats a 3-prompt set, so req0,
req3, req6 ... are replicas of each other.

The defect these tests pin: with fewer than 6 requests no prompt has two replicas, every
loop in the function is skipped, and it used to print

    PASS (0 replica tensor pairs, worst max|d|=0.000000e+00)

which is the exact string shape this branch removed everywhere else -- it reads as perfect
agreement and means "nothing was compared". It was reachable two ways: any 3-request tree
(this file builds one), and a real `full_batch` leg that ever stopped repeating its prompt
set, which would take the real run from 180 compared pairs to 0 with the verdict staying
green.
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

from tests.mia.parity.t2_invariants import replica_identity  # noqa: E402

_LAYERS = (1, 16, 32)


def _tree(root: Path, n_requests: int, *, corrupt: tuple | None = None) -> Path:
    """`n_requests` requests cycling a 3-prompt set: req0/req3 are replicas, req1/req4, ...

    `corrupt=(req, layer, delta)` perturbs one replica so a real disagreement is detectable.
    """
    for index in range(n_requests):
        request = root / f"req{index}"
        request.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(index % 3)   # replicas share a seed
        for layer in _LAYERS:
            values = torch.rand((4, 5, 8), generator=generator).float()
            if corrupt and corrupt[0] == index and corrupt[1] == layer:
                values = values.clone()
                values[0, 0, 0] += corrupt[2]
            save_file({"hidden_states": values,
                       "layer_num": torch.tensor(int(layer), dtype=torch.int64)},
                      str(request / f"layer{int(layer):03d}.safetensors"))
    return root


def _run(tree: Path | None, *, fatal: bool = True):
    verdicts: list = []
    replica_identity(tree, "unit", verdicts, fatal=fatal)
    return verdicts[0]


def test_a_tree_with_no_replica_pair_fails_instead_of_passing_vacuously(tmp_path, capsys):
    """THE defect. Three requests, three distinct prompts, zero replica pairs."""
    label, ok, fatal = _run(_tree(tmp_path / "three", 3))
    assert ok is False and fatal is True
    out = capsys.readouterr().out
    assert "VACUOUS" in out
    assert "0 replica tensor pairs" in out
    # and it must not be mistakable for the healthy line
    assert "PASS" not in out.split("VERDICT")[-1]


def test_identical_replicas_pass_and_say_how_much_was_compared(tmp_path, capsys):
    label, ok, _fatal = _run(_tree(tmp_path / "six", 6))
    assert ok is True
    out = capsys.readouterr().out
    assert "0 replica tensor pairs" not in out
    assert "replica group(s)" in out       # the count is visible, not just the verdict


def test_a_disagreeing_replica_fails(tmp_path):
    """The other half of a discrimination test: it must still catch a real difference."""
    tree = _tree(tmp_path / "six_bad", 6, corrupt=(3, 16, 1e-3))
    _label, ok, _fatal = _run(tree)
    assert ok is False


@pytest.mark.parametrize("n_requests", (0, 1, 2, 3, 5))
def test_every_too_small_tree_is_reported_as_vacuous(tmp_path, n_requests):
    """Fewer than 6 requests can never form a replica pair for all three prompts; below 4
    there is no pair at all. None of these may report agreement."""
    tree = _tree(tmp_path / f"n{n_requests}", n_requests)
    _label, ok, _fatal = _run(tree)
    if n_requests < 4:
        assert ok is False, f"{n_requests} requests produced no pair but did not fail"


def test_a_missing_arm_still_fails(tmp_path):
    """Unchanged behaviour, pinned so the vacuity fix did not disturb it."""
    _label, ok, _fatal = _run(None)
    assert ok is False


def test_the_non_fatal_form_still_reports_the_vacuity(tmp_path, capsys):
    """`fatal=False` makes the verdict INFO; it must not make an empty comparison look fine."""
    _label, ok, fatal = _run(_tree(tmp_path / "info", 3), fatal=False)
    assert ok is False and fatal is False
    assert "VACUOUS" in capsys.readouterr().out
