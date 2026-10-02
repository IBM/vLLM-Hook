"""qk_small's `routing-identity-w33-distinct` gate is now BANDED (task D9).

task-D8-report.md's separate finding: this leg failed bit-exact with 156/396 float tensors
differing, worst tensor-global relative 1.870e-03, concentrated on layer031 q/k_full across
all 33 requests -- generations IDENTICAL on every request, shapes and integer metadata
bit-exact. That is the SAME width-33 boot nondeterminism already banded for the
`routing-identity-w32` sibling (`_ROUTING_IDENTITY_QK_REL_BAND` /
`_ROUTING_IDENTITY_QK_ROW_BAND`), not a mis-route -- see that band's comment in
`tests/mia/parity/t2_invariants.py` for the full derivation. `judge_capture` now runs this leg
through `_compare_replay_band` with those SAME constants instead of a bit-exact `_compare`,
exactly as `routing-identity-w32` already does for QK. `hs_small`'s `routing-identity-w33-
distinct` is UNCHANGED (HS is bit-reproducible on this leg).

The band's own discrimination (accepts boot spread, rejects an off-by-one row / zeroed row /
shape change / integer-metadata drift) is pinned generically by
`tests/test_qk_routing_identity_band.py`. This file pins the ROUTING decision specifically:
which gate label `judge_capture` emits for which workload, and that the measured 1.870e-03
delta -- which would fail a bit-exact `_compare` -- now passes through `judge_capture`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    EXPECTED_GATES,
    LEGS,
    judge_capture,
)

_MEASURED_WORST_REL = 1.870e-03  # task-D8-report.md, layer031 q/k_full, all 33 requests


def _generation(root: Path, req: str) -> None:
    d = root / req
    d.mkdir(parents=True, exist_ok=True)
    save_file({"prompt_token_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
               "token_ids": torch.tensor([4, 5, 6], dtype=torch.int64)},
              str(d / "generation.safetensors"))


def _qk_layer(root: Path, req: str, q: torch.Tensor, layer_num: int = 31) -> None:
    d = root / req
    d.mkdir(parents=True, exist_ok=True)
    save_file({"q": q.contiguous().clone(), "k_full": q.contiguous().clone(),
               "k_prefix_ends": torch.tensor([q.shape[0]], dtype=torch.int64),
               "layer_num": torch.tensor(layer_num, dtype=torch.int64)},
              str(d / f"layer{layer_num:03d}.safetensors"))


def _identical_tree(root: Path, leg_names) -> dict:
    """One tree per leg name, all BIT-IDENTICAL to each other.

    The generator is reseeded to the SAME value at the start of every leg's build (rather
    than shared across legs), so every leg produces the identical sequence of tensors --
    the same trick `tests/test_parity_expected_gates.py::_capture_tree` uses to make a
    baseline where every gate passes cleanly before a test perturbs one specific value.
    """
    out = {}
    for name in leg_names:
        leg_root = root / name
        g = torch.Generator().manual_seed(0)
        for i in range(3):
            _generation(leg_root, f"req{i}")
            _qk_layer(leg_root, f"req{i}",
                      torch.randn(5, 8, generator=g, dtype=torch.float32).to(torch.float16))
        out[name] = leg_root
    return out


def test_qk_w33_distinct_gate_is_banded_not_bitexact():
    assert "qk_small routing-identity-w33-distinct [BANDED]" in EXPECTED_GATES["qk_small"]
    assert "qk_small RECORD routing-identity-w33-distinct (bit-exact)" in \
        EXPECTED_GATES["qk_small"]
    assert "qk_small routing-identity-w33-distinct" not in EXPECTED_GATES["qk_small"]


def test_hs_w33_distinct_gate_stays_bitexact_not_banded():
    assert "hs_small routing-identity-w33-distinct" in EXPECTED_GATES["hs_small"]
    assert "hs_small routing-identity-w33-distinct [BANDED]" not in EXPECTED_GATES["hs_small"]
    assert "hs_small RECORD routing-identity-w33-distinct (bit-exact)" not in \
        EXPECTED_GATES["hs_small"]


def test_the_measured_qk_boot_spread_passes_through_judge_capture(tmp_path):
    """The finding this task closes: 1.870e-03 tensor-global relative on layer031 q/k_full,
    the SAME cluster `_ROUTING_IDENTITY_QK_REL_BAND` was built on, must PASS `judge_capture`
    for qk_small even though a bit-exact `_compare` on the same artifacts would FAIL."""
    legs = _identical_tree(tmp_path, LEGS["qk_small"])
    # Perturb ONLY the aperture_batch_distinct req0 q tensor by the measured worst relative.
    path = legs["aperture_batch_distinct"] / "req0" / "layer031.safetensors"
    data = load_file(str(path))
    q = data["q"].float() * (1.0 + _MEASURED_WORST_REL)
    save_file({"q": q.to(torch.float16).contiguous(), "k_full": data["k_full"],
               "k_prefix_ends": data["k_prefix_ends"], "layer_num": data["layer_num"]},
              str(path))

    verdicts: list = []
    judge_capture("qk_small", legs, verdicts)
    by_label = {label: (ok, fatal) for label, ok, fatal in verdicts}

    ok, fatal = by_label["qk_small routing-identity-w33-distinct [BANDED]"]
    assert ok, "the measured boot-spread magnitude failed the banded gate"
    assert fatal

    record_ok, record_fatal = by_label[
        "qk_small RECORD routing-identity-w33-distinct (bit-exact)"]
    assert not record_ok, "the RECORD line should show the (expected) bit-exact mismatch"
    assert not record_fatal, "RECORD lines must stay non-fatal, informational only"


def test_an_off_by_one_row_still_fails_the_qk_w33_distinct_band(tmp_path):
    """The band must not turn into a rubber stamp: a real mis-route signature still fails."""
    legs = _identical_tree(tmp_path, LEGS["qk_small"])
    path = legs["aperture_batch_distinct"] / "req0" / "layer031.safetensors"
    data = load_file(str(path))
    shifted = torch.roll(data["q"], shifts=1, dims=0)
    save_file({"q": shifted.contiguous(), "k_full": data["k_full"],
               "k_prefix_ends": data["k_prefix_ends"], "layer_num": data["layer_num"]},
              str(path))

    verdicts: list = []
    judge_capture("qk_small", legs, verdicts)
    by_label = {label: ok for label, ok, _fatal in verdicts}
    assert not by_label["qk_small routing-identity-w33-distinct [BANDED]"], (
        "an off-by-one row slipped through the banded qk_small w33-distinct gate")
