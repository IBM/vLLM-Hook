"""QK's `routing-identity-w32` gate must tolerate boot noise and still reject a mis-route.

Task D4c. `tests/mia/parity/t2_invariants.py`'s `qk_small routing-identity-w32` compared MIA's
two capture mechanisms (forward-hook probes vs the baked aperture op) at width 32 and used
to be bit-exact. It passed in LSF 1705469 and 1705794, then failed in 1710144 with no code
change in between. The diagnosis (LSF 1710884, 3 boots per arm) is that NEITHER arm is
self-reproducible boot-to-boot, and the cross-arm delta is the same magnitude as the
within-arm delta -- the comparison was measuring vLLM's attention-kernel nondeterminism
(split-K / atomics / per-boot autotuning), not a MIA defect. `hs_small`'s
`routing-identity-w32` stayed bit-exact across the SAME runs, and the `w33-distinct` legs
passed bit-exact for BOTH `hs` and `qk` in LSF 1710144 -- so this is QK-at-width-32
specifically, not "QK is always noisy."

The gate is now `_compare_replay_band` (reused, not reimplemented) with its own bound,
`_ROUTING_IDENTITY_QK_REL_BAND` / `_ROUTING_IDENTITY_QK_ROW_BAND`. These tests pin that
bound's discrimination the way `tests/test_parity_gates.py` pins `_REPLAY_REL_BAND` /
`_REPLAY_ROW_BAND`: it must ACCEPT the measured boot-to-boot spread and REJECT an
off-by-one row and a zeroed row. A bound with no test can drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    _ROUTING_IDENTITY_QK_REL_BAND,
    _ROUTING_IDENTITY_QK_ROW_BAND,
    _compare_replay_band,
)

# Measured directly on the LSF 1710884 artifacts (`that job’s artifacts (
# A1,A2,A3,B1,B2,B3)), with `_compare_replay_band`'s own metric, over all 3 same-arm-A,
# 3 same-arm-B and 3 matched-boot cross-arm (A1/B1, A2/B2, A3/B3) pairs on the qk_small
# `layer*.safetensors` artifacts (q, k_full; 198 float tensors per pair):
#   worst tensor-global relative (max|a-b|/max|a|): 1.446e-03
#   worst per-row relative (max_r max|a[r]-b[r]|/max|a[r]|): 1.880e-03
# All 9 pairs land in [1.216e-03, 1.880e-03] on both metrics -- same order same-arm or
# cross-arm, which is the evidence that this is a shared noise floor, not a mis-route.
_MEASURED_WORST_REL = 1.446e-03
_MEASURED_WORST_ROW = 1.880e-03

# The off-by-one / zeroed-row corruption floors, measured elsewhere in this suite (LSF
# 1705469) and already pinned by `tests/test_parity_gates.py::
# test_band_bounds_sit_between_the_measured_rounding_and_the_corruption_floors`. Reused here
# as reference values, not re-derived.
_OFF_BY_ONE_REL_FLOOR = 4.512e-01
_OFF_BY_ONE_ROW_FLOOR = 5.959e-01
_ZEROED_ROW = 1.0


def _tree(root: Path, q: torch.Tensor, k_full: torch.Tensor, layer_num: int = 31) -> Path:
    from safetensors.torch import save_file

    d = root / "req0"
    d.mkdir(parents=True, exist_ok=True)
    save_file({"q": q.contiguous(), "k_full": k_full.contiguous(),
               "k_prefix_ends": torch.tensor([q.shape[0]], dtype=torch.int64),
               "layer_num": torch.tensor(layer_num, dtype=torch.int64)},
              str(d / f"layer{layer_num:03d}.safetensors"))
    return root


def _qk_rows(n: int = 20, width: int = 3072, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # A magnitude spread like real q/k projections: a few large channels dominate max|v|,
    # matching the measured artifacts (worst tensors are always req0/layer031::k_full).
    x = torch.randn(n, width, generator=g, dtype=torch.float32)
    x[:, 0] *= 30.0
    return x.to(torch.float16)


def _verdict(a: Path, b: Path) -> bool:
    verdicts: list = []
    _compare_replay_band("unit qk routing-identity-w32", a, b, name="layer*.safetensors",
                         verdicts=verdicts, rel=_ROUTING_IDENTITY_QK_REL_BAND,
                         row=_ROUTING_IDENTITY_QK_ROW_BAND)
    return verdicts[0][1]


def test_band_accepts_the_measured_boot_to_boot_spread(tmp_path):
    """LSF 1710884: same arm, different boot, differs by up to 1.446e-03 tensor-global and
    1.880e-03 per-row. The gate exists to tolerate exactly this -- vLLM's own attention
    kernel nondeterminism -- so it must PASS."""
    q, k = _qk_rows(), _qk_rows(seed=1)
    a = _tree(tmp_path / "a", q, k)
    q2 = (q.float() * (1.0 + _MEASURED_WORST_REL)).to(torch.float16)
    k2 = (k.float() * (1.0 + _MEASURED_WORST_ROW)).to(torch.float16)
    b = _tree(tmp_path / "b", q2, k2)
    assert _verdict(a, b), "the QK routing-identity band rejected the measured boot spread"


def test_band_rejects_an_off_by_one_row(tmp_path):
    """A one-row shift inside the request -- measured floor 4.512e-01 tensor-global /
    5.959e-01 per-row, 22.6x / 4.0x above the band. This is the mis-route shape the gate
    exists to catch; boot noise must never be confused with it."""
    q, k = _qk_rows(), _qk_rows(seed=1)
    a = _tree(tmp_path / "a", q, k)
    b = _tree(tmp_path / "b", torch.roll(q, shifts=1, dims=0), k)
    assert not _verdict(a, b), "an off-by-one row shift slipped through the QK band"


def test_band_rejects_a_zeroed_row(tmp_path):
    """A zeroed / sentinel row scores exactly 1.0 per-row (the padding-reads-stale-routing
    hazard the whole replay-band design exists for) -- 6.7x above the row band."""
    q, k = _qk_rows(), _qk_rows(seed=1)
    a = _tree(tmp_path / "a", q, k)
    corrupt_k = k.clone()
    corrupt_k[3] = 0
    b = _tree(tmp_path / "b", q, corrupt_k)
    assert not _verdict(a, b), "a zeroed row slipped through the QK band"


def test_band_rejects_a_shape_change_from_a_cross_request_misroute(tmp_path):
    """A mis-route between two requests of different length is a SHAPE mismatch, caught with
    no tolerance at all -- the same guarantee `_compare_replay_band` gives every caller."""
    a = _tree(tmp_path / "a", _qk_rows(n=20, seed=1), _qk_rows(n=20, seed=2))
    b = _tree(tmp_path / "b", _qk_rows(n=25, seed=3), _qk_rows(n=25, seed=4))
    assert not _verdict(a, b), "a differently-shaped artifact slipped through the QK band"


def test_band_holds_integer_metadata_bit_exact(tmp_path):
    """`layer_num` / `k_prefix_ends` are metadata, not signal; the band must never touch
    them -- a wrong layer number is a silent mislabel, not noise."""
    from safetensors.torch import save_file

    q, k = _qk_rows(), _qk_rows(seed=1)
    a = _tree(tmp_path / "a", q, k, layer_num=31)
    d = tmp_path / "b" / "req0"
    d.mkdir(parents=True)
    save_file({"q": q.contiguous(), "k_full": k.contiguous(),
               "k_prefix_ends": torch.tensor([q.shape[0] + 1], dtype=torch.int64),
               "layer_num": torch.tensor(31, dtype=torch.int64)},
              str(d / "layer031.safetensors"))
    assert not _verdict(a, tmp_path / "b"), "a wrong k_prefix_ends slipped through"


def test_bounds_sit_between_the_measured_spread_and_the_corruption_floors():
    """The bound is only defensible while it separates the two measured populations: the
    boot-to-boot noise floor (LSF 1710884) below it, the mis-route floors (LSF 1705469)
    above it."""
    assert _MEASURED_WORST_REL < _ROUTING_IDENTITY_QK_REL_BAND < _OFF_BY_ONE_REL_FLOOR
    assert _MEASURED_WORST_ROW < _ROUTING_IDENTITY_QK_ROW_BAND < _OFF_BY_ONE_ROW_FLOOR
    assert _ROUTING_IDENTITY_QK_ROW_BAND < _ZEROED_ROW

    # Margins, restated numerically so a future change to either constant trips this test
    # rather than silently drifting: see `_ROUTING_IDENTITY_QK_REL_BAND`'s comment in
    # tests/mia/parity/t2_invariants.py for the same numbers spelled out.
    assert _ROUTING_IDENTITY_QK_REL_BAND / _MEASURED_WORST_REL > 13.0       # ~13.8x
    assert _OFF_BY_ONE_REL_FLOOR / _ROUTING_IDENTITY_QK_REL_BAND > 22.0     # ~22.6x
    assert _ROUTING_IDENTITY_QK_ROW_BAND / _MEASURED_WORST_ROW > 79.0       # ~79.8x
    assert _OFF_BY_ONE_ROW_FLOOR / _ROUTING_IDENTITY_QK_ROW_BAND > 3.9      # ~4.0x, thinnest
