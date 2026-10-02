"""`T1-batched capture_hs` / `T1-batched capture_qk` must tolerate the SAME boot noise QK's
T2 `routing-identity-w32` gate already tolerates, and still reject a mis-route (task D7).

`run_parity.sh`'s T1-batched section compares MIA's batched capture (33 distinct-length
requests, width 33) against an INDEPENDENT oracle -- a plain vanilla-vLLM forward hook that
never imports MIA -- and used to require bit-exactness. LSF 1725139 failed BOTH capture
kinds:

    capture_qk worst rel = 7.457e-04  (req0/layer031::q)
    capture_hs worst rel = 1.272e-03  (req0, growing with depth)

Both sit AT OR BELOW the re-derived vLLM boot-nondeterminism floor -- ALL 15 pairwise boots
of LSF 1710884 (`_compare_replay_band`'s own metric): worst tensor-global rel 1.446e-03,
worst per-row rel 1.880e-03. So this is the SAME noise `_ROUTING_IDENTITY_QK_REL_BAND`
already documents, not a new regime, and the fix reuses that band's IDENTICAL magnitude
(2e-2 / 1.5e-1) as `_T1_BATCHED_REL_BAND` / `_T1_BATCHED_ROW_BAND` rather than a new, looser
bound.

These tests pin that reused bound's discrimination the way `test_qk_routing_identity_band.py`
pins the T2 gate it is copied from: it must ACCEPT the measured spread (both capture kinds)
and REJECT an off-by-one row, a zeroed row, and a cross-request shape mismatch. A bound with
no test can drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    _T1_BATCHED_REL_BAND,
    _T1_BATCHED_ROW_BAND,
    _compare_replay_band,
)

# Measured directly on LSF 1725139's T1-batched artifacts (refb_capture_qk/refb_capture_hs
# vs miab_capture_qk/miab_capture_hs), with `_compare_replay_band`'s own metric:
_MEASURED_QK_REL = 7.457e-04
_MEASURED_HS_REL = 1.272e-03

# The boot-nondeterminism floor this sits under, re-derived over ALL 15 pairwise boots of
# LSF 1710884 (task D6) -- already pinned by `test_qk_routing_identity_band.py` for the T2
# gate this band is copied from. Reused here as a reference value, not re-derived.
_BOOT_FLOOR_REL = 1.446e-03
_BOOT_FLOOR_ROW = 1.880e-03

# The off-by-one / zeroed-row corruption floors (LSF 1705469), already pinned by
# `tests/test_parity_gates.py` and reused by `test_qk_routing_identity_band.py`.
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


def _rows(n: int = 20, width: int = 3072, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # A magnitude spread like real q/k or hidden-state activations: a few large channels
    # dominate max|v|, matching the real artifacts (worst tensors are always req0).
    x = torch.randn(n, width, generator=g, dtype=torch.float32)
    x[:, 0] *= 30.0
    return x.to(torch.float16)


def _verdict(a: Path, b: Path) -> bool:
    verdicts: list = []
    _compare_replay_band("unit T1-batched capture", a, b, name="layer*.safetensors",
                         verdicts=verdicts, rel=_T1_BATCHED_REL_BAND,
                         row=_T1_BATCHED_ROW_BAND, tag="T1")
    return verdicts[0][1]


def test_band_accepts_the_measured_qk_batched_oracle_delta(tmp_path):
    """LSF 1725139: T1-batched capture_qk differs from the independent oracle by 7.457e-04
    relative -- boot noise, not a mis-route -- so the gate must PASS."""
    q, k = _rows(), _rows(seed=1)
    a = _tree(tmp_path / "a", q, k)
    q2 = (q.float() * (1.0 + _MEASURED_QK_REL)).to(torch.float16)
    b = _tree(tmp_path / "b", q2, k)
    assert _verdict(a, b), "the T1-batched band rejected the measured QK boot spread"


def test_band_accepts_the_measured_hs_batched_oracle_delta(tmp_path):
    """LSF 1725139: T1-batched capture_hs differs from the independent oracle by 1.272e-03
    relative -- the same order of noise, now also measured on HS's batched capture."""
    q, k = _rows(), _rows(seed=1)
    a = _tree(tmp_path / "a", q, k)
    q2 = (q.float() * (1.0 + _MEASURED_HS_REL)).to(torch.float16)
    b = _tree(tmp_path / "b", q2, k)
    assert _verdict(a, b), "the T1-batched band rejected the measured HS boot spread"


def test_band_rejects_an_off_by_one_row(tmp_path):
    """A one-row shift inside the request -- measured floor 4.512e-01 tensor-global, 22.6x
    above the band. This is the mis-route shape the gate exists to catch."""
    q, k = _rows(), _rows(seed=1)
    a = _tree(tmp_path / "a", q, k)
    b = _tree(tmp_path / "b", torch.roll(q, shifts=1, dims=0), k)
    assert not _verdict(a, b), "an off-by-one row shift slipped through the T1-batched band"


def test_band_rejects_a_zeroed_row(tmp_path):
    """A zeroed / sentinel row scores exactly 1.0 per-row -- the padding-reads-stale-routing
    hazard the whole replay-band design exists for."""
    q, k = _rows(), _rows(seed=1)
    a = _tree(tmp_path / "a", q, k)
    corrupt_k = k.clone()
    corrupt_k[3] = 0
    b = _tree(tmp_path / "b", q, corrupt_k)
    assert not _verdict(a, b), "a zeroed row slipped through the T1-batched band"


def test_band_rejects_a_shape_change_from_a_cross_request_misroute(tmp_path):
    """A mis-route between two requests of different length is a SHAPE mismatch, caught with
    no tolerance at all -- distinct prompt lengths are what makes this true in the real
    batched-oracle workload (task D5 item 8)."""
    a = _tree(tmp_path / "a", _rows(n=20, seed=1), _rows(n=20, seed=2))
    b = _tree(tmp_path / "b", _rows(n=25, seed=3), _rows(n=25, seed=4))
    assert not _verdict(a, b), "a differently-shaped artifact slipped through the T1-batched band"


def test_band_holds_integer_metadata_bit_exact(tmp_path):
    """`layer_num` / `k_prefix_ends` are metadata, not signal; the band must never touch
    them -- a wrong layer number is a silent mislabel, not noise."""
    from safetensors.torch import save_file

    q, k = _rows(), _rows(seed=1)
    a = _tree(tmp_path / "a", q, k, layer_num=31)
    d = tmp_path / "b" / "req0"
    d.mkdir(parents=True)
    save_file({"q": q.contiguous(), "k_full": k.contiguous(),
               "k_prefix_ends": torch.tensor([q.shape[0] + 1], dtype=torch.int64),
               "layer_num": torch.tensor(31, dtype=torch.int64)},
              str(d / "layer031.safetensors"))
    assert not _verdict(a, tmp_path / "b"), "a wrong k_prefix_ends slipped through"


def test_bound_is_the_identical_magnitude_as_the_qk_t2_band():
    """Task D7's instruction: reuse the QK T2 band's magnitude exactly, do not invent a
    looser one. Two independently-named constants, same numbers."""
    from tests.mia.parity.t2_invariants import (
        _ROUTING_IDENTITY_QK_REL_BAND,
        _ROUTING_IDENTITY_QK_ROW_BAND,
    )

    assert _T1_BATCHED_REL_BAND == _ROUTING_IDENTITY_QK_REL_BAND
    assert _T1_BATCHED_ROW_BAND == _ROUTING_IDENTITY_QK_ROW_BAND


def test_bounds_sit_between_the_measured_spread_and_the_corruption_floors():
    """The bound is only defensible while it separates the two measured populations: the
    boot-to-boot noise floor (LSF 1710884 / LSF 1725139) below it, the mis-route floors
    (LSF 1705469) above it."""
    assert _MEASURED_QK_REL < _BOOT_FLOOR_REL < _T1_BATCHED_REL_BAND < _OFF_BY_ONE_REL_FLOOR
    assert _MEASURED_HS_REL < _T1_BATCHED_REL_BAND < _OFF_BY_ONE_REL_FLOOR
    assert _BOOT_FLOOR_ROW < _T1_BATCHED_ROW_BAND < _OFF_BY_ONE_ROW_FLOOR
    assert _T1_BATCHED_ROW_BAND < _ZEROED_ROW

    # Margins, restated numerically so a future change to either constant trips this test
    # rather than silently drifting -- mirrors the QK T2 band's own margin test.
    assert _T1_BATCHED_REL_BAND / _MEASURED_QK_REL > 26.0        # ~26.8x
    assert _T1_BATCHED_REL_BAND / _MEASURED_HS_REL > 15.0        # ~15.7x
    assert _OFF_BY_ONE_REL_FLOOR / _T1_BATCHED_REL_BAND > 22.0   # ~22.6x
    assert _OFF_BY_ONE_ROW_FLOOR / _T1_BATCHED_ROW_BAND > 3.9    # ~4.0x, thinnest
