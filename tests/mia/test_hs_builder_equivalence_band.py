"""The HS routing-builder-equivalence gates must tolerate boot noise and still reject a
mis-route.

Task D4f. `tests/mia/parity/t2_invariants.py`'s `hs_small builder legacy vs default` and
`hs_small builder vectorized vs default` compare MIA's three interchangeable HS
routing-table builders -- legacy (`_build_routing_hs`), vectorized
(`_build_routing_hs_vectorized`), and decode-cache (`_build_routing_hs_decode_cache`, the
default) -- and used to require bit-exact agreement.

The diagnosis (LSF 1722805, 3 boots per builder, 40 comparison pairs; NOT re-run here --
the diagnosis is complete) found that EVERY builder disagrees with ITSELF boot-to-boot at
the same tail requests (reqs 30-32, the second prefill wave): SELF default FAIL 3/9/9
problems, SELF legacy FAIL 3/6/3, SELF vectorized PASS-then-FAIL 0/9/9, SELF w33 default
FAIL 6. A bit-exact CROSS-builder gate was never achievable -- the noise floor already
exceeds it WITHIN a single builder. Several cross-builder pairs land bit-exact anyway
(`defaultboot3-vs-vectorizedboot1/2`, `legacyboot3-vs-vectorizedboot1/2`,
`defaultboot3-vs-legacyboot3`), which is positive evidence that legacy and vectorized
compute the SAME routing, not different routing that happens to be close. Token ids are
identical on all 40 pairs.

The gate is now `_compare_replay_band` (reused, not reimplemented) with its own bound,
`_BUILDER_EQUIV_REL_BAND` / `_BUILDER_EQUIV_ROW_BAND`. These tests pin that bound's
discrimination the way `tests/test_parity_gates.py` pins `_REPLAY_REL_BAND` /
`_REPLAY_ROW_BAND` and `tests/test_qk_routing_identity_band.py` pins the QK routing-identity
band: it must ACCEPT the measured 40-pair spread and REJECT an off-by-one row and a zeroed
row. A bound with no test can drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    _BUILDER_EQUIV_REL_BAND,
    _BUILDER_EQUIV_ROW_BAND,
    _compare_replay_band,
)

# Measured directly on the LSF 1722805 artifacts (3 boots per builder: default, legacy,
# vectorized; 40 comparison pairs total: same-builder cross-boot + cross-builder
# matched-and-mismatched-boot), with `_compare_replay_band`'s own metric on the `hs_small`
# `layer*.safetensors` artifacts (hidden_states):
#   worst tensor-global relative (max|a-b|/max|a|): 2.230e-03
#   worst per-row relative (max_r max|a[r]-b[r]|/max|a[r]|): 8.348e-03
# token_ids_differ=no on all 40 pairs -- generation is unaffected; the noise is sub-argmax.
_MEASURED_WORST_REL = 2.230e-03
_MEASURED_WORST_ROW = 8.348e-03

# The off-by-one / zeroed-row corruption floors, measured elsewhere in this suite (LSF
# 1705469) and already pinned by `tests/test_parity_gates.py::
# test_band_bounds_sit_between_the_measured_rounding_and_the_corruption_floors` and
# `tests/test_qk_routing_identity_band.py`. Reused here as reference values, not re-derived.
_OFF_BY_ONE_REL_FLOOR = 4.512e-01
_OFF_BY_ONE_ROW_FLOOR = 5.959e-01
_ZEROED_ROW = 1.0


def _tree(root: Path, hidden: torch.Tensor, layer_num: int = 16) -> Path:
    from safetensors.torch import save_file

    d = root / "req0"
    d.mkdir(parents=True, exist_ok=True)
    save_file({"hidden_states": hidden.contiguous(),
               "layer_num": torch.tensor(layer_num, dtype=torch.int64)},
              str(d / f"layer{layer_num:03d}.safetensors"))
    return root


def _rows(n: int = 20, width: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # A magnitude spread like a real residual stream: a few large channels dominate max|v|.
    x = torch.randn(n, width, generator=g, dtype=torch.float32)
    x[:, 0] *= 500.0
    return x.to(torch.float16)


def _verdict(a: Path, b: Path) -> bool:
    verdicts: list = []
    _compare_replay_band("unit hs builder-equivalence", a, b, name="layer*.safetensors",
                         verdicts=verdicts, rel=_BUILDER_EQUIV_REL_BAND,
                         row=_BUILDER_EQUIV_ROW_BAND)
    return verdicts[0][1]


def test_band_accepts_the_measured_boot_to_boot_spread(tmp_path):
    """LSF 1722805: 40 cross-boot / cross-builder pairs differ by up to 2.230e-03
    tensor-global and 8.348e-03 per-row -- the same vLLM boot-to-boot nondeterminism
    `_ROUTING_IDENTITY_QK_REL_BAND` documents for QK, now measured on HS's own builders. The
    gate exists to tolerate exactly this, so it must PASS."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    perturbed = (x.float() * (1.0 + _MEASURED_WORST_REL)).to(torch.float16)
    b = _tree(tmp_path / "b", perturbed)
    assert _verdict(a, b), "the HS builder-equivalence band rejected the measured boot spread"


def test_per_row_band_accepts_the_measured_per_row_spread(tmp_path):
    """The per-row metric's own measured worst (8.348e-03) must also pass, independent of
    the tensor-global check above."""
    x = _rows()
    x[3] *= 1e-3
    a = _tree(tmp_path / "a", x)
    perturbed = x.clone().float()
    perturbed[3] *= 1.0 + _MEASURED_WORST_ROW
    b = _tree(tmp_path / "b", perturbed.to(torch.float16))
    assert _verdict(a, b), "the per-row band rejected the measured per-row boot spread"


def test_band_rejects_an_off_by_one_row(tmp_path):
    """A one-row shift inside the request -- measured floor ~4.5e-01 tensor-global / ~6.0e-01
    per-row, ~22.5x / ~4.0x above the band. This is the mis-route shape the gate exists to
    catch; boot noise must never be confused with it."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    b = _tree(tmp_path / "b", torch.roll(x, shifts=1, dims=0))
    assert not _verdict(a, b), "an off-by-one row shift slipped through the builder band"


def test_band_rejects_a_zeroed_row(tmp_path):
    """A zeroed / sentinel row scores exactly 1.0 per-row (the padding-reads-stale-routing
    hazard the whole replay-band design exists for) -- ~6.7x above the row band."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    corrupt = x.clone()
    corrupt[7] = 0
    b = _tree(tmp_path / "b", corrupt)
    assert not _verdict(a, b), "a zeroed row slipped through the builder band"


def test_band_rejects_a_shape_change_from_a_cross_request_misroute(tmp_path):
    """A mis-route between two requests of different length is a SHAPE mismatch, caught with
    no tolerance at all -- the same guarantee `_compare_replay_band` gives every caller."""
    a = _tree(tmp_path / "a", _rows(n=20, seed=1))
    b = _tree(tmp_path / "b", _rows(n=25, seed=2))
    assert not _verdict(a, b), "a differently-shaped artifact slipped through the builder band"


def test_band_holds_integer_metadata_bit_exact(tmp_path):
    """`layer_num` is metadata, not signal; the band must never touch it -- a wrong layer
    number is a silent mislabel, not noise."""
    from safetensors.torch import save_file

    x = _rows()
    a = _tree(tmp_path / "a", x, layer_num=16)
    d = tmp_path / "b" / "req0"
    d.mkdir(parents=True)
    save_file({"hidden_states": x.contiguous(),
               "layer_num": torch.tensor(15, dtype=torch.int64)},
              str(d / "layer016.safetensors"))
    assert not _verdict(a, tmp_path / "b"), "a wrong layer_num slipped through"


def test_bounds_sit_between_the_measured_spread_and_the_corruption_floors():
    """The bound is only defensible while it separates the two measured populations: the
    boot-to-boot noise floor (LSF 1722805) below it, the mis-route floors (LSF 1705469)
    above it."""
    assert _MEASURED_WORST_REL < _BUILDER_EQUIV_REL_BAND < _OFF_BY_ONE_REL_FLOOR
    assert _MEASURED_WORST_ROW < _BUILDER_EQUIV_ROW_BAND < _OFF_BY_ONE_ROW_FLOOR
    assert _BUILDER_EQUIV_ROW_BAND < _ZEROED_ROW

    # Margins, restated numerically so a future change to either constant trips this test
    # rather than silently drifting: see `_BUILDER_EQUIV_REL_BAND`'s comment in
    # tests/mia/parity/t2_invariants.py for the same numbers spelled out.
    assert _BUILDER_EQUIV_REL_BAND / _MEASURED_WORST_REL > 8.0          # ~9.0x (8.97x)
    assert _OFF_BY_ONE_REL_FLOOR / _BUILDER_EQUIV_REL_BAND > 22.0       # ~22.5x
    assert _BUILDER_EQUIV_ROW_BAND / _MEASURED_WORST_ROW > 17.0         # ~18.0x (17.97x)
    assert _OFF_BY_ONE_ROW_FLOOR / _BUILDER_EQUIV_ROW_BAND > 3.9        # ~4.0x, thinnest
