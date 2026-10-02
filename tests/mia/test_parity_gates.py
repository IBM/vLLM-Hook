"""The T2 replay-band gate must accept rounding and reject corruption.

`tests/mia/parity/t2_invariants.py`'s bit-exact gates (`op-identity`, `routing-identity-w32`)
buy their exactness by running MIA's baked op against an UNCOMPILED model, which replays no
CUDA graph and pads to nothing. The one leg that touches real replay-time routing --
forward hooks vs a real FULL-cudagraph run -- cannot be bit-exact, because it inherits
vLLM's compiled-vs-uncompiled numerics. It is therefore a MAGNITUDE band, and a band is only
worth anything if the distance between "rounding" and "corruption" is real.

These tests pin that distance with synthetic trees, so the bound cannot quietly drift into
either uselessness (accepting a mis-route) or flakiness (rejecting fp16 rounding) without a
failure here. The magnitudes are the ones measured on GPU in LSF 1702981 and recorded in
`_REPLAY_REL_BAND`'s comment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    _REPLAY_REL_BAND,
    _REPLAY_ROW_BAND,
    _compare_replay_band,
)


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
    _compare_replay_band("unit", a, b, name="layer*.safetensors", verdicts=verdicts)
    return verdicts[0][1]


def test_band_accepts_fp_rounding_at_the_measured_scale(tmp_path):
    """The real replay leg measured 6.12e-03 tensor-global and 2.74e-02 per-row at worst
    (LSF 1705469). Rounding is PROPORTIONAL to each value, not a constant offset, so that is
    how it is modelled here. Both bands must accept it or the gate is flaky against the
    numerics it exists to tolerate."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    perturbed = (x.float() * (1.0 + 6.0e-03)).to(torch.float16)
    b = _tree(tmp_path / "b", perturbed)
    assert _verdict(a, b), "the gate rejected rounding at the measured real-replay scale"


def test_per_row_band_has_headroom_for_a_small_rows_own_rounding(tmp_path):
    """Per-row normalisation amplifies a small row's rounding -- that is the whole point, and
    it is also why the per-row band (1.5e-1) is looser than the global one. The measured
    per-row worst on real artifacts is 2.74e-02; that must still PASS."""
    x = _rows()
    x[3] *= 1e-3
    a = _tree(tmp_path / "a", x)
    perturbed = x.clone().float()
    perturbed[3] *= 1.0 + 2.7e-02        # only the small row moves, at the measured level
    b = _tree(tmp_path / "b", perturbed.to(torch.float16))
    assert _verdict(a, b), "the per-row band rejected a small row's own measured rounding"


def test_per_row_band_catches_a_zeroed_row_the_global_metric_would_miss(tmp_path):
    """The reason the per-row metric exists (task D4 fix round 2).

    The tensor-global metric divides by the tensor's LARGEST value, so zeroing a row whose
    own magnitude is small scores small: measured 1.11e-02 on real HS artifacts, UNDER the
    2e-2 global band. Per-row normalisation makes any destroyed row score exactly 1.0.
    """
    x = _rows()
    x[3] *= 1e-3                      # a row that is small next to the rest of the tensor
    a = _tree(tmp_path / "a", x)
    corrupt = x.clone()
    corrupt[3] = 0
    b = _tree(tmp_path / "b", corrupt)

    glob = (x.float() - corrupt.float()).abs().max().item() / x.abs().max().item()
    assert glob < _REPLAY_REL_BAND, (
        f"precondition: this corruption must be invisible to the global metric, got {glob:.3e}")
    assert not _verdict(a, b), "a zeroed low-magnitude row slipped past the per-row band"


def test_gate_reports_an_artifact_present_only_on_side_b(tmp_path):
    """A leaked EXTRA request lives only on side B; a side-A-only walk never visits it."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    b = _tree(tmp_path / "b", x)
    _tree(tmp_path / "b" / "extra", x)          # an artifact with no counterpart on A
    (tmp_path / "b" / "req1").mkdir(exist_ok=True)
    from safetensors.torch import save_file
    save_file({"hidden_states": x.contiguous(),
               "layer_num": torch.tensor(16, dtype=torch.int64)},
              str(tmp_path / "b" / "req1" / "layer016.safetensors"))
    assert not _verdict(a, b), "an artifact present only on side B was not reported"


def test_band_rejects_a_zeroed_or_sentinel_row(tmp_path):
    """The exact shape of the hazard the gate exists for: a padding row that read stale
    routing scatters into a live aperture slot, or a re-allocated slab leaves the sentinel
    behind. Relative signature ~1.0, i.e. 50x the band."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    corrupt = x.clone()
    corrupt[7] = 0
    b = _tree(tmp_path / "b", corrupt)
    assert not _verdict(a, b), "a zeroed row slipped through the band"


def test_band_rejects_an_off_by_one_row(tmp_path):
    """A one-row shift inside the request -- measured >= 4.58e-01 relative, 23x the band."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    b = _tree(tmp_path / "b", torch.roll(x, shifts=1, dims=0))
    assert not _verdict(a, b), "an off-by-one row shift slipped through the band"


def test_band_rejects_another_requests_rows_by_shape(tmp_path):
    """Distinct prompts have distinct row counts in this workload, so a mis-route between
    them is a SHAPE mismatch -- caught with no tolerance involved."""
    a = _tree(tmp_path / "a", _rows(n=20, seed=1))
    b = _tree(tmp_path / "b", _rows(n=25, seed=2))
    assert not _verdict(a, b), "a differently shaped artifact slipped through"


def test_band_holds_integer_metadata_bit_exact(tmp_path):
    """`layer_num` / `k_prefix_ends` are metadata; a band would be meaningless for them, and
    a wrong layer number is exactly the silent-mislabel failure the suite exists to stop."""
    x = _rows()
    a = _tree(tmp_path / "a", x, layer_num=16)
    from safetensors.torch import save_file

    d = tmp_path / "b" / "req0"
    d.mkdir(parents=True)
    save_file({"hidden_states": x.contiguous(),
               "layer_num": torch.tensor(15, dtype=torch.int64)},
              str(d / "layer016.safetensors"))
    assert not _verdict(a, tmp_path / "b"), "a wrong layer_num slipped through"


def test_band_fails_loudly_on_an_empty_side(tmp_path):
    """Two empty trees agree on everything. That must never read as a PASS."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not _verdict(empty, _tree(tmp_path / "b", _rows()))


def test_band_bounds_sit_between_the_measured_rounding_and_the_corruption_floors():
    """Each bound is only defensible while it separates the two measured populations.

    All figures LSF 1705469, and they are FLOORS over the corruption population, not the
    best case -- the distinction that made the previous revision of the sentinel margin wrong.
    """
    # tensor-global: worst observed rounding  <  band  <  off-by-one FLOOR
    assert 6.117e-03 < _REPLAY_REL_BAND < 4.512e-01
    # per-row: worst observed rounding  <  band  <  off-by-one FLOOR (and a zeroed row = 1.0)
    assert 2.736e-02 < _REPLAY_ROW_BAND < 5.959e-01

    # The documented blind spot, asserted so it cannot be quietly forgotten: a same-shape
    # (same-prompt replica) mis-route sits UNDER both bands and is invisible to value gates.
    # It is closed by construction instead, by the distinct-prompt-length legs.
    assert 2.226e-03 < _REPLAY_REL_BAND
    assert 1.107e-02 < _REPLAY_ROW_BAND


@pytest.mark.parametrize("scale", [1e-4, 1e-3])
def test_band_accepts_across_the_magnitude_range_it_must(tmp_path, scale):
    x = _rows()
    a = _tree(tmp_path / f"a{scale}", x)
    b = _tree(tmp_path / f"b{scale}", (x.float() * (1.0 + scale)).to(torch.float16))
    assert _verdict(a, b)
