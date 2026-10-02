"""`T1-batched capture_hs` / `capture_qk` GENERATION must tolerate the SAME cross-boot
logprob noise `_T0_GRAPH_LOGPROB_BAND` already documents, with TOKEN IDS held bit-exact and
FATAL (task D10 item 1).

`run_parity.sh`'s T1-batched section used to compare `generation.safetensors` (prompts,
token ids, logprobs) bit-exact between the independent vanilla-vLLM oracle and MIA's batched
capture. LSF 1725139 never showed it move; LSF 1731353 did, on BOTH capture kinds, on the
SAME two channels:

    req0/generation.safetensors::cumulative_logprob  max|d| = 1.854e-02
    req0/generation.safetensors::token_logprobs      max|d| = 1.269e-02

`token_ids` and `prompt_token_ids` were NOT among the mismatches on either capture kind --
the two arms produced the identical sequence on all 33 requests; only the emitted logprobs
moved. This is the SAME cross-boot logprob noise `_T0_GRAPH_LOGPROB_BAND` documents (worst
measured HS 1.815024e-02 there), now observed between the vanilla-vLLM T1 reference boot and
the MIA boot rather than between two MIA boots.

`_T1_BATCHED_LOGPROB_BAND` reuses `_T0_GRAPH_LOGPROB_BAND`'s VALUE (5e-2) as its own,
independently-retunable constant, gating `_compare_generation_band` (task D5 item 3's
silent-vacuity-closed gate, reused here through `--compare-t1-batched-generation`, not
reimplemented). These tests pin that reuse and the accept/reject boundary the way
`test_parity_band_discrimination.py` pins `_T0_GRAPH_LOGPROB_BAND` itself, and the way
`test_parity_t1_batched_band.py` pins the sibling capture-payload band. A bound with no test
can drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    _BANDS,
    _T0_GRAPH_LOGPROB_BAND,
    _T1_BATCHED_LOGPROB_BAND,
    _compare_generation_band,
)

# Measured directly on LSF 1731353's T1-batched artifacts (refb_capture_hs/refb_capture_qk
# vs miab_capture_hs/miab_capture_qk, req0) -- IDENTICAL on both capture kinds.
_MEASURED_CUM_LOGPROB = 1.854e-02
_MEASURED_TOKEN_LOGPROBS = 1.269e-02

# The scale of a real steering intervention on this exact logprob channel -- already pinned
# by test_parity_band_discrimination.py for _T0_GRAPH_LOGPROB_BAND -- reused as the "this is
# not noise any more" ceiling.
_STEER_EFFECT = 7.632304e-01


def _generation(root: Path, *, cum_shift: float, tok_shift: float,
                token_ids=(11, 12, 13), req: str = "req0") -> Path:
    from safetensors.torch import save_file

    d = root / req
    d.mkdir(parents=True, exist_ok=True)
    save_file({"prompt_token_ids": torch.tensor([5, 6, 7], dtype=torch.int64),
               "token_ids": torch.tensor(list(token_ids), dtype=torch.int64),
               "token_logprobs": torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
                   + tok_shift,
               "cumulative_logprob": torch.tensor(-6.0 + cum_shift, dtype=torch.float64)},
              str(d / "generation.safetensors"))
    return root


def _verdict(a: Path, b: Path) -> bool:
    verdicts: list = []
    _compare_generation_band("unit T1-batched generation", a, b, verdicts=verdicts,
                             bound=_T1_BATCHED_LOGPROB_BAND, note="unit test", tag="T1")
    assert verdicts[0][0] == "unit T1-batched generation"
    return verdicts[0][1]


def test_band_accepts_the_measured_cumulative_logprob_delta(tmp_path):
    """LSF 1731353: req0/generation.safetensors::cumulative_logprob moved 1.854e-02 on BOTH
    capture kinds, with token ids unchanged -- cross-boot noise, not a routing defect."""
    a = _generation(tmp_path / "a", cum_shift=0.0, tok_shift=0.0)
    b = _generation(tmp_path / "b", cum_shift=_MEASURED_CUM_LOGPROB, tok_shift=0.0)
    assert _verdict(a, b), "the T1-batched generation band rejected the measured delta"


def test_band_accepts_the_measured_token_logprobs_delta(tmp_path):
    """Same job, the sibling channel: token_logprobs moved 1.269e-02."""
    a = _generation(tmp_path / "a", cum_shift=0.0, tok_shift=0.0)
    b = _generation(tmp_path / "b", cum_shift=0.0, tok_shift=_MEASURED_TOKEN_LOGPROBS)
    assert _verdict(a, b), "the T1-batched generation band rejected the measured delta"


def test_band_rejects_an_intervention_scale_delta(tmp_path):
    """The reject side: if generation started genuinely diverging, the scale to beat is a
    real steering intervention on these same logprobs (7.632304e-01, ~15x the band)."""
    a = _generation(tmp_path / "a", cum_shift=0.0, tok_shift=0.0)
    b = _generation(tmp_path / "b", cum_shift=_STEER_EFFECT, tok_shift=0.0)
    assert not _verdict(a, b), "an intervention-scale delta slipped through the band"


def test_band_is_fatal_on_a_token_id_mismatch_no_matter_the_logprob_gap(tmp_path):
    """The one thing this gate must never tolerate: a routing/generation bug that also
    moves the sampled token must fail even under an arbitrarily wide band -- this is what
    keeps the batched oracle's independence load-bearing (do NOT band token ids)."""
    a = _generation(tmp_path / "a", cum_shift=0.0, tok_shift=0.0, token_ids=(11, 12, 13))
    b = _generation(tmp_path / "b", cum_shift=0.0, tok_shift=0.0, token_ids=(11, 12, 14))
    verdicts: list = []
    _compare_generation_band("unit", a, b, verdicts=verdicts, bound=1.0, note="wide open",
                             tag="T1")
    label, ok, fatal = verdicts[0]
    assert not ok, "a token-id mismatch must fail even under an arbitrarily wide band"
    assert fatal, "the token-id check must be FATAL"


def test_bound_reuses_the_t0_graph_bands_value_as_a_separate_named_constant():
    """Task D10's instruction: reuse `_T0_GRAPH_LOGPROB_BAND`'s VALUE rather than invent a
    new one, but keep it a separate, independently-retunable constant -- it gates a
    DIFFERENT comparison (reference-vs-MIA generation at width 33, not capture-off-vs-on at
    width 1)."""
    assert _T1_BATCHED_LOGPROB_BAND == _T0_GRAPH_LOGPROB_BAND == 5e-2
    assert "MIA_T1_BATCHED_LOGPROB_BAND" in _BANDS
    assert "MIA_T2_T0_GRAPH_BAND" in _BANDS
    assert (_BANDS["MIA_T1_BATCHED_LOGPROB_BAND"]["what"]
            != _BANDS["MIA_T2_T0_GRAPH_BAND"]["what"]), (
        "the two bands must be documented as gating DIFFERENT comparisons, even though "
        "they resolve to the same value today")


def test_bound_sits_between_the_measurement_and_the_corruption():
    assert _MEASURED_CUM_LOGPROB < _T1_BATCHED_LOGPROB_BAND < _STEER_EFFECT
    assert _MEASURED_TOKEN_LOGPROBS < _T1_BATCHED_LOGPROB_BAND
    assert _T1_BATCHED_LOGPROB_BAND / _MEASURED_CUM_LOGPROB > 2.0     # ~2.7x
    assert _T1_BATCHED_LOGPROB_BAND / _MEASURED_TOKEN_LOGPROBS > 3.9  # ~3.9x
    assert _STEER_EFFECT / _T1_BATCHED_LOGPROB_BAND > 10.0            # ~15x
