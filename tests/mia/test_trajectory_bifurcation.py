"""The trajectory-bifurcation blind spot in `_compare_replay_band` (task D9).

task-D8-report.md diagnosed `hs_small routing-identity-w33-distinct-replay` req25
(LSF 1726855): the FULL-cudagraph arm's compiled kernels moved a near-tie greedy step's
logprob by 1.15e-02 nats (a step whose top probability was only ~9%), flipping the argmax
from decode index 3 onward. `prompt_len=36` predicted the first affected row at 39; the
measured first bad row WAS 39, and rows 0-38 sat at the ~1-ULP floor (8.1e-04). MIA's
capture was proven correct -- the third arm, MIA's own baked-op capture, was bit-exact to
the forward-hook path on this same request.

THE GAP: `_compare_replay_band` never looks at `generation.safetensors`, so with a fixed
`max_tokens` a bifurcation keeps row counts equal and cannot surface as the shape mismatch
the suite otherwise relies on -- it surfaces as an unabsorbable magnitude blowup on the rows
that fed the divergent token, indistinguishable at a glance from real tensor corruption.

`_compare_replay_band_checked` (`tests/mia/parity/t2_invariants.py`) is the fix: it reads
generation FIRST, reports a bifurcation explicitly, and compares captured tensors only for
the rows before the divergence. These tests pin the three behaviours the task specified:

  * a synthetic bifurcation is detected and reported at the right row, and does NOT by
    itself fail the leg (it is a published, non-fatal FULL-graph limitation);
  * corruption in the rows BEFORE the divergence still FAILS -- a bifurcation only excuses
    the rows it cannot explain;
  * a bifurcation with NO explanatory logprob delta FAILS -- greedy sampling from
    bit-identical logits cannot produce two different argmaxes, so an unexplained flip is a
    different bug, not a tie-flip, and must not be waved through.
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
    _BIFURCATION_LOGPROB_FLOOR,
    _REPLAY_REL_BAND,
    _compare_replay_band,
    _compare_replay_band_checked,
)

# A near-tie flip on the scale task-D8-report.md measured (1.15e-02 nats, LSF 1726855) --
# comfortably above `_BIFURCATION_LOGPROB_FLOOR` (reused `_STEER_ATOL`, 1e-5) and comfortably
# below a real steering intervention (~7.7e-01), the same headroom `_T0_GRAPH_LOGPROB_BAND`
# already stands on.
_MEASURED_TIE_FLIP_DELTA = 1.15e-02

_PROMPT_LEN = 5
_N_GEN = 6                          # so total_rows = 5 + 6 - 1 = 10
_TOTAL_ROWS = _PROMPT_LEN + _N_GEN - 1
_DIVERGE_IDX = 3                    # decode index of the first differing token
_DIVERGE_ROW = _PROMPT_LEN + _DIVERGE_IDX   # == 8, matching prompt_len + divergence_index


def _generation(root: Path, req: str, *, token_ids: list[int],
               token_logprobs: list[float] | None = None) -> None:
    d = root / req
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_token_ids": torch.arange(1, _PROMPT_LEN + 1, dtype=torch.int64),
        "token_ids": torch.tensor(token_ids, dtype=torch.int64),
    }
    if token_logprobs is not None:
        lp = torch.tensor(token_logprobs, dtype=torch.float64)
        payload["token_logprobs"] = lp
        payload["cumulative_logprob"] = lp.sum().to(torch.float64)
    save_file(payload, str(d / "generation.safetensors"))


def _layer(root: Path, req: str, hidden: torch.Tensor, layer_num: int = 1) -> None:
    d = root / req
    d.mkdir(parents=True, exist_ok=True)
    save_file({"hidden_states": hidden.contiguous(),
               "layer_num": torch.tensor(layer_num, dtype=torch.int64)},
              str(d / f"layer{layer_num:03d}.safetensors"))


def _hidden(seed: int, width: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(_TOTAL_ROWS, width, generator=g, dtype=torch.float32).to(torch.float16)


def _bifurcated_pair(tmp_path: Path, *, corrupt_pre_row: int | None = None,
                     explanatory_delta: float | None = _MEASURED_TIE_FLIP_DELTA):
    """Two trees whose req0 trajectory diverges at `_DIVERGE_IDX`.

    Rows before `_DIVERGE_ROW` are bit-identical (the shared prefix); rows from
    `_DIVERGE_ROW` on are wildly different (a different token's activations), which would
    fail the band outright if compared -- the whole point of the cutoff is that they must
    NOT be compared. `corrupt_pre_row`, if given, additionally breaks agreement on a row
    strictly BEFORE the divergence, which must still fail regardless of the bifurcation.
    """
    a_ids = [10, 11, 12, 13, 14, 15]
    b_ids = list(a_ids)
    b_ids[_DIVERGE_IDX] = 99  # a different sampled token from decode index 3 on
    lp_a = [-1.0] * _N_GEN
    lp_b = list(lp_a)
    if explanatory_delta is not None:
        lp_b[_DIVERGE_IDX] = lp_a[_DIVERGE_IDX] - explanatory_delta

    a, b = tmp_path / "a", tmp_path / "b"
    _generation(a, "req0", token_ids=a_ids, token_logprobs=lp_a)
    _generation(b, "req0", token_ids=b_ids,
               token_logprobs=(lp_b if explanatory_delta is not None else None))

    hidden = _hidden(seed=0)
    hidden_b = hidden.clone()
    # Post-divergence rows hold a DIFFERENT token's activation on side B -- large, not a
    # rounding difference -- exactly what the diagnosis found (rel 4.53e-01 at layer 1).
    hidden_b[_DIVERGE_ROW:] = hidden_b[_DIVERGE_ROW:] * 5.0 + 3.0
    if corrupt_pre_row is not None:
        assert corrupt_pre_row < _DIVERGE_ROW, "fixture bug: this must be a PRE-divergence row"
        hidden_b[corrupt_pre_row] = hidden_b[corrupt_pre_row] * 5.0 + 3.0

    _layer(a, "req0", hidden)
    _layer(b, "req0", hidden_b)
    return a, b


def _verdict(a: Path, b: Path) -> bool:
    verdicts: list = []
    _compare_replay_band_checked("unit bifurcation", a, b, name="layer*.safetensors",
                                 verdicts=verdicts)
    return verdicts[0][1]


# ---------------------------------------------------------------------------
# 1. A synthetic bifurcation is detected and reported at the right row.
# ---------------------------------------------------------------------------

def test_bifurcation_is_detected_and_reported_at_the_right_row(tmp_path, capsys):
    a, b = _bifurcated_pair(tmp_path)
    assert _verdict(a, b), "an EXPLAINED bifurcation (published FULL-graph limitation) must not by itself fail the leg"
    out = capsys.readouterr().out
    assert "BIFURCATION" in out
    assert "req0" in out
    assert f"decode index {_DIVERGE_IDX}" in out
    assert f"row {_DIVERGE_ROW}" in out
    assert "token 13 (side A) vs 99 (side B)" in out
    assert f"|d(logprob)|={_MEASURED_TIE_FLIP_DELTA:.3e}" in out
    # The verdict line must say what was and was not compared.
    assert "VERDICT T2 unit bifurcation: PASS" in out
    assert "1 request(s) bifurcated" in out
    assert f"{_DIVERGE_ROW} pre-divergence rows compared" in out
    assert f"{_TOTAL_ROWS - _DIVERGE_ROW} post-divergence rows skipped" in out


def test_a_non_diverging_pair_is_judged_exactly_like_the_unchecked_gate(tmp_path):
    """No generation.safetensors divergence -> the wrapper must not change the verdict of a
    plain `_compare_replay_band` call (an off-by-one-row corruption, no bifurcation at all)."""
    a, b = tmp_path / "a", tmp_path / "b"
    ids = [10, 11, 12, 13, 14, 15]
    _generation(a, "req0", token_ids=ids)
    _generation(b, "req0", token_ids=ids)
    hidden = _hidden(seed=1)
    corrupt = torch.roll(hidden, shifts=1, dims=0)
    _layer(a, "req0", hidden)
    _layer(b, "req0", corrupt)

    verdicts_checked: list = []
    _compare_replay_band_checked("unit checked", a, b, name="layer*.safetensors",
                                 verdicts=verdicts_checked)
    verdicts_plain: list = []
    _compare_replay_band("unit plain", a, b, name="layer*.safetensors",
                         verdicts=verdicts_plain)
    assert verdicts_checked[0][1] == verdicts_plain[0][1] == False


# ---------------------------------------------------------------------------
# 2. Pre-divergence corruption still FAILS.
# ---------------------------------------------------------------------------

def test_pre_divergence_corruption_still_fails(tmp_path, capsys):
    """A bifurcation only excuses the rows it cannot explain. Row 2 (< `_DIVERGE_ROW`) is
    part of the shared prefix that consumed the SAME tokens on both arms -- corrupting it
    must fail exactly as if no divergence had ever happened."""
    a, b = _bifurcated_pair(tmp_path, corrupt_pre_row=2)
    assert not _verdict(a, b), "corruption strictly before the divergence row slipped through"
    out = capsys.readouterr().out
    assert "BIFURCATION" in out          # still reported, not silently reclassified
    assert "VERDICT T2 unit bifurcation: FAIL" in out


def test_the_pre_divergence_corruption_fixture_actually_exceeds_the_band(tmp_path):
    """Precondition check: the corruption in the test above must be large enough to trip
    `_REPLAY_REL_BAND` on its own, or the test above would prove nothing."""
    a, b = _bifurcated_pair(tmp_path, corrupt_pre_row=2)
    from safetensors.torch import load_file
    x = load_file(str(a / "req0" / "layer001.safetensors"))["hidden_states"].float()
    y = load_file(str(b / "req0" / "layer001.safetensors"))["hidden_states"].float()
    rel = (x[2] - y[2]).abs().max().item() / x.abs().max().item()
    assert rel > _REPLAY_REL_BAND, f"fixture corruption too small to test anything: {rel:.3e}"


# ---------------------------------------------------------------------------
# 3. A bifurcation with NO explanatory logprob delta FAILS.
# ---------------------------------------------------------------------------

def test_bifurcation_without_a_logprob_delta_fails(tmp_path, capsys):
    """Same token-id divergence, but the two arms' logprobs at that step are IDENTICAL
    (delta exactly 0.0, at/under `_BIFURCATION_LOGPROB_FLOOR`). Greedy sampling from
    bit-identical logits cannot produce two different argmaxes, so this must be treated as a
    different bug, not a tie-flip -- and it must FAIL, not pass as a documented limitation."""
    a, b = _bifurcated_pair(tmp_path, explanatory_delta=0.0)
    assert not _verdict(a, b), "an unexplained argmax flip (zero logprob delta) was waved through"
    out = capsys.readouterr().out
    assert "does not clear the bit-exact floor" in out
    assert "FATAL" in out
    assert "VERDICT T2 unit bifurcation: FAIL" in out


def test_bifurcation_with_no_logprob_channel_at_all_fails(tmp_path, capsys):
    """The other half of 'unexplained': no `token_logprobs` channel exists to judge the flip
    by at all (e.g. a workload that never captured logprobs). This must also fail loudly
    rather than defaulting to "no evidence, so assume it is fine"."""
    a, b = _bifurcated_pair(tmp_path, explanatory_delta=None)
    assert not _verdict(a, b), "a bifurcation with no logprob channel was waved through"
    out = capsys.readouterr().out
    assert "no token_logprobs channel exists to explain it" in out
    assert "FATAL" in out


def test_the_floor_sits_strictly_between_zero_and_the_measured_tie_flip():
    """The discrimination the floor exists for: it must reject an exact-zero delta (no
    numeric cause at all) while accepting the scale of a real, measured tie-flip."""
    assert 0.0 < _BIFURCATION_LOGPROB_FLOOR < _MEASURED_TIE_FLIP_DELTA
    assert _MEASURED_TIE_FLIP_DELTA / _BIFURCATION_LOGPROB_FLOOR > 100.0


# ---------------------------------------------------------------------------
# ...and the legs that NEED the checked comparator must actually use it
# ---------------------------------------------------------------------------
# D9 built `_compare_replay_band_checked` and pointed the three legs that replay a graph at
# it. The whole-branch review then found a fourth family that needs it and did not have it:
# the two `gpu-routing` band gates, whose own companion generation gate records the two arms
# generating DIFFERENTLY on 33/33 requests at 1.98e-02..2.14e-02 -- larger than the 1.15e-02
# delta that produced the one tie-flip this suite has ever observed.
#
# It could not cause a false PASS (the companion gate holds token ids fatal), which is
# exactly why a value-based test cannot catch it. The cost is diagnostic: a tie-flip there
# would present as unexplained tensor corruption and cost the D8-style bug hunt D9 closed.
# So the check is structural, and it is on the CALL, not on the source text.


def _band_calls(func) -> list:
    """[(label fragment, callee name)] for every `_compare_replay_band*` call in `func`."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(func))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.id if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", None))
        if name not in ("_compare_replay_band", "_compare_replay_band_checked"):
            continue
        label = node.args[0] if node.args else None
        pieces = [c.value for c in ast.walk(label)
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)] if label else []
        out.append(("".join(pieces), name))
    return out


def test_the_gpu_routing_band_gates_are_bifurcation_aware():
    """Both `gpu-routing` band gates must go through the CHECKED comparator."""
    import tests.mia.parity.t2_invariants as t2mod

    calls = _band_calls(t2mod.judge_capture)
    gpu = [(label, callee) for label, callee in calls if "gpu-routing" in label]
    assert len(gpu) == 2, f"expected 2 gpu-routing band calls, found {gpu}"
    for label, callee in gpu:
        assert callee == "_compare_replay_band_checked", (
            f"{label!r} uses {callee}, which never reads generation.safetensors. These two "
            f"legs' arms generate differently on 33/33 requests, so a greedy tie-flip there "
            f"would present as unexplained tensor corruption instead of being named.")


# Which comparator each band gate is SUPPOSED to use, and why. Declared rather than inferred,
# so adding a band call forces a decision instead of inheriting whichever default was handy.
#   CHECKED -- the two arms can generate DIFFERENT tokens, so a divergence must be named
#              before any tensor is compared;
#   PLAIN   -- the two arms share one forward (they differ only in which capture MECHANISM
#              reads it), so generations are identical by construction and there is nothing
#              to detect. Keeping these plain is what makes the CHECKED assertion meaningful.
_EXPECTED_COMPARATOR = {
    "routing-identity-w33-distinct-replay [BANDED]": "checked",   # real graph replay
    "replay-band-w32": "checked",                                 # real graph replay
    "replay-band-w1": "checked",                                  # real graph replay
    "gpu-routing vs host-routing [BANDED]": "checked",            # generation differs 33/33
    "gpu-routing vs host-routing, no tail wave [BANDED]": "checked",
    "routing-identity-w32 [BANDED]": "plain",                     # one forward, two readers
    "routing-identity-w33-distinct [BANDED]": "plain",            # one forward, two readers
    "vs default [BANDED]": "plain",                               # builder equivalence
}


def test_every_band_gate_uses_the_comparator_its_arms_require():
    """The converse of the test above, so this never becomes 'use the checked one everywhere'.

    An exact table: every band call is classified, and a new one fails here until someone
    decides which kind it is. That decision is the whole content of the D9 fix -- a leg whose
    arms can bifurcate needs generation read first; a leg whose arms share a forward does not.
    """
    import tests.mia.parity.t2_invariants as t2mod

    actual = {label.strip(): callee for label, callee in _band_calls(t2mod.judge_capture)}
    assert set(actual) == set(_EXPECTED_COMPARATOR), (
        f"band gates changed: only-in-code={sorted(set(actual) - set(_EXPECTED_COMPARATOR))}, "
        f"only-in-table={sorted(set(_EXPECTED_COMPARATOR) - set(actual))}. Classify the new "
        f"one: can its two arms generate different tokens?")
    for label, kind in sorted(_EXPECTED_COMPARATOR.items()):
        want = "_compare_replay_band_checked" if kind == "checked" else "_compare_replay_band"
        assert actual[label] == want, (
            f"{label!r} is declared {kind} but calls {actual[label]}")

