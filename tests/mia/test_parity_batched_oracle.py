"""The batched oracle's row segmentation, pinned without a GPU.

Task D5 item 8, and the deepest finding of the structural review: every T2 gate is
MIA-vs-MIA. Both arms import `StepView` / `step_view` from `mia/runner.py`, which IS the
V1->V2 port surface, so a bug there moves both arms identically and every T2 gate still
passes. The only oracle outside MIA ran ONE request, eager -- so batching, the thing the
port actually changed, was never checked against anything but MIA itself.

`t1_reference.reference_capture_batched` closes that by running 33 distinct-length requests
in one batch. The part that makes it an ORACLE rather than a second copy of the code under
test is `segment_batch_rows`: it decides which rows of a batched forward pass belong to
which request from the TOKEN IDS the model was fed, using no `query_start_loc`, no
`idx_mapping` and nothing from `mia/runner.py`. That function is pure, so it is pinned here
-- including the ways it must REFUSE rather than guess, because a silently wrong
segmentation would compare the wrong rows and report bit-exact agreement about nothing.

THE IRREDUCIBLE LIMIT, asserted as documentation rather than behaviour at the end of this
file: a Python forward hook cannot observe a CUDA-graph replay (a replay is a graph launch;
no Python runs), so an independent oracle for graph mode is not constructible this way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.capture_workload import (  # noqa: E402
    BATCH_ORACLE_PROMPTS,
    WORKLOADS,
    assert_batch_oracle_prompts,
)
from tests.mia.parity.t1_reference import segment_batch_rows  # noqa: E402


class _FakeTokenizer:
    """Tokenizes on whitespace with a stable per-word id. Enough for the two properties."""

    def __call__(self, text):
        return {"input_ids": [abs(hash(w)) % 50000 + 1 for w in text.split()]}


def _ids(*lengths, base=1000):
    """One token-id list per request, each with a UNIQUE first token."""
    return [[base + 100 * i] + [base + 100 * i + 1 + j for j in range(n - 1)]
            for i, n in enumerate(lengths)]


def test_segments_a_single_prefill_wave_in_arrival_order():
    prompts = _ids(3, 5, 2)
    flat = [t for p in prompts for t in p]
    assert segment_batch_rows(flat, prompts) == {0: (0, 3), 1: (3, 8), 2: (8, 10)}


def test_segments_a_wave_the_scheduler_reordered():
    """The oracle must not assume arrival order -- it DISCOVERS the order, which is half the
    point: assuming the layout is what the code under test does."""
    prompts = _ids(3, 5, 2)
    flat = prompts[2] + prompts[0] + prompts[1]
    assert segment_batch_rows(flat, prompts) == {2: (0, 2), 0: (2, 5), 1: (5, 10)}


def test_skips_decode_rows_and_padding_between_prefills():
    """A pass can mix decode rows (one token per running request) and padding with prefill.
    Those rows belong to no prompt and are stepped over one at a time."""
    prompts = _ids(3, 4)
    flat = [7] + prompts[1] + [9, 9] + prompts[0] + [0, 0]
    assert segment_batch_rows(flat, prompts) == {1: (1, 5), 0: (7, 10)}


def test_a_request_absent_from_the_pass_is_simply_not_placed():
    """Segmentation reports what it FOUND. The caller is what refuses to compare a subset --
    `reference_capture_batched` raises when any request went unplaced across all passes."""
    prompts = _ids(3, 4)
    assert segment_batch_rows(prompts[0], prompts) == {0: (0, 3)}


def test_a_prompt_is_never_matched_twice():
    """Two requests with identical prompts would be indistinguishable; the batched oracle's
    prompt set forbids that, and the walk places each request at most once regardless."""
    prompts = _ids(3)
    found = segment_batch_rows(prompts[0] + prompts[0], prompts)
    assert found == {0: (0, 3)}


def test_a_truncated_trailing_prompt_is_not_matched():
    """A chunked prefill splits a request's prompt across passes. A partial match must NOT
    be accepted as the whole request -- it would silently compare too few rows."""
    prompts = _ids(5)
    assert segment_batch_rows(prompts[0][:3], prompts) == {}


def test_shared_first_tokens_are_REFUSED_not_guessed():
    """THE AMBIGUITY THIS ORACLE CANNOT TOLERATE. With the `DISTINCT_PROMPTS` set every
    prompt is a prefix of the next, so a greedy walk could place the wrong request at a
    boundary and report bit-exact agreement about the wrong rows. Rather than guess, the
    segmentation refuses -- and `assert_batch_oracle_prompts` is what makes sure it never
    has to."""
    prompts = [[500, 1, 2], [500, 1, 2, 3]]
    with pytest.raises(RuntimeError, match="share first token"):
        segment_batch_rows([500, 1, 2, 3], prompts)


# ---------------------------------------------------------------------------
# The prompt set the oracle rests on
# ---------------------------------------------------------------------------

def test_batch_oracle_prompt_set_is_the_right_size_and_all_different():
    assert len(BATCH_ORACLE_PROMPTS) == 33
    assert len(set(BATCH_ORACLE_PROMPTS)) == 33


def test_batch_oracle_assertion_rejects_colliding_lengths():
    """Enforced against the tokenizer in use, not assumed from word counts: a different
    model would re-tokenize these strings and the solved lengths would not hold."""
    with pytest.raises(RuntimeError, match="DIFFERENT prompt token counts"):
        assert_batch_oracle_prompts(_FakeTokenizer(), ["alpha beta", "gamma delta"])


def test_batch_oracle_assertion_rejects_colliding_first_tokens():
    with pytest.raises(RuntimeError, match="DIFFERENT first tokens"):
        assert_batch_oracle_prompts(_FakeTokenizer(), ["alpha beta", "alpha beta gamma"])


def test_batch_oracle_assertion_accepts_a_well_formed_set():
    ids = assert_batch_oracle_prompts(_FakeTokenizer(), ["alpha beta", "gamma delta epsilon"])
    assert [len(i) for i in ids] == [2, 3]


@pytest.mark.parametrize("name", ("hs_batch", "qk_batch"))
def test_the_batched_workloads_capture_prefill_only(name):
    """A forward hook sees one flat tensor per pass and nothing that says which rows belong
    to whom; token-id matching recovers that EXACTLY for the prefill wave, where each
    request contributes a contiguous run equal to its own prompt. Decode rows are one token
    per running request and are not identifiable that way, so they are not claimed."""
    wl = WORKLOADS[name]
    assert wl.hooks_on == "prefill"
    assert wl.prompts == BATCH_ORACLE_PROMPTS
    assert wl.granularity == "all_tokens"


@pytest.mark.parametrize("name", ("hs_batch", "qk_batch"))
def test_the_batched_workloads_reuse_their_small_twins_layers(name):
    """Same physical blocks as the single-request legs, in each subsystem's own numbering
    (HS 1-based, QK 0-based) -- so a failure here is about batching and nothing else."""
    twin = WORKLOADS[name.replace("_batch", "_small")]
    assert WORKLOADS[name].layers == twin.layers
    assert WORKLOADS[name].subsystem == twin.subsystem


# ---------------------------------------------------------------------------
# The irreducible limit, kept in the repository rather than in a commit message
# ---------------------------------------------------------------------------

def test_the_graph_mode_limit_is_stated_in_the_code_and_in_tolerances():
    """A Python forward hook CANNOT observe a CUDA-graph replay: a replay is a graph launch
    and no Python runs during it. So an independent oracle for GRAPH mode is not
    constructible this way, and graph-mode correctness still rests on MIA-vs-MIA plus the
    replay band. The task brief asked for that to be said plainly in the code AND in
    TOLERANCES.md rather than pretended otherwise; this test is what keeps it said."""
    ref = (REPO_ROOT / "tests" / "mia" / "parity" / "t1_reference.py").read_text()
    tol = (REPO_ROOT / "tests" / "mia" / "parity" / "TOLERANCES.md").read_text()
    assert "CANNOT OBSERVE A CUDA-GRAPH REPLAY" in ref
    assert "cannot observe a CUDA-graph replay" in tol
    assert "not constructible this way" in ref and "not constructible this way" in tol
    assert "MIA-vs-MIA" in tol
