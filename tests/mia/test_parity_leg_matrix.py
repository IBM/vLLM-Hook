"""The leg matrix must contain the combinations the gates claim to cover.

Task D5 items 6 and 7. The structural review's finding was not that a gate was wrong but
that a CONFIGURATION was missing, so no gate could see it:

  * the distinct-prompt-length legs ran `cudagraph=False`, so no leg combined distinct
    lengths with REAL capture+replay -- the same-shape hole was closed exactly where padded
    rows do not exist and left open exactly where they do;
  * every steer leg ran at `max_num_seqs=1`, so a steer mis-route at width > 1 was
    unmeasured.

A missing configuration cannot be caught by any assertion about numbers, so it is asserted
about the matrix itself. These tests are cheap and they fail the moment someone drops the
`cudagraph=True` from the distinct leg or the width from the steer legs -- which is how both
holes opened in the first place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.capture_workload import DISTINCT_PROMPTS  # noqa: E402
from tests.mia.parity.t2_invariants import EXPECTED_GATES, LEGS  # noqa: E402

_CAPTURE_WORKLOADS = ("hs_small", "qk_small")


@pytest.mark.parametrize("workload", _CAPTURE_WORKLOADS)
def test_a_leg_runs_distinct_lengths_under_real_replay(workload):
    """The hole item 6 closes: distinct prompt lengths AND a real CUDA-graph replay.

    `graph=True` arms MIA's baked op and aperture; `cudagraph=True` is what makes vLLM
    compile the model and replay graphs, which is the only configuration in which padded
    rows exist at all. Without both on the same leg, replay-time routing is only ever
    checked against a batch whose only same-shape pairs are same-prompt replicas -- and a
    mis-route between two replicas measures under every band in the suite.
    """
    combined = [name for name, spec in LEGS[workload].items()
                if spec["graph"] and spec["cudagraph"]
                and spec["env"].get("MIA_PARITY_PROMPTS") == "distinct"]
    assert combined, (
        f"{workload}: no leg combines distinct prompt lengths with real capture+replay "
        f"(graph=True, cudagraph=True). The same-shape blind spot is then closed only "
        f"where padded rows do not exist.")


@pytest.mark.parametrize("workload", _CAPTURE_WORKLOADS)
def test_the_distinct_replay_leg_is_gated(workload):
    """A leg that runs and is not adjudicated is an engine boot, not evidence."""
    assert f"{workload} routing-identity-w33-distinct-replay [BANDED]" in \
        EXPECTED_GATES[workload]


@pytest.mark.parametrize("workload", _CAPTURE_WORKLOADS)
def test_the_distinct_legs_are_wide_enough_to_hold_every_prompt(workload):
    """33 prompts in 33 slots. One slot short and the last request forms a SECOND prefill
    wave, whose padded graph size differs -- which is what made the earlier gpu-routing legs
    diverge on their tail requests. The point of the distinct set is that all 33 share the
    step."""
    for name, spec in LEGS[workload].items():
        if spec["env"].get("MIA_PARITY_PROMPTS") != "distinct":
            continue
        assert spec["seqs"] >= len(DISTINCT_PROMPTS), (
            f"{workload}/{name}: max_num_seqs={spec['seqs']} cannot hold "
            f"{len(DISTINCT_PROMPTS)} distinct-length requests in one wave")


def test_a_steer_leg_runs_above_width_one():
    """The hole item 7 closes. Every steer gate used to run at max_num_seqs=1, so the steer
    routing plane -- which is per-token-column and therefore per-request -- was never
    checked at a width where it can put a request's steering on its neighbour."""
    wide = [name for name, spec in LEGS["steer_small"].items() if spec["seqs"] > 1]
    assert wide, "no steer leg runs above max_num_seqs=1"


def test_the_width_steer_gate_is_declared():
    assert "steer_small width [BANDED]" in EXPECTED_GATES["steer_small"]


def test_a_steer_leg_arms_only_some_requests_in_one_batch():
    """The discriminator itself: steering ALL requests in a batch with the SAME config
    cannot distinguish "each request got its own steering" from "each request got its
    neighbour's" -- they are the same intervention. Arming only some of them can: an
    unarmed request that moves has received a neighbour's steer.
    """
    mixed = [name for name, spec in LEGS["steer_small"].items()
             if spec["env"].get("MIA_PARITY_STEER_ARM") == "alternate"]
    assert mixed, (
        "no steer leg arms only part of its batch, so no gate can tell a correct per-request "
        "steer from a mis-routed one at width > 1")
    for name in mixed:
        spec = LEGS["steer_small"][name]
        assert spec["seqs"] > 1, f"{name} is mixed-armed but runs at width 1"
        assert spec["graph"] and spec["cudagraph"], (
            f"{name} must run under real capture + replay: a stale steer config surviving "
            f"into a real step is only possible where vLLM replays the compiled graph")
        assert spec["env"].get("MIA_PARITY_PROMPTS") == "distinct", (
            f"{name} must use the distinct-length prompts so each request is individually "
            f"identifiable rather than a replica of its neighbour")


def test_the_per_request_arming_gate_is_declared():
    for label in ("steer_small per-request-arming",
                  "steer_small INFO per-request-arming bit-exact"):
        assert label in EXPECTED_GATES["steer_small"]
