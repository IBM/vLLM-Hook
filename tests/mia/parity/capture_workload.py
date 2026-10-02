"""One workload definition, used by every parity tier, so comparisons are apples-to-apples.

Deliberately small and deterministic: greedy sampling, fixed prompts, few layers,
ignore_eos so every request decodes the same number of steps regardless of content.

The module has two halves, and the split is load-bearing:

* Everything from ``PROMPTS`` down to ``write_request_artifacts`` is ARM-NEUTRAL. It names
  no MIA symbol and reads no MIA environment variable, so a driver written against the
  PRE-rename package (task A6's throwaway ``legacy_run.py``, which lives only in a worktree
  of ``main`` and is never committed) loads this same file by path and reuses it verbatim.
  That is what makes the A/B a controlled experiment: only the plugin under test differs --
  not the prompts, not the request-level arguments, not the serialization.
* ``run_workload`` is the MIA-side driver and is the interface the later parity tiers
  (C4 cross-version, D4 alone-vs-batch, E1) call.

Artifact layout is ``<out_dir>/req<N>/<name>.safetensors`` -- the per-request directory
level is part of the contract, because D4 compares one request captured alone against the
same request captured inside a wider batch and indexes it as ``<out_dir>/req0``.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# DO NOT ADD `sys.path.insert(0, str(REPO_ROOT))` HERE.
# ---------------------------------------------------------------------------
# Every other driver in this directory has that line (`t1_reference.py`, `t2_invariants.py`,
# `t3_crossbranch.py`), so adding it here "for consistency" looks like a tidy-up. It is not.
#
# T3's 0.21 arms (`tests/mia/parity/run_crossbranch.sh`) run THIS file against a DIFFERENT `mia`
# package -- a git worktree pinned at the Phase-A SHA -- because MIA at HEAD refuses vLLM
# 0.21's V1 runner. They arrange that by running the file by absolute path (so `sys.path[0]`
# is this directory, which holds no `mia`) with `PYTHONPATH=<shadow>:<repo>`. Inserting
# REPO_ROOT at position 0 would put the LIVE `mia` ahead of the pinned one and the old-world
# arm would silently become a second new-world arm.
#
# The failure would be LOUD, not silent -- HEAD's `mia/runner.py::require_v2_runner` raises
# on a V1 runner, so the arm would crash rather than emit wrong numbers, and T3 fails a
# crashed arm's gates instead of dropping them. But loud is not free, and the arm is the only
# definition of "the old world" T3 has. `tests/test_parity_t3_crossbranch.py::
# test_capture_workload_has_no_sys_path_insert` fails if this ever changes, and the arm
# itself re-checks the resolved package in-process (see `_assert_mia_provenance`).
# ---------------------------------------------------------------------------

# Set by run_crossbranch.sh's 0.21 arms to the directory the pinned `mia` MUST come from.
# Unset (the normal case) means "wherever the environment puts it" and nothing is enforced.
REQUIRE_MIA_UNDER = "MIA_PARITY_REQUIRE_MIA_UNDER"


def _assert_mia_provenance(module) -> str:
    """Print which `mia` this process actually imported, and enforce it when asked.

    The job script probes the shadow in a SEPARATE process before booting anything, which
    proves the resolution for that process's `sys.path`, not for the engine's. This runs
    inside the arm, so the log records what was actually loaded, and with
    ``MIA_PARITY_REQUIRE_MIA_UNDER`` set it refuses to continue when the answer is wrong --
    a cross-version comparison against the wrong old world is worse than no comparison.

    SYMLINKS ARE NOT A MISMATCH. The pinned package is reached through a directory holding a
    single symlink to a git worktree's `mia/`, so `__file__` and its `resolve()`d target are
    two different real paths and BOTH are the right answer. Comparing only the resolved form
    against only the literal root rejects exactly the arrangement this check exists to
    approve, so both forms of the path are matched against both forms of the root.
    """
    raw = getattr(module, "__file__", "") or "<namespace package>"
    print(f"[workload] mia imported from {raw}", flush=True)
    required = os.environ.get(REQUIRE_MIA_UNDER)
    if required:
        candidates = {os.path.abspath(raw)}
        roots = {os.path.abspath(required)}
        try:                                   # a broken symlink must not mask the check
            candidates.add(str(Path(raw).resolve()))
            roots.add(str(Path(required).resolve()))
        except OSError:
            pass
        under = any(c == r or c.startswith(r.rstrip("/") + "/")
                    for c in candidates for r in roots)
        if not under:
            raise RuntimeError(
                f"{REQUIRE_MIA_UNDER}={required} demands the `mia` package come from there "
                f"(or its symlink target), but this process imported {raw} "
                f"(candidates {sorted(candidates)} against roots {sorted(roots)}). Something "
                f"put another `mia` ahead of it on sys.path -- refusing to run, because an "
                f"arm that silently swaps which MIA it is testing produces numbers that look "
                f"valid and mean nothing.")
        print(f"[workload] mia provenance ENFORCED: under {sorted(roots)}", flush=True)
    return raw


@dataclasses.dataclass(frozen=True)
class Workload:
    subsystem: str            # "hs" | "qk" | "steer"
    prompts: tuple[str, ...]
    layers: tuple[int, ...]
    max_tokens: int
    hooks_on: str             # "prefill" | "decode" | "both"
    granularity: str = "all_tokens"


PROMPTS = (
    "The capital of France is",
    "In a shocking finding, scientists discovered that",
    "def fibonacci(n):",
)

# Layer numbering is PER SUBSYSTEM and is not a free choice. The hidden-state path reports
# and filters on a 1-based layer number (PyTorch module index + 1); the QK path filters on
# the 0-based module index. So hs_small's (1, 16, 32) and qk_small's (0, 15, 31) name the
# SAME three physical blocks of a 32-layer model. Asking the HS path for "layer 0" would
# match nothing and produce an empty artifact tree -- which a naive A/B would read as a
# trivially identical PASS. That false pass is precisely what this gate exists to catch, so
# the numbering is written out rather than shared between the two workloads.
WORKLOADS = {
    "hs_small":    Workload("hs",    PROMPTS, (1, 16, 32), 16, "both"),
    "qk_small":    Workload("qk",    PROMPTS, (0, 15, 31), 16, "both"),
    "steer_small": Workload("steer", PROMPTS, (15,),       16, "decode"),
}

MODEL = os.environ.get("MIA_PARITY_MODEL", "microsoft/Phi-3-mini-4k-instruct")

# Absolute, and resolved against THIS file's repository, so both arms of a cross-branch A/B
# load the very same bytes instead of each arm's own checkout of the vector.
STEER_VECTOR = REPO_ROOT / "steering_vectors" / "phi3_adjust_rs_test.pt"

# Engine arguments are part of the workload: a parity comparison only means something if
# both sides boot the same engine. Prefix caching off + greedy + a fixed seed keep
# generation reproducible; eager-vs-graph is the one axis a caller varies.
ENGINE_KWARGS: dict[str, Any] = dict(
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.60,
    enable_prefix_caching=False,
    tensor_parallel_size=1,
    trust_remote_code=True,
    seed=0,
)

# The running-batch width is the one engine argument a tier may need to pin, and it is
# pinned through the environment so BOTH arms of a comparison read the same value from
# this same dict. T1 (task C4) sets it to 1: with one request resident at a time every
# forward pass holds exactly one request's rows, so T1's independent oracle needs no
# batch-row arithmetic and therefore shares none of the index logic it is checking.
# Unset -> vLLM's own default, leaving A6's tree and D4's alone-vs-batch untouched.
_MAX_NUM_SEQS = os.environ.get("MIA_PARITY_MAX_NUM_SEQS")
if _MAX_NUM_SEQS:
    ENGINE_KWARGS["max_num_seqs"] = int(_MAX_NUM_SEQS)


# ---------------------------------------------------------------------------
# Arm-neutral: request construction
# ---------------------------------------------------------------------------

def steer_config(wl: Workload) -> dict:
    """The per-request steering config for a steer workload."""
    return {
        "method": "adjust_rs",
        "coefficient": 1.0,
        "optimal_layer": list(wl.layers),
        "vector_path": str(STEER_VECTOR),
        "phase": wl.hooks_on,
        "positions": "all_tokens",
    }


def steer_fingerprint(cfg: dict) -> str:
    """A printable identity for one steering intervention, including the vector's BYTES.

    Both arms of the steer comparison print this line. Two matching fingerprints in one log
    are the evidence that the arms applied the SAME vector at the SAME layer with the SAME
    coefficient and method -- rather than two different interventions that were never
    comparable. The digest is over the file contents, so two checkouts holding different
    bytes under the same filename are caught.
    """
    import hashlib

    path = Path(cfg["vector_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return (f"method={cfg.get('method')} coefficient={cfg.get('coefficient')} "
            f"optimal_layer={cfg.get('optimal_layer')} phase={cfg.get('phase')} "
            f"positions={cfg.get('positions')} vector={path.name} sha256:{digest}")


def extra_args_for(wl: Workload) -> dict:
    """The ``SamplingParams.extra_args`` that arm this workload's subsystem.

    These keys are the plugin's REQUEST-level API and were deliberately not renamed, so one
    dict drives both arms of the rename A/B.
    """
    if wl.subsystem == "hs":
        return {"output_hidden_states": list(wl.layers),
                "hs_mode": wl.granularity,
                "hooks_on": wl.hooks_on}
    if wl.subsystem == "qk":
        return {"output_qk": list(wl.layers),
                "hookq_mode": wl.granularity,
                "hooks_on": wl.hooks_on}
    if wl.subsystem == "steer":
        return {"steer": steer_config(wl)}
    raise KeyError(f"unknown subsystem {wl.subsystem!r}")


def sampling_params_for(wl: Workload, extra: dict | None):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=wl.max_tokens,
        ignore_eos=True,      # every request decodes exactly max_tokens steps
        logprobs=1,           # a float channel that catches drift the token ids would hide
        extra_args=extra,
    )


# 33 prompts whose token counts are all DIFFERENT, built by growing a sentence one word at a
# time. Why this exists (task D4 fix round 2): the default workload repeats a 3-prompt set,
# so the only SAME-SHAPE request pairs in a wide batch are same-prompt replicas -- and a
# whole-request mis-route between two replicas produces a difference of only ~2e-03 relative,
# which is under any usable magnitude band. No value comparison can see that swap. With 33
# distinct lengths every whole-request mis-route changes the captured row count instead, and
# a shape mismatch is fatal with no tolerance involved. The property is ENFORCED at runtime
# by `assert_distinct_prompt_lengths` rather than assumed from the word counts, because
# tokenizers are free to merge and two different sentences could still tokenize to the same
# length.
_DISTINCT_STEM = ("data model layer token vector matrix kernel buffer stream cursor window "
                  "batch prompt decoder encoder residual attention gradient parameter "
                  "checkpoint inference throughput latency scheduler allocator partition "
                  "embedding normalizer activation projection quantizer accelerator "
                  "dispatcher").split()
# Fix round 2 (task D4): the stem above must have >= 33 words. ``range(1, 34)`` below
# builds 33 prefixes _DISTINCT_STEM[:1..33]; once ``n`` exceeds the stem's length, slicing
# silently returns the SAME full list for every larger ``n`` (Python does not error), so a
# too-short stem produces identical prompts -- not merely similar-length ones -- for every
# n beyond its length. That is exactly what happened with a 31-word stem: n=31, 32 and 33
# all sliced to the same 31 words, giving three byte-identical prompts that tokenized to
# the same 43 tokens and tripped `assert_distinct_prompt_lengths` (LSF job 1705794). The
# fix is two more words, not a smaller range or a guard change -- see that function's
# docstring for why the guard itself must stay. Verified lengths (33 distinct):
# [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 22, 23, 24, 26, 27, 29, 31, 33,
#  35, 36, 37, 39, 40, 41, 43, 45, 47]

DISTINCT_PROMPTS: tuple[str, ...] = tuple(
    "List: " + " ".join(_DISTINCT_STEM[:n]) for n in range(1, 34))


# ---------------------------------------------------------------------------
# The BATCHED-ORACLE prompt set (task D5 item 8)
# ---------------------------------------------------------------------------
# WHY A SECOND DISTINCT SET. The structural review's deepest finding: every T2 gate is
# MIA-vs-MIA. Both arms import ``StepView`` / ``step_view`` from ``mia/runner.py``, which IS
# the V1->V2 port surface, so a bug there moves both arms identically and every T2 gate
# still passes. The only independent oracle is T1 -- vanilla vLLM with ``VLLM_PLUGINS=""``
# and torch forward hooks -- and it ran ONE request, eager. So batching, the thing the port
# actually changed, was never checked against anything outside MIA.
#
# The batched oracle fixes that, and to do it the reference has to answer a question it
# previously sidestepped by running at ``max_num_seqs=1``: WHICH ROWS OF A BATCHED FORWARD
# PASS BELONG TO WHICH REQUEST. It must answer it WITHOUT MIA's routing, or the oracle is
# not independent -- so it answers it from the TOKEN IDS the model was given, matching each
# request's own prompt against the flat input of the pass
# (``t1_reference.segment_batch_rows``). That needs two properties the default set does not
# have:
#
#   * DISTINCT FIRST TOKENS, so at any row boundary at most one request can start there.
#     ``DISTINCT_PROMPTS`` above are prefixes of one another ("List: data",
#     "List: data model", ...), which makes greedy matching ambiguous. Each prompt here
#     opens with its own unique word instead.
#   * DISTINCT LENGTHS, for the same reason the `*_batch_distinct` legs need them: a
#     whole-request mis-route then changes the captured row count and lands as a SHAPE
#     mismatch, with no tolerance involved.
#
# The word counts below are not decorative: they were SOLVED against the real Phi-3
# tokenizer so that the 33 lengths come out all-different (4, 5, 6, 7, 8, 10, 11, 12, 13,
# 14, 15, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 36, 37, 38, 40, 41, 42, 44,
# 46, 48; 805 tokens total, one prefill wave). They are ENFORCED at runtime by
# ``assert_batch_oracle_prompts`` against the tokenizer actually in use, because a different
# model would re-tokenize these strings and the solution would not hold -- and a collision
# must fail loudly rather than silently make two requests interchangeable.
_BATCH_ORACLE_HEADS = ("Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo "
                       "Lima Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform "
                       "Victor Whiskey Xray Yankee Zulu Anchor Beacon Cobalt Dynamo Ember "
                       "Falcon Granite").split()
_BATCH_ORACLE_WORDS = (1, 2, 4, 4, 5, 6, 8, 10, 11, 11, 12, 14, 15, 16, 17, 18, 19, 19, 20,
                       20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)

BATCH_ORACLE_PROMPTS: tuple[str, ...] = tuple(
    f"{head}: " + " ".join(_DISTINCT_STEM[:n])
    for head, n in zip(_BATCH_ORACLE_HEADS, _BATCH_ORACLE_WORDS))


def assert_batch_oracle_prompts(tokenizer, prompts) -> list[list[int]]:
    """Enforce the two properties the batched oracle rests on. Returns the token ids.

    Both are checked against the tokenizer in use rather than assumed from the word counts:

      * DISTINCT LENGTHS -- so a whole-request mis-route changes the captured row count and
        fails as a shape mismatch, not as a magnitude a band has to catch;
      * DISTINCT FIRST TOKENS -- so the oracle can find each request's rows in a batched
        forward pass by matching token ids, with no appeal to MIA's routing. Without this
        the segmentation is ambiguous wherever one prompt is a prefix of another, and an
        ambiguous segmentation would silently compare the wrong rows.
    """
    ids = [list(tokenizer(p)["input_ids"]) for p in prompts]
    lengths = [len(i) for i in ids]
    if len(set(lengths)) != len(lengths):
        dupes = sorted({n for n in lengths if lengths.count(n) > 1})
        raise RuntimeError(
            f"the batched oracle requires {len(prompts)} DIFFERENT prompt token counts, but "
            f"these collide: {dupes} (lengths={lengths}). Two same-length requests are "
            f"interchangeable to every value gate, which is what this set exists to prevent.")
    firsts = [i[0] for i in ids]
    if len(set(firsts)) != len(firsts):
        dupes = sorted({t for t in firsts if firsts.count(t) > 1})
        raise RuntimeError(
            f"the batched oracle requires {len(prompts)} DIFFERENT first tokens so that row "
            f"segmentation is unambiguous, but these collide: {dupes}. The oracle finds each "
            f"request's rows by matching token ids -- that is what makes it independent of "
            f"MIA's routing -- and it cannot do so when two requests can start at the same "
            f"row.")
    return ids


# The BATCHED-ORACLE workloads (task D5 item 8). Same layers, same model, same engine as
# their `*_small` twins; two differences, both load-bearing:
#
#   * 33 distinct-length, distinct-first-token prompts (see BATCH_ORACLE_PROMPTS) run in ONE
#     batch, which is what makes this the first multi-row comparison against an oracle
#     outside MIA;
#   * hooks_on="prefill". A Python forward hook sees one flat [num_tokens, ...] tensor per
#     pass and nothing that says which rows belong to which request; the reference recovers
#     that from the TOKEN IDS, which works exactly for the prefill wave, where each request
#     contributes a contiguous run equal to its own prompt. Decode rows carry one token per
#     running request and are not identifiable that way, so they are not claimed: the gate
#     covers the prefill wave, which is also where multi-row-per-request routing lives
#     (query_start_loc, idx_mapping, the padded slice). Generation still runs to
#     max_tokens, so the generation comparison covers all 16 steps at width 33.
_BATCH_ORACLE_TOKENS = 16
WORKLOADS["hs_batch"] = Workload("hs", BATCH_ORACLE_PROMPTS, (1, 16, 32),
                                 _BATCH_ORACLE_TOKENS, "prefill")
WORKLOADS["qk_batch"] = Workload("qk", BATCH_ORACLE_PROMPTS, (0, 15, 31),
                                 _BATCH_ORACLE_TOKENS, "prefill")


def assert_distinct_prompt_lengths(tokenizer, prompts) -> list[int]:
    """Enforce the property the distinct-length legs rest on. Returns the token counts."""
    lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    if len(set(lengths)) != len(lengths):
        dupes = sorted({n for n in lengths if lengths.count(n) > 1})
        raise RuntimeError(
            f"the distinct-length workload requires {len(prompts)} DIFFERENT prompt token "
            f"counts, but these collide: {dupes} (lengths={lengths}). A whole-request "
            f"mis-route between two same-length requests would be invisible to every value "
            f"gate, which is the entire reason this leg exists.")
    return lengths


def steer_arm_mask(n: int) -> list[bool]:
    """Which requests of a mixed-armed batch carry a steer config. ALTERNATING, from 1.

    Task D5 item 7. Every steer gate in this suite ran at ``max_num_seqs=1``, so a steer
    mis-route at width > 1 was unmeasured. Simply widening the batch does not fix that:
    steering ALL requests with the SAME config cannot distinguish "each request received its
    own steering" from "each request received its neighbour's" -- they are the same
    intervention, and the two hypotheses predict identical logprobs.

    Arming only PART of the batch can distinguish them, and it is the sharpest available
    discriminator for the steer routing plane, which is per-token-column and therefore
    per-request (``coeff_all`` / ``vec_id_all`` / ``mode_all`` in
    ``mia/graph/install_steer.py``):

      * an ARMED request that does NOT move  -> its steering was routed away;
      * an UNARMED request that DOES move    -> it received a neighbour's steering.

    Alternating (rather than, say, the first half) puts an unarmed request on BOTH sides of
    almost every armed one, so an off-by-one in the column mapping lands on an unarmed row
    in either direction.
    """
    return [bool(i % 2) for i in range(int(n))]


def prompts_for(wl: Workload, batch: int) -> list[str]:
    """``batch`` repeats the prompt set to force a wider running batch (D4's alone-vs-batch).

    ``MIA_PARITY_PROMPTS=distinct`` swaps in ``DISTINCT_PROMPTS`` instead, ignoring ``batch``
    -- see that constant for why.
    """
    if os.environ.get("MIA_PARITY_PROMPTS") == "distinct":
        return list(DISTINCT_PROMPTS)
    return list(wl.prompts) * int(batch)


# ---------------------------------------------------------------------------
# Arm-neutral: artifact serialization
# ---------------------------------------------------------------------------

def _save(path: Path, tensors: dict) -> None:
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    # clone(): safetensors refuses tensors that share storage, and every captured tensor
    # may be a view into a shared host buffer. Cloning copies values, so it is bit-exact.
    payload = {k: v.detach().cpu().contiguous().clone()
               for k, v in tensors.items() if torch.is_tensor(v)}
    save_file(payload, str(path))


def _ascii(value: str):
    import torch
    return torch.tensor(list(str(value).encode("ascii", "replace")), dtype=torch.uint8)


def generation_tensors(output) -> dict:
    """Token ids + logprobs for one request: the artifact every workload produces.

    For steer this IS the artifact (steering has no captured payload, only an effect on
    generation); for the capture workloads it additionally proves capture did not perturb
    the generation it observed.
    """
    import torch

    comp = output.outputs[0]
    out = {
        "prompt_token_ids": torch.tensor(list(output.prompt_token_ids), dtype=torch.int64),
        "token_ids": torch.tensor(list(comp.token_ids), dtype=torch.int64),
        "cumulative_logprob": torch.tensor(float(comp.cumulative_logprob or 0.0),
                                           dtype=torch.float64),
    }
    steps = getattr(comp, "logprobs", None) or []
    vals = [float(step[tid].logprob) for step, tid in zip(steps, comp.token_ids)
            if tid in step]
    if len(vals) == len(comp.token_ids):
        out["token_logprobs"] = torch.tensor(vals, dtype=torch.float64)
    return out


# The float channels that carry a steer's EFFECT. Steering emits no artifact -- it mutates
# the residual stream -- so the only observable is what it does to the output distribution.
LOGPROB_KEYS = ("token_logprobs", "cumulative_logprob")

# Liveness floor: how far a steered run must sit from its unsteered control before we call
# the steer alive. NOT a measurement tolerance -- a floor over float noise.
#
# Why 1e-3. The control is the SAME engine, in the SAME process, decoding the SAME prompts
# greedily with a fixed seed and prefix caching off, so a steer that had genuinely become a
# no-op would reproduce the control EXACTLY: the honest no-op delta is 0.0, not "small".
# The threshold therefore only has to clear bit-noise, and 1e-3 clears it by two orders of
# magnitude while sitting two orders of magnitude BELOW the effect an adjust_rs steer with
# avg_proj=20 actually produces (measured: see the C5 report). It is also 100x the 1e-5
# parity tolerance used to compare the two ARMS, so "alive" and "equal within tolerance"
# can never overlap: a difference big enough to count as liveness is always bigger than a
# difference small enough to count as agreement.
STEER_LIVENESS_ATOL = 1e-3

# Positive-control floor for `steer_small per-request-arming`'s ARMED check (task D12,
# RE-ANCHORED task D13). `STEER_LIVENESS_ATOL` above assumes a genuinely unsteered no-op
# control reproduces its steered pass EXACTLY (delta 0.0), which held for the single-request
# liveness gates it was built for. It does NOT hold for the mixed-arm A/B gate's positive
# control (real-vs-zero `dir` at width 33): the null-control experiment
# (tests/mia/parity/run_steer_ab_null_control.py, LSF 1733997, ONE engine boot) measured its
# PATH pass -- the SAME real vector bytes, only `vector_path` renamed, i.e. NO steering
# difference at all -- moving ARMED rows by 5.830579e-02, already 58x above
# STEER_LIVENESS_ATOL. A floor at 1e-3 cannot tell "Arm B silently failed to be a real
# zero-effect control" from "nothing happened, but vector_path noise moved the rows" apart.
#
# Task D12 anchored the UPPER bound on 1.216572e+01, that SAME job's "GATE" pass (real `dir`
# vs all-zero `dir`) -- but that number is the WORST effect across a whole real-vs-zero BATCH
# comparison, not a bound on any single request. The weakest genuine PER-REQUEST armed
# effect this gate actually produces is far below it: 2.757719e-01, measured on THIS gate's
# own full mixed-arm batch (LSF 1737784), agreeing to 6 significant figures with 2.757723e-01
# from the original decisive A/B experiment (LSF 1725670, a different job on a different
# node). Anchoring on 1.216572e+01 put the old floor (10x the noise ceiling = 5.830579e-01)
# ABOVE both of those, so no genuine per-request steer could ever clear it -- see LSF
# 1737784: "req29: ARMED but did not clear the positive-control floor
# (2.757719e-01 <= 5.830579e-01)".
#
# A per-request floor must instead sit strictly between:
#   LOWER anchor (noise it must exceed): the PATH-noise ceiling above, 5.830579e-02 -- larger
#   than the same job's worst UNARMED pass-to-pass logprob envelope
#   (STEER_UNARMED_LOGPROB_ENVELOPE, 3.265401e-02), so it is the binding lower anchor.
#   UPPER anchor (weakest real signal it must not exceed): 2.757719e-01 (LSF 1737784) /
#   2.757723e-01 (LSF 1725670).
#
# Floor = 2.0 * 5.830579e-02 = 1.166116e-01, giving:
#   - 2.00x margin ABOVE the noise ceiling (5.830579e-02) -- 3.57x above the unarmed envelope
#   - 2.36x margin BELOW the weakest measured armed effect (2.757719e-01 / 2.757723e-01)
# comfortable clearance on both sides. See
# test_arming_gate_atol_sits_between_noise_and_signal in tests/test_parity_steer_bands.py,
# which asserts both inequalities and both margins hermetically off these same recorded
# constants, so a future edit that pushes this floor back above the real signal (or below
# the noise) fails a unit test instead of a GPU job an hour later.
STEER_ARM_POSITIVE_CONTROL_ATOL = 2.0 * 5.830579e-02  # = 1.166116e-01

# The measured pass-to-pass envelope for UNARMED-row logprob movement inside the
# `steer_small per-request-arming` A/B design (task D12) -- recorded for the INFO line only,
# NOT a tolerance anyone may spend. GATE 1 and GATE 2 in
# tests/mia/parity/run_steer_ab_null_control.py (LSF 1733997) are the SAME comparison (Arm A
# real vs Arm B zero) sampled twice inside ONE engine boot: one draw measured 3.265401e-02
# on the worst unarmed row, the other measured EXACTLY 0.0. Two passes differing in NOTHING
# disagreeing on this channel is what proves unarmed-row logprobs are not reproducible
# pass-to-pass even within a single boot -- see STEER_ARM_POSITIVE_CONTROL_ATOL above and
# tests/mia/parity/TOLERANCES.md. The gate's own no-leakage property rests on unarmed TOKEN IDS
# (bit-exact, fatal) and the armed positive control above, not on this number.
STEER_UNARMED_LOGPROB_ENVELOPE = 3.265401e-02


def logprob_delta(a: dict, b: dict) -> float:
    """max |a - b| over the logprob channels of two ``generation_tensors`` dicts.

    Raises when the two dicts share no comparable logprob channel: a liveness check that
    silently compared nothing would report "no difference" for a run that never produced a
    float channel at all, which is the same false PASS the empty-tree guard exists to stop.
    """
    import torch

    worst, seen = 0.0, False
    for key in LOGPROB_KEYS:
        x, y = a.get(key), b.get(key)
        if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.shape != y.shape:
            continue
        seen = True
        worst = max(worst, float((x.double() - y.double()).abs().max().item()))
    if not seen:
        raise RuntimeError(
            f"no comparable logprob channel in these two results (looked for {LOGPROB_KEYS}, "
            f"A has {sorted(a)}, B has {sorted(b)}): a steer liveness check would be vacuous")
    return worst


def _entry_tensors(entry: dict) -> dict:
    """The comparable content of one captured layer entry.

    Tensors plus the small numeric/enum metadata that decides how a reader interprets them.
    Free-form strings (module names, the worker's own config blob) are left out on purpose:
    they can carry filesystem paths and package names, which differ between two checkouts
    for reasons that have nothing to do with the numbers.
    """
    import torch

    out = {}
    for key in ("hidden_states", "q", "k_all", "scores"):
        value = entry.get(key)
        if torch.is_tensor(value):
            out[key] = value
    out["layer_num"] = torch.tensor(int(entry["layer_num"]), dtype=torch.int64)
    for key in ("hs_mode", "hookq_mode", "capture"):
        if isinstance(entry.get(key), str):
            out[key] = _ascii(entry[key])
    return out


def write_request_artifacts(out_dir: Path, outputs, *, label: str = "generation") -> int:
    """Normalize one ``generate()`` call to ``<out_dir>/req<N>/*.safetensors``.

    Returns the number of files written. Request N is the Nth prompt of the call.
    """
    written = 0
    for index, output in enumerate(outputs):
        req_dir = Path(out_dir) / f"req{index}"
        _save(req_dir / f"{label}.safetensors", generation_tensors(output))
        written += 1
        probes = getattr(output, "probes", None) or {}
        for cache_key in ("hs_cache", "qk_cache"):
            for entry in (probes.get(cache_key) or {}).values():
                if not isinstance(entry, dict) or "layer_num" not in entry:
                    continue
                tensors = _entry_tensors(entry)
                _save(req_dir / f"layer{int(entry['layer_num']):03d}.safetensors", tensors)
                written += 1
    return written


def count_artifacts(out_dir: Path) -> int:
    return len(list(Path(out_dir).rglob("*.safetensors")))


def scratch_dir(name: str) -> Path:
    """A per-run working directory that is NEVER inside an artifact tree.

    The aperture spill directory and the disk-sink directory both have to point somewhere
    (unset means they default under $HOME and blow the quota), but anything they leave
    behind must not be mistaken for a normalized artifact by a reader — or picked up by the
    comparator's recursive glob. MIA_PARITY_SCRATCH points this at node NVMe on LSF.
    """
    import tempfile

    root = os.environ.get("MIA_PARITY_SCRATCH")
    base = Path(root) if root else Path(tempfile.mkdtemp(prefix="mia_parity_"))
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Arm-neutral: which plugin actually loaded
# ---------------------------------------------------------------------------

def assert_arm(expected_top_level: str, worker_path: str) -> None:
    """Print, and then enforce, which distribution's plugin this process is running.

    vLLM selects plugins from the ``vllm.general_plugins`` entry-point group, filtered by
    the ``VLLM_PLUGINS`` allowlist -- and it logs that decision at DEBUG, which a normal
    run never shows. With two disjoint distributions installed side by side, a run that
    silently loaded the wrong one (or none, capturing nothing) would still "succeed", so
    the check is made structural here rather than left to a grep of the log.
    """
    from importlib.metadata import entry_points

    import vllm.envs as envs

    allowed = envs.VLLM_PLUGINS
    print(f"[A6] VLLM_PLUGINS allowlist = {allowed}")
    if allowed is None:
        raise RuntimeError(
            "VLLM_PLUGINS is unset: every installed plugin would load, so this process "
            "cannot be attributed to one arm. Set the allowlist explicitly.")

    discovered = sorted(entry_points(group="vllm.general_plugins"), key=lambda e: e.name)
    for ep in discovered:
        state = "LOADED" if ep.name in allowed else "skipped"
        print(f"[A6] plugin entry-point {ep.name} -> {ep.value} : {state}")
        top = ep.value.split(".", 1)[0].split(":", 1)[0]
        if state == "LOADED" and top not in (expected_top_level, "vllm"):
            raise RuntimeError(
                f"WRONG ARM: entry point {ep.name} from {top!r} is allowed, but this run "
                f"must load only {expected_top_level!r}")

    print(f"[A6] worker_extension_cls = {worker_path}")
    worker_top = worker_path.split(".", 1)[0]
    if worker_top != expected_top_level:
        raise RuntimeError(
            f"WRONG ARM: worker class {worker_path} comes from {worker_top!r}, "
            f"expected {expected_top_level!r}")


# ---------------------------------------------------------------------------
# MIA-side driver
# ---------------------------------------------------------------------------

WORKER_KIND = {"hs": "capture_hs", "qk": "capture_qk", "steer": "steer"}


def run_workload(name: str, out_dir: Path, *, graph: bool, batch: int = 1,
                 drive=None, cudagraph: bool | None = None) -> Path:
    """Run one workload under MIA and return the directory holding its artifacts.

    ``batch`` repeats the prompt set to force a wider running batch (D4's alone-vs-batch).

    ``cudagraph`` separates two things ``graph`` used to conflate: whether MIA arms its
    BAKED-OP + capture-aperture path (``graph``, i.e. ``MIA_ALLOW_CUDAGRAPH``), and whether
    vLLM compiles the model and replays CUDA graphs (``cudagraph``, i.e. ``not
    enforce_eager``). Defaults to ``graph``, which is the shipped pairing. Setting
    ``graph=True, cudagraph=False`` runs MIA's in-graph capture op against the SAME
    uncompiled kernels the forward-hook path runs against -- the only configuration in which
    the two capture mechanisms can be compared without vLLM's compiled-vs-eager numerics
    sitting in the middle of the comparison (task D4).

    ``drive`` overrides the post-generate half (default :func:`_drive`). The engine setup --
    model, worker, engine kwargs, arm assertion, scratch layout -- is the part that MUST be
    identical across arms of a comparison, so it stays here; RETRIEVAL is not: the eager path
    reads ``output.probes`` (forward hooks -> ``get_captured_states``) while the FULL-graph
    path reads the durable capture aperture (``flush_aperture`` + ``aperture_reader``), and
    task D4 needs both normalized to one comparable form. See ``tests/mia/parity/t2_invariants.py``.
    """
    wl = WORKLOADS[name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Both are set before the package is imported: the module-level guard that decides
    # TORCHDYNAMO_DISABLE reads the cudagraph flag at import time, and the aperture
    # directory is never left unset (unset means dumps land in $HOME and blow the quota).
    if cudagraph is None:
        cudagraph = graph
    scratch = scratch_dir(f"{name}_{'graph' if graph else 'eager'}"
                          f"{'' if cudagraph == graph else '_nocg'}")
    os.environ["MIA_ALLOW_CUDAGRAPH"] = "1" if graph else "0"
    os.environ["MIA_APERTURE_DIR"] = str(scratch / "aperture")
    os.makedirs(os.environ["MIA_APERTURE_DIR"], exist_ok=True)

    import vllm.plugins

    import mia

    from mia import MiaLLM
    from mia.registry import PluginRegistry

    # Which MIA is this, really? Printed from inside the arm process, so the log proves what
    # the job script's preflight probe can only infer from a different process's sys.path.
    _assert_mia_provenance(mia)

    vllm.plugins.load_general_plugins()   # idempotent; honours VLLM_PLUGINS
    kind = WORKER_KIND[wl.subsystem]
    assert_arm("mia", PluginRegistry.get_worker(kind).path)

    # vLLM 0.29 resolves cudagraph_mode to FULL_AND_PIECEWISE by DEFAULT, and MIA refuses that
    # mode by design (PIECEWISE dispatches some batch shapes through graphs where the in-graph
    # capture op is unvalidated). Every graph-mode run must therefore name FULL explicitly, or
    # the engine never boots -- see mia/_plugin.py::validate_graph_mode.
    graph_kwargs = {"compilation_config": {"cudagraph_mode": "FULL"}} if cudagraph else {}

    llm = MiaLLM(
        model=MODEL,
        worker_name=kind,
        enforce_eager=not cudagraph,
        hook_dir=str(scratch / "sink"),
        **graph_kwargs,
        **ENGINE_KWARGS,
    )
    return (drive or _drive)(llm.llm, wl, out_dir, batch)


def _drive(engine, wl: Workload, out_dir: Path, batch: int) -> Path:
    """Run the requests and normalize the results. Shared with the pre-rename arm.

    ``engine`` is the plugin-patched vLLM ``LLM``. We call it directly rather than going
    through the wrapper's ``generate()`` because that wrapper, for a multi-request call,
    overwrites ``outputs[0].probes`` with a convenience merge that keeps only the FIRST
    forward pass of each request. A parity gate needs every request's full per-step capture,
    and D4 needs request 0 at the same fidelity as the rest.
    """
    prompts = prompts_for(wl, batch)
    extra = extra_args_for(wl)
    if wl.subsystem == "steer":
        print(f"[A6] steer config: {extra['steer']}")
        print(f"[A6] steer fingerprint {steer_fingerprint(extra['steer'])}")
    outputs = engine.generate(prompts, sampling_params_for(wl, extra), use_tqdm=False)
    written = write_request_artifacts(out_dir, outputs)

    if wl.subsystem == "steer":
        # Steering leaves no captured payload, so an inert steer would look like a clean
        # PASS on both arms. Re-run the same prompts with steering disarmed and require the
        # two to differ by MORE THAN FLOAT NOISE (STEER_LIVENESS_ATOL).
        #
        # The test is on the LOGPROBS, not the token ids. Measured in task A6 on this exact
        # workload: steering moved the next-token logprobs on 3/3 requests and the sampled
        # token ids on 0/3. A token-identity liveness check -- the obvious one -- would
        # therefore read a perfectly working steer path as dead.
        import torch

        control = engine.generate(prompts, sampling_params_for(wl, None), use_tqdm=False)
        written += write_request_artifacts(out_dir, control, label="control")
        changed_tokens = 0
        worst = 0.0
        for steered, plain in zip(outputs, control):
            a, b = generation_tensors(steered), generation_tensors(plain)
            if not torch.equal(a["token_ids"], b["token_ids"]):
                changed_tokens += 1
            worst = max(worst, logprob_delta(a, b))
        print(f"[A6] steer liveness: max|d(logprob)| steered-vs-control = {worst:.6e} "
              f"(floor {STEER_LIVENESS_ATOL:.1e}); {changed_tokens}/{len(prompts)} requests "
              f"also changed their token ids")
        if worst <= STEER_LIVENESS_ATOL:
            raise RuntimeError(
                f"steering changed NOTHING measurable: max|d(logprob)| = {worst:.6e} <= "
                f"{STEER_LIVENESS_ATOL:.1e} across {len(prompts)} requests with and without "
                f"the steer config, so the steer op never reached the residual and a steer "
                f"A/B would pass vacuously")

    print(f"[A6] wrote {written} artifact files under {out_dir}")
    if written == 0:
        raise RuntimeError(f"capture produced NOTHING under {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run one parity workload under MIA.")
    ap.add_argument("workload", choices=sorted(WORKLOADS))
    ap.add_argument("out_dir")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args(argv)
    run_workload(args.workload, Path(args.out_dir), graph=args.graph, batch=args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
