"""T3 — the old world (MIA on vLLM 0.21 / V1) against the new one (MIA on vLLM 0.29 / V2).

This is the last validation tier of the port: T0 said capture is an observer, T1 said MIA
agrees with vanilla vLLM + naive forward hooks on 0.29, T2 said MIA agrees with itself
across its own mechanisms — and T3 asks the only question left. Does the ported system
still produce what the pre-port system produced?

WHY THIS FILE IS NOT A ONE-CALL `compare()` WITH A ROUND `atol`
---------------------------------------------------------------
The obvious T3 is `compare(old, new, atol=2e-2)` with the problems split into "structural"
and "value" by substring matching. That design reintroduces every defect class Phase D
spent weeks removing from this suite:

  * a comparator that skips a channel missing on one side and then prints `0.000000e+00`,
    which reads as perfect agreement and means "nothing was compared";
  * gates that VANISH when a leg fails to run, so the suite gets shorter instead of redder;
  * an absolute-only metric that cannot tell 1-ULP noise from a destroyed row;
  * a band with no derivation and no discrimination test, which catches nothing.

So T3 is built on the Phase-D comparator stack, by import and never by copy:
`_compare` (bit-exact), `_compare_replay_band_checked` (tensor-global RELATIVE + PER-ROW
bands, bifurcation-aware), `_compare_generation_band` (token ids bit-exact, logprobs
banded), `_bifurcation_report` (token-id divergence detection), `gate_problems` (the
declared-vs-emitted gate-set machinery) and `steer_logprob_delta` (which RAISES rather than
returning a vacuous 0.0). The zero-artifact guard rides along inside all of them, and
`EXPECTED_ARTIFACTS` below adds the one thing none of them can know: which files and tensor
keys each workload is SUPPOSED to produce.

THE ARMS — five, in ONE job on ONE node, so no gate crosses a node or a job boundary
------------------------------------------------------------------------------------
    ref   the on-disk 0.21 reference minted by task A6 (LSF 1696165, node p5-r25-n3),
          MIA @ 9cfbef6 on vLLM 0.21 / V1, at vLLM's DEFAULT running-batch width
    c21   vanilla vLLM 0.21 + naive `register_forward_hook`, MIA NEVER IMPORTED, width 1
    a21   MIA @ 9cfbef6 on vLLM 0.21 / V1, width 1
    a21r  MIA @ 9cfbef6 on vLLM 0.21 / V1, at the reference's DEFAULT width
    c29   vanilla vLLM 0.29 + naive `register_forward_hook`, MIA NEVER IMPORTED, width 1
    a29   MIA @ HEAD on vLLM 0.29 / V2, width 1

`a21`/`a21r` exist because MIA at HEAD REFUSES the V1 runner (`mia/runner.py`
`require_v2_runner`), so the old world can only be re-run from a worktree pinned at the
Phase-A SHA — see `run_crossbranch.sh` for how that package is put in front of the editable
install without touching it.

WIDTH IS NOT A FREE VARIABLE. The reference tree was minted with `MIA_PARITY_MAX_NUM_SEQS`
unset, i.e. all three requests resident at once, while the vanilla oracle
(`t1_reference.py`) can only run at width 1 — it locates a request's rows by "this request
owns the leading rows", which is true only when one request is resident. And the model is
NOT batch-invariant: T2's own `CONTROL eager alone-vs-batch` fails bit-exactness on every
workload in every job that has run it. So a width-1 arm compared against the width-default
reference carries a batch-variance confound that no gate here could attribute. That is why
`a21` exists at width 1 (every FATAL gate is matched-width) and `a21r` exists at the
reference's width (the reference check is matched-width too), and why the two REF gates
that unavoidably cross widths are INFO.

THE GATES
---------
  G1  FATAL, bit-exact   a21 == c21    MIA is inert on 0.21: what it captured IS the
                                       model's own activation, not something it perturbed.
                                       Over EVERY channel the workload writes, not just the
                                       captured payload -- see the note beside the call.
  G2  FATAL, bit-exact   a29 == c29    The same claim on 0.29. This re-confirms T1 on this
                                       node; T1 already passes bit-exact at width 1 eager,
                                       so a failure here is about the JOB'S SETUP, not a
                                       new port bug — the log says so.
  G3  INFO               ref vs a21r   Cross-job / cross-node reproducibility of the A6
                                       reference, at matched width and matched MIA SHA.
                                       Structure is reported; nothing here gates T3,
                                       because no measurement in this suite establishes
                                       cross-boot bit-exactness at 0.21 — this leg is that
                                       measurement.
  G4  CONTROL, MEASURED  c21 vs c29    Neither side ever imports MIA, so this is PURE vLLM
                                       0.21->0.29 drift: different attention backend,
                                       different kernels, different torch. Structure is
                                       FATAL (the band is meaningless over trees that do
                                       not line up); the values are the band's anchor.
  G5  T3 ITSELF          a21 vs a29    Structure FATAL. Values BANDED against G4 -- on the
                                       captured payload (hs/qk), on the generation channel,
                                       and on `steer_small`'s unsteered CONTROL channel,
                                       which has its own three legs because it is a
                                       different measurement from the steered one.
  G5b INFO               ref vs a29    The same question asked directly of the on-disk
                                       reference. INFO because it crosses both a job/node
                                       boundary and a width boundary.

WHY THE EXTRA ARMS ARE WORTH IT. If G1 and G2 are both bit-exact then, by transitivity,
G5's delta IS G4's delta, and the result is a real statement — "MIA contributes exactly zero
on both vLLM versions; every difference between the old world and the new world is vLLM's
own cross-version drift" — instead of the unfalsifiable "we banded it at 2e-2". If G1 or G2
is NOT bit-exact, that is the finding, and it matters more than T3's verdict: it is printed
under its own heading and it is never absorbed into a band.

WHERE THE BAND COMES FROM (the derivation, also in tests/mia/parity/TOLERANCES.md)
------------------------------------------------------------------------------
Not a round number, and not a constant at all: the band is DERIVED, per workload, from the
three legs measured in the same job, by the triangle inequality.

    |a21 - a29|  <=  |a21 - c21|  +  |c21 - c29|  +  |c29 - a29|
                       (G1)            (G4)            (G2)

    band = MARGIN * (G1 + G4 + G2), capped at a CEILING

so when G1 and G2 are bit-exact — the expected case — the band is exactly `MARGIN x G4`,
i.e. twice the pure-vLLM drift measured on this node in this job with MIA absent from both
sides. `MARGIN = 2.0` is the same headroom `capture_workload.STEER_ARM_POSITIVE_CONTROL_ATOL`
uses over its own measurement, and it also absorbs the one approximation in the triangle
above: each leg's RELATIVE metric normalises by its own side-A magnitude, and those
denominators are equal only to within the drift itself.

The CEILING is what stops a derived band from silently becoming a licence. Bands wide
enough to pass a mis-route catch nothing: a whole-request off-by-one row measures ~4.5e-01
tensor-global (`tests/test_parity_band_discrimination.py::OFF_BY_ONE_ROW`) and ~1.0 per-row,
and a real steering intervention moves a logprob by ~7.6e-01. The ceilings below sit an
order of magnitude under those, so the gate can still fail for the reasons it exists for. If
a control ever measures above `CEILING / MARGIN`, the band is capped, a `CONTROL ANOMALY`
line is printed, and G5 is allowed to fail — a red T3 with a named cause beats a green one
whose band was widened to fit.

There is NO environment override for any of this. `t2_invariants._band()` exists because
those bounds are constants that something might want to widen from the outside; these are
derived in-process from measurements taken in the same job, and the two ceilings are source
constants with no reader of `os.environ` anywhere near them.

TOKEN IDS COME FIRST ON EVERY CROSS-VERSION LEG
-----------------------------------------------
Across two vLLM versions a greedy argmax can legitimately flip at a near-tie, and every row
after such a flip holds the activations of a DIFFERENT TOKEN — not comparable under any
band. Phase D lost a full bug hunt to exactly this (a 45% relative delta at layer 1 that
looked like a mis-route and was a tie-flip). So `_bifurcation_report` runs before any tensor
comparison on G3/G4/G5/G5b, only pre-divergence rows are compared, and the skipped rows are
counted in the log. A bifurcation is not a blanket excuse: the leg still FAILS if the
pre-divergence rows disagree, or if the divergence carries no explanatory logprob delta
(greedy sampling from bit-identical logits cannot produce two different argmaxes).

Run (no GPU needed — this file only reads artifact trees):

    python tests/mia/parity/t3_crossbranch.py hs_small \\
        --ref DIR --c21 DIR --a21 DIR --a21r DIR --c29 DIR --a29 DIR
"""
from __future__ import annotations

import argparse
import io
import math
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.capture_workload import (  # noqa: E402
    STEER_LIVENESS_ATOL,
    WORKLOADS,
)
from tests.mia.parity.compare_artifacts import steer_logprob_delta  # noqa: E402
from tests.mia.parity.t1_reference import block_index  # noqa: E402
from tests.mia.parity.t2_invariants import (  # noqa: E402
    _bifurcation_report,
    _compare,
    _compare_generation_band,
    _compare_replay_band_checked,
    gate_problems,
)

TAG = "T3"
ARMS = ("ref", "c21", "a21", "a21r", "c29", "a29")

# The arm that minted the on-disk reference, echoed in the log so a tree can always be
# traced back to the job that produced it.
REFERENCE_JOB = "LSF 1696165, node p5-r25-n3, 2026-09-16, MIA @ 9cfbef6, vLLM 0.21.0 / V1"

# ---------------------------------------------------------------------------
# THE BAND POLICY — derived per workload, never read from the environment
# ---------------------------------------------------------------------------
# See the module docstring for the derivation. These three constants are the whole policy,
# and they are the ONLY numbers in this file that were chosen rather than measured.
_MARGIN = 2.0

# Ceilings, chosen so the band can still fail for the reasons the gate exists for:
#   * an off-by-one whole-request row shift measures ~4.512e-01 tensor-global and ~1.0
#     per-row (tests/test_parity_band_discrimination.py, LSF 1705469);
#   * a real steering intervention moves a logprob by ~7.632e-01 (task D4's measurement).
# Each ceiling sits at least 4x under the signature it must still catch.
_REL_CEILING = 1.0e-1
_ROW_CEILING = 5.0e-1
_LOGPROB_CEILING = 1.0e-1

# ---------------------------------------------------------------------------
# DECLARED ARTIFACTS — what each workload is SUPPOSED to produce (ruling E-3)
# ---------------------------------------------------------------------------
# `steer_small` is why this table exists. It writes NO layer tensors at all — only
# `generation` and its unsteered `control` — so a comparison that walked "whatever files are
# there" would compare two files, find them equal, and report a clean PASS for a subsystem
# whose payload it never looked at. Worse, an arm whose capture silently produced nothing
# would match another arm that produced nothing. Declaring the tree makes both vacuities
# impossible: every arm is checked against this table before any arm is compared to another.
_GENERATION_KEYS = ("cumulative_logprob", "prompt_token_ids", "token_ids", "token_logprobs")
_HS_LAYER_KEYS = ("hidden_states", "hs_mode", "layer_num")
_QK_LAYER_KEYS = ("hookq_mode", "k_all", "layer_num", "q")


def _expected_tree(workload: str) -> dict[str, tuple[str, ...]]:
    """`{relative path: sorted tensor keys}` for one workload's artifact tree."""
    wl = WORKLOADS[workload]
    tree: dict[str, tuple[str, ...]] = {}
    for index in range(len(wl.prompts)):
        tree[f"req{index}/generation.safetensors"] = _GENERATION_KEYS
        if wl.subsystem == "steer":
            tree[f"req{index}/control.safetensors"] = _GENERATION_KEYS
            continue
        keys = _HS_LAYER_KEYS if wl.subsystem == "hs" else _QK_LAYER_KEYS
        for layer in wl.layers:
            tree[f"req{index}/layer{int(layer):03d}.safetensors"] = keys
    return tree


EXPECTED_ARTIFACTS = {name: _expected_tree(name)
                      for name in ("hs_small", "qk_small", "steer_small")}

# LAYER NUMBERING, enforced rather than assumed. Getting this wrong is silent in BOTH
# directions: comparing HS layer 1 against QK layer 1 compares two different physical
# blocks, and asking the HS path for layer 0 matches no module at all and yields an empty
# tree that a naive A/B reads as a trivial PASS. `hs_small`'s (1, 16, 32) and `qk_small`'s
# (0, 15, 31) must name the SAME three blocks, and `steer_small`'s (15,) must be the middle
# one. The mapping itself lives in t1_reference (HS is 1-based, QK and steer are 0-based)
# and is imported, never re-derived here.
_PHYSICAL_BLOCKS = (0, 15, 31)


def assert_layer_conventions() -> str:
    """Prove the three workloads' layer numbers name the blocks this tier thinks they do."""
    lines = []
    for kind, workload in (("capture_hs", "hs_small"), ("capture_qk", "qk_small")):
        blocks = tuple(block_index(kind, layer) for layer in WORKLOADS[workload].layers)
        if blocks != _PHYSICAL_BLOCKS:
            raise RuntimeError(
                f"{workload} layers {WORKLOADS[workload].layers} map to blocks {blocks}, "
                f"expected {_PHYSICAL_BLOCKS}. HS layer numbers are 1-BASED and QK layer "
                f"numbers are 0-BASED; if these two workloads stop naming the same physical "
                f"blocks, every cross-tier comparison in this file compares different layers.")
        lines.append(f"{workload} {WORKLOADS[workload].layers} -> blocks {blocks}")
    steer_layers = WORKLOADS["steer_small"].layers
    steer_blocks = tuple(block_index("steer", layer) for layer in steer_layers)
    if steer_blocks != (15,):
        raise RuntimeError(
            f"steer_small optimal_layer {steer_layers} maps to blocks {steer_blocks}, "
            f"expected (15,) — the steer path filters on the RAW 0-based block index.")
    lines.append(f"steer_small {steer_layers} -> blocks {steer_blocks}")
    return "; ".join(lines)


def inventory_problems(tree: Path | None, workload: str) -> list[str]:
    """Every way one arm's artifact tree can fail to be the tree this workload declares.

    Includes the "produced NOTHING" guard explicitly rather than inheriting it: an empty
    tree is reported as an empty tree, not as a long list of missing files.
    """
    from safetensors.torch import load_file

    expected = EXPECTED_ARTIFACTS[workload]
    if tree is None:
        return [f"the arm did not run, so it produced no tree to check against the "
                f"{len(expected)} declared artifacts"]
    root = Path(tree)
    if not root.exists():
        return [f"{root} does not exist -- this arm produced NOTHING"]
    found = {p.relative_to(root).as_posix() for p in root.rglob("*.safetensors")}
    if not found:
        return [f"{root} contains no *.safetensors at all -- this arm produced NOTHING"]

    problems = []
    for missing in sorted(set(expected) - found):
        problems.append(f"{missing}: DECLARED for {workload} but absent from {root}")
    for extra in sorted(found - set(expected)):
        problems.append(f"{extra}: present under {root} but NOT declared for {workload} "
                        f"(a leaked request, or a stale tree from another run)")
    for rel in sorted(set(expected) & found):
        keys = tuple(sorted(load_file(str(root / rel)).keys()))
        if keys != tuple(sorted(expected[rel])):
            problems.append(f"{rel}: tensor keys {keys} != declared "
                            f"{tuple(sorted(expected[rel]))}")
    return problems


def judge_inventory(workload: str, arms: dict, verdicts: list) -> None:
    """One gate over EVERY arm: each tree is the tree this workload declares."""
    label = f"{workload} inventory (all arms)"
    total = 0
    for name in ARMS:
        problems = inventory_problems(arms.get(name), workload)
        total += len(problems)
        state = "OK" if not problems else f"{len(problems)} problems"
        print(f"INVENTORY {workload} {name}: {state} "
              f"({len(EXPECTED_ARTIFACTS[workload])} declared artifacts)", flush=True)
        for problem in problems[:10]:
            print(f"MISMATCH {name}: {problem}", flush=True)
        if len(problems) > 10:
            print(f"MISMATCH {name}: ... and {len(problems) - 10} more", flush=True)
    ok = total == 0
    print(f"VERDICT {TAG} {label}: {'PASS' if ok else 'FAIL'} ({total} problems across "
          f"{len(ARMS)} arms)", flush=True)
    verdicts.append((label, ok, True))


# ---------------------------------------------------------------------------
# MEASUREMENT — by running the GATE with its bands set to infinity
# ---------------------------------------------------------------------------
# The control's number and the gate's number have to be produced by the SAME code, or
# "banded against the measured control" means nothing: two comparators that normalise
# differently, or truncate a bifurcation differently, would silently compare different
# quantities. So nothing here re-implements the metric. `measure` calls
# `_compare_replay_band_checked` — the very function that gates G5 — with `rel=inf` and
# `row=inf`, which leaves exactly the STRUCTURAL findings (file sets, tensor key sets,
# shapes, integer metadata, an unexplained bifurcation) able to fail, and reads the worst
# relative and worst per-row values out of the verdict line it prints.
#
# Parsing a printed line is only safe if a parse failure is LOUD. It is: no match raises,
# and the captured text is re-emitted to the log either way, so nothing is hidden.
#
# ONE KNOWN IMPRECISION, stated rather than discovered later: the comparator renders its
# worst values with `%.3e`, i.e. four significant figures, so a reading can understate the
# true worst by up to 5e-04 RELATIVE (0.05%). That is four orders of magnitude smaller than
# the `_MARGIN` the band multiplies it by, so it cannot move a verdict; it is recorded here
# because a band derived from a rounded number should say that it is.
_WORST_RE = re.compile(
    r"worst rel=(?P<rel>[-+0-9.eE]+|inf|nan) at (?P<rel_key>\S+) \[band [^\]]*\], "
    r"worst per-row=(?P<row>[-+0-9.eE]+|inf|nan) at (?P<row_key>\S+) \[band")
_TENSORS_RE = re.compile(r"\((?P<n>\d+) tensors,")


def measure(label: str, a: Path | None, b: Path | None, *, name: str, verdicts: list | None,
            structure_fatal: bool) -> dict | None:
    """Measure `a` vs `b`; gate only its STRUCTURE. Returns None when an arm is missing.

    `verdicts=None` measures without emitting a gate (used for the per-layer breakdown and
    for G1/G2, whose gate is the bit-exact comparison itself).
    """
    if a is None or b is None:
        if verdicts is not None:
            print(f"VERDICT {TAG} {label}: FAIL (an arm did not run, so this leg was never "
                  f"measured -- a missing measurement is a failure, not an absence)",
                  flush=True)
            verdicts.append((label, False, structure_fatal))
        return None

    buffer = io.StringIO()
    inner: list = []
    with redirect_stdout(buffer):
        _compare_replay_band_checked(label, a, b, name=name, verdicts=inner,
                                     rel=math.inf, row=math.inf, tag=TAG)
    text = buffer.getvalue()
    # The comparator always prints `VERDICT` -- but only a leg with a gate of its own IS a
    # verdict. A measurement re-labels itself so a reader (or a grep for `VERDICT T3`) can
    # never mistake an informational number for a gate, and a non-fatal gate says INFO, the
    # same convention `_compare` uses one tier down.
    prefix = "MEASURE-ONLY" if verdicts is None else ("VERDICT" if structure_fatal else "INFO")
    printable = (text if prefix == "VERDICT"
                 else text.replace(f"VERDICT {TAG} ", f"{prefix} {TAG} "))
    print(printable, end="", flush=True)

    match = _WORST_RE.search(text)
    if match is None:
        raise RuntimeError(
            f"could not read the measurement out of the comparator's verdict line for "
            f"{label!r}. T3 derives its band from this number, so a measurement it cannot "
            f"read is a hard failure rather than a zero. Output was:\n{text}")
    count = _TENSORS_RE.search(text)
    structure_ok = inner[0][1] if inner else False
    result = {
        "label": label,
        "rel": float(match.group("rel")),
        "rel_key": match.group("rel_key"),
        "row": float(match.group("row")),
        "row_key": match.group("row_key"),
        "tensors": int(count.group("n")) if count else 0,
        "structure_ok": structure_ok,
    }
    print(f"MEASURED {TAG} {label}: rel={result['rel']:.6e} at {result['rel_key']} "
          f"per-row={result['row']:.6e} at {result['row_key']} "
          f"({result['tensors']} tensors, structure "
          f"{'OK' if structure_ok else 'PROBLEMS'})", flush=True)
    if verdicts is not None:
        verdicts.append((label, structure_ok, structure_fatal))
    return result


def measure_per_layer(label: str, a: Path | None, b: Path | None, workload: str) -> None:
    """The same measurement, one layer file at a time — reported, never gated.

    Depth matters for the reading: cross-version numerics accumulate with depth while a
    mis-route does not, so "layer 1 is as bad as layer 32" and "layer 32 is 30x layer 1"
    are different findings and the per-workload worst hides which one happened.
    """
    if a is None or b is None:
        return
    layers = WORKLOADS[workload].layers
    if WORKLOADS[workload].subsystem == "steer":
        return
    for layer in layers:
        measure(f"{label} layer{int(layer):03d}", a, b,
                name=f"layer{int(layer):03d}.safetensors",
                verdicts=None, structure_fatal=False)


def measure_logprobs(label: str, a: Path | None, b: Path | None,
                     *, channel: str = "generation") -> float | None:
    """max |d| over the LOGPROB channels of two trees' `<channel>.safetensors`.

    `steer_logprob_delta` RAISES when the two sides share no comparable logprob tensor, so
    this cannot quietly return 0.0 for trees with nothing to compare. `channel="control"`
    measures `steer_small`'s UNSTEERED arm, which is a different measurement from the
    steered one and gets its own band.
    """
    if a is None or b is None:
        return None
    delta = steer_logprob_delta(a, b, label_a=channel, label_b=channel)
    print(f"MEASURED {TAG} {label}: max|d(logprob)|={delta:.6e}", flush=True)
    return delta


# ---------------------------------------------------------------------------
# THE BAND, derived
# ---------------------------------------------------------------------------

def _finite(value: float, what: str, term: str) -> bool:
    """Refuse a band term that is not a finite number. Fails CLOSED.

    `min(nan, ceiling)` is `nan`, and every band comparison in this stack is `delta > band`,
    which is False for every delta against NaN -- so a single NaN term would silently turn
    the gate into an unconditional PASS printing `band=nan`. `_WORST_RE` accepts `nan` by
    construction (it parses whatever the comparator printed), so the parse path can produce
    one even though today's arithmetic cannot: `worst_rel` only updates through `r > worst`,
    which NaN never satisfies, and `steer_logprob_delta`'s `max(worst, nan)` returns `worst`.
    A guard that costs two comparisons is cheaper than relying on that staying true.
    """
    if math.isfinite(value):
        return True
    print(f"BAND {TAG} {what}: NOT DERIVABLE -- the {term} leg measured {value!r}, which is "
          f"not finite. A NaN band passes EVERY input (`delta > nan` is always False) and an "
          f"infinite one bounds nothing, so the band fails closed and the gate it feeds "
          f"FAILS rather than silently disarming.", flush=True)
    return False


def derive_band(control: dict | None, g1: dict | None, g2: dict | None,
                *, key: str, ceiling: float, what: str) -> float | None:
    """`MARGIN * (G1 + G4 + G2)`, capped at `ceiling`. See the module docstring."""
    if control is None or g1 is None or g2 is None:
        missing = [n for n, m in (("G1", g1), ("G4 control", control), ("G2", g2))
                   if m is None]
        print(f"BAND {TAG} {what}: NOT DERIVABLE -- {', '.join(missing)} did not run. The "
              f"band is the sum of three measured legs; with one missing there is no bound "
              f"to enforce and the gate FAILS rather than inventing one.", flush=True)
        return None
    for term, leg in (("G1", g1), ("G4 control", control), ("G2", g2)):
        if not _finite(float(leg[key]), what, term):
            return None
    total = float(control[key]) + float(g1[key]) + float(g2[key])
    raw = _MARGIN * total
    if not _finite(raw, what, "sum"):
        return None
    band = min(raw, ceiling)
    print(f"BAND {TAG} {what}: {band:.6e} = min({_MARGIN} x (G1 {g1[key]:.6e} + G4 "
          f"{control[key]:.6e} + G2 {g2[key]:.6e}) = {raw:.6e}, ceiling {ceiling:.6e})",
          flush=True)
    if raw > ceiling:
        print(f"CONTROL ANOMALY {TAG} {what}: the three measured legs sum to {total:.6e}, so "
              f"the derived band {raw:.6e} EXCEEDS the ceiling {ceiling:.6e} and has been "
              f"capped. A band wide enough to pass a mis-route would catch nothing, so the "
              f"cap stands and this gate may fail. That failure is a real finding about "
              f"vLLM's cross-version drift -- do not widen the ceiling to absorb it.",
              flush=True)
    return band


def derive_logprob_band(control: float | None, g1: float | None, g2: float | None,
                        *, what: str = "logprob") -> float | None:
    """The same derivation for an absolute logprob channel.

    ``what`` names the channel in the log: ``steer_small`` derives this TWICE, once for the
    steered `generation` channel and once for its unsteered `control` channel, which are
    separate measurements and must not share a bound.
    """
    if control is None or g1 is None or g2 is None:
        print(f"BAND {TAG} {what}: NOT DERIVABLE -- a leg did not run.", flush=True)
        return None
    for term, leg in (("G1", g1), ("G4 control", control), ("G2", g2)):
        if not _finite(leg, what, term):
            return None
    total = control + g1 + g2
    raw = _MARGIN * total
    if not _finite(raw, what, "sum"):
        return None
    band = min(raw, _LOGPROB_CEILING)
    print(f"BAND {TAG} {what}: {band:.6e} = min({_MARGIN} x (G1 {g1:.6e} + G4 "
          f"{control:.6e} + G2 {g2:.6e}) = {raw:.6e}, ceiling {_LOGPROB_CEILING:.6e})",
          flush=True)
    if raw > _LOGPROB_CEILING:
        print(f"CONTROL ANOMALY {TAG} {what}: derived {raw:.6e} exceeds the ceiling "
              f"{_LOGPROB_CEILING:.6e} and has been capped; a steering intervention moves a "
              f"logprob by ~7.6e-01 and a band above the ceiling could no longer tell one "
              f"from cross-version noise.", flush=True)
    return band


# ---------------------------------------------------------------------------
# GENERATION, made bifurcation-aware
# ---------------------------------------------------------------------------

def _truncated_generation_tree(root: Path, cutoffs: dict, out: Path,
                               name: str = "generation.safetensors") -> Path:
    """Copy `root`'s generation files, cut to the rows before each request's divergence.

    Everything else about the comparison is unchanged, which is the point: the comparator
    that judges the result is `_compare_generation_band`, unmodified. `cumulative_logprob` is
    a scalar over the WHOLE sequence and cannot be cut, so it is RE-DERIVED on both sides as
    the sum of the surviving `token_logprobs` — still a real float channel over exactly the
    rows both arms agree they generated, rather than a key dropped (which the comparator
    would correctly call vacuous) or kept (which would compare two different sequences).
    """
    import torch
    from safetensors.torch import load_file, save_file

    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(root).rglob(name)):
        req = path.parent.name
        tensors = dict(load_file(str(path)))
        cutoff = cutoffs.get(req)
        if cutoff is not None:
            prompt_len = int(tensors["prompt_token_ids"].shape[0])
            keep = max(cutoff[0] - prompt_len, 0)
            tensors["token_ids"] = tensors["token_ids"][:keep].clone()
            tensors["token_logprobs"] = tensors["token_logprobs"][:keep].clone()
            tensors["cumulative_logprob"] = torch.tensor(
                float(tensors["token_logprobs"].double().sum()),
                dtype=tensors["cumulative_logprob"].dtype)
        target = out / req / name
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(target))
    return out


def compare_generation_band_checked(label: str, a: Path | None, b: Path | None, *,
                                    verdicts: list, bound: float | None, note: str,
                                    name: str = "generation.safetensors") -> None:
    """`_compare_generation_band`, with the cross-version tie-flip accounted for.

    Token ids stay bit-exact and FATAL — that is the claim T3 exists to make — for every row
    both arms actually generated. What changes is only which rows those are: a divergence
    that carries an explanatory logprob delta is a greedy tie-flip, and the rows after it
    hold a DIFFERENT token's activations on the two sides. They are excluded, counted, and
    named in the log. A divergence with no explanatory delta, or a disagreement before the
    divergence, still fails.
    """
    if bound is None:
        print(f"VERDICT {TAG} {label}: FAIL (no band could be derived -- see the BAND line "
              f"above; an ungrounded bound is not a bound)", flush=True)
        verdicts.append((label, False, True))
        return
    if a is None or b is None:
        _compare_generation_band(label, a, b, verdicts=verdicts, bound=bound, note=note,
                                 tag=TAG, name=name)
        return

    reqs = {p.parent.name for p in Path(a).rglob(name)}
    reqs |= {p.parent.name for p in Path(b).rglob(name)}
    cutoffs, problems, notes = _bifurcation_report(Path(a), Path(b), reqs, name)
    for line in notes:
        print(line, flush=True)
    if problems:
        for problem in problems[:10]:
            print("MISMATCH", problem, flush=True)
        print(f"VERDICT {TAG} {label}: FAIL ({len(problems)} unexplained trajectory "
              f"divergences -- a bifurcation is not a blanket excuse)", flush=True)
        verdicts.append((label, False, True))
        return
    if not cutoffs:
        _compare_generation_band(label, a, b, verdicts=verdicts, bound=bound, note=note,
                                 tag=TAG, name=name)
        return

    scratch = Path(tempfile.mkdtemp(prefix="mia_t3_bifurcation_"))
    trimmed_a = _truncated_generation_tree(Path(a), cutoffs, scratch / "a", name)
    trimmed_b = _truncated_generation_tree(Path(b), cutoffs, scratch / "b", name)
    skipped = sum(total - row for row, total in cutoffs.values())
    _compare_generation_band(
        label, trimmed_a, trimmed_b, verdicts=verdicts, bound=bound,
        note=f"{note}; {len(cutoffs)} request(s) bifurcated, compared on pre-divergence "
             f"rows only ({skipped} rows skipped, cumulative_logprob re-derived over the "
             f"surviving rows on BOTH sides)", tag=TAG, name=name)


def judge_steer_liveness(arms: dict, verdicts: list) -> None:
    """Every steer arm must differ from its OWN unsteered control by more than float noise.

    Without this, two arms whose steering had silently become a no-op would agree perfectly
    and T3 would report a clean PASS over an intervention that never happened. The test is
    on the LOGPROBS, not the token ids: task A6 measured this exact workload moving the
    logprobs on 3/3 requests and the sampled token ids on 0/3, so a token-identity liveness
    check would read a perfectly working steer path as dead.
    """
    for name in ARMS:
        label = f"steer_small liveness {name}"
        tree = arms.get(name)
        if tree is None:
            print(f"VERDICT {TAG} {label}: FAIL (arm did not run)", flush=True)
            verdicts.append((label, False, True))
            continue
        try:
            delta = steer_logprob_delta(tree, tree, label_b="control")
        except RuntimeError as exc:
            print(f"MISMATCH {label}: {exc}", flush=True)
            print(f"VERDICT {TAG} {label}: FAIL (no comparable steered-vs-control channel)",
                  flush=True)
            verdicts.append((label, False, True))
            continue
        alive = delta > STEER_LIVENESS_ATOL
        print(f"VERDICT {TAG} {label}: {'PASS' if alive else 'FAIL'} "
              f"(max|d(logprob)| steered-vs-control = {delta:.6e}, floor "
              f"{STEER_LIVENESS_ATOL:.1e}) -> {'ALIVE' if alive else 'INERT'}", flush=True)
        verdicts.append((label, alive, True))


# ---------------------------------------------------------------------------
# The gate table and the judge
# ---------------------------------------------------------------------------

def expected_gates(workload: str) -> tuple[str, ...]:
    """Every gate `judge` must emit for `workload`.

    Derived from the same facts the judge reads (does this workload have layer tensors at
    all?) rather than written out three times, so the table and the judge cannot drift; the
    hermetic test asserts SET EQUALITY between this and what a full judge run emits.
    """
    w = workload
    gates = [
        f"{w} inventory (all arms)",
        f"{w} G1 MIA-inert-0.21 a21-vs-c21 (bit-exact)",
        f"{w} G2 MIA-inert-0.29 a29-vs-c29 (bit-exact)",
        f"{w} G3 reference reproducibility ref-vs-a21r [INFO]",
        f"{w} WIDTH CONTROL a21-vs-a21r [INFO]",
        f"{w} G4 CONTROL vanilla 0.21-vs-0.29 [MEASURED]",
        f"{w} G5 T3 structure a21-vs-a29",
        f"{w} G5 T3 generation a21-vs-a29 [BANDED]",
    ]
    if WORKLOADS[w].subsystem == "steer":
        gates.append(f"{w} G5 T3 control a21-vs-a29 [BANDED]")
    else:
        gates.append(f"{w} G5 T3 capture a21-vs-a29 [BANDED from G4]")
    gates.append(f"{w} G5b ref-vs-a29 [INFO]")
    if WORKLOADS[w].subsystem == "steer":
        gates += [f"steer_small liveness {name}" for name in ARMS]
    return tuple(gates)


EXPECTED_GATES = {name: expected_gates(name)
                  for name in ("hs_small", "qk_small", "steer_small")}


def judge(workload: str, arms: dict, verdicts: list) -> None:
    """Emit every gate for one workload, in the order the argument needs them."""
    L = "layer*.safetensors"      # the captured payload
    ALL = "*.safetensors"         # every channel this workload writes
    subsystem = WORKLOADS[workload].subsystem
    # What the BAND is measured over: the captured payload for hs/qk, and for steer (which
    # writes no layer tensors) everything it does write. Deliberately NOT `ALL` for hs/qk --
    # the relative band gates `layer*.safetensors`, and folding the generation channel's
    # drift into its anchor would bound the captured activations with a number measured on
    # something else.
    payload = L if subsystem != "steer" else ALL

    print(f"\n=== [{TAG}] {workload}: arms "
          f"{ {name: (arms.get(name) is not None) for name in ARMS} } ===", flush=True)

    # ---- the declared tree, per arm, before anything is compared to anything -------
    judge_inventory(workload, arms, verdicts)
    if subsystem == "steer":
        judge_steer_liveness(arms, verdicts)

    # ---- G1 / G2: is MIA inert on each vLLM version? ------------------------------
    # Bit-exact, and FATAL. If either fails, say so under its own heading: it is a bigger
    # finding than T3's own verdict and must not be absorbed into a band.
    #
    # `ALL`, NOT `payload`. These two gates must cover EVERY channel that feeds the band,
    # or a defect in MIA makes the gate that judges it more permissive: the band's G1/G2
    # terms include `g1_lp`/`g2_lp`, measured over `generation.safetensors`, so while these
    # gates covered `layer*.safetensors` only, MIA shifting the generation logprobs on 0.29
    # by 3e-2 would have passed G2, pushed the band to 2 x (3e-2 + 1.166e-06) = 6.0e-2 --
    # still under the 1e-1 ceiling, so no CONTROL ANOMALY and no HEADLINE -- and then passed
    # G5 at the band its own defect had widened. Exactly the shape of the Phase-D `steer-gap`
    # defect, where a budget computed from a floor self-widened to match it.
    #
    # With the gates covering everything, the band's G1/G2 terms can be nonzero ONLY when the
    # corresponding FATAL gate has already failed and the HEADLINE has already fired, so no
    # defect can widen the band without first turning the job red. Verified on job 1749270's
    # stored artifacts: all 24 `generation.safetensors` + `control.safetensors` files are
    # byte-identical for both pairs on all three workloads, so this strengthens the claim
    # rather than changing it (task E1 review, finding 1).
    _compare(f"{workload} G1 MIA-inert-0.21 a21-vs-c21 (bit-exact)",
             arms.get("a21"), arms.get("c21"), name=ALL, verdicts=verdicts, tag=TAG)
    _compare(f"{workload} G2 MIA-inert-0.29 a29-vs-c29 (bit-exact)",
             arms.get("a29"), arms.get("c29"), name=ALL, verdicts=verdicts, tag=TAG)

    # Measured even when bit-exact, because the band is the SUM of these two and the
    # control, and a zero has to be a measured zero.
    g1 = measure(f"{workload} G1 measured", arms.get("a21"), arms.get("c21"),
                 name=payload, verdicts=None, structure_fatal=False)
    g2 = measure(f"{workload} G2 measured", arms.get("a29"), arms.get("c29"),
                 name=payload, verdicts=None, structure_fatal=False)
    g1_lp = measure_logprobs(f"{workload} G1 logprob", arms.get("a21"), arms.get("c21"))
    g2_lp = measure_logprobs(f"{workload} G2 logprob", arms.get("a29"), arms.get("c29"))
    # `steer_small` writes a SECOND logprob file -- `control.safetensors`, the unsteered arm
    # the liveness gate compares against. It needs its own band legs: a cross-version drift
    # confined to the control is a different quantity from one in the steered generation,
    # and averaging them would bound each with the other's noise.
    g1_ctl = g2_ctl = None
    if subsystem == "steer":
        g1_ctl = measure_logprobs(f"{workload} G1 control logprob", arms.get("a21"),
                                  arms.get("c21"), channel="control")
        g2_ctl = measure_logprobs(f"{workload} G2 control logprob", arms.get("a29"),
                                  arms.get("c29"), channel="control")

    # ---- G3: is the on-disk reference reproducible at all? ------------------------
    # Same MIA SHA, same vLLM, same width; different job, different node, different boot.
    # INFO by construction: nothing in this suite establishes cross-boot bit-exactness at
    # 0.21, so this leg is the measurement, not a gate resting on one.
    measure(f"{workload} G3 reference reproducibility ref-vs-a21r [INFO]",
            arms.get("ref"), arms.get("a21r"), name=payload, verdicts=verdicts,
            structure_fatal=False)
    measure_per_layer(f"{workload} G3", arms.get("ref"), arms.get("a21r"), workload)
    measure_logprobs(f"{workload} G3 logprob", arms.get("ref"), arms.get("a21r"))

    # ---- the WIDTH control: why a21 and a21r both exist ---------------------------
    # Same MIA SHA, same vLLM, same job, same node, same boot -- the ONLY difference is the
    # running-batch width (1 vs vLLM's default, which holds all three requests at once). So
    # this number IS the model's batch-variance on this workload, and it is the reason the
    # reference tree (width default) cannot be gated bit-exact against a width-1 arm: T2's
    # own `CONTROL eager alone-vs-batch` fails bit-exactness on every workload in every job
    # that has run it. Reported, never gated: it is a property of the model, not of MIA.
    measure(f"{workload} WIDTH CONTROL a21-vs-a21r [INFO]",
            arms.get("a21"), arms.get("a21r"), name=payload, verdicts=verdicts,
            structure_fatal=False)
    measure_logprobs(f"{workload} WIDTH CONTROL logprob", arms.get("a21"), arms.get("a21r"))

    # ---- G4: the CONTROL. Pure vLLM drift, MIA absent from both sides -------------
    control = measure(f"{workload} G4 CONTROL vanilla 0.21-vs-0.29 [MEASURED]",
                      arms.get("c21"), arms.get("c29"), name=payload, verdicts=verdicts,
                      structure_fatal=True)
    measure_per_layer(f"{workload} G4 CONTROL", arms.get("c21"), arms.get("c29"), workload)
    control_lp = measure_logprobs(f"{workload} G4 CONTROL logprob",
                                  arms.get("c21"), arms.get("c29"))
    control_ctl = None
    if subsystem == "steer":
        control_ctl = measure_logprobs(f"{workload} G4 CONTROL control logprob",
                                       arms.get("c21"), arms.get("c29"), channel="control")

    # ---- the band, derived from the three legs above ------------------------------
    rel_band = derive_band(control, g1, g2, key="rel", ceiling=_REL_CEILING,
                           what=f"{workload} relative")
    row_band = derive_band(control, g1, g2, key="row", ceiling=_ROW_CEILING,
                           what=f"{workload} per-row")
    lp_band = derive_logprob_band(control_lp, g1_lp, g2_lp,
                                  what=f"{workload} logprob")
    ctl_band = (derive_logprob_band(control_ctl, g1_ctl, g2_ctl,
                                    what=f"{workload} control logprob")
                if subsystem == "steer" else None)

    # ---- G5: T3 itself ------------------------------------------------------------
    # STRUCTURE first and separately, so a structural failure is attributable rather than
    # buried in a banded verdict: same requests, same layer files, same tensor keys, same
    # shapes, same row counts, same integer metadata. The infinite bands leave exactly those
    # findings able to fail.
    measure(f"{workload} G5 T3 structure a21-vs-a29", arms.get("a21"), arms.get("a29"),
            name=payload, verdicts=verdicts, structure_fatal=True)
    measure_per_layer(f"{workload} G5 T3", arms.get("a21"), arms.get("a29"), workload)

    # TOKEN IDS BEFORE TENSORS. Across two vLLM versions a greedy argmax can flip at a
    # near-tie and the rows after it are not comparable under any band.
    compare_generation_band_checked(
        f"{workload} G5 T3 generation a21-vs-a29 [BANDED]",
        arms.get("a21"), arms.get("a29"), verdicts=verdicts, bound=lp_band,
        note="cross-version logprob drift, derived from the G4 control measured in this job")

    # `steer_small` has no layer tensors, so without this its UNSTEERED control channel is
    # bit-exact WITHIN a version (G1/G2) and structurally checked ACROSS versions (G5
    # structure, bands `inf`) but never VALUE-banded across them -- a drift confined to the
    # control would fail nothing, while the `*.safetensors` payload made it look covered.
    # It does drift: all six control files differ between 0.21 and 0.29 (task E1 review,
    # finding 9).
    if subsystem == "steer":
        compare_generation_band_checked(
            f"{workload} G5 T3 control a21-vs-a29 [BANDED]",
            arms.get("a21"), arms.get("a29"), verdicts=verdicts, bound=ctl_band,
            note="cross-version drift of the UNSTEERED control channel, derived from the "
                 "same three legs measured on that channel",
            name="control.safetensors")

    if subsystem != "steer":
        if rel_band is None or row_band is None:
            label = f"{workload} G5 T3 capture a21-vs-a29 [BANDED from G4]"
            print(f"VERDICT {TAG} {label}: FAIL (no band could be derived -- see the BAND "
                  f"lines above)", flush=True)
            verdicts.append((label, False, True))
        else:
            _compare_replay_band_checked(
                f"{workload} G5 T3 capture a21-vs-a29 [BANDED from G4]",
                arms.get("a21"), arms.get("a29"), name=L, verdicts=verdicts,
                rel=rel_band, row=row_band, tag=TAG)

    # ---- G5b: the same question asked of the on-disk reference directly ------------
    # INFO: it crosses a job/node boundary AND a width boundary (the reference ran at
    # vLLM's default running batch, every gated arm here runs at width 1, and the model is
    # not batch-invariant), so it is evidence, not a gate.
    measure(f"{workload} G5b ref-vs-a29 [INFO]", arms.get("ref"), arms.get("a29"),
            name=payload, verdicts=verdicts, structure_fatal=False)
    measure_logprobs(f"{workload} G5b logprob", arms.get("ref"), arms.get("a29"))


def adjudicate(workload: str, verdicts: list) -> int:
    """The gate-set assertion + the exit code. Separated so it is hermetically testable.

    Counting only the verdicts that WERE emitted is how a suite reports PASS with four of
    its gates deleted: the list simply gets shorter and nothing notices. This asserts
    against the DECLARED set, in both directions.
    """
    missing, undeclared = gate_problems(EXPECTED_GATES[workload], verdicts)
    for label in missing:
        print(f"MISSING GATE {label}: declared in EXPECTED_GATES but NO verdict was emitted",
              flush=True)
    for label in undeclared:
        print(f"UNDECLARED GATE {label}: emitted a verdict but is not in EXPECTED_GATES -- "
              f"declare it, or it can vanish later without anyone noticing", flush=True)
    gates_ok = not missing and not undeclared
    print(f"VERDICT {TAG} {workload} expected-gates: {'PASS' if gates_ok else 'FAIL'} "
          f"({len(EXPECTED_GATES[workload])} declared, {len(verdicts)} emitted, "
          f"{len(missing)} missing, {len(undeclared)} undeclared)", flush=True)
    verdicts.append((f"{workload} expected-gates", gates_ok, True))

    # The inertness claims get their own heading whatever else happened: a non-inert MIA on
    # either version is a bigger finding than T3's verdict and must not read as a footnote.
    for gate in (f"{workload} G1 MIA-inert-0.21 a21-vs-c21 (bit-exact)",
                 f"{workload} G2 MIA-inert-0.29 a29-vs-c29 (bit-exact)"):
        for label, ok, _fatal in verdicts:
            if label == gate and not ok:
                print(f"HEADLINE {TAG} {label}: FAILED. MIA is NOT bit-inert on this vLLM "
                      f"version, so its captured values are not provably the model's own "
                      f"activations and T3's band -- which is derived assuming inertness -- "
                      f"no longer means what its derivation says. Investigate this BEFORE "
                      f"reading the T3 verdict.", flush=True)

    failed = [label for label, ok, fatal in verdicts if fatal and not ok]
    print(f"VERDICT {TAG} {workload}: {'PASS' if not failed else 'FAIL'} "
          f"({len(failed)} failing gates: {failed})", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T3: the old world against the new one.")
    ap.add_argument("workload", nargs="?", choices=sorted(EXPECTED_GATES))
    ap.add_argument("--print-policy", action="store_true",
                    help="echo the band policy and the layer conventions, then exit (the "
                         "job script calls this before any engine boots, so the log records "
                         "what was enforced)")
    for arm in ARMS:
        ap.add_argument(f"--{arm}", default=None,
                        help=f"artifact tree for the {arm} arm (omitted -> its gates FAIL)")
    args = ap.parse_args(argv)

    if args.print_policy:
        print(f"[{TAG}] layer conventions: {assert_layer_conventions()}", flush=True)
        print(f"[{TAG}] band policy: margin={_MARGIN} ceilings rel={_REL_CEILING:.3e} "
              f"per-row={_ROW_CEILING:.3e} logprob={_LOGPROB_CEILING:.3e}; the band itself "
              f"is DERIVED per workload as margin x (G1 + G4 + G2) and has NO environment "
              f"override", flush=True)
        print(f"[{TAG}] reference tree provenance: {REFERENCE_JOB}", flush=True)
        for name in sorted(EXPECTED_ARTIFACTS):
            print(f"[{TAG}] declared artifacts {name}: "
                  f"{len(EXPECTED_ARTIFACTS[name])} files, "
                  f"{len(EXPECTED_GATES[name])} gates", flush=True)
        return 0

    if not args.workload:
        ap.error("a workload is required unless --print-policy is given")

    print(f"[{TAG}] layer conventions: {assert_layer_conventions()}", flush=True)
    arms = {name: (Path(getattr(args, name)) if getattr(args, name) else None)
            for name in ARMS}
    for name in ARMS:
        print(f"[{TAG}] arm {name} = {arms[name]}", flush=True)

    verdicts: list = []
    judge(args.workload, arms, verdicts)
    return adjudicate(args.workload, verdicts)


if __name__ == "__main__":
    raise SystemExit(main())
