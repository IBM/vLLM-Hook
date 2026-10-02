"""Compare two artifact trees. Bit-exact by default; tolerance mode for cross-version.

The load-bearing detail is the empty case. Two trees that both contain nothing compare
equal on every tensor, so a run where capture silently did nothing -- the wrong plugin
loaded, a layer filter that matched no module, an allowlist that loaded no plugin at all --
would otherwise report a clean PASS. An empty side is reported as a problem, loudly, and
the exit code is non-zero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


# Kept in sync with capture_workload.LOGPROB_KEYS by hand: this module is deliberately
# import-free so it can be run as a bare script against any two trees.
LOGPROB_KEYS = ("token_logprobs", "cumulative_logprob")


def _tensors(root: Path, name: str = "*.safetensors") -> dict[str, torch.Tensor]:
    out = {}
    for path in sorted(root.rglob(name)):
        for key, value in load_file(str(path)).items():
            out[f"{path.relative_to(root).as_posix()}::{key}"] = value
    return out


def steer_logprob_delta(arm_a: Path, arm_b: Path, *,
                        label_a: str = "generation", label_b: str = "generation") -> float:
    """max |a - b| over the LOGPROB channels of two artifact trees.

    The single number a steering comparison turns on, because steering writes no artifact:
    its only observable is what it does to the output distribution. Two uses, one function:

      * ACROSS ARMS  -- ``steer_logprob_delta(ref_tree, mia_tree)``: how far MIA's steer
        sits from the independent reference's. Must be within the recorded tolerance.
      * LIVENESS     -- ``steer_logprob_delta(tree, tree, label_b="control")``: how far a
        steered run sits from its own unsteered control. Must EXCEED the liveness floor,
        or the steer became a no-op and any across-arm agreement is vacuous.

    Raises when the two sides share no comparable logprob tensor: a silent 0.0 there would
    read as "identical" for trees that in fact contain nothing to compare.
    Consumed by tasks D4 and E1.
    """
    a = {k.replace(f"/{label_a}.safetensors::", "::"): v
         for k, v in _tensors(Path(arm_a), f"{label_a}.safetensors").items()
         if k.rsplit("::", 1)[1] in LOGPROB_KEYS}
    b = {k.replace(f"/{label_b}.safetensors::", "::"): v
         for k, v in _tensors(Path(arm_b), f"{label_b}.safetensors").items()
         if k.rsplit("::", 1)[1] in LOGPROB_KEYS}
    shared = sorted(a.keys() & b.keys())
    if not shared:
        raise RuntimeError(
            f"no comparable logprob tensor between {arm_a} ({label_a}) and {arm_b} "
            f"({label_b}): A={sorted(a)} B={sorted(b)}. A steer comparison over these two "
            f"trees would be vacuous.")
    worst = 0.0
    for key in shared:
        x, y = a[key], b[key]
        if x.shape != y.shape:
            raise RuntimeError(f"{key}: shape {tuple(x.shape)} != {tuple(y.shape)}")
        worst = max(worst, float((x.double() - y.double()).abs().max().item()))
    return worst


def compare(a_root: Path, b_root: Path, *, atol: float | None,
            name: str = "*.safetensors", summary: bool = False) -> list[str]:
    a, b = _tensors(a_root, name), _tensors(b_root, name)
    problems = []

    # Emptiness first: everything below is vacuous when a side has no artifacts.
    for label, root, tensors in (("A", a_root, a), ("B", b_root, b)):
        if not tensors:
            problems.append(
                f"side {label}: no artifacts matching {name!r} found under {root} -- "
                f"this arm produced NOTHING")

    if a.keys() != b.keys():
        only_a = sorted(a.keys() - b.keys())[:5]
        only_b = sorted(b.keys() - a.keys())[:5]
        problems.append(f"key sets differ ({len(a)} vs {len(b)} tensors): "
                        f"only-A={only_a} only-B={only_b}")

    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        if x.dtype != y.dtype:
            problems.append(f"{key}: dtype {x.dtype} != {y.dtype}")
            continue
        if x.shape != y.shape:
            problems.append(f"{key}: shape {tuple(x.shape)} != {tuple(y.shape)}")
            continue
        delta = (x.double() - y.double()).abs().max().item()
        if summary:
            # Printed for EVERY shared key, not only the failing ones: a tolerance is only
            # meaningful next to the deltas actually observed under it.
            print(f"DELTA {key}: max|d|={delta:.3e}")
        if atol is None:
            if not torch.equal(x, y):
                problems.append(f"{key}: NOT bit-exact, max|d|={delta:.3e}")
        elif delta > atol:
            problems.append(f"{key}: max|d|={delta:.3e} > atol={atol:.3e}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--require-bit-exact", action="store_true")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--label", default=None,
                    help="name for this comparison, echoed on the VERDICT line")
    ap.add_argument("--name", default="*.safetensors",
                    help="compare only files matching this glob (e.g. generation.safetensors "
                         "to compare generation alone, ignoring captured payloads)")
    ap.add_argument("--summary", action="store_true",
                    help="print max|d| for every shared key, not only the failing ones")
    args = ap.parse_args()

    atol = None if args.require_bit_exact else args.atol
    a_root, b_root = Path(args.a), Path(args.b)
    problems = compare(a_root, b_root, atol=atol, name=args.name, summary=args.summary)

    label = f" {args.label}" if args.label else ""
    n_a = len(list(a_root.rglob(args.name)))
    n_b = len(list(b_root.rglob(args.name)))
    print(f"COUNT{label}: A={n_a} files ({a_root})  B={n_b} files ({b_root})")
    for problem in problems[:40]:
        print("MISMATCH", problem)
    if len(problems) > 40:
        print(f"MISMATCH ... and {len(problems) - 40} more")
    print(f"VERDICT{label}: {'PASS' if not problems else 'FAIL'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
