#!/usr/bin/env python3
"""Offline stopping-signal calibration from full-depth trajectory parquet.

Reads a ``--capture-trajectory`` run (ideally ``rho=0`` so every token runs the
full recurrence), labels each decode token's *earliest safe depth*, then replays
candidate exit policies against that trajectory to trade false exits against
depth savings.

The replay is a counterfactual on the full-depth trajectory: it assumes exiting
at depth ``d`` yields the depth-``d`` latent. That is the standard calibration
assumption and is what makes threshold selection possible without one run per
threshold.

Pipeline placement::

    run_lm_eval (--capture-trajectory, rho=0)  →  trajectories/*.parquet
    analyze_signals.py (this module)           →  signal_report.json + CalibrationManifest
    run_lm_eval (--policy-manifest)             →  real adaptive eval + exit_stats
    plot_pareto.py                             →  quality vs mean_effective_r_decode

Function flow inside this module::

    load_trajectory  → decode-only parquet rows
    to_arrays        → [n_tokens, n_iterations] metric matrices (+ contraction)
    earliest_safe_depth → oracle label per token (when top-1 stabilizes)
    threshold_grid + sweep_single/sweep_pairs → replay each candidate policy
    replay + score   → exit depth vs false-exit rate / depth savings
    best_under_constraint → pick policy under --max-false-exit
    CalibrationManifest   → frozen thresholds for run_lm_eval --policy-manifest

Example::

    python benchmarks/recurrent_depth/analyze_signals.py \\
        --trajectory benchmarks/recurrent_depth/results/vllm/trajectories/vllm_fixed_rho0.0_r32.parquet \\
        --max-false-exit 0.02 --pairs \\
        --out-json benchmarks/recurrent_depth/results/vllm/signal_report.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "vllm_hook_plugins"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vllm_hook_plugins.protocols.recurrent_policy import (  # pyright: ignore[reportMissingImports]
    READOUT_METRICS,
    CalibrationManifest,
    MetricCondition,
    Readout,
    StoppingPolicyConfig,
    readout_column,
    readout_direction,
    CONTRACTION_COLUMN
)

TRAJECTORY_KEYS = ("forward_index", "token_index")

@dataclass(frozen=True)
class Candidate:
    """One replayable policy: conditions plus loop-control settings."""

    conditions: Tuple[Tuple[Readout, float], ...]
    combine: str = "all"
    min_steps: int = 1
    patience: int = 1

    @property
    def label(self) -> str:
        sep = "&" if self.combine == "all" else "|"
        body = sep.join(
            f"{r.value}{'<' if readout_direction(r) == 'lt' else '>='}{t:g}"
            for r, t in self.conditions
        )
        return f"{body} (min_steps={self.min_steps},patience={self.patience})"

    def to_policy(self) -> StoppingPolicyConfig:
        return StoppingPolicyConfig(
            enabled=True,
            conditions=tuple(
                MetricCondition(readout, float(threshold))
                for readout, threshold in self.conditions
            ),
            combine="all" if self.combine == "all" else "any",
            min_steps=self.min_steps,
            patience=self.patience,
        )


def load_trajectory(paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate parquet trajectories, keeping decode rows and unique ids."""
    frames = []
    for offset, path in enumerate(paths):
        frame = pd.read_parquet(path)
        if "is_decode" in frame.columns:
            frame = frame[frame["is_decode"].fillna(True).astype(bool)]
        # Keep forward ids disjoint across files so tokens never collide.
        frame = frame.assign(forward_index=frame["forward_index"] + offset * 10**7)
        frames.append(frame)
    if not frames:
        raise SystemExit("no trajectory rows found")
    return pd.concat(frames, ignore_index=True)


def to_arrays(frame: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], int]:
    """Pivot to ``{metric: [n_tokens, n_iterations]}`` float arrays (NaN = invalid)."""
    if frame.empty:
        raise SystemExit("trajectory contains no decode rows")
    n_iter = int(frame["iteration"].max()) + 1  # pyright: ignore[reportArgumentType]
    keys = list[Literal['forward_index', 'token_index']](TRAJECTORY_KEYS)
    metric_columns = [
        c
        for c in frame.columns
        if c not in set(keys) | {"iteration", "is_decode", "has_prediction_metrics"}
    ]
    arrays: Dict[str, np.ndarray] = {}
    for column in metric_columns:
        pivot = frame.pivot_table(
            index=keys, columns="iteration", values=column, aggfunc="last"
        ).reindex(columns=range(n_iter))
        arrays[column] = pivot.to_numpy(dtype=float)
    arrays[CONTRACTION_COLUMN] = _contraction_array(arrays)
    return arrays, n_iter


def _contraction_array(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Relative remaining movement implied by the observed contraction rate."""
    delta = arrays["hidden_delta"]
    h_norm = arrays["hidden_norm"]
    previous = np.full_like(delta, np.nan)
    previous[:, 1:] = delta[:, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = delta / np.clip(previous, 1e-9, None)
        r_safe = np.clip(r_hat, None, 0.999)
        remaining = delta * r_safe / (1.0 - r_safe) # Assuming constant geometric contraction rate
        relative = remaining / np.clip(h_norm, 1e-9, None)
    # Non-converging steps must never satisfy a contraction threshold.
    return np.where(r_hat < 1.0, relative, np.inf)


def earliest_safe_depth(
    arrays: Dict[str, np.ndarray],
    *,
    kl_tolerance: Optional[float] = None,
    margin_fraction: Optional[float] = None,
) -> np.ndarray:
    """Smallest depth whose prediction already matches the full-depth outcome.

    A depth ``d`` is safe when every later iteration keeps the final top-1 token
    (optionally also a small predictive KL and a margin close to the final one).
    """
    top1 = arrays["top1_token"]
    final_top1 = top1[:, -1]
    unstable = top1 != final_top1[:, None]

    if kl_tolerance is not None and "predictive_kl" in arrays:
        kl = arrays["predictive_kl"]
        unstable |= np.nan_to_num(kl, nan=0.0) > kl_tolerance
    if margin_fraction is not None and "logit_margin" in arrays:
        margin = arrays["logit_margin"]
        final_margin = margin[:, -1]
        unstable |= margin < margin_fraction * final_margin[:, None]

    n_iter = top1.shape[1]
    index = np.arange(n_iter)[None, :]
    last_unstable = np.where(unstable.any(axis=1), (index * unstable).max(axis=1), -1)
    # Depths are 1-based: the iteration after the last unstable one.
    return last_unstable + 2


def replay(
    arrays: Dict[str, np.ndarray],
    candidate: Candidate,
    n_iter: int,
) -> np.ndarray:
    """Return the exit depth each token would take under ``candidate``."""
    n_tokens = arrays["hidden_delta"].shape[0]
    hit = np.ones((n_tokens, n_iter), dtype=bool) if candidate.combine == "all" else (
        np.zeros((n_tokens, n_iter), dtype=bool)
    )
    for readout, threshold in candidate.conditions:
        values = arrays[readout_column(readout)]
        valid = ~np.isnan(values)
        if readout_direction(readout) == "lt":
            condition = (values < threshold) & valid
        else:
            condition = (values >= threshold) & valid
        hit = (hit & condition) if candidate.combine == "all" else (hit | condition)

    # The runtime analyzer only tests conditions once iteration >= min_steps.
    hit[:, :candidate.min_steps] = False

    run = np.zeros(n_tokens, dtype=np.int64)
    exit_index = np.full(n_tokens, -1, dtype=np.int64)
    for j in range(n_iter):
        run = np.where(hit[:, j], run + 1, 0)
        fired = (run >= candidate.patience) & (exit_index < 0)
        exit_index[fired] = j
    return np.where(exit_index < 0, n_iter, exit_index + 1)


def score(depths: np.ndarray, safe: np.ndarray, n_iter: int) -> Dict[str, float]:
    early = depths < safe
    return {
        "n_tokens": int(depths.size),
        "false_exit_rate": float(early.mean()),
        "mean_depth": float(depths.mean()),
        "median_depth": float(np.median(depths)),
        "depth_savings": float(1.0 - depths.mean() / n_iter),
        "wasted_depth": float(np.clip(depths - safe, 0, None).mean()),
        "no_exit_rate": float((depths >= n_iter).mean()),
    }


def threshold_grid(
    arrays: Dict[str, np.ndarray],
    readout: Readout,
    n_quantiles: int,
) -> List[float]:
    """Data-driven candidate thresholds so no constants are hardcoded."""
    values = arrays[readout_column(readout)]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    quantiles = np.linspace(0.02, 0.98, n_quantiles)
    return sorted({float(v) for v in np.quantile(finite, quantiles)})


def available_readouts(arrays: Dict[str, np.ndarray]) -> List[Readout]:
    readouts = [Readout.CONTRACTION]
    readouts += [readout for readout in READOUT_METRICS if readout_column(readout) in arrays
                 and np.isfinite(arrays[readout_column(readout)]).any()]
    return readouts


def sweep_single(
    arrays: Dict[str, np.ndarray],
    safe: np.ndarray,
    n_iter: int,
    readouts: Sequence[Readout],
    *,
    n_quantiles: int,
    min_steps: int,
    patience: int,
) -> List[Dict[str, Any]]:
    rows = []
    for readout in readouts:
        for threshold in threshold_grid(arrays, readout, n_quantiles):
            candidate = Candidate(
                conditions=((readout, threshold),),
                min_steps=min_steps,
                patience=patience,
            )
            rows.append(_row(arrays, safe, n_iter, candidate))
    return rows


def sweep_pairs(
    arrays: Dict[str, np.ndarray],
    safe: np.ndarray,
    n_iter: int,
    readouts: Sequence[Readout],
    *,
    n_quantiles: int,
    min_steps: int,
    patience: int,
) -> List[Dict[str, Any]]:
    rows = []
    for first, second in itertools.combinations(readouts, 2):
        grid_a = threshold_grid(arrays, first, n_quantiles)
        grid_b = threshold_grid(arrays, second, n_quantiles)
        for threshold_a, threshold_b in itertools.product(grid_a, grid_b):
            candidate = Candidate(
                conditions=((first, threshold_a), (second, threshold_b)),
                combine="all",
                min_steps=min_steps,
                patience=patience,
            )
            rows.append(_row(arrays, safe, n_iter, candidate))
    return rows


def _row(
    arrays: Dict[str, np.ndarray],
    safe: np.ndarray,
    n_iter: int,
    candidate: Candidate,
) -> Dict[str, Any]:
    depths = replay(arrays, candidate, n_iter)
    return {
        "label": candidate.label,
        "readouts": [r.value for r, _ in candidate.conditions],
        "thresholds": [float(t) for _, t in candidate.conditions],
        "combine": candidate.combine,
        "min_steps": candidate.min_steps,
        "patience": candidate.patience,
        **score(depths, safe, n_iter),
    }


def best_under_constraint(
    rows: Sequence[Dict[str, Any]],
    max_false_exit: float,
) -> Optional[Dict[str, Any]]:
    feasible = [r for r in rows if r["false_exit_rate"] <= max_false_exit]
    if not feasible:
        return None
    return min(feasible, key=lambda r: (r["mean_depth"], r["false_exit_rate"]))


def candidate_from_row(row: Dict[str, Any]) -> Candidate:
    return Candidate(
        conditions=tuple(
            (Readout(name), float(threshold))
            for name, threshold in zip(row["readouts"], row["thresholds"])
        ),
        combine=row["combine"],
        min_steps=int(row["min_steps"]),
        patience=int(row["patience"]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--trajectory", type=Path, nargs="+", required=True)
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--quantiles", type=int, default=13, help="thresholds per readout")
    p.add_argument("--max-false-exit", type=float, default=0.02)
    p.add_argument("--pairs", action="store_true", help="also sweep 2-readout AND policies")
    p.add_argument("--stability-kl", type=float, default=None, help="tighter safe-depth label")
    p.add_argument("--stability-margin-frac", type=float, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--top", type=int, default=10, help="rows to print / keep per sweep")
    p.add_argument("--write-manifest", type=Path, default=None)
    p.add_argument("--checkpoint-revision", default="tomg-group-umd/huginn-0125")
    p.add_argument("--precision", default="bfloat16")
    p.add_argument("--domain", default="gsm8k")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_trajectory(args.trajectory)
    arrays, n_iter = to_arrays(frame)
    safe = earliest_safe_depth(
        arrays,
        kl_tolerance=args.stability_kl,
        margin_fraction=args.stability_margin_frac,
    )
    readouts = available_readouts(arrays)

    rows = sweep_single(
        arrays,
        safe,
        n_iter,
        readouts,
        n_quantiles=args.quantiles,
        min_steps=args.min_steps,
        patience=args.patience,
    )
    pair_rows: List[Dict[str, Any]] = []
    if args.pairs:
        pair_rows = sweep_pairs(
            arrays,
            safe,
            n_iter,
            readouts,
            n_quantiles=args.quantiles,
            min_steps=args.min_steps,
            patience=args.patience,
        )

    all_rows = rows + pair_rows
    best = best_under_constraint(all_rows, args.max_false_exit)
    best_per_readout = {}
    for readout in readouts:
        subset = [r for r in rows if r["readouts"] == [readout.value]]
        choice = best_under_constraint(subset, args.max_false_exit)
        best_per_readout[readout.value] = choice if choice is not None else {}

    oracle = {
        "n_tokens": int(safe.size),
        "recurrence_cap": n_iter,
        "mean_earliest_safe_depth": float(safe.mean()),
        "median_earliest_safe_depth": float(np.median(safe)),
        "oracle_depth_savings": float(1.0 - safe.mean() / n_iter),
    }

    print(
        f"tokens={oracle['n_tokens']} cap={n_iter} "
        f"oracle mean safe depth={oracle['mean_earliest_safe_depth']:.2f} "
        f"(savings {oracle['oracle_depth_savings']:.1%})"
    )
    print(f"\nbest single readouts (false exit <= {args.max_false_exit:.1%}):")
    for name, row in sorted(best_per_readout.items(), key=lambda kv: kv[1]["mean_depth"]):
        print(
            f"  {name:<18} thr={row['thresholds'][0]:<12.6g} "
            f"mean_depth={row['mean_depth']:.2f} "
            f"savings={row['depth_savings']:.1%} "
            f"false_exit={row['false_exit_rate']:.2%}"
        )
    if best is not None:
        print(f"\nselected: {best['label']}")
        print(
            f"  mean_depth={best['mean_depth']:.2f} savings={best['depth_savings']:.1%} "
            f"false_exit={best['false_exit_rate']:.2%} no_exit={best['no_exit_rate']:.1%}"
        )
    else:
        print("\nno policy met the false-exit constraint; loosen --max-false-exit")

    def _top(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        feasible = [r for r in items if r["false_exit_rate"] <= args.max_false_exit]
        return sorted(feasible, key=lambda r: r["mean_depth"])[: args.top]

    report = {
        "trajectories": [str(p) for p in args.trajectory],
        "safe_depth_label": {
            "stability_kl": args.stability_kl,
            "stability_margin_frac": args.stability_margin_frac,
        },
        "oracle": oracle,
        "constraint": {"max_false_exit_rate": args.max_false_exit},
        "best": best,
        "best_per_readout": best_per_readout,
        "top_single": _top(rows),
        "top_pairs": _top(pair_rows),
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out_json}")

    if args.write_manifest:
        if best is None:
            raise SystemExit("cannot write a manifest without a feasible policy")
        manifest = CalibrationManifest(
            schema_version=1,
            checkpoint_revision=args.checkpoint_revision,
            recurrence_cap=n_iter,
            precision=args.precision,
            domain=args.domain,
            policy=candidate_from_row(best).to_policy(),
        )
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(manifest.to_dict(), indent=2))
        print(f"wrote {args.write_manifest}")


if __name__ == "__main__":
    main()
