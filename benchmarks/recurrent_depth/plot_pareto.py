#!/usr/bin/env python3
"""Plot quality vs mean effective recurrence from run_lm_eval JSON results.

Example::

    python benchmarks/recurrent_depth/plot_pareto.py \\
        --results-dir benchmarks/recurrent_depth/results/vllm \\
        --task gsm8k --metric exact_match \\
        --out benchmarks/recurrent_depth/results/vllm/pareto_gsm8k.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _task_block(results: dict, task: str) -> Optional[dict]:
    if not results:
        return None
    block = results.get(task) or results.get(f"{task}")
    if block is None:
        for k, v in results.items():
            if task in k and isinstance(v, dict):
                return v
    return block if isinstance(block, dict) else None


def _metric_key(block: dict, metric: str) -> Optional[str]:
    if metric in block and isinstance(block[metric], (int, float)):
        return metric
    for k, v in block.items():
        if k.startswith(metric) and "_stderr" not in k and isinstance(v, (int, float)):
            return k
    return None


def _stderr_key(quality_key: str) -> str:
    if "," in quality_key:
        prefix, suffix = quality_key.split(",", 1)
        return f"{prefix}_stderr,{suffix}"
    return f"{quality_key}_stderr"


def _metric_from_results(
    results: dict, task: str, metric: str
) -> Tuple[Optional[float], Optional[float]]:
    """Return (quality, stderr) from an lm-eval results block."""
    block = _task_block(results, task)
    if block is None:
        return None, None
    key = _metric_key(block, metric)
    if key is None:
        return None, None
    quality = float(block[key])
    stderr_key = _stderr_key(key)
    raw = block.get(stderr_key)
    stderr = float(raw) if isinstance(raw, (int, float)) else None
    return quality, stderr


def load_points(results_dir: Path, task: str, metric: str) -> List[Dict[str, Any]]:
    points = []
    files = sorted(results_dir.glob("*.json"))
    for path in files:
        if path.name in {"sweep_summary.json", "pareto_points.json"}:
            continue
        data = json.loads(path.read_text())
        cfg = data.get("config") or {}
        stats = data.get("exit_stats") or {}
        # Adaptive exit is decode-only, so decode recurrence is the honest axis;
        # older result files only carry the all-token mean.
        r_bar = stats.get("mean_effective_r_decode")
        cost_basis = "decode"
        if r_bar is None:
            r_bar = stats.get("mean_effective_r")
            cost_basis = "all_tokens"
        if r_bar is None and cfg.get("rho", 0) == 0 and cfg.get("num_steps") is not None:
            r_bar = float(cfg["num_steps"])
            cost_basis = "configured"
        quality, stderr = _metric_from_results(data.get("results") or {}, task, metric)
        if r_bar is None or quality is None:
            continue
        points.append(
            {
                "file": path.name,
                "backend": data.get("backend"),
                "arm": cfg.get("arm")
                or ("fixed" if float(cfg.get("rho", 0)) == 0.0 else "adaptive"),
                "rho": cfg.get("rho"),
                "num_steps": cfg.get("num_steps"),
                "mean_effective_r": float(r_bar),
                "cost_basis": cost_basis,
                "mlp_token_steps": stats.get("mlp_token_steps"),
                "attn_token_steps": stats.get("attn_token_steps"),
                "quality": float(quality),
                "quality_stderr": stderr,
                "metric": metric,
                "task": task,
            }
        )
    return points


def _yerr(points: List[Dict[str, Any]], ci_scale: float) -> Optional[List[float]]:
    if any(p.get("quality_stderr") is None for p in points):
        return None
    return [ci_scale * float(p["quality_stderr"]) for p in points]


def plot(
    points: List[Dict[str, Any]],
    out: Path,
    title: str,
    *,
    ci_scale: float,
) -> None:
    import matplotlib.pyplot as plt

    fixed = [p for p in points if p["arm"] == "fixed"]
    adaptive = [p for p in points if p["arm"] == "adaptive"]
    fixed.sort(key=lambda p: p["mean_effective_r"])
    adaptive.sort(key=lambda p: p["mean_effective_r"])

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    series = (
        (fixed, "o", "-", "Fixed Depth (ρ = 0)", "#1f4e79"),
        (adaptive, "s", "--", "Adaptive Exit", "#c45c26"),
    )
    for group, marker, linestyle, label, color in series:
        if not group:
            continue
        ax.errorbar(
            [p["mean_effective_r"] for p in group],
            [p["quality"] for p in group],
            yerr=_yerr(group, ci_scale),
            marker=marker,
            linestyle=linestyle,
            label=label,
            color=color,
            capsize=3,
            elinewidth=1,
        )
    basis = {p.get("cost_basis") for p in points}
    xlabel = (
        r"Mean decode recurrence $\bar{r}_{\mathrm{decode}}$"
        if basis == {"decode"}
        else r"Mean effective recurrence $\bar{r}$"
    )
    ax.set_xlabel(xlabel)
    if abs(ci_scale - 1.96) < 0.02:
        ylabel = "Quality (95% CI)"
    elif abs(ci_scale - 1.0) < 0.02:
        ylabel = r"Quality ($\pm$1 SE)"
    else:
        ylabel = f"Quality (±{ci_scale:g} SE)"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--metric", default="exact_match", help="lm-eval metric key prefix")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", default=None)
    p.add_argument(
        "--ci-scale",
        type=float,
        default=1.96,
        help="multiply lm-eval stderr by this for vertical bars (1.96 ≈ 95% CI)",
    )
    args = p.parse_args()

    points = load_points(args.results_dir, args.task, args.metric)
    if not points:
        raise SystemExit(f"No plottable points in {args.results_dir} for {args.task}/{args.metric}")

    table = args.results_dir / "pareto_points.json"
    table.write_text(json.dumps(points, indent=2))
    print(f"wrote {table} ({len(points)} points)")

    out = args.out or (args.results_dir / f"pareto_{args.task}.pdf")
    title = args.title or f"{args.task}: quality vs mean recurrence"
    plot(points, out, title, ci_scale=args.ci_scale)


if __name__ == "__main__":
    main()
