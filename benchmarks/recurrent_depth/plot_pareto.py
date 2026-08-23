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


def _metric_from_results(results: dict, task: str, metric: str) -> Optional[float]:
    if not results:
        return None
    # lm-eval nests as results[task][metric] or results[task][metric,none]
    block = results.get(task) or results.get(f"{task}")
    if block is None:
        # try first key containing task name
        for k, v in results.items():
            if task in k and isinstance(v, dict):
                block = v
                break
    if not isinstance(block, dict):
        return None
    if metric in block and isinstance(block[metric], (int, float)):
        return float(block[metric])
    for k, v in block.items():
        if k.startswith(metric) and isinstance(v, (int, float)):
            return float(v)
    return None


def load_points(results_dir: Path, task: str, metric: str) -> List[Dict[str, Any]]:
    points = []
    files = sorted(results_dir.glob("*.json"))
    for path in files:
        if path.name == "sweep_summary.json":
            continue
        data = json.loads(path.read_text())
        cfg = data.get("config") or {}
        stats = data.get("exit_stats") or {}
        r_bar = stats.get("mean_effective_r")
        if r_bar is None and cfg.get("rho", 0) == 0 and cfg.get("num_steps") is not None:
            r_bar = float(cfg["num_steps"])
        quality = _metric_from_results(data.get("results") or {}, task, metric)
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
                "quality": float(quality),
                "metric": metric,
                "task": task,
            }
        )
    return points


def plot(points: List[Dict[str, Any]], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fixed = [p for p in points if p["arm"] == "fixed"]
    adaptive = [p for p in points if p["arm"] == "adaptive"]
    fixed.sort(key=lambda p: p["mean_effective_r"])
    adaptive.sort(key=lambda p: p["mean_effective_r"])

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    if fixed:
        ax.plot(
            [p["mean_effective_r"] for p in fixed],
            [p["quality"] for p in fixed],
            marker="o",
            linestyle="-",
            label="Fixed depth (ρ=0)",
            color="#1f4e79",
        )
    if adaptive:
        ax.plot(
            [p["mean_effective_r"] for p in adaptive],
            [p["quality"] for p in adaptive],
            marker="s",
            linestyle="--",
            label="Adaptive exit (ρ sweep)",
            color="#c45c26",
        )
    ax.set_xlabel(r"Mean effective recurrence $\bar{r}$")
    ax.set_ylabel("Quality")
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
    args = p.parse_args()

    points = load_points(args.results_dir, args.task, args.metric)
    if not points:
        raise SystemExit(f"No plottable points in {args.results_dir} for {args.task}/{args.metric}")

    table = args.results_dir / "pareto_points.json"
    table.write_text(json.dumps(points, indent=2))
    print(f"wrote {table} ({len(points)} points)")

    out = args.out or (args.results_dir / f"pareto_{args.task}.pdf")
    title = args.title or f"{args.task}: quality vs mean recurrence"
    plot(points, out, title)


if __name__ == "__main__":
    main()
