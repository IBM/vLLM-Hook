"""
Plot grid benchmark results: four metrics vs number of layers,
one figure per prompt length, three lines per plot:
  - vLLM-Hook (last_token)
  - vLLM-Hook (all_tokens)
  - vLLM Eagle (native, last_token)

Usage:
  python benchmarks/plot_grid_results.py --input grid_results.csv --output-dir plots/
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _float(val):
    return float(val) if val not in ("", "None", None) else None


def extract(rows, prompt_len, system, token_mode):
    """Return list of dicts with n_layers and all four metrics + stds, sorted by n_layers."""
    points = []
    for r in rows:
        if (int(r["prompt_len"]) != prompt_len
                or r["system"] != system
                or r["token_mode"] != token_mode):
            continue
        if r["gen_lat_mean"] in ("", "None", None):
            continue
        points.append({
            "n_layers":       len(eval(r["layers"])),
            "gen_lat":        _float(r["gen_lat_mean"]) * 1000,
            "gen_lat_std":    (_float(r["gen_lat_std"]) or 0.0) * 1000,
            "total_lat":      _float(r["total_lat_mean"]) * 1000,
            "total_lat_std":  (_float(r["total_lat_std"]) or 0.0) * 1000,
            "peak_mem":       _float(r["peak_mem_mean"]),
            "peak_mem_std":   _float(r["peak_mem_mean"]) and 0.0,  # no std column for mem
            "art_kb":         _float(r["art_kb_mean"]),
            "art_kb_std":     0.0,
        })
    points.sort(key=lambda x: x["n_layers"])
    return points


METRICS = [
    ("gen_lat",   "gen_lat_std",  "Generation latency (ms)"),
    ("total_lat", "total_lat_std","Total latency incl. read (ms)"),
    ("peak_mem",  None,           "GPU memory in use (MB)"),
    ("art_kb",    None,           "Artifact size (KB)"),
]

SERIES = [
    ("hook",   "last_token", "vLLM-Hook (last_token)", "#1f77b4", "-",  "o"),
    ("hook",   "all_tokens", "vLLM-Hook (all_tokens)", "#ff7f0e", "--", "s"),
    ("native", "last_token", "vLLM Eagle (native)",    "#2ca02c", "-.", "^"),
]


def plot_all(rows, prompt_lens, output_dir):
    n_rows = len(prompt_lens)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle("vLLM-Hook vs Native vLLM Eagle", fontsize=22)

    for row_idx, prompt_len in enumerate(prompt_lens):
        for col_idx, (metric, std_col, ylabel) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]

            for system, token_mode, label, color, ls, marker in SERIES:
                points = extract(rows, prompt_len, system, token_mode)
                valid = [p for p in points if p[metric] is not None]
                if not valid:
                    continue
                xs   = [p["n_layers"] for p in valid]
                ys   = [p[metric]     for p in valid]
                errs = [p[std_col]    for p in valid] if std_col else None

                ax.plot(xs, ys, label=label, color=color, linestyle=ls, marker=marker, markersize=5)
                if errs and any(e > 0 for e in errs):
                    ax.fill_between(xs,
                                    [y - e for y, e in zip(ys, errs)],
                                    [y + e for y, e in zip(ys, errs)],
                                    color=color, alpha=0.15)

            if row_idx == 0:
                ax.set_title(ylabel, fontsize=16)
            if col_idx == 0:
                ax.set_ylabel(f"prompt_len={prompt_len}\n{ylabel}", fontsize=14)
            else:
                ax.set_ylabel(ylabel, fontsize=14)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Number of layers extracted", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
            ax.ticklabel_format(useOffset=False, style="plain", axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "grid_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="grid_results.csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    rows = load(args.input)
    prompt_lens = sorted(set(int(r["prompt_len"]) for r in rows))
    plot_all(rows, prompt_lens, args.output_dir)


if __name__ == "__main__":
    main()
