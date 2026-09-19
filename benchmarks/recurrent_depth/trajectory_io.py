"""Compact recurrent-trajectory I/O for threshold calibration.

One parquet row = one (forward, iteration, token) observation with flat
scalar metrics. JSON run files only keep a short summary + parquet path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

# Stable column order for offline analysis / threshold sweeps.
META_COLUMNS: tuple[str, ...] = (
    "forward_index",
    "iteration",
    "token_index",
    "is_decode",
    "has_prediction_metrics",
)

LATENT_COLUMNS: tuple[str, ...] = (
    "hidden_delta",
    "hidden_norm",
    "normalized_displacement",
    "cosine_distance",
    "normalized_acceleration",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "entropy",
    "entropy_delta",
    "top1_confidence",
    "logit_margin",
    "top1_token",
    "top1_stability",
    "predictive_kl",
)

ALL_METRIC_COLUMNS: tuple[str, ...] = LATENT_COLUMNS + PREDICTION_COLUMNS


def summarize_trajectory_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Tiny JSON-safe summary for run / sweep index files."""
    if not rows:
        return {
            "n_rows": 0,
            "n_forwards": 0,
            "n_iterations": 0,
            "n_decode_rows": 0,
            "n_prediction_rows": 0,
            "columns": list(META_COLUMNS + ALL_METRIC_COLUMNS),
        }
    forwards = {int(r["forward_index"]) for r in rows}
    iterations = {int(r["iteration"]) for r in rows}
    return {
        "n_rows": len(rows),
        "n_forwards": len(forwards),
        "n_iterations": len(iterations),
        "n_decode_rows": sum(1 for r in rows if r.get("is_decode")),
        "n_prediction_rows": sum(1 for r in rows if r.get("has_prediction_metrics")),
        "columns": list(META_COLUMNS + ALL_METRIC_COLUMNS),
    }


def trajectory_rows_to_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a flat DataFrame with a stable schema for calibration."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        cols = list(META_COLUMNS + ALL_METRIC_COLUMNS)
        return pd.DataFrame(columns=cols)

    for col in META_COLUMNS + ALL_METRIC_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    ordered = list(META_COLUMNS + ALL_METRIC_COLUMNS)
    extras = [c for c in frame.columns if c not in ordered]
    frame = frame[ordered + extras]

    for col in ("forward_index", "iteration", "token_index"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    for col in ("is_decode", "has_prediction_metrics"):
        frame[col] = frame[col].astype("boolean")
    for col in ALL_METRIC_COLUMNS:
        if col == "top1_token":
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def write_trajectory_parquet(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> Dict[str, Any]:
    """Write flat trajectory rows to parquet and return a JSON summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = trajectory_rows_to_frame(rows)
    frame.to_parquet(path, index=False)
    summary = summarize_trajectory_rows(rows)
    summary["path"] = str(path)
    return summary
