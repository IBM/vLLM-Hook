"""Two-stage (cascade) detection for probe-style detectors. Numpy only.

Stage 1 is a cheap gate run on every item. Stage 2 runs only on the items
stage 1 flags, and an item is flagged only when both stages fire. Stage 2 can
be anything that returns a score, such as a second probe at another layer, a
probe trained on the items stage 1 flags, or a slower guard model.

Requiring both stages can lower the false-positive rate of stage 1 but can
never raise its recall. The two thresholds trade one against the other, so
they have to be calibrated on data that is disjoint from the data used to
report results. ``calibrate_cascade`` does that for a target recall, and
``threshold_for_fpr`` recalibrates a single score to a target false-positive
rate from negatives alone (for example in-domain benign traffic).

Scores follow one convention throughout: higher means riskier, and an item is
flagged when its score is at or above the threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

Scorer = Callable[[Any], Sequence[float]]

_EPS = 1e-9


def _needed(n: int, rate: float, finite_sample: bool) -> int:
    """Positives that must clear the threshold. ``finite_sample`` uses the split-conformal
    count ceil((n + 1) * rate), so that a new exchangeable positive clears it with
    probability at least ``rate``. That count can exceed ``n``."""
    k = int(np.ceil(((n + 1) if finite_sample else n) * rate - _EPS))
    return max(k, 1)


def threshold_for_recall(pos_scores: Sequence[float], recall: float, finite_sample: bool = False) -> float:
    """Largest threshold that flags at least ``recall`` of the positives.

    With ``finite_sample=True`` the threshold is the split-conformal one, which holds the
    recall in expectation on new positives. It is ``-inf`` (flag everything) when there are
    too few positives for the requested recall.
    """
    s = np.sort(np.asarray(pos_scores, dtype=np.float64))[::-1]
    if s.size == 0:
        raise ValueError("threshold_for_recall needs at least one positive score")
    if not 0.0 < recall <= 1.0:
        raise ValueError(f"recall must be in (0, 1], got {recall}")
    k = _needed(s.size, recall, finite_sample)
    return float(s[k - 1]) if k <= s.size else float("-inf")


def threshold_for_fpr(neg_scores: Sequence[float], fpr: float) -> float:
    """Smallest threshold that flags at most ``fpr`` of the negatives.

    Needs no positives, so a deployment can recalibrate from its own benign
    traffic.
    """
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    if s.size == 0:
        raise ValueError("threshold_for_fpr needs at least one negative score")
    if not 0.0 <= fpr <= 1.0:
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")
    m = int(np.floor(fpr * s.size + _EPS))  # negatives allowed above the threshold
    if m >= s.size:
        return float(s[-1])
    return float(np.nextafter(s[m], np.inf))


def cascade_flags(s1: Sequence[float], s2: Sequence[float], t1: float, t2: float) -> np.ndarray:
    """Boolean flags of the cascade. NaN in ``s2`` (stage 2 not run) never flags."""
    a = np.asarray(s1, dtype=np.float64)
    b = np.asarray(s2, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (a >= t1) & (b >= t2)


@dataclass(frozen=True)
class OperatingPoint:
    """Thresholds of a cascade and what they achieve on one labelled set."""
    stage1_threshold: float
    stage2_threshold: float
    recall: float
    fpr: float
    stage2_rate: float   # fraction of all items that reach stage 2
    n_pos: int
    n_neg: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_cascade(s1, s2, y, t1: float, t2: float) -> OperatingPoint:
    """Recall, false-positive rate and stage-2 rate of fixed thresholds."""
    y = np.asarray(y).astype(bool)
    flags = cascade_flags(s1, s2, t1, t2)
    reached = np.asarray(s1, dtype=np.float64) >= t1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return OperatingPoint(
        stage1_threshold=float(t1),
        stage2_threshold=float(t2),
        recall=float(flags[y].mean()) if n_pos else float("nan"),
        fpr=float(flags[~y].mean()) if n_neg else float("nan"),
        stage2_rate=float(reached.mean()) if reached.size else float("nan"),
        n_pos=n_pos,
        n_neg=n_neg,
    )


def _check_calibration_inputs(s1, s2, y):
    a = np.asarray(s1, dtype=np.float64)
    b = np.asarray(s2, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    if not (a.shape == b.shape == y.shape) or a.ndim != 1:
        raise ValueError("s1, s2 and y must be 1-D and the same length")
    if np.isnan(b).any():
        raise ValueError("calibration needs stage-2 scores on every calibration item")
    if y.all() or not y.any():
        raise ValueError("calibration needs both positives and negatives")
    return a, b, y


def calibrate_cascade_for_fpr(s1, s2, y, target_fpr: float) -> OperatingPoint:
    """Choose (t1, t2) with the highest recall at a false-positive rate of at most ``target_fpr``.

    This is the fixed-alarm-budget counterpart of ``calibrate_cascade``. Every stage-1
    threshold is tried. For each, ``t2`` is the smallest value that keeps the negatives
    flagged by both stages within budget. Ties go to the lower false-positive rate, then to
    the lower stage-2 rate. Report results on items the calibration never saw.
    """
    a, b, y = _check_calibration_inputs(s1, s2, y)
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    allowed = int(np.floor(target_fpr * int((~y).sum()) + _EPS))
    best: Optional[OperatingPoint] = None
    for t1 in np.unique(a)[::-1]:
        gate = a >= t1
        neg = np.sort(b[gate & ~y])[::-1]
        t2 = float(np.nextafter(neg[allowed], np.inf)) if allowed < neg.size else float("-inf")
        point = evaluate_cascade(a, b, y, t1, t2)
        if best is None or (-point.recall, point.fpr, point.stage2_rate) < (-best.recall, best.fpr, best.stage2_rate):
            best = point
    return best


def calibrate_cascade(
    s1,
    s2,
    y,
    target_recall: float,
    max_stage2_rate: Optional[float] = None,
    finite_sample: bool = False,
    stage1_recall: Optional[float] = None,
) -> OperatingPoint:
    """Choose (t1, t2) with the lowest false-positive rate at ``target_recall``.

    ``s1`` and ``s2`` are both stage scores on the same calibration items, so
    stage 2 must be scored on every calibration item, not only on the stage-1
    flags. Every stage-1 threshold that still reaches the target recall is
    tried. For each, ``t2`` is the largest value that keeps overall recall at
    or above the target. Ties go to the lower stage-2 rate. ``max_stage2_rate``
    caps the fraction of items sent to stage 2, for a stage 2 that is
    expensive to run. ``finite_sample=True`` uses the split-conformal count of
    positives (see ``threshold_for_recall``), which is slightly stricter.
    ``stage1_recall`` fixes the stage-1 gate at that recall on the positives
    and searches only ``t2``. Searching one threshold instead of two gives
    the calibration less room to overfit, so recall holds up better on new
    data.

    The returned point describes the calibration set. Report results on items
    the calibration never saw.
    """
    a, b, y = _check_calibration_inputs(s1, s2, y)
    n_pos = int(y.sum())
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    need = _needed(n_pos, target_recall, finite_sample)
    if need > n_pos:
        raise ValueError("too few calibration positives for target_recall with finite_sample=True")

    t1_max = threshold_for_recall(a[y], target_recall, finite_sample)
    candidates = np.unique(a[y][a[y] <= t1_max])[::-1]   # strictest first
    if stage1_recall is not None:
        if not target_recall <= stage1_recall <= 1.0:
            raise ValueError("stage1_recall must be between target_recall and 1")
        candidates = np.array([threshold_for_recall(a[y], stage1_recall, finite_sample)])

    best: Optional[OperatingPoint] = None
    for t1 in candidates:
        gate = a >= t1
        rate = float(gate.mean())
        if max_stage2_rate is not None and rate > max_stage2_rate + _EPS:
            break   # lower t1 only sends more items to stage 2
        passed = np.sort(b[gate & y])[::-1]
        if passed.size < need:
            continue
        t2 = float(passed[need - 1])
        point = evaluate_cascade(a, b, y, t1, t2)
        if best is None or (point.fpr, point.stage2_rate) < (best.fpr, best.stage2_rate):
            best = point
    if best is None:
        raise ValueError(
            "no thresholds reach the target recall"
            + ("" if max_stage2_rate is None else " within max_stage2_rate")
        )
    return best


@dataclass
class CascadeResult:
    flags: np.ndarray            # final verdicts
    stage1_scores: np.ndarray
    stage2_scores: np.ndarray    # NaN where stage 2 did not run
    reached_stage2: np.ndarray


class TwoStageDetector:
    """Runs ``stage1`` on every item and ``stage2`` only on stage-1 flags.

    Each stage maps a batch of items (an array, or a list of any objects) to
    one score per item. Stage 2 can be expensive, since it only sees the items
    stage 1 lets through. With ``score_all=True`` stage 2 runs on every item,
    which is what ``calibrate_cascade`` needs.
    """

    def __init__(self, stage1: Scorer, stage2: Scorer, t1: float, t2: float):
        self.stage1, self.stage2 = stage1, stage2
        self.t1, self.t2 = float(t1), float(t2)

    @classmethod
    def from_operating_point(cls, stage1: Scorer, stage2: Scorer, point: OperatingPoint):
        return cls(stage1, stage2, point.stage1_threshold, point.stage2_threshold)

    @staticmethod
    def _take(items, idx: np.ndarray):
        if isinstance(items, np.ndarray) or hasattr(items, "shape"):
            return items[idx]
        return [items[i] for i in idx]

    def __call__(self, items, score_all: bool = False) -> CascadeResult:
        s1 = np.asarray(self.stage1(items), dtype=np.float64).reshape(-1)
        reached = s1 >= self.t1
        s2 = np.full(s1.shape, np.nan)
        idx = np.arange(s1.size) if score_all else np.flatnonzero(reached)
        if idx.size:
            out = np.asarray(self.stage2(self._take(items, idx)), dtype=np.float64).reshape(-1)
            if out.size != idx.size:
                raise ValueError(f"stage 2 returned {out.size} scores for {idx.size} items")
            s2[idx] = out
        return CascadeResult(
            flags=cascade_flags(s1, s2, self.t1, self.t2),
            stage1_scores=s1,
            stage2_scores=s2,
            reached_stage2=reached,
        )


def save_cascade_config(path: str, stage1_probe, stage2_probe,
                        point: OperatingPoint, **meta) -> dict:
    """Write the ``cascade.json`` that ``two_stage_probe`` reads.

    A stage given as a list of probe paths becomes a ``product`` stage, whose score is the
    product of its probes' probabilities.

    Probe paths inside the directory of ``path`` are stored relative to it, so the folder can
    be moved as a unit. ``meta`` (data, split, target) is stored under "calibration" with the
    operating point measured on the calibration set.
    """
    import json
    import os

    base = os.path.dirname(os.path.abspath(path))

    def rel(p):
        p = os.path.abspath(p)
        return os.path.relpath(p, base) if os.path.commonpath([p, base]) == base else p

    def stage(probe, threshold):
        if isinstance(probe, (list, tuple)):
            return {"product": [{"probe_path": rel(q)} for q in probe], "threshold": threshold}
        return {"probe_path": rel(probe), "threshold": threshold}

    cfg = {
        "stage1": stage(stage1_probe, point.stage1_threshold),
        "stage2": stage(stage2_probe, point.stage2_threshold),
        "calibration": {"operating_point": point.to_dict(), **meta},
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg
