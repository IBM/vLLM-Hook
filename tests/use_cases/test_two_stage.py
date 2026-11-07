# tests/use_cases/test_two_stage.py
# CPU only, no model download. Checks the cascade calibration rules, the lazy
# two-stage detector and the two_stage_probe analyzer against synthetic probes.
# The "synthetic" tests show the mechanism on generated data. They are not
# evidence about any real model.
import json

import numpy as np
import pytest
import torch

from mia import PluginRegistry, register_plugins
from mia.analyzers.tool_call_risk_analyzer import ToolCallRiskAnalyzer
from mia.analyzers.two_stage_probe_analyzer import TwoStageProbeAnalyzer
from mia.utils.two_stage import (
    TwoStageDetector,
    calibrate_cascade,
    calibrate_cascade_for_fpr,
    cascade_flags,
    evaluate_cascade,
    save_cascade_config,
    threshold_for_fpr,
    threshold_for_recall,
)

HIDDEN = 16


# ---- thresholds -------------------------------------------------------------------------

@pytest.mark.parametrize("recall", [0.1, 0.5, 0.9, 0.95, 1.0])
def test_threshold_for_recall_keeps_the_recall(recall):
    pos = np.random.default_rng(0).normal(size=37)
    t = threshold_for_recall(pos, recall)
    assert np.mean(pos >= t) >= recall
    # strictest such threshold: the next score up loses the recall
    above = np.sort(pos[pos > t])
    if above.size:
        assert np.mean(pos >= above[0]) < recall


def test_threshold_for_recall_handles_float_edge_and_ties():
    pos = np.arange(10, dtype=float)   # 0.9 * 10 is 9.000000000000002 in floating point
    assert threshold_for_recall(pos, 0.9) == 1.0
    assert np.mean(np.array([1, 1, 1, 0.0]) >= threshold_for_recall([1, 1, 1, 0.0], 0.5)) >= 0.5


@pytest.mark.parametrize("fpr", [0.0, 0.05, 0.2, 1.0])
def test_threshold_for_fpr_caps_false_positives(fpr):
    neg = np.random.default_rng(1).normal(size=41)
    neg[:5] = neg[5]   # ties at one value
    t = threshold_for_fpr(neg, fpr)
    assert np.mean(neg >= t) <= fpr + 1e-12


def test_threshold_inputs_are_validated():
    with pytest.raises(ValueError):
        threshold_for_recall([], 0.9)
    with pytest.raises(ValueError):
        threshold_for_recall([0.1], 0.0)
    with pytest.raises(ValueError):
        threshold_for_fpr([0.1], 1.5)


# ---- cascade calibration ----------------------------------------------------------------

def _synthetic(n, seed, informative_stage2=True):
    """Positives score high on both stages. Half the negatives also score high on stage 1
    (the false positives stage 1 cannot separate); stage 2 separates them when informative."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.2
    hard = (~y) & (rng.random(n) < 0.5)
    s1 = rng.normal(size=n) + 2.0 * (y | hard)
    s2 = rng.normal(size=n) + (2.0 * y if informative_stage2 else 0.0)
    return s1, s2, y


def test_calibrated_cascade_reaches_target_recall_on_calibration():
    s1, s2, y = _synthetic(2000, 0)
    for target in (0.8, 0.9, 0.95):
        op = calibrate_cascade(s1, s2, y, target)
        assert op.recall >= target
        assert op == evaluate_cascade(s1, s2, y, op.stage1_threshold, op.stage2_threshold)


def test_cascade_never_flags_what_stage1_passes():
    s1, s2, y = _synthetic(500, 1)
    op = calibrate_cascade(s1, s2, y, 0.9)
    flags = cascade_flags(s1, s2, op.stage1_threshold, op.stage2_threshold)
    assert not np.any(flags & (s1 < op.stage1_threshold))


def test_calibration_never_does_worse_than_stage1_alone():
    # Single-stage stage 1 is one of the candidates (t1 at the target, t2 at or below every positive).
    for seed, informative in ((2, True), (3, False)):
        s1, s2, y = _synthetic(1000, seed, informative)
        op = calibrate_cascade(s1, s2, y, 0.9)
        t = threshold_for_recall(s1[y], 0.9)
        assert op.fpr <= np.mean(s1[~y] >= t) + 1e-12


@pytest.mark.parametrize("seed", range(20))
def test_calibration_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    n = 30
    s1, s2 = rng.integers(0, 6, n).astype(float), rng.integers(0, 6, n).astype(float)  # many ties
    y = rng.random(n) < 0.4
    y[:2] = [True, False]
    target = 0.8
    best = min(
        evaluate_cascade(s1, s2, y, t1, t2).fpr
        for t1 in np.unique(s1) for t2 in np.unique(s2)
        if evaluate_cascade(s1, s2, y, t1, t2).recall >= target
    )
    op = calibrate_cascade(s1, s2, y, target)
    assert op.recall >= target and op.fpr == pytest.approx(best)


def test_synthetic_holdout_cascade_cuts_fpr_when_stage2_is_informative():
    """Synthetic demonstration with thresholds from a calibration split, scored on a fresh split."""
    cal, test = _synthetic(3000, 4), _synthetic(3000, 5)
    op = calibrate_cascade(*cal, 0.9)
    held = evaluate_cascade(*test, op.stage1_threshold, op.stage2_threshold)
    single_t = threshold_for_recall(cal[0][cal[2]], 0.9)
    single_fpr = np.mean(test[0][~test[2]] >= single_t)
    assert held.recall >= 0.85               # recall moves a little off the calibration split
    assert held.fpr < 0.6 * single_fpr       # most stage-1 false positives are removed


def test_max_stage2_rate_is_respected():
    s1, s2, y = _synthetic(1000, 6)
    free = calibrate_cascade(s1, s2, y, 0.8)
    cap = free.stage2_rate - 0.03   # make the cap bind
    op = calibrate_cascade(s1, s2, y, 0.8, max_stage2_rate=cap)
    assert op.stage2_rate <= cap and op.recall >= 0.8 and op.fpr >= free.fpr
    with pytest.raises(ValueError, match="max_stage2_rate"):
        calibrate_cascade(s1, s2, y, 0.95, max_stage2_rate=0.05)


def test_calibration_inputs_are_validated():
    s1, s2, y = _synthetic(200, 7)
    with pytest.raises(ValueError, match="every calibration item"):
        calibrate_cascade(s1, np.where(y, s2, np.nan), y, 0.9)
    with pytest.raises(ValueError, match="both positives and negatives"):
        calibrate_cascade(s1, s2, np.zeros_like(y), 0.9)
    with pytest.raises(ValueError, match="same length"):
        calibrate_cascade(s1, s2[:-1], y, 0.9)


# ---- lazy detector ----------------------------------------------------------------------

def test_detector_runs_stage2_only_on_stage1_flags():
    seen = []

    def stage2(batch):
        seen.append(list(batch))
        return [10.0 if item == "b" else -10.0 for item in batch]

    det = TwoStageDetector(lambda b: [{"a": 1.0, "b": 1.0, "c": -1.0}[i] for i in b], stage2, 0.0, 0.0)
    out = det(["a", "b", "c"])
    assert seen == [["a", "b"]]
    assert out.flags.tolist() == [False, True, False]
    assert out.reached_stage2.tolist() == [True, True, False]
    assert np.isnan(out.stage2_scores[2])

    out_all = det(["a", "b", "c"], score_all=True)
    assert seen[-1] == ["a", "b", "c"] and out_all.flags.tolist() == out.flags.tolist()


def test_detector_skips_stage2_when_nothing_is_flagged():
    det = TwoStageDetector(lambda b: np.full(len(b), -1.0), lambda b: pytest.fail("stage 2 ran"), 0.0, 0.0)
    assert not det(np.zeros((4, 2))).flags.any()


def test_detector_rejects_wrong_stage2_length():
    det = TwoStageDetector(lambda b: np.ones(len(b)), lambda b: [1.0], 0.0, 0.0)
    with pytest.raises(ValueError, match="stage 2 returned"):
        det(np.zeros((3, 2)))


# ---- analyzer ---------------------------------------------------------------------------

def _write_probe(d, layer, threshold, seed):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    params = dict(mean=rng.normal(size=HIDDEN).astype(np.float32),
                  scale=rng.uniform(0.5, 2.0, size=HIDDEN).astype(np.float32),
                  weights=rng.normal(size=HIDDEN).astype(np.float32), bias=np.float32(0.1))
    np.savez(d / "probe.npz", layer=np.int64(layer), **params)
    (d / "probe.json").write_text(json.dumps({"model_name": "test/model", "layer": layer,
                                              "threshold": threshold}))
    return str(d / "probe.npz"), params


def _prob(x, p):
    return 1.0 / (1.0 + np.exp(-(((x - p["mean"]) / p["scale"]) @ p["weights"] + p["bias"])))


def _cache(x14, x21):
    return {"model.layers.13": {"hidden_states": [torch.from_numpy(r) for r in x14], "layer_num": 14},
            "model.layers.20": {"hidden_states": [torch.from_numpy(r) for r in x21], "layer_num": 21}}


def test_analyzer_flags_only_when_both_stages_fire(tmp_path):
    p1, a = _write_probe(tmp_path / "s1", 21, 0.5, 0)
    p2, b = _write_probe(tmp_path / "s2", 14, 0.5, 1)
    rng = np.random.default_rng(2)
    x21, x14 = rng.normal(size=(64, HIDDEN)).astype(np.float32), rng.normal(size=(64, HIDDEN)).astype(np.float32)
    out = TwoStageProbeAnalyzer(str(tmp_path), {}).analyze(
        analyzer_spec={"stage1": {"probe_path": p1, "threshold": 0.5},
                       "stage2": {"probe_path": p2, "threshold": 0.5}},
        probes={"hs_cache": _cache(x14, x21)})
    q1, q2 = _prob(x21, a), _prob(x14, b)
    np.testing.assert_allclose(out["stage1"]["probabilities"], q1, rtol=1e-5)
    reached = q1 >= 0.5
    assert out["reached_stage2"] == reached.tolist()
    assert out["verdicts"] == ["flag" if r and s >= 0.5 else "pass" for r, s in zip(reached, q2)]
    got2 = out["stage2"]["probabilities"]
    assert all((g is None) == (not r) for g, r in zip(got2, reached))
    np.testing.assert_allclose([g for g in got2 if g is not None], q2[reached], rtol=1e-5)
    assert (out["stage1"]["layer"], out["stage2"]["layer"]) == (21, 14)
    assert 0 < reached.sum() < 64   # the fixture exercises both branches


def test_analyzer_reads_cascade_file_and_spec_overrides(tmp_path):
    _write_probe(tmp_path / "s1", 21, 0.5, 0)
    _write_probe(tmp_path / "s2", 14, 0.5, 1)
    (tmp_path / "cascade.json").write_text(json.dumps({
        "stage1": {"probe_path": "s1/probe.npz", "threshold": 0.2},
        "stage2": {"probe_path": "s2/probe.npz", "threshold": 0.7}}))
    rng = np.random.default_rng(3)
    cache = _cache(*rng.normal(size=(2, 8, HIDDEN)).astype(np.float32))
    an = TwoStageProbeAnalyzer(str(tmp_path), {})
    out = an.analyze(analyzer_spec={"cascade_path": str(tmp_path / "cascade.json"), "score_all": True},
                     probes={"hs_cache": cache})
    assert (out["stage1"]["threshold"], out["stage2"]["threshold"]) == (0.2, 0.7)
    assert all(p is not None for p in out["stage2"]["probabilities"])
    out = an.analyze(analyzer_spec={"cascade_path": str(tmp_path / "cascade.json"),
                                    "stage2": {"threshold": 1.0}}, probes={"hs_cache": cache})
    assert out["stage2"]["threshold"] == 1.0 and set(out["verdicts"]) == {"pass"}


def test_analyzer_with_open_stage2_matches_tool_call_risk(tmp_path):
    """Backward compatibility: stage 2 at threshold 0 reproduces the single-probe analyzer."""
    p1, _ = _write_probe(tmp_path / "s1", 21, 0.5, 0)
    p2, _ = _write_probe(tmp_path / "s2", 14, 0.5, 1)
    cache = _cache(*np.random.default_rng(4).normal(size=(2, 32, HIDDEN)).astype(np.float32))
    single = ToolCallRiskAnalyzer(str(tmp_path), {}).analyze(
        analyzer_spec={"probe_path": p1}, probes={"hs_cache": cache})
    two = TwoStageProbeAnalyzer(str(tmp_path), {}).analyze(
        analyzer_spec={"stage1": {"probe_path": p1, "threshold": 0.5},
                       "stage2": {"probe_path": p2, "threshold": 0.0}},
        probes={"hs_cache": cache})
    assert two["verdicts"] == single["verdicts"]
    np.testing.assert_allclose(two["stage1"]["probabilities"], single["probabilities"], rtol=1e-6)


def test_analyzer_errors_name_the_fix(tmp_path):
    p1, _ = _write_probe(tmp_path / "s1", 21, 0.5, 0)
    p2, _ = _write_probe(tmp_path / "s2", 7, 0.5, 1)
    cache = _cache(*np.zeros((2, 2, HIDDEN), dtype=np.float32))
    an = TwoStageProbeAnalyzer(str(tmp_path), {})
    with pytest.raises(ValueError, match="missing stage2"):
        an.analyze(analyzer_spec={"stage1": {"probe_path": p1}}, probes={"hs_cache": cache})
    with pytest.raises(RuntimeError, match="Add layer 7"):
        an.analyze(analyzer_spec={"stage1": {"probe_path": p1, "threshold": 0.5},
                                  "stage2": {"probe_path": p2, "threshold": 0.0}},
                   probes={"hs_cache": cache})


def test_both_analyzers_are_registered():
    register_plugins()
    assert PluginRegistry.get_analyzer("tool_call_risk") is not None
    assert PluginRegistry.get_analyzer("two_stage_probe") is not None


# ---- r1 additions: release contract ------------------------------------------------------

def test_analyzer_refuses_uncalibrated_thresholds(tmp_path):
    """probe.json thresholds belong to each probe alone, so the cascade must be given its own."""
    p1, _ = _write_probe(tmp_path / "s1", 21, 0.5, 0)
    p2, _ = _write_probe(tmp_path / "s2", 14, 0.5, 1)
    cache = _cache(*np.zeros((2, 2, HIDDEN), dtype=np.float32))
    with pytest.raises(ValueError, match="calibrated threshold for stage2"):
        TwoStageProbeAnalyzer(str(tmp_path), {}).analyze(
            analyzer_spec={"stage1": {"probe_path": p1, "threshold": 0.5}, "stage2": {"probe_path": p2}},
            probes={"hs_cache": cache})


def test_calibrate_save_load_round_trip(tmp_path):
    """calibrate_cascade -> save_cascade_config -> analyzer gives the calibrated verdicts."""
    p1, a = _write_probe(tmp_path / "s1", 21, 0.5, 0)
    p2, b = _write_probe(tmp_path / "s2", 14, 0.5, 1)
    rng = np.random.default_rng(5)
    x21, x14 = rng.normal(size=(200, HIDDEN)).astype(np.float32), rng.normal(size=(200, HIDDEN)).astype(np.float32)
    q1, q2 = _prob(x21, a), _prob(x14, b)
    y = (q1 + q2 + rng.normal(scale=0.3, size=200)) > 1.0
    op = calibrate_cascade(q1, q2, y, 0.9)
    cfg = save_cascade_config(str(tmp_path / "cascade.json"), p1, p2, op, data="synthetic")
    assert cfg["stage1"]["probe_path"] == "s1/probe.npz" and cfg["calibration"]["data"] == "synthetic"
    out = TwoStageProbeAnalyzer(str(tmp_path), {}).analyze(
        analyzer_spec={"cascade_path": str(tmp_path / "cascade.json")}, probes={"hs_cache": _cache(x14, x21)})
    want = cascade_flags(q1, q2, op.stage1_threshold, op.stage2_threshold)
    assert out["verdicts"] == ["flag" if f else "pass" for f in want]


@pytest.mark.parametrize("seed", range(20))
def test_fpr_calibration_matches_brute_force(seed):
    rng = np.random.default_rng(300 + seed)
    n = 30
    s1, s2 = rng.integers(0, 6, n).astype(float), rng.integers(0, 6, n).astype(float)
    y = rng.random(n) < 0.4
    y[:2] = [True, False]
    target = 0.25
    pts = [evaluate_cascade(s1, s2, y, t1, t2)
           for t1 in np.unique(s1) for t2 in np.r_[np.unique(s2), np.inf]]
    best = max(p.recall for p in pts if p.fpr <= target)
    op = calibrate_cascade_for_fpr(s1, s2, y, target)
    assert op.fpr <= target + 1e-12 and op.recall == pytest.approx(best)


def test_finite_sample_threshold_is_conformal_and_stricter():
    pos = np.arange(1, 20, dtype=float)                   # 19 positives
    assert threshold_for_recall(pos, 0.9, finite_sample=True) == 2.0    # ceil(20 * 0.9) = 18th largest
    assert threshold_for_recall(pos, 0.9) == 2.0                        # ceil(19 * 0.9) = 18th largest
    assert threshold_for_recall(pos, 0.95, finite_sample=True) == 1.0   # ceil(20 * 0.95) = 19
    assert threshold_for_recall(pos, 0.99, finite_sample=True) == float("-inf")
    s1, s2, y = _synthetic(400, 8)
    loose = calibrate_cascade(s1, s2, y, 0.9)
    strict = calibrate_cascade(s1, s2, y, 0.9, finite_sample=True)
    assert strict.recall >= loose.recall
    with pytest.raises(ValueError, match="too few calibration positives"):
        calibrate_cascade(s1[:20], s2[:20], y[:20] | (np.arange(20) < 2), 0.99, finite_sample=True)
