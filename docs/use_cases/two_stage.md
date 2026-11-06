# Two-Stage Probe Cascade

**Contributor:** [@rishabhsinha17](https://github.com/rishabhsinha17)

---

## What it does

Lowers the false-positive rate of a probe by adding a second check that only runs on the steps the first probe flags. The final verdict flags a step only when both stages fire. Stage 1 is a cheap, high-recall gate, and stage 2 can be a probe at another layer, a probe trained on the steps stage 1 lets through, or a slower detector such as a guard model.

Requiring both stages can lower the false-positive rate of stage 1, but it can never raise its recall. The thresholds trade one against the other, so calibrate them together on held-out data for a target recall.

---

## Pieces

`mia.utils.two_stage` (numpy only):

| Function | Use |
|---|---|
| `calibrate_cascade(s1, s2, y, target_recall, max_stage2_rate=None)` | Picks stage thresholds with the lowest false-positive rate at the target recall on calibration data. `max_stage2_rate` caps the share of steps sent to an expensive stage 2. |
| `threshold_for_recall(pos_scores, recall)` | Single-stage threshold for a target recall. |
| `threshold_for_fpr(neg_scores, fpr)` | Threshold for a target false-positive rate from negatives alone, such as one deployment's benign traffic. |
| `TwoStageDetector(stage1, stage2, t1, t2)` | Runs any two scorers and calls `stage2` only on the stage-1 flags. |

`two_stage_probe` analyzer (worker `capture_hs`) runs two linear probes in the `tool_call_risk` artifact format on one capture:

```python
spec = {"stage1": {"probe_path": "stage1/probe.npz"},
        "stage2": {"probe_path": "stage2/probe.npz"}}
# or {"cascade_path": "cascade.json"} with the same keys plus calibrated thresholds
out = llm.analyze(analyzer_spec=spec, run_id=run_id)
# out["verdicts"], out["reached_stage2"], out["stage1"]["probabilities"], out["stage2"]["probabilities"]
```

Each stage's threshold defaults to its `probe.json` and can be overridden in the spec. The infer config must capture both probe layers. `"score_all": true` scores stage 2 on every step, which recalibration needs.

---

## Evaluation on the tool-call risk probe

These numbers come from the AgentDojo steps behind the [tool-call risk probe](tool_call_risk.md): 1,729 attacked steps, 358 of them unsafe, plus 97 clean steps, all from Qwen2.5-7B-Instruct. Folds group steps by user task, so training never sees a test task. Thresholds come from out-of-fold scores inside the training split, the target recall is 90%, and every number is the mean of 4 fold seeds.

| Detector | Recall | FPR, attacked but safe | FPR, clean | Steps reaching stage 2 |
|---|---|---|---|---|
| Released probe alone (layer 28) | 0.899 | 0.394 | 0.387 | 100% |
| Cascade, stage 2 = layer-21 probe | 0.883 | 0.382 | 0.332 | 60% |
| Cascade, stage 2 trained on stage-1 flags | 0.891 | 0.384 | 0.381 | 52% |

On this data a second probe on the same prompt changes the false-positive rate by about one point, and the 95% bootstrap intervals over user tasks include zero. At a 95% recall target the layer-21 cascade lowers it from 0.562 to 0.535, and averaging the two probes' scores lowers it to 0.516. Both cost about one point of recall. The prefill views share most of their information, so a useful stage 2 needs a different signal. Candidates are hidden states at the first generated tokens of the action (`hooks_on="both"`), scored before the tool runs, or an independent detector through `TwoStageDetector`.

At one global threshold the false-positive rate also varies by tool suite, from 0.18 on workspace to 0.79 on slack. Thresholds set per suite with `threshold_for_fpr` even this out but lower pooled recall from 0.90 to 0.84 at the same target rate. Recalibrate per deployment if its false-positive budget is fixed and pooled recall matters less.

The evaluation scripts and per-seed results are not published yet, so these numbers cannot be reproduced from public code until they are.

---

## Limitations

- The cascade only removes false positives that stage 2 can tell apart. With two probes on the same prompt it barely helps, as the table shows.
- Calibrate on data disjoint from what you report, and refit thresholds when the agent, tools or prompt format change.
- Both stages read the last prompt token. Decision-time features need a different capture mode and a new probe.
