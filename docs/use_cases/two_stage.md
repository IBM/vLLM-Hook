# Two-Stage Probe Cascade

**Contributor:** [@rishabhsinha17](https://github.com/rishabhsinha17)

---

## What it does

Lowers the false-positive rate of a detector by adding a second check that only runs on the steps the first one flags. The final verdict flags a step only when both stages fire. Stage 1 is a cheap, high-recall gate. Stage 2 can be another probe on the same capture, or any slower scorer such as a guard model.

Requiring both stages can lower the false-positive rate of stage 1 but can never raise its recall, so calibrate the two thresholds together on held-out data. The second stage only helps if it sees something stage 1 does not.

---

## Pieces

`mia.utils.two_stage` (numpy only):

| Function | Use |
|---|---|
| `calibrate_cascade(s1, s2, y, target_recall, max_stage2_rate=None, finite_sample=False)` | Lowest false-positive rate at a target recall. `max_stage2_rate` caps the share of steps sent to an expensive stage 2. |
| `calibrate_cascade_for_fpr(s1, s2, y, target_fpr)` | Highest recall within a false-positive budget. |
| `threshold_for_recall(pos_scores, recall, finite_sample=False)` | Single-stage threshold for a target recall. `finite_sample=True` uses the split-conformal count. |
| `threshold_for_fpr(neg_scores, fpr)` | Single-stage threshold for a false-positive budget, from negatives alone such as one deployment's benign traffic. |
| `TwoStageDetector(stage1, stage2, t1, t2)` | Runs any two scorers and calls `stage2` only on the stage-1 flags. |
| `save_cascade_config(path, stage1_probe, stage2_probe, point, **meta)` | Writes the `cascade.json` the analyzer reads. |

The `two_stage_probe` analyzer (worker `capture_hs`) runs two linear probes in the `tool_call_risk` artifact format on one capture:

```python
from mia.utils.two_stage import calibrate_cascade, save_cascade_config

point = calibrate_cascade(s1_cal, s2_cal, y_cal, target_recall=0.9)   # held-out calibration scores
save_cascade_config("cascade.json", "stage1/probe.npz", "stage2/probe.npz", point)

out = llm.analyze(analyzer_spec={"cascade_path": "cascade.json"}, run_id=run_id)
# out["verdicts"], out["reached_stage2"], out["stage1"]["probabilities"], out["stage2"]["probabilities"]
```

Both thresholds must come from `cascade.json` or the spec. The analyzer ignores each `probe.json` threshold, because that one was set for the probe on its own. The infer config must capture every probe layer. `"score_all": true` scores stage 2 on every step, which recalibration needs.

---

## Evaluation on the tool-call risk probe

These numbers come from the AgentDojo steps behind the [tool-call risk probe](tool_call_risk.md), all from Qwen2.5-7B-Instruct: 1,729 attacked steps, 358 of them unsafe, and 97 clean steps with no injection. Folds group steps by user task, so training never sees a test task. Thresholds come from out-of-fold scores inside the training split, and every number is the mean of 4 fold seeds. Intervals are paired 95% bootstrap intervals over user tasks for the change from the released probe.

| Detector, 90% recall target | Recall | FPR, attacked but safe | FPR, clean |
|---|---|---|---|
| Released probe alone (layer 28) | 0.899 | 0.394 | 0.387 |
| Stage 2 = injection-presence probe on layer 28 | 0.898 [−0.007, +0.006] | 0.387 [−0.021, +0.003] | 0.134 [−0.335, −0.173] |
| Stage 2 = a second risk probe on layer 21 | 0.885 [−0.035, +0.004] | 0.383 [−0.031, +0.008] | 0.330 [−0.098, −0.015] |

The presence probe is trained only to tell steps with an injection in context from their clean twins, with no outcome label. Requiring it to agree removes about two thirds of the false alarms on clean steps at an unchanged recall, and it reads the layer stage 1 already captures. It cannot help on attacked-but-safe steps, which contain an injection by definition. At a 95% recall target the clean false-positive rate falls from 0.513 to 0.214. With a false-positive budget of 0.2, it falls from 0.175 to 0.077 while recall stays at 0.79 and the attacked-but-safe rate rises from 0.201 to 0.209. The gain is uneven across suites. Clean false-positive rates fall from 0.94 to 0.11 on slack and from 0.19 to 0.01 on travel, but only from 0.52 to 0.41 on banking and from 0.14 to 0.10 on workspace. With only 16 to 40 clean steps per suite, the per-suite rates are rough. Length alone separates attacked from clean steps at 0.52 AUROC, so prompt length does not explain the gain, and in a check on a three-layer version of the presence probe, removing length from its score left the gain unchanged.

A second risk probe barely helps, because the two probes rank steps alike (Spearman 0.81). Even thresholds tuned on the evaluated labels only lower the attacked-but-safe rate from 0.385 to 0.373. The released probe flags about 45% of safe steps whose next action is a tool call, attacked or clean, against 7 to 20% of steps that end in text. So reducing the attacked-but-safe false positives needs a stage 2 that sees the action itself, such as hidden states at the first generated tokens (`hooks_on="both"`), scored before the tool runs. In a simulation that keeps the released probe's real scores, a stage 2 with an AUROC of 0.9 among the flagged steps roughly halves that rate, and one with 0.7 barely moves it.

At one global threshold the false-positive rate also varies by tool suite, from 0.18 on workspace to 0.79 on slack. Thresholds set per suite with `threshold_for_fpr` even this out but lower pooled recall from 0.90 to 0.84 at the same budget.

The evaluation scripts and per-seed results are not published yet, so these numbers cannot be reproduced from public code until they are.

---

## Limitations

- The evaluation covers Qwen2.5-7B-Instruct on AgentDojo only, with 97 clean steps. The clean-step result needs checking on real agent traffic before anyone relies on it.
- The presence probe is a linear probe trained on AgentDojo injections. Other injection styles or tools need their own validation, and a dedicated injection detector can stand in as stage 2 through `TwoStageDetector`.
- Calibrate on data disjoint from what you report, and refit thresholds when the agent, tools or prompt format change.
- Both stages read the last prompt token. Decision-time features need a different capture mode and a new probe.
