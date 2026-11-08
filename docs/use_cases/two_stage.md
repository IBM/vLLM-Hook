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
| `calibrate_cascade(s1, s2, y, target_recall, max_stage2_rate=None, finite_sample=False, stage1_recall=None)` | Lowest false-positive rate at a target recall. `max_stage2_rate` caps the share of steps sent to an expensive stage 2. `stage1_recall` fixes the stage-1 gate and searches only the stage-2 threshold, which holds recall better on new data. |
| `calibrate_cascade_for_fpr(s1, s2, y, target_fpr)` | Highest recall within a false-positive budget. |
| `threshold_for_recall(pos_scores, recall, finite_sample=False)` | Single-stage threshold for a target recall. `finite_sample=True` uses the split-conformal count. |
| `threshold_for_fpr(neg_scores, fpr)` | Single-stage threshold for a false-positive budget, from negatives alone such as one deployment's benign traffic. |
| `TwoStageDetector(stage1, stage2, t1, t2)` | Runs any two scorers and calls `stage2` only on the stage-1 flags. |
| `save_cascade_config(path, stage1_probe, stage2_probe, point, **meta)` | Writes the `cascade.json` the analyzer reads. |

The `two_stage_probe` analyzer (worker `capture_hs`) runs linear probes in the `tool_call_risk` artifact format on one capture. A stage is one probe or a product of probes (`{"product": [...]}`). Each probe reads either the last prompt token (`"position": "prompt_last"`, the default) or the mean of the states for the first generated tokens (`"decode_mean"`), as set in its probe.json or the spec:

```python
from mia.utils.two_stage import calibrate_cascade, save_cascade_config

point = calibrate_cascade(s1_cal, s2_cal, y_cal, target_recall=0.9)   # held-out calibration scores
save_cascade_config("cascade.json", "stage1/probe.npz", "stage2/probe.npz", point)

out = llm.analyze(analyzer_spec={"cascade_path": "cascade.json"}, run_id=run_id)
# out["verdicts"], out["reached_stage2"], out["stage1"]["probabilities"], out["stage2"]["probabilities"]
```

Both thresholds must come from `cascade.json` or the spec. The analyzer ignores each `probe.json` threshold, because that one was set for the probe on its own. The infer config must capture every probe layer. `"score_all": true` scores stage 2 on every step, which recalibration needs. A `decode_mean` probe needs each request's own capture, so generate with `extra_args={"hooks_on": "both"}` and `merge_probes=False`, then pass `probes=[o.probes for o in outputs]`. The default merged capture keeps only the prompt passes.

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

## Other models, tool suites and attack templates

The same pipeline ran on two more 8B models: ibm-granite/granite-3.1-8b-instruct and NousResearch/Hermes-3-Llama-3.1-8B, with Qwen2.5-7B-Instruct rerun alongside. For each model one GPU pass captures the states and continues the same generation, and the labels describe the exact action the features saw. The probes read the model's last decoder layer (28, 40 and 32). The evaluation was fixed before any of these captures existed. Split A groups folds by user task, as above, split B holds out a whole tool suite, and split C holds out a whole attack template. Brackets are 95% intervals of the change from the risk probe alone, at a 90% recall target. Split C has no clean steps in its test sets (n.a.).

| Model (unsafe steps) | Split | Risk probe alone: recall / FPR attacked-safe / FPR clean | Presence cascade: FPR clean | Decision cascade: FPR attacked-safe |
|---|---|---|---|---|
| Qwen2.5-7B (353) | A | 0.900 / 0.432 / 0.405 | 0.165 [−0.314, −0.165] | 0.331 [−0.160, −0.040] |
| | B | 0.907 / 0.485 / 0.412 | 0.144 [−0.361, −0.175] | 0.685 [+0.102, +0.301] |
| | C | 0.904 / 0.385 / n.a. | n.a. | 0.312 [−0.130, −0.015] |
| Granite-3.1-8B (150) | A | 0.893 / 0.346 / 0.345 | 0.211 [−0.191, −0.080] | 0.439 [+0.033, +0.152] |
| | B | 0.847 / 0.456 / 0.433 | 0.289 [−0.216, −0.082] | 0.400 [−0.152, +0.037] |
| | C | 0.953 / 0.372 / n.a. | n.a. | 0.377 [−0.052, +0.064] |
| Hermes-3-Llama-3.1-8B (104) | A | 0.904 / 0.347 / 0.371 | 0.144 [−0.299, −0.152] | 0.144 [−0.261, −0.144] |
| | B | 0.856 / 0.433 / 0.412 | 0.381 [−0.072, 0.000] | 0.242 [−0.269, −0.120] |
| | C | 0.904 / 0.403 / n.a. | n.a. | 0.114 [−0.348, −0.232] |

The presence cascade cuts clean-step false alarms on all three models in split A, and on held-out tool suites for Qwen and Granite. Its recall cost stays within about two points, except Hermes in split A at 0.865.

The decision cascade in this table reads the first 16 generated tokens. Hermes gains on every split. Qwen gains in splits A and C, but on a held-out suite its thresholds do not transfer and the rate rises. Recalibrating only the thresholds on half of the new suite's user tasks brings Qwen back to 0.409 against 0.503 [−0.161, −0.032]. Granite does not gain. The next section gives the cause and tests a stage that reads the call itself.

vLLM-Hook's own Attention Tracker analyzer can serve as stage 2 with no training. On Granite, with the head configuration in model_configs/attention_tracker and the user's task as the instruction span, every second step was scored. A cascade from the risk probe to one minus the tracker's score lowers clean-step false alarms from 0.578 to 0.318 [−0.377, −0.162] and attacked-but-safe ones from 0.550 to 0.482 [−0.118, −0.021], at recall 0.848 against 0.885 [−0.085, +0.003]. On the same steps it matches the trained presence probe. The tracker needs an eager engine in its score mode, and its Q/K mode spilled too much to disk for 5,000-token agent prompts.

Granite 3.1 needed two accommodations. Its chat template drops `tool_calls` and replaces the tool-calling instruction whenever a system message is given. So prior calls are written in its own `<|tool_call|>` syntax and that instruction is prepended. Even so, Granite never emitted `<|tool_call|>`, and its actions were read from any JSON call in the text. Strict parsing finds only 13 unsafe steps, too few to evaluate.

### Reading the decision stage at the tool call

The first 16 tokens fail on Granite for a reason other than slow calls. Granite starts its call within the first 16 tokens in 106 of its 150 unsafe actions, about as often as Qwen (268 of 353) and Hermes (88 of 104). But it writes the arguments before the tool name. The name comes after token 16 in 116 of the 150, and 33 of its unsafe calls start after token 16, behind some text. Most of its safe calls start after token 16 too (162 of 253).

A decision probe can read the call itself instead. `tool_call_window` finds the first call in the generated token ids, and a probe with `"position": "decode_window"` averages the decode states inside the window passed for each request. To compare windows on the actions evaluated above, each state was recomputed by teacher forcing, feeding the prompt plus the first t action tokens and keeping the last state. These states match the original decode capture at a median cosine of 0.9999 on all three models. The windows compared are 16 tokens from the call's first token, 16 tokens from the tool name, and the whole call up to 64 tokens. Steps without a call fall back to the first 16 tokens. On Granite the window from the call's first token, which equals the first 16 tokens whenever the call starts at the first token, lowers the attacked-but-safe rate on all steps from 0.438 to 0.190 [−0.306, −0.191] at the same recall. Those late calls are where the first tokens fail. A new split D holds out injection tasks, so the attacker's goal (such as its addresses or accounts) is new at test time. The protocol was fixed before these states existed.

A decision-time stage runs after the action is written, so the baseline for it is the simplest decision-time rule, which flags a step only if the risk probe fires and the action contains a tool call. The rule misses unsafe actions that are only text, which happen for attacker goals that are messages (11 of Granite's 150 unsafe steps). On Granite it lowers the attacked-but-safe rate from 0.346 to 0.139 at recall 0.897, because most attacked-but-safe Granite steps end in text, and in split A no hidden-state stage does better on all steps. On Hermes it changes almost nothing (0.344), while the decision probe brings the rate to 0.144. On Qwen the rule lowers the rate only from 0.432 to 0.392, while the whole-call window brings it to 0.210 [−0.289, −0.158] at an unchanged recall of 0.897, against 0.331 for the first 16 tokens.

So the table counts only steps whose action contains a tool call, the steps that a gate on tool execution reviews. Probes and thresholds are fit on those steps only, and tested on them. Each cell is recall / FPR on attacked-but-safe calls, and brackets are 95% intervals of the whole-call window's change from the risk probe alone.

| Model | Split | Risk probe alone | First 16 tokens | Whole call | Whole call minus risk probe (FPR / recall) | Tool name only (control) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | A | 0.892 / 0.460 | 0.895 / 0.338 | 0.900 / 0.197 | [−0.332, −0.192] / [−0.046, +0.064] | 0.902 / 0.263 |
| | D | 0.895 / 0.399 | 0.830 / 0.304 | 0.630 / 0.091 | [−0.382, −0.249] / [−0.455, −0.080] | 0.404 / 0.221 |
| Granite-3.1-8B | A | 0.912 / 0.724 | 0.888 / 0.731 | 0.894 / 0.597 | [−0.268, +0.001] / [−0.074, +0.042] | 0.890 / 0.669 |
| | D | 0.876 / 0.624 | 0.905 / 0.666 | 0.840 / 0.410 | [−0.307, −0.096] / [−0.113, +0.062] | 0.716 / 0.442 |
| Hermes-3-Llama-3.1-8B | A | 0.913 / 0.359 | 0.893 / 0.189 | 0.938 / 0.212 | [−0.239, −0.068] / [+0.003, +0.065] | 0.919 / 0.269 |
| | D | 0.921 / 0.336 | 0.593 / 0.086 | 0.784 / 0.122 | [−0.292, −0.155] / [−0.359, −0.031] | 0.472 / 0.217 |

With the attacker goals of the test steps also seen in training (split A), the whole-call window lowers the false-positive rate on attacked-but-safe calls for all three models. On Qwen it falls from 0.460 to 0.197. On Granite the interval reaches zero, but the whole call is lower than the first 16 tokens [−0.225, −0.045]. On Hermes, whose calls start at the first token, the two windows agree. Held-out attack templates (split C) give the same picture.

New attacker goals change it. Split D holds out injection tasks, and split B does too in effect, because each suite has its own. In both, the whole-call window loses recall on Qwen (0.630 in D, 0.754 against 0.896 in B) and on Hermes (0.784 in D, 0.865 against 0.989 in B), while on Granite its recall intervals include no change. In split D a stage that knows only the tool name loses even more. In splits A and C the same attacker goals, such as addresses and accounts, appear in training and in test, and much of the decision stage's gain there comes from recognizing them. The 16-token window from the tool name shows this most, with the lowest false-positive rates in split A on Qwen and Granite and the largest recall losses of the windows in split D. So numbers from splits A and C overstate a decision stage's effect on new attacks, and it needs validating on attacks whose goals were not in its training data.

On all steps in split D, the presence cascade still cuts clean-step false alarms on all three models (Qwen 0.407 to 0.173, Granite 0.361 to 0.222, Hermes 0.374 to 0.237). Recall does not change measurably on Granite, and it falls from 0.909 to 0.865 on Qwen [−0.077, −0.011] and from 0.798 to 0.704 on Hermes [−0.205, −0.019]. The first-16-token decision cascade lowers Qwen's attacked-but-safe rate from 0.435 to 0.285 at a recall of 0.832 against 0.909 [−0.167, −0.019], and Hermes' from 0.348 to 0.121 at an unchanged recall.

To serve a decision probe on the call, generate with `hooks_on="both"` and `merge_probes=False`, find each request's window with `tool_call_window(tokenizer, output.token_ids)`, and pass the windows as `"decode_windows"` in the analyzer spec. On 64 Granite steps generated this way on an A40, 41 reproduced the evaluated action, and on those the served features match the teacher-forced ones at a median cosine of 0.99993 (minimum 0.994). In 12 of the 41 the served window starts one token after the teacher-forced one, whose positions come from a re-tokenization of the text. No probe trained on call windows is released with this page, because on new attacker goals they lost recall on two of the three models.

The evaluation scripts, their outputs and the action texts and labels of all three models are in [latent-state-auditing/serving/vllm_hook/two_stage](https://github.com/rishabhsinha17/latent-state-auditing/tree/main/serving/vllm_hook/two_stage). The hidden-state captures are too large for git and are regenerated by its GPU scripts.

---

## On GPU with the current backend

These runs used vLLM 0.32 on one NVIDIA A40 with `gpu_memory_utilization=0.42` and CUDA graphs on.

`examples/demo_two_stage.py` runs end to end and gives the three demo steps the verdicts pass, pass and flag, with risk scores of 0.075, 0.048 and 0.733. Running both analyzers in batches of 32 over all 1,826 steps, the cascade's verdicts differ from a one-step-at-a-time capture on 19 steps and from the stored training features on 17. The `tool_call_risk` analyzer gives the same stage-1 scores. Prefill states from this backend match the training features at a median cosine of 0.99966 (minimum 0.9956), and the released probe's in-sample AUROC on attacked steps is 0.903 on them against 0.902 on the stored ones. At its 90%-recall threshold of 0.11, 29 of 1,826 verdicts flip between backends and 19 between batched and one-at-a-time capture on this backend, because 155 steps score within 0.02 of that threshold. At a threshold of 0.5 the counts are 9 and 7. Retraining the probes on this backend's states changes held-out recall and false-positive rates by at most 0.023.

Capture latency stays within run-to-run noise. On 64 real steps, each generating 32 tokens, the median time per step at batch size 1 is shown below, with all 64 steps in one batch in the last column.

| Configuration | Batch size 1 | 64 steps batched |
|---|---|---|
| Capture off | 0.930 s | 8.11 s |
| Capture at prefill (`tool_call_risk`, `two_stage_probe`) | 0.928 s | 8.06 s |
| Capture at prefill and every decode pass | 0.935 s | 8.15 s |
| Plain vLLM without vLLM-Hook | 0.949 s | 8.17 s |

Both analyzers score a batch of 64 steps on CPU in 1.5 ms and 0.5 ms. The GPU was shared with another job, and a lock kept the two from running at once.

### Decision-time stage 2

A second set of hidden states comes from the first 16 generated tokens of the action (`hooks_on="both"`, layer 28, one request per call). This check runs after generation starts but before the tool executes. The probe averages those states and is trained on attacked steps with the outcome label. The evaluation fixed the detectors and protocol in advance and uses the same split and calibration as above. Of the 1,826 steps, 1,626 regenerate the labelled action on this backend and are kept. The table compares each cascade with the released probe at a 90% recall target on those steps.

| Detector | Recall | FPR, attacked but safe | FPR, clean |
|---|---|---|---|
| Released probe alone | 0.901 | 0.418 | 0.430 |
| Stage 2 = injection-presence probe | 0.891 | 0.409 | 0.148 |
| Stage 2 = decision-time probe | 0.880 [−0.051, +0.006] | 0.292 [−0.188, −0.066] | 0.360 [−0.136, −0.006] |
| Stage 2 = presence × decision-time probe | 0.880 [−0.052, +0.008] | 0.277 [−0.203, −0.080] | 0.230 [−0.270, −0.130] |

For about two points of recall, the decision-time stage cuts the false-positive rate on attacked-but-safe steps (the model saw an injection and ignored it) by about a third. These rates count every attacked step, including those whose action is plain text, which a gate on tool execution never reviews. [Reading the decision stage at the tool call](#reading-the-decision-stage-at-the-tool-call) compares the stage with a rule that only reviews tool calls. Multiplying in the presence probe keeps most of the clean-step gain. As a control, a stage that only knows the name of the tool being called reaches 0.343 on attacked-but-safe steps. In a comparison run after these results, the presence-times-decision cascade is lower than that control on both false-positive rates, with intervals that exclude zero, while the decision-time stage alone is not clearly lower on attacked-but-safe steps. A fairer control keeps the same structure and replaces only the decision probe with the tool-name model, so stage 2 is presence times tool name. Compared at matched recall on held-out scores, the decision probe's attacked-but-safe rate is lower by 0.09 to 0.17 between recall 0.86 and 0.92 (0.297 against 0.441 at 0.90). That comparison places thresholds on test labels, so it measures score quality rather than a deployable operating point.

A second check adds back the steps whose action changed on this backend. They are regenerated in full and relabelled with AgentDojo 0.1.35, which reproduces every original label. On the resulting 1,774 steps, the presence-times-decision cascade lowers the attacked-but-safe rate from 0.404 to 0.282 [−0.184, −0.062] and the clean rate from 0.399 to 0.222 [−0.253, −0.106]. Recall is 0.872 against 0.902 [−0.064, +0.002]. With a false-positive budget of 0.2 instead of a recall target, the same cascade raises recall from 0.798 to 0.837.

Calibration matters for this cascade. Searching both thresholds on out-of-fold scores leaves realized recall two to three points under the target. Fixing the stage-1 gate at 97% recall (`stage1_recall=0.97`) and searching only the stage-2 threshold brings realized recall to within a point of the target. With every probe retrained on this backend's states, including the released recipe used as the baseline, the 1,774-step results are below.

| | Released recipe, retrained | Decision cascade with fixed gate | Interval of the change |
|---|---|---|---|
| Recall | 0.899 | 0.893 | [−0.038, +0.028] |
| FPR, attacked but safe | 0.402 | 0.289 | [−0.167, −0.057] |
| FPR, clean | 0.389 | 0.232 | [−0.232, −0.080] |

The fixed gate costs about two points of false-positive rate against the joint search, and it is the wrong choice for the presence-only cascade, whose attacked-but-safe rate rises to 0.57 under it.

The analyzer serves this stage in batches. Generate with `hooks_on="both"` and `merge_probes=False`, and give the decision probe `"position": "decode_mean"` in its probe.json. In batches of 32 over all 1,826 steps, every request kept its 17 passes. On 1,641 steps the batched run generated the same first 17 tokens as a one-at-a-time run, and on those the verdicts differ on 10, with stage-2 scores within 0.016. On the other 185 steps the model wrote a different action, which the stage then scores.

Reading each request's decode rows back is slow in batches. Across five measurements on two sets of 12 batches, a batch of 32 took 1.8 to 3.1 s against 0.84 to 0.86 s with capture off, and 0.93 s with the decode rows captured but not read. One request on its own takes 0.56 s against 0.50 s. Neither a 5 ms delivery poll nor six gather workers changed the batched time, and reading in parallel threads made it slower. Those runs rotated the modes over the same prompts with vLLM's prefix cache on, so whichever mode ran first on a batch also paid its cold prefill. A later run on another A40 host warmed every batch before timing it. There, capture at prefill added 0.12 s to a 0.82 s batch, and decode capture with every request's rows read added 0.79 s. Moving the capture directory (`MIA_APERTURE_DIR`) to a RAM disk left 0.71 s, so the cost does not come from writing the rows to disk. Its cause is not identified.

`examples/demo_two_stage.py --decision` runs this cascade. On the three demo steps it passes the clean step and flags both injected ones. One of the two is the demo step in which the model ignores the injection (stage 2 at 0.205 against a threshold of 0.096), and the prompt-only cascade passes it.

---

## Limitations

- The evaluation covers three 8B-class models on AgentDojo v1 only, with 97 clean steps per model. The clean-step result needs checking on real agent traffic before anyone relies on it.
- The presence probe is a linear probe trained on AgentDojo injections. Other injection styles or tools need their own validation, and a dedicated injection detector can stand in as stage 2 through `TwoStageDetector`.
- Calibrate on data disjoint from what you report, and refit thresholds when the agent, tools or prompt format change.
- Decision-time probes work with offline `MiaLLM` and `merge_probes=False`. The served path (`MiaClient`) does not return per-request decode passes, and reading every request's decode rows adds 0.7 to 2 s per batch of 32 on an A40, depending on the host.
