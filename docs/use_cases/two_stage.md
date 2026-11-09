# Two-Stage Probe Cascade

**Contributor:** [@rishabhsinha17](https://github.com/rishabhsinha17)  
**Evaluation code, outputs and the detailed evaluation:** [latent-state-auditing/serving/vllm_hook/two_stage](https://github.com/rishabhsinha17/latent-state-auditing/tree/main/serving/vllm_hook/two_stage) ([EVALUATION.md](https://github.com/rishabhsinha17/latent-state-auditing/blob/main/serving/vllm_hook/two_stage/EVALUATION.md))

---

## What it does

Lowers the false-positive rate of a detector by adding a second check that only runs on the steps the first one flags. The final verdict flags a step only when both stages fire. Stage 1 is a cheap, high-recall gate. Stage 2 can be another probe on the same capture, or any slower scorer such as a guard model.

Requiring both stages can lower the false-positive rate of stage 1 but can never raise its recall, so calibrate the two thresholds together on held-out data. The second stage only helps if it sees something stage 1 does not.

---

## How it integrates

`mia.utils.two_stage` needs only numpy.

| Function | Use |
|---|---|
| `calibrate_cascade(s1, s2, y, target_recall, max_stage2_rate=None, finite_sample=False, stage1_recall=None)` | Lowest false-positive rate at a target recall. `max_stage2_rate` caps the share of steps sent to an expensive stage 2. `stage1_recall` fixes the stage-1 gate and searches only the stage-2 threshold, which held recall better for the decision-time cascade but raised the presence cascade's attacked-but-safe false alarms. |
| `calibrate_cascade_for_fpr(s1, s2, y, target_fpr)` | Highest recall within a false-positive budget. |
| `threshold_for_recall(pos_scores, recall, finite_sample=False)` | Single-stage threshold for a target recall. `finite_sample=True` uses the split-conformal count. |
| `threshold_for_fpr(neg_scores, fpr)` | Single-stage threshold for a false-positive budget, from negatives alone such as one deployment's benign traffic. |
| `TwoStageDetector(stage1, stage2, t1, t2)` | Runs any two scorers and calls `stage2` only on the stage-1 flags. |
| `save_cascade_config(path, stage1_probe, stage2_probe, point, **meta)` | Writes the `cascade.json` the analyzer reads. |
| `tool_call_window(tokenizer, token_ids, max_tokens=64)` | Generated-token window of the first tool call in an action, for a `decode_window` probe. |

The `two_stage_probe` analyzer (worker `capture_hs`) runs linear probes in the `tool_call_risk` artifact format on one capture. A stage is one probe or a product of probes (`{"product": [...]}`). Each probe reads the last prompt token (`"position": "prompt_last"`, the default), the mean of the states for the first generated tokens (`"decode_mean"`), or their mean over a per-request window of generated tokens (`"decode_window"`), as set in its probe.json or the spec. A cascade is calibrated, saved and served as in this sketch.

```python
from mia.utils.two_stage import calibrate_cascade, save_cascade_config

point = calibrate_cascade(s1_cal, s2_cal, y_cal, target_recall=0.9)   # held-out calibration scores
save_cascade_config("cascade.json", "stage1/probe.npz", "stage2/probe.npz", point)

out = llm.analyze(analyzer_spec={"cascade_path": "cascade.json"}, run_id=run_id)
# out["verdicts"], out["reached_stage2"], out["stage1"]["probabilities"], out["stage2"]["probabilities"]
```

Both thresholds must come from `cascade.json` or the spec. The analyzer ignores each `probe.json` threshold, because that one was set for the probe on its own. The infer config must capture every probe layer. `"score_all": true` scores stage 2 on every step, which recalibration needs. A `decode_mean` probe needs each request's own capture, so generate with `extra_args={"hooks_on": "both"}` and `merge_probes=False`, then pass `probes=[o.probes for o in outputs]`. The default merged capture keeps only the prompt passes.

---

## Pre-built cascade

`examples/demo_two_stage.py` downloads a cascade for Qwen2.5-7B-Instruct from the evaluation repo. Stage 1 is the [tool-call risk probe](tool_call_risk.md), called the risk probe alone below when it runs without stage 2. Stage 2 is an injection-presence probe, and the two together are the presence cascade. Both read the last prompt token of layer 28, and the cascade uses the same capture as stage 1. The presence probe is trained to tell steps with an injection in context from clean ones, with no outcome label, on the 1,826 AgentDojo v1 steps behind the risk probe. Both thresholds in `cascade.json` were calibrated together at a 90% recall target on out-of-fold scores, with folds grouped by user task. This is the recipe of the Qwen rows in the results below, fit on an earlier capture of the same steps, with the risk probe fit on the 1,729 attacked steps. `--decision` loads the experimental decision-time cascade described below, whose stage-1 probe was refit on vLLM 0.32 features and gated at 97% recall.

## Quick start

```
python examples/demo_two_stage.py
```

The demo scores three AgentDojo steps, one clean, one whose action ignores an injection and one whose action follows it. It prints both stage scores and the verdict for each, and on an A40 the verdicts are pass, pass and flag. Stage 1 alone already passes the first two, and the demo shows the interface rather than a false alarm that stage 2 removes. With `--decision` the verdicts are pass, flag and flag, and that cascade flags the step whose action ignores the injection.

---

## Results

AgentDojo pairs each user task, the request the agent serves, with injection tasks, the goals an injected instruction tries to make it carry out. These results use the AgentDojo v1 steps of three 8B-class models, with folds grouped by user task so that training never sees a test user task, though each test step's injection task and attack template also appear in training, and a 90% recall target. Recall at arrival counts each attack at the first step with the injection in its context. Recall after arrival counts the later steps of the same task with the injection still in the history, taken from the task's ground-truth calls, for the three attack templates used in training. The AgentDojo clean steps are 97 per model, one per user task at the depth where attacks arrive. The ToolACE column uses 1,201 clean steps from a different tool-use benchmark, scored by probes fit on all AgentDojo v1 steps. Intervals, other splits and the checks behind every number are in [EVALUATION.md](https://github.com/rishabhsinha17/latent-state-auditing/blob/main/serving/vllm_hook/two_stage/EVALUATION.md), and the raw outputs are in `out_r11/` and `out_r15/` next to it.

| Model | Detector | Recall at arrival | Recall after arrival | FPR, attacked but safe, at arrival | FPR, AgentDojo clean | FPR, ToolACE clean |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | risk probe alone | 0.900 | 0.896 | 0.432 | 0.405 | 0.484 |
| | presence cascade | 0.894 | 0.562 | 0.426 | 0.165 | 0.120 |
| Granite-3.1-8B | risk probe alone | 0.893 | 0.757 | 0.346 | 0.345 | 0.153 |
| | presence cascade | 0.882 | 0.569 | 0.341 | 0.211 | 0.047 |
| Hermes-3-Llama-3.1-8B | risk probe alone | 0.904 | 0.915 | 0.347 | 0.371 | 0.387 |
| | presence cascade | 0.865 | 0.683 | 0.329 | 0.144 | 0.059 |

The presence cascade cuts clean-step false alarms on all three models, by 39 to 61% on AgentDojo and 69 to 85% on ToolACE. With held-out user tasks its recall at arrival stays within about two points on Qwen and Granite and falls four points on Hermes. With held-out attack templates or injection tasks it falls by up to about nine points (Hermes 0.798 to 0.704 with held-out injection tasks). It does not lower false alarms on attacked-but-safe steps.

Once an injection stays in the history, its recall is 19 to 33 points below the risk probe's with the original templates and 25 to 34 points with two held-out ones (point estimates). Changes that bring back most of that recall, among them carrying the presence score forward within a task, added false alarms on clean steps and on some models still lost recall, failing criteria set before each run, three of them partly after some outcomes were seen ([details](https://github.com/rishabhsinha17/latent-state-auditing/blob/main/serving/vllm_hook/two_stage/EVALUATION.md#what-to-use-when)).

False alarms also depend on the step's position in its task. Across 337 more AgentDojo clean steps at every depth of each task, the presence cascade flags 0.336, 0.539 and 0.433 of them and the risk probe alone 0.713, 0.670 and 0.790. Most of the presence cascade's excess is at a task's first step, before any tool output. There the presence cascade flags 0.778, 1.000 and 0.933 and the risk probe alone 0.977, 1.000 and 0.961. On the steps with a tool output the presence cascade still flags 0.157, 0.352 and 0.231.

vLLM-Hook's own Attention Tracker analyzer can serve as a training-free stage 2. On half of Granite's steps its clean-step false alarms were not shown to differ from the trained presence probe's, and it needs an eager engine in its score mode ([EVALUATION.md](https://github.com/rishabhsinha17/latent-state-auditing/blob/main/serving/vllm_hook/two_stage/EVALUATION.md)).

## When to use it

- Use the presence cascade to cut false alarms on clean steps for agents that act on an injection, if at all, at the step where it arrives. In Qwen's own rollouts about half of the unsafe steps came after arrival.
- Use the risk probe alone if injections can stay in the agent's context across steps and missing them later costs more than the extra false alarms.
- If untrusted text reaches the prompt only through tool outputs, the caller can skip scoring steps without a tool output. This is a recommendation the evaluation did not test, and the analyzer scores every request it is given.
- The presence cascade does not lower false alarms on attacked-but-safe steps. That needs a stage that reads the action, such as the experimental one below.

## A decision-time stage 2 (experimental)

A stage-2 probe can read the decode states of the first 16 generated tokens (`"position": "decode_mean"`), which still come before the tool executes, and `"decode_window"` with `tool_call_window` reads the tool call instead (no call-window probe is released). The decision-time cascade multiplies the presence probe by a probe on the first 16 tokens, with the stage-1 gate fixed at 97% recall, the configuration `--decision` runs. On 1,774 Qwen steps it lowers attacked-but-safe false alarms from 0.402 to 0.289 and clean ones from 0.389 to 0.232, at recall 0.893 against 0.899. On Granite, with held-out user tasks, it raised attacked-but-safe false alarms (0.346 to 0.439), and with a held-out tool suite Qwen's rose (0.485 to 0.685). On held-out injection tasks it lost recall on Qwen, and on Hermes among steps with a tool call, and a test with criteria fixed in advance adopted it on no model. It needs offline `MiaLLM` with `hooks_on="both"` and `merge_probes=False`, since the served path does not return per-request decode passes.

## Overhead

In a run that warmed every batch first (Qwen, one A40, 17 generated tokens), capture at prefill for both probes added 0.12 s to a 0.82 s batch of 32, and an earlier 64-step benchmark found it within run-to-run noise. Capture at each decode pass with each request's rows read, which only the decision-time cascade needs, added 0.79 s per batch over capture off. The `tool_call_risk` and `two_stage_probe` analyzers each score 64 steps on CPU in under 2 ms.

---

## Limitations

- Three 8B-class models. Recall comes from AgentDojo only, apart from one InjecAgent test of the decision-time stage on Qwen, and false alarms from AgentDojo and ToolACE, whose tool outputs are synthetic. Check the clean-step rate on real traffic before relying on it.
- The presence cascade's recall drops once an injection stays in the context, by 19 to 34 points against the risk probe alone, and none of the changes tried was adopted ([EVALUATION.md](https://github.com/rishabhsinha17/latent-state-auditing/blob/main/serving/vllm_hook/two_stage/EVALUATION.md#injections-that-stay-in-the-context)).
- With held-out attack templates or injection tasks, the presence cascade's recall at arrival falls up to about nine points below the risk probe alone's, which is itself below its 90% target there. On AgentDojo v1.2's newer injection tasks the risk probe alone caught 27 of Qwen's 39 unsafe tool-call steps. Validate on attacks that were not in the training data.
- The presence probe is a linear probe trained on AgentDojo injections. Other injection styles or tools need their own validation, and a dedicated injection detector can stand in as stage 2 through `TwoStageDetector`.
- Calibrate on data disjoint from what you report, and refit thresholds when the agent, tools or prompt format change.
