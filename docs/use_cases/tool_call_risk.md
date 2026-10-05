# Tool-Call Risk Probe

**Contributor:** [@rishabhsinha17](https://github.com/rishabhsinha17)  
**Training and evaluation code:** [latent-state-auditing](https://github.com/rishabhsinha17/latent-state-auditing)

---

## What it does

Scores an agent step before the model acts. The step is the full prompt the agent sends to the model (system prompt, tool schemas, the user task, earlier tool calls and their results). A logistic probe reads the hidden state of the last prompt token at one layer and returns the probability that the next action is an unsafe tool call, for example one that follows an instruction injected into a tool result.

Because the score comes from prefill, it is available before any output token is sampled. A serving stack can hold the step, ask for confirmation, or route it to a stricter policy.

---

## How it integrates with vLLM-Hook

`ProbeHiddenStatesWorker` captures the last-token hidden state at the probe layer during prefill (`hooks_on="prefill"`, `mode="last_token"`). `ToolCallRiskAnalyzer` loads the probe and returns:

```python
{
  "probabilities": [0.04, 0.91],   # P(unsafe tool call) per step
  "margins":       [-3.1, 2.3],    # raw logits
  "verdicts":      ["pass", "flag"],
  "layer": 21,
  "threshold": 0.5,
}
```

The infer config (`model_configs/tool_call_risk/Qwen2.5-7B-Instruct.infer.json`) captures only the probe layer.

---

## Probe artifact

`probe.npz` holds `mean`, `scale`, `weights`, `bias` and `layer`. `probe.json` beside it holds `model_name`, `layer` and the calibrated `threshold`. `layer` uses the vLLM-Hook convention (1-based output of the Nth decoder block, same as HuggingFace `hidden_states[N]`). The analyzer reads the threshold from `probe.json` unless `analyzer_spec["threshold"]` overrides it.

No artifact is published yet. It will be hosted outside this repo, the same way H-Node does it.

---

## Quick start

```bash
python examples/demo_toolcallrisk.py --probe path/to/probe.npz
```

The demo builds two agent steps with the model's tool-calling chat template, one with a clean file and one with an injected instruction in the file, and prints each step's score next to the action the model generated.

---

## Limitations

- The probe predicts the action, it does not detect injection. Clean traffic can score above attacked-but-safe traffic, so it should sit behind an injection or exposure gate rather than replace one.
- Pick the threshold from the false-positive rate you can tolerate on attacked-but-safe steps, not from AUROC alone.
- A probe is tied to one model and one layer. Probes for other models need their own artifacts.
