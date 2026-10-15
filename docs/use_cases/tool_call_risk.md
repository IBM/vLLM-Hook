# Tool-Call Risk Probe

**Contributor:** [@rishabhsinha17](https://github.com/rishabhsinha17)  
**Training and evaluation code:** [latent-state-auditing/serving/vllm_hook](https://github.com/rishabhsinha17/latent-state-auditing/tree/main/serving/vllm_hook)

---

## What it does

Scores an agent step before the model acts. The step is the full prompt the agent sends to the model (system prompt, tool schemas, the user task, earlier tool calls and their results). A logistic probe reads the hidden state of the last prompt token at one layer and returns the probability that the next action follows a prompt injection that arrived in a tool result.

Because the score comes from prefill, it is available before the model samples any output token. A serving stack can hold the step, ask for confirmation, or route it to a stricter policy.

---

## How it integrates with vLLM-Hook

MIA's hidden-state worker (`capture_hs`) captures the residual stream at the probe layer and keeps the last-token vector of each prompt. `ToolCallRiskAnalyzer` loads the probe and returns:

```python
{
  "probabilities": [0.073, 0.048, 0.748],   # P(next action follows the injection) per step
  "margins":       [-2.54, -2.99, 1.09],   # raw logits
  "verdicts":      ["pass", "pass", "flag"],
  "layer": 28,
  "threshold": 0.1116,
}
```

The infer config (`model_configs/tool_call_risk/Qwen2.5-7B-Instruct.infer.json`) captures only the probe layer.

---

## Pre-built probe

The training repo hosts the artifact, and `examples/demo_toolcallrisk.py` downloads it on demand into `./cache/tool_call_risk/`.

| Model | Layer | Probe | AUROC (held-out user tasks) | FPR at 90% recall | Trained on |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 28 | PCA-32 + logistic | 0.869 | 0.325 | 1,729 attacked AgentDojo v1 steps, 358 unsafe |

The training steps come from all four AgentDojo suites and three attack templates. Each step uses the Qwen2.5 tool-calling chat template and goes through vLLM-Hook, so the probe is fit on the same features this analyzer reads. Labels come from AgentDojo's own injection-task ground truth and security checks applied to the action the model generated.

| File | Contents |
|---|---|
| `probe.npz` | `mean`, `scale`, `weights`, `bias`, `layer` |
| `probe.json` | model name, layer, calibrated `threshold`, metrics, provenance |
| `demo_steps.jsonl` | three AgentDojo steps for the demo |

`layer` uses the vLLM-Hook convention (1-based output of the Nth decoder block). The analyzer reads the threshold from `probe.json` unless `analyzer_spec["threshold"]` overrides it.

---

## Quick start

```bash
python examples/demo_toolcallrisk.py
```

The demo scores a clean step and two injected steps (the model ignores the injection in one and follows it in the other), and prints each score next to the generated action.

---

## Capture parity and serving cost

Hidden states captured through vLLM-Hook match HuggingFace. With identical token ids on the same GPU, the two agree to 0.13% relative L2 in fp32, and in bf16 they differ by as much as two HuggingFace runs do (about 2.5%).

On 280 agent steps of about 4,500 prompt tokens each (Qwen2.5-7B-Instruct, bf16, one RTX 4090, vLLM 0.21.0), capturing one layer to memory took 159.1 s against 158.6 s with capture off and 158.5 s for plain vLLM with CUDA graphs.

These measurements, and the probe itself, come from the pre-MIA capture path on vLLM 0.21.0 in eager mode. Recapturing 440 of the training steps through MIA on vLLM 0.29 (RTX 4090, CUDA graphs) reproduces those features. The median cosine to the training vectors is 0.99966 (minimum 0.998), the largest probe score shift is 0.06, and AUROC on the subset is 0.902 for both. Only 4 of 440 verdicts change at the stored threshold.

On a 24 GB GPU the demo sets `MIA_APERTURE_GPU_BYTES` to 512 MiB. One captured layer needs about 60 MB per 8192-token step, and the 4 GiB default aperture leaves a 7B model no room for KV cache at `gpu_memory_utilization=0.8`.

Full numbers and scripts are in the training repo.

---

## Limitations

- The probe predicts the action, it does not detect injection. Its AUROC for separating attacked-but-safe steps from clean ones is 0.53, so deploy it next to an injection filter rather than instead of one.
- At 90% recall it still flags about a third of attacked-but-safe steps. Pick the threshold from the false-positive rate you can accept.
- Each probe belongs to one model, one layer and vLLM-Hook captures. Layer 28 is the last block of Qwen2.5-7B, and vLLM-Hook captures it before the final norm, unlike HuggingFace `hidden_states[28]`.
- AgentDojo is the only training distribution. Other agents, tools or prompt formats need their own validation.
