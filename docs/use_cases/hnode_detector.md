# H-Node Hallucination Detector

**Contributor:** [@Samarpit-bhatia](https://github.com/Samarpit-bhatia)  
**Method:** [H-Node Attack and Defense in Large Language Models](https://arxiv.org/abs/2506.07230) — Yocam, Vaidyan, Wang, 2026  
**Config-building repo:** [hnode-probe-builder](https://github.com/Samarpit-bhatia/hnode-probe-builder)

---

## What it does

Detects hallucination at inference time by reading the model's internal activations — before the output token is sampled. A logistic-regression probe trained on last-token hidden states at a specific transformer layer assigns a hallucination probability to each prompt.

The probe is trained to distinguish between:
- **Grounded** responses — correct answers (label 0)
- **Hallucinated** responses — incorrect answers (label 1)

Using TruthfulQA as the training signal, the best layer is selected by held-out AUC. The top-N hidden dimensions most associated with hallucination are identified as **H-Nodes**.

---

## How it integrates with vLLM-Hook

vLLM-Hook installs a forward hook on the model's transformer layers at startup (`ProbeHiddenStatesWorker`). During inference, when the residual stream passes through the probe's best layer, the hook captures the last-token hidden state vector and saves it to disk.

After generation, `HallucinationAnalyzer` reads that vector, runs it through the probe, and returns:

```python
{
  "probabilities": [0.02, 0.97, ...],   # P(hallucinated) per prompt
  "h_node_excess": [0.01, 0.31, ...],   # mean H-Node activation above baseline
  "verdicts":      ["grounded", "hallucinated", ...],
  "best_layer": 14,
  "threshold": 0.5,
}
```

The model config (`model_configs/hallucination_detection/Qwen2.5-1.5B-Instruct.infer.json`) tells the worker to capture only layer 14 — so there is minimal overhead compared to capturing all layers during training.

---

## Pre-built probe artifact

A trained probe for `Qwen/Qwen2.5-1.5B-Instruct` is shipped in `vllm_hook_plugins/vllm_hook_plugins/utils/hnode/artifacts/`:

| Model | Best layer | AUC | H-Nodes | Trained on |
|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 14 | 0.902 | 50 | TruthfulQA (300 questions, 70/30 split) |

| File | Contents |
|---|---|
| `probe.npz` | Weights, bias, scaler parameters, H-Node indices and baselines |
| `probe.json` | Metadata: model name, best layer, per-layer AUC scores, H-Node count |

---

## Quick start

```python
from vllm_hook_plugins import HookLLM
from vllm import SamplingParams
import torch

llm = HookLLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    worker_name="probe_hidden_states",
    analyzer_name="hallucination",
    config_file="model_configs/hallucination_detection/Qwen2.5-1.5B-Instruct.infer.json",
    gpu_memory_utilization=0.85,
    max_model_len=1024,
    dtype=torch.float16,
    enable_hook=True,
)

prompts = [
    "Q: What is the capital of France?\nA: Paris",
    "Q: What is the capital of France?\nA: London",
]

run_id = "my_run"
llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1),
             save_to_disk=True, run_id=run_id)

result = llm.analyze(
    analyzer_spec={"probe_path": "vllm_hook_plugins/vllm_hook_plugins/utils/hnode/artifacts/probe.npz", "threshold": 0.5},
    run_id=run_id,
)

for prompt, verdict, prob in zip(prompts, result["verdicts"], result["probabilities"]):
    print(f"[{verdict}] P={prob:.3f}  {prompt}")
```

Or run the demo directly:

```bash
python examples/demo_halludetect.py
```

---

## Building your own probe

To train a probe for a different model or dataset, use the config-building repository:

**[github.com/Samarpit-bhatia/hnode-probe-builder](https://github.com/Samarpit-bhatia/hnode-probe-builder)**

The workflow:
1. **Extract** — run prompts through vLLM-Hook to dump per-layer hidden states (`activations.pt`)
2. **Train** — fit per-layer logistic-regression probes, select best layer by AUC, identify H-Nodes
3. **Drop in** — copy `probe.npz` + `probe.json` into `vllm_hook_plugins/vllm_hook_plugins/utils/hnode/artifacts/` and update `best_layer` in the infer config

See the repo's README for setup and usage instructions.

---

## Two-repo architecture

```
hnode-probe-builder                vLLM-Hook
(config-building, one-time)        (inference, every prompt)

data.py  ─────────────────────┐
extract.py  ──────────────────┤  probe.npz ──► score.py
train_probe.py  ──────────────┘  probe.json     HNodeProbe.score()
                    train once        │               │
                         │            │         hallucination_analyzer.py
                         ▼            │               │
                    probe.npz  ───────┘               ▼
                    probe.json              verdict per prompt
```

vLLM-Hook ships the inference side only. The training pipeline is kept separate so vLLM-Hook has no dependency on `datasets` or sklearn training code.

---

## Citation

```bibtex
@article{yocam2026hnode,
  title   = {H-Node Attack and Defense in Large Language Models},
  author  = {Yocam, E. and Vaidyan, A. and Wang, P.},
  year    = {2026},
  url     = {https://arxiv.org/abs/2506.07230}
}
```
