# H-Node Probe Artifacts

This directory ships the pre-built probe artifact for Qwen/Qwen2.5-1.5B-Instruct:

| File | Description |
|---|---|
| `probe.npz` | Numpy arrays: weights, bias, scaler_mean, scaler_scale, h_node_indices, h_node_baselines |
| `probe.json` | Metadata: model_name, best_layer, auc_per_layer, hidden_size, n_h_nodes, baseline_percentile, train_size, eval_size |

`activations.pt` (extracted hidden states, ~100 MB) is gitignored and not shipped.

## Validated artifact

| Model | Best layer | AUC | H-Nodes | Baseline percentile |
|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 14 | 0.902 | 50 | 80th |

Trained on TruthfulQA (300 questions, question-grouped 70/30 split).

## Dropping in your own probe

1. Train a probe using [hnode-probe-builder](https://github.com/Samarpit-bhatia/hnode-probe-builder)
2. Copy the output `probe.npz` + `probe.json` into this directory
3. Update `model_configs/hallucination_detection/Qwen2.5-1.5B-Instruct.infer.json` so `hidden_states.layers` matches the new `best_layer`

## Method

"H-Node Attack and Defense in Large Language Models"
Yocam, Vaidyan, Wang, 2026 — https://arxiv.org/abs/2506.07230
