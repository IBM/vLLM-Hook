# Tests

This directory contains model compatibility tests for the `vllm_hook_plugins` package.
The tests validate that hooks, workers, and analyzers work correctly with vLLM models.

These tests are **resource-aware** and do assume enough access to GPU resources. To reduce contention on shared systems:
- tests use low `gpu_memory_utilization` values
- only small or mid-sized models are enabled by default

If the GPU is heavily loaded, model initialization may fail. Current tests assume enough compute to host a 7B model and have `gpu_memory_utilization=0.2~0.5`.

---
## Run Tests
From the project root:

```bash
pytest -vv
```

Run only attention tracker tests:

```bash
pytest tests/use_cases/test_attntracker.py -vv
```

Run a single model:

```bash
pytest tests/use_cases/test_attntracker.py::test_attention_tracker[gpt2] -vv
```

---
## Run Platform Parity Experiments

The abstract specs in `tests/system_prompts/` are implemented by
`tests/vllm_hook_experiments.py` only where this repo has paired Metal and
non-Metal support. Run each backend separately with the same `--benchmark-id`;
W&B uses that id as the run group and stores reports plus hook artifacts.
For non-Metal runs, pass `--device cuda` or `--device cpu` when you want the
runtime choice recorded and forced explicitly.

```bash
pip install wandb

export WANDB_API_KEY=...
export BENCHMARK_ID=attn-parity-001

python tests/vllm_hook_experiments.py attn-tracker \
  --backend non-metal \
  --device cuda \
  --benchmark-id "$BENCHMARK_ID" \
  --wandb-mode online \
  --wandb-project vllm-hook-platform-parity

python tests/vllm_hook_experiments.py attn-tracker \
  --backend metal \
  --benchmark-id "$BENCHMARK_ID" \
  --wandb-mode online \
  --wandb-project vllm-hook-platform-parity
```

Available experiments:

- `attn-tracker` uses `probe_hook_qk` with `attn_tracker` / `AttntrackerAnalyzerMetal`.
- `core-reranker` uses `probe_hook_qk` with `core_reranker` / `CorerAnalyzerMetal`.
- `steer-activation` uses `steer_hook_act` on both backends and compares baseline vs steered output behavior.

Run the other paired Metal notebook experiments the same way:

```bash
export WANDB_API_KEY=...

for EXPERIMENT in attn-tracker core-reranker steer-activation; do
  export BENCHMARK_ID="${EXPERIMENT}-parity-001"

  python tests/vllm_hook_experiments.py "$EXPERIMENT" \
    --backend non-metal \
    --device cuda \
    --benchmark-id "$BENCHMARK_ID" \
    --wandb-mode online \
    --wandb-project vllm-hook-platform-parity

  python tests/vllm_hook_experiments.py "$EXPERIMENT" \
    --backend metal \
    --benchmark-id "$BENCHMARK_ID" \
    --wandb-mode online \
    --wandb-project vllm-hook-platform-parity
done
```

Local JSON, CSV, artifact manifests, and hook artifacts are written under
`tests/experiment_runs/` and ignored by git.

The model-backed experiment runner smoke tests are skipped by default. Enable
them explicitly:

```bash
RUN_VLLM_HOOK_EXPERIMENT_SMOKE=1 pytest tests/test_vllm_hook_experiment_runner.py -vv
RUN_VLLM_HOOK_METAL_EXPERIMENT_SMOKE=1 pytest tests/test_vllm_hook_experiment_runner.py -vv
```

---

## Common Failures

- **Installed 0 hooks**  
  Model architecture not matched or config contains no heads.
