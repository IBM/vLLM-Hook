# Parity Tests

Run the local MLX/Metal side of the minimal parity suite from the repository
root:

```bash
python tests/parity_tests/run_all_minimal_parity.py --skip-colab
```

To run a single MLX/Metal parity experiment:

```bash
python tests/parity_tests/minimal_parity_benchmarks.py hidden-states \
  --backend metal \
  --benchmark-id minimal-parity-hidden-states \
  --hardware-label apple-metal \
  --hardware-kind metal
```

Use the same `--benchmark-id` when running the matching non-Metal/GPU side.

## Run All Command Sequence

From the repository root, run the full local MLX/Metal plus Colab GPU parity
sequence:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook

python tests/parity_tests/run_all_minimal_parity.py \
  --benchmark-prefix minimal-parity \
  --wandb-mode online \
  --gpu T4 \
  --prompt-wandb-key
```

The runner executes all configured local Metal experiments first, then starts
the matching Colab non-Metal runs with the same benchmark prefix.

## Generate Report

After the runs finish, generate the W&B parity report:

```bash
python tests/parity_tests/generate_wandb_parity_report.py \
  --wandb-entity tm8ctgzqj8-georgia-institute-of-technology \
  --report-project vllm-hook-platform-parity \
  --report-width fluid \
  --project hiddenstates \
  --project attntracker \
  --project corereranker
```
