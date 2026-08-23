# Recurrent-depth lm-eval (fixed vs adaptive Raven)

Paper-oriented evaluation of **fixed recurrence** vs **adaptive early exit** via
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

**Default / production path:** `--backend vllm` → `AdaptiveRavenForvLLM` under
vLLM-Hook. Optional `--backend hf` → `AdaptiveRavenForCausalLM` (numerical oracle).

## Install

```bash
conda activate vllm_hook_env_py312   # or your env with vllm
pip install "lm_eval[hf]" datasets matplotlib
# Raven HF pin used elsewhere: transformers==4.51.0
```

## Pareto axes

- **X:** `exit_stats.mean_effective_r` (mean effective recurrence)
- **Y:** lm-eval task metric (e.g. GSM8K `exact_match`)


| Arm      | How                                                           |
| -------- | ------------------------------------------------------------- |
| Fixed    | `--sweep-fixed 4,8,16,32` → `rho=0`, vary `num_steps`         |
| Adaptive | `--sweep-rho 0,0.01,… --num-steps 32` → vary `ρ`, cap `r_max` |


Both sweeps can run in **one** invocation.

## Quick start (vLLM)

```bash
# Smoke
python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \
  --tasks gsm8k --num-fewshot 5 --limit 32 --rho 0 --num-steps 32

# Publication sweep + plot
bash benchmarks/recurrent_depth/run_sweep.sh
# or with a limit first:
LIMIT=100 bash benchmarks/recurrent_depth/run_sweep.sh
```

HF reference curve:

```bash
BACKEND=hf OUT=benchmarks/recurrent_depth/results/hf \
  bash benchmarks/recurrent_depth/run_sweep.sh
```



## Files


| File                       | Role                         |
| -------------------------- | ---------------------------- |
| `raven_lm_eval.py`         | HF `adaptive_raven`          |
| `raven_lm_eval_vllm.py`    | vLLM `adaptive_raven_vllm`   |
| `run_lm_eval.py`           | sweeps (`--backend hf|vllm`) |
| `plot_pareto.py`           | quality vs \bar{r} PDF/PNG   |
| `run_publication_sweep.sh` | fixed + ρ grids + plot       |