#!/usr/bin/env bash
# Pareto sweep on AdaptiveRavenForvLLM (default) + optional HF.
# Requires: conda env with vllm + lm_eval + GPU.

export VLLM_ENABLE_V1_MULTIPROCESSING=0
OUT=benchmarks/recurrent_depth/results/vllm_cal
mkdir -p "$OUT/trajectories"

# 1. Calibration trajectory (~1–1.5 hr for limit 64)
python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \
  --tasks gsm8k --num-fewshot 5 --num-steps 32 --rho 0 --limit 64 \
  --capture-trajectory --prediction-metrics --output-dir "$OUT"

# 2. Threshold selection
python benchmarks/recurrent_depth/analyze_signals.py \
  --trajectory "$OUT/trajectories/vllm_fixed_rho0.0_r32.parquet" \
  --min-steps 2 --patience 2 --max-false-exit 0.02 --pairs \
  --out-json "$OUT/signal_report.json" \
  --write-manifest benchmarks/recurrent_depth/calibration/huginn_gsm8k_r32.json

# 3. Fixed-depth Pareto arm (~4× step 1 without trajectory, still slow)
python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \
  --tasks gsm8k --num-fewshot 5 --limit 64 \
  --sweep-fixed 4,8,16,32 --output-dir "$OUT"

# 4. Calibrated policy eval (1 run)
python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \
  --tasks gsm8k --num-fewshot 5 --num-steps 32 --limit 64 \
  --policy-manifest benchmarks/recurrent_depth/calibration/huginn_gsm8k_r32.json \
  --output-dir "$OUT"

# 5. Plot
python benchmarks/recurrent_depth/plot_pareto.py \
  --results-dir "$OUT" --task gsm8k --metric exact_match \
  --out "$OUT/pareto_gsm8k.pdf"