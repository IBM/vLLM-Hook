#!/usr/bin/env bash
# Calibrate on 64 GSM8K examples, then plot fixed vs adaptive on LIMIT examples.
# Adaptive = calibration winner (margin AND top1_stability), sweep threshold T only.
# Requires: conda env with vllm + lm_eval + GPU.

export VLLM_ENABLE_V1_MULTIPROCESSING=0
OUT=benchmarks/recurrent_depth/results/vllm_cal
LIMIT="${LIMIT:-256}"
mkdir -p "$OUT/trajectories"

# 1. Calibration trajectory (~1–1.5 hr for limit 64)
python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \
  --tasks gsm8k --num-fewshot 5 --num-steps 32 --rho 0 --limit 64 \
  --capture-trajectory --prediction-metrics --output-dir "$OUT"

# 2. Threshold selection (writes the winning pair manifest)
python benchmarks/recurrent_depth/analyze_signals.py \
  --trajectory "$OUT/trajectories/vllm_fixed_rho0.0_r32.parquet" \
  --min-steps 2 --patience 2 --max-false-exit 0.02 --pairs \
  --out-json "$OUT/signal_report.json" \
  --write-manifest benchmarks/recurrent_depth/calibration/huginn_gsm8k_r32.json

# 3. Fixed-depth arm on the eval set
python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \
  --tasks gsm8k --num-fewshot 5 --limit "$LIMIT" \
  --sweep-fixed 4,8,16,32 --output-dir "$OUT"

# 4. Adaptive arm: freeze margin=0.125, sweep stability T from the cal grid.
#    Offline mean depths on the 64-example trajectory: 2→4.9, 4→7.1 (winner),
#    7→10.3, 9→12.3, 12→15.3, 19→22.2
for T in 2 4 7 9 12 19; do
  python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \
    --tasks gsm8k --num-fewshot 5 --num-steps 32 --limit "$LIMIT" \
    --min-steps 2 --patience 2 --condition-combine all \
    --condition margin=0.125 --condition top1_stability="$T" \
    --output-dir "$OUT"
done

# 5. Plot
python benchmarks/recurrent_depth/plot_pareto.py \
  --results-dir "$OUT" --task gsm8k --metric exact_match \
  --out "$OUT/pareto_gsm8k.pdf"