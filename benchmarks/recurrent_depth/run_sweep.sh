#!/usr/bin/env bash
# Pareto sweep on AdaptiveRavenForvLLM (default) + optional HF.
# Requires: conda env with vllm + lm_eval + GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BACKEND="${BACKEND:-vllm}"
MODEL="${MODEL:-tomg-group-umd/huginn-0125}"
TASKS="${TASKS:-gsm8k}"
LIMIT="${LIMIT:-}"          # empty = full task
FEWSHOT="${FEWSHOT:-5}"
RMAX="${RMAX:-32}"
FIXED_GRID="${FIXED_GRID:-4,8,16,32}"
RHO_GRID="${RHO_GRID:-0,0.001,0.005,0.01,0.02,0.05}"
OUT="${OUT:-$ROOT/benchmarks/recurrent_depth/results/${BACKEND}}"

EXTRA=()
if [[ -n "$LIMIT" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

echo "backend=$BACKEND model=$MODEL tasks=$TASKS fewshot=$FEWSHOT rmax=$RMAX"
echo "fixed=$FIXED_GRID rho=$RHO_GRID out=$OUT"

python benchmarks/recurrent_depth/run_lm_eval.py \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --num-fewshot "$FEWSHOT" \
  --num-steps "$RMAX" \
  --sweep-fixed "$FIXED_GRID" \
  --sweep-rho "$RHO_GRID" \
  --batch-size 1 \
  --dtype bfloat16 \
  --output-dir "$OUT" \
  "${EXTRA[@]}"

python benchmarks/recurrent_depth/plot_pareto.py \
  --results-dir "$OUT" \
  --task gsm8k \
  --metric exact_match \
  --out "$OUT/pareto_gsm8k.pdf" \
  --title "GSM8K (5-shot): AdaptiveRaven vs fixed depth [${BACKEND}]"
