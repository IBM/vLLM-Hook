#!/bin/bash
# run_steer_ab_leak_control.sh — decisive A/B experiment for task D6.
#
# LSF 1725139's `steer_small per-request-arming` gate found 5/17 UNARMED requests moved by
# up to 2.575421e-02 (floor 1e-3) while the weakest ARMED effect was 2.757737e-01 (~10x
# larger). Is that a real per-row steering leak, or the batch-composition/boot noise this
# project has measured repeatedly at this exact magnitude elsewhere (T0-graph 1.87e-02,
# steer width 1.66e-02, QK boot spread up to 3.5e-02 absolute)?
#
# Runs tests/mia/parity/run_steer_ab_leak_control.py: ONE engine boot, the SAME 33
# distinct-length prompts, SAME alternating arming mask, SAME batch composition in both
# passes -- Arm A arms 16 requests with the REAL adjust_rs steer, Arm B arms the SAME 16
# with the SAME config except an ALL-ZERO-DIRECTION clone of the vector (verified NOT to
# skip the op -- see the script's docstring for why "coefficient: 0.0" would have been a
# worthless control for this method). Unarmed rows get no config in either arm, exactly as
# the real leg does.
#
# Reads off:
#   - the 17 UNARMED requests, A vs B: bit-identical -> no leak (b); differ -> a real leak (a).
#   - the 16 ARMED requests, A vs B: positive control, must be LARGE (~2.76e-01).
#
# Submit:  bash -lc 'cd /u/fangyunh/vLLM-Hook && bsub < tests/mia/parity/run_steer_ab_leak_control.sh'
# Watch:   bjobs ; bpeek <JOBID>
# Result:  grep -E 'VERDICT|SUMMARY|MISMATCH|\[AB\]' bluevela/logs/steer_ab.<JOBID>.out
#
#BSUB -J mia_steer_ab
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o bluevela/logs/steer_ab.%J.out
#BSUB -e bluevela/logs/steer_ab.%J.err
set -uo pipefail

REPO=/u/fangyunh/vLLM-Hook
PY=/proj/dmfexp/fangyunh/envs/mia_v029/bin/python
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/proj/dmfexp/fangyunh/vLLM-Hook-offload/home_hf_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Same distinct-length, width-33 batch composition the real leg uses.
export MIA_PARITY_PROMPTS=distinct
export MIA_PARITY_MAX_NUM_SEQS=33

JOB="${LSB_JOBID:-manual}"
OUT=/proj/dmfexp/fangyunh/mia_parity/$JOB/steer_ab_leak_control
mkdir -p "$OUT"

SCRATCH="/opt/nvme/${USER}/mia_steer_ab_${JOB}"
mkdir -p "$SCRATCH" 2>/dev/null || SCRATCH="$OUT/_scratch"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
export TRITON_CACHE_DIR="$SCRATCH/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
export VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"
export MIA_PARITY_SCRATCH="$SCRATCH/parity_scratch"
export MIA_APERTURE_DIR="$SCRATCH/aperture"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$MIA_PARITY_SCRATCH" "$MIA_APERTURE_DIR" "$VLLM_CACHE_ROOT"

echo "[AB] host=$(hostname) job=$JOB"
echo "[AB] branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) sha=$(git -C "$REPO" rev-parse --short HEAD)"
echo "[AB] python=$PY"
echo "[AB] vllm=$("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
echo "[AB] out=$OUT scratch=$SCRATCH"

MIA_EPS=$("$PY" - <<'EOF' 2>/dev/null
from importlib.metadata import entry_points
names = sorted(e.name for e in entry_points(group="vllm.general_plugins")
               if e.value.split(".", 1)[0].split(":", 1)[0] == "mia")
print(",".join(names))
EOF
)
if [ -z "$MIA_EPS" ]; then
  echo "[AB] FATAL: no vllm.general_plugins entry point belongs to the mia distribution"
  exit 1
fi
echo "[AB] MIA plugin allowlist = $MIA_EPS"
export VLLM_PLUGINS="$MIA_EPS"

"$PY" tests/mia/parity/run_steer_ab_leak_control.py "$OUT"
rc=$?

if [ "$rc" -eq 0 ]; then
  echo "[AB] script exited 0"
else
  echo "[AB] script exited $rc"
fi
exit "$rc"
