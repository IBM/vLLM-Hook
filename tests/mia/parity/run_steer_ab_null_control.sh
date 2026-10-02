#!/bin/bash
# run_steer_ab_null_control.sh — the NULL control for the steer A/B design (task D11).
#
# The in-suite gate `steer_small per-request-arming` and the standalone
# run_steer_ab_leak_control.py now DRIVE THE SAME FUNCTION (run_steer_ab_pass, task D10) at
# the same configuration -- same 33 distinct-length prompts, same alternating mask, same
# max_num_seqs=33, same graph+cudagraph, same fingerprints -- and still disagree: LSF
# 1725670 measured all 17 unarmed requests bit-exact 0.0, LSF 1731353 flagged
# {req0,req2,req30}, LSF 1732150 flagged {req14,16,18,20,22,24,26,28,30,32}. Two runs of one
# implementation flagging two nearly disjoint sets is not a parameter difference.
#
# The A/B design has no pass that differs from another pass in NOTHING, so it cannot tell
# "steering leaked onto an unarmed row" from "two passes of this engine do not reproduce
# each other bit-for-bit". This job adds those passes: ONE boot, five passes over the same
# batch -- P1 real, P2 zero-dir, P3 real (null for P1), P4 zero-dir (null for P2), P5 the
# real vector's bytes at a DIFFERENT vector_path (isolates Arm B's path change from its
# value change). See the script's docstring for how each outcome is read.
#
# Submit:  bash -lc 'cd /u/fangyunh/vLLM-Hook && bsub < tests/mia/parity/run_steer_ab_null_control.sh'
# Watch:   bjobs ; bpeek <JOBID>
# Result:  grep -E 'VERDICT|SUMMARY|\[NULL\]' bluevela/logs/steer_null.<JOBID>.out
#
#BSUB -J mia_steer_null
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o bluevela/logs/steer_null.%J.out
#BSUB -e bluevela/logs/steer_null.%J.err
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
OUT=/proj/dmfexp/fangyunh/mia_parity/$JOB/steer_ab_null_control
mkdir -p "$OUT"

SCRATCH="/opt/nvme/${USER}/mia_steer_null_${JOB}"
mkdir -p "$SCRATCH" 2>/dev/null || SCRATCH="$OUT/_scratch"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
export TRITON_CACHE_DIR="$SCRATCH/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
export VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"
export MIA_PARITY_SCRATCH="$SCRATCH/parity_scratch"
export MIA_APERTURE_DIR="$SCRATCH/aperture"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$MIA_PARITY_SCRATCH" "$MIA_APERTURE_DIR" "$VLLM_CACHE_ROOT"

echo "[NULL] host=$(hostname) job=$JOB"
echo "[NULL] branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) sha=$(git -C "$REPO" rev-parse --short HEAD)"
echo "[NULL] python=$PY"
echo "[NULL] vllm=$("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
echo "[NULL] out=$OUT scratch=$SCRATCH"

MIA_EPS=$("$PY" - <<'EOF' 2>/dev/null
from importlib.metadata import entry_points
names = sorted(e.name for e in entry_points(group="vllm.general_plugins")
               if e.value.split(".", 1)[0].split(":", 1)[0] == "mia")
print(",".join(names))
EOF
)
if [ -z "$MIA_EPS" ]; then
  echo "[NULL] FATAL: no vllm.general_plugins entry point belongs to the mia distribution"
  exit 1
fi
echo "[NULL] MIA plugin allowlist = $MIA_EPS"
export VLLM_PLUGINS="$MIA_EPS"

"$PY" tests/mia/parity/run_steer_ab_null_control.py "$OUT"
rc=$?

if [ "$rc" -eq 0 ]; then
  echo "[NULL] script exited 0"
else
  echo "[NULL] script exited $rc"
fi
exit "$rc"
