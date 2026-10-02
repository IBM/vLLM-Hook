#!/bin/bash
# run_parity.sh — T0 + T1: the GPU proof that capture and steering on the V2 runner are CORRECT.
#
# Two arms on one GPU, in one job, one engine per process:
#   ref — vanilla vLLM 0.29 with VLLM_PLUGINS="" (MIA never loads), capturing — or steering —
#         through torch's own register_forward_hook (tests/mia/parity/t1_reference.py)
#   mia — this branch's plugin, driven by the shared workload (tests/mia/parity/capture_workload.py)
# Same weights, same kernels, same version, same V2 runner, same prompts, greedy, eager,
# one request resident at a time — so the two artifact trees must be BIT-EXACT. A
# tolerance is never introduced here: a real difference means the port is broken.
#
# FIVE LEGS:
#   T1 capture_hs / capture_qk — bit-exact artifact comparison (task C4).
#   T1 BATCHED oracle          — the same bit-exact comparison with 33 DISTINCT-length
#                                requests in ONE batch (task D5 item 8). Every T2 gate is
#                                MIA-vs-MIA — both arms import StepView from mia/runner.py,
#                                which IS the port surface — so this is the only place
#                                MULTI-ROW routing is checked against something outside MIA.
#                                The reference finds each request's rows from the TOKEN IDS
#                                the model was fed, not from any routing table. Eager only,
#                                irreducibly: a Python forward hook cannot observe a
#                                CUDA-graph replay (no Python runs during a graph launch),
#                                so an independent oracle for graph mode is not
#                                constructible this way. See TOLERANCES.md.
#   T1 steer                   — steering writes NO artifact; it mutates the residual, so the
#                                observable is its EFFECT on the next-token logprobs. Compared
#                                at STEER_ATOL (below) — the ONE tolerance in this suite, see
#                                tests/mia/parity/TOLERANCES.md. Each arm also carries its own
#                                unsteered control: a steer that matched the other arm while
#                                doing NOTHING would be a vacuous pass, so both arms must
#                                differ from their control by more than float noise.
#   T0 non-interference        — the MIA capture trees above vs a run with NO plugin and NO
#                                hooks at all: generation must be bit-identical.
#
# The oracle shares no routing, no aperture and no custom op with MIA. It does share the
# arm-neutral half of capture_workload.py (prompts, engine kwargs, sampling params,
# safetensors serialization), which is what makes this a controlled experiment.
#
# LAYER NUMBERING IS NOT SHARED BETWEEN THE TWO CAPTURE PATHS: MIA's hidden-state layer
# numbers are 1-based and its q/k layer numbers are 0-based, while torch's block index is
# 0-based in both cases. hs_small's (1, 16, 32) and qk_small's (0, 15, 31) therefore name
# the same three physical blocks. t1_reference.assert_layer_mapping() prints and enforces
# the mapping; grep "[T1] layer map" in this log to see it.
#
# Submit:  bash -lc 'cd /u/fangyunh/vLLM-Hook && bsub < tests/mia/parity/run_parity.sh'
# Watch:   bjobs ; bpeek <JOBID>
# Result:  grep -E 'VERDICT|MISMATCH|COUNT|\[T0\]|\[T1\]' bluevela/logs/parity.<JOBID>.out
#
#BSUB -J mia_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o bluevela/logs/parity.%J.out
#BSUB -e bluevela/logs/parity.%J.err
set -uo pipefail

REPO=/u/fangyunh/vLLM-Hook
PY=/proj/dmfexp/fangyunh/envs/mia_v029/bin/python
cd "$REPO"

# Model resolution pings the HF API even for a cached model, and a 429 surfaces as an
# opaque startup failure. The weights live on /proj, not in the tight $HOME quota.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/proj/dmfexp/fangyunh/vLLM-Hook-offload/home_hf_cache
# An OpenBLAS thread storm against the node's RLIMIT_NPROC segfaults natively and reads
# like a code bug.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Eager capture needs Dynamo off; set explicitly so the two arms cannot differ on it.
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# One request resident at a time, in BOTH arms: every forward pass then holds exactly one
# request's rows, so the reference needs no batch-row arithmetic and shares none of the
# index logic under test. Read by capture_workload.ENGINE_KWARGS, which both arms use.
export MIA_PARITY_MAX_NUM_SEQS=1
# The legs below import tests.mia.parity.* from inline python; the repo root is not on a
# script's sys.path automatically (sys.path[0] is the SCRIPT's directory).
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# The only tolerance in this suite, and it applies to the steer leg alone: steering is
# compared through its effect on float logprobs, not as a bit-exact artifact. Rationale,
# date, versions, model and dtype are recorded in tests/mia/parity/TOLERANCES.md. Widening it
# to make a run pass is exactly the move that document exists to prevent.
STEER_ATOL=1e-5

# Which tiers this job runs. T0/T1 (the vanilla-vLLM reference legs) are eager-only and
# already GPU-proven; T2 (task D4) is the FULL-cudagraph tier and boots ~16 engines, so the
# two are separable: RUN_TIERS=t2 re-runs only the graph invariants.
RUN_TIERS="${RUN_TIERS:-t0t1,t2}"
case ",$RUN_TIERS," in *",t0t1,"*) RUN_T0T1=1 ;; *) RUN_T0T1=0 ;; esac
case ",$RUN_TIERS," in *",t2,"*)   RUN_T2=1   ;; *) RUN_T2=0   ;; esac

JOB="${LSB_JOBID:-manual}"
OUT=/proj/dmfexp/fangyunh/mia_parity/$JOB
mkdir -p "$OUT"

SCRATCH="/opt/nvme/${USER}/mia_parity_${JOB}"
mkdir -p "$SCRATCH" 2>/dev/null || SCRATCH="$OUT/_scratch"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
export TRITON_CACHE_DIR="$SCRATCH/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
# vLLM's torch.compile / AOT cache defaults to $HOME/.cache/vllm, which is the wrong side
# of the quota AND persists across jobs -- a stale entry from another branch would be
# loaded silently. Node-local and per-job. NOT per-LEG on purpose: the HS legs compile
# first and the QK/steer legs must then miss, which is the regression test for the
# compile-cache key stamp (see mia/_plugin.py::stamp_compile_cache_key).
export VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"
# Never left unset: the capture spill and disk-sink directories default under $HOME and
# blow the quota. run_workload() repoints MIA_APERTURE_DIR inside MIA_PARITY_SCRATCH.
export MIA_PARITY_SCRATCH="$SCRATCH/parity_scratch"
export MIA_APERTURE_DIR="$SCRATCH/aperture"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$MIA_PARITY_SCRATCH" "$MIA_APERTURE_DIR" "$VLLM_CACHE_ROOT"

echo "[T1] host=$(hostname) job=$JOB"
echo "[T1] branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) sha=$(git -C "$REPO" rev-parse --short HEAD)"
echo "[T1] python=$PY"
echo "[T1] vllm=$("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
echo "[T1] torch=$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "[T1] out=$OUT scratch=$SCRATCH"

# MIA's entry-point names are discovered, not hard-coded: a run that allowed a name that
# does not exist would load NO plugin, capture nothing, and compare "equal" to another
# empty tree. assert_arm() in the workload then re-checks the arm from inside the process.
MIA_EPS=$("$PY" - <<'EOF' 2>/dev/null
from importlib.metadata import entry_points
names = sorted(e.name for e in entry_points(group="vllm.general_plugins")
               if e.value.split(".", 1)[0].split(":", 1)[0] == "mia")
print(",".join(names))
EOF
)
if [ -z "$MIA_EPS" ]; then
  echo "[T1] FATAL: no vllm.general_plugins entry point belongs to the mia distribution"
  exit 1
fi
echo "[T1] MIA plugin allowlist = $MIA_EPS"

# WHAT THIS RUN ENFORCES, recorded before anything boots (task D5 item 4). Three of these
# bounds were round numbers with no discrimination test and a live environment override:
# `MIA_T2_T0_GRAPH_BAND=10 MIA_T2_STEER_FUSED_BAND=10 MIA_T2_GPU_ROUTING_BAND=10 pytest`
# passed 88 tests, and at STEER_FUSED_BAND=10 a completely broken fused steer passed too.
# This job now REFUSES to start if any override widens a documented band -- `--print-bands`
# resolves and validates every one of them at import, so reaching the VERDICT line below
# already means nothing was widened -- and the resolved values go in the log either way, so
# a green run can be read against the bounds it actually ran under.
echo "=== [T2] enforced bounds ==="
if ! "$PY" tests/mia/parity/t2_invariants.py --print-bands; then
  echo "[T1] FATAL: a band override was refused (see the message above). This job enforces"
  echo "[T1] the documented bounds; widening one from the environment would make its PASS"
  echo "[T1] mean something other than what tests/mia/parity/TOLERANCES.md says."
  exit 1
fi

# A clean slate: a stale tree would let an arm that captured nothing look identical to an
# arm that captured something.
case "$OUT" in /proj/dmfexp/fangyunh/mia_parity/*) rm -rf "${OUT:?}"/ref_* "${OUT:?}"/mia_* "${OUT:?}"/refb_* "${OUT:?}"/miab_* "${OUT:?}"/t0_* "${OUT:?}"/t2_* ;;
  *) echo "[T1] FATAL: refusing to clear $OUT"; exit 1 ;; esac

rc=0
if [ "$RUN_T0T1" = 1 ]; then
for kind in capture_hs capture_qk; do
  workload="${kind#capture_}_small"

  echo "=== [T1] reference (vanilla vLLM, no plugin) :: $kind ==="
  ( cd "$REPO" && VLLM_PLUGINS="" "$PY" tests/mia/parity/t1_reference.py "$kind" "$OUT/ref_$kind" ) \
    || { echo "[T1] reference arm FAILED for $kind"; rc=1; continue; }

  echo "=== [T1] MIA (this branch) :: $workload ==="
  ( cd "$REPO" && VLLM_PLUGINS="$MIA_EPS" "$PY" tests/mia/parity/capture_workload.py "$workload" "$OUT/mia_$kind" ) \
    || { echo "[T1] MIA arm FAILED for $kind"; rc=1; continue; }

  echo "=== [T1] inventory :: $kind ==="
  "$PY" - "$OUT/ref_$kind" "$OUT/mia_$kind" <<'EOF'
import sys
from pathlib import Path
from safetensors.torch import load_file
for label, root in (("ref", Path(sys.argv[1])), ("mia", Path(sys.argv[2]))):
    paths = sorted(root.rglob("*.safetensors"))
    print(f"[T1] {label}: {len(paths)} artifact files under {root}")
    for path in paths:
        rel = path.relative_to(root).as_posix()
        for key, value in load_file(str(path)).items():
            print(f"[T1] {label}  {rel}::{key}  {tuple(value.shape)} {value.dtype}")
EOF

  echo "=== [T1] compare :: $kind ==="
  "$PY" tests/mia/parity/compare_artifacts.py "$OUT/ref_$kind" "$OUT/mia_$kind" \
      --require-bit-exact --label "T1 $kind" || rc=1
done


cap_rc="$rc"
if [ "$cap_rc" -eq 0 ]; then echo "VERDICT T1 capture: PASS"; else echo "VERDICT T1 capture: FAIL"; fi

# =============================================================================
# T1 BATCHED — the independent oracle at width > 1 (task D5 item 8)
# =============================================================================
# THE COMMON MODE THIS BREAKS. Every T2 gate is MIA-vs-MIA: both arms import StepView /
# step_view from mia/runner.py, which IS the V1->V2 port surface, so a bug there moves both
# arms identically and every T2 gate still passes. T1 was the only oracle outside MIA and it
# ran ONE request at a time, so BATCHING -- the thing the port actually changed -- had never
# been checked against anything but MIA itself.
#
# These two legs run 33 requests with 33 DISTINCT prompt lengths in ONE batch, vanilla vLLM
# with torch forward hooks against MIA's capture. The reference decides which rows of a
# batched forward pass belong to which request from the TOKEN IDS the model was fed
# (t1_reference.segment_batch_rows) -- no query_start_loc, no idx_mapping, nothing from
# mia/runner.py -- which is what keeps it an oracle rather than a second copy of the code
# under test. Distinct lengths also mean a whole-request mis-route lands as a SHAPE
# mismatch, with no tolerance involved, at ANY band width.
#
# GENERATION vs CAPTURED PAYLOAD are judged separately, and BOTH are now banded (task D10
# item 1 revised task D7's claim that generation stayed bit-exact). `layer*.safetensors`
# (the captured HS/QK payload) is BANDED with `_T1_BATCHED_REL_BAND` / `_T1_BATCHED_ROW_BAND`
# (2e-2 / 1.5e-1, the SAME magnitude as QK's T2 `routing-identity-w32` band): both capture
# kinds failed bit-exact in LSF 1725139 (QK worst rel 7.457e-04, HS worst rel 1.272e-03) at
# or below the re-derived vLLM boot-nondeterminism floor (1.446e-03 / 1.880e-03
# tensor-global/per-row, all 15 pairwise boots of LSF 1710884) -- the same kernel noise
# already documented for QK's own two-mechanism T2 gate, not a new regime. Integer metadata
# and shape stay bit-exact and fatal INSIDE the banded comparison too, so this
# distinct-length construction still turns a whole-request mis-route into a shape failure
# regardless of the band.
#
# `generation.safetensors` (prompts, token ids, logprobs) was believed bit-exact in LSF
# 1725139, but LSF 1731353 failed it on BOTH capture kinds' `cumulative_logprob` and
# `token_logprobs` (max|d| 1.854e-02 / 1.269e-02) with `token_ids` and `prompt_token_ids`
# UNCHANGED on every request -- the SAME cross-boot logprob noise `_T0_GRAPH_LOGPROB_BAND`
# already documents, now seen between the vanilla-vLLM reference boot and the MIA boot.
# `--compare-t1-batched-generation` (`_compare_generation_band`, `_T1_BATCHED_LOGPROB_BAND`)
# keeps token ids and prompt token ids BIT-EXACT and FATAL -- that is the claim this gate
# exists to make: MIA does not change what the model generates -- and bands only the two
# logprob channels. See `t2_invariants.py`'s band-registry comment item 8/8b and
# tests/mia/parity/TOLERANCES.md for the full derivation and margins.
#
# WIDTH 33 IN BOTH ARMS: all 33 resident at once, so there is no second prefill wave and the
# comparison is about routing rather than scheduling. The reference REFUSES to run at any
# other width rather than silently comparing the wrong rows.
#
# THE IRREDUCIBLE LIMIT, stated rather than papered over: a Python forward hook CANNOT
# OBSERVE A CUDA-GRAPH REPLAY -- a replay is a graph launch and no Python runs during it, so
# register_forward_hook never fires. An independent oracle for GRAPH mode is NOT
# CONSTRUCTIBLE this way, by any hook on any module. These legs are eager, and graph-mode
# correctness still rests on MIA-vs-MIA (T2's op-identity / routing-identity) plus the
# replay band. See tests/mia/parity/TOLERANCES.md.
batch_rc=0
for kind in capture_hs capture_qk; do
  workload="${kind#capture_}_batch"

  echo "=== [T1] BATCHED reference (vanilla vLLM, no plugin, width 33) :: $kind ==="
  ( cd "$REPO" && MIA_PARITY_MAX_NUM_SEQS=33 VLLM_PLUGINS="" \
      "$PY" tests/mia/parity/t1_reference.py "$kind" "$OUT/refb_$kind" --batched ) \
    || { echo "[T1] BATCHED reference arm FAILED for $kind"; batch_rc=1; continue; }

  echo "=== [T1] BATCHED MIA (this branch, width 33) :: $workload ==="
  ( cd "$REPO" && MIA_PARITY_MAX_NUM_SEQS=33 VLLM_PLUGINS="$MIA_EPS" \
      "$PY" tests/mia/parity/capture_workload.py "$workload" "$OUT/miab_$kind" ) \
    || { echo "[T1] BATCHED MIA arm FAILED for $kind"; batch_rc=1; continue; }

  echo "=== [T1] BATCHED compare (generation, token ids bit-exact / logprobs BANDED) :: $kind ==="
  "$PY" tests/mia/parity/t2_invariants.py \
      --compare-t1-batched-generation "$OUT/refb_$kind" "$OUT/miab_$kind" \
      --label "T1-batched $kind generation [BANDED]" || batch_rc=1

  echo "=== [T1] BATCHED compare (captured payload, [BANDED]) :: $kind ==="
  "$PY" tests/mia/parity/t2_invariants.py --compare-t1-batched "$OUT/refb_$kind" "$OUT/miab_$kind" \
      --name "layer*.safetensors" --label "T1-batched $kind capture [BANDED]" || batch_rc=1
done
[ "$batch_rc" -eq 0 ] || rc=1
if [ "$batch_rc" -eq 0 ]; then echo "VERDICT T1 batched-oracle: PASS"
else echo "VERDICT T1 batched-oracle: FAIL"; fi

# =============================================================================
# T1 steer — the effect, not an artifact
# =============================================================================
# The reference applies the SAME steering vector, at the SAME physical block, with the
# SAME method and coefficient, via a naive register_forward_hook in a process where MIA is
# not importable. Both arms take that intervention from ONE function
# (capture_workload.steer_config) and print a fingerprint of it — including a sha256 of the
# vector FILE — so "both arms applied the same intervention" is evidence in this log, not
# an assumption. grep "steer fingerprint".
#
# Layer numbering, AGAIN a separate convention: MIA's steer path filters on the RAW 0-based
# PyTorch block index (steer_worker registers ln=layer_num, with no +1 — unlike HS). So
# steer_small's optimal_layer=[15] is model.layers.15. grep "[T1] layer map steer".
steer_rc=0
echo "=== [T1] reference steer (vanilla vLLM, naive forward hook) ==="
( cd "$REPO" && VLLM_PLUGINS="" "$PY" tests/mia/parity/t1_reference.py steer "$OUT/ref_steer" ) \
  || { echo "[T1] reference arm FAILED for steer"; steer_rc=1; }

if [ "$steer_rc" -eq 0 ]; then
  echo "=== [T1] MIA steer (this branch) :: steer_small ==="
  ( cd "$REPO" && VLLM_PLUGINS="$MIA_EPS" "$PY" tests/mia/parity/capture_workload.py steer_small "$OUT/mia_steer" ) \
    || { echo "[T1] MIA arm FAILED for steer"; steer_rc=1; }
fi

if [ "$steer_rc" -eq 0 ]; then
  echo "=== [T1] steer liveness :: each arm vs its OWN unsteered control ==="
  # Non-vacuity, in the direction the artifact comparison cannot see: if steering silently
  # became a no-op in BOTH arms they would still agree perfectly. Assert on logprobs, never
  # on token ids — task A6 measured this exact workload moving the logprobs on 3/3 requests
  # and the token ids on 0/3.
  "$PY" - "$OUT/ref_steer" "$OUT/mia_steer" <<'EOF'
import sys
from pathlib import Path

from tests.mia.parity.capture_workload import STEER_LIVENESS_ATOL
from tests.mia.parity.compare_artifacts import steer_logprob_delta

bad = 0
for label, root in (("ref", Path(sys.argv[1])), ("mia", Path(sys.argv[2]))):
    delta = steer_logprob_delta(root, root, label_b="control")
    alive = delta > STEER_LIVENESS_ATOL
    bad += 0 if alive else 1
    print(f"[T1] steer liveness {label}: max|d(logprob)| steered-vs-control = {delta:.6e} "
          f"(floor {STEER_LIVENESS_ATOL:.1e}) -> {'ALIVE' if alive else 'INERT'}")
print(f"VERDICT T1 steer liveness: {'PASS' if bad == 0 else 'FAIL'} ({bad} inert arms)")
sys.exit(1 if bad else 0)
EOF
  [ $? -eq 0 ] || steer_rc=1

  echo "=== [T1] compare :: steer (atol=$STEER_ATOL, the suite's only tolerance) ==="
  "$PY" tests/mia/parity/compare_artifacts.py "$OUT/ref_steer" "$OUT/mia_steer" \
      --atol "$STEER_ATOL" --summary --label "T1 steer" || steer_rc=1
fi
[ "$steer_rc" -eq 0 ] || rc=1

# =============================================================================
# T0 — capture is an observer
# =============================================================================
# The capture-ON arms are the MIA trees the T1 legs already produced (same workloads, same
# engine, capture armed). The capture-OFF control is fresh: no plugin, no hooks, nothing
# attached anywhere — which is what T1's reference arm, carrying torch hooks of its own,
# cannot be. Only the generation files are compared; the capture arm's payload has no
# counterpart on the off side by definition.
t0_rc=0
echo "=== [T0] capture-off arm (no plugin, no hooks) ==="
( cd "$REPO" && VLLM_PLUGINS="" "$PY" tests/mia/parity/test_t0_noninterference.py off "$OUT/t0_off" ) \
  || { echo "[T0] capture-off arm FAILED"; t0_rc=1; }

if [ "$t0_rc" -eq 0 ]; then
  for kind in capture_hs capture_qk; do
    if [ ! -d "$OUT/mia_$kind" ]; then
      echo "[T0] no capture-on tree for $kind (its T1 leg failed); cannot judge T0"
      t0_rc=1; continue
    fi
    "$PY" tests/mia/parity/compare_artifacts.py "$OUT/t0_off" "$OUT/mia_$kind" \
        --require-bit-exact --name generation.safetensors \
        --label "T0 ${kind#capture_}" || t0_rc=1
  done
fi
[ "$t0_rc" -eq 0 ] || rc=1

echo "[T1] steer/capture artifacts kept under $OUT"
if [ "$steer_rc" -eq 0 ]; then echo "VERDICT T1 steer: PASS"; else echo "VERDICT T1 steer: FAIL"; fi
if [ "$t0_rc" -eq 0 ]; then echo "VERDICT T0: PASS"; else echo "VERDICT T0: FAIL"; fi
if [ "$rc" -eq 0 ]; then echo "VERDICT T0+T1: PASS"; else echo "VERDICT T0+T1: FAIL"; fi
fi   # RUN_T0T1

# =============================================================================
# T2 — MIA's OWN invariants (task D4): eager == FULL, and alone == in-batch
# =============================================================================
# No external reference can check these two. T1 proves MIA agrees with vanilla vLLM at
# max_num_seqs=1 and eager; T2 is the first FULL-cudagraph tier and the first time more than
# one request shares a forward pass, which is where V2's padded rows / idx_mapping / num_reqs
# slicing actually get exercised.
#
# Every arm is its OWN PROCESS (an exclusive_process GPU refuses a second engine in one
# process, and each routing lever is read once at install), so t2_invariants.py orchestrates
# and holds no engine itself. TORCHDYNAMO_DISABLE is set PER LEG there, not here: the eager
# path needs it on (torch.compile replaces the forward, so register_forward_hook never fires)
# and the graph path needs it off (its work is baked into a custom op that survives compile).
# vLLM 0.29's default cudagraph_mode is FULL_AND_PIECEWISE, which MIA refuses; every graph arm
# names FULL explicitly (capture_workload.run_workload).
if [ "$RUN_T2" = 1 ]; then
  t2_rc=0
  export VLLM_PLUGINS="$MIA_EPS"
  export STEER_ATOL
  # Per-leg subdirectories are created under these two roots by the orchestrator.
  unset MIA_PARITY_MAX_NUM_SEQS
  for w in hs_small qk_small steer_small; do
    echo "============================================================"
    echo "=== [T2] workload $w ==="
    echo "============================================================"
    "$PY" -u tests/mia/parity/t2_invariants.py "$w" "$OUT/t2_$w" || t2_rc=1
  done
  [ "$t2_rc" -eq 0 ] || rc=1
  if [ "$t2_rc" -eq 0 ]; then echo "VERDICT T2: PASS"; else echo "VERDICT T2: FAIL"; fi
fi

echo "[T2] artifacts kept under $OUT"
if [ "$rc" -eq 0 ]; then echo "VERDICT PARITY ($RUN_TIERS): PASS"; else echo "VERDICT PARITY ($RUN_TIERS): FAIL"; fi
exit "$rc"
