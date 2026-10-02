#!/bin/bash
# run_crossbranch.sh — T3: the old world (MIA on vLLM 0.21/V1) against the new one
# (MIA on vLLM 0.29/V2), with the control that says what the difference MEANS.
#
# FIVE ARMS, ONE JOB, ONE NODE, ONE GPU — so no gate crosses a node or a job boundary:
#
#   c21   vanilla vLLM 0.21 + naive register_forward_hook, MIA NEVER IMPORTED, width 1
#   a21   MIA @ the Phase-A SHA on vLLM 0.21 / V1, width 1
#   a21r  MIA @ the Phase-A SHA on vLLM 0.21 / V1, at the REFERENCE's default width
#   c29   vanilla vLLM 0.29 + naive register_forward_hook, MIA NEVER IMPORTED, width 1
#   a29   MIA @ HEAD on vLLM 0.29 / V2, width 1
#
# plus `ref`, the on-disk tree task A6 minted (read, never written).
#
# THE GATES (tests/mia/parity/t3_crossbranch.py holds the derivations):
#   G1  FATAL  a21 == c21   MIA is inert on 0.21
#   G2  FATAL  a29 == c29   MIA is inert on 0.29
#   G3  INFO   ref vs a21r  is the on-disk reference reproducible across jobs and nodes?
#   G4  CONTROL c21 vs c29  PURE vLLM 0.21->0.29 drift, MIA absent from both sides —
#                           the band's empirical anchor
#   G5  T3     a21 vs a29   structure FATAL, values BANDED at 2 x (G1 + G4 + G2), on the
#                           captured payload, the generation channel, and steer's control
#   G5b INFO   ref vs a29   the same question asked of the on-disk reference directly
#
# If G1 and G2 are both bit-exact then G5's delta IS G4's delta, and T3's result is a real
# statement rather than an unfalsifiable tolerance: MIA contributes exactly zero on both
# vLLM versions, and every remaining difference is vLLM's own.
#
# HOW THE OLD MIA IS RUN AT ALL. MIA at HEAD REFUSES the V1 runner (mia/runner.py's
# require_v2_runner, called by every worker at install time), so the 0.21 arms cannot use the
# live tree. They use a read-only git worktree pinned at the SHA that minted the reference,
# put in front of the editable install by a directory holding a single symlink to its `mia`
# package. Verified before any engine boots: the arm PRINTS the resolved mia.__file__ and
# refuses to run if it is not the worktree's. Nothing is installed, uninstalled or
# reinstalled — the shared conda env is not mutated.
#
# WIDTH IS NOT A FREE VARIABLE. The reference was minted with MIA_PARITY_MAX_NUM_SEQS unset
# (all three requests resident at once) while the vanilla oracle can only run at width 1, and
# the model is NOT batch-invariant (T2's own `CONTROL eager alone-vs-batch` fails
# bit-exactness in every job that has run it). Every FATAL gate here is therefore
# matched-width, `a21r` reproduces the reference's width so G3 is matched-width too, and the
# `WIDTH CONTROL a21-vs-a21r` line measures exactly what the width is worth.
#
# Submit:  bash -lc 'cd /u/fangyunh/vLLM-Hook && bsub -G grp_exploratory < tests/mia/parity/run_crossbranch.sh'
# Watch:   bjobs ; bpeek <JOBID>
# Result:  grep -E 'VERDICT T3|INFO T3|HEADLINE|MEASURED T3|BAND T3|CONTROL ANOMALY|BIFURCATION|MISMATCH|MISSING GATE|\[T3\]' bluevela/logs/crossbranch.<JOBID>.out
#
#BSUB -J mia_crossbranch
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o bluevela/logs/crossbranch.%J.out
#BSUB -e bluevela/logs/crossbranch.%J.err
set -uo pipefail

REPO=/u/fangyunh/vLLM-Hook
# The 0.29 env is the one MIA targets; the 0.21 env is the pre-existing validation arm.
PY29=/proj/dmfexp/fangyunh/envs/mia_v029/bin/python
PY21=/u/fangyunh/miniconda3/envs/vllm_hook_env/bin/python
REF=/proj/dmfexp/fangyunh/mia_ref/0.21_v1/B_v2sup
# The SHA that minted $REF (its MANIFEST.txt records it as arm_B_sha). The 0.21 MIA arms run
# THIS code, so "the old world" is the old world and not a guess at it.
PHASE_A_SHA=9cfbef60f72a0e8429a56d19ed60c965705193f3
# Shared storage, not node-local /tmp: the login node and the compute nodes do not share
# /tmp, so a worktree created at submit time would not exist where the job runs.
WORKTREE=/proj/dmfexp/fangyunh/mia_ref/_phaseA_worktree
SHADOW=/proj/dmfexp/fangyunh/mia_ref/_phaseA_pkg
NEUTRAL=/proj/dmfexp/fangyunh/mia_ref/_phaseA_cwd
WORKLOADS="hs_small qk_small steer_small"

cd "$REPO" || exit 1

# Model resolution pings the HF API even for a cached model, and a 429 surfaces as an opaque
# startup failure with exit 0. The weights live on /proj, not in the tight $HOME quota.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/proj/dmfexp/fangyunh/vLLM-Hook-offload/home_hf_cache
# An OpenBLAS thread storm against the node's RLIMIT_NPROC segfaults natively and reads like
# a code bug.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Eager capture needs Dynamo off; set explicitly so no two arms can differ on it.
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# The arms import tests.mia.parity.* ; a script's sys.path[0] is the SCRIPT's directory, not the
# repo root.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

JOB="${LSB_JOBID:-manual}"
# /u is at its quota ceiling — artifact trees go to /proj, scratch to node NVMe, and only the
# ~200KB LSF log lands under $HOME.
OUT=/proj/dmfexp/fangyunh/mia_crossbranch/$JOB
mkdir -p "$OUT" || exit 1

SCRATCH="/opt/nvme/${USER}/mia_crossbranch_${JOB}"
mkdir -p "$SCRATCH" 2>/dev/null || SCRATCH="$OUT/_scratch"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH/tmp"
export TRITON_CACHE_DIR="$SCRATCH/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
# vLLM's compile/AOT cache defaults to $HOME/.cache/vllm — the wrong side of the quota, and
# it persists across jobs, so a stale entry from another branch would load silently.
export VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"
# Never left unset: the capture spill and disk-sink directories default under $HOME.
export MIA_PARITY_SCRATCH="$SCRATCH/parity_scratch"
export MIA_APERTURE_DIR="$SCRATCH/aperture"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$MIA_PARITY_SCRATCH" "$MIA_APERTURE_DIR" "$VLLM_CACHE_ROOT"

echo "[T3] host=$(hostname) job=$JOB"
echo "[T3] branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) sha=$(git -C "$REPO" rev-parse --short HEAD)"
echo "[T3] out=$OUT scratch=$SCRATCH"
echo "[T3] vllm(0.29 env)=$("$PY29" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
echo "[T3] vllm(0.21 env)=$("$PY21" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
echo "[T3] reference tree = $REF"
[ -d "$REF" ] || { echo "[T3] FATAL: the 0.21 reference tree $REF does not exist"; exit 1; }
sed 's/^/[T3] MANIFEST /' "$(dirname "$REF")/MANIFEST.txt" 2>/dev/null

# ---------------------------------------------------------------------------
# The old MIA, from a pinned worktree, without touching the editable install
# ---------------------------------------------------------------------------
if [ ! -d "$WORKTREE" ]; then
  echo "[T3] creating the read-only worktree at $WORKTREE"
  git -C "$REPO" worktree add --detach "$WORKTREE" "$PHASE_A_SHA" || exit 1
fi
WT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)
echo "[T3] worktree sha=$WT_SHA (want $PHASE_A_SHA)"
if [ "$WT_SHA" != "$PHASE_A_SHA" ]; then
  echo "[T3] FATAL: $WORKTREE is at $WT_SHA, not the SHA that minted the reference."
  echo "[T3] Comparing against a DIFFERENT old world would answer a question nobody asked."
  exit 1
fi
# A SHA pins which commit was checked out, NOT what is on disk now. This worktree is the
# SOLE definition of "the old world" for the whole tier, and editing one file under
# $WORKTREE/mia/ changes that definition while `rev-parse HEAD` keeps reporting the right
# commit. Content, not just provenance.
WT_DIRTY=$(git -C "$WORKTREE" status --porcelain)
if [ -n "$WT_DIRTY" ]; then
  echo "[T3] FATAL: $WORKTREE has uncommitted modifications -- it is not the Phase-A tree:"
  echo "$WT_DIRTY" | sed 's/^/[T3]   /'
  echo "[T3] The old world must be the code that minted the reference, byte for byte."
  exit 1
fi
echo "[T3] worktree is clean (git status --porcelain empty)"
mkdir -p "$SHADOW" "$NEUTRAL" || exit 1
ln -sfn "$WORKTREE/mia" "$SHADOW/mia" || exit 1
# `$NEUTRAL` must NOT contain a `mia` package: sys.path[0] (the script's own directory, and
# the cwd for `python -c`) is searched BEFORE PYTHONPATH, so a stray copy there would shadow
# the shadow.
[ -e "$NEUTRAL/mia" ] && { echo "[T3] FATAL: $NEUTRAL contains a mia package"; exit 1; }

# The resolution is PROVEN before any engine boots, not assumed: an arm that silently ran the
# live V2-only tree on 0.21 would crash, and one that silently ran the wrong old SHA would
# compare the wrong thing and look fine.
RESOLVED=$( cd "$NEUTRAL" && PYTHONPATH="$SHADOW:$REPO" "$PY21" -c \
  'import mia, sys; print(mia.__file__)' 2>/dev/null | tail -1 )
echo "[T3] 0.21 arm resolves mia -> $RESOLVED"
case "$RESOLVED" in
  "$SHADOW"/*) : ;;
  *) echo "[T3] FATAL: the 0.21 MIA arm would import $RESOLVED, not the pinned worktree."
     exit 1 ;;
esac

# Entry-point names are DISCOVERED, never hard-coded: allowing a name that does not exist
# loads no plugin, captures nothing, and compares "equal" to another empty tree.
discover_eps() {
  "$1" - <<'EOF' 2>/dev/null
from importlib.metadata import entry_points
names = sorted(e.name for e in entry_points(group="vllm.general_plugins")
               if e.value.split(".", 1)[0].split(":", 1)[0] == "mia")
print(",".join(names))
EOF
}
EPS29=$(discover_eps "$PY29")
EPS21=$(discover_eps "$PY21")
echo "[T3] MIA allowlist (0.29 env) = $EPS29"
echo "[T3] MIA allowlist (0.21 env) = $EPS21"
[ -n "$EPS29" ] || { echo "[T3] FATAL: no mia entry point in the 0.29 env"; exit 1; }
[ -n "$EPS21" ] || { echo "[T3] FATAL: no mia entry point in the 0.21 env"; exit 1; }

echo "=== [T3] enforced policy ==="
"$PY29" tests/mia/parity/t3_crossbranch.py --print-policy || exit 1

# A clean slate: a stale tree would let an arm that captured nothing look identical to an arm
# that captured something.
case "$OUT" in /proj/dmfexp/fangyunh/mia_crossbranch/*)
  rm -rf "${OUT:?}"/c21_* "${OUT:?}"/a21_* "${OUT:?}"/a21r_* "${OUT:?}"/c29_* "${OUT:?}"/a29_* ;;
  *) echo "[T3] FATAL: refusing to clear $OUT"; exit 1 ;; esac

# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------
# One engine per process (an LSF exclusive_process GPU rejects a second engine in the same
# process), and every capture arm runs at MIA_PARITY_MAX_NUM_SEQS=1 except `a21r`, which
# reproduces the reference's default width on purpose.
rc=0
OKDIR="$OUT/_ran"
rm -rf "$OKDIR"; mkdir -p "$OKDIR"

run_arm() {                      # run_arm <arm> <workload> <cwd> <command...>
  local arm="$1" w="$2" cwd="$3"; shift 3
  echo "=== [T3] arm $arm :: $w ==="
  if ( cd "$cwd" && "$@" ); then
    : > "$OKDIR/${arm}_${w}"
  else
    echo "[T3] arm $arm FAILED for $w (its gates will FAIL, not vanish)"
    rc=1
  fi
}

for w in $WORKLOADS; do
  case "$w" in
    hs_small)    kind=capture_hs ;;
    qk_small)    kind=capture_qk ;;
    steer_small) kind=steer ;;
  esac

  # --- vanilla vLLM 0.29, MIA never imported (t1_reference sets VLLM_PLUGINS="" itself
  #     and assert_no_plugin() PROVES no mia module is in the process).
  run_arm c29 "$w" "$REPO" env MIA_PARITY_MAX_NUM_SEQS=1 VLLM_PLUGINS="" \
      "$PY29" tests/mia/parity/t1_reference.py "$kind" "$OUT/c29_$w"

  # --- MIA @ HEAD on 0.29 / V2.
  run_arm a29 "$w" "$REPO" env MIA_PARITY_MAX_NUM_SEQS=1 VLLM_PLUGINS="$EPS29" \
      "$PY29" tests/mia/parity/capture_workload.py "$w" "$OUT/a29_$w"

  # --- vanilla vLLM 0.21, MIA never imported. The 0.21 env HAS mia editable-installed, so
  #     the empty allowlist and assert_no_plugin() matter here more than anywhere else.
  #     MIA_PARITY_ALLOW_LEGACY_RUNNER=1 is the narrow, pinned opt-in that lets the oracle
  #     accept 0.21's V1 runner (t1_reference._model_of); it cannot excuse anything on 0.29.
  run_arm c21 "$w" "$REPO" env MIA_PARITY_MAX_NUM_SEQS=1 VLLM_PLUGINS="" VLLM_USE_V1=1 \
      MIA_PARITY_ALLOW_LEGACY_RUNNER=1 \
      "$PY21" tests/mia/parity/t1_reference.py "$kind" "$OUT/c21_$w"

  # --- MIA @ the Phase-A SHA on 0.21 / V1, width 1. Run from a directory with no `mia` in
  #     it, with the pinned worktree's package ahead of the editable install. The script is
  #     invoked by ABSOLUTE PATH, so sys.path[0] is $REPO/tests/mia/parity (which holds no `mia`
  #     either) and PYTHONPATH decides -- cwd is belt-and-braces.
  #
  #     That resolution rests on capture_workload.py NOT doing `sys.path.insert(0, REPO_ROOT)`
  #     the way every other driver in that directory does. The preflight probe above proves it
  #     for a DIFFERENT process, so MIA_PARITY_REQUIRE_MIA_UNDER makes the arm re-check its
  #     own imported package and refuse to run if anything got in front of the shadow; the
  #     file carries a matching comment at the point of dependence and a test pins it.
  run_arm a21 "$w" "$NEUTRAL" env MIA_PARITY_MAX_NUM_SEQS=1 VLLM_USE_V1=1 \
      VLLM_PLUGINS="$EPS21" PYTHONPATH="$SHADOW:$REPO" \
      MIA_PARITY_REQUIRE_MIA_UNDER="$SHADOW" \
      "$PY21" "$REPO/tests/mia/parity/capture_workload.py" "$w" "$OUT/a21_$w"

  # --- ... and the same thing at the REFERENCE's width. MIA_PARITY_MAX_NUM_SEQS is never
  #     exported by this script, so simply not setting it here is vLLM's default width --
  #     exactly how run_rename_ab.sh minted the reference.
  run_arm a21r "$w" "$NEUTRAL" env VLLM_USE_V1=1 \
      VLLM_PLUGINS="$EPS21" PYTHONPATH="$SHADOW:$REPO" \
      MIA_PARITY_REQUIRE_MIA_UNDER="$SHADOW" \
      "$PY21" "$REPO/tests/mia/parity/capture_workload.py" "$w" "$OUT/a21r_$w"
done

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
# An arm that failed contributes NO path, so its gates fail with "an arm did not run" rather
# than quietly disappearing from the tally (tests/mia/parity/t3_crossbranch.py::adjudicate
# asserts the emitted gate set against the declared one in both directions).
for w in $WORKLOADS; do
  args=(--ref "$REF/$w")
  for arm in c21 a21 a21r c29 a29; do
    if [ -e "$OKDIR/${arm}_${w}" ]; then
      args+=(--"$arm" "$OUT/${arm}_$w")
    fi
  done
  echo "=== [T3] judge :: $w ==="
  "$PY29" tests/mia/parity/t3_crossbranch.py "$w" "${args[@]}" || rc=1
done

{
  echo "job=$JOB host=$(hostname)"
  echo "date=$(date -Is)"
  echo "repo_sha=$(git -C "$REPO" rev-parse HEAD) branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "phase_a_sha=$WT_SHA"
  echo "vllm_new=$("$PY29" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  echo "vllm_old=$("$PY21" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  echo "reference=$REF"
  echo "verdict=$([ $rc -eq 0 ] && echo PASS || echo FAIL)"
} > "$OUT/MANIFEST.txt"
cat "$OUT/MANIFEST.txt"

if [ $rc -eq 0 ]; then
  echo "VERDICT T3: PASS (all workloads)"
else
  echo "VERDICT T3: FAIL — see the VERDICT T3 / HEADLINE / MISMATCH lines above"
fi
exit $rc
