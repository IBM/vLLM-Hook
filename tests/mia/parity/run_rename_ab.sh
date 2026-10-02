#!/bin/bash
# run_rename_ab.sh — the Phase-A exit gate: prove the rename is byte-inert.
#
# Runs the SAME three workloads twice on one GPU, in one job:
#   arm A — the PRE-rename package, from a read-only worktree of `main`
#   arm B — this branch's renamed package
# same vLLM, same V1 runner, same node, same GPU, same prompts — then compares the two
# artifact trees for BIT-EXACT equality. Arm B's tree is also the reference the later
# cross-version tiers are measured against.
#
# How each arm loads its own plugin: vLLM discovers plugins through the
# `vllm.general_plugins` entry-point group and filters them with the VLLM_PLUGINS
# allowlist, which is keyed on the entry-point NAME. The two distributions are disjoint on
# every axis (distribution name, top-level package, entry-point names), so both can be
# installed at once and each arm allows only its own. This file never spells the retired
# names out: it discovers them, which keeps the naming policy (tests/test_naming_policy.py
# scans every .sh here) honest AND fails loudly if the pre-rename package is missing
# instead of silently running with no plugin at all — a run that captures nothing would
# otherwise compare "equal" to another run that captures nothing.
#
# Submit:  bash -lc 'bsub -G grp_exploratory < tests/mia/parity/run_rename_ab.sh'
# Watch:   bjobs ; bpeek <JOBID>
# Result:  grep -E 'VERDICT|MISMATCH|COUNT|\[A6\]' bluevela/logs/rename_ab.<JOBID>.out
#
#BSUB -J mia_rename_ab
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o bluevela/logs/rename_ab.%J.out
#BSUB -e bluevela/logs/rename_ab.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

REPO=/u/fangyunh/vLLM-Hook
PY=/u/fangyunh/miniconda3/envs/vllm_hook_env/bin/python
PIP=/u/fangyunh/miniconda3/envs/vllm_hook_env/bin/pip
REF=/proj/dmfexp/fangyunh/mia_ref/0.21_v1
# Shared storage, not node-local /tmp: the login node and the compute nodes do not share
# /tmp, so a worktree created at submit time would not exist where the job runs.
WORKTREE="${LEGACY_WORKTREE:-/proj/dmfexp/fangyunh/mia_ref/_main_worktree}"
SHIM=/proj/dmfexp/fangyunh/mia_ref/_legacy_run.py
WORKLOADS="hs_small qk_small steer_small"

# Model resolution pings the HF API even for a cached model; a 429 surfaces as an opaque
# startup failure with exit 0. The models live on /proj, not in the tight $HOME quota.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/proj/dmfexp/fangyunh/vLLM-Hook-offload/home_hf_cache
# An OpenBLAS thread storm against the node's RLIMIT_NPROC segfaults natively and reads
# like a code bug.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Set explicitly rather than left to each package's own import-time default, so the two
# arms cannot differ on compilation. Eager capture needs Dynamo off.
export TORCHDYNAMO_DISABLE=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

TMPDIR="/opt/nvme/${USER}/rename_ab_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR" 2>/dev/null || TMPDIR="/tmp/mia_${USER}/rename_ab_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR"
export TMPDIR
export TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "[A6] host=$(hostname) job=${LSB_JOBID:-manual}"
echo "[A6] branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) sha=$(git -C "$REPO" rev-parse --short HEAD)"
echo "[A6] vllm=$("$PY" -c 'import vllm; print(vllm.__version__)')"

# ---------------------------------------------------------------------------
# Arm A's plugin: the pre-rename distribution, installed alongside this one.
# ---------------------------------------------------------------------------
if [ ! -d "$WORKTREE" ]; then
  echo "[A6] creating the read-only worktree of main at $WORKTREE"
  git -C "$REPO" worktree add --detach "$WORKTREE" main
fi
echo "[A6] arm-A worktree sha=$(git -C "$WORKTREE" rev-parse --short HEAD)"

# Discover, never hard-code: the legacy distribution is the one publishing
# vllm.general_plugins entry points that are neither this package's nor vLLM's own.
discover_legacy() {
  "$PY" -c '
import importlib.metadata as md
rows = []
for dist in md.distributions():
    name = (dist.metadata["Name"] or "").lower()
    if name in ("mia", "vllm"):
        continue
    eps = [e.name for e in dist.entry_points if e.group == "vllm.general_plugins"]
    if eps:
        rows.append((dist.metadata["Name"], ",".join(sorted(eps))))
print(*(rows[0] if len(rows) == 1 else ("NONE", "NONE")))
'
}

LEGACY_INFO=$(discover_legacy)
if [ "${LEGACY_INFO%% *}" = "NONE" ]; then
  echo "[A6] installing the pre-rename package from the worktree"
  "$PIP" install -e "$WORKTREE"/*_plugins --no-deps --no-build-isolation
  LEGACY_INFO=$(discover_legacy)
fi
LEGACY_DIST=${LEGACY_INFO%% *}
LEGACY_EPS=${LEGACY_INFO##* }
[ "$LEGACY_DIST" != "NONE" ] || { echo "[A6] FATAL: no pre-rename distribution installed"; exit 1; }
MIA_EPS=$("$PY" -c 'import importlib.metadata as m; print(",".join(sorted(e.name for e in m.distribution("mia").entry_points if e.group=="vllm.general_plugins")))')
echo "[A6] arm A dist=$LEGACY_DIST allowlist=$LEGACY_EPS"
echo "[A6] arm B dist=mia allowlist=$MIA_EPS"

# Arm A's driver: the same shared workload definition, driven through the pre-rename API.
# Written here, never committed, and deliberately generic — it resolves the old package,
# its wrapper class and its registry from the allowlist at run time.
mkdir -p "$(dirname "$SHIM")"
cat > "$SHIM" <<'SHIMEOF'
"""Arm A driver: run a shared parity workload through the PRE-rename package.

Generated by tests/mia/parity/run_rename_ab.sh; never committed. Everything about the old
package is discovered from the VLLM_PLUGINS allowlist, so this file makes no assumption
beyond "the allowed plugin package exposes a *LLM wrapper class and a .registry module".
"""
import importlib
import importlib.util
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

# The pre-rename worker-kind names. These are not brand names and survive verbatim.
WORKER_KIND = {"hs": "probe_hidden_states", "qk": "probe_hook_qk", "steer": "steer_hook_act"}


def main():
    workload_name, out_dir = sys.argv[1], sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    spec = importlib.util.spec_from_file_location("parity_shared",
                                                  os.environ["PARITY_SHARED"])
    shared = importlib.util.module_from_spec(spec)
    # Registered before exec: the shared module pairs `from __future__ import annotations`
    # with a dataclass, and dataclasses resolves those through sys.modules[__module__].
    sys.modules[spec.name] = shared
    spec.loader.exec_module(shared)

    import vllm.envs as envs
    import vllm.plugins

    allowed = set(envs.VLLM_PLUGINS or [])
    tops = {ep.value.split(".", 1)[0].split(":", 1)[0]
            for ep in entry_points(group="vllm.general_plugins") if ep.name in allowed}
    tops.discard("vllm")
    if len(tops) != 1:
        raise SystemExit(f"expected exactly one allowed plugin package, got {sorted(tops)}")
    top = tops.pop()

    package = importlib.import_module(top)
    wrappers = [v for k, v in vars(package).items()
                if isinstance(v, type) and k.endswith("LLM")]
    if len(wrappers) != 1:
        raise SystemExit(f"expected exactly one *LLM wrapper in {top}, got {wrappers}")
    wrapper = wrappers[0]

    vllm.plugins.load_general_plugins()
    registry = importlib.import_module(top + ".registry").PluginRegistry
    wl = shared.WORKLOADS[workload_name]
    kind = WORKER_KIND[wl.subsystem]
    shared.assert_arm(top, registry.get_worker(kind).path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scratch = shared.scratch_dir(f"{workload_name}_eager_legacy")
    llm = wrapper(
        model=shared.MODEL,
        worker_name=kind,
        enforce_eager=True,
        hook_dir=str(scratch / "sink"),
        **shared.ENGINE_KWARGS,
    )
    shared._drive(llm.llm, wl, out, batch)


# vLLM spawns its engine-core process, and a spawned child re-imports this module. Without
# this guard the child would boot a second engine on an exclusive_process GPU.
if __name__ == "__main__":
    main()
SHIMEOF

# ---------------------------------------------------------------------------
# A clean slate: a stale tree here would let an arm that captured nothing look identical
# to an arm that captured something.
# ---------------------------------------------------------------------------
case "$REF" in */0.21_v1) rm -rf "$REF" ;; *) echo "[A6] FATAL: refusing to clear $REF"; exit 1 ;; esac
mkdir -p "$REF"

export PARITY_SHARED="$REPO/tests/mia/parity/capture_workload.py"
# Aperture spill + disk-sink directories: node NVMe, and never inside an artifact tree.
export MIA_PARITY_SCRATCH="$TMPDIR/parity_scratch"
mkdir -p "$MIA_PARITY_SCRATCH"

for w in $WORKLOADS; do
  echo "=== [A6] arm A (pre-rename) :: $w ==="
  ( cd "$WORKTREE" && VLLM_PLUGINS="$LEGACY_EPS" "$PY" "$SHIM" "$w" "$REF/A_main/$w" )
done

for w in $WORKLOADS; do
  echo "=== [A6] arm B (this branch) :: $w ==="
  ( cd "$REPO" && VLLM_PLUGINS="$MIA_EPS" "$PY" tests/mia/parity/capture_workload.py "$w" "$REF/B_v2sup/$w" )
done

# ---------------------------------------------------------------------------
# Compare: per workload (so one failure is attributable) and then the whole tree.
# ---------------------------------------------------------------------------
rc=0
for w in $WORKLOADS; do
  "$PY" "$REPO/tests/mia/parity/compare_artifacts.py" \
      "$REF/A_main/$w" "$REF/B_v2sup/$w" --require-bit-exact --label "$w" || rc=1
done
"$PY" "$REPO/tests/mia/parity/compare_artifacts.py" \
    "$REF/A_main" "$REF/B_v2sup" --require-bit-exact --label "all" || rc=1

{
  echo "job=${LSB_JOBID:-manual} host=$(hostname)"
  echo "date=$(date -Is)"
  echo "vllm=$("$PY" -c 'import vllm; print(vllm.__version__)') runner=V1"
  echo "model=${MIA_PARITY_MODEL:-microsoft/Phi-3-mini-4k-instruct}"
  echo "arm_A_sha=$(git -C "$WORKTREE" rev-parse HEAD)  dist=$LEGACY_DIST"
  echo "arm_B_sha=$(git -C "$REPO" rev-parse HEAD)  dist=mia branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "bit_exact=$([ $rc -eq 0 ] && echo yes || echo NO)"
} > "$REF/MANIFEST.txt"
cat "$REF/MANIFEST.txt"

if [ $rc -eq 0 ]; then
  echo "VERDICT: rename byte-inert"
else
  echo "VERDICT: rename NOT byte-inert — see MISMATCH lines above"
fi
exit $rc
