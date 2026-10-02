"""T0 -- capture must be an OBSERVER: identical generation with hooks on and off.

The claim is narrow and absolute: turning MIA's capture on changes nothing about what the
model produces. Not "changes little" -- nothing. So the comparison is bit-exact (no
tolerance) over the generated token ids, their per-token logprobs and the cumulative
logprob, for the same prompts under the same greedy settings.

WHY THIS IS NOT ALREADY COVERED BY T1. T1's reference arm carries torch forward hooks of
its own, so it compares capture against capture; its `generation.safetensors` agreement
shows the two capture paths perturb generation identically, which is not the same as not
perturbing it. T0's control has NOTHING attached: no plugin, no hooks, no worker
extension. That is the arm that can catch "some hook, anywhere, cost a token".

TWO PROCESSES, ALWAYS. An LSF `exclusive_process` GPU rejects a second engine in the same
process (`cudaErrorDevicesUnavailable`), so the arms cannot both run inside one pytest
process. This module is therefore dual-purpose:

  * as a SCRIPT it runs exactly one arm and writes its tree
        python tests/mia/parity/test_t0_noninterference.py {off|hs|qk} <out_dir>
  * as a TEST it spawns those scripts as child processes and compares the trees.

`run_parity.sh` drives the script form (so the job log carries the VERDICT line); the test
form is the same contract, self-contained, for a developer with a GPU.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Arm name -> the workload whose capture configuration that arm runs. "off" is the
# control: no plugin, no hooks, nothing attached.
ARMS = {"hs": "hs_small", "qk": "qk_small"}

# Only generation is comparable across the arms: the capture arm additionally writes its
# captured payload, and the control arm has none by definition. Filtering to the
# generation files is what makes the two trees line up key-for-key -- and the comparator
# still fails loudly if either side turns out to be empty.
GENERATION = "generation.safetensors"


def mia_plugin_allowlist() -> str:
    """The `VLLM_PLUGINS` value that loads MIA and nothing else.

    Discovered, never hard-coded: an allowlist naming an entry point that does not exist
    loads NO plugin, captures nothing, and would make the capture arm identical to the
    control for exactly the wrong reason.
    """
    from importlib.metadata import entry_points

    names = sorted(ep.name for ep in entry_points(group="vllm.general_plugins")
                   if ep.value.split(".", 1)[0].split(":", 1)[0] == "mia")
    if not names:
        raise RuntimeError(
            "no vllm.general_plugins entry point belongs to the `mia` distribution; the "
            "capture-on arm would run with no plugin at all")
    return ",".join(names)


# ---------------------------------------------------------------------------
# One arm, in this process (script form)
# ---------------------------------------------------------------------------

def run_arm(arm: str, out_dir: Path) -> Path:
    """Run one T0 arm here and return its tree. ONE engine, so one process per call."""
    out_dir = Path(out_dir)
    if arm == "off":
        # Imported lazily: importing this module sets VLLM_PLUGINS="" process-wide, which
        # would disarm the capture arm if it were imported there.
        from tests.mia.parity.t1_reference import reference_generate
        return reference_generate(out_dir)

    from tests.mia.parity.capture_workload import run_workload
    return run_workload(ARMS[arm], out_dir, graph=False)


# ---------------------------------------------------------------------------
# Both arms, as child processes (test form)
# ---------------------------------------------------------------------------

def _spawn_arm(arm: str, out_dir: Path) -> Path:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "" if arm == "off" else mia_plugin_allowlist()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["MIA_PARITY_MAX_NUM_SEQS"] = "1"
    subprocess.run([sys.executable, str(Path(__file__).resolve()), arm, str(out_dir)],
                   cwd=str(REPO_ROOT), env=env, check=True)
    return Path(out_dir)


def _scratch(tmp_path: Path) -> Path:
    root = os.environ.get("MIA_PARITY_SCRATCH")
    return Path(root) / "t0" if root else tmp_path


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_one_control_arm_serves_both_capture_arms():
    """The capture-off arm is generated once and compared against both capture arms.

    That is only legitimate while the two capture workloads decode the SAME prompts for the
    SAME number of steps -- they differ in `extra_args` alone. If someone re-tunes one
    workload, this fails here rather than producing a mysterious T0 mismatch on a GPU node.
    """
    from tests.mia.parity.capture_workload import WORKLOADS

    hs, qk = WORKLOADS["hs_small"], WORKLOADS["qk_small"]
    assert hs.prompts == qk.prompts
    assert hs.max_tokens == qk.max_tokens


@pytest.mark.gpu
@pytest.mark.skipif(not _has_gpu(), reason="T0 boots real engines; needs a GPU")
def test_capture_does_not_change_generation(tmp_path):
    from tests.mia.parity.compare_artifacts import compare

    base = _scratch(tmp_path)
    off = _spawn_arm("off", base / "off")
    for arm in ARMS:
        on = _spawn_arm(arm, base / f"on_{arm}")
        problems = compare(off, on, atol=None, name=GENERATION)
        assert not problems, f"T0 {arm}: {problems}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run ONE T0 arm in this process.")
    ap.add_argument("arm", choices=("off", *ARMS))
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)
    run_arm(args.arm, Path(args.out_dir))
    return 0


# vLLM may spawn helper processes that re-import this module; without the guard a child
# would boot a second engine on an exclusive_process GPU.
if __name__ == "__main__":
    raise SystemExit(main())
