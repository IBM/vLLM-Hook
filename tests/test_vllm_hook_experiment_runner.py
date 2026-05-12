import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT_SMOKE = os.environ.get("RUN_VLLM_HOOK_EXPERIMENT_SMOKE") == "1"
RUN_METAL_EXPERIMENT_SMOKE = os.environ.get("RUN_VLLM_HOOK_METAL_EXPERIMENT_SMOKE") == "1"


@pytest.mark.skipif(
    not RUN_EXPERIMENT_SMOKE,
    reason="Set RUN_VLLM_HOOK_EXPERIMENT_SMOKE=1 to run model-backed experiments.",
)
@pytest.mark.parametrize(
    ("experiment", "extra_args"),
    [
        ("attn-tracker", ["--max-tokens", "1"]),
        ("core-reranker", []),
        ("steer-activation", ["--max-tokens", "8"]),
    ],
)
def test_non_metal_experiment_runner_smoke(tmp_path, experiment, extra_args):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "vllm_hook_experiments.py"),
        experiment,
        "--backend",
        "non-metal",
        "--benchmark-id",
        f"pytest-{experiment}",
        "--run-id",
        f"pytest-{experiment}-non-metal",
        "--output-dir",
        str(tmp_path),
        "--wandb-mode",
        "disabled",
        *extra_args,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


@pytest.mark.skipif(
    not RUN_METAL_EXPERIMENT_SMOKE,
    reason="Set RUN_VLLM_HOOK_METAL_EXPERIMENT_SMOKE=1 to run Metal model-backed experiments.",
)
@pytest.mark.parametrize(
    ("experiment", "extra_args"),
    [
        ("attn-tracker", ["--max-tokens", "1"]),
        ("core-reranker", []),
        ("steer-activation", ["--max-tokens", "8"]),
    ],
)
def test_metal_experiment_runner_smoke(tmp_path, experiment, extra_args):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "vllm_hook_experiments.py"),
        experiment,
        "--backend",
        "metal",
        "--benchmark-id",
        f"pytest-{experiment}",
        "--run-id",
        f"pytest-{experiment}-metal",
        "--output-dir",
        str(tmp_path),
        "--wandb-mode",
        "disabled",
        *extra_args,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
