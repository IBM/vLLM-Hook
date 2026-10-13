"""Optional pytest wrapper for the vLLM-backed CPU behavior fixtures."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu_behavior

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = [
    "run_hidden_states_cpu.py",
    "run_attention_tracker_cpu.py",
    "run_activation_steer_cpu.py",
    "run_core_reranker_cpu.py",
]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_vllm_cpu_fixture_script(fixture):
    if os.environ.get("MIA_RUN_VLLM_CPU_FIXTURES") != "1":
        pytest.skip("set MIA_RUN_VLLM_CPU_FIXTURES=1 to boot vLLM CPU fixtures")

    script = REPO_ROOT / "tests" / "cpu_behavioral" / fixture
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("VLLM_CPU_OMP_THREADS_BIND", "nobind")
    env.setdefault("TRITON_DEFAULT_BACKEND", "cpu")
    env.setdefault("TRITON_CACHE_DIR", "/private/tmp/triton-cache")
    env.setdefault("TRITON_DISABLE_OPENMP", "1")
    env.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")
    env.setdefault("MIA_WRITER_PROCESS", "0")
    env.pop("TRITON_LOCAL_LIBOMP_PATH", None)
    env.pop("DYLD_INSERT_LIBRARIES", None)

    print(f"\n[vllm-cpu-fixture] running {fixture}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, f"{fixture} failed with exit code {completed.returncode}"
