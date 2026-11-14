"""Run the existing demo scenarios automatically on compatible backends."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import torch

from backend_support import backend_environment


pytestmark = pytest.mark.behavioral
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent


def run_child(command, env, log_path, timeout):
    # Stream the log while subprocess.run retains responsibility for the timeout.
    finished = threading.Event()

    def show_output():
        with log_path.open() as reader:
            while True:
                chunk = reader.read(4096)
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                elif finished.is_set():
                    break
                else:
                    finished.wait(0.05)

    with log_path.open("w") as log:
        display = threading.Thread(target=show_output)
        display.start()
        try:
            completed = subprocess.run(command, cwd=REPO_ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"Timed out after {timeout}s. Log: {log_path}", pytrace=False)
        finally:
            finished.set()
            display.join()
    if completed.returncode != 0:
        pytest.fail(
            f"Fixture/process failed ({completed.returncode}). Log: {log_path}", pytrace=False)


# A CPU-built vLLM works on any OS it supports; this is not a macOS check.
# Do not hide a broken installation: import errors should fail collection.
if importlib.util.find_spec("vllm") is None:
    current_platform = None
else:
    from vllm.platforms import current_platform

CPU_AVAILABLE = current_platform is not None and current_platform.is_cpu()
CUDA_AVAILABLE = (current_platform is not None and current_platform.is_cuda()
                  and torch.cuda.is_available())
METAL_AVAILABLE = False  # MIA's Metal scenario adapter is not implemented yet.

REQUIRED = set(filter(None, (name.strip() for name in
                           os.environ.get("MIA_REQUIRED_BACKENDS", "").split(","))))
if REQUIRED - {"cpu", "cuda", "metal"}:
    raise pytest.UsageError(f"Unknown required backends: {sorted(REQUIRED)}")

DEMO_SCENARIOS = {
    "hidden_states": "run_hidden_states_cpu.py",
    "attention_tracker": "run_attention_tracker_cpu.py",
    "activation_steer": "run_activation_steer_cpu.py",
    "core_reranker": "run_core_reranker_cpu.py",
}


@pytest.mark.parametrize("backend,available", [
    pytest.param("cpu", CPU_AVAILABLE, id="cpu", marks=pytest.mark.skipif(
        not CPU_AVAILABLE and "cpu" not in REQUIRED,
        reason="requires a CPU-capable vLLM installation (on any supported OS)")),
    pytest.param("cuda", CUDA_AVAILABLE, id="cuda", marks=pytest.mark.skipif(
        not CUDA_AVAILABLE and "cuda" not in REQUIRED,
        reason="requires CUDA vLLM and a visible CUDA device")),
    pytest.param("metal", METAL_AVAILABLE, id="metal", marks=pytest.mark.skipif(
        not METAL_AVAILABLE and "metal" not in REQUIRED,
        reason="MIA Metal scenario adapter is not implemented yet")),
])
@pytest.mark.parametrize("demo_scenario", DEMO_SCENARIOS)
def test_vllm_demo_scenario(demo_scenario, backend, available, tmp_path, capsys):
    if not available:
        pytest.fail(f"Required backend {backend} is unavailable for {demo_scenario}", pytrace=False)

    env = backend_environment(backend)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MIA_BEHAVIOR_HOOK_DIR"] = str(tmp_path / "hooks")
    if "MIA_BEHAVIOR_MODEL" in env:
        env["MIA_CPU_MODEL"] = env["MIA_BEHAVIOR_MODEL"]
    script = FIXTURE_DIR / DEMO_SCENARIOS[demo_scenario]
    with capsys.disabled():
        print(f"\n--- {demo_scenario} ({backend}) ---", flush=True)
        run_child([sys.executable, "-u", str(script)], env, tmp_path / "fixture.log",
                  int(os.environ.get("MIA_BEHAVIOR_TIMEOUT", "1800")))
