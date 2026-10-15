"""Run the existing demo scenarios automatically on compatible backends."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend_support import BACKENDS, DEMO_SCENARIOS, backend_environment


pytestmark = pytest.mark.behavioral
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent


def run_child(command, env, log_path, timeout):
    with log_path.open("w") as log:
        try:
            completed = subprocess.run(command, cwd=REPO_ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"Timed out after {timeout}s. Log: {log_path}\n{log_path.read_text()[-8000:]}", pytrace=False)
    if completed.returncode != 0:
        pytest.fail(
            f"Fixture/process failed ({completed.returncode}). Log: {log_path}\n"
            f"{log_path.read_text()[-8000:]}", pytrace=False)


@pytest.fixture(scope="session")
def backend_matrix(tmp_path_factory):
    required = set(filter(None, (name.strip() for name in os.environ.get("MIA_REQUIRED_BACKENDS", "").split(","))))
    unknown = required - set(BACKENDS)
    if unknown:
        pytest.fail(f"Unknown required backend(s): {sorted(unknown)}", pytrace=False)
    work = tmp_path_factory.mktemp("backend-discovery")
    report = work / "backends.json"
    env = backend_environment("cpu") if sys.platform == "darwin" else dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    run_child([sys.executable, str(FIXTURE_DIR / "backend_support.py"), str(report)],
              env, work / "discovery.log", 120)
    return json.loads(report.read_text()), required


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("demo_scenario", DEMO_SCENARIOS)
def test_vllm_demo_scenario(demo_scenario, backend, backend_matrix, tmp_path, record_property):
    matrix, required = backend_matrix
    detected = matrix[demo_scenario][backend]
    record_property("demo_scenario", demo_scenario)
    record_property("backend", backend)
    if detected["status"] == "error":
        pytest.fail(detected["reason"], pytrace=False)
    if detected["status"] != "ready":
        if backend in required:
            pytest.fail(f"Required {demo_scenario}/{backend}: {detected['reason']}", pytrace=False)
        pytest.skip(detected["reason"])

    env = backend_environment(backend)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MIA_BEHAVIOR_HOOK_DIR"] = str(tmp_path / "hooks")
    if "MIA_BEHAVIOR_MODEL" in env:
        env["MIA_CPU_MODEL"] = env["MIA_BEHAVIOR_MODEL"]
    script = FIXTURE_DIR / DEMO_SCENARIOS[demo_scenario][0]
    run_child([sys.executable, str(script)], env, tmp_path / "fixture.log",
              int(os.environ.get("MIA_BEHAVIOR_TIMEOUT", "1800")))
