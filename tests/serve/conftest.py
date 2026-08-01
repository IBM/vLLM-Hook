# tests/serve/conftest.py
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("vllm")
import torch

if not torch.cuda.is_available():
    collect_ignore_glob = ["test_*.py"]

TEST_MODEL = os.environ.get("VLLM_HOOK_TEST_MODEL", "facebook/opt-125m")
API_KEY = "hook-test-key"


def _entry_points_installed() -> bool:
    """The serve subprocess only loads the plugin through the installed
    entry points; skip when the package is not pip-installed.
    """
    from importlib.metadata import entry_points

    return any(
        ep.value.startswith("vllm_hook_plugins")
        for ep in entry_points(group="vllm.general_plugins")
    )


@pytest.fixture(scope="session")
def registry_dir():
    return os.environ.setdefault(
        "VLLM_HOOK_REGISTRY_DIR", tempfile.mkdtemp(prefix="vllm_hook_registry_")
    )


@pytest.fixture(scope="session")
def server(registry_dir):
    """One `vllm serve` subprocess with the unified worker and an API key."""
    if not _entry_points_installed():
        pytest.skip("vllm-hook-plugins entry points not installed; pip install -e vllm_hook_plugins")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = dict(os.environ)
    env["VLLM_HOOK_WORKER"] = "unified"
    env["VLLM_HOOK_REGISTRY_DIR"] = registry_dir
    process = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", TEST_MODEL, "--enforce-eager", "--dtype", "float16",
         "--gpu-memory-utilization", "0.3", "--port", str(port),
         "--api-key", API_KEY],
        env=env,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 600
    while time.time() < deadline:
        if process.poll() is not None:
            pytest.fail(f"vllm serve exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    else:
        process.terminate()
        pytest.fail("vllm serve did not become healthy in time")

    yield base_url
    process.terminate()
    process.wait(timeout=60)


def http_get(url, api_key=None):
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def http_post(url, body, api_key=None):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
