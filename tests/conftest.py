# tests/conftest.py
import os
import sys
import json
import multiprocessing as mp
from pathlib import Path
from typing import Literal

import pytest

# NO `pytest.importorskip("vllm")` HERE, AND NEVER AGAIN. It used to be the first
# statement in this file, which made the ENTIRE suite hostage to an optional import:
# a Skipped raised while this conftest is being imported aborts collection of all of
# tests/, so `pytest tests/ -q -m "not gpu"` -- this plan's exit criterion -- ran ZERO
# tests. Depending on pytest version and invocation shape that surfaces either as a raw
# traceback or as a directory-level skip with a green exit 0; neither is a gate. The
# overwhelming majority of these tests are pure Python and never touch vLLM, so the
# dependency belongs at the tests that actually need it: modules importing `vllm.*` or
# `mia` call `pytest.importorskip("vllm")` themselves, at their own top. The root
# conftest.py additionally refuses to exit 0 having collected nothing, from a file that
# still loads when this one does not.

# ORDER MATTERS BELOW. The spawn pin must precede any CUDA probe (has_gpu() imports
# torch further down), and both env vars must be set before any test module imports
# vLLM -- conftest import always precedes test-module import, so this is the right
# place for them even though the vLLM guard is not.
mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

for p in (PROJECT_ROOT, EXAMPLES_DIR):
    sys.path.insert(0, str(p))


def pytest_configure(config):
    # Registered so `-m "not gpu"` works and the mark stops warning. The GPU tests skip
    # themselves when no device is present; the mark is for selecting them deliberately.
    config.addinivalue_line(
        "markers", "gpu: boots a real vLLM engine; needs a GPU (skipped otherwise)")


# ---------------------------------------------------------------------------------------
# THE HERMETIC GATE IS  `pytest tests/ -q -m "not gpu"`  -- WITH -m, NEVER -k.
# ---------------------------------------------------------------------------------------
# `-k` is a SUBSTRING filter over test ids. It has no idea what a GPU test is, and it got
# the answer wrong in BOTH directions:
#
#   * It INCLUDED the real engine tests in tests/use_cases/, whose names say nothing about
#     GPUs -- so they ran on a login node and the gate reported "11 failed" as routine.
#   * It EXCLUDED six pure-CPU tests whose names merely CONTAIN "gpu":
#     tests/test_parity_band_discrimination.py::test_gpu_routing_band_* (4) and
#     tests/test_parity_band_index.py's [MIA_T2_GPU_ROUTING_BAND] parametrization (2) --
#     the latter meaning the guard that enforces band documentation was silently skipping
#     one of the fourteen bands it exists to police. Measured on a login node:
#     -k "not gpu" -> 471 passed / 18 deselected;  -m "not gpu" -> 477 passed / 12.
#
# Now that the `gpu` marker is actually applied, marker selection is what "not gpu" was
# always supposed to mean: `-m` deselects exactly the 12 marked tests and nothing else.
# Do not reintroduce `-k` for this purpose.


def has_gpu() -> bool:
    """True when a CUDA device is actually visible to this process."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 -- torch absent/broken is "no GPU", not a test error
        return False


#: Apply to any test that constructs a real ``LLM``/``MiaLLM``, TOGETHER with
#: ``@pytest.mark.gpu``. The marker is what ``-m "not gpu"`` selects on; this skipif is what
#: makes the test explain itself when someone runs it directly on a CPU-only node instead of
#: dying inside vLLM with ``RuntimeError: Device string must not be empty``.
requires_gpu = pytest.mark.skipif(
    not has_gpu(),
    reason="constructs a real vLLM engine; no CUDA device visible (vLLM raises "
           "'Device string must not be empty'). Run on a GPU node to exercise it.")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def cache_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("vllm_cache")
    return root


@pytest.fixture
def cache_dir(cache_root: Path, request) -> Path:
    sub = cache_root / request.node.name
    sub.mkdir(parents=True, exist_ok=True)
    return sub


ConfigKind = Literal[
    "attention_tracker",
    "activation_steer",
    "core_reranker",
    "hidden_states",
]


def ensure_config_for_model(project_root: Path, kind: ConfigKind, model_id: str) -> Path:
    """Ensure a config JSON exists for (kind, model_id). Create a random one if missing."""
    config_dir = project_root / "model_configs" / kind
    config_dir.mkdir(parents=True, exist_ok=True)

    short = model_id.split("/")[-1]
    target1 = config_dir / f"{short}.json"
    target2 = config_dir / f"{short}.RANDOM_TEST.json"

    if target1.exists():
        return target1
    if target2.exists():
        return target2

    # Pick a template config if exists
    templates = sorted(config_dir.glob("*.json"))
    if templates:
        template = templates[0]
        with open(template, "r") as f:
            data = json.load(f)

        data["random_generated"] = True

    # make important_heads the first few layers to avoid out of range lists
    if kind == "hidden_states":
        data = {
            "hidden_states": {
                "layers": [1, 2],
                "mode": "last_token",
                "random_generated": True,
            },
        }
    else:
        data = {
            "params": {
                "important_heads": [[1, 2],[3, 4]],
                "random_generated": True,
            },
        }
    with open(target2, "w") as f:
        json.dump(data, f, indent=2)

    return target2
