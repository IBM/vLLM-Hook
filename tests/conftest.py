"""Shared test setup: the spawn start method, the `gpu` marker, engine teardown and test configs."""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch

# Engines start worker processes: pin spawn before any test module imports vLLM.
mp.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

for p in (PROJECT_ROOT, EXAMPLES_DIR):
    sys.path.insert(0, str(p))


def pytest_configure(config):
    # Select engine tests with -m (`-m gpu` / `-m "not gpu"`), never -k: -k matches names only.
    config.addinivalue_line(
        "markers", "gpu: boots a real vLLM engine; needs a GPU (skipped otherwise)")


def has_gpu() -> bool:
    """True when a CUDA device is visible to this process."""
    try:
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 -- a broken CUDA setup counts as no GPU
        return False


#: Pair with ``@pytest.mark.gpu`` on a test that builds a real engine; skips it without a GPU.
requires_gpu = pytest.mark.skipif(
    not has_gpu(),
    reason="builds a real vLLM engine; no CUDA device visible. Run it on a GPU node.")


@pytest.fixture
def engines():
    """A list a test appends its MiaLLMs to; their engines are shut down after the test."""
    built = []
    yield built
    for llm in built:
        llm.llm_engine.engine_core.shutdown()


ConfigKind = Literal[
    "attention_tracker",
    "activation_steer",
    "core_reranker",
    "hidden_states",
]


def ensure_config_for_model(kind: ConfigKind, model_id: str, tmp_dir: Path) -> Path:
    """The shipped config for (kind, model_id), else a random one written under ``tmp_dir``."""
    short = model_id.split("/")[-1]
    shipped = PROJECT_ROOT / "model_configs" / kind / f"{short}.json"
    if shipped.exists():
        return shipped
    # Low layer/head indices stay in range for every test model.
    if kind == "hidden_states":
        data = {"hidden_states": {"layers": [1, 2], "mode": "last_token"}}
    else:
        data = {"params": {"important_heads": [[1, 2], [3, 4]]}}
    if kind == "attention_tracker":
        data["hookq"] = {"hookq_mode": "last_token"}  # the tracker scores the last token's Q
    target = Path(tmp_dir) / f"{kind}-{short}.RANDOM_TEST.json"
    target.write_text(json.dumps(data, indent=2))
    return target
