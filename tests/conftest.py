# tests/conftest.py
import os
import sys
import json
import multiprocessing as mp
from pathlib import Path
from typing import Literal

import pytest

pytest.importorskip("vllm")

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

for p in (PROJECT_ROOT, EXAMPLES_DIR):
    sys.path.insert(0, str(p))


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
    data = {
        "params": {
            "important_heads": [[1, 2],[3, 4]],
            "random_generated": True,
        },
    }
    with open(target2, "w") as f:
        json.dump(data, f, indent=2)

    return target2
