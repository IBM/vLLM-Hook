# tests/engine/conftest.py
import json
import os
import tempfile

import pytest

pytest.importorskip("vllm")
import torch

if not torch.cuda.is_available():
    collect_ignore_glob = ["test_*.py"]

# Both the tests (writing artifacts) and the worker (loading them) must
# resolve the same registry root, and the env vars must be set before the
# engine process is configured.
os.environ.setdefault(
    "VLLM_HOOK_REGISTRY_DIR", tempfile.mkdtemp(prefix="vllm_hook_registry_")
)
os.environ["VLLM_HOOK_WORKER"] = "unified"

# A fused-residual model (LLaMA-family): its decoder layers return
# (hidden_states, residual) pairs, so the tests exercise the stream
# materialization and per-row residual write-back — OPT/GPT-2 layers
# return a single tensor and would leave that path uncovered.
TEST_MODEL = os.environ.get("VLLM_HOOK_TEST_MODEL", "Qwen/Qwen2-0.5B-Instruct")


@pytest.fixture(scope="session")
def registry():
    from vllm_hook_plugins.core.artifacts import ArtifactRegistry

    return ArtifactRegistry(os.environ["VLLM_HOOK_REGISTRY_DIR"])


@pytest.fixture(scope="session")
def unified_llm():
    """One unified-worker engine for the whole session, prefix caching on
    (the salt rule and the isolation tests depend on it).
    """
    from vllm import LLM
    from vllm_hook_plugins import _hook_plugin

    _hook_plugin.register()
    llm = LLM(
        model=TEST_MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.3,
        dtype="float16",
    )
    yield llm


@pytest.fixture(scope="session")
def model_info(unified_llm):
    cfg = unified_llm.llm_engine.model_config.hf_config
    text_cfg = getattr(cfg, "text_config", cfg)
    return {
        "num_layers": int(text_cfg.num_hidden_layers),
        "hidden_size": int(text_cfg.hidden_size),
    }


def generate_one(llm, prompt, extra=None, salt=None, max_tokens=8):
    """Greedy single-prompt generate with optional extra_args/cache_salt."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, extra_args=extra)
    request = {"prompt": prompt}
    if salt is not None:
        request["cache_salt"] = salt
    return llm.generate([request], params, use_tqdm=False)[0]


def load_capture(output):
    """(manifest dict, {name: tensor}) from output.captures."""
    import safetensors.torch

    manifest_json, data = output.captures
    return json.loads(manifest_json), safetensors.torch.load(data)


def prompt_rows(manifest, tensors, layer, n_prompt):
    """Rows of a captured layer at prompt positions, with their positions."""
    positions = manifest["positions"][str(layer)]
    rows = tensors[f"layer_{layer}"]
    keep = [i for i, p in enumerate(positions) if p < n_prompt]
    return rows[keep], [positions[i] for i in keep]
