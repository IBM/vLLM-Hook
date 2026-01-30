# tests/controls/test_actsteer.py
import pytest
import torch

from vllm_hook_plugins import HookLLM, register_plugins
from tests.conftest import ensure_config_for_model

TEST_MODELS = [
    "facebook/opt-125m",
    "gpt2",
    "Qwen/Qwen2-1.5B-Instruct",
]


@pytest.mark.parametrize("model_id", TEST_MODELS)
def test_activation_steer(cache_dir, project_root, model_id):
    register_plugins()

    cfg = ensure_config_for_model(project_root, "activation_steer", model_id)

    llm = HookLLM(
        model=model_id,
        worker_name="steer_hook_act",
        analyzer_name=None,
        config_file=str(cfg),
        download_dir=str(cache_dir),
        gpu_memory_utilization=0.5,
        dtype=torch.float16,
        enable_hook=True,
    )

    prompt = "This is for testing only."

    _ = llm.generate(prompt, max_tokens=10, temperature=0.0, use_hook=True)
