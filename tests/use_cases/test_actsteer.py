"""Activation steering end to end through a real MiaLLM engine (GPU)."""
import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM; skip, never error the whole collection

from mia import MiaLLM, register_plugins
from tests.conftest import ensure_config_for_model, requires_gpu

TEST_MODELS = [
    "facebook/opt-125m",
    "gpt2",
    "Qwen/Qwen2-1.5B-Instruct",
]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("model_id", TEST_MODELS)
def test_activation_steer(tmp_path, engines, model_id):
    """A steered generate runs on each model."""
    register_plugins()

    cfg = ensure_config_for_model("activation_steer", model_id, tmp_path)

    llm = MiaLLM(
        model=model_id,
        worker_name="steer",
        analyzer_name=None,
        config_file=str(cfg),
        hook_dir=str(tmp_path / "hooks"),
        gpu_memory_utilization=0.5,
        dtype=torch.float16,
        enable_hook=True,
    )
    engines.append(llm)

    prompt = "This is for testing only."

    _ = llm.generate(prompt, max_tokens=10, temperature=0.0, use_hook=True)
