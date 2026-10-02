# tests/controls/test_actsteer.py
import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from mia import MiaLLM, register_plugins
from tests.conftest import ensure_config_for_model, requires_gpu

TEST_MODELS = [
    "facebook/opt-125m",
    "gpt2",
    "Qwen/Qwen2-1.5B-Instruct",
]


@pytest.mark.gpu
@requires_gpu          # builds a real MiaLLM; see tests/conftest.py::requires_gpu
@pytest.mark.parametrize("model_id", TEST_MODELS)
def test_activation_steer(cache_dir, project_root, model_id):
    """End-to-end activation steering through a real MiaLLM engine.

    GPU-only: it boots vLLM, so on a CPU-only node vLLM raises
    "Device string must not be empty" before the test can assert anything. Skipped
    (not failed) there by @requires_gpu -- see tests/conftest.py.
    """
    register_plugins()

    cfg = ensure_config_for_model(project_root, "activation_steer", model_id)

    llm = MiaLLM(
        model=model_id,
        worker_name="steer",
        analyzer_name=None,
        config_file=str(cfg),
        download_dir=str(cache_dir),
        gpu_memory_utilization=0.5,
        dtype=torch.float16,
        enable_hook=True,
    )

    prompt = "This is for testing only."

    _ = llm.generate(prompt, max_tokens=10, temperature=0.0, use_hook=True)
