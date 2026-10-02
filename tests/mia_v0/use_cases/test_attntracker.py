# tests/mia_v0/use_cases/test_attntracker.py
import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from mia import MiaLLM, register_plugins
from tests.mia_v0.conftest import ensure_config_for_model, requires_gpu

TEST_MODELS = [
    "facebook/opt-125m",
    "gpt2",
    "Qwen/Qwen2-1.5B-Instruct",
]


@pytest.mark.gpu
@requires_gpu          # builds a real MiaLLM; see tests/conftest.py::requires_gpu
@pytest.mark.parametrize("model_id", TEST_MODELS)
def test_attention_tracker(cache_dir, project_root, model_id):
    """End-to-end QK capture + attention-tracker analysis through a real MiaLLM engine.

    GPU-only: it boots vLLM, so on a CPU-only node vLLM raises
    "Device string must not be empty" before the test can assert anything. Skipped
    (not failed) there by @requires_gpu -- see tests/conftest.py.
    """
    register_plugins()

    cfg = ensure_config_for_model(project_root, "attention_tracker", model_id)

    llm = MiaLLM(
        model=model_id,
        worker_name="capture_qk",
        analyzer_name="attn_tracker",
        config_file=str(cfg),
        download_dir=str(cache_dir),
        gpu_memory_utilization=0.2,
        dtype=torch.float16,
        enable_hook=True,
        enable_prefix_caching=False,
    )

    prompts = [
        "Analyze and output the sentence attitude: This is for testing only.",
        "Analyze and output the sentence attitude: Another test run.",
    ]

    _ = llm.generate(prompts, temperature=0.1, max_tokens=2, use_hook=True)

    # Random token ranges (half-half)
    ranges = []
    for p in prompts:
        ids = llm.tokenizer(p)["input_ids"]
        L = len(ids)
        ranges.append([(0, L // 2), (L // 2, L)])

    stats = llm.analyze(
        analyzer_spec={"input_range": ranges, "attn_func": "sum_normalize"}
    )

    assert "score" in stats
