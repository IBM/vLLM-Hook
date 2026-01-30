# tests/controls/test_attntracker.py
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
def test_attention_tracker(cache_dir, project_root, model_id):
    register_plugins()

    cfg = ensure_config_for_model(project_root, "attention_tracker", model_id)

    llm = HookLLM(
        model=model_id,
        worker_name="probe_hook_qk",
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
