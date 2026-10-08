"""Q/K capture and the attention-tracker analyzer through a real MiaLLM engine (GPU)."""
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
def test_attention_tracker(tmp_path, engines, model_id):
    """In-memory capture of two prompts, read with `analyze(probes=...)`, scores each prompt."""
    register_plugins()

    cfg = ensure_config_for_model("attention_tracker", model_id, tmp_path)

    llm = MiaLLM(
        model=model_id,
        worker_name="capture_qk",
        analyzer_name="attn_tracker",
        config_file=str(cfg),
        hook_dir=str(tmp_path / "hooks"),
        gpu_memory_utilization=0.2,
        dtype=torch.float16,
        enable_hook=True,
        enable_prefix_caching=False,
    )
    engines.append(llm)

    prompts = [
        "Analyze and output the sentence attitude: This is for testing only.",
        "Analyze and output the sentence attitude: Another test run.",
    ]

    out = llm.generate(prompts, temperature=0.1, max_tokens=2, use_hook=True)

    # Each prompt's tokens split into two halves.
    ranges = []
    for p in prompts:
        ids = llm.tokenizer(p)["input_ids"]
        L = len(ids)
        ranges.append([(0, L // 2), (L // 2, L)])

    stats = llm.analyze(
        analyzer_spec={"input_range": ranges, "attn_func": "sum_normalize"},
        probes=out[0].probes,
    )

    assert "score" in stats
    assert len(stats["score"]) == len(prompts)
