"""Hidden-state capture and the hidden_states analyzer through a real MiaLLM engine (GPU)."""
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
def test_hidden_states_extraction(tmp_path, engines, model_id):
    """In-memory last-token capture of two prompts: a (hidden_size,) tensor per prompt and layer."""
    register_plugins()

    cfg = ensure_config_for_model("hidden_states", model_id, tmp_path)

    llm = MiaLLM(
        model=model_id,
        worker_name="capture_hs",
        analyzer_name="hidden_states",
        config_file=str(cfg),
        hook_dir=str(tmp_path / "hooks"),
        gpu_memory_utilization=0.2,
        dtype=torch.float16,
        enable_hook=True,
        enable_prefix_caching=False,
    )
    engines.append(llm)

    prompts = [
        "Hidden states test prompt one.",
        "Hidden states test prompt two.",
    ]

    out = llm.generate(prompts, temperature=0.0, max_tokens=1, use_hook=True)

    stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out[0].probes)

    assert "hidden_states" in stats
    hs = stats["hidden_states"]
    assert len(hs) > 0, "Expected at least one layer in hidden_states output"

    hidden_size = llm.llm.llm_engine.model_config.hf_config.hidden_size

    for layer_name, tensors in hs.items():
        assert len(tensors) == len(prompts), (
            f"Expected {len(prompts)} tensors for layer {layer_name}, got {len(tensors)}"
        )
        for t in tensors:
            assert isinstance(t, torch.Tensor)
            assert t.shape == torch.Size([hidden_size]), (
                f"Expected shape ({hidden_size},) for last_token mode, got {t.shape}"
            )
