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


def _mean_tokens_run(tmp_path, model_id, enforce_eager, chunk):
    """One all_tokens capture of a long prompt; returns (mean_tokens, mean, none) for one layer."""
    import json
    cfg = tmp_path / f"hs_all_{enforce_eager}_{chunk}.json"
    cfg.write_text(json.dumps({"model_info": {"name": model_id},
                               "hidden_states": {"layers": [2], "mode": "all_tokens"}}))
    llm = MiaLLM(model=model_id, worker_name="capture_hs", analyzer_name="hidden_states",
                 config_file=str(cfg), hook_dir=str(tmp_path / "hooks"), enforce_eager=enforce_eager,
                 gpu_memory_utilization=0.2, dtype=torch.float16, enable_prefix_caching=False,
                 max_model_len=1024, max_num_batched_tokens=chunk, max_num_seqs=1)
    try:
        prompt = "The quick brown fox jumps over the lazy dog. " * 50      # ~500 tokens
        out = llm.generate([prompt], temperature=0.0, max_tokens=1, use_hook=True)
        n = len(out[0].prompt_token_ids)
        res = {r: next(iter(llm.analyze(analyzer_spec={"reduce": r},
                                        probes=out[0].probes)["hidden_states"].values()))
               for r in ("mean_tokens", "mean", "none")}
        return n, res
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_mean_tokens_ignores_chunked_prefill_padding(tmp_path, enforce_eager):
    """reduce='mean_tokens' is the mean over the prompt's real rows, chunked prefill or not."""
    register_plugins()
    model_id = "facebook/opt-125m"
    n, chunked = _mean_tokens_run(tmp_path, model_id, enforce_eager, chunk=128)
    n1, whole = _mean_tokens_run(tmp_path, model_id, enforce_eager, chunk=1024)
    assert n == n1 and n > 128 and n % 128, "the prompt must span several chunks, the last short"

    a, b = chunked["mean_tokens"], whole["mean_tokens"]
    assert len(a) == 1 and a[0].dim() == 1, f"one vector per request, got {[t.shape for t in a]}"
    cos = torch.nn.functional.cosine_similarity(a[0].float(), b[0].float(), dim=0)
    assert cos > 0.999, f"mean_tokens differs between chunked and whole prefill (cos={cos:.5f})"

    # the same mean from reduce='none': drop zero padding, average the real rows
    raw = chunked["none"]
    rows = raw.reshape(-1, raw.shape[-1]) if torch.is_tensor(raw) else torch.cat(list(raw))
    rows = rows[rows.abs().sum(-1) != 0][:n]
    torch.testing.assert_close(a[0].float(), rows.float().mean(0), rtol=1e-2, atol=1e-2)

    # reduce='mean' is per pass, but no pass may be diluted by padding
    for t in chunked["mean"]:
        assert t is not None and torch.isfinite(t).all()
