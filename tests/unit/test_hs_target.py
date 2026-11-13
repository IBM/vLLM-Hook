"""MIA_HS_TARGET: capture decoder-layer outputs (default) or MLP neurons (mlp.act_fn)."""
from types import SimpleNamespace

import pytest

from mia.core.runtime.hs_target import (
    HS_TARGET_ENV, hs_capture_width, hs_match_fn, hs_target, match_mlp_act)
from mia.errors import MiaConfigurationError

CFG = SimpleNamespace(hidden_size=1536, intermediate_size=8960, num_hidden_layers=28)


def _layer_matcher(name):
    return 7 if name == "model.layers.7" else None


def test_default_is_layer(monkeypatch):
    monkeypatch.delenv(HS_TARGET_ENV, raising=False)
    assert hs_target() == "layer"
    assert hs_match_fn(_layer_matcher) is _layer_matcher
    assert hs_capture_width(CFG) == 1536


def test_mlp_target_matches_act_fn_and_uses_intermediate_size(monkeypatch):
    monkeypatch.setenv(HS_TARGET_ENV, "mlp")
    match = hs_match_fn(_layer_matcher)
    assert match is match_mlp_act
    assert match("model.layers.14.mlp.act_fn") == 14
    assert match("language_model.model.layers.3.mlp.act_fn") == 3
    for other in ("model.layers.14", "model.layers.14.mlp", "model.layers.14.mlp.down_proj",
                  "model.layers.14.self_attn.o_proj", "model.layers.14.mlp.act_fn.extra"):
        assert match(other) is None
    assert hs_capture_width(CFG) == 8960


def test_mlp_target_refuses_tensor_parallel(monkeypatch):
    monkeypatch.setenv(HS_TARGET_ENV, "mlp")
    with pytest.raises(MiaConfigurationError, match="tensor_parallel_size=1"):
        hs_capture_width(CFG, tp_size=2)


@pytest.mark.parametrize("inter", [None, [8960, 8960], 0, True])
def test_mlp_target_refuses_non_dense_intermediate_size(monkeypatch, inter):
    monkeypatch.setenv(HS_TARGET_ENV, "mlp")
    cfg = SimpleNamespace(hidden_size=1536, intermediate_size=inter)
    with pytest.raises(MiaConfigurationError, match="intermediate_size"):
        hs_capture_width(cfg)


def test_unknown_target_is_refused(monkeypatch):
    monkeypatch.setenv(HS_TARGET_ENV, "attention")
    with pytest.raises(MiaConfigurationError, match="not a capture target"):
        hs_target()


def _vllm_config(tp=1):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=CFG, dtype="float16"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, tensor_parallel_size=tp),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.8))


@pytest.mark.parametrize("target,width", [("layer", 1536), ("mlp", 8960)])
def test_memory_plan_sizes_hs_rows_to_the_target_width(monkeypatch, target, width):
    from mia.core.runtime.memory_plan import build_memory_plan
    monkeypatch.setenv(HS_TARGET_ENV, target)
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(64 << 20))
    plan = build_memory_plan(_vllm_config(), "hidden_states", total_bytes=24 << 30, tp_rank=0)
    assert plan.hidden == width
    hs = [b for b in plan.buffers if b.name.startswith("hs.") and b.name[3:].isdigit()]
    assert len(hs) == 28 and all(b.shape == (plan.rows + 1, width) for b in hs)
    assert plan.rows == (64 << 20) // (28 * width * 2) - 1
