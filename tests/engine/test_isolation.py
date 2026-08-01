# tests/engine/test_isolation.py
"""Prefix-cache isolation: salted steered requests must not pollute the
shared-prefix KV, and unsalted spec-bearing requests are rejected.
"""
import pytest
import torch

from tests.engine.conftest import generate_one
from vllm_hook_plugins.core.canonical import request_salt
from vllm_hook_plugins.core.schema import SpecError

# Long enough to span multiple KV blocks so the prefix cache engages.
SHARED_PREFIX = (
    "In a quiet village nestled between two mountains, the baker rose before "
    "dawn each morning to knead the day's bread. The scent drifted through "
    "the narrow streets and woke the neighbors one by one, and"
)


def _spec(registry, hidden_size, layer, strength=16.0):
    vector = torch.randn(hidden_size, dtype=torch.float32) * 2
    artifact_id = registry.write({"vector": vector})
    return {"ops": [{
        "layers": [layer],
        "transform": {"kind": "additive", "strength": strength, "artifact": artifact_id},
        "scope": {"kind": "all"},
        "gate": None,
    }]}


def test_salt_required_without_cache_salt(unified_llm, registry, model_info):
    spec = _spec(registry, model_info["hidden_size"], model_info["num_layers"] // 2)
    with pytest.raises(SpecError) as excinfo:
        generate_one(unified_llm, SHARED_PREFIX, extra={"intervention_spec": spec})
    assert excinfo.value.code == "E_SALT_REQUIRED"
    # the engine keeps serving after the rejection
    ok = generate_one(unified_llm, "hello world", max_tokens=2)
    assert ok.outputs[0].text


def test_steered_after_baseline_shared_prefix(unified_llm, registry, model_info):
    spec = _spec(registry, model_info["hidden_size"], model_info["num_layers"] // 2)
    salt = request_salt(spec, [op["transform"]["artifact"] for op in spec["ops"]])

    baseline_first = generate_one(unified_llm, SHARED_PREFIX, max_tokens=16)
    steered = generate_one(
        unified_llm, SHARED_PREFIX, extra={"intervention_spec": spec}, salt=salt, max_tokens=16
    )
    baseline_again = generate_one(unified_llm, SHARED_PREFIX, max_tokens=16)

    # a large intervention must change greedy output...
    assert steered.outputs[0].text != baseline_first.outputs[0].text
    # ...without contaminating the unsalted shared prefix
    assert baseline_again.outputs[0].text == baseline_first.outputs[0].text


def test_distinct_specs_need_distinct_salts(unified_llm, registry, model_info):
    """Two different interventions over one prefix, each salted with the
    reference derivation, land on distinct KV and produce their own
    outputs.
    """
    layer = model_info["num_layers"] // 2
    spec_a = _spec(registry, model_info["hidden_size"], layer, strength=16.0)
    spec_b = _spec(registry, model_info["hidden_size"], layer, strength=-16.0)
    salt_a = request_salt(spec_a, [spec_a["ops"][0]["transform"]["artifact"]])
    salt_b = request_salt(spec_b, [spec_b["ops"][0]["transform"]["artifact"]])
    assert salt_a != salt_b

    out_a = generate_one(unified_llm, SHARED_PREFIX, extra={"intervention_spec": spec_a},
                         salt=salt_a, max_tokens=16)
    out_b = generate_one(unified_llm, SHARED_PREFIX, extra={"intervention_spec": spec_b},
                         salt=salt_b, max_tokens=16)
    assert out_a.outputs[0].text != out_b.outputs[0].text
