# tests/serve/test_discovery.py
"""Discovery payload shape, auth inheritance, fingerprint agreement with a
client-side recomputation, and the xargs round trip over HTTP.
"""
import json
import uuid

import torch

from tests.serve.conftest import API_KEY, TEST_MODEL, http_get, http_post
from vllm_hook_plugins.core.artifacts import ArtifactRegistry
from vllm_hook_plugins.core.canonical import request_salt
from vllm_hook_plugins.core.fingerprints import config_fingerprint


def test_capabilities_requires_auth(server):
    status, _ = http_get(f"{server}/v1/hook/capabilities")
    assert status == 401


def test_capabilities_payload_shape(server):
    status, payload = http_get(f"{server}/v1/hook/capabilities", api_key=API_KEY)
    assert status == 200

    assert payload["active_worker"] == "unified"
    assert payload["plugin_version"]
    assert payload["vllm_version"]

    kinds = payload["intervention_kinds"]
    assert set(kinds["transforms"]) == {"additive", "directional_ablation", "rotation", "head_additive"}
    assert set(kinds["modifiers"]) == {"norm_preserving", "alignment_adaptive"}
    assert set(kinds["scopes"]) == {"all", "after_prompt", "last_k", "from_position"}
    assert set(kinds["gates"]) == {"null", "cache_once", "probe_sum", "multi_key_threshold"}
    assert kinds["constraints"] == {"head_additive": "tensor_parallel_size==1"}

    assert payload["processor_kinds"] == {"processors": []}
    assert payload["capture_kinds"]["kinds"] == ["residual"]
    assert set(payload["capture_kinds"]["locations"]) == {"layer_output", "layer_input"}
    assert set(payload["capture_kinds"]["modes"]) == {"all_tokens", "last_token"}
    assert set(payload["artifact_transports"]) == {"shared_fs", "http"}
    assert payload["artifact_registry_root"]

    engine = payload["engine"]
    assert engine["enforce_eager"] is True
    assert engine["speculative_decoding"] is False
    assert engine["tensor_parallel_size"] == 1

    model = payload["model"]
    assert model["id"] == TEST_MODEL
    for key in ("config_fingerprint", "tokenizer_fingerprint", "chat_template_fingerprint"):
        assert model[key].startswith("sha256:")
    assert model["num_layers"] > 0
    assert model["hidden_size"] > 0


def test_config_fingerprint_agrees_with_client_recomputation(server):
    from transformers import AutoConfig

    _, payload = http_get(f"{server}/v1/hook/capabilities", api_key=API_KEY)
    client_side = config_fingerprint(AutoConfig.from_pretrained(TEST_MODEL).to_dict())
    assert payload["model"]["config_fingerprint"] == client_side


def test_xargs_round_trip(server, registry_dir):
    _, payload = http_get(f"{server}/v1/hook/capabilities", api_key=API_KEY)
    hidden_size = payload["model"]["hidden_size"]
    layer = payload["model"]["num_layers"] // 2

    registry = ArtifactRegistry(registry_dir)
    artifact_id = registry.write({"vector": torch.randn(hidden_size)})
    spec = {"ops": [{
        "layers": [layer],
        "transform": {"kind": "additive", "strength": 4.0, "artifact": artifact_id},
        "scope": {"kind": "all"},
        "gate": None,
    }]}

    status, body = http_post(
        f"{server}/v1/completions",
        {
            "model": TEST_MODEL,
            "prompt": "The weather today is",
            "max_tokens": 8,
            "temperature": 0.0,
            "cache_salt": request_salt(spec, [artifact_id]),
            # vllm_xargs is scalar-typed: nested specs go over as JSON strings
            "vllm_xargs": {"intervention_spec": json.dumps(spec)},
        },
        api_key=API_KEY,
    )
    assert status == 200, body
    assert body["choices"][0]["text"]


def test_bad_spec_rejected_with_code(server):
    status, body = http_post(
        f"{server}/v1/completions",
        {
            "model": TEST_MODEL,
            "prompt": "The weather today is",
            "max_tokens": 4,
            "cache_salt": str(uuid.uuid4()),
            "vllm_xargs": {"intervention_spec": json.dumps({"ops": [{
                "layers": [0],
                "transform": {"kind": "multiply", "artifact": "sha256:" + "ab" * 32},
                "scope": {"kind": "all"},
            }]})},
        },
        api_key=API_KEY,
    )
    assert status != 200
    assert "E_UNKNOWN_KIND" in str(body)
