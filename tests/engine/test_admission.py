# tests/engine/test_admission.py
"""Admission rejections carry their E_* codes and never wedge the engine:
after every rejection a plain request still completes.
"""
import os
import uuid

import pytest
import torch

from tests.engine.conftest import TEST_MODEL, generate_one
from vllm_hook_plugins.core.schema import SpecError

PROMPT = "A short prompt for admission tests that"


def _assert_engine_alive(unified_llm):
    output = generate_one(unified_llm, "still alive?", max_tokens=2)
    assert output.finished


def test_legacy_key_rejected_on_unified_worker(unified_llm):
    with pytest.raises(SpecError) as excinfo:
        generate_one(unified_llm, PROMPT, extra={"output_hidden_states": True})
    assert excinfo.value.code == "E_LEGACY_KEY"
    assert excinfo.value.path == "output_hidden_states"
    _assert_engine_alive(unified_llm)


def test_unknown_kind_rejected(unified_llm):
    with pytest.raises(SpecError) as excinfo:
        generate_one(
            unified_llm, PROMPT,
            extra={"intervention_spec": {"ops": [{
                "layers": [0],
                "transform": {"kind": "multiply", "artifact": "sha256:" + "ab" * 32},
                "scope": {"kind": "all"},
            }]}},
            salt=str(uuid.uuid4()),
        )
    assert excinfo.value.code == "E_UNKNOWN_KIND"
    assert excinfo.value.path == "ops[0].transform.kind"
    _assert_engine_alive(unified_llm)


def test_artifact_missing_rejected(unified_llm):
    with pytest.raises(SpecError) as excinfo:
        generate_one(
            unified_llm, PROMPT,
            extra={"intervention_spec": {"ops": [{
                "layers": [0],
                "transform": {"kind": "additive", "strength": 1.0,
                              "artifact": "sha256:" + "00" * 32},
                "scope": {"kind": "all"},
            }]}},
            salt=str(uuid.uuid4()),
        )
    assert excinfo.value.code == "E_ARTIFACT_MISSING"
    _assert_engine_alive(unified_llm)


def test_artifact_shape_mismatch_rejected(unified_llm, registry, model_info):
    """A wrong-shaped tensor must reject at staging, where it costs one
    request — inside a hook it would abort the whole forward pass.
    """
    bad_id = registry.write({"vector": torch.randn(model_info["hidden_size"] + 1)})
    with pytest.raises(SpecError) as excinfo:
        generate_one(
            unified_llm, PROMPT,
            extra={"intervention_spec": {"ops": [{
                "layers": [0],
                "transform": {"kind": "additive", "strength": 1.0, "artifact": bad_id},
                "scope": {"kind": "all"},
            }]}},
            salt=str(uuid.uuid4()),
        )
    assert excinfo.value.code == "E_BAD_PARAM"
    _assert_engine_alive(unified_llm)


def test_artifact_hash_tamper_rejected(unified_llm, registry, model_info):
    artifact_id = registry.write({"vector": torch.randn(model_info["hidden_size"])})
    file_path = registry.path_for(artifact_id)
    with open(file_path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        f.write(b"\x00")
    with pytest.raises(SpecError) as excinfo:
        generate_one(
            unified_llm, PROMPT,
            extra={"intervention_spec": {"ops": [{
                "layers": [0],
                "transform": {"kind": "additive", "strength": 1.0, "artifact": artifact_id},
                "scope": {"kind": "all"},
            }]}},
            salt=str(uuid.uuid4()),
        )
    assert excinfo.value.code == "E_ARTIFACT_HASH"
    _assert_engine_alive(unified_llm)


def test_batch_survives_one_bad_request(unified_llm, registry, model_info):
    """The engine (and its prepared state) is unaffected by a rejected
    request: a good spec-bearing request right after a rejection works.
    """
    vector_id = registry.write({"vector": torch.randn(model_info["hidden_size"])})
    with pytest.raises(SpecError):
        generate_one(unified_llm, PROMPT,
                     extra={"intervention_spec": {"ops": [{"layers": [9999],
                                                           "transform": {"kind": "additive",
                                                                         "strength": 1.0,
                                                                         "artifact": vector_id},
                                                           "scope": {"kind": "all"}}]}},
                     salt=str(uuid.uuid4()))
    good = generate_one(
        unified_llm, PROMPT,
        extra={"intervention_spec": {"ops": [{
            "layers": [model_info["num_layers"] // 2],
            "transform": {"kind": "additive", "strength": 2.0, "artifact": vector_id},
            "scope": {"kind": "all"},
        }]}},
        salt=str(uuid.uuid4()),
    )
    assert good.finished


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="head_additive TP constraint needs 2 GPUs")
def test_head_additive_constraint_under_tp(registry, model_info):
    """E_CONSTRAINT: head_additive requires tensor_parallel_size==1."""
    from vllm import LLM
    from vllm_hook_plugins import _hook_plugin

    _hook_plugin.register()
    llm = LLM(model=TEST_MODEL, enforce_eager=True, tensor_parallel_size=2,
              gpu_memory_utilization=0.3, dtype="float16")
    vector_id = registry.write({"vector": torch.randn(64)})
    with pytest.raises(SpecError) as excinfo:
        generate_one(
            llm, PROMPT,
            extra={"intervention_spec": {"ops": [{
                "layers": [0],
                "transform": {"kind": "head_additive", "strength": 1.0, "artifact": vector_id},
                "scope": {"kind": "all"},
            }]}},
            salt=str(uuid.uuid4()),
        )
    assert excinfo.value.code == "E_CONSTRAINT"
