# tests/engine/test_interventions.py
"""Steered greedy decoding against the interpreter-applied reference, and
the (hidden, residual) re-split against the layer-boundary value.
"""
import uuid

import torch

from tests.engine.conftest import generate_one, load_capture, prompt_rows
from vllm_hook_plugins.core.interpreter.transforms import additive, directional_ablation

PROMPT = "The quick brown fox jumps over the lazy dog and then"


def _capture(layers, location="layer_output"):
    return {"layers": layers, "mode": "all_tokens", "location": location}


def test_steered_prompt_rows_match_interpreter_reference(unified_llm, registry, model_info):
    layer = model_info["num_layers"] // 2
    vector = torch.randn(model_info["hidden_size"], dtype=torch.float32)
    artifact_id = registry.write({"vector": vector})

    baseline = generate_one(
        unified_llm, PROMPT,
        extra={"capture": _capture([layer])},
        salt=str(uuid.uuid4()),
    )
    steered = generate_one(
        unified_llm, PROMPT,
        extra={
            "intervention_spec": {"ops": [{
                "layers": [layer],
                "transform": {"kind": "additive", "strength": 4.0, "artifact": artifact_id},
                "scope": {"kind": "all"},
                "gate": None,
            }]},
            "capture": _capture([layer]),
        },
        salt=str(uuid.uuid4()),
    )

    n_prompt = len(baseline.prompt_token_ids)
    base_manifest, base_tensors = load_capture(baseline)
    steer_manifest, steer_tensors = load_capture(steered)
    base_rows, base_positions = prompt_rows(base_manifest, base_tensors, layer, n_prompt)
    steer_rows, steer_positions = prompt_rows(steer_manifest, steer_tensors, layer, n_prompt)
    assert base_positions == steer_positions

    # Same tokens, eager mode: the pre-op layer output is identical, so the
    # engine-steered rows must equal the reference math applied outside.
    reference = additive(
        base_rows.float(), vector=vector, strength=4.0
    ).to(base_rows.dtype)
    assert torch.allclose(steer_rows, reference, atol=2e-2, rtol=2e-2)


def test_directional_ablation_reference(unified_llm, registry, model_info):
    layer = model_info["num_layers"] // 2
    vector = torch.randn(model_info["hidden_size"], dtype=torch.float32)
    artifact_id = registry.write({"vector": vector})

    baseline = generate_one(
        unified_llm, PROMPT, extra={"capture": _capture([layer])}, salt=str(uuid.uuid4())
    )
    steered = generate_one(
        unified_llm, PROMPT,
        extra={
            "intervention_spec": {"ops": [{
                "layers": [layer],
                "transform": {"kind": "directional_ablation", "artifact": artifact_id},
                "scope": {"kind": "all"},
                "gate": None,
            }]},
            "capture": _capture([layer]),
        },
        salt=str(uuid.uuid4()),
    )

    n_prompt = len(baseline.prompt_token_ids)
    base_manifest, base_tensors = load_capture(baseline)
    steer_manifest, steer_tensors = load_capture(steered)
    base_rows, _ = prompt_rows(base_manifest, base_tensors, layer, n_prompt)
    steer_rows, _ = prompt_rows(steer_manifest, steer_tensors, layer, n_prompt)

    reference = directional_ablation(base_rows.float(), vector=vector).to(base_rows.dtype)
    assert torch.allclose(steer_rows, reference, atol=2e-2, rtol=2e-2)


def test_stream_resplit_matches_layer_boundary(unified_llm, registry, model_info):
    """The steered rows' residual write-back must reach the next layer as
    hidden + residual' == stream': layer_input capture at layer+1 equals
    layer_output capture at layer, under the same intervention.
    """
    layer = model_info["num_layers"] // 2
    vector = torch.randn(model_info["hidden_size"], dtype=torch.float32)
    artifact_id = registry.write({"vector": vector})
    spec = {"ops": [{
        "layers": [layer],
        "transform": {"kind": "additive", "strength": 4.0, "artifact": artifact_id},
        "scope": {"kind": "all"},
        "gate": None,
    }]}

    out_run = generate_one(
        unified_llm, PROMPT,
        extra={"intervention_spec": spec, "capture": _capture([layer], "layer_output")},
        salt=str(uuid.uuid4()),
    )
    in_run = generate_one(
        unified_llm, PROMPT,
        extra={"intervention_spec": spec, "capture": _capture([layer + 1], "layer_input")},
        salt=str(uuid.uuid4()),
    )

    n_prompt = len(out_run.prompt_token_ids)
    out_manifest, out_tensors = load_capture(out_run)
    in_manifest, in_tensors = load_capture(in_run)
    out_rows, _ = prompt_rows(out_manifest, out_tensors, layer, n_prompt)
    in_rows, _ = prompt_rows(in_manifest, in_tensors, layer + 1, n_prompt)
    assert torch.allclose(out_rows, in_rows, atol=2e-2, rtol=2e-2)


def test_steering_changes_greedy_text(unified_llm, registry, model_info):
    layer = model_info["num_layers"] // 2
    vector = torch.randn(model_info["hidden_size"], dtype=torch.float32) * 4
    artifact_id = registry.write({"vector": vector})

    baseline = generate_one(unified_llm, PROMPT, max_tokens=16)
    steered = generate_one(
        unified_llm, PROMPT,
        extra={"intervention_spec": {"ops": [{
            "layers": [layer],
            "transform": {"kind": "additive", "strength": 12.0, "artifact": artifact_id},
            "scope": {"kind": "all"},
            "gate": None,
        }]}},
        salt=str(uuid.uuid4()),
        max_tokens=16,
    )
    assert steered.outputs[0].text != baseline.outputs[0].text
