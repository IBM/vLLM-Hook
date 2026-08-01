# tests/engine/test_gates.py
"""Gate decisions against evidence precomputed from a capture run, and the
cache_once hold.
"""
import uuid

import torch

from tests.engine.conftest import generate_one, load_capture, prompt_rows

PROMPT = "The committee reviewed the proposal carefully before deciding that"


def _calibrate(unified_llm, model_info, condition_layer):
    """Capture the condition layer's input for the prompt and return a
    probe weight plus the exact evidence sum the gate will see at the end
    of the prefill pass.
    """
    output = generate_one(
        unified_llm, PROMPT,
        extra={"capture": {"layers": [condition_layer], "mode": "all_tokens",
                           "location": "layer_input"}},
        salt=str(uuid.uuid4()),
    )
    n_prompt = len(output.prompt_token_ids)
    manifest, tensors = load_capture(output)
    rows, _ = prompt_rows(manifest, tensors, condition_layer, n_prompt)
    weight = rows.float().mean(dim=0)
    weight = weight / weight.norm()
    prompt_sum = float((rows.float() @ weight).sum())
    return weight, prompt_sum


def _gated_spec(registry, hidden_size, op_layer, condition_layer, weight, threshold,
                cache_once=False):
    vector = torch.randn(hidden_size, dtype=torch.float32) * 2
    vector_id = registry.write({"vector": vector})
    probe_id = registry.write({"weight": weight})
    inner = {"kind": "probe_sum", "threshold": threshold,
             "condition_layers": [condition_layer], "artifact": probe_id}
    gate = {"kind": "cache_once", "inner": inner} if cache_once else inner
    return {"ops": [{
        "layers": [op_layer],
        "transform": {"kind": "additive", "strength": 16.0, "artifact": vector_id},
        "scope": {"kind": "all"},
        "gate": gate,
    }]}


def test_probe_sum_gate_open_and_closed(unified_llm, registry, model_info):
    condition_layer = 1
    op_layer = model_info["num_layers"] // 2
    weight, prompt_sum = _calibrate(unified_llm, model_info, condition_layer)
    margin = abs(prompt_sum) * 0.5 + 1.0

    baseline = generate_one(unified_llm, PROMPT, max_tokens=16)

    open_spec = _gated_spec(registry, model_info["hidden_size"], op_layer,
                            condition_layer, weight, threshold=prompt_sum - margin)
    opened = generate_one(unified_llm, PROMPT, extra={"intervention_spec": open_spec},
                          salt=str(uuid.uuid4()), max_tokens=16)
    assert opened.outputs[0].text != baseline.outputs[0].text

    closed_spec = _gated_spec(registry, model_info["hidden_size"], op_layer,
                              condition_layer, weight, threshold=prompt_sum + 1e6)
    closed = generate_one(unified_llm, PROMPT, extra={"intervention_spec": closed_spec},
                          salt=str(uuid.uuid4()), max_tokens=16)
    assert closed.outputs[0].text == baseline.outputs[0].text


def test_cache_once_decision_holds_for_the_request(unified_llm, registry, model_info):
    """cache_once freezes the prompt-end decision: an open decision keeps
    steering all decode steps, a closed one never steers, regardless of
    how decode-time evidence would move the inner gate.
    """
    condition_layer = 1
    op_layer = model_info["num_layers"] // 2
    weight, prompt_sum = _calibrate(unified_llm, model_info, condition_layer)
    margin = abs(prompt_sum) * 0.5 + 1.0

    baseline = generate_one(unified_llm, PROMPT, max_tokens=16)

    open_spec = _gated_spec(registry, model_info["hidden_size"], op_layer, condition_layer,
                            weight, threshold=prompt_sum - margin, cache_once=True)
    opened = generate_one(unified_llm, PROMPT, extra={"intervention_spec": open_spec},
                          salt=str(uuid.uuid4()), max_tokens=16)
    assert opened.outputs[0].text != baseline.outputs[0].text

    closed_spec = _gated_spec(registry, model_info["hidden_size"], op_layer, condition_layer,
                              weight, threshold=prompt_sum + 1e6, cache_once=True)
    closed = generate_one(unified_llm, PROMPT, extra={"intervention_spec": closed_spec},
                          salt=str(uuid.uuid4()), max_tokens=16)
    assert closed.outputs[0].text == baseline.outputs[0].text
