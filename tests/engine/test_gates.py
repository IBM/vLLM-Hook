# tests/engine/test_gates.py
"""Structured gate decisions end to end against evidence precomputed from a
capture run: the prompt-end freeze under real prefill, and per-request
isolation when two gated requests share a batched step.
"""
import uuid

import torch

from tests.engine.conftest import generate_one, load_capture, prompt_rows

PROMPT = "The committee reviewed the proposal carefully before deciding that"


def _calibrate(unified_llm, model_info, condition_layer):
    """Capture the condition layer's input for the prompt and return a
    probe weight plus the exact mean-pooled evidence the gate will see at
    the end of the prefill pass.
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
    prompt_mean = float((rows.float() @ weight).mean())
    return weight, prompt_mean


def _gated_spec(registry, hidden_size, op_layer, condition_layer, weight, bias):
    """A structured gate: one affine weight row over the condition layer,
    decided by sum_threshold with the bias inline in the rule. The affine
    readout artifact carries only weights; the bias is not an artifact.
    """
    vector = torch.randn(hidden_size, dtype=torch.float32) * 2
    vector_id = registry.write({"vector": vector})
    probe_id = registry.write({"weights": weight.unsqueeze(0)})
    return {"ops": [{
        "layers": [op_layer],
        "transform": {"kind": "additive", "strength": 16.0, "artifact": vector_id},
        "scope": {"kind": "all"},
        "gate": {
            "layers": [condition_layer],
            "pooling": "mean",
            "readout": {"kind": "affine", "artifact": probe_id},
            "rule": {"kind": "sum_threshold", "bias": bias},
        },
    }]}


def test_gate_open_and_closed(unified_llm, registry, model_info):
    """The prompt-end freeze decides the gate: a bias that clears the
    pooled evidence steers every token, one that never can leaves the
    baseline untouched.
    """
    condition_layer = 1
    op_layer = model_info["num_layers"] // 2
    weight, prompt_mean = _calibrate(unified_llm, model_info, condition_layer)
    margin = abs(prompt_mean) * 0.5 + 1.0

    baseline = generate_one(unified_llm, PROMPT, max_tokens=16)

    open_spec = _gated_spec(registry, model_info["hidden_size"], op_layer,
                            condition_layer, weight, bias=margin - prompt_mean)
    opened = generate_one(unified_llm, PROMPT, extra={"intervention_spec": open_spec},
                          salt=str(uuid.uuid4()), max_tokens=16)
    assert opened.outputs[0].text != baseline.outputs[0].text

    closed_spec = _gated_spec(registry, model_info["hidden_size"], op_layer,
                              condition_layer, weight, bias=-prompt_mean - 1e6)
    closed = generate_one(unified_llm, PROMPT, extra={"intervention_spec": closed_spec},
                          salt=str(uuid.uuid4()), max_tokens=16)
    assert closed.outputs[0].text == baseline.outputs[0].text


def test_batched_requests_gate_independently(unified_llm, registry, model_info):
    """Two gated requests in one generate call must decide independently:
    each request's GateState sees only its own rows, so an open spec steers
    while a closed spec in the same batched step does not.
    """
    from vllm import SamplingParams

    condition_layer = 1
    op_layer = model_info["num_layers"] // 2
    weight, prompt_mean = _calibrate(unified_llm, model_info, condition_layer)
    margin = abs(prompt_mean) * 0.5 + 1.0

    baseline = generate_one(unified_llm, PROMPT, max_tokens=16)

    open_spec = _gated_spec(registry, model_info["hidden_size"], op_layer,
                            condition_layer, weight, bias=margin - prompt_mean)
    closed_spec = _gated_spec(registry, model_info["hidden_size"], op_layer,
                              condition_layer, weight, bias=-prompt_mean - 1e6)

    requests = [
        {"prompt": PROMPT, "cache_salt": str(uuid.uuid4())},
        {"prompt": PROMPT, "cache_salt": str(uuid.uuid4())},
    ]
    params = [
        SamplingParams(temperature=0.0, max_tokens=16,
                       extra_args={"intervention_spec": open_spec}),
        SamplingParams(temperature=0.0, max_tokens=16,
                       extra_args={"intervention_spec": closed_spec}),
    ]
    opened, closed = unified_llm.generate(requests, params, use_tqdm=False)
    assert opened.outputs[0].text != baseline.outputs[0].text
    assert closed.outputs[0].text == baseline.outputs[0].text
