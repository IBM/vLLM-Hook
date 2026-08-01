# tests/engine/test_capture.py
"""Capture completeness under prefix caching (fresh salt per request),
both boundary locations, and last_token mode.
"""
import os
import uuid

import torch

from tests.engine.conftest import generate_one, load_capture

PROMPT = "Counting stars is easier than counting the reasons why the sky"


def _capture(layers, mode="all_tokens", location="layer_output", save_dir=None):
    capture = {"layers": layers, "mode": mode, "location": location}
    if save_dir is not None:
        capture["save_dir"] = save_dir
    return capture


def test_all_tokens_complete_under_prefix_caching(unified_llm, model_info):
    """A fresh random salt forces recompute, so every position is captured
    even when an identical prompt was fully cached by an earlier request.
    """
    layers = [1, model_info["num_layers"] - 1]
    first = generate_one(unified_llm, PROMPT,
                         extra={"capture": _capture(layers)}, salt=str(uuid.uuid4()))
    second = generate_one(unified_llm, PROMPT,
                          extra={"capture": _capture(layers)}, salt=str(uuid.uuid4()))

    for output in (first, second):
        n_prompt = len(output.prompt_token_ids)
        n_gen = len(output.outputs[0].token_ids)
        expected_len = n_prompt + n_gen - 1
        manifest, tensors = load_capture(output)
        assert manifest["mode"] == "all_tokens"
        assert manifest["location"] == "layer_output"
        for layer in layers:
            assert manifest["positions"][str(layer)] == list(range(expected_len))
            assert tensors[f"layer_{layer}"].shape == (expected_len, model_info["hidden_size"])


def test_last_token_mode(unified_llm, model_info):
    layer = model_info["num_layers"] // 2
    output = generate_one(unified_llm, PROMPT,
                          extra={"capture": _capture([layer], mode="last_token")},
                          salt=str(uuid.uuid4()))
    n_prompt = len(output.prompt_token_ids)
    n_gen = len(output.outputs[0].token_ids)
    expected_len = n_prompt + n_gen - 1
    manifest, tensors = load_capture(output)
    # one position per pass: the final prompt position, then each decode step
    assert manifest["positions"][str(layer)] == list(range(n_prompt - 1, expected_len))
    assert tensors[f"layer_{layer}"].shape[0] == n_gen


def test_layer_input_equals_previous_layer_output(unified_llm, model_info):
    layer = model_info["num_layers"] // 2
    out_run = generate_one(unified_llm, PROMPT,
                           extra={"capture": _capture([layer], location="layer_output")},
                           salt=str(uuid.uuid4()))
    in_run = generate_one(unified_llm, PROMPT,
                          extra={"capture": _capture([layer + 1], location="layer_input")},
                          salt=str(uuid.uuid4()))
    _, out_tensors = load_capture(out_run)
    _, in_tensors = load_capture(in_run)
    assert torch.allclose(
        out_tensors[f"layer_{layer}"], in_tensors[f"layer_{layer + 1}"], atol=2e-2, rtol=2e-2
    )


def test_capture_all_layers(unified_llm, model_info):
    output = generate_one(unified_llm, PROMPT,
                          extra={"capture": {"layers": "all"}}, salt=str(uuid.uuid4()))
    manifest, tensors = load_capture(output)
    assert manifest["layers"] == list(range(model_info["num_layers"]))
    assert len(tensors) == model_info["num_layers"]


def test_save_dir_writes_artifact_instead_of_rpc(unified_llm, model_info, tmp_path):
    layer = model_info["num_layers"] // 2
    save_dir = str(tmp_path / "captures")
    output = generate_one(unified_llm, PROMPT,
                          extra={"capture": _capture([layer], save_dir=save_dir)},
                          salt=str(uuid.uuid4()))
    assert getattr(output, "captures", None) is None
    req_id = output.request_id
    assert os.path.exists(os.path.join(save_dir, f"capture_{req_id}.safetensors"))
    assert os.path.exists(os.path.join(save_dir, f"capture_{req_id}.json"))
