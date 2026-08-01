"""Numerical parity at the layer_input boundary.

The unified worker materializes the layer boundary twice: once in each
layer's forward hook (location "layer_output") and once in the next
layer's forward pre-hook (location "layer_input"). Both must observe the
same value — layer N's output IS layer N+1's input. This script captures
both boundaries for the same prompts and reports the differences, layer
by layer, in the same format as verify_artifact_parity.py.

Run on a GPU host with the plugin installed:
    VLLM_HOOK_WORKER=unified python docs/numerical_analysis/verify_layer_input_parity.py
"""
import json
import os
import sys
import uuid
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "vllm_hook_plugins"))

MODEL_ID = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct"
    "/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
)
DOWNLOAD_DIR = os.path.expanduser("~/.cache")
LAYERS = list(range(1, 21))  # 0-based unified-surface indices; layer_input read at N+1
TARGET_PROMPT_LEN = 64
N_PROMPTS = 8

sys.path.insert(0, str(PROJECT_ROOT / "docs" / "numerical_analysis"))
from benchmark_hidden_states import _prompts_for_length

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_HOOK_WORKER", "unified")


def _build_prompts():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=DOWNLOAD_DIR)
    return _prompts_for_length(TARGET_PROMPT_LEN, tokenizer, n=N_PROMPTS)


def _capture_runs(llm, prompts, layers, location):
    """One capture-bearing request per prompt; returns layer -> list of
    (seq_len, hidden) tensors in prompt order.
    """
    import safetensors.torch
    from vllm import SamplingParams

    result = {layer: [] for layer in layers}
    for prompt in prompts:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"capture": {"layers": layers, "mode": "all_tokens",
                                    "location": location}},
        )
        output = llm.generate(
            [{"prompt": prompt, "cache_salt": str(uuid.uuid4())}], params, use_tqdm=False
        )[0]
        manifest_json, data = output.captures
        tensors = safetensors.torch.load(data)
        manifest = json.loads(manifest_json)
        for layer in layers:
            assert manifest["positions"][str(layer)], f"no positions for layer {layer}"
            result[layer].append(tensors[f"layer_{layer}"].float())
    return result


def compare(output_result, input_result):
    print(f"\n{'='*72}")
    print(f"{'Layer':>6}  {'Prompt':>8}  {'Shape':>16}  {'MaxAbsDiff':>12}  {'CosSim':>8}")
    print(f"{'-'*72}")

    for layer_idx in sorted(LAYERS):
        out_tensors = output_result.get(layer_idx, [])
        in_tensors = input_result.get(layer_idx + 1, [])
        if not out_tensors or not in_tensors:
            print(f"{layer_idx:>6}  {'N/A':>8}  {'missing data':>16}")
            continue

        for pi, (h, n) in enumerate(zip(out_tensors, in_tensors)):
            min_len = min(h.shape[0], n.shape[0])
            h = h[:min_len]
            n = n[:min_len]

            max_diff = (h - n).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                h.reshape(1, -1), n.reshape(1, -1)
            ).item()
            print(f"{layer_idx:>6}  {pi:>8}  {str(tuple(h.shape)):>16}  "
                  f"{max_diff:>12.6f}  {cos_sim:>8.6f}")

    print(f"{'='*72}")


if __name__ == "__main__":
    from vllm import LLM
    from vllm_hook_plugins import _hook_plugin

    _hook_plugin.register()
    prompts = _build_prompts()
    print(f"Using {len(prompts)} prompts of ~{TARGET_PROMPT_LEN} tokens each")

    llm = LLM(
        model=MODEL_ID,
        download_dir=DOWNLOAD_DIR,
        dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=True,
    )

    print("\nCapturing layer_output at layers", LAYERS)
    output_result = _capture_runs(llm, prompts, LAYERS, "layer_output")
    print("Capturing layer_input at layers", [l + 1 for l in LAYERS])
    input_result = _capture_runs(llm, prompts, [l + 1 for l in LAYERS], "layer_input")

    compare(output_result, input_result)
