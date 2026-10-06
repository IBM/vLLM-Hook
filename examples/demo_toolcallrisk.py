"""Tool-call risk demo (inference-only).

Downloads a pre-built probe for Qwen2.5-7B-Instruct and three AgentDojo agent
steps on first run, renders each step with the model's tool-calling chat
template, scores the last prompt token with the registered ``tool_call_risk``
analyzer, and prints the score next to the action the model generated.

Usage:
    python examples/demo_toolcallrisk.py
    python examples/demo_toolcallrisk.py --probe path/to/probe.npz   # your own probe

The artifact (probe.npz + probe.json + demo_steps.jsonl) is hosted with the
training code and cached under ./cache/tool_call_risk/:

    https://github.com/rishabhsinha17/latent-state-auditing/tree/main/serving/vllm_hook
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.append(_root)

import torch
from vllm import SamplingParams

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "./cache/"
HOOK_DIR = "/dev/shm/vllm_hook"
INFER_CFG = "model_configs/tool_call_risk/Qwen2.5-7B-Instruct.infer.json"

ARTIFACT_BASE_URL = (
    "https://raw.githubusercontent.com/rishabhsinha17/latent-state-auditing/"
    "main/serving/vllm_hook/artifacts/qwen2.5-7b-instruct"
)
ART_DIR = "./cache/tool_call_risk"


def ensure_artifacts():
    """Download probe.npz, probe.json and demo_steps.jsonl into ART_DIR if missing."""
    import urllib.error
    import urllib.request

    os.makedirs(ART_DIR, exist_ok=True)
    for name in ("probe.npz", "probe.json", "demo_steps.jsonl"):
        dest = os.path.join(ART_DIR, name)
        if os.path.exists(dest):
            continue
        url = f"{ARTIFACT_BASE_URL}/{name}"
        print(f"Downloading {name} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as exc:
            sys.exit(
                f"Could not download {name} ({exc}).\n"
                f"Download it manually from {url}\n"
                f"and place it in {ART_DIR}/."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default=None, help="Path to probe.npz (probe.json beside it).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the threshold stored in probe.json.")
    args = parser.parse_args()

    ensure_artifacts()
    probe_path = args.probe or os.path.join(ART_DIR, "probe.npz")
    with open(os.path.join(ART_DIR, "demo_steps.jsonl")) as f:
        steps = [json.loads(line) for line in f]

    from vllm_hook_plugins import HookLLM

    llm = HookLLM(
        model=MODEL,
        worker_name="probe_hidden_states",
        analyzer_name="tool_call_risk",
        config_file=INFER_CFG,
        download_dir=CACHE_DIR,
        hook_dir=HOOK_DIR,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        dtype=torch.bfloat16,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    prompts = [
        {"prompt_token_ids": llm.tokenizer.apply_chat_template(
            s["messages"], tools=s["tools"], add_generation_prompt=True, tokenize=True)}
        for s in steps
    ]

    run_id = "toolcallrisk_demo"
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256),
                           save_to_disk=True, run_id=run_id)

    spec = {"probe_path": probe_path}
    if args.threshold is not None:
        spec["threshold"] = args.threshold
    result = llm.analyze(analyzer_spec=spec, run_id=run_id)

    print(f"Layer: {result['layer']}  |  threshold: {result['threshold']}")
    print("-" * 78)
    for s, out, p, verdict in zip(steps, outputs, result["probabilities"], result["verdicts"]):
        action = out.outputs[0].text.strip().replace("\n", " ")
        print(f"[{verdict:>4s}]  P(unsafe)={p:.3f}  {s['label']}")
        print(f"        task:   {s['messages'][1]['content'].strip()[:100]}")
        print(f"        action: {action[:160]}\n")


if __name__ == "__main__":
    main()
