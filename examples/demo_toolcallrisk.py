"""Tool-call risk scoring of AgentDojo agent steps (inference only).

Runs offline with `MiaLLM`. Downloads a pre-built Qwen2.5-7B-Instruct probe and three
AgentDojo steps on first run, renders each step with the model's tool-calling chat template,
scores the last prompt token, and prints the score next to the action the model generated.

The artifact is hosted with the training code and cached under ./cache/tool_call_risk/:
https://github.com/rishabhsinha17/latent-state-auditing/tree/main/serving/vllm_hook
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import urllib.error
import urllib.request

from vllm import SamplingParams

from mia import MiaLLM
from _paths import config_path

MODEL = "Qwen/Qwen2.5-7B-Instruct"   # the probe is trained on this model's layer 28
INFER_CFG = config_path("tool_call_risk/Qwen2.5-7B-Instruct.infer.json")

ARTIFACT_BASE_URL = (
    "https://raw.githubusercontent.com/rishabhsinha17/latent-state-auditing/"
    "main/serving/vllm_hook/artifacts/qwen2.5-7b-instruct"
)
ART_DIR = "./cache/tool_call_risk"


def ensure_artifacts():
    """Download probe.npz, probe.json and demo_steps.jsonl into ART_DIR if missing."""
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


def report(steps, outputs, result):
    """Print the verdict and probability per step, with the generated action."""
    print(f"Layer: {result['layer']}  |  threshold: {result['threshold']}")
    print("-" * 78)
    for s, out, p, verdict in zip(steps, outputs, result["probabilities"], result["verdicts"]):
        action = out.outputs[0].text.strip().replace("\n", " ")
        print(f"[{verdict:>4s}]  P(unsafe)={p:.3f}  {s['label']}")
        print(f"        task:   {s['messages'][1]['content'].strip()[:100]}")
        print(f"        action: {action[:160]}\n")


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

    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="tool_call_risk",
                 config_file=INFER_CFG, hook_dir="/dev/shm/mia",
                 gpu_memory_utilization=0.85, max_model_len=16384)

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
    report(steps, outputs, llm.analyze(analyzer_spec=spec, run_id=run_id))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
