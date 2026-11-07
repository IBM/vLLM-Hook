"""Two-stage tool-call risk scoring of AgentDojo agent steps (inference only).

Stage 1 is the tool-call risk probe. Stage 2 is an injection-presence probe on the same layer, so
the cascade needs no extra capture. A step is flagged only when both fire, which removes most
false alarms on steps without an injection (see docs/use_cases/two_stage.md).

Downloads the probes, the calibrated cascade.json and three AgentDojo steps on first run into
./cache/two_stage/.
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

MODEL = "Qwen/Qwen2.5-7B-Instruct"   # both probes read this model's layer 28
INFER_CFG = config_path("tool_call_risk/Qwen2.5-7B-Instruct.infer.json")

ARTIFACT_BASE_URL = (
    "https://raw.githubusercontent.com/rishabhsinha17/latent-state-auditing/"
    "main/serving/vllm_hook/artifacts/qwen2.5-7b-instruct"
)
ART_DIR = "./cache/two_stage"
FILES = ("two_stage/cascade.json", "two_stage/stage1/probe.npz", "two_stage/stage1/probe.json",
         "two_stage/stage2/probe.npz", "two_stage/stage2/probe.json", "demo_steps.jsonl")


def ensure_artifacts():
    """Download the cascade files and demo steps into ART_DIR if missing."""
    for name in FILES:
        dest = os.path.join(ART_DIR, name)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{ARTIFACT_BASE_URL}/{name}"
        print(f"Downloading {name} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as exc:
            sys.exit(f"Could not download {name} ({exc}).\nDownload it manually from {url}\n"
                     f"and place it in {ART_DIR}/.")


def report(steps, actions, result):
    """Print each stage's probability, the verdict and the generated action."""
    s1, s2 = result["stage1"], result["stage2"]
    print(f"Layers {s1['layer']} and {s2['layer']}  |  thresholds {s1['threshold']:.4f} and {s2['threshold']:.4f}")
    print("-" * 78)
    for i, (s, text) in enumerate(zip(steps, actions)):
        p2 = s2["probabilities"][i]
        p2_text = "not run" if p2 is None else f"{p2:.3f}"
        print(f"[{result['verdicts'][i]:>4s}]  risk={s1['probabilities'][i]:.3f}  injection={p2_text}  {s['label']}")
        print(f"        action: {text.strip().replace(chr(10), ' ')[:160]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    ensure_artifacts()
    with open(os.path.join(ART_DIR, "demo_steps.jsonl")) as f:
        steps = [json.loads(line) for line in f]

    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="two_stage_probe",
                 config_file=INFER_CFG, hook_dir="/dev/shm/mia",
                 gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    prompts = [
        {"prompt_token_ids": llm.tokenizer.apply_chat_template(
            s["messages"], tools=s["tools"], add_generation_prompt=True, tokenize=True,
            return_dict=False)}
        for s in steps
    ]
    run_id = "two_stage_demo"
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256),
                           save_to_disk=True, run_id=run_id)
    spec = {"cascade_path": os.path.join(ART_DIR, "two_stage", "cascade.json"), "score_all": True}
    report(steps, [o.outputs[0].text for o in outputs], llm.analyze(analyzer_spec=spec, run_id=run_id))


if __name__ == "__main__":
    # Same capture budget as demo_toolcallrisk.py: one captured layer on a 24 GB GPU.
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
