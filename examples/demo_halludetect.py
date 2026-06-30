"""H-Node hallucination detection demo (inference-only).

Loads a pre-built probe artifact and scores example prompts via the
registered ``hallucination`` analyzer.

Usage:
    python examples/demo_halludetect.py

Requires:
    vllm_hook_plugins/vllm_hook_plugins/utils/hnode/artifacts/probe.npz  \\
    vllm_hook_plugins/vllm_hook_plugins/utils/hnode/artifacts/probe.json  — pre-built probe artifact.

    To build your own probe, see:
    https://github.com/Samarpit-bhatia/hnode-probe-builder

Method: "H-Node Attack and Defense in Large Language Models"
        Yocam, Vaidyan, Wang, 2026 — https://arxiv.org/abs/2506.07230
"""
from __future__ import annotations

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

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = "./cache/"
HOOK_DIR = "/dev/shm/vllm_hook"
ART_DIR = "./vllm_hook_plugins/vllm_hook_plugins/utils/hnode/artifacts"
PROBE_PATH = os.path.join(ART_DIR, "probe.npz")
INFER_CFG = "model_configs/hallucination_detection/Qwen2.5-1.5B-Instruct.infer.json"


def _make_llm(config_file: str, analyzer_name: str = "hidden_states"):
    from vllm_hook_plugins import HookLLM

    # Tuned for 8 GB laptop GPUs (RTX 4060 etc.). Qwen2.5-1.5B fp16 weights are
    # ~3 GB. gpu_memory_utilization=0.85 (~7 GB budget) leaves room for KV cache
    # and the captured activations; max_model_len/max_num_batched_tokens are
    # capped so vLLM doesn't reserve KV blocks for huge hypothetical batches.
    return HookLLM(
        model=MODEL,
        worker_name="probe_hidden_states",
        analyzer_name=analyzer_name,
        config_file=config_file,
        download_dir=CACHE_DIR,
        hook_dir=HOOK_DIR,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        max_num_batched_tokens=2048,
        trust_remote_code=True,
        dtype=torch.float16,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


def stage_detect():
    if not os.path.exists(PROBE_PATH):
        sys.exit(
            f"Probe artifact not found at {PROBE_PATH}.\n"
            "Download probe.npz + probe.json from:\n"
            "  https://github.com/Samarpit-bhatia/hnode-probe-builder\n"
            "and place them in vllm_hook_plugins/vllm_hook_plugins/utils/hnode/artifacts/."
        )

    examples = [
        "Q: What is the capital of France?\nA: Paris",
        "Q: What is the capital of France?\nA: London",
        "Q: Who wrote Hamlet?\nA: William Shakespeare",
        "Q: Who wrote Hamlet?\nA: Charles Dickens",
        "Q: What is 2 + 2?\nA: 4",
        "Q: What is 2 + 2?\nA: 5",
    ]

    print("Loading model with inference config (best layer only) + hallucination analyzer...")
    llm = _make_llm(INFER_CFG, analyzer_name="hallucination")

    print("Running detection on example prompts...\n")
    run_id = "halludetect_detect"
    llm.generate(examples, SamplingParams(temperature=0.0, max_tokens=1),
                 save_to_disk=True, run_id=run_id)
    result = llm.analyze(
        analyzer_spec={"probe_path": PROBE_PATH, "threshold": 0.5},
        run_id=run_id,
    )

    print(f"Best layer: {result['best_layer']}  |  threshold: {result['threshold']}")
    print("-" * 78)
    for prompt, p, exc, verdict in zip(
        examples, result["probabilities"], result["h_node_excess"], result["verdicts"]
    ):
        line = prompt.replace("\n", "  ")
        print(f"[{verdict:>12s}]  P(hall)={p:.3f}  H-excess={exc:.3f}  |  {line}")


if __name__ == "__main__":
    stage_detect()
