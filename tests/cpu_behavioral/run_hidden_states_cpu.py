"""Run the smallest local CPU hidden-state capture fixture.

This script intentionally uses the real MIA/vLLM path. It should be run only
inside an environment where vLLM CPU support has been installed.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import torch
from vllm import SamplingParams

from mia import MiaLLM, register_plugins


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = os.environ.get("MIA_CPU_MODEL", "facebook/opt-125m")
CONFIG_FILE = REPO_ROOT / "model_configs" / "hidden_states" / "Qwen2-1.5B-Instruct.json"


def main() -> None:
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("MIA_WRITER_PROCESS", "0")

    register_plugins()

    llm = MiaLLM(
        model=MODEL_ID,
        worker_name="capture_hs",
        analyzer_name="hidden_states",
        config_file=str(CONFIG_FILE),
        hook_dir=str(REPO_ROOT / "tests" / "cpu_behavioral" / ".hooks"),
        dtype=torch.float32,
        max_model_len=64,
        gpu_memory_utilization=0.25,
        enable_hook=True,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    prompts = [
        "Hidden states test.",
    ]

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=1),
        use_hook=True,
    )
    stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=outputs[0].probes)

    hidden_states = stats["hidden_states"]
    assert hidden_states, "Expected at least one captured hidden-state layer."

    hidden_size = llm.llm.llm_engine.model_config.hf_config.hidden_size
    for layer_name, tensors in sorted(hidden_states.items()):
        assert len(tensors) == len(prompts), (
            f"{layer_name}: expected {len(prompts)} tensors, got {len(tensors)}"
        )
        for tensor in tensors:
            assert tensor.device.type == "cpu"
            assert tensor.shape == torch.Size([hidden_size])
            assert torch.isfinite(tensor).all()
        print(f"{layer_name}: {len(tensors)} tensors, shape=({hidden_size},)")

    llm.llm_engine.engine_core.shutdown()
    print("CPU hidden-state behavior fixture passed.")


if __name__ == "__main__":
    main()

