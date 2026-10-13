"""Start the local CPU engine and install MIA hooks without generation.

This is intentionally smaller than ``run_hidden_states_cpu.py``. It verifies
that vLLM CPU can start, MIA can inject into CPUWorker, and the hidden-state
worker can install hooks on the CPU V2 runner. It stops before token generation,
which currently enters vLLM's Triton CPU slot-mapping kernel.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import torch

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
        max_model_len=128,
        gpu_memory_utilization=0.25,
        enable_hook=True,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    llm.llm.collective_rpc("install_hooks")
    print("CPU hook-install probe passed.")

    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
