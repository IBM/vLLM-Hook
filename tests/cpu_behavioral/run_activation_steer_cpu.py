"""Run the smallest local CPU activation-steering fixture through vLLM."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import torch
from vllm import SamplingParams

from mia import MiaLLM, register_plugins


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = os.environ.get("MIA_CPU_MODEL", "facebook/opt-125m")
HOOK_DIR = REPO_ROOT / "tests" / "cpu_behavioral" / ".hooks"


def log(message: str) -> None:
    print(f"[activation-steer-cpu] {message}", flush=True)


def _write_config() -> Path:
    cfg = HOOK_DIR / "activation_steer_cpu_config.json"
    vec_path = HOOK_DIR / "activation_steer_cpu_vector.pt"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dir": torch.ones(768, dtype=torch.float32)}, vec_path)
    log(f"wrote dummy steering vector={vec_path}")
    cfg.write_text(
        json.dumps(
            {
                "steering": {
                    "method": "add_vector",
                    "coefficient": 0.0,
                    "optimal_layer": 1,
                    "vector_path": str(vec_path),
                    "apply_at_all_positions": True,
                }
            },
            indent=2,
        )
    )
    log(f"wrote config={cfg}")
    return cfg


def main() -> None:
    log(f"repo={REPO_ROOT}")
    log(f"model={MODEL_ID}")
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("MIA_WRITER_PROCESS", "0")

    log("registering MIA plugins")
    register_plugins()

    log("building vLLM CPU engine with steer worker")
    llm = MiaLLM(
        model=MODEL_ID,
        worker_name="steer",
        analyzer_name=None,
        config_file=str(_write_config()),
        hook_dir=str(HOOK_DIR),
        dtype=torch.float32,
        max_model_len=64,
        gpu_memory_utilization=0.25,
        enable_hook=True,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    log("running generate() with steering enabled")
    outputs = llm.generate(
        ["Activation steering CPU fixture."],
        SamplingParams(temperature=0.0, max_tokens=1),
        use_hook=True,
    )

    assert len(outputs) == 1
    assert outputs[0].outputs
    assert outputs[0].outputs[0].text is not None

    log("shutting down vLLM engine")
    llm.llm_engine.engine_core.shutdown()
    log("passed")


if __name__ == "__main__":
    main()
