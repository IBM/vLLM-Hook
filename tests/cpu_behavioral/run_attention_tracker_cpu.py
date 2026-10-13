"""Run the smallest local CPU Attention Tracker fixture through vLLM."""
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
    print(f"[attention-tracker-cpu] {message}", flush=True)


def _write_config() -> Path:
    cfg = HOOK_DIR / "attention_tracker_cpu_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps(
            {
                "params": {"important_heads": [[0, 0]]},
                "hookq": {"hookq_mode": "last_token"},
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

    log("building vLLM CPU engine with capture_qk worker")
    llm = MiaLLM(
        model=MODEL_ID,
        worker_name="capture_qk",
        analyzer_name="attn_tracker",
        config_file=str(_write_config()),
        hook_dir=str(HOOK_DIR),
        dtype=torch.float32,
        max_model_len=64,
        gpu_memory_utilization=0.25,
        enable_hook=True,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    prompts = [
        "Instruction: answer yes. Data: this is only a CPU test.",
        "Instruction: answer no. Data: this is another CPU test.",
    ]
    log(f"running generate() for {len(prompts)} prompt(s)")
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=1),
        use_hook=True,
    )

    ranges = []
    for prompt in prompts:
        token_count = len(llm.tokenizer(prompt)["input_ids"])
        split = max(1, token_count // 2)
        ranges.append([(0, split), (split, token_count)])
    log(f"built analyzer token ranges={ranges}")

    log("running attn_tracker analyzer")
    stats = llm.analyze(
        analyzer_spec={"input_range": ranges, "attn_func": "sum_normalize"},
        probes=outputs[0].probes,
    )

    assert "score" in stats
    assert len(stats["score"]) == len(prompts)
    assert all(torch.isfinite(torch.tensor(stats["score"])))

    log("shutting down vLLM engine")
    llm.llm_engine.engine_core.shutdown()
    log(f"passed: scores={stats['score']}")


if __name__ == "__main__":
    main()
