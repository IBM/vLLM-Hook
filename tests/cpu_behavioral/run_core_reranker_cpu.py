"""Run the smallest local CPU CoRe reranker fixture through vLLM."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import torch
from vllm import SamplingParams

from mia import MiaLLM, register_plugins
from backend_support import assert_backend, hook_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = os.environ.get("MIA_CPU_MODEL", "facebook/opt-125m")
HOOK_DIR = hook_directory(REPO_ROOT)


def log(message: str) -> None:
    print(f"[core-reranker-cpu] {message}", flush=True)


def _write_config() -> Path:
    cfg = HOOK_DIR / "core_reranker_cpu_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps(
            {
                "params": {"important_heads": [[0, 0]]},
                "hookq": {"hookq_mode": "all_tokens"},
            },
            indent=2,
        )
    )
    log(f"wrote config={cfg}")
    return cfg


def _prompt_and_spec(tokenizer, query: str):
    prefix = "Documents:\n"
    docs = ["Rome was founded in 753 BC.", "New York was founded in 1624."]
    prompt = prefix
    spans = []
    for i, doc in enumerate(docs, start=1):
        prompt += f"[document {i}] "
        start = len(tokenizer(prompt).input_ids)
        prompt += doc
        end = len(tokenizer(prompt).input_ids) - 1
        spans.append((start, end))
        prompt += "\n"
    prompt += "Question: "
    query_start = len(tokenizer(prompt).input_ids)
    after_instruction = query_start
    prompt += query
    query_end = len(tokenizer(prompt).input_ids) - 1
    return prompt, (spans, query_start, after_instruction, query_end), len(docs)


def main() -> None:
    log(f"repo={REPO_ROOT}")
    log(f"model={MODEL_ID}")
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("MIA_WRITER_PROCESS", "0")

    log("registering MIA plugins")
    register_plugins()

    log("building vLLM CPU engine with capture_qk worker")
    assert_backend()
    llm = MiaLLM(
        model=MODEL_ID,
        worker_name="capture_qk",
        analyzer_name="core_reranker",
        config_file=str(_write_config()),
        hook_dir=str(HOOK_DIR),
        dtype=torch.float32,
        max_model_len=96,
        gpu_memory_utilization=0.25,
        enable_hook=True,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    try:
        doc_prompt, query_spec, doc_count = _prompt_and_spec(
            llm.tokenizer,
            "Which city is older: Rome or New York?",
        )
        na_prompt, na_spec, _ = _prompt_and_spec(llm.tokenizer, "N/A")
        log(f"built document prompt with {doc_count} document span(s)")

        log("running document Q/K capture generation")
        llm.generate(
            doc_prompt,
            SamplingParams(temperature=0.0, max_tokens=1),
            save_to_disk=True,
            run_id="core-reranker-cpu-doc",
        )
        log("running N/A baseline Q/K capture generation")
        llm.generate(
            na_prompt,
            SamplingParams(temperature=0.0, max_tokens=1),
            save_to_disk=True,
            run_id="core-reranker-cpu-na",
        )
        log("running core_reranker analyzer")
        stats = llm.analyze(
            analyzer_spec={"query_spec": query_spec, "na_spec": na_spec},
            run_ids=["core-reranker-cpu-doc", "core-reranker-cpu-na"],
        )

        assert "scores" in stats
        assert "ranking" in stats
        assert len(stats["scores"][0]) == doc_count
        assert len(stats["ranking"][0]) == doc_count
    finally:
        log("shutting down vLLM engine")
        llm.llm_engine.engine_core.shutdown()
    log(f"passed: scores={stats['scores']} ranking={stats['ranking']}")


if __name__ == "__main__":
    main()
