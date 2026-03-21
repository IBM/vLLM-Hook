import argparse
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from vllm_hook_plugins import HookLLM


def apply_attntracker_template(
    tokenizer,
    model_name: str,
    instruction: str,
    data: str,
):
    off_set = 0
    if "granite" in model_name.lower():
        prompt_prefix = "<|start_of_role|>user<|end_of_role|>"
        prompt_suffix = "<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
    elif "llama" in model_name.lower():
        prompt_prefix = "<|start_header_id|>user<|end_header_id|>"
        prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif "mistral" in model_name.lower():
        prompt_prefix = "[INST]"
        prompt_suffix = "[/INST]"
        off_set = 1
    elif "phi" in model_name.lower():
        prompt_prefix = "<|im_start|>user<|im_sep|>"
        prompt_suffix = "<|im_end|><|im_start|>assistant<|im_sep|>"
    else:
        prompt_prefix = ""
        prompt_suffix = ""

    full_prompt = f"{prompt_prefix}{instruction}{data}{prompt_suffix}"
    instruction_len = len(tokenizer(prompt_prefix + instruction).input_ids)
    data_end = len(tokenizer(full_prompt).input_ids) - off_set
    input_range = ((0, instruction_len), (instruction_len, data_end))
    return full_prompt, input_range


def apply_corer_template(
    tokenizer,
    model_name: str,
    query: str,
    documents: list[list[str]],
):
    off_set = 0
    if "granite" in model_name.lower():
        prompt_prefix = "<|start_of_role|>user<|end_of_role|>"
        prompt_suffix = "<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
    elif "llama" in model_name.lower():
        prompt_prefix = "<|start_header_id|>user<|end_header_id|>"
        prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif "mistral" in model_name.lower():
        prompt_prefix = "[INST]"
        prompt_suffix = "[/INST]"
        off_set = 1
    elif "phi" in model_name.lower():
        prompt_prefix = "<|im_start|>user<|im_sep|>"
        prompt_suffix = "<|im_end|><|im_start|>assistant<|im_sep|>"
    else:
        prompt_prefix = ""
        prompt_suffix = ""

    retrieval_instruction = " Here are some paragraphs:\n\n"
    retrieval_instruction_late = (
        "Please find information that are relevant to the following query in the "
        "paragraphs above.\n\nQuery: "
    )

    doc_span = []
    prompt = prompt_prefix + retrieval_instruction
    for i, doc in enumerate(documents):
        prompt += f"[document {i + 1}]"
        start_len = len(tokenizer(prompt).input_ids)
        prompt += " " + " ".join(doc)
        end_len = len(tokenizer(prompt).input_ids) - off_set
        doc_span.append((start_len, end_len))
        prompt += "\n\n"

    start_len = len(tokenizer(prompt).input_ids)
    prompt += retrieval_instruction_late
    after_instruct = len(tokenizer(prompt).input_ids) - off_set
    prompt += query.strip()
    end_len = len(tokenizer(prompt).input_ids) - off_set
    prompt += prompt_suffix

    query_spec = (doc_span, start_len, after_instruct, end_len)
    return prompt, query_spec


def latest_run_id(run_id_file: Path) -> str:
    run_ids = [line.strip() for line in run_id_file.read_text().splitlines() if line.strip()]
    if not run_ids:
        raise RuntimeError(f"No run IDs found in {run_id_file}")
    return run_ids[-1]


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def dtype_for_model(model: str):
    if model == "Qwen/Qwen2-1.5B-Instruct":
        return torch.float32
    return torch.float16


def default_config_file(project_root: Path, analyzer_mode: str, model: str) -> Path:
    model_name = model.split("/")[-1]
    if analyzer_mode == "attntracker":
        return project_root / "model_configs" / "attention_tracker" / f"{model_name}.json"
    return project_root / "model_configs" / "core_reranker" / f"{model_name}.json"


def flatten_layer_heads(important_heads: list[list[int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer, head in important_heads:
        out.setdefault(int(layer), []).append(int(head))
    for heads in out.values():
        heads.sort()
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", required=True, choices=["attntracker", "corer"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-json", required=True, help="Single-example JSON payload")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download-dir", default="~/.cache/vllm-hook")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.input_json).read_text())
    config_file = (
        Path(args.config_file).expanduser().resolve()
        if args.config_file
        else default_config_file(project_root, args.analyzer, args.model)
    )
    config = json.loads(config_file.read_text())
    important_heads = config.get("params", {}).get("important_heads", [])
    layer_to_heads = flatten_layer_heads(important_heads)
    hookq_mode = config.get("hookq", {}).get("hookq_mode")

    analyzer_name = "attention_tracker" if args.analyzer == "attntracker" else "core_reranker"
    llm = HookLLM(
        model=args.model,
        worker_name="probe_hook_qk",
        analyzer_name=analyzer_name,
        config_file=str(config_file),
        download_dir=os.path.expanduser(args.download_dir),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype_for_model(args.model),
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    if args.analyzer == "attntracker":
        prompt, input_range = apply_attntracker_template(
            llm.tokenizer,
            args.model,
            payload["instruction"],
            payload["data"],
        )
        analyzer_spec = {
            "input_range": input_range,
            "attn_func": payload.get("attn_func", "sum_normalize"),
        }
        llm.generate(prompt, temperature=args.temperature, max_tokens=args.max_tokens)
        analyzer_output = llm.analyze(analyzer_spec=analyzer_spec)
        spec_files = {
            "attn_spec.json": analyzer_spec,
        }
    else:
        prompt, query_spec = apply_corer_template(
            llm.tokenizer,
            args.model,
            payload["query"],
            payload["documents"],
        )
        _, na_spec = apply_corer_template(
            llm.tokenizer,
            args.model,
            payload.get("na_query", "N/A"),
            payload["documents"],
        )
        analyzer_spec = {
            "query_spec": query_spec,
            "na_spec": na_spec,
        }
        llm.generate(prompt, temperature=args.temperature, max_tokens=args.max_tokens)
        na_prompt, _ = apply_corer_template(
            llm.tokenizer,
            args.model,
            payload.get("na_query", "N/A"),
            payload["documents"],
        )
        llm.generate(na_prompt, cleanup=False, temperature=args.temperature, max_tokens=args.max_tokens)
        analyzer_output = llm.analyze(analyzer_spec=analyzer_spec)
        query_doc_span, query_start, after_instruct, query_end = analyzer_spec["query_spec"]
        _, na_query_start, na_after_instruct, na_query_end = analyzer_spec["na_spec"]
        spec_files = {
            "corer_spec.json": {
                "doc_span": [query_doc_span],
                "query_start": [query_start],
                "after_instruct": [after_instruct],
                "query_end": [query_end],
            },
            "corer_na_spec.json": {
                "doc_span": [query_doc_span],
                "query_start": [na_query_start],
                "after_instruct": [na_after_instruct],
                "query_end": [na_query_end],
            },
        }

    run_id = latest_run_id(Path(llm._run_id_file))
    artifact_paths = glob.glob(os.path.join(llm._hook_dir, run_id, "**", "qk.pt"), recursive=True)
    if not artifact_paths:
        raise FileNotFoundError(f"No qk.pt found under run_id={run_id} in {llm._hook_dir}")
    if len(artifact_paths) != 1:
        raise RuntimeError(f"Expected exactly one qk.pt, found {artifact_paths}")

    bundle_artifact_path = output_dir / "qk.pt"
    shutil.copy2(artifact_paths[0], bundle_artifact_path)

    for filename, data in spec_files.items():
        (output_dir / filename).write_text(json.dumps(data, indent=2))

    analyzer_output_path = output_dir / "analyzer_outputs.json"
    analyzer_output_path.write_text(json.dumps(analyzer_output, indent=2))

    metadata = {
        "model": args.model,
        "analyzer": args.analyzer,
        "analyzer_name": analyzer_name,
        "config_file": str(config_file),
        "git_commit": git_commit(project_root),
        "run_id": run_id,
        "hook_dir": llm._hook_dir,
        "artifact_source_path": artifact_paths[0],
        "artifact_bundle_path": str(bundle_artifact_path),
        "layer_to_heads": layer_to_heads,
        "hookq_mode": hookq_mode,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "input_payload": payload,
        "prompt_preview": prompt[:500],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"wrote_bundle={output_dir}")
    print(f"analyzer={args.analyzer} run_id={run_id}")
    print(f"qk={bundle_artifact_path}")
    for filename in sorted(spec_files):
        print(f"spec={output_dir / filename}")
    print(f"outputs={analyzer_output_path}")
    print(f"metadata={output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
