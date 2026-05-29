"""Minimal MLX/GPU parity benchmarks for vLLM-Hook.

Run the same experiment twice with the same --benchmark-id:

  python tests/parity_tests/minimal_parity_benchmarks.py attn-tracker --backend metal --benchmark-id attn-tracker-001
  python tests/parity_tests/minimal_parity_benchmarks.py attn-tracker --backend non-metal --benchmark-id attn-tracker-001
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "experiment_runs" / "minimal_parity"

for import_path in (PROJECT_ROOT, PROJECT_ROOT / "vllm_hook_plugins"):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


ATTN_PROMPTS = [
    "Analyze and output the sentence attitude: This is for testing only.",
    "Analyze and output the sentence attitude: Another test run.",
]

HIDDEN_PROMPTS = [
    "Hidden states test prompt one.",
    "Hidden states test prompt two.",
]

DEFAULT_GRANITE_QUANTIZED_MODEL = "RedHatAI/granite-3.1-2b-instruct-quantized.w4a16"

CORE_QUERY = "Which city is older: Rome or New York?"
CORE_DOCUMENTS = [
    ["Rome was founded in 753 BC."],
    ["New York was founded in 1624."],
]

STEER_PROMPT = "This is for testing only."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=("hidden-states", "attn-tracker", "core-reranker", "steer-activation"),
    )
    parser.add_argument("--backend", choices=("metal", "non-metal"), required=True)
    parser.add_argument(
        "--benchmark-id",
        default=None,
        help="Shared W&B group id. Use the same value for Metal and non-Metal.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--hardware-label",
        default=os.environ.get("VLLM_HOOK_HARDWARE_LABEL", ""),
        help="Human-readable accelerator label, e.g. apple-metal, colab-T4.",
    )
    parser.add_argument(
        "--hardware-kind",
        default=os.environ.get("VLLM_HOOK_HARDWARE_KIND", ""),
        help="Hardware family, e.g. metal, cuda, cpu.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vllm-hook-platform-parity"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-api-key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    args = parser.parse_args()
    args.temperature = 0.0
    return args


def model_for(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    return {
        "hidden-states": DEFAULT_GRANITE_QUANTIZED_MODEL,
        "attn-tracker": DEFAULT_GRANITE_QUANTIZED_MODEL,
        "core-reranker": "mistralai/Mistral-7B-Instruct-v0.3",
        "steer-activation": "microsoft/Phi-3-mini-4k-instruct",
    }[args.experiment]


def config_kind(args: argparse.Namespace) -> str:
    return {
        "hidden-states": "hidden_states",
        "attn-tracker": "attention_tracker",
        "core-reranker": "core_reranker",
        "steer-activation": "activation_steer",
    }[args.experiment]


def ensure_config(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.config_file:
        return args.config_file

    kind = config_kind(args)
    short = model_for(args).split("/")[-1]
    config_dir = PROJECT_ROOT / "model_configs" / kind
    direct = config_dir / f"{short}.json"
    if direct.exists():
        source = direct
    else:
        source = out_dir / f"{kind}_{short}.json"
        if kind == "hidden_states":
            data = {"hidden_states": {"layers": [1, 2], "mode": "last_token"}}
        elif kind == "activation_steer":
            data = {
                "steering": {
                    "method": "adjust_rs",
                    "coefficient": 1,
                    "optimal_layer": 8,
                    "vector_path": str(PROJECT_ROOT / "steering_vectors" / "phi3_format.pt"),
                    "apply_at_all_positions": True,
                }
            }
        else:
            data = {
                "params": {
                    "temperature": args.temperature,
                    "max_output_tokens": args.max_tokens,
                    "important_heads": [[1, 2], [3, 4]],
                },
                "hookq": {"hookq_mode": "last_token" if args.experiment == "attn-tracker" else "all_tokens"},
            }
        source.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if kind != "activation_steer":
        return source

    data = json.loads(source.read_text(encoding="utf-8"))
    steering = data.setdefault("steering", {})
    vector_path = Path(str(steering.get("vector_path", ""))).expanduser()
    local_vector = PROJECT_ROOT / "steering_vectors" / "phi3_format.pt"
    if not vector_path.exists() and local_vector.exists():
        steering["vector_path"] = str(local_vector)
    normalized = out_dir / "activation_steer_config.json"
    normalized.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return normalized


def dtype_for(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }.get(name, torch.float16)


def worker_analyzer(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.experiment == "hidden-states":
        return "probe_hidden_states", "hidden_states"
    if args.experiment == "attn-tracker":
        return "probe_hook_qk", "attn_tracker"
    if args.experiment == "core-reranker":
        return "probe_hook_qk", "core_reranker"
    if args.experiment == "steer-activation":
        return "steer_hook_act", None
    raise ValueError(args.experiment)


def make_out_dir(args: argparse.Namespace) -> Path:
    group = args.benchmark_id or args.experiment
    path = args.output_dir / args.experiment / group / args.backend
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_llm(args: argparse.Namespace, out_dir: Path):
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    config_file = ensure_config(args, out_dir)
    args._resolved_config_file = str(config_file)
    worker_name, analyzer_name = worker_analyzer(args)
    hook_dir = out_dir / "hook_artifacts"
    hook_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        model=model_for(args),
        worker_name=worker_name,
        analyzer_name=analyzer_name,
        config_file=str(config_file),
        download_dir=str(args.download_dir or (PROJECT_ROOT / "cache")),
        hook_dir=str(hook_dir),
        enable_hook=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype_for(args.dtype),
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    if args.backend == "metal":
        os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "auto")
        from vllm_hook_plugins.metal.hook_llm_metal import HookLLMMetal

        return HookLLMMetal(**kwargs)

    from vllm_hook_plugins import register_plugins
    register_plugins()

    from vllm_hook_plugins import HookLLM

    return HookLLM(**kwargs)


def tensor_l2(value: Any) -> float:
    import torch

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return float(torch.norm(value.float()).item())


def flatten_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def read_run_ids(llm: Any) -> list[str]:
    path = Path(getattr(llm, "_run_id_file", ""))
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def qk_artifact_paths(llm: Any, run_id: str) -> list[Path]:
    hook_dir = Path(getattr(llm, "_hook_dir", ""))
    paths = []
    for filename in ("qk.pt", "qkv.pt"):
        paths.extend(hook_dir.glob(f"{run_id}/**/{filename}"))
    return sorted(path for path in paths if path.is_file())


def load_qk_like(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def qk_l2_from_artifacts(paths: list[Path]) -> dict[str, Any]:
    q_l2 = []
    k_l2 = []
    x_l2 = []
    by_layer: dict[str, dict[str, float]] = {}

    for path in paths:
        cache = load_qk_like(path)
        if "qk_cache" in cache:
            modules = cache["qk_cache"]
            for module_name, data in modules.items():
                layer = str(data.get("layer_num", module_name))
                q_vals = [tensor_l2(t) for t in data.get("q", [])]
                k_vals = [tensor_l2(t) for t in data.get("k_all", [])]
                q_l2.extend(q_vals)
                k_l2.extend(k_vals)
                by_layer.setdefault(layer, {})["q_l2_mean"] = flatten_mean(q_vals)
                by_layer.setdefault(layer, {})["k_l2_mean"] = flatten_mean(k_vals)
        elif "qkv_cache" in cache:
            for _, data in cache["qkv_cache"].items():
                layer = str(data.get("layer_num", "unknown"))
                kind = data.get("proj_kind")
                vals = [tensor_l2(t) for t in data.get("tokens", [])]
                if kind == "q":
                    q_l2.extend(vals)
                    by_layer.setdefault(layer, {})["q_l2_mean"] = flatten_mean(vals)
                elif kind == "k":
                    k_l2.extend(vals)
                    by_layer.setdefault(layer, {})["k_l2_mean"] = flatten_mean(vals)
                elif kind == "x":
                    x_l2.extend(vals)
                    by_layer.setdefault(layer, {})["hidden_state_l2_mean"] = flatten_mean(vals)

    return {
        "q_l2_mean": flatten_mean(q_l2),
        "k_l2_mean": flatten_mean(k_l2),
        "hidden_state_l2_mean_from_qkv_x": flatten_mean(x_l2),
        "l2_by_layer": by_layer,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def input_signature(args: argparse.Namespace, **values: Any) -> dict[str, Any]:
    config_file = Path(getattr(args, "_resolved_config_file", "")) if getattr(args, "_resolved_config_file", None) else None
    config_sha = file_sha256(config_file) if config_file and config_file.exists() else ""
    return {
        "experiment": args.experiment,
        "model": model_for(args),
        "temperature": 0.0 if args.experiment == "steer-activation" else args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "config_sha256": config_sha,
        **values,
    }


def set_comparison_identity(args: argparse.Namespace, signature: dict[str, Any]) -> str:
    key = sha256_text(canonical_json(signature))
    args._input_signature = signature
    args._comparison_key = key
    return key


def comparison_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "comparison_key": getattr(args, "_comparison_key", ""),
        "input_signature": getattr(args, "_input_signature", {}),
    }


def output_text(output: Any) -> str:
    return output[0].outputs[0].text


def output_tokens(output: Any) -> list[int]:
    return list(output[0].outputs[0].token_ids)


def reset_prefix_cache(llm: Any) -> None:
    try:
        llm.llm_engine.reset_prefix_cache()
    except Exception:
        pass


def attention_ranges(llm: Any, prompts: list[str]) -> list[list[tuple[int, int]]]:
    ranges = []
    for prompt in prompts:
        ids = llm.tokenizer(prompt)["input_ids"]
        length = len(ids)
        ranges.append([(0, length // 2), (length // 2, length)])
    return ranges


def run_attn_tracker(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    llm = make_llm(args, out_dir)
    start = time.perf_counter()
    _ = llm.generate(ATTN_PROMPTS, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, use_hook=True)
    generation_ms = (time.perf_counter() - start) * 1000
    ranges = attention_ranges(llm, ATTN_PROMPTS)
    comparison_key = set_comparison_identity(
        args,
        input_signature(
            args,
            prompts=ATTN_PROMPTS,
            input_ranges=ranges,
            attn_func="sum_normalize",
        ),
    )
    start = time.perf_counter()
    stats = llm.analyze(analyzer_spec={"input_range": ranges, "attn_func": "sum_normalize"})
    analysis_ms = (time.perf_counter() - start) * 1000
    run_id = read_run_ids(llm)[-1]
    qk_metrics = qk_l2_from_artifacts(qk_artifact_paths(llm, run_id))
    scores = [float(score) for score in stats["score"]]
    records = [
        {
            "case_index": i,
            "backend": args.backend,
            "comparison_key": comparison_key,
            "attn_score": score,
        }
        for i, score in enumerate(scores)
    ]
    metrics = {
        **comparison_fields(args),
        "attn_score_mean": flatten_mean(scores),
        "attn_score_by_case": scores,
        "generation_ms": generation_ms,
        "analysis_ms": analysis_ms,
        **qk_metrics,
    }
    reset_prefix_cache(llm)
    return metrics, records


def run_hidden_states(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    llm = make_llm(args, out_dir)
    comparison_key = set_comparison_identity(
        args,
        input_signature(
            args,
            config_sha256="hidden-states-minimal-v1",
            prompts=HIDDEN_PROMPTS,
            hidden_states={"metric": "hidden_state_l2_mean"},
            temperature=0.0,
            max_tokens=1,
        ),
    )
    _ = llm.generate(
        HIDDEN_PROMPTS,
        temperature=0.0,
        max_tokens=1,
        use_hook=True,
        save_to_disk=True,
    )
    stats = llm.analyze(analyzer_spec={"reduce": "norm"})
    hidden_by_layer = {
        layer: flatten_mean([float(v) for v in values])
        for layer, values in stats["hidden_states"].items()
    }
    metrics = {
        **comparison_fields(args),
        "hidden_state_l2_mean": flatten_mean(list(hidden_by_layer.values())),
        "hidden_state_l2_by_layer": hidden_by_layer,
        "hidden_state_source": "probe_hidden_states",
    }
    records = [
        {
            "backend": args.backend,
            "comparison_key": comparison_key,
            "layer": layer,
            "hidden_state_l2_mean": value,
        }
        for layer, value in hidden_by_layer.items()
    ]
    reset_prefix_cache(llm)
    return metrics, records


def core_prompt(tokenizer: Any, model_name: str, query: str, documents: list[list[str]]):
    offset = 0
    lower = model_name.lower()
    if "mistral" in lower:
        prefix, suffix, offset = "[INST]", "[/INST]", 1
    elif "phi" in lower:
        prefix, suffix = "<|im_start|>user<|im_sep|>", "<|im_end|><|im_start|>assistant<|im_sep|>"
    elif "granite" in lower:
        prefix, suffix = "<|start_of_role|>user<|end_of_role|>", "<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
    else:
        prefix, suffix = "", ""

    prompt = prefix + " Here are some paragraphs:\n\n"
    doc_span = []
    doc_tokens = []
    for index, doc in enumerate(documents):
        prompt += f"[document {index + 1}]"
        start = len(tokenizer(prompt).input_ids)
        prompt += " " + " ".join(doc)
        end = len(tokenizer(prompt).input_ids) - offset
        doc_span.append((start, end))
        doc_tokens.append(max(0, end - start + 1))
        prompt += "\n\n"
    query_start = len(tokenizer(prompt).input_ids)
    prompt += "Please find information that are relevant to the following query in the paragraphs above.\n\nQuery: "
    after_instruction = len(tokenizer(prompt).input_ids) - offset
    prompt += query.strip()
    query_end = len(tokenizer(prompt).input_ids) - offset
    prompt += suffix
    return prompt, (doc_span, query_start, after_instruction, query_end), doc_tokens


def run_core_reranker(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    llm = make_llm(args, out_dir)
    model_name = model_for(args)
    text_q, query_spec, doc_tokens = core_prompt(llm.tokenizer, model_name, CORE_QUERY, CORE_DOCUMENTS)
    text_na, na_spec, _ = core_prompt(llm.tokenizer, model_name, "N/A", CORE_DOCUMENTS)
    prompt_context_tokens = len(llm.tokenizer(text_q).input_ids)
    query_tokens = len(llm.tokenizer(CORE_QUERY).input_ids)
    comparison_key = set_comparison_identity(
        args,
        input_signature(
            args,
            query=CORE_QUERY,
            documents=CORE_DOCUMENTS,
            query_spec=query_spec,
            na_spec=na_spec,
            prompt_context_tokens=prompt_context_tokens,
            query_tokens=query_tokens,
            doc_tokens=doc_tokens,
        ),
    )
    llm.generate(text_q, temperature=args.temperature, max_tokens=1)
    llm.generate(text_na, cleanup=False, temperature=args.temperature, max_tokens=1)
    stats = llm.analyze(analyzer_spec={"query_spec": query_spec, "na_spec": na_spec})
    scores = [float(score) for score in stats["scores"][0]]
    ranking = [int(rank) for rank in stats["ranking"][0]]
    sorted_scores = sorted(scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    score_l2 = math.sqrt(sum(score * score for score in scores))
    metrics = {
        **comparison_fields(args),
        "scores": scores,
        "ranking": ranking,
        "score_margin_top1_top2": float(margin),
        "score_l2": float(score_l2),
        "prompt_context_tokens": prompt_context_tokens,
        "query_tokens": query_tokens,
        "doc_tokens": doc_tokens,
    }
    records = [
        {
            "backend": args.backend,
            "comparison_key": comparison_key,
            "document_index": i,
            "score": score,
            "doc_tokens": doc_tokens[i],
        }
        for i, score in enumerate(scores)
    ]
    reset_prefix_cache(llm)
    return metrics, records


def steering_vector_metrics(config_path: Path) -> dict[str, Any]:
    import torch

    data = json.loads(config_path.read_text(encoding="utf-8"))
    steering = data.get("steering", {})
    vector_path = Path(str(steering.get("vector_path", ""))).expanduser()
    metrics: dict[str, Any] = {
        "steering_method": steering.get("method", ""),
        "steering_optimal_layer": steering.get("optimal_layer"),
        "steering_coefficient": steering.get("coefficient"),
        "steering_apply_at_all_positions": steering.get("apply_at_all_positions"),
        "steering_vector_path": str(vector_path),
    }
    if vector_path.exists():
        loaded = torch.load(vector_path, map_location="cpu", weights_only=False)
        direction = loaded.get("dir") if isinstance(loaded, dict) else loaded
        metrics["steering_vector_l2"] = tensor_l2(direction)
        metrics["steering_vector_sha256"] = file_sha256(vector_path)
    return metrics


def run_steer_activation(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from vllm import SamplingParams

    llm = make_llm(args, out_dir)
    text = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": STEER_PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    steering_metrics = steering_vector_metrics(Path(args._resolved_config_file))
    comparison_key = set_comparison_identity(
        args,
        input_signature(
            args,
            prompt=STEER_PROMPT,
            rendered_prompt=text,
            steering_config_sha256=file_sha256(Path(args._resolved_config_file)),
            steering_vector_sha256=steering_metrics.get("steering_vector_sha256", ""),
            temperature=0.0,
        ),
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    steered = llm.generate(text, sampling_params=sampling_params, use_hook=True)
    reset_prefix_cache(llm)
    baseline = llm.generate(text, sampling_params=sampling_params, use_hook=False)
    steered_text = output_text(steered)
    baseline_text = output_text(baseline)
    metrics = {
        **comparison_fields(args),
        "temperature": 0.0,
        "text_changed_by_steering": steered_text != baseline_text,
        "baseline_tokens": output_tokens(baseline),
        "steered_tokens": output_tokens(steered),
        **steering_metrics,
    }
    records = [
        {
            "backend": args.backend,
            "comparison_key": comparison_key,
            "prompt": STEER_PROMPT,
            "baseline_text": baseline_text,
            "steered_text": steered_text,
            "text_changed_by_steering": metrics["text_changed_by_steering"],
        }
    ]
    reset_prefix_cache(llm)
    return metrics, records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_outputs(args: argparse.Namespace, out_dir: Path, metrics: dict[str, Any], records: list[dict[str, Any]]) -> tuple[Path, Path]:
    report = {
        "experiment": args.experiment,
        "backend": args.backend,
        "benchmark_id": args.benchmark_id or args.experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": base_params(args),
        "metrics": metrics,
        "records": records,
    }
    json_path = out_dir / f"{args.experiment}_{args.backend}.json"
    json_path.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    csv_path = out_dir / f"{args.experiment}_{args.backend}.csv"
    if records:
        fields = sorted({key for row in records for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({
                    key: json.dumps(value, default=json_default) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                })
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def base_params(args: argparse.Namespace) -> dict[str, Any]:
    worker_name, analyzer_name = worker_analyzer(args)
    return {
        "experiment": args.experiment,
        "backend": args.backend,
        "benchmark_id": args.benchmark_id or args.experiment,
        "model": model_for(args),
        "config_file": getattr(args, "_resolved_config_file", str(args.config_file)),
        "worker_name": worker_name,
        "analyzer_name": analyzer_name,
        "temperature": 0.0 if args.experiment == "steer-activation" else args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "hardware_label": args.hardware_label or inferred_hardware_label(args),
        "hardware_kind": args.hardware_kind or inferred_hardware_kind(args),
        "hardware_specs": hardware_specs(args),
        "comparison_key": getattr(args, "_comparison_key", ""),
        "input_signature": getattr(args, "_input_signature", {}),
        "platform": platform.platform(),
        "python": sys.version,
    }


def inferred_hardware_label(args: argparse.Namespace) -> str:
    if args.backend == "metal":
        return "apple-metal"
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "non-metal"


def inferred_hardware_kind(args: argparse.Namespace) -> str:
    if args.backend == "metal":
        return "metal"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def hardware_specs(args: argparse.Namespace) -> dict[str, Any]:
    specs: dict[str, Any] = {
        "label": args.hardware_label or inferred_hardware_label(args),
        "kind": args.hardware_kind or inferred_hardware_kind(args),
        "backend": args.backend,
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
    }
    try:
        import torch

        specs["torch_version"] = getattr(torch, "__version__", "")
        specs["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            specs["cuda_device_count"] = torch.cuda.device_count()
            specs["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            specs["cuda_total_memory_gb"] = round(props.total_memory / 1024**3, 3)
            specs["cuda_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        specs["torch_probe_error"] = type(exc).__name__
    try:
        import mlx.core as mx

        specs["mlx_available"] = True
        specs["mlx_default_device"] = str(mx.default_device())
    except Exception:
        specs["mlx_available"] = False
    if platform.system() == "Darwin":
        specs.update(mac_specs())
    return specs


def sysctl_value(name: str) -> str:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def mac_specs() -> dict[str, Any]:
    memsize = sysctl_value("hw.memsize")
    specs: dict[str, Any] = {
        "mac_model": sysctl_value("hw.model"),
        "mac_chip": sysctl_value("machdep.cpu.brand_string"),
        "mac_cpu_cores_physical": sysctl_value("hw.physicalcpu"),
        "mac_cpu_cores_logical": sysctl_value("hw.logicalcpu"),
        "mac_os_version": platform.mac_ver()[0],
    }
    if memsize.isdigit():
        specs["mac_memory_gb"] = round(int(memsize) / 1024**3, 3)
    return specs


def log_wandb(args: argparse.Namespace, out_dir: Path, metrics: dict[str, Any], records: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    if args.wandb_mode == "disabled":
        return
    import wandb
    try:
        import weave
    except ImportError:
        weave = None

    os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_api_key and args.wandb_mode == "online":
        if hasattr(wandb, "login"):
            wandb.login(key=args.wandb_api_key)
        else:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key

    project_name = wandb_project_for(args)
    run_name = wandb_run_name(args)

    if weave is not None and args.wandb_mode != "disabled":
        try:
            weave.init(project_name)
        except Exception as exc:
            print(f"weave init skipped: {type(exc).__name__}: {exc}", flush=True)

    group = args.benchmark_id or args.experiment
    run = wandb.init(
        project=project_name,
        entity=args.wandb_entity,
        group=group,
        name=run_name,
        job_type=args.experiment,
        tags=[args.experiment, args.backend, "minimal-parity"],
        config={**base_params(args), "wandb_project_requested": args.wandb_project},
    )
    try:
        wandb.log(metrics)
        if records:
            columns = sorted({key for row in records for key in row})
            wandb.log({
                f"{args.experiment}_records": wandb.Table(
                    columns=columns,
                    data=[[row.get(column) for column in columns] for row in records],
                )
            })
        artifact = wandb.Artifact(
            name=f"{args.experiment}-{args.backend}-{group}",
            type="vllm-hook-minimal-parity",
            metadata={
                "experiment": args.experiment,
                "backend": args.backend,
                "benchmark_id": group,
            },
        )
        artifact.add_file(str(json_path), name=json_path.name)
        artifact.add_file(str(csv_path), name=csv_path.name)
        for path in sorted(out_dir.glob("hook_artifacts/**/*")):
            if path.is_file() and path.name in {"qk.pt", "qkv.pt", "hidden_states.pt", "hidden_states.safetensors", "hidden_states.json"}:
                artifact.add_file(str(path), name=str(path.relative_to(out_dir)))
        run.log_artifact(artifact)
    finally:
        wandb.finish()


def wandb_project_for(args: argparse.Namespace) -> str:
    return {
        "attn-tracker": "attntracker",
        "core-reranker": "corereranker",
        "steer-activation": "steering",
        "hidden-states": "hiddenstates",
    }[args.experiment]


def sanitize_name(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("-")
    return "".join(cleaned).strip("-").lower()


def wandb_run_name(args: argparse.Namespace) -> str:
    platform_name = "metal" if args.backend == "metal" else "gpu"
    hardware = sanitize_name(args.hardware_label or inferred_hardware_label(args))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{platform_name}_{hardware}-{stamp}"


def main() -> int:
    args = parse_args()
    out_dir = make_out_dir(args)
    dispatch = {
        "hidden-states": run_hidden_states,
        "attn-tracker": run_attn_tracker,
        "core-reranker": run_core_reranker,
        "steer-activation": run_steer_activation,
    }
    metrics, records = dispatch[args.experiment](args, out_dir)
    metrics = {
        "backend": args.backend,
        "hardware_label": args.hardware_label or inferred_hardware_label(args),
        "hardware_kind": args.hardware_kind or inferred_hardware_kind(args),
        **metrics,
    }
    json_path, csv_path = write_outputs(args, out_dir, metrics, records)
    log_wandb(args, out_dir, metrics, records, json_path, csv_path)
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV: {csv_path}")
    for key, value in sorted(metrics.items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
