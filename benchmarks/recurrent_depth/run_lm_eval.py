#!/usr/bin/env python3
"""Run lm-eval on Adaptive Raven (HF and/or vLLM; single config or Pareto sweep).

Examples
--------
vLLM fixed-depth curve (production path)::

    python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \\
        --tasks gsm8k --num-fewshot 5 --sweep-fixed 4,8,16,32

vLLM adaptive ρ curve::

    python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \\
        --tasks gsm8k --num-fewshot 5 --num-steps 32 \\
        --sweep-rho 0,0.001,0.01,0.02,0.05

Both arms in one invocation (fixed grid + rho grid)::

    python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \\
        --tasks gsm8k --num-fewshot 5 --num-steps 32 \\
        --sweep-fixed 4,8,16,32 --sweep-rho 0,0.001,0.01,0.02,0.05
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import lm_eval
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vllm_hook_plugins"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _parse_float_list(s: Optional[str]) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: Optional[str]) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="vllm",
        help="Inference stack (default: vllm — production / Hook path)",
    )
    p.add_argument("--model", default="tomg-group-umd/huginn-0125", help="HF checkpoint id or path")
    p.add_argument("--tasks", default="gsm8k", help="Comma-separated lm-eval task names (gsm8k,mmlu,…)")
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--batch-size", default="1")
    p.add_argument("--limit", type=float, default=None, help="Cap examples per task (e.g. 32 or 0.1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda", help="HF only")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM only")
    p.add_argument("--max-model-len", type=int, default=4096, help="vLLM only")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--log-samples", action="store_true")

    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=None, help="Recurrence cap / fixed depth")
    p.add_argument("--baseline", default=None, help="HF only: latent-diff|kl|…")

    p.add_argument("--sweep-fixed", default=None, help="e.g. 4,8,16,32 → rho=0, num_steps=r")
    p.add_argument("--sweep-rho", default=None, help="e.g. 0,0.01,0.02 with --num-steps as r_max")
    return p.parse_args()


def _task_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def _configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build run list. Fixed and rho sweeps may be combined (Pareto both arms)."""
    fixed = _parse_int_list(args.sweep_fixed)
    rhos = _parse_float_list(args.sweep_rho)
    cfgs: List[Dict[str, Any]] = []
    if fixed:
        for r in fixed:
            cfgs.append(
                {
                    "arm": "fixed",
                    "rho": 0.0,
                    "num_steps": r,
                    "min_steps": args.min_steps,
                    "baseline": args.baseline,
                }
            )
    if rhos:
        if args.num_steps is None:
            raise SystemExit("--sweep-rho requires --num-steps (r_max cap)")
        for rho in rhos:
            cfgs.append(
                {
                    "arm": "adaptive",
                    "rho": rho,
                    "num_steps": args.num_steps,
                    "min_steps": args.min_steps,
                    "baseline": args.baseline,
                }
            )
    if cfgs:
        return cfgs
    return [
        {
            "arm": "fixed" if args.rho == 0.0 else "adaptive",
            "rho": args.rho,
            "num_steps": args.num_steps,
            "min_steps": args.min_steps,
            "baseline": args.baseline,
        }
    ]


def _slug(cfg: Dict[str, Any], backend: str) -> str:
    parts = [backend, cfg.get("arm", "run"), f"rho{cfg['rho']}"]
    if cfg.get("num_steps") is not None:
        parts.append(f"r{cfg['num_steps']}")
    if cfg.get("baseline"):
        parts.append(f"base-{cfg['baseline']}")
    return "_".join(parts)


def _build_lm(args: argparse.Namespace, cfg: Dict[str, Any]):
    if args.backend == "hf":
        from raven_lm_eval import AdaptiveRavenLM

        return AdaptiveRavenLM(
            pretrained=args.model,
            rho=cfg["rho"],
            min_steps=cfg["min_steps"],
            num_steps=cfg["num_steps"],
            baseline=cfg["baseline"],
            dtype=args.dtype,
            device=args.device,
            batch_size=args.batch_size,
            trust_remote_code=True,
        )

    from raven_lm_eval_vllm import AdaptiveRavenVLLMLM

    return AdaptiveRavenVLLMLM(
        pretrained=args.model,
        rho=cfg["rho"],
        min_steps=cfg["min_steps"],
        num_steps=cfg["num_steps"],
        dtype=args.dtype,
        batch_size=args.batch_size,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


def run_one(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:

    lm = _build_lm(args, cfg)
    try:
        lm.reset_exit_stats()

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=_task_list(args.tasks),
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            log_samples=args.log_samples,
        )

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "model": args.model,
            "tasks": _task_list(args.tasks),
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "seed": args.seed,
            "config": cfg,
            "exit_stats": lm.exit_stats(),
            "results": results.get("results") if isinstance(results, dict) else results,
            "n-shot": results.get("n-shot") if isinstance(results, dict) else None,
        }
        if args.log_samples and isinstance(results, dict) and "samples" in results:
            payload["samples"] = results["samples"]

        return payload

    finally:
        # Memory freeing and vLLM core engine shutdown
        model = getattr(lm, "model", None)
        engine = getattr(model, "llm_engine", None)
        if engine is not None:
            if hasattr(engine, "reset_prefix_cache"):
                try:
                    engine.reset_prefix_cache()
                except Exception:
                    pass
            core = getattr(engine, "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                try:
                    core.shutdown()
                except Exception:
                    pass

        # Explicit garbage collection
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    try:
        import lm_eval  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            'lm_eval is not installed. Run: pip install "lm_eval[hf]" datasets\n'
            f"Original error: {e}"
        ) from e

    if args.output_dir is None:
        args.output_dir = _ROOT / "benchmarks" / "recurrent_depth" / "results" / args.backend
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for cfg in _configs(args):
        slug = _slug(cfg, args.backend)
        print(f"=== run {slug} ===", flush=True)
        payload = run_one(args, cfg)
        out = args.output_dir / f"{slug}.json"
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {out}", flush=True)
        print("exit_stats:", payload["exit_stats"], flush=True)
        summary.append(
            {
                "file": str(out),
                "backend": args.backend,
                "config": cfg,
                "exit_stats": payload["exit_stats"],
                "results": payload["results"],
            }
        )

    index = args.output_dir / "sweep_summary.json"
    index.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": args.backend,
                "model": args.model,
                "tasks": _task_list(args.tasks),
                "runs": summary,
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {index}", flush=True)


if __name__ == "__main__":
    main()
