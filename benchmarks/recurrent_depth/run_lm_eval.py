#!/usr/bin/env python3
"""Run lm-eval on Adaptive Raven (fixed depth, calibrated policy, or legacy ρ sweep).

Examples::

    # Fixed-depth Pareto arm
    python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \\
        --tasks gsm8k --num-fewshot 5 --sweep-fixed 4,8,16,32

    # Calibrated adaptive policy (after analyze_signals.py)
    python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \\
        --tasks gsm8k --num-fewshot 5 --num-steps 32 \\
        --policy-manifest benchmarks/recurrent_depth/calibration/huginn_gsm8k_r32.json

    # Full-depth trajectory for offline calibration (use --limit 64–256)
    python benchmarks/recurrent_depth/run_lm_eval.py --backend vllm \\
        --tasks gsm8k --num-fewshot 5 --num-steps 32 --rho 0 --limit 64 \\
        --capture-trajectory --prediction-metrics
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import lm_eval
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vllm_hook_plugins"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# pyright: reportUnknownVariableType=false
# pyright: reportCallIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

_T = TypeVar("_T")


def _parse_csv(s: Optional[str], cast: Callable[[str], _T]) -> List[_T]:
    if not s:
        return []
    return [cast(part.strip()) for part in s.split(",") if part.strip()]


def _parse_condition(spec: str) -> Dict[str, Any]:
    try:
        readout, threshold = spec.rsplit("=", 1)
        return {"readout": readout.strip(), "threshold": float(threshold)}
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("condition must be READOUT=THRESHOLD") from exc


def _task_list(s: str) -> List[str]:
    return _parse_csv(s, str)


def _resolve_policy(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.policy_manifest is not None:
        if args.condition:
            raise SystemExit("--policy-manifest cannot be combined with --condition")
        manifest = json.loads(args.policy_manifest.read_text())
        policy = manifest.get("policy")
        if not isinstance(policy, dict):
            raise SystemExit(f"{args.policy_manifest} has no 'policy' block")
        return policy
    if not args.condition:
        return None
    return {
        "enabled": True,
        "conditions": args.condition,
        "combine": args.condition_combine,
        "confirmation_conditions": args.confirmation_condition,
        "confirmation_combine": args.confirmation_combine,
        "min_steps": args.min_steps,
        "patience": args.patience,
    }


def _run_cfg(
    args: argparse.Namespace,
    *,
    arm: str,
    rho: float = 0.0,
    num_steps: Optional[int] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "arm": arm,
        "rho": rho,
        "num_steps": num_steps,
        "min_steps": args.min_steps,
        "baseline": args.baseline,
        "policy": policy,
    }


def _configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    policy = _resolve_policy(args)
    fixed = _parse_csv(args.sweep_fixed, int)
    rhos = _parse_csv(args.sweep_rho, float)
    if policy and rhos:
        raise SystemExit("an explicit exit policy cannot be combined with --sweep-rho")

    cfgs = [_run_cfg(args, arm="fixed", num_steps=r) for r in fixed]
    if rhos:
        if args.num_steps is None:
            raise SystemExit("--sweep-rho requires --num-steps (r_max cap)")
        cfgs.extend(
            _run_cfg(args, arm="adaptive", rho=rho, num_steps=args.num_steps)
            for rho in rhos
        )
    if policy:
        if args.num_steps is None:
            raise SystemExit("an explicit exit policy requires --num-steps (r_max cap)")
        cfgs.append(_run_cfg(args, arm="adaptive", num_steps=args.num_steps, policy=policy))

    if cfgs:
        return cfgs
    return [
        _run_cfg(
            args,
            arm="adaptive" if policy or args.rho else "fixed",
            rho=args.rho,
            num_steps=args.num_steps,
            policy=policy,
        )
    ]


def _slug(cfg: Dict[str, Any], backend: str) -> str:
    parts = [backend, cfg["arm"], f"rho{cfg['rho']}"]
    if cfg.get("num_steps") is not None:
        parts.append(f"r{cfg['num_steps']}")
    if cfg.get("baseline"):
        parts.append(f"base-{cfg['baseline']}")
    if policy := cfg.get("policy"):
        parts.append(
            "policy-"
            + "-".join(f"{c['readout']}{c['threshold']}" for c in policy["conditions"])
        )
    return "_".join(parts)


def _build_lm(args: argparse.Namespace, cfg: Dict[str, Any]):
    if args.backend == "hf":
        if cfg.get("policy") is not None:
            raise SystemExit("composite policies require --backend vllm")
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
        capture_trajectory=args.capture_trajectory,
        compute_prediction_metrics=args.prediction_metrics,
        policy=cfg["policy"],
        dtype=args.dtype,
        batch_size=args.batch_size,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


def _shutdown_vllm(lm) -> None:
    model = getattr(lm, "model", None)
    engine = getattr(model, "llm_engine", None)
    if engine is None:
        return
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


def run_one(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    *,
    trajectory_path: Optional[Path] = None,
) -> Dict[str, Any]:
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

        exit_stats = lm.exit_stats()
        trajectory_summary = None
        if args.capture_trajectory and trajectory_path is not None:
            from trajectory_io import write_trajectory_parquet

            trajectory_summary = write_trajectory_parquet(
                lm.take_trajectory_samples() or [],
                trajectory_path,
            )
            exit_stats["trajectory"] = {
                k: trajectory_summary[k]
                for k in ("path", "n_rows", "n_forwards", "n_decode_rows", "n_prediction_rows")
            }

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "model": args.model,
            "tasks": _task_list(args.tasks),
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "seed": args.seed,
            "config": cfg,
            "exit_stats": exit_stats,
            "results": results.get("results") if isinstance(results, dict) else results,
            "n-shot": results.get("n-shot") if isinstance(results, dict) else None,
        }
        if trajectory_summary is not None:
            payload["trajectory"] = trajectory_summary
        if args.log_samples and isinstance(results, dict) and "samples" in results:
            payload["samples"] = results["samples"]
        return payload
    finally:
        _shutdown_vllm(lm)
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("hf", "vllm"), default="vllm")
    p.add_argument("--model", default="tomg-group-umd/huginn-0125")
    p.add_argument("--tasks", default="gsm8k")
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--batch-size", default="1")
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda", help="HF only")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM only")
    p.add_argument("--max-model-len", type=int, default=4096, help="vLLM only")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--log-samples", action="store_true")

    p.add_argument("--rho", type=float, default=0.0, help="legacy contraction threshold (single run)")
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=None, help="Recurrence cap / fixed depth")
    p.add_argument("--baseline", default=None, help="HF only")
    p.add_argument("--capture-trajectory", action="store_true")
    p.add_argument("--prediction-metrics", action="store_true")
    p.add_argument("--condition", action="append", type=_parse_condition, default=[])
    p.add_argument("--confirmation-condition", action="append", type=_parse_condition, default=[])
    p.add_argument("--condition-combine", choices=("all", "any"), default="all")
    p.add_argument("--confirmation-combine", choices=("all", "any"), default="all")
    p.add_argument("--patience", type=int, default=1)
    p.add_argument(
        "--policy-manifest",
        type=Path,
        default=None,
        help="Frozen CalibrationManifest JSON from analyze_signals.py",
    )

    p.add_argument("--sweep-fixed", default=None, help="e.g. 4,8,16,32")
    p.add_argument(
        "--sweep-rho",
        default=None,
        help="legacy contraction ρ grid; prefer --policy-manifest after calibration",
    )
    return p.parse_args()


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
    traj_dir = args.output_dir / "trajectories"
    if args.capture_trajectory:
        traj_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for cfg in _configs(args):
        slug = _slug(cfg, args.backend)
        print(f"=== run {slug} ===", flush=True)
        traj_path = (traj_dir / f"{slug}.parquet") if args.capture_trajectory else None
        payload = run_one(args, cfg, trajectory_path=traj_path)
        out = args.output_dir / f"{slug}.json"
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {out}", flush=True)
        if traj_path and payload.get("trajectory"):
            print(f"wrote {traj_path} ({payload['trajectory']['n_rows']} rows)", flush=True)
        summary.append(
            {
                "file": str(out),
                "backend": args.backend,
                "config": cfg,
                "exit_stats": payload["exit_stats"],
                "trajectory": payload.get("trajectory"),
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
