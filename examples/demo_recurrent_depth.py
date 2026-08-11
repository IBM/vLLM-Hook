#!/usr/bin/env python3
"""Stage-1 recurrent-depth adaptive exit demo (HF and/or vLLM).

Single-backend smoke::

    python examples/demo_recurrent_depth.py --model tomg-group-umd/huginn-0125 --rho 0.0
    python examples/demo_recurrent_depth.py --backend vllm --rho 0.02

Compare HF in-process vs AdaptiveRavenForvLLM (same prompts, greedy generate)::

    python examples/demo_recurrent_depth.py --backend both --rho 0.02 --max-tokens 16

Also useful: ``--rho 0`` (no early exit) vs ``--rho 0.02`` on one backend to
see adaptive depth savings within the same runtime.

Requires a Raven / Huginn / retrofitted checkpoint.

HF Raven modeling is validated against ``transformers==4.51.0`` (see
``docs/use_cases/RecurrentDepth.md``). The shared vLLM env typically needs
``transformers>=4.56`` (vLLM 0.22); this demo uses a TF5-tolerant HF forward
path so ``--backend both`` works without downgrading transformers. For a
strict HF oracle env, pin 4.51.0 separately.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# Must precede any vLLM import (including HookLLM).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vllm_hook_plugins"))

# Fixed set for HF ↔ vLLM A/B (override with --prompt for a single string).
DEFAULT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "In mathematics, the derivative of x^2 is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "The mitochondria is the powerhouse of the",
    "Write a one-sentence summary of photosynthesis:",
)


@dataclass
class PromptResult:
    prompt: str
    text: str
    latency_s: float
    num_tokens: int
    # Mean recurrence exit index over positions seen on the last forward (-1 = never).
    mean_exit_iter: Optional[float] = None
    # Steps actually executed in the last recurrence loop (vLLM); HF ≈ max over exits.
    steps_run: Optional[int] = None
    nonconverging: Optional[int] = None


@dataclass
class BackendReport:
    backend: str
    rho: float
    mean_recurrence: Optional[int] = None
    results: list[PromptResult] = field(default_factory=list)
    load_s: float = 0.0

    @property
    def total_latency_s(self) -> float:
        return sum(r.latency_s for r in self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.num_tokens for r in self.results)

    @property
    def tokens_per_s(self) -> float:
        t = self.total_latency_s
        return self.total_tokens / t if t > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=("hf", "vllm", "both"),
        default="hf",
        help="hf / vllm smoke, or both = timed comparison",
    )
    p.add_argument(
        "--model",
        default="tomg-group-umd/huginn-0125",
        help="HF repo or local Raven checkpoint",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Single prompt (default: built-in comparison set)",
    )
    p.add_argument("--rho", type=float, default=0.0, help="0 → no exit (exact-match check)")
    p.add_argument(
        "--baseline",
        default=None,
        help="HF only. Huginn baseline: latent-diff|kl|entropy-diff|argmax-stability|none",
    )
    p.add_argument("--num-steps", type=int, default=None, help="HF only. Override recurrence steps.")
    p.add_argument("--max-tokens", type=int, default=16, help="Greedy tokens to generate per prompt")
    p.add_argument("--warmup", type=int, default=1, help="Warmup generates discarded from timings")
    return p.parse_args()


def _prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        return [args.prompt]
    return list(DEFAULT_PROMPTS)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _exit_stats_from_tensor(exits: Any) -> tuple[Optional[float], Optional[int]]:
    """Return (mean exit iter over exited positions, max exit+1 as steps proxy)."""
    if exits is None:
        return None, None
    import torch

    if not isinstance(exits, torch.Tensor):
        return None, None
    t = exits.detach().float().view(-1)
    exited = t[t >= 0]
    mean = float(exited.mean().item()) if exited.numel() else None
    steps = int(t.max().item()) + 1 if t.numel() and float(t.max()) >= 0 else None
    return mean, steps


def _try_vllm_raven_model(llm: Any) -> Any:
    """Best-effort handle to AdaptiveRavenForvLLM / AdaptiveRavenModel (in-proc)."""
    candidates: list[Any] = []
    engine = getattr(llm, "llm_engine", None) or getattr(getattr(llm, "llm", None), "llm_engine", None)
    if engine is None:
        return None
    candidates.append(engine.model_executor.driver_worker.model_runner.model)
    # for path in (
    #     lambda e: e.model_executor.driver_worker.model_runner.model,
    #     lambda e: e.model_executor.driver_worker.worker.model_runner.model,
    #     lambda e: e.engine_core.model_executor.driver_worker.model_runner.model,
    # ):
    #     try:
    #         candidates.append(path(engine))
    #     except Exception:
    #         continue
    for m in candidates:
        if m is None:
            continue
        if hasattr(m, "controller") or hasattr(m, "last_exit_iteration"):
            return m
        inner = getattr(m, "model", None)
        if inner is not None and (
            hasattr(inner, "controller") or hasattr(inner, "last_exit_iteration")
        ):
            return inner
    return None


def _print_report(report: BackendReport) -> None:
    print()
    print("=" * 72)
    print(f"backend={report.backend}  rho={report.rho}  mean_recurrence={report.mean_recurrence}")
    print(f"load_s={report.load_s:.2f}  gen_s={report.total_latency_s:.3f}  "
          f"tokens={report.total_tokens}  tok/s={report.tokens_per_s:.2f}")
    print("-" * 72)
    for i, r in enumerate(report.results):
        exit_s = f"{r.mean_exit_iter:.2f}" if r.mean_exit_iter is not None else "n/a"
        steps_s = str(r.steps_run) if r.steps_run is not None else "n/a"
        preview = r.text.replace("\n", "\\n")[:48]
        print(
            f"[{i}] {r.latency_s:6.3f}s  {r.num_tokens:3d} tok  "
            f"exit̄={exit_s:>6}  steps={steps_s:>4}  | {preview!r}"
        )
    print("=" * 72)


def _print_comparison(hf: BackendReport, vllm: BackendReport) -> None:
    print()
    print("#" * 72)
    print("HF (in-process AdaptiveRavenForCausalLM)  vs  "
          "vLLM (AdaptiveRavenForvLLM)")
    print("#" * 72)
    print(f"{'metric':<28} {'HF':>14} {'vLLM':>14} {'Δ (vLLM/HF)':>14}")
    rows = [
        ("load_s", hf.load_s, vllm.load_s),
        ("generate_s (sum)", hf.total_latency_s, vllm.total_latency_s),
        ("tokens", float(hf.total_tokens), float(vllm.total_tokens)),
        ("tokens/s", hf.tokens_per_s, vllm.tokens_per_s),
    ]
    for name, a, b in rows:
        ratio = (b / a) if a else float("nan")
        print(f"{name:<28} {a:14.3f} {b:14.3f} {ratio:14.3f}")

    hf_exits = [r.mean_exit_iter for r in hf.results if r.mean_exit_iter is not None]
    v_exits = [r.mean_exit_iter for r in vllm.results if r.mean_exit_iter is not None]
    if hf_exits:
        print(f"{'mean exit iter (HF)':<28} {statistics.mean(hf_exits):14.3f}")
    if v_exits:
        print(f"{'mean exit iter (vLLM)':<28} {statistics.mean(v_exits):14.3f}")
    elif vllm.rho > 0:
        print(
            "note: vLLM exit̄ unavailable if the worker model handle is not "
            "exposed in-process; compare generate_s at rho=0 vs rho>0 instead."
        )
    print()
    print(
        "Protocol: both call RecurrentDepthWorker → RecurrentConvergenceAnalyzer "
        "→ ExitController. HF keeps [B,S,D]; vLLM flattens to [T,D] and "
        "unsqueeze(1) to [T,1,D] inside RecurrentStepController. HF may also "
        "row-slice decode + HuginnDynamicCache; vLLM skips Attn for inactive "
        "rows only via MLP gating + hidden freeze today."
    )
    print("#" * 72)


def run_hf(args: argparse.Namespace, prompts: Sequence[str]) -> BackendReport:
    import torch
    from transformers import AutoTokenizer

    from model_adapters.hf import (
        AdaptiveRavenForCausalLM,
        RavenAdapterConfig,
        block_geometry_from_config,
    )
    from vllm_hook_plugins.protocols.recurrent_depth import attach_recurrent_depth

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AdaptiveRavenForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval()
    load_s = time.perf_counter() - t0

    r, prelude = block_geometry_from_config(model.config)
    mean_rec = int(getattr(model.config, "mean_recurrence", -1))
    print(f"[hf] geometry prelude={prelude} R={r} mean_recurrence={mean_rec}")

    raven_cfg = RavenAdapterConfig(
        slice_decode=True,
        baseline_criterion=args.baseline,
        cache_lookup_strategy="latest-m4",
    )
    proto = attach_recurrent_depth(model, rho=args.rho, min_steps=1, raven_cfg=raven_cfg)

    report = BackendReport(
        backend="hf", rho=args.rho, mean_recurrence=mean_rec, load_s=load_s
    )

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if args.num_steps is not None:
        gen_kwargs["num_steps"] = args.num_steps

    warm = prompts[0]
    for _ in range(max(0, args.warmup)):
        ids = tok(warm, return_tensors="pt")
        if torch.cuda.is_available():
            ids = {k: v.cuda() for k, v in ids.items()}
        with torch.no_grad():
            model.generate(**ids, **gen_kwargs)
        _sync_cuda()

    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            ids = {k: v.cuda() for k, v in ids.items()}
        prompt_len = int(ids["input_ids"].shape[-1])

        _sync_cuda()
        t1 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(**ids, **gen_kwargs)
        _sync_cuda()
        latency = time.perf_counter() - t1

        new_ids = out_ids[0, prompt_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        mean_exit, steps = _exit_stats_from_tensor(proto.last_exit_iteration)
        nonconv = proto.last_nonconverging
        nonconv_n = int(nonconv.sum().item()) if nonconv is not None else None

        report.results.append(
            PromptResult(
                prompt=prompt,
                text=text,
                latency_s=latency,
                num_tokens=int(new_ids.numel()),
                mean_exit_iter=mean_exit,
                steps_run=steps,
                nonconverging=nonconv_n,
            )
        )

    _print_report(report)
    del model
    _free_cuda()
    return report


def run_vllm(args: argparse.Namespace, prompts: Sequence[str]) -> BackendReport:
    from vllm import SamplingParams

    from model_adapters.vllm import ADAPTIVE_RAVEN_ARCH, register_adaptive_raven
    from vllm_hook_plugins import HookLLM

    register_adaptive_raven()

    t0 = time.perf_counter()
    llm = HookLLM(
        model=args.model,
        download_dir=str(ROOT / "cache"),
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=max(128, args.max_tokens + 64),
        gpu_memory_utilization=0.8,
        hf_overrides={
            "architectures": [ADAPTIVE_RAVEN_ARCH],
            "recurrent_depth": {"rho": args.rho, "min_steps": 1},
        },
    )
    load_s = time.perf_counter() - t0

    mean_rec = None
    raven = _try_vllm_raven_model(llm)
    if raven is not None:
        mean_rec = int(getattr(getattr(raven, "config", None), "mean_recurrence", -1))

    print(f"[vllm] arch={ADAPTIVE_RAVEN_ARCH} mean_recurrence={mean_rec}")

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    report = BackendReport(
        backend="vllm", rho=args.rho, mean_recurrence=mean_rec, load_s=load_s
    )

    for _ in range(max(0, args.warmup)):
        llm.generate([prompts[0]], sampling_params=sp)
        _sync_cuda()

    for prompt in prompts:
        _sync_cuda()
        t1 = time.perf_counter()
        outs = llm.generate([prompt], sampling_params=sp)
        _sync_cuda()
        latency = time.perf_counter() - t1

        out = outs[0].outputs[0]
        text = out.text
        n_tok = len(out.token_ids)

        mean_exit, steps = None, None
        nonconv_n = None
        raven = _try_vllm_raven_model(llm)
        if raven is not None:
            mean_exit, steps = _exit_stats_from_tensor(
                getattr(raven, "last_exit_iteration", None)
            )
            steps = getattr(raven, "last_recurrence_steps_run", steps)
            nc = getattr(raven, "last_nonconverging", None)
            if nc is not None:
                nonconv_n = int(nc.sum().item())

        report.results.append(
            PromptResult(
                prompt=prompt,
                text=text,
                latency_s=latency,
                num_tokens=n_tok,
                mean_exit_iter=mean_exit,
                steps_run=steps,
                nonconverging=nonconv_n,
            )
        )

    _print_report(report)
    del llm
    _free_cuda()
    return report


def test_vllm(args: argparse.Namespace) -> None:
    run_vllm(args, _prompts(args))


def test_HF(args: argparse.Namespace) -> None:
    run_hf(args, _prompts(args))


if __name__ == "__main__":
    args = parse_args()
    prompts = _prompts(args)
    if args.backend == "hf":
        test_HF(args)
    elif args.backend == "vllm":
        test_vllm(args)
    else:
        hf_report = run_hf(args, prompts)
        vllm_report = run_vllm(args, prompts)
        _print_comparison(hf_report, vllm_report)
