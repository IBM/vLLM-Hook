#!/usr/bin/env python3
"""Stage-1 recurrent-depth adaptive exit demo (HF and/or vLLM).

HF::

    python examples/demo_recurrent_depth.py --model tomg-group-umd/huginn-0125 --rho 0.0

vLLM (registers only AdaptiveRavenForvLLM, not the full Hook plugin set)::

    python examples/demo_recurrent_depth.py --backend vllm --model tomg-group-umd/huginn-0125 --rho 0.0

Requires a Raven / Huginn / retrofitted checkpoint. HF path: transformers==4.51.0.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must precede any vLLM import (including HookLLM).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vllm_hook_plugins"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=("hf", "vllm", "both"),
        default="hf",
        help="Which runtime to exercise (HF or vLLM)",
    )
    p.add_argument(
        "--model",
        default="tomg-group-umd/huginn-0125",
        help="HF repo or local Raven checkpoint",
    )
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--rho", type=float, default=0.0, help="0 → no exit (exact-match check)")
    p.add_argument(
        "--baseline",
        default=None,
        help="HF only. Huginn baseline: latent-diff|kl|entropy-diff|argmax-stability|none",
    )
    p.add_argument("--num-steps", type=int, default=None, help="HF only. Override recurrence steps.")
    return p.parse_args()


def test_vllm(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    from model_adapters.vllm import ADAPTIVE_RAVEN_ARCH, register_adaptive_raven
    from vllm_hook_plugins import HookLLM

    register_adaptive_raven()

    llm = HookLLM(
        model=args.model,
        download_dir=str(ROOT / "cache"),
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        hf_overrides={
            "architectures": [ADAPTIVE_RAVEN_ARCH],
            "recurrent_depth": {"rho": args.rho, "min_steps": 1},
        },
    )
    out = llm.generate(
        [args.prompt],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=20),
    )
    print("backend: vllm")
    print("arch:", ADAPTIVE_RAVEN_ARCH)
    print("prompt:", args.prompt)
    print("rho:", args.rho)
    print(out[0].outputs[0].text)


def test_HF(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from model_adapters.hf import (
        AdaptiveRavenForCausalLM,
        HuginnDynamicCache,
        RavenAdapterConfig,
        block_geometry_from_config,
    )
    from vllm_hook_plugins.protocols.recurrent_depth import attach_recurrent_depth

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AdaptiveRavenForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

    r, prelude = block_geometry_from_config(model.config)
    # Upstream HuginnDynamicCache still uses the stock latest-m4 name; our
    # RowSliceCacheProxy reads R/prelude from config for congruent fill.
    print(f"geometry: prelude={prelude}, recurrent_block={r}; cache lookup=latest-m4 (upstream)")

    raven_cfg = RavenAdapterConfig(
        slice_decode=True,
        baseline_criterion=args.baseline,
        cache_lookup_strategy="latest-m4",
    )
    proto = attach_recurrent_depth(model, rho=args.rho, min_steps=1, raven_cfg=raven_cfg)

    ids = tok(args.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        ids = {k: v.cuda() for k, v in ids.items()}

    cache = HuginnDynamicCache(lookup_strategy="latest-m4")
    with torch.no_grad():
        out = model(
            input_ids=ids["input_ids"],
            use_cache=True,
            past_key_values=cache,
            num_steps=args.num_steps,
        )

    exits = proto.last_exit_iteration
    nonconv = proto.last_nonconverging
    print("backend: hf")
    print("prompt:", args.prompt)
    print("rho:", args.rho, "baseline:", args.baseline)
    if exits is not None:
        print("exit_iteration [B,S]:", exits.tolist())
    if nonconv is not None:
        print("nonconverging count:", int(nonconv.sum().item()))
    next_id = out.logits[:, -1, :].argmax(-1)
    print("greedy next:", tok.decode(next_id.tolist()))


if __name__ == "__main__":
    args = parse_args()
    if args.backend in ("hf", "both"):
        test_HF(args)
    if args.backend in ("vllm", "both"):
        test_vllm(args)
