#!/usr/bin/env python3
"""Stage-1 recurrent-depth adaptive exit demo (HF AdaptiveRaven path).

Wiring::

    model = AdaptiveRavenForCausalLM.from_pretrained(...)
    attach_recurrent_depth(model, rho=0.0)  # contraction; exact-match check
    # or A/B vs Huginn:
    # attach_recurrent_depth(model, raven_cfg=RavenAdapterConfig(baseline_criterion="latent-diff"))

Requires a Raven / Huginn / retrofitted checkpoint and transformers==4.51.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vllm_hook_plugins"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF repo or local Raven checkpoint")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--rho", type=float, default=0.0, help="0 → no exit (exact-match check)")
    p.add_argument(
        "--baseline",
        default=None,
        help="Huginn baseline instead of contraction: latent-diff|kl|entropy-diff|argmax-stability|none",
    )
    p.add_argument("--num-steps", type=int, default=None)
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer

    from model_adapters import (
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
    print("prompt:", args.prompt)
    print("rho:", args.rho, "baseline:", args.baseline)
    if exits is not None:
        print("exit_iteration [B,S]:", exits.tolist())
    if nonconv is not None:
        print("nonconverging count:", int(nonconv.sum().item()))
    next_id = out.logits[:, -1, :].argmax(-1)
    print("greedy next:", tok.decode(next_id.tolist()))


if __name__ == "__main__":
    main()
