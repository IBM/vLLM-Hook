import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb


def resolve_artifact(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_file():
        return path

    candidates = [
        *path.rglob("qkv.pt"),
        *path.rglob("qk.pt"),
    ]
    if not candidates:
        raise FileNotFoundError(f"No qkv.pt or qk.pt found under {path}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple artifacts found under {path}; pass the exact file path.\n"
            + "\n".join(str(p) for p in candidates)
        )
    return candidates[0]


def find_rotary_module(model):
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    if hasattr(model, "rotary_emb"):
        return model.rotary_emb
    raise AttributeError("Could not locate rotary embedding module on model")


def get_num_heads(attn, config) -> int:
    for attr in ("num_heads", "n_heads", "heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    for attr in ("num_attention_heads",):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("Could not determine number of attention heads")


def get_num_kv_heads(attn, config) -> int:
    for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    for attr in ("num_key_value_heads",):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("Could not determine number of key/value heads")


def flatten_attn(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)


def summarize_diff(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    diff = (ref - got).abs().float()
    ref_f = ref.float().reshape(-1)
    got_f = got.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        ref_f.unsqueeze(0), got_f.unsqueeze(0)
    ).item()
    print(
        f"{name}: shape={tuple(ref.shape)} "
        f"max_abs={diff.max().item():.6f} "
        f"mean_abs={diff.mean().item():.6f} "
        f"cosine={cosine:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Path to qkv.pt or its run dir")
    parser.add_argument("--model", required=True, help="HF model id or local model path")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    artifact = resolve_artifact(args.artifact)
    cache = torch.load(artifact, map_location="cpu")

    x_key = f"model.layers.{args.layer}.self_attn.attn.x"
    q_key = f"model.layers.{args.layer}.self_attn.attn.q"
    k_key = f"model.layers.{args.layer}.self_attn.attn.k"

    if "qkv_cache" not in cache:
        raise ValueError(f"{artifact} does not contain Metal qkv_cache")

    qkv_cache = cache["qkv_cache"]
    if x_key not in qkv_cache or q_key not in qkv_cache or k_key not in qkv_cache:
        raise KeyError(
            f"Layer {args.layer} not found. Available keys:\n"
            + "\n".join(sorted(qkv_cache.keys()))
        )

    x = qkv_cache[x_key]["tokens"][args.sample].unsqueeze(0)
    q_metal = qkv_cache[q_key]["tokens"][args.sample].unsqueeze(0)
    k_metal = qkv_cache[k_key]["tokens"][args.sample].unsqueeze(0)

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()

    attn = model.model.layers[args.layer].self_attn
    rotary_emb = find_rotary_module(model)

    x = x.to(dtype)
    head_dim = attn.head_dim
    num_heads = get_num_heads(attn, model.config)
    num_kv_heads = get_num_kv_heads(attn, model.config)
    seq_len = x.shape[1]

    q_ref = attn.q_proj(x).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k_ref = attn.k_proj(x).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(q_ref, position_ids)
    q_ref, k_ref = apply_rotary_pos_emb(q_ref, k_ref, cos, sin)

    q_ref = flatten_attn(q_ref).cpu()
    k_ref = flatten_attn(k_ref).cpu()

    print(f"artifact={artifact}")
    print(f"layer={args.layer} sample={args.sample} seq_len={seq_len}")
    summarize_diff("q", q_ref, q_metal)
    summarize_diff("k", k_ref, k_metal)


if __name__ == "__main__":
    main()
