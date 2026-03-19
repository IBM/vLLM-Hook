import argparse
from pathlib import Path

import torch

from vllm_hook_plugins.run_utils import _normalize_qkv_cache


def load_cache(path_str: str):
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    cache = torch.load(path, map_location="cpu")
    return _normalize_qkv_cache(cache), path


def summarize(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a - b).abs().float()
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0)
    ).item()
    print(
        f"{name}: shape={tuple(a.shape)} "
        f"max_abs={diff.max().item():.6f} "
        f"mean_abs={diff.mean().item():.6f} "
        f"cosine={cosine:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", required=True, help="Metal qkv.pt artifact")
    parser.add_argument("--nonmetal", required=True, help="Non-metal qk.pt artifact")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    metal_cache, metal_path = load_cache(args.metal)
    nonmetal_cache, nonmetal_path = load_cache(args.nonmetal)

    module_name = f"model.layers.{args.layer}.self_attn.attn"
    metal_qk = metal_cache["qk_cache"].get(module_name)
    nonmetal_qk = nonmetal_cache["qk_cache"].get(module_name)
    if metal_qk is None:
        raise KeyError(f"{module_name} not found in Metal artifact {metal_path}")
    if nonmetal_qk is None:
        available = "\n".join(sorted(nonmetal_cache["qk_cache"].keys()))
        raise KeyError(
            f"{module_name} not found in non-metal artifact {nonmetal_path}\n"
            f"Available modules:\n{available}"
        )

    q_metal = metal_qk["q"][args.sample]
    q_nonmetal = nonmetal_qk["q"][args.sample]
    k_metal = metal_qk["k_all"][args.sample]
    k_nonmetal = nonmetal_qk["k_all"][args.sample]

    print(f"metal={metal_path}")
    print(f"nonmetal={nonmetal_path}")
    print(f"layer={args.layer} sample={args.sample}")

    if q_metal.shape != q_nonmetal.shape:
        print(f"q shape mismatch: metal={tuple(q_metal.shape)} nonmetal={tuple(q_nonmetal.shape)}")
    else:
        summarize("q", q_metal, q_nonmetal)

    if k_metal.shape != k_nonmetal.shape:
        print(f"k shape mismatch: metal={tuple(k_metal.shape)} nonmetal={tuple(k_nonmetal.shape)}")
    else:
        summarize("k", k_metal, k_nonmetal)


if __name__ == "__main__":
    main()
