import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch


def flatten_layer_heads(layer_to_heads: dict[str, list[int]] | dict[int, list[int]]) -> str:
    parts = []
    for layer, heads in sorted(((int(k), v) for k, v in layer_to_heads.items()), key=lambda x: x[0]):
        parts.append(f"{layer}:{','.join(str(head) for head in heads)}")
    return ";".join(parts)


def resolve_default_metal_path(project_root: Path, analyzer: str) -> Path:
    bundle_dir = project_root / "sandbox" / "colab_sandbox" / "output" / f"metal_bundle_{analyzer}"
    qkv = bundle_dir / "qkv.pt"
    if not qkv.exists():
        raise FileNotFoundError(
            f"No default metal bundle found at {qkv}. "
            f"Run sandbox/colab_sandbox/export_metal_bundle.py first."
        )
    return qkv


def available_metal_layers(metal_path: Path) -> set[int]:
    cache = torch.load(metal_path, map_location="cpu")
    qkv = cache.get("qkv_cache", {})
    layers = set()
    for proj_data in qkv.values():
        layers.add(int(proj_data["layer_num"]))
    return layers


def filter_layer_heads_to_metal(
    layer_to_heads: dict[str, list[int]] | dict[int, list[int]],
    metal_layers: set[int],
) -> dict[int, list[int]]:
    filtered: dict[int, list[int]] = {}
    for layer, heads in layer_to_heads.items():
        layer_idx = int(layer)
        if layer_idx in metal_layers:
            filtered[layer_idx] = heads
    if not filtered:
        raise ValueError(
            f"No configured layers overlap with available metal layers: {sorted(metal_layers)}"
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", default=None, help="Path to Metal qkv.pt or containing run dir")
    parser.add_argument("--bundle-dir", required=True, help="Exported Colab bundle directory")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-rope-offset", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    analyzer = metadata["analyzer"]
    project_root = Path(__file__).resolve().parents[2]
    metal_path = (
        Path(args.metal).expanduser().resolve()
        if args.metal is not None
        else resolve_default_metal_path(project_root, analyzer)
    )
    qk_path = bundle_dir / "qk.pt"
    model = metadata["model"]
    layer_heads = filter_layer_heads_to_metal(
        metadata["layer_to_heads"],
        available_metal_layers(metal_path),
    )

    cmd = [
        sys.executable,
        str(project_root / "sandbox" / "scripts" / "analyzer_score_probe.py"),
        "--metal",
        str(metal_path),
        "--nonmetal",
        str(qk_path),
        "--model",
        model,
        "--layer-heads",
        flatten_layer_heads(layer_heads),
        "--sample",
        str(args.sample),
        "--max-rope-offset",
        str(args.max_rope_offset),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])

    attn_spec = bundle_dir / "attn_spec.json"
    corer_spec = bundle_dir / "corer_spec.json"
    if analyzer == "attntracker" and attn_spec.exists():
        cmd.extend(["--attn-spec", str(attn_spec)])
    if analyzer == "corer" and corer_spec.exists():
        cmd.extend(["--corer-spec", str(corer_spec)])

    print("running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
