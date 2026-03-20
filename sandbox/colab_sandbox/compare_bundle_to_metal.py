import argparse
import json
import subprocess
import sys
from pathlib import Path


def flatten_layer_heads(layer_to_heads: dict[str, list[int]] | dict[int, list[int]]) -> str:
    parts = []
    for layer, heads in sorted(((int(k), v) for k, v in layer_to_heads.items()), key=lambda x: x[0]):
        parts.append(f"{layer}:{','.join(str(head) for head in heads)}")
    return ";".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", required=True, help="Path to Metal qkv.pt or containing run dir")
    parser.add_argument("--bundle-dir", required=True, help="Exported Colab bundle directory")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-rope-offset", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    analyzer = metadata["analyzer"]
    qk_path = bundle_dir / "qk.pt"
    model = metadata["model"]
    layer_heads = flatten_layer_heads(metadata["layer_to_heads"])

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "analyzer_score_probe.py"),
        "--metal",
        args.metal,
        "--nonmetal",
        str(qk_path),
        "--model",
        model,
        "--layer-heads",
        layer_heads,
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
