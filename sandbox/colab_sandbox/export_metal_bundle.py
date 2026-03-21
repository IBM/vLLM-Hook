import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def latest_run_id(run_id_file: Path) -> str:
    run_ids = [line.strip() for line in run_id_file.read_text().splitlines() if line.strip()]
    if not run_ids:
        raise RuntimeError(f"No run IDs found in {run_id_file}")
    return run_ids[-1]


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", required=True, choices=["attntracker", "corer"])
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    project_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_json = Path(args.input_json).expanduser().resolve()
    if not input_json.exists():
        raise FileNotFoundError(input_json)

    if args.analyzer == "corer":
        runner = project_root / "examples" / "demo_corer.py"
        model = "ibm-granite/granite-4.0-micro"
    else:
        runner = project_root / "examples" / "demo_attntracker.py"
        model = "ibm-granite/granite-4.0-micro"

    env = os.environ.copy()
    env["VLLM_HOOK_BACKEND"] = "metal"

    subprocess.run(
        [
            "python",
            str(runner),
            "--debug-input-json",
            str(input_json),
        ],
        cwd=str(project_root),
        env=env,
        check=True,
    )

    hook_root = Path(os.path.expanduser("~/.cache/vllm-hook/_v1_qk_peeks"))
    run_id = latest_run_id(hook_root / "RUN_ID.txt")
    source_qkv = hook_root / run_id / "tp_rank_0" / "qkv.pt"
    if not source_qkv.exists():
        raise FileNotFoundError(source_qkv)

    bundle_qkv = output_dir / "qkv.pt"
    shutil.copy2(source_qkv, bundle_qkv)

    metadata = {
        "analyzer": args.analyzer,
        "model": model,
        "input_json": str(input_json),
        "git_commit": git_commit(project_root),
        "run_id": run_id,
        "artifact_source_path": str(source_qkv),
        "artifact_bundle_path": str(bundle_qkv),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"wrote_bundle={output_dir}")
    print(f"analyzer={args.analyzer} run_id={run_id}")
    print(f"qkv={bundle_qkv}")
    print(f"metadata={output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
