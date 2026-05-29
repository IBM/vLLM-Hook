"""Remote payload for Colab-side minimal parity runs.

This file is intended to run inside a Colab VM through google-colab-cli. The
local orchestrator can prepend environment values before executing it remotely.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = ("hidden-states", "attn-tracker", "core-reranker", "steer-activation")
DEFAULT_GRANITE_QUANTIZED_MODEL = "RedHatAI/granite-3.1-2b-instruct-quantized.w4a16"
DEFAULT_MODELS = {
    "hidden-states": DEFAULT_GRANITE_QUANTIZED_MODEL,
    "attn-tracker": DEFAULT_GRANITE_QUANTIZED_MODEL,
    "core-reranker": "mistralai/Mistral-7B-Instruct-v0.3",
    "steer-activation": "microsoft/Phi-3-mini-4k-instruct",
}


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_shell(command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + command, flush=True)
    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=True,
        executable="/bin/bash",
        check=True,
    )


def env_value(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value if value is not None else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-url",
        default=env_value("VLLM_HOOK_REPO_URL", ""),
        help="Optional remote repo URL. Empty means reuse --workdir and never clone.",
    )
    parser.add_argument("--repo-branch", default=env_value("VLLM_HOOK_REPO_BRANCH", "vllm-hook-mlx"))
    parser.add_argument("--workdir", default=env_value("VLLM_HOOK_COLAB_WORKDIR", "/content/vLLM-Hook"))
    parser.add_argument("--benchmark-prefix", default=env_value("BENCHMARK_PREFIX", "minimal-parity"))
    parser.add_argument("--hidden-model", default=env_value("HIDDEN_STATES_MODEL", DEFAULT_MODELS["hidden-states"]))
    parser.add_argument("--attn-model", default=env_value("ATTN_TRACKER_MODEL", DEFAULT_MODELS["attn-tracker"]))
    parser.add_argument("--core-model", default=env_value("CORE_RERANKER_MODEL", DEFAULT_MODELS["core-reranker"]))
    parser.add_argument("--steer-model", default=env_value("STEER_ACTIVATION_MODEL", DEFAULT_MODELS["steer-activation"]))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=env_value("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=env_value("WANDB_PROJECT", "vllm-hook-platform-parity"))
    parser.add_argument("--wandb-entity", default=env_value("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-api-key", default=env_value("WANDB_API_KEY", ""))
    parser.add_argument("--wandb-secret-project", default=env_value("WANDB_SECRET_PROJECT", ""))
    parser.add_argument("--wandb-secret-name", default=env_value("WANDB_SECRET_NAME", ""))
    parser.add_argument("--wandb-secret-version", default=env_value("WANDB_SECRET_VERSION", "latest"))
    parser.add_argument("--hardware-label", default=env_value("VLLM_HOOK_HARDWARE_LABEL", "colab-gpu"))
    parser.add_argument("--hardware-kind", default=env_value("VLLM_HOOK_HARDWARE_KIND", "cuda"))
    parser.add_argument("--max-tokens", type=int, default=int(env_value("MAX_TOKENS", "2")))
    parser.add_argument("--temperature", type=float, default=float(env_value("TEMPERATURE", "0.0")))
    parser.add_argument("--top-p", type=float, default=float(env_value("TOP_P", "1.0")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(env_value("GPU_MEMORY_UTILIZATION", "0.8")),
    )
    parser.add_argument("--max-model-len", type=int, default=int(env_value("MAX_MODEL_LEN", "2048")))
    parser.add_argument("--dtype", default=env_value("DTYPE", "float16"))
    parser.add_argument(
        "--colab-install-vllm",
        default=env_value("COLAB_INSTALL_VLLM", ""),
        help=(
            "Optional extra pip spec installed on Colab for GPU vLLM runs. "
            "By default requirement.txt installs vLLM."
        ),
    )
    args = parser.parse_args()
    args.temperature = 0.0
    return args


def model_for(args: argparse.Namespace, experiment: str) -> str:
    return {
        "hidden-states": args.hidden_model,
        "attn-tracker": args.attn_model,
        "core-reranker": args.core_model,
        "steer-activation": args.steer_model,
    }[experiment]


def repo_remote_matches(repo_root: Path, expected_remote: str) -> bool:
    try:
        origin_url = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().removesuffix(".git")
    except Exception:
        return False
    return origin_url == expected_remote


def find_existing_repo_root(start_dir: Path, expected_remote: str) -> Path | None:
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / ".git").exists() and repo_remote_matches(candidate, expected_remote):
            return candidate
    return None


def prepare_repo(repo_url: str, branch: str, requested_workdir: Path) -> Path:
    if not repo_url:
        if requested_workdir.exists():
            print(f"Using existing repo at {requested_workdir}", flush=True)
            return requested_workdir
        raise FileNotFoundError(
            f"Repo URL was not provided and {requested_workdir} does not exist. "
            "Create/upload the repo in Colab first, or pass --repo-url explicitly."
        )

    expected_remote = repo_url.removesuffix(".git")
    existing = find_existing_repo_root(Path.cwd(), expected_remote)
    if existing is not None:
        print(f"Reusing existing repo at {existing}", flush=True)
        repo_root = existing
    else:
        repo_root = requested_workdir
        if not repo_root.exists():
            print(f"Cloning {repo_url} ({branch}) into {repo_root} ...", flush=True)
            run(["git", "clone", "--branch", branch, repo_url, str(repo_root)])
        elif not repo_remote_matches(repo_root, expected_remote):
            print(f"Remote mismatch under {repo_root}; replacing clone with {expected_remote}", flush=True)
            shutil.rmtree(repo_root)
            run(["git", "clone", "--branch", branch, repo_url, str(repo_root)])
        else:
            print(f"Reusing existing clone at {repo_root}", flush=True)

    run(["git", "-C", str(repo_root), "fetch", "origin", branch])
    run(["git", "-C", str(repo_root), "checkout", branch])
    run(["git", "-C", str(repo_root), "pull", "--ff-only", "origin", branch])
    return repo_root


def assert_cuda_runtime() -> None:
    try:
        import torch
    except Exception:
        torch = None
    has_cuda = bool(torch is not None and torch.cuda.is_available())
    has_cudart = importlib.util.find_spec("nvidia.cuda_runtime") is not None
    if not has_cuda and not has_cudart:
        raise RuntimeError(
            "This parity run requires a Colab GPU runtime with CUDA available. "
            "Choose a GPU runtime such as T4, then rerun from a fresh runtime."
        )


def read_secret_manager_secret(project: str, secret_name: str, version: str) -> str:
    try:
        from google.colab import auth

        auth.authenticate_user()
    except Exception as exc:
        print(f"Colab auth skipped or unavailable: {type(exc).__name__}: {exc}", flush=True)

    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    resource = f"projects/{project}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(request={"name": resource})
    return response.payload.data.decode("utf-8").strip()


def main() -> int:
    args = parse_args()
    repo_url = args.repo_url
    branch = args.repo_branch
    workdir = Path(args.workdir)
    benchmark_prefix = args.benchmark_prefix
    wandb_mode = args.wandb_mode
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity

    if shutil.which("git") is None:
        raise RuntimeError("git is unavailable in the Colab runtime.")

    workdir = prepare_repo(repo_url, branch, workdir)
    os.chdir(workdir)
    print(f"Changed working directory to {workdir}", flush=True)

    plugin_dir = workdir / "vllm_hook_plugins"
    req = workdir / "requirement.txt"
    if not plugin_dir.exists():
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")

    assert_cuda_runtime()

    # Keep setup as shell commands so Colab streams package-manager output.
    # Heavy installs can otherwise appear idle and trigger connection loss.
    run_shell(f"{sys.executable} -m pip install -U pip")
    if req.exists():
        run_shell(f"{sys.executable} -m pip install -r {req}")
    else:
        print("Warning: requirement.txt not found; skipping dependency install.", flush=True)
    run_shell(f"{sys.executable} -m pip install --force-reinstall 'protobuf>=5.29.6,<6.30'")
    run_shell(f"{sys.executable} -m pip install wandb weave pytest")
    if args.wandb_secret_name:
        run_shell(f"{sys.executable} -m pip install google-cloud-secret-manager")
    if args.colab_install_vllm:
        run_shell(f"{sys.executable} -m pip install {args.colab_install_vllm}")
    run_shell(f"{sys.executable} -m pip install -e {plugin_dir}")
    plugin_src = str(plugin_dir.resolve())
    if plugin_src not in sys.path:
        sys.path.insert(0, plugin_src)
    importlib.invalidate_caches()
    print(f"Plugin source: {plugin_src}", flush=True)
    print(f"Python exec  : {sys.executable}", flush=True)

    env = os.environ.copy()
    env["WANDB_MODE"] = wandb_mode
    env["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if args.wandb_secret_name:
        if not args.wandb_secret_project:
            raise ValueError("--wandb-secret-project is required with --wandb-secret-name")
        env["WANDB_API_KEY"] = read_secret_manager_secret(
            args.wandb_secret_project,
            args.wandb_secret_name,
            args.wandb_secret_version,
        )
    elif args.wandb_api_key:
        env["WANDB_API_KEY"] = args.wandb_api_key
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    runner = workdir / "tests" / "parity_tests" / "minimal_parity_benchmarks.py"
    for experiment in EXPERIMENTS:
        benchmark_id = f"{benchmark_prefix}-{experiment}"
        cmd = [
            sys.executable,
            str(runner),
            experiment,
            "--backend",
            "non-metal",
            "--benchmark-id",
            benchmark_id,
            "--model",
            model_for(args, experiment),
            "--wandb-mode",
            wandb_mode,
            "--wandb-project",
            wandb_project,
            "--hardware-label",
            args.hardware_label,
            "--hardware-kind",
            args.hardware_kind,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--dtype",
            args.dtype,
        ]
        if wandb_entity:
            cmd.extend(["--wandb-entity", wandb_entity])
        run(cmd, cwd=workdir, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
