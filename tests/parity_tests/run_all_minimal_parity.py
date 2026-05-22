"""Run all minimal parity experiments on local Metal and Colab GPU.

Requires google-colab-cli on the local machine:
  uv tool install git+https://github.com/googlecolab/google-colab-cli

The script runs local Metal experiments directly, then uses `colab run` to
execute matching non-Metal experiments on a fresh Colab VM. Matching W&B groups
are created by using the same benchmark prefix and experiment name.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = ("attn-tracker", "core-reranker", "steer-activation")
DEFAULT_MODELS = {
    "attn-tracker": "Qwen/Qwen2-1.5B-Instruct",
    "core-reranker": "mistralai/Mistral-7B-Instruct-v0.3",
    "steer-activation": "microsoft/Phi-3-mini-4k-instruct",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = Path(__file__).resolve().parent
REMOTE_TEMPLATE = PARITY_DIR / "colab_minimal_parity_remote.py"
LOCAL_RUNNER = PARITY_DIR / "minimal_parity_benchmarks.py"
LOCAL_WANDB_CONFIG = PARITY_DIR / "local_wandb_config.py"
DEFAULT_LOCAL_PYTHON = (
    Path("/Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/bin/python")
)
LOCAL_COLAB_CLI = Path("/Users/timothyburley/opensource/google-colab-cli")


def load_local_wandb_config() -> dict[str, str]:
    if not LOCAL_WANDB_CONFIG.exists():
        return {}
    spec = importlib.util.spec_from_file_location("local_wandb_config", LOCAL_WANDB_CONFIG)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {}
    placeholders = {"paste_your_wandb_key_here", "your_key_here", ""}
    for name in (
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_SECRET_PROJECT",
        "WANDB_SECRET_NAME",
        "WANDB_SECRET_VERSION",
    ):
        value = getattr(module, name, "")
        if str(value) not in placeholders and "paste_your" not in str(value):
            values[name] = str(value)
    return values


def clean_wandb_key(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().strip("\"'")
    if value in {"your_key_here", "paste_your_wandb_key_here"}:
        return ""
    return value


def parse_args() -> argparse.Namespace:
    local_config = load_local_wandb_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-prefix",
        default=os.environ.get("BENCHMARK_PREFIX", "minimal-parity"),
        help="Prefix used for W&B groups: <prefix>-<experiment>.",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument(
        "--wandb-project",
        default=local_config.get(
            "WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "vllm-hook-platform-parity")
        ),
        help=(
            "Fallback/requested W&B project value. Benchmark runs are logged "
            "to app-specific projects: attntracker, corereranker, steering."
        ),
    )
    parser.add_argument(
        "--wandb-entity",
        default=local_config.get("WANDB_ENTITY", os.environ.get("WANDB_ENTITY", "")),
    )
    parser.add_argument(
        "--wandb-api-key",
        default=clean_wandb_key(
            local_config.get("WANDB_API_KEY", os.environ.get("WANDB_API_KEY", ""))
        ),
    )
    parser.add_argument(
        "--prompt-wandb-key",
        action="store_true",
        default=os.environ.get("WANDB_PROMPT_KEY", "0") == "1",
        help=(
            "Prompt for the W&B API key at runtime. This overrides any key "
            "loaded from local config or WANDB_API_KEY for Colab runs."
        ),
    )
    parser.add_argument(
        "--wandb-secret-project",
        default=local_config.get("WANDB_SECRET_PROJECT", os.environ.get("WANDB_SECRET_PROJECT", "")),
        help="Google Cloud project containing the W&B API key secret for Colab runs.",
    )
    parser.add_argument(
        "--wandb-secret-name",
        default=local_config.get("WANDB_SECRET_NAME", os.environ.get("WANDB_SECRET_NAME", "")),
        help="Secret Manager secret name containing the W&B API key for Colab runs.",
    )
    parser.add_argument(
        "--wandb-secret-version",
        default=local_config.get("WANDB_SECRET_VERSION", os.environ.get("WANDB_SECRET_VERSION", "latest")),
        help="Secret Manager version for --wandb-secret-name.",
    )
    parser.add_argument(
        "--repo-url",
        default=os.environ.get("VLLM_HOOK_REPO_URL", "https://github.com/tburleyinfo/vLLM-Hook.git"),
        help=(
            "Remote repo URL for Colab. Pass '' to reuse an existing "
            "/content/vLLM-Hook instead of cloning."
        ),
    )
    parser.add_argument("--repo-branch", default=os.environ.get("VLLM_HOOK_REPO_BRANCH", current_branch()))
    parser.add_argument("--session", default=os.environ.get("COLAB_SESSION", "vllm-hook-gpu"))
    parser.add_argument(
        "--colab-auth",
        choices=("oauth2", "adc"),
        default=os.environ.get("COLAB_AUTH", "oauth2"),
        help="Auth strategy passed to google-colab-cli.",
    )
    parser.add_argument(
        "--colab-client-oauth-config",
        default=os.environ.get(
            "COLAB_CLIENT_OAUTH_CONFIG",
            str(Path.home() / ".colab-cli-oauth-config.json"),
        ),
        help="OAuth client config JSON used when --colab-auth oauth2.",
    )
    parser.add_argument(
        "--colab-cli-source",
        default=os.environ.get(
            "COLAB_CLI_SOURCE",
            str(LOCAL_COLAB_CLI) if LOCAL_COLAB_CLI.exists() else "git+https://github.com/googlecolab/google-colab-cli",
        ),
        help="Local path or pip spec used to install google-colab-cli if `colab` is missing.",
    )
    parser.add_argument("--gpu", default=os.environ.get("COLAB_GPU", "T4"), choices=("T4", "L4", "A100", "H100"))
    parser.add_argument(
        "--local-python",
        default=os.environ.get(
            "VLLM_HOOK_LOCAL_PYTHON",
            str(DEFAULT_LOCAL_PYTHON if DEFAULT_LOCAL_PYTHON.exists() else Path(sys.executable)),
        ),
        help="Python executable for local Metal runs.",
    )
    parser.add_argument(
        "--colab-install-vllm",
        default=os.environ.get("COLAB_INSTALL_VLLM", ""),
        help=(
            "Optional extra pip spec installed on Colab for GPU vLLM runs. "
            "By default requirement.txt installs vLLM."
        ),
    )
    parser.add_argument("--local-hardware-label", default=os.environ.get("LOCAL_HARDWARE_LABEL", "apple-metal"))
    parser.add_argument("--local-hardware-kind", default=os.environ.get("LOCAL_HARDWARE_KIND", "metal"))
    parser.add_argument("--attn-model", default=os.environ.get("ATTN_TRACKER_MODEL", DEFAULT_MODELS["attn-tracker"]))
    parser.add_argument("--core-model", default=os.environ.get("CORE_RERANKER_MODEL", DEFAULT_MODELS["core-reranker"]))
    parser.add_argument("--steer-model", default=os.environ.get("STEER_ACTIVATION_MODEL", DEFAULT_MODELS["steer-activation"]))
    parser.add_argument("--skip-local", action="store_true", help="Only run Colab non-Metal experiments.")
    parser.add_argument("--skip-colab", action="store_true", help="Only run local Metal experiments.")
    parser.add_argument(
        "--keep-colab",
        action="store_true",
        default=os.environ.get("COLAB_KEEP", "1") != "0",
        help="Pass --keep to colab run. Defaults on so setup disconnects do not erase the VM.",
    )
    parser.add_argument(
        "--no-keep-colab",
        action="store_false",
        dest="keep_colab",
        help="Stop the Colab session after the remote script finishes.",
    )
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    args.temperature = 0.0
    return args


def model_for(args: argparse.Namespace, experiment: str) -> str:
    return {
        "attn-tracker": args.attn_model,
        "core-reranker": args.core_model,
        "steer-activation": args.steer_model,
    }[experiment]


def current_branch() -> str:
    try:
        completed = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "vllm-hook-mlx"
    except Exception:
        return "vllm-hook-mlx"


def display_command(cmd: list[str]) -> str:
    redacted = []
    redact_next = False
    for part in cmd:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(part)
        if part in {"--wandb-api-key", "--api-key"}:
            redact_next = True
    return " ".join(redacted)


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + display_command(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if completed.returncode:
        raise SystemExit(
            f"Command failed with exit status {completed.returncode}: {display_command(cmd)}"
        )


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def install_local_package(python: str, package: str) -> None:
    run([python, "-m", "pip", "install", package], cwd=PROJECT_ROOT)


def ensure_colab_cli(args: argparse.Namespace) -> None:
    if shutil.which("colab") is None:
        if command_exists("uv"):
            run(["uv", "tool", "install", args.colab_cli_source])
        else:
            install_local_package(sys.executable, args.colab_cli_source)
    if shutil.which("colab") is None:
        raise SystemExit("Failed to install or locate `colab` CLI.")


def preflight_colab_auth(args: argparse.Namespace) -> None:
    if args.skip_colab:
        return
    if args.colab_auth != "oauth2":
        return
    path = Path(args.colab_client_oauth_config).expanduser()
    if path.exists():
        return
    raise SystemExit(
        "Colab OAuth2 client config is missing.\n\n"
        f"Expected: {path}\n\n"
        "Fix one of these ways:\n"
        "  1. Put a Google OAuth desktop-client JSON at that path, then rerun.\n"
        "  2. Pass --colab-client-oauth-config /path/to/client_secret.json.\n"
        "  3. Use ADC instead: run `gcloud auth application-default login "
        "--scopes=https://www.googleapis.com/auth/cloud-platform,"
        "https://www.googleapis.com/auth/colaboratory,"
        "https://www.googleapis.com/auth/drive.file` then rerun with "
        "--colab-auth adc.\n"
    )


def python_can_import(python: str, module: str) -> bool:
    probe = subprocess.run(
        [python, "-c", f"import {module}"],
        cwd="/private/tmp",
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def ensure_local_python_packages(local_python: str, wandb_mode: str) -> None:
    if wandb_mode == "disabled":
        return
    for module, package in (("wandb", "wandb"), ("weave", "weave")):
        if not python_can_import(local_python, module):
            install_local_package(local_python, package)


def ensure_local_wandb(local_python: str, wandb_mode: str) -> None:
    ensure_local_python_packages(local_python, wandb_mode)


def run_local(args: argparse.Namespace) -> None:
    ensure_local_wandb(args.local_python, args.wandb_mode)
    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode
    env["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_api_key:
        env["WANDB_API_KEY"] = args.wandb_api_key

    for experiment in EXPERIMENTS:
        benchmark_id = f"{args.benchmark_prefix}-{experiment}"
        cmd = [
            args.local_python,
            str(LOCAL_RUNNER),
            experiment,
            "--backend",
            "metal",
            "--benchmark-id",
            benchmark_id,
            "--model",
            model_for(args, experiment),
            "--wandb-mode",
            args.wandb_mode,
            "--wandb-project",
            args.wandb_project,
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
            "--hardware-label",
            args.local_hardware_label,
            "--hardware-kind",
            args.local_hardware_kind,
        ]
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_api_key:
            cmd.extend(["--wandb-api-key", args.wandb_api_key])
        run(cmd, cwd=PROJECT_ROOT, env=env)


def run_colab(args: argparse.Namespace) -> None:
    ensure_colab_cli(args)
    wandb_api_key = args.wandb_api_key
    if args.prompt_wandb_key and args.wandb_mode == "online" and not args.wandb_secret_name:
        wandb_api_key = clean_wandb_key(
            getpass.getpass("Paste W&B API key for Colab run: ").strip()
        )
        if not wandb_api_key:
            raise SystemExit("No W&B API key was entered.")
    cmd = [
        "colab",
        "--auth",
        args.colab_auth,
    ]
    if args.colab_auth == "oauth2":
        cmd.extend(["--client-oauth-config", args.colab_client_oauth_config])
    cmd.extend(["run", "-s", args.session, "--gpu", args.gpu])
    if args.keep_colab:
        cmd.append("--keep")
    cmd.extend(
        [
            str(REMOTE_TEMPLATE),
            "--repo-url",
            args.repo_url,
            "--repo-branch",
            args.repo_branch,
            "--benchmark-prefix",
            args.benchmark_prefix,
            "--attn-model",
            args.attn_model,
            "--core-model",
            args.core_model,
            "--steer-model",
            args.steer_model,
            "--wandb-mode",
            args.wandb_mode,
            "--wandb-project",
            args.wandb_project,
            "--wandb-secret-project",
            args.wandb_secret_project,
            "--wandb-secret-name",
            args.wandb_secret_name,
            "--wandb-secret-version",
            args.wandb_secret_version,
            "--colab-install-vllm",
            args.colab_install_vllm,
            "--hardware-label",
            f"colab-{args.gpu}",
            "--hardware-kind",
            "cuda",
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
    )
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if wandb_api_key and not args.wandb_secret_name:
        cmd.extend(["--wandb-api-key", wandb_api_key])
    run(cmd, cwd=PROJECT_ROOT)


def main() -> int:
    args = parse_args()
    if args.skip_local and args.skip_colab:
        raise SystemExit("Nothing to run: both --skip-local and --skip-colab were set.")
    preflight_colab_auth(args)
    if not args.skip_local:
        run_local(args)
    if not args.skip_colab:
        run_colab(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
