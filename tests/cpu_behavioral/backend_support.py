"""Environment and output paths for the existing generation fixtures."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def backend_environment(backend: str) -> dict[str, str]:
    """Give each engine its own environment; never mutate pytest's environment."""
    env = dict(os.environ)
    env["MIA_BEHAVIOR_BACKEND"] = backend
    env.setdefault("MIA_WRITER_PROCESS", "0")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if backend == "cpu":
        env.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("VLLM_CPU_OMP_THREADS_BIND", "nobind")
        env["TRITON_DEFAULT_BACKEND"] = "cpu"
        if sys.platform == "darwin":
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            env.setdefault("TRITON_DISABLE_OPENMP", "1")
            env.setdefault("TRITON_CACHE_DIR", str(Path("/private/tmp") / "triton-cache"))
            env.pop("TRITON_LOCAL_LIBOMP_PATH", None)
            env.pop("DYLD_INSERT_LIBRARIES", None)
    elif backend == "cuda":
        env.pop("TRITON_DEFAULT_BACKEND", None)
        env.pop("TRITON_DISABLE_OPENMP", None)
    return env


def assert_backend() -> None:
    """Guard the selected platform before constructing a demo's engine."""
    from vllm.platforms import current_platform

    expected = os.environ.get("MIA_BEHAVIOR_BACKEND", "cpu")
    actual = "cpu" if current_platform.is_cpu() else "cuda" if current_platform.is_cuda() else "other"
    if actual != expected:
        raise RuntimeError(f"Requested {expected} fixture, but vLLM selected {actual}")


def hook_directory(repo_root: Path) -> Path:
    return Path(os.environ.get("MIA_BEHAVIOR_HOOK_DIR", repo_root / "tests/cpu_behavioral/.hooks"))
