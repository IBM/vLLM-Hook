"""Runtime discovery shared by pytest and the original demo fixtures.

No torch/vLLM imports at module scope: discovery runs in a separate process.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


BACKENDS = ("cpu", "cuda", "metal")
# Scenario -> (existing fixture, worker, analyzer).
DEMO_SCENARIOS = {
    "hidden_states": ("run_hidden_states_cpu.py", "capture_hs", "hidden_states"),
    "attention_tracker": ("run_attention_tracker_cpu.py", "capture_qk", "attn_tracker"),
    "activation_steer": ("run_activation_steer_cpu.py", "steer", None),
    "core_reranker": ("run_core_reranker_cpu.py", "capture_qk", "core_reranker"),
}


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


def discover() -> dict:
    """Find compatible scenario candidates. Only a passing fixture proves usability."""
    result = {
        scenario: {backend: {"status": "unavailable", "reason": "no matching vLLM runtime"}
                   for backend in BACKENDS}
        for scenario in DEMO_SCENARIOS
    }
    for row in result.values():
        row["metal"]["reason"] = "MIA Metal scenario adapter is not implemented on this branch"
    if importlib.util.find_spec("vllm") is None:
        for row in result.values():
            for backend in ("cpu", "cuda"):
                row[backend]["reason"] = "vLLM is not installed in this Python environment"
        return result

    try:
        from vllm.platforms import current_platform

        backend = "cpu" if current_platform.is_cpu() else "cuda" if current_platform.is_cuda() else None
    except Exception as exc:
        for row in result.values():
            for name in ("cpu", "cuda"):
                row[name] = {"status": "error", "reason": f"vLLM platform discovery failed: {exc!r}"}
        return result
    if backend is None:
        return result

    try:
        import torch
        # A CPU on a CUDA host does not imply a CPU-built vLLM installation.
        # Exercise the selected device before claiming it is a candidate.
        value = torch.ones(1, device=backend).sum().item()
        assert value == 1
        from mia import register_plugins
        from mia.registry import PluginRegistry

        register_plugins()
    except Exception as exc:
        for row in result.values():
            row[backend] = {"status": "error", "reason": f"{backend} runtime check failed: {exc!r}"}
        return result

    for scenario, (_, worker, analyzer) in DEMO_SCENARIOS.items():
        if PluginRegistry.get_worker(worker) is None or (
            analyzer is not None and PluginRegistry.get_analyzer(analyzer) is None
        ):
            result[scenario][backend] = {"status": "error", "reason": "scenario worker/analyzer is missing"}
        else:
            result[scenario][backend] = {"status": "ready", "reason": "runtime detected; fixture must validate generation"}
    return result


def assert_backend() -> None:
    """Guard the selected platform before constructing a demo's engine."""
    from vllm.platforms import current_platform

    expected = os.environ.get("MIA_BEHAVIOR_BACKEND", "cpu")
    actual = "cpu" if current_platform.is_cpu() else "cuda" if current_platform.is_cuda() else "other"
    if actual != expected:
        raise RuntimeError(f"Requested {expected} fixture, but vLLM selected {actual}")


def hook_directory(repo_root: Path) -> Path:
    return Path(os.environ.get("MIA_BEHAVIOR_HOOK_DIR", repo_root / "tests/cpu_behavioral/.hooks"))


if __name__ == "__main__":
    # Ensure the checkout, rather than an editable install elsewhere, supplies MIA.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    Path(sys.argv[1]).write_text(json.dumps(discover(), indent=2))
