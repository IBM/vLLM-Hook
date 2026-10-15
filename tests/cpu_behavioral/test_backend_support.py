"""Discovery must distinguish unavailable runtimes from broken scenarios."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

import backend_support as support


def fake_runtime(monkeypatch, *, backend="cpu", missing_worker=None):
    monkeypatch.setattr(support.importlib.util, "find_spec", lambda name: object())
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        is_cpu=lambda: backend == "cpu", is_cuda=lambda: backend == "cuda")
    torch = ModuleType("torch")
    torch.ones = lambda *args, **kwargs: SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1))
    mia = ModuleType("mia")
    mia.register_plugins = lambda: None
    registry = ModuleType("mia.registry")
    registry.PluginRegistry = SimpleNamespace(
        get_worker=lambda name: None if name == missing_worker else object(),
        get_analyzer=lambda name: object())
    for name, module in {"vllm.platforms": platforms, "torch": torch,
                         "mia": mia, "mia.registry": registry}.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_missing_runtime_is_unavailable(monkeypatch):
    monkeypatch.setattr(support.importlib.util, "find_spec", lambda name: None)
    matrix = support.discover()
    assert all(cell["status"] == "unavailable" for row in matrix.values() for cell in row.values())


@pytest.mark.parametrize("backend,other", [("cpu", "cuda"), ("cuda", "cpu")])
def test_only_installed_platform_is_candidate(monkeypatch, backend, other):
    fake_runtime(monkeypatch, backend=backend)
    matrix = support.discover()
    assert all(row[backend]["status"] == "ready" for row in matrix.values())
    assert all(row[other]["status"] == "unavailable" for row in matrix.values())
    assert all(row["metal"]["status"] == "unavailable" for row in matrix.values())


def test_missing_worker_affects_only_its_scenarios(monkeypatch):
    fake_runtime(monkeypatch, missing_worker="capture_qk")
    matrix = support.discover()
    assert matrix["attention_tracker"]["cpu"]["status"] == "error"
    assert matrix["core_reranker"]["cpu"]["status"] == "error"
    assert matrix["hidden_states"]["cpu"]["status"] == "ready"
    assert matrix["activation_steer"]["cpu"]["status"] == "ready"


def test_broken_device_is_error_not_skip(monkeypatch):
    fake_runtime(monkeypatch)
    def broken(*args, **kwargs):
        raise RuntimeError("device initialization failed")
    monkeypatch.setattr(sys.modules["torch"], "ones", broken)
    assert all(row["cpu"]["status"] == "error" for row in support.discover().values())


def test_backend_guard_rejects_wrong_platform(monkeypatch):
    fake_runtime(monkeypatch, backend="cuda")
    monkeypatch.setenv("MIA_BEHAVIOR_BACKEND", "cpu")
    with pytest.raises(RuntimeError, match="vLLM selected cuda"):
        support.assert_backend()


def test_cuda_does_not_inherit_cpu_triton_settings(monkeypatch):
    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "cpu")
    monkeypatch.setenv("TRITON_DISABLE_OPENMP", "1")
    env = support.backend_environment("cuda")
    assert "TRITON_DEFAULT_BACKEND" not in env
    assert "TRITON_DISABLE_OPENMP" not in env
    assert support.os.environ["TRITON_DEFAULT_BACKEND"] == "cpu"


def test_required_unavailable_backend_fails(tmp_path):
    from test_vllm_cpu_fixtures import test_vllm_demo_scenario
    matrix = {"hidden_states": {"metal": {"status": "unavailable", "reason": "adapter missing"}}}
    with pytest.raises(pytest.fail.Exception, match="Required hidden_states/metal"):
        test_vllm_demo_scenario("hidden_states", "metal", (matrix, {"metal"}), tmp_path, lambda *args: None)


def test_broken_runtime_is_not_skipped(tmp_path):
    from test_vllm_cpu_fixtures import test_vllm_demo_scenario
    matrix = {"hidden_states": {"cpu": {"status": "error", "reason": "broken runtime"}}}
    with pytest.raises(pytest.fail.Exception, match="broken runtime"):
        test_vllm_demo_scenario("hidden_states", "cpu", (matrix, set()), tmp_path, lambda *args: None)


def test_failed_generation_is_not_reclassified_as_unavailable(monkeypatch, tmp_path):
    import test_vllm_cpu_fixtures as suite
    matrix = {"hidden_states": {"cpu": {"status": "ready", "reason": "candidate"}}}
    monkeypatch.setattr(suite.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(pytest.fail.Exception, match="Fixture/process failed"):
        suite.test_vllm_demo_scenario("hidden_states", "cpu", (matrix, set()), tmp_path, lambda *args: None)


def test_cpu_preserves_native_compilation_settings(monkeypatch):
    monkeypatch.delenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setattr(support.sys, "platform", "darwin")
    env = support.backend_environment("cpu")
    assert env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "1800"
    assert env["TRITON_CACHE_DIR"] == "/private/tmp/triton-cache"
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "2400")
    assert support.backend_environment("cpu")["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "2400"
