# tests/plugin/test_engine_config_patch.py
"""The engine-config patch owns the model-runner constraint.

Every worker in this package reads the legacy GPU model runner's
``input_batch`` / ``requests`` surface, and vLLM 0.28+ defaults common
architectures to the V2 runner. ``_patched_create_engine_config`` must
therefore pin ``VLLM_USE_V2_MODEL_RUNNER=0`` before the config is built
whenever a plugin worker is selected — offline and under ``vllm serve``
alike — while leaving an operator's explicit value alone (the unified
worker then fails loud at install; see tests/workers/test_unified_install.py).

``_hook_plugin`` imports no vLLM at module level, so this runs on the
bare package.
"""
import os
import types

import pytest

from vllm_hook_plugins import _hook_plugin

V2 = _hook_plugin._V2_MODEL_RUNNER_ENV
WORKER_ENV = "VLLM_HOOK_WORKER"


class FakeEngineArgs:
    def __init__(self, worker_extension_cls=""):
        self.worker_extension_cls = worker_extension_cls
        self.enforce_eager = False


def _fake_config():
    return types.SimpleNamespace(
        speculative_config=None,
        cache_config=types.SimpleNamespace(enable_prefix_caching=False),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        model_config=types.SimpleNamespace(model="m", dtype="torch.float16"),
    )


@pytest.fixture
def patched(monkeypatch):
    """A recording stand-in for vLLM's create_engine_config, with the env
    vars and module stash the patch touches isolated per test (including
    keys the patch adds, which monkeypatch.delenv alone would not undo).
    """
    saved = {name: os.environ.get(name) for name in (V2, WORKER_ENV)}
    for name in saved:
        os.environ.pop(name, None)
    seen = {}

    def original(self, *args, **kwargs):
        # What the worker will see: the property reads the env var live
        # while VllmConfig is built, before the engine core is spawned.
        seen["v2_at_build"] = os.environ.get(V2)
        seen["worker"] = self.worker_extension_cls
        seen["enforce_eager"] = self.enforce_eager
        return _fake_config()

    monkeypatch.setattr(_hook_plugin, "_original_create_engine_config", original)
    monkeypatch.setattr(_hook_plugin, "_ENGINE_FACTS", {})
    yield seen
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_unified_worker_pins_legacy_runner_before_config_is_built(patched):
    os.environ[WORKER_ENV] = "unified"
    args = FakeEngineArgs()
    _hook_plugin._patched_create_engine_config(args)
    assert args.worker_extension_cls == _hook_plugin._WORKER_EXT_UNIFIED
    assert patched["worker"] == _hook_plugin._WORKER_EXT_UNIFIED
    assert patched["enforce_eager"] is True
    assert patched["v2_at_build"] == "0"
    # still set for the engine core / worker processes spawned next
    assert os.environ[V2] == "0"
    assert _hook_plugin._ENGINE_FACTS["active_worker"] == "unified"


def test_explicit_operator_v2_is_respected(patched):
    os.environ[WORKER_ENV] = "unified"
    os.environ[V2] = "1"
    _hook_plugin._patched_create_engine_config(FakeEngineArgs())
    assert patched["v2_at_build"] == "1"
    assert os.environ[V2] == "1"


@pytest.mark.parametrize(
    "env_worker, preset_cls",
    [
        (None, ""),  # env unset: the default hidden_states worker
        ("qk", ""),
        ("steer", ""),
        # HookLLM passes PluginRegistry.get_worker(name).path explicitly.
        (None, "vllm_hook_plugins.workers.spotlight_worker.SpotlightWorker"),
    ],
)
def test_every_plugin_worker_pins_the_runner(patched, env_worker, preset_cls):
    # They all read the same input_batch / requests surface.
    if env_worker is not None:
        os.environ[WORKER_ENV] = env_worker
    _hook_plugin._patched_create_engine_config(FakeEngineArgs(preset_cls))
    assert patched["worker"].startswith("vllm_hook_plugins.workers.")
    assert patched["v2_at_build"] == "0"


def test_foreign_worker_extension_leaves_the_runner_choice_to_vllm(patched):
    _hook_plugin._patched_create_engine_config(FakeEngineArgs("other_pkg.workers.Worker"))
    assert patched["v2_at_build"] is None
    assert V2 not in os.environ

