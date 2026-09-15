# tests/workers/test_unified_install.py
"""Hook installation fails loud on an unsupported model runner.

On vLLM's V2 runner the unified worker used to install cleanly, find no
rows to map in every hook, and let steered requests complete unsteered
with only a warning in the worker log. ``install_hooks`` now checks the
runner surface once and raises; the legitimate empty-views passes
(warmup, CUDA-graph capture) stay silent.

The worker module imports ``vllm.forward_context`` at load time; when vLLM
is absent (the CI isolation job) a stand-in for that one module is
installed for the duration of this file.
"""
import importlib
import sys
import types

import pytest
import torch


@pytest.fixture(scope="module")
def unified_worker():
    stubbed = []
    try:
        import vllm.forward_context  # noqa: F401
    except ImportError:
        vllm_pkg = types.ModuleType("vllm")
        forward_context = types.ModuleType("vllm.forward_context")
        forward_context.get_forward_context = lambda: None
        vllm_pkg.forward_context = forward_context
        for name, module in (("vllm", vllm_pkg), ("vllm.forward_context", forward_context)):
            sys.modules[name] = module
            stubbed.append(name)
    module = importlib.import_module("vllm_hook_plugins.workers.unified_worker")
    yield module
    if stubbed:
        for name in stubbed:
            sys.modules.pop(name, None)
        # Re-import against the real engine if a later test has one.
        sys.modules.pop("vllm_hook_plugins.workers.unified_worker", None)


class TinyModel(torch.nn.Module):
    """One decoder layer at ``model.layers.0`` so the layer hooks have a
    module to land on.
    """

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2, head_dim=4
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8)])


class LegacyRunner:
    def __init__(self, model=None):
        self.model = model
        self.input_batch = types.SimpleNamespace(
            req_ids=[], num_computed_tokens_cpu=[], num_prompt_tokens=[]
        )
        self.requests = {}


class V2Runner:
    def __init__(self, model=None):
        self.model = model
        self.req_states = object()
        self.input_buffers = object()


class FakeMetadata:
    def __init__(self, query_start_loc):
        self.query_start_loc = torch.tensor(query_start_loc)
        self.seq_lens = None


def make_worker(module, runner):
    class Worker(module.UnifiedHookWorker):
        pass

    worker = Worker()
    worker.model_runner = runner
    worker.rank = 0
    worker.parallel_config = types.SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
    return worker


def test_legacy_runner_installs_hooks_once(unified_worker):
    worker = make_worker(unified_worker, LegacyRunner(TinyModel()))
    worker.install_hooks()
    assert worker._hooks_installed
    assert worker._hooked_layers == {0}
    assert len(worker._hooks) == 2  # pre- and post-hook on the one layer
    worker.install_hooks()
    assert len(worker._hooks) == 2


def test_v2_runner_raises_naming_constraint_and_is_not_remembered_as_installed(unified_worker):
    worker = make_worker(unified_worker, V2Runner(TinyModel()))
    with pytest.raises(RuntimeError) as info:
        worker.install_hooks()
    message = str(info.value)
    assert "V2Runner has no input_batch, requests" in message
    assert "legacy GPU model runner" in message
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in message
    assert not worker._hooks_installed
    assert worker._hooks == []
    # every retry re-raises rather than silently doing nothing
    with pytest.raises(RuntimeError):
        worker.install_hooks()


def test_prepare_requests_propagates_install_failure(unified_worker):
    worker = make_worker(unified_worker, V2Runner(TinyModel()))
    with pytest.raises(RuntimeError, match="legacy GPU model runner"):
        worker.prepare_requests({"r": {"intervention_spec": None, "capture": None}})


def test_runner_without_loaded_model_raises(unified_worker):
    worker = make_worker(unified_worker, LegacyRunner(model=None))
    with pytest.raises(RuntimeError, match="no loaded model"):
        worker.install_hooks()
    assert not worker._hooks_installed


def _installed_worker(module, runner):
    worker = make_worker(module, runner)
    worker.install_hooks()
    return worker


def test_pass_views_silent_without_attention_metadata(unified_worker, monkeypatch):
    worker = _installed_worker(unified_worker, LegacyRunner(TinyModel()))
    monkeypatch.setattr(
        unified_worker, "get_forward_context", lambda: types.SimpleNamespace(attn_metadata=None)
    )
    assert worker._pass_views() == []


def test_pass_views_silent_on_warmup_metadata(unified_worker, monkeypatch):
    worker = _installed_worker(unified_worker, LegacyRunner(TinyModel()))
    monkeypatch.setattr(
        unified_worker, "get_forward_context", lambda: types.SimpleNamespace(attn_metadata={})
    )
    assert worker._pass_views() == []


def test_pass_views_silent_during_cuda_graph_capture(unified_worker, monkeypatch):
    worker = _installed_worker(unified_worker, LegacyRunner(TinyModel()))
    monkeypatch.setattr(
        unified_worker,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=FakeMetadata([0, 4])),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert worker._pass_views() == []


def test_pass_views_map_rows_on_legacy_runner(unified_worker, monkeypatch):
    runner = LegacyRunner(TinyModel())
    runner.input_batch = types.SimpleNamespace(
        req_ids=["a"], num_computed_tokens_cpu=[0], num_prompt_tokens=[4]
    )
    worker = _installed_worker(unified_worker, runner)
    monkeypatch.setattr(
        unified_worker,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=FakeMetadata([0, 4])),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (view,) = worker._pass_views()
    assert view.req_id == "a"
    assert view.positions == range(0, 4)
