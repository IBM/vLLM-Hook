# tests/workers/test_layer_output_forms.py
"""The layer hook steers every form a vLLM decoder layer returns its stream in.

vLLM decoder layers hand the residual stream on in three forms (vLLM 0.32):
Llama-style and Qwen layers return ``(hidden, residual)`` whose sum is the
stream; Granite returns the stream alone; Gemma 4 returns ``(stream, None)``.
Each form must move the stream by exactly the transform's shift. Before the
``(stream, None)`` case was handled, ``hidden + None`` raised inside the hook,
the request was disabled, and generation went on unsteered.

Runs on CPU without vLLM (the forward context is stubbed when vLLM is absent),
with a hand-made pass view standing in for the model runner's batch mapping.
"""
import importlib
import sys
import types

import pytest
import torch

from vllm_hook_plugins.core.interpreter.transforms import projection_target
from vllm_hook_plugins.core.schema import parse_intervention_spec

VEC = "sha256:" + "cd" * 32
HIDDEN = 8


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
        sys.modules.pop("vllm_hook_plugins.workers.unified_worker", None)


def _worker(module, vector):
    spec = parse_intervention_spec({"ops": [{
        "layers": [0],
        "transform": {"kind": "projection_target", "threshold": 0.0, "target": 3.0, "artifact": VEC},
        "scope": {"kind": "all"},
        "gate": None,
    }]}, num_layers=1)
    state = module._StagedRequest(spec, None, {VEC: {"vector": vector}})
    view = types.SimpleNamespace(req_id="r", row_slice=slice(0, 4), positions=range(0, 4), prompt_len=4,
                                 is_restart=False)
    worker = object.__new__(module.UnifiedHookWorker)
    worker._pass_views = lambda: [view]
    worker._state_for = lambda req_id: state
    worker._should_capture = False
    worker._disable_request = lambda state, exc, req_id=None: (_ for _ in ()).throw(exc)
    return worker


@pytest.mark.parametrize("form", ["pair", "single", "stream_none"])
def test_layer_hook_moves_the_stream_in_every_output_form(unified_worker, form):
    torch.manual_seed(0)
    vector = torch.randn(HIDDEN)
    stream = torch.randn(4, HIDDEN)
    stream[0] -= 5 * vector / vector.norm()  # at least one row below the threshold
    expected = projection_target(stream, vector=vector, threshold=0.0, target=3.0)

    if form == "pair":
        hidden = torch.randn(4, HIDDEN)
        output = (hidden, stream - hidden)
    elif form == "single":
        output = stream.clone()
    else:
        output = (stream.clone(), None)

    worker = _worker(unified_worker, vector)
    returned = worker._layer_hook(None, (), output, 0)

    if form == "pair":
        got = returned[0] + returned[1]
    elif form == "single":
        got = returned
    else:
        assert returned[1] is None
        got = returned[0]
    assert torch.allclose(got, expected, atol=1e-5)
    assert not torch.allclose(got, stream)  # the hook did steer
