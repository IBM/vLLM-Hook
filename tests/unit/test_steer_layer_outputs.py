"""
CPU-Only: the CUDA graph steering wrap picks the residual stream for every decoder-layer output shape
"""

import pytest
import torch
from mia.core.hooks import install_steer
from mia.core.hooks.install_steer import _STEER_HOST_ATTR, _wrap_layer_class

class _Host:
    do_steer = True

    def __init__(self):
        self.steered = None

    def steer(self, t):
        self.steered = t

def _layer(output):
    cls = type(f"Layer{id(output)}", (torch.nn.Module,), {"forward": lambda self: output})
    _wrap_layer_class(cls)
    layer = cls()
    host = _Host()
    setattr(layer, _STEER_HOST_ATTR, host)
    return layer, host

def test_hidden_and_residual_steers_residual():
    h, r = torch.zeros(2, 4), torch.ones(2, 4)
    layer, host = _layer((h, r))
    layer()
    assert host.steered is r

def test_no_residual_steers_hidden():
    h = torch.zeros(2, 4)
    layer, host = _layer((h, None))
    layer()
    assert host.steered is h

def test_bare_tensor_is_steered():
    h = torch.zeros(2, 4)
    layer, host = _layer(h)
    layer()
    assert host.steered is h


@pytest.fixture(autouse=True)
def _restore_wrapped():
    yield
    install_steer._WRAPPED_LAYER_CLASSES.clear()
