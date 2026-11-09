"""
Tests that adjust_rs sets the projection of the full residual stream to avg_proj.

vLLM decoder layers return (hidden_states, residual), and the next layer adds the two.
adjust_rs must therefore control (hidden + residual) . d, not residual . d alone.
"""

import pytest
import torch

from mia.core.hooks.ops import _steer_buffer_impl
from mia.workers.steer_worker import _steer_rows

N, H = 6, 64


def _setup(dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(N, H, generator=g).to(device, dtype)     # layer output (e.g. MLP)
    residual = torch.randn(N, H, generator=g).to(device, dtype)
    d = torch.randn(H, generator=g)
    d = (d / d.norm()).to(device, dtype)
    return hidden, residual, d


def _proj(t, d):
    return (t.float() @ d.float())


# ---- eager path: _steer_rows ------------------------------------------------------------

@pytest.mark.parametrize("avg", [0.0, 1.5, -2.0])
def test_eager_adjust_rs_sets_full_stream_projection(avg):
    hidden, residual, d = _setup()
    data = {"dir": d, "avg_proj": torch.tensor(avg)}
    new_res = _steer_rows(residual, {"method": "adjust_rs"}, data, hidden=hidden)
    assert torch.allclose(_proj(hidden + new_res, d), torch.full((N,), avg), atol=1e-4)


def test_eager_adjust_rs_only_moves_along_d():
    hidden, residual, d = _setup()
    data = {"dir": d, "avg_proj": torch.tensor(0.0)}
    delta = _steer_rows(residual, {"method": "adjust_rs"}, data, hidden=hidden) - residual
    # the edit is a multiple of d on every row
    perp = delta - _proj(delta, d).unsqueeze(-1) * d
    assert perp.abs().max() < 1e-5


def test_eager_without_hidden_keeps_whole_stream_semantics():
    # a layer that returns one tensor (or (hidden, None)): that tensor is the whole stream
    _, residual, d = _setup()
    data = {"dir": d, "avg_proj": torch.tensor(0.7)}
    new_res = _steer_rows(residual, {"method": "adjust_rs"}, data)
    assert torch.allclose(_proj(new_res, d), torch.full((N,), 0.7), atol=1e-4)


def test_eager_add_vector_ignores_hidden():
    hidden, residual, d = _setup()
    data = {"dir": d}
    cfg = {"method": "add_vector", "coefficient": 3.0}
    assert torch.equal(_steer_rows(residual, cfg, data, hidden=hidden),
                       _steer_rows(residual, cfg, data))


# ---- graph path: the steer_buffer op -----------------------------------------------------

def _tables(d, avg):
    vec_table = torch.stack([torch.zeros_like(d), d])    # row 1 holds the vector
    avg_table = torch.tensor([0.0, avg])
    return vec_table, avg_table


def test_graph_op_adjust_rs_sets_full_stream_projection():
    hidden, residual, d = _setup()
    vec_table, avg_table = _tables(d, 0.0)
    coeff = torch.zeros(N)
    vec_id = torch.ones(N, dtype=torch.int64)
    mode = torch.ones(N, dtype=torch.int64)                # adjust_rs everywhere
    res = residual.clone()
    _steer_buffer_impl(res, coeff, vec_id, vec_table, avg_table, mode, hidden)
    assert _proj(hidden + res, d).abs().max() < 1e-4


def test_graph_op_mixed_rows():
    hidden, residual, d = _setup()
    vec_table, avg_table = _tables(d, -1.0)
    coeff = torch.tensor([0.0, 0.0, 2.0, 2.0, 0.0, 0.0])
    vec_id = torch.tensor([1, 1, 1, 1, 0, 0])
    mode = torch.tensor([1, 1, 0, 0, 0, 0])               # adjust, adjust, add, add, off, off
    res = residual.clone()
    _steer_buffer_impl(res, coeff, vec_id, vec_table, avg_table, mode, hidden)
    assert torch.allclose(_proj(hidden[:2] + res[:2], d), torch.full((2,), -1.0), atol=1e-4)
    assert torch.allclose(res[2:4], residual[2:4] + 2.0 * d, atol=1e-6)
    assert torch.equal(res[4:], residual[4:])


def test_graph_op_without_hidden_unchanged():
    _, residual, d = _setup()
    vec_table, avg_table = _tables(d, 0.3)
    res = residual.clone()
    _steer_buffer_impl(res, torch.zeros(N), torch.ones(N, dtype=torch.int64), vec_table,
                       avg_table, torch.ones(N, dtype=torch.int64))
    assert torch.allclose(_proj(res, d), torch.full((N,), 0.3), atol=1e-4)

