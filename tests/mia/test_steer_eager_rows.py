"""Eager steering is per-request: each request's OWN rows, with its OWN config.

The contract (``SteerWorker`` docstring, docs/configs.md): requests in one batch may carry
different vectors / methods / coefficients / layers, and an unsteered request is never
touched. The eager default-mode branch broke it -- it ran ``entries[0]``'s config over the
WHOLE residual, so every co-scheduled request (steered or not, plus any padding row) was
steered with the config of the lowest batch row that targeted the layer. Nothing caught
it: the eager parity leg runs at width 1 with ``phase="decode"`` (a non-default mode, so it
never reached that branch) and the mixed-arm leakage gate is graph-only.

Proven against fakes, all on CPU: a 2-layer model whose blocks return vLLM's fused-residual
``(hidden, residual)`` pair, a hand-built ``StepView``, and ``_install_hooks()`` called
directly (``install_hooks()`` demands a real V2 runner). A hook fires by calling a block.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

import mia
import mia.workers.steer_worker as steer
from mia.runner import StepView

_REPO = Path(__file__).resolve().parents[2]
H = 8   # hidden size


@pytest.fixture(autouse=True)
def _mia_is_this_checkout():
    """``mia`` is an editable install: a checkout that is not the install's own tree
    imports the INSTALLED tree unless it wins on ``sys.path``. Every test below would
    then exercise some other revision's hook and pass or fail on its behalf."""
    where = Path(mia.__file__).resolve()
    if not where.is_relative_to(_REPO):
        pytest.fail(f"imported mia from {where}, not from the checkout under test "
                    f"({_REPO}); put the checkout first on PYTHONPATH")


# ---------------------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------------------

class _Block(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x), x            # vLLM's fused-residual (hidden, residual)


class _Layers(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block() for _ in range(n))


class _Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.model = _Layers(n)                  # module names: model.layers.<i>


class _Worker(steer.SteerWorker):
    def __init__(self, n_layers=2):
        self.model_runner = types.SimpleNamespace(model=_Model(n_layers))
        self._install_hooks()

    def fire(self, layer, x):
        """Run block ``layer`` on residual ``x``; return the residual after the hook."""
        return self.model_runner.model.model.layers[layer](x.clone())[1]


class _Req(NamedTuple):
    rid: str
    n: int                          # tokens scheduled this step
    prefill: bool
    steer: Optional[dict] = None    # None: the request carries no steer config
    computed: int = 0               # tokens computed before this step
    prompt_len: Optional[int] = None  # default: this step ends the prompt


def _step(*reqs: _Req) -> StepView:
    n = len(reqs)
    sched = np.asarray([r.n for r in reqs], dtype=np.int32)
    qsl = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(sched, out=qsl[1:])
    computed = np.asarray([r.computed for r in reqs], dtype=np.int32)
    prompt = np.asarray([r.computed + r.n if r.prompt_len is None else r.prompt_len
                         for r in reqs], dtype=np.int32)
    return StepView(
        req_ids=[r.rid for r in reqs],
        num_reqs=n,
        num_scheduled_tokens=sched,
        query_start_loc=torch.from_numpy(qsl.copy()),
        query_start_loc_np=qsl,
        num_computed_tokens_np=computed,
        prefill_len_np=prompt.copy(),
        prompt_len_np=prompt,
        is_prefilling_np=np.asarray([r.prefill for r in reqs], dtype=bool),
        seq_lens=torch.from_numpy(computed + sched),
        block_tables=(),
        extra_args={r.rid: {"steer": r.steer} for r in reqs if r.steer is not None},
    )


def _vector(tmp_path, name, seed, avg_proj=0.5):
    """Write a steering-vector file in the shipped format; return (path, unit dir)."""
    d = torch.randn(H, generator=torch.Generator().manual_seed(seed))
    d = d / d.norm()
    path = tmp_path / f"{name}.pt"
    torch.save({"dir": d.numpy(), "avg_proj": torch.tensor(avg_proj)}, path)
    return str(path), d


def _add(path, coeff, layer=0, **modes):
    return {"method": "add_vector", "coefficient": coeff, "optimal_layer": layer,
            "vector_path": path, **modes}


def _adjust(path, layer=0, **modes):
    return {"method": "adjust_rs", "optimal_layer": layer, "vector_path": path, **modes}


def _residual(n_rows):
    """A reproducible residual: ``n_rows`` flat token rows."""
    return torch.randn(n_rows, H, generator=torch.Generator().manual_seed(1000 + n_rows))


def _whole(w, x, cfg):
    """The whole-tensor op the default path always applied, for bit-identity checks."""
    return steer._steer_rows(x, cfg, w._vector_cache_for(cfg))


# ---------------------------------------------------------------------------------------
# (a) an unsteered request is never touched
# ---------------------------------------------------------------------------------------

def test_unsteered_request_is_untouched_next_to_a_steered_one(tmp_path):
    path, _ = _vector(tmp_path, "a", 0)
    w = _Worker()
    cfg = _adjust(path)                                      # default phase/positions
    w._step = _step(_Req("s", 3, True, cfg),                 # rows 0-2
                    _Req("u", 1, False),                     # row 3: unsteered decode
                    _Req("t", 2, True))                      # rows 4-5: unsteered prefill
    x = _residual(6)
    y = w.fire(0, x)
    assert torch.equal(y[3:], x[3:]), "an UNSTEERED request's rows were steered"
    # The steered request's rows are the whole-tensor op's rows, bit for bit (a row-slice
    # of adjust_rs is not: matmul blocking depends on the row count).
    assert torch.equal(y[:3], _whole(w, x, cfg)[:3])
    assert not torch.equal(y[:3], x[:3])


def test_padding_rows_beyond_the_last_request_are_never_steered(tmp_path):
    path, d = _vector(tmp_path, "a", 1)
    w = _Worker()
    cfg = _add(path, 1.0)
    w._step = _step(_Req("a", 1, False, cfg), _Req("b", 1, False, cfg))
    x = _residual(4)                                         # 2 real rows + 2 padding rows
    y = w.fire(0, x)
    assert torch.equal(y[:2], x[:2] + 1.0 * d.view(1, -1))
    assert torch.equal(y[2:], x[2:]), "padding rows were steered"


# ---------------------------------------------------------------------------------------
# (b) each steered request gets its OWN config
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("differs", ["vector", "coefficient"])
def test_each_request_gets_its_own_vector_and_coefficient(tmp_path, differs):
    """Vary ONE field at a time: when vector and coefficient differ together, either one
    alone separates the two requests, and a config that ignored the other would pass."""
    pa, da = _vector(tmp_path, "a", 2)
    pb, db = _vector(tmp_path, "b", 3)
    if differs == "vector":
        cb, eb = _add(pb, 1.0), 1.0 * db                  # same coefficient, other vector
    else:
        cb, eb = _add(pa, 2.0), 2.0 * da                  # same vector, other coefficient
    w = _Worker()
    w._step = _step(_Req("a", 2, True, _add(pa, 1.0)), _Req("b", 1, False, cb))
    x = _residual(3)
    y = w.fire(0, x)
    assert torch.equal(y[:2], x[:2] + 1.0 * da.view(1, -1))
    assert torch.equal(y[2:], x[2:] + eb.view(1, -1)), f"b was steered with a's {differs}"


def test_each_request_gets_its_own_method(tmp_path):
    pa, da = _vector(tmp_path, "a", 4, avg_proj=0.25)
    pb, db = _vector(tmp_path, "b", 5)
    w = _Worker()
    ca, cb = _adjust(pa), _add(pb, 3.0)
    w._step = _step(_Req("a", 2, True, ca), _Req("b", 1, False, cb))
    x = _residual(3)
    y = w.fire(0, x)
    assert torch.equal(y[:2], _whole(w, x, ca)[:2])
    assert torch.allclose(y[:2], x[:2] + (0.25 - x[:2] @ da).unsqueeze(-1) * da.view(1, -1),
                          rtol=0, atol=1e-6)
    assert torch.equal(y[2:], x[2:] + 3.0 * db.view(1, -1)), "b was steered with a's method"


def test_one_vector_file_serves_both_methods(tmp_path):
    """The vector cache is keyed by path alone, so an add_vector request that loads a file
    first must not leave a later adjust_rs on the same file without its avg_proj."""
    path, d = _vector(tmp_path, "a", 12, avg_proj=0.25)
    w = _Worker()
    w._step = _step(_Req("a", 1, False, _add(path, 2.0)), _Req("b", 2, True, _adjust(path)))
    x = _residual(3)
    y = w.fire(0, x)
    assert torch.equal(y[:1], x[:1] + 2.0 * d.view(1, -1))
    assert torch.allclose(y[1:], x[1:] + (0.25 - x[1:] @ d).unsqueeze(-1) * d.view(1, -1),
                          rtol=0, atol=1e-6)


def test_requests_sharing_a_config_share_one_op(tmp_path, monkeypatch):
    """Cost is one op per distinct EFFECTIVE config, not one per request. Grouping is by
    what the op reads, not by dict identity: adjust_rs ignores ``coefficient``, so a
    differing coefficient on it does not split the group."""
    path, _ = _vector(tmp_path, "a", 6)
    w = _Worker()
    ca, cb = _adjust(path), {**_adjust(path), "coefficient": 7.0}
    w._step = _step(_Req("a", 2, True, ca), _Req("u", 1, False), _Req("b", 2, True, cb))
    x = _residual(5)
    ref = _whole(w, x, ca)
    calls = []
    real = steer._steer_rows
    monkeypatch.setattr(steer, "_steer_rows",
                        lambda rows, cfg, data: calls.append(cfg) or real(rows, cfg, data))
    y = w.fire(0, x)
    assert len(calls) == 1, f"{len(calls)} steer ops for one effective config"
    assert torch.equal(y[:2], ref[:2]) and torch.equal(y[3:], ref[3:])
    assert torch.equal(y[2], x[2]), "the UNSTEERED request between them was steered"


# ---------------------------------------------------------------------------------------
# (c) layer targeting is per-request
# ---------------------------------------------------------------------------------------

def test_each_request_is_steered_only_at_its_own_layer(tmp_path):
    pa, da = _vector(tmp_path, "a", 7)
    pb, db = _vector(tmp_path, "b", 8)
    w = _Worker()
    w._step = _step(_Req("a", 1, False, _add(pa, 1.0, layer=0)),
                    _Req("b", 1, False, _add(pb, 2.0, layer=1)))
    x = _residual(2)
    y0 = w.fire(0, x)
    assert torch.equal(y0[0], x[0] + 1.0 * da)
    assert torch.equal(y0[1], x[1]), "layer 0 steered b, which targets layer 1"
    y1 = w.fire(1, x)
    assert torch.equal(y1[0], x[0]), "layer 1 steered a, which targets layer 0"
    assert torch.equal(y1[1], x[1] + 2.0 * db)


# ---------------------------------------------------------------------------------------
# (d) one config owning every row is exactly the whole-tensor op
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["adjust_rs", "add_vector"])
def test_one_config_on_every_row_is_the_whole_tensor_op(tmp_path, method):
    """The batch today's default path already steered CORRECTLY must not move a bit."""
    path, _ = _vector(tmp_path, "a", 9)
    w = _Worker()
    cfg = _adjust(path) if method == "adjust_rs" else _add(path, 1.5)
    w._step = _step(_Req("a", 3, True, dict(cfg)), _Req("b", 1, False, dict(cfg)),
                    _Req("c", 2, True, dict(cfg)))
    x = _residual(6)
    assert torch.equal(w.fire(0, x), _whole(w, x, cfg))


# ---------------------------------------------------------------------------------------
# (e) rows that cannot be mapped to requests are never steered
# ---------------------------------------------------------------------------------------

def test_unmappable_rows_leave_the_residual_unchanged(tmp_path, capsys):
    """The step claims more tokens than the tensor has rows (a stale StepView on a dummy
    run, a sharded residual): no row can be attributed, so none may be steered."""
    path, _ = _vector(tmp_path, "a", 10)
    w = _Worker()
    cfg = _add(path, 1.0)
    w._step = _step(_Req("a", 3, True, cfg), _Req("b", 2, True, cfg))   # claims 5 rows
    x = _residual(4)
    assert torch.equal(w.fire(0, x), x), "rows no request owns were steered"
    assert torch.equal(w.fire(1, x), x), "rows no request owns were steered"
    assert capsys.readouterr().out.count("[steer]") == 1, "warn once, not per forward"


# ---------------------------------------------------------------------------------------
# (f) non-default phase x positions keep their exact semantics
# ---------------------------------------------------------------------------------------

def test_phase_and_positions_select_the_same_rows_as_before(tmp_path):
    """Rows steered per ``steer_span``, the single source of truth all three paths share.
    It needs no forward context: rows are read from the StepView's host query_start_loc."""
    path, d = _vector(tmp_path, "a", 11)
    w = _Worker()
    w._step = _step(
        # rows 0-2: FINAL prefill chunk, last_token -> only row 2 (the last prompt token)
        _Req("final", 3, True, _add(path, 1.0, positions="last_token"),
             computed=5, prompt_len=8),
        # rows 3-4: a NON-final chunk, last_token -> nothing (chunk-invariant)
        _Req("chunk", 2, True, _add(path, 2.0, positions="last_token"),
             computed=0, prompt_len=9),
        # row 5: a decode, last_token -> its one row
        _Req("dec_last", 1, False, _add(path, 3.0, positions="last_token")),
        # row 6: a decode under phase=prefill -> nothing
        _Req("dec_pre", 1, False, _add(path, 4.0, phase="prefill")),
        # rows 7-8: a prefill under phase=decode -> nothing
        _Req("pre_dec", 2, True, _add(path, 5.0, phase="decode")),
        # row 9: a decode under phase=decode -> steered
        _Req("dec_dec", 1, False, _add(path, 6.0, phase="decode")),
        # rows 10-11: a prefill under phase=prefill -> both rows
        _Req("pre_pre", 2, True, _add(path, 7.0, phase="prefill")),
        # row 12: default modes next to all of the above -> steered
        _Req("default", 1, False, _add(path, 8.0)),
        # row 13: unsteered
        _Req("u", 1, False),
    )
    x = _residual(14)
    y = w.fire(0, x)
    expect = x.clone()
    for row, coeff in ((2, 1.0), (5, 3.0), (9, 6.0), (10, 7.0), (11, 7.0), (12, 8.0)):
        expect[row] = x[row] + coeff * d
    assert torch.equal(y, expect)
