"""Graph-mode routing wrapper: builds routing from `prepare_inputs`'s RETURN VALUE.

V2's InputBatch is TRANSIENT (built and returned by `prepare_inputs`, never stored on
the runner), so the old `_prepare_inputs` wrapper that reached into
`model_runner.input_batch` / `model_runner.requests` has nothing left to reach into.
These tests prove the replacement wrapper (`install_prepare_inputs_routing`):

  * wraps `prepare_inputs` (not `_prepare_inputs`) and hands every builder a `StepView`
    snapshot of the RETURNED batch, not a live runner reference;
  * still returns vLLM's InputBatch untouched;
  * skips the builder entirely when no registry is armed (`should_capture` false/absent);
  * lets `ApertureBackpressureError` PROPAGATE out of the wrapper (never-drop);
  * degrades every OTHER exception to a printed warning + "no capture this step";
  * skips during a CUDA-graph capture pass (buffers populate at replay, not capture);
  * is idempotent per label;
  * and that the per-subsystem registry dict (`set_registry`/`get_registry`) lets two
    subsystems coexist without one clobbering the other's registry.

Fakes follow the `tests/test_runner_adapter.py` convention: oversized arrays with a
sentinel tail, so a dropped `[:num_reqs]` slice fails loudly instead of passing by
coincidence.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from mia.graph.capture_aperture import ApertureBackpressureError
from mia.graph.install import install_prepare_inputs_routing
from mia.graph.registry import get_registry, set_registry
from mia.runner import StepView

_STALE = 999999


class _Runner:
    """V2-shaped fake: real `add_requests`/`finish_requests` (for the arg stash) and a
    real `block_tables.input_block_tables` (for `step_view`), oversized past `num_reqs`
    with a sentinel tail so a dropped `[:n]` slice in `step_view` cannot pass silently.
    """

    __module__ = "vllm.v1.worker.gpu.model_runner"

    def __init__(self, num_reqs=1, pad=3):
        max_num_reqs = num_reqs + pad
        self.block_tables = types.SimpleNamespace(
            input_block_tables=(
                torch.full((max_num_reqs, 3), _STALE, dtype=torch.int32),
            )
        )
        self.block_tables.input_block_tables[0][:num_reqs] = torch.arange(
            num_reqs * 3, dtype=torch.int32).reshape(num_reqs, 3)
        self.added, self.finished = [], []

    def add_requests(self, so):
        self.added.append(so)

    def finish_requests(self, so):
        self.finished.append(so)

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        return types.SimpleNamespace(
            req_ids=["r0"], num_reqs=1,
            num_scheduled_tokens=np.array([4], dtype=np.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            query_start_loc_np=np.array([0, 4], dtype=np.int32),
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            prefill_len_np=np.array([4], dtype=np.int32),
            is_prefilling_np=np.array([True]),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            idx_mapping_np=np.array([0], dtype=np.intp),
        )


class _V1Runner:
    __module__ = "vllm.v1.worker.gpu_model_runner"


def _fake_registry(**overrides):
    """Minimal-but-COMPLETE registry double: exercises the real (legacy) reset ->
    build -> upload branch of the wrapper (not just the builder call), since a bare
    `should_capture`-only stub would skip straight past `_upload_width`'s `registry.cap`
    read. `incremental_enabled`/`gpu_routing` default False so the legacy branch runs.
    """
    base = dict(
        should_capture=True,
        cap=8,
        incremental_enabled=False,
        gpu_routing=False,
        begin_step=lambda: None,
        routing_key=lambda step: None,
        reset_pinned=lambda width=None: None,
        upload=lambda width=None: None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# A fake with req_states.prompt_len, matching the runner-adapter's idx_mapping
# indirection that step_view()'s prompt_len_np field needs.
def _runner_with_req_states(num_reqs=1):
    runner = _Runner(num_reqs=num_reqs)
    runner.req_states = types.SimpleNamespace(
        prompt_len=types.SimpleNamespace(np=np.array([4] * (num_reqs + 3), dtype=np.int32))
    )
    return runner


def test_builder_receives_a_step_view_and_the_batch_is_passed_through():
    """No registry armed => the builder never runs, but vLLM's InputBatch still
    passes through unmodified (the wrapper must never swallow the real return value)."""
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()  # no registry for label "test" at all
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    result = runner.prepare_inputs(object(), object(), object())
    assert result.req_ids == ["r0"], "the wrapper must return vLLM's InputBatch untouched"
    assert seen == []


def test_builder_runs_when_capture_is_armed():
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    runner.prepare_inputs(object(), object(), object())
    assert len(seen) == 1 and isinstance(seen[0], StepView)
    assert seen[0].is_prefilling_np.tolist() == [True]
    assert seen[0].req_ids == ["r0"]


def test_registry_absent_for_this_label_skips_even_if_another_label_is_armed():
    """Two subsystems must not cross-wire: routing installed under label "hs" must
    never see the "qk" registry (or vice versa)."""
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "qk", _fake_registry())
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="hs")
    runner.prepare_inputs(object(), object(), object())
    assert seen == []


def test_should_capture_false_skips_the_builder():
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry(should_capture=False))
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    runner.prepare_inputs(object(), object(), object())
    assert seen == []


def test_never_drop_aperture_backpressure_propagates():
    """The never-drop contract: ApertureBackpressureError must reach the engine, not
    be swallowed by the wrapper's defensive except. This is the ONE exception that
    outranks "no capture this step"."""
    runner = _runner_with_req_states()
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())

    def _boom(step, registry):
        raise ApertureBackpressureError("aperture full; consumer dead")

    install_prepare_inputs_routing(runner, worker, _boom, label="test")
    with pytest.raises(ApertureBackpressureError):
        runner.prepare_inputs(object(), object(), object())


def test_other_exceptions_also_propagate_rather_than_capturing_nothing(capsys):
    """Every other routing failure propagates too.

    This test used to assert the opposite -- that a routing failure degraded to "no capture
    this step" plus a one-shot warning -- and that behaviour was the bug. A run whose routing
    broke at step 3 finished successfully, wrote short artifacts, and looked exactly like a
    complete one; the single warning had scrolled away and the PROF counter that carried the
    real rate is a no-op unless MIA_PROFILE=1.

    ApertureBackpressureError was already exempted (the test above) because the never-drop
    contract outranks availability. But every OTHER routing failure drops the same capturing
    request just as silently, so the exemption was covering one anticipated cause rather than
    the contract. And the realistic failure is deterministic, not transient: this wrapper
    reads V2-internal fields, so whatever breaks it breaks every subsequent step -- degrading
    buys a complete-looking run with no artifacts, not a degraded run.
    """
    runner = _runner_with_req_states()
    worker = types.SimpleNamespace()
    registry = _fake_registry()
    set_registry(worker, "test", registry)

    def _boom(step, registry):
        raise RuntimeError("routing build blew up")

    install_prepare_inputs_routing(runner, worker, _boom, label="test")
    with pytest.raises(RuntimeError, match="routing build blew up"):
        runner.prepare_inputs(object(), object(), object())
    # the original exception, not a wrapper: the traceback still points at the real failure
    out = capsys.readouterr().out
    assert "FATAL" in out and "routing failed" in out.lower()
    # registry state is still reset on the way out, so a caller that catches upstream does
    # not inherit a half-built routing plane
    assert registry._pending_plans == []
    assert registry._last_route_key is None


def test_skips_during_cudagraph_capture_pass(monkeypatch):
    """A host sync / pinned write while vLLM is capturing a graph is illegal -- the
    aperture is populated at REPLAY, never at capture."""
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True,
                        raising=False)
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    runner.prepare_inputs(object(), object(), object())
    assert seen == []


def test_install_is_idempotent_per_label():
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    build_fn = lambda step, registry: seen.append(step)
    install_prepare_inputs_routing(runner, worker, build_fn, label="test")
    wrapped_once = runner.prepare_inputs
    install_prepare_inputs_routing(runner, worker, build_fn, label="test")
    assert runner.prepare_inputs is wrapped_once, "a second install must be a no-op"


def test_v1_runner_is_refused_at_install_not_silently_skipped():
    from mia.runner import UnsupportedRunnerError

    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    with pytest.raises(UnsupportedRunnerError):
        install_prepare_inputs_routing(_V1Runner(), worker, lambda step, registry: None,
                                       label="test")


def test_two_subsystems_can_register_without_clobbering():
    worker = types.SimpleNamespace()
    set_registry(worker, "hs", "HS-REG")
    set_registry(worker, "qk", "QK-REG")
    assert get_registry(worker, "hs") == "HS-REG"   # not overwritten by the later install
    assert get_registry(worker, "qk") == "QK-REG"
    assert get_registry(worker, "steer") is None


def test_set_registry_overwrite_replaces_only_its_own_subsystem():
    worker = types.SimpleNamespace()
    set_registry(worker, "hs", "HS-REG-1")
    set_registry(worker, "qk", "QK-REG")
    set_registry(worker, "hs", "HS-REG-2")
    assert get_registry(worker, "hs") == "HS-REG-2"
    assert get_registry(worker, "qk") == "QK-REG"


# ---------------------------------------------------------------------------
# prefix_block_ids: the QK prefix-K block-table read (Task D2). V2's
# `runner.block_tables.input_block_tables[g]` is already gathered into BATCH order
# (see mia.runner.step_view), so `step.block_tables[group][req_index]` is req_index's
# OWN row directly -- no slot-index indirection like V1's
# `input_batch.block_table.block_tables[0].get_device_tensor(num_reqs)[slot]`.
# ---------------------------------------------------------------------------


def _step_with_block_tables(*group_rows, req_index_live_rows=2):
    """A StepView whose `block_tables` is exactly the tuple `group_rows` (each a 2-D
    tensor). Only `block_tables` is exercised by `prefix_block_ids`; every other field
    is filled with the minimum `step_view()`-shaped placeholder for `req_index_live_rows`
    requests so the StepView itself stays realistic."""
    n = req_index_live_rows
    return StepView(
        req_ids=[f"r{i}" for i in range(n)],
        num_reqs=n,
        num_scheduled_tokens=np.array([1] * n, dtype=np.int32),
        query_start_loc=torch.arange(n + 1, dtype=torch.int32),
        query_start_loc_np=np.arange(n + 1, dtype=np.int32),
        num_computed_tokens_np=np.zeros(n, dtype=np.int32),
        prefill_len_np=np.array([1] * n, dtype=np.int32),
        prompt_len_np=np.array([1] * n, dtype=np.int32),
        is_prefilling_np=np.array([True] * n),
        seq_lens=torch.ones(n, dtype=torch.int32),
        block_tables=group_rows,
        extra_args={},
    )


def test_prefix_block_ids_reads_batch_ordered_v2_tables():
    """V2's input_block_tables are already gathered into BATCH order by
    gather_block_tables(), so batch row `i` in `step.block_tables[g]` belongs to
    `step.req_ids[i]` -- prefix_block_ids does a plain row/column slice, no slot-index
    indirection. Row 0 and row 1 hold DIFFERENT block ids (not a shared arange) so a
    swapped-row bug fails instead of coincidentally matching; a padded tail row (never
    sliced to, since num_reqs=2) carries a sentinel that must never leak in."""
    from mia.graph.install import prefix_block_ids

    group0 = torch.tensor(
        [[7, 8, 9], [1, 2, 3], [_STALE, _STALE, _STALE], [_STALE, _STALE, _STALE]],
        dtype=torch.int32,
    )
    step = _step_with_block_tables(group0)

    assert prefix_block_ids(step, 0, num_blocks=2).tolist() == [7, 8]
    assert prefix_block_ids(step, 1, num_blocks=3).tolist() == [1, 2, 3]


def test_prefix_block_ids_selects_the_right_kv_cache_group():
    """`group` picks which KV-cache group's block table to read (hybrid models have
    more than one). Group 0 and group 1 hold disjoint values so reading the wrong
    group's tensor fails instead of passing by coincidence."""
    from mia.graph.install import prefix_block_ids

    group0 = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    group1 = torch.tensor([[70, 80, 90]], dtype=torch.int32)
    step = _step_with_block_tables(group0, group1, req_index_live_rows=1)

    assert prefix_block_ids(step, 0, num_blocks=2, group=0).tolist() == [7, 8]
    assert prefix_block_ids(step, 0, num_blocks=2, group=1).tolist() == [70, 80]


def test_prefix_block_ids_returns_none_without_block_tables():
    """No KV-cache group at all (e.g. attention-free model) -> None, never an index
    error into an empty tuple."""
    from mia.graph.install import prefix_block_ids

    step = _step_with_block_tables(req_index_live_rows=1)
    assert prefix_block_ids(step, 0, num_blocks=2) is None


# ---------------------------------------------------------------------------
# Task D3: the per-step drain wrapper (`execute_model`) — V2 dummy/profile passes
# and the stale-STEPVIEW hazard.
#
# V2 drives warmup, cudagraph capture and memory profiling through the SAME
# `execute_model` entry point real steps use, distinguished only by keyword
# arguments (`vllm/v1/worker/gpu/model_runner.py::execute_model`). Those passes
# carry no real requests, so draining them would push warmup garbage into the
# aperture. Skipping the drain is NECESSARY but NOT SUFFICIENT: `execute_model`
# never calls `prepare_inputs` for a dummy pass (`InputBatch.make_dummy` bypasses
# it — see `mia.runner.step_view`'s docstring), so `registry`'s per-step routing
# is not refreshed for that pass. Under FULL cudagraph, `execute_model` still
# unconditionally replays the exact compiled graph for a dummy run
# (`cudagraph_manager.run_fullgraph`, gated only on `batch_desc.cg_mode`, never on
# `dummy_run`) — so the baked in-graph scatter op fires anyway, reading
# `registry.capture_index_all` (the device-resident table it uses to decide where
# to write). Left alone, that stale table — last written by the PREVIOUS real
# step — would route a dummy pass's garbage hidden states into real, possibly
# not-yet-drained aperture rows. The wrapper must blind that table (zero =
# "every column discards", the registry's own documented safe/inert value) for
# exactly the duration of a dummy/profile call and restore it byte-for-byte
# afterwards (including on exception), so the next REAL step's W1 idle-skip still
# sees the correct last-real-routing state.
# ---------------------------------------------------------------------------


def _fake_capture_registry(**overrides):
    """A HostRegistry double shaped for the execute_model wrapper (not the routing
    wrapper): `capture_index_all` stands in for the device-resident routing table
    the baked op reads. Filled with a distinctive non-zero pattern (never 0, the
    real registry's own "inactive" sentinel) so a wrapper that fails to blind it
    during a dummy pass, or fails to restore it byte-for-byte after, is caught
    instead of coincidentally matching.
    """
    base = dict(
        should_capture=True,
        capture_index_all=torch.tensor([[5, 6, 0, 0], [1, 2, 3, 0]], dtype=torch.int64),
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _ExecRunner:
    """V2-shaped fake carrying only what the execute_model wrapper's SETUP path
    touches (it must not need a real aperture/host/model to install cleanly):
    `add_requests`/`finish_requests` (arg stash), `prepare_inputs` (routing wrap),
    and a settable `execute_model` the wrapper closes over as `orig_execute_model`.
    """

    __module__ = "vllm.v1.worker.gpu.model_runner"

    def __init__(self):
        self.added, self.finished = [], []

    def add_requests(self, so):
        self.added.append(so)

    def finish_requests(self, so):
        self.finished.append(so)

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        return types.SimpleNamespace(
            req_ids=["r0"], num_reqs=1,
            num_scheduled_tokens=np.array([4], dtype=np.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            query_start_loc_np=np.array([0, 4], dtype=np.int32),
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            prefill_len_np=np.array([4], dtype=np.int32),
            is_prefilling_np=np.array([True]),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            idx_mapping_np=np.array([0], dtype=np.intp),
        )


@pytest.mark.parametrize("subsystem,install_name,plans_attr,entries_attr,rows_attr,start_attr", [
    ("hs", "install_execute_model_wrapper_hs", "_pending_plans",
     "_hs_step_entries", "_hs_step_rows", "_hs_step_start"),
    ("qk", "install_execute_model_wrapper", "_pending_plans",
     "_qk_step_entries", "_qk_step_rows", "_qk_step_start"),
])
def test_drain_wrapper_skips_dummy_runs_and_forwards_v2_kwargs(
        subsystem, install_name, plans_attr, entries_attr, rows_attr, start_attr):
    """A dummy/profile pass must never reach the drain, and V2's extra kwargs must
    reach the real execute_model untouched."""
    if subsystem == "hs":
        from mia.graph.install_hs import install_execute_model_wrapper_hs as install_fn
    else:
        from mia.graph.install import install_execute_model_wrapper as install_fn

    runner = _ExecRunner()
    calls = []

    def fake_orig(scheduler_output, **kw):
        calls.append(kw)
        return "out"

    runner.execute_model = fake_orig

    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    registry = _fake_capture_registry()
    # Sentinel state left over from a previous REAL step: if a dummy pass reads
    # (drains) this, the test must fail loud, never pass by coincidence.
    setattr(registry, plans_attr, ["STALE_PLAN"])
    setattr(registry, entries_attr, ["STALE_RECORD"])
    setattr(registry, rows_attr, _STALE)
    setattr(registry, start_attr, _STALE)
    set_registry(worker, subsystem, registry)

    install_fn(runner, worker)
    drained = []
    drain_attr = "_hs_drain" if subsystem == "hs" else "_qk_drain"
    setattr(worker, drain_attr,
            types.SimpleNamespace(enqueue=lambda *a, **k: drained.append(a),
                                  per_request=False))

    # -- dummy_run: must not drain, must forward kwargs untouched --
    result = runner.execute_model(object(), dummy_run=True)
    assert result == "out"
    assert drained == [], "a dummy run must never reach the drain"
    assert calls[-1] == {"dummy_run": True}
    assert getattr(registry, plans_attr) == ["STALE_PLAN"], (
        "stale state from before the dummy pass must survive it unchanged")

    # -- is_profile (always paired with dummy_run=True in real V2, but checked
    # independently per the brief's "handle every non-real-work flag") --
    result = runner.execute_model(object(), dummy_run=True, is_profile=True,
                                  skip_attn_for_dummy_run=True, context_len=7)
    assert result == "out"
    assert drained == []
    assert calls[-1] == {"dummy_run": True, "is_profile": True,
                          "skip_attn_for_dummy_run": True, "context_len": 7}

    # -- a REAL step (dummy_run=False) with the SAME leftover plan must reach the
    # drain: proves the guard is keyed on dummy_run/is_profile, not "never drains" --
    result = runner.execute_model(object(), dummy_run=False, is_profile=False)
    assert result == "out"
    assert drained, "a real step with pending plans must reach the drain"
    assert calls[-1] == {"dummy_run": False, "is_profile": False}


@pytest.mark.parametrize("subsystem,install_name", [
    ("hs", "install_execute_model_wrapper_hs"),
    ("qk", "install_execute_model_wrapper"),
])
def test_drain_wrapper_blinds_stale_routing_during_dummy_pass_and_restores(subsystem, install_name):
    """The baked in-graph scatter op reads `registry.capture_index_all` on EVERY
    graph replay, including a dummy one (FULL cudagraph replays the compiled graph
    unconditionally on `dummy_run` — see vllm's model_runner.execute_model). The
    wrapper must zero it for exactly the duration of a dummy call (0 = every
    column discards, the registry's own documented inert value) and restore the
    ORIGINAL table byte-for-byte once the call returns."""
    if subsystem == "hs":
        from mia.graph.install_hs import install_execute_model_wrapper_hs as install_fn
    else:
        from mia.graph.install import install_execute_model_wrapper as install_fn

    runner = _ExecRunner()
    registry = _fake_capture_registry()
    stale = registry.capture_index_all.clone()
    seen_during_call = {}

    def fake_orig(scheduler_output, **kw):
        # Snapshot what the baked op would read AT REPLAY TIME, mid-call.
        seen_during_call["ci"] = registry.capture_index_all.clone()
        return "out"

    runner.execute_model = fake_orig
    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    set_registry(worker, subsystem, registry)

    install_fn(runner, worker)
    runner.execute_model(object(), dummy_run=True)

    assert seen_during_call["ci"].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]], (
        "capture_index_all must read all-inactive DURING a dummy pass")
    assert registry.capture_index_all.tolist() == stale.tolist(), (
        "the real routing table must be restored byte-for-byte after the dummy pass")


@pytest.mark.parametrize("subsystem,install_name", [
    ("hs", "install_execute_model_wrapper_hs"),
    ("qk", "install_execute_model_wrapper"),
])
def test_drain_wrapper_restores_stale_routing_on_exception(subsystem, install_name):
    """A dummy pass that raises mid-forward must still restore the real routing
    table — the never-drop / byte-identity contract cannot depend on the happy
    path."""
    if subsystem == "hs":
        from mia.graph.install_hs import install_execute_model_wrapper_hs as install_fn
    else:
        from mia.graph.install import install_execute_model_wrapper as install_fn

    runner = _ExecRunner()
    registry = _fake_capture_registry()
    stale = registry.capture_index_all.clone()

    def fake_orig(scheduler_output, **kw):
        raise RuntimeError("boom mid dummy forward")

    runner.execute_model = fake_orig
    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    set_registry(worker, subsystem, registry)

    install_fn(runner, worker)
    with pytest.raises(RuntimeError, match="boom mid dummy forward"):
        runner.execute_model(object(), dummy_run=True)

    assert registry.capture_index_all.tolist() == stale.tolist(), (
        "an exception mid dummy-forward must still restore the real routing table")


@pytest.mark.parametrize("subsystem,install_name,wrapped_attr", [
    ("hs", "install_execute_model_wrapper_hs", "_mia_hs_wrapped"),
    ("qk", "install_execute_model_wrapper", "_mia_qk_wrapped"),
])
def test_execute_model_wrapper_is_idempotent(subsystem, install_name, wrapped_attr):
    if subsystem == "hs":
        from mia.graph.install_hs import install_execute_model_wrapper_hs as install_fn
    else:
        from mia.graph.install import install_execute_model_wrapper as install_fn

    runner = _ExecRunner()
    runner.execute_model = lambda so, **kw: "out"
    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    set_registry(worker, subsystem, _fake_capture_registry())

    install_fn(runner, worker)
    wrapped_once = runner.execute_model
    install_fn(runner, worker)
    assert runner.execute_model is wrapped_once, "a second install must be a no-op"
    assert getattr(runner, wrapped_attr) is True
