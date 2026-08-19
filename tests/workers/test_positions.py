# tests/workers/test_positions.py
"""Pass-view construction against faked engine structures: chunked
prefill, mixed batches, restart reset, post-stop pass idempotence, and
the scope truth tables on real views.
"""
import torch

from vllm_hook_plugins.core.interpreter.scopes import scope_rows
from vllm_hook_plugins.core.schema import ScopeSpec
from vllm_hook_plugins.workers.positions import (
    PositionTracker,
    RequestPassView,
    build_pass_views,
)


class FakeMetadata:
    def __init__(self, query_start_loc):
        self.query_start_loc = torch.tensor(query_start_loc)
        self.seq_lens = None


class FakeInputBatch:
    def __init__(self, req_ids, num_computed, num_prompt):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed
        self.num_prompt_tokens = num_prompt


class FakeModelRunner:
    def __init__(self, input_batch):
        self.input_batch = input_batch


def _pass(tracker, req_specs):
    """req_specs: [(req_id, num_computed, chunk_len, prompt_len)] -> views."""
    starts = [0]
    for _, _, chunk_len, _ in req_specs:
        starts.append(starts[-1] + chunk_len)
    runner = FakeModelRunner(FakeInputBatch(
        [r[0] for r in req_specs],
        [r[1] for r in req_specs],
        [r[3] for r in req_specs],
    ))
    return build_pass_views(runner, FakeMetadata(starts), tracker)


def test_warmup_pass_returns_empty():
    tracker = PositionTracker()
    runner = FakeModelRunner(FakeInputBatch(["a"], [0], [4]))
    assert build_pass_views(runner, {}, tracker) == []


def test_single_prefill_pass():
    tracker = PositionTracker()
    (view,) = _pass(tracker, [("a", 0, 6, 6)])
    assert view == RequestPassView(
        req_id="a", row_slice=slice(0, 6), positions=range(0, 6), prompt_len=6, is_restart=False
    )


def test_chunked_prefill_positions_continue():
    tracker = PositionTracker()
    (first,) = _pass(tracker, [("a", 0, 4, 8)])
    (second,) = _pass(tracker, [("a", 4, 4, 8)])
    assert first.positions == range(0, 4)
    assert second.positions == range(4, 8)
    assert not first.is_restart and not second.is_restart


def test_mixed_batch_row_slices():
    tracker = PositionTracker()
    views = _pass(tracker, [("a", 0, 5, 5), ("b", 7, 1, 4), ("c", 0, 3, 6)])
    assert [v.req_id for v in views] == ["a", "b", "c"]
    assert views[0].row_slice == slice(0, 5)
    assert views[1].row_slice == slice(5, 6)
    assert views[1].positions == range(7, 8)  # decode: one row past the prompt
    assert views[2].row_slice == slice(6, 9)
    assert views[2].prompt_len == 6


def test_restart_detected_and_mark_rewound():
    tracker = PositionTracker()
    _pass(tracker, [("a", 0, 8, 8)])
    _pass(tracker, [("a", 8, 1, 8)])
    # preemption: vLLM recomputes from position 0
    (view,) = _pass(tracker, [("a", 0, 8, 8)])
    assert view.is_restart
    # after the rewind the request proceeds normally
    (view,) = _pass(tracker, [("a", 8, 1, 8)])
    assert not view.is_restart


def test_post_stop_extra_pass_is_not_a_restart():
    tracker = PositionTracker()
    _pass(tracker, [("a", 0, 4, 4)])
    _pass(tracker, [("a", 4, 1, 4)])
    _pass(tracker, [("a", 5, 1, 4)])
    # the post-stop pass computes the next position like any decode step
    (view,) = _pass(tracker, [("a", 6, 1, 4)])
    assert not view.is_restart
    assert view.positions == range(6, 7)


def test_position_keyed_state_is_idempotent_across_replay():
    """Replayed positions overwrite; final assembly truncated to the
    reported token count drops the post-stop surplus row.
    """
    tracker = PositionTracker()
    buffer = {}

    def consume(views):
        for view in views:
            for offset, position in enumerate(view.positions):
                buffer[position] = ("pass", position)

    consume(_pass(tracker, [("a", 0, 4, 4)]))
    consume(_pass(tracker, [("a", 4, 1, 4)]))
    consume(_pass(tracker, [("a", 5, 1, 4)]))  # post-stop extra pass
    # replay after preemption: same keys, same values — idempotent
    consume(_pass(tracker, [("a", 0, 5, 4)]))
    expected_len = 5  # prompt 4 + 2 generated - 1
    kept = sorted(p for p in buffer if p < expected_len)
    assert kept == [0, 1, 2, 3, 4]


def test_views_cached_per_metadata_object():
    tracker = PositionTracker()
    runner = FakeModelRunner(FakeInputBatch(["a"], [0], [4]))
    metadata = FakeMetadata([0, 4])
    first = build_pass_views(runner, metadata, tracker)
    second = build_pass_views(runner, metadata, tracker)
    assert first is second
    assert tracker.pass_key == id(metadata)
    # a second build did not double-observe: no restart flagged next pass
    (view,) = _pass(tracker, [("a", 4, 1, 4)])
    assert not view.is_restart


def test_drop_forgets_request():
    tracker = PositionTracker()
    _pass(tracker, [("a", 0, 8, 8)])
    tracker.drop("a")
    (view,) = _pass(tracker, [("a", 0, 8, 8)])
    assert not view.is_restart


def test_missing_input_batch_fields_yield_no_views():
    tracker = PositionTracker()

    class BareRunner:
        pass

    assert build_pass_views(BareRunner(), FakeMetadata([0, 4]), tracker) == []


# ---------------------------------------------------------------------------
# Scope truth tables over realistic pass shapes
# ---------------------------------------------------------------------------


def _views_for_generation(tracker, prompt_len, chunks, decode_steps):
    views = []
    computed = 0
    for chunk in chunks:
        views.extend(_pass(tracker, [("a", computed, chunk, prompt_len)]))
        computed += chunk
    for _ in range(decode_steps):
        views.extend(_pass(tracker, [("a", computed, 1, prompt_len)]))
        computed += 1
    return views


def _selected_positions(views, scope):
    selected = []
    for view in views:
        rows = scope_rows(view, scope)
        if rows is None:
            continue
        selected.extend(list(view.positions)[rows])
    return selected


def test_scope_truth_table_chunked_prefill_plus_decode():
    prompt_len = 8
    chunks = [5, 3]
    decode_steps = 3  # positions 8, 9, 10

    def positions_for(scope):
        return _selected_positions(
            _views_for_generation(PositionTracker(), prompt_len, chunks, decode_steps), scope
        )

    assert positions_for(ScopeSpec("all", {})) == list(range(0, 11))
    assert positions_for(ScopeSpec("after_prompt", {})) == [8, 9, 10]
    assert positions_for(ScopeSpec("from_position", {"position": 6})) == [6, 7, 8, 9, 10]
    # last_k selects absolute positions >= prompt_len - k, invariant to chunking
    assert positions_for(ScopeSpec("last_k", {"k": 2})) == [6, 7, 8, 9, 10]
    assert positions_for(ScopeSpec("last_k", {"k": 1})) == [7, 8, 9, 10]


def test_scope_truth_table_decode_only_pass():
    tracker = PositionTracker()
    _pass(tracker, [("a", 0, 8, 8)])
    (decode_view,) = _pass(tracker, [("a", 8, 1, 8)])
    assert scope_rows(decode_view, ScopeSpec("all", {})) == slice(0, 1)
    assert scope_rows(decode_view, ScopeSpec("after_prompt", {})) == slice(0, 1)
    assert scope_rows(decode_view, ScopeSpec("last_k", {"k": 5})) == slice(0, 1)
    assert scope_rows(decode_view, ScopeSpec("from_position", {"position": 9})) is None
