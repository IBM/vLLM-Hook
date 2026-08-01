"""Per-forward-pass request view: the single position authority.

Every consumer of positions — intervention scopes, gates, capture —
reads ``RequestPassView``; nothing else derives positions independently.
Built from ``ForwardContext.attn_metadata`` (``get_query_metadata`` in
``_common`` handles the hybrid-model dict walk) plus
``input_batch.num_computed_tokens_cpu`` / ``num_prompt_tokens``, and
cached by ``id(attn_metadata)`` — one build per forward step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from vllm_hook_plugins.workers._common import get_query_metadata

logger = logging.getLogger("vllm_hook.positions")


@dataclass(frozen=True)
class RequestPassView:
    req_id: str
    row_slice: slice  # rows of this pass's flattened token batch
    positions: range  # absolute positions those rows occupy
    prompt_len: int
    is_restart: bool  # pass starts below the request's high-water mark


class PositionTracker:
    """Per-request high-water marks of computed positions.

    ``observe(req_id, positions)`` returns True when ``positions.start``
    falls below the mark — vLLM recomputed the request after preemption —
    in which case callers must reset that request's accumulated state
    before consuming the pass. ``drop(req_id)`` forgets the request.
    """

    def __init__(self):
        self._marks: dict = {}  # req_id -> next absolute position expected
        # One build_pass_views result per forward step, keyed by
        # id(attn_metadata): vLLM allocates fresh metadata per step, so the
        # id is a cheap fingerprint. Caching here also keeps observe() at
        # exactly one call per (request, pass). The reference to the keyed
        # object is held alongside so its id cannot be reused by a later
        # allocation while the cache entry is live.
        self._views_key = None
        self._views_obj = None
        self._views: list = []

    def observe(self, req_id: str, positions: range) -> bool:
        mark = self._marks.get(req_id, 0)
        is_restart = positions.start < mark
        self._marks[req_id] = positions.stop if is_restart else max(mark, positions.stop)
        return is_restart

    @property
    def pass_key(self):
        """Fingerprint of the pass whose views are cached — stable across
        every hook fire within one forward step. Valid after
        ``build_pass_views``.
        """
        return self._views_key

    def drop(self, req_id: str) -> None:
        self._marks.pop(req_id, None)


def build_pass_views(model_runner, attn_metadata, tracker: PositionTracker) -> list:
    """One view per request in this pass. Returns [] on warmup passes
    (no ``query_start_loc``).
    """
    cache_key = id(attn_metadata)
    if tracker._views_key == cache_key and tracker._views_obj is attn_metadata:
        return tracker._views

    query_start_loc, _seq_lens = get_query_metadata(attn_metadata)
    if query_start_loc is None:
        return []

    try:
        input_batch = model_runner.input_batch
        req_ids = input_batch.req_ids
        num_computed = input_batch.num_computed_tokens_cpu
        num_prompt = input_batch.num_prompt_tokens
    except AttributeError as exc:
        logger.warning("cannot build pass views; input_batch is missing %s", exc)
        return []

    views = []
    num_reqs = min(len(query_start_loc) - 1, len(req_ids))
    for i in range(num_reqs):
        start_row = int(query_start_loc[i].item())
        end_row = int(query_start_loc[i + 1].item())
        if end_row <= start_row:
            continue
        req_id = req_ids[i]
        pos_start = int(num_computed[i])
        positions = range(pos_start, pos_start + (end_row - start_row))
        views.append(
            RequestPassView(
                req_id=req_id,
                row_slice=slice(start_row, end_row),
                positions=positions,
                prompt_len=int(num_prompt[i]),
                is_restart=tracker.observe(req_id, positions),
            )
        )

    tracker._views_key = cache_key
    tracker._views_obj = attn_metadata
    tracker._views = views
    return views
