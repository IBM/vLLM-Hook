"""Per-request demux + finish-tracking + assembly for the off-loop delivery pipeline.

Rows arrive step-by-step keyed by (req_id, layer); PerRequestIndex buffers them in append
order, and once a request is marked finished its rows are ready to assemble (torch.cat in
append order) and hand off. FINISH is enqueued after a request's last step (FIFO), so by the
time mark_finished fires all of that request's rows are already present. Pure logic — no
torch ops beyond cat, no GPU, no disk.

QK ASSEMBLY: a single (req, layer) needs TWO independent row streams: q (post-RoPE query, emitted
only on emit_q steps) and k_full (the growing key history, appended every step). Both stage
through the SAME unmodified `note_rows` by using DISTINCT tuple layer keys — `("q", layer)` /
`("k", layer)`. `assemble_qk` reads those two streams off a request's entry and rebuilds `k_all`
via the same growing-prefix pattern `aperture_reader.load_multilayer_qk_aperture_artifact` uses: `k_full =
torch.cat(k_blocks)`, `k_all = [k_full[:L] for L in prefix_ends]`.

prefix_ends CONTRACT: the caller passes `kmeta={"prefix_ends": [...]}` on the `("k", layer)`
`note_rows` call as the FULL CUMULATIVE list-so-far, not a per-step delta (step 1 passes `[3]`,
step 2 `[3, 4]`, step 3 `[3, 4, 5]`); `kmeta=None` on a step with no new prefix boundary. This
rides `note_rows`'s existing LAST-WRITE-WINS kmeta store as-is — the final call's list is the one
that survives at finish time, so no accumulation logic is needed in this module."""
from __future__ import annotations
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def assemble_qk(entry: dict) -> dict:
    """Assemble one request's per-layer Q/K capture from a `PerRequestIndex` entry (the dict shape
    `PerRequestIndex._entry` builds / stores in `_entries[req_id]`: `{"layers", "finished", "kmeta"}`).

    Reads the `("q", layer)` / `("k", layer)` staged streams (see the module docstring for the
    staging + prefix_ends contracts) and returns
    `{layer: {"q": <cat q rows>, "k_all": [k_full[:L] for L in prefix_ends]}}` for every layer that
    has a `("k", layer)` stream. `k_full` and `q` are each `torch.cat` of their blocks in append
    (step) order — the same order `note_rows` recorded them, matching `pop_deliverable`'s plain-cat
    convention. A captured layer always emits at least one q row (all_tokens: every step; last_token:
    the final prefill chunk + every decode step), so a layer with k rows but zero q rows is a genuine
    anomaly, not a normal empty case: this raises `ValueError` rather than fabricating an empty-q
    tensor (unlike `aperture_reader`, which reads a dedicated `q_row_shape` from its sidecar header and can
    legitimately report an empty q).
    """
    layers = entry["layers"]
    kmeta = entry["kmeta"]

    layer_nums = {key[1] for key in layers if isinstance(key, tuple) and len(key) == 2
                  and key[0] in ("q", "k")}

    out: dict = {}
    for layer in layer_nums:
        k_blocks = layers.get(("k", layer))
        if not k_blocks:
            raise ValueError(f"assemble_qk: layer {layer} has no ('k', {layer}) rows "
                              f"(k is required every step)")
        k_full = k_blocks[0] if len(k_blocks) == 1 else torch.cat(k_blocks, 0)

        q_blocks = layers.get(("q", layer))
        if not q_blocks:
            raise ValueError(f"assemble_qk: layer {layer} has no ('q', {layer}) rows "
                              f"(q is required at least once per captured layer)")
        q = q_blocks[0] if len(q_blocks) == 1 else torch.cat(q_blocks, 0)

        prefix_ends = kmeta.get(("k", layer), {}).get("prefix_ends", [])
        k_all = [k_full[:L] for L in prefix_ends]
        out[layer] = {"q": q, "k_all": k_all}
    return out


class PerRequestIndex:
    def __init__(self):
        self._entries: dict[str, dict[str, Any]] = {}
        self._deliverable: list[str] = []

    def _entry(self, req_id):
        e = self._entries.get(req_id)
        if e is None:
            e = {"layers": {}, "finished": False, "kmeta": {}}
            self._entries[req_id] = e
        return e

    def note_rows(self, req_id, layer, rows_cpu, kmeta=None):
        e = self._entry(req_id)
        e["layers"].setdefault(layer, []).append(rows_cpu)
        if kmeta is not None:
            e["kmeta"][layer] = kmeta

    def mark_finished(self, req_id):
        e = self._entry(req_id)
        if not e["finished"]:
            e["finished"] = True
            self._deliverable.append(req_id)

    def pop_deliverable(self) -> list[tuple[str, dict]]:
        out = []
        for req_id in self._deliverable:
            e = self._entries[req_id]
            assembled = {}
            for layer, blocks in e["layers"].items():
                assembled[layer] = blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)
            out.append((req_id, assembled))
        self._deliverable = []
        return out

    def pop_deliverable_qk(self) -> list[tuple[str, dict]]:
        """QK counterpart of `pop_deliverable`: same finish-drain (FIFO order, `_deliverable`
        cleared after read) but assembles each entry via `assemble_qk` (per-layer `{"q", "k_all"}`)
        instead of `pop_deliverable`'s plain per-layer `torch.cat`. Use this to pop entries staged
        through the `("q", layer)` / `("k", layer)` convention (a QK worker's requests);
        `pop_deliverable` is unchanged and remains correct for plain single-stream-per-layer entries
        (e.g. HS).

        POISON ISOLATION: one malformed finished entry -- e.g. a `last_token` request marked finished
        mid-prefill (k rows, zero q rows) via `finalize_all` at shutdown -- makes `assemble_qk` raise
        `ValueError`. That raise must NOT escape this pop: it would leave `_deliverable` uncleared (the
        clear runs only after the loop), so every later pop re-processes the same poison entry and
        re-raises == a PERMANENT WEDGE of QK RPC delivery, and the entry is never freed so residency
        never drops. So each entry is isolated: on failure log LOUD + DROP and FREE that one entry (so
        residency still falls and it can never re-poison a later pop) and continue with the rest;
        `_deliverable` is ALWAYS cleared. `assemble_qk`'s fail-loud validation is unchanged -- it just
        no longer wedges the pipeline. Mirrors the "one bad entry never wedges the consumer" discipline
        the drain's finalize/demux paths already use."""
        out = []
        try:
            for req_id in self._deliverable:
                try:
                    assembled = assemble_qk(self._entries[req_id])
                except Exception:  # noqa: BLE001 -- isolate ONE poison entry; never wedge QK delivery
                    logger.exception(
                        "pop_deliverable_qk: DROPPING poison QK entry req_id=%r (assemble_qk failed); "
                        "its delivery is dropped and its residency freed, the pipeline continues",
                        req_id)
                    self._entries.pop(req_id, None)   # FREE so residency drops + it never re-poisons
                    continue
                out.append((req_id, assembled))
        finally:
            self._deliverable = []   # ALWAYS clear -- a poison entry can never re-enter a later pop
        return out

    def free(self, req_id):
        self._entries.pop(req_id, None)

    def live_req_ids(self) -> set:
        return set(self._entries.keys())
