"""Per-request demux + finish-tracking + assembly for the off-loop delivery pipeline."""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class QKDeliveryRefused(ValueError):
    """A request's Q/K cannot be assembled into the history eager would report."""


def assemble_qk(entry: dict) -> dict:
    """Assemble one request's per-layer Q/K capture from a PerRequestIndex entry."""
    layers = entry["layers"]
    kmeta = entry["kmeta"]
    first_nc = entry.get("first_nc", {})

    layer_nums = sorted({key[1] for key in layers if isinstance(key, tuple) and len(key) == 2
                         and key[0] in ("q", "k")})

    out: dict = {}
    empty = []
    for layer in layer_nums:
        k_blocks = layers.get(("k", layer))
        if not k_blocks:
            raise ValueError(f"assemble_qk: layer {layer} has no ('k', {layer}) rows "
                              f"(k is required every step)")
        nc = int(first_nc.get(("k", layer), 0) or 0)
        if nc > 0:
            raise QKDeliveryRefused(
                f"layer {layer}: its first captured step already had {nc} cached tokens "
                f"(prefix caching), whose keys were never captured")
        k_full = k_blocks[0] if len(k_blocks) == 1 else torch.cat(k_blocks, 0)
        prefix_ends = kmeta.get(("k", layer), {}).get("prefix_ends", [])

        q_blocks = layers.get(("q", layer))
        if not q_blocks:
            if prefix_ends:
                raise ValueError(f"assemble_qk: layer {layer} has prefix ends but no q rows")
            empty.append(layer)
            continue
        q = q_blocks[0] if len(q_blocks) == 1 else torch.cat(q_blocks, 0)
        if prefix_ends and k_full.shape[0] != int(prefix_ends[-1]):
            raise QKDeliveryRefused(
                f"layer {layer}: {k_full.shape[0]} key rows for a {prefix_ends[-1]}-token history "
                f"(its prefill ran again after preemption)")
        k_all = [k_full[:L] for L in prefix_ends]
        out[layer] = {"q": q, "k_all": k_all}
    if empty and out:
        raise ValueError(f"assemble_qk: layers {empty} have no q rows while {sorted(out)} do "
                         f"(q is required at least once per captured layer)")
    return out


class PerRequestIndex:
    def __init__(self):
        self._entries: dict[str, dict[str, Any]] = {}
        self._deliverable: list[str] = []

    def _entry(self, req_id):
        e = self._entries.get(req_id)
        if e is None:
            e = {"layers": {}, "finished": False, "kmeta": {}, "first_nc": {}}
            self._entries[req_id] = e
        return e

    def note_rows(self, req_id, layer, rows_cpu, kmeta=None, num_computed=None):
        e = self._entry(req_id)
        e["layers"].setdefault(layer, []).append(rows_cpu)
        if kmeta is not None:
            e["kmeta"][layer] = kmeta
        if num_computed is not None:
            e["first_nc"].setdefault(layer, int(num_computed))

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
        """QK counterpart of ``pop_deliverable``: finished requests assembled via ``assemble_qk``."""
        out = []
        try:
            for req_id in self._deliverable:
                try:
                    assembled = assemble_qk(self._entries[req_id])
                except Exception as e:  # noqa: BLE001
                    logger.warning("pop_deliverable_qk: QK entry req_id=%r not assembled: %s",
                                   req_id, e)
                    self._entries.pop(req_id, None)
                    out.append((req_id, e))
                    continue
                out.append((req_id, assembled))
        finally:
            self._deliverable = []
        return out

    def free(self, req_id):
        self._entries.pop(req_id, None)

    def live_req_ids(self) -> set:
        return set(self._entries.keys())

