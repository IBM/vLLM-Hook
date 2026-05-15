"""Shared helpers for probe workers.

Both probe_hookqk_worker and probe_hidden_states_worker do the same things:
- match internal request IDs by ``{external_req_id}-`` prefix
- write atomic .pt artifacts with tmp+rename
- spawn a background save thread for VLLM_HOOK_ASYNC_SAVE=1
- pull query_start_loc/seq_lens from ForwardContext.attn_metadata,
  walking the per-layer dict for hybrid models

These helpers factor out that duplication. They are stateless utilities
where possible; the async-save thread setup mutates ``self`` because the
queue and thread live on the worker instance.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Any, Iterator

import torch


# ---------------------------------------------------------------------------
# Per-request bookkeeping
# ---------------------------------------------------------------------------


def iter_matching_req_ids(state_dict: dict, external_req_id: str) -> Iterator[str]:
    """Yield internal req_ids in ``state_dict`` that match ``external_req_id``.

    vLLM internally transforms the user-provided request_id into either the
    same id (v0.12+) or ``{request_id}-{random_suffix}`` (older versions).
    We accept both: exact equality OR ``{external_req_id}-`` prefix.
    """
    prefix = f"{external_req_id}-"
    for req_id in list(state_dict):
        if req_id == external_req_id or req_id.startswith(prefix):
            yield req_id


def clear_states_for_req(state_dict: dict, external_req_id: str) -> None:
    """Pop all internal req_ids matching ``external_req_id`` from ``state_dict``."""
    for req_id in iter_matching_req_ids(state_dict, external_req_id):
        del state_dict[req_id]


# ---------------------------------------------------------------------------
# Forward-context metadata extraction
# ---------------------------------------------------------------------------


def get_query_metadata(metadata: Any) -> tuple:
    """Return (query_start_loc, seq_lens) from ``attn_metadata``.

    For hybrid models (e.g. Qwen3.5), linear-attention layers have no entry
    keyed by their own module name, so we walk the dict and grab the metadata
    from any entry that has ``query_start_loc``. Returns (None, None) when no
    such entry exists (warmup, non-attention pass).
    """
    query_start_loc = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if query_start_loc is None and isinstance(metadata, dict):
        for entry in metadata.values():
            query_start_loc = getattr(entry, "query_start_loc", None)
            if query_start_loc is not None:
                seq_lens = getattr(entry, "seq_lens", None)
                break
    return query_start_loc, seq_lens


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def save_pt_atomic(cpu_cache: dict, out_path: str) -> None:
    """Write ``cpu_cache`` to ``out_path`` via tmp+fsync+rename for atomicity."""
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(cpu_cache, f)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, out_path)


def init_async_save_thread(worker, target, thread_name: str) -> None:
    """Start a background save thread on ``worker`` if not already started.

    Activated by VLLM_HOOK_ASYNC_SAVE=1. The thread runs ``target`` (a bound
    method on ``worker`` that consumes ``worker._save_queue``).
    """
    if os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") != "1":
        return
    if getattr(worker, "_io_thread_started", False):
        return
    worker._save_queue = queue.Queue(maxsize=4)
    worker._io_thread = threading.Thread(target=target, daemon=True, name=thread_name)
    worker._io_thread.start()
    worker._io_thread_started = True
