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
import re
import threading
from typing import Any, Iterator

import torch


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


LAYER_PATTERNS = [
    # LLaMA / Qwen2.x / Granite: model.layers.<i>
    re.compile(r"^model\.layers\.(\d+)$"),
    # Qwen3.5 multimodal (Qwen3_5ForConditionalGeneration): language_model.model.layers.<i>
    re.compile(r"^language_model\.model\.layers\.(\d+)$"),
    # GPT-2: transformer.h.<i>
    re.compile(r"^transformer\.h\.(\d+)$"),
    # OPT: model.decoder.layers.<i>
    re.compile(r"^model\.decoder\.layers\.(\d+)$"),
]


def match_layer(name: str):
    for pat in LAYER_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


ATTN_PATTERNS = [
    # GPT-2: transformer.h.<i>.attn
    re.compile(r"^transformer\.h\.(\d+)\.attn.attn$"),

    # OPT: model.decoder.layers.<i>.self_attn
    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn.attn$"),

    # Qwen/LLaMA: model.layers.<i>.self_attn
    re.compile(r"^model\.layers\.(\d+)\.self_attn.attn$"),
]

def match_attn(name: str):
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


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


def iter_matched_modules(model, match_fn, layer_filter=None):
    """Yield (name, module, layer_num) for modules matching ``match_fn``.

    ``match_fn(name)`` returns the layer index or None. ``layer_filter`` is
    optional; when truthy, only modules whose layer index is in the filter
    set are yielded. ``layer_filter`` is checked with ``layer_num in filter``.
    """
    for name, module in model.named_modules():
        layer_num = match_fn(name)
        if layer_num is None:
            continue
        if layer_filter and layer_num not in layer_filter:
            continue
        yield name, module, layer_num


def save_safetensors_atomic(flat_dict: dict, meta: dict, run_dir: str, basename: str) -> None:
    """Write ``flat_dict`` to ``{run_dir}/{basename}.safetensors`` and ``meta``
    to ``{run_dir}/{basename}.json``, both via tmp+rename for atomicity.
    """
    import json as _json
    from safetensors.torch import save_file as _st_save

    out_path = os.path.join(run_dir, f"{basename}.safetensors")
    meta_path = os.path.join(run_dir, f"{basename}.json")
    tmp_st = out_path + ".tmp"
    tmp_meta = meta_path + ".tmp"

    _st_save(flat_dict, tmp_st)
    os.rename(tmp_st, out_path)

    with open(tmp_meta, "w") as f:
        _json.dump(meta, f)
    os.rename(tmp_meta, meta_path)


def background_save_loop(worker, pt_filename: str) -> None:
    """Drain ``worker._save_queue`` and write artifacts to disk.

    Activated by VLLM_HOOK_ASYNC_SAVE=1. Each queue item is
    (run_id, cpu_cache, run_dir). Workers must define
    ``_save_safetensors(cpu_cache, run_dir)`` — used when
    VLLM_HOOK_USE_SAFETENSORS=1.
    """
    while True:
        run_id, cpu_cache, run_dir = worker._save_queue.get()
        try:
            os.makedirs(run_dir, exist_ok=True)
            if os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1":
                worker._save_safetensors(cpu_cache, run_dir)
            else:
                save_pt_atomic(cpu_cache, os.path.join(run_dir, pt_filename))
        except Exception as e:
            print(f"background save failed for {run_id}: {e}")
        finally:
            worker._save_queue.task_done()
