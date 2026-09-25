"""Stateless helpers shared by qk_capture_worker and hs_capture_worker:
matching internal request IDs by ``{external_req_id}-`` prefix, writing atomic
artifacts via tmp+rename, and pulling query_start_loc/seq_lens from
ForwardContext.attn_metadata (walking the per-layer dict for hybrid models).
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterator

import torch

from mia._profiler import PROF


# ---------------------------------------------------------------------------
# Artifact quantization (opt-in via MIA_ARTIFACT_DTYPE; default off/no-op)
# ---------------------------------------------------------------------------
# Quantizes the captured GPU clone at capture (shrinks GPU residency and the
# deferred D2H copy) and dequantizes back to float inside the worker at
# retrieval/flush, so every downstream consumer sees a normal float tensor and
# stays byte-identical.


def resolve_capture_quant(artifact: str):
    """Return ``(tag, gran, group_size)`` for an artifact family
    (``"qk"``/``"hs"``/``"score"``). ``tag is None`` means native/off (no-op)."""
    from mia.artifact_quant import (
        resolve_dtype, resolve_granularity, resolve_group_size)
    return resolve_dtype(artifact), resolve_granularity(), resolve_group_size()


def quant_clone(x, tag, gran, group_size=128):
    """Quantize a captured GPU clone. Returns ``(packed, scale, qmeta)``
    (``qmeta is None`` when ``tag is None`` → ``packed is x`` unchanged)."""
    if tag is None:
        return x, None, None
    from mia.artifact_quant import quantize
    packed, scale, _zp, qmeta = quantize(x, tag, gran, group_size)
    return packed, scale, qmeta


def capture_bytes(*tensors):
    """Resident bytes of a (possibly quantized) captured artifact — packed + scale."""
    from mia.artifact_quant import quant_nbytes
    return quant_nbytes(*tensors)


# ---------------------------------------------------------------------------
# Pinned host staging buffer
# ---------------------------------------------------------------------------
# Grow-only pinned buffer, keyed by dtype, for a batched GPU->host D2H move. Safe to
# reuse across calls because there is one worker per process and retrieval runs one
# call at a time on the serial engine loop, syncing+cloning before returning -> no
# per-call cudaHostAlloc. Unused directly by this module; the capture bank's GPU->host
# mover owns the batching logic and reuses this buffer.
_PINNED_STAGING: dict = {}


def _pinned_staging(dtype: torch.dtype, numel: int) -> torch.Tensor:
    buf = _PINNED_STAGING.get(dtype)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, pin_memory=torch.cuda.is_available())
        _PINNED_STAGING[dtype] = buf
    return buf[:numel]


def cpu_list_batched(tensors: list) -> list:
    """Batched byte-identical replacement for ``[t.cpu() for t in tensors]``.

    cat on device -> one ``non_blocking`` copy into a reused pinned staging buffer -> sync
    -> split -> owned clones. Collapses many launch-bound per-tensor D2H copies into one
    bandwidth-bound transfer; each clone owns its storage so the staging pool is safe to
    reuse on the next call.

    Falls back to the exact per-tensor ``.cpu()`` when the list is empty, holds a non-tensor,
    the first element is already host, or the elements differ in device / dtype / trailing
    shape (not cat-able -- e.g. a streaming-drain CUDA/CPU mix). Bit-for-bit the old path in
    every fallback case.
    """
    if not tensors:
        return []
    t0 = tensors[0]
    if not torch.is_tensor(t0) or not t0.is_cuda:
        return [t.cpu() if torch.is_tensor(t) else t for t in tensors]
    dev, dt, trail = t0.device, t0.dtype, t0.shape[1:]
    for t in tensors:
        if (not torch.is_tensor(t)) or t.device != dev or t.dtype != dt \
                or t.dim() < 1 or t.shape[1:] != trail:
            return [t.cpu() if torch.is_tensor(t) else t for t in tensors]
    lengths = [t.shape[0] for t in tensors]
    flat = torch.cat(tensors, dim=0)
    staging = _pinned_staging(dt, flat.numel()).view(flat.shape)
    staging.copy_(flat, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return [s.clone() for s in staging.split(lengths, dim=0)]


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
    re.compile(r"^transformer\.h\.(\d+)\.attn\.attn$"),

    # OPT: model.decoder.layers.<i>.self_attn
    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.attn$"),

    # Qwen/LLaMA: model.layers.<i>.self_attn
    re.compile(r"^model\.layers\.(\d+)\.self_attn\.attn$"),
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


def compact_page_backed_cache(cpu_cache: dict) -> None:
    """Clone every list-valued tensor leaf in ``cpu_cache`` IN PLACE to owned storage,
    breaking any sharing with a capture-aperture host page.

    ``pickle``/``torch.save`` of a tensor that is a narrow VIEW into a multi-MiB pinned aperture
    page re-serializes the WHOLE page per view -- up to ``page_bytes`` per tensor on disk. The
    DEFAULT disk path never hits this: the writer process packs via ``torch.cat``
    (``graph/artifact_writer.py``) into fresh storage before it ever touches disk. This helper
    exists for the RARE inline-fallback path only (writer process off, or its child
    unavailable), which serializes the raw ``cpu_cache`` directly -- call it right before
    ``save_pt_atomic``/``save_safetensors_atomic`` there, then release the request's aperture
    pages (now safe: the values are cloned + about to be written).

    ``.clone()`` allocates fresh storage with the same values -> byte-identical, just no
    longer aliasing a page. Non-tensor entries pass through untouched. Shape is exactly
    ``cpu_cache``'s two-levels-of-dict nesting
    (``{"hs_cache"/"qk_cache": {module_name: {key: [tensor, ...], ...}}}``)."""
    for top_val in cpu_cache.values():
        if not isinstance(top_val, dict):
            continue
        for mod_entry in top_val.values():
            if not isinstance(mod_entry, dict):
                continue
            for key, val in mod_entry.items():
                if isinstance(val, list):
                    mod_entry[key] = [t.clone() if torch.is_tensor(t) else t for t in val]


def save_pt_atomic(cpu_cache: dict, out_path: str) -> None:
    """Write ``cpu_cache`` to ``out_path`` via tmp+fsync+rename for atomicity."""
    with PROF.timed("worker.disk_write.pt"):
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            torch.save(cpu_cache, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, out_path)
    try:
        PROF.gauge("disk.bytes.pt", os.path.getsize(out_path))
    except OSError:
        pass


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

    When MIA_PROFILE=1 the JSON sidecar is enriched with a snapshot
    of the worker-side profiler so post-hoc analysis doesn't need a
    separate RPC fetch.
    """
    import json as _json
    from safetensors.torch import save_file as _st_save

    out_path = os.path.join(run_dir, f"{basename}.safetensors")
    meta_path = os.path.join(run_dir, f"{basename}.json")
    tmp_st = out_path + ".tmp"
    tmp_meta = meta_path + ".tmp"

    with PROF.timed("worker.disk_write.safetensors"):
        _st_save(flat_dict, tmp_st)
        os.rename(tmp_st, out_path)

    # Bake a profile snapshot into the JSON meta so the harness reads perf data without
    # a separate fetch. Cumulative at write time -- readers subtract the previous
    # snapshot for per-cell metrics, or call PROF.reset() between cells.
    try:
        from mia._profiler import PROF as _PROF, is_enabled as _en
        if _en():
            meta = dict(meta)
            meta["profile"] = _PROF.summary_only()
    except Exception:
        pass

    with open(tmp_meta, "w") as f:
        _json.dump(meta, f)
    os.rename(tmp_meta, meta_path)

    try:
        PROF.gauge("disk.bytes.safetensors", os.path.getsize(out_path))
        PROF.gauge("disk.bytes.json", os.path.getsize(meta_path))
    except OSError:
        pass
