"""Stateless helpers shared by the capture workers: request matching, atomic writes, metadata."""
from __future__ import annotations

import json as _json
import os
import re
from typing import Any, Iterator

import torch
import vllm.envs as envs
from safetensors.torch import save_file as _st_save

from mia._profiler import PROF, is_enabled
from mia.artifact_quant import (
    quant_nbytes,
    quantize,
    resolve_dtype,
    resolve_granularity,
    resolve_group_size,
)


def resolve_capture_quant(artifact: str):
    """Return ``(tag, gran, group_size)`` for an artifact family (``"qk"``/``"hs"``/``"score"``)."""
    return resolve_dtype(artifact), resolve_granularity(), resolve_group_size()


def quant_clone(x, tag, gran, group_size=128):
    """Quantize a captured GPU clone."""
    if tag is None:
        return x, None, None
    packed, scale, _zp, qmeta = quantize(x, tag, gran, group_size)
    return packed, scale, qmeta


def capture_bytes(*tensors):
    """Resident bytes of a (possibly quantized) captured artifact — packed + scale."""
    return quant_nbytes(*tensors)


_PINNED_STAGING: dict = {}


def _pinned_staging(dtype: torch.dtype, numel: int) -> torch.Tensor:
    buf = _PINNED_STAGING.get(dtype)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, pin_memory=torch.cuda.is_available())
        _PINNED_STAGING[dtype] = buf
    return buf[:numel]


def cpu_list_batched(tensors: list) -> list:
    """Batched byte-identical replacement for ``[t.cpu() for t in tensors]``."""
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


LAYER_PATTERNS = [
    re.compile(r"^model\.layers\.(\d+)$"),
    re.compile(r"^language_model\.model\.layers\.(\d+)$"),
    re.compile(r"^transformer\.h\.(\d+)$"),
    re.compile(r"^model\.decoder\.layers\.(\d+)$"),
]


def match_layer(name: str):
    for pat in LAYER_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


ATTN_PATTERNS = [
    re.compile(r"^transformer\.h\.(\d+)\.attn\.attn$"),

    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.attn$"),

    re.compile(r"^model\.layers\.(\d+)\.self_attn\.attn$"),
]

def match_attn(name: str):
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


def iter_matching_req_ids(state_dict: dict, external_req_id: str) -> Iterator[str]:
    """Yield internal req_ids in ``state_dict`` that match ``external_req_id``."""
    prefix = f"{external_req_id}-"
    for req_id in list(state_dict):
        if req_id == external_req_id or req_id.startswith(prefix):
            yield req_id


_HEX = frozenset("0123456789abcdef")


def _randomized() -> bool:
    return not envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION


def request_id_base(rid: str, randomized: bool | None = None):
    """``rid`` without vLLM's random ``-<8 hex>`` suffix, or None when it carries none."""
    if randomized is None:
        randomized = _randomized()
    if not randomized or len(rid) < 10 or rid[-9] != "-" or not set(rid[-8:]) <= _HEX:
        return None
    return rid[:-9]


def match_internal_ids(keys, key: str) -> list:
    """Ids in ``keys`` that are request ``key``: itself, or it plus vLLM's 8-hex suffix."""
    key, rnd = str(key), _randomized()
    return [k for k in keys if str(k) == key or request_id_base(str(k), rnd) == key]


def clear_states_for_req(state_dict: dict, external_req_id: str) -> None:
    """Pop all internal req_ids matching ``external_req_id`` from ``state_dict``."""
    for req_id in iter_matching_req_ids(state_dict, external_req_id):
        del state_dict[req_id]


def get_query_metadata(metadata: Any) -> tuple:
    """Return (query_start_loc, seq_lens) from ``attn_metadata``."""
    query_start_loc = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if query_start_loc is None and isinstance(metadata, dict):
        for entry in metadata.values():
            query_start_loc = getattr(entry, "query_start_loc", None)
            if query_start_loc is not None:
                seq_lens = getattr(entry, "seq_lens", None)
                break
    return query_start_loc, seq_lens


def compact_page_backed_cache(cpu_cache: dict) -> None:
    """Clone list-valued tensor leaves in ``cpu_cache`` in place, detaching them from aperture pages."""
    for top_val in cpu_cache.values():
        if not isinstance(top_val, dict):
            continue
        for mod_entry in top_val.values():
            if not isinstance(mod_entry, dict):
                continue
            for key, val in mod_entry.items():
                if isinstance(val, list):
                    mod_entry[key] = [t.clone() if torch.is_tensor(t) else t for t in val]


def clear_rank_artifact(run_dir: str, basename: str) -> None:
    """Remove a previous ``basename`` artifact from ``run_dir`` before a new one is written."""
    for ext in (".safetensors", ".json", ".pt"):
        try:
            os.remove(os.path.join(run_dir, basename + ext))
        except FileNotFoundError:
            pass


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
    """Yield (name, module, layer_num) for modules matching ``match_fn``."""
    for name, module in model.named_modules():
        layer_num = match_fn(name)
        if layer_num is None:
            continue
        if layer_filter and layer_num not in layer_filter:
            continue
        yield name, module, layer_num


def save_safetensors_atomic(flat_dict: dict, meta: dict, run_dir: str, basename: str) -> None:
    """Write ``flat_dict`` as safetensors plus a JSON ``meta`` sidecar, atomically."""
    out_path = os.path.join(run_dir, f"{basename}.safetensors")
    meta_path = os.path.join(run_dir, f"{basename}.json")
    tmp_st = out_path + ".tmp"
    tmp_meta = meta_path + ".tmp"

    with PROF.timed("worker.disk_write.safetensors"):
        _st_save(flat_dict, tmp_st)
        os.rename(tmp_st, out_path)

    try:
        if is_enabled():
            meta = dict(meta)
            meta["profile"] = PROF.summary_only()
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

