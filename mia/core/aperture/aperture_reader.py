"""Rebuild each request's per-layer tensors from an aperture raw dump and its sidecar."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .aperture_metadata import read_qk_sidecar, read_sidecar
from .aperture_run_index import INDEX_NAME, read_run_index, read_run_segments, run_slices
from .aperture_trim import is_reclaimed, refuse_trimmed_rows, trimmed_floor_rows
from .aperture_trim import TrimmedRegionError, trim_status  # noqa: F401  (re-exported)
from mia.core.runtime.tp_shard import (
    TPShardError,
    check_complete_shard_set,
    check_hs_shard_set,
    discover_rank_dirs,
    hs_expected_ranks,
    hs_requested_layers,
    hs_shard_from_header,
    merge_head_tensors,
    merge_hs_layer_maps,
    parse_rank_dir,
    qk_shard_from_header,
)


def _file_row(e) -> int:
    fr = getattr(e, "file_row", None)
    return e.logical_start if fr is None else fr


_NUMPY_DTYPE_BY_NAME = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def load_aperture_artifact(raw_path: str, meta_path: str) -> dict:
    """Reconstruct `{req_id: {layer: Tensor}}` from a raw aperture dump + its metadata sidecar."""
    header, steps = read_sidecar(meta_path)
    dtype_name = header["dtype"]
    row_shape = tuple(header["row_shape"])
    is_bf16 = dtype_name == "bfloat16"
    if is_bf16:
        np_dtype = np.uint16
    else:
        try:
            np_dtype = _NUMPY_DTYPE_BY_NAME[dtype_name]
        except KeyError:
            raise ValueError(f"unsupported dtype {dtype_name!r} in aperture header")

    mmap = np.memmap(raw_path, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)

    entries = [e for s in steps for e in s.entries]

    blocks: dict = {}
    for e in entries:
        fr = _file_row(e)
        block = np.array(mmap[fr : fr + e.n_rows])
        tensor = torch.from_numpy(block)
        if is_bf16:
            tensor = tensor.view(torch.bfloat16)
        blocks.setdefault((e.req_id, e.layer), []).append((fr, tensor))

    out: dict = {}
    for (req_id, layer), parts in blocks.items():
        parts.sort(key=lambda p: p[0])
        tensors = [t for _, t in parts]
        merged = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        out.setdefault(req_id, {})[layer] = merged

    return out


def _np_dtype_for(dtype_name: str):
    if dtype_name == "bfloat16":
        return np.uint16, True
    try:
        return _NUMPY_DTYPE_BY_NAME[dtype_name], False
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype_name!r} in aperture header")


def load_multilayer_qk_aperture_artifact(run_dir: str, meta_path: str | None = None) -> dict:
    """Reconstruct per-request, per-layer QK tensors from a QK aperture dump and its sidecar."""
    if meta_path is None:
        meta_path = os.path.join(run_dir, "qk_aperture_meta.jsonl")
    header, entries = read_qk_sidecar(meta_path)
    q_dtype, q_is_bf16 = _np_dtype_for(header["dtype"])
    k_dtype, k_is_bf16 = q_dtype, q_is_bf16
    q_row_shape = tuple(header["q_row_shape"])
    k_row_shape = tuple(header["k_row_shape"])

    q_mmaps: dict = {}
    k_mmaps: dict = {}

    def _mm(cache: dict, fname: str, layer: int, np_dtype, row_shape):
        mm = cache.get(layer)
        if mm is None:
            raw = os.path.join(run_dir, fname)
            mm = np.memmap(raw, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)
            cache[layer] = mm
        return mm

    grouped: dict = {}
    for e in entries:
        grouped.setdefault((e.req_id, e.layer), []).append(e)

    out: dict = {}
    for (req_id, layer), es in grouped.items():
        es.sort(key=lambda e: e.k_start)
        if es and es[0].num_computed > 0:
            raise NotImplementedError(
                f"QK capture-aperture prefix reconstruction is deferred (v1): request {req_id!r} "
                f"layer {layer} first-step num_computed={es[0].num_computed} > 0 "
                f"(prefix caching). k_full would be short by the cached prefix; refusing to "
                f"return a wrong k_all. Capture with prefix caching off (the graph default) or "
                f"eager.")

        kmm = _mm(k_mmaps, f"qk_k_layer_{layer}.raw", layer, k_dtype, k_row_shape)

        k_parts, q_parts, prefix_ends = [], [], []
        for e in es:
            kb = torch.from_numpy(np.array(kmm[e.k_start:e.k_start + e.k_rows]))
            if k_is_bf16:
                kb = kb.view(torch.bfloat16)
            k_parts.append(kb)
            if e.q_rows > 0 and e.q_start >= 0:
                qmm = _mm(q_mmaps, f"qk_q_layer_{layer}.raw", layer, q_dtype, q_row_shape)
                qb = torch.from_numpy(np.array(qmm[e.q_start:e.q_start + e.q_rows]))
                if q_is_bf16:
                    qb = qb.view(torch.bfloat16)
                q_parts.append(qb)
            if e.prefix_end >= 0:
                prefix_ends.append(int(e.prefix_end))

        k_full = k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=0)
        q_cat = (q_parts[0] if len(q_parts) == 1
                 else (torch.cat(q_parts, dim=0) if q_parts else k_full.new_empty((0,) + q_row_shape)))
        k_all = [k_full[:L] for L in prefix_ends]
        out.setdefault(req_id, {})[layer] = {
            "q": q_cat,
            "k_all": k_all,
            "k_full": k_full,
            "k_prefix_ends": prefix_ends,
            "hookq_mode": header.get("hookq_mode"),
        }
    return out


def _note_reclaimed(out: dict | None, req_id, layer) -> None:
    if out is not None:
        out.setdefault(req_id, []).append(int(layer))


def load_multilayer_aperture_artifact(run_dir: str, meta_path: str | None = None, *,
                                      skip_trimmed: bool = False,
                                      reclaimed_out: dict | None = None) -> dict:
    """Reconstruct ``{req_id: {layer: Tensor}}`` from an HS aperture dump and its sidecar."""
    if meta_path is None:
        meta_path = os.path.join(run_dir, "hs_aperture_meta.jsonl")
    header, steps = read_sidecar(meta_path)
    dtype_name = header["dtype"]
    row_shape = tuple(header["row_shape"])
    is_bf16 = dtype_name == "bfloat16"
    if is_bf16:
        np_dtype = np.uint16
    else:
        try:
            np_dtype = _NUMPY_DTYPE_BY_NAME[dtype_name]
        except KeyError:
            raise ValueError(f"unsupported dtype {dtype_name!r} in aperture header")

    mmaps: dict = {}

    def _mm(layer: int):
        mm = mmaps.get(layer)
        if mm is None:
            raw = os.path.join(run_dir, f"hs_layer_{layer}.raw")
            mm = np.memmap(raw, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)
            mmaps[layer] = mm
        return mm

    entries = [e for s in steps for e in s.entries]
    floors = trimmed_floor_rows(run_dir)
    lowest: dict = {}
    if floors:
        for e in entries:
            k = (e.req_id, e.layer)
            fr = _file_row(e)
            if k not in lowest or fr < lowest[k]:
                lowest[k] = fr
    reclaimed_keys = {k for k, fr in lowest.items() if is_reclaimed(k[1], fr, floors)}
    if reclaimed_keys and not skip_trimmed:
        rid, layer = sorted(reclaimed_keys, key=lambda k: (str(k[0]), int(k[1])))[0]
        refuse_trimmed_rows(run_dir, layer, lowest[(rid, layer)], floors)
    for k in sorted(reclaimed_keys, key=lambda k: (str(k[0]), int(k[1]))):
        _note_reclaimed(reclaimed_out, k[0], k[1])
    blocks: dict = {}
    for e in entries:
        fr = _file_row(e)
        if (e.req_id, e.layer) in reclaimed_keys:
            continue
        mm = _mm(e.layer)
        block = np.array(mm[fr: fr + e.n_rows])
        tensor = torch.from_numpy(block)
        if is_bf16:
            tensor = tensor.view(torch.bfloat16)
        blocks.setdefault((e.req_id, e.layer), []).append((fr, tensor))

    out: dict = {}
    for (req_id, layer), parts in blocks.items():
        parts.sort(key=lambda p: p[0])
        tensors = [t for _, t in parts]
        merged = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        out.setdefault(req_id, {})[layer] = merged

    return out


def load_from_run_index(run_dir: str, index_path: str | None = None, *,
                        skip_trimmed: bool = False, reclaimed_out: dict | None = None) -> dict:
    """Reconstruct ``{req_id: {layer: Tensor}}`` from the run index instead of the sidecar."""
    if index_path is None:
        index_path = os.path.join(run_dir, INDEX_NAME)
    header, reqs = read_run_index(index_path)
    return _reconstruct_from_runs(run_dir, header, reqs, skip_trimmed=skip_trimmed,
                                  reclaimed_out=reclaimed_out)


def load_from_run_segments(run_dir: str, *, only_complete: bool = False,
                           skip_trimmed: bool = False, reclaimed_out: dict | None = None) -> dict:
    """Reconstruct ``{req_id: {layer: Tensor}}`` from the mid-run index segment chain."""
    header, reqs, complete = read_run_segments(run_dir)
    if only_complete:
        reqs = {r: v for r, v in reqs.items() if r in complete}
    return _reconstruct_from_runs(run_dir, header, reqs, skip_trimmed=skip_trimmed,
                                  reclaimed_out=reclaimed_out)


def _reconstruct_from_runs(run_dir: str, header: dict, reqs: dict, *, skip_trimmed: bool = False,
                           reclaimed_out: dict | None = None) -> dict:
    np_dtype, is_bf16 = _np_dtype_for(header["dtype"])
    row_shape = tuple(header["row_shape"])
    floors = trimmed_floor_rows(run_dir)

    mmaps: dict = {}

    def _mm(layer: int):
        mm = mmaps.get(layer)
        if mm is None:
            raw = os.path.join(run_dir, f"hs_layer_{layer}.raw")
            mm = np.memmap(raw, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)
            mmaps[layer] = mm
        return mm

    out: dict = {}
    for req_id, r in reqs.items():
        for layer in r.layers:
            slices = list(run_slices(r.runs))
            if floors and slices and is_reclaimed(layer, min(a for a, _b, _s in slices), floors):
                if not skip_trimmed:
                    refuse_trimmed_rows(run_dir, layer, min(a for a, _b, _s in slices), floors)
                _note_reclaimed(reclaimed_out, req_id, layer)
                continue
            mm = _mm(layer)
            parts = [np.array(mm[a:b:st]) for a, b, st in slices]
            if not parts:
                block = np.array(mm[0:0])
            elif len(parts) == 1:
                block = parts[0]
            else:
                block = np.concatenate(parts, axis=0)
            tensor = torch.from_numpy(block)
            if is_bf16:
                tensor = tensor.view(torch.bfloat16)
            out.setdefault(req_id, {})[int(layer)] = tensor
    return out


QK_SIDECAR_NAME = "qk_aperture_meta.jsonl"
HS_SIDECAR_NAME = "hs_aperture_meta.jsonl"


def read_sidecar_header(meta_path: str) -> dict:
    """Line 0 of an aperture sidecar (HS or QK), without reading the entries."""
    with open(meta_path, "r", encoding="utf-8") as f:
        first = f.readline()
    return json.loads(first)["__header__"]


def merge_qk_aperture_ranks(rank_dirs, meta_paths=None, *, check_replicas: bool = False) -> dict:
    """Merge per-rank QK aperture dumps into ONE artifact in the global head layout."""
    rank_dirs = [str(d) for d in rank_dirs if d]
    if not rank_dirs:
        raise TPShardError("no QK rank dirs to merge (flush_aperture returned nothing)")
    if meta_paths is None:
        meta_paths = [os.path.join(d, QK_SIDECAR_NAME) for d in rank_dirs]
    if len(meta_paths) != len(rank_dirs):
        raise ValueError("meta_paths must align with rank_dirs")
    headers = [read_sidecar_header(m) for m in meta_paths]
    shards = [qk_shard_from_header(h) for h in headers]
    if all(s is None for s in shards):
        if len(rank_dirs) != 1:
            raise TPShardError(
                f"{len(rank_dirs)} QK dirs without TP shard headers: cannot order their heads")
        return load_multilayer_qk_aperture_artifact(rank_dirs[0], meta_paths[0])
    if any(s is None for s in shards):
        raise TPShardError("some QK dirs carry TP shard headers and some do not")
    for d, h, s in zip(rank_dirs, headers, shards):
        q_w = int(np.prod(h["q_row_shape"]))
        k_w = int(np.prod(h["k_row_shape"]))
        if (q_w, k_w) != (s.q_width, s.k_width):
            raise TPShardError(
                f"{d}: row widths q={q_w} k={k_w} contradict its shard header "
                f"(expected q={s.q_width} k={s.k_width})")
    order = check_complete_shard_set(shards)
    arts = [load_multilayer_qk_aperture_artifact(rank_dirs[i], meta_paths[i]) for i in order]
    ranked = [shards[i] for i in order]

    keys0 = {(r, L) for r, per in arts[0].items() for L in per}
    for s, art in zip(ranked[1:], arts[1:]):
        keys = {(r, L) for r, per in art.items() for L in per}
        if keys != keys0:
            only0 = sorted(keys0 - keys)[:4]
            onlyr = sorted(keys - keys0)[:4]
            raise TPShardError(
                f"tp_rank {s.tp_rank} captured a different (request, layer) set than tp_rank "
                f"{ranked[0].tp_rank}: missing {only0} extra {onlyr}")

    out: dict = {}
    for req_id, layer in sorted(keys0, key=lambda x: (str(x[0]), int(x[1]))):
        per_rank = [art[req_id][layer] for art in arts]
        ends0 = list(per_rank[0]["k_prefix_ends"])
        for s, e in zip(ranked[1:], per_rank[1:]):
            if list(e["k_prefix_ends"]) != ends0:
                raise TPShardError(
                    f"req {req_id!r} layer {layer}: tp_rank {s.tp_rank} k_prefix_ends "
                    f"{list(e['k_prefix_ends'])[:4]}... != rank {ranked[0].tp_rank}'s {ends0[:4]}...")
        q = merge_head_tensors("q", [(s, e["q"]) for s, e in zip(ranked, per_rank)],
                               check_replicas)
        k_full = merge_head_tensors("k", [(s, e["k_full"]) for s, e in zip(ranked, per_rank)],
                                    check_replicas)
        out.setdefault(req_id, {})[layer] = {
            "q": q,
            "k_all": [k_full[:L] for L in ends0],
            "k_full": k_full,
            "k_prefix_ends": ends0,
            "hookq_mode": per_rank[0].get("hookq_mode"),
        }
    return out


def load_qk_aperture_tp(aperture_dir: str, *, check_replicas: bool = False) -> dict:
    """Load every ``tp_rank_<r>/`` QK dump under ``aperture_dir`` and merge them."""
    found = discover_rank_dirs(aperture_dir, QK_SIDECAR_NAME)
    if not found:
        raise TPShardError(f"no QK aperture sidecar under {aperture_dir}")
    return merge_qk_aperture_ranks([d for _, d in found], check_replicas=check_replicas)


def merge_hs_aperture_ranks(rank_dirs, meta_paths=None, *, expected_layers=None,
                            skip_trimmed: bool = False, reclaimed_out: dict | None = None) -> dict:
    """Union per-rank HS aperture dumps of the TP LAYER shard into ONE artifact."""
    rank_dirs = [str(d) for d in rank_dirs if d]
    if not rank_dirs:
        raise TPShardError("no HS rank dirs to merge (flush_aperture returned nothing)")
    if meta_paths is None:
        meta_paths = [os.path.join(d, HS_SIDECAR_NAME) for d in rank_dirs]
    if len(meta_paths) != len(rank_dirs):
        raise ValueError("meta_paths must align with rank_dirs")
    headers = [read_sidecar_header(m) for m in meta_paths]
    shards = [hs_shard_from_header(h) for h in headers]
    if all(s is None for s in shards):
        if len(rank_dirs) != 1:
            raise TPShardError(
                f"{len(rank_dirs)} HS dirs without a layer-shard header: each holds every layer "
                f"(TP = 1, MIA_HS_TP_SHARD=0, or the all-ranks diagnostic's replicas) -- read one "
                f"with load_multilayer_aperture_artifact, or the run with load_hs_aperture_tp")
        return load_multilayer_aperture_artifact(rank_dirs[0], meta_paths[0],
                                                 skip_trimmed=skip_trimmed,
                                                 reclaimed_out=reclaimed_out)
    if any(s is None for s in shards):
        raise TPShardError("some HS dirs carry a layer-shard header and some do not: two "
                           "different captures, or a rank-0-only dir mixed into a sharded run")
    geom0 = (headers[0].get("dtype"), list(headers[0].get("row_shape") or []))
    for d, h, sh in zip(rank_dirs, headers, shards):
        g = (h.get("dtype"), list(h.get("row_shape") or []))
        if g != geom0:
            raise TPShardError(f"{d}: dtype/row_shape {g} differs from {rank_dirs[0]}'s {geom0}")
        named = parse_rank_dir(d)
        if named is not None and named != sh.tp_rank:
            raise TPShardError(f"{d} is named for tp_rank {named} but its header is tp_rank "
                               f"{sh.tp_rank}'s")
    expected = None
    if expected_layers is not None:
        expected = hs_expected_ranks(
            hs_requested_layers(expected_layers, shards[0].num_layers), shards[0].tp_size)
    order = check_hs_shard_set(shards, expected)
    items = [(shards[i], load_multilayer_aperture_artifact(rank_dirs[i], meta_paths[i],
                                                           skip_trimmed=skip_trimmed,
                                                           reclaimed_out=reclaimed_out))
             for i in order]
    return merge_hs_layer_maps(items)


def _hs_replicas_equal(found, headers, *, skip_trimmed: bool = False) -> None:
    tp = int(headers[0].get("tp_size", 1) or 1)
    ranks = [r for r, _ in found]
    if sorted(ranks) != list(range(tp)):
        raise TPShardError(f"all-ranks HS replicas: expected tp_rank_0..{tp - 1}, found {ranks}")
    base = load_multilayer_aperture_artifact(dict(found)[0], skip_trimmed=skip_trimmed)
    keys0 = {(q, L) for q, per in base.items() for L in per}
    for r, d in found:
        if r == 0:
            continue
        art = load_multilayer_aperture_artifact(d, skip_trimmed=skip_trimmed)
        keys = {(q, L) for q, per in art.items() for L in per}
        if keys != keys0:
            raise TPShardError(
                f"all-ranks HS replicas: tp_rank {r} captured a different (request, layer) set "
                f"than tp_rank 0: missing {sorted(keys0 - keys)[:4]} extra {sorted(keys - keys0)[:4]}")
        for q, L in sorted(keys0, key=lambda x: (str(x[0]), int(x[1]))):
            a, b = art[q][L], base[q][L]
            if a.shape != b.shape or not torch.equal(a, b):
                raise TPShardError(
                    f"all-ranks HS replicas differ: req {q!r} layer {L} on tp_rank {r} is not "
                    f"bitwise equal to tp_rank 0's (shapes {tuple(a.shape)} vs {tuple(b.shape)})")


def load_hs_aperture_tp(aperture_dir: str, *, check_replicas: bool = False,
                        expected_layers=None, skip_trimmed: bool = False,
                        reclaimed_out: dict | None = None) -> dict:
    """Load one run's HS capture from its aperture dir, a rank dir, or a bare TP=1 dump."""
    found = discover_rank_dirs(aperture_dir, HS_SIDECAR_NAME)
    if not found:
        raise TPShardError(f"no HS aperture sidecar under {aperture_dir}")
    headers = [read_sidecar_header(os.path.join(d, HS_SIDECAR_NAME)) for _, d in found]
    if any(hs_shard_from_header(h) is not None for h in headers):
        return merge_hs_aperture_ranks([d for _, d in found], expected_layers=expected_layers,
                                       skip_trimmed=skip_trimmed, reclaimed_out=reclaimed_out)
    for (r, d), h in zip(found, headers):
        if int(h.get("tp_size", 1)) > 1 and r != 0 and not h.get("capture_all_ranks", False):
            raise TPShardError(
                f"HS capture dir {d} belongs to tp_rank {r} of {h.get('tp_size')} but declares no "
                f"layer shard: without MIA_HS_TP_SHARD only tp_rank 0 captures the (replicated) "
                f"residual stream")
    rank0 = [d for r, d in found if r == 0]
    if not rank0:
        raise TPShardError(f"no tp_rank_0 HS dir under {aperture_dir} (found ranks "
                           f"{[r for r, _ in found]})")
    if check_replicas and any(h.get("capture_all_ranks", False) for h in headers):
        _hs_replicas_equal(found, headers, skip_trimmed=skip_trimmed)
    return load_multilayer_aperture_artifact(rank0[0], skip_trimmed=skip_trimmed,
                                             reclaimed_out=reclaimed_out)

