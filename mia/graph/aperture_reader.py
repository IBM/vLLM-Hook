"""Read-side reconstruction for the capture-aperture raw dump: pair the row-major raw file with the
`aperture_metadata` sidecar to rebuild each request's per-layer tensors, byte-identical to what was
written. Pure host-side (no GPU): `numpy.memmap` + `torch.from_numpy`.

Contract (see `aperture_metadata.py::LayerEntry`): `read_sidecar` returns entries grouped into
`StepMeta`s, but step index/position is NOT meaningful — reconstruction keys on each entry's
`file_row`, the row offset into THAT ENTRY'S OWN raw file (the per-layer file for the multi-layer
reader; the single shared file for `load_aperture_artifact`), NEVER on `logical_start` (the aperture-wide
reservation position — only equal to `file_row` while every layer receives every row) nor on list
position. `file_row` is always populated (`LayerEntry.__post_init__` / `read_sidecar`'s "fr"-else-"o"
fallback resolve it), so this module reads it unconditionally; `_file_row()` below adds one more layer
of defensiveness for an entry that somehow lacks the attribute. Multiple blocks for the same
(req_id, layer) are sorted by `file_row` (ascending) before concatenation, so the reconstructed token
order is correct regardless of the order entries appear in the sidecar — the reader does not rely on
the upstream drain writing monotonically.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .aperture_metadata import read_qk_sidecar, read_sidecar


def _file_row(e) -> int:
    """The row offset into `e`'s OWN raw file: prefer the explicit per-layer `file_row`, else fall
    back to `logical_start`. `LayerEntry.__post_init__` / `read_sidecar` already guarantee
    `file_row` is populated on every entry this module sees in practice, but keying through this
    helper (rather than `e.file_row` directly) keeps the reader correct even against an entry that
    somehow lacks the attribute — "keys on file_row when present, else logical_start", literally."""
    fr = getattr(e, "file_row", None)
    return e.logical_start if fr is None else fr


# Map the header's dtype string to a numpy dtype. bfloat16 has no native numpy dtype: read its raw
# bytes as uint16 and reinterpret via `torch.view(torch.bfloat16)` after the tensor is built (numpy
# has no bf16 type to view into directly).
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
    """Reconstruct `{req_id: {layer: Tensor}}` from a raw aperture dump + its metadata sidecar.

    `raw_path` is the row-major dump of aperture rows in logical order (row `i` at byte offset
    `i * row_bytes`); `meta_path` is the `aperture_metadata` sidecar written alongside it.
    """
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

    # Flatten in write order (steps are organizational only; grouping below is keyed on
    # file_row, not on this order).
    entries = [e for s in steps for e in s.entries]

    # Collect every block per (req_id, layer) tagged with its file_row, so multi-block concatenation
    # can be sorted into file row order below — self-contained-correct regardless of the order
    # entries happen to appear in the sidecar (do not rely on writer monotonicity).
    blocks: dict = {}
    for e in entries:
        fr = _file_row(e)
        block = np.array(mmap[fr : fr + e.n_rows])  # copy out of the mmap
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
    """numpy dtype for a header dtype string; bfloat16 reads as uint16 (reinterpreted after)."""
    if dtype_name == "bfloat16":
        return np.uint16, True
    try:
        return _NUMPY_DTYPE_BY_NAME[dtype_name], False
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype_name!r} in aperture header")


def load_multilayer_qk_aperture_artifact(run_dir: str, meta_path: str | None = None) -> dict:
    """Reconstruct ``{req_id: {layer: {"q", "k_all", "k_full", "k_prefix_ends", "hookq_mode"}}}``
    from the QK capture-aperture dump: one q raw file + one k raw file per layer
    (``qk_q_layer_<L>.raw`` / ``qk_k_layer_<L>.raw``, row-major, written by
    :class:`MultiLayerQKApertureDrain`) + ONE shared QK sidecar (``qk_aperture_meta.jsonl``).

    Both per-layer files grow in lockstep with the shared aperture cursor, so a
    ``QKStepEntry.k_start`` / ``q_start`` is the row offset into ITS file (same invariant as
    :func:`load_multilayer_aperture_artifact`, split across the q and k files). Per (req, layer):

      * ``k_full`` = the forwarded key history (cat of each step's ``k_file[k_start:k_start+k_rows]``,
        in ``k_start`` / logical order);
      * ``q``      = cat of the emit_q ``q_file[q_start:q_start+q_rows]`` slices;
      * ``k_all``  = ``[k_full[:L] for L in k_prefix_ends]`` — the growing-prefix reconstruction the
        worker's ``_k_all_cpu_list`` produces (byte-identical to the eager path).

    v1 PREFIX-CACHE / hooks_on=decode LIMIT (deferred, see the module + QKStepEntry docstrings): the
    trimmed cached prefix ``[0, num_computed)`` is NOT written into the aperture, so when a request's
    FIRST capture step has ``num_computed > 0`` this raises ``NotImplementedError`` rather than
    returning a k_all that is short by the cached prefix. Fresh prefills (clean / hooks_on=both) have
    first-step ``num_computed == 0`` and reconstruct exactly.
    """
    if meta_path is None:
        meta_path = os.path.join(run_dir, "qk_aperture_meta.jsonl")
    header, entries = read_qk_sidecar(meta_path)
    q_dtype, q_is_bf16 = _np_dtype_for(header["dtype"])
    k_dtype, k_is_bf16 = q_dtype, q_is_bf16  # q and k share the model dtype
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

    # Group entries by (req_id, layer), preserving per-entry step order via k_start.
    grouped: dict = {}
    for e in entries:
        grouped.setdefault((e.req_id, e.layer), []).append(e)

    out: dict = {}
    for (req_id, layer), es in grouped.items():
        es.sort(key=lambda e: e.k_start)
        # First capture step = smallest k_start. v1 does not reconstruct a trimmed prefix.
        if es and es[0].num_computed > 0:
            raise NotImplementedError(
                f"QK capture-aperture prefix reconstruction is deferred (v1): request {req_id!r} "
                f"layer {layer} first-step num_computed={es[0].num_computed} > 0 "
                f"(prefix caching or hooks_on=decode). k_full would be short by the cached prefix; "
                f"refusing to return a wrong k_all. Use a fresh prefill / hooks_on in "
                f"{{prefill, both}}, or extend the QK aperture path to prepend cached keys.")

        qmm = _mm(q_mmaps, f"qk_q_layer_{layer}.raw", layer, q_dtype, q_row_shape)
        kmm = _mm(k_mmaps, f"qk_k_layer_{layer}.raw", layer, k_dtype, k_row_shape)

        k_parts, q_parts, prefix_ends = [], [], []
        for e in es:
            kb = torch.from_numpy(np.array(kmm[e.k_start:e.k_start + e.k_rows]))
            if k_is_bf16:
                kb = kb.view(torch.bfloat16)
            k_parts.append(kb)
            if e.q_rows > 0 and e.q_start >= 0:
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


def load_multilayer_aperture_artifact(run_dir: str, meta_path: str | None = None) -> dict:
    """Reconstruct ``{req_id: {layer: Tensor}}`` from the HS capture-aperture dump: one raw file per
    layer (``hs_layer_<L>.raw``, row-major, written by ``MultiLayerApertureDrain``) + ONE shared
    ``aperture_metadata`` sidecar (``hs_aperture_meta.jsonl``).

    Each ``LayerEntry``'s ``file_row`` is the row offset into ITS OWN layer's file (same invariant as
    the single-file ``load_aperture_artifact``, extended to per-layer files) — NOT ``logical_start``, the
    aperture-wide reservation position, which only coincides with ``file_row`` today because every
    installed layer's file receives every step's rows (see ``LayerEntry``). Multi-block
    ``(req_id, layer)`` groups are sorted by ``file_row`` before concatenation.
    """
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

    # One memmap per layer file, opened lazily on first reference (a request may touch only a
    # subset of layers; non-referenced layer files are never opened).
    mmaps: dict = {}

    def _mm(layer: int):
        mm = mmaps.get(layer)
        if mm is None:
            raw = os.path.join(run_dir, f"hs_layer_{layer}.raw")
            mm = np.memmap(raw, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)
            mmaps[layer] = mm
        return mm

    entries = [e for s in steps for e in s.entries]
    blocks: dict = {}
    for e in entries:
        mm = _mm(e.layer)
        fr = _file_row(e)
        block = np.array(mm[fr: fr + e.n_rows])  # copy out of the mmap
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


# ---------------------------------------------------------------------------
# Tensor parallelism: per-rank QK dirs -> ONE global-head-order artifact
# ---------------------------------------------------------------------------
# QK is captured on EVERY TP rank, each rank writing its OWN local heads to its own
# ``<MIA_APERTURE_DIR>/tp_rank_<r>/`` (see ``mia/graph/tp_shard.py`` for which heads a rank
# holds). Nothing downstream should ever read one rank dir and call it the layer: the merge
# below is the reader's TP contract. HS is different -- the residual stream is REPLICATED, so a
# rank's copy of a layer IS the layer, and under the TP LAYER shard (the default at TP > 1)
# every rank writes a DIFFERENT subset of the layers (round-robin) to its own ``tp_rank_<r>/``:
# ``merge_hs_aperture_ranks`` / ``load_hs_aperture_tp`` union them, refusing a gap or a
# duplicate. Without the shard (TP = 1, ``MIA_HS_TP_SHARD=0``) only ``tp_rank_0`` holds data.

QK_SIDECAR_NAME = "qk_aperture_meta.jsonl"
HS_SIDECAR_NAME = "hs_aperture_meta.jsonl"


def read_sidecar_header(meta_path: str) -> dict:
    """Line 0 of an aperture sidecar (HS or QK), without reading the entries."""
    import json

    with open(meta_path, "r", encoding="utf-8") as f:
        first = f.readline()
    return json.loads(first)["__header__"]


def merge_qk_aperture_ranks(rank_dirs, meta_paths=None, *, check_replicas: bool = False) -> dict:
    """Merge per-rank QK aperture dumps into ONE artifact in the global head layout.

    ``rank_dirs``: the per-rank directories (``tp_rank_<r>``) of ONE capture, in any order --
    typically ``[d for d in llm.collective_rpc("flush_aperture") if d]``. ``meta_paths``
    (optional, aligned with ``rank_dirs``) overrides each rank's sidecar, e.g. a filtered
    sidecar holding only a sample of requests (the header line must be kept).

    Returns the SAME shape as :func:`load_multilayer_qk_aperture_artifact`:
    ``{req_id: {layer(0-based): {"q", "k_all", "k_full", "k_prefix_ends", "hookq_mode"}}}``
    with ``q`` of width ``num_attention_heads * head_dim`` and ``k_full`` / every ``k_all``
    element of width ``num_key_value_heads * head_dim``, heads in GLOBAL order (rank order for
    q; for k, one copy of each KV head -- replicas vLLM made when ``num_key_value_heads <
    tp_size`` are de-duplicated, lowest rank wins; ``check_replicas=True`` also requires the
    replicas to be bitwise equal).

    Raises :class:`mia.graph.tp_shard.TPShardError` when the set is not exactly one dir per rank
    ``0..tp_size-1`` (a rank-0-only capture fails HERE), when headers disagree on the global
    geometry, when a rank's file widths contradict its header, or when the ranks disagree on
    which (request, layer) pairs they captured or on their row structure. A dir whose header
    carries no TP fields (a pre-TP artifact) is accepted only alone, as a full-width TP=1 dump.
    """
    from .tp_shard import TPShardError, check_complete_shard_set, merge_head_tensors, \
        qk_shard_from_header

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
    """Discover every ``tp_rank_<r>/`` under ``aperture_dir`` (the ``MIA_APERTURE_DIR`` of one
    run) and merge them with :func:`merge_qk_aperture_ranks`. ``aperture_dir`` may also be a
    single rank dir / bare TP=1 dump. The expected rank count comes from the headers, so a
    missing rank dir raises rather than returning a narrower layer."""
    from .tp_shard import TPShardError, discover_rank_dirs

    found = discover_rank_dirs(aperture_dir, QK_SIDECAR_NAME)
    if not found:
        raise TPShardError(f"no QK aperture sidecar under {aperture_dir}")
    return merge_qk_aperture_ranks([d for _, d in found], check_replicas=check_replicas)


def merge_hs_aperture_ranks(rank_dirs, meta_paths=None, *, expected_layers=None) -> dict:
    """Union per-rank HS aperture dumps of the TP LAYER shard into ONE artifact.

    ``rank_dirs``: the ``tp_rank_<r>`` dirs of ONE capture, in any order. Each holds its rank's
    round-robin share of the layers and a header naming it (``layer_shard`` / ``owned_layers``,
    recomputed and checked here -- a header that lies is refused). ``meta_paths`` (optional,
    aligned) overrides each rank's sidecar, as for the QK merge.

    Returns :func:`load_multilayer_aperture_artifact`'s shape, ``{req_id: {layer (1-based):
    Tensor}}``, layers ascending, byte-identical to a single-dir capture of the same layers.

    ``expected_layers`` (1-based list, or True for all) is a GAP RELAXATION, NOT A FILTER. It
    narrows WHICH RANKS must be present -- those owning one of its layers -- for a capture that
    legitimately covered only some layers (e.g. a per-request disk delivery of a subset request),
    and nothing else. It does not restrict what is RETURNED, and it cannot tell a subset capture
    from a real loss: passing ``[1, 2]`` to a TP4 run whose rank 2 dir has been DELETED accepts
    the remaining three ranks and returns their layers. A caller that asked for specific layers
    must still check the ones it got.

    Raises :class:`mia.graph.tp_shard.TPShardError`, never returning a partial union, on: a GAP
    (a rank owning an expected layer is missing -- by default every owning rank must be present),
    a DUPLICATE (two dirs of
    one rank, or a (request, layer) present on two ranks), a rank holding a layer it does not own,
    headers that disagree on geometry / dtype / row width, a mix of sharded and unsharded dirs, and
    a request whose layers hold different row counts (the ranks captured different tokens).

    A single dir whose header declares no layer shard (TP = 1, ``MIA_HS_TP_SHARD=0``) is read
    as-is."""
    from .tp_shard import (
        TPShardError, check_hs_shard_set, hs_expected_ranks, hs_requested_layers,
        hs_shard_from_header, merge_hs_layer_maps)

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
        return load_multilayer_aperture_artifact(rank_dirs[0], meta_paths[0])
    if any(s is None for s in shards):
        raise TPShardError("some HS dirs carry a layer-shard header and some do not: two "
                           "different captures, or a rank-0-only dir mixed into a sharded run")
    from .tp_shard import parse_rank_dir
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
    items = [(shards[i], load_multilayer_aperture_artifact(rank_dirs[i], meta_paths[i]))
             for i in order]
    return merge_hs_layer_maps(items)


def _hs_replicas_equal(found, headers) -> None:
    """``MIA_HS_CAPTURE_ALL_RANKS`` diagnostic: every rank ``0..tp_size-1`` present, holding the
    same (request, layer) set as rank 0, each tensor BITWISE equal to rank 0's. Raises
    ``TPShardError`` naming the first difference."""
    import torch as _torch
    from .tp_shard import TPShardError

    tp = int(headers[0].get("tp_size", 1) or 1)
    ranks = [r for r, _ in found]
    if sorted(ranks) != list(range(tp)):
        raise TPShardError(f"all-ranks HS replicas: expected tp_rank_0..{tp - 1}, found {ranks}")
    base = load_multilayer_aperture_artifact(dict(found)[0])
    keys0 = {(q, L) for q, per in base.items() for L in per}
    for r, d in found:
        if r == 0:
            continue
        art = load_multilayer_aperture_artifact(d)
        keys = {(q, L) for q, per in art.items() for L in per}
        if keys != keys0:
            raise TPShardError(
                f"all-ranks HS replicas: tp_rank {r} captured a different (request, layer) set "
                f"than tp_rank 0: missing {sorted(keys0 - keys)[:4]} extra {sorted(keys - keys0)[:4]}")
        for q, L in sorted(keys0, key=lambda x: (str(x[0]), int(x[1]))):
            a, b = art[q][L], base[q][L]
            if a.shape != b.shape or not _torch.equal(a, b):
                raise TPShardError(
                    f"all-ranks HS replicas differ: req {q!r} layer {L} on tp_rank {r} is not "
                    f"bitwise equal to tp_rank 0's (shapes {tuple(a.shape)} vs {tuple(b.shape)})")


def load_hs_aperture_tp(aperture_dir: str, *, check_replicas: bool = False,
                        expected_layers=None) -> dict:
    """Locate and load the HS capture of one run (``aperture_dir`` = its ``MIA_APERTURE_DIR``, or
    one rank dir / a bare TP = 1 dump). Returns :func:`load_multilayer_aperture_artifact`'s shape.

    * TP LAYER SHARD (headers carry ``owned_layers``; the default at TP > 1): every rank dir is
      discovered -- a rank that captured nothing still has its header-only sidecar -- and unioned
      with :func:`merge_hs_aperture_ranks` (a gap or duplicate raises ``TPShardError``).
      ``expected_layers`` passes straight through, and is a gap RELAXATION rather than a filter --
      see that function; it neither restricts the returned layers nor distinguishes a subset
      capture from a lost rank dir.
    * No layer shard (TP = 1, ``MIA_HS_TP_SHARD=0``): exactly ``tp_rank_0`` holds the capture. A
      header saying ``tp_size > 1`` on a dir other than rank 0 means a rank that should have
      written nothing wrote data; it is refused unless the header records the
      ``MIA_HS_CAPTURE_ALL_RANKS`` diagnostic, in which case rank 0's copy is returned (the
      replicas hold the same residual) and ``check_replicas=True`` first requires every rank's
      copy to be BITWISE equal to rank 0's."""
    from .tp_shard import TPShardError, discover_rank_dirs, hs_shard_from_header

    found = discover_rank_dirs(aperture_dir, HS_SIDECAR_NAME)
    if not found:
        raise TPShardError(f"no HS aperture sidecar under {aperture_dir}")
    headers = [read_sidecar_header(os.path.join(d, HS_SIDECAR_NAME)) for _, d in found]
    if any(hs_shard_from_header(h) is not None for h in headers):
        return merge_hs_aperture_ranks([d for _, d in found], expected_layers=expected_layers)
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
        _hs_replicas_equal(found, headers)
    return load_multilayer_aperture_artifact(rank0[0])
