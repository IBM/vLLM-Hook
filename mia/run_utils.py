from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, List, Optional

from mia._profiler import PROF


def qk_score_size_select(prompt_len: int, mode: str, layer_to_heads: dict,
                         H_q: int, H_kv: int, d: int) -> str:
    """Size model: return "qk" or "score" — the smaller capture artifact.

    Shared by MiaLLM offline admission (`_qk_capture_for`) and the serve-path patch
    (`_plugin._patched_generate`) so both decide identically. Element counts (both
    representations store 2 bytes/elem, so counts compare directly), summed over the
    analyzer's layers with per-layer head count ``n``; the prefill pass dominates so
    ``max_tokens`` cancels out of the crossover:

      all_tokens: QK = S·(H_q+H_kv)·d   score = n·S²   -> score iff S < (H_q+H_kv)·d/n
      last_token: QK ≈ (H_q + S·H_kv)·d score = n·S    -> score iff n < ~H_kv·d

    No analyzer head set (``layer_to_heads`` empty/None) => score would need ALL heads
    and is ~never smaller, so keep raw QK.
    """
    if not layer_to_heads:
        return "qk"
    S = int(prompt_len)
    qk_elems = sc_elems = 0
    for _layer, heads in layer_to_heads.items():
        n = len(heads) or 1
        if mode == "all_tokens":
            qk_elems += S * (H_q + H_kv) * d
            sc_elems += n * S * S
        else:  # last_token
            qk_elems += (H_q + S * H_kv) * d
            sc_elems += n * S
    return "score" if sc_elems < qk_elems else "qk"


# ---------------------------------------------------------------------------
# Per-request storage router (MIA_STORAGE_ROUTER)
#
# Pick save_to_disk=True/False for the fastest engine-loop throughput, from the predicted
# artifact size + model config. The two paths are not symmetric in ms:
#
#   RPC  (get_captured_states, save_to_disk=False): pad + pickle + zstd + ship run
#        synchronously on the EngineCore step loop -> a per-request blocking stall that
#        grows with artifact size (~RPC_INTERCEPT + RPC_SLOPE[w]*KB).
#   DISK (flush_disk, save_to_disk=True): hands cpu_cache to the writer and returns -> the
#        on-loop cost is the handoff (queue_put + the D2H both paths already pay) plus the
#        per-KB write it commits to; the rest of the serialize overlaps decode off-loop.
#
# The load-bearing correction: the ~200 ms "disk floor" is overlappable async-saver GIL
# contention, not a per-request blocking cost -- so it must not be what RPC is compared
# against. RPC blocks the loop; disk defers. RPC therefore wins only when its blocking ship
# is smaller than disk's on-loop handoff -- i.e. only for tiny artifacts. Comparing RPC's
# block to disk's handoff (not the 200 ms floor) reproduces the empirically-proven optimum:
# HS last_token (tiny) -> RPC; HS/QK all_tokens and QK last_token (MB+) -> disk. (An earlier
# model compared RPC's block to the 200 ms floor, which mis-routed mid-size QK to RPC.)
#
# The size predictor below is closed-form and was validated against measured artifact_kb.
#
# BOTH SIDES ARE A MODEL: `intercept + slope * KB`, one pair of coefficients per worker kind, and
# the crossover is SOLVED from them (`rpc_disk_crossover_kb`) rather than written down separately.
# Every coefficient below says where its number came from, and the ONE that is not measured says
# so in capitals. Every one has an env override, listed in docs/configs.md.
# ---------------------------------------------------------------------------

# --- RPC side: the on-loop BLOCKING ship (get_captured_states) ------------------------------
# MEASURED on granite-3.1-8b, FULL+buffer, in the cb campaigns that produced the proven optimum
# (HS last_token -> RPC; QK last_token / HS all_tokens / QK all_tokens -> disk). NOT re-measured
# since; they describe a response-shipping cost the 2026-09-20 writer work did not touch.
_RPC_INTERCEPT_MS = 5.0            # get_captured_states fixed on-loop setup (pickle/ship start)
# RPC BLOCKING ship, ms/KB, per worker. QK ships a padded high-entropy k_all (pickles/zstds
# dearer per KB); HS ships a dense contiguous slab (cheaper) -- anchored so HS-last 104 KB ~= 8 ms.
_RPC_SLOPE_MS_PER_KB = {"qk": 0.157, "hs": 0.03}

# --- DISK side: the on-loop handoff PLUS the per-KB write ------------------------------------
# Until 2026-09-20 this side was ONE flat number (20 ms) with no size term at all, which is why
# the same 512 KiB threshold was applied to HS and QK although their RPC slopes differ 5x.
#
# NOT MEASURED, and it is the dominant term: the on-loop HANDOFF -- what the engine thread spends
# handing the request to the disk route (queue_put of cpu_cache in the classic path; in the
# capture-aperture path, registering the route, after which the consumer thread does the work).
# The 20.0 predates the 2026-09-20 writer fix and never described the per-request write either.
# It is KEPT rather than replaced with a guess. What a GPU run must measure to replace it: the
# engine-loop time between a disk-routed request being admitted and the loop continuing, at a
# fixed artifact size, against the same request routed to RPC -- see docs/configs.md.
# NOTE the direction, if it is ever measured near the write cost below: the crossover collapses
# toward zero and essentially everything routes to disk (CAPTURE_IO.md section 6.3 item 4).
_DISK_HANDOFF_MS = 20.0
# MEASURED (job 1802981; CAPTURE_IO.md section 6.1): the per-request disk staging's own write, on
# the zero-copy buffered writer it now shares with the shared-file drain. Two points, 8 KiB ->
# 0.0174 ms and 16 KiB -> 0.0197 ms per write, give a per-write model of
#     0.0151 ms fixed + 0.000288 ms/KB        (the slope is ~3.4 GB/s, and a bare-write control in
#                                              the same job independently reads 3.2 GB/s)
# The staging writes ONE captured row per file per step, so bytes-per-write is the row width and
# the per-KB cost is `0.0151 / row_KB + 0.000288`:
#   HS  rows are the hidden size x 2 B -- 8 KiB at 8B, 16 KiB at 70B -> 0.0022 ms/KB (the 8 KiB,
#       i.e. more-writes-per-KB, i.e. conservative, end)
#   QK  the k rows dominate the artifact and are the narrowest -- 2 KiB at 8B TP1 -> 0.0078 ms/KB
# A whole HS last_token artifact is therefore 32 x 8 KiB = 0.56 ms of writing at 8B and
# 20 x 16 KiB = 0.39 ms per rank at 70B (legacy: 0.77 / 0.63 ms) -- not the 20 ms above.
# SENSITIVITY, since the slope is per-ROW and the router is handed only KB: a narrower row means
# more writes per KB. QK's k row is 512 B at 70B TP4, which would read 0.0305 ms/KB and move that
# crossover 100.5 -> 118.6 KB (+18 %) -- within the noise of the un-measured handoff above, so the
# defaults are the 8B TP1 shapes and the env overrides exist for a deployment that cares.
_DISK_SLOPE_MS_PER_KB = {"qk": 0.0078, "hs": 0.0022}


#: Decode tokens assumed for a request that pinned no ``max_tokens`` -- see :func:`estimate_gen_len`.
#: 256 is a round number near this study's ShareGPT stream (mean max_tokens 299.8). It is an
#: ESTIMATE, not a measurement; the function's docstring argues why an estimate is sufficient.
DEFAULT_GEN_LEN = 256


def estimate_gen_len(max_tokens) -> int:
    """Decode tokens to price a request at: ``max_tokens`` when the caller pinned one, else a
    rough estimate (:data:`DEFAULT_GEN_LEN`, override ``MIA_ROUTER_DEFAULT_GEN_LEN``).

    WHY A ROUGH GUESS IS ENOUGH, AND WHY ZERO WAS NOT. Every caller is a router, and a router
    only asks which SIDE of a crossover the artifact falls on -- never what it weighs. Since the
    artifact grows by at least one row per layer per decode step, the answer stops changing
    within a few tokens: an 8B HS ``last_token`` capture hooked on decode adds 256 KB per step
    against a 539.6 KB crossover, so it routes to disk from two decode tokens upward and every
    plausible estimate -- 16, 256, 4096 -- picks the same route. What the callers passed before
    was not a coarse estimate but ``0``, which is not a decode length any request has: it priced
    a whole generation as a single step.

    The bias is deliberate and one-directional. Over-estimating routes a large artifact to disk,
    which costs a file; under-estimating ships it over RPC on the engine loop, which is the
    failure this router exists to prevent."""
    import os
    try:
        n = int(max_tokens or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return n
    try:
        return max(1, int(os.environ.get("MIA_ROUTER_DEFAULT_GEN_LEN", DEFAULT_GEN_LEN)))
    except ValueError:
        return DEFAULT_GEN_LEN


def predict_artifact_kb(worker_kind: str, gran: str, prompt_len: int, n_layers: int,
                        heads_per_layer: int, head_dim: int, hidden: int,
                        dtype_bytes: int = 2, gen_len: int = 0,
                        hooks_on: str = "prefill") -> float:
    """Closed-form RAW artifact size (KB) for one request's captured tensors.

    Prefill-dominant: with ``hooks_on="prefill"`` decode captures nothing, so the
    size is a pure function of the prompt length ``prompt_len`` (the regime the cb
    campaigns run and the only one anchored to measured data). For hooks_on that
    include decode, ``gen_len`` (an UPPER BOUND -- ``max_tokens``, or
    :func:`estimate_gen_len` when the request pinned none) is added so the router
    over-estimates rather than under-routes big artifacts to RPC.

    Shapes follow the RPC (padded) format documented above -- that is what
    ``get_captured_states`` ships and hence what the RPC tax is charged on:
      HS last_token -> L * steps * hidden               (one vector / layer / CAPTURED STEP)
      HS all_tokens -> L * seq * hidden
      QK last_token -> L * H * seq * head_dim           (k_all dominates; q ~O(L*H))
      QK all_tokens -> L * H * seq * 2 * head_dim        (q[seq] + k[seq])

    ``last_token`` IS PER STEP, NOT PER REQUEST (fixed 2026-09-21; this read ``L * hidden``
    before, independent of ``gen_len``). The name says which TOKEN of a step is kept, not how
    often the hook fires -- it fires on every forward step the request is in, and each fire
    appends one more vector per layer (``mia/workers/hs_capture_worker.py``:
    ``ls["hidden_states"].append(...)``; the graph path reserves ``n = 1`` aperture row per
    layer per step, ``mia/graph/install_hs.py``). So a ``hooks_on: prefill`` request really is
    ``L * hidden`` -- its one captured step is the prefill -- while the same request at
    ``hooks_on: both``, the serve default, captures ``1 + gen_len`` steps and is ~300x larger on
    a 300-token generation. QK needed no matching fix: its ``L*seq*Hd + L*Hd`` already grows
    with the context and with the per-step q row.
    """
    P = int(prompt_len)
    # Tokens that actually get captured. prefill-only -> P; else P + generated.
    seq = P if hooks_on == "prefill" else P + int(gen_len)
    # Forward STEPS the hook fires on: the prefill counts once whatever the prompt length (in
    # last_token mode chunked prefill emits only on the final chunk), plus one per decode step.
    if hooks_on == "prefill":
        steps = 1
    elif hooks_on == "decode":
        steps = max(0, int(gen_len))
    else:                                   # "both" -- the serve default: prefill AND decode
        steps = 1 + max(0, int(gen_len))
    L = int(n_layers)
    if worker_kind == "hs":
        elems = L * steps * hidden if gran == "last_token" else L * seq * hidden
    else:  # qk
        Hd = int(heads_per_layer) * int(head_dim)
        if gran == "last_token":
            elems = L * seq * Hd + L * Hd          # k_all[seq] + q[1]  (per layer,head)
        else:  # all_tokens
            elems = L * seq * 2 * Hd               # q[seq] + k[seq]
    return elems * int(dtype_bytes) / 1024.0


def predicted_rpc_ms(worker_kind: str, predicted_kb: float) -> float:
    """On-loop RPC (get_captured_states) BLOCKING ship time in ms for an artifact size.
    Per-worker slope (QK's padded k_all is dearer/KB than HS's dense slab). The per-worker
    default is overridable by MIA_ROUTER_RPC_SLOPE_MS_PER_KB_{QK,HS}."""
    import os
    intercept = float(os.environ.get("MIA_ROUTER_RPC_INTERCEPT_MS", _RPC_INTERCEPT_MS))
    default_slope = _RPC_SLOPE_MS_PER_KB.get(worker_kind, _RPC_SLOPE_MS_PER_KB["hs"])
    slope = float(os.environ.get(
        f"MIA_ROUTER_RPC_SLOPE_MS_PER_KB_{worker_kind.upper()}", default_slope))
    return intercept + slope * float(predicted_kb)


def predicted_disk_ms(worker_kind: str, predicted_kb: float) -> float:
    """On-loop DISK cost in ms for an artifact size: the handoff plus the per-KB write.

    ``MIA_ROUTER_DISK_HANDOFF_MS`` (default 20.0, the one coefficient here that is NOT measured --
    see the block comment) plus ``MIA_ROUTER_DISK_SLOPE_MS_PER_KB_{HS,QK}`` (defaults measured in
    job 1802981 from the per-request staging's own writes: 0.0022 ms/KB for HS's 8 KiB rows,
    0.0078 for QK's 2 KiB k rows). This is what RPC's blocking ship must beat -- NOT the ~200 ms
    async-saver contention floor, which overlaps decode."""
    import os
    handoff = float(os.environ.get("MIA_ROUTER_DISK_HANDOFF_MS", _DISK_HANDOFF_MS))
    default_slope = _DISK_SLOPE_MS_PER_KB.get(worker_kind, _DISK_SLOPE_MS_PER_KB["hs"])
    slope = float(os.environ.get(
        f"MIA_ROUTER_DISK_SLOPE_MS_PER_KB_{worker_kind.upper()}", default_slope))
    return handoff + slope * float(predicted_kb)


def route_to_disk(worker_kind: str, predicted_kb: float) -> bool:
    """Fastest-path storage choice: DISK iff RPC's on-loop BLOCKING ship costs more than the disk
    route's on-loop cost at this size -- i.e. RPC (save_to_disk=False) wins only while the artifact
    is small enough that shipping it on the loop is cheaper than handing it to the writer. Both
    sides are the models above; the size where they cross is :func:`rpc_disk_crossover_kb`.
    Reproduces the proven optimum: HS-last -> RPC (tiny), QK-last / HS-all / QK-all -> disk."""
    return (predicted_rpc_ms(worker_kind, predicted_kb)
            > predicted_disk_ms(worker_kind, predicted_kb))


#: What ``rpc_disk_crossover_kb`` returns when RPC is never the dearer side, so there is no
#: crossover: every predictable artifact goes by RPC. 1 TiB in KB -- larger than any artifact a
#: request can produce, and finite so callers can keep doing arithmetic on it.
NO_CROSSOVER_KB = float(1 << 30)


def rpc_disk_crossover_kb(worker_kind: str) -> float:
    """The artifact size (KB) at which the RPC and disk models cross -- SOLVED from them, never a
    separate constant, so a retuned coefficient moves the threshold with it.

        rpc_intercept + rpc_slope*KB  ==  disk_handoff + disk_slope*KB

    At the shipped defaults: HS (5.0 + 0.03 KB vs 20.0 + 0.0022 KB) crosses at **539.6 KB**, QK
    (slope 0.157 vs 0.0078) at **100.5 KB** -- QK five times earlier than HS, which the single
    hardcoded 512 KiB both of them used before 2026-09-20 could not express. Returns
    :data:`NO_CROSSOVER_KB` when RPC's slope does not exceed disk's (RPC then wins at every size)
    or when RPC already starts dearer (disk wins at every size -> 0.0)."""
    import os
    rpc_intercept = float(os.environ.get("MIA_ROUTER_RPC_INTERCEPT_MS", _RPC_INTERCEPT_MS))
    rpc_slope = float(os.environ.get(
        f"MIA_ROUTER_RPC_SLOPE_MS_PER_KB_{worker_kind.upper()}",
        _RPC_SLOPE_MS_PER_KB.get(worker_kind, _RPC_SLOPE_MS_PER_KB["hs"])))
    disk_handoff = float(os.environ.get("MIA_ROUTER_DISK_HANDOFF_MS", _DISK_HANDOFF_MS))
    disk_slope = float(os.environ.get(
        f"MIA_ROUTER_DISK_SLOPE_MS_PER_KB_{worker_kind.upper()}",
        _DISK_SLOPE_MS_PER_KB.get(worker_kind, _DISK_SLOPE_MS_PER_KB["hs"])))
    if rpc_intercept >= disk_handoff:
        return 0.0
    if rpc_slope <= disk_slope:
        return NO_CROSSOVER_KB
    return (disk_handoff - rpc_intercept) / (rpc_slope - disk_slope)


# ---------------------------------------------------------------------------
# Artifact format conversion utilities
#
# Two retrieval paths produce different formats for the same underlying data:
#
# RPC path (get_captured_states / output.probes):
#   Optimised for API consumers that want a single array to index directly.
#   Hidden states are stacked/padded into one tensor:
#     last_token → Tensor(N_passes, hidden)
#     all_tokens → Tensor(N_passes, max_seq, hidden)
#   QK:
#     all_tokens → q: Tensor(N_passes, max_seq, hidden), k: same
#     last_token → q: Tensor(N_passes, hidden),          k: Tensor(N_passes, max_seq, hidden)
#
# Disk path (flush_disk / .pt file hs_cache / qk_cache):
#   Optimised for offline analysis — preserves variable-length per-pass
#   tensors without padding:
#     last_token → hidden_states: List[Tensor(hidden,)]
#     all_tokens → hidden_states: List[Tensor(seq_i, hidden)]
#   QK:
#     all_tokens → q: List[Tensor(seq_i, hidden)], k_all: List[Tensor(seq_i, kv_hidden)]
#     last_token → q: List[Tensor(hidden,)],       k_all: List[Tensor(seq_i, kv_hidden)]
#
# The functions below normalise either format to a list of per-pass tensors
# so that analyzers (AttntrackerAnalyzer, CorerAnalyzer, HiddenStatesAnalyzer)
# can consume both retrieval paths with the same code.
# ---------------------------------------------------------------------------


def unpack_hidden_states(entry: dict) -> "List[Any]":
    """Return hidden states as a list of per-pass tensors.

    Accepts both the RPC format (single stacked/padded tensor) and the disk
    format (list of tensors). The returned list always has one element per
    forward pass that was captured.

    Example::

        # works on both output.probes["hidden_states"]["model.layers.11"]
        # and cache["hs_cache"]["model.layers.11"] loaded from disk
        tensors = unpack_hidden_states(entry)
        for t in tensors:
            print(t.shape)  # (hidden,) for last_token, (seq_len, hidden) for all_tokens
    """
    import torch
    hs = entry["hidden_states"]
    if isinstance(hs, list):
        return hs  # disk format
    if isinstance(hs, torch.Tensor):
        return list(hs.unbind(0))  # RPC format: split along pass dim
    raise TypeError(f"Unexpected hidden_states type: {type(hs)}")


def unpack_qk(entry: dict) -> "tuple[List[Any], List[Any]]":
    """Return Q and K as lists of per-pass tensors.

    Accepts both the RPC format (stacked/padded tensors) and the disk format
    (lists of tensors). Returns (q_list, k_list) where each element is one
    forward pass.

    Example::

        q_passes, k_passes = unpack_qk(entry)
        for q, k in zip(q_passes, k_passes):
            print(q.shape, k.shape)
    """
    import torch

    def _unpack(t):
        if isinstance(t, list):
            return t
        if isinstance(t, torch.Tensor):
            return list(t.unbind(0))
        raise TypeError(f"Unexpected tensor type: {type(t)}")

    return _unpack(entry["q"]), _unpack(entry["k_all"])


def _artifact_glob(hook_dir: str, run_id: str, filename: str, timeout: float = 0.0,
                   expected_ranks: Optional[int] = None) -> List[str]:
    """Return artifact paths under run_id, optionally polling.

    Pass timeout > 0 (e.g. 5.0) to poll until the file (and, for safetensors, EACH path's .json
    sidecar) appears or the deadline is reached — for a disk-save path that may not have finished
    writing yet by the time analyze() is called.

    ``expected_ranks`` (with timeout > 0): keep polling until that many distinct ``tp_rank_<r>``
    dirs hold the file. At TP > 1 every QK rank lands its own shard on its own writer's schedule,
    so the first match is one rank's heads, not the artifact. On timeout the paths found so far are
    returned (possibly a partial set, which the QK loader refuses loudly), [] when there are none.
    """
    patt = os.path.join(hook_dir, run_id, "**", filename)
    paths = glob.glob(patt, recursive=True)
    if timeout <= 0:
        return paths

    def _complete(ps) -> bool:
        if not ps:
            return False
        if filename.endswith(".safetensors") and not all(
                os.path.exists(p[:-len(".safetensors")] + ".json") for p in ps):
            return False
        if expected_ranks:
            from mia.graph.tp_shard import parse_rank_dir
            ranks = {parse_rank_dir(os.path.dirname(p)) for p in ps}
            return len(ranks - {None}) >= int(expected_ranks)
        return True

    deadline = time.monotonic() + timeout
    while not _complete(paths):
        if time.monotonic() >= deadline:
            return paths
        time.sleep(0.001)
        paths = glob.glob(patt, recursive=True)
    return paths


def _load_safetensors_shards(st_paths: List[str], cache_key: str, basename: str,
                              build_entries):
    """Load a list of safetensors files, parse the JSON sidecar for each, and
    call ``build_entries(sf, meta)`` to construct the per-shard cache dict.

    Returns a list of (tp_rank, shard_dict) tuples sorted by tp_rank, where
    each ``shard_dict`` has keys ``"config"`` and ``cache_key``. Extra keys
    (e.g. ``peak_gpu_mb``) can be added by ``build_entries``.

    ``build_entries(safe_open_handle, meta)`` is the per-shard tensor extractor
    that returns a dict {module_name: entry}.
    """
    import json
    from safetensors import safe_open

    shards = []
    with PROF.timed("io.artifact_load.safetensors"):
        for p in st_paths:
            meta_path = p.replace(f"{basename}.safetensors", f"{basename}.json")
            with open(meta_path) as f:
                meta = json.load(f)
            tp_rank = meta.get("tp_rank", 0)
            try:
                PROF.gauge("io.bytes_read.safetensors", os.path.getsize(p))
                PROF.gauge("io.bytes_read.json", os.path.getsize(meta_path))
            except OSError:
                pass
            with safe_open(p, framework="pt", device="cpu") as sf:
                shard_cache = build_entries(sf, meta)
            shard = {"config": meta["config"], cache_key: shard_cache}
            if "peak_gpu_mb" in meta:
                shard["peak_gpu_mb"] = meta.get("peak_gpu_mb", 0.0)
            if "tp_shard" in meta:
                shard["tp_shard"] = meta["tp_shard"]
            shards.append((tp_rank, shard))
    shards.sort(key=lambda x: x[0])
    return shards


def _shard_tp_rank(cache: dict, path: str) -> int:
    """The TP rank a disk artifact came from: its ``meta.tp_rank``, else its ``tp_shard``, else
    the ``tp_rank_<r>`` directory it sits in, else 0. (The ``.pt`` path carries no ``meta``, so
    reading only ``meta`` made every shard rank 0 and the merge order the glob order.)"""
    meta = cache.get("meta") if isinstance(cache, dict) else None
    if isinstance(meta, dict) and "tp_rank" in meta:
        return int(meta["tp_rank"])
    shard = cache.get("tp_shard") if isinstance(cache, dict) else None
    if isinstance(shard, dict) and "tp_rank" in shard:
        return int(shard["tp_rank"])
    from mia.graph.tp_shard import parse_rank_dir
    r = parse_rank_dir(os.path.dirname(path))
    return 0 if r is None else int(r)


def _refuse_partial_qk_shard_set(shards, where: str = "") -> None:
    """Raise :class:`mia.graph.tp_shard.TPShardError` when QK disk shards written at TP > 1 do not
    cover every rank -- a LONE shard included.

    Each TP rank's QK artifact holds only that rank's heads (``tp_shard`` in the .pt / the
    safetensors sidecar). The loaders used to return a lone shard as the whole artifact after
    dropping its ``tp_shard`` key: q at ``H_q/tp`` heads, k at ``H_kv/tp``, no error. With the
    writer process on every rank the shards land on their own schedules, so that is the normal
    state of a run dir read too early, not a crash leftover. A TP=1 artifact (no ``tp_shard``, or
    ``tp_size == 1``) passes unchanged.

    HS needs no such check: its residual stream is REPLICATED and the eager HS worker captures on
    tp_rank 0 only, so one HS shard is the whole answer at any TP (``_replicated_hs_shard``)."""
    from mia.graph.tp_shard import (TP_SHARD_KEY, TPShardError, check_complete_shard_set,
                                    qk_shard_from_header)
    geoms = [qk_shard_from_header(sh.get(TP_SHARD_KEY)) if isinstance(sh, dict) else None
             for _, sh in shards]
    if not any(g is not None and g.tp_size > 1 for g in geoms):
        return
    if any(g is None for g in geoms):
        return  # mixed geometry / no geometry: merge_qk_payloads refuses it with its own message
    try:
        check_complete_shard_set(geoms)
    except TPShardError as e:
        raise TPShardError(
            f"{where + ': ' if where else ''}{e} Each rank's writer lands its own shard; read the "
            f"run after the save_to_disk barrier (generate() / durable_wait) returns, or the "
            f"missing rank's artifact was lost.") from e


def _replicated_hs_shard(shards):
    """HS shards from several TP ranks are REPLICAS (the residual stream is all-reduced, so
    every rank holds the same hidden state): return the lowest rank's, after checking the
    replicas agree on shape. Concatenating them along ``hidden`` -- what this loader used to do
    -- produced a ``tp x hidden``-wide tensor of duplicated data."""
    base = shards[0][1]
    for rank, shard in shards[1:]:
        for name, entry in (shard.get("hs_cache") or {}).items():
            ref = (base.get("hs_cache") or {}).get(name)
            if ref is None:
                continue
            a, b = ref["hidden_states"], entry["hidden_states"]
            if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
                raise ValueError(
                    f"HS replicas disagree on shape for {name} (tp_rank {rank} vs "
                    f"{shards[0][0]}): the residual stream should be identical on every rank")
    PROF.gauge("io.tp_shard_count", len(shards))
    return base


def _merge_shards_by_module(shards, cache_key: str, tensor_keys, where: str = ""):
    """Merge TP shards into a single cache.

    * HS (``cache_key == "hs_cache"``): the residual stream is REPLICATED across TP ranks, so
      shards are copies -- the lowest rank's is returned (``_replicated_hs_shard``).
    * QK shards that carry ``tp_shard`` geometry (written at TP > 1): merged in GLOBAL head
      order with replicated KV heads de-duplicated (``graph/tp_shard.merge_qk_payloads``).
    * Legacy QK shards without geometry: concatenated along the last dim in rank order.
    * A QK set written at TP > 1 that does not cover every rank (a lone shard included) is
      refused (``_refuse_partial_qk_shard_set``); ``where`` names the run in that error.

    ``tensor_keys`` is the list of tensor list keys per entry (e.g. ``["q","k_all"]``
    or ``["hidden_states"]``). Other keys (e.g. ``layer_num``) are taken from
    the first non-empty shard.
    """
    import torch

    if cache_key == "qk_cache":
        _refuse_partial_qk_shard_set(shards, where)
    if len(shards) == 1:
        out = shards[0][1]
        out.pop("tp_shard", None)
        return out
    if cache_key == "hs_cache":
        return _replicated_hs_shard(shards)
    if cache_key == "qk_cache" and any("tp_shard" in sh for _, sh in shards):
        from mia.graph.tp_shard import merge_qk_payloads
        PROF.gauge("io.tp_shard_count", len(shards))
        with PROF.timed("io.tp_shard_merge"):
            return merge_qk_payloads([sh for _, sh in shards])

    PROF.gauge("io.tp_shard_count", len(shards))

    base_cfg = shards[0][1]["config"]
    merged: Dict[str, Any] = {"config": base_cfg, cache_key: {}}
    if "peak_gpu_mb" in shards[0][1]:
        merged["peak_gpu_mb"] = shards[0][1].get("peak_gpu_mb", 0.0)

    with PROF.timed("io.tp_shard_merge"):
        module_names: set = set()
        for _, shard in shards:
            module_names.update(shard.get(cache_key, {}).keys())

        for module_name in module_names:
            layer_num = None
            per_shard_lists: Dict[str, List[List[Any]]] = {k: [] for k in tensor_keys}
            for _, shard in shards:
                entry = shard.get(cache_key, {}).get(module_name)
                if entry is None:
                    continue
                if layer_num is None:
                    layer_num = entry.get("layer_num")
                for k in tensor_keys:
                    per_shard_lists[k].append(entry[k])

            bs = len(next(iter(per_shard_lists.values()))[0])
            merged_entry = {"layer_num": layer_num}
            for k in tensor_keys:
                merged_entry[k] = [
                    torch.cat([s[i] for s in per_shard_lists[k]], dim=-1) for i in range(bs)
                ]
            merged[cache_key][module_name] = merged_entry

    return merged


def _load_and_merge_hs_safetensors(
    hook_dir: str, run_id: str, st_paths: List[str]
) -> Dict[str, Any]:
    """Load safetensors artifacts and merge across TP shards (same logic as .pt path)."""
    def build_hs_entries(sf, meta):
        batch_size = meta["batch_size"]
        seq_lens = meta.get("seq_lens")  # present for all_tokens, None otherwise
        default_mode = meta.get("hs_mode", "last_token")
        out = {}
        for item in meta["layer_order"]:
            t = sf.get_tensor(item["key"])
            hs_mode = item.get("hs_mode", default_mode)
            if hs_mode == "all_tokens" and seq_lens is not None:
                if t.dim() == 2:
                    # O(seq) layout: t is (sum_seq, hidden) — split by per-entry lengths.
                    hidden_states, off = [], 0
                    for L in seq_lens:
                        hidden_states.append(t[off:off + L, :]); off += L
                else:
                    # legacy padded layout: t is (bs, max_seq, hidden) — unpad.
                    hidden_states = [t[i, :seq_lens[i], :] for i in range(batch_size)]
            else:
                # last_token: t is (bs, hidden_size)
                hidden_states = [t[i] for i in range(batch_size)]
            out[item["module_name"]] = {
                "hidden_states": hidden_states,
                "layer_num": item["layer_num"],
            }
        return out

    shards = _load_safetensors_shards(st_paths, "hs_cache", "hidden_states", build_hs_entries)
    return _merge_shards_by_module(shards, "hs_cache", ["hidden_states"])


def load_and_merge_hs_cache(hook_dir: str, run_id: str) -> Dict[str, Any]:
    """
    Load all hidden-state artifacts for run_id and merge across TP ranks.
    Returns a dict with keys: config, hs_cache.
    TP shards are concatenated along the hidden_size (last) dimension.
    Auto-detects safetensors format when MIA_USE_SAFETENSORS=1.
    """
    import torch

    if os.environ.get("MIA_USE_SAFETENSORS", "0") == "1":
        st_paths = _artifact_glob(hook_dir, run_id, "hidden_states.safetensors")
        if st_paths:
            return _load_and_merge_hs_safetensors(hook_dir, run_id, st_paths)
        # A quantized HS cache is saved as .pt even when safetensors is on (the packed uint8 +
        # scale + qmeta don't fit fixed-shape safetensors) -> fall through to the .pt loader
        # below instead of failing (mirrors load_and_merge_qk_cache).

    paths = _artifact_glob(hook_dir, run_id, "hidden_states.pt")
    if not paths:
        raise FileNotFoundError(
            f"No hidden-state artifacts found for run_id={run_id} under {hook_dir}"
        )

    shards = []
    with PROF.timed("io.artifact_load.pt"):
        for p in paths:
            try:
                PROF.gauge("io.bytes_read.pt", os.path.getsize(p))
            except OSError:
                pass
            cache = torch.load(p, map_location="cpu")
            shards.append((_shard_tp_rank(cache, p), cache))
    shards.sort(key=lambda x: x[0])

    from mia.artifact_quant import dequantize_cache_inplace
    # Dequantize at the disk-read (analysis) boundary. No-op when native.
    for _, shard in shards:
        dequantize_cache_inplace(shard.get("hs_cache", {}), ("hidden_states",))
    # One HS shard is the whole answer at any TP (the residual is replicated and the eager HS
    # worker captures on tp_rank 0 only), so unlike QK there is no partial set to refuse.
    if len(shards) == 1:
        return shards[0][1]
    # Several HS shards can only be replicas of ONE residual stream (HS captures on tp_rank 0;
    # more ranks means the all-ranks diagnostic). Return rank 0's; never concatenate them.
    return _replicated_hs_shard(shards)


def _qk_rebuild_kall(entry: dict) -> None:
    """Rebuild the growing-prefix ``k_all`` list from the compact (``k_full`` +
    ``k_prefix_ends``) per-request form the worker now writes. ``[full[:L] for L in ends]``
    is byte-identical to the old expanded k_all (keys are append-only), at O(seq) storage
    instead of O(seq^2). No-op when the entry is already expanded (eager/quant/score)."""
    if not isinstance(entry, dict) or "k_full" not in entry:
        return
    k_list = []
    for full, ends in zip(entry["k_full"], entry["k_prefix_ends"]):
        k_list.extend(full[:int(L)] for L in ends)
    entry["k_all"] = k_list
    entry.pop("k_full", None)
    entry.pop("k_prefix_ends", None)


def _load_and_merge_qk_safetensors(hook_dir: str, run_id: str, st_paths: List[str]) -> Dict[str, Any]:
    """Load safetensors Q/K artifacts and merge across TP shards."""
    def build_qk_entries(sf, meta):
        # Top-level hookq_mode is the worker default at save time; the actual
        # per-request mode is stored on each layer entry.
        default_mode = meta.get("hookq_mode", "all_tokens")
        # Newer saves track q and k seq_lens separately (they can differ when
        # prefix caching reconstructs k_all to the full sequence). Fall back to
        # legacy "seq_lens" (used for both) for older artifacts.
        legacy_seq_lens = meta.get("seq_lens")
        q_seq_lens = meta.get("q_seq_lens", legacy_seq_lens)
        k_seq_lens = meta.get("k_seq_lens", legacy_seq_lens)
        # NATIVE-COMPACT all_tokens layout: q and the unique keys k_full are stored
        # CONCATENATED (O(seq)), not padded. Split q by q_seq_lens; rebuild the
        # growing-prefix k_all from k_full split by k_full_lens then sliced by
        # k_prefix_ends -- byte-identical to the padded form (keys are append-only).
        compact = meta.get("compact_all_tokens", False)
        k_full_lens = meta.get("k_full_lens")
        k_prefix_ends = meta.get("k_prefix_ends")
        out = {}
        for item in meta["layer_order"]:
            t_q = sf.get_tensor(item["key_q"])
            t_k = sf.get_tensor(item["key_k"])
            hookq_mode = item.get("hookq_mode", default_mode)
            if compact and hookq_mode == "all_tokens":
                # rebuild the growing-prefix k_all from the concatenated k_full
                k_list = []
                off = 0
                for flen, ends in zip(k_full_lens, k_prefix_ends):
                    full_i = t_k[off:off + flen]
                    off += flen
                    for L in ends:
                        k_list.append(full_i[:L])
                # split the concatenated q back into its per-step chunks
                q_list = []
                off = 0
                for ql in q_seq_lens:
                    q_list.append(t_q[off:off + ql])
                    off += ql
            else:
                # legacy padded layout: unpad each row by its recorded length
                # k_all is always 2D — unpad if we have k_seq_lens
                if k_seq_lens is not None:
                    k_list = [t_k[i, :k_seq_lens[i], :] for i in range(t_k.shape[0])]
                else:
                    k_list = [t_k[i] for i in range(t_k.shape[0])]
                # q is 2D in all_tokens (unpad with q_seq_lens) or 1D in last_token
                if hookq_mode == "all_tokens" and q_seq_lens is not None:
                    q_list = [t_q[i, :q_seq_lens[i], :] for i in range(t_q.shape[0])]
                else:
                    q_list = [t_q[i] for i in range(t_q.shape[0])]
            out[item["module_name"]] = {
                "q": q_list,
                "k_all": k_list,
                "layer_num": item["layer_num"],
            }
        return out

    shards = _load_safetensors_shards(st_paths, "qk_cache", "qk", build_qk_entries)
    return _merge_shards_by_module(shards, "qk_cache", ["q", "k_all"],
                                   where=os.path.join(hook_dir, run_id))


def load_and_merge_qk_cache(hook_dir: str, run_id: str):
    """
    Load all shareds for run_id and merge into a single cache.
    Returns a dict with keys: config, qk_cache, meta.
    Auto-detects safetensors format when MIA_USE_SAFETENSORS=1.
    """
    import torch

    if os.environ.get("MIA_USE_SAFETENSORS", "0") == "1":
        st_paths = _artifact_glob(hook_dir, run_id, "qk.safetensors")
        if st_paths:
            return _load_and_merge_qk_safetensors(hook_dir, run_id, st_paths)
        # Score caches are ragged [S_q,S_k] and always save as .pt even when safetensors is
        # on (see QKCaptureWorker._save_safetensors), so a missing qk.safetensors is
        # expected for a score run -- fall through to the .pt loader below.

    paths = _artifact_glob(hook_dir, run_id, "qk.pt")
    if not paths:
        raise FileNotFoundError(
            f"No Q/K cache artifacts found for run_id={run_id} under {hook_dir}"
        )

    shareds = []
    with PROF.timed("io.artifact_load.pt"):
        for p in paths:
            try:
                PROF.gauge("io.bytes_read.pt", os.path.getsize(p))
            except OSError:
                pass
            cache = torch.load(p, map_location="cpu")
            shareds.append((_shard_tp_rank(cache, p), cache))
    shareds.sort(key=lambda x: x[0])
    # A TP > 1 set must cover every rank -- one shard alone is 1/tp of the heads.
    _refuse_partial_qk_shard_set(shareds, os.path.join(hook_dir, run_id))

    # Single shared: return as-is.
    if len(shareds) == 1:
        cache = shareds[0][1]
        cache.pop("tp_shard", None)
        cache.setdefault("meta", {})
        cache["meta"].setdefault("num_shareds", 1)
        for _entry in cache.get("qk_cache", {}).values():
            _qk_rebuild_kall(_entry)      # compact (k_full+k_prefix_ends) -> growing-prefix k_all
        # Dequantize at the disk-read (analysis) boundary. No-op when native.
        from mia.artifact_quant import dequantize_cache_inplace
        dequantize_cache_inplace(cache.get("qk_cache", {}), ("q", "k_all"))
        return cache

    # TP > 1 shards written with their head geometry: rebuild + dequantize each rank's entries,
    # then merge in GLOBAL head order (replicated KV heads de-duplicated). Every rank must be
    # present -- graph/tp_shard.merge_qk_payloads raises on a missing one.
    if any("tp_shard" in sh for _, sh in shareds):
        from mia.artifact_quant import dequantize_cache_inplace
        from mia.graph.tp_shard import merge_qk_payloads
        for _, sh in shareds:
            for _entry in sh.get("qk_cache", {}).values():
                _qk_rebuild_kall(_entry)
            dequantize_cache_inplace(sh.get("qk_cache", {}), ("q", "k_all"))
        with PROF.timed("io.tp_shard_merge"):
            merged = merge_qk_payloads([sh for _, sh in shareds])
        merged["meta"] = {"num_shareds": len(shareds), "tp_ranks": [tp for tp, _ in shareds]}
        return merged

    base_cfg = shareds[0][1]["config"]
    merged: Dict[str, Any] = {
        "config": base_cfg,
        "qk_cache": {},
        "meta": {
            "num_shareds": len(shareds),
            "tp_ranks": [tp for tp, _ in shareds],
        },
    }

    # Collect per-module per-shared entries
    # Each shared contains the same module names for the local partition
    module_names = set()
    for _, shared in shareds:
        module_names.update(shared.get("qk_cache", {}).keys())

    for module_name in module_names:
        # Find first available shared for metadata like layer_num
        layer_num = None
        per_shared_q: List[List[Any]] = []
        per_shared_k: List[List[Any]] = []
        for _, shared in shareds:
            qk = shared.get("qk_cache", {}).get(module_name)
            if qk is None:
                continue
            _qk_rebuild_kall(qk)   # compact (k_full+k_prefix_ends) -> growing-prefix k_all
            # Dequantize each shard before the hidden-dim cat below (concatenating packed
            # sub-byte tensors would corrupt the packing). No-op when native.
            if qk.get("q_qmeta") is not None:
                from mia.artifact_quant import dequantize_cache_inplace
                dequantize_cache_inplace({module_name: qk}, ("q", "k_all"))
            if layer_num is None:
                layer_num = qk.get("layer_num")
            per_shared_q.append(qk["q"])
            per_shared_k.append(qk["k_all"])

        bs = len(per_shared_q[0])
        q_merged: List[Any] = []
        k_merged: List[Any] = []
        for i in range(bs):
            q_parts = [qs[i] for qs in per_shared_q]
            k_parts = [ks[i] for ks in per_shared_k]

            # Validate token dimensions 
            # q: [hidden] or [seq, hidden]
            q_token_shape = q_parts[0].shape[:-1]
            if any(q.shape[:-1] != q_token_shape for q in q_parts):
                raise ValueError(
                    f"Mismatched q token dims across shareds for {module_name}"
                )
            k_token_shape = k_parts[0].shape[:-1]
            if any(k.shape[:-1] != k_token_shape for k in k_parts):
                raise ValueError(
                    f"Mismatched k token dims across shareds for {module_name}"
                )

            q_merged.append(torch.cat(q_parts, dim=-1))
            k_merged.append(torch.cat(k_parts, dim=-1))

        merged["qk_cache"][module_name] = {
            "q": q_merged,
            "k_all": k_merged,
            "layer_num": layer_num,
        }

    return merged


def dispatch_disk_analyze(analyzer, analyzer_spec, run_id=None, run_ids=None):
    """Call ``analyzer.analyze`` with whichever of run_id/run_ids it accepts.

    Used by MiaLLM and MiaClient to dispatch the disk-path analyze call.
    Two-pass analyzers (CoRer) take ``run_ids=[doc, na]``; single-pass
    analyzers take ``run_id`` or no run_id at all.
    """
    import inspect
    sig = inspect.signature(analyzer.analyze)
    kwargs = {"analyzer_spec": analyzer_spec}
    if "run_ids" in sig.parameters and run_ids is not None:
        kwargs["run_ids"] = run_ids
    elif "run_id" in sig.parameters and run_id is not None:
        kwargs["run_id"] = run_id
    return analyzer.analyze(**kwargs)
