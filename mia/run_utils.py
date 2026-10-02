"""Artifact-size prediction, RPC-vs-disk routing and captured-artifact loading helpers."""
from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, List, Optional

from mia._profiler import PROF


def qk_score_size_select(prompt_len: int, mode: str, layer_to_heads: dict,
                         H_q: int, H_kv: int, d: int) -> str:
    """Size model: return "qk" or "score" — the smaller capture artifact."""
    if not layer_to_heads:
        return "qk"
    S = int(prompt_len)
    qk_elems = sc_elems = 0
    for _layer, heads in layer_to_heads.items():
        n = len(heads) or 1
        if mode == "all_tokens":
            qk_elems += S * (H_q + H_kv) * d
            sc_elems += n * S * S
        else:
            qk_elems += (H_q + S * H_kv) * d
            sc_elems += n * S
    return "score" if sc_elems < qk_elems else "qk"


_RPC_INTERCEPT_MS = 5.0
_RPC_SLOPE_MS_PER_KB = {"qk": 0.157, "hs": 0.03}

_DISK_HANDOFF_MS = 20.0
_DISK_SLOPE_MS_PER_KB = {"qk": 0.0078, "hs": 0.0022}

_ARTIFACT_WAIT_S = float(os.environ.get("MIA_ARTIFACT_WAIT_S", "10") or 10)


DEFAULT_GEN_LEN = 256


def estimate_gen_len(max_tokens) -> int:
    """Decode length to price a request at: max_tokens if pinned, else a default estimate."""
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
    """Closed-form RAW artifact size (KB) for one request's captured tensors."""
    P = int(prompt_len)
    seq = P if hooks_on == "prefill" else P + int(gen_len)
    if hooks_on == "prefill":
        steps = 1
    elif hooks_on == "decode":
        steps = max(0, int(gen_len))
    else:
        steps = 1 + max(0, int(gen_len))
    L = int(n_layers)
    if worker_kind == "hs":
        elems = L * steps * hidden if gran == "last_token" else L * seq * hidden
    else:
        Hd = int(heads_per_layer) * int(head_dim)
        if gran == "last_token":
            elems = L * seq * Hd + L * Hd
        else:
            elems = L * seq * 2 * Hd
    return elems * int(dtype_bytes) / 1024.0


def predicted_rpc_ms(worker_kind: str, predicted_kb: float) -> float:
    """Predicted blocking RPC ship time (ms) for an artifact size."""
    import os
    intercept = float(os.environ.get("MIA_ROUTER_RPC_INTERCEPT_MS", _RPC_INTERCEPT_MS))
    default_slope = _RPC_SLOPE_MS_PER_KB.get(worker_kind, _RPC_SLOPE_MS_PER_KB["hs"])
    slope = float(os.environ.get(
        f"MIA_ROUTER_RPC_SLOPE_MS_PER_KB_{worker_kind.upper()}", default_slope))
    return intercept + slope * float(predicted_kb)


def predicted_disk_ms(worker_kind: str, predicted_kb: float) -> float:
    """Predicted on-loop disk cost (ms) for an artifact size."""
    import os
    handoff = float(os.environ.get("MIA_ROUTER_DISK_HANDOFF_MS", _DISK_HANDOFF_MS))
    default_slope = _DISK_SLOPE_MS_PER_KB.get(worker_kind, _DISK_SLOPE_MS_PER_KB["hs"])
    slope = float(os.environ.get(
        f"MIA_ROUTER_DISK_SLOPE_MS_PER_KB_{worker_kind.upper()}", default_slope))
    return handoff + slope * float(predicted_kb)


def route_to_disk(worker_kind: str, predicted_kb: float) -> bool:
    """True when shipping over RPC costs more than the disk route at this artifact size."""
    return (predicted_rpc_ms(worker_kind, predicted_kb)
            > predicted_disk_ms(worker_kind, predicted_kb))


NO_CROSSOVER_KB = float(1 << 30)


def rpc_disk_crossover_kb(worker_kind: str) -> float:
    """Artifact size (KB) where the RPC and disk cost models cross."""
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


def unpack_hidden_states(entry: dict) -> "List[Any]":
    """Return hidden states as a list of per-pass tensors."""
    import torch
    hs = entry["hidden_states"]
    if isinstance(hs, list):
        return hs
    if isinstance(hs, torch.Tensor):
        return list(hs.unbind(0))
    raise TypeError(f"Unexpected hidden_states type: {type(hs)}")


def unpack_qk(entry: dict) -> "tuple[List[Any], List[Any]]":
    """Return Q and K as lists of per-pass tensors."""
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
    from mia.graph.tp_shard import (TP_SHARD_KEY, TPShardError, check_complete_shard_set,
                                    qk_shard_from_header)
    geoms = [qk_shard_from_header(sh.get(TP_SHARD_KEY)) if isinstance(sh, dict) else None
             for _, sh in shards]
    if not any(g is not None and g.tp_size > 1 for g in geoms):
        return
    if any(g is None for g in geoms):
        return
    try:
        check_complete_shard_set(geoms)
    except TPShardError as e:
        raise TPShardError(
            f"{where + ': ' if where else ''}{e} Each rank's writer lands its own shard; read the "
            f"run after the save_to_disk barrier (generate() / durable_wait) returns, or the "
            f"missing rank's artifact was lost.") from e


def _replicated_hs_shard(shards):
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
    def build_hs_entries(sf, meta):
        batch_size = meta["batch_size"]
        seq_lens = meta.get("seq_lens")
        default_mode = meta.get("hs_mode", "last_token")
        out = {}
        for item in meta["layer_order"]:
            t = sf.get_tensor(item["key"])
            hs_mode = item.get("hs_mode", default_mode)
            if hs_mode == "all_tokens" and seq_lens is not None:
                if t.dim() == 2:
                    hidden_states, off = [], 0
                    for L in seq_lens:
                        hidden_states.append(t[off:off + L, :]); off += L
                else:
                    hidden_states = [t[i, :seq_lens[i], :] for i in range(batch_size)]
            else:
                hidden_states = [t[i] for i in range(batch_size)]
            out[item["module_name"]] = {
                "hidden_states": hidden_states,
                "layer_num": item["layer_num"],
            }
        return out

    shards = _load_safetensors_shards(st_paths, "hs_cache", "hidden_states", build_hs_entries)
    return _merge_shards_by_module(shards, "hs_cache", ["hidden_states"])


def load_and_merge_hs_cache(hook_dir: str, run_id: str) -> Dict[str, Any]:
    """Load all hidden-state artifacts for run_id and merge across TP ranks."""
    import torch

    safetensors = os.environ.get("MIA_USE_SAFETENSORS", "0") == "1"
    if safetensors:
        st_paths = _artifact_glob(hook_dir, run_id, "hidden_states.safetensors", timeout=_ARTIFACT_WAIT_S)
        if st_paths:
            return _load_and_merge_hs_safetensors(hook_dir, run_id, st_paths)

    paths = _artifact_glob(hook_dir, run_id, "hidden_states.pt",
                           timeout=0.0 if safetensors else _ARTIFACT_WAIT_S)
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
    for _, shard in shards:
        dequantize_cache_inplace(shard.get("hs_cache", {}), ("hidden_states",))
    if len(shards) == 1:
        return shards[0][1]
    return _replicated_hs_shard(shards)


def _qk_rebuild_kall(entry: dict) -> None:
    if not isinstance(entry, dict) or "k_full" not in entry:
        return
    k_list = []
    for full, ends in zip(entry["k_full"], entry["k_prefix_ends"]):
        k_list.extend(full[:int(L)] for L in ends)
    entry["k_all"] = k_list
    entry.pop("k_full", None)
    entry.pop("k_prefix_ends", None)


def _load_and_merge_qk_safetensors(hook_dir: str, run_id: str, st_paths: List[str]) -> Dict[str, Any]:
    def build_qk_entries(sf, meta):
        default_mode = meta.get("hookq_mode", "all_tokens")
        legacy_seq_lens = meta.get("seq_lens")
        q_seq_lens = meta.get("q_seq_lens", legacy_seq_lens)
        k_seq_lens = meta.get("k_seq_lens", legacy_seq_lens)
        compact = meta.get("compact_all_tokens", False)
        k_full_lens = meta.get("k_full_lens")
        k_prefix_ends = meta.get("k_prefix_ends")
        out = {}
        for item in meta["layer_order"]:
            t_q = sf.get_tensor(item["key_q"])
            t_k = sf.get_tensor(item["key_k"])
            hookq_mode = item.get("hookq_mode", default_mode)
            if compact and hookq_mode == "all_tokens":
                k_list = []
                off = 0
                for flen, ends in zip(k_full_lens, k_prefix_ends):
                    full_i = t_k[off:off + flen]
                    off += flen
                    for L in ends:
                        k_list.append(full_i[:L])
                q_list = []
                off = 0
                for ql in q_seq_lens:
                    q_list.append(t_q[off:off + ql])
                    off += ql
            else:
                if k_seq_lens is not None:
                    k_list = [t_k[i, :k_seq_lens[i], :] for i in range(t_k.shape[0])]
                else:
                    k_list = [t_k[i] for i in range(t_k.shape[0])]
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
    """Load all QK shards for run_id and merge them into a single cache."""
    import torch

    safetensors = os.environ.get("MIA_USE_SAFETENSORS", "0") == "1"
    if safetensors:
        st_paths = _artifact_glob(hook_dir, run_id, "qk.safetensors", timeout=_ARTIFACT_WAIT_S)
        if st_paths:
            return _load_and_merge_qk_safetensors(hook_dir, run_id, st_paths)

    paths = _artifact_glob(hook_dir, run_id, "qk.pt",
                           timeout=0.0 if safetensors else _ARTIFACT_WAIT_S)
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
    _refuse_partial_qk_shard_set(shareds, os.path.join(hook_dir, run_id))

    if len(shareds) == 1:
        cache = shareds[0][1]
        cache.pop("tp_shard", None)
        cache.setdefault("meta", {})
        cache["meta"].setdefault("num_shareds", 1)
        for _entry in cache.get("qk_cache", {}).values():
            _qk_rebuild_kall(_entry)
        from mia.artifact_quant import dequantize_cache_inplace
        dequantize_cache_inplace(cache.get("qk_cache", {}), ("q", "k_all"))
        return cache

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

    module_names = set()
    for _, shared in shareds:
        module_names.update(shared.get("qk_cache", {}).keys())

    for module_name in module_names:
        layer_num = None
        per_shared_q: List[List[Any]] = []
        per_shared_k: List[List[Any]] = []
        for _, shared in shareds:
            qk = shared.get("qk_cache", {}).get(module_name)
            if qk is None:
                continue
            _qk_rebuild_kall(qk)
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
    """Call ``analyzer.analyze`` with whichever of run_id/run_ids it accepts."""
    import inspect
    sig = inspect.signature(analyzer.analyze)
    kwargs = {"analyzer_spec": analyzer_spec}
    if "run_ids" in sig.parameters and run_ids is not None:
        kwargs["run_ids"] = run_ids
    elif "run_id" in sig.parameters and run_id is not None:
        kwargs["run_id"] = run_id
    return analyzer.analyze(**kwargs)

