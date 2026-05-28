from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Any


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


def _artifact_glob(hook_dir: str, run_id: str, filename: str, timeout: float = 0.0) -> List[str]:
    """Return artifact paths under run_id, optionally polling.

    When VLLM_HOOK_ASYNC_SAVE=1 the worker hands the save off to a background
    thread before execute_model() returns, so the file may not exist yet when
    analyze() is called. Pass timeout > 0 (e.g. 5.0) to poll until the file
    (and its .json sidecar, for safetensors) appears or the deadline is reached.
    """
    patt = os.path.join(hook_dir, run_id, "**", filename)
    paths = glob.glob(patt, recursive=True)
    if paths or timeout <= 0:
        return paths
    json_patt = (
        os.path.join(hook_dir, run_id, "**", filename.replace(".safetensors", ".json"))
        if filename.endswith(".safetensors") else None
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.001)
        paths = glob.glob(patt, recursive=True)
        # main tensor exists AND (no sidecar needed OR sidecar also exists)
        if paths and (json_patt is None or glob.glob(json_patt, recursive=True)):
            return paths
    return []


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
    for p in st_paths:
        meta_path = p.replace(f"{basename}.safetensors", f"{basename}.json")
        with open(meta_path) as f:
            meta = json.load(f)
        tp_rank = meta.get("tp_rank", 0)
        with safe_open(p, framework="pt", device="cpu") as sf:
            shard_cache = build_entries(sf, meta)
        shard = {"config": meta["config"], cache_key: shard_cache}
        if "peak_gpu_mb" in meta:
            shard["peak_gpu_mb"] = meta.get("peak_gpu_mb", 0.0)
        shards.append((tp_rank, shard))
    shards.sort(key=lambda x: x[0])
    return shards


def _merge_shards_by_module(shards, cache_key: str, tensor_keys):
    """Merge TP shards into a single cache by concatenating per-request tensors
    along the last dim (the hidden / kv-hidden dimension is sharded across TP).

    ``tensor_keys`` is the list of tensor list keys per entry (e.g. ``["q","k_all"]``
    or ``["hidden_states"]``). Other keys (e.g. ``layer_num``) are taken from
    the first non-empty shard.
    """
    import torch

    if len(shards) == 1:
        return shards[0][1]

    base_cfg = shards[0][1]["config"]
    merged: Dict[str, Any] = {"config": base_cfg, cache_key: {}}
    if "peak_gpu_mb" in shards[0][1]:
        merged["peak_gpu_mb"] = shards[0][1].get("peak_gpu_mb", 0.0)

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
                # t is (bs, max_seq_len, hidden_size) — unpad
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
    Auto-detects safetensors format when VLLM_HOOK_USE_SAFETENSORS=1.
    """
    import torch

    async_save = os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1"
    poll_timeout = 5.0 if async_save else 0.0

    if os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1":
        st_paths = _artifact_glob(hook_dir, run_id, "hidden_states.safetensors", timeout=poll_timeout)
        if not st_paths:
            raise FileNotFoundError(
                f"No safetensors artifacts found for run_id={run_id} under {hook_dir}"
            )
        return _load_and_merge_hs_safetensors(hook_dir, run_id, st_paths)

    paths = _artifact_glob(hook_dir, run_id, "hidden_states.pt", timeout=poll_timeout)
    if not paths:
        raise FileNotFoundError(
            f"No hidden-state artifacts found for run_id={run_id} under {hook_dir}"
        )

    shards = []
    for p in paths:
        cache = torch.load(p, map_location="cpu")
        meta = cache.get("meta", {})
        tp_rank = int(meta.get("tp_rank", 0))
        shards.append((tp_rank, cache))
    shards.sort(key=lambda x: x[0])

    if len(shards) == 1:
        return shards[0][1]

    base_cfg = shards[0][1]["config"]
    merged: Dict[str, Any] = {
        "config": base_cfg,
        "hs_cache": {},
    }

    module_names: set = set()
    for _, shard in shards:
        module_names.update(shard.get("hs_cache", {}).keys())

    for module_name in module_names:
        layer_num = None
        per_shard_hs: List[List[Any]] = []
        for _, shard in shards:
            entry = shard.get("hs_cache", {}).get(module_name)
            if entry is None:
                continue
            if layer_num is None:
                layer_num = entry.get("layer_num")
            per_shard_hs.append(entry["hidden_states"])

        bs = len(per_shard_hs[0])
        merged_hs: List[Any] = []
        for i in range(bs):
            parts = [hs[i] for hs in per_shard_hs]
            merged_hs.append(torch.cat(parts, dim=-1))

        merged["hs_cache"][module_name] = {
            "hidden_states": merged_hs,
            "layer_num": layer_num,
        }

    return merged


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
        out = {}
        for item in meta["layer_order"]:
            t_q = sf.get_tensor(item["key_q"])
            t_k = sf.get_tensor(item["key_k"])
            hookq_mode = item.get("hookq_mode", default_mode)
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
    return _merge_shards_by_module(shards, "qk_cache", ["q", "k_all"])


def load_and_merge_qk_cache(hook_dir: str, run_id: str):
    """
    Load all shareds for run_id and merge into a single cache.
    Returns a dict with keys: config, qk_cache, meta.
    Auto-detects safetensors format when VLLM_HOOK_USE_SAFETENSORS=1.
    """
    import torch

    async_save = os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1"
    poll_timeout = 5.0 if async_save else 0.0

    if os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1":
        st_paths = _artifact_glob(hook_dir, run_id, "qk.safetensors", timeout=poll_timeout)
        if not st_paths:
            raise FileNotFoundError(
               f"No safetensors artifacts found for run_id={run_id} under {hook_dir}"
            )
        return _load_and_merge_qk_safetensors(hook_dir, run_id, st_paths)

    paths = _artifact_glob(hook_dir, run_id, "qk.pt", timeout=poll_timeout)
    if not paths:
        raise FileNotFoundError(
            f"No Q/K cache artifacts found for run_id={run_id} under {hook_dir}"
        )

    shareds = []
    for p in paths:
        cache = torch.load(p, map_location="cpu")
        meta = cache.get("meta", {})
        tp_rank = int(meta.get("tp_rank", 0))
        shareds.append((tp_rank, cache))
    shareds.sort(key=lambda x: x[0])

    # Single shared: return as-is.
    if len(shareds) == 1:
        cache = shareds[0][1]
        cache.setdefault("meta", {})
        cache["meta"].setdefault("num_shareds", 1)
        return cache

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

    Used by HookLLM and HookClient to dispatch the disk-path analyze call.
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
