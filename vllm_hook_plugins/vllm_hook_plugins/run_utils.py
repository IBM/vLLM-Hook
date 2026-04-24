from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Any


def read_run_ids(run_id_file: str) -> List[str]:
    """Read all run IDs from the RUN_ID file."""
    if not run_id_file or not os.path.exists(run_id_file):
        return []
    with open(run_id_file, "r") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]


def latest_run_id(run_id_file: str) -> str:
    ids = read_run_ids(run_id_file)
    if not ids:
        raise FileNotFoundError(f"No run IDs found in {run_id_file!r}.")
    return ids[-1]


def _qk_glob(hook_dir: str, run_id: str, filename: str, timeout: float = 0.0) -> List[str]:
    """Return a list of shared artifact files for a run."""
    patt = os.path.join(hook_dir, run_id, "**", filename)
    paths = glob.glob(patt, recursive=True)
    if paths or timeout <= 0:
        return paths
    json_patt = (
        os.path.join(hook_dir, run_id, "**", "qk.json")
        if filename == "qk.safetensors" else None
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.001)
        paths = glob.glob(patt, recursive=True)
        if paths and (json_patt is None or glob.glob(json_patt, recursive=True)):
            return paths
    return []


def _hs_glob(hook_dir: str, run_id: str, filename: str, timeout: float = 0.0) -> List[str]:
    """Return artifact under run_id, optionally polling.

    When VLLM_HOOK_ASYNC_SAVE=1 the worker hands the save off to a background
    thread before execute_model() returns, so the file may not exist yet when
    analyze() is called. Pass timeout > 0 (e.g. 5.0) to poll until the file
    appears or the deadline is reached.
    """
    patt = os.path.join(hook_dir, run_id, "**", filename)
    paths = glob.glob(patt, recursive=True)
    if paths or timeout <= 0:
        return paths
    json_patt = (
        os.path.join(hook_dir, run_id, "**", "hidden_states.json")
        if filename == "hidden_states.safetensors" else None
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.001)
        paths = glob.glob(patt, recursive=True)
        # the main tensor exist and either we're not looking for a JSON sidecar (non-safetensors case), or the JSON also exists 
        if paths and (json_patt is None or glob.glob(json_patt, recursive=True)):
            return paths
    return []


def _load_and_merge_hs_safetensors(
    hook_dir: str, run_id: str, st_paths: List[str]
) -> Dict[str, Any]:
    """Load safetensors artifacts and merge across TP shards (same logic as .pt path)."""
    import json
    import torch
    from safetensors import safe_open

    shards = []
    for p in st_paths:
        meta_path = p.replace("hidden_states.safetensors", "hidden_states.json")
        with open(meta_path) as f:
            meta = json.load(f)
        tp_rank = meta.get("tp_rank", 0)
        batch_size = meta["batch_size"]

        seq_lens = meta.get("seq_lens")  # present for all_tokens mode, None otherwise

        hs_cache: Dict[str, Any] = {}
        with safe_open(p, framework="pt", device="cpu") as sf:
            for item in meta["layer_order"]:
                t = sf.get_tensor(item["key"])
                if seq_lens is not None:
                    # all_tokens: t is (bs, max_seq_len, hidden_size) — unpad
                    hidden_states = [t[i, :seq_lens[i], :] for i in range(batch_size)]
                else:
                    # last_token: t is (bs, hidden_size)
                    hidden_states = [t[i] for i in range(batch_size)]
                hs_cache[item["module_name"]] = {
                    "hidden_states": hidden_states,
                    "layer_num": item["layer_num"],
                }

        shards.append((tp_rank, {
            "config": meta["config"],
            "hs_cache": hs_cache,
            "peak_gpu_mb": meta.get("peak_gpu_mb", 0.0),
        }))
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
        st_paths = _hs_glob(hook_dir, run_id, "hidden_states.safetensors", timeout=poll_timeout)
        if not st_paths:
            raise FileNotFoundError(
                f"No safetensors artifacts found for run_id={run_id} under {hook_dir}"
            )
        return _load_and_merge_hs_safetensors(hook_dir, run_id, st_paths)

    paths = _hs_glob(hook_dir, run_id, "hidden_states.pt", timeout=poll_timeout)
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
    import json
    import torch
    from safetensors import safe_open
    from torch.nn.utils.rnn import pad_sequence

    shards = []
    for p in st_paths:
        meta_path = p.replace("qk.safetensors", "qk.json")
        with open(meta_path) as f:
            meta = json.load(f)
        tp_rank = meta.get("tp_rank", 0)
        hookq_mode = meta.get("hookq_mode", "all_tokens")
        batch_size = meta.get("batch_size")

        qk_cache: Dict[str, Any] = {}
        with safe_open(p, framework="pt", device="cpu") as sf:
            for item in meta["layer_order"]:
                t_q = sf.get_tensor(item["key_q"])
                t_k = sf.get_tensor(item["key_k"])
                seq_lens = meta.get("seq_lens")
                if hookq_mode == "all_tokens":
                    # q and k are padded (bs, max_seq, heads*head_dim) — unpad using seq_lens
                    if seq_lens is not None:
                        q_list = [t_q[i, :seq_lens[i], :] for i in range(t_q.shape[0])]
                        k_list = [t_k[i, :seq_lens[i], :] for i in range(t_k.shape[0])]
                    else:
                        q_list = [t_q[i] for i in range(t_q.shape[0])]
                        k_list = [t_k[i] for i in range(t_k.shape[0])]
                else:
                    # last_token: q is (bs, head_dim); k is padded (bs, max_seq, head_dim) — unpad k
                    q_list = [t_q[i] for i in range(t_q.shape[0])]
                    if seq_lens is not None:
                        k_list = [t_k[i, :seq_lens[i], :] for i in range(t_k.shape[0])]
                    else:
                        k_list = [t_k[i] for i in range(t_k.shape[0])]
                qk_cache[item["module_name"]] = {
                    "q": q_list,
                    "k_all": k_list,
                    "layer_num": item["layer_num"],
                }

        shards.append((tp_rank, {"config": meta["config"], "qk_cache": qk_cache}))
    shards.sort(key=lambda x: x[0])

    if len(shards) == 1:
        return shards[0][1]

    base_cfg = shards[0][1]["config"]
    merged: Dict[str, Any] = {"config": base_cfg, "qk_cache": {}}
    module_names: set = set()
    for _, shard in shards:
        module_names.update(shard["qk_cache"].keys())

    for module_name in module_names:
        layer_num = None
        per_shard_q: List[List[Any]] = []
        per_shard_k: List[List[Any]] = []
        for _, shard in shards:
            entry = shard["qk_cache"].get(module_name)
            if entry is None:
                continue
            if layer_num is None:
                layer_num = entry["layer_num"]
            per_shard_q.append(entry["q"])
            per_shard_k.append(entry["k_all"])

        bs = len(per_shard_q[0])
        merged["qk_cache"][module_name] = {
            "q": [torch.cat([qs[i] for qs in per_shard_q], dim=-1) for i in range(bs)],
            "k_all": [torch.cat([ks[i] for ks in per_shard_k], dim=-1) for i in range(bs)],
            "layer_num": layer_num,
        }

    return merged


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
        st_paths = _qk_glob(hook_dir, run_id, "qk.safetensors", timeout=poll_timeout)
        if not st_paths:
            raise FileNotFoundError(
               f"No safetensors artifacts found for run_id={run_id} under {hook_dir}"
            )
        return _load_and_merge_qk_safetensors(hook_dir, run_id, st_paths)

    paths = _qk_glob(hook_dir, run_id, "qk.pt", timeout=poll_timeout)
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
