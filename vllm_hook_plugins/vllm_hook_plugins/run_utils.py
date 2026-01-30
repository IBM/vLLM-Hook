from __future__ import annotations

import glob
import os
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


def _artifact_glob(hook_dir: str, run_id: str) -> List[str]:
    """Return a list of shared artifact files for a run."""
    patt_new = os.path.join(hook_dir, run_id, "**", "qk.pt")
    paths = glob.glob(patt_new, recursive=True)
    return paths


def load_and_merge_qk_cache(hook_dir: str, run_id: str):
    """
    Load all shareds for run_id and merge into a single cache.
    Returns a dict with keys: config, qk_cache, meta.
    """
    import torch

    shared_paths = _artifact_glob(hook_dir, run_id)
    if not shared_paths:
        raise FileNotFoundError(
            f"No Q/K cache artifacts found for run_id={run_id} under {hook_dir}"
        )

    shareds = []
    for p in shared_paths:
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
