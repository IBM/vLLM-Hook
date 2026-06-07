from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

from vllm_hook_plugins.run_utils import (
    load_and_merge_hs_cache as shared_load_and_merge_hs_cache,
)


def _artifact_glob(hook_dir: str, run_id: str) -> List[str]:
    patterns = [
        os.path.join(hook_dir, run_id, "**", "qk.pt"),
        os.path.join(hook_dir, run_id, "**", "qkv.pt"),
    ]
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    return paths


def _normalize_qkv_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    qkv_cache = cache.get("qkv_cache")
    if not qkv_cache:
        return cache

    normalized: Dict[str, Any] = {
        "config": cache["config"],
        "qk_cache": {},
        "meta": cache.get("meta", {}),
    }

    grouped: Dict[int, Dict[str, Any]] = {}
    for module_name, proj_data in qkv_cache.items():
        layer_num = proj_data["layer_num"]
        proj_kind = proj_data["proj_kind"]
        layer_entry = grouped.setdefault(layer_num, {"layer_num": layer_num})
        layer_entry[proj_kind] = proj_data["tokens"]
        layer_entry.setdefault("module_name", module_name.rsplit(".", 1)[0])

    for layer_data in grouped.values():
        if "q" not in layer_data or "k" not in layer_data:
            continue
        normalized["qk_cache"][layer_data["module_name"]] = {
            "q": layer_data["q"],
            "k_all": layer_data["k"],
            "layer_num": layer_data["layer_num"],
        }

    return normalized


def load_and_merge_qk_cache(hook_dir: str, run_id: str):
    import torch

    shared_paths = _artifact_glob(hook_dir, run_id)
    if not shared_paths:
        raise FileNotFoundError(
            f"No Q/K cache artifacts found for run_id={run_id} under {hook_dir}"
        )

    shareds = []
    for path in shared_paths:
        cache = torch.load(path, map_location="cpu")
        cache = _normalize_qkv_cache(cache)
        meta = cache.get("meta", {})
        tp_rank = int(meta.get("tp_rank", 0))
        shareds.append((tp_rank, cache))
    shareds.sort(key=lambda item: item[0])

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


def load_and_merge_hs_cache(hook_dir: str, run_id: str) -> Dict[str, Any]:
    """Load Metal hidden-state artifacts through the Metal analyzer boundary.

    Metal currently emits the shared hidden_states.pt schema, so this delegates
    to the common loader while keeping analyzer imports backend-specific.
    """
    return shared_load_and_merge_hs_cache(hook_dir, run_id)
