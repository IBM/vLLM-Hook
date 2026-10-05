"""Pure serialize+write functions for captured artifacts."""
from __future__ import annotations

import os

import torch
from torch.nn.utils.rnn import pad_sequence

from mia._profiler import PROF
from mia.workers._common import save_pt_atomic, save_safetensors_atomic


def reconstruct_k_all(parts: list, prefix_ends) -> list:
    """Rebuild the growing-prefix k_all from per-step key chunks + absolute prefix lengths."""
    if not prefix_ends:
        return parts
    full = torch.cat(parts, dim=0)
    return [full[:int(L)] for L in prefix_ends]


def save_qk_cache_safetensors(cpu_cache: dict, run_dir: str,
                              default_hookq_mode: str, tp_rank: int) -> None:
    """Write a QK ``cpu_cache`` as safetensors + JSON sidecar (or .pt for score/quant caches)."""
    if any("scores" in e or e.get("capture") == "score" or "q_qmeta" in e
           for e in cpu_cache["qk_cache"].values()):
        save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
        return

    at_entries = [e for e in cpu_cache["qk_cache"].values()
                  if e.get("hookq_mode", default_hookq_mode) == "all_tokens"]
    compact = bool(at_entries) and all("k_full" in e for e in at_entries)
    if not compact:
        for _e in cpu_cache["qk_cache"].values():
            if isinstance(_e, dict) and "k_full" in _e and "k_all" not in _e:
                _e["k_all"] = [f[:int(L)] for f, ends in zip(_e["k_full"], _e["k_prefix_ends"])
                               for L in ends]

    flat_dict: dict = {}
    layer_order: list = []
    batch_size = 0
    q_seq_lens: list = []
    k_seq_lens: list = []
    k_full_lens: list = []
    k_prefix_ends: list = []

    with PROF.timed("writer.serialize.pad"):
        for mod_name, entry in cpu_cache["qk_cache"].items():
            safe_key_q = mod_name.replace(".", "__") + "__q"
            safe_key_k = mod_name.replace(".", "__") + "__k"
            mode = entry.get("hookq_mode", default_hookq_mode)

            if mode == "all_tokens" and compact:
                flat_dict[safe_key_q] = torch.cat(entry['q'], dim=0)
                flat_dict[safe_key_k] = torch.cat(entry['k_full'], dim=0)
                if not q_seq_lens:
                    q_seq_lens = [int(t.shape[0]) for t in entry['q']]
                    k_full_lens = [int(t.shape[0]) for t in entry['k_full']]
                    k_prefix_ends = [[int(L) for L in ends] for ends in entry['k_prefix_ends']]
            elif mode == "all_tokens":
                flat_dict[safe_key_q] = pad_sequence(entry['q'], batch_first=True)
                flat_dict[safe_key_k] = pad_sequence(entry['k_all'], batch_first=True)
                if not q_seq_lens:
                    q_seq_lens = [t.shape[0] for t in entry['q']]
                if not k_seq_lens:
                    k_seq_lens = [t.shape[0] for t in entry['k_all']]
            else:
                flat_dict[safe_key_q] = torch.stack(entry['q'])
                flat_dict[safe_key_k] = pad_sequence(entry['k_all'], batch_first=True)
                if not k_seq_lens:
                    k_seq_lens = [t.shape[0] for t in entry['k_all']]

            batch_size = flat_dict[safe_key_q].shape[0]
            layer_order.append({
                "key_q": safe_key_q,
                "key_k": safe_key_k,
                "module_name": mod_name,
                "layer_num": entry["layer_num"],
                "hookq_mode": mode,
            })

    meta = {
        "config": cpu_cache["config"],
        "layer_order": layer_order,
        "batch_size": batch_size,
        "hookq_mode": default_hookq_mode,
        "tp_rank": int(tp_rank),
    }
    if q_seq_lens:
        meta["q_seq_lens"] = q_seq_lens
    if k_seq_lens:
        meta["k_seq_lens"] = k_seq_lens
    if compact:
        meta["compact_all_tokens"] = True
        meta["k_full_lens"] = k_full_lens
        meta["k_prefix_ends"] = k_prefix_ends
    if isinstance(cpu_cache.get("tp_shard"), dict):
        meta["tp_shard"] = dict(cpu_cache["tp_shard"])
    save_safetensors_atomic(flat_dict, meta, run_dir, "qk")


def save_hs_cache_safetensors(cpu_cache: dict, run_dir: str,
                              default_hs_mode: str, tp_rank: int) -> None:
    """Write an HS ``cpu_cache`` as safetensors + JSON sidecar (or .pt for quant caches)."""
    if any("hidden_states_qmeta" in e for e in cpu_cache["hs_cache"].values()):
        save_pt_atomic(cpu_cache, os.path.join(run_dir, "hidden_states.pt"))
        return

    flat_dict: dict = {}
    layer_order: list = []
    batch_size = 0
    seq_lens: list = []

    for mod_name, entry in cpu_cache["hs_cache"].items():
        hs_list = entry["hidden_states"]
        safe_key = mod_name.replace(".", "__")
        mode = entry.get("hs_mode", default_hs_mode)

        if mode == "all_tokens":
            stacked = torch.cat(hs_list, dim=0)
            if not seq_lens:
                seq_lens = [int(t.shape[0]) for t in hs_list]
        else:
            stacked = torch.stack(hs_list)

        batch_size = stacked.shape[0]
        flat_dict[safe_key] = stacked
        layer_order.append({
            "key": safe_key,
            "module_name": mod_name,
            "layer_num": entry["layer_num"],
            "hs_mode": mode,
        })

    meta: dict = {
        "config": cpu_cache["config"],
        "layer_order": layer_order,
        "batch_size": batch_size,
        "hs_mode": default_hs_mode,
        "peak_gpu_mb": cpu_cache.get("peak_gpu_mb", 0.0),
        "tp_rank": int(tp_rank),
    }
    if seq_lens:
        meta["seq_lens"] = seq_lens
    save_safetensors_atomic(flat_dict, meta, run_dir, "hidden_states")


def write_artifact(worker_kind: str, cpu_cache: dict, run_dir: str, default_mode: str,
                   tp_rank: int, use_safetensors: bool, force_pt: bool, pt_filename: str) -> None:
    """Serialize and write one cpu_cache."""
    os.makedirs(run_dir, exist_ok=True)
    if use_safetensors and not force_pt:
        if worker_kind == "qk":
            save_qk_cache_safetensors(cpu_cache, run_dir, default_mode, tp_rank)
        else:
            save_hs_cache_safetensors(cpu_cache, run_dir, default_mode, tp_rank)
    else:
        save_pt_atomic(cpu_cache, os.path.join(run_dir, pt_filename))

