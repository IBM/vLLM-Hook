"""Pure serialize+write functions for captured artifacts.

The disk-save serialize (pad + safetensors-encode / pickle) is the ~200 ms flat contention
floor on the EngineCore busy-loop thread: doing it inline holds the CPython GIL and starves
decode. Running it in a separate *process* (own interpreter, own GIL) means that child
process can't call worker methods, so the serialize bodies live here as PURE functions taking
everything as arguments (no ``self``, no vLLM/CUDA imports beyond torch-CPU + safetensors).
Both the inline path and the writer-process path call the SAME function, so the on-disk
artifact is byte-identical either way.

These are verbatim extractions of ``qk_capture_worker._save_safetensors`` /
``hs_capture_worker._save_safetensors`` — ``self.hookq_mode``/``self.hs_mode`` become
the ``default_*_mode`` argument and ``ps.get_tensor_model_parallel_rank()`` becomes ``tp_rank``.
"""
from __future__ import annotations

import os

import torch

from mia._profiler import PROF
from mia.workers._common import save_pt_atomic, save_safetensors_atomic


def reconstruct_k_all(parts: list, prefix_ends) -> list:
    """Rebuild the growing-prefix k_all from per-step key chunks + absolute prefix lengths.

    Byte-identical to the tail of ``qk_capture_worker._k_all_cpu_list``: with no
    ``prefix_ends`` (eager path) the chunks are already full prefixes; otherwise concat the
    per-step new-key blocks into one ``full`` and slice each growing prefix ``full[:L]``.
    ``parts`` must be HOST tensors (callers that run off the GPU are CPU-only)."""
    if not prefix_ends:
        return parts
    full = torch.cat(parts, dim=0)
    return [full[:int(L)] for L in prefix_ends]


def save_qk_cache_safetensors(cpu_cache: dict, run_dir: str,
                              default_hookq_mode: str, tp_rank: int) -> None:
    """Write a QK ``cpu_cache`` as safetensors + JSON sidecar (or .pt for score/quant caches).

    Pure form of ``QKCaptureWorker._save_safetensors``."""
    # Score caches are ragged [S_q,S_k] lists that don't fit the fixed-shape safetensors
    # format — fall back to .pt (pickle). Also covers the async/writer-process save path,
    # which calls this function directly.
    if any("scores" in e or e.get("capture") == "score" or "q_qmeta" in e
           for e in cpu_cache["qk_cache"].values()):
        save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
        return
    # safetensors stores fixed-shape tensors only, so per-request variable-length
    # q/k are packed into a single batched tensor + per-entry lengths in the JSON
    # sidecar. Two all_tokens layouts (disambiguated by the sidecar ``compact_all_tokens``):
    #
    #   NATIVE-COMPACT (the buffer-graph path, what the worker writes): store q and the
    #   unique keys ``k_full`` each CONCATENATED into one (sum_len, dim) tensor -> O(seq).
    #   The reader splits q by ``q_seq_lens`` and rebuilds the growing-prefix k_all from
    #   ``k_full`` split by ``k_full_lens`` then sliced by ``k_prefix_ends``
    #   ([full[:L] for L in ends]) -- byte-identical (keys are append-only).
    #
    #   LEGACY PADDED (eager path with pre-expanded k_all, or older artifacts): pad each
    #   variable-length entry to the longest -> (batch, max_len, dim). This is O(seq^2) for
    #   the growing prefixes (a ~900-token request -> 100+ GB, the disk-sink blowup); only
    #   reached when an all_tokens entry lacks k_full.
    from torch.nn.utils.rnn import pad_sequence

    at_entries = [e for e in cpu_cache["qk_cache"].values()
                  if e.get("hookq_mode", default_hookq_mode) == "all_tokens"]
    compact = bool(at_entries) and all("k_full" in e for e in at_entries)
    if not compact:
        # Mixed/eager corner: expand any compact entry so the legacy pad path is uniform.
        for _e in cpu_cache["qk_cache"].values():
            if isinstance(_e, dict) and "k_full" in _e and "k_all" not in _e:
                _e["k_all"] = [f[:int(L)] for f, ends in zip(_e["k_full"], _e["k_prefix_ends"])
                               for L in ends]

    flat_dict: dict = {}
    layer_order: list = []
    batch_size = 0
    q_seq_lens: list = []    # per-step q lengths (all_tokens; splits/unpads q)
    k_seq_lens: list = []    # legacy padded k_all lengths (per growing prefix)
    k_full_lens: list = []   # compact: per-request unique-key row counts (splits k_full)
    k_prefix_ends: list = [] # compact: per-request [prefix end lengths] to rebuild k_all

    with PROF.timed("writer.serialize.pad"):
        for mod_name, entry in cpu_cache["qk_cache"].items():
            safe_key_q = mod_name.replace(".", "__") + "__q"
            safe_key_k = mod_name.replace(".", "__") + "__k"
            mode = entry.get("hookq_mode", default_hookq_mode)

            if mode == "all_tokens" and compact:
                # concatenate (O(seq)) instead of pad (O(seq^2))
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
                # last_token: q is 1D (head_dim,) per request — stack; k_all is one
                # snapshot per request, padded across the (bounded) batch, not O(seq^2).
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
    # TP > 1: this rank's head geometry (graph/tp_shard.py), so the loader can merge the ranks in
    # global head order. Absent at TP=1 -> the sidecar is byte-identical to before.
    if isinstance(cpu_cache.get("tp_shard"), dict):
        meta["tp_shard"] = dict(cpu_cache["tp_shard"])
    save_safetensors_atomic(flat_dict, meta, run_dir, "qk")


def save_hs_cache_safetensors(cpu_cache: dict, run_dir: str,
                              default_hs_mode: str, tp_rank: int) -> None:
    """Write an HS ``cpu_cache`` as safetensors + JSON sidecar (or .pt for quant caches).

    Pure form of ``HSCaptureWorker._save_safetensors``."""
    # A quantized cache (packed uint8 + scale + qmeta) doesn't fit the fixed-shape safetensors
    # format -> .pt. Covers the async/writer-process save path, which calls here.
    if any("hidden_states_qmeta" in e for e in cpu_cache["hs_cache"].values()):
        save_pt_atomic(cpu_cache, os.path.join(run_dir, "hidden_states.pt"))
        return

    flat_dict: dict = {}
    layer_order: list = []
    batch_size = 0
    seq_lens: list = []  # only populated for all_tokens mode

    for mod_name, entry in cpu_cache["hs_cache"].items():
        hs_list = entry["hidden_states"]
        safe_key = mod_name.replace(".", "__")
        mode = entry.get("hs_mode", default_hs_mode)

        if mode == "all_tokens":
            # NOT pad_sequence: it pads every entry out to the longest one, so a per-step
            # capture (prefill [P,H] + one [1,H] per decode step) explodes to
            # O(steps x max_seq) = O(seq^2) (a ~900-token request -> 100+ GB). Concatenate
            # into one (sum_seq, hidden) tensor + record per-entry lengths (O(seq)); the
            # reader splits by cumsum back into the same per-entry tensors (byte-identical).
            stacked = torch.cat(hs_list, dim=0)  # (sum_seq, hidden)
            if not seq_lens:
                # int() so the sidecar is plain-JSON safe.
                seq_lens = [int(t.shape[0]) for t in hs_list]
        else:
            stacked = torch.stack(hs_list)  # (bs, hidden_size)

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


# Dispatch used by both the in-process wrapper and the writer child process, so the two paths
# are guaranteed to serialize identically. worker_kind is "qk" or "hs".
def write_artifact(worker_kind: str, cpu_cache: dict, run_dir: str, default_mode: str,
                   tp_rank: int, use_safetensors: bool, force_pt: bool, pt_filename: str) -> None:
    """Serialize+write one cpu_cache. ``force_pt`` (score/quant caches) or ``not use_safetensors``
    -> .pt; else the fixed-shape safetensors + sidecar. Byte-identical to the thread path."""
    os.makedirs(run_dir, exist_ok=True)
    if use_safetensors and not force_pt:
        if worker_kind == "qk":
            save_qk_cache_safetensors(cpu_cache, run_dir, default_mode, tp_rank)
        else:
            save_hs_cache_safetensors(cpu_cache, run_dir, default_mode, tp_rank)
    else:
        save_pt_atomic(cpu_cache, os.path.join(run_dir, pt_filename))
