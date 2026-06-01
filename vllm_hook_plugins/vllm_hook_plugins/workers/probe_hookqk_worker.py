import os
import math
import pickle
import torch
from typing import TYPE_CHECKING, Any, Dict, List
import zstandard as zstd
from vllm.forward_context import get_forward_context
from vllm.distributed import parallel_state as ps

from vllm_hook_plugins.workers._common import (
    background_save_loop,
    clear_states_for_req,
    get_query_metadata,
    init_async_save_thread,
    iter_matched_modules,
    iter_matching_req_ids,
    match_attn,
    save_pt_atomic,
    save_safetensors_atomic,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _read_cached_keys(
    module_name,
    attn_metadata,
    req_idx: int,
    num_cached: int,
    total_len: int,
):
    """Read cached prefix keys from vLLM's paged KV cache.

    When prefix caching is active, the hook only fires for non-cached tokens.
    This function reconstructs the missing prefix keys by reading directly from
    vLLM's KV cache blocks, keyed by the block_table entry for this request.

    Returns a tensor of shape (num_cached, num_kv_heads * head_size) on the
    same device as the KV cache, or None on any error (caller falls back to
    new-tokens-only capture).
    """
    try:
        ctx = get_forward_context()
        # kv_cache is bound to the vLLM Attention wrapper in the forward context,
        # not to the PyTorch module. Access via no_compile_layers[layer_name].kv_cache.
        kv_cache = ctx.no_compile_layers[module_name].kv_cache
        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]

        block_size   = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size    = key_cache.shape[3]

        # block_table: [batch_size, max_blocks_per_seq]
        block_table = attn_metadata.block_table
        num_blocks_needed = math.ceil(total_len / block_size)
        block_ids = block_table[req_idx, :num_blocks_needed]  # [num_blocks_needed]

        # Gather and flatten: [num_blocks_needed * block_size, kv_hidden]
        prefix_keys = key_cache[block_ids].reshape(-1, num_kv_heads * head_size)

        # Trim to exact cached token count (last block may be partially filled)
        return prefix_keys[:num_cached].detach()
    except Exception:
        return None

class ProbeHookQKWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (ProbeHookQKWorker,) at runtime,
    so self is the Worker instance. Methods are callable via collective_rpc.
    """

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    # Default capture phase — matches the old hooks_on=(True, False) registry entry.
    # Can be overridden per-request via extra_args["hooks_on"].
    _default_hooks_on: str = "prefill"

    # Per-request captured QK states (API serving path):
    # internal_req_id -> {module_name -> {"q": [...], "k_all": [...], "layer_num": int}}
    _captured_states: dict = {}
    _hooks_installed: bool = False

    def install_hooks(self):
        """Install forward hooks on all target attention modules. Idempotent.

        Callable via collective_rpc("install_hooks") — the plugin calls this
        lazily on the first request that sets output_qk in extra_args.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}  # RPC path
        self._disk_states = {}      # disk path: same shape, written via flush_disk()
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # Worker-wide fallback when extra_args["hookq_mode"] is missing.
        self.hookq_mode = "all_tokens"

        # Background I/O thread (shared by all disk-save requests on this worker).
        init_async_save_thread(self, self._background_save_loop, "vllm-hook-qk-io")

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        num_h = int(getattr(text_cfg, "num_attention_heads"))
        num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
        hidden = int(getattr(text_cfg, "hidden_size"))
        head_dim = hidden // num_h
        attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=attn_mult,
        )

        # Only TP rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        def qkv_hook(input, module_name, attn_module=None):
            # Fast-path: only rank 0 captures (RPC and disk paths both need it).
            if not self._should_capture:
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)

            # Warmup or non-attention passes: nothing to do
            if metadata is None:
                return
            if torch.cuda.is_current_stream_capturing():
                return None

            query_start_loc, seq_lens = get_query_metadata(metadata)
            if query_start_loc is None:
                return

            bs = len(query_start_loc) - 1
            last_indices = query_start_loc

            layer_num = match_attn(module_name)

            # Per-request capture. Each request in the batch may route to
            # either _captured_states (RPC path) or _disk_states (disk path)
            # based on its extra_args.
            try:
                req_ids = self.model_runner.input_batch.req_ids
            except Exception:
                return

            for i in range(bs):
                req_id = req_ids[i]
                req_state = self.model_runner.requests.get(req_id)
                if req_state is None or req_state.sampling_params is None:
                    continue
                extra = req_state.sampling_params.extra_args
                if not extra or extra.get("output_qk") is None:
                    continue

                # output_qk accepts three forms, matching the old layer_to_heads config:
                #   True             -> capture all layers
                #   [layer_ids]      -> capture specific layers
                #   {layer: [heads]} -> capture specific layers (heads used downstream by analyzer)
                # The worker only uses the keys for layer filtering — head info is
                # forwarded to the analyzer by the caller, same as the old env-var flow.
                output_spec = extra.get("output_qk")
                if isinstance(output_spec, dict):
                    layer_set = {int(k) for k in output_spec.keys()}
                    if layer_num not in layer_set:
                        continue
                elif isinstance(output_spec, list):
                    if layer_num not in output_spec:
                        continue

                # hooks_on: "prefill" (default) | "decode" | "both"
                # Uses output_token_ids == [] on the worker-side CachedRequestState
                # to detect the first (prefill) pass. This is robust to prefix
                # caching where query_len < seq_len even on the first pass.
                hooks_on = extra.get("hooks_on", self._default_hooks_on)
                if hooks_on != "both":
                    is_prefill = len(req_state.output_token_ids) == 0
                    if hooks_on == "prefill" and not is_prefill:
                        continue
                    if hooks_on == "decode" and is_prefill:
                        continue

                # Per-request mode: extra_args["hookq_mode"] overrides the worker default.
                req_mode = extra.get("hookq_mode", self.hookq_mode)

                start = int(last_indices[i].item())
                end = int(last_indices[i + 1].item())

                # Accumulate GPU tensors — clone() copies data immediately so we
                # own the buffer; .cpu() is deferred to retrieval/flush.
                if req_mode == "all_tokens":
                    q_tok = input[0][start:end, :].detach().clone()
                else:
                    q_tok = input[0][end - 1, :].detach().clone()
                k_tok = input[1][start:end, :].detach().clone()

                # Reconstruct full k_all when prefix caching is active.
                # seq_lens[i] = total sequence length (cached + new tokens).
                # query_len = new tokens only (what the hook captured above).
                # If num_cached > 0, read the missing prefix keys directly from
                # vLLM's paged KV cache and prepend them to k_tok.
                if seq_lens is not None and attn_module is not None:
                    try:
                        total_len = int(seq_lens[i].item()) if hasattr(seq_lens[i], 'item') else int(seq_lens[i])
                        query_len = end - start
                        num_cached = total_len - query_len
                        if num_cached > 0:
                            prefix_k = _read_cached_keys(module_name, metadata if not isinstance(metadata, dict) else next(iter(metadata.values())), i, num_cached, total_len)
                            if prefix_k is not None:
                                k_tok = torch.cat([prefix_k.to(k_tok.device, dtype=k_tok.dtype), k_tok], dim=0)
                    except Exception:
                        pass

                # Route to disk or RPC bucket based on save_to_disk flag.
                bucket = self._disk_states if extra.get("save_to_disk") else self._captured_states
                if req_id not in bucket:
                    bucket[req_id] = {}
                layer_states = bucket[req_id]
                if module_name not in layer_states:
                    layer_states[module_name] = {"q": [], "k_all": [], "layer_num": layer_num, "hookq_mode": req_mode}
                layer_states[module_name]["q"].append(q_tok)
                layer_states[module_name]["k_all"].append(k_tok)

        # Hook every attention module. Per-request layer filtering via
        # extra_args['output_qk'] happens inside the hook closure.
        self._hooks = []
        matched = []
        for name, module, _ in iter_matched_modules(model, match_attn):
            hook = module.register_forward_hook(
                lambda _m, i, _o, n=name: qkv_hook(i, n, _m)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} hooks on layers: {matched}")

    # ------------------------------------------------------------------
    # API serving: collective_rpc-callable artifact retrieval
    # ------------------------------------------------------------------

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve and remove captured QK states for a completed request.

        Matches by "{external_req_id}-" prefix because vLLM internally
        transforms the user-provided request_id into "{request_id}-{random_suffix}".

        CPU transfer happens here (once per request, not per hook).
        Returns zstd-compressed pickle, or None if nothing was captured.
        """
        for req_id in iter_matching_req_ids(self._captured_states, external_req_id):
            layer_dict = self._captured_states.pop(req_id)
            cpu_dict = {}
            for mod_name, entry in layer_dict.items():
                from torch.nn.utils.rnn import pad_sequence
                mode = entry.get("hookq_mode", self.hookq_mode)
                if mode == "all_tokens":
                    q_stacked = pad_sequence([t.cpu() for t in entry["q"]], batch_first=True)
                else:
                    q_stacked = torch.stack([t.cpu() for t in entry["q"]])
                k_stacked = pad_sequence([t.cpu() for t in entry["k_all"]], batch_first=True)
                cpu_dict[mod_name] = {"q": q_stacked, "k_all": k_stacked, "layer_num": entry["layer_num"], "hookq_mode": mode}
            payload = {"qk_cache": cpu_dict, "config": self._conf}
            return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured states without returning them (cleanup on abort/disconnect)."""
        clear_states_for_req(self._captured_states, external_req_id)

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """Write captured Q/K for all requests in the batch to one artifact.

        Accepts a list of external_req_ids so all requests sharing a run_id
        are merged into one cpu_cache before writing — matching the old
        execute_model() behavior where the full batch was saved atomically.

        Returns True if any artifacts were written, False if nothing captured.
        """
        cpu_cache: dict = {"config": self._conf, "qk_cache": {}}
        found_any = False

        for external_req_id in external_req_ids:
            for req_id in iter_matching_req_ids(self._disk_states, external_req_id):
                layer_dict = self._disk_states.pop(req_id)
                if not layer_dict:
                    continue
                found_any = True
                for mod_name, entry in layer_dict.items():
                    cpu_entry = {
                        "q": [t.cpu() for t in entry["q"]],
                        "k_all": [t.cpu() for t in entry["k_all"]],
                        "layer_num": entry["layer_num"],
                        "hookq_mode": entry.get("hookq_mode", self.hookq_mode),
                    }
                    if mod_name in cpu_cache["qk_cache"]:
                        cpu_cache["qk_cache"][mod_name]["q"].extend(cpu_entry["q"])
                        cpu_cache["qk_cache"][mod_name]["k_all"].extend(cpu_entry["k_all"])
                    else:
                        cpu_cache["qk_cache"][mod_name] = cpu_entry

        if not found_any:
            return False

        tp_rank = int(ps.get_tensor_model_parallel_rank())
        run_dir = os.path.join(hook_dir, run_id, f"tp_rank_{tp_rank}")
        os.makedirs(run_dir, exist_ok=True)

        if os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1":
            self._save_queue.put((run_id, cpu_cache, run_dir))
        elif os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1":
            self._save_safetensors(cpu_cache, run_dir)
        else:
            save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
        return found_any

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        # safetensors stores fixed-shape tensors only, so per-request variable-length
        # q/k_all are padded into a single batched tensor and unpadded on load using
        # the seq_lens stored in the JSON sidecar. q and k_all may have different
        # lengths when prefix caching is active (q = new tokens, k_all = new+cached),
        # so we track q_seq_lens and k_seq_lens separately.
        from torch.nn.utils.rnn import pad_sequence

        flat_dict: dict = {}
        layer_order: list = []
        batch_size = 0
        q_seq_lens: list = []  # only populated for all_tokens mode (q is 2D then)
        k_seq_lens: list = []  # always populated (k_all is always 2D)

        for mod_name, entry in cpu_cache["qk_cache"].items():
            safe_key_q = mod_name.replace(".", "__") + "__q"
            safe_key_k = mod_name.replace(".", "__") + "__k"
            mode = entry.get("hookq_mode", self.hookq_mode)

            if mode == "all_tokens":
                flat_dict[safe_key_q] = pad_sequence(entry['q'], batch_first=True)
                flat_dict[safe_key_k] = pad_sequence(entry['k_all'], batch_first=True)
                if not q_seq_lens:
                    q_seq_lens = [t.shape[0] for t in entry['q']]
            else:
                # last_token: q is 1D (head_dim,) per request — stack instead of pad.
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
            "hookq_mode": self.hookq_mode,
            "tp_rank": int(ps.get_tensor_model_parallel_rank()),
        }
        if q_seq_lens:
            meta["q_seq_lens"] = q_seq_lens
        if k_seq_lens:
            meta["k_seq_lens"] = k_seq_lens
        save_safetensors_atomic(flat_dict, meta, run_dir, "qk")

    def _background_save_loop(self):
        background_save_loop(self, "qk.pt")

