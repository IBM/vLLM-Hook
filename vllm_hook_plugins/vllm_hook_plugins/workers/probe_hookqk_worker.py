import os
import math
import pickle
import queue
import re
import threading
import torch
from typing import TYPE_CHECKING, Any, Dict, List
import zstandard as zstd
from vllm.forward_context import get_forward_context
from vllm.distributed import parallel_state as ps

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)

ATTN_PATTERNS = [
    # GPT-2: transformer.h.<i>.attn
    re.compile(r"^transformer\.h\.(\d+)\.attn.attn$"),

    # OPT: model.decoder.layers.<i>.self_attn
    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn.attn$"),

    # Qwen/LLaMA: model.layers.<i>.self_attn
    re.compile(r"^model\.layers\.(\d+)\.self_attn.attn$"),
]

def match_attn(name: str):
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


def _read_cached_keys(
    attn_module,
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
        virtual_engine = get_forward_context().virtual_engine
        # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        kv_cache = attn_module.kv_cache[virtual_engine]
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

        self.hookq_mode = os.environ.get("VLLM_HOOKQ_MODE", "all_tokens") # ["last_token", "all_tokens"]

        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())

        # Background I/O thread (shared by all disk-save requests on this worker).
        if os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1":
            if not getattr(self, '_io_thread_started', False):
                self._save_queue: queue.Queue = queue.Queue(maxsize=4)
                self._io_thread = threading.Thread(
                    target=self._background_save_loop,
                    daemon=True,
                    name="vllm-hook-qk-io",
                )
                self._io_thread.start()
                self._io_thread_started = True

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

            # The HS worker hooks on "model.layers.<i>", so we look up the corresponding attention key.
            # query_start_loc is the cumulative sum of per-request *query* token counts (shape [bs+1]).
            # Unlike cumsum(seq_lens), it excludes prefix-cached tokens and is always within hidden.shape[0].
            # For hybrid models (e.g. Qwen3.5), linear_attention layers have no entry in the metadata dict under their own key,
            # so we grab query_start_loc from any available entry rather than only the current layer's key.
            query_start_loc = getattr(metadata, "query_start_loc", None)
            seq_lens = getattr(metadata, "seq_lens", None)
            if query_start_loc is None and isinstance(metadata, dict):
                for entry in metadata.values():
                    query_start_loc = getattr(entry, "query_start_loc", None)
                    if query_start_loc is not None:
                        seq_lens = getattr(entry, "seq_lens", None)
                        break

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
                            prefix_k = _read_cached_keys(attn_module, metadata if not isinstance(metadata, dict) else next(iter(metadata.values())), i, num_cached, total_len)
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

        # register hooks on attention modules 
        self._hooks = []
        matched = []
        # When important_layers is empty (API path, no VLLM_HOOK_LAYER_HEADS env var),
        # hook every attention module. Per-request filtering via extra_args['output_qk']
        # happens inside the hook closure.
        for name, module in model.named_modules():
            layer_num = match_attn(name)
            if layer_num is None: # not an attention module
                continue
            if self.important_layers and layer_num not in self.important_layers:
                continue
            hook = module.register_forward_hook(
                lambda _m, i, _o, n=name: qkv_hook(i, n, _m)
            )
            self._hooks.append(hook)
            matched.append(name)
        
        print(f"Installed {len(self._hooks)} hooks on layers: {matched}")

    def _parse_layer_heads(self) -> Dict[int, List[int]]:
        ## Parse 'VLLM_HOOK_LAYER_HEADS' env var from string to dict: '0:0,3,6;15:2' → {0:[0,3,6], 15:[2]}
        layer_heads = os.environ.get("VLLM_HOOK_LAYER_HEADS", "")
        result = {}
        
        for part in layer_heads.split(";"):
            part = part.strip()
            if not part:
                continue
            
            layer_str, heads_str = part.split(":")
            layer_idx = int(layer_str)
            head_indices = sorted([int(h) for h in heads_str.split(",") if h])
            result[layer_idx] = head_indices
        
        return result

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
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id == external_req_id or req_id.startswith(prefix):
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
                payload = {
                    "qk_cache": cpu_dict,
                    "config": self._conf,
                }
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured states without returning them (cleanup on abort/disconnect)."""
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id == external_req_id or req_id.startswith(prefix):
                del self._captured_states[req_id]

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
            prefix = f"{external_req_id}-"
            for req_id in list(self._disk_states):
                if req_id != external_req_id and not req_id.startswith(prefix):
                    continue
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
            out_path = os.path.join(run_dir, "qk.pt")
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as f:
                torch.save(cpu_cache, f)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, out_path)
        return found_any

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        import json as _json
        from torch.nn.utils.rnn import pad_sequence
        from safetensors.torch import save_file as _st_save

        out_path = os.path.join(run_dir, "qk.safetensors")
        tmp_path = out_path + ".tmp"
        meta_path = os.path.join(run_dir, "qk.json")

        flat_dict: dict = {}
        layer_order: list = []
        batch_size = 0
        seq_lens: list = []

        for mod_name, entry in cpu_cache["qk_cache"].items():
            safe_key_q = mod_name.replace(".", "__") + "__q"
            safe_key_k = mod_name.replace(".", "__") + "__k"
            mode = entry.get("hookq_mode", self.hookq_mode)

            if mode == "all_tokens":
                flat_dict[safe_key_q] = pad_sequence(entry['q'], batch_first=True)
                flat_dict[safe_key_k] = pad_sequence(entry['k_all'], batch_first=True)
                if not seq_lens:
                    seq_lens = [t.shape[0] for t in entry['q']]
            else:
                # last_token: q is 1D (head_dim,) per request; k_all is 2D (seq_len, head_dim)
                # with variable seq_len across the batch — pad k, stack q.
                flat_dict[safe_key_q] = torch.stack(entry['q'])
                flat_dict[safe_key_k] = pad_sequence(entry['k_all'], batch_first=True)
                if not seq_lens:
                    seq_lens = [t.shape[0] for t in entry['k_all']]

            batch_size = flat_dict[safe_key_q].shape[0]
            layer_order.append({
                "key_q": safe_key_q,
                "key_k": safe_key_k,
                "module_name": mod_name,
                "layer_num": entry["layer_num"],
                "hookq_mode": mode,
            })

        _st_save(flat_dict, tmp_path)
        os.rename(tmp_path, out_path)

        meta = {
            "config": cpu_cache["config"],
            "layer_order": layer_order,
            "batch_size": batch_size,
            "hookq_mode": self.hookq_mode,
            "tp_rank": int(ps.get_tensor_model_parallel_rank()),
        }
        if seq_lens:
            meta["seq_lens"] = seq_lens
        meta_tmp = meta_path + ".tmp"
        with open(meta_tmp, "w") as f:
            _json.dump(meta, f)
        os.rename(meta_tmp, meta_path)

    def _background_save_loop(self):
        """Drain the save queue and write artifacts to disk.

        Activated when VLLM_HOOK_ASYNC_SAVE=1. Runs as a daemon thread in the
        worker subprocess. Each item is (run_id, cpu_cache, run_dir).
        """
        while True:
            run_id, cpu_cache, run_dir = self._save_queue.get()
            try:
                os.makedirs(run_dir, exist_ok=True)
                if os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1":
                    self._save_safetensors(cpu_cache, run_dir)
                else:
                    out_path = os.path.join(run_dir, "qk.pt")
                    tmp_path = out_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        torch.save(cpu_cache, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.rename(tmp_path, out_path)
            except Exception as e:
                print(f"background save failed for {run_id}: {e}")
            finally:
                self._save_queue.task_done()

