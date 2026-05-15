import os
import pickle
import re
from typing import TYPE_CHECKING, Any

import torch
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
    save_pt_atomic,
    save_safetensors_atomic,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)

LAYER_PATTERNS = [
    # LLaMA / Qwen2.x / Granite: model.layers.<i>
    re.compile(r"^model\.layers\.(\d+)$"),
    # Qwen3.5 multimodal (Qwen3_5ForConditionalGeneration): language_model.model.layers.<i>
    re.compile(r"^language_model\.model\.layers\.(\d+)$"),
    # GPT-2: transformer.h.<i>
    re.compile(r"^transformer\.h\.(\d+)$"),
    # OPT: model.decoder.layers.<i>
    re.compile(r"^model\.decoder\.layers\.(\d+)$"),
]


def match_layer(name: str):
    for pat in LAYER_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


class ProbeHiddenStatesWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (ProbeHiddenStatesWorker,) at runtime,
    so self is the Worker instance. Methods are callable via collective_rpc.
    """

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    # Default capture phase — matches the old hooks_on=(True, False) registry entry.
    # Can be overridden per-request via extra_args["hooks_on"].
    _default_hooks_on: str = "prefill"

    # Per-request captured hidden states (API serving path):
    # internal_req_id -> {module_name -> {"hidden_states": [...], "layer_num": int}}
    _captured_states: dict = {}
    _hooks_installed: bool = False

    def install_hooks(self):
        """Install forward hooks on all target decoder layers. Idempotent.

        Callable via collective_rpc("install_hooks") — the plugin calls this
        lazily on the first request that sets output_hidden_states in extra_args.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}  # RPC path: req_id -> {module: {hidden_states, layer_num}}
        self._disk_states = {}      # disk path: req_id -> {module: {hidden_states, layer_num}} + {"_meta": {run_id, hook_dir}}
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hs_mode = os.environ.get("VLLM_HOOK_HS_MODE", "last_token")
        self.target_layers = self._parse_target_layers()

        # SHM path is a specialized same-machine transport, independent of the
        # per-request RPC/disk paths. Kept as-is for backward compat.
        self._shm = None
        if os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            try:
                from multiprocessing.shared_memory import SharedMemory
                shm_name = os.environ["VLLM_HOOK_SHM_NAME"]
                self._shm = SharedMemory(create=False, name=shm_name)
                self._shm_hidden_size = int(os.environ["VLLM_HOOK_SHM_HIDDEN_SIZE"])
                self._shm_num_layers = int(os.environ["VLLM_HOOK_SHM_NUM_LAYERS"])
                self._shm_max_batch = int(os.environ["VLLM_HOOK_SHM_MAX_BATCH"])
                self._shm_ready_flag = os.environ["VLLM_HOOK_SHM_READY_FLAG"]
                layer_order_str = os.environ.get("VLLM_HOOK_SHM_LAYER_ORDER", "")
                self._shm_layer_order = [int(x) for x in layer_order_str.split(";") if x]
            except Exception as e:
                print(f"SHM attach failed: {e} — falling back to disk path")
                self._shm = None

        # Background I/O thread for async safetensors/.pt writes. Shared by
        # all disk-save requests on this worker.
        init_async_save_thread(self, self._background_save_loop, "vllm-hook-hs-io")

        cfg = model.config
        # Multimodal models (e.g. Qwen3.5) nest text config under text_config.
        text_cfg = getattr(cfg, "text_config", cfg)
        hidden_size = int(getattr(text_cfg, "hidden_size"))
        num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
        self._conf = {"hidden_size": hidden_size, "num_layers": num_layers}

        # Only TP rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        def hs_hook(output, module_name, layer_num):
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

            # vLLM uses a fused residual pattern: transformer blocks return
            # (hidden_states, residual) where the residual has not yet been added. 
            if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], torch.Tensor):
                hidden = output[0] + output[1]
            elif isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

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
                if not extra or extra.get("output_hidden_states") is None:
                    continue

                output_layers = extra.get("output_hidden_states")
                if isinstance(output_layers, list) and (layer_num not in output_layers):
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

                # Per-request mode: extra_args["hs_mode"] overrides the worker default.
                req_mode = extra.get("hs_mode", self.hs_mode)

                start = int(last_indices[i].item())
                end = int(last_indices[i + 1].item())

                # Accumulate GPU tensors — clone() copies data immediately so we
                # own the buffer; .cpu() is deferred to the retrieval/flush call.
                if req_mode == "last_token":
                    activation = hidden[end - 1].detach().clone()
                else:
                    activation = hidden[start:end].detach().clone()

                # Route to disk or RPC bucket based on save_to_disk flag.
                bucket = self._disk_states if extra.get("save_to_disk") else self._captured_states
                if req_id not in bucket:
                    bucket[req_id] = {}
                layer_states = bucket[req_id]
                if module_name not in layer_states:
                    layer_states[module_name] = {"hidden_states": [], "layer_num": layer_num, "hs_mode": req_mode}
                layer_states[module_name]["hidden_states"].append(activation)

        # When target_layers is empty (API path, no VLLM_HOOK_LAYERS env var),
        # hook every decoder layer. Per-request filtering via extra_args['output_hidden_states']
        # happens inside the hook closure.
        # target_layers uses the HuggingFace/Eagle 1-based convention (layer N = output
        # after the Nth transformer block), but iter_matched_modules / match_layer return
        # 0-based PyTorch indices (model.layers.N-1). Subtract 1 to align them.
        layer_filter_0based = {l - 1 for l in self.target_layers} if self.target_layers else None
        self._hooks = []
        matched = []
        for name, module, layer_num in iter_matched_modules(model, match_layer, layer_filter_0based):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name, ln=layer_num+1: hs_hook(o, n, ln)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} hidden-state hooks on layers: {matched}")

    def _parse_target_layers(self):
        raw = os.environ.get("VLLM_HOOK_LAYERS", "")
        result = set()
        for part in raw.split(";"):
            part = part.strip()
            if part:
                result.add(int(part))
        return result

    # ------------------------------------------------------------------
    # API serving: collective_rpc-callable artifact retrieval
    # ------------------------------------------------------------------

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve and remove captured hidden states for a completed request.

        Matches either by exact equality (vLLM v0.12+ uses the same id internally)
        or by "{external_req_id}-" prefix (older versions append a random suffix).

        CPU transfer happens here (once per request, not per hook).
        Returns zstd-compressed pickle, or None if nothing was captured.
        """
        for req_id in iter_matching_req_ids(self._captured_states, external_req_id):
            layer_dict = self._captured_states.pop(req_id)
            cpu_dict = {}
            for mod_name, entry in layer_dict.items():
                tensors = entry["hidden_states"]
                mode = entry.get("hs_mode", self.hs_mode)
                if mode == "last_token":
                    stacked = torch.stack([t.cpu() for t in tensors])
                else:
                    from torch.nn.utils.rnn import pad_sequence
                    stacked = pad_sequence([t.cpu() for t in tensors], batch_first=True)
                cpu_dict[mod_name] = {"hidden_states": stacked, "layer_num": entry["layer_num"], "hs_mode": mode}
            payload = {"hs_cache": cpu_dict, "config": self._conf}
            return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured states without returning them (cleanup on abort/disconnect)."""
        clear_states_for_req(self._captured_states, external_req_id)

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """Write captured hidden states for all requests in the batch to one artifact.

        Accepts a list of external_req_ids so all requests sharing a run_id
        are merged into one cpu_cache before writing — matching the old
        execute_model() behavior where the full batch was saved atomically.

        Returns True if any artifacts were written, False if nothing captured.
        """
        cpu_cache: dict = {"config": self._conf, "hs_cache": {}}
        found_any = False

        for external_req_id in external_req_ids:
            for req_id in iter_matching_req_ids(self._disk_states, external_req_id):
                layer_dict = self._disk_states.pop(req_id)
                if not layer_dict:
                    continue
                found_any = True
                for mod_name, entry in layer_dict.items():
                    cpu_entry = {
                        "hidden_states": [t.cpu() for t in entry["hidden_states"]],
                        "layer_num": entry["layer_num"],
                        "hs_mode": entry.get("hs_mode", self.hs_mode),
                    }
                    if mod_name in cpu_cache["hs_cache"]:
                        cpu_cache["hs_cache"][mod_name]["hidden_states"].extend(
                            cpu_entry["hidden_states"]
                        )
                    else:
                        cpu_cache["hs_cache"][mod_name] = cpu_entry

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
            save_pt_atomic(cpu_cache, os.path.join(run_dir, "hidden_states.pt"))
        return found_any

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        """Write cpu_cache as a safetensors file + JSON sidecar.

        last_token mode: tensors are (hidden_size,) per prompt — stacked to
          (bs, hidden_size).
        all_tokens mode: tensors are (seq_len, hidden_size) with variable
          seq_len — padded to (bs, max_seq_len, hidden_size) and seq_lens
          stored in the sidecar JSON so the reader can unpad.
        """
        from torch.nn.utils.rnn import pad_sequence

        flat_dict: dict = {}
        layer_order: list = []
        batch_size = 0
        seq_lens: list = []  # only populated for all_tokens mode

        for mod_name, entry in cpu_cache["hs_cache"].items():
            hs_list = entry["hidden_states"]
            safe_key = mod_name.replace(".", "__")
            mode = entry.get("hs_mode", self.hs_mode)

            if mode == "all_tokens":
                stacked = pad_sequence(hs_list, batch_first=True)  # (bs, max_seq, hidden)
                if not seq_lens:
                    seq_lens = [t.shape[0] for t in hs_list]
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
            "hs_mode": self.hs_mode,
            "peak_gpu_mb": cpu_cache.get("peak_gpu_mb", 0.0),
            "tp_rank": int(ps.get_tensor_model_parallel_rank()),
        }
        if seq_lens:
            meta["seq_lens"] = seq_lens
        save_safetensors_atomic(flat_dict, meta, run_dir, "hidden_states")

    def _background_save_loop(self):
        background_save_loop(self, "hidden_states.pt")

