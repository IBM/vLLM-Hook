import os
import math
import queue
import re
import threading
import torch
from typing import Dict, List
import torch
if torch.cuda.is_available():
    from vllm.v1.worker.gpu_worker import Worker as V1Worker
else:
    from vllm.v1.worker.cpu_worker import CPUWorker as V1Worker
from vllm.forward_context import get_forward_context
from vllm.distributed import parallel_state as ps

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

class ProbeHookQKWorker(V1Worker):

    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)
        
        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")
            
        return r

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self.hookq_mode = os.environ.get("VLLM_HOOKQ_MODE", "all_tokens") # ["last_token", "all_tokens"]

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return            

        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())

        self._run_cache = {}

        # Background I/O thread (only when VLLM_HOOK_ASYNC_SAVE=1).
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

        # Per-pass state written by execute_model(), read by the hook.
        # Avoids all filesystem I/O inside the hook closure.
        self._hook_active = False
        self._current_run_id = None

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

        def qkv_hook(input, module_name):
            # Fast-path: checked once per execute_model() call, not per hook.
            if not self._hook_active:
                return None

            run_id = self._current_run_id

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
            if query_start_loc is None and isinstance(metadata, dict):
                for entry in metadata.values():
                    query_start_loc = getattr(entry, "query_start_loc", None)
                    if query_start_loc is not None:
                        break

            if query_start_loc is None:
                return

            bs = len(query_start_loc) - 1
            last_indices = query_start_loc

            cache = self._run_cache.get(run_id)
            if cache is None:
                cache = {"config": self._conf, "qk_cache": {}}
                self._run_cache[run_id] = cache
            if module_name not in cache["qk_cache"]:     # this means it is the first time the hook is called for this layer under the run ID
                q_tokens = []
                k_all_tokens = []
            else:
                q_tokens = cache["qk_cache"][module_name]['q']
                k_all_tokens = cache["qk_cache"][module_name]['k_all']

            layer_num = match_attn(module_name)
            # Accumulate GPU tensors — clone() copies the data immediately so
            # we own the buffer and don't need a synchronize() before post-pass.
            # No .cpu() here; transfer happens once in execute_model().
            if self.hookq_mode == "all_tokens":
                q_tokens.extend([
                    input[0][last_indices[i]:last_indices[i+1], :].detach().clone()
                    for i in range(bs)
                ])
            elif self.hookq_mode == "last_token":
                q_tokens.extend([
                    input[0][last_indices[i+1] - 1, :].detach().clone()
                    for i in range(bs)
                ])
            else:
                raise NotImplementedError
            k_all_tokens.extend([
                input[1][last_indices[i]:last_indices[i+1], :].detach().clone()
                for i in range(bs)
            ])

            cache["qk_cache"][module_name] = {
                'q': q_tokens,
                'k_all': k_all_tokens,
                'layer_num': layer_num
            }

        # register hooks on attention modules 
        self._hooks = []
        matched = []
        for name, module in model.named_modules():
            layer_num = match_attn(name)
            if layer_num is None: # not an attention module 
                continue
            if layer_num not in self.important_layers:
                continue
            hook = module.register_forward_hook(
                lambda _m, i, _o, n=name: qkv_hook(i, n)
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

            if self.hookq_mode == "all_tokens":
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

    def execute_model(self, *args, **kwargs):
        hooks_configured = all([
            getattr(self, "hook_dir", None),
            getattr(self, "hook_flag", None),
            getattr(self, "run_id_file", None),
        ])

        if hooks_configured and os.path.exists(self.hook_flag):
            if os.path.exists(self.run_id_file):
                run_id = open(self.run_id_file).read().strip().split("\n")[-1]
            else:
                run_id = None
            self._hook_active = run_id is not None
            self._current_run_id = run_id
        else:
            self._hook_active = False
            self._current_run_id = None

        result = super().execute_model(*args, **kwargs)

        if self._hook_active:
            # Read req_ids after the forward pass — input_batch is populated
            # by super().execute_model() from the scheduler_output.
            try:
                num_reqs = self.model_runner.input_batch.num_reqs
                _req_ids_raw = self.model_runner.input_batch.req_ids
                self._current_req_ids = list(_req_ids_raw[:num_reqs])
            except Exception:
                self._current_req_ids = None

        # --- Post-pass: GPU→CPU transfer and save (once per pass, not per layer) ---
        run_id = self._current_run_id
        if self._hook_active and run_id is not None:
            cache = self._run_cache.get(run_id)
            has_data = cache is not None and any(
                entry['q'] for entry in cache.get("qk_cache", {}).values()
            )
            req_ids = getattr(self, "_current_req_ids", None)
            call_bs = len(req_ids) if req_ids is not None else None

            if has_data and call_bs != 0:
                tp_rank = int(ps.get_tensor_model_parallel_rank())
                run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
                os.makedirs(run_dir, exist_ok=True)

                if call_bs is not None:
                    perm = sorted(range(call_bs), key=lambda idx: req_ids[idx])
                else:
                    perm = None

                def _to_cpu_sorted(tensors):
                    if call_bs is None or perm is None:
                        return [t.cpu() for t in tensors]
                    prefix = [t.cpu() for t in tensors[:-call_bs]]
                    tail = tensors[-call_bs:]
                    return prefix + [tail[i].cpu() for i in perm]

                cpu_cache = {
                    "config": cache["config"],
                    "qk_cache": {
                        mod: {
                            "q": _to_cpu_sorted(entry["q"]),
                            "k_all": _to_cpu_sorted(entry["k_all"]),
                            "layer_num": entry["layer_num"],
                        }
                        for mod, entry in cache["qk_cache"].items()
                    },
                }

                for stale_id in [k for k in self._run_cache if k != run_id]:
                    del self._run_cache[stale_id]

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

        return result
