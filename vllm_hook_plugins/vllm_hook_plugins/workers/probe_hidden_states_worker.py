import os
import queue
import re
import threading
import torch
from vllm.v1.worker.gpu_worker import Worker as V1Worker
from vllm.forward_context import get_forward_context
from vllm.distributed import parallel_state as ps

LAYER_PATTERNS = [
    # LLaMA / Qwen / Granite: model.layers.<i>
    re.compile(r"^model\.layers\.(\d+)$"),
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


class ProbeHiddenStatesWorker(V1Worker):

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
        self.hs_mode = os.environ.get("VLLM_HOOK_HS_MODE", "last_token")
        # tp_rank = int(ps.get_tensor_model_parallel_rank())

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

        self.target_layers = self._parse_target_layers()

        self._run_cache = {}

        # attach to pre-allocated SharedMemory block if enabled.
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
                # print(f"SHM attached: {shm_name} "
                #       f"({self._shm_num_layers} layers, hidden_size={self._shm_hidden_size})")
            except Exception as e:
                print(f"SHM attach failed: {e} — falling back to disk path")
                self._shm = None

        # Background I/O thread (only when VLLM_HOOK_ASYNC_SAVE=1, and SHM is not used).
        elif os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1":
            if not getattr(self, '_io_thread_started', False):
                self._save_queue: queue.Queue = queue.Queue(maxsize=4)
                self._io_thread = threading.Thread(
                    target=self._background_save_loop,
                    daemon=True,
                    name="vllm-hook-io",
                )
                self._io_thread.start()
                self._io_thread_started = True

        # Per-pass state written by execute_model(), read by the hook.
        # Avoids all filesystem I/O inside the hook closure.
        self._hook_active = False
        self._current_run_id = None

        cfg = model.config
        hidden_size = int(getattr(cfg, "hidden_size"))
        num_layers = int(getattr(cfg, "num_hidden_layers", 0))
        self._conf = {"hidden_size": hidden_size, "num_layers": num_layers}

        def hs_hook(output, module_name, layer_num):
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

            # The HS worker hooks on "model.layers.<i>",
            # so we look up the corresponding attention key.
            seq_lens = getattr(metadata, "seq_lens", None)
            if seq_lens is None and isinstance(metadata, dict):
                attn_key = f"{module_name}.self_attn.attn"
                entry = metadata.get(attn_key)
                if entry is not None:
                    seq_lens = entry.seq_lens

            if seq_lens is None:
                return

            last_indices = torch.cumsum(seq_lens, dim=0)
            bs = len(last_indices)
            last_indices = torch.cat(
                [torch.tensor([0]).to(last_indices.device), last_indices]
            )

            # vLLM uses a fused residual pattern: transformer blocks return
            # (hidden_states, residual) where the residual has not yet been added. 
            if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], torch.Tensor):
                hidden = output[0] + output[1]
            elif isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            cache = self._run_cache.get(run_id)
            if cache is None:
                cache = {"config": self._conf, "hs_cache": {}}
                self._run_cache[run_id] = cache

            if module_name not in cache["hs_cache"]:
                batch_hs = []
            else:
                batch_hs = cache["hs_cache"][module_name]["hidden_states"]

            # Accumulate GPU tensors — clone() copies the data immediately so
            # we own the buffer and don't need a synchronize() before post-pass.
            # No .cpu() here; transfer happens once in execute_model().
            if self.hs_mode == "last_token":
                batch_hs.extend(
                    [
                        hidden[last_indices[i + 1] - 1].detach().clone()
                        for i in range(bs)
                    ]
                )
            elif self.hs_mode == "all_tokens":
                batch_hs.extend(
                    [
                        hidden[last_indices[i] : last_indices[i + 1]].detach().clone()
                        for i in range(bs)
                    ]
                )
            else:
                raise NotImplementedError(f"Unknown hs_mode: {self.hs_mode}")

            cache["hs_cache"][module_name] = {
                "hidden_states": batch_hs,
                "layer_num": layer_num,
            }

        self._hooks = []
        matched = []
        for name, module in model.named_modules():
            layer_num = match_layer(name)
            if layer_num is None:
                continue
            if layer_num not in self.target_layers:
                continue
            hook = module.register_forward_hook(
                lambda m, i, o, n=name, ln=layer_num: hs_hook(o, n, ln)
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

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        """Optionally write cpu_cache as a safetensors file + JSON sidecar.

        last_token mode: tensors are (hidden_size,) per prompt — stacked to
          (bs, hidden_size).
        all_tokens mode: tensors are (seq_len, hidden_size) with variable
          seq_len — padded to (bs, max_seq_len, hidden_size) and seq_lens
          stored in the sidecar JSON so the reader can unpad.
        """
        import json as _json
        from torch.nn.utils.rnn import pad_sequence
        from safetensors.torch import save_file as _st_save

        out_path = os.path.join(run_dir, "hidden_states.safetensors")
        tmp_path = out_path + ".tmp"
        meta_path = os.path.join(run_dir, "hidden_states.json")

        flat_dict: dict = {}
        layer_order: list = []
        batch_size = 0
        seq_lens: list = []  # only populated for all_tokens mode

        for mod_name, entry in cpu_cache["hs_cache"].items():
            hs_list = entry["hidden_states"]
            safe_key = mod_name.replace(".", "__")

            if self.hs_mode == "all_tokens":
                # Variable seq_len per prompt — pad to (bs, max_seq_len, hidden_size)
                # pad_sequence expects (seq_len, hidden_size) inputs, pads along dim 0
                stacked = pad_sequence(hs_list, batch_first=True)  # (bs, max_seq, hidden)
                if not seq_lens:
                    seq_lens = [t.shape[0] for t in hs_list]
            else:
                # last_token: uniform (hidden_size,) per prompt
                stacked = torch.stack(hs_list)  # (bs, hidden_size)

            batch_size = stacked.shape[0]
            flat_dict[safe_key] = stacked
            layer_order.append({
                "key": safe_key,
                "module_name": mod_name,
                "layer_num": entry["layer_num"],
            })

        _st_save(flat_dict, tmp_path)
        os.rename(tmp_path, out_path)

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
                    out_path = os.path.join(run_dir, "hidden_states.pt")
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
        # --- Refresh per-pass state (filesystem I/O done once here, not in hook) ---
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

        if self._hook_active:
            peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()

        result = super().execute_model(*args, **kwargs)

        # --- Post-pass: GPU→CPU transfer and save (once per pass, not per layer) ---
        run_id = self._current_run_id
        if self._hook_active and run_id is not None:
            cache = self._run_cache.get(run_id)
            if cache is not None:
                tp_rank = int(ps.get_tensor_model_parallel_rank())
                run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
                os.makedirs(run_dir, exist_ok=True)

                # Move all accumulated GPU tensors to CPU in one pass
                cpu_cache = {
                    "config": cache["config"],
                    "hs_cache": {
                        mod: {
                            "hidden_states": [t.cpu() for t in entry["hidden_states"]],
                            "layer_num": entry["layer_num"],
                        }
                        for mod, entry in cache["hs_cache"].items()
                    },
                    "peak_gpu_mb": peak_gpu_mb,
                }
                # Evict stale run entries to prevent unbounded memory growth
                for stale_id in [k for k in self._run_cache if k != run_id]:
                    del self._run_cache[stale_id]

                if self._shm is not None:
                    # write tensors directly into shared memory — no disk I/O.
                    # Layout: [0:4] uint32 num_layers, [4:8] uint32 batch_size,
                    #          [8:] float16 data row-major (layer_slot, batch_item, hidden_dim).
                    # Sync via a tiny flag file on /dev/shm (ms-range latency).
                    # Only last_token mode is supported (all tensors are 1-D, same shape).
                    layer_order = self._shm_layer_order
                    hidden_size = self._shm_hidden_size
                    lnum_to_mod = {
                        entry["layer_num"]: mod
                        for mod, entry in cpu_cache["hs_cache"].items()
                    }
                    actual_layers = sum(1 for ln in layer_order if ln in lnum_to_mod)
                    first_entry = next(iter(cpu_cache["hs_cache"].values()))
                    actual_bs = len(first_entry["hidden_states"])

                    import struct as _struct
                    self._shm.buf[0:8] = _struct.pack("II", actual_layers, actual_bs)

                    data_offset = 8
                    slot = 0
                    for lnum in layer_order:
                        mod = lnum_to_mod.get(lnum)
                        if mod is None:
                            continue
                        tensors = cpu_cache["hs_cache"][mod]["hidden_states"]
                        stacked = torch.stack(tensors).to(torch.float16)  # (bs, hidden_size)
                        raw = stacked.numpy().tobytes()
                        start = data_offset + slot * actual_bs * hidden_size * 2
                        end = start + len(raw)
                        self._shm.buf[start:end] = raw
                        slot += 1

                    # Signal readiness via flag file (main process polls for it).
                    # Write peak_gpu_mb into the flag file so the analyzer can
                    # include it in the persisted artifact.
                    with open(self._shm_ready_flag, "w") as _f:
                        _f.write(str(peak_gpu_mb))

                elif os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1":
                    self._save_queue.put((run_id, cpu_cache, run_dir))
                elif os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1":
                    self._save_safetensors(cpu_cache, run_dir)
                else:
                    # Default synchronous path
                    out_path = os.path.join(run_dir, "hidden_states.pt")
                    tmp_path = out_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        torch.save(cpu_cache, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.rename(tmp_path, out_path)

        return result
