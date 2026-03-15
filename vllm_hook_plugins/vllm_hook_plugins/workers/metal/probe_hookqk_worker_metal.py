import math
import os
from typing import Dict, List

import mlx.nn as nn
import torch
from vllm.utils.torch_utils import set_random_seed
from vllm_metal.platform import MetalPlatform
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.utils import set_wired_limit
from vllm_metal.v1.worker import MetalWorker


class MLXHookWrapper(nn.Module):
    # Housing analogy:
    # - the wrapped address is a house listed in the building directory
    # - once we replace that address, this wrapper becomes the new outer house
    # - the original module still lives inside as the inner house at
    #   `self.module`
    # - visitors reach the outer house first, step into the recording room
    #   via `hook_fn`, and then continue into the original inner house
    def __init__(self, module, name, hook_fn):
        super().__init__()
        self.module = module
        self.name = name
        self.hook_fn = hook_fn

    def __call__(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        self.hook_fn(args, output, self.name)
        return output


class ProbeHookQKWorkerMetal(MetalWorker):
    def _stage(self, message: str) -> None:
        if not getattr(self, "_debug_hook", False):
            return
        pid = os.getpid()
        rank = getattr(self, "rank", "?")
        local_rank = getattr(self, "local_rank", "?")
        print(
            f"[metal-worker pid={pid} rank={rank} local_rank={local_rank}] {message}",
            flush=True,
        )

    def __init__(self, *args, **kwargs):
        self._execute_logged = False
        self._capture_active = False
        self._debug_hook = os.environ.get("VLLM_HOOK_DEBUG", "") == "1"
        super().__init__(*args, **kwargs)
        self._stage("worker __init__ complete")

    def init_device(self) -> None:
        self._stage(
            "init_device start "
            f"distributed_init_method={self.distributed_init_method}"
        )
        try:
            world_size = self.parallel_config.world_size
            if world_size == 1:
                self._stage("init_device using single-process fast path")
                self._init_device_single_process()
            else:
                super().init_device()
        except Exception as exc:
            self._stage(f"init_device failed: {type(exc).__name__}: {exc}")
            raise
        self._stage("init_device complete")

    def _init_device_single_process(self) -> None:
        if self.metal_config.use_mlx:
            import mlx.core as mx

            device_type = (
                mx.DeviceType.gpu
                if self.metal_config.mlx_device == "gpu"
                else mx.DeviceType.cpu
            )
            mx.set_default_device(mx.Device(device_type))
            self._stage(f"MLX device set to: {mx.default_device()}")
            set_wired_limit()

        self.device = MetalPlatform.get_torch_device(0)
        self._stage(f"PyTorch device set to: {self.device}")

        set_random_seed(self.model_config.seed)

        from vllm_metal.v1.model_runner import MetalModelRunner

        self.model_runner = MetalModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def load_model(self, *args, **kwargs):
        self._stage("load_model start")
        try:
            result = super().load_model(*args, **kwargs)
        except Exception as exc:
            self._stage(f"load_model failed: {type(exc).__name__}: {exc}")
            raise

        try:
            self._stage("install_hooks start")
            self._install_hooks()
            self._stage("install_hooks complete")
            print("Hooks installed successfully", flush=True)
        except Exception as exc:
            self._stage(f"install_hooks failed: {type(exc).__name__}: {exc}")
            print(f"Hook installation failed: {exc}", flush=True)

        self._stage("load_model complete")
        return result

    def _current_run_id(self) -> str | None:
        # Housing analogy:
        # - the recording room only writes when the building manager has posted
        #   an "open notebook" sign for the current visit
        # - `_capture_active` means the worker is in the real tenant visit, not
        #   warmup or teardown traffic
        # - `hook_flag` and `run_id_file` together tell the room which notebook
        #   to write into for this run
        if not self._capture_active or not os.path.exists(self.hook_flag):
            return None
        if not os.path.exists(self.run_id_file):
            raise RuntimeError("run_id not found")
        return open(self.run_id_file).read().strip().split("\n")[-1]

    def _ensure_run_cache(self, run_id: str):
        # Housing analogy:
        # - each run gets its own notebook
        # - the notebook keeps both the recorded visitor packets and the
        #   building facts the scoring desk needs later
        cache = self._run_cache.get(run_id)
        if cache is None:
            cache = {
                "config": self._conf,
                "qkv_cache": {},
                "meta": {
                    "tp_rank": int(self.rank),
                    "capture_boundary": self._capture_boundary,
                },
            }
            self._run_cache[run_id] = cache
        return cache

    def _append_proj_tokens(
        self,
        run_id: str,
        layer_num: int,
        proj_kind: str,
        tokens: torch.Tensor,
    ) -> None:
        # Housing analogy:
        # - each layer's attention house gets a notebook in the recording room
        # - each projection kind (`x`, `q`, `k`, `v`) gets its own section
        # - each sample in the batch is stored as a separate packet so the
        #   analyzer can later inspect each tenant visit independently
        cache = self._ensure_run_cache(run_id)
        module_name = f"model.layers.{layer_num}.self_attn.attn.{proj_kind}"
        layer_cache = cache["qkv_cache"].setdefault(
            module_name,
            {
                "layer_num": layer_num,
                "proj_kind": proj_kind,
                "tokens": [],
            },
        )
        layer_cache["tokens"].extend([tokens[i].clone() for i in range(tokens.shape[0])])

    def _record_qkv(
        self,
        run_id: str,
        layer_num: int,
        raw_x,
        queries,
        keys,
        values,
    ) -> None:
        # Housing analogy:
        # - `raw_x` is the visitor as it first enters the self-attention house
        # - `queries`, `keys`, and `values` are the specialized copies of that
        #   visitor after the house has routed it through q_proj/k_proj/v_proj
        # - this method files both the original visitor packet and the
        #   specialized packets into the notebook under the same floor tab
        x_torch = mlx_to_torch(raw_x, device="cpu")
        q_flat = queries.transpose(0, 2, 1, 3).reshape(queries.shape[0], queries.shape[2], -1)
        k_flat = keys.transpose(0, 2, 1, 3).reshape(keys.shape[0], keys.shape[2], -1)
        v_flat = values.transpose(0, 2, 1, 3).reshape(values.shape[0], values.shape[2], -1)
        q_torch = mlx_to_torch(q_flat, device="cpu")
        k_torch = mlx_to_torch(k_flat, device="cpu")
        v_torch = mlx_to_torch(v_flat, device="cpu")

        self._append_proj_tokens(run_id, layer_num, "x", x_torch)
        if self.hookq_mode == "all_tokens":
            self._append_proj_tokens(run_id, layer_num, "q", q_torch)
        elif self.hookq_mode == "last_token":
            self._append_proj_tokens(run_id, layer_num, "q", q_torch[:, -1:, :])
        else:
            raise NotImplementedError(self.hookq_mode)
        self._append_proj_tokens(run_id, layer_num, "k", k_torch)
        self._append_proj_tokens(run_id, layer_num, "v", v_torch)

        if self._debug_hook:
            print(
                f"[qkv_boundary] module=model.layers.{layer_num}.self_attn.attn "
                f"x_shape={tuple(x_torch.shape)} q_shape={tuple(q_torch.shape)} "
                f"k_shape={tuple(k_torch.shape)} v_shape={tuple(v_torch.shape)}",
                flush=True,
            )

    def _capture_from_self_attn(self, run_id: str, layer_num: int, attn_module, input_args) -> None:
        # Housing analogy:
        # - this is the recording room attached to the `self_attn` house door
        # - visitors arrive as raw hidden states `x`
        # - we let the recording room compute the same q/k/v routes the house
        #   itself would compute internally
        # - that keeps the notebook aligned with the real house behavior while
        #   avoiding any need for a deeper inner door that does not exist in
        #   the installed Metal runtime
        raw_x = input_args[0]
        cache = input_args[2] if len(input_args) > 2 else None
        batch, seq_len, _ = raw_x.shape

        queries = attn_module.q_proj(raw_x).reshape(batch, seq_len, attn_module.n_heads, -1)
        keys = attn_module.k_proj(raw_x).reshape(batch, seq_len, attn_module.n_kv_heads, -1)
        values = attn_module.v_proj(raw_x).reshape(batch, seq_len, attn_module.n_kv_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = attn_module.rope(queries, offset=cache.offset)
            keys = attn_module.rope(keys, offset=cache.offset)
        else:
            queries = attn_module.rope(queries)
            keys = attn_module.rope(keys)

        self._record_qkv(run_id, layer_num, raw_x, queries, keys, values)

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self.hookq_mode = os.environ.get("VLLM_HOOKQ_MODE", "all_tokens")

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

        # Housing analogy:
        # - `layer_to_heads` tells us which apartment floors matter
        # - we only install wrapper houses on those self-attention addresses
        # - later, the analyzer will only score the tracked heads on those
        #   same floors
        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())
        self._run_cache = {}
        self._hooks = []
        self._matched_hook_modules = []
        self._capture_boundary = "self_attn"

        cfg = getattr(self.model_runner, "model_args", None) or {}
        layers_obj = getattr(getattr(model, "model", model), "layers", None)
        if layers_obj is None:
            layers_obj = getattr(model, "layers", None)
        self._num_hidden_layers = int(
            cfg.get("num_hidden_layers", len(layers_obj) if layers_obj is not None else 0)
        )
        num_h = int(cfg.get("num_attention_heads", getattr(model, "n_heads", 0)))
        num_kv = int(cfg.get("num_key_value_heads", num_h))
        hidden = int(cfg.get("hidden_size", getattr(model, "dim", 0)))
        if not all([num_h, num_kv, hidden]):
            print("Could not infer model dimensions for Metal hook installation")
            return
        head_dim = hidden // num_h
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=float(
                cfg.get("attention_multiplier", 1 / math.sqrt(head_dim))
            ),
            q_proj_output_width=hidden,
            kv_proj_output_width=head_dim * num_kv,
        )

        # Housing analogy:
        # - `named_modules()` is the building directory
        # - each entry maps an address to the house currently living there
        # - we search that directory for `*.self_attn` houses and replace each
        #   tracked address with a wrapper outer house
        # - each wrapper keeps the original attention house inside so teardown
        #   can later restore the exact original resident to that address
        named_modules = dict(model.named_modules())
        for name, module in named_modules.items():
            if not name.endswith(".self_attn"):
                continue

            parts = name.split(".")
            if "layers" not in parts:
                continue
            layers_pos = parts.index("layers")
            if layers_pos + 1 >= len(parts) or not parts[layers_pos + 1].isdigit():
                continue

            layer_idx = int(parts[layers_pos + 1])
            if layer_idx not in self.important_layers:
                continue
            if not all(
                hasattr(module, attr)
                for attr in ("q_proj", "k_proj", "v_proj", "rope", "n_heads", "n_kv_heads")
            ):
                continue

            parent_name, target_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name)
            if parent is None:
                continue

            original_attn = module

            def attention_hook(input_args, _output, _module_name, layer_num=layer_idx, attn=original_attn):
                # Housing analogy:
                # - the wrapper's recording room checks whether there is an open
                #   notebook for this visit
                # - if so, it records the raw visitor and the projected Q/K/V
                #   copies for this floor
                run_id = self._current_run_id()
                if run_id is None:
                    return None
                self._capture_from_self_attn(run_id, layer_num, attn, input_args)
                return None

            wrapped_attn = MLXHookWrapper(
                module=original_attn,
                name=name,
                hook_fn=attention_hook,
            )
            setattr(parent, target_name, wrapped_attn)
            self._hooks.append(
                {
                    "parent": parent,
                    "target_name": target_name,
                    "original_module": original_attn,
                }
            )
            self._matched_hook_modules.append(name)

        if not self._matched_hook_modules:
            print("Could not locate self_attn modules for Metal attention hook")
            return

        print(
            f"Installed {len(self._matched_hook_modules)} hooks on layers: "
            f"{self._matched_hook_modules}",
            flush=True,
        )
        if self._debug_hook:
            print(
                "Selected Metal hook layers: "
                f"{sorted(self.important_layers)} "
                f"({self._capture_boundary} boundary)",
                flush=True,
            )
            print("Installed Granite attention hook", flush=True)

    def _flush_run_cache(self) -> None:
        # Housing analogy:
        # - when the visit ends, the worker carries each in-memory notebook to
        #   the archive room on disk and files it as `qkv.pt`
        if not self._run_cache:
            return

        tp_rank = int(self.rank)
        for run_id, cache in self._run_cache.items():
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qkv.pt")
            torch.save(cache, cache_path)
            if self._debug_hook:
                sample_counts = {
                    module_name: len(module_cache.get("tokens", []))
                    for module_name, module_cache in cache["qkv_cache"].items()
                }
                print(
                    f"[metal-worker] flushed qkv cache for run_id={run_id} "
                    f"modules={list(cache['qkv_cache'].keys())} "
                    f"sample_counts={sample_counts}",
                    flush=True,
                )

    def _uninstall_hooks(self):
        # Housing analogy:
        # - teardown removes the temporary outer houses from each tracked
        #   address
        # - the original inner houses are moved back to the public address so
        #   the next temporary engine starts from a clean building directory
        for entry in reversed(self._hooks):
            setattr(entry["parent"], entry["target_name"], entry["original_module"])
        self._hooks.clear()

    def _parse_layer_heads(self) -> Dict[int, List[int]]:
        layer_heads = os.environ.get("VLLM_HOOK_LAYER_HEADS", "")
        result = {}
        for part in layer_heads.split(";"):
            part = part.strip()
            if not part:
                continue
            layer_str, heads_str = part.split(":")
            layer_idx = int(layer_str)
            head_indices = sorted(int(h) for h in heads_str.split(",") if h)
            result[layer_idx] = head_indices
        return result

    def execute_model(self, *args, **kwargs):
        # Housing analogy:
        # - `execute_model` is when the building opens for a real tenant visit
        # - capture is only active during this window
        # - once the visit ends, any filled notebooks are sent to the archive
        #   room before returning control
        if not self._execute_logged:
            self._stage("execute_model first entry")
            self._execute_logged = True
        self._capture_active = True
        try:
            result = super().execute_model(*args, **kwargs)
        finally:
            self._capture_active = False
        try:
            self._flush_run_cache()
        except Exception as exc:
            self._stage(f"flush_run_cache failed: {type(exc).__name__}: {exc}")
            raise
        return result
