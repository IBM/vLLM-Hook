import os
import re
from typing import Dict, List, Optional, Tuple

import mlx.nn as nn
import torch
import math
from vllm.utils.torch_utils import set_random_seed
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.platform import MetalPlatform
from vllm_metal.v1.worker import MetalWorker
from vllm_metal.utils import set_wired_limit

# House analogy:
# - outer house: model.layers.<i>.self_attn
# - inner houses: wrappers installed at q_proj / k_proj / v_proj
# - inner-inner houses: original q_proj / k_proj / v_proj modules
# - recording room: qkv_hook
# Refined from sandbox inspection:
# - Attention exposes q_proj, k_proj, v_proj, o_proj, rope as named_modules()
# - q_proj returns full hidden width
# - k_proj / v_proj return KV width
PROJ_PATTERNS = [
    re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj)$"),
]


def match_qkv_proj(name: str) -> Optional[Tuple[int, str]]:
    # Example names we want to match:
    # - model.layers.22.self_attn.q_proj
    # - model.layers.22.self_attn.k_proj
    # - model.layers.22.self_attn.k_proj
    #
    # House analogy:
    # - outer house address: model.layers.22.self_attn
    # - inner house address after wrapping: model.layers.22.self_attn.q_proj
    # - inner-inner house after wrapping: the original q_proj stored inside the wrapper
    for pat in PROJ_PATTERNS:
        match = pat.match(name)
        if match:
            return int(match.group(1)), match.group(2)[0]
    return None


class MLXHookWrapper(nn.Module):
    # House analogy:
    # Relative to self_attn, this wrapper becomes the new inner house at the
    # q_proj / k_proj address.
    # The original projection module becomes the inner-inner house inside the
    # wrapper. Visitors enter the new inner house first, get recorded, and then
    # continue unchanged into the original inner-inner house.
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
        pid = os.getpid()
        rank = getattr(self, "rank", "?")
        local_rank = getattr(self, "local_rank", "?")
        print(
            f"[metal-worker pid={pid} rank={rank} local_rank={local_rank}] {message}",
            flush=True,
        )

    def __init__(self, *args, **kwargs):
        self._execute_logged = False
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
            print("Projection hooks installed successfully", flush=True)
        except Exception as exc:
            self._stage(f"install_hooks failed: {type(exc).__name__}: {exc}")
            print(f"Hook installation failed: {exc}", flush=True)

        self._stage("load_model complete")
        return result

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self.hookq_mode = os.environ.get("VLLM_HOOKQ_MODE", "all_tokens")
        tp_rank = int(self.rank)

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())
        self._run_cache = {}
        self._hooks = []
        self._original_modules = {}
        self._selected_layer = (
            min(self.important_layers) if self.important_layers else None
        )

        cfg = getattr(self.model_runner, "model_args", None) or {}
        num_h = int(cfg.get("num_attention_heads", getattr(model, "n_heads", 0)))
        num_kv = int(cfg.get("num_key_value_heads", num_h))
        hidden = int(cfg.get("hidden_size", getattr(model, "dim", 0)))
        if not all([num_h, num_kv, hidden]):
            print("Could not infer model dimensions for Metal hook installation")
            return
        head_dim = hidden // num_h
        q_width = hidden
        kv_width = head_dim * num_kv
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=1 / math.sqrt(head_dim),
            q_proj_output_width=q_width,
            kv_proj_output_width=kv_width,
        )

        # House analogy:
        # This is the recording room inside the wrapper inner house. It should
        # see visitors after they leave the original q_proj / k_proj
        # inner-inner house, not before they enter it.
        def qkv_hook(_input_args, output, module_name):
            try:
                # Recording room is closed unless the hook flag is present.
                if not os.path.exists(self.hook_flag):
                    return None
                if not os.path.exists(self.run_id_file):
                    raise RuntimeError("run_id not found")

                # Each run gets its own notebook so visitors from different runs are
                # not mixed together.
                run_id = open(self.run_id_file).read().strip().split("\n")[-1]

                proj_info = match_qkv_proj(module_name)
                if proj_info is None:
                    return None
                layer_num, proj_kind = proj_info

                # 'output' is the projected visitor that has already left q_proj
                # or k_proj. This is the person we want to write down.
                value = mlx_to_torch(output, device="cpu")
                print(
                    f"[qkv_hook] module={module_name} proj={proj_kind} "
                    f"torch_shape={tuple(value.shape)}"
                )
                if value.ndim < 2:
                    return None

                cache = self._run_cache.get(run_id)
                if cache is None:
                    cache = {"config": self._conf, "qk_cache": {}}
                    self._run_cache[run_id] = cache

                layer_name = module_name.rsplit(".", 1)[0]
                layer_cache = cache["qk_cache"].setdefault(
                    layer_name,
                    {"layer_num": layer_num, "q": [], "k_all": []},
                )

                # Projection outputs on Metal are available directly at the hook
                # boundary, so we can slice by shape without forward_context.
                if value.ndim == 3:
                    # Prefill shape: (batch, seq, dim)
                    batch_size = value.shape[0]
                    q_tokens_all = [value[i].clone() for i in range(batch_size)]
                    q_tokens_last = [value[i, -1, :].clone() for i in range(batch_size)]
                elif value.ndim == 2:
                    # Decode shape may already be (batch, dim).
                    q_tokens_all = [value[i].clone() for i in range(value.shape[0])]
                    q_tokens_last = q_tokens_all
                else:
                    raise RuntimeError(
                        f"Unexpected projection output rank {value.ndim} for {module_name}"
                    )

                if proj_kind == "q":
                    if self.hookq_mode == "all_tokens":
                        layer_cache["q"].extend(q_tokens_all)
                    elif self.hookq_mode == "last_token":
                        layer_cache["q"].extend(q_tokens_last)
                    else:
                        raise NotImplementedError(self.hookq_mode)
                elif proj_kind == "k":
                    layer_cache["k_all"].extend(q_tokens_all)

            except Exception as exc:
                shape = getattr(output, "shape", None)
                print(
                    f"[qkv_hook] failure module={module_name} "
                    f"output_type={type(output).__name__} "
                    f"output_shape={shape} error={type(exc).__name__}: {exc}"
                )
                raise

        # register hooks on projection modules
        matched = []
        discovered_proj_modules = []
        for name, module in model.named_modules():
            # model.named_modules() gives full street addresses for every house
            # and sub-house in the model tree. Here we keep only projection
            # addresses inside the self_attn outer house.
            proj_info = match_qkv_proj(name)
            if proj_info is None:
                continue

            layer_num, _proj_kind = proj_info
            discovered_proj_modules.append(name)
            if self._selected_layer is None:
                self._selected_layer = layer_num
            if layer_num != self._selected_layer:
                continue

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                # Walk the address one step at a time until we reach the parent
                # outer house that owns q_proj / k_proj.
                #
                # Example:
                # name = model.layers.22.self_attn.q_proj
                # parts[:-1] = ["model", "layers", "22", "self_attn"]
                #
                # After this loop:
                # parent == model.layers[22].self_attn
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)

            # target_name is the specific projection address hanging off the
            # self_attn outer house.
            target_name = parts[-1]
            wrapped_module = MLXHookWrapper(module=module, name=name, hook_fn=qkv_hook)
            self._original_modules[name] = module

            # This is the actual swap:
            # before: parent.q_proj == original inner house
            # after:  parent.q_proj == wrapper inner house
            #
            # The original projection module is still kept inside the wrapper as
            # the inner-inner house: wrapped_module.module.
            setattr(parent, target_name, wrapped_module)
            self._hooks.append(
                {
                    "name": name,
                    "parent": parent,
                    "target_name": target_name,
                    "original_module": module,
                }
            )
            matched.append(name)

        print(
            "Discovered projection houses inside self_attn: "
            f"{discovered_proj_modules}"
        )
        print(
            "Selected Metal hook layer: "
            f"{self._selected_layer} (q_proj and k_proj only)"
        )
        print(f"Installed {len(self._hooks)} projection hooks on layers: {matched}")

    def _flush_run_cache(self) -> None:
        if not self._run_cache:
            return

        tp_rank = int(self.rank)
        for run_id, cache in self._run_cache.items():
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qk.pt")
            torch.save(cache, cache_path)
            print(
                f"[metal-worker] flushed qk cache for run_id={run_id} "
                f"modules={list(cache['qk_cache'].keys())}",
                flush=True,
            )

    def _uninstall_hooks(self):
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
        if not self._execute_logged:
            self._stage("execute_model first entry")
            self._execute_logged = True
        result = super().execute_model(*args, **kwargs)
        try:
            self._flush_run_cache()
        except Exception as exc:
            self._stage(f"flush_run_cache failed: {type(exc).__name__}: {exc}")
            raise
        return result
