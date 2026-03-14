import os
import re
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
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
        self._sdpa_call_index = 0

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
        # The original PyTorch worker recorded the documents handed into the
        # inner control room `self_attn.attn`. Granite on MLX does not expose
        # that room as a child module, so we stand at the actual doorway into
        # the attention meeting: the `scaled_dot_product_attention(...)` call.
        def cache_qk(run_id: str, layer_num: int, queries, keys) -> None:
            module_name = f"model.layers.{layer_num}.self_attn.attn"
            try:
                cache = self._run_cache.get(run_id)
                if cache is None:
                    cache = {"config": self._conf, "qk_cache": {}}
                    self._run_cache[run_id] = cache

                layer_cache = cache["qk_cache"].setdefault(
                    module_name,
                    {"layer_num": layer_num, "q": [], "k_all": [], "max_seq_len": -1},
                )

                # House analogy:
                # `queries` and `keys` here are the attention-ready documents
                # already prepared for the meeting room. We flatten them back to
                # the same notebook format the analyzer expects from the
                # original worker.
                q_flat = queries.transpose(0, 2, 1, 3).reshape(
                    queries.shape[0], queries.shape[2], -1
                )
                k_flat = keys.transpose(0, 2, 1, 3).reshape(
                    keys.shape[0], keys.shape[2], -1
                )

                q_torch = mlx_to_torch(q_flat, device="cpu")
                k_torch = mlx_to_torch(k_flat, device="cpu")
                current_seq_len = int(q_torch.shape[1])
                print(
                    f"[qk_boundary] module={module_name} "
                    f"q_shape={tuple(q_torch.shape)} k_shape={tuple(k_torch.shape)} "
                    f"seq_len={current_seq_len}",
                    flush=True,
                )

                # House analogy:
                # The same room may host a few small side meetings as well as the
                # main full-prompt meeting. For the analyzer we want the complete
                # packet, so we keep only the longest sequence seen for a run/layer.
                if current_seq_len < layer_cache["max_seq_len"]:
                    print(
                        f"[qk_boundary] skipping shorter capture for {module_name} "
                        f"seq_len={current_seq_len} < max_seq_len={layer_cache['max_seq_len']}",
                        flush=True,
                    )
                    return
                if current_seq_len > layer_cache["max_seq_len"]:
                    layer_cache["q"].clear()
                    layer_cache["k_all"].clear()
                    layer_cache["max_seq_len"] = current_seq_len

                if self.hookq_mode == "all_tokens":
                    layer_cache["q"].extend(
                        [q_torch[i].clone() for i in range(q_torch.shape[0])]
                    )
                elif self.hookq_mode == "last_token":
                    layer_cache["q"].extend(
                        [q_torch[i, -1, :].clone() for i in range(q_torch.shape[0])]
                    )
                else:
                    raise NotImplementedError(self.hookq_mode)

                layer_cache["k_all"].extend(
                    [k_torch[i].clone() for i in range(k_torch.shape[0])]
                )

            except Exception as exc:
                print(
                    f"[qk_boundary] failure module={module_name} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise

        import mlx_lm.models.granite as granite_mod

        self._original_sdpa = granite_mod.scaled_dot_product_attention

        def probing_sdpa(queries, keys, values, cache=None, scale=None, mask=None):
            # House analogy:
            # Each call to `scaled_dot_product_attention(...)` is one layer's
            # control-room meeting. We count meetings as they happen and record
            # only the selected layer's documents.
            current_layer = self._sdpa_call_index
            self._sdpa_call_index += 1

            if self._selected_layer is None:
                self._selected_layer = current_layer

            if (
                current_layer == self._selected_layer
                and os.path.exists(self.hook_flag)
                and os.path.exists(self.run_id_file)
            ):
                run_id = open(self.run_id_file).read().strip().split("\n")[-1]
                cache_qk(run_id, current_layer, queries, keys)

            return self._original_sdpa(
                queries, keys, values, cache=cache, scale=scale, mask=mask
            )

        granite_mod.scaled_dot_product_attention = probing_sdpa
        print(
            "Selected Metal hook layer: "
            f"{self._selected_layer} (scaled_dot_product_attention boundary)"
        )
        print("Installed Granite attention boundary hook", flush=True)

    def _flush_run_cache(self) -> None:
        if not self._run_cache:
            return

        tp_rank = int(self.rank)
        for run_id, cache in self._run_cache.items():
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qk.pt")
            torch.save(cache, cache_path)
            sample_counts = {
                module_name: {
                    "q": len(module_cache.get("q", [])),
                    "k_all": len(module_cache.get("k_all", [])),
                    "max_seq_len": module_cache.get("max_seq_len"),
                }
                for module_name, module_cache in cache["qk_cache"].items()
            }
            print(
                f"[metal-worker] flushed qk cache for run_id={run_id} "
                f"modules={list(cache['qk_cache'].keys())} "
                f"sample_counts={sample_counts}",
                flush=True,
            )

    def _uninstall_hooks(self):
        if hasattr(self, "_original_sdpa"):
            import mlx_lm.models.granite as granite_mod

            granite_mod.scaled_dot_product_attention = self._original_sdpa
            del self._original_sdpa
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
        self._sdpa_call_index = 0
        result = super().execute_model(*args, **kwargs)
        try:
            self._flush_run_cache()
        except Exception as exc:
            self._stage(f"flush_run_cache failed: {type(exc).__name__}: {exc}")
            raise
        return result
