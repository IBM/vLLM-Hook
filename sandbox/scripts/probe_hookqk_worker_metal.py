import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from vllm.distributed import parallel_state as ps
from vllm.forward_context import get_forward_context
from vllm.v1.worker.gpu_worker import Worker as V1Worker

import mlx.nn as nn

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
    # - model.layers.22.self_attn.v_proj
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


def to_torch_cpu(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    return torch.from_numpy(np.asarray(value))


class MLXHookWrapper(nn.Module):
    # House analogy:
    # Relative to self_attn, this wrapper becomes the new inner house at the
    # q_proj / k_proj / v_proj address.
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


class ProbeHookQKWorker(V1Worker):

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)

        try:
            self._install_hooks()
            print("Projection hooks installed successfully")
        except Exception as exc:
            print(f"Hook installation failed: {exc}")

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
        tp_rank = int(ps.get_tensor_model_parallel_rank())

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())
        self._run_cache = {}
        self._hooks = []
        self._original_modules = {}

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
            q_proj_output_width=q_width,
            kv_proj_output_width=kv_width,
        )

        # House analogy:
        # This is the recording room inside the wrapper inner house. It should
        # see visitors after they leave the original q_proj / k_proj / v_proj
        # inner-inner house, not before they enter it.
        def qkv_hook(input_args, output, module_name):
            # Recording room is closed unless the hook flag is present.
            if not os.path.exists(self.hook_flag):
                return None
            if not os.path.exists(self.run_id_file):
                raise RuntimeError("run_id not found")

            # Each run gets its own notebook so visitors from different runs are
            # not mixed together.
            run_id = open(self.run_id_file).read().strip().split("\n")[-1]
            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)
            if metadata is None:
                return None

            # seq_lens tells the recording room where each request begins and
            # ends inside the combined stream of visitors.
            seq_lens = getattr(metadata, "seq_lens", None)
            if seq_lens is None:
                return None

            proj_info = match_qkv_proj(module_name)
            if proj_info is None:
                return None
            layer_num, proj_kind = proj_info

            # 'output' is the projected visitor that has already left q_proj,
            # k_proj, or v_proj. This is the person we want to write down.
            value = to_torch_cpu(output)
            if value.ndim < 2:
                return None
            seq_lens = to_torch_cpu(seq_lens)
            last_indices = torch.cumsum(seq_lens, dim=0)
            batch_size = len(last_indices)
            last_indices = torch.cat(
                [torch.tensor([0], dtype=last_indices.dtype), last_indices]
            )

            cache = self._run_cache.get(run_id)
            if cache is None:
                cache = {"config": self._conf, "qkv_cache": {}}
                self._run_cache[run_id] = cache

            # Keep a separate notebook per projection house:
            # q_proj visitors go in the q notebook, k_proj in the k notebook,
            # and v_proj in the v notebook.
            layer_cache = cache["qkv_cache"].setdefault(
                module_name,
                {
                    "layer_num": layer_num,
                    "proj_kind": proj_kind,
                    "shape": tuple(value.shape),
                    "tokens": [],
                },
            )

            if self.hookq_mode == "all_tokens":
                layer_cache["tokens"].extend(
                    [
                        value[last_indices[i] : last_indices[i + 1], :].clone()
                        for i in range(batch_size)
                    ]
                )
            elif self.hookq_mode == "last_token":
                layer_cache["tokens"].extend(
                    list(value[last_indices[1:] - 1, :].clone())
                )
            else:
                raise NotImplementedError(self.hookq_mode)

            # Save the notes to disk. The recorded visitor is a copy; the real
            # visitor already continues through attention unchanged.
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qkv.pt")
            torch.save(cache, cache_path)

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

            layer_num, proj_kind = proj_info
            discovered_proj_modules.append(name)
            if self.important_layers and layer_num not in self.important_layers:
                continue

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                # Walk the address one step at a time until we reach the parent
                # outer house that owns q_proj / k_proj / v_proj.
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
            # For example:
            # - q_proj
            # - k_proj
            # - v_proj
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
        print(f"Installed {len(self._hooks)} projection hooks on layers: {matched}")

    def _uninstall_hooks(self):
        # House analogy:
        # Put the original inner-inner houses back at their original addresses.
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
        return super().execute_model(*args, **kwargs)
