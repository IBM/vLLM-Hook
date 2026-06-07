import os
import re

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.utils.torch_utils import set_random_seed
from vllm_metal.platform import MetalPlatform
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.utils import set_wired_limit


LAYER_PATTERNS = [
    re.compile(r"^model\.layers\.(\d+)$"),
    re.compile(r"^model\.model\.layers\.(\d+)$"),
    re.compile(r"^language_model\.model\.layers\.(\d+)$"),
]


def match_layer(name: str):
    for pat in LAYER_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


class MLXHiddenStatesWrapper(nn.Module):
    def __init__(self, module, name, hook_fn):
        super().__init__()
        self.module = module
        self.name = name
        self.hook_fn = hook_fn

    def __call__(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        self.hook_fn(output, self.name)
        return output


class ProbeHiddenStatesWorkerMetal:
    def _ensure_extension_state(self) -> None:
        if getattr(self, "_metal_hidden_extension_ready", False):
            return
        self._execute_logged = False
        self._capture_active = False
        self._hooks_installed = False
        self._debug_hook = os.environ.get(
            "HOOK_DEBUG", os.environ.get("VLLM_HOOK_DEBUG", "")
        ) == "1"
        self._metal_hidden_extension_ready = True

    def __init__(self, *args, **kwargs):
        self._ensure_extension_state()
        self._stage("worker __init__ complete")

    def _stage(self, message: str) -> None:
        self._ensure_extension_state()
        if getattr(self, "_debug_hook", False):
            print(f"[metal-hidden-worker pid={os.getpid()}] {message}", flush=True)

    def install_hooks(self):
        self._ensure_extension_state()
        if getattr(self, "_hooks_installed", False):
            return
        self._hooks_installed = True
        self._capture_active = True
        self._stage("install_hooks start")
        try:
            self._install_hooks()
            print("Hooks installed successfully", flush=True)
        except Exception as exc:
            self._stage(f"install_hooks failed: {type(exc).__name__}: {exc}")
            print(f"Hook installation failed: {exc}", flush=True)
            raise
        self._stage("install_hooks complete")

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("HOOK_FLAG", os.environ.get("VLLM_HOOK_FLAG"))
        self.hook_dir = os.environ.get("HOOK_DIR", os.environ.get("VLLM_HOOK_DIR"))
        self.run_id_file = os.environ.get("HOOK_RUN_ID", os.environ.get("VLLM_RUN_ID"))
        self.hs_mode = os.environ.get("HOOK_HS_MODE", os.environ.get("VLLM_HOOK_HS_MODE", "last_token"))

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

        self.important_layers = self._parse_layers()
        self._run_cache = {}
        self._hooks = []
        self._matched_hook_modules = []

        cfg = getattr(self.model_runner, "model_args", None) or {}
        layers_obj = getattr(getattr(model, "model", model), "layers", None)
        if layers_obj is None:
            layers_obj = getattr(model, "layers", None)

        num_layers = int(
            cfg.get("num_hidden_layers", len(layers_obj) if layers_obj is not None else 0)
        )
        hidden = int(cfg.get("hidden_size", getattr(model, "dim", 0)))
        self._conf = {"hidden_size": hidden, "num_layers": num_layers}

        named_modules = dict(model.named_modules())
        seen_targets = set()

        def replace_child(parent, target_name, wrapped):
            if hasattr(parent, "__setitem__"):
                try:
                    parent[target_name] = wrapped
                    return
                except Exception:
                    pass
            if isinstance(target_name, str):
                setattr(parent, target_name, wrapped)
                return
            raise TypeError(
                f"Cannot replace child {target_name!r} on parent type {type(parent).__name__}"
            )

        def install_wrapper(name, parent, target_name, layer_idx, module) -> bool:
            target_key = (id(parent), target_name)
            if target_key in seen_targets:
                return False

            wrapped = MLXHiddenStatesWrapper(
                module=module,
                name=name,
                hook_fn=lambda output, module_name, ln=layer_idx + 1: (
                    self._hidden_states_hook(output, module_name, ln)
                ),
            )
            replace_child(parent, target_name, wrapped)
            seen_targets.add(target_key)
            self._hooks.append(
                {
                    "parent": parent,
                    "target_name": target_name,
                    "original_module": module,
                }
            )
            self._matched_hook_modules.append(name)
            return True

        for name, module in named_modules.items():
            layer_idx = match_layer(name)
            if layer_idx is None:
                continue
            exposed_layer_num = layer_idx + 1
            if self.important_layers and exposed_layer_num not in self.important_layers:
                continue
            parent_name, target_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name)
            if parent is None:
                continue
            install_wrapper(name, parent, target_name, layer_idx, module)

        if not self._matched_hook_modules and layers_obj is not None:
            try:
                indexed_layers = list(layers_obj)
            except TypeError:
                indexed_layers = []
            for layer_idx, module in enumerate(indexed_layers):
                exposed_layer_num = layer_idx + 1
                if (
                    self.important_layers
                    and exposed_layer_num not in self.important_layers
                ):
                    continue
                install_wrapper(
                    f"layers.{layer_idx}",
                    layers_obj,
                    layer_idx,
                    layer_idx,
                    module,
                )

        if not self._matched_hook_modules:
            try:
                from vllm_metal.paged_attention_common import find_layers
            except Exception:
                find_layers = None

            if find_layers is not None:
                layers = find_layers(model)
                for layer_idx, module in enumerate(layers):
                    exposed_layer_num = layer_idx + 1
                    if (
                        self.important_layers
                        and exposed_layer_num not in self.important_layers
                    ):
                        continue
                    install_wrapper(
                        f"model.layers.{layer_idx}",
                        layers,
                        layer_idx,
                        layer_idx,
                        module,
                    )

        if not self._matched_hook_modules:
            raise RuntimeError(
                "Could not locate decoder layers for Metal hidden-state hook. "
                f"Requested layers={sorted(self.important_layers)}."
            )

        print(
            f"Installed {len(self._matched_hook_modules)} hidden-state hooks on "
            f"layers: {self._matched_hook_modules}",
            flush=True,
        )
        if getattr(self, "_debug_hook", False):
            print(
                "Selected Metal hidden-state hook layers: "
                f"{sorted(self.important_layers)}",
                flush=True,
            )

    def _parse_layers(self) -> set[int]:
        raw = os.environ.get("HOOK_LAYERS", os.environ.get("VLLM_HOOK_LAYERS", ""))
        return {int(part) for part in raw.split(";") if part.strip()}

    def _current_run_id(self) -> str | None:
        if not getattr(self, "_capture_active", False) or not os.path.exists(self.hook_flag):
            return None
        if not os.path.exists(self.run_id_file):
            raise RuntimeError("run_id not found")
        with open(self.run_id_file) as f:
            return f.read().strip().split("\n")[-1]

    def _ensure_run_cache(self, run_id: str) -> dict:
        cache = self._run_cache.get(run_id)
        if cache is None:
            cache = {
                "config": self._conf,
                "hs_cache": {},
                "meta": {"tp_rank": int(self.rank)},
            }
            self._run_cache[run_id] = cache
        return cache

    def _to_torch_cpu(self, value) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().cpu()
        return mlx_to_torch(value, device="cpu").detach()

    def _extract_hidden(self, output):
        if isinstance(output, tuple) and len(output) == 2:
            first, second = output
            if torch.is_tensor(second):
                return first + second
            return first + second
        if isinstance(output, tuple):
            return output[0]
        return output

    def _hidden_states_hook(self, output, module_name: str, layer_num: int) -> None:
        run_id = self._current_run_id()
        if run_id is None:
            return

        hidden = self._to_torch_cpu(self._extract_hidden(output))

        per_request_activations = None
        if self.hs_mode == "last_token":
            if hidden.dim() == 3:
                activation = hidden[:, -1, :].clone()
                per_request_activations = [
                    activation[i].clone() for i in range(activation.shape[0])
                ]
            else:
                activation = hidden[-1:, :].clone()
                per_request_activations = [activation.squeeze(0).clone()]
        elif self.hs_mode == "all_tokens":
            activation = hidden.clone()
        else:
            raise NotImplementedError(self.hs_mode)

        cache = self._ensure_run_cache(run_id)
        layer_cache = cache["hs_cache"].setdefault(
            module_name,
            {
                "hidden_states": [],
                "layer_num": layer_num,
                "hs_mode": self.hs_mode,
            },
        )

        if per_request_activations is not None:
            layer_cache["hidden_states"].extend(per_request_activations)
        elif activation.dim() == 3:
            layer_cache["hidden_states"].extend(
                [activation[i].clone() for i in range(activation.shape[0])]
            )
        else:
            layer_cache["hidden_states"].append(activation.clone())

    def _uninstall_hooks(self):
        self._ensure_extension_state()
        self._capture_active = False
        self._flush_run_cache()
        for entry in reversed(getattr(self, "_hooks", [])):
            parent = entry["parent"]
            if isinstance(parent, list):
                parent[entry["target_name"]] = entry["original_module"]
            else:
                setattr(parent, entry["target_name"], entry["original_module"])
        if hasattr(self, "_hooks"):
            self._hooks.clear()
        self._hooks_installed = False

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """
        Compatibility RPC for vLLM's post-generate artifact collection path.

        Metal hidden-state capture is run-scoped and disk-backed. Flush the
        current run cache so the analyzer can read hidden_states.pt, but return
        no per-request in-memory payload.
        """
        self._ensure_extension_state()
        self._stage(
            "get_captured_states compatibility flush "
            f"for request_id={external_req_id}"
        )
        self._flush_run_cache()
        return None

    def clear_captured_states(self, external_req_id: str) -> None:
        """Drop any run-scoped hidden-state cache for request cleanup RPCs."""
        self._ensure_extension_state()
        self._stage(
            "clear_captured_states compatibility clear "
            f"for request_id={external_req_id}"
        )
        if getattr(self, "_run_cache", None):
            self._run_cache.clear()

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """
        Compatibility RPC for save_to_disk collection.

        The Metal worker writes to the hook directory configured by HOOK_DIR and
        RUN_ID. Arguments are accepted for API parity with non-Metal workers.
        """
        self._ensure_extension_state()
        self._stage(
            "flush_disk compatibility flush "
            f"for run_id={run_id} request_ids={external_req_ids}"
        )
        return self._flush_run_cache()

    def _flush_run_cache(self) -> bool:
        if not getattr(self, "_run_cache", None):
            self._stage("flush_run_cache found no captured hidden states")
            return False

        tp_rank = int(self.rank)
        for run_id, cache in self._run_cache.items():
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            torch.save(cache, os.path.join(run_dir, "hidden_states.pt"))
            self._stage(
                f"flushed hidden states for run_id={run_id} "
                f"modules={list(cache['hs_cache'].keys())}"
            )
        self._run_cache.clear()
        return True
