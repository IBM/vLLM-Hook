import math
import os
import re
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.utils.torch_utils import set_random_seed
from vllm_metal.platform import MetalPlatform
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.utils import set_wired_limit

ATTN_PATTERNS = [
    re.compile(r"^model\.layers\.(\d+)\.self_attn$"),
    re.compile(r"^model\.model\.layers\.(\d+)\.self_attn$"),
]

PROJ_MODULE_NAME_TEMPLATE = "model.layers.{layer_num}.self_attn.attn.{proj_kind}"


def match_attn(name: str):
    """
    Return the layer index for a Metal self-attention module name.

    Args:
        name (str): Fully qualified module name to inspect.

    Returns:
        int | None: Parsed layer index if the module matches a Metal attention
        pattern, otherwise ``None``.
    """
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


class MLXHookWrapper(nn.Module):
    """
    Wrap an MLX attention module and invoke hook callbacks around calls.

    Args:
        None.

    Returns:
        None: Instances proxy calls to the wrapped MLX module.
    """

    def __init__(self, module, name, hook_fn):
        """
        Store the wrapped module metadata and hook callback.

        Args:
            module: MLX module being wrapped.
            name (str): Fully qualified module name.
            hook_fn: Callback invoked around wrapped module execution.

        Returns:
            None: Initializes wrapper state.
        """
        super().__init__()
        self.module = module
        self.name = name
        self.hook_fn = hook_fn

    def __call__(self, *args, **kwargs):
        """
        Invoke the hook before and after delegating to the wrapped module.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Any: Output produced by the wrapped module.
        """
        self.hook_fn("pre", args, kwargs, None, self.name)
        output = self.module(*args, **kwargs)
        self.hook_fn("post", args, kwargs, output, self.name)
        return output


class ProbeHookQKWorkerMetal:
    def _ensure_extension_state(self) -> None:
        if getattr(self, "_metal_qk_extension_ready", False):
            return
        self._execute_logged = False
        self._capture_active = False
        self._hooks_installed = False
        self._debug_hook = os.environ.get(
            "HOOK_DEBUG", os.environ.get("VLLM_HOOK_DEBUG", "")
        ) == "1"
        self._metal_qk_extension_ready = True

    def _stage(self, message: str) -> None:
        """
        Emit a standardized debug trace when hook debugging is enabled.

        Args:
            message (str): Debug message to print.

        Returns:
            None: The message is printed only when debug mode is enabled.
        """
        self._ensure_extension_state()
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
        """
        Initialize capture state and optional debug logging.

        Args:
            *args: Positional arguments forwarded to the base Metal worker.
            **kwargs: Keyword arguments forwarded to the base Metal worker.

        Returns:
            None: Worker state is initialized in-place.
        """
        self._ensure_extension_state()
        self._stage("worker __init__ complete")

    def install_hooks(self):
        """
        Install temporary self-attention wrappers for Metal capture.

        Returns:
            None: Wrappers are installed in-place.
        """
        self._ensure_extension_state()
        if getattr(self, "_hooks_installed", False):
            return
        self._hooks_installed = True
        self._capture_active = True
        self._stage("install_hooks start")
        try:
            self._install_hooks()
            self._stage("install_hooks complete")
            print("Hooks installed successfully", flush=True)
        except Exception as exc:
            self._stage(f"install_hooks failed: {type(exc).__name__}: {exc}")
            print(f"Hook installation failed: {exc}", flush=True)

    def _current_run_id(self) -> str | None:
        """
        Return the active run id when capture is armed for this execution.

        Args:
            None.

        Returns:
            str | None: Active run identifier, or ``None`` when capture is
            disabled for the current execution.
        """
        if not getattr(self, "_capture_active", False) or not os.path.exists(self.hook_flag):
            return None
        if not os.path.exists(self.run_id_file):
            raise RuntimeError("run_id not found")
        return open(self.run_id_file).read().strip().split("\n")[-1]

    def _ensure_run_cache(self, run_id: str):
        """
        Create or return the per-run cache used for recorded tensors.

        Args:
            run_id (str): Active run identifier.

        Returns:
            dict: Mutable cache entry for the requested run id.
        """
        cache = self._run_cache.get(run_id)
        if cache is None:
            cache = {
                "config": self._conf,
                # This remains `qkv_cache` rather than the non-Metal `qk_cache`
                # because the Metal path reconstructs and stores x/q/k/v data.
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
        """
        Append batched projection tensors to the per-layer cache.

        Args:
            run_id (str): Active run identifier.
            layer_num (int): Transformer layer index being recorded.
            proj_kind (str): Projection kind, such as ``x`` or ``q``.
            tokens (torch.Tensor): Batched projection tensor to append.

        Returns:
            None: Tokens are appended to the run cache in-place.
        """
        cache = self._ensure_run_cache(run_id)
        module_name = PROJ_MODULE_NAME_TEMPLATE.format(
            layer_num=layer_num,
            proj_kind=proj_kind,
        )
        layer_cache = cache["qkv_cache"].setdefault(
            module_name,
            {
                "layer_num": layer_num,
                "proj_kind": proj_kind,
                "tokens": [],
            },
        )
        layer_cache["tokens"].extend(
            [tokens[i].clone() for i in range(tokens.shape[0])]
        )

    def _append_proj_token_list(
        self,
        run_id: str,
        layer_num: int,
        proj_kind: str,
        token_list: list[torch.Tensor],
    ) -> None:
        """
        Append a pre-split list of projection tensors to the per-layer cache.

        Args:
            run_id (str): Active run identifier.
            layer_num (int): Transformer layer index being recorded.
            proj_kind (str): Projection kind, such as ``k`` or ``v``.
            token_list (list[torch.Tensor]): Per-sample tensors to append.

        Returns:
            None: Tensors are appended to the run cache in-place.
        """
        cache = self._ensure_run_cache(run_id)
        module_name = PROJ_MODULE_NAME_TEMPLATE.format(
            layer_num=layer_num,
            proj_kind=proj_kind,
        )
        layer_cache = cache["qkv_cache"].setdefault(
            module_name,
            {
                "layer_num": layer_num,
                "proj_kind": proj_kind,
                "tokens": [],
            },
        )
        layer_cache["tokens"].extend([tokens.clone() for tokens in token_list])

    def _mx_offsets_to_int_list(self, value, batch_size: int) -> list[int]:
        """
        Normalize MLX or scalar offset metadata to a Python integer list.

        Args:
            value: Offset metadata value from the MLX cache.
            batch_size (int): Number of samples in the batch.

        Returns:
            list[int]: Integer offsets for each sample in the batch.
        """
        if value is None:
            return [0] * batch_size
        if isinstance(value, int):
            return [int(value)] * batch_size
        if hasattr(value, "shape"):
            value_torch = mlx_to_torch(value, device="cpu")
            if value_torch.ndim == 0:
                return [int(value_torch.item())] * batch_size
            return [int(x) for x in value_torch.tolist()]
        return [int(value)] * batch_size

    def _flatten_attention_sample(self, sample) -> torch.Tensor:
        """
        Convert `[heads, seq, head_dim]` data to `[seq, heads * head_dim]`.

        Args:
            sample: Single attention sample in MLX head-major layout.

        Returns:
            torch.Tensor: Flattened tensor on CPU for analyzer consumption.
        """
        flat = sample.transpose(1, 0, 2).reshape(sample.shape[1], -1)
        return mlx_to_torch(flat, device="cpu")

    def _build_full_kv_without_mutation(self, cache, keys, values):
        """
        Rebuild full-history K/V tensors without mutating the live cache.

        Args:
            cache: Active MLX cache object, if present.
            keys: Current-step key tensor batch.
            values: Current-step value tensor batch.

        Returns:
            tuple[list, list]: Full key and value tensors for each sample.
        """
        batch_size = keys.shape[0]
        if cache is None or not hasattr(cache, "keys") or cache.keys is None:
            return [keys[i] for i in range(batch_size)], [values[i] for i in range(batch_size)]

        lengths = self._mx_offsets_to_int_list(getattr(cache, "offset", None), batch_size)
        left_padding = self._mx_offsets_to_int_list(
            getattr(cache, "left_padding", None), batch_size
        )

        full_keys = []
        full_values = []
        for i in range(batch_size):
            old_len = max(0, lengths[i])
            pad = max(0, left_padding[i])
            if old_len > 0:
                old_k = cache.keys[i, :, pad : pad + old_len, :]
                old_v = cache.values[i, :, pad : pad + old_len, :]
                full_keys.append(mx.concatenate([old_k, keys[i]], axis=1))
                full_values.append(mx.concatenate([old_v, values[i]], axis=1))
            else:
                full_keys.append(keys[i])
                full_values.append(values[i])
        return full_keys, full_values

    def _record_qkv(
        self,
        run_id: str,
        layer_num: int,
        raw_x,
        queries,
        key_samples,
        value_samples,
    ) -> None:
        """
        Store raw inputs and projected Q/K/V tensors for one layer.

        Args:
            run_id (str): Active run identifier.
            layer_num (int): Transformer layer index being recorded.
            raw_x: Original hidden-state tensor entering self-attention.
            queries: Query tensor after projection and rotary embedding.
            key_samples: Per-sample full-history key tensors.
            value_samples: Per-sample full-history value tensors.

        Returns:
            None: Recorded tensors are appended to the run cache.
        """
        x_torch = mlx_to_torch(raw_x, device="cpu")
        q_flat = queries.transpose(0, 2, 1, 3).reshape(
            queries.shape[0], queries.shape[2], -1
        )
        q_torch = mlx_to_torch(q_flat, device="cpu")
        k_torch = [self._flatten_attention_sample(sample) for sample in key_samples]
        v_torch = [self._flatten_attention_sample(sample) for sample in value_samples]

        self._append_proj_tokens(run_id, layer_num, "x", x_torch)
        if self.hookq_mode == "all_tokens":
            self._append_proj_tokens(run_id, layer_num, "q", q_torch)
        elif self.hookq_mode == "last_token":
            self._append_proj_tokens(run_id, layer_num, "q", q_torch[:, -1:, :])
        else:
            raise NotImplementedError(self.hookq_mode)
        self._append_proj_token_list(run_id, layer_num, "k", k_torch)
        self._append_proj_token_list(run_id, layer_num, "v", v_torch)

        if getattr(self, "_debug_hook", False):
            print(
                f"[qkv_boundary] module=model.layers.{layer_num}.self_attn.attn "
                f"x_shape={tuple(x_torch.shape)} q_shape={tuple(q_torch.shape)} "
                f"k_shape={tuple(k_torch.shape)} v_shape={tuple(v_torch.shape)}",
                flush=True,
            )

    def _capture_from_self_attn(
        self, run_id: str, layer_num: int, attn_module, input_args, input_kwargs
    ) -> None:
        """
        Recompute and record the Q/K/V tensors used by a self-attention call.

        Args:
            run_id (str): Active run identifier.
            layer_num (int): Transformer layer index being recorded.
            attn_module: Wrapped MLX self-attention module.
            input_args: Positional arguments passed to the wrapped module.

        Returns:
            None: Recomputed tensors are recorded into the run cache.
        """
        raw_x = input_args[0]
        cache = input_kwargs.get("cache") if input_kwargs else None
        if cache is None:
            cache = input_args[2] if len(input_args) > 2 else None
        batch, seq_len, _ = raw_x.shape

        queries = attn_module.q_proj(raw_x).reshape(
            batch, seq_len, attn_module.n_heads, -1
        )
        keys = attn_module.k_proj(raw_x).reshape(
            batch, seq_len, attn_module.n_kv_heads, -1
        )
        values = attn_module.v_proj(raw_x).reshape(
            batch, seq_len, attn_module.n_kv_heads, -1
        )

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = attn_module.rope(queries, offset=cache.offset)
            keys = attn_module.rope(keys, offset=cache.offset)
        else:
            queries = attn_module.rope(queries)
            keys = attn_module.rope(keys)

        full_keys, full_values = self._build_full_kv_without_mutation(cache, keys, values)

        self._record_qkv(run_id, layer_num, raw_x, queries, full_keys, full_values)

    def _parse_layer_heads(self) -> Dict[int, List[int]]:
        """
        Parse `layer:head,head;layer:head` into a layer-to-head mapping.

        Args:
            None.

        Returns:
            Dict[int, List[int]]: Mapping from layer index to sorted head indices.
        """
        layer_heads = os.environ.get("HOOK_LAYER_HEADS", os.environ.get("VLLM_HOOK_LAYER_HEADS", ""))
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

    def _install_hooks(self):
        """
        Install wrappers on tracked self-attention modules.

        Args:
            None.

        Returns:
            None: Wrappers are installed in-place on the loaded model.
        """
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("HOOK_FLAG", os.environ.get("VLLM_HOOK_FLAG"))
        self.hook_dir = os.environ.get("HOOK_DIR", os.environ.get("VLLM_HOOK_DIR"))
        self.run_id_file = os.environ.get("HOOK_RUN_ID", os.environ.get("VLLM_RUN_ID"))
        self.hookq_mode = os.environ.get("HOOKQ_MODE", os.environ.get("VLLM_HOOKQ_MODE", "all_tokens"))
        # This remains configurable because the Metal wrapper can observe both
        # pre- and post-call boundaries, unlike the non-Metal forward hook.
        self.capture_phase = os.environ.get("HOOK_CAPTURE_PHASE", os.environ.get("VLLM_HOOK_CAPTURE_PHASE", "pre"))

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

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
            cfg.get(
                "num_hidden_layers", len(layers_obj) if layers_obj is not None else 0
            )
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

        named_modules = dict(model.named_modules())
        seen_targets = set()

        def capture_module_for(module):
            return getattr(module, "_inner", module)

        def has_capture_api(module) -> bool:
            return all(
                hasattr(module, attr)
                for attr in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "rope",
                    "n_heads",
                    "n_kv_heads",
                )
            )

        def install_wrapper(name, parent, target_name, layer_idx, module) -> bool:
            target_key = (id(parent), target_name)
            if target_key in seen_targets:
                return False
            capture_attn = capture_module_for(module)
            if not has_capture_api(capture_attn):
                return False

            original_attn = module

            def attention_hook(
                phase,
                input_args,
                input_kwargs,
                _output,
                _module_name,
                layer_num=layer_idx,
                attn=capture_attn,
            ):
                """
                Capture one self-attention call when the active run is armed.

                Args:
                    phase: Wrapper callback phase, such as ``pre`` or ``post``.
                    input_args: Positional arguments passed to self-attention.
                    input_kwargs: Keyword arguments passed to self-attention.
                    _output: Output produced by the wrapped module.
                    _module_name: Hook-reported module name.
                    layer_num (int): Bound transformer layer index.
                    attn: Bound original attention module.

                Returns:
                    None: Capture side effects are written into the run cache.
                """
                run_id = self._current_run_id()
                if run_id is None:
                    return None
                if phase != self.capture_phase:
                    return None
                self._capture_from_self_attn(
                    run_id, layer_num, attn, input_args, input_kwargs or {}
                )
                return None

            # This remains a wrapper replacement instead of `register_forward_hook`
            # because the Metal MLX modules do not expose the same hook API as
            # the non-Metal PyTorch modules.
            wrapped_attn = MLXHookWrapper(
                module=original_attn,
                name=name,
                hook_fn=attention_hook,
            )
            setattr(parent, target_name, wrapped_attn)
            seen_targets.add(target_key)
            self._hooks.append(
                {
                    "parent": parent,
                    "target_name": target_name,
                    "original_module": original_attn,
                }
            )
            self._matched_hook_modules.append(name)
            return True

        for name, module in named_modules.items():
            layer_idx = match_attn(name)
            if layer_idx is None:
                continue
            if layer_idx not in self.important_layers:
                continue

            parent_name, target_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name)
            if parent is None:
                continue

            install_wrapper(name, parent, target_name, layer_idx, module)

        if not self._matched_hook_modules:
            try:
                from vllm_metal.paged_attention_common import (
                    find_attn_attr,
                    find_layers,
                )
            except Exception:
                find_attn_attr = None
                find_layers = None

            if find_layers is not None and find_attn_attr is not None:
                for layer_idx, layer in enumerate(find_layers(model)):
                    if layer_idx not in self.important_layers:
                        continue
                    attn_attr = find_attn_attr(layer)
                    if attn_attr is None:
                        continue
                    module = getattr(layer, attn_attr)
                    name = f"model.layers.{layer_idx}.{attn_attr}"
                    install_wrapper(name, layer, attn_attr, layer_idx, module)

        if not self._matched_hook_modules:
            print("Could not locate self_attn modules for Metal attention hook")
            return

        print(
            f"Installed {len(self._matched_hook_modules)} hooks on layers: "
            f"{self._matched_hook_modules}",
            flush=True,
        )
        if getattr(self, "_debug_hook", False):
            print(
                "Selected Metal hook layers: "
                f"{sorted(self.important_layers)} "
                f"({self._capture_boundary} boundary)",
                flush=True,
            )
            print("Installed Granite attention hook", flush=True)

    def _uninstall_hooks(self):
        """
        Restore the original attention modules after temporary wrapping.

        Args:
            None.

        Returns:
            None: Wrapped modules are restored in-place.
        """
        self._ensure_extension_state()
        self._capture_active = False
        self._flush_run_cache()
        for entry in reversed(self._hooks):
            setattr(entry["parent"], entry["target_name"], entry["original_module"])
        self._hooks.clear()

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """
        Compatibility RPC for vLLM's post-generate artifact collection path.

        Metal Q/K capture is keyed by the run-id flag file rather than vLLM's
        request id, so there is no per-request in-memory payload to return.
        Flushing here makes the shared plugin path produce the disk artifact the
        Metal analyzer expects without keeping the captured tensors resident.
        """
        self._ensure_extension_state()
        self._stage(
            "get_captured_states compatibility flush "
            f"for request_id={external_req_id}"
        )
        self._flush_run_cache()
        return None

    def clear_captured_states(self, external_req_id: str) -> None:
        """
        Compatibility RPC for aborted in-memory collection requests.

        Metal captures are run-scoped. If vLLM asks to clear request-scoped
        states, drop the current run cache to avoid retaining tensors.
        """
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

        The Metal worker already writes to the hook directory configured through
        HOOK_DIR/RUN_ID. The request ids and hook_dir arguments are accepted so
        the shared vLLM hook plugin can call the same RPC on all backends.
        """
        self._ensure_extension_state()
        self._stage(
            "flush_disk compatibility flush "
            f"for run_id={run_id} request_ids={external_req_ids}"
        )
        return self._flush_run_cache()

    def _flush_run_cache(self) -> bool:
        """
        Persist any captured run cache entries to disk.

        Args:
            None.

        Returns:
            bool: True when at least one run cache artifact was serialized.
        """
        if not getattr(self, "_run_cache", None):
            return False

        tp_rank = int(self.rank)
        for run_id, cache in self._run_cache.items():
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qkv.pt")
            # This stays as a post-execution flush instead of per-hook `torch.save`
            # because the Metal worker reconstructs and batches multiple tensors
            # per layer, making immediate writes much more expensive than in the
            # non-Metal forward-hook path.
            torch.save(cache, cache_path)
            if getattr(self, "_debug_hook", False):
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
        self._run_cache.clear()
        return True
