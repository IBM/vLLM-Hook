import math
import os
from typing import Dict, List

import mlx.core as mx
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
        self.hook_fn("pre", args, None, self.name)
        output = self.module(*args, **kwargs)
        self.hook_fn("post", args, output, self.name)
        return output


class ProbeHookQKWorkerMetal(MetalWorker):
    def _stage(self, message: str) -> None:
        # Shared debug-print helper so all optional worker traces use the same
        # prefix and can be silenced together.
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
        # Track whether the current execution window should be recorded and
        # whether we have already emitted the one-time execute_model trace.
        self._execute_logged = False
        self._capture_active = False
        self._debug_hook = os.environ.get("VLLM_HOOK_DEBUG", "") == "1"
        super().__init__(*args, **kwargs)
        self._stage("worker __init__ complete")

    def init_device(self) -> None:
        # Use the simplified Metal setup when running a single process, but
        # preserve the normal distributed worker path for larger world sizes.
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
        # Set up MLX, the torch-facing Metal device, and the model runner for
        # the common single-process case.
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
        # Load the underlying model first, then install the temporary wrappers
        # that let this worker observe self_attn traffic for one hooked engine.
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
        layer_cache["tokens"].extend([tokens.clone() for tokens in token_list])

    def _mx_offsets_to_int_list(self, value, batch_size: int) -> list[int]:
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
        # Convert one `[heads, seq, head_dim]` packet into the legacy
        # `[seq, heads * head_dim]` notebook layout.
        flat = sample.transpose(1, 0, 2).reshape(sample.shape[1], -1)
        return mlx_to_torch(flat, device="cpu")

    def _build_full_kv_without_mutation(self, cache, keys, values):
        # Reconstruct the attention-ready full-history K/V packets that the
        # live MLX attention path will use, without mutating the real cache.
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

    def _record_qkv(  # qkv_hook in non metal worker.
        self,
        run_id: str,
        layer_num: int,
        raw_x,
        queries,
        key_samples,
        value_samples,
    ) -> None:
        # Housing analogy:
        # - `raw_x` is the visitor as it first enters the self-attention house
        # - `queries`, `keys`, and `values` are the specialized copies of that
        #   visitor after the house has routed it through q_proj/k_proj/v_proj
        # - this method files both the original visitor packet and the
        #   specialized packets into the notebook under the same floor tab
        # `raw_x` already has the analyzer-friendly `[batch, seq, hidden]`
        # layout, so it can be bridged to CPU as-is.
        x_torch = mlx_to_torch(raw_x, device="cpu")
        # Q/K/V arrive here in the attention-worker layout
        # `[batch, heads, seq, head_dim]`.
        # The notebook stores the flatter analyzer layout
        # `[batch, seq, heads * head_dim]`, so we transpose seq back to the
        # middle and then collapse the head dimensions.
        q_flat = queries.transpose(0, 2, 1, 3).reshape(
            queries.shape[0], queries.shape[2], -1
        )
        # Archive everything on CPU so the on-disk notebook does not depend on
        # the lifetime of MLX tensors or Metal device state.
        q_torch = mlx_to_torch(q_flat, device="cpu")
        k_torch = [self._flatten_attention_sample(sample) for sample in key_samples]
        v_torch = [self._flatten_attention_sample(sample) for sample in value_samples]

        self._append_proj_tokens(run_id, layer_num, "x", x_torch)
        # Q is the only packet kind whose storage width changes with
        # `hookq_mode`: either keep the whole packet or just the last page.
        if self.hookq_mode == "all_tokens":
            self._append_proj_tokens(run_id, layer_num, "q", q_torch)
        elif self.hookq_mode == "last_token":
            self._append_proj_tokens(run_id, layer_num, "q", q_torch[:, -1:, :])
        else:
            raise NotImplementedError(self.hookq_mode)
        self._append_proj_token_list(run_id, layer_num, "k", k_torch)
        self._append_proj_token_list(run_id, layer_num, "v", v_torch)

        if self._debug_hook:
            print(
                f"[qkv_boundary] module=model.layers.{layer_num}.self_attn.attn "
                f"x_shape={tuple(x_torch.shape)} q_shape={tuple(q_torch.shape)} "
                f"k_shape={tuple(k_torch.shape)} v_shape={tuple(v_torch.shape)}",
                flush=True,
            )

    def _capture_from_self_attn(
        self, run_id: str, layer_num: int, attn_module, input_args
    ) -> None:
        # Housing analogy:
        # - this is the recording room attached to the `self_attn` house door
        # - visitors arrive as raw hidden states `x`
        # - we let the recording room compute the same q/k/v routes the house
        #   itself would compute internally
        # - that keeps the notebook aligned with the real house behavior while
        #   avoiding any need for a deeper inner door that does not exist in
        #   the installed Metal runtime
        # The wrapped `self_attn` call receives `(x, mask, cache)`. We only
        # need the raw hidden-state visitor plus the cache offset information
        # to rebuild the same attention-ready Q/K/V packets the house will use.
        raw_x = input_args[0]
        cache = input_args[2] if len(input_args) > 2 else None
        batch, seq_len, _ = raw_x.shape

        # Run the same projection occupants the real house uses. Right after
        # projection the tensors are still in `[batch, seq, heads, head_dim]`.
        queries = attn_module.q_proj(raw_x).reshape(
            batch, seq_len, attn_module.n_heads, -1
        )
        keys = attn_module.k_proj(raw_x).reshape(
            batch, seq_len, attn_module.n_kv_heads, -1
        )
        values = attn_module.v_proj(raw_x).reshape(
            batch, seq_len, attn_module.n_kv_heads, -1
        )

        # Switch into the attention-worker layout `[batch, heads, seq, head_dim]`
        # before applying rope, because that is the layout the live house uses
        # internally for attention.
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Q and K must pass through the same positional checkpoint as the live
        # attention path. During prefill/decode, `cache.offset` keeps rope
        # aligned with the already-seen prefix.
        if cache is not None:
            queries = attn_module.rope(queries, offset=cache.offset)
            keys = attn_module.rope(keys, offset=cache.offset)
        else:
            queries = attn_module.rope(queries)
            keys = attn_module.rope(keys)

        full_keys, full_values = self._build_full_kv_without_mutation(cache, keys, values)

        # At this point the packets match the real attention-ready values, so
        # the notebook can archive them without depending on a deeper hook.
        self._record_qkv(run_id, layer_num, raw_x, queries, full_keys, full_values)

    def _install_hooks(self):
        # `model_runner.model` is the live MLX model object that will actually
        # be executed. If it is missing, there is nothing concrete to wrap.
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # These env vars are the worker-side control plane for one hooked run:
        # - `hook_flag` says whether capture is currently armed
        # - `hook_dir` tells us where notebooks should be archived
        # - `run_id_file` tells us which notebook belongs to the active visit
        # - `hookq_mode` controls whether Q is stored for all tokens or only
        #   the last token
        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self.hookq_mode = os.environ.get("VLLM_HOOKQ_MODE", "all_tokens")
        self.capture_phase = os.environ.get("VLLM_HOOK_CAPTURE_PHASE", "pre")

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

        # Gather the structural facts the analyzer will need later to interpret
        # the recorded Q/K/V packets. We prefer `model_args`, but fall back to
        # live model attributes when necessary because different runtime builds
        # expose these values in different places.
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

        # `_conf` is the notebook header: enough shape metadata for the
        # analyzer to reconstruct attention from archived Q/K/V packets later.
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
            # Only `*.self_attn` addresses are valid outer-house targets for
            # this worker. Everything else in the building directory is ignored.
            if not name.endswith(".self_attn"):
                continue

            parts = name.split(".")
            # We only want modules that clearly belong to `layers.<idx>` so we
            # can map them back to the configured tracked floors.
            if "layers" not in parts:
                continue
            layers_pos = parts.index("layers")
            if layers_pos + 1 >= len(parts) or not parts[layers_pos + 1].isdigit():
                continue

            layer_idx = int(parts[layers_pos + 1])
            # Skip floors that are not in the configured layer/head shortlist.
            if layer_idx not in self.important_layers:
                continue
            # The runtime must expose the projection and rope occupants we need
            # to rebuild attention-ready Q/K/V at the self_attn door.
            if not all(
                hasattr(module, attr)
                for attr in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "rope",
                    "n_heads",
                    "n_kv_heads",
                )
            ):
                continue

            # We replace the module at `parent.target_name`, so we need the
            # parent house object as well as the current resident.
            parent_name, target_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name)
            if parent is None:
                continue

            original_attn = module

            def attention_hook(
                phase,
                input_args,
                _output,
                _module_name,
                layer_num=layer_idx,
                attn=original_attn,
            ):
                # Housing analogy:
                # - the wrapper's recording room checks whether there is an open
                #   notebook for this visit
                # - if so, it records the raw visitor and the projected Q/K/V
                #   copies for this floor
                run_id = self._current_run_id()
                if run_id is None:
                    return None
                if phase != self.capture_phase:
                    return None
                self._capture_from_self_attn(run_id, layer_num, attn, input_args)
                return None

            # Install the wrapper at the public address and remember enough
            # information to restore the original house during teardown.
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

        # If nothing matched, the worker cannot observe any self_attn traffic,
        # so later analysis would have no notebook to read.
        if not self._matched_hook_modules:
            print("Could not locate self_attn modules for Metal attention hook")
            return

        # Report the exact wrapped addresses so it is obvious which floors were
        # actually instrumented for this temporary engine.
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
        # - the archive is grouped first by run, then by tensor-parallel rank,
        #   so multiple workers can file notebooks for the same visit without
        #   overwriting each other
        if not self._run_cache:
            return

        tp_rank = int(self.rank)
        for run_id, cache in self._run_cache.items():
            # Each run gets its own directory, and each worker writes inside
            # `tp_rank_<n>` so later merge logic can collect the right notebook
            # fragments for that run.
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qkv.pt")
            # `torch.save` is the archive step: the entire notebook header plus
            # all filed packet sections are serialized in one artifact.
            torch.save(cache, cache_path)
            if self._debug_hook:
                # Debug mode summarizes how many request packets were filed in
                # each notebook section so batch behavior is easy to inspect.
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
        # Parse the `layer:head,head;layer:head` env-var format into the layer
        # lookup the worker uses to decide which floors to wrap and score.
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
        # The one-time `_execute_logged` banner keeps debug logs readable
        # without repeating the same startup line on every execution call.
        if not self._execute_logged:
            self._stage("execute_model first entry")
            self._execute_logged = True
        # Wrappers consult `_capture_active` through `_current_run_id()`, so
        # this flag is the top-level switch that distinguishes real execution
        # traffic from warmup/setup/teardown traffic.
        self._capture_active = True
        try:
            # The underlying Metal worker performs the real forward/generation
            # work while wrapper houses opportunistically record visits.
            result = super().execute_model(*args, **kwargs)
        finally:
            # Always drop the capture flag, even if model execution raises, so
            # later calls do not accidentally keep writing into the notebook.
            self._capture_active = False
        try:
            # Archive any notebooks filled during this execution window before
            # returning the model result to the caller.
            self._flush_run_cache()
        except Exception as exc:
            self._stage(f"flush_run_cache failed: {type(exc).__name__}: {exc}")
            raise
        return result
