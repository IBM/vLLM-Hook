import os
import math
import torch
from typing import Dict, List
from vllm.v1.worker.gpu_worker import Worker as V1Worker
from vllm.forward_context import get_forward_context
import re
from vllm.distributed import parallel_state as ps

PROJ_MODULE_NAME_TEMPLATE = "{attn_name}.{proj_kind}"

ATTN_PATTERNS = [
    # GPT-2: transformer.h.<i>.attn
    re.compile(r"^transformer\.h\.(\d+)\.attn.attn$"),

    # OPT: model.decoder.layers.<i>.self_attn
    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn.attn$"),

    # Qwen/LLaMA: model.layers.<i>.self_attn
    re.compile(r"^model\.layers\.(\d+)\.self_attn$"),
    re.compile(r"^model\.layers\.(\d+)\.self_attn.attn$"),
]

def match_attn(name: str):
    """
    Return the layer index for a supported attention module name.

    Args:
        name (str): Fully qualified module name to inspect.

    Returns:
        int | None: Parsed layer index if the module matches an attention
        pattern, otherwise ``None``.
    """
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


def _segment_bounds_from_metadata(metadata, module_name: str, device: torch.device):
    """
    Extract flattened request boundaries from vLLM attention metadata.

    Args:
        metadata: Attention metadata object provided by the forward context.
        module_name (str): Name of the hooked attention module.
        device (torch.device): Device used for any temporary tensors.

    Returns:
        torch.Tensor | None: Boundary tensor of shape ``[batch + 1]`` when
        bounds can be recovered, otherwise ``None``.
    """
    if metadata is None:
        return None

    candidates = [metadata]
    for attr in ("prefill", "decode", "_cached_prefill_metadata", "_cached_decode_metadata"):
        nested = getattr(metadata, attr, None)
        if nested is not None:
            candidates.append(nested)

    if hasattr(metadata, "__contains__"):
        try:
            if module_name in metadata:
                candidates.append(metadata[module_name])
        except Exception:
            pass

    for candidate in candidates:
        if candidate is None:
            continue

        query_start_loc = getattr(candidate, "query_start_loc", None)
        if query_start_loc is not None:
            if not torch.is_tensor(query_start_loc):
                query_start_loc = torch.tensor(query_start_loc, device=device)
            else:
                query_start_loc = query_start_loc.to(device)
            if query_start_loc.numel() >= 2:
                return query_start_loc.to(dtype=torch.long)

        for attr in ("seq_lens_tensor", "seq_lens"):
            seq_lens = getattr(candidate, attr, None)
            if seq_lens is None:
                continue
            if not torch.is_tensor(seq_lens):
                seq_lens = torch.tensor(seq_lens, device=device)
            else:
                seq_lens = seq_lens.to(device)
            if seq_lens.numel() == 0:
                continue
            seq_lens = seq_lens.to(dtype=torch.long)
            return torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    torch.cumsum(seq_lens, dim=0),
                ]
            )

    return None

class ProbeHookQKWorker(V1Worker):

    def load_model(self, *args, **kwargs):
        """
        Load the model and install forward hooks for tracked layers.

        Args:
            *args: Positional arguments forwarded to the base worker.
            **kwargs: Keyword arguments forwarded to the base worker.

        Returns:
            Any: Result returned by the base worker ``load_model`` call.
        """
        r = super().load_model(*args, **kwargs)
        
        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")
            
        return r

    def _install_hooks(self):
        """
        Register forward hooks on configured attention modules.

        Args:
            None.

        Returns:
            None: Hooks are installed in-place on the loaded model.
        """
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self.hookq_mode = os.environ.get("VLLM_HOOKQ_MODE", "all_tokens") # ["last_token", "all_tokens"]
        self._debug_hook = os.environ.get("VLLM_HOOK_DEBUG", "") == "1"
        self._debug_seen = set()
        tp_rank = int(ps.get_tensor_model_parallel_rank())
        
        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return            

        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())
        
        self._run_cache = {}
        
        cfg = model.config
        num_h = int(getattr(cfg, "num_attention_heads"))
        num_kv = int(getattr(cfg, "num_key_value_heads", num_h))
        hidden = int(getattr(cfg, "hidden_size"))
        head_dim = hidden // num_h
        attn_mult = float(getattr(cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=attn_mult,
        )

        def _active_run_id():
            if not os.path.exists(self.hook_flag):
                return None
            if os.path.exists(self.run_id_file):
                return (open(self.run_id_file).read().strip().split('\n'))[-1]
            raise Exception("run_id not found.")

        def _get_or_create_cache(run_id):
            cache = self._run_cache.get(run_id)
            if cache is None:
                cache = {"config": self._conf, "qk_cache": {}}
                self._run_cache[run_id] = cache
            return cache

        def _save_cache(run_id, cache):
            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qk.pt")
            torch.save(cache, cache_path)
            if self._debug_hook:
                print(
                    f"[hook-debug] saved qk cache run_id={run_id} path={cache_path} "
                    f"modules={list(cache['qk_cache'].keys())}",
                    flush=True,
                )

        def _debug_once(reason, module_name, extra=""):
            if not self._debug_hook:
                return
            key = (reason, module_name)
            if key in self._debug_seen:
                return
            self._debug_seen.add(key)
            suffix = f" {extra}" if extra else ""
            print(
                f"[hook-debug] {reason} module={module_name}{suffix}",
                flush=True,
            )

        def _ensure_layer_cache(cache, module_name, layer_num):
            layer_cache = cache["qk_cache"].get(module_name)
            if layer_cache is None:
                layer_cache = {"q": [], "k_all": [], "layer_num": layer_num}
                cache["qk_cache"][module_name] = layer_cache
            return layer_cache

        def qkv_hook(input, module_name):
            """
            Capture Q/K tensors for one attention module invocation.

            Args:
                input: Forward-hook input tuple for the attention module.
                module_name (str): Fully qualified hooked module name.

            Returns:
                None: Captured tensors are written into the worker cache.
            """

            run_id = _active_run_id()
            if run_id is None:
                _debug_once("skip_no_active_run", module_name)
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)

            # Warmup or non-attention passes: nothing to do
            if metadata is None:
                _debug_once("skip_no_attn_metadata", module_name)
                return

            # Some Granite/vLLM attention wrappers do not expose q/k tensors in
            # the forward-hook input tuple. In those cases, let the q_proj/k_proj
            # fallback hooks capture tensors instead of crashing here.
            if len(input) < 2 or not hasattr(input[0], "device"):
                _debug_once(
                    "skip_unusable_hook_input",
                    module_name,
                    extra=f"input_len={len(input)}",
                )
                return
        
            bounds = _segment_bounds_from_metadata(
                metadata,
                module_name,
                input[0].device,
            )
            if bounds is None or bounds.numel() < 2:
                _debug_once("skip_no_segment_bounds", module_name)
                return

            bs = bounds.numel() - 1

            layer_num = match_attn(module_name)
            cache = _get_or_create_cache(run_id)
            layer_cache = _ensure_layer_cache(cache, module_name, layer_num)
            q_tokens = layer_cache["q"]
            k_all_tokens = layer_cache["k_all"]

            if self.hookq_mode == "all_tokens":
                q_tokens.extend(
                    [
                        input[0][bounds[i]:bounds[i + 1], :].detach().cpu()
                        for i in range(bs)
                    ]
                )
            elif self.hookq_mode == "last_token":
                q_tokens.extend(list(input[0][bounds[1:] - 1, :].detach().cpu()))
            else:
                raise NotImplementedError
            k_all_tokens.extend(
                [
                    input[1][bounds[i]:bounds[i + 1], :].detach().cpu()
                    for i in range(bs)
                ]
            )
            _save_cache(run_id, cache)

        def proj_hook(input, output, module_name, layer_num, proj_kind):
            """
            Capture projected Q or K tensors for attention modules whose wrapper
            no longer passes both tensors in the forward-hook input tuple.
            """

            run_id = _active_run_id()
            if run_id is None:
                _debug_once(f"skip_no_active_run_{proj_kind}", module_name)
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)
            if metadata is None:
                _debug_once(f"skip_no_attn_metadata_{proj_kind}", module_name)
                return None

            device = getattr(output, "device", None)
            if device is None:
                _debug_once(f"skip_no_output_device_{proj_kind}", module_name)
                return None

            bounds = _segment_bounds_from_metadata(
                metadata,
                module_name,
                device,
            )
            if bounds is None or bounds.numel() < 2:
                _debug_once(f"skip_no_segment_bounds_{proj_kind}", module_name)
                return None

            bs = bounds.numel() - 1
            cache = _get_or_create_cache(run_id)
            layer_cache = _ensure_layer_cache(cache, module_name, layer_num)

            if proj_kind == "q":
                if self.hookq_mode == "all_tokens":
                    layer_cache["q"].extend(
                        [
                            output[bounds[i]:bounds[i + 1], :].detach().cpu()
                            for i in range(bs)
                        ]
                    )
                elif self.hookq_mode == "last_token":
                    layer_cache["q"].extend(
                        list(output[bounds[1:] - 1, :].detach().cpu())
                    )
                else:
                    raise NotImplementedError
            elif proj_kind == "k":
                layer_cache["k_all"].extend(
                    [
                        output[bounds[i]:bounds[i + 1], :].detach().cpu()
                        for i in range(bs)
                    ]
                )
            else:
                raise NotImplementedError

            _save_cache(run_id, cache)

        # register hooks on attention modules 
        self._hooks = []
        matched = []
        for name, module in model.named_modules():
            layer_num = match_attn(name)
            if layer_num is None: # not an attention module 
                continue
            if layer_num not in self.important_layers:
                continue

            # Fallback for Granite-style modules whose self_attn wrapper computes
            # q/k internally instead of exposing them as the hook input tuple.
            if name.endswith(".self_attn") and hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                if self._debug_hook:
                    print(
                        f"[hook-debug] using proj fallback for module={name}",
                        flush=True,
                    )
                for proj_kind in ("q", "k"):
                    proj_module = getattr(module, f"{proj_kind}_proj", None)
                    if proj_module is None:
                        _debug_once(
                            f"missing_proj_module_{proj_kind}",
                            name,
                        )
                        continue
                    hook = proj_module.register_forward_hook(
                        lambda m, i, o, n=name, l=layer_num, p=proj_kind: proj_hook(i, o, n, l, p)
                    )
                    self._hooks.append(hook)
                    matched.append(PROJ_MODULE_NAME_TEMPLATE.format(attn_name=name, proj_kind=f"{proj_kind}_proj"))
                continue

            hook = module.register_forward_hook(
                lambda m, i, o, n=name: qkv_hook(i, n)
            )
            self._hooks.append(hook)
            matched.append(name)
        
        print(f"Installed {len(self._hooks)} hooks on layers: {matched}")

    def _parse_layer_heads(self) -> Dict[int, List[int]]:
        """
        Parse `VLLM_HOOK_LAYER_HEADS` into a layer-to-head mapping.

        Args:
            None.

        Returns:
            Dict[int, List[int]]: Mapping from layer index to sorted head indices.
        """
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

    def _uninstall_hooks(self):
        """
        Remove any registered forward hooks from the model.

        Args:
            None.

        Returns:
            None: Stored hooks are removed and the hook list is cleared.
        """
        for hook in getattr(self, "_hooks", []):
            try:
                hook.remove()
            except Exception:
                pass
        if hasattr(self, "_hooks"):
            self._hooks.clear()

    def execute_model(self, *args, **kwargs):
        """
        Delegate model execution to the base vLLM worker.

        Args:
            *args: Positional arguments forwarded to the base worker.
            **kwargs: Keyword arguments forwarded to the base worker.

        Returns:
            Any: Result returned by the base worker ``execute_model`` call.
        """
        return super().execute_model(*args, **kwargs)
