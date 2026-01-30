import os
import math
import torch
from typing import Dict, List
from vllm.v1.worker.gpu_worker import Worker as V1Worker
from vllm.forward_context import get_forward_context
import re
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

        def qkv_hook(input, module_name):

            if not os.path.exists(self.hook_flag): # hooks deactivated
                return None

            elif os.path.exists(self.run_id_file): # hooks activated
                run_id = (open(self.run_id_file).read().strip().split('\n'))[-1]

            else:
                raise Exception("run_id not found.")

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)

            # Warmup or non-attention passes: nothing to do
            if metadata is None:
                return
        
            # seq_lens = metadata.seq_lens
            seq_lens = getattr(metadata, "seq_lens", None)
            if seq_lens is None and module_name in metadata:
                seq_lens = metadata[module_name].seq_lens
            
            # assert (seq_lens).sum() == metadata.num_actual_tokens, "Please set enable_prefix_caching=False for batch processing."
            last_indices = torch.cumsum(seq_lens, dim=0)
            bs = len(last_indices)
            last_indices = torch.cat([torch.tensor([0]).to(last_indices.device), last_indices])

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
            if self.hookq_mode == "all_tokens":
                q_tokens.extend([input[0][last_indices[i]:last_indices[i+1],:].detach().cpu() for i in range(bs)])
            elif self.hookq_mode == "last_token":
                q_tokens.extend(list(input[0][last_indices[1:] - 1,:].detach().cpu()))
            else:
                raise NotImplementedError
            k_all_tokens.extend([input[1][last_indices[i]:last_indices[i+1],:].detach().cpu() for i in range(bs)])     

            cache["qk_cache"][module_name] = {
                'q': q_tokens,
                'k_all': k_all_tokens,
                'layer_num': layer_num
            }

            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qk.pt")
            torch.save(cache, cache_path)

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
                lambda m, i, o, n=name: qkv_hook(i, n)
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

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)