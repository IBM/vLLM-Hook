import os
import re
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
        tp_rank = int(ps.get_tensor_model_parallel_rank())

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

        self.target_layers = self._parse_target_layers()

        self._run_cache = {}

        cfg = model.config
        hidden_size = int(getattr(cfg, "hidden_size"))
        num_layers = int(getattr(cfg, "num_hidden_layers", 0))
        self._conf = {"hidden_size": hidden_size, "num_layers": num_layers}

        def hs_hook(output, module_name, layer_num):
            if not os.path.exists(self.hook_flag):
                return None

            if os.path.exists(self.run_id_file):
                run_id = open(self.run_id_file).read().strip().split("\n")[-1]
            else:
                raise Exception("run_id not found.")

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

            if self.hs_mode == "last_token":
                batch_hs.extend(
                    [
                        hidden[last_indices[i + 1] - 1].detach().cpu()
                        for i in range(bs)
                    ]
                )
            elif self.hs_mode == "all_tokens":
                batch_hs.extend(
                    [
                        hidden[last_indices[i] : last_indices[i + 1]].detach().cpu()
                        for i in range(bs)
                    ]
                )
            else:
                raise NotImplementedError(f"Unknown hs_mode: {self.hs_mode}")

            cache["hs_cache"][module_name] = {
                "hidden_states": batch_hs,
                "layer_num": layer_num,
            }

            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            torch.save(cache, os.path.join(run_dir, "hidden_states.pt"))

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

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)
