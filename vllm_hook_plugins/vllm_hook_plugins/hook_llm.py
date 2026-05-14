import copy
import inspect
import os
import json
import uuid
from typing import Optional, Dict, List
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from vllm import LLM, SamplingParams
from vllm_hook_plugins.registry import PluginRegistry

class HookLLM:
    def __init__(
        self,
        model: str,
        worker_name: str = None,
        analyzer_name: str = None,
        config_file: str = None,
        download_dir: str = '~/.cache',
        enable_hook: bool = True,
        hook_dir: str = None,
        enforce_eager: bool = True,
        **vllm_kwargs
    ):

        self.model_name = model
        self.worker_name = worker_name
        self.analyzer_name = analyzer_name
        self.enable_hook = enable_hook
        self.enforce_eager = enforce_eager

        if hook_dir is not None:
            HOOK_DIR = hook_dir
        else:
            HOOK_DIR = os.path.join(download_dir, '_v1_qk_peeks')
        os.makedirs(HOOK_DIR, exist_ok=True)
        self._hook_dir = HOOK_DIR

        os.environ["VLLM_HOOK_DIR"] = os.path.abspath(self._hook_dir)

        self.layer_to_heads = {}
        self._output_layers = None   # set by load_config for HS worker
        if config_file:
            self.load_config(config_file)

        # (Optional) pre-allocate shared memory before worker subprocess is spawned.
        self._hook_shm = None
        if os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            from vllm_hook_plugins.shm_utils import setup_shm
            self._hook_shm = setup_shm(config_file, worker_name)

        worker = None
        if worker_name:
            import vllm.plugins
            vllm.plugins.load_general_plugins()
            worker = PluginRegistry.get_worker(worker_name).path

        self.llm = LLM(
            model=model,
            download_dir=download_dir,
            worker_extension_cls=worker,
            enforce_eager=enforce_eager,
            **vllm_kwargs
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.llm_engine = self.llm.llm_engine

        self.analyzer = None
        if analyzer_name:
            self.analyzer = PluginRegistry.get_analyzer(analyzer_name).analyzer
            self.analyzer = self.analyzer(self._hook_dir, self.layer_to_heads)


    def load_config(self, config_file: str):
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        if "params" in config_data and "important_heads" in config_data["params"]:
            self.important_heads = config_data["params"]["important_heads"]
            self.layer_to_heads = {}
            for layer_idx, head_idx in self.important_heads:
                if layer_idx not in self.layer_to_heads:
                    self.layer_to_heads[layer_idx] = []
                self.layer_to_heads[layer_idx].append(head_idx)

            layer_to_heads_string = ";".join([
                f"{layer}:{','.join(map(str, heads))}"
                for layer, heads in sorted(self.layer_to_heads.items())
            ])
            os.environ["VLLM_HOOK_LAYER_HEADS"] = layer_to_heads_string

        if "hookq" in config_data:
            hookq_mode = config_data["hookq"]["hookq_mode"]
            os.environ["VLLM_HOOKQ_MODE"] = hookq_mode

        if "steering" in config_data:
            os.environ["VLLM_ACTSTEER_CONFIG"] = os.path.abspath(config_file)

        if "hidden_states" in config_data:
            hs_cfg = config_data["hidden_states"]
            layers = hs_cfg.get("layers", [])
            os.environ["VLLM_HOOK_LAYERS"] = ";".join(map(str, layers))
            mode = hs_cfg.get("mode", "last_token")
            os.environ["VLLM_HOOK_HS_MODE"] = mode
            self._output_layers = layers if layers else True

    def _build_extra_args(self, save_to_disk: bool, run_id: str) -> dict:
        """Build extra_args for the probe worker based on worker_name and config."""
        extra = {}
        if self.worker_name == "probe_hidden_states":
            extra["output_hidden_states"] = self._output_layers if self._output_layers else True
        elif self.worker_name == "probe_hook_qk":
            # Pass layer_to_heads dict for head-level metadata; worker uses keys for layer filtering.
            extra["output_qk"] = self.layer_to_heads if self.layer_to_heads else True
        elif self.worker_name == "steer_hook_act":
            extra["steer"] = True

        if save_to_disk:
            extra["save_to_disk"] = True
            extra["run_id"] = run_id
            extra["hook_dir"] = self._hook_dir

        return extra

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_hook: Optional[bool] = None,
        save_to_disk: bool = False,
        run_id: Optional[str] = None,
        **kwargs
    ):
        hook = use_hook if use_hook is not None else self.enable_hook

        if not isinstance(prompts, list):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)

        if hook and self.worker_name:
            if run_id is None:
                run_id = str(uuid.uuid4())
            sampling_params = copy.copy(sampling_params)
            extra = dict(sampling_params.extra_args or {})
            extra.update(self._build_extra_args(save_to_disk, run_id))
            sampling_params.extra_args = extra
            # Store last run_id so analyze() can find the artifact without
            # the caller needing to track it.
            self._last_run_id = run_id

        outputs = self.llm.generate(prompts, sampling_params)

        if hook and self.worker_name and not save_to_disk and len(outputs) > 1 and getattr(outputs[0], "probes", None) is not None:
            # Merge per-request probes onto outputs[0] so callers always use
            # output[0].probes regardless of batch size. q/k_all become lists
            # of per-request tensors (unbind dim 0), matching the disk format
            # that unpack_qk expects. k_all varies in seq_len so can't be cat'd.
            all_probes = [o.probes for o in outputs]
            merged = {k: v for k, v in all_probes[0].items() if k not in ("qk_cache", "hs_cache")}
            for cache_key in ("qk_cache", "hs_cache"):
                if cache_key not in all_probes[0]:
                    continue
                merged[cache_key] = {}
                for layer in all_probes[0][cache_key]:
                    first = all_probes[0][cache_key][layer]
                    entry = {k: v for k, v in first.items() if k not in ("q", "k_all", "hs")}
                    for tensor_key in ("q", "k_all", "hs"):
                        if tensor_key not in first:
                            continue
                        entry[tensor_key] = [p[cache_key][layer][tensor_key][0] for p in all_probes]
                    merged[cache_key][layer] = entry
            outputs[0].probes = merged

        return outputs

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        probes: Optional[Dict] = None,
        run_id: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Run the configured analyzer.

        Two paths:
        - In-memory: pass ``probes=output.probes`` from a generate() call that
          used save_to_disk=False. The analyzer receives the RPC artifacts directly.
        - Disk: pass ``run_id`` for single-pass analyzers (HS, AttnTracker), or
          ``run_ids=[doc_run_id, na_run_id]`` for CoRer. Omitting both falls back
          to the last generate()'s run_id.
        """
        if self.analyzer is None:
            print("No analyzer configured")
            return None

        if probes is not None:
            return self.analyzer.analyze(analyzer_spec=analyzer_spec, probes=probes)

        # Disk path: resolve run_id from args or _last_run_id fallback.
        effective_run_id = run_id or getattr(self, "_last_run_id", None)
        effective_run_ids = run_ids

        sig = inspect.signature(self.analyzer.analyze)
        kwargs = {"analyzer_spec": analyzer_spec}
        if "run_ids" in sig.parameters and effective_run_ids is not None:
            kwargs["run_ids"] = effective_run_ids
        elif "run_id" in sig.parameters and effective_run_id is not None:
            kwargs["run_id"] = effective_run_id

        return self.analyzer.analyze(**kwargs)

    def __del__(self):
        from vllm_hook_plugins.shm_utils import teardown_shm
        teardown_shm(getattr(self, "_hook_shm", None))
