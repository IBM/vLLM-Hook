import os
import json
import glob
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
            HOOK_DIR = os.path.join(download_dir,'_v1_qk_peeks')
        os.makedirs(HOOK_DIR, exist_ok=True)
        self._hook_dir = HOOK_DIR
        self._hook_flag = os.path.join(self._hook_dir, "EXTRACT.flag")
        self._run_id_file = os.path.join(self._hook_dir, "RUN_ID.txt")
        
        os.environ["VLLM_HOOK_DIR"] = os.path.abspath(self._hook_dir)
        os.environ["VLLM_HOOK_FLAG"] = os.path.abspath(self._hook_flag)
        os.environ["VLLM_RUN_ID"] = os.path.abspath(self._run_id_file)
        
        self.layer_to_heads = {}
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
            worker_cls=worker,
            enforce_eager = enforce_eager,
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
            # self.important_heads = [[i, j] for i in range(32) for j in range(32)]
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

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_hook: Optional[bool] = None,
        cleanup: Optional[bool] = True,
        **kwargs
    ):
        hook = use_hook if use_hook is not None else self.enable_hook
        
        if not isinstance(prompts, list):
            prompts = [prompts]

        if not hook or not self.worker_name:
            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            return self.llm.generate(prompts, sampling_params)
        
        worker_entry = PluginRegistry.get_worker(self.worker_name)
        hooks_on_prefill, hooks_on_generate = worker_entry.hooks_on if worker_entry else (False, False)

        return self._generate_with_hooks(
            prompts, sampling_params, cleanup,
            hooks_on_prefill=hooks_on_prefill,
            hooks_on_generate=hooks_on_generate,
            **kwargs
        )

    def _generate_with_hooks(self, prompts, sampling_params, cleanup,
                              hooks_on_prefill: bool, hooks_on_generate: bool, **kwargs):
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)

        if hooks_on_prefill and hooks_on_generate:
            # Single-pass: hooks active throughout
            self._setup_hooks(cleanup)
            try:
                return self.llm.generate(prompts, sampling_params)
            finally:
                self._cleanup_hooks()
        else:
            # Two-pass: prefill (max_tokens=1) then full generation
            prefill_params = SamplingParams(temperature=0.1, max_tokens=1)

            if hooks_on_prefill:
                self._setup_hooks(cleanup)
            self.llm.generate(prompts, prefill_params)
            if hooks_on_prefill:
                self._cleanup_hooks()

            if hooks_on_generate:
                self._setup_hooks(cleanup)
            output = self.llm.generate(prompts, sampling_params)
            if hooks_on_generate:
                self._cleanup_hooks()

            return output

    ####### depreciated ####### 
    def generate_with_encode_hook(self, prompts, sampling_params, cleanup, **kwargs):
        return self._generate_with_hooks(prompts, sampling_params, cleanup,
                                          hooks_on_prefill=True, hooks_on_generate=False, **kwargs)
    ####### depreciated ####### 
    def generate_with_decode_hook(self, prompts, sampling_params, cleanup, **kwargs):
        return self._generate_with_hooks(prompts, sampling_params, cleanup,
                                          hooks_on_prefill=False, hooks_on_generate=True, **kwargs)
    
    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None
    ) -> Optional[Dict]:

        if self.analyzer is None:
            print("No analyzer configured")
            return None
        
        return self.analyzer.analyze(analyzer_spec)
    
    def __del__(self):
        from vllm_hook_plugins.shm_utils import teardown_shm
        teardown_shm(getattr(self, "_hook_shm", None))

    def _setup_hooks(self, cleanup):
        if cleanup:
            for ext in ("*.pt", "*.safetensors", "*.json"):
                for p in glob.glob(
                    os.path.join(self._hook_dir, "**", ext), recursive=True
                ):
                    os.remove(p)
            if os.path.exists(self._run_id_file):
                os.remove(self._run_id_file)

        run_id = str(uuid.uuid4())
        with open(self._run_id_file, "a") as f:
            f.write(run_id+ "\n")

        open(self._hook_flag, "a").close()

    def _cleanup_hooks(self):
        if os.path.exists(self._hook_flag):
            os.remove(self._hook_flag)
    