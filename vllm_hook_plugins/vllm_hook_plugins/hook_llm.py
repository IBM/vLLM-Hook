import os
import json
import glob
import uuid
import platform
from importlib.util import find_spec
from typing import Optional, Dict, List
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from vllm import LLM, SamplingParams

class HookLLM:
    def __init__(
        self,
        model: str,
        worker_name: str = None,
        backend: Optional[str] = None,
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
        self.backend = self._resolve_backend(backend, vllm_kwargs)
        self._vllm_kwargs = dict(vllm_kwargs)
        self._plugin_registry = None
        self._last_generate_used_hooks = False

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

        if self.backend == "metal":
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        

        if worker_name:
            import vllm.plugins
            from vllm_hook_plugins import PluginRegistry
            vllm.plugins.load_general_plugins()
            self._plugin_registry = PluginRegistry
            self.worker_name = self._resolve_worker_name(
                PluginRegistry, worker_name, self.backend
            )
        self._resolved_download_dir = download_dir
        self.llm = self._build_llm(use_hook_worker=False)
            
        self.tokenizer = self.llm.get_tokenizer()
        self.llm_engine = self.llm.llm_engine

        self.analyzer = None
        if analyzer_name:
            if self._plugin_registry is None:
                import vllm.plugins
                from vllm_hook_plugins import PluginRegistry
                vllm.plugins.load_general_plugins()
                self._plugin_registry = PluginRegistry
            self.analyzer = self._plugin_registry.get_analyzer(analyzer_name).analyzer
            self.analyzer = self.analyzer(self._hook_dir, self.layer_to_heads)

    @staticmethod
    def _resolve_backend(
        backend: Optional[str], vllm_kwargs: Dict
    ) -> str:
        if backend:
            return backend.lower()

        device = vllm_kwargs.get("device")
        if isinstance(device, str) and device:
            return device.lower()

        env_backend = os.environ.get("VLLM_HOOK_BACKEND")
        if env_backend:
            return env_backend.lower()

        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and find_spec("vllm_metal") is not None
        ):
            return "metal"

        return "default"

    @staticmethod
    def _resolve_worker_name(
        PluginRegistry, worker_name: str, backend: str
    ) -> str:
        candidates = []
        if backend not in {"", "default", "auto", None}:
            candidates.append(f"{worker_name}_{backend}")
        candidates.append(worker_name)

        for candidate in candidates:
            if PluginRegistry.get_worker(candidate) is not None:
                return candidate

        available = ", ".join(sorted(PluginRegistry.list_workers()))
        raise ValueError(
            f"No worker registered for '{worker_name}' with backend '{backend}'. "
            f"Tried: {', '.join(candidates)}. Available workers: {available}"
        )

    def _should_use_hook_worker(self) -> bool:
        if not self.worker_name:
            return False
        if self.backend == "metal":
            return os.environ.get("VLLM_DISABLE_METAL_HOOKS", "0") != "1"
        return True

    def _build_llm(self, use_hook_worker: bool) -> LLM:
        worker = None
        if use_hook_worker and self._should_use_hook_worker():
            worker_entry = self._plugin_registry.get_worker(self.worker_name)
            if worker_entry is None:
                available = ", ".join(sorted(self._plugin_registry.list_workers()))
                raise ValueError(
                    f"Worker '{self.worker_name}' is not registered. "
                    f"Available workers: {available}"
                )
            worker = worker_entry.path

        llm_kwargs = dict(
            model=self.model_name,
            download_dir=self._resolved_download_dir,
            enforce_eager=self.enforce_eager,
            **self._vllm_kwargs,
        )
        if worker is not None:
            llm_kwargs["worker_cls"] = worker

        return LLM(**llm_kwargs)

    def _dispose_llm(self, llm: Optional[LLM]) -> None:
        if llm is None:
            return
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()

    
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

        if hook and not self._should_use_hook_worker():
            self._last_generate_used_hooks = False
            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            return self.llm.generate(prompts, sampling_params)

        if hook:
            self._last_generate_used_hooks = True
            if self.worker_name and "probe" in self.worker_name:
                return self.generate_with_encode_hook(prompts, sampling_params, cleanup, **kwargs)
            elif self.worker_name and "steer" in self.worker_name:
                return self.generate_with_decode_hook(prompts, sampling_params, cleanup, **kwargs)

        else:
            self._last_generate_used_hooks = False
            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            return self.llm.generate(prompts, sampling_params)
    
    def generate_with_encode_hook(self, prompts, sampling_params, cleanup, **kwargs):
        hook_llm = None
        try:
            self._setup_hooks(cleanup)

            # On Metal, keep the hooked prefill isolated from the normal engine.
            hook_llm = self._build_llm(use_hook_worker=True)
            prefill_params = SamplingParams(temperature=0.1, max_tokens=1)
            hook_llm.generate(prompts, prefill_params)
        finally:
            self._cleanup_hooks()
            self._dispose_llm(hook_llm)

        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, sampling_params)
    
    def generate_with_decode_hook(self, prompts, sampling_params, cleanup, **kwargs):
        
        # prefill without hooks
        prefill_params = SamplingParams(temperature=0.1, max_tokens=1)
        self.llm.generate(prompts, prefill_params)

        hook_llm = None
        try:
            self._setup_hooks(cleanup)
            hook_llm = self._build_llm(use_hook_worker=True)

            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            output = hook_llm.generate(prompts, sampling_params)
        finally:
            self._cleanup_hooks()
            self._dispose_llm(hook_llm)

        return output
    
    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None
    ) -> Optional[Dict]:

        if self.analyzer is None:
            print("No analyzer configured")
            return None
        if not self._last_generate_used_hooks:
            raise RuntimeError(
                "No hook artifacts are available for analysis. "
                "Hook workers were disabled for the last generate call; "
                "unset VLLM_DISABLE_METAL_HOOKS or rerun with hooks enabled."
            )
        
        return self.analyzer.analyze(analyzer_spec)
    
    
    def _setup_hooks(self, cleanup):
        if cleanup:
            for p in glob.glob(os.path.join(self._hook_dir, "**", "qk.pt"), recursive=True):
                os.remove(p)
                print("Cleaned up previous qk cache.")
            for p in glob.glob(os.path.join(self._hook_dir, "**", "qkv.pt"), recursive=True):
                os.remove(p)
                print("Cleaned up previous qkv cache.")
            if os.path.exists(self._run_id_file):
                os.remove(self._run_id_file)

        run_id = str(uuid.uuid4())
        with open(self._run_id_file, "a") as f:
            f.write(run_id+ "\n")
            print("Logged run ID.")

        open(self._hook_flag, "a").close()
        print("Created hook flag.")
        

    def _cleanup_hooks(self):
        if os.path.exists(self._hook_flag):
            os.remove(self._hook_flag)
            print("Hooks deactivated.")
        else:
            print("No hooks to be deactivated.")
    
