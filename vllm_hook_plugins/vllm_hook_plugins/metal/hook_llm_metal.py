import glob
import json
import os
import tempfile
import uuid
from typing import Dict, List, Optional

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from vllm import LLM, SamplingParams


class HookLLMMetal:
    def __init__(
        self,
        model: str,
        worker_name: str = None,
        analyzer_name: str = None,
        config_file: str = None,
        download_dir: str = "~/.cache",
        enable_hook: bool = True,
        hook_dir: str = None,
        enforce_eager: bool = True,
        **vllm_kwargs,
    ):
        self.model_name = model
        self.worker_name = worker_name
        self.analyzer_name = analyzer_name
        self.enable_hook = enable_hook
        self.enforce_eager = enforce_eager
        self._vllm_kwargs = dict(vllm_kwargs)
        self._last_generate_used_hooks = False

        self._hook_dir = self._resolve_hook_dir(download_dir, hook_dir)
        self._hook_flag = os.path.join(self._hook_dir, "EXTRACT.flag")
        self._run_id_file = os.path.join(self._hook_dir, "RUN_ID.txt")

        os.environ["VLLM_HOOK_DIR"] = os.path.abspath(self._hook_dir)
        os.environ["VLLM_HOOK_FLAG"] = os.path.abspath(self._hook_flag)
        os.environ["VLLM_RUN_ID"] = os.path.abspath(self._run_id_file)
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

        self.layer_to_heads = {}
        if config_file:
            self.load_config(config_file)

        self._hook_shm = None
        if os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            from vllm_hook_plugins.shm_utils import setup_shm

            self._hook_shm = setup_shm(config_file, worker_name)

        self._resolved_download_dir = download_dir
        self._worker_cls_path = self._resolve_worker_class_path(worker_name)

        print(
            f"HookLLMMetal worker={self.worker_name or 'none'} "
            f"hooks_enabled={self.enable_hook}",
            flush=True,
        )
        self.llm = self._build_llm(use_hook_worker=False)

        self.tokenizer = self.llm.get_tokenizer()
        self.llm_engine = self.llm.llm_engine

        self.analyzer = None
        if analyzer_name:
            analyzer_cls = self._resolve_analyzer_class(analyzer_name)
            self.analyzer = analyzer_cls(self._hook_dir, self.layer_to_heads)

    @staticmethod
    def _resolve_hook_dir(download_dir: str, hook_dir: Optional[str]) -> str:
        candidates = []
        if hook_dir is not None:
            candidates.append(os.path.expanduser(hook_dir))
        else:
            candidates.append(
                os.path.join(os.path.expanduser(download_dir), "_v1_qk_peeks")
            )

        candidates.append(os.path.join(os.path.expanduser("~/.cache"), "_v1_qk_peeks"))
        candidates.append(os.path.join(tempfile.gettempdir(), "_v1_qk_peeks"))

        seen = set()
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                os.makedirs(candidate, exist_ok=True)
                return candidate
            except OSError:
                continue

        raise OSError(
            "Could not create a writable hook cache directory. "
            f"Tried: {', '.join(seen)}"
        )

    @staticmethod
    def _resolve_worker_class_path(worker_name: Optional[str]) -> Optional[str]:
        if not worker_name:
            return None

        if worker_name in {"probe_hook_qk", "probe_hook_qk_metal"}:
            from vllm_hook_plugins.workers.metal import ProbeHookQKWorkerMetal

            return f"{ProbeHookQKWorkerMetal.__module__}.{ProbeHookQKWorkerMetal.__name__}"

        if worker_name in {"steer_hook_act", "steer_hook_act_metal"}:
            from vllm_hook_plugins.workers.metal import SteerHookActWorkerMetal

            return f"{SteerHookActWorkerMetal.__module__}.{SteerHookActWorkerMetal.__name__}"

        raise ValueError(
            f"Unsupported Metal worker '{worker_name}'. "
            "Use 'probe_hook_qk' or 'steer_hook_act'."
        )

    @staticmethod
    def _resolve_analyzer_class(analyzer_name: str):
        if analyzer_name in {"attn_tracker", "attention_tracker"}:
            from vllm_hook_plugins.analyzers.metal import AttntrackerAnalyzerMetal

            return AttntrackerAnalyzerMetal

        if analyzer_name == "core_reranker":
            from vllm_hook_plugins.analyzers.metal import CorerAnalyzerMetal

            return CorerAnalyzerMetal

        from vllm_hook_plugins.registry import PluginRegistry

        analyzer = PluginRegistry.get_analyzer(analyzer_name)
        if analyzer is None:
            raise ValueError(f"Unknown analyzer '{analyzer_name}'.")
        return analyzer.analyzer

    def _build_llm(self, use_hook_worker: bool) -> LLM:
        llm_kwargs = dict(
            model=self.model_name,
            download_dir=self._resolved_download_dir,
            enforce_eager=self.enforce_eager,
            **self._vllm_kwargs,
        )
        if use_hook_worker and self._worker_cls_path is not None:
            llm_kwargs["worker_cls"] = self._worker_cls_path
        return LLM(**llm_kwargs)

    def _dispose_llm(self, llm: Optional[LLM]) -> None:
        if llm is None:
            return
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "collective_rpc"):
            try:
                engine.collective_rpc("_uninstall_hooks")
            except Exception:
                pass
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()

    def load_config(self, config_file: str):
        with open(config_file, "r") as f:
            config_data = json.load(f)

        if "params" in config_data and "important_heads" in config_data["params"]:
            self.important_heads = config_data["params"]["important_heads"]
            self.layer_to_heads = {}
            for layer_idx, head_idx in self.important_heads:
                if layer_idx not in self.layer_to_heads:
                    self.layer_to_heads[layer_idx] = []
                self.layer_to_heads[layer_idx].append(head_idx)

            layer_to_heads_string = ";".join(
                [
                    f"{layer}:{','.join(map(str, heads))}"
                    for layer, heads in sorted(self.layer_to_heads.items())
                ]
            )
            os.environ["VLLM_HOOK_LAYER_HEADS"] = layer_to_heads_string

        if "hookq" in config_data:
            os.environ["VLLM_HOOKQ_MODE"] = config_data["hookq"]["hookq_mode"]

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
        **kwargs,
    ):
        hook = use_hook if use_hook is not None else self.enable_hook

        if not isinstance(prompts, list):
            prompts = [prompts]

        if not hook or not self.worker_name:
            self._last_generate_used_hooks = False
            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            return self.llm.generate(prompts, sampling_params)

        self._last_generate_used_hooks = True
        if "probe" in self.worker_name:
            return self.generate_with_encode_hook(
                prompts, sampling_params, cleanup, **kwargs
            )
        if "steer" in self.worker_name:
            return self.generate_with_decode_hook(
                prompts, sampling_params, cleanup, **kwargs
            )
        raise ValueError(f"Unsupported Metal worker '{self.worker_name}'.")

    def generate_with_encode_hook(self, prompts, sampling_params, cleanup, **kwargs):
        hook_llm = None
        try:
            self._setup_hooks(cleanup)
            hook_llm = self._build_llm(use_hook_worker=True)
            prefill_params = SamplingParams(temperature=0.1, max_tokens=1)
            hook_llm.generate(prompts, prefill_params)
            self._assert_hook_artifacts_exist()
        finally:
            self._cleanup_hooks()
            self._dispose_llm(hook_llm)

        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, sampling_params)

    def generate_with_decode_hook(self, prompts, sampling_params, cleanup, **kwargs):
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

    def analyze(self, analyzer_spec: Optional[Dict] = None) -> Optional[Dict]:
        if self.analyzer is None:
            print("No analyzer configured")
            return None
        if not self._last_generate_used_hooks:
            raise RuntimeError(
                "No hook artifacts are available for analysis. "
                "Rerun generate with Metal hooks enabled."
            )
        return self.analyzer.analyze(analyzer_spec)

    def __del__(self):
        try:
            from vllm_hook_plugins.shm_utils import teardown_shm

            teardown_shm(getattr(self, "_hook_shm", None))
        except Exception:
            pass

    def _setup_hooks(self, cleanup):
        if cleanup:
            for path in glob.glob(
                os.path.join(self._hook_dir, "**", "qk.pt"), recursive=True
            ):
                os.remove(path)
            for path in glob.glob(
                os.path.join(self._hook_dir, "**", "qkv.pt"), recursive=True
            ):
                os.remove(path)
            if os.path.exists(self._run_id_file):
                os.remove(self._run_id_file)

        run_id = str(uuid.uuid4())
        with open(self._run_id_file, "a") as f:
            f.write(run_id + "\n")

        open(self._hook_flag, "a").close()

    def _cleanup_hooks(self):
        if os.path.exists(self._hook_flag):
            os.remove(self._hook_flag)

    def _assert_hook_artifacts_exist(self) -> None:
        run_ids = []
        if os.path.exists(self._run_id_file):
            with open(self._run_id_file, "r") as f:
                run_ids = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        run_id = run_ids[-1] if run_ids else None
        if not run_id:
            raise RuntimeError("Hook run completed without a recorded run ID.")

        qk_paths = glob.glob(
            os.path.join(self._hook_dir, run_id, "**", "qk.pt"),
            recursive=True,
        )
        qkv_paths = glob.glob(
            os.path.join(self._hook_dir, run_id, "**", "qkv.pt"),
            recursive=True,
        )
        if qk_paths or qkv_paths:
            return

        raise RuntimeError(
            "Hooked generation completed but produced no qk/qkv artifact for "
            f"run_id={run_id} under {self._hook_dir}."
        )
