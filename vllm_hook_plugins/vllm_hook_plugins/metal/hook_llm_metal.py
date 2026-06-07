import glob
import copy
import gc
import json
import os
import tempfile
import uuid
from typing import Dict, List, Optional

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from vllm import LLM, SamplingParams
from vllm_hook_plugins.run_utils import dispatch_disk_analyze


class _LazyBaseEngineProxy:
    def __init__(self, owner: "HookLLMMetal"):
        self._owner = owner

    def __getattr__(self, name):
        engine = self._owner._ensure_base_llm().llm_engine
        return getattr(engine, name)


class HookLLMMetal:
    MEMORY_GUARD_MIN_AVAILABLE_MB = int(
        float(os.environ.get("VLLM_HOOK_MIN_AVAILABLE_GB", "4")) * 1024
    )
    MEMORY_GUARD_MAX_RSS_MB = int(
        float(os.environ.get("VLLM_HOOK_MAX_RSS_GB", "24")) * 1024
    )
    @staticmethod
    def _format_mb(value_mb: float | None) -> str:
        if value_mb is None:
            return "n/a"
        return f"{value_mb / 1024:.2f} GB"

    def _memory_snapshot(self) -> Dict[str, float | None]:
        rss_mb = None
        available_mb = None
        total_mb = None

        try:
            import psutil

            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)
            total_mb = vm.total / (1024 * 1024)
        except Exception:
            try:
                import resource

                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_mb = float(rss_kb) / 1024.0
            except Exception:
                pass

        return {
            "rss_mb": rss_mb,
            "available_mb": available_mb,
            "total_mb": total_mb,
        }

    def _log_memory(self, stage: str) -> None:
        snap = self._memory_snapshot()
        print(
            f"Metal steer: {stage} "
            f"rss={self._format_mb(snap['rss_mb'])} "
            f"available={self._format_mb(snap['available_mb'])} "
            f"total={self._format_mb(snap['total_mb'])}",
            flush=True,
        )

    def _ensure_memory_headroom(self, stage: str) -> None:
        snap = self._memory_snapshot()
        rss_mb = snap["rss_mb"]
        available_mb = snap["available_mb"]
        if available_mb is not None and available_mb < self.MEMORY_GUARD_MIN_AVAILABLE_MB:
            raise MemoryError(
                f"Metal memory guard triggered at {stage}: "
                f"available={self._format_mb(available_mb)} "
                f"threshold={self._format_mb(self.MEMORY_GUARD_MIN_AVAILABLE_MB)}"
            )
        if rss_mb is not None and rss_mb > self.MEMORY_GUARD_MAX_RSS_MB:
            raise MemoryError(
                f"Metal memory guard triggered at {stage}: "
                f"rss={self._format_mb(rss_mb)} "
                f"threshold={self._format_mb(self.MEMORY_GUARD_MAX_RSS_MB)}"
            )

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

        os.environ["HOOK_DIR"] = os.path.abspath(self._hook_dir)
        os.environ["HOOK_FLAG"] = os.path.abspath(self._hook_flag)
        os.environ["HOOK_RUN_ID"] = os.path.abspath(self._run_id_file)
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

        self.layer_to_heads = {}
        self._output_layers = None
        self._hs_mode = "last_token"
        if config_file:
            self.load_config(config_file)

        self._hook_shm = None
        if os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            from vllm_hook_plugins.shm_utils import setup_shm

            self._hook_shm = setup_shm(config_file, worker_name)

        self._resolved_download_dir = download_dir
        self._last_run_id: Optional[str] = None
        self._last_run_ids: List[str] = []
        self._set_hook_worker_env(worker_name)
        os.environ.pop("VLLM_METAL_MIN_KV_BLOCKS", None)
        os.environ.pop("VLLM_METAL_RESERVED_KV_GB", None)
        os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "auto")
        if self._resolve_hook_worker_env(worker_name) == "qk":
            os.environ.setdefault("VLLM_HOOK_RECLAIM_BASE_FOR_ENCODE", "1")
        self._register_metal_plugin()

        self._ensure_memory_headroom("HookLLMMetal initialization before engine build")
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
    def _resolve_hook_worker_env(worker_name: Optional[str]) -> str:
        if worker_name in {"probe_hook_qk", "probe_hook_qk_metal"}:
            return "qk"
        if worker_name in {"steer_hook_act", "steer_hook_act_metal"}:
            return "steer"
        if worker_name in {"probe_hidden_states", "probe_hidden_states_metal"}:
            return "hidden_states"
        return "hidden_states"

    def _set_hook_worker_env(self, worker_name: Optional[str]) -> None:
        worker_env = self._resolve_hook_worker_env(worker_name)
        os.environ["HOOK_WORKER"] = worker_env
        os.environ["VLLM_HOOK_WORKER"] = worker_env

    def _register_metal_plugin(self) -> None:
        from vllm_hook_plugins.metal._hook_plugin import register as register_metal_plugin

        register_metal_plugin()

    @staticmethod
    def _resolve_worker_class_path(worker_name: Optional[str]) -> Optional[str]:
        worker_env = HookLLMMetal._resolve_hook_worker_env(worker_name)
        if worker_env == "qk":
            return (
                "vllm_hook_plugins.workers.metal.probe_hookqk_worker_metal."
                "ProbeHookQKWorkerMetal"
            )
        if worker_env == "steer":
            return (
                "vllm_hook_plugins.workers.metal.steer_activation_worker_metal."
                "SteerHookActWorkerMetal"
            )
        if worker_env == "hidden_states":
            return (
                "vllm_hook_plugins.workers.metal.probe_hidden_states_worker_metal."
                "ProbeHiddenStatesWorkerMetal"
            )
        return None

    @staticmethod
    def _resolve_analyzer_class(analyzer_name: str):
        if analyzer_name in {"attn_tracker", "attention_tracker"}:
            from vllm_hook_plugins.analyzers.metal import AttntrackerAnalyzerMetal

            return AttntrackerAnalyzerMetal

        if analyzer_name == "core_reranker":
            from vllm_hook_plugins.analyzers.metal import CorerAnalyzerMetal

            return CorerAnalyzerMetal

        if analyzer_name in {"hidden_states", "hidden_states_metal"}:
            from vllm_hook_plugins.analyzers.metal import HiddenStatesAnalyzerMetal

            return HiddenStatesAnalyzerMetal

        from vllm_hook_plugins.registry import PluginRegistry

        analyzer = PluginRegistry.get_analyzer(analyzer_name)
        if analyzer is None:
            raise ValueError(f"Unknown analyzer '{analyzer_name}'.")
        return analyzer.analyzer

    def _build_llm(self, use_hook_worker: bool) -> LLM:
        vllm_kwargs = dict(self._vllm_kwargs)
        if (
            "max_model_len" in vllm_kwargs
            and "max_num_batched_tokens" not in vllm_kwargs
        ):
            max_num_seqs = int(vllm_kwargs.get("max_num_seqs", 1))
            vllm_kwargs["max_num_batched_tokens"] = (
                int(vllm_kwargs["max_model_len"]) * max_num_seqs
            )
        llm_kwargs = dict(
            model=self.model_name,
            download_dir=self._resolved_download_dir,
            enforce_eager=self.enforce_eager,
            **vllm_kwargs,
        )
        return LLM(**llm_kwargs)

    def _ensure_base_llm(self) -> LLM:
        if self.llm is None:
            self.llm = self._build_llm(use_hook_worker=False)
            self.tokenizer = self.llm.get_tokenizer()
            self.llm_engine = self.llm.llm_engine
        return self.llm

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
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Release the active engines and clear local references."""
        self._cleanup_hooks()
        if isinstance(self.llm, LLM):
            self._dispose_llm(self.llm)
        self.llm = None
        self.tokenizer = None
        self.llm_engine = None
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            pass

    def _build_hook_llm_with_memory_retry(self) -> LLM:
        try:
            return self._build_llm(use_hook_worker=True)
        except ValueError as exc:
            if "Paged attention: computed num_blocks too low" not in str(exc):
                raise
            if self.llm is None:
                raise
            print(
                "Hook worker load hit Metal KV-cache pressure; "
                "releasing base engine and retrying hook capture.",
                flush=True,
            )
            self._dispose_llm(self.llm)
            self.llm = None
            self.llm_engine = None
            return self._build_llm(use_hook_worker=True)

    @staticmethod
    def _reset_metal_config_cache() -> None:
        """Force vllm_metal to re-read Metal env vars on the next access."""
        try:
            from vllm_metal.config import reset_config

            reset_config()
        except Exception:
            pass

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
            os.environ["HOOK_LAYER_HEADS"] = layer_to_heads_string

        if "hookq" in config_data:
            os.environ["HOOKQ_MODE"] = config_data["hookq"]["hookq_mode"]

        if "steering" in config_data:
            os.environ["ACTSTEER_CONFIG"] = os.path.abspath(config_file)

        if "hidden_states" in config_data:
            hs_cfg = config_data["hidden_states"]
            layers = hs_cfg.get("layers", [])
            os.environ["HOOK_LAYERS"] = ";".join(map(str, layers))
            self._output_layers = layers if layers else True
            self._hs_mode = hs_cfg.get("mode", "last_token")
            os.environ["HOOK_HS_MODE"] = self._hs_mode

    def _build_hidden_states_prefill_params(self) -> SamplingParams:
        extra_args = {
            "output_hidden_states": self._output_layers if self._output_layers is not None else True,
            "hs_mode": self._hs_mode,
        }
        return SamplingParams(temperature=0.1, max_tokens=1, extra_args=extra_args)

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_hook: Optional[bool] = None,
        save_to_disk: bool = False,
        run_id: Optional[str] = None,
        cleanup: Optional[bool] = True,
        **kwargs,
    ):
        hook = use_hook if use_hook is not None else self.enable_hook

        if not isinstance(prompts, list):
            prompts = [prompts]

        if not hook or not self.worker_name:
            self._last_generate_used_hooks = False
            self._cleanup_hooks()
            kwargs.pop("save_to_disk", None)
            kwargs.pop("run_id", None)
            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            elif isinstance(sampling_params, list):
                if len(sampling_params) != len(prompts):
                    raise ValueError(
                        f"sampling_params list length ({len(sampling_params)}) "
                        f"must match prompts length ({len(prompts)})"
                    )
                stripped_params = []
                for sp in sampling_params:
                    if sp.extra_args:
                        sp = copy.copy(sp)
                        sp.extra_args = None
                    stripped_params.append(sp)
                sampling_params = stripped_params
            elif sampling_params.extra_args:
                sampling_params = copy.copy(sampling_params)
                sampling_params.extra_args = None
            base_llm = self._ensure_base_llm()
            return base_llm.generate(prompts, sampling_params)

        self._last_generate_used_hooks = True
        if "probe" in self.worker_name:
            return self.generate_with_encode_hook(
                prompts,
                sampling_params,
                cleanup,
                save_to_disk=save_to_disk,
                run_id=run_id,
                **kwargs,
            )
        if "steer" in self.worker_name:
            if (
                len(prompts) > 1
                and os.environ.get("VLLM_HOOK_METAL_STEER_BATCH", "0") != "1"
            ):
                if isinstance(sampling_params, list):
                    if len(sampling_params) != len(prompts):
                        raise ValueError(
                            f"sampling_params list length ({len(sampling_params)}) "
                            f"must match prompts length ({len(prompts)})"
                        )
                    params_list = sampling_params
                else:
                    params_list = [sampling_params] * len(prompts)

                outputs = []
                for prompt, params in zip(prompts, params_list):
                    outputs.extend(
                        self.generate_with_decode_hook(
                            [prompt],
                            [params] if params is not None else None,
                            cleanup,
                            **kwargs,
                        )
                    )
                return outputs
            return self.generate_with_decode_hook(
                prompts, sampling_params, cleanup, **kwargs
            )
        raise ValueError(f"Unsupported Metal worker '{self.worker_name}'.")

    def generate_with_encode_hook(
        self,
        prompts,
        sampling_params,
        cleanup,
        save_to_disk: bool = False,
        run_id: Optional[str] = None,
        **kwargs,
    ):
        hook_llm = None
        try:
            self._setup_hooks(cleanup)
            if (
                os.environ.get("VLLM_HOOK_RECLAIM_BASE_FOR_ENCODE", "0") == "1"
                and self.llm is not None
            ):
                print(
                    "Releasing base engine before Metal encode-hook capture.",
                    flush=True,
                )
                self._dispose_llm(self.llm)
                self.llm = None
                self.llm_engine = None
            hook_llm = self._build_hook_llm_with_memory_retry()
            hook_llm.llm_engine.collective_rpc("install_hooks")
            prefill_params = self._build_hidden_states_prefill_params()
            hook_llm.generate(prompts, prefill_params)
            self._assert_hook_artifacts_exist()
        finally:
            self._cleanup_hooks()
            self._dispose_llm(hook_llm)

        if sampling_params is None:
            kwargs.pop("save_to_disk", None)
            kwargs.pop("run_id", None)
            sampling_params = SamplingParams(**kwargs)
        if self.llm is None:
            self.llm = self._build_llm(use_hook_worker=False)
            self.tokenizer = self.llm.get_tokenizer()
            self.llm_engine = self.llm.llm_engine
        outputs = self.llm.generate(prompts, sampling_params)
        if not save_to_disk and outputs:
            probes = self._load_probes_for_run(self._last_run_id)
            if probes is not None:
                outputs[0].probes = probes
        return outputs

    def generate_with_decode_hook(self, prompts, sampling_params, cleanup, **kwargs):
        hook_llm = None
        output = None
        generation_exc = None
        try:
            print("Metal steer: preparing hook run", flush=True)
            self._log_memory("host before hook setup")
            self._ensure_memory_headroom("before hook setup")
            self._setup_hooks(cleanup)
            if os.environ.get("VLLM_HOOK_RECLAIM_BASE_FOR_STEER", "1") == "1":
                print(
                    "Releasing base engine before Metal steer-hook capture.",
                    flush=True,
                )
                self._dispose_llm(self.llm)
                self.llm = None
                self.llm_engine = None

            self._reset_metal_config_cache()
            print("Metal steer: building hook engine", flush=True)
            self._log_memory("host before hook engine build")
            self._ensure_memory_headroom("before hook engine build")
            hook_llm = self._build_hook_llm_with_memory_retry()
            print("Metal steer: installing hooks", flush=True)
            hook_llm.llm_engine.collective_rpc("install_hooks")

            if sampling_params is None:
                kwargs.pop("save_to_disk", None)
                kwargs.pop("run_id", None)
                sampling_params = SamplingParams(**kwargs)
            print("Metal steer: running hooked generation", flush=True)
            try:
                output = hook_llm.generate(prompts, sampling_params)
                print("Metal steer: hooked generation complete", flush=True)
            except Exception as exc:
                generation_exc = exc
        finally:
            print("Metal steer: cleaning up hook run", flush=True)
            self._cleanup_hooks()
            self._dispose_llm(hook_llm)
            self._reset_metal_config_cache()
            self._log_memory("host after hook exit")

        if generation_exc is not None:
            if self._is_memory_guard_exception(generation_exc):
                self.shutdown()
                raise SystemExit(
                    f"Metal memory guard triggered during hooked generation: "
                    f"{generation_exc}"
                ) from generation_exc
            self.llm = None
            self.tokenizer = None
            self.llm_engine = _LazyBaseEngineProxy(self)
            raise generation_exc

        self.llm = None
        self.tokenizer = None
        self.llm_engine = _LazyBaseEngineProxy(self)
        return output

    @staticmethod
    def _is_memory_guard_exception(exc: Exception) -> bool:
        message = str(exc).lower()
        return isinstance(exc, MemoryError) or (
            "memory guard" in message or "available=" in message or "rss=" in message
        )

    def _load_probes_for_run(self, run_id: Optional[str]) -> Optional[Dict]:
        """Load the latest Metal disk artifact into non-Metal output.probes shape."""
        if run_id is None:
            return None
        if self.worker_name in {"probe_hook_qk", "probe_hook_qk_metal"}:
            from vllm_hook_plugins.metal.run_utils_metal import load_and_merge_qk_cache

            return load_and_merge_qk_cache(self._hook_dir, run_id)
        if self.worker_name in {"probe_hidden_states", "probe_hidden_states_metal"}:
            from vllm_hook_plugins.metal.run_utils_metal import load_and_merge_hs_cache

            return load_and_merge_hs_cache(self._hook_dir, run_id)
        return None

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        probes: Optional[Dict] = None,
    ) -> Optional[Dict]:
        if self.analyzer is None:
            print("No analyzer configured")
            return None
        if not self._last_generate_used_hooks:
            raise RuntimeError(
                "No hook artifacts are available for analysis. "
                "Rerun generate with Metal hooks enabled."
            )
        if probes is not None:
            return self.analyzer.analyze(analyzer_spec=analyzer_spec, probes=probes)

        run_id = self._last_run_id
        run_ids = list(self._last_run_ids)
        if os.path.exists(self._run_id_file):
            with open(self._run_id_file, "r") as f:
                file_run_ids = [
                    ln.strip() for ln in f.read().splitlines() if ln.strip()
                ]
            if file_run_ids and not run_ids:
                run_ids = file_run_ids
            if run_id is None:
                run_id = run_ids[-1] if run_ids else None

        return dispatch_disk_analyze(
            self.analyzer,
            analyzer_spec,
            run_id=run_id,
            run_ids=run_ids[-2:] if len(run_ids) >= 2 else None,
        )

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
            for path in glob.glob(
                os.path.join(self._hook_dir, "**", "hidden_states.pt"), recursive=True
            ):
                os.remove(path)
            if os.path.exists(self._run_id_file):
                os.remove(self._run_id_file)

        run_id = str(uuid.uuid4())
        self._last_run_id = run_id
        self._last_run_ids.append(run_id)
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

        if self.worker_name in {"probe_hidden_states", "probe_hidden_states_metal"}:
            hs_paths = glob.glob(
                os.path.join(self._hook_dir, run_id, "**", "hidden_states.pt"),
                recursive=True,
            )
            if hs_paths:
                return
            raise RuntimeError(
                "Hooked hidden-states generation completed but produced no "
                f"hidden_states.pt artifact for run_id={run_id} under "
                f"{self._hook_dir}."
            )

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
