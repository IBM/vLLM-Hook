"""MiaLLM: a vllm.LLM wrapper that arms MIA capture or steering and runs analyzers."""
import copy
import os
import json
import uuid
from typing import Optional, Dict, List

import vllm.plugins
from vllm import LLM, SamplingParams

from mia.optimizations import apply_optimizations
from mia.registry import PluginRegistry
from mia.core.delivery.sizing import qk_score_size_select
from mia.artifacts import dispatch_disk_analyze
from mia._profiler import PROF
from mia.workers.steer_worker import resolve_steer_modes
from mia.core.delivery.artifact_quant import dequantize_cache_inplace


def _merge_probes(all_probes: list) -> dict:
    first = next((p for p in all_probes if p), {})
    merged = {k: v for k, v in first.items()
              if k not in ("qk_cache", "hs_cache")}
    TENSOR_KEYS = ("q", "k_all", "hidden_states", "scores",
                   "q_scale", "k_all_scale", "hidden_states_scale")
    for cache_key in ("qk_cache", "hs_cache"):
        if not any(cache_key in p for p in all_probes):
            continue
        merged[cache_key] = {}
        layers = []
        for p in all_probes:
            for layer in p.get(cache_key, {}):
                if layer not in layers:
                    layers.append(layer)
        for layer in layers:
            entry = {}
            for p in all_probes:
                le = p.get(cache_key, {}).get(layer, {})
                for k, v in le.items():
                    if k not in TENSOR_KEYS:
                        entry.setdefault(k, v)
            for tensor_key in TENSOR_KEYS:
                vals = []
                for p in all_probes:
                    le = p.get(cache_key, {}).get(layer, {})
                    v = le.get(tensor_key)
                    vals.append(v[0] if v is not None and len(v) > 0 else None)
                if any(v is not None for v in vals):
                    entry[tensor_key] = vals
            merged[cache_key][layer] = entry
    return merged


def _dequantized(p0):
    if p0:
        if isinstance(p0.get("qk_cache"), dict):
            dequantize_cache_inplace(p0["qk_cache"], ("q", "k_all"))
        if isinstance(p0.get("hs_cache"), dict):
            dequantize_cache_inplace(p0["hs_cache"], ("hidden_states",))
    return p0


class MiaLLM:
    """An in-process ``vllm.LLM`` that runs one MIA worker (capture or steering) and an analyzer."""

    def __init__(
        self,
        model: str,
        worker_name: str = None,
        analyzer_name: str = None,
        config_file: str = None,
        download_dir: Optional[str] = None,
        enable_hook: bool = True,
        hook_dir: str = None,
        enforce_eager: bool = False,
        **vllm_kwargs
    ):
        """Build the engine.

        Args:
            model: model name or path, as for ``vllm.LLM``.
            worker_name: "capture_hs", "capture_qk" or "steer"; None runs no MIA worker.
            analyzer_name: a registered analyzer for ``analyze()``; None for none.
            config_file: JSON with the worker's settings (heads, Q/K mode, steering, HS layers).
            download_dir: model download dir; also the root of the default ``hook_dir``.
            enable_hook: default for ``generate(use_hook=...)``.
            hook_dir: dir for ``save_to_disk`` runs; default ``<download_dir>/mia``, else
                ``~/.cache/mia``.
            enforce_eager: run eagerly instead of with CUDA graphs.
            **vllm_kwargs: passed to ``vllm.LLM``.
        """
        self.model_name = model
        self.worker_name = worker_name
        self.analyzer_name = analyzer_name
        self.enable_hook = enable_hook
        self.enforce_eager = enforce_eager

        if download_dir is not None:
            download_dir = os.path.expanduser(download_dir)
        self._download_dir = download_dir

        if hook_dir is not None:
            HOOK_DIR = hook_dir
        else:
            fallback_root = download_dir or os.path.expanduser('~/.cache')
            HOOK_DIR = os.path.join(fallback_root, 'mia')
        os.makedirs(HOOK_DIR, exist_ok=True)
        self._hook_dir = HOOK_DIR

        self.layer_to_heads = {}
        self._output_layers = None
        self._hookq_mode = "all_tokens"
        self._qk_capture = "qk"
        self._score_head = 0
        self._hs_mode = "last_token"
        self._steering_config: Optional[Dict] = None
        if config_file:
            self.load_config(config_file)

        # Unknown names are refused before the engine is built.
        worker = analyzer_entry = None
        if worker_name or analyzer_name:
            vllm.plugins.load_general_plugins()
        if worker_name:
            worker_entry = PluginRegistry.get_worker(worker_name)
            if worker_entry is None:
                raise ValueError(f"Unknown worker: {worker_name!r}. "
                                 f"Available: {PluginRegistry.list_workers()}")
            worker = worker_entry.path
        if analyzer_name:
            analyzer_entry = PluginRegistry.get_analyzer(analyzer_name)
            if analyzer_entry is None:
                raise ValueError(f"Unknown analyzer: {analyzer_name!r}. "
                                 f"Available: {PluginRegistry.list_analyzers()}")

        llm_kwargs = dict(vllm_kwargs)
        if download_dir is not None:
            llm_kwargs['download_dir'] = download_dir
        # lazy: keep mia._plugin and mia.core.hooks out of import mia
        from mia._plugin import engine_hints
        with engine_hints(qk_score=self._qk_capture == "score"):
            self.llm = LLM(
                model=model,
                worker_extension_cls=worker,
                enforce_eager=enforce_eager,
                **llm_kwargs,
            )

        self.tokenizer = self.llm.get_tokenizer()
        self.llm_engine = self.llm.llm_engine

        self._model_dims = None
        try:
            tc = self.llm_engine.model_config.hf_text_config
            H_q = int(getattr(tc, "num_attention_heads"))
            H_kv = int(getattr(tc, "num_key_value_heads", H_q))
            hidden = int(getattr(tc, "hidden_size"))
            self._model_dims = {"H_q": H_q, "H_kv": H_kv, "d": hidden // H_q}
        except Exception:
            self._model_dims = None
        self._autosel_log_n = 0

        self.analyzer = None
        self._analyzer_accepts = "qk"
        if analyzer_entry is not None:
            analyzer_cls = analyzer_entry.analyzer
            self._analyzer_accepts = getattr(analyzer_cls, "ACCEPTS", "qk")
            self.analyzer = analyzer_cls(self._hook_dir, self.layer_to_heads)


    def load_config(self, config_file: str):
        """Load worker settings (important heads, Q/K mode, steering, HS layers) from JSON."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        apply_optimizations(config_data)

        if "params" in config_data and "important_heads" in config_data["params"]:
            self.important_heads = config_data["params"]["important_heads"]
            self.layer_to_heads = {}
            for layer_idx, head_idx in self.important_heads:
                if layer_idx not in self.layer_to_heads:
                    self.layer_to_heads[layer_idx] = []
                self.layer_to_heads[layer_idx].append(head_idx)

        if "hookq" in config_data:
            self._hookq_mode = config_data["hookq"].get("hookq_mode", self._hookq_mode)
            self._qk_capture = config_data["hookq"].get("capture", self._qk_capture)
            self._score_head = int(config_data["hookq"].get("score_head", self._score_head))

        if "steering" in config_data:
            self._steering_config = dict(config_data["steering"])
            resolve_steer_modes(self._steering_config)

        if "hidden_states" in config_data:
            hs_cfg = config_data["hidden_states"]
            layers = hs_cfg.get("layers", [])
            self._hs_mode = hs_cfg.get("mode", "last_token")
            self._output_layers = layers if layers else True

    def _build_extra_args(self, save_to_disk: bool, run_id: str,
                            request_extra_args: Optional[dict] = None) -> dict:
        extra = {}
        request_extra_args = request_extra_args or {}
        if self.worker_name == "capture_hs":
            extra["output_hidden_states"] = self._output_layers if self._output_layers else True
            extra["hs_mode"] = self._hs_mode
        elif self.worker_name == "capture_qk":
            extra["output_qk"] = self.layer_to_heads if self.layer_to_heads else True
            extra["hookq_mode"] = self._hookq_mode
            cap = request_extra_args.get("qk_capture", self._qk_capture)
            if cap == "score":
                extra["qk_capture"] = "score"
                extra["score_head"] = self._score_head
        elif self.worker_name == "steer":
            base = dict(self._steering_config) if self._steering_config else {}
            override = request_extra_args.get("steer")
            if isinstance(override, dict):
                base.update(override)
            extra["steer"] = base or True
        if save_to_disk:
            extra["save_to_disk"] = True
            extra["run_id"] = run_id
            extra["hook_dir"] = self._hook_dir
        return extra

    def _prompt_token_len(self, prompt) -> Optional[int]:
        try:
            if isinstance(prompt, str):
                return len(self.tokenizer.encode(prompt))
            if isinstance(prompt, (list, tuple)):
                return len(prompt)
            if isinstance(prompt, dict):
                toks = prompt.get("prompt_token_ids")
                if toks is not None:
                    return len(toks)
                txt = prompt.get("prompt")
                if isinstance(txt, str):
                    return len(self.tokenizer.encode(txt))
        except Exception:
            return None
        return None

    def _qk_capture_for(self, prompt_len: int, mode: str) -> str:
        dims = self._model_dims
        if not dims or not self.layer_to_heads:
            return "qk"
        return qk_score_size_select(prompt_len, mode, self.layer_to_heads,
                                    dims["H_q"], dims["H_kv"], dims["d"])

    def _maybe_auto_select(self, prompt, sp, extra: dict) -> None:
        if self.worker_name != "capture_qk":
            return
        if os.environ.get("MIA_QK_AUTO_SELECT", "0") != "1":
            return
        if self._analyzer_accepts not in ("score", "either"):
            return
        if "qk_capture" in extra:
            return
        if self._qk_capture != "qk":
            return
        if self._model_dims is None:
            return
        # lazy: keep mia._plugin and mia.core.hooks out of import mia
        from mia._plugin import _engine_graph, _engine_tp_size
        if _engine_tp_size(self.llm) > 1 or _engine_graph(self.llm):
            return
        prompt_len = self._prompt_token_len(prompt)
        if prompt_len is None:
            return
        req_mode = extra.get("hookq_mode", self._hookq_mode)
        pick = self._qk_capture_for(prompt_len, req_mode)
        extra["qk_capture"] = pick
        if self._autosel_log_n < 16:
            self._autosel_log_n += 1
            print(f"[miallm/D2] auto-select qk_capture={pick} (S={prompt_len} "
                  f"mode={req_mode} accepts={self._analyzer_accepts})", flush=True)

    def generate(
        self,
        prompts: List[str],
        sampling_params=None,
        use_hook: Optional[bool] = None,
        save_to_disk: bool = False,
        run_id: Optional[str] = None,
        **kwargs
    ):
        """Generate as ``vllm.LLM.generate`` does, with the worker armed on every prompt.

        Args:
            prompts: one prompt or a list.
            sampling_params: a ``SamplingParams`` or one per prompt (default: from ``**kwargs``).
            use_hook: arm the worker for this call (default: ``enable_hook``).
            save_to_disk: write captured data to ``<hook_dir>/<run_id>/`` instead of ``probes``.
            run_id: the run's name (default: a new uuid).
        Returns:
            The ``RequestOutput`` list; ``out[i].probes`` holds prompt i's capture, and with
            several prompts ``out[0].probes`` holds the whole batch.
        """
        hook = use_hook if use_hook is not None else self.enable_hook

        if not isinstance(prompts, list):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)

        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                raise ValueError(
                    f"sampling_params list length ({len(sampling_params)}) "
                    f"must match prompts length ({len(prompts)})"
                )
            sp_list = list(sampling_params)
        else:
            sp_list = [sampling_params] * len(prompts)

        if hook and self.worker_name:
            if run_id is None:
                run_id = str(uuid.uuid4())
            with PROF.timed("miallm.build_extra"):
                new_sp_list = []
                for prompt, sp in zip(prompts, sp_list):
                    sp = copy.copy(sp)
                    extra = dict(sp.extra_args or {})
                    self._maybe_auto_select(prompt, sp, extra)
                    defaults = self._build_extra_args(save_to_disk, run_id,
                                                       request_extra_args=extra)
                    for k, v in defaults.items():
                        extra.setdefault(k, v)
                    if "steer" in defaults:
                        extra["steer"] = defaults["steer"]
                    sp.extra_args = extra
                    new_sp_list.append(sp)
                sp_list = new_sp_list
            self._last_run_id = run_id
        else:
            new_sp_list = []
            for sp in sp_list:
                if sp.extra_args:
                    sp = copy.copy(sp)
                    sp.extra_args = None
                new_sp_list.append(sp)
            sp_list = new_sp_list

        passthrough = {k: v for k, v in kwargs.items()
                       if k not in ("temperature", "max_tokens", "top_p", "top_k",
                                    "min_p", "n", "seed", "stop", "stop_token_ids",
                                    "presence_penalty", "frequency_penalty",
                                    "repetition_penalty")}
        PROF.incr("miallm.generate.calls")
        PROF.gauge("miallm.prompts", len(prompts))
        with PROF.timed("miallm.generate"):
            if all(sp is sp_list[0] for sp in sp_list):
                outputs = self.llm.generate(prompts, sp_list[0], **passthrough)
            else:
                outputs = self.llm.generate(prompts, sp_list, **passthrough)

        if hook and self.worker_name and not save_to_disk:
            # lazy: keep mia.core.delivery out of import mia
            from mia.core.delivery.delivered_probes import attach_lazy, merge_source, pending
            srcs = [merge_source(o) for o in outputs]
            if len(outputs) > 1 and any(s is not None for s in srcs):
                def merged():
                    with PROF.timed("miallm.merge_probes"):
                        parts = [(s() if s is not None else None) or {} for s in srcs]
                        return _dequantized(_merge_probes(parts)) if any(parts) else None
                if any(pending(o) for o in outputs):
                    try:
                        attach_lazy(outputs[0], merged)
                    except TypeError:
                        outputs[0].probes = merged()
                else:
                    outputs[0].probes = merged()
            elif not pending(outputs[0]):
                _dequantized(getattr(outputs[0], "probes", None))

        return outputs

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        probes: Optional[Dict] = None,
        run_id: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Run the analyzer on ``probes``, else on disk run ``run_id`` (default: the last run)."""
        if self.analyzer is None:
            print("No analyzer configured")
            return None

        PROF.incr("miallm.analyze.calls")

        if probes is not None:
            with PROF.timed("miallm.analyze"):
                with PROF.timed("analyzer.kernel"):
                    return self.analyzer.analyze(analyzer_spec=analyzer_spec, probes=probes)

        effective_run_id = run_id or getattr(self, "_last_run_id", None)
        with PROF.timed("miallm.analyze"):
            return dispatch_disk_analyze(self.analyzer, analyzer_spec,
                                         run_id=effective_run_id, run_ids=run_ids)

    def close(self):
        """No-op; kept for API compatibility."""

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

