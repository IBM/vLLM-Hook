"""MiaClient: OpenAI-compatible client for vllm serve with probe capture and analysis."""
from __future__ import annotations

import glob
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import torch

from mia._profiler import PROF
from mia.run_utils import dispatch_disk_analyze


class MiaClient:
    def __init__(
        self,
        base_url: str,
        analyzer_name: str,
        config_file: str,
        api_key: str = "EMPTY",
        hook_dir: str = None,
    ):
        from mia.registry import PluginRegistry
        from mia import register_plugins
        register_plugins()

        self._load_config(config_file)

        analyzer_entry = PluginRegistry.get_analyzer(analyzer_name)
        if analyzer_entry is None:
            raise ValueError(
                f"Unknown analyzer: {analyzer_name!r}. "
                f"Available: {PluginRegistry.list_analyzers()}"
            )
        self._hook_dir = hook_dir or "/dev/shm/mia"
        self.analyzer = analyzer_entry.analyzer(self._hook_dir, self.layer_to_heads)

        import openai
        self._openai = openai.OpenAI(base_url=base_url, api_key=api_key)

        self._last_response: Any = None
        self._last_run_id: Optional[str] = None
        self._last_save_to_disk: bool = False


    def generate(
        self,
        messages: List[Dict],
        model: str,
        save_to_disk: Optional[bool] = None,
        run_id: Optional[str] = None,
        extra_xargs: Optional[Dict] = None,
        **openai_kwargs,
    ):
        """Send a chat completion request with probe capture.

        ``extra_xargs`` carries per-request knobs the config file does not cover -- the
        serve-path equivalent of ``SamplingParams.extra_args`` offline, e.g.
        ``{"hooks_on": "both"}``. vLLM's ``vllm_xargs`` only accepts scalars, so dicts and
        lists are JSON-encoded here, exactly as the plugin expects to decode them.
        """
        extra_body = self._build_extra_body()

        run_id = run_id or str(uuid.uuid4())
        os.makedirs(self._hook_dir, exist_ok=True)
        extra_body["vllm_xargs"].update({
            "run_id": run_id,
            "hook_dir": self._hook_dir,
        })
        if save_to_disk is not None:
            extra_body["vllm_xargs"]["save_to_disk"] = bool(save_to_disk)
        for key, value in (extra_xargs or {}).items():
            extra_body["vllm_xargs"][key] = (
                json.dumps(value) if isinstance(value, (dict, list)) else value)

        PROF.incr("client.request.calls")
        with PROF.timed("client.request"):
            response = self._openai.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                **openai_kwargs,
            )

        try:
            size = len(getattr(response, "_raw_response", None).text)  # type: ignore[union-attr]
            PROF.gauge("client.response_bytes", size)
        except Exception:
            try:
                PROF.gauge("client.response_bytes_est",
                           len(response.model_dump_json()))
            except Exception:
                pass

        self._last_response = response
        self._last_run_id = run_id
        self._last_save_to_disk = save_to_disk
        return response

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Run the configured analyzer on the last generate() result."""
        if self._last_response is None:
            raise RuntimeError("No generate() call has been made yet.")

        raw_probes = getattr(self._last_response, "probes", None)
        if raw_probes is None:
            effective_run_id = run_id or self._last_run_id
            if not self._last_save_to_disk and not self._artifact_dir_exists(effective_run_id):
                raise RuntimeError(
                    "Response has no .probes field. Make sure the server was started "
                    "with the mia plugin loaded "
                    "(check MIA_WORKER env var and plugin entry point)."
                )
            self._wait_artifact_dir(effective_run_id)
            return dispatch_disk_analyze(self.analyzer, analyzer_spec,
                                         run_id=effective_run_id, run_ids=run_ids)
        with PROF.timed("client.deserialize"):
            probes = self._deserialize_probes(raw_probes)

        if "qk_cache" not in probes and "hs_cache" not in probes:
            raise RuntimeError(f"Unexpected probes keys: {list(probes.keys())}")

        with PROF.timed("analyzer.kernel"):
            return self.analyzer.analyze(analyzer_spec, probes=probes)


    def _wait_artifact_dir(self, run_id, timeout_s: float = 10.0, poll_s: float = 0.05) -> bool:
        base = os.path.join(self._hook_dir, run_id)
        prev = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            files = sorted(glob.glob(os.path.join(base, "**", "*"), recursive=True))
            files = [f for f in files if os.path.isfile(f) and not f.endswith(".tmp")
                     and not os.path.basename(f).startswith(".tmp")]
            if files and files == prev:
                return True
            prev = files
            time.sleep(poll_s)
        return bool(prev)

    def _artifact_dir_exists(self, run_id: Optional[str]) -> bool:
        if not run_id:
            return False
        path = os.path.join(self._hook_dir, run_id)
        try:
            return os.path.isdir(path) and bool(os.listdir(path))
        except OSError:
            return False

    def _load_config(self, config_file: str):
        with open(config_file) as f:
            cfg = json.load(f)

        self.layer_to_heads: Dict[int, list] = {}
        self._output_layers = None

        if "params" in cfg and "important_heads" in cfg["params"]:
            for layer_idx, head_idx in cfg["params"]["important_heads"]:
                self.layer_to_heads.setdefault(layer_idx, []).append(head_idx)

        self._hookq_mode = cfg.get("hookq", {}).get("hookq_mode", "last_token")

        if "hidden_states" in cfg:
            layers = cfg["hidden_states"].get("layers", [])
            self._output_layers = layers if layers else True

    def _build_extra_body(self) -> Dict:
        if self._output_layers is not None:
            layers = self._output_layers
            xargs = {"output_hidden_states": json.dumps(layers) if isinstance(layers, list) else layers}
        elif self.layer_to_heads:
            xargs = {
                "output_qk": json.dumps({str(k): v for k, v in self.layer_to_heads.items()}),
                "hookq_mode": self._hookq_mode,
            }
        else:
            xargs = {"output_hidden_states": True}
        return {"vllm_xargs": xargs}

    def _deserialize_probes(self, raw: dict) -> dict:
        result = {}
        for cache_key, cache_val in raw.items():
            if cache_key == "config":
                result[cache_key] = cache_val
                continue
            if not isinstance(cache_val, dict):
                result[cache_key] = cache_val
                continue
            result[cache_key] = {}
            for mod_name, entry in cache_val.items():
                if not isinstance(entry, dict):
                    result[cache_key][mod_name] = entry
                    continue
                restored = {}
                for k, v in entry.items():
                    if isinstance(v, list):
                        restored[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        restored[k] = v
                result[cache_key][mod_name] = restored
        return result

