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
        tokenizer_for: Optional[str] = None,
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
        self._tokenizer_for = tokenizer_for
        self._tokenizer = None


    #: The only `vllm_xargs` keys the plugin JSON-decodes back into Python objects
    #: (`_plugin.py`). A dict or list under any other key would arrive as a string and be
    #: read as one, so this client refuses to send one rather than let it pass silently.
    _JSON_DECODED_XARGS = ("output_qk", "output_hidden_states", "steer")

    def _build_xargs(
        self,
        run_id: str,
        save_to_disk: Optional[bool],
        extra_xargs: Optional[Dict],
        steer: Optional[Dict],
        capture: bool,
    ) -> Optional[Dict]:
        """The `vllm_xargs` for one request, or None when it asks for nothing of MIA."""
        xargs: Dict[str, Any] = {}
        if capture:
            xargs.update(self._build_extra_body()["vllm_xargs"])
            os.makedirs(self._hook_dir, exist_ok=True)
            xargs["run_id"] = run_id
            xargs["hook_dir"] = self._hook_dir
            if save_to_disk is not None:
                xargs["save_to_disk"] = bool(save_to_disk)

        if steer is not None:
            xargs["steer"] = json.dumps(steer)

        for key, value in (extra_xargs or {}).items():
            if isinstance(value, (dict, list, tuple)):
                if key not in self._JSON_DECODED_XARGS:
                    raise ValueError(
                        f"extra_xargs[{key!r}] is a {type(value).__name__}, but the plugin "
                        f"only JSON-decodes {list(self._JSON_DECODED_XARGS)}; anything else "
                        f"would reach the worker as a string. Pass a scalar, or use the "
                        f"dedicated argument (steer=...) where one exists.")
                xargs[key] = json.dumps(value)
            else:
                xargs[key] = value

        return xargs

    @staticmethod
    def _body(xargs: Dict, extra_body: Optional[Dict]) -> Optional[Dict]:
        """Merge MIA's `vllm_xargs` with a caller's own extra_body, without either losing.

        A caller needs this for vLLM's own request extensions -- `return_token_ids` is the
        one that matters here, since it is how a client learns the exact token ids a
        request prompted on and generated.
        """
        body = dict(extra_body or {})
        caller_xargs = body.pop("vllm_xargs", None) or {}
        merged = {**caller_xargs, **xargs}
        if merged:
            body["vllm_xargs"] = merged
        return body or None

    def _record(self, response, run_id: Optional[str], save_to_disk: Optional[bool]):
        try:
            size = len(getattr(response, "_raw_response", None).text)  # type: ignore[union-attr]
            PROF.gauge("client.response_bytes", size)
        except Exception:
            try:
                PROF.gauge("client.response_bytes_est", len(response.model_dump_json()))
            except Exception:
                pass
        self._last_response = response
        self._last_run_id = run_id
        self._last_save_to_disk = save_to_disk
        return response

    def generate(
        self,
        messages: List[Dict],
        model: str,
        save_to_disk: Optional[bool] = None,
        run_id: Optional[str] = None,
        extra_xargs: Optional[Dict] = None,
        steer: Optional[Dict] = None,
        capture: bool = True,
        extra_body: Optional[Dict] = None,
        **openai_kwargs,
    ):
        """Send a chat completion with probe capture.

        The server applies the model's chat template, so the token layout is the server's.
        When an analyzer needs exact token spans, use :meth:`generate_tokens` instead.

        ``extra_xargs`` carries per-request knobs the config file does not cover -- the
        serve-path equivalent of ``SamplingParams.extra_args`` offline, e.g.
        ``{"hooks_on": "both"}``. ``steer`` sends a steering config for this request alone.
        ``capture=False`` is the equivalent of offline ``use_hook=False``: a plain request
        that arms nothing.
        """
        run_id = run_id or str(uuid.uuid4())
        body = self._body(
            self._build_xargs(run_id, save_to_disk, extra_xargs, steer, capture), extra_body)

        PROF.incr("client.request.calls")
        with PROF.timed("client.request"):
            response = self._openai.chat.completions.create(
                model=model, messages=messages, extra_body=body, **openai_kwargs)

        return self._record(response, run_id if capture else None,
                            save_to_disk if capture else False)

    def generate_tokens(
        self,
        prompt_token_ids,
        model: str,
        save_to_disk: Optional[bool] = None,
        run_id: Optional[str] = None,
        extra_xargs: Optional[Dict] = None,
        steer: Optional[Dict] = None,
        capture: bool = True,
        extra_body: Optional[Dict] = None,
        **openai_kwargs,
    ):
        """Capture against **exact token ids**, via the completions endpoint.

        No chat template is applied, so the tokens the model sees are the tokens passed in
        -- which is what an analyzer scoring token spans requires. Accepts one sequence
        (``[int, ...]``) or a batch (``[[int, ...], ...]``); a batch shares one ``run_id``,
        the way a list passed to the offline ``generate`` does.
        """
        return self._completions(prompt_token_ids, model, save_to_disk, run_id,
                                 extra_xargs, steer, capture, extra_body, **openai_kwargs)

    def generate_text(
        self,
        prompt,
        model: str,
        save_to_disk: Optional[bool] = None,
        run_id: Optional[str] = None,
        extra_xargs: Optional[Dict] = None,
        steer: Optional[Dict] = None,
        capture: bool = True,
        extra_body: Optional[Dict] = None,
        **openai_kwargs,
    ):
        """Capture against raw text, via the completions endpoint -- no chat template.

        Use this for a prompt you have already templated yourself. Accepts one string or a
        list of strings, and a list shares one ``run_id``.
        """
        return self._completions(prompt, model, save_to_disk, run_id, extra_xargs, steer,
                                 capture, extra_body, **openai_kwargs)

    def _completions(self, prompt, model, save_to_disk, run_id, extra_xargs, steer,
                     capture, extra_body=None, **openai_kwargs):
        run_id = run_id or str(uuid.uuid4())
        body = self._body(
            self._build_xargs(run_id, save_to_disk, extra_xargs, steer, capture), extra_body)

        PROF.incr("client.request.calls")
        with PROF.timed("client.request"):
            response = self._openai.completions.create(
                model=model, prompt=prompt, extra_body=body, **openai_kwargs)

        return self._record(response, run_id if capture else None,
                            save_to_disk if capture else False)

    @property
    def tokenizer(self):
        """The served model's tokenizer, for computing the spans an analyzer scores.

        Loaded locally and lazily; the offline entry point exposes the engine's own.
        """
        if self._tokenizer is None:
            if not self._tokenizer_for:
                raise RuntimeError(
                    "no tokenizer bound: construct the client with "
                    "tokenizer_for=<model id>, or load one yourself with "
                    "transformers.AutoTokenizer.")
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_for)
        return self._tokenizer

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
        probes: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Run the configured analyzer on the last generate() result.

        ``probes`` analyzes a payload you already hold instead of the last response -- the
        serve-path equivalent of passing ``probes=`` to the offline ``analyze``.
        """
        if probes is not None:
            with PROF.timed("analyzer.kernel"):
                return self.analyzer.analyze(analyzer_spec, probes=probes)

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

