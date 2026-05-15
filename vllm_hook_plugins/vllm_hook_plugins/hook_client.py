"""
HookClient — OpenAI-compatible client for vllm serve with probe analysis.

Mirrors the HookLLM offline API over HTTP:
    hook = HookClient(base_url=..., analyzer_name=..., config_file=...)
    response = hook.generate(model=..., messages=[...])
    result   = hook.analyze(analyzer_spec=...)

Server setup:
    VLLM_HOOK_WORKER=qk vllm serve <model> --enforce-eager
    VLLM_HOOK_WORKER=hidden_states vllm serve <model> --enforce-eager
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional

import torch


class HookClient:
    def __init__(
        self,
        base_url: str,
        analyzer_name: str,
        config_file: str,
        api_key: str = "EMPTY",
        hook_dir: str = None,
    ):
        from vllm_hook_plugins.registry import PluginRegistry
        from vllm_hook_plugins import register_plugins
        register_plugins()

        self._load_config(config_file)

        analyzer_entry = PluginRegistry.get_analyzer(analyzer_name)
        if analyzer_entry is None:
            raise ValueError(
                f"Unknown analyzer: {analyzer_name!r}. "
                f"Available: {PluginRegistry.list_analyzers()}"
            )
        self._hook_dir = hook_dir or "/dev/shm/vllm_hook"
        self.analyzer = analyzer_entry.analyzer(self._hook_dir, self.layer_to_heads)

        import openai
        self._openai = openai.OpenAI(base_url=base_url, api_key=api_key)

        self._last_response: Any = None
        self._last_run_id: Optional[str] = None
        self._last_save_to_disk: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict],
        model: str,
        save_to_disk: bool = False,
        run_id: Optional[str] = None,
        **openai_kwargs,
    ):
        """Send a chat completion request with probe capture.

        Args:
            messages: OpenAI-style message list.
            model: Model name as registered with vllm serve.
            save_to_disk: If True, server writes artifacts to hook_dir.
                Server and client must share the filesystem path (same host).
                hook_dir defaults to /dev/shm/vllm_hook (shared memory).
            run_id: Artifact directory name under hook_dir. Auto-generated if
                omitted and save_to_disk=True.
            **openai_kwargs: Forwarded to openai.chat.completions.create()
                (e.g. max_tokens, temperature).

        Returns:
            Raw openai ChatCompletion response. Text is at
            response.choices[0].message.content; probes (in-memory path) at
            response.probes.
        """
        extra_body = self._build_extra_body()

        if save_to_disk:
            run_id = run_id or str(uuid.uuid4())
            os.makedirs(self._hook_dir, exist_ok=True)
            extra_body["vllm_xargs"].update({
                "save_to_disk": True,
                "run_id": run_id,
                "hook_dir": self._hook_dir,
            })

        response = self._openai.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body,
            **openai_kwargs,
        )

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
        """Run the configured analyzer on the last generate() result.

        In-memory path (save_to_disk=False): reads response.probes,
        deserializes tensors, runs analyzer in-process.

        Disk path (save_to_disk=True): reads artifacts from hook_dir/run_id,
        runs analyzer. Falls back to last generate()'s run_id if omitted.
        For two-pass analyzers (CoRer), pass run_ids=[pass1_id, pass2_id].
        """
        if self._last_response is None:
            raise RuntimeError("No generate() call has been made yet.")

        if self._last_save_to_disk:
            import inspect
            effective_run_id = run_id or self._last_run_id
            sig = inspect.signature(self.analyzer.analyze)
            kwargs = {"analyzer_spec": analyzer_spec}
            if "run_ids" in sig.parameters and run_ids is not None:
                kwargs["run_ids"] = run_ids
            elif "run_id" in sig.parameters and effective_run_id is not None:
                kwargs["run_id"] = effective_run_id
            return self.analyzer.analyze(**kwargs)

        # In-memory path: probes come back in response.probes as serialized JSON.
        raw_probes = getattr(self._last_response, "probes", None)
        if raw_probes is None:
            raise RuntimeError(
                "Response has no .probes field. Make sure the server was started "
                "with the vllm_hook_plugins plugin loaded "
                "(check VLLM_HOOK_WORKER env var and plugin entry point)."
            )
        probes = self._deserialize_probes(raw_probes)

        if "qk_cache" not in probes and "hs_cache" not in probes:
            raise RuntimeError(f"Unexpected probes keys: {list(probes.keys())}")

        return self.analyzer.analyze(analyzer_spec, probes=probes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        # vLLM v0.12+ maps extra_body["vllm_xargs"] -> SamplingParams.extra_args.
        # vllm_xargs is typed as dict[str, str|int|float|list[scalar]], so nested
        # structures (layer_to_heads dict, list-of-ints) must be JSON-encoded as
        # strings. _hook_plugin._patched_generate decodes them back.
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
        """Convert serialized probes (JSON-safe) back to torch tensors.

        _serialize_probes in _hook_plugin.py converts tensors to lists via
        .tolist() and passes config (flat dict of scalars) through unchanged.
        This reverses that: lists -> tensors, scalars/config left as-is.
        """
        result = {}
        for cache_key, cache_val in raw.items():
            # config is a flat dict of scalars — pass through directly.
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
