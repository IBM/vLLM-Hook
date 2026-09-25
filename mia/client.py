"""
MiaClient — OpenAI-compatible client for vllm serve with probe analysis.

Mirrors the MiaLLM offline API over HTTP:
    hook = MiaClient(base_url=..., analyzer_name=..., config_file=...)
    response = hook.generate(model=..., messages=[...])
    result   = hook.analyze(analyzer_spec=...)

Server setup:
    MIA_WORKER=qk vllm serve <model> --enforce-eager
    MIA_WORKER=hidden_states vllm serve <model> --enforce-eager
"""
from __future__ import annotations

import json
import os
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict],
        model: str,
        save_to_disk: Optional[bool] = None,
        run_id: Optional[str] = None,
        **openai_kwargs,
    ):
        """Send a chat completion request with probe capture.

        Args:
            messages: OpenAI-style message list.
            model: Model name as registered with vllm serve.
            save_to_disk: Where the artifact goes. THREE states:
                ``None`` (default) -- no preference: the server's storage router picks the
                    fastest path per request. ``analyze()`` reads whichever it chose, so this
                    is transparent unless you need the file itself.
                ``True``  -- REQUIRE a durable artifact file under hook_dir/run_id. Honored
                    exactly; the router never overrides it (that is how you save e.g. HS
                    last_token, which the router would otherwise price as RPC).
                ``False`` -- require the in-memory path; artifacts come back on the response.
                Server and client must share the filesystem path (same host) for the disk
                path. hook_dir defaults to /dev/shm/mia (shared memory).
            run_id: Artifact directory name under hook_dir. Auto-generated if omitted.
            **openai_kwargs: Forwarded to openai.chat.completions.create()
                (e.g. max_tokens, temperature).

        Returns:
            Raw openai ChatCompletion response. Text is at
            response.choices[0].message.content; probes (in-memory path) at
            response.probes.
        """
        extra_body = self._build_extra_body()

        # Always carry a run_id + hook_dir, even when we are not asking for disk. With
        # save_to_disk=None the server's router may send this request to disk; without a run_id
        # of our own the server falls back to its internal request_id, which we never learn ->
        # the artifact lands somewhere we cannot read. Both keys are inert unless the request
        # ends up on disk, so sending them always costs nothing.
        run_id = run_id or str(uuid.uuid4())
        os.makedirs(self._hook_dir, exist_ok=True)
        extra_body["vllm_xargs"].update({
            "run_id": run_id,
            "hook_dir": self._hook_dir,
        })
        # Send save_to_disk ONLY when the caller stated a preference: the server routes only
        # when the key is ABSENT, and treats any present value as a requirement to honor.
        if save_to_disk is not None:
            extra_body["vllm_xargs"]["save_to_disk"] = bool(save_to_disk)

        PROF.incr("client.request.calls")
        with PROF.timed("client.request"):
            response = self._openai.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                **openai_kwargs,
            )

        # Capture the wire size of the response when available.
        try:
            size = len(getattr(response, "_raw_response", None).text)  # type: ignore[union-attr]
            PROF.gauge("client.response_bytes", size)
        except Exception:
            # Fallback: estimate via the OpenAI model_dump_json output.
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
        """Run the configured analyzer on the last generate() result.

        Reads whichever path the artifact ACTUALLY took: probes on the response => in-memory;
        no probes => the worker wrote hook_dir/run_id. That matters when ``generate`` was called
        with ``save_to_disk=None`` (the default), where the server's storage router chooses per
        request. An explicit True/False is honored by the server, so there the two always agree.

        For two-pass analyzers (CoRer), pass run_ids=[pass1_id, pass2_id].
        """
        if self._last_response is None:
            raise RuntimeError("No generate() call has been made yet.")

        raw_probes = getattr(self._last_response, "probes", None)
        if raw_probes is None:
            # Either we asked for disk, or the router sent this request there behind us.
            effective_run_id = run_id or self._last_run_id
            if not self._last_save_to_disk and not self._artifact_dir_exists(effective_run_id):
                # We asked for RPC, got no probes, and nothing was written -> the server is not
                # actually hooking. Keep the precise diagnosis rather than a disk read failure.
                raise RuntimeError(
                    "Response has no .probes field. Make sure the server was started "
                    "with the mia plugin loaded "
                    "(check MIA_WORKER env var and plugin entry point)."
                )
            # The generate() durability barrier is gone (fire-and-forget disk flush), so wait
            # here (bounded, on the atomic-write guarantee) for the artifact before reading.
            self._wait_artifact_dir(effective_run_id)
            return dispatch_disk_analyze(self.analyzer, analyzer_spec,
                                         run_id=effective_run_id, run_ids=run_ids)
        with PROF.timed("client.deserialize"):
            probes = self._deserialize_probes(raw_probes)

        if "qk_cache" not in probes and "hs_cache" not in probes:
            raise RuntimeError(f"Unexpected probes keys: {list(probes.keys())}")

        with PROF.timed("analyzer.kernel"):
            return self.analyzer.analyze(analyzer_spec, probes=probes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_artifact_dir(self, run_id, timeout_s: float = 10.0, poll_s: float = 0.05) -> bool:
        import os, time, glob
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
        """True if the worker wrote artifacts for ``run_id`` under hook_dir.

        Used only to tell "the router rerouted us to disk" from "the server never hooked at
        all", so the latter keeps its precise error instead of a confusing disk-read failure.
        """
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
        # vLLM v0.12+ maps extra_body["vllm_xargs"] -> SamplingParams.extra_args.
        # vllm_xargs is typed as dict[str, str|int|float|list[scalar]], so nested
        # structures (layer_to_heads dict, list-of-ints) must be JSON-encoded as
        # strings. _plugin._patched_generate decodes them back.
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

        _serialize_probes in _plugin.py converts tensors to lists via
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
