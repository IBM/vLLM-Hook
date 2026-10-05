"""MiaClient: OpenAI-compatible client for vllm serve with probe capture and analysis."""
from __future__ import annotations

import copy
import glob
import inspect
import json
import os
import shutil
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from mia._profiler import PROF
from mia.errors import MiaDeliveryError
from mia.registry import PluginRegistry
from mia.run_utils import dispatch_disk_analyze


_UNSET = object()


class MiaClient:
    """OpenAI-compatible client for a MIA ``vllm serve``: capture/steer requests and analysis."""

    def __init__(
        self,
        base_url: str,
        analyzer_name: str,
        config_file: str,
        api_key: str = "EMPTY",
        hook_dir: str = None,
        tokenizer_for: Optional[str] = None,
    ):
        """Connect to a server.

        Args:
            base_url: the server's OpenAI base URL, e.g. ``http://localhost:8000/v1``.
            analyzer_name: a registered analyzer for ``analyze()``.
            config_file: JSON with the capture settings each request sends.
            api_key: the server's API key.
            hook_dir: where the server writes ``save_to_disk`` runs (default ``/dev/shm/mia``).
            tokenizer_for: model id whose tokenizer ``.tokenizer`` loads.
        """
        # lazy: import cycle (mia/__init__ imports mia.client); optional dependency (openai)
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
        # Hybrid save_to_disk runs: delivered keys, reads and write nonces per run_id.
        self._run_keys: Dict[str, List[str]] = {}
        self._run_reads: Dict[str, List[tuple]] = {}
        self._run_nonces: Dict[str, Dict[str, str]] = {}


    #: The only vllm_xargs keys the plugin JSON-decodes; a dict/list under any other is refused.
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
        """Merge MIA's `vllm_xargs` into the caller's extra_body (e.g. `return_token_ids`)."""
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
        self._attach_delivery(response, run_id)
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
        """Send a chat completion with capture; the server applies the chat template.

        Args:
            messages, model: as for ``chat.completions.create``.
            save_to_disk: True writes the capture to ``<hook_dir>/<run_id>/``, False returns it on
                ``probes``; None (default) omits the setting and the server decides.
            run_id: the run's name (default: a new uuid).
            extra_xargs: per-request settings the config file lacks, e.g. ``{"hooks_on": "both"}``.
            steer: a steering config for this request alone.
            capture: False sends a plain request that arms nothing (offline ``use_hook=False``).
            extra_body: the caller's own extra_body, merged with MIA's.
        Returns:
            The OpenAI response; its capture is ``response.probes``.
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
        """Capture against exact token ids via the completions endpoint (no chat template).

        Takes one sequence (``[int, ...]``) or a batch (``[[int, ...], ...]``) sharing one
        ``run_id``; other arguments as for :meth:`generate`.
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
        """Capture against raw, already-templated text via the completions endpoint.

        Takes one string or a list sharing one ``run_id``; other arguments as for :meth:`generate`.
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
        """The served model's tokenizer (loaded locally on first use), for an analyzer's spans."""
        if self._tokenizer is None:
            if not self._tokenizer_for:
                raise RuntimeError(
                    "no tokenizer bound: construct the client with "
                    "tokenizer_for=<model id>, or load one yourself with "
                    "transformers.AutoTokenizer.")
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_for)
        return self._tokenizer

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
        probes: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Run the analyzer on the last response, on ``probes`` you hold, or on disk runs."""
        if probes is not None:
            with PROF.timed("analyzer.kernel"):
                return self.analyzer.analyze(analyzer_spec, probes=probes)

        if self._last_response is None:
            raise RuntimeError("No generate() call has been made yet.")

        raw_probes = getattr(self._last_response, "probes", None)
        if raw_probes is None and any(r in self._run_keys
                                      for r in (run_ids or [run_id or self._last_run_id])):
            return self._analyze_delivered_run(analyzer_spec, run_id or self._last_run_id,
                                               run_ids)
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


    def _attach_delivery(self, response, run_id: str) -> None:
        """A hybrid server's response: lazy ``probes``, or with save_to_disk the run's keys."""
        # lazy: keep mia.graph (reads env at import) out of import mia
        from mia.graph.delivered_probes import attach_lazy, check_id, external_id

        extra = getattr(response, "__pydantic_extra__", None)
        mark = extra.pop("mia", None) if isinstance(extra, dict) else None
        if not isinstance(mark, dict) or mark.get("delivery") != "hybrid":
            return
        rid, kind, items = getattr(response, "id", None), mark.get("kind"), mark.get("items") or []
        try:
            check_id(rid)
        except ValueError:
            def unreadable():
                raise MiaDeliveryError(f"response id {rid!r} cannot be read back "
                                       f"(use ids of A-Za-z0-9._:- only)")
            attach_lazy(response, unreadable)
            return
        if any(it.get("save_to_disk") is True for it in items):
            self._run_keys[run_id], self._run_reads[run_id] = [], []
            nonces = self._run_nonces.setdefault(run_id, {})
            nonces.clear()
            for it in items:
                ext, n = external_id(rid, kind, int(it["item"])), int(it["n"])
                keys = [ext] if n == 1 else [f"{j}_{ext}" for j in range(n)]
                self._run_keys[run_id].extend(keys)
                self._run_reads[run_id].append((rid, kind, it))
                if it.get("nonce"):
                    nonces.update({k: str(it["nonce"]) for k in keys})
            return
        fetch = {int(it["item"]): _ItemFetch(self, rid, kind, it) for it in items}
        n = int(items[0]["n"]) if items else 1
        targets = [(response, lambda: fetch[0]()[0])]
        if len(items) > 1 or n > 1:
            for c, ch in enumerate(getattr(response, "choices", None) or []):
                c = int(getattr(ch, "index", c))
                targets.append((ch, lambda i=c // n, j=c % n: fetch[i]()[j]))
        try:
            for obj, loader in targets:
                attach_lazy(obj, loader)
        except Exception as e:  # noqa: BLE001
            print(f"[mia] response.probes read eagerly ({e!r})", flush=True)
            for obj, loader in targets:
                setattr(obj, "probes", loader())

    def _read_delivered(self, rid: str, kind: str, it: dict):
        """``(samples, keys, names, config)`` of one response item, from the server's route."""
        # lazy: optional httpx/openai; mia.graph out of import mia; patchable DELIVERY_HARD_CAP_S
        import httpx
        import openai
        from mia.graph.aperture_gather import DELIVERY_HARD_CAP_S
        from mia.graph.delivered_probes import DeliveryReadTimeout, decode_delivery

        params = {"id": rid, "item": int(it["item"]), "n": int(it["n"]), "kind": kind}
        if it.get("key"):
            params["key"] = str(it["key"])
        if it.get("layers"):
            params["layers"] = ",".join(str(int(x)) for x in it["layers"])
        try:
            with PROF.timed("client.delivered_read"):
                r = self._openai.get("mia/delivered", cast_to=httpx.Response, options={
                    "params": params, "max_retries": 0, "timeout": DELIVERY_HARD_CAP_S + 60})
        except openai.APIStatusError as e:
            try:
                msg = e.response.json().get("error") or e.response.text
            except Exception:  # noqa: BLE001
                msg = e.response.text
            if e.status_code == 504:
                raise DeliveryReadTimeout(f"{rid!r}: {msg}", missing={rid: msg}) from None
            raise MiaDeliveryError(
                f"delivered probes for {rid!r}: HTTP {e.status_code}: {msg}") from None
        return decode_delivery(r.content)

    def _analyze_delivered_run(self, analyzer_spec, run_id, run_ids):
        """Disk analyze of hybrid save_to_disk runs, once each run holds its requests."""
        # lazy: keep mia.graph out of import mia; hs_probes patchable at call time
        from mia.graph import run_artifact
        from mia.graph.aperture_gather import delivery_timeout_s
        from mia.graph.delivered_probes import hs_probes, merge_disk

        nonces = {r: self._run_nonces.get(r) or None for r in (run_ids or [run_id])}
        # As long as the server's writer may wait for a delivery.
        idle = max(run_artifact.artifact_wait_s(), delivery_timeout_s())
        for rid_ in (run_ids or [run_id]):
            try:
                run_artifact.wait_run(self._hook_dir, rid_, self._run_keys.get(rid_, []),
                                      timeout_s=idle, nonces=nonces[rid_])
            except run_artifact.RunArtifactError:
                remote = not os.path.isdir(run_artifact.run_dir(self._hook_dir, rid_))
                takes = "probes" in inspect.signature(self.analyzer.analyze).parameters
                if not (remote and takes and not run_ids):
                    raise
                cache = None
                for rid, kind, it in self._run_reads.get(rid_, []):
                    samples, _k, names, config = self._read_delivered(rid, kind, it)
                    for j, rows in enumerate(samples):
                        if rows:
                            cache = merge_disk(cache, hs_probes(
                                rows, {**_item_meta(it, j), "config": config}, names,
                                layout="disk"))
                with PROF.timed("analyzer.kernel"):
                    return self.analyzer.analyze(analyzer_spec, probes=cache)
        runs = run_ids or [run_id]
        while True:
            with run_artifact.read_lock(self._hook_dir, runs):
                if all(not nonces[r] or run_artifact.holds(
                        run_artifact.read_manifest(self._hook_dir, r),
                        self._run_keys.get(r, []), nonces[r]) for r in runs):
                    snap = run_artifact.snapshot(self._hook_dir, runs)
                    if snap is None:
                        return dispatch_disk_analyze(self.analyzer, analyzer_spec,
                                                     run_id=run_id, run_ids=run_ids)
                    break
            for r in runs:
                run_artifact.wait_run(self._hook_dir, r, self._run_keys.get(r, []),
                                      timeout_s=idle, nonces=nonces[r])
        an = copy.copy(self.analyzer)
        an.hook_dir = snap
        before = dict(vars(an))
        try:
            return dispatch_disk_analyze(an, analyzer_spec, run_id=run_id, run_ids=run_ids)
        finally:
            shutil.rmtree(snap, ignore_errors=True)
            for k, v in vars(an).items():
                if k != "hook_dir" and before.get(k, _UNSET) is not v:
                    setattr(self.analyzer, k, v)

    def _wait_artifact_dir(self, run_id, timeout_s: float = 10.0, poll_s: float = 0.05) -> bool:
        base = os.path.join(self._hook_dir, run_id)
        prev = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            files = sorted(glob.glob(os.path.join(base, "**", "*"), recursive=True))
            files = [f for f in files if os.path.isfile(f) and not f.endswith(".tmp")
                     and not os.path.basename(f).startswith(".")]
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
        # A steering config asks for no capture.
        self._captures = "steering" not in cfg

        if "params" in cfg and "important_heads" in cfg["params"]:
            for layer_idx, head_idx in cfg["params"]["important_heads"]:
                self.layer_to_heads.setdefault(layer_idx, []).append(head_idx)

        # Same defaults as MiaLLM.load_config.
        hookq = cfg.get("hookq", {})
        self._wants_qk = bool(self.layer_to_heads) or "hookq" in cfg
        self._hookq_mode = hookq.get("hookq_mode", "all_tokens")
        self._qk_capture = hookq.get("capture", "qk")
        self._score_head = int(hookq.get("score_head", 0))
        self._hs_mode = "last_token"

        if "hidden_states" in cfg:
            layers = cfg["hidden_states"].get("layers", [])
            self._output_layers = layers if layers else True
            self._hs_mode = cfg["hidden_states"].get("mode", "last_token")

    def _build_extra_body(self) -> Dict:
        if not self._captures:
            xargs = {}
        elif self._output_layers is not None or not self._wants_qk:
            layers = self._output_layers if self._output_layers is not None else True
            xargs = {"output_hidden_states": json.dumps(layers) if isinstance(layers, list) else layers,
                     "hs_mode": self._hs_mode}
        else:
            xargs = {
                "output_qk": (json.dumps({str(k): v for k, v in self.layer_to_heads.items()})
                              if self.layer_to_heads else True),
                "hookq_mode": self._hookq_mode,
            }
            if self._qk_capture == "score":
                xargs.update(qk_capture="score", score_head=self._score_head)
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


def _item_meta(it: dict, j: int) -> dict:
    per = it.get("n_cached_samples")
    return {"hs_mode": it["hs_mode"], "hooks_on": it["hooks_on"], "n_prompt": it["n_prompt"],
            "n_gen": it["n_gen"][j],
            "n_cached": per[j] if isinstance(per, list) else it.get("n_cached", 0)}


class _ItemFetch:
    """One response item's samples as client-visible probes, read once on first use."""

    def __init__(self, client: "MiaClient", rid: str, kind: str, it: dict):
        self._client, self._rid, self._kind, self._it = client, rid, kind, it
        self._lock = threading.Lock()
        self._value: Optional[list] = None

    def __call__(self) -> list:
        # lazy: keep mia.graph (reads env at import) out of import mia
        from mia.graph.delivered_probes import client_probes
        with self._lock:
            if self._value is None:
                samples, _k, names, config = self._client._read_delivered(
                    self._rid, self._kind, self._it)
                with PROF.timed("client.delivered_convert"):
                    self._value = [client_probes(rows, {**_item_meta(self._it, j),
                                                        "config": config}, names) if rows else None
                                   for j, rows in enumerate(samples)]
            return self._value
