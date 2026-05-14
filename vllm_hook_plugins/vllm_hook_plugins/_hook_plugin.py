"""
vLLM general plugin that exposes hidden-state and QK probe outputs via the
OpenAI-compatible API when ``output_hidden_states`` or ``output_qk`` is
passed in ``SamplingParams.extra_args``.

Installed automatically via the ``vllm.general_plugins`` entry point
(configured in setup.py). Patches ``EngineArgs.create_engine_config``
to inject the worker extension and eager mode, and patches
``AsyncLLM.generate`` and ``LLM.generate`` to retrieve per-request probe
outputs for both online (async) and offline (sync) usage.

For ``vllm serve``, also patches the OpenAI response builders so probe
outputs are included in HTTP responses as ``response.probes``.
"""

from __future__ import annotations

import pickle
from collections.abc import AsyncIterator, Callable
from typing import Any

import zstandard as zstd

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

# Populated by register() with the original unpatched methods.
_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None

_WORKER_EXT_HS = "vllm_hook_plugins.workers.probe_hidden_states_worker.ProbeHiddenStatesWorker"
_WORKER_EXT_QK = "vllm_hook_plugins.workers.probe_hookqk_worker.ProbeHookQKWorker"
_WORKER_EXT_STEER = "vllm_hook_plugins.workers.steer_activation_worker.SteerHookActWorker"


def _decompress(data: bytes) -> Any:
    if data[:4] == _ZSTD_MAGIC:
        return pickle.loads(_ZSTD_DECOMPRESSOR.decompress(data))
    return pickle.loads(data)


def _trim_probes(probes: dict, key: str, expected_len: int) -> None:
    """Trim probe tensors to expected_len along the sequence dimension.

    The vLLM v1 scheduler may execute one extra forward pass after the EOS
    stop condition is hit. vLLM discards the extra output token, but our
    capture hooks still fire during that pass. This trims the surplus so
    probe shapes are always deterministic.

    Only applies to tensors with a sequence dim (all_tokens mode). last_token
    tensors are shape (bs, hidden) and have no sequence dim to trim.
    """
    for entry in probes.get(key, {}).values():
        for tkey in ("hidden_states", "q", "k_all"):
            t = entry.get(tkey)
            if t is None:
                continue
            # 3D = (bs, seq, hidden) — trim seq (dim 1)
            if t.dim() == 3 and t.shape[1] > expected_len:
                entry[tkey] = t[:, :expected_len, :]


# ---------------------------------------------------------------------------
# Engine config patch — inject worker extension + eager mode
# ---------------------------------------------------------------------------


def _patched_create_engine_config(self, *args, **kwargs):
    """Inject worker extension and force eager mode before VllmConfig is built."""
    if not self.worker_extension_cls:
        # Default to hidden states worker; users can override via env var.
        import os
        worker_type = os.environ.get("VLLM_HOOK_WORKER", "hidden_states")
        if worker_type == "qk":
            self.worker_extension_cls = _WORKER_EXT_QK
        elif worker_type == "steer":
            self.worker_extension_cls = _WORKER_EXT_STEER
        else:
            self.worker_extension_cls = _WORKER_EXT_HS
    self.enforce_eager = True

    assert _original_create_engine_config is not None
    return _original_create_engine_config(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Generate patch — install hooks and attach probes to output
# ---------------------------------------------------------------------------


async def _patched_generate(
    self,
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    """Wrap AsyncLLM.generate to install hooks and attach probes on finish."""
    # In vLLM v1, the chat endpoint clones SamplingParams into EngineCoreRequest
    # before calling generate(). We must read/modify the clone so our changes take effect.
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:
            effective_params = prompt.sampling_params
    except ImportError:
        pass

    extra = dict(effective_params.extra_args or {})
    # vllm_xargs only allows scalar values, so HookClient JSON-encodes nested
    # structures. Decode them back here before the worker reads extra_args.
    import json as _json
    for _k in ("output_qk", "output_hidden_states"):
        if isinstance(extra.get(_k), str):
            _decoded = _json.loads(extra[_k])
            # output_qk comes back as {str_key: list} — restore int keys
            if _k == "output_qk" and isinstance(_decoded, dict):
                _decoded = {int(k): v for k, v in _decoded.items()}
            extra[_k] = _decoded
    effective_params.extra_args = extra

    wants_hs = extra.get("output_hidden_states") is not None
    wants_qk = extra.get("output_qk") is not None
    wants_steer = bool(extra.get("steer"))
    needs_hooks = wants_hs or wants_qk or wants_steer
    save_to_disk = bool(extra.get("save_to_disk"))

    if needs_hooks and not getattr(self, "_vllm_hook_installed", False):
        await self.collective_rpc("install_hooks")
        setattr(self, "_vllm_hook_installed", True)

    assert _original_generate is not None
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if output.finished and needs_hooks and not wants_steer:
                if save_to_disk:
                    import os
                    run_id = extra.get("run_id") or request_id
                    hook_dir = extra.get("hook_dir") or os.environ.get("VLLM_HOOK_DIR", "")
                    if not hook_dir:
                        raise ValueError(
                            "save_to_disk=True but no hook_dir provided in extra_args "
                            "and VLLM_HOOK_DIR env var is not set."
                        )
                    await self.collective_rpc("flush_disk", args=([request_id], run_id, hook_dir))
                    # Leave output.probes unset — caller reads artifacts from disk.
                else:
                    states = await self.collective_rpc("get_captured_states", args=(request_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    if parts:
                        probes = parts[0]  # rank 0 result; PP merge not yet needed
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes
            yield output
    finally:
        if needs_hooks and not wants_steer and not save_to_disk:
            await self.collective_rpc("clear_captured_states", args=(request_id,))


# ---------------------------------------------------------------------------
# Offline (sync) LLM.generate patch
# ---------------------------------------------------------------------------


def _patched_llm_generate(self, prompts: Any, sampling_params: Any = None, **kwargs) -> list:
    """Wrap LLM.generate to install hooks and dispatch post-generation.

    Each request is dispatched based on its own extra_args:
    - save_to_disk=True -> collective_rpc("flush_disk"); output.probes unset.
    - otherwise        -> collective_rpc("get_captured_states"); attach to output.probes.
    """
    if isinstance(sampling_params, (list, tuple)):
        params_list = list(sampling_params)
    elif sampling_params is not None:
        params_list = [sampling_params]
    else:
        params_list = []

    needs_hooks = any(
        (sp.extra_args or {}).get("output_hidden_states") is not None
        or (sp.extra_args or {}).get("output_qk") is not None
        or bool((sp.extra_args or {}).get("steer"))
        for sp in params_list
    )

    if needs_hooks and not getattr(self, "_vllm_hook_installed", False):
        self.collective_rpc("install_hooks")
        self._vllm_hook_installed = True

    assert _original_llm_generate is not None
    outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)

    if needs_hooks:
        import os

        # First pass: handle RPC (in-memory) requests immediately, and collect
        # disk-save requests grouped by run_id so all requests sharing the same
        # run_id are flushed together — preventing the second flush from
        # overwriting the first when a batch shares one run_id.
        disk_by_run: dict = {}  # run_id -> [(req_id, hook_dir)]

        for idx, output in enumerate(outputs):
            req_id = output.request_id
            sp = params_list[idx] if idx < len(params_list) else params_list[0] if params_list else None
            extra = (sp.extra_args if sp is not None else None) or {}

            wants_artifacts = extra.get("output_hidden_states") is not None or extra.get("output_qk") is not None
            if extra.get("save_to_disk"):
                run_id = extra.get("run_id") or req_id
                hook_dir = extra.get("hook_dir") or os.environ.get("VLLM_HOOK_DIR", "")
                if not hook_dir:
                    raise ValueError(
                        "save_to_disk=True but no hook_dir provided in extra_args "
                        "and VLLM_HOOK_DIR env var is not set."
                    )
                disk_by_run.setdefault(run_id, []).append((req_id, hook_dir))
            elif wants_artifacts:
                states = self.collective_rpc("get_captured_states", args=(req_id,))
                parts = [_decompress(s) for s in states if s is not None]
                if parts:
                    probes = parts[0]
                    n_prompt = len(output.prompt_token_ids)
                    n_gen = len(output.outputs[0].token_ids)
                    expected_len = n_prompt + n_gen - 1
                    _trim_probes(probes, "hs_cache", expected_len)
                    _trim_probes(probes, "qk_cache", expected_len)
                    output.probes = probes

        # Second pass: flush disk-save requests. All req_ids sharing a run_id
        # are sent in one collective_rpc call so the worker merges the full
        # batch into a single artifact — matching old execute_model() behavior.
        for run_id, req_list in disk_by_run.items():
            req_ids = [r for r, _ in req_list]
            _, hook_dir = req_list[0]
            self.collective_rpc("flush_disk", args=(req_ids, run_id, hook_dir))

    return outputs


# ---------------------------------------------------------------------------
# Response builder patches for vllm serve (OpenAI-compatible API)
# ---------------------------------------------------------------------------


def _serialize_probes(probes: dict) -> dict:
    """Serialize probe tensors to lists for JSON transport."""
    import torch
    result = {}
    for key, cache in probes.items():
        # config is a flat dict of scalars — pass through as-is.
        if key == "config" and isinstance(cache, dict):
            result[key] = cache
            continue
        if not isinstance(cache, dict):
            continue
        result[key] = {}
        for mod_name, entry in cache.items():
            result[key][mod_name] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in entry.items()
            }
    return result


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    """Inject serialized probes into completion responses."""
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        probes = getattr(res, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)
            break
    return response


async def _patched_chat_full_generator(self, request, result_generator, *args, **kwargs):
    """Inject serialized probes into chat completion responses."""
    assert _original_chat_full_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    response = await _original_chat_full_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    )

    if last_output is not None and hasattr(response, "model_dump"):
        probes = getattr(last_output, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)

    return response


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup.

    Patches EngineArgs, AsyncLLM.generate, LLM.generate, and the OpenAI
    response builders to enable activation capture via extra_args.

    Usage:
        # Hidden states
        SamplingParams(extra_args={"output_hidden_states": True})
        SamplingParams(extra_args={"output_hidden_states": [layer1, layer2]})

        # QK weights
        SamplingParams(extra_args={"output_qk": True})
        SamplingParams(extra_args={"output_qk": [layer1, layer2]})

    Probe outputs are returned in output.probes and, when using
    vllm serve, injected into the HTTP response body as response.probes.
    """
    global _original_create_engine_config
    global _original_generate, _original_llm_generate
    global _original_completion_response, _original_chat_full_generator

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    # Patch OpenAI-compatible response builders (only available with vllm serve).
    # Module paths differ across vLLM versions; try all known locations.
    for _completion_module in (
        "vllm.entrypoints.openai.completion.serving",   # <0.12
        "vllm.entrypoints.openai.serving_completion",   # ≥0.12
    ):
        try:
            import importlib as _il
            _mod = _il.import_module(_completion_module)
            _cls = _mod.OpenAIServingCompletion
            _original_completion_response = (
                _cls.request_output_to_completion_response
            )
            _cls.request_output_to_completion_response = _patched_completion_response
            break
        except Exception:
            pass

    for _chat_module in (
        "vllm.entrypoints.openai.chat_completion.serving",  # <0.12
        "vllm.entrypoints.openai.serving_chat",             # ≥0.12
    ):
        try:
            import importlib as _il
            _mod = _il.import_module(_chat_module)
            _cls = _mod.OpenAIServingChat
            _original_chat_full_generator = _cls.chat_completion_full_generator
            _cls.chat_completion_full_generator = _patched_chat_full_generator
            break
        except Exception:
            pass
