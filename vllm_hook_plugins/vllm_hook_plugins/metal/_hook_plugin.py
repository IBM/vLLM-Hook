from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

_original_create_engine_config: Callable | None = None
_original_awq_for_model: Callable | None = None
_original_maybe_override_with_speculators: Callable | None = None


def _resolve_metal_worker_class(worker_type: str):
    if worker_type == "qk":
        from vllm_hook_plugins.workers.metal import ProbeHookQKWorkerMetal

        return f"{ProbeHookQKWorkerMetal.__module__}.{ProbeHookQKWorkerMetal.__name__}"
    if worker_type == "steer":
        from vllm_hook_plugins.workers.metal import SteerHookActWorkerMetal

        return f"{SteerHookActWorkerMetal.__module__}.{SteerHookActWorkerMetal.__name__}"
    if worker_type == "hidden_states":
        from vllm_hook_plugins.workers.metal import ProbeHiddenStatesWorkerMetal

        return (
            f"{ProbeHiddenStatesWorkerMetal.__module__}."
            f"{ProbeHiddenStatesWorkerMetal.__name__}"
        )
    raise ValueError(f"Unsupported Metal hook worker type: {worker_type}")


def _patched_create_engine_config(self, *args, **kwargs):
    """Force the Metal worker extension before vLLM finalizes engine config."""
    import os

    worker_type = os.environ.get(
        "HOOK_WORKER",
        os.environ.get("VLLM_HOOK_WORKER", "hidden_states"),
    )
    self.worker_extension_cls = _resolve_metal_worker_class(worker_type)

    assert _original_create_engine_config is not None
    return _original_create_engine_config(self, *args, **kwargs)


def _patched_awq_for_model(cls, model_name: str):
    """Skip AWQ preflight for GGUF refs so the GGUF loader can own them."""
    if Path(str(model_name)).suffix == ".gguf" or ":" in str(model_name):
        return None

    assert _original_awq_for_model is not None
    return _original_awq_for_model(cls, model_name)


def _patched_maybe_override_with_speculators(
    model: str,
    tokenizer: str | None,
    trust_remote_code: bool,
    revision: str | None = None,
    vllm_speculative_config: dict[str, Any] | None = None,
    hf_token: bool | str | None = None,
    **kwargs,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Leave GGUF refs untouched so they do not hit HF config preflight."""
    from vllm.transformers_utils.gguf_utils import is_gguf

    if is_gguf(model):
        return model, tokenizer, vllm_speculative_config

    assert _original_maybe_override_with_speculators is not None
    return _original_maybe_override_with_speculators(
        model,
        tokenizer,
        trust_remote_code,
        revision=revision,
        vllm_speculative_config=vllm_speculative_config,
        hf_token=hf_token,
        **kwargs,
    )


def register() -> None:
    """Install the Metal-specific engine config patch once."""
    global _original_create_engine_config, _original_awq_for_model
    global _original_maybe_override_with_speculators
    if _original_create_engine_config is not None:
        return

    import vllm.engine.arg_utils as arg_utils
    import vllm.transformers_utils.config as config_utils
    from vllm.engine.arg_utils import EngineArgs

    try:
        from vllm_metal.quant.awq_loader import AWQQuantLoader
    except ModuleNotFoundError:
        AWQQuantLoader = None

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    _original_maybe_override_with_speculators = (
        arg_utils.maybe_override_with_speculators
    )
    arg_utils.maybe_override_with_speculators = (
        _patched_maybe_override_with_speculators
    )
    config_utils.maybe_override_with_speculators = (
        _patched_maybe_override_with_speculators
    )

    if AWQQuantLoader is not None:
        _original_awq_for_model = AWQQuantLoader.for_model.__func__
        AWQQuantLoader.for_model = classmethod(_patched_awq_for_model)
