"""lm-eval HFLM wrapper for AdaptiveRaven (fixed depth + contraction exit).

Registers model name ``adaptive_raven`` for use with ``lm_eval`` / ``run_lm_eval.py``.

Requires:
    pip install "lm_eval[hf]" datasets
    transformers==4.51.0   # Raven / Huginn pin used in this repo

CLI model_args (all strings from the harness) include::

    pretrained=tomg-group-umd/huginn-0125
    rho=0.0                 # 0 → no exit (fixed-depth / oracle)
    num_steps=32            # caps recurrence; also sets config.mean_recurrence
    min_steps=1
    baseline=latent-diff    # optional Huginn native criterion (HF path)
    dtype=bfloat16
    trust_remote_code=True

Pareto arms
-----------
* Fixed depth r:  ``rho=0,num_steps=r``
* Adaptive ρ:     ``rho=ρ,num_steps=r_max`` (usually 32)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

# Repo root + plugin package on path (same pattern as examples/demo_recurrent_depth.py).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PLUGINS = _ROOT / "vllm_hook_plugins"
if str(_PLUGINS) not in sys.path:
    sys.path.insert(0, str(_PLUGINS))

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import get_dtype

from model_adapters.hf import AdaptiveRavenForCausalLM, RavenAdapterConfig
from vllm_hook_plugins.protocols.recurrent_depth import attach_recurrent_depth  # pyright: ignore[reportMissingImports]

eval_logger = logging.getLogger(__name__)


def _as_bool(v: Union[bool, str, None], default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _as_optional_int(v: Any) -> Optional[int]:
    if v is None or str(v).strip().lower() in {"", "none", "null"}:
        return None
    return int(v)


def _as_optional_str(v: Any) -> Optional[str]:
    if v is None or str(v).strip().lower() in {"", "none", "null"}:
        return None
    return str(v)


def effective_recurrence(
    exit_iteration: Optional[torch.Tensor],
    max_steps: int,
) -> Optional[torch.Tensor]:
    """Map ExitController indices to per-token recurrence count.

    ``exit_iteration == k`` (≥0) means the token finished core step ``k``
    (0-based) and stopped → effective depth ``k + 1``. ``-1`` means never
    exited → ``max_steps``.
    """
    if exit_iteration is None:
        return None
    out = exit_iteration.to(dtype=torch.long).clone()
    never = out < 0
    out = out + 1
    out[never] = int(max_steps)
    return out


@register_model("adaptive_raven")
class AdaptiveRavenLM(HFLM):
    """HuggingFace lm-eval backend that loads ``AdaptiveRavenForCausalLM``.

    Tracks mean effective recurrence from ``last_exit_iteration`` after each
    forward (sidecar efficiency metric for Pareto plots; not used by lm_eval
    scoring itself).
    """

    def __init__(
        self,
        pretrained: str = "tomg-group-umd/huginn-0125",
        rho: Union[float, str] = 0.0,
        min_steps: Union[int, str] = 1,
        num_steps: Union[int, str, None] = None,
        baseline: Union[str, None] = None,
        slice_decode: Union[bool, str] = True,
        trust_remote_code: Union[bool, str] = True,
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        **kwargs,
    ) -> None:
        self._rho = float(rho)
        self._min_steps = int(min_steps)
        self._num_steps = _as_optional_int(num_steps)
        self._baseline = _as_optional_str(baseline)
        self._slice_decode = _as_bool(slice_decode, True)
        self._proto = None
        self._max_steps_runtime: Optional[int] = None
        # Running stats for efficiency axis (quality comes from lm_eval).
        self.exit_depth_samples: List[float] = []

        super().__init__(
            pretrained=pretrained,
            trust_remote_code=_as_bool(trust_remote_code, True),
            dtype=dtype,
            **kwargs,
        )

        self._install_exit_probe()
        eval_logger.info(
            "AdaptiveRavenLM ready: rho=%s min_steps=%s num_steps=%s baseline=%s "
            "mean_recurrence=%s",
            self._rho,
            self._min_steps,
            self._num_steps,
            self._baseline,
            getattr(self.config, "mean_recurrence", None),
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        if peft or delta or autogptq or gptqmodel or gguf_file:
            raise ValueError(
                "AdaptiveRavenLM does not support peft/delta/autogptq/gptqmodel/gguf; "
                "load a full Raven / Huginn checkpoint."
            )

        model_kwargs = dict(kwargs) if kwargs else {}
        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=model_kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        self._model = AdaptiveRavenForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=bool(trust_remote_code),
            **model_kwargs,
        )

        if self._num_steps is not None:
            # Fixed-depth arm without threading num_steps through every HFLM call.
            self._model.config.mean_recurrence = int(self._num_steps)

        self._max_steps_runtime = int(self._model.config.mean_recurrence)

        raven_cfg = RavenAdapterConfig(
            slice_decode=self._slice_decode,
            baseline_criterion=self._baseline,
            cache_lookup_strategy="latest-m4",
        )
        self._proto = attach_recurrent_depth(
            self._model,
            rho=self._rho,
            min_steps=self._min_steps,
            raven_cfg=raven_cfg,
        )

    def _install_exit_probe(self) -> None:
        """Record effective recurrence after each forward that updates exits."""
        model = self._model
        if getattr(model, "_adaptive_raven_exit_probe", False):
            return
        max_steps = int(
            self._max_steps_runtime
            or getattr(model.config, "mean_recurrence", 0)
            or 0
        )
        lm = self
        orig_forward = model.forward

        def forward_with_probe(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            exits = getattr(model, "last_exit_iteration", None)
            depths = effective_recurrence(exits, max_steps)
            if depths is not None:
                # Decode / last-token focus: mean over batch × seq for this call.
                lm.exit_depth_samples.append(float(depths.float().mean().item()))
            return out

        model.forward = forward_with_probe  # type: ignore[method-assign]
        model._adaptive_raven_exit_probe = True

    def reset_exit_stats(self) -> None:
        self.exit_depth_samples.clear()

    def exit_stats(self) -> dict:
        xs = self.exit_depth_samples
        if not xs:
            return {
                "n_forwards": 0,
                "mean_effective_r": None,
                "min_effective_r": None,
                "max_effective_r": None,
                "rho": self._rho,
                "num_steps_cap": self._max_steps_runtime,
                "baseline": self._baseline,
            }
        return {
            "n_forwards": len(xs),
            "mean_effective_r": sum(xs) / len(xs),
            "min_effective_r": min(xs),
            "max_effective_r": max(xs),
            "rho": self._rho,
            "num_steps_cap": self._max_steps_runtime,
            "baseline": self._baseline,
        }
