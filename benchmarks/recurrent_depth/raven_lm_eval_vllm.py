"""lm-eval vLLM backend for AdaptiveRavenForvLLM (production / Hook path).

Registers ``adaptive_raven_vllm``. Builds stock lm-eval ``VLLM`` with
``hf_overrides`` so the OOT executor + contraction exit load under vLLM.

Requires GPU + ``vllm`` + ``lm_eval``. Call ``register_adaptive_raven()`` is
done inside this module before ``LLM(...)``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PLUGINS = _ROOT / "vllm_hook_plugins"
if str(_PLUGINS) not in sys.path:
    sys.path.insert(0, str(_PLUGINS))

# Must precede vLLM imports (matches examples/demo_recurrent_depth.py).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM

from model_adapters.vllm import ADAPTIVE_RAVEN_ARCH, register_adaptive_raven

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


def _exit_stats_from_samples(
    xs: List[float],
    *,
    rho: float,
    num_steps_cap: Optional[int],
) -> dict:
    if not xs:
        return {
            "n_forwards": 0,
            "mean_effective_r": None,
            "min_effective_r": None,
            "max_effective_r": None,
            "rho": rho,
            "num_steps_cap": num_steps_cap,
            "baseline": None,
        }
    return {
        "n_forwards": len(xs),
        "mean_effective_r": sum(xs) / len(xs),
        "min_effective_r": min(xs),
        "max_effective_r": max(xs),
        "rho": rho,
        "num_steps_cap": num_steps_cap,
        "baseline": None,
    }


@register_model("adaptive_raven_vllm")
class AdaptiveRavenVLLMLM(VLLM):
    """lm-eval ``VLLM`` subclass that loads ``AdaptiveRavenForvLLM``.

    model_args (CLI strings ok)::

        pretrained=tomg-group-umd/huginn-0125
        rho=0.02
        num_steps=32
        min_steps=1
        dtype=bfloat16
        gpu_memory_utilization=0.85
        max_model_len=4096
    """

    def __init__(
        self,
        pretrained: str = "tomg-group-umd/huginn-0125",
        rho: Union[float, str] = 0.0,
        min_steps: Union[int, str] = 1,
        num_steps: Union[int, str, None] = None,
        trust_remote_code: Union[bool, str] = True,
        enforce_eager: Union[bool, str] = True,
        gpu_memory_utilization: Union[float, str] = 0.85,
        dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        register_adaptive_raven()

        self._rho = float(rho)
        self._min_steps = int(min_steps)
        self._num_steps = _as_optional_int(num_steps)
        self._max_steps_runtime = self._num_steps
        self.exit_depth_samples: List[float] = []

        # Drop HF-only knobs if a shared CLI forwarded them.
        kwargs.pop("baseline", None)
        kwargs.pop("slice_decode", None)
        kwargs.pop("device", None)

        overrides = kwargs.pop("hf_overrides", None)
        if not isinstance(overrides, dict):
            overrides = {}
        overrides = dict(overrides)
        overrides["architectures"] = [ADAPTIVE_RAVEN_ARCH]
        overrides["recurrent_depth"] = {
            "rho": self._rho,
            "min_steps": self._min_steps,
        }
        if self._num_steps is not None:
            overrides["mean_recurrence"] = int(self._num_steps)

        kwargs["hf_overrides"] = overrides
        kwargs["enforce_eager"] = _as_bool(enforce_eager, True)
        kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)

        super().__init__(
            pretrained=pretrained,
            trust_remote_code=_as_bool(trust_remote_code, True),
            dtype=dtype,
            **kwargs,
        )

        if self._max_steps_runtime is None:
            self._max_steps_runtime = int(
                getattr(self._config, "mean_recurrence", 32) or 32
            )

        eval_logger.info(
            "AdaptiveRavenVLLMLM ready: arch=%s rho=%s min_steps=%s num_steps=%s "
            "mean_recurrence=%s",
            ADAPTIVE_RAVEN_ARCH,
            self._rho,
            self._min_steps,
            self._num_steps,
            self._max_steps_runtime,
        )

    def reset_exit_stats(self) -> None:
        self.exit_depth_samples.clear()
        try:
            self.model.apply_model(
                lambda m: m.reset_exit_depth_samples()
                if hasattr(m, "reset_exit_depth_samples")
                else None
            )
        except Exception as e:  # noqa: BLE001 — best-effort across vLLM versions
            eval_logger.debug("reset_exit_stats apply_model failed: %s", e)

    def _harvest_exit_samples(self) -> None:
        try:
            harvested = self.model.apply_model(
                lambda m: m.pop_exit_depth_samples()
                if hasattr(m, "pop_exit_depth_samples")
                else []
            )
        except Exception as e:  # noqa: BLE001
            eval_logger.debug("harvest exit samples failed: %s", e)
            return
        # apply_model returns list[R] (one entry per worker).
        if not harvested:
            return
        for part in harvested:
            if isinstance(part, list):
                self.exit_depth_samples.extend(part)

    def exit_stats(self) -> dict:
        self._harvest_exit_samples()
        stats = _exit_stats_from_samples(
            self.exit_depth_samples,
            rho=self._rho,
            num_steps_cap=self._max_steps_runtime,
        )
        # Fixed rho=0 with no harvested samples → report configured depth.
        if (
            stats["mean_effective_r"] is None
            and self._rho == 0.0
            and self._max_steps_runtime is not None
        ):
            stats["mean_effective_r"] = float(self._max_steps_runtime)
            stats["min_effective_r"] = float(self._max_steps_runtime)
            stats["max_effective_r"] = float(self._max_steps_runtime)
        return stats
