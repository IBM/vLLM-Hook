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

# pyright: reportUnknownVariableType=false
# pyright: reportCallIssue=false
# pyright: reportAttributeAccessIssue=false

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


def _as_optional_int(v: Any) -> Optional[int]:
    if v is None or str(v).strip().lower() in {"", "none", "null"}:
        return None
    return int(v)


def _exit_stats_from_samples(
    records: List[dict],
    *,
    rho: float,
    num_steps_cap: Optional[int],
) -> dict:
    """Aggregate per-forward compute records into Pareto-ready statistics.

    ``mean_effective_r_decode`` is the primary cost axis: adaptive exit is
    decode-only, so prefill rows pinned at the cap would otherwise mask the
    savings. ``mean_effective_r`` keeps the legacy per-forward mean over all
    rows for comparison with older result files.
    """
    stats = {
        "n_forwards": len(records),
        "n_tokens": 0,
        "n_decode_tokens": 0,
        "mean_effective_r_decode": None,
        "min_effective_r_decode": None,
        "max_effective_r_decode": None,
        "mean_effective_r_token": None,
        "mean_effective_r": None,
        "min_effective_r": None,
        "max_effective_r": None,
        "attn_token_steps": 0,
        "mlp_token_steps": 0,
        "rho": rho,
        "num_steps_cap": num_steps_cap,
        "baseline": None,
    }
    if not records:
        return stats

    tokens = sum(int(r["n_tokens"]) for r in records)
    decode_tokens = sum(int(r["n_decode_tokens"]) for r in records)
    per_forward_means = [
        int(r["depth_sum"]) / int(r["n_tokens"]) for r in records if int(r["n_tokens"])
    ]
    decode_mins = [r["decode_min_depth"] for r in records if r["decode_min_depth"] is not None]
    decode_maxes = [r["decode_max_depth"] for r in records if r["decode_max_depth"] is not None]

    stats.update(
        {
            "n_tokens": tokens,
            "n_decode_tokens": decode_tokens,
            "attn_token_steps": sum(int(r["attn_token_steps"]) for r in records),
            "mlp_token_steps": sum(int(r["mlp_token_steps"]) for r in records),
        }
    )
    if decode_tokens:
        decode_depth_sum = sum(int(r["decode_depth_sum"]) for r in records)
        stats["mean_effective_r_decode"] = decode_depth_sum / decode_tokens
        stats["min_effective_r_decode"] = float(min(decode_mins))
        stats["max_effective_r_decode"] = float(max(decode_maxes))
    if tokens:
        stats["mean_effective_r_token"] = (
            sum(int(r["depth_sum"]) for r in records) / tokens
        )
    if per_forward_means:
        stats["mean_effective_r"] = sum(per_forward_means) / len(per_forward_means)
        stats["min_effective_r"] = min(per_forward_means)
        stats["max_effective_r"] = max(per_forward_means)
    return stats


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
        capture_trajectory: Union[bool, str] = False,
        compute_prediction_metrics: Union[bool, str] = False,
        policy: Optional[dict] = None,
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
        self._capture_trajectory = capture_trajectory
        self._compute_prediction_metrics = compute_prediction_metrics
        self._policy = policy
        self.exit_depth_samples: List[dict] = []
        self.trajectory_samples: List[dict] = []

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
            "capture_trajectory": self._capture_trajectory,
            "compute_prediction_metrics": self._compute_prediction_metrics,
        }
        if self._policy is not None:
            overrides["recurrent_depth"]["policy"] = self._policy
        if self._num_steps is not None:
            overrides["mean_recurrence"] = int(self._num_steps)

        kwargs["hf_overrides"] = overrides
        kwargs["enforce_eager"] = enforce_eager if enforce_eager is not None else True
        kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)

        super().__init__(
            pretrained=pretrained,
            trust_remote_code=trust_remote_code if trust_remote_code is not None else True,
            dtype=dtype,
            **kwargs,
        )

        self._max_steps_runtime = int(
            getattr(self._config, "mean_recurrence", 32) or 32
        ) if self._max_steps_runtime is None else self._max_steps_runtime

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
        self.trajectory_samples.clear()
        try:
            self.model.apply_model(
                lambda m: (
                    m.reset_exit_depth_samples(),
                    m.reset_trajectory_samples(),
                )
                if (
                    hasattr(m, "reset_exit_depth_samples")
                    and hasattr(m, "reset_trajectory_samples")
                )
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
        except Exception as e:
            eval_logger.debug("harvest exit samples failed: %s", e)
            return
        if not harvested:
            return
        for part in harvested:
            if isinstance(part, list):
                self.exit_depth_samples.extend(part)

    def _harvest_trajectory_samples(self) -> None:
        if not self._capture_trajectory:
            return
        try:
            harvested = self.model.apply_model(
                lambda m: m.pop_trajectory_samples()
                if hasattr(m, "pop_trajectory_samples")
                else []
            )
        except Exception as e:
            eval_logger.debug("harvest trajectory samples failed: %s", e)
            return
        for part in harvested or []:
            if isinstance(part, list):
                self.trajectory_samples.extend(part)

    def take_trajectory_samples(self) -> List[dict]:
        """Return and clear harvested flat trajectory rows for parquet logging."""
        self._harvest_trajectory_samples()
        rows = list(self.trajectory_samples)
        self.trajectory_samples.clear()
        return rows

    def exit_stats(self) -> dict:
        self._harvest_exit_samples()
        stats = _exit_stats_from_samples(
            self.exit_depth_samples,
            rho=self._rho,
            num_steps_cap=self._max_steps_runtime,
        )
        # Fixed rho=0 with no harvested records → report the configured depth.
        if (
            stats["mean_effective_r"] is None
            and self._rho == 0.0
            and self._max_steps_runtime is not None
        ):
            depth = float(self._max_steps_runtime)
            for key in (
                "mean_effective_r",
                "min_effective_r",
                "max_effective_r",
                "mean_effective_r_token",
                "mean_effective_r_decode",
                "min_effective_r_decode",
                "max_effective_r_decode",
            ):
                stats[key] = depth
        return stats
