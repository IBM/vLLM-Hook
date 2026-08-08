"""Shared knobs for the model-agnostic recurrent-depth stack.

Model-specific options (Huginn baseline criteria, decode row-slicing, cache
lookup strategy) live on the adapter, not here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional

# Keys in a ``recurrent_depth`` dict that name classes, not config knobs.
_CLASS_PATH_KEYS = frozenset({"worker", "analyzer"})


@dataclass
class RecurrentDepthConfig:
    """Knobs for :class:`RecurrentDepthWorker` / :class:`RecurrentConvergenceAnalyzer`.

    Exit uses the training-free contraction criterion only. Safety steering is
    independent (Stage 2) and optional.

    Unknown keys (e.g. a custom analyzer's ``kl_threshold``) land in ``extra``
    and are also set as attributes so ``getattr(cfg, "kl_threshold", …)`` works.
    Import-string keys ``worker`` / ``analyzer`` are stripped here and resolved
    by :func:`build_recurrent_stack`.
    """

    # Contraction-rate relative remaining threshold (RECUR_DEPTH ρ).
    # ρ = 0 → nothing exits (exact-match for baseline looped Raven).
    rho: float = 0.0
    min_steps: int = 1  # never exit before this many recurrence steps
    compute_kl: bool = False  # opt-in metric; adapter supplies predict_from_latents
    compute_colsum: bool = False  # Stage 2 (steering); needs Q/K capture

    # ---- Stage 2 scaffolding (optional; unused for exit) ----
    enable_steering: bool = False
    refusal_ids: List[int] = field(default_factory=list)
    harmful_ids: List[int] = field(default_factory=list)

    # Out-of-tree / custom analyzer knobs that are not first-class fields.
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "RecurrentDepthConfig":
        if not d:
            return RecurrentDepthConfig()
        if not isinstance(d, dict):
            if hasattr(d, "items"):
                d = dict(d)
            elif hasattr(d, "__dict__"):
                d = {k: v for k, v in vars(d).items() if not k.startswith("_")}
            else:
                raise TypeError(f"recurrent_depth must be a mapping, got {type(d).__name__}")
        known = {f.name for f in fields(RecurrentDepthConfig)} - {"extra"}
        extra = {k: v for k, v in d.items() if k not in known and k not in _CLASS_PATH_KEYS}
        cfg = RecurrentDepthConfig(
            **{k: v for k, v in d.items() if k in known},
            extra=extra,
        )
        for k, v in extra.items():
            setattr(cfg, k, v)
        return cfg


def load_cls(path: str) -> type:
    """Import a class from ``pkg.mod:Class`` or ``pkg.mod.Class``."""
    if not isinstance(path, str) or not path.strip():
        raise TypeError(f"class path must be a non-empty string, got {path!r}")
    if ":" in path:
        module_name, class_name = path.rsplit(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import recurrent class module {module_name!r}: {exc}"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module {module_name!r} has no attribute {class_name!r}"
        ) from exc


def _coerce_worker(spec: Any, model: Any, cfg: RecurrentDepthConfig) -> Any:
    from vllm_hook_plugins.workers.recurrent_depth_worker import RecurrentDepthWorker

    if spec is None:
        # Default worker
        return RecurrentDepthWorker(model, cfg)
    if isinstance(spec, str):
        spec = load_cls(spec)
    if isinstance(spec, type):
        return spec(model, cfg)
    return spec


def _coerce_analyzer(spec: Any, cfg: RecurrentDepthConfig) -> Any:
    from vllm_hook_plugins.analyzers.recurrent_conv_analyzer import RecurrentConvergenceAnalyzer

    if spec is None:
        # Default analyzer
        return RecurrentConvergenceAnalyzer(cfg)
    if isinstance(spec, str):
        spec = load_cls(spec)
    if isinstance(spec, type):
        return spec(cfg)
    return spec


def build_recurrent_stack(
    model: Any,
    recur_dict: Optional[dict] = None,
    *,
    worker: Any = None,
    analyzer: Any = None,
) -> tuple[RecurrentDepthConfig, Any, Any]:
    """Build ``(cfg, worker, analyzer)`` from a ``recurrent_depth`` dict.

    ``worker`` / ``analyzer`` may be omitted (stock defaults), import strings,
    classes, or already-constructed instances. Explicit ``worker=`` / ``analyzer=``
    kwargs override keys in ``recur_dict``.
    """
    d = dict(recur_dict or {})
    if worker is not None:
        d["worker"] = worker
    if analyzer is not None:
        d["analyzer"] = analyzer
    cfg = RecurrentDepthConfig.from_dict(d)
    return cfg, _coerce_worker(d.get("worker"), model, cfg), _coerce_analyzer(d.get("analyzer"), cfg)
