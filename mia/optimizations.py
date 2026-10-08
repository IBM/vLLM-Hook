"""The public optimization levers (PUBLIC_LEVERS), set from env or a config file."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

PUBLIC_LEVERS: Dict[str, Tuple[str, str, str]] = {
    "steer_fused": (
        "MIA_STEER_FUSED", "on",
        "Run the steering op as one fused kernel; 0 gives bit-reproducible steering. Default on.",
    ),
    "compact_kall": (
        "MIA_QK_COMPACT_KALL", "auto",
        "Q/K only: send each request's keys once instead of padded per-row copies. Default auto.",
    ),
    "writer_process": (
        "MIA_WRITER_PROCESS", "on",
        "Disk path only: serialize and write artifacts in a child process. Default on.",
    ),
    "storage_router": (
        "MIA_STORAGE_ROUTER", "on",
        "Server only: a request with no save_to_disk goes to RPC or disk by its size. Default on.",
    ),
    "artifact_dtype": (
        "MIA_ARTIFACT_DTYPE", "native",
        "Quantize saved artifacts (int2, int4, int8, fp8_e4m3, fp8_e5m2, bf16, fp16, fp32); "
        "lossy. Default native (no quantization).",
    ),
    "aperture_mmap": (
        "MIA_APERTURE_MMAP", "off",
        "Write capture files through a memory map; only with MIA_APERTURE_WRITE_MODE=legacy. "
        "Default off.",
    ),
    "aperture_max_batched_tokens": (
        "MIA_APERTURE_MAX_BATCHED_TOKENS", "off",
        "Graph mode: lower max_num_batched_tokens (auto, or an int) so a capture step fits in "
        "GPU memory; never raises it. Default off.",
    ),
}

# Removed levers and why, so a stale config gets one clear message.
_REMOVED = {
    "batched_egress": "it had no effect (nothing read MIA_BATCHED_EGRESS); delete it from the config",
}

_TRUE = ("1", "true", "on", "yes")
_FALSE = ("0", "false", "off", "no")
_DEFER = ("auto", "default", "native")


def _to_env_value(key: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if key == "aperture_max_batched_tokens":
        if isinstance(value, bool):
            return "auto" if value else None
        s = str(value).strip().lower()
        if s in _FALSE:
            return None
        if s in _DEFER or s in _TRUE:
            return "auto"
        return str(value)
    if isinstance(value, bool):
        truthy = value
    else:
        s = str(value).strip().lower()
        if s in _DEFER:
            return None
        if s in _TRUE:
            truthy = True
        elif s in _FALSE:
            truthy = False
        else:
            return str(value)
    if key == "artifact_dtype":
        return "1" if truthy else None
    return "1" if truthy else "0"


def apply_optimizations(config_data: Mapping[str, Any]) -> Dict[str, str]:
    """Apply a config file's ``optimizations`` block to the environment."""
    opts = (config_data or {}).get("optimizations") or {}
    if not isinstance(opts, dict):
        raise ValueError(
            f"config 'optimizations' must be an object, got {type(opts).__name__}")

    gone = sorted(set(opts) & set(_REMOVED))
    if gone:
        raise ValueError("; ".join(
            f"optimization {k!r} was removed: {_REMOVED[k]}" for k in gone))

    unknown = sorted(set(opts) - set(PUBLIC_LEVERS))
    if unknown:
        raise ValueError(
            f"unknown optimization(s) {unknown}; supported: {sorted(PUBLIC_LEVERS)}. "
            "Advanced/diagnostic knobs are env-only by design -- see optimizations.py.")

    applied: Dict[str, str] = {}
    for key, value in opts.items():
        env_name = PUBLIC_LEVERS[key][0]
        env_value = _to_env_value(key, value)
        if env_value is None:
            continue
        if env_name in os.environ:
            continue
        os.environ[env_name] = env_value
        applied[key] = env_value
    return applied


def env_is_on(key: str) -> bool:
    """Resolve a boolean public lever: env if set, else the shipped default."""
    env_name, default, _doc = PUBLIC_LEVERS[key]
    raw = os.environ.get(env_name)
    if raw is None:
        return default == "on"
    return raw.strip().lower() in _TRUE


def describe() -> str:
    """One-screen reference of the public levers and their shipped defaults."""
    width = max(len(k) for k in PUBLIC_LEVERS)
    lines = ["optimization".ljust(width) + "  default   env var",
             "-" * (width + 40)]
    for key, (env_name, default, _doc) in PUBLIC_LEVERS.items():
        lines.append(f"{key.ljust(width)}  {default.ljust(8)}  {env_name}")
    return "\n".join(lines)


__all__ = ["PUBLIC_LEVERS", "apply_optimizations", "env_is_on", "describe"]

