"""The public optimization levers (PUBLIC_LEVERS), set from env or a config file."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

PUBLIC_LEVERS: Dict[str, Tuple[str, str, str]] = {
    "batched_egress": (
        "MIA_BATCHED_EGRESS", "on",
        "Reduce-then-free egress: ONE index_select per layer instead of one clone per "
        "(layer, request). -2.9..-3.9% decode ms/step in 3/3 regimes. Byte-identical.",
    ),
    "steer_fused": (
        "MIA_STEER_FUSED", "on",
        "Fuse the buffer-mode steer op into one Triton kernel. Idle decode tax 0.56 -> 0.10 "
        "ms/step (~5.4x). Falls back to the aten reference on any error. NOT byte-identical: "
        "the fused projection reduction moves steered next-token logprobs by up to ~2.3e-02 vs "
        "the aten path (~3% of the steering effect). '=0' reproduces the eager forward-hook "
        "steer bit-exactly; use it when steering must be bit-reproducible.",
    ),
    "compact_kall": (
        "MIA_QK_COMPACT_KALL", "auto",
        "QK only. Ship k_full + prefix_ends (O(seq)) instead of the padded O(seq^2) k_all; "
        "the driver rebuilds it. ~130x worker retrieval on trajectories. Byte-identical. "
        "'auto' = compact only when a request accumulated >=2 growing-prefix rows.",
    ),
    "writer_process": (
        "MIA_WRITER_PROCESS", "on",
        "Disk path only. Serialize + write in a child process, off the engine GIL. Moved the "
        "disk SLO knee (QK 12->16, HS 12->20). Byte-identical.",
    ),
    "storage_router": (
        "MIA_STORAGE_ROUTER", "on",
        "Serve only -- strictly inert offline (LLM.generate never calls it). Predicts a "
        "request's artifact size and picks RPC vs disk, reproducing the optimum HS-last->RPC / "
        "QK+HS-all->disk. Fires ONLY when the caller set no save_to_disk: an explicit value is a "
        "requirement (True = 'I need the artifact FILE') and is never overridden.",
    ),
    "artifact_dtype": (
        "MIA_ARTIFACT_DTYPE", "native",
        "Quantize saved artifacts: int2|int4|int8|fp8_e4m3|fp8_e5m2|bf16|fp16|fp32. "
        "ORTHOGONAL capability, OFF by default because it is LOSSY -- the only lever here "
        "that does not preserve values. 'native'/false = no quantization.",
    ),
    "aperture_mmap": (
        "MIA_APERTURE_MMAP", "off",
        "Capture-aperture durable sink. 'off' (default) writes each layer's raw file "
        "with plain open(ab)+write(), which RELEASES the GIL; 'on' memcpys into a pre-sized "
        "MAP_SHARED mapping, which holds it for the whole copy on the drain consumer thread. Same "
        "bytes either way -- byte-identical, a scheduling choice only. Off recovered ~98% of the "
        "phase=both serve gap: SLO knee 4->8, saturation 16->32, replicated K=3 across three nodes. "
        "Turn it on only for a genuinely networked-GPFS run dir, where the per-step open(ab) cost "
        "the mmap path was built to remove outweighs the GIL it holds. LEGACY WRITE PATH ONLY: "
        "accepted only with MIA_APERTURE_WRITE_MODE=legacy and refused otherwise, because the "
        "default write path keeps every raw file open for the run (no per-step open at all).",
    ),
    "aperture_max_batched_tokens": (
        "MIA_APERTURE_MAX_BATCHED_TOKENS", "off",
        "Graph mode only. Auto-derive (or pin) max_num_batched_tokens so heavy full-graph capture's "
        "per-step transient cannot CUDA-OOM at high batch. MIN-ONLY -- it only ever LOWERS the "
        "budget, so it is byte-identical when the derived cap >= what vLLM would use. 'auto' derives "
        "from model dims + GPU + aperture; an int pins the cap (still min'd); off/unset leaves the budget "
        "untouched. Opt-in (default off).",
    ),
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

