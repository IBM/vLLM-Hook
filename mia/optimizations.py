"""The user-facing optimization surface for FULL CUDA-graph mode.

ONE table (``PUBLIC_LEVERS``) is the whole public API. Everything else the package reads from
the environment is INTERNAL — tuning constants (aperture depths, poll intervals, queue sizes),
regime-specific levers that are not a default win, and diagnostics that deliberately corrupt or
perturb. Those keep working for anyone who knows the name; they are simply not part of the
supported surface and are not accepted here.

Two ways to set a lever, in precedence order:

1. **Environment variable** — always wins. Every existing ``run_*.sh`` keeps working unchanged.
2. **Config file** — an ``"optimizations"`` block in ``model_configs/<use_case>/<model>.json``,
   alongside the ``hidden_states`` / ``params`` / ``hookq`` / ``steering`` sections::

       {
         "model_info":     {"name": "Qwen/Qwen2-1.5B-Instruct"},
         "hidden_states":  {"layers": [], "mode": "last_token"},
         "optimizations":  {"artifact_dtype": "int8"}
       }

3. Unset in both -> the shipped default below.

**Defaults are the proven stack.** A lever ships ON only where a GPU measurement says it is a
gain; the rest ship OFF and stay toggleable. Nothing here is lossy or behavior-changing by
default: captured values are byte-identical to the eager path unless you opt into
``artifact_dtype``.

**Scope.** ``load_config`` runs in the DRIVER before ``LLM(...)`` spawns the workers, so the env
it sets is inherited by every worker process. That makes the config block an OFFLINE
(``MiaLLM``) path: ``vllm serve`` never parses a config file, so serve callers set the env
directly (which is the same knob).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

# key -> (env var, shipped default, one-line doc)
#
# Shipped default is what you get with the key absent AND the env unset. "on"/"off" are plain
# booleans; "auto" means the lever decides per request; "native" means no transform.
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
        "GPU-measured (LSF 1703642, Phi-3-mini fp16, adjust_rs) the fused kernel's projection "
        "reduction moves the steered next-token logprobs max|d| = 2.34e-02 vs the aten path, "
        "on a steering effect of 7.72e-01 (~3%). '=0' reproduces the eager forward-hook steer "
        "BIT-EXACTLY (0.000e+00). Use '=0' when steering must be bit-reproducible.",
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
        "Capture-aperture durable sink. 'off' (default since 2026-08-14) writes each layer's raw file "
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
        "untouched. Opt-in (default off) pending a serve/cb A/B; default-ON auto is the intended end.",
    ),
}

_TRUE = ("1", "true", "on", "yes")
_FALSE = ("0", "false", "off", "no")
# Values that mean "use the built-in default" -> leave the env unset rather than forcing a value.
_DEFER = ("auto", "default", "native")


def _to_env_value(key: str, value: Any) -> Optional[str]:
    """Map a JSON config value to an env string, or None = leave unset (built-in default).

    Booleans and the usual on/off spellings collapse to "1"/"0". ``artifact_dtype`` is a
    free-form dtype name, so anything unrecognized passes through for the quant module to
    validate -- except the off-spellings, which mean "native" (unset), not the string "0".
    """
    if value is None:
        return None
    if key == "aperture_max_batched_tokens":
        # Tri-state: 'auto' (or any truthy spelling) enables derivation; an int pins the cap and
        # passes through; off-spellings leave the env UNSET (the OFF default). Handled before the
        # generic _DEFER path, which would wrongly map 'auto' -> unset for this opt-in lever.
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
            return str(value)          # e.g. artifact_dtype: "int8"
    if key == "artifact_dtype":
        # This lever has no "0" state: off means native, i.e. the env stays unset.
        return "1" if truthy else None
    return "1" if truthy else "0"


def apply_optimizations(config_data: Mapping[str, Any]) -> Dict[str, str]:
    """Apply a config file's ``optimizations`` block to the environment. Returns what it set.

    Precedence: an explicit env var ALWAYS wins (so a run script overrides the config file, and
    every existing harness is unaffected). Unknown keys raise -- a typo that silently changed
    nothing is the failure mode this table exists to prevent.

    Must run BEFORE the workers are spawned; ``MiaLLM.__init__`` calls ``load_config`` before
    ``LLM(...)`` for exactly that reason.
    """
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
            continue                    # 'auto'/'native' -> defer to the built-in default
        if env_name in os.environ:
            continue                    # explicit env wins
        os.environ[env_name] = env_value
        applied[key] = env_value
    return applied


def env_is_on(key: str) -> bool:
    """Resolve a boolean public lever: the env if set, else the shipped default in the table.

    Call this instead of re-spelling ``os.environ.get(NAME, "1") == "1"`` at the use site, so the
    table above is the SINGLE source of truth for the default rather than a second copy that can
    drift out of agreement with the code.
    """
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
