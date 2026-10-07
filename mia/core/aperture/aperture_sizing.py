"""Capture-aperture byte budgets and the safe max_num_batched_tokens cap."""
import os
from typing import Optional, Tuple

from mia.errors import MiaSizingError

DEFAULT_APERTURE_GPU_BYTES = 4 * (1 << 30)

DEFAULT_AUTOCAP_SAFETY = 3
DEFAULT_AUTOCAP_HEADROOM_BYTES = 1 << 30

_TRUE = ("1", "true", "on", "yes", "auto")
_FALSE = ("", "0", "false", "off", "no")


def resolve_aperture_bytes_fixed(total_gpu_bytes: int, fixed_bytes: int, gpu_mem_util: float) -> int:
    """Fixed-size aperture path: return ``fixed_bytes`` verbatim after a fit check."""
    fixed_bytes = int(fixed_bytes)
    total_gpu_bytes = int(total_gpu_bytes)
    free_margin = (1.0 - gpu_mem_util) * total_gpu_bytes
    if fixed_bytes > free_margin:
        raise MiaSizingError(
            f"aperture {fixed_bytes/(1<<30):.2f} GiB + gpu_memory_utilization={gpu_mem_util} leaves no "
            f"room (free margin {free_margin/(1<<30):.2f} GiB): lower gpu_memory_utilization or "
            f"MIA_APERTURE_GPU_BYTES")
    return fixed_bytes


def resolve_aperture_bytes_auto(total_gpu_bytes: int, gpu_mem_util: float,
                                rows_needed: Optional[int] = None,
                                row_bytes: Optional[int] = None,
                                what: str = "capture") -> int:
    """Resolve the aperture byte budget: ``MIA_APERTURE_GPU_BYTES`` if set, else the default."""
    raw_fixed = os.environ.get("MIA_APERTURE_GPU_BYTES")
    if raw_fixed is not None and raw_fixed.strip() != "":
        return resolve_aperture_bytes_fixed(total_gpu_bytes, int(raw_fixed), gpu_mem_util)
    budget = DEFAULT_APERTURE_GPU_BYTES
    need = model_sized_aperture_bytes(rows_needed, row_bytes)
    if need is not None and need > budget:
        free_margin = (1.0 - gpu_mem_util) * int(total_gpu_bytes)
        if need > free_margin:
            raise MiaSizingError(
                f"{what} aperture cannot hold one max-token step by default: "
                f"{int(rows_needed)} rows (max_num_batched_tokens) x {int(row_bytes)} B/row "
                f"(every captured layer) = {need / (1 << 30):.2f} GiB, but "
                f"gpu_memory_utilization={gpu_mem_util} leaves {free_margin / (1 << 30):.2f} GiB. "
                f"Lower gpu_memory_utilization, lower max_num_batched_tokens, capture fewer "
                f"layers, or set MIA_APERTURE_GPU_BYTES explicitly (an explicit value always "
                f"wins; one smaller than {need} B risks ApertureBackpressureError on a "
                f"max-token step).")
        budget = need
    return resolve_aperture_bytes_fixed(total_gpu_bytes, budget, gpu_mem_util)


def model_sized_aperture_bytes(rows_needed: Optional[int], row_bytes: Optional[int]) -> Optional[int]:
    """Bytes one max-token step fills (rows x row bytes), or None when unknown."""
    try:
        rows, rb = int(rows_needed), int(row_bytes)
    except (TypeError, ValueError):
        return None
    if rows <= 0 or rb <= 0:
        return None
    return rows * rb


def aperture_bytes_is_explicit() -> bool:
    """True when ``MIA_APERTURE_GPU_BYTES`` is set (and therefore wins over any derived size)."""
    raw = os.environ.get("MIA_APERTURE_GPU_BYTES")
    return raw is not None and raw.strip() != ""


def per_layer_token_bytes_hs(hidden_size: int, dtype_size: int) -> int:
    """HS: one captured token costs one residual-stream row per layer."""
    return int(hidden_size) * int(dtype_size)


def per_layer_token_bytes_qk(n_q_heads: int, n_kv_heads: int, head_dim: int, dtype_size: int) -> int:
    """QK: one captured token costs (H_q + H_kv) * head_dim elements per layer ON ONE RANK."""
    return (int(n_q_heads) + int(n_kv_heads)) * int(head_dim) * int(dtype_size)


def compute_safe_max_batched_tokens(
    total_gpu_bytes: int,
    gpu_mem_util: float,
    aperture_gpu_bytes: int,
    n_layers_captured: int,
    per_layer_token_bytes: int,
    safety: int = DEFAULT_AUTOCAP_SAFETY,
    headroom_bytes: int = DEFAULT_AUTOCAP_HEADROOM_BYTES,
) -> Optional[int]:
    """Largest per-step token budget whose worst-case capture fits the free GPU margin."""
    total_gpu_bytes = int(total_gpu_bytes)
    free_margin = round((1.0 - gpu_mem_util) * total_gpu_bytes) - int(aperture_gpu_bytes) - int(headroom_bytes)
    bytes_per_token = int(n_layers_captured) * int(per_layer_token_bytes)
    if free_margin <= 0 or bytes_per_token <= 0 or int(safety) <= 0:
        return None
    cap = int(free_margin // (bytes_per_token * int(safety)))
    return cap if cap >= 1 else None


def safe_cap_with_model_sized_aperture(
    total_gpu_bytes: int,
    gpu_mem_util: float,
    default_aperture_bytes: int,
    n_layers_captured: int,
    per_layer_token_bytes: int,
    safety: int = DEFAULT_AUTOCAP_SAFETY,
    headroom_bytes: int = DEFAULT_AUTOCAP_HEADROOM_BYTES,
) -> Optional[int]:
    """The safe token cap when the install will size a DEFAULT aperture to one cap-sized step."""
    b = int(n_layers_captured) * int(per_layer_token_bytes)
    margin = round((1.0 - gpu_mem_util) * int(total_gpu_bytes)) - int(headroom_bytes)
    if b <= 0 or margin <= 0 or int(safety) <= 0:
        return None
    grown = int(margin // (b * (int(safety) + 1)))
    if grown * b > int(default_aperture_bytes):
        return grown if grown >= 1 else None
    boundary = int(int(default_aperture_bytes) // b)
    return boundary if boundary >= 1 else None


def apply_min_only(current: Optional[int], safe: Optional[int]) -> Optional[int]:
    """The min-only decision: only ever LOWER ``max_num_batched_tokens``."""
    if safe is None:
        return None
    if current is None:
        return int(safe)
    return int(safe) if int(safe) < int(current) else None


def parse_autocap_setting(raw: Optional[str]) -> Tuple[str, Optional[int]]:
    """Parse the tri-state MIA_APERTURE_MAX_BATCHED_TOKENS knob."""
    if raw is None:
        return ("off", None)
    s = raw.strip().lower()
    if s in _FALSE:
        return ("off", None)
    if s in _TRUE:
        return ("auto", None)
    try:
        return ("explicit", int(s))
    except ValueError:
        return ("off", None)


CAPTURE_SUBSYSTEMS = ("hs", "qk")
KNOWN_SUBSYSTEMS = ("hs", "qk", "steer")


def per_token_row_bytes(subsystem: str, model_dims) -> int:
    """Bytes ONE captured token costs ONE layer, for ``subsystem``."""
    dims = dict(model_dims or {})
    dtype_size = int(_first(dims, ("dtype_bytes", "dtype_size"), default=2))
    if subsystem == "hs":
        hidden = _first(dims, ("hidden", "hidden_size"))
        if hidden is None:
            raise ValueError("model_dims needs 'hidden' (residual width) to size the HS aperture")
        return per_layer_token_bytes_hs(int(hidden), dtype_size)
    if subsystem == "qk":
        q_dim = _first(dims, ("q", "q_dim"))
        k_dim = _first(dims, ("k", "k_dim"))
        if q_dim is None or k_dim is None:
            h_q = _first(dims, ("n_q_heads", "num_attention_heads"))
            h_kv = _first(dims, ("n_kv_heads", "num_key_value_heads"), default=h_q)
            head_dim = _first(dims, ("head_dim",))
            if h_q is None or head_dim is None:
                raise ValueError(
                    "model_dims needs 'q'/'k' widths (or n_q_heads/n_kv_heads/head_dim) "
                    "to size the QK aperture")
            return per_layer_token_bytes_qk(int(h_q), int(h_kv), int(head_dim), dtype_size)
        return per_layer_token_bytes_qk(int(q_dim), int(k_dim), 1, dtype_size)
    if subsystem == "steer":
        raise KeyError("steer has no capture aperture: it mutates the residual and captures nothing")
    raise KeyError(f"unknown subsystem {subsystem!r}; known: {', '.join(KNOWN_SUBSYSTEMS)}")


def _first(dims, names, default=None):
    for n in names:
        v = dims.get(n)
        if v is not None:
            return v
    return default


def resolve_aperture_bytes(subsystems, *, gpu_bytes_budget: int, model_dims=None):
    """Split ONE fixed aperture budget across the enabled *capture* subsystems."""
    budget = int(gpu_bytes_budget)
    if budget <= 0:
        raise ValueError(f"gpu_bytes_budget must be positive, got {gpu_bytes_budget!r}")
    requested = list(dict.fromkeys(subsystems))
    unknown = [s for s in requested if s not in KNOWN_SUBSYSTEMS]
    if unknown:
        raise ValueError(
            f"unknown subsystem(s) {unknown}; known: {', '.join(KNOWN_SUBSYSTEMS)}")
    capture = sorted(s for s in requested if s in CAPTURE_SUBSYSTEMS)
    if not capture:
        return {}
    if len(capture) == 1:
        return {capture[0]: budget}
    weights = {s: per_token_row_bytes(s, model_dims) for s in capture}
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError(f"per-token row bytes summed to {total_w} for {capture}; cannot split")
    sizes = {s: (budget * w) // total_w for s, w in weights.items()}
    zero = [s for s, b in sizes.items() if b <= 0]
    if zero:
        raise ValueError(
            f"aperture budget {budget} B is too small to give {zero} a non-empty slice across "
            f"{capture}: raise MIA_APERTURE_GPU_BYTES or enable fewer capture subsystems")
    return sizes

