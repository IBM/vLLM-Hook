"""Artifact-size prediction and the RPC-vs-disk routing decision."""
from __future__ import annotations

import os


def qk_score_size_select(prompt_len: int, mode: str, layer_to_heads: dict,
                         H_q: int, H_kv: int, d: int) -> str:
    """Size model: return "qk" or "score" — the smaller capture artifact."""
    if not layer_to_heads:
        return "qk"
    S = int(prompt_len)
    qk_elems = sc_elems = 0
    for _layer, heads in layer_to_heads.items():
        n = len(heads) or 1
        if mode == "all_tokens":
            qk_elems += S * (H_q + H_kv) * d
            sc_elems += n * S * S
        else:
            qk_elems += (H_q + S * H_kv) * d
            sc_elems += n * S
    return "score" if sc_elems < qk_elems else "qk"


_RPC_INTERCEPT_MS = 5.0
_RPC_SLOPE_MS_PER_KB = {"qk": 0.157, "hs": 0.03}

_DISK_HANDOFF_MS = 20.0
_DISK_SLOPE_MS_PER_KB = {"qk": 0.0078, "hs": 0.0022}
DEFAULT_GEN_LEN = 256


def estimate_gen_len(max_tokens) -> int:
    """Decode length to price a request at: max_tokens if pinned, else a default estimate."""
    try:
        n = int(max_tokens or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return n
    try:
        return max(1, int(os.environ.get("MIA_ROUTER_DEFAULT_GEN_LEN", DEFAULT_GEN_LEN)))
    except ValueError:
        return DEFAULT_GEN_LEN


def predict_artifact_kb(worker_kind: str, gran: str, prompt_len: int, n_layers: int,
                        heads_per_layer: int, head_dim: int, hidden: int,
                        dtype_bytes: int = 2, gen_len: int = 0,
                        hooks_on: str = "prefill") -> float:
    """Closed-form RAW artifact size (KB) for one request's captured tensors."""
    P = int(prompt_len)
    seq = P if hooks_on == "prefill" else P + int(gen_len)
    if hooks_on == "prefill":
        steps = 1
    elif hooks_on == "decode":
        steps = max(0, int(gen_len))
    else:
        steps = 1 + max(0, int(gen_len))
    L = int(n_layers)
    if worker_kind == "hs":
        elems = L * steps * hidden if gran == "last_token" else L * seq * hidden
    else:
        Hd = int(heads_per_layer) * int(head_dim)
        if gran == "last_token":
            elems = L * seq * Hd + L * Hd
        else:
            elems = L * seq * 2 * Hd
    return elems * int(dtype_bytes) / 1024.0


def predicted_rpc_ms(worker_kind: str, predicted_kb: float) -> float:
    """Predicted blocking RPC ship time (ms) for an artifact size."""
    intercept = float(os.environ.get("MIA_ROUTER_RPC_INTERCEPT_MS", _RPC_INTERCEPT_MS))
    default_slope = _RPC_SLOPE_MS_PER_KB.get(worker_kind, _RPC_SLOPE_MS_PER_KB["hs"])
    slope = float(os.environ.get(
        f"MIA_ROUTER_RPC_SLOPE_MS_PER_KB_{worker_kind.upper()}", default_slope))
    return intercept + slope * float(predicted_kb)


def predicted_disk_ms(worker_kind: str, predicted_kb: float) -> float:
    """Predicted on-loop disk cost (ms) for an artifact size."""
    handoff = float(os.environ.get("MIA_ROUTER_DISK_HANDOFF_MS", _DISK_HANDOFF_MS))
    default_slope = _DISK_SLOPE_MS_PER_KB.get(worker_kind, _DISK_SLOPE_MS_PER_KB["hs"])
    slope = float(os.environ.get(
        f"MIA_ROUTER_DISK_SLOPE_MS_PER_KB_{worker_kind.upper()}", default_slope))
    return handoff + slope * float(predicted_kb)


def route_to_disk(worker_kind: str, predicted_kb: float) -> bool:
    """True when shipping over RPC costs more than the disk route at this artifact size."""
    return (predicted_rpc_ms(worker_kind, predicted_kb)
            > predicted_disk_ms(worker_kind, predicted_kb))


NO_CROSSOVER_KB = float(1 << 30)


def rpc_disk_crossover_kb(worker_kind: str) -> float:
    """Artifact size (KB) where the RPC and disk cost models cross."""
    rpc_intercept = float(os.environ.get("MIA_ROUTER_RPC_INTERCEPT_MS", _RPC_INTERCEPT_MS))
    rpc_slope = float(os.environ.get(
        f"MIA_ROUTER_RPC_SLOPE_MS_PER_KB_{worker_kind.upper()}",
        _RPC_SLOPE_MS_PER_KB.get(worker_kind, _RPC_SLOPE_MS_PER_KB["hs"])))
    disk_handoff = float(os.environ.get("MIA_ROUTER_DISK_HANDOFF_MS", _DISK_HANDOFF_MS))
    disk_slope = float(os.environ.get(
        f"MIA_ROUTER_DISK_SLOPE_MS_PER_KB_{worker_kind.upper()}",
        _DISK_SLOPE_MS_PER_KB.get(worker_kind, _DISK_SLOPE_MS_PER_KB["hs"])))
    if rpc_intercept >= disk_handoff:
        return 0.0
    if rpc_slope <= disk_slope:
        return NO_CROSSOVER_KB
    return (disk_handoff - rpc_intercept) / (rpc_slope - disk_slope)

