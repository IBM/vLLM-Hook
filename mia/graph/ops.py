"""Custom ops for CUDA-graph QK/HS capture and steering (buffer mode).

Three static-buffer ops, all absorbed into the decode cudagraph (NOT splitting ops) so they
replay every step with no Python on the replay path:

  * ``capture_qk`` / ``capture_hs`` — scatter this step's routed q/k / residual rows into a
    static per-layer buffer, in place.
  * ``steer_buffer`` — masked in-place ``residual += coeff * vec`` (unifies add_vector + adjust_rs).

Each mutates its output buffer under a ``mutates_args`` contract so the piecewise/full compiler
threads the write into the downstream graph.
"""
from __future__ import annotations

import os

import torch

# Triton fusion for the buffer-mode steer op. Default ON; set MIA_STEER_FUSED=0 to fall
# back to the aten control. When enabled AND the residual is on CUDA, _steer_buffer_impl
# dispatches to a single fused kernel instead of gather + reduction + where + scaled-add; any
# Triton error (or a CPU/meta residual) falls back to the aten path, so fusion is never
# load-bearing.
_STEER_FUSED = os.environ.get("MIA_STEER_FUSED", "1") == "1"

# Triton-fused capture scatter. Default ON: byte-identical to the aten path and strictly less
# work (fewer memory passes and kernel launches). Any Triton error (or a CPU/unsupported-dtype
# tensor) falls through to the aten body, so fusion is never load-bearing;
# MIA_CAPTURE_FUSED=0 is the kill switch / aten-fallback control.
_CAPTURE_FUSED = os.environ.get("MIA_CAPTURE_FUSED", "1") == "1"

# Mirrors capture_triton.FUSED_OK_DTYPES. Deliberately duplicated, NOT imported: a module-level
# import of capture_triton would make ops.py require Triton at import time, and ops.py must
# import on a driver-less box (the login node). The dispatch import stays inside the function.
_FUSED_OK_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# The dedicated mia library (held alive); populated by register_graph_ops.
_LIB = None
_OPS_REGISTERED = False

# Real impl executions (prefill + graph-capture passes, NOT cudagraph replays).
# Read via get_fire_count to confirm the op is in the executed graph. Diagnostic.
_FIRE_COUNT = [0]


def get_fire_count() -> int:
    return _FIRE_COUNT[0]


# Successful `capture_hs_fused` completions only — bumped AFTER the call returns, inside the
# try in _capture_hs_impl, never on the except-fallthrough. _CAPTURE_FUSED being armed does not
# by itself prove the fused kernel ran (Triton absence, a JIT error, or an unsupported dtype all
# fall through silently to the aten body); this counter tells the difference. Diagnostic only,
# mirrors _FIRE_COUNT.
_FUSED_FIRE_COUNT = [0]


def get_fused_fire_count() -> int:
    return _FUSED_FIRE_COUNT[0]


# steer_buffer: the MUTATING counterpart of capture_qk/capture_hs. A real CUDA kernel (NOT a
# splitting op) that reads static routing buffers and does a masked, in-place
# `residual += coeff * vec` (the `add_vector` method). It is absorbed into the decode cudagraph
# and replays every step; the in-place mutation under the `mutates_args=["residual"]` contract is
# what threads the steered residual into the downstream layers — a functional variant that
# returns a new residual instead has its return value dropped across the compiled segment seam,
# so in-place + the mutates contract is load-bearing, not a style choice.
#
# This is the LoRA-slot model: a fixed `vec_table[V_max, hidden]` library, a per-row
# `token → vec_id` map, and a per-row `coeff` that gates no-steer rows to 0. Different
# requests pick different (layer, vector, coefficient) purely as buffer contents.


def _steer_buffer_impl(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    """Masked in-place steering add — unified ``add_vector`` + ``adjust_rs``.

    Per token ``t`` (``unit = vec_table[vec_id[t]]``):
      * ``steer_mode[t] == 0`` (add_vector / no-op): ``c = coeff[t]`` (host-supplied;
        0 = no-op).
      * ``steer_mode[t] == 1`` (adjust_rs): ``c = avg_proj[vec_id[t]] - (residual[t]·unit)``
        — the per-token coefficient is computed IN-KERNEL from the live residual (the
        projection ``residual·unit`` only exists at replay), so adjust_rs cannot be a
        static host coefficient like add_vector.
    Then ``residual[t] += c · unit``. Vectorised + graph-safe (gather + reduction +
    ``where`` + scaled add, no host sync); the routing tensors are length CAP and the
    leading ``n = residual.shape[0]`` rows align with this step's tokens.

    (The arg is ``steer_mode``, not ``mode``: ``mode`` collides with a reserved
    parameter of torch's ``auto_functionalized`` HOP — the wrapper for mutating ops
    under torch.compile — and raises ``auto_functionalized_fake() got multiple values
    for argument 'mode'`` at compile time.)
    """
    # Buffer-mode static buffers are Python-global tensors baked as graph consts; they can
    # deserialize to None on an AOT-compile reload. Skip the add rather than crash (defensive).
    if coeff is None or vec_id is None or vec_table is None:
        return None
    _FIRE_COUNT[0] += 1  # diagnostic only
    n = residual.shape[0]
    # Triton fusion (MIA_STEER_FUSED, default ON). CUDA-only; any failure (Triton absent,
    # kernel compile/launch error) falls through to the aten reference below — fusion is
    # additive, never load-bearing. The gate sits AFTER the None-guard + _FIRE_COUNT bump so
    # fused and aten paths keep identical guard + fire-count behaviour.
    if _STEER_FUSED and residual.is_cuda:
        try:
            from mia.graph.steer_triton import steer_buffer_fused
            steer_buffer_fused(residual, coeff, vec_id, vec_table, avg_proj, steer_mode, n)
            return None
        except Exception:  # noqa: BLE001 — Triton unavailable / kernel error: fall back to aten
            pass
    vids = vec_id[:n].to(torch.long)                            # (n,)
    units = vec_table.index_select(0, vids).to(residual.dtype)  # (n, hidden)
    c_add = coeff[:n].to(residual.dtype)                        # (n,) host coeff
    # adjust_rs coefficient, computed for every row but selected only where mode==1.
    proj = (residual[:n] * units).sum(dim=-1)                   # (n,) residual·unit
    avg = avg_proj.index_select(0, vids).to(residual.dtype)     # (n,) per-vector target
    c_adj = avg - proj                                         # (n,)
    is_adj = steer_mode[:n].to(torch.bool)                     # (n,)
    c = torch.where(is_adj, c_adj, c_add)                      # (n,)
    residual[:n].add_(c.unsqueeze(-1) * units)                 # in-place — mutates contract
    return None


def _steer_buffer_fake(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    """Meta/fake impl: in-place mutation, no new tensor under tracing."""
    return None


# Constants so downstream files import rather than hard-code op names.
LIB_NAMESPACE = "mia"
CAPTURE_QK_OP_NAME = "capture_qk"
CAPTURE_HS_OP_NAME = "capture_hs"
STEER_BUFFER_OP_NAME = "steer_buffer"


def _import_direct_register_custom_op():
    """Locate ``direct_register_custom_op``; it has moved between vLLM releases."""
    errs = []
    for modpath in (
        "vllm.utils",
        "vllm.utils.torch_utils",
        "vllm.utils._custom_ops",
        "vllm.compilation.decorators",
    ):
        try:
            mod = __import__(modpath, fromlist=["direct_register_custom_op"])
            fn = getattr(mod, "direct_register_custom_op", None)
            if fn is not None:
                return fn
            errs.append(f"  {modpath}: imported but no symbol")
        except ImportError as e:
            errs.append(f"  {modpath}: {e}")
    raise ImportError(
        "Could not locate direct_register_custom_op in any known vLLM path.\n"
        + "\n".join(errs)
    )


# ---------------------------------------------------------------------------
# capture_qk: real (CUDA) and fake (meta) impls
# ---------------------------------------------------------------------------


def _capture_qk_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Scatter post-RoPE q/k rows into the static buffers, in place.

    Vectorized and graph-safe (no host sync). ``index`` is length CAP but q/k have
    only ``n = q.shape[0]`` rows this step, and ``index_copy_`` requires the index
    length to match; the leading ``n`` routing entries align row-for-row with q/k,
    and ``n`` is a graph-constant slice. ``active`` is unused — sentinel-row
    routing already no-ops inactive tokens, and a multiply would allocate a
    per-step temporary and could zero real data on a routing miscompute. The dtype
    cast is a defensive fallback (q/k should already match buf dtype).
    """
    del active  # unused: sentinel-row routing supersedes the multiplicative gate
    _FIRE_COUNT[0] += 1  # diagnostic only
    # Static buffers are Python-global tensors baked as graph consts; they can deserialize to
    # None on an AOT-compile reload. Skip the scatter rather than crash (defensive).
    if q_buf is None or k_buf is None or index is None:
        return None
    n = q.shape[0]
    idx = index[:n].to(torch.long)
    q_src = q if q.dtype == q_buf.dtype else q.to(q_buf.dtype)
    k_src = k if k.dtype == k_buf.dtype else k.to(k_buf.dtype)
    q_buf.index_copy_(0, idx, q_src)
    k_buf.index_copy_(0, idx, k_src)
    return None


def _capture_qk_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Meta/fake impl: in-place mutation, no new tensor under tracing."""
    return None


# ---------------------------------------------------------------------------
# capture_hs: the HS analogue of capture_qk — scatter the residual stream into
# a static per-layer sink, in place, as a graph-recorded kernel. NOT a splitting
# op, so it is absorbed into the decode cudagraph and replays every step (no
# Python on replay). Unlike QK there is no prefix-K — the residual stream is
# per-token, so the scattered rows ARE the artifact.
# ---------------------------------------------------------------------------


def _capture_hs_impl(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hs_buf: torch.Tensor,
    index: torch.Tensor,
    has_residual: int,
) -> None:
    """Form the residual stream and scatter its rows into ``hs_buf``, in place.

    ``combined = hidden + residual`` for fused-residual layers (``has_residual``
    truthy), else ``hidden``. ``index`` is length CAP but ``hidden`` has only
    ``n = hidden.shape[0]`` rows this step; the leading ``n`` routing entries
    align row-for-row, and row 0 is the discard sentinel for unrequested tokens.
    Vectorised, no host sync — graph-legal. ``has_residual`` is a per-layer
    constant baked at the call site, so this adds no data-dependent control flow.
    """
    _FIRE_COUNT[0] += 1  # diagnostic only
    # Static buffers are Python-global tensors baked as graph consts; they can
    # deserialize to None on an AOT-compile reload. Skip the scatter rather than crash.
    if hs_buf is None or index is None:
        return None
    # Triton fusion, default ON (MIA_CAPTURE_FUSED=0 disables). CUDA-only; any failure
    # (Triton absent, JIT error, unsupported dtype) falls through to the aten reference below —
    # never load-bearing.
    #
    # `hs_buf.dtype == hidden.dtype` is a BYTE-IDENTITY gate, not a capability one: the two paths
    # round DIFFERENTLY when the dtypes differ. aten computes `hidden + residual` in the HIDDEN
    # dtype (one rounding) then casts to `hs_buf.dtype` (a second rounding); the kernel widens to
    # fp32, adds, and rounds ONCE on store. Requiring equal dtypes keeps the fused path exactly
    # `torch.equal` to aten rather than merely close; do not relax this gate to "any dtype in
    # _FUSED_OK_DTYPES" without accounting for the double-rounding difference.
    if (_CAPTURE_FUSED and hidden.is_cuda
            and hidden.dtype == residual.dtype
            and hs_buf.dtype == hidden.dtype
            and hs_buf.dtype in _FUSED_OK_DTYPES):
        try:
            from mia.graph.capture_triton import capture_hs_fused
            capture_hs_fused(hidden, residual, hs_buf, index, has_residual)
            _FUSED_FIRE_COUNT[0] += 1  # diagnostic only: fused kernel actually completed
            return None
        except Exception:  # noqa: BLE001
            pass
    n = hidden.shape[0]
    idx = index[:n].to(torch.long)
    combined = hidden + residual if has_residual else hidden
    src = combined if combined.dtype == hs_buf.dtype else combined.to(hs_buf.dtype)
    hs_buf.index_copy_(0, idx, src)
    return None


def _capture_hs_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hs_buf: torch.Tensor,
    index: torch.Tensor,
    has_residual: int,
) -> None:
    """Meta/fake impl: in-place mutation, no new tensor under tracing."""
    return None


# ---------------------------------------------------------------------------
# registration entry point
# ---------------------------------------------------------------------------


def register_graph_ops() -> None:
    """Register all graph-capture custom ops under ``mia``. Idempotent:
    "already registered" errors are swallowed."""
    global _LIB, _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    from torch.library import Library

    direct_register_custom_op = _import_direct_register_custom_op()

    # Dedicated library so ops live at torch.ops.mia.*, not torch.ops.vllm.*.
    if _LIB is None:
        _LIB = Library(LIB_NAMESPACE, "FRAGMENT")

    try:
        direct_register_custom_op(
            op_name=CAPTURE_QK_OP_NAME,
            op_func=_capture_qk_impl,
            mutates_args=["q_buf", "k_buf"],
            fake_impl=_capture_qk_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
        # Already registered in this process — fine.
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    try:
        direct_register_custom_op(
            op_name=CAPTURE_HS_OP_NAME,
            op_func=_capture_hs_impl,
            mutates_args=["hs_buf"],
            fake_impl=_capture_hs_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    # The FULL-mode (buffer) steering op: masked in-place residual add, NOT a
    # splitting op — absorbed into the decode cudagraph and replayed every step.
    try:
        direct_register_custom_op(
            op_name=STEER_BUFFER_OP_NAME,
            op_func=_steer_buffer_impl,
            mutates_args=["residual"],
            fake_impl=_steer_buffer_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    # Mark registered only once the handles resolve, so a partial failure surfaces
    # here rather than as a confusing AttributeError at capture time.
    ns = getattr(torch.ops, LIB_NAMESPACE, None)
    if ns is None or not hasattr(ns, CAPTURE_QK_OP_NAME) or not hasattr(
        ns, CAPTURE_HS_OP_NAME
    ) or not hasattr(ns, STEER_BUFFER_OP_NAME):
        raise RuntimeError(
            f"register_graph_ops: ops under torch.ops.{LIB_NAMESPACE} "
            "did not resolve after registration"
        )

    _OPS_REGISTERED = True


def capture_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Thin wrapper over ``torch.ops.mia.capture_qk`` (resolved at call time),
    so callers can import and call it directly; traced as the opaque graph node."""
    return torch.ops.mia.capture_qk(q, k, q_buf, k_buf, index, active)


def capture_hs(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hs_buf: torch.Tensor,
    index: torch.Tensor,
    has_residual: int,
) -> None:
    """Thin wrapper over ``torch.ops.mia.capture_hs`` (resolved at call time)."""
    return torch.ops.mia.capture_hs(hidden, residual, hs_buf, index, has_residual)


def steer_buffer(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    """Thin wrapper over ``torch.ops.mia.steer_buffer`` (resolved at call time)."""
    return torch.ops.mia.steer_buffer(
        residual, coeff, vec_id, vec_table, avg_proj, steer_mode)
