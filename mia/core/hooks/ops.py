"""Custom ops for CUDA-graph QK/HS capture and steering (buffer mode)."""
from __future__ import annotations

import os

import torch
from torch.library import Library

_STEER_FUSED = os.environ.get("MIA_STEER_FUSED", "1") == "1"

_CAPTURE_FUSED = os.environ.get("MIA_CAPTURE_FUSED", "1") == "1"

_FUSED_OK_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_LIB = None
_OPS_REGISTERED = False

_FIRE_COUNT = [0]


def get_fire_count() -> int:
    return _FIRE_COUNT[0]


_FUSED_FIRE_COUNT = [0]


def get_fused_fire_count() -> int:
    return _FUSED_FIRE_COUNT[0]


def _steer_buffer_impl(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    if coeff is None or vec_id is None or vec_table is None:
        return None
    _FIRE_COUNT[0] += 1
    n = residual.shape[0]
    if _STEER_FUSED and residual.is_cuda:
        try:
            # lazy: Triton kernels load on first use; a failed import falls back below
            from mia.core.hooks.steer_triton import steer_buffer_fused
            steer_buffer_fused(residual, coeff, vec_id, vec_table, avg_proj, steer_mode, n)
            return None
        except Exception:  # noqa: BLE001
            pass
    vids = vec_id[:n].to(torch.long)
    units = vec_table.index_select(0, vids).to(residual.dtype)
    c_add = coeff[:n].to(residual.dtype)
    proj = (residual[:n] * units).sum(dim=-1)
    avg = avg_proj.index_select(0, vids).to(residual.dtype)
    c_adj = avg - proj
    is_adj = steer_mode[:n].to(torch.bool)
    c = torch.where(is_adj, c_adj, c_add)
    residual[:n].add_(c.unsqueeze(-1) * units)
    return None


def _steer_buffer_fake(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    return None


LIB_NAMESPACE = "mia"
CAPTURE_QK_OP_NAME = "capture_qk"
CAPTURE_HS_OP_NAME = "capture_hs"
STEER_BUFFER_OP_NAME = "steer_buffer"


def _import_direct_register_custom_op():
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


def _capture_qk_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    del active
    _FIRE_COUNT[0] += 1
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
    return None


def _capture_hs_impl(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hs_buf: torch.Tensor,
    index: torch.Tensor,
    has_residual: int,
) -> None:
    _FIRE_COUNT[0] += 1
    if hs_buf is None or index is None:
        return None
    if (_CAPTURE_FUSED and hidden.is_cuda
            and hidden.dtype == residual.dtype
            and hs_buf.dtype == hidden.dtype
            and hs_buf.dtype in _FUSED_OK_DTYPES):
        try:
            # lazy: Triton kernels load on first use; a failed import falls back below
            from mia.core.hooks.capture_triton import capture_hs_fused
            capture_hs_fused(hidden, residual, hs_buf, index, has_residual)
            _FUSED_FIRE_COUNT[0] += 1
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
    return None


def register_graph_ops() -> None:
    """Register all graph-capture custom ops under ``mia``."""
    global _LIB, _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    direct_register_custom_op = _import_direct_register_custom_op()

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
    """Thin wrapper over ``torch.ops.mia.capture_qk`` (resolved at call time)."""
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

