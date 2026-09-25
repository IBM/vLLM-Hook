"""Triton-fused capture scatter — one kernel for ``capture_hs``."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4

FUSED_OK_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def capture_hs_fused_kernel(
    hidden_ptr,
    residual_ptr,
    buf_ptr,
    index_ptr,
    hidden,
    stride_h, stride_r, stride_buf,
    HAS_RESIDUAL: tl.constexpr,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    dst = tl.load(index_ptr + row)

    dst = tl.where((dst >= 0) & (dst <= SENTINEL), dst, SENTINEL)

    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden

    v = tl.load(hidden_ptr + row * stride_h + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        v += tl.load(residual_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(buf_ptr + dst * stride_buf + offs, v, mask=mask)


def capture_hs_fused(hidden, residual, hs_buf, index, has_residual, n=None) -> None:
    """Launch the fused capture scatter (in-place on ``hs_buf``)."""
    if n is None:
        n = hidden.shape[0]
    n = int(n)
    if n == 0:
        return
    h = int(hidden.shape[1])
    sentinel = int(hs_buf.shape[0]) - 1
    grid = (n, triton.cdiv(h, _BLOCK))
    capture_hs_fused_kernel[grid](
        hidden, residual, hs_buf,
        index,
        h,
        hidden.stride(0), residual.stride(0), hs_buf.stride(0),
        HAS_RESIDUAL=1 if has_residual else 0,
        SENTINEL=sentinel,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )

