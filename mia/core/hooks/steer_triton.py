"""Triton-fused ``steer_buffer`` op: the FULL-mode buffer-steering kernel."""
from __future__ import annotations

import os

import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4

_EARLY_EXIT = os.environ.get("MIA_STEER_EARLY_EXIT", "0") == "1"


@triton.jit
def steer_buffer_fused_kernel(
    residual_ptr,
    coeff_ptr,
    vec_id_ptr,
    vec_table_ptr,
    avg_proj_ptr,
    steer_mode_ptr,
    hid_ptr,
    n,
    hidden,
    res_row_stride,
    vec_row_stride,
    hid_row_stride,
    BLOCK: tl.constexpr,
    EARLY_EXIT: tl.constexpr,
    HAS_HID: tl.constexpr,
):
    # The residual stream is hidden_states + residual (the next vLLM layer adds them).
    # When HAS_HID is set, adjust_rs computes the projection on that sum; the update is
    # always written to the residual.
    t = tl.program_id(0)

    res_dtype = residual_ptr.dtype.element_ty

    mode = tl.load(steer_mode_ptr + t)
    cf = tl.load(coeff_ptr + t)

    if EARLY_EXIT:
        do_work = (mode != 0) or (cf != 0.0)
        if do_work:
            vid = tl.load(vec_id_ptr + t)
            res_row = residual_ptr + t * res_row_stride
            vec_row = vec_table_ptr + vid * vec_row_stride
            if mode != 0:
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                for off in range(0, hidden, BLOCK):
                    cols = off + tl.arange(0, BLOCK)
                    m = cols < hidden
                    r = tl.load(res_row + cols, mask=m, other=0.0).to(tl.float32)
                    if HAS_HID:
                        h = tl.load(hid_ptr + t * hid_row_stride + cols, mask=m,
                                    other=0.0).to(tl.float32)
                        # Round the sum to the model dtype to match the next layer's add.
                        r = (r + h).to(res_dtype).to(tl.float32)
                    u = tl.load(vec_row + cols, mask=m, other=0.0).to(tl.float32)
                    acc += (r * u).to(res_dtype).to(tl.float32)
                proj = tl.sum(acc, axis=0).to(res_dtype).to(tl.float32)
                avg = tl.load(avg_proj_ptr + vid).to(res_dtype).to(tl.float32)
                c = (avg - proj).to(res_dtype).to(tl.float32)
            else:
                c = cf.to(res_dtype).to(tl.float32)
            for off in range(0, hidden, BLOCK):
                cols = off + tl.arange(0, BLOCK)
                m = cols < hidden
                u = tl.load(vec_row + cols, mask=m, other=0.0).to(tl.float32)
                r = tl.load(res_row + cols, mask=m, other=0.0).to(tl.float32)
                prod = (c * u).to(res_dtype).to(tl.float32)
                tl.store(res_row + cols, (r + prod).to(res_dtype), mask=m)
        return

    vid = tl.load(vec_id_ptr + t)
    res_row = residual_ptr + t * res_row_stride
    vec_row = vec_table_ptr + vid * vec_row_stride

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, hidden, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < hidden
        r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_HID:
            h = tl.load(hid_ptr + t * hid_row_stride + cols, mask=mask,
                        other=0.0).to(tl.float32)
            r = (r + h).to(res_dtype).to(tl.float32)
        u = tl.load(vec_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += (r * u).to(res_dtype).to(tl.float32)
    proj = tl.sum(acc, axis=0).to(res_dtype).to(tl.float32)

    avg = tl.load(avg_proj_ptr + vid).to(res_dtype).to(tl.float32)
    c = tl.where(mode != 0, avg - proj, cf)
    c = c.to(res_dtype).to(tl.float32)

    for off in range(0, hidden, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < hidden
        u = tl.load(vec_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        prod = (c * u).to(res_dtype).to(tl.float32)
        new = r + prod
        tl.store(res_row + cols, new.to(res_dtype), mask=mask)


def steer_buffer_fused(residual, coeff, vec_id, vec_table, avg_proj, steer_mode, n=None,
                       hidden=None):
    """Launch the fused steer kernel (in-place on ``residual[:n]``).

    ``hidden``: optional hidden states returned alongside ``residual``. Only ``adjust_rs``
    reads it, to compute the projection on ``hidden + residual``.
    """
    if n is None:
        n = residual.shape[0]
    n = int(n)
    if n == 0:
        return
    hidden_size = int(residual.shape[1])
    has_hid = hidden is not None
    if has_hid and (hidden.shape[-1] != hidden_size or hidden.dtype != residual.dtype
                    or hidden.stride(-1) != 1):
        raise RuntimeError("steer_buffer: hidden must match residual's width and dtype "
                           "and be contiguous along the hidden dimension")
    hid = hidden if has_hid else residual  # placeholder; not read when HAS_HID is False
    grid = (n,)
    steer_buffer_fused_kernel[grid](
        residual, coeff, vec_id, vec_table, avg_proj, steer_mode, hid,
        n, hidden_size,
        residual.stride(0), vec_table.stride(0), hid.stride(0),
        BLOCK=_BLOCK,
        EARLY_EXIT=_EARLY_EXIT,
        HAS_HID=has_hid,
        num_warps=_NUM_WARPS,
    )
