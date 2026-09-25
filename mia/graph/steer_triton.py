"""Triton-fused ``steer_buffer`` op — the FULL-mode buffer-steering kernel.

Gated behind ``MIA_STEER_FUSED`` (default ON -- see ``mia/graph/ops.py``). When armed,
``ops._steer_buffer_impl`` dispatches to :func:`steer_buffer_fused` instead of the aten
reference. A single Triton program per token row does BOTH the ``adjust_rs`` projection
reduction and the in-place masked add in ONE launch (the aten path issues gather +
reduction + ``where`` + scaled-add as separate kernels).

Reference math matched verbatim (``graph/ops.py:_steer_buffer_impl``), per token ``t`` with
``unit = vec_table[vec_id[t]]``:
  * ``steer_mode[t] == 0`` (add_vector / no-op): ``c = coeff[t]`` (host, cast to residual dtype)
  * ``steer_mode[t] == 1`` (adjust_rs):          ``c = avg_proj[vec_id[t]] - (residual[t] . unit)``
  then ``residual[t] += c * unit``   (in place — the ``mutates_args=["residual"]`` contract).

Sentinel / padding / unsteered rows carry ``coeff=0, steer_mode=0`` (``vec_id`` arbitrary) →
``c=0`` → residual unchanged. The no-op does NOT depend on ``vec_table[vec_id]`` being zero, so
the kernel gates purely on ``coeff``/``steer_mode`` (it does: ``c=0`` → ``+0``).

Bit-exactness: add_vector rows must match aten to the last bit. aten forms ``c * unit`` in the
residual dtype (a rounded product) then ``.add_`` (a second rounding). The kernel mirrors the
two roundings — it casts the product to the residual dtype BEFORE the add and stores through the
residual dtype (auto-rounds the sum) — instead of a single fp32 FMA. The adjust_rs projection
reduces over ``hidden`` and reorders vs ``torch.sum``, so adjust_rs matches only to ~1e-3 (the
tolerated divergence).

cudagraph-safety: NO ``@triton.autotune`` (multi-config timing launches are illegal during
graph capture), no host sync in the launcher, fixed ``num_warps``/``BLOCK``. The op is baked
into the decode cudagraph; vLLM's pre-capture warmup forwards JIT-compile the kernel before
capture, so replay never re-enters Python.

Importable without a GPU: importing ``triton`` is fine; nothing launches at import time.
"""
from __future__ import annotations

import os

import triton
import triton.language as tl

# Fixed launch config (NO autotune — a single-config launch is cudagraph-legal).
# BLOCK tiles ``hidden``; a power of two so ``tl.arange`` is legal, and the masked tail
# handles ``hidden % BLOCK != 0`` (e.g. Qwen2-1.5B hidden=1536, Granite hidden=4096).
_BLOCK = 1024
_NUM_WARPS = 4

# Lever B (MIA_STEER_EARLY_EXIT): per-token idle early-exit + per-mode split. When ON, a
# fully-inactive row (mode==0 AND coeff==0) skips BOTH passes (byte-identical: writing r + 0 = r
# is a lossless bf16 round-trip), and an add_vector row (mode==0, coeff!=0) skips the projection
# pass (unused for add_vector). adjust_rs (mode!=0) always runs the full 2-pass. Default OFF ->
# the kernel compiles the unconditional body verbatim (bit-for-bit).
_EARLY_EXIT = os.environ.get("MIA_STEER_EARLY_EXIT", "0") == "1"


@triton.jit
def steer_buffer_fused_kernel(
    residual_ptr,      # (n, hidden)      model dtype (bf16/fp16/fp32) — mutated IN PLACE
    coeff_ptr,         # (>= n,)          fp32   host add_vector coeff (0 = no-op)
    vec_id_ptr,        # (>= n,)          int64  row index into vec_table
    vec_table_ptr,     # (v_max, hidden)  model dtype  resident vector library
    avg_proj_ptr,      # (v_max,)         fp32   adjust_rs per-vector target projection
    steer_mode_ptr,    # (>= n,)          int64  0 = add_vector/no-op, 1 = adjust_rs
    n,                 # rows this step (padded token count) — graph constant
    hidden,            # feature width
    res_row_stride,    # residual.stride(0)
    vec_row_stride,    # vec_table.stride(0)
    BLOCK: tl.constexpr,
    EARLY_EXIT: tl.constexpr,
):
    t = tl.program_id(0)

    # Residual/vec_table element type (bf16/fp16/fp32); used to reproduce aten's rounding.
    res_dtype = residual_ptr.dtype.element_ty

    mode = tl.load(steer_mode_ptr + t)     # int64
    cf = tl.load(coeff_ptr + t)            # fp32

    if EARLY_EXIT:
        # Lever B: skip fully-inactive rows entirely (byte-identical to residual += 0), and
        # skip the projection for add_vector rows (they never use it). Per-program scalar
        # branches -> cudagraph-legal (data-dependent control flow INSIDE the captured kernel;
        # the graph captures the fixed launch, not the branch outcome).
        do_work = (mode != 0) or (cf != 0.0)
        if do_work:
            vid = tl.load(vec_id_ptr + t)
            res_row = residual_ptr + t * res_row_stride
            vec_row = vec_table_ptr + vid * vec_row_stride
            if mode != 0:
                # adjust_rs: proj = sum_j res*vec, aten's bf16 rounding (see the else block).
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                for off in range(0, hidden, BLOCK):
                    cols = off + tl.arange(0, BLOCK)
                    m = cols < hidden
                    r = tl.load(res_row + cols, mask=m, other=0.0).to(tl.float32)
                    u = tl.load(vec_row + cols, mask=m, other=0.0).to(tl.float32)
                    acc += (r * u).to(res_dtype).to(tl.float32)
                proj = tl.sum(acc, axis=0).to(res_dtype).to(tl.float32)
                avg = tl.load(avg_proj_ptr + vid).to(res_dtype).to(tl.float32)
                c = (avg - proj).to(res_dtype).to(tl.float32)
            else:
                # add_vector: c = host coeff, NO projection needed (identical to the where(...)
                # picking cf for mode==0 -> the discarded proj is pure waste today).
                c = cf.to(res_dtype).to(tl.float32)
            for off in range(0, hidden, BLOCK):
                cols = off + tl.arange(0, BLOCK)
                m = cols < hidden
                u = tl.load(vec_row + cols, mask=m, other=0.0).to(tl.float32)
                r = tl.load(res_row + cols, mask=m, other=0.0).to(tl.float32)
                prod = (c * u).to(res_dtype).to(tl.float32)
                tl.store(res_row + cols, (r + prod).to(res_dtype), mask=m)
        # else: inactive row — residual untouched (== today's r + 0 = r, same bytes).
        return

    vid = tl.load(vec_id_ptr + t)          # int64
    res_row = residual_ptr + t * res_row_stride
    vec_row = vec_table_ptr + vid * vec_row_stride

    # ------- Pass 1: proj = sum_j residual[t, j] * vec_table[vid, j] -------
    # Reproduce the aten reference's rounding EXACTLY, else low-precision (bf16) adjust_rs
    # diverges hard: aten forms `residual * units` as a res_dtype tensor (EACH product rounded
    # to res_dtype), `.sum()`s it (fp32 accumulate, result rounded back to res_dtype), and casts
    # avg_proj to res_dtype before the subtract. Keeping the dot product in fp32 out-precisions
    # aten and, over ~hidden cancelling terms, breaks parity (adj |Δ|≈2.0 @bf16). For fp32
    # residual every `.to(res_dtype)` here is a no-op, so the fp32 path stays byte-identical.
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, hidden, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < hidden
        r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(vec_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += (r * u).to(res_dtype).to(tl.float32)   # round each product to res_dtype (aten's bf16 product tensor)
    proj = tl.sum(acc, axis=0).to(res_dtype).to(tl.float32)         # aten's .sum() returns res_dtype

    avg = tl.load(avg_proj_ptr + vid).to(res_dtype).to(tl.float32)  # aten casts avg_proj to res_dtype
    # adjust_rs (mode != 0): c = avg - proj ; add_vector / no-op (mode == 0): c = host coeff.
    c = tl.where(mode != 0, avg - proj, cf)                  # fp32
    # aten casts the selected coefficient to the residual dtype before the add; mirror it so
    # add_vector's c is the EXACT rounded host coeff (needed for bit-identical add_vector).
    c = c.to(res_dtype).to(tl.float32)

    # ------- Pass 2: residual[t, j] += c * vec_table[vid, j], in place (two roundings) -------
    for off in range(0, hidden, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < hidden
        u = tl.load(vec_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        # Round the product to the residual dtype BEFORE the add (matches aten's rounded
        # bf16 product); materializing it also blocks a single-rounding fp32 FMA.
        prod = (c * u).to(res_dtype).to(tl.float32)
        new = r + prod                                       # fp32 sum
        tl.store(res_row + cols, new.to(res_dtype), mask=mask)  # store rounds the sum


def steer_buffer_fused(residual, coeff, vec_id, vec_table, avg_proj, steer_mode, n=None):
    """Launch the fused steer kernel (in-place on ``residual[:n]``).

    ``n`` defaults to ``residual.shape[0]`` (the padded token count this step). One program
    per row (grid ``(n,)``); no host sync, fixed launch config → cudagraph-legal. ``n == 0``
    is a no-op (no launch). The routing tensors ``coeff``/``vec_id``/``steer_mode`` are length
    CAP ≥ n contiguous views; only the leading ``n`` entries are read.
    """
    if n is None:
        n = residual.shape[0]
    n = int(n)
    if n == 0:
        return
    hidden = int(residual.shape[1])
    grid = (n,)
    steer_buffer_fused_kernel[grid](
        residual, coeff, vec_id, vec_table, avg_proj, steer_mode,
        n, hidden,
        residual.stride(0), vec_table.stride(0),
        BLOCK=_BLOCK,
        EARLY_EXIT=_EARLY_EXIT,
        num_warps=_NUM_WARPS,
    )
