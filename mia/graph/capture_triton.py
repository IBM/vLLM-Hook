"""Triton-fused capture scatter — one kernel for ``capture_hs``.

Gated behind ``MIA_CAPTURE_FUSED`` (default ON; ``=0`` is the kill switch / aten-fallback
control). Unless disabled, ``ops._capture_hs_impl`` dispatches here instead of the aten body.

WHY. The aten body issues TWO kernels per layer: ``hidden + residual`` (reads 2, writes 1) then
``hs_buf.index_copy_`` (reads 1, writes 1) — 5 passes over ``n x hidden`` where a plain copy
needs 2. ``index_copy_`` is also aten's generic TensorIterator scatter, measurably slower than a
contiguous copy of the same bytes: the penalty is per-element index arithmetic and lost
vectorization, NOT coalescing (each destination row is ``hidden`` contiguous elements). Hoisting
the index load to ONE PER ROW and tiling the row recovers copy bandwidth with arbitrary indices,
and folding the add in cuts the memory traffic by roughly 40%.

Reference math matched verbatim (``graph/ops.py:_capture_hs_impl``):
    combined = hidden + residual if has_residual else hidden
    hs_buf.index_copy_(0, index[:n].to(int64), combined.to(hs_buf.dtype))

BIT-EXACTNESS. aten's bf16/fp16 add uses an fp32 opmath type internally and rounds ONCE on
store. The kernel mirrors that exactly: load both operands, widen to fp32, add in fp32, and let
``tl.store`` round once into the buffer dtype. Result is bit-identical, not merely close — the
GPU oracle asserts ``torch.equal``.

DESTINATION BOUNDS (load-bearing since fusion became the DEFAULT). The aten body this kernel
replaced, ``hs_buf.index_copy_(0, idx, src)``, bounds-checks EVERY index: an out-of-range
destination raises ``IndexError`` on CPU and trips ``CUDA_KERNEL_ASSERT`` inside
``indexCopy{Small,Large}Index`` on GPU. ``tl.store``'s ``mask`` covers only the HIDDEN dimension,
so a fused kernel without an explicit destination check turns that loud crash into a SILENT write
into whatever allocation follows ``hs_buf`` — KV cache, weights. The kernel therefore clamps
``dst`` into ``[0, SENTINEL]`` — aten's own legal domain — so the select is the identity for
every legally-routed index (byte-identity preserved, proven cell-by-cell by
``capture_fused_unit.py`` on the capture_aperture branch) and an illegal one lands on the
discard row instead of another allocation.

  Why SILENT-degrade and not loud. Loud is not reachable from inside a captured cudagraph: a
  host-side check of a device value needs a sync (forbidden on this path and illegal during
  capture), and ``tl.device_assert`` is compiled out unless Triton debug is armed and, when it
  does fire, poisons the CUDA context for the whole process rather than raising a catchable
  error. Silent-degrade is also what the padding bound one line above already does, so the two
  out-of-domain cases stay in one category. A DEVICE-side out-of-range counter was declined: it
  needs an atomic plus a new fixed-address buffer that must be allocated before graph capture and
  never during it, putting fresh capture-time allocation risk on the DEFAULT path to observe a
  condition that is unreachable without a separate builder bug (routing destinations already come
  bounded, ``% n_slots``, from ``CaptureAperture.physical_slots``). The condition is instead made
  OBSERVABLE by proof rather than by telemetry: ``run_oob_cells()`` in the oracle drives
  genuinely out-of-range destinations (negative, ``> SENTINEL``, and far past the buffer) and
  asserts every such row lands on SENTINEL with all in-range rows untouched.

SENTINEL ROWS. Unselected tokens route to ``hs_buf``'s last row (``CaptureAperture.SENTINEL``),
so several source rows may target the same destination concurrently. Those writes race — exactly
as they do under ``index_copy_`` with duplicate indices — and the row is discarded at drain. The
sentinel row's CONTENT is therefore undefined on both paths and must be excluded from any
equality assertion.

cudagraph-safety: NO ``@triton.autotune`` (multi-config timing launches are illegal during graph
capture), no host sync in the launcher, fixed ``BLOCK``/``num_warps``, ``n == 0`` performs no
launch. vLLM's pre-capture warmup forwards JIT-compile the kernel before capture, so replay never
re-enters Python.

Importable without a GPU: importing ``triton`` is fine; nothing launches at import time.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Fixed launch config (NO autotune — a single-config launch is cudagraph-legal). BLOCK tiles the
# hidden dimension; the grid's second axis covers hidden/BLOCK tiles and a masked tail handles
# hidden % BLOCK != 0 (e.g. Qwen2-1.5B hidden=1536).
_BLOCK = 1024
_NUM_WARPS = 4

# Buffer dtypes whose tl.store rounding we have verified against aten. Anything else (fp8, int8
# quantized sinks) falls back to the aten body rather than risk a different rounding.
FUSED_OK_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def capture_hs_fused_kernel(
    hidden_ptr,      # (n, H)    model dtype
    residual_ptr,    # (n, H)    model dtype — aliases hidden_ptr when HAS_RESIDUAL == 0
    buf_ptr,         # (R+1, H)  buffer dtype — mutated IN PLACE
    index_ptr,       # (>= n,)   int64 destination row per source row
    hidden,
    stride_h, stride_r, stride_buf,
    HAS_RESIDUAL: tl.constexpr,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    # ONE destination load per ROW (not per element) — this is the whole point of the kernel.
    dst = tl.load(index_ptr + row)

    # DESTINATION BOUNDS CLAMP -- restores the bounds check the aten body had. See the module
    # docstring's "DESTINATION BOUNDS" section for why this is load-bearing now that fusion is
    # the DEFAULT. `[0, SENTINEL]` is exactly aten `index_copy_`'s legal destination domain
    # (`hs_buf` is `(R+1, H)` and `SENTINEL == R`), so this select is the IDENTITY on every
    # index the routing builders can legally produce -- byte-identity is unaffected. Out of
    # that range it degrades to the SENTINEL discard row, the same category the padding clamp
    # above already uses, instead of storing past `hs_buf`'s allocation. One compare-pair +
    # select per ROW (not per element), and cudagraph-legal: no host read, no sync, no branch
    # on a device value.
    dst = tl.where((dst >= 0) & (dst <= SENTINEL), dst, SENTINEL)

    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden

    v = tl.load(hidden_ptr + row * stride_h + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        v += tl.load(residual_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)

    # Single rounding on store — matches aten's fp32-opmath add + one round.
    tl.store(buf_ptr + dst * stride_buf + offs, v, mask=mask)


def capture_hs_fused(hidden, residual, hs_buf, index, has_residual, n=None) -> None:
    """Launch the fused capture scatter (in-place on ``hs_buf``).

    ``n`` defaults to ``hidden.shape[0]`` (the PADDED token count this step, which is what the
    cudagraph is captured for — same semantics as the aten body's ``n = hidden.shape[0]``).
    ``index`` is length CAP >= n; only the leading ``n`` entries are read.
    """
    if n is None:
        n = hidden.shape[0]
    n = int(n)
    if n == 0:
        return
    h = int(hidden.shape[1])
    sentinel = int(hs_buf.shape[0]) - 1   # hs_buf is (R+1, H); row R is the discard/pad sentinel
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
