"""GPU scatter of the steer and capture routing slabs (MIA_STEER_GPU_ROUTING, MIA_CAPTURE_GPU_ROUTING)."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


if _HAVE_TRITON:
    @triton.jit
    def _request_for_column(qsl_ptr, cols, nreq, BLOCK: tl.constexpr):
        # searchsorted(qsl, cols, right=True) - 1, clamped, without a device temporary
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), nreq + 1, tl.int32)
        remaining = nreq + 1
        while remaining > 0:
            active = lo < hi
            mid = (lo + hi) // 2
            boundary = tl.load(qsl_ptr + mid, mask=active, other=0)
            right = active & (boundary <= cols)
            hi = tl.where(active & ~right, mid, hi)
            lo = tl.where(right, mid + 1, lo)
            remaining = remaining // 2
        return tl.minimum(tl.maximum(lo - 1, 0), nreq - 1)

    @triton.jit
    def _scatter_routing_kernel(
        qsl_ptr,
        svid_ptr,
        smode_ptr,
        scoeff_ptr,
        smask_ptr,
        scol_ptr,
        coeff_ptr,
        vecid_ptr,
        mode_ptr,
        real_n,
        nreq,
        width,
        num_layers,
        cap,
        BLOCK: tl.constexpr,
        HAS_COL: tl.constexpr,
    ):
        L = tl.program_id(0)
        pid = tl.program_id(1)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        cmask = cols < width
        valid = cols < real_n
        req = _request_for_column(qsl_ptr, cols, nreq, BLOCK)
        m = tl.load(smask_ptr + req * num_layers + L, mask=cmask, other=0)
        active = valid & (m != 0)
        if HAS_COL:
            sc = tl.load(scol_ptr + req, mask=cmask, other=-2)
            active = active & ((sc == -1) | (cols.to(tl.int64) == sc))
        coeff = tl.load(scoeff_ptr + req, mask=cmask, other=0.0)
        vid = tl.load(svid_ptr + req, mask=cmask, other=0)
        mode = tl.load(smode_ptr + req, mask=cmask, other=0)
        base = L * cap + cols
        tl.store(coeff_ptr + base, tl.where(active, coeff, 0.0), mask=cmask)
        tl.store(vecid_ptr + base, tl.where(active, vid, 0), mask=cmask)
        tl.store(mode_ptr + base, tl.where(active, mode, 0), mask=cmask)


_BLOCK = 256
_NUM_WARPS = 4


def _scatter_torch(qsl_dev, slot_vid, slot_mode, slot_coeff, slot_layer_mask,
                   coeff_all, vec_id_all, mode_all, real_n, width, slot_col=None):
    # Host-side fallback: in-place slab writes, no CUDA temporaries
    qsl = qsl_dev.cpu().tolist()
    masks = slot_layer_mask.cpu().tolist()
    vids, modes = slot_vid.cpu().tolist(), slot_mode.cpu().tolist()
    coeffs = slot_coeff.cpu().tolist()
    cols = slot_col.cpu().tolist() if slot_col is not None else None
    w = max(1, min(int(width), coeff_all.shape[1]))
    coeff_all[:, :w].zero_(); vec_id_all[:, :w].zero_(); mode_all[:, :w].zero_()
    for req in range(len(qsl) - 1):
        start, end = max(0, int(qsl[req])), min(w, int(real_n), int(qsl[req + 1]))
        if cols is not None and cols[req] != -1:
            start, end = max(start, cols[req]), min(end, cols[req] + 1)
        if start >= end:
            continue
        for layer in range(coeff_all.shape[0]):
            if masks[req][layer]:
                coeff_all[layer, start:end].fill_(coeffs[req])
                vec_id_all[layer, start:end].fill_(vids[req])
                mode_all[layer, start:end].fill_(modes[req])


def scatter_routing(qsl_dev, slot_vid, slot_mode, slot_coeff, slot_layer_mask,
                    coeff_all, vec_id_all, mode_all, real_n, width, slot_col=None,
                    backend="auto"):
    """Fill ``coeff_all/vec_id_all/mode_all[:, :width]`` from the slot config on the GPU."""
    num_layers, cap = coeff_all.shape
    w = max(1, min(int(width), cap))
    use_triton = (_HAVE_TRITON and coeff_all.is_cuda
                  and backend in ("auto", "triton"))
    if backend == "triton" and not (_HAVE_TRITON and coeff_all.is_cuda):
        raise RuntimeError("triton backend requested but unavailable / not on cuda")
    if not use_triton:
        _scatter_torch(qsl_dev, slot_vid, slot_mode, slot_coeff, slot_layer_mask,
                       coeff_all, vec_id_all, mode_all, real_n, w, slot_col)
        return
    if int(qsl_dev.numel()) <= 1:
        coeff_all[:, :w].zero_(); vec_id_all[:, :w].zero_(); mode_all[:, :w].zero_()
        return
    grid = (num_layers, (w + _BLOCK - 1) // _BLOCK)
    _scatter_routing_kernel[grid](
        qsl_dev, slot_vid, slot_mode, slot_coeff, slot_layer_mask,
        slot_col if slot_col is not None else slot_vid,
        coeff_all, vec_id_all, mode_all,
        int(real_n), int(qsl_dev.numel()) - 1, w, num_layers, cap,
        BLOCK=_BLOCK, HAS_COL=slot_col is not None, num_warps=_NUM_WARPS,
    )


if _HAVE_TRITON:
    @triton.jit
    def _scatter_capture_kernel(
        qsl_ptr,
        smask_ptr,
        out_ptr,
        real_n, nreq, width, num_layers, cap,
        BLOCK: tl.constexpr,
    ):
        L = tl.program_id(0)
        pid = tl.program_id(1)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        cmask = cols < width
        valid = cols < real_n
        req = _request_for_column(qsl_ptr, cols, nreq, BLOCK)
        m = tl.load(smask_ptr + req * num_layers + L, mask=cmask, other=0)
        active = valid & (m != 0)
        idx = (cols + 1).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + L * cap + cols, tl.where(active, idx, 0), mask=cmask)


def _scatter_capture_torch(qsl_dev, slot_layer_mask, capture_index_all, real_n, w):
    qsl = qsl_dev.cpu().tolist()
    masks = slot_layer_mask.cpu().tolist()
    capture_index_all[:, :w].zero_()
    for req in range(len(qsl) - 1):
        start, end = max(0, int(qsl[req])), min(w, int(real_n), int(qsl[req + 1]))
        if start >= end:
            continue
        indices = torch.arange(start + 1, end + 1, dtype=capture_index_all.dtype, device="cpu")
        for layer in range(capture_index_all.shape[0]):
            if masks[req][layer]:
                capture_index_all[layer, start:end].copy_(indices)


def scatter_capture_routing(qsl_dev, slot_layer_mask, capture_index_all, real_n, width,
                            backend="auto"):
    """Fill ``capture_index_all[:, :width]`` from a per-slot capture layer mask on the GPU."""
    num_layers, cap = capture_index_all.shape
    w = max(1, min(int(width), cap))
    use_triton = (_HAVE_TRITON and capture_index_all.is_cuda and backend in ("auto", "triton"))
    if backend == "triton" and not (_HAVE_TRITON and capture_index_all.is_cuda):
        raise RuntimeError("triton backend requested but unavailable / not on cuda")
    if not use_triton:
        _scatter_capture_torch(qsl_dev, slot_layer_mask, capture_index_all, real_n, w)
        return
    if int(qsl_dev.numel()) <= 1:
        capture_index_all[:, :w].zero_(); return
    grid = (num_layers, (w + _BLOCK - 1) // _BLOCK)
    _scatter_capture_kernel[grid](qsl_dev, slot_layer_mask, capture_index_all,
                                  int(real_n), int(qsl_dev.numel()) - 1, w, num_layers, cap,
                                  BLOCK=_BLOCK, num_warps=_NUM_WARPS)


__all__ = ["scatter_routing", "scatter_capture_routing"]

