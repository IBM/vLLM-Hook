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
    def _scatter_routing_kernel(
        req_ptr,
        svid_ptr,
        smode_ptr,
        scoeff_ptr,
        smask_ptr,
        scol_ptr,
        coeff_ptr,
        vecid_ptr,
        mode_ptr,
        real_n,
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
        req = tl.load(req_ptr + cols, mask=cmask, other=0)
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


def _req_of_col(qsl_dev, width):
    bs = int(qsl_dev.numel()) - 1
    col = torch.arange(width, device=qsl_dev.device, dtype=qsl_dev.dtype)
    r = torch.searchsorted(qsl_dev, col, right=True) - 1
    return r.clamp_(0, max(bs - 1, 0)).to(torch.int32)


def _scatter_torch(qsl_dev, slot_vid, slot_mode, slot_coeff, slot_layer_mask,
                   coeff_all, vec_id_all, mode_all, real_n, width, slot_col=None):
    dev = coeff_all.device
    num_layers, cap = coeff_all.shape
    w = max(1, min(int(width), cap))
    bs = int(qsl_dev.numel()) - 1
    if bs <= 0:
        coeff_all[:, :w].zero_(); vec_id_all[:, :w].zero_(); mode_all[:, :w].zero_()
        return
    col = torch.arange(w, device=dev, dtype=qsl_dev.dtype)
    req = (torch.searchsorted(qsl_dev, col, right=True) - 1).clamp_(0, bs - 1)
    valid = col < int(real_n)
    if slot_col is not None:
        sc = slot_col[req]
        valid = valid & ((sc == -1) | (col.to(sc.dtype) == sc))
    vid_c = torch.where(valid, slot_vid[req], torch.zeros_like(req))
    mode_c = torch.where(valid, slot_mode[req], torch.zeros_like(req))
    coef_c = torch.where(valid, slot_coeff[req], torch.zeros(w, dtype=torch.float32, device=dev))
    mask = slot_layer_mask[req].transpose(0, 1) & valid.unsqueeze(0)
    coeff_all[:, :w] = torch.where(mask, coef_c.unsqueeze(0).expand(num_layers, -1), 0.0)
    vec_id_all[:, :w] = torch.where(mask, vid_c.unsqueeze(0).expand(num_layers, -1), 0)
    mode_all[:, :w] = torch.where(mask, mode_c.unsqueeze(0).expand(num_layers, -1), 0)


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
    req = _req_of_col(qsl_dev, w)
    smask_i8 = slot_layer_mask.to(torch.int8)
    grid = (num_layers, (w + _BLOCK - 1) // _BLOCK)
    _scatter_routing_kernel[grid](
        req, slot_vid, slot_mode, slot_coeff, smask_i8,
        slot_col if slot_col is not None else slot_vid,
        coeff_all, vec_id_all, mode_all,
        int(real_n), w, num_layers, cap,
        BLOCK=_BLOCK, HAS_COL=slot_col is not None, num_warps=_NUM_WARPS,
    )


if _HAVE_TRITON:
    @triton.jit
    def _scatter_capture_kernel(
        req_ptr,
        smask_ptr,
        out_ptr,
        real_n, width, num_layers, cap,
        BLOCK: tl.constexpr,
    ):
        L = tl.program_id(0)
        pid = tl.program_id(1)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        cmask = cols < width
        valid = cols < real_n
        req = tl.load(req_ptr + cols, mask=cmask, other=0)
        m = tl.load(smask_ptr + req * num_layers + L, mask=cmask, other=0)
        active = valid & (m != 0)
        idx = (cols + 1).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + L * cap + cols, tl.where(active, idx, 0), mask=cmask)


def _scatter_capture_torch(qsl_dev, slot_layer_mask, capture_index_all, real_n, w):
    dev = capture_index_all.device
    num_layers, cap = capture_index_all.shape
    bs = int(qsl_dev.numel()) - 1
    if bs <= 0:
        capture_index_all[:, :w].zero_(); return
    col = torch.arange(w, device=dev, dtype=qsl_dev.dtype)
    req = (torch.searchsorted(qsl_dev, col, right=True) - 1).clamp_(0, bs - 1)
    valid = col < int(real_n)
    mask = slot_layer_mask[req].transpose(0, 1) & valid.unsqueeze(0)
    idx = (torch.arange(w, device=dev, dtype=capture_index_all.dtype) + 1).unsqueeze(0)
    capture_index_all[:, :w] = torch.where(mask, idx.expand(num_layers, -1),
                                           torch.zeros((), dtype=capture_index_all.dtype, device=dev))


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
    req = _req_of_col(qsl_dev, w)
    smask_i8 = slot_layer_mask.to(torch.int8)
    grid = (num_layers, (w + _BLOCK - 1) // _BLOCK)
    _scatter_capture_kernel[grid](req, smask_i8, capture_index_all,
                                  int(real_n), w, num_layers, cap,
                                  BLOCK=_BLOCK, num_warps=_NUM_WARPS)


__all__ = ["scatter_routing", "scatter_capture_routing"]

