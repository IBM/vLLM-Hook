"""Lever A production scatter: fill the steer routing slabs on the GPU (env-gated).

``scatter_routing`` expands the per-slot config table (``SteerRegistry.slot_vid/slot_mode/
slot_coeff/slot_layer_mask``) + the on-GPU ``query_start_loc`` into the SAME
``coeff_all/vec_id_all/mode_all`` slabs the baked ``steer_buffer`` op reads — replacing the
host ``apply_incremental_routing`` build. The result is byte-identical to the host build: it is
a pure gather/scatter of exact int/float values, no arithmetic.

Two backends, same output:
  * **triton** (default when available + cuda): ONE ``searchsorted`` (req-of-column) + ONE
    fused kernel over a ``(num_layers, columns)`` grid -> a small, ~constant launch cost that
    does NOT scale with N or B (the original torch scatter was launch-bound; this collapses
    many small launches into two).
  * **torch** (fallback): the original ``scatter_into`` body, kept as the byte-identical
    reference and the no-Triton path.

``real_n`` is passed as a HOST int (from ``qsl_cpu[-1]``) so the kernel needs no ``.item()``
D2H sync. ``n==0`` -> zero the read region, no gather. Any error -> raise (the caller in
``install.py`` catches and falls back to the host routing path; never load-bearing).
"""
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
        req_ptr,            # (width,)            int32  request index owning each column
        svid_ptr,           # (>=bs,)             int64  per-slot vec id
        smode_ptr,          # (>=bs,)             int64  per-slot mode (0 add_vector, 1 adjust_rs)
        scoeff_ptr,         # (>=bs,)             fp32   per-slot coefficient
        smask_ptr,          # (>=bs, num_layers)  int8   per-slot layer mask (row-major)
        scol_ptr,           # (>=bs,)             int64  per-slot steer column (see HAS_COL)
        coeff_ptr,          # (num_layers, cap)   fp32   OUT
        vecid_ptr,          # (num_layers, cap)   int64  OUT
        mode_ptr,           # (num_layers, cap)   int64  OUT
        real_n,             # host int: columns [0, real_n) are real, [real_n, width) padding
        width,              # host int: columns to write this step
        num_layers,
        cap,
        BLOCK: tl.constexpr,
        HAS_COL: tl.constexpr,
    ):
        L = tl.program_id(0)                      # layer row
        pid = tl.program_id(1)                    # column block
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        cmask = cols < width
        valid = cols < real_n                     # real vs cudagraph-padding columns
        req = tl.load(req_ptr + cols, mask=cmask, other=0)
        m = tl.load(smask_ptr + req * num_layers + L, mask=cmask, other=0)
        active = valid & (m != 0)
        if HAS_COL:
            # phase x positions gate: -1 = whole span, -2 = nothing, >=0 = that column.
            # cols is always >= 0 so the -2 sentinel can never match. Compare in int64 —
            # an unsigned promotion would make cols == -2 reachable and steer a padding
            # column.
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
    """Column -> owning request index via one searchsorted (GPU). Clamped in-range; padding
    columns get an arbitrary in-range index and are zeroed by the `valid` mask downstream."""
    bs = int(qsl_dev.numel()) - 1
    # col dtype must match qsl (production query_start_loc.gpu is int32) so searchsorted agrees.
    col = torch.arange(width, device=qsl_dev.device, dtype=qsl_dev.dtype)
    r = torch.searchsorted(qsl_dev, col, right=True) - 1
    return r.clamp_(0, max(bs - 1, 0)).to(torch.int32)


def _scatter_torch(qsl_dev, slot_vid, slot_mode, slot_coeff, slot_layer_mask,
                   coeff_all, vec_id_all, mode_all, real_n, width, slot_col=None):
    """Byte-identical fallback (the proven Phase-0 body), into resident slabs, no host sync."""
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
        # phase x positions gate; see the Triton kernel's HAS_COL branch.
        sc = slot_col[req]
        valid = valid & ((sc == -1) | (col.to(sc.dtype) == sc))
    vid_c = torch.where(valid, slot_vid[req], torch.zeros_like(req))
    mode_c = torch.where(valid, slot_mode[req], torch.zeros_like(req))
    coef_c = torch.where(valid, slot_coeff[req], torch.zeros(w, dtype=torch.float32, device=dev))
    mask = slot_layer_mask[req].transpose(0, 1) & valid.unsqueeze(0)     # (num_layers, w)
    coeff_all[:, :w] = torch.where(mask, coef_c.unsqueeze(0).expand(num_layers, -1), 0.0)
    vec_id_all[:, :w] = torch.where(mask, vid_c.unsqueeze(0).expand(num_layers, -1), 0)
    mode_all[:, :w] = torch.where(mask, mode_c.unsqueeze(0).expand(num_layers, -1), 0)


def scatter_routing(qsl_dev, slot_vid, slot_mode, slot_coeff, slot_layer_mask,
                    coeff_all, vec_id_all, mode_all, real_n, width, slot_col=None,
                    backend="auto"):
    """Fill ``coeff_all/vec_id_all/mode_all[:, :width]`` from the slot config + qsl (GPU).

    ``slot_col`` is the optional per-slot phase x positions gate: a ``(>=bs,)`` int64
    tensor whose entries are ``-1`` (steer the whole span — today's behaviour), ``-2``
    (steer nothing this step), or an absolute flat column. ``None`` means every slot
    steers its whole span and is BYTE-IDENTICAL to the pre-gate kernel — the gate load
    and compare are elided at compile time via the ``HAS_COL`` constexpr, so the default
    path costs nothing.

    ``backend``: "auto" (triton on cuda, else torch), "triton", or "torch".
    Writes in place; the tail beyond ``width`` is left stale (the op reads only ``[:, :n<=width]``).
    """
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
    smask_i8 = slot_layer_mask.to(torch.int8)              # bool -> int8 for the kernel load
    grid = (num_layers, (w + _BLOCK - 1) // _BLOCK)
    _scatter_routing_kernel[grid](
        req, slot_vid, slot_mode, slot_coeff, smask_i8,
        slot_col if slot_col is not None else slot_vid,    # dummy ptr when HAS_COL=False
        coeff_all, vec_id_all, mode_all,
        int(real_n), w, num_layers, cap,
        BLOCK=_BLOCK, HAS_COL=slot_col is not None, num_warps=_NUM_WARPS,
    )


if _HAVE_TRITON:
    @triton.jit
    def _scatter_capture_kernel(
        req_ptr,            # (width,)            int32  request index owning each column
        smask_ptr,          # (>=bs, num_layers)  int8   per-slot capture layer mask (row-major)
        out_ptr,            # (num_layers, cap)   int64  OUT: capture_index_all
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
        # capture_index_all[L, c] = c+1 iff layer L active at column c, else 0 (position-
        # deterministic, matching HostRegistry.apply_incremental_routing).
        idx = (cols + 1).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + L * cap + cols, tl.where(active, idx, 0), mask=cmask)


def _scatter_capture_torch(qsl_dev, slot_layer_mask, capture_index_all, real_n, w):
    """Byte-identical fallback for the capture scatter (into the resident slab)."""
    dev = capture_index_all.device
    num_layers, cap = capture_index_all.shape
    bs = int(qsl_dev.numel()) - 1
    if bs <= 0:
        capture_index_all[:, :w].zero_(); return
    col = torch.arange(w, device=dev, dtype=qsl_dev.dtype)
    req = (torch.searchsorted(qsl_dev, col, right=True) - 1).clamp_(0, bs - 1)
    valid = col < int(real_n)
    mask = slot_layer_mask[req].transpose(0, 1) & valid.unsqueeze(0)          # (num_layers, w)
    idx = (torch.arange(w, device=dev, dtype=capture_index_all.dtype) + 1).unsqueeze(0)
    capture_index_all[:, :w] = torch.where(mask, idx.expand(num_layers, -1),
                                           torch.zeros((), dtype=capture_index_all.dtype, device=dev))


def scatter_capture_routing(qsl_dev, slot_layer_mask, capture_index_all, real_n, width,
                            backend="auto"):
    """Fill ``capture_index_all[:, :width]`` (QK/HS routing) from a per-slot capture layer-mask
    + qsl (GPU). Value at (L, c) is ``c+1`` iff column c's request captures layer L (and c is
    real), else 0 — byte-identical to ``HostRegistry.apply_incremental_routing``. Same 2-launch
    Triton path as the steer scatter; ``backend`` = "auto"/"triton"/"torch"."""
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
