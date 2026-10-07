"""Per-layer static-buffer hosts for CUDA-graph QK and HS capture and buffer-mode steering."""
from __future__ import annotations

import torch
import torch.nn as nn


class QKCaptureHost(nn.Module):
    """Static-buffer host for a single attention module's QK capture."""

    def __init__(
        self,
        module_name: str,
        layer_num: int,
        cap: int,
        q_dim: int,
        k_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        do_capture: bool = True,
        q_buf: torch.Tensor | None = None,
        k_buf: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.layer_num = int(layer_num)
        self.cap = int(cap)
        self.q_dim = int(q_dim)
        self.k_dim = int(k_dim)
        self.do_capture = bool(do_capture)

        self.register_buffer(
            "q_buf",
            q_buf if q_buf is not None
            else torch.zeros(self.cap + 1, self.q_dim, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "k_buf",
            k_buf if k_buf is not None
            else torch.zeros(self.cap + 1, self.k_dim, dtype=dtype, device=device),
            persistent=False,
        )

        self.capture_index: torch.Tensor | None = None
        self.any_active: torch.Tensor | None = None


    def bind_views(self, capture_index: torch.Tensor, any_active: torch.Tensor) -> None:
        """Attach this layer's row-views into the registry's global slabs."""
        self.capture_index = capture_index
        self.any_active = any_active


    def capture(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Scatter post-RoPE ``q``/``k`` rows into the static buffers."""
        torch.ops.mia.capture_qk(
            q, k, self.q_buf, self.k_buf, self.capture_index, self.any_active
        )


class HSCaptureHost(nn.Module):
    """Static-buffer host for a single decoder layer's hidden-state capture."""

    def __init__(
        self,
        module_name: str,
        layer_num: int,
        egress_layer_num: int,
        cap: int,
        hidden: int,
        dtype: torch.dtype,
        device: torch.device | str,
        has_residual: int = 1,
        do_capture: bool = True,
        hs_buf: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.layer_num = int(layer_num)
        self.egress_layer_num = int(egress_layer_num)
        self.cap = int(cap)
        self.hidden = int(hidden)
        self.has_residual = int(has_residual)
        self.do_capture = bool(do_capture)

        self.register_buffer(
            "hs_buf",
            hs_buf if hs_buf is not None
            else torch.zeros(self.cap + 1, self.hidden, dtype=dtype, device=device),
            persistent=False,
        )

        self.capture_index: torch.Tensor | None = None
        self.any_active: torch.Tensor | None = None

    def bind_views(self, capture_index: torch.Tensor, any_active: torch.Tensor) -> None:
        """Attach this layer's row-views into the registry's global slabs."""
        self.capture_index = capture_index
        self.any_active = any_active

    def capture(self, hidden: torch.Tensor, residual: torch.Tensor) -> None:
        """Scatter the residual stream rows into the static buffer."""
        torch.ops.mia.capture_hs(
            hidden, residual, self.hs_buf, self.capture_index, self.has_residual
        )


class SteerHost(nn.Module):
    """Per-decoder-layer host for buffer-mode activation steering."""

    def __init__(
        self,
        module_name: str,
        layer_num: int,
        cap: int,
        do_steer: bool = True,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.layer_num = int(layer_num)
        self.cap = int(cap)
        self.do_steer = bool(do_steer)

        self.coeff: torch.Tensor | None = None
        self.vec_id: torch.Tensor | None = None
        self.mode: torch.Tensor | None = None
        self.vec_table: torch.Tensor | None = None
        self.avg_proj: torch.Tensor | None = None

    def bind_views(self, coeff: torch.Tensor, vec_id: torch.Tensor,
                   mode: torch.Tensor, vec_table: torch.Tensor,
                   avg_proj: torch.Tensor) -> None:
        """Attach this layer's coeff/vec_id/mode row-views + the shared tables."""
        self.coeff = coeff
        self.vec_id = vec_id
        self.mode = mode
        self.vec_table = vec_table
        self.avg_proj = avg_proj

    def steer(self, residual: torch.Tensor) -> None:
        """Apply the masked in-place steering add (single ``steer_buffer`` op)."""
        torch.ops.mia.steer_buffer(
            residual, self.coeff, self.vec_id, self.vec_table,
            self.avg_proj, self.mode,
        )

