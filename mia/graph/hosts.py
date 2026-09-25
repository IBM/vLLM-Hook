"""Per-attention-layer static-buffer host for CUDA-graph QK capture."""
from __future__ import annotations

import torch
import torch.nn as nn


class QKCaptureHost(nn.Module):
    """Static-buffer host for a single attention module's QK capture.

    Owns static ``q_buf``/``k_buf`` sized ``(cap + 1, dim)`` with row 0 a
    write-only discard sentinel. ``capture_index``/``any_active`` are views into
    the registry slabs (bound later, not owned here). ``do_capture`` is an
    install-time gate (True only on the capture rank) read by install.py to
    decide whether to call ``capture()`` — never a branch inside the op.
    ``module_name`` is the egress bucket key, matching the eager ``qkv_hook``
    so analyzers/disk-save read the same buckets (the reuse contract).
    """

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

        # Row 0 = discard sentinel (never read by egress). register_buffer pins
        # data_ptr across replays, but only if the host is built BEFORE vLLM's
        # cudagraph pool is allocated — the load_model patch guarantees that.
        # persistent=False keeps these out of the state_dict. A caller MAY pass a
        # slab ROW-VIEW (q_buf/k_buf) instead of letting the host allocate its own —
        # every layer's buffer then aliases one contiguous slab, so egress can gather
        # all layers in ONE index_select; a slab view has a fixed data_ptr (slab base
        # + layer offset) since the slab is allocated before the pool, exactly like an
        # owned buffer. The op scatters into the view, which lands in the slab.
        # (Currently unused for QK — see HSCaptureHost.hs_buf for the live consumer,
        # the vectorized HS egress path.)
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

        # Routing views, populated later by bind_views(). Plain attributes, not
        # buffers: the registry owns the backing slabs, so registering would
        # double-register the same storage.
        self.capture_index: torch.Tensor | None = None
        self.any_active: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # View binding (called by the registry, once, after construction).
    # ------------------------------------------------------------------

    def bind_views(self, capture_index: torch.Tensor, any_active: torch.Tensor) -> None:
        """Attach this layer's row-views into the registry's global slabs.

        ``capture_index`` is a ``(cap,)`` int64 view; ``any_active`` is 0-d. The
        registry refreshes both each step; ``capture()`` only reads them, so the
        bindings hold for the host's lifetime.
        """
        self.capture_index = capture_index
        self.any_active = any_active

    # ------------------------------------------------------------------
    # Graph-recorded capture (the only thing the wrap calls).
    # ------------------------------------------------------------------

    def capture(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Scatter post-RoPE ``q``/``k`` rows into the static buffers.

        Single ``mia::capture_qk`` op call (graph-legal). The op aligns
        ``capture_index`` to the live token count and writes each row to its
        routed slot; tokens routed to row 0 (the sentinel) are discarded.
        ``any_active`` is passed for schema stability only — sentinel-row routing
        already makes unrequested tokens a no-op (see graph/ops.py).
        """
        torch.ops.mia.capture_qk(
            q, k, self.q_buf, self.k_buf, self.capture_index, self.any_active
        )


class HSCaptureHost(nn.Module):
    """Static-buffer host for a single decoder layer's hidden-state capture.

    The HS analogue of :class:`QKCaptureHost`. Owns a static ``hs_buf`` sized
    ``(cap + 1, hidden)`` with row 0 a write-only discard sentinel. The
    ``capture_hs`` op forms the residual stream (``hidden + residual`` for
    fused-residual layers) and scatters its rows into ``hs_buf`` by the routed
    ``capture_index`` — a graph-recorded kernel that replays under FULL decode.

    Unlike QK there is no key history to reconstruct: the residual stream is
    per-token, so the scattered rows ARE the artifact (no prefix-K plane).

    ``layer_num`` is the 0-based registry-slab row (matched-module order);
    ``egress_layer_num`` is the 1-based number stored in the worker bucket so the
    artifact matches the eager hidden-states path. ``has_residual`` records the
    layer's output structure (fused (hidden, residual) tuple vs single tensor),
    determined at install time so the op call site bakes a constant.
    """

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

        # Row 0 = discard sentinel (never read by egress). register_buffer pins
        # data_ptr across replays, but only if built BEFORE vLLM's cudagraph pool
        # is allocated — the load_model patch guarantees that. persistent=False
        # keeps it out of the state_dict. ``hs_buf`` accepts an optional pre-built
        # view (see QKCaptureHost for the slab rationale used on the QK side); no
        # current HS caller passes one, so each host owns its own buffer.
        self.register_buffer(
            "hs_buf",
            hs_buf if hs_buf is not None
            else torch.zeros(self.cap + 1, self.hidden, dtype=dtype, device=device),
            persistent=False,
        )

        # Routing views, bound later by the registry (it owns the backing slabs).
        self.capture_index: torch.Tensor | None = None
        self.any_active: torch.Tensor | None = None

    def bind_views(self, capture_index: torch.Tensor, any_active: torch.Tensor) -> None:
        """Attach this layer's row-views into the registry's global slabs."""
        self.capture_index = capture_index
        self.any_active = any_active

    def capture(self, hidden: torch.Tensor, residual: torch.Tensor) -> None:
        """Scatter the residual stream rows into the static buffer.

        Single ``mia::capture_hs`` op call (graph-legal). ``has_residual``
        is baked at install time, so the op site carries no data-dependent branch.
        """
        torch.ops.mia.capture_hs(
            hidden, residual, self.hs_buf, self.capture_index, self.has_residual
        )


class SteerHost(nn.Module):
    """Per-decoder-layer host for buffer-mode activation steering (the MUTATING
    counterpart of QKCaptureHost/HSCaptureHost).

    Owns NO per-layer sink buffer — steering writes the residual in place, so the
    only static GPU state is the per-(layer, token) routing views (``coeff``,
    ``vec_id``, ``mode``, registry-owned) and the shared ``vec_table`` + ``avg_proj``
    tables. ``steer()`` calls the ``steer_buffer`` op (masked in-place add); ``mode``
    selects add_vector (host ``coeff``) vs adjust_rs (in-kernel ``avg_proj - proj``)
    per token, and ``coeff == 0`` / ``mode == 0`` rows are no-ops, so a layer no
    request steers costs one masked add. ``do_steer`` is an install-time gate.
    ``layer_num`` is 0-based (matches the eager worker's ``this_layer`` compared
    against config ``optimal_layer``).
    """

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

        # Routing views + shared tables, bound later by the registry.
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
