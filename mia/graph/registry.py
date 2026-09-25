"""Per-worker device routing slabs for CUDA-graph QK capture."""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from mia._profiler import PROF
from mia.graph.hosts import QKCaptureHost

# Incremental/persistent capture routing. Default ON; set to 0 to fall back to
# the legacy per-step full-width reset+build+upload. Only HostRegistry (QK/HS
# capture) honours it; SteerRegistry keeps its own key-skip.
INCREMENTAL_ROUTING = os.environ.get("MIA_INCREMENTAL_ROUTING", "1") != "0"

# Depth of the pinned-mirror aperture. >=2 lets reset_pinned() reuse a slot whose
# H2D upload was enqueued >=2 steps ago -- its event is already complete, so
# the wait never stalls the decode hot path. 1 == a single mirror, which forces
# a per-step CPU<->GPU serialization. vLLM's async scheduler runs the CPU at
# most ~1 step ahead of the GPU, so 2 is sufficient.
APERTURE_DEPTH = max(1, int(os.environ.get("MIA_PINNED_MIRROR", "2")))


# ---------------------------------------------------------------------------
# Per-subsystem registry storage.
# ---------------------------------------------------------------------------
# A single scalar attribute on the worker cannot hold three subsystems at once: QK
# writes it at install (install.py), then HS installing after QK would silently
# overwrite QK's registry (steer the same), so whichever subsystem installs LAST
# silently wins and the others route nothing. Keyed storage under one dict attribute
# makes co-installation expressible now (Task D1) and correct later, when one engine
# serves a mixed HS+QK+steer batch (Plan 2) -- do not collapse this back to a single
# slot.
_REGISTRIES_ATTR = "_mia_registries"


def set_registry(worker, subsystem: str, registry) -> None:
    """Register one subsystem's HostRegistry/SteerRegistry on the worker, keyed by
    subsystem name ("hs" | "qk" | "steer"). Idempotent to call repeatedly; each
    subsystem's slot is independent of the others."""
    registries = getattr(worker, _REGISTRIES_ATTR, None)
    if registries is None:
        registries = {}
        setattr(worker, _REGISTRIES_ATTR, registries)
    registries[subsystem] = registry


def get_registry(worker, subsystem: str):
    """The registry registered for `subsystem`, or None if that subsystem never
    installed (or installed with no capturable model)."""
    return getattr(worker, _REGISTRIES_ATTR, {}).get(subsystem)


class PinnedMirror:
    """Depth-K aperture of pinned-host mirror *sets* + per-slot CUDA upload events.

    The capture/steer routing is staged in pinned host memory, then copied H2D into
    the device slabs each step. With a SINGLE mirror, ``reset_pinned`` must
    ``event.synchronize()`` before zeroing it (zeroing a mirror whose H2D copy is
    still in flight would latch zeros = silent capture/steer loss) — a per-step
    CPU<->GPU serialization that breaks vLLM's async decode overlap. With K>=2 slots
    the slot reused this step was last uploaded K steps ago, so its event is already
    complete and the wait returns immediately (kept only as a correctness guard).

    Each slot holds one pinned tensor per ``(name, shape, dtype)`` spec. The owning
    registry exposes ``registry.<name>_pinned`` as the CURRENT slot's tensor (a
    property), so build code writes the active mirror unchanged; ``upload`` copies the
    current slot to the device slabs (registry-specific) and advances the cursor.
    """

    def __init__(self, specs, pin: bool, depth: int = APERTURE_DEPTH) -> None:
        self.depth = max(1, int(depth))
        self.idx = 0
        self.slots: List[Dict[str, torch.Tensor]] = [
            {
                name: torch.zeros(shape, dtype=dtype, pin_memory=pin)
                for name, shape, dtype in specs
            }
            for _ in range(self.depth)
        ]
        # One event per slot; None off-CUDA (copies are synchronous there).
        self.events: List[Optional[torch.cuda.Event]] = [
            torch.cuda.Event() if pin else None for _ in range(self.depth)
        ]

    def cur(self, name: str) -> torch.Tensor:
        """The current slot's tensor for ``name`` (what build code writes/upload reads)."""
        return self.slots[self.idx][name]

    def wait_current(self) -> None:
        """Block until the current slot's last upload (K steps ago) is done.

        On a warm aperture this event is already complete, so this does not stall. The
        wait is an UNCONDITIONAL synchronize, so an under-deep aperture (or a CPU that
        runs further ahead than the aperture covers) only degrades to a stall — it can
        never overwrite a mirror mid-H2D. MIA_PINNED_MIRROR is thus a pure perf
        knob with no correctness floor; a never-recorded event reports complete, so
        the depth-2 warm-up and any never-reused slot are safe.
        """
        ev = self.events[self.idx]
        if ev is not None:
            ev.synchronize()

    def record_advance(self) -> None:
        """Record the current slot's upload event, then rotate to the next slot."""
        ev = self.events[self.idx]
        if ev is not None:
            ev.record()
        self.idx = (self.idx + 1) % self.depth


class HostRegistry:
    """Per-worker owner of the QK routing slabs and host directory.

    Rows = ``num_layers`` (hosts index by ``layer_num``); columns = ``cap`` token
    capacity (= ``max_num_batched_tokens``). Owns two device slabs, each with a
    pinned mirror: ``capture_index_all`` (num_layers, cap) int64 destination rows,
    and ``any_active`` (num_layers,) int32 per-layer active markers.
    """

    def __init__(
        self,
        num_layers: int,
        cap: int,
        device: torch.device | str,
        should_capture: bool = True,
    ) -> None:
        self.num_layers = int(num_layers)
        self.cap = int(cap)
        self.device = torch.device(device)
        self.should_capture = bool(should_capture)

        # Sentinel destination row for pad / no-capture lanes. Default 0 ("row 0
        # discard": reset_pinned fills with 0, byte-identical to zero_()). The HS
        # capture-aperture path overrides this to the aperture's SENTINEL row (== R, the
        # extra (R+1)th row of each per-layer hs_buf), since aperture slot 0 is a REAL
        # storage row and a zero-filled pad lane would corrupt it. See
        # graph/install_hs._install_hs_buffer.
        self.sentinel_row = 0

        # int64 (index_copy_ needs Long rows); any_active int32. Zeroed, so
        # pre-first-upload every token routes to sentinel row 0 -- a safe no-op.
        self.capture_index_all = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, device=self.device
        )
        self.any_active = torch.zeros(
            self.num_layers, dtype=torch.int32, device=self.device
        )

        # Pinned-CPU mirrors for non-blocking uploads, aperture-buffered so reset_pinned
        # never waits on the in-flight copy; plain CPU off-CUDA so the registry
        # stays constructible for tests. capture_index_pinned / any_active_pinned
        # are properties onto the CURRENT aperture slot.
        pin = self.device.type == "cuda"
        self._aperture = PinnedMirror(
            [
                ("capture_index", (self.num_layers, self.cap), torch.int64),
                ("any_active", (self.num_layers,), torch.int32),
            ],
            pin=pin,
        )

        # layer_num -> QKCaptureHost (or HSCaptureHost — duck-typed: layer_num/cap/bind_views)
        self.hosts: Dict[int, QKCaptureHost] = {}

        # Plans built by the prepare_inputs routing wrapper this step, consumed by
        # the execute_model egress wrapper post-forward (the two run in one step).
        self._pending_plans: list = []
        # Per-active-request (start, end, layers_or_None) assignments the incremental
        # router diffs (set by _build_routing each step in incremental mode).
        self._pending_assignments: list = []

        # Forward-context stash: the attention wrap snapshots the live context on
        # its first fire each step (it's only valid inside the forward), so egress
        # can read it post-forward for prefix-K. See graph/install.py.
        self.fwd_ctx = None          # the live ForwardContext object
        self.fwd_attn_metadata = None  # ctx.attn_metadata snapshot
        self._ctx_step_token = 0     # bumped each step so the wrap re-snapshots

        # Incremental/persistent routing state (lazy; cuda + opt-in only).
        # ``capture_index_all[L, c]`` is c+1 iff layer L is active at column c, else 0 —
        # a POSITION-deterministic value (independent of WHICH request occupies c). So the
        # device slab is a pure function of (column -> active-layer-set), and a step only
        # needs to rewrite/upload the columns whose active-layer-set CHANGED since last
        # step. ``_col_layerset`` is that per-column shadow (interned set id; 0 = inactive);
        # ``_inc_mirror`` is a PERSISTENT pinned mirror (single, not aperture-depth — the
        # incremental upload is tiny so its wait is cheap); ``_inc_event`` guards the mirror
        # against the previous (in-flight) H2D. ``any_active`` is vestigial (the op does
        # ``del active``), so only ``capture_index_all`` is maintained here.
        self.incremental_enabled = (INCREMENTAL_ROUTING
                                    and self.device.type == "cuda"
                                    and self.should_capture)
        self._col_layerset: Optional[np.ndarray] = None   # (cap,) int64 shadow
        self._inc_mirror: Optional[torch.Tensor] = None   # (num_layers, cap) int64 pinned
        self._inc_event: Optional[torch.cuda.Event] = None
        self._inc_force_full = False                      # next apply rewrites [0,w) fully
        # Interned layer-set ids (id>=1; 0 reserved for "inactive"). "all layers" gets a
        # fixed id so the common homogeneous batch never builds/hashes a per-request tuple.
        self._ls_intern: Dict[tuple, int] = {}
        self._ls_layers: Dict[int, list] = {}
        self._ls_next = 1
        self._all_sid = self._intern_layerset(tuple(range(self.num_layers)))

        # GPU-side capture routing (MIA_CAPTURE_GPU_ROUTING): per-slot capture
        # layer-mask (which layers each request captures THIS step, from _build_routing's
        # gating) resident on GPU + a GPU scatter that writes capture_index_all[L,c]=c+1 for
        # active cells — replacing the O(num_layers x cap) numpy build. Simpler than steer (no
        # vid/coeff/mode, one slab). `gpu_routing` (the wrapper gate) requires cuda; arrays are
        # built whenever the env is set so CPU unit tests can drive it. Capture rebuilds every
        # step (routing_key=None), so the mask is refreshed per step (O(reqs) host).
        self._gpu_routing_env = os.environ.get("MIA_CAPTURE_GPU_ROUTING", "0") == "1"
        self.gpu_routing = self._gpu_routing_env and self.device.type == "cuda"
        self.slot_layer_mask = None
        if self._gpu_routing_env:
            pin = self.device.type == "cuda"
            self.slot_layer_mask = torch.zeros(cap, self.num_layers, dtype=torch.bool,
                                               device=self.device)
            self._slot_mask_h = torch.zeros(cap, self.num_layers, dtype=torch.bool,
                                            pin_memory=pin)

    # ------------------------------------------------------------------
    # Host registration + view assignment
    # ------------------------------------------------------------------

    def register_host(self, host: QKCaptureHost) -> None:
        """Record a host under its ``layer_num`` (slab row); idempotent per layer."""
        if not (0 <= host.layer_num < self.num_layers):
            raise ValueError(
                f"host layer_num {host.layer_num} out of range "
                f"[0, {self.num_layers}) for module {host.module_name!r}"
            )
        if host.cap != self.cap:
            raise ValueError(
                f"host cap {host.cap} != registry cap {self.cap} "
                f"for module {host.module_name!r}"
            )
        self.hosts[host.layer_num] = host

    def assign_views(self) -> None:
        """Bind each host to its slab row-views, so one ``upload()`` reaches all.

        Call once after all hosts register (the load_model patch does). The views
        alias slab storage, so the refresh is visible to every ``capture()`` on
        the next replay.
        """
        for layer_num, host in self.hosts.items():
            host.bind_views(
                self.capture_index_all[layer_num],   # (cap,) int64 view
                self.any_active[layer_num],           # 0-d int32 view
            )

    # ------------------------------------------------------------------
    # Per-step upload (fan-out by view) — called from the execute_model wrapper.
    # ------------------------------------------------------------------

    @property
    def capture_index_pinned(self) -> torch.Tensor:
        """The current aperture slot's (num_layers, cap) int64 routing mirror."""
        return self._aperture.cur("capture_index")

    @property
    def any_active_pinned(self) -> torch.Tensor:
        """The current aperture slot's (num_layers,) int32 active-marker mirror."""
        return self._aperture.cur("any_active")

    def routing_key(self, step) -> Optional[tuple]:
        """Invalidation key for the routing wrapper. ``None`` = never skip.

        Capture (QK/HS) returns ``None`` so the routing is rebuilt every step — the
        conservative, latency-neutral choice. With FLAT per-step routing the index is
        actually bit-identical on a stable decode batch (each request's column is
        fixed), so a real signature like SteerRegistry's would let this collapse decode
        to a pure replay; that is an opt-in perf follow-on, deliberately left off here.

        ``step`` (the StepView the wrapper just built) is accepted for a uniform
        call convention with ``routing_key_fn`` free functions (which DO need it,
        e.g. ``_capture_idle_key``); unused here.
        """
        return None

    def upload(self, width: Optional[int] = None) -> None:
        """Push the current pinned mirror slot to the device slabs: one copy per step.

        ``width`` (when given) copies only the ``[:, :width]`` column prefix of the
        capture_index slab — the prefix the cudagraph-padded forward actually reads
        (a 2D strided async copy, much smaller than the full ``cap`` width on decode).
        Columns beyond ``width`` are never read by the op, so the stale device tail is
        harmless. ``any_active`` is tiny (num_layers,) so it always copies in full.
        Records the slot's upload event and rotates the aperture.
        """
        ci = self._aperture.cur("capture_index")
        aa = self._aperture.cur("any_active")
        if width is None:
            self.capture_index_all.copy_(ci, non_blocking=True)
        else:
            w = max(1, min(int(width), self.cap))
            self.capture_index_all[:, :w].copy_(ci[:, :w], non_blocking=True)
        self.any_active.copy_(aa, non_blocking=True)
        self._aperture.record_advance()

    def reset_pinned(self, width: Optional[int] = None) -> None:
        """Zero the current pinned mirror slot to the sentinel no-op, before writes.

        Defaults any unrequested (layer, token) to row 0 + inactive. ``width`` zeros
        only the ``[:, :width]`` prefix (the part this step writes + uploads); the op
        never reads beyond ``width``, so the tail need not be re-zeroed. Waits on this
        slot's own (K-steps-stale, already-complete) upload event first — a guard that
        no longer stalls the hot path now that the aperture gives a fresh slot each step.

        Under ``MIA_PROFILE_FINE=1`` the wait and the fill are timed separately
        (``graph.route.reset.wait`` / ``.fill``) since the wait's cost is the prior
        in-flight GPU work it absorbs, not plugin work. Both are no-ops otherwise.
        """
        with PROF.timed("graph.route.reset.wait", tier=2):
            self._aperture.wait_current()
        with PROF.timed("graph.route.reset.fill", tier=2):
            ci = self._aperture.cur("capture_index")
            aa = self._aperture.cur("any_active")
            s = self.sentinel_row
            if width is None:
                ci.fill_(s)
            else:
                w = max(1, min(int(width), self.cap))
                ci[:, :w].fill_(s)
            aa.zero_()

    # ------------------------------------------------------------------
    # Incremental/persistent routing (per-column diff + upload).
    # ------------------------------------------------------------------

    def _intern_layerset(self, layers: tuple) -> int:
        """Map a sorted tuple of layer rows to a stable id (>=1); cache the rows."""
        sid = self._ls_intern.get(layers)
        if sid is None:
            sid = self._ls_next
            self._ls_next += 1
            self._ls_intern[layers] = sid
            self._ls_layers[sid] = list(layers)
        return sid

    def force_full_routing(self) -> None:
        """Force the next ``apply_incremental_routing`` to rewrite [0,width) in full.

        Used after a wrapper error (device slab in an unknown state) so recovery
        re-establishes the whole prefix rather than trusting the stale shadow.
        """
        self._inc_force_full = True

    def apply_incremental_routing(self, assignments: list, width: int) -> bool:
        """Write + upload ONLY the capture-index columns whose active-layer-set changed.

        ``assignments`` is one ``(start, end, layers_or_None)`` per ACTIVE request this
        step (``None`` = all layers; else a sorted tuple of layer rows). Because the slab
        value at column c is the position-deterministic c+1, the device content of a
        column is fully described by its active-layer-set id, so a column needs a rewrite
        iff that id changed vs ``_col_layerset``. Steady-state stable decode (and every
        idle step) changes 0 columns -> 0 upload, a pure replay. A +1-prefill / finish /
        condense changes only the affected columns -> O(changed) work. Returns True iff an
        upload was issued (diagnostic).

        Correctness under input-batch condensation: when a finished request's slot is
        back-filled, the moved request's column shifts, but the slab VALUE there is still
        c+1 — so if its layer-set matches the vacated one (the common all-layers case) the
        column legitimately needs no rewrite; egress correctness rides on the
        per-step-rebuilt plans, not the slab. A layer-set change at a column IS detected
        (different id) and rewritten.
        """
        cap = self.cap
        w = max(1, min(int(width), cap))
        if self._col_layerset is None:
            self._col_layerset = np.zeros(cap, dtype=np.int64)
            pin = self.device.type == "cuda"
            self._inc_mirror = torch.zeros(self.num_layers, cap, dtype=torch.int64,
                                           pin_memory=pin)
            self._inc_event = torch.cuda.Event() if pin else None
            # Device slab + shadow + mirror all start zeroed (constructor) -> consistent.

        # Desired per-column active-layer-set id over [0, cap) (only [0,w) is read/diffed;
        # active columns are always < real_n <= w). Disjoint flat ranges -> no overlap.
        new_col = np.zeros(cap, dtype=np.int64)
        for (start, end, layers) in assignments:
            if end <= start:
                continue
            sid = self._all_sid if layers is None else self._intern_layerset(layers)
            new_col[start:end] = sid

        old = self._col_layerset
        if self._inc_force_full:
            # Treat the whole prefix as dirty so the device is fully re-established.
            changed = np.arange(0, w, dtype=np.int64)
            self._inc_force_full = False
        else:
            changed = np.nonzero(new_col[:w] != old[:w])[0]
        if changed.size == 0:
            return False

        lo = int(changed[0])
        hi = int(changed[-1]) + 1
        if self._inc_event is not None:
            self._inc_event.synchronize()  # last upload of this mirror is done
        mir = self._inc_mirror
        cc = torch.from_numpy(changed)                      # int64 column indices
        mir[:, cc] = 0                                       # clear changed cols, all layers
        new_changed = new_col[changed]
        for sid in np.unique(new_changed):
            if sid == 0:
                continue
            cols = changed[new_changed == sid]
            layer_t = torch.tensor(self._ls_layers[int(sid)], dtype=torch.long)
            cols_t = torch.from_numpy(cols)
            mir[layer_t[:, None], cols_t] = (cols_t + 1)[None, :]

        self.capture_index_all[:, lo:hi].copy_(mir[:, lo:hi], non_blocking=True)
        if self._inc_event is not None:
            self._inc_event.record()
        old[:w] = new_col[:w]
        return True

    def _qsl_device(self, step):
        """This step's cumulative query_start_loc on the DEVICE, ``[:num_reqs+1]``.

        ``step.query_start_loc`` IS the tensor `prepare_inputs` returned (V2 keeps
        `InputBatch.query_start_loc` device-resident for attention to read directly),
        already sliced to `[:num_reqs+1]` by `step_view` -- so no resident fallback
        buffer or extra H2D copy is needed here now that routing reads the RETURNED
        batch directly instead of reconstructing qsl from the runner post hoc.
        """
        return step.query_start_loc

    def build_and_upload_gpu(self, step, width, build_routing_fn):
        """GPU-side capture routing: run ``build_routing_fn`` (the gating — output_qk filter,
        hooks_on, per-request layer set — which sets ``_pending_assignments`` + returns plans),
        build the per-slot capture layer-mask from those assignments (O(reqs)), then GPU-scatter
        ``capture_index_all[L,c]=c+1`` for active cells — replacing the O(num_layers x cap) numpy
        ``apply_incremental_routing``. Returns the plans (egress consumes them). Byte-identical
        device slab vs the host path (the slab value is the position-deterministic c+1)."""
        from mia.graph.steer_routing_gpu import scatter_capture_routing
        plans = build_routing_fn(step, self)
        bs = step.num_reqs
        qsl_np = step.query_start_loc_np
        self._slot_mask_h[:bs].zero_()
        for (start, end, layers) in self._pending_assignments:
            if end <= start:
                continue
            i = int(np.searchsorted(qsl_np, start))     # request position (start == qsl[i])
            if i >= bs:
                continue
            if layers is None:
                self._slot_mask_h[i, :] = True           # all layers
            else:
                self._slot_mask_h[i, list(layers)] = True
        self.slot_layer_mask[:bs].copy_(self._slot_mask_h[:bs], non_blocking=True)
        real_n = int(qsl_np[-1]) if qsl_np.size else 0
        qsl_dev = self._qsl_device(step)
        scatter_capture_routing(qsl_dev, self.slot_layer_mask, self.capture_index_all,
                                real_n, width)
        return plans

    # ------------------------------------------------------------------
    # Forward-context stash (set by the attention wrap, read by egress).
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Clear the stash and bump the step token so the wrap re-snapshots once.

        Called by the execute_model wrapper before the forward.
        """
        self.fwd_ctx = None
        self.fwd_attn_metadata = None
        self._ctx_step_token += 1

    def stash_forward_context(self, ctx, attn_metadata) -> None:
        """Snapshot the live forward context for post-forward egress reads.

        First call per step wins (wrap guards on the step token), capturing the
        first layer's batch-wide query_start_loc / block_table that egress needs.
        """
        self.fwd_ctx = ctx
        self.fwd_attn_metadata = attn_metadata

    # ------------------------------------------------------------------
    # Egress accessors (read static buffers after the forward returns).
    # ------------------------------------------------------------------

    def iter_hosts(self) -> Iterator[Tuple[int, QKCaptureHost]]:
        """Yield ``(layer_num, host)`` for every registered host, layer-ordered."""
        for layer_num in sorted(self.hosts):
            yield layer_num, self.hosts[layer_num]
