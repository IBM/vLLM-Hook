"""Per-worker device routing slabs and host registry for CUDA-graph capture."""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from mia._profiler import PROF
from mia.core.hooks.hosts import QKCaptureHost

INCREMENTAL_ROUTING = os.environ.get("MIA_INCREMENTAL_ROUTING", "1") != "0"

APERTURE_DEPTH = max(1, int(os.environ.get("MIA_PINNED_MIRROR", "2")))


_REGISTRIES_ATTR = "_mia_registries"


def set_registry(worker, subsystem: str, registry) -> None:
    """Register one subsystem's registry on the worker ("hs", "qk" or "steer")."""
    registries = getattr(worker, _REGISTRIES_ATTR, None)
    if registries is None:
        registries = {}
        setattr(worker, _REGISTRIES_ATTR, registries)
    registries[subsystem] = registry


def get_registry(worker, subsystem: str):
    """The registry for ``subsystem``, or None if it never installed."""
    return getattr(worker, _REGISTRIES_ATTR, {}).get(subsystem)


class PinnedMirror:
    """Depth-K set of pinned-host mirror slots with per-slot CUDA upload events."""

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
        self.events: List[Optional[torch.cuda.Event]] = [
            torch.cuda.Event() if pin else None for _ in range(self.depth)
        ]

    def cur(self, name: str) -> torch.Tensor:
        """The current slot's tensor for ``name`` (what build code writes/upload reads)."""
        return self.slots[self.idx][name]

    def wait_current(self) -> None:
        """Block until the current slot's last upload (K steps ago) is done."""
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
    """Per-worker owner of the QK routing slabs and host directory."""

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

        self.sentinel_row = 0

        self.capture_index_all = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, device=self.device
        )
        self.any_active = torch.zeros(
            self.num_layers, dtype=torch.int32, device=self.device
        )

        pin = self.device.type == "cuda"
        self._aperture = PinnedMirror(
            [
                ("capture_index", (self.num_layers, self.cap), torch.int64),
                ("any_active", (self.num_layers,), torch.int32),
            ],
            pin=pin,
        )

        self.hosts: Dict[int, QKCaptureHost] = {}

        self._pending_plans: list = []
        self._pending_assignments: list = []

        self.fwd_ctx = None
        self.fwd_attn_metadata = None
        self._ctx_step_token = 0

        self.incremental_enabled = (INCREMENTAL_ROUTING
                                    and self.device.type == "cuda"
                                    and self.should_capture)
        self._col_layerset: Optional[np.ndarray] = None
        self._inc_mirror: Optional[torch.Tensor] = None
        self._inc_event: Optional[torch.cuda.Event] = None
        self._inc_force_full = False
        self._ls_intern: Dict[tuple, int] = {}
        self._ls_layers: Dict[int, list] = {}
        self._ls_next = 1
        self._all_sid = self._intern_layerset(tuple(range(self.num_layers)))

        self._gpu_routing_env = os.environ.get("MIA_CAPTURE_GPU_ROUTING", "0") == "1"
        self.gpu_routing = self._gpu_routing_env and self.device.type == "cuda"
        self.slot_layer_mask = None
        if self._gpu_routing_env:
            pin = self.device.type == "cuda"
            self.slot_layer_mask = torch.zeros(cap, self.num_layers, dtype=torch.bool,
                                               device=self.device)
            self._slot_mask_h = torch.zeros(cap, self.num_layers, dtype=torch.bool,
                                            pin_memory=pin)


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
        """Bind each host to its slab row-views, so one ``upload()`` reaches all."""
        for layer_num, host in self.hosts.items():
            host.bind_views(
                self.capture_index_all[layer_num],
                self.any_active[layer_num],
            )


    @property
    def capture_index_pinned(self) -> torch.Tensor:
        """The current aperture slot's (num_layers, cap) int64 routing mirror."""
        return self._aperture.cur("capture_index")

    @property
    def any_active_pinned(self) -> torch.Tensor:
        """The current aperture slot's (num_layers,) int32 active-marker mirror."""
        return self._aperture.cur("any_active")

    def routing_key(self, step) -> Optional[tuple]:
        """Invalidation key for the routing wrapper."""
        return None

    def upload(self, width: Optional[int] = None) -> None:
        """Push the current pinned mirror slot to the device slabs: one copy per step."""
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
        """Zero the current pinned mirror slot to the sentinel no-op, before writes."""
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


    def _intern_layerset(self, layers: tuple) -> int:
        sid = self._ls_intern.get(layers)
        if sid is None:
            sid = self._ls_next
            self._ls_next += 1
            self._ls_intern[layers] = sid
            self._ls_layers[sid] = list(layers)
        return sid

    def force_full_routing(self) -> None:
        """Force the next ``apply_incremental_routing`` to rewrite [0,width) in full."""
        self._inc_force_full = True

    def apply_incremental_routing(self, assignments: list, width: int) -> bool:
        """Write and upload only the capture-index columns whose active-layer set changed."""
        cap = self.cap
        w = max(1, min(int(width), cap))
        if self._col_layerset is None:
            self._col_layerset = np.zeros(cap, dtype=np.int64)
            pin = self.device.type == "cuda"
            self._inc_mirror = torch.zeros(self.num_layers, cap, dtype=torch.int64,
                                           pin_memory=pin)
            self._inc_event = torch.cuda.Event() if pin else None

        new_col = np.zeros(cap, dtype=np.int64)
        for (start, end, layers) in assignments:
            if end <= start:
                continue
            sid = self._all_sid if layers is None else self._intern_layerset(layers)
            new_col[start:end] = sid

        old = self._col_layerset
        if self._inc_force_full:
            changed = np.arange(0, w, dtype=np.int64)
            self._inc_force_full = False
        else:
            changed = np.nonzero(new_col[:w] != old[:w])[0]
        if changed.size == 0:
            return False

        lo = int(changed[0])
        hi = int(changed[-1]) + 1
        if self._inc_event is not None:
            self._inc_event.synchronize()
        mir = self._inc_mirror
        cc = torch.from_numpy(changed)
        mir[:, cc] = 0
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
        return step.query_start_loc

    def build_and_upload_gpu(self, step, width, build_routing_fn):
        """GPU-side capture routing: build per-slot layer masks, then GPU-scatter the capture index."""
        # lazy: Triton kernels stay out of plugin load
        from mia.core.hooks.steer_routing_gpu import scatter_capture_routing
        plans = build_routing_fn(step, self)
        bs = step.num_reqs
        qsl_np = step.query_start_loc_np
        self._slot_mask_h[:bs].zero_()
        for (start, end, layers) in self._pending_assignments:
            if end <= start:
                continue
            i = int(np.searchsorted(qsl_np, start))
            if i >= bs:
                continue
            if layers is None:
                self._slot_mask_h[i, :] = True
            else:
                self._slot_mask_h[i, list(layers)] = True
        self.slot_layer_mask[:bs].copy_(self._slot_mask_h[:bs], non_blocking=True)
        real_n = int(qsl_np[-1]) if qsl_np.size else 0
        qsl_dev = self._qsl_device(step)
        scatter_capture_routing(qsl_dev, self.slot_layer_mask, self.capture_index_all,
                                real_n, width)
        return plans


    def begin_step(self) -> None:
        """Clear the stash and bump the step token so the wrap re-snapshots once."""
        self.fwd_ctx = None
        self.fwd_attn_metadata = None
        self._ctx_step_token += 1

    def stash_forward_context(self, ctx, attn_metadata) -> None:
        """Snapshot the live forward context for post-forward egress reads."""
        self.fwd_ctx = ctx
        self.fwd_attn_metadata = attn_metadata


    def iter_hosts(self) -> Iterator[Tuple[int, QKCaptureHost]]:
        """Yield ``(layer_num, host)`` for every registered host, layer-ordered."""
        for layer_num in sorted(self.hosts):
            yield layer_num, self.hosts[layer_num]

