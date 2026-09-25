"""Fixed-size GPU capture aperture: the in-graph scatter writes rows here at an advancing cursor, so
nothing is overwritten within a drain window and NO per-step egress copy is needed. Logical cursors
are monotonic ints; physical slot = logical % n_slots. Reservations are whole rows (row-aligned), so
a row never straddles the wrap. One SENTINEL slot absorbs pad / no-capture lanes.

THREAD-SAFETY (off-loop consumer drain). The aperture is a single-producer / single-consumer cursor:
the ENGINE thread is the ONLY writer of ``_write`` (via ``reserve``); the DRAIN CONSUMER thread is
the ONLY writer of ``_drain`` (via ``advance_drain``). No lock is taken because each cursor has one
writer and a CPython attribute read is atomic under the GIL, so:
  * ``reserve``'s ``free_rows()`` may read a slightly STALE (smaller) ``_drain`` — that only
    UNDER-counts free rows, so a reserve is refused a touch early and retried; it can never
    over-admit into un-drained rows (conservative, safe).
  * ``advance_drain`` reads ``_write`` for its assert; the consumer only drains steps whose reserve
    already completed (STORE done before the step was enqueued), so ``_drain <= _write`` holds even
    against a concurrent in-flight ``reserve``.
This invariant is load-bearing for the off-loop drain and is validated by the GPU parity oracle."""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch


class ApertureBackpressureError(RuntimeError):
    """Raised when a step's rows cannot be admitted to the capture aperture within the block timeout.

    NEVER-DROP contract: a capturing request is never silently skipped. The engine BLOCKS on a full
    aperture (polling while the off-loop consumer drains and frees rows); this fires only if the aperture is
    mis-sized for the workload (a single step needs more than the whole aperture holds, unrelievable) or
    the off-loop consumer DIED (no one will ever free rows). It is loud and MUST PROPAGATE out of the
    routing wrapper — ``graph/install.py`` re-raises it rather than swallowing it into a silent drop.
    Defined here (not in install_hs) so ``install.py`` can import it without a circular import.
    """


class CaptureAperture:
    def __init__(self, row_bytes: int, n_slots: int, device="cpu", dtype=None, row_shape=None):
        assert n_slots >= 1
        self.row_bytes = int(row_bytes)
        # n_slots USABLE rows [0, n_slots); one EXTRA sentinel row at index n_slots absorbs pad /
        # no-capture lanes (never returned by reserve/physical_slots). Usable capacity == n_slots.
        self.n_slots = int(n_slots)
        self.SENTINEL = int(n_slots)
        self._write = 0        # monotonic logical write cursor (rows)
        self._drain = 0        # monotonic logical drain cursor (rows)
        self.device = device
        self.dtype = dtype
        self.row_shape = row_shape
        self.buf = None        # allocated by alloc_gpu() on the real device

    def alloc_gpu(self) -> None:
        """Allocate the (n_slots_total, *row_shape) GPU tensor. Called at install on the model device."""
        total = self.n_slots + 1  # + sentinel
        shape = (total, *self.row_shape) if self.row_shape else (total, self.row_bytes)
        self.buf = torch.empty(shape, dtype=self.dtype or torch.uint8, device=self.device)

    def free_rows(self) -> int:
        return self.n_slots - (self._write - self._drain)

    def pending_rows(self) -> int:
        return self._write - self._drain

    def reserve(self, n_rows: int) -> Optional[int]:
        if n_rows <= 0:
            return self._write
        if n_rows > self.free_rows():
            return None                     # aperture full -> caller backpressures
        start = self._write
        self._write += n_rows
        return start

    def advance_drain(self, n_rows: int) -> None:
        self._drain += int(n_rows)
        assert self._drain <= self._write

    def physical_slots(self, logical_start: int, n_rows: int) -> List[int]:
        return [(logical_start + j) % self.n_slots for j in range(n_rows)]

    def segments_at(self, logical_start: int, n_rows: int) -> List[Tuple[int, int]]:
        """Physical [start,end) segments covering the ``n_rows`` logical rows beginning at
        ``logical_start``; ≤2 on wrap. Used by the OFF-LOOP consumer to read ONE step's rows
        (``[start_logical, start_logical + n_rows)``) rather than the whole pending region — the
        engine may have reserved several later steps ahead, so ``drained_segments`` (which spans
        the entire ``[drain, write)``) would over-read."""
        if n_rows <= 0:
            return []
        ps = logical_start % self.n_slots
        if ps + n_rows <= self.n_slots:
            return [(ps, ps + n_rows)]
        return [(ps, self.n_slots), (0, n_rows - (self.n_slots - ps))]

    def drained_segments(self) -> List[Tuple[int, int]]:
        """Physical [start,end) segments covering the pending region [drain, write); ≤2 on wrap.
        (The SYNCHRONOUS per-step drain uses this — after a sync step the pending region IS exactly
        that step's rows.)"""
        return self.segments_at(self._drain, self._write - self._drain)
