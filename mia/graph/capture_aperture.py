"""Fixed-size GPU capture aperture written in-graph at an advancing cursor, drained off-loop."""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch


class ApertureBackpressureError(RuntimeError):
    """Raised when a step's rows cannot be admitted to the aperture within the block timeout."""


class CaptureAperture:
    def __init__(self, row_bytes: int, n_slots: int, device="cpu", dtype=None, row_shape=None):
        assert n_slots >= 1
        self.row_bytes = int(row_bytes)
        self.n_slots = int(n_slots)
        self.SENTINEL = int(n_slots)
        self._write = 0
        self._drain = 0
        self.device = device
        self.dtype = dtype
        self.row_shape = row_shape
        self.buf = None

    def alloc_gpu(self) -> None:
        """Allocate the (n_slots_total, *row_shape) GPU tensor."""
        total = self.n_slots + 1
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
            return None
        start = self._write
        self._write += n_rows
        return start

    def advance_drain(self, n_rows: int) -> None:
        self._drain += int(n_rows)
        assert self._drain <= self._write

    def physical_slots(self, logical_start: int, n_rows: int) -> List[int]:
        return [(logical_start + j) % self.n_slots for j in range(n_rows)]

    def segments_at(self, logical_start: int, n_rows: int) -> List[Tuple[int, int]]:
        """Physical [start, end) segments for ``n_rows`` rows from ``logical_start``; at most 2."""
        if n_rows <= 0:
            return []
        ps = logical_start % self.n_slots
        if ps + n_rows <= self.n_slots:
            return [(ps, ps + n_rows)]
        return [(ps, self.n_slots), (0, n_rows - (self.n_slots - ps))]

    def drained_segments(self) -> List[Tuple[int, int]]:
        """Physical [start,end) segments covering the pending region [drain, write); ≤2 on wrap."""
        return self.segments_at(self._drain, self._write - self._drain)

