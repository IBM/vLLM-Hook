"""Hybrid delivery gather: scatters the shared layer files into per-request artifacts."""
from __future__ import annotations

import glob
import json
import logging
import multiprocessing
import os
import queue as _queue
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np
import torch

from .aperture_run_index import (RUN_ID_KEY, ChainCursor, RunIndexError, read_run_index,
                                 segment_name, segment_paths)
from .aperture_trim import (TRIM_ENV as GATHER_TRIM_ENV, TrimError, TrimLog, align_down,
                            floor_row_for_bytes, is_reclaimed, published_cursors, punch_hole,
                            punch_supported, trim_align, trim_chunk_bytes, trim_enabled,
                            trim_explicit, trim_lag_bytes, trimmed_floor_rows)
from mia.core.runtime.cpu_budget import allocated_cpus
from mia.core.runtime.tp_shard import (
    RANK_DIR_PREFIX,
    HSShard,
    TPShardError,
    check_hs_shard_set,
    dp_dir_name,
    hs_expected_ranks,
    hs_owned_rows,
    hs_requested_layers,
    hs_shard_from_header,
    merge_hs_layer_maps,
    parse_dp_dir,
    parse_rank_dir,
    rank_dir_name,
)
from mia.errors import MiaRefusal

logger = logging.getLogger(__name__)

DELIVER_ENV = "MIA_APERTURE_GATHER_DELIVER"
WORKERS_ENV = "MIA_APERTURE_GATHER_WORKERS"
DIR_ENV = "MIA_APERTURE_GATHER_DIR"
BATCH_BYTES_ENV = "MIA_APERTURE_GATHER_BATCH_BYTES"
POLL_MS_ENV = "MIA_APERTURE_GATHER_POLL_MS"

DELIVERY_DIRNAME = "delivered"
MANIFEST_FMT = "delivery.w%02dof%02d.json"
MANIFEST_GLOB = "delivery.w*of*.json"
DELIVERY_FORMAT = "hs-layer-runs-v1"

RANK_MARKER_NAME = "delivery.rank.json"
RANK_MARKER_FORMAT = "hs-delivery-rank-v1"

DEFAULT_BATCH_BYTES = 256 << 20

_TRIM_REPORT_S = 0.5

DEFAULT_POLL_MS = 100


class GatherError(RuntimeError):
    """The gather cannot deliver what it was asked for, and will not deliver part of it."""


DELIVERY_TIMEOUT_ENV = "MIA_DELIVERY_TIMEOUT_S"
DEFAULT_DELIVERY_TIMEOUT_S = 60.0
DELIVERY_HARD_CAP_S = 1800.0
_WAIT_POLL_S = 0.1


class TrimRefusedError(GatherError, MiaRefusal):
    """The trim was asked for explicitly on a capture dir that cannot punch holes."""


class NoDeliveryError(GatherError):
    """No delivery at a path yet: no root, no delivery dir, or a rank's marker missing."""

    def __init__(self, msg: str, what: str = ""):
        super().__init__(msg)
        self.what = what or msg


class DeliveryTimeoutError(GatherError):
    """:func:`wait_delivered` gave up; ``missing`` maps each id to what it lacks."""

    def __init__(self, msg: str, missing: Optional[Dict[str, str]] = None,
                 roots: Optional[List[str]] = None):
        super().__init__(msg)
        self.missing = dict(missing or {})
        self.roots = list(roots or [])


def delivery_enabled() -> bool:
    """``MIA_APERTURE_GATHER_DELIVER``: default off; ``"1"`` is the only on-spelling."""
    return os.environ.get(DELIVER_ENV, "0") == "1"


@dataclass(frozen=True)
class GatherWorkers:
    """How many gather workers this rank runs, and where the number came from."""
    n: int
    requested: int
    cap: int
    capped: bool
    source: str
    note: str


DEFAULT_WORKERS = 2


def resolve_gather_workers(tp_size: int = 1) -> GatherWorkers:
    """Resolve this rank's gather worker count once, capped by its CPU share."""
    tp = max(1, int(tp_size))
    cap = max(1, (allocated_cpus() - 1) // tp)
    raw = os.environ.get(WORKERS_ENV)
    if raw is None or raw.strip() == "":
        requested, source = DEFAULT_WORKERS, "default"
    else:
        source = "explicit"
        try:
            requested = int(raw.strip())
        except ValueError:
            raise GatherError(
                f"{WORKERS_ENV}={raw!r} is not a whole number of gather workers. Unset it for the "
                f"default of {DEFAULT_WORKERS}."
            ) from None
        if requested < 1:
            raise GatherError(
                f"{WORKERS_ENV}={raw!r} must be at least 1. To run no gather at all, unset "
                f"{DELIVER_ENV} -- a zero-worker gather would arm the flag and deliver nothing.")
    n = min(requested, cap)
    bits = [f"{n} gather worker(s) per rank ({source}"
            + (f", default {DEFAULT_WORKERS}" if source == "default" else f", asked {requested}")
            + ")"]
    bits.append(f"cap {cap} = (allocated_cpus {allocated_cpus()} - 1) // tp_size {tp}")
    if n < requested:
        bits.append(
            f"CAPPED from {requested}: this rank's share of the allocation is {cap}. The gather has "
            f"no backpressure to the engine, so an under-provisioned one shows up as delivery "
            f"LATENESS and retention, never as request latency -- give the job more slots if the "
            f"delivery falls behind")
    if n == 1:
        bits.append("W=1 is untested at production payloads; the default is 2")
    return GatherWorkers(n=n, requested=requested, cap=cap, capped=n < requested, source=source,
                         note="; ".join(bits))


def gather_workers(tp_size: int = 1) -> int:
    """:func:`resolve_gather_workers`'s count."""
    return resolve_gather_workers(tp_size).n


def gather_batch_bytes() -> int:
    """``MIA_APERTURE_GATHER_BATCH_BYTES``: the resident cap per chunk."""
    raw = os.environ.get(BATCH_BYTES_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_BATCH_BYTES
    try:
        n = int(raw.strip())
    except ValueError:
        raise GatherError(
            f"{BATCH_BYTES_ENV}={raw!r} is not a whole number of bytes") from None
    if n < 1:
        raise GatherError(f"{BATCH_BYTES_ENV}={raw!r} must be positive")
    return n


def gather_poll_s() -> float:
    raw = os.environ.get(POLL_MS_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_POLL_MS / 1000.0
    try:
        ms = int(raw.strip())
    except ValueError:
        raise GatherError(f"{POLL_MS_ENV}={raw!r} is not a whole number of milliseconds") from None
    if ms < 1:
        raise GatherError(f"{POLL_MS_ENV}={raw!r} must be positive")
    return ms / 1000.0


def delivery_timeout_s() -> float:
    """``MIA_DELIVERY_TIMEOUT_S`` (default 60): :func:`wait_delivered`'s no-progress bound."""
    raw = os.environ.get(DELIVERY_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_DELIVERY_TIMEOUT_S
    try:
        v = float(raw)
    except ValueError:
        v = float("nan")
    if not v > 0:
        raise GatherError(f"{DELIVERY_TIMEOUT_ENV}={raw!r} must be a positive number of seconds")
    return v


def delivery_dir(run_dir: str, *, tp_rank: Optional[int] = None,
                 dp_rank: Optional[int] = None) -> str:
    """Delivery root: ``MIA_APERTURE_GATHER_DIR`` or ``<run_dir>/delivered``, per rank if sharded."""
    d = os.environ.get(DIR_ENV)
    if d and dp_rank is not None:
        d = os.path.join(d, dp_dir_name(int(dp_rank)))
    if tp_rank is None:
        return d if d else os.path.join(run_dir, DELIVERY_DIRNAME)
    if d:
        root = d
    else:
        named = parse_rank_dir(run_dir)
        if named is None:
            raise GatherError(
                f"{run_dir} is the run dir of a layer-SHARDED HS capture (tp_rank "
                f"{int(tp_rank)}), but it is not a {RANK_DIR_PREFIX}<N> directory, so there is no "
                f"run-wide parent to put the delivery under. Set {DIR_ENV} to a delivery root, or "
                f"let graph/install_hs.py build the run dir (it always names it for the rank).")
        if int(named) != int(tp_rank):
            raise GatherError(
                f"{run_dir} is named for tp_rank {int(named)} but this gather is rank "
                f"{int(tp_rank)}'s -- refusing to put one rank's delivery under another's run dir.")
        root = os.path.join(os.path.dirname(os.path.normpath(run_dir)), DELIVERY_DIRNAME)
    return os.path.join(root, rank_dir_name(int(tp_rank)))


def req_dir_name(req_id: str) -> str:
    """One request's directory name, percent-encoded so it is injective and stays in the root."""
    return quote(str(req_id), safe="")


def layers_for_worker(layers: Sequence[int], worker: int, n_workers: int) -> List[int]:
    """The layers worker ``worker`` owns, by install-position round-robin."""
    return [int(L) for i, L in enumerate(layers) if i % int(n_workers) == int(worker)]


def manifest_name(worker: int, n_workers: int) -> str:
    return MANIFEST_FMT % (int(worker), int(n_workers))


def clip_run(run: Tuple[int, int, int], lo: int, hi: int) -> Optional[Tuple[int, int, int]]:
    """``(base, stride, n)`` restricted to rows in ``[lo, hi)``, or None if it has none."""
    base, stride, n = int(run[0]), int(run[1]), int(run[2])
    st = stride if stride > 0 else 1
    if n <= 0:
        return None
    last = base + st * (n - 1)
    if last < lo or base >= hi:
        return None
    i0 = 0 if base >= lo else -((base - lo) // st)
    i1 = (n - 1) if last < hi else (hi - 1 - base) // st
    if i1 < i0:
        return None
    return (base + i0 * st, st, i1 - i0 + 1)


def _row_bytes(header: dict) -> int:
    shape = [int(d) for d in header["row_shape"]]
    width = 1
    for d in shape:
        width *= d
    return width * int(np.dtype(_np_name(header["dtype"])).itemsize)


def _np_name(dtype_name: str) -> str:
    return "uint16" if dtype_name == "bfloat16" else dtype_name


def _no_punch_message(run_dir: str, why: str) -> str:
    return (f"{run_dir} cannot free space while data is delivered ({why}), so captured data would "
            f"be kept twice; put the capture dir (MIA_APERTURE_DIR) on a local disk, or set "
            f"{GATHER_TRIM_ENV}=0 to accept that.")


def _no_punch_warning(run_dir: str, why: str) -> str:
    return (f"WARNING: {run_dir} cannot free space while data is delivered ({why}), so captured "
            f"data is kept twice there; put the capture dir (MIA_APERTURE_DIR) on a local disk to "
            f"avoid it.")


class _ReqState:
    __slots__ = ("layers", "mine", "written", "written_all", "touched")

    def __init__(self, layers: Tuple[int, ...], mine: List[int]):
        self.layers = layers
        self.mine = mine
        self.written = 0
        self.written_all = 0
        self.touched = False


class GatherPass:
    """One worker's streaming pass over the index's segment chain."""

    def __init__(self, run_dir: str, out_dir: Optional[str] = None, *, worker: int = 0,
                 n_workers: int = 1, batch_bytes: Optional[int] = None,
                 trim: Optional[bool] = None, trim_chunk: Optional[int] = None,
                 trim_lag: Optional[int] = None, tp_rank: Optional[int] = None,
                 tp_size: int = 1, num_layers: Optional[int] = None,
                 run_id: Optional[str] = None):
        if not (0 <= int(worker) < int(n_workers)):
            raise GatherError(f"worker {worker} is not in range(0, {n_workers})")
        self.run_dir = str(run_dir)
        self.tp_rank = None if tp_rank is None else int(tp_rank)
        self.tp_size = int(tp_size)
        self.num_layers = None if num_layers is None else int(num_layers)
        self.shard = None
        if self.tp_rank is not None:
            if self.num_layers is None:
                raise GatherError(
                    "a layer-sharded gather needs num_layers: the shard rule (and so which layers "
                    "this rank owns, and which ranks a reader must find) is a function of it.")
            self.shard = HSShard.of(self.tp_rank, self.tp_size, self.num_layers)
        self.out_dir = str(out_dir) if out_dir is not None else delivery_dir(
            self.run_dir, tp_rank=self.tp_rank)
        if self.tp_rank is not None:
            named = parse_rank_dir(self.out_dir)
            if named is None or int(named) != self.tp_rank:
                raise GatherError(
                    f"{self.out_dir} is not tp_rank {self.tp_rank}'s delivery root (its last path "
                    f"component must be tp_rank_{self.tp_rank}). Under the HS layer shard the rank "
                    f"is part of the PATH: without it every rank writes the same manifest name "
                    f"into the same request directory and a reader hands back whichever wrote "
                    f"last as the whole artifact.")
        self.worker = int(worker)
        self.n_workers = int(n_workers)
        self.batch_bytes = int(batch_bytes) if batch_bytes is not None else gather_batch_bytes()
        self.final_seen = False
        self.on_batch: Optional[Callable[[], None]] = None
        self.run_id = None if run_id is None else str(run_id)
        self._cursor = ChainCursor(expect_run_id=self.run_id)
        self._reqs: Dict[str, _ReqState] = {}
        self._header: Optional[dict] = None
        self._row_bytes = 0
        self._fds: Dict[int, int] = {}
        self.trim = trim_enabled() if trim is None else bool(trim)
        self.trim_chunk = int(trim_chunk) if trim_chunk is not None else trim_chunk_bytes()
        self.trim_lag = int(trim_lag) if trim_lag is not None else trim_lag_bytes()
        self._trim_align = 0
        self._trimmed: Dict[int, int] = {}
        self._trimlog: Optional[TrimLog] = None
        self._trim_layers: Optional[List[int]] = None
        self._trim_reported = -1.0
        self._floors: Dict[int, int] = {}
        self._stats = {"segments": 0, "windows": 0, "batches": 0, "chunks": 0, "rows_read": 0,
                       "rows_delivered": 0, "bytes_read": 0, "bytes_delivered": 0,
                       "finalized": 0, "skipped_stamps": 0, "trim_punches": 0, "trim_bytes": 0,
                       "trim_rounds": 0}
        if self.shard is not None:
            self._publish_rank_marker()

    def _publish_rank_marker(self) -> None:
        assert self.shard is not None
        os.makedirs(self.out_dir, exist_ok=True)
        man = dict(self.shard.as_header())
        man.update({"format": RANK_MARKER_FORMAT, "delivery_format": DELIVERY_FORMAT,
                    "n_workers": self.n_workers})
        path = os.path.join(self.out_dir, RANK_MARKER_NAME)
        tmp = f"{path}.{os.getpid()}.{self.worker}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(man, sort_keys=True))
        os.replace(tmp, path)

    def _check_shard_header(self, hdr: dict) -> None:
        try:
            got = hs_shard_from_header(hdr)
        except TPShardError as e:
            raise GatherError(f"{self.run_dir}: the index header's HS layer shard is unusable: "
                              f"{e}") from None
        if got is None and self.shard is None:
            return
        if got is None:
            raise GatherError(
                f"{self.run_dir}: this gather was armed for tp_rank {self.tp_rank} of "
                f"{self.tp_size} under the HS layer shard, but the index header declares NO shard "
                f"-- so that chain holds every layer it captured and is not a shard at all. "
                f"Refusing rather than delivering it as one rank's quarter.")
        if self.shard is None:
            raise GatherError(
                f"{self.run_dir}: the index header declares an HS layer shard (tp_rank "
                f"{got.tp_rank} of {got.tp_size}, layers {list(got.owned_layers)[:6]}...) but this "
                f"gather was armed without one, so its delivery would be a PART of every request "
                f"with nothing recording that. Refusing.")
        if got != self.shard:
            raise GatherError(
                f"{self.run_dir}: the index header is tp_rank {got.tp_rank} of {got.tp_size} "
                f"({got.num_layers} layers) but this gather is tp_rank {self.shard.tp_rank} of "
                f"{self.shard.tp_size} ({self.shard.num_layers} layers) -- it is reading another "
                f"rank's chain. Refusing.")

    def stats(self) -> dict:
        d = dict(self._stats)
        d.update({"worker": self.worker, "n_workers": self.n_workers,
                  "tp_rank": self.tp_rank, "tp_size": self.tp_size,
                  "final_seen": self.final_seen, "at_row": self._cursor.lo,
                  "live": len(self._reqs), "trim": self.trim,
                  "trim_floor_rows": (floor_row_for_bytes(min(self._trimmed.values()),
                                                          self._row_bytes)
                                      if self._trimmed and self._row_bytes else 0)})
        return d

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def poll_once(self) -> int:
        """Consume every segment available now, in order; return how many were consumed."""
        n = 0
        while not self.final_seen:
            batch = self._take_batch()
            if not batch:
                break
            self._consume_batch(batch)
            n += len(batch)
        return n

    def _take_batch(self) -> List[tuple]:
        out: List[tuple] = []
        rows = 0
        while True:
            path = os.path.join(self.run_dir, segment_name(self._cursor.seq + len(out)))
            if not os.path.exists(path):
                if not out:
                    self._refuse_a_hole()
                break
            hdr, reqs, stamps = read_run_index(path, with_stamps=True)
            out.append((path, hdr, reqs, stamps))
            if self._row_bytes == 0:
                self._row_bytes = _row_bytes(hdr)
            rows += int(hdr["up_to"]) - int(hdr["since"])
            if bool(hdr.get("final")) or rows >= max(1, self.batch_bytes // self._row_bytes):
                break
        return out

    def _refuse_a_hole(self) -> None:
        want = self._cursor.seq
        later = [p for p in segment_paths(self.run_dir)
                 if int(os.path.basename(p).split(".")[2]) > want]
        if later:
            raise RunIndexError(
                f"{self.run_dir}: index segment {want} is MISSING while {os.path.basename(later[0])} "
                f"exists -- the chain is written strictly in seq order, so this is a gap, and the "
                f"rows of segment {want} are named by no segment at all. Refusing to skip them.")

    def _consume_batch(self, batch: List[tuple]) -> None:
        ready: List[Tuple[str, dict]] = []
        merged: Dict[str, Tuple[Tuple[int, ...], list]] = {}
        lo = hi = self._cursor.lo
        final = False
        for path, hdr, reqs, stamps in batch:
            ready.extend(self._cursor.feed(path, hdr, reqs, stamps))
            if self._header is None:
                self._check_shard_header(hdr)
                self._header = dict(hdr)
                self._row_bytes = _row_bytes(hdr)
            hi = int(hdr["up_to"])
            final = final or bool(hdr.get("final"))
            for rid, r in reqs.items():
                layers = tuple(int(x) for x in r.layers)
                if rid not in self._reqs:
                    self._reqs[rid] = _ReqState(
                        layers, layers_for_worker(list(layers), self.worker, self.n_workers))
                prev = merged.get(rid)
                merged[rid] = (layers, (list(prev[1]) if prev else []) + list(r.runs))
            self._stats["segments"] += 1
            self._stats["windows"] += 1
        self._stats["batches"] += 1
        self._floors = trimmed_floor_rows(self.run_dir)
        if hi > lo:
            self._scatter_window(lo, hi, merged)
        # Manifests only after the window's rows are written: a reader trusts a manifest.
        for rid, stamp in ready:
            self._finalize(rid, int(stamp["n_total"]))
        if final:
            self.final_seen = True
            self.close()
        self._maybe_trim()
        if self.on_batch is not None:
            self.on_batch()

    def _scatter_window(self, lo: int, hi: int,
                        merged: Dict[str, Tuple[Tuple[int, ...], list]]) -> None:
        plan: Dict[int, List[Tuple[str, list]]] = {}
        for rid, (_layers, runs) in merged.items():
            st = self._reqs[rid]
            if not st.mine:
                continue
            self._touch(rid, st)
            if not runs:
                continue
            for L in st.mine:
                plan.setdefault(L, []).append((rid, list(runs)))
        if not plan:
            return
        chunk_rows = max(1, self.batch_bytes // max(1, self._row_bytes))
        for c0 in range(lo, hi, chunk_rows):
            c1 = min(c0 + chunk_rows, hi)
            self._stats["chunks"] += 1
            for L, owners in plan.items():
                clipped = [(rid, [c for c in (clip_run(run, c0, c1) for run in runs)
                                  if c is not None]) for rid, runs in owners]
                clipped = [(rid, cs) for rid, cs in clipped if cs]
                if not clipped:
                    continue
                buf = self._read_rows(L, c0, c1 - c0)
                for rid, cs in clipped:
                    parts = [buf[b - c0: b - c0 + st * n: st] for b, st, n in cs]
                    block = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
                    self._append(rid, L, block)

    def _resolve_trim(self) -> bool:
        if self._trim_align:
            return True
        ok, why = punch_supported(self.run_dir)
        if not ok:
            raise GatherError(_no_punch_message(self.run_dir, why))
        self._trim_align = trim_align(self.run_dir)
        self._trimlog = TrimLog(self.run_dir, self.worker, self.n_workers)
        return True

    def _trim_layer_files(self) -> List[int]:
        if self._trim_layers is None:
            found = []
            for path in glob.glob(os.path.join(self.run_dir, "hs_layer_*.raw")):
                m = re.search(r"hs_layer_(\d+)\.raw$", os.path.basename(path))
                if m:
                    found.append(int(m.group(1)))
            if not found:
                return []
            self._trim_layers = sorted(L for L in found
                                       if L % int(self.n_workers) == int(self.worker))
        return self._trim_layers

    def _maybe_trim(self) -> None:
        if not self.trim or self._row_bytes <= 0:
            return
        self._resolve_trim()
        now = time.monotonic()
        if self.final_seen or self._trim_reported < 0 or (now - self._trim_reported) >= _TRIM_REPORT_S:
            self._trimlog.publish({}, row_bytes=self._row_bytes, at_row=int(self._cursor.lo))
            self._trim_reported = now
        layers = self._trim_layer_files()
        if not layers:
            return
        cursors = published_cursors(self.run_dir, self.n_workers)
        cursors[int(self.worker)] = int(self._cursor.lo)
        if len(cursors) < int(self.n_workers):
            return
        # Bound by the slowest worker: workers read the same layer files for other requests.
        slowest = min(cursors.values())
        lag, chunk = self.trim_lag, self.trim_chunk
        safe = int(slowest) * int(self._row_bytes) - int(lag)
        if safe <= 0:
            return
        target = align_down(safe, self._trim_align)
        due = [L for L in layers if target - self._trimmed.get(L, 0) >= chunk]
        if not due:
            return
        floor_row = floor_row_for_bytes(target, self._row_bytes)
        self._trimlog.publish({L: floor_row for L in due}, row_bytes=self._row_bytes,
                              at_row=int(self._cursor.lo))
        self._stats["trim_rounds"] += 1
        for L in due:
            frm = self._trimmed.get(L, 0)
            path = os.path.join(self.run_dir, f"hs_layer_{L}.raw")
            try:
                fd = os.open(path, os.O_WRONLY)
            except OSError as e:
                raise GatherError(
                    f"{path}: {GATHER_TRIM_ENV} is ON but this pass cannot open the layer file for "
                    f"writing to free it ({e}). Refusing rather than growing an unbounded second "
                    f"copy silently; set {GATHER_TRIM_ENV}=0 to run without the trim.") from None
            try:
                punch_hole(fd, frm, target - frm)
            except (OSError, TrimError) as e:
                raise GatherError(
                    f"{path}: fallocate(PUNCH_HOLE) of [{frm}, {target}) failed ({e}). The marker "
                    f"already names these rows as reclaimed, which is the SAFE direction -- a "
                    f"reader refuses rows that are in fact still there -- but the space was not "
                    f"freed, so this is a failure, not a warning.") from None
            finally:
                os.close(fd)
            self._trimmed[L] = target
            self._stats["trim_punches"] += 1
            self._stats["trim_bytes"] += target - frm

    def _read_rows(self, layer: int, start: int, n: int) -> np.ndarray:
        fd = self._fds.get(layer)
        if fd is None:
            p = os.path.join(self.run_dir, f"hs_layer_{layer}.raw")
            try:
                fd = os.open(p, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            except FileNotFoundError:
                raise GatherError(
                    f"{p}: the index names layer {layer} but the capture wrote no such layer file. "
                    f"The gather reads the SHARED layer files; a request cannot name a layer that "
                    f"was never installed.") from None
            self._fds[layer] = fd
        if self._floors and is_reclaimed(layer, int(start), self._floors):
            raise GatherError(
                f"hs_layer_{layer}.raw: rows [{start}, {start + n}) are below the TRIM FLOOR "
                f"{self._floors.get(int(layer))} -- those blocks were already freed with "
                f"fallocate(PUNCH_HOLE) ({GATHER_TRIM_ENV} is ON by default) and the region reads "
                f"back as ZEROS, not short, so delivering it would write a plausible, wrong "
                f"artifact. A gather cannot be re-run over a chain it has already trimmed; re-run "
                f"the capture, or set {GATHER_TRIM_ENV}=0 to keep the shared files whole.")
        want = int(n) * self._row_bytes
        got = os.pread(fd, want, int(start) * self._row_bytes)
        if len(got) != want:
            raise GatherError(
                f"hs_layer_{layer}.raw is SHORTER than the index says: rows "
                f"[{start}, {start + n}) are {want} B but only {len(got)} B could be read. A "
                f"published segment promises every row below its watermark is already written in "
                f"every layer file (invariant G1); refusing rather than delivering a short or "
                f"zero-padded artifact.")
        self._stats["rows_read"] += int(n)
        self._stats["bytes_read"] += want
        return np.frombuffer(got, dtype=np.uint8).reshape(int(n), self._row_bytes)

    def _append(self, rid: str, layer: int, block: np.ndarray) -> None:
        path = os.path.join(self.out_dir, req_dir_name(rid), f"hs_layer_{layer}.raw")
        with open(path, "ab") as f:
            f.write(block.tobytes())
        n = int(block.shape[0])
        st = self._reqs[rid]
        st.written_all += n
        if int(layer) == int(st.mine[0]):
            st.written += n
        self._stats["rows_delivered"] += n
        self._stats["bytes_delivered"] += n * self._row_bytes

    def _touch(self, rid: str, st: "_ReqState") -> None:
        if st.touched:
            return
        d = os.path.join(self.out_dir, req_dir_name(rid))
        os.makedirs(d, exist_ok=True)
        for L in st.mine:
            with open(os.path.join(d, f"hs_layer_{L}.raw"), "wb"):
                pass
        st.touched = True

    def _stamp_run(self, man: dict) -> None:
        if self._cursor.run_id is not None:
            man[RUN_ID_KEY] = self._cursor.run_id

    def _write_empty(self, rid: str) -> None:
        d = os.path.join(self.out_dir, req_dir_name(rid))
        os.makedirs(d, exist_ok=True)
        hdr = self._header or {}
        man = {"format": DELIVERY_FORMAT, "req_id": rid, "n_rows": 0, "layers": [],
               "layers_all": [], "worker": 0, "n_workers": self.n_workers,
               "dtype": hdr.get("dtype"), "row_shape": [int(x) for x in hdr.get("row_shape", [])],
               "row_bytes": int(self._row_bytes), "complete": True}
        self._stamp_run(man)
        if self.shard is not None:
            man.update(self.shard.as_header())
        path = os.path.join(d, manifest_name(0, self.n_workers))
        with open(f"{path}.tmp", "w", encoding="utf-8") as f:
            f.write(json.dumps(man, sort_keys=True))
        os.replace(f"{path}.tmp", path)
        self._stats["finalized"] += 1

    def _finalize(self, rid: str, n_total: int) -> None:
        st = self._reqs.pop(rid, None)
        if st is None:
            if int(n_total) == 0:
                if self.worker == 0:
                    self._write_empty(rid)
                return
            self._stats["skipped_stamps"] += 1
            return
        if not st.mine:
            self._stats["skipped_stamps"] += 1
            return
        if st.written != int(n_total):
            raise GatherError(
                f"{rid!r}: the chain stamps {n_total} rows but this pass WROTE {st.written} of them "
                f"to layer {st.mine[0]}. The artifact would be SHORT; refusing to write a manifest "
                f"over it.")
        if st.written_all != st.written * len(st.mine):
            raise GatherError(
                f"{rid!r}: wrote {st.written_all} rows over {len(st.mine)} layers, which is not "
                f"{st.written} per layer -- the layers of one request have diverged, so at least one "
                f"of its layer files is short. Refusing.")
        self._touch(rid, st)
        d = os.path.join(self.out_dir, req_dir_name(rid))
        hdr = self._header or {}
        man = {"format": DELIVERY_FORMAT, "req_id": rid, "n_rows": int(n_total),
               "layers": [int(L) for L in st.mine],
               "layers_all": [int(L) for L in st.layers],
               "worker": self.worker, "n_workers": self.n_workers,
               "dtype": hdr.get("dtype"), "row_shape": [int(x) for x in hdr.get("row_shape", [])],
               "row_bytes": int(self._row_bytes), "complete": True}
        self._stamp_run(man)
        if self.shard is not None:
            man.update(self.shard.as_header())
        path = os.path.join(d, manifest_name(self.worker, self.n_workers))
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(man, sort_keys=True))
        os.replace(tmp, path)
        self._stats["finalized"] += 1


def _as_id_set(req_ids):
    return None if req_ids is None else {str(x) for x in req_ids}


def _manifest_shard(man: dict, where: str):
    try:
        return hs_shard_from_header(man)
    except TPShardError as e:
        raise GatherError(f"{where}: its TP layer shard is unusable: {e}") from None


def discover_delivery_ranks(root: str) -> List[Tuple[int, str]]:
    """``[(tp_rank, dir), ...]`` for each marked ``tp_rank_<N>`` delivery root under ``root``."""
    for base in (root, os.path.join(root, DELIVERY_DIRNAME)):
        out: List[Tuple[int, str]] = []
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        for n in names:
            r = parse_rank_dir(n)
            d = os.path.join(base, n)
            if r is not None and os.path.isdir(d) and os.path.exists(
                    os.path.join(d, RANK_MARKER_NAME)):
                out.append((int(r), d))
        if out:
            return sorted(out)
    return []


def read_manifests(req_dir: str, run_ids=None) -> List[dict]:
    """Every manifest in one request's delivery directory, in worker order."""
    out = []
    for p in sorted(glob.glob(os.path.join(req_dir, MANIFEST_GLOB))):
        with open(p, "r", encoding="utf-8") as f:
            out.append(json.loads(f.read()))
    if run_ids is not None:
        out = [m for m in out if m.get(RUN_ID_KEY) in run_ids]
    return sorted(out, key=lambda m: int(m.get("worker", 0)))


def _foreign_note(req_dir: str, run_ids) -> str:
    if run_ids is None or not glob.glob(os.path.join(req_dir, MANIFEST_GLOB)):
        return ""
    return " (only a previous launch's delivery is here)"


_IOV_MAX = 1024


def _read_into(path: str, bufs, want: int) -> None:
    views = [memoryview(b).cast("B") for b in bufs]
    if sum(len(v) for v in views) != want:
        raise GatherError(f"{path}: {sum(len(v) for v in views)} B of buffers for {want} B")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        pos, i = 0, 0
        while i < len(views):
            n = os.preadv(fd, views[i:i + _IOV_MAX], pos)
            if n <= 0:
                raise GatherError(f"{path}: SHORT -- {pos} of the {want} B its manifest claims "
                                  f"could be read. Refusing.")
            pos += n
            while n:
                if n >= len(views[i]):
                    n -= len(views[i])
                    i += 1
                else:
                    views[i], n = views[i][n:], 0
    finally:
        os.close(fd)


def load_delivery(req_dir: str, *, allow_shard: bool = False, run_ids=None, into=None) -> dict:
    """One request's delivered artifact as ``{layer: Tensor}``, or :class:`GatherError`."""
    mans = read_manifests(req_dir, run_ids)
    if not mans:
        raise GatherError(
            f"{req_dir}: no manifest ({MANIFEST_GLOB}) -- this request was never stamped complete, "
            f"so what is here is a PREFIX of its capture, not its artifact. Not delivered."
            + _foreign_note(req_dir, run_ids))
    first = mans[0]
    n_workers = int(first.get("n_workers", 1))
    n_rows = int(first["n_rows"])
    layers_all = [int(x) for x in first["layers_all"]]
    shard = _manifest_shard(first, req_dir)
    if shard is not None and not allow_shard:
        raise GatherError(
            f"{req_dir}: this is tp_rank {shard.tp_rank} of {shard.tp_size}'s SHARE of the "
            f"request -- layers {list(shard.owned_layers)[:6]}"
            f"{'...' if len(shard.owned_layers) > 6 else ''} of {shard.num_layers}, under the HS "
            f"round-robin layer shard. The request's other layers are in the other ranks' delivery "
            f"dirs. Read the run with load_delivered(<the delivery root, or MIA_APERTURE_DIR>), "
            f"which unions every rank and refuses a gap or a duplicate.")
    for m in mans[1:]:
        if _manifest_shard(m, req_dir) != shard:
            raise GatherError(
                f"{req_dir}: its manifests disagree on the TP layer shard -- worker "
                f"{m.get('worker')} says {_manifest_shard(m, req_dir)}, worker "
                f"{first.get('worker')} says {shard}. Two different runs, or two ranks writing one "
                f"directory.")
    for m in mans:
        if m.get("complete") is not True:
            raise GatherError(
                f"{req_dir}: manifest for worker {m.get('worker')} is not marked complete -- the "
                f"gather was interrupted mid-finalize. Refusing rather than reading what it covers.")
        if int(m.get("n_workers", 1)) != n_workers or int(m["n_rows"]) != n_rows:
            raise GatherError(
                f"{req_dir}: manifests disagree ({n_workers} workers / {n_rows} rows vs "
                f"{m.get('n_workers')} / {m['n_rows']}) -- they describe two different runs.")
        if [int(x) for x in m["layers_all"]] != layers_all:
            raise GatherError(
                f"{req_dir}: manifests disagree on the request's layer list {layers_all} vs "
                f"{m['layers_all']}")
    expected = ({w for w in range(n_workers) if layers_for_worker(layers_all, w, n_workers)}
                if layers_all else {0})
    present = {int(m["worker"]) for m in mans}
    if present != expected:
        raise GatherError(
            f"{req_dir}: the layer partition needs manifests from workers {sorted(expected)} but "
            f"only {sorted(present)} are here -- some layer was never finished, and the layers the "
            f"surviving manifests name would read back perfectly. Refusing.")
    covered: List[int] = []
    for m in mans:
        covered.extend(int(x) for x in m["layers"])
    if sorted(covered) != sorted(layers_all) or len(set(covered)) != len(covered):
        raise GatherError(
            f"{req_dir}: the manifests cover layers {sorted(covered)}, not the request's "
            f"{sorted(layers_all)} exactly once")

    row_shape = tuple(int(x) for x in first["row_shape"])
    dtype_name = first["dtype"]
    np_dtype = np.dtype(_np_name(dtype_name))
    width = 1
    for d in row_shape:
        width *= d
    row_bytes = width * np_dtype.itemsize
    sink = into(list(layers_all), n_rows, row_shape, dtype_name) if into is not None else None
    out: dict = {}
    for L in layers_all:
        p = os.path.join(req_dir, f"hs_layer_{L}.raw")
        try:
            size = os.path.getsize(p)
        except FileNotFoundError:
            raise GatherError(
                f"{p}: the manifest names layer {L} but the file is missing") from None
        want = n_rows * row_bytes
        if size != want:
            how = "SHORT" if size < want else "LONG"
            raise GatherError(
                f"{p}: the manifest claims {n_rows} rows ({want} B) but the file is {size} B -- "
                f"{how}. A manifest is written only after every row it claims is in the file, so "
                f"this artifact is not the one the manifest describes. Refusing.")
        if sink is not None:
            value, bufs = sink[int(L)]
            _read_into(p, bufs, want)
            out[int(L)] = value
            continue
        with open(p, "rb") as f:
            raw = f.read(want)
        arr = np.frombuffer(raw, dtype=np_dtype).reshape((n_rows,) + row_shape)
        t = torch.from_numpy(arr.copy())
        if dtype_name == "bfloat16":
            t = t.view(torch.bfloat16)
        out[int(L)] = t
    return out


def _load_delivered_one_rank(out_dir: str, *, allow_shard: bool = False, req_ids=None,
                             run_ids=None, into=None) -> dict:
    out: dict = {}
    if not os.path.isdir(out_dir):
        return out
    if req_ids is not None:
        for rid in sorted(req_ids):
            d = os.path.join(out_dir, req_dir_name(rid))
            mans = read_manifests(d, run_ids) if os.path.isdir(d) else []
            if not mans:
                continue
            named = str(mans[0]["req_id"])
            if named != rid:
                raise GatherError(f"{d}: its manifest names request {named!r}, not {rid!r}")
            out[rid] = load_delivery(d, allow_shard=allow_shard, run_ids=run_ids, into=into)
        return out
    for name in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, name)
        if not os.path.isdir(d):
            continue
        mans = read_manifests(d, run_ids)
        if not mans:
            continue
        rid = str(mans[0]["req_id"])
        if req_ids is not None and rid not in req_ids:
            continue
        out[rid] = load_delivery(d, allow_shard=allow_shard, run_ids=run_ids, into=into)
    return out


def _rank_shards(ranks: Sequence[Tuple[int, str]]):
    out = []
    for r, d in ranks:
        path = os.path.join(d, RANK_MARKER_NAME)
        with open(path, "r", encoding="utf-8") as f:
            man = json.loads(f.read())
        sh = _manifest_shard(man, path)
        if sh is None:
            raise GatherError(
                f"{path}: a delivery rank marker that declares no HS layer shard. A single-rank "
                f"delivery writes no marker at all, so this one was written by something else or "
                f"edited; refusing to treat it as a rank of a sharded run.")
        if int(sh.tp_rank) != int(r):
            raise GatherError(
                f"{d} is named for tp_rank {int(r)} but its marker is tp_rank {sh.tp_rank}'s")
        out.append(sh)
    return out


def _expected_ranks_for(ask, tp_size: int, num_layers: int):
    if ask is None:
        return sorted(r for r in range(int(tp_size))
                      if hs_owned_rows(int(num_layers), int(tp_size), r))
    return hs_expected_ranks(hs_requested_layers(list(ask), int(num_layers)), int(tp_size))


def _listdir(d: str) -> List[str]:
    try:
        return sorted(os.listdir(d))
    except OSError:
        return []


def _aperture_contents(d: str) -> List[str]:
    out = []
    for n in _listdir(d):
        p = os.path.join(d, n)
        if parse_rank_dir(n) is not None and os.path.isdir(p):
            out.append(n + "/")
        elif n.endswith((".raw", ".jsonl")) and os.path.isfile(p):
            out.append(n)
    return out


def _level_root(d: str) -> Optional[str]:
    ranks = discover_delivery_ranks(d)
    if ranks:
        return os.path.dirname(ranks[0][1])
    sub = os.path.join(d, DELIVERY_DIRNAME)
    if os.path.isdir(sub):
        return sub
    tps = sorted((r, os.path.join(d, n, DELIVERY_DIRNAME)) for n in _listdir(d)
                 for r in [parse_rank_dir(n)]
                 if r is not None and os.path.isdir(os.path.join(d, n, DELIVERY_DIRNAME)))
    if len(tps) > 1 or (tps and tps[0][0] != 0):
        raise GatherError(
            f"{d}: per-rank deliveries {[p for _, p in tps]} carry no {RANK_MARKER_NAME}, so they "
            f"are not one sharded run (the all-ranks diagnostic writes replicas). Read one of them "
            f"explicitly.")
    if tps:
        return tps[0][1]
    return None


def _check_delivery_root(root: str) -> None:
    if discover_delivery_ranks(root):
        return
    found = _aperture_contents(root)
    if found:
        raise NoDeliveryError(
            f"{root} is a capture-aperture dir (it holds {found[:4]}) with no per-request delivery "
            f"under it: no {RANK_MARKER_NAME}, no {DELIVERY_DIRNAME}/, no "
            f"tp_rank_0/{DELIVERY_DIRNAME}. Its capture is in the shared layer files "
            f"(aperture_reader.load_hs_aperture_tp), or its delivery is under {DIR_ENV}.",
            "no delivery under the capture dir yet")


def _resolve_roots(d, skipped: Optional[List[str]] = None) -> List[str]:
    d = os.path.abspath(os.fspath(d))
    if not os.path.isdir(d):
        raise NoDeliveryError(f"{d}: no such delivery root or aperture dir",
                              "the delivery root does not exist yet")
    own = _level_root(d)
    dps = [os.path.join(d, n) for n in _listdir(d)
           if parse_dp_dir(n) is not None and os.path.isdir(os.path.join(d, n))]
    dps.sort(key=parse_dp_dir)
    if not dps:
        roots = [own if own is not None else d]
    else:
        loose = [n for n in _listdir(d) if os.path.isdir(os.path.join(d, n))
                 and parse_dp_dir(n) is None and parse_rank_dir(n) is None
                 and n != DELIVERY_DIRNAME]
        roots = [own] if own is not None else ([d] if loose else [])
        for c in dps:
            r = _level_root(c)
            if r is None and not _aperture_contents(c):
                r = c
            if r is not None:
                roots.append(r)
            elif skipped is not None:
                skipped.append(c)
        if not roots:
            raise NoDeliveryError(f"{d}: none of its DP engine dirs {dps} holds a delivery",
                                  "no DP engine dir holds a delivery yet")
    for r in roots:
        _check_delivery_root(r)
    return roots


def _as_paths(roots) -> List[str]:
    return [roots] if isinstance(roots, (str, os.PathLike)) else list(roots)


def _resolve_all(roots, skipped: Optional[List[str]] = None) -> List[str]:
    out, seen = [], set()
    for item in _as_paths(roots):
        for r in _resolve_roots(item, skipped):
            k = os.path.realpath(r)
            if k not in seen:
                seen.add(k)
                out.append(r)
    if not out:
        raise GatherError("no delivery roots given")
    return out


def delivery_base(aperture_dir: Optional[str] = None) -> str:
    """The dir a run's delivery roots resolve from (``MIA_APERTURE_GATHER_DIR`` first)."""
    return (os.environ.get(DIR_ENV) or aperture_dir or os.environ.get("MIA_APERTURE_DIR")
            or "./hs_aperture_dump")


def delivery_root(aperture_dir: Optional[str] = None) -> List[str]:
    """Every delivery root of a run, one per DP engine."""
    return _resolve_roots(delivery_base(aperture_dir))


def _dup_message(rid: str, a: str, b: str) -> str:
    return (f"request {rid!r} is delivered under both {a} and {b} -- two runs or two DP engines "
            f"used one id. Refusing to pick one; read one root explicitly.")


def load_delivered(out_dir, *, expected_layers=None, req_ids=None, run_ids=None,
                   into=None) -> dict:
    """Every complete delivery under ``out_dir`` as ``{req_id: {layer: Tensor}}``, TP-aware."""
    roots = _resolve_all(out_dir)
    want = _as_id_set(req_ids)
    if len(roots) == 1:
        return _load_delivered_root(roots[0], expected_layers=expected_layers, req_ids=want,
                                    run_ids=run_ids, into=into)
    out: dict = {}
    where: dict = {}
    for root in roots:
        for rid, art in _load_delivered_root(root, expected_layers=expected_layers,
                                             req_ids=want, run_ids=run_ids, into=into).items():
            if rid in where:
                raise GatherError(_dup_message(rid, where[rid], root))
            out[rid] = art
            where[rid] = root
    return out


def _load_delivered_root(out_dir: str, *, expected_layers=None, req_ids=None,
                         run_ids=None, into=None) -> dict:
    ranks = discover_delivery_ranks(out_dir)
    if not ranks:
        return _load_delivered_one_rank(out_dir, req_ids=_as_id_set(req_ids), run_ids=run_ids,
                                        into=into)
    want_ids = _as_id_set(req_ids)
    shards = _rank_shards(ranks)
    tp_size, num_layers = shards[0].tp_size, shards[0].num_layers
    whole = None if isinstance(expected_layers, dict) else expected_layers
    check_hs_shard_set(shards, _expected_ranks_for(whole, tp_size, num_layers)
                       if whole is not None else None)
    per_rank = {sh.tp_rank: _load_delivered_one_rank(d, allow_shard=True, req_ids=want_ids,
                                                     run_ids=run_ids, into=into)
                for sh, (_r, d) in zip(shards, ranks)}
    present = {sh.tp_rank: sh for sh in shards}
    seen: dict = {}
    for r, art in per_rank.items():
        for rid, layers in art.items():
            if layers:
                seen.setdefault(str(rid), set()).add(int(r))
    if into is not None:
        dirs = {int(sh.tp_rank): d for sh, (_r, d) in zip(shards, ranks)}
        for rid, have in sorted(seen.items()):
            rows = {r: max((int(m["n_rows"]) for m in read_manifests(
                os.path.join(dirs[r], req_dir_name(rid)), run_ids)), default=0)
                for r in sorted(have)}
            if len(set(rows.values())) > 1:
                raise TPShardError(
                    f"req {rid!r}: its layers hold different row counts across ranks (rank -> "
                    f"rows {rows}); every layer of a request captures the same tokens, so the "
                    f"ranks captured different steps")
    for rid, have in sorted(seen.items()):
        ask = expected_layers.get(rid) if isinstance(expected_layers, dict) else whole
        want = set(_expected_ranks_for(ask, tp_size, num_layers)).intersection(present)
        if have != want:
            missing = sorted(want - have)
            lost = sorted(L for r in missing for L in present[r].owned_layers)
            raise GatherError(
                f"req {rid!r}: delivered by tp_rank(s) {sorted(have)} but tp_rank(s) {missing} "
                f"{'have' if len(missing) != 1 else 'has'} not stamped it, so layers {lost[:8]}"
                f"{'...' if len(lost) > 8 else ''} ({len(lost)} of {num_layers}) are MISSING from "
                f"an artifact whose other ranks read back perfectly. Refusing. (If this request "
                f"legitimately asked only for other ranks' layers, pass expected_layers={{req_id: "
                f"[its layers]}} -- it is a gap relaxation, not a filter.)"
                + (f" Extra: tp_rank(s) {sorted(have - want)} delivered it and were not expected."
                   if have - want else ""))
    return merge_hs_layer_maps([(sh, per_rank[sh.tp_rank]) for sh in shards])


def _worker_gap(req_dir: str, run_ids=None) -> str:
    if not os.path.isdir(req_dir):
        return "no rows delivered yet"
    mans = read_manifests(req_dir, run_ids)
    if not mans:
        return "rows arriving, no manifest yet" + _foreign_note(req_dir, run_ids)
    first = mans[0]
    nw = int(first.get("n_workers", 1))
    layers_all = [int(x) for x in first["layers_all"]]
    need = {w for w in range(nw) if layers_for_worker(layers_all, w, nw)}
    have = {int(m.get("worker", 0)) for m in mans}
    if have < need:
        return (f"gather worker manifest(s) {sorted(have)} present, {sorted(need - have)} "
                f"missing")
    return ""


class _WaitRoot:

    def __init__(self, root: str, expected_layers):
        self.root = root
        ranks = discover_delivery_ranks(root)
        self.ranks: Dict[int, str] = {}
        self.missing: List[int] = []
        if ranks:
            shards = _rank_shards(ranks)
            self.tp_size, self.num_layers = shards[0].tp_size, shards[0].num_layers
            whole = None if isinstance(expected_layers, dict) else expected_layers
            need = _expected_ranks_for(whole, self.tp_size, self.num_layers)
            present = sorted(int(sh.tp_rank) for sh in shards)
            check_hs_shard_set(shards, present)
            self.missing = sorted(set(need) - set(present))
            self.ranks = {int(sh.tp_rank): d for sh, (_r, d) in zip(shards, ranks)}

    def dirs(self) -> List[str]:
        return list(self.ranks.values()) if self.ranks else [self.root]

    def holds(self, rid: str) -> bool:
        q = req_dir_name(rid)
        return any(os.path.isdir(os.path.join(d, q)) for d in self.dirs())

    def gap(self, rid: str, expected_layers, run_ids=None) -> str:
        q = req_dir_name(rid)
        if not self.ranks:
            return _worker_gap(os.path.join(self.root, q), run_ids)
        if self.missing:
            return f"tp_rank(s) {self.missing} have no {RANK_MARKER_NAME} yet"
        ask = expected_layers.get(rid) if isinstance(expected_layers, dict) else expected_layers
        want = sorted(set(_expected_ranks_for(ask, self.tp_size, self.num_layers))
                      & set(self.ranks))
        gaps = {r: _worker_gap(os.path.join(self.ranks[r], q), run_ids) for r in want}
        late = [r for r in want if gaps[r]]
        if not late:
            return ""
        return (f"tp_rank(s) {[r for r in want if not gaps[r]]} stamped it, {late} not yet ("
                + "; ".join(f"tp_rank {r}: {gaps[r]}" for r in late) + ")")


def _pending(views: List[_WaitRoot], ids: List[str], expected_layers,
             run_ids=None) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for rid in ids:
        homes = [v for v in views if v.holds(rid)]
        if len(homes) > 1:
            raise GatherError(_dup_message(rid, homes[0].root, homes[1].root))
        if not homes:
            out[rid] = ("no delivery yet under any root", None)
            continue
        why = homes[0].gap(rid, expected_layers, run_ids)
        if why:
            out[rid] = (why, homes[0].root)
    return out


def _progress(views: List[_WaitRoot], ids) -> tuple:
    sig: list = []
    for v in views:
        for d in v.dirs():
            try:
                sig.append(os.stat(d).st_mtime_ns)
            except OSError:
                sig.append(None)
            for rid in ids:
                try:
                    with os.scandir(os.path.join(d, req_dir_name(rid))) as it:
                        sig.append(sum(e.stat().st_size if e.name.endswith(".raw") else 1
                                       for e in it))
                except OSError:
                    sig.append(None)
    return tuple(sig)


class _Poll:
    __slots__ = ("got", "pending", "views", "dirs", "absent", "skipped")

    def __init__(self, got, pending, views, dirs, absent, skipped):
        self.got, self.pending, self.views = got, pending, views
        self.dirs, self.absent, self.skipped = dirs, absent, skipped

    def mark(self) -> tuple:
        """Changes when the gather writes into the roots or the pending requests' dirs."""
        if self.absent is not None:
            return (str(self.absent),)
        return _progress(self.views, list(self.pending)) + (tuple(v.missing for v in self.views),)


def _manifest_rows(views: List[_WaitRoot], rid: str, run_ids=None) -> int:
    q = req_dir_name(rid)
    return max((int(m["n_rows"]) for v in views if v.holds(rid) for d in v.dirs()
                for m in read_manifests(os.path.join(d, q), run_ids)), default=0)


def _poll(roots, ids: List[str], expected_layers, given: List[str], run_ids=None,
          load: bool = True) -> _Poll:
    skipped: List[str] = []
    try:
        dirs = _resolve_all(roots, skipped)
        views = [_WaitRoot(r, expected_layers) for r in dirs]
    except NoDeliveryError as e:
        pending = {rid: (e.what, None) for rid in ids}
        return _Poll(None, pending, [], given, e, skipped)
    pending = _pending(views, ids, expected_layers, run_ids)
    if pending:
        return _Poll(None, pending, views, dirs, None, skipped)
    if not load:
        return _Poll({rid: _manifest_rows(views, rid, run_ids) for rid in ids}, {}, views, dirs,
                     None, skipped)
    homes = list(dict.fromkeys(v.root for rid in ids for v in views if v.holds(rid)))
    got = load_delivered(homes, expected_layers=expected_layers, req_ids=ids, run_ids=run_ids)
    lost = [r for r in ids if r not in got]
    if lost:
        raise GatherError(f"requests {lost} were complete on disk under {homes} but "
                          f"did not read back")
    return _Poll(got, {}, views, dirs, None, skipped)


def poll_delivered(roots, req_ids, expected_layers=None, run_ids=None):
    """One non-blocking look at whether every id is delivered."""
    ids = list(dict.fromkeys(str(x) for x in req_ids))
    given = [os.path.abspath(os.fspath(r)) for r in _as_paths(roots)]
    p = _poll(roots, ids, expected_layers, given, run_ids)
    if p.got is not None:
        return p.got, {}, ()
    return None, {rid: msg for rid, (msg, _w) in p.pending.items()}, p.mark()


def delivery_backlog(roots, req_ids, expected_layers=None, run_ids=None):
    """``({req_id: what it lacks}, mark)`` for ids not fully delivered; ``mark`` tracks progress."""
    ids = list(dict.fromkeys(str(x) for x in req_ids))
    if not ids:
        return {}, ()
    given = [os.path.abspath(os.fspath(r)) for r in _as_paths(roots)]
    p = _poll(roots, ids, expected_layers, given, run_ids, load=False)
    if p.got is not None:
        return {}, ()
    return {rid: msg for rid, (msg, _w) in p.pending.items()}, p.mark()


def wait_delivered(roots, req_ids, *, expected_layers=None, timeout_s=None, run_ids=None,
                   load: bool = True) -> dict:
    """Poll until every id in ``req_ids`` is delivered, then load exactly those ids."""
    idle = delivery_timeout_s() if timeout_s is None else float(timeout_s)
    if not idle > 0:
        raise GatherError(f"timeout_s={timeout_s!r} must be a positive number of seconds")
    ids = list(dict.fromkeys(str(x) for x in req_ids))
    if not ids:
        return {}
    given = [os.path.abspath(os.fspath(r)) for r in _as_paths(roots)]
    probe_every = min(1.0, idle / 4)
    start = last = probed = time.monotonic()
    seen = None
    while True:
        p = _poll(roots, ids, expected_layers, given, run_ids, load)
        if p.got is not None:
            return p.got
        now = time.monotonic()
        if seen is None or now - probed >= probe_every:
            probed = now
            mark = p.mark()
            if mark != seen:
                seen, last = mark, now
        capped = now - start >= DELIVERY_HARD_CAP_S
        if capped or now - last >= idle:
            why = (f"the {DELIVERY_HARD_CAP_S:.0f} s hard cap" if capped
                   else f"{idle:g} s with no new delivery")
            notes = [str(p.absent)] if p.absent is not None else []
            notes += [f"{v.root}: tp_rank(s) {v.missing} have no {RANK_MARKER_NAME}"
                      for v in p.views if v.missing]
            if p.skipped:
                notes.append(f"skipped DP engine dirs holding only a capture: {p.skipped}")
            raise DeliveryTimeoutError(
                f"{len(p.pending)} of {len(ids)} request(s) not delivered after "
                f"{now - start:.1f} s ({why}) under {p.dirs}:\n"
                + "\n".join(f"  {rid!r}: {msg}" + (f" (under {where})" if where else "")
                            for rid, (msg, where) in p.pending.items())
                + "".join(f"\n  note: {n}" for n in notes)
                + f"\nIf the gather is only slow, raise {DELIVERY_TIMEOUT_ENV} (now {idle:g} s; "
                  f"hard cap {DELIVERY_HARD_CAP_S:.0f} s).",
                missing={rid: msg for rid, (msg, _w) in p.pending.items()}, roots=p.dirs)
        time.sleep(_WAIT_POLL_S)


_CHILD_THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def _gather_child(ctl_q, run_dir: str, out_dir: str, worker: int, n_workers: int,
                  batch_bytes: int, poll_s: float, trim: bool = True,
                  trim_chunk: int = 0, trim_lag: int = 0, tp_rank: Optional[int] = None,
                  tp_size: int = 1, num_layers: Optional[int] = None,
                  run_id: Optional[str] = None, progress=None) -> None:
    for k, v in _CHILD_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    parent = multiprocessing.parent_process()
    p = GatherPass(run_dir, out_dir, worker=worker, n_workers=n_workers, batch_bytes=batch_bytes,
                   trim=trim, trim_chunk=trim_chunk or None, trim_lag=trim_lag if trim_lag >= 0
                   else None, tp_rank=tp_rank, tp_size=tp_size, num_layers=num_layers,
                   run_id=run_id)
    if progress is not None:
        def _bump():
            progress.value += 1
        p.on_batch = _bump
    stop = False
    try:
        while not stop:
            try:
                item = ctl_q.get(timeout=poll_s)
                stop = item is None
            except _queue.Empty:
                if parent is not None and not parent.is_alive():
                    stop = True
            try:
                p.poll_once()
            except Exception as e:  # noqa: BLE001
                print(f"[aperture-gather] worker {worker}/{n_workers} stopped on {e!r}", flush=True)
                return
            if p.final_seen:
                stop = True
        try:
            p.poll_once()
        except Exception as e:  # noqa: BLE001
            print(f"[aperture-gather] worker {worker}/{n_workers} final pass failed: {e!r}",
                  flush=True)
        print(f"[aperture-gather] rank {tp_rank} worker {worker}/{n_workers} done: {p.stats()}",
              flush=True)
    finally:
        p.close()


class ApertureGatherProcess:
    """``n_workers`` spawned children following one run's index chain, layer-partitioned."""

    def __init__(self, run_dir: str, out_dir: Optional[str] = None, *,
                 n_workers: Optional[int] = None,
                 batch_bytes: Optional[int] = None, poll_s: Optional[float] = None,
                 trim: Optional[bool] = None, header: Optional[dict] = None,
                 run_id: Optional[str] = None):
        self.run_dir = str(run_dir)
        self.run_id = None if run_id is None else str(run_id)
        self.shard = _manifest_shard(header or {}, f"{self.run_dir} index header")
        self.tp_rank = None if self.shard is None else int(self.shard.tp_rank)
        self.tp_size = 1 if self.shard is None else int(self.shard.tp_size)
        self.num_layers = None if self.shard is None else int(self.shard.num_layers)
        _dp = (header or {}).get("dp_rank")
        self.dp_rank = None if _dp is None else int(_dp)
        self.out_dir = str(out_dir) if out_dir is not None else delivery_dir(
            self.run_dir, tp_rank=self.tp_rank, dp_rank=self.dp_rank)
        if n_workers is None:
            _w = resolve_gather_workers(self.tp_size)
            self.n_workers, self.workers_note = _w.n, _w.note
        else:
            self.n_workers = max(1, int(n_workers))
            self.workers_note = f"{self.n_workers} gather worker(s) per rank (passed in)"
        self.batch_bytes = int(batch_bytes) if batch_bytes is not None else gather_batch_bytes()
        self.poll_s = float(poll_s) if poll_s is not None else gather_poll_s()
        self.trim = trim_enabled() if trim is None else bool(trim)
        # Only an asked-for trim refuses a dir that cannot punch; the default turns off.
        self.trim_asked = trim is not None or trim_explicit()
        self.no_punch: Optional[str] = None
        self.trim_chunk = trim_chunk_bytes()
        self.trim_lag = trim_lag_bytes()
        self.parent_daemonic = False
        self._procs: list = []
        self._qs: list = []
        self._progress: list = []
        self._closed = False

    @classmethod
    def from_env(cls, run_dir: str, out_dir: Optional[str] = None,
                 header: Optional[dict] = None, run_id: Optional[str] = None):
        """A started gather, or None when ``MIA_APERTURE_GATHER_DELIVER`` is not ``1``."""
        if not delivery_enabled():
            return None
        gp = cls(run_dir, out_dir, header=header, run_id=run_id)
        gp.start()
        return gp

    def start(self) -> None:
        # lazy: child_process reads env at import; keep it out of plugin load
        from mia.core.runtime.child_process import register_shutdown, start_child

        if self._procs:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        trim_note = f"trim OFF ({GATHER_TRIM_ENV}=0): the shared layer files are kept WHOLE"
        if self.trim:
            ok, why = punch_supported(self.run_dir)
            if not ok and self.trim_asked:
                raise TrimRefusedError(_no_punch_message(self.run_dir, why))
            if not ok:
                self.trim, self.no_punch = False, why
                trim_note = _no_punch_warning(self.run_dir, why)
        if self.trim:
            trim_note = (f"trim ON by default ({GATHER_TRIM_ENV}=0 turns it off): the shared "
                         f"hs_layer_*.raw will be fallocate(PUNCH_HOLE)d behind the SLOWEST "
                         f"worker's cursor in {self.trim_chunk} B chunks, {self.trim_lag} B lag, "
                         f"{trim_align(self.run_dir)} B aligned. The files keep their LENGTH; rows "
                         f"below the published floor in hs_trim.w*of*.json are RECLAIMED and "
                         f"aperture_reader names them instead of returning zeros. The delivered "
                         f"artifacts under {self.out_dir} are this run's product")
        ctx = multiprocessing.get_context("spawn")
        saved = {k: os.environ.get(k) for k in (*_CHILD_THREAD_ENV, "CUDA_VISIBLE_DEVICES")}
        try:
            for k, v in _CHILD_THREAD_ENV.items():
                os.environ.setdefault(k, v)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            for w in range(self.n_workers):
                q = ctx.Queue()
                done = ctx.Value("q", 0, lock=False)
                proc = ctx.Process(target=_gather_child,
                                   args=(q, self.run_dir, self.out_dir, w, self.n_workers,
                                         self.batch_bytes, self.poll_s, self.trim,
                                         self.trim_chunk, self.trim_lag, self.tp_rank,
                                         self.tp_size, self.num_layers, self.run_id, done),
                                   daemon=True, name=f"mia-gather-{w}")
                self.parent_daemonic = start_child(proc)
                self._qs.append(q)
                self._procs.append(proc)
                self._progress.append(done)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        register_shutdown(self.close)
        print(f"[aperture-gather] streaming delivery ON: {self.workers_note} over "
              f"{self.run_dir} -> {self.out_dir}, child pid(s) {[p.pid for p in self._procs]}"
              + (f", run id {self.run_id}" if self.run_id else ", NO run id (a chain left in this "
                 "directory by a previous launch cannot be told from this one's)"),
              flush=True)
        if self.shard is not None:
            print(f"[aperture-gather] TP layer shard: tp_rank {self.shard.tp_rank} of "
                  f"{self.shard.tp_size} delivers layers "
                  f"{list(self.shard.owned_layers)[:8]}"
                  f"{'...' if len(self.shard.owned_layers) > 8 else ''} "
                  f"({len(self.shard.owned_layers)} of {self.shard.num_layers}) into "
                  f"{self.out_dir}; a request's artifact is the UNION over the ranks -- read it "
                  f"with aperture_gather.load_delivered(<the run's delivery root>), which refuses a "
                  f"gap or a duplicate instead of returning one rank's share", flush=True)
        print(f"[aperture-gather] {trim_note}", flush=True)
        (logger.warning if self.no_punch else logger.info)("aperture gather trim: %s", trim_note)

    def alive(self) -> bool:
        return bool(self._procs) and any(p.is_alive() for p in self._procs)

    def close(self, timeout: float = 60.0) -> None:
        """Stop every child after one last pass, bounded."""
        if self._closed:
            return
        self._closed = True
        for q in self._qs:
            try:
                q.put(None, timeout=10)
            except Exception:  # noqa: BLE001
                pass
        start = last = time.monotonic()
        seen, why = None, ""
        while any(p.is_alive() for p in self._procs):
            now = time.monotonic()
            mark = tuple(v.value for v in self._progress)
            if mark != seen:
                seen, last = mark, now
            if now - start >= DELIVERY_HARD_CAP_S:
                why = f"reached the {DELIVERY_HARD_CAP_S:g} s hard cap"
                break
            if now - last >= timeout:
                why = f"made no progress for {timeout:g} s"
                break
            for p in self._procs:
                try:
                    p.join(timeout=0.2)
                except Exception:  # noqa: BLE001
                    pass
        for p in self._procs:
            if p.is_alive():
                logger.error("aperture gather child %s %s; terminating (its unflushed deliveries "
                             "are lost)", p.pid, why)
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass


__all__ = ["ApertureGatherProcess", "BATCH_BYTES_ENV", "DELIVER_ENV", "DELIVERY_FORMAT",
           "DEFAULT_WORKERS", "DIR_ENV", "GATHER_TRIM_ENV", "GatherError", "GatherPass",
           "DELIVERY_TIMEOUT_ENV", "DeliveryTimeoutError", "NoDeliveryError", "TrimRefusedError",
           "delivery_root",
           "delivery_backlog", "delivery_timeout_s",
           "wait_delivered",
           "GatherWorkers", "MANIFEST_GLOB", "resolve_gather_workers",
           "POLL_MS_ENV", "RANK_MARKER_FORMAT", "RANK_MARKER_NAME", "WORKERS_ENV",
           "clip_run", "delivery_dir", "delivery_enabled", "discover_delivery_ranks",
           "gather_batch_bytes", "gather_poll_s",
           "gather_workers", "layers_for_worker", "load_delivered", "load_delivery",
           "manifest_name", "read_manifests", "req_dir_name"]
