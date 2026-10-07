"""Run-encoded per-request row index into the shared HS layer files (``MIA_APERTURE_GATHER``)."""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aperture_trim import MARKER_GLOB

logger = logging.getLogger(__name__)

GATHER_ENV = "MIA_APERTURE_GATHER"
FLUSH_MS_ENV = "MIA_APERTURE_GATHER_FLUSH_MS"

RUN_ID_KEY = "run_id"
INDEX_NAME = "hs_run_index.jsonl"
SEGMENT_FMT = "hs_run_index.seg.%06d.jsonl"
SEGMENT_RE = re.compile(r"hs_run_index\.seg\.(\d{6})\.jsonl$")


class RunIndexError(RuntimeError):
    """The index's layer-independence invariant was violated, or a step cannot be represented."""


def new_run_id() -> str:
    """An identity for one launch's index chain: ``<host>-<pid>-<ns>-<rand>``."""
    try:
        host = socket.gethostname().split(".", 1)[0]
    except OSError:
        host = "?"
    return f"{host}-{os.getpid()}-{time.time_ns():x}-{uuid.uuid4().hex[:6]}"


def gather_enabled() -> bool:
    """``MIA_APERTURE_GATHER``: default off."""
    return os.environ.get(GATHER_ENV, "0") == "1"


def flush_interval_ms() -> int:
    """``MIA_APERTURE_GATHER_FLUSH_MS``: mid-run publish cadence; ``0`` writes one segment at close."""
    raw = os.environ.get(FLUSH_MS_ENV)
    if raw is None or raw.strip() == "0":
        return 0
    try:
        ms = int(raw.strip())
    except ValueError:
        raise RunIndexError(
            f"{FLUSH_MS_ENV}={raw!r} is not a whole number of milliseconds. It is the cadence at "
            f"which the run index is published mid-run; unset it (or set it to 0) to write one "
            f"segment at close.") from None
    if ms < 0:
        raise RunIndexError(
            f"{FLUSH_MS_ENV}={raw!r} is negative. Unset it (or set 0) to disable the incremental "
            f"flush; a positive value is the cadence in milliseconds.")
    return ms


def segment_name(seq: int) -> str:
    return SEGMENT_FMT % int(seq)


def clear_chain(run_dir: str) -> None:
    """Remove a previous launch's index chain and trim markers from ``run_dir``."""
    stale = glob.glob(os.path.join(run_dir, "hs_run_index.seg.*.jsonl"))
    stale += glob.glob(os.path.join(run_dir, MARKER_GLOB)) + [os.path.join(run_dir, INDEX_NAME)]
    for p in stale:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def segment_paths(run_dir: str) -> List[str]:
    """Every index segment in ``run_dir``, in ``seq`` order."""
    out = [p for p in glob.glob(os.path.join(run_dir, "hs_run_index.seg.*.jsonl"))
           if SEGMENT_RE.search(os.path.basename(p))]
    return sorted(out, key=lambda p: int(SEGMENT_RE.search(os.path.basename(p)).group(1)))


@dataclass(frozen=True)
class ReqRuns:
    """One request's rows, as runs of row indices into every layer file it captured."""
    req_id: str
    layers: Tuple[int, ...]
    runs: List[Tuple[int, int, int]]
    n_rows: int

    def rows(self) -> List[int]:
        """The expanded row indices, ascending."""
        out: List[int] = []
        for base, stride, n in self.runs:
            out.extend(range(base, base + stride * n, stride) if stride else [base] * n)
        return out


class RunIndex:
    """Run index of one shared-file HS capture, derived from its sidecar's per-step arrays."""

    def __init__(self, sidecar_log):
        self._log = sidecar_log
        self.rows_total = 0
        self.n_steps = 0
        self._finished: Deque[Tuple[str, int]] = deque()
        self._asked: set = set()

    def note_step(self, step_start: int, n_rows: int, cursor, plans=None, block=None) -> None:
        """Check this step keeps the index's invariants, and advance the file watermark."""
        if plans is not None:
            raise RunIndexError(
                f"{GATHER_ENV}=1 requires a FULL drain, but a COMPACTING (selective) step reached "
                f"the index at logical row {step_start}: a compacted layer's file rows are not the "
                f"aperture's logical rows, so one run list cannot serve every layer. Unset "
                f"MIA_DRAIN_SELECTIVE, or unset {GATHER_ENV}.")
        if not isinstance(cursor, (int, np.integer)) or int(cursor) != int(step_start):
            raise RunIndexError(
                f"{GATHER_ENV}=1 requires every layer's file cursor to equal the aperture's "
                f"logical cursor, but the step at logical row {step_start} was appended at file "
                f"cursor {cursor!r}. The index is layer-independent only while every installed "
                f"layer receives every row.")
        if int(step_start) != int(self.rows_total):
            raise RunIndexError(
                f"the index's file watermark is {self.rows_total} but the step reaching it starts "
                f"at logical row {step_start}: a step was dropped, re-ordered, or indexed twice "
                f"(the drain's FIFO invariant is what makes the file cursor the logical cursor)")
        if block is not None and block[0] != "rows":
            raise RunIndexError(
                f"the sidecar log kept the step at logical row {step_start} as LayerEntry objects "
                f"rather than as arrays (a record with a non-int or unhashable field). The run "
                f"index derives from the arrays and will not guess at the fallback shape.")
        self.rows_total = int(step_start) + int(n_rows)
        self.n_steps += 1

    def note_finish(self, req_id, asked: bool = False) -> None:
        """Record that ``req_id`` has finished, at the current file watermark."""
        if asked:
            self._asked.add(str(req_id))
        self._finished.append((str(req_id), int(self.rows_total)))

    def pop_asked(self, req_id) -> bool:
        """Whether a finished request had asked for capture (so zero rows is its answer)."""
        try:
            self._asked.remove(str(req_id))
            return True
        except KeyError:
            return False

    def take_finished(self, up_to: int) -> List[Tuple[str, int]]:
        """Pop every recorded finish at or below ``up_to``, in order."""
        out: List[Tuple[str, int]] = []
        q = self._finished
        while q:
            rid, wm = q[0]
            if wm > int(up_to):
                break
            q.popleft()
            out.append((rid, wm))
        return out

    def has_finished(self, up_to: int) -> bool:
        """Whether a recorded finish could be stamped by a segment published at ``up_to``."""
        q = self._finished
        return bool(q) and q[0][1] <= int(up_to)

    def snapshot(self, block_lo: int = 0) -> Tuple[int, list]:
        """``(watermark, blocks[block_lo:])``, taken in the one order that is safe."""
        wm = self.rows_total
        blocks = self._log.blocks[int(block_lo):]
        return int(wm), blocks

    def encode(self, *, up_to: Optional[int] = None, since: int = 0,
               blocks: Optional[list] = None) -> Dict[str, ReqRuns]:
        """``{req_id: ReqRuns}`` for every request with rows in ``[since, up_to)``."""
        if blocks is not None and up_to is None:
            raise RunIndexError(
                "encode(blocks=...) must be given the up_to the snapshot was taken with: reading "
                "rows_total after a caller's own block snapshot is the unsafe order of the pair "
                "(it can name rows that no block in the snapshot describes). Use snapshot().")
        if blocks is None:
            snap_wm, blocks = self.snapshot()
        else:
            snap_wm = int(up_to)
        wm = snap_wm if up_to is None else int(up_to)
        lo = int(since)
        bad = [b for b in blocks if b[0] != "rows"]
        if bad:
            raise RunIndexError(
                f"{len(bad)} step(s) were kept as LayerEntry objects rather than arrays; the run "
                f"index does not represent them (see note_step)")
        blocks = [b for b in blocks if b[0] == "rows"]
        rid_vals = self._log.req_ids()
        lay_lists = self._log.layer_lists()
        if not blocks or wm <= lo:
            return {}
        big = np.concatenate([b[2] for b in blocks]) if len(blocks) > 1 else blocks[0][2]
        rid = big[:, 0]
        start = big[:, 1]
        nrow = big[:, 2]
        lidx = big[:, 4]

        vis = (start < wm) & (np.maximum(start + nrow, start + 1) > lo)
        seen: Dict[int, int] = {}
        n_lay = max(1, len(lay_lists))
        for k in np.unique(rid[vis] * n_lay + lidx[vis]).tolist():
            r, li = divmod(int(k), n_lay)
            prev = seen.get(r)
            if prev is None:
                seen[r] = li
            elif sorted(lay_lists[prev]) != sorted(lay_lists[li]):
                raise RunIndexError(
                    f"request {rid_vals[r]!r} was recorded with two different layer SETS "
                    f"({list(lay_lists[prev])} then {list(lay_lists[li])}); a request's layer list "
                    f"is fixed when it is made, so this is a corrupt or re-used request id")

        keep = vis & (nrow > 0)
        s_eff = np.maximum(start[keep], lo)
        n_eff = np.minimum(start[keep] + nrow[keep], wm) - s_eff
        r_eff = rid[keep]
        if n_eff.size:
            inw = n_eff > 0
            s_eff, n_eff, r_eff = s_eff[inw], n_eff[inw], r_eff[inw]
        runs_by_rid: Dict[int, List[Tuple[int, int, int]]] = {}
        rows_by_rid: Dict[int, int] = {}
        if s_eff.size:
            order = np.lexsort((s_eff, r_eff))
            s_eff, n_eff, r_eff = s_eff[order], n_eff[order], r_eff[order]
            total = int(n_eff.sum())
            off = np.concatenate(([0], np.cumsum(n_eff)[:-1]))
            pos = np.arange(total) - np.repeat(off, n_eff)
            rows = np.repeat(s_eff, n_eff) + pos
            owner = np.repeat(r_eff, n_eff)
            cnts = np.bincount(owner)
            for r in np.flatnonzero(cnts).tolist():
                rows_by_rid[int(r)] = int(cnts[r])
            new_req = np.empty(total, dtype=bool)
            new_req[0] = True
            if total > 1:
                new_req[1:] = owner[1:] != owner[:-1]
            start_run = new_req.copy()
            if total > 2:
                d = np.diff(rows)
                same = ~new_req
                start_run[2:] |= same[2:] & same[1:-1] & (d[1:] != d[:-1])
            at = np.flatnonzero(start_run)
            ends = np.concatenate((at[1:], [total]))
            counts = ends - at
            bases = rows[at]
            strides = np.where(counts > 1, rows[np.minimum(at + 1, total - 1)] - bases, 1)
            for r, b, st, c in zip(owner[at].tolist(), bases.tolist(), strides.tolist(),
                                   counts.tolist()):
                runs_by_rid.setdefault(r, []).append((int(b), int(st), int(c)))

        out: Dict[str, ReqRuns] = {}
        for r, li in seen.items():
            out[rid_vals[r]] = ReqRuns(req_id=rid_vals[r], layers=tuple(lay_lists[li]),
                                       runs=runs_by_rid.get(r, []), n_rows=rows_by_rid.get(r, 0))
        return out

    def write(self, path: str, header: dict, *, up_to: Optional[int] = None,
              since: int = 0) -> None:
        """Write one index segment covering ``[since, up_to)``."""
        if up_to is None:
            wm, blocks = self.snapshot()
        else:
            wm, blocks = int(up_to), None
        reqs = self.encode(up_to=wm, since=since, blocks=blocks)
        write_segment(path, header, reqs, since=int(since), up_to=wm)


def write_segment(path: str, header: dict, reqs: Dict[str, ReqRuns], *, since: int, up_to: int,
                  seq: Optional[int] = None, final: Optional[bool] = None,
                  stamps: Optional[Sequence[Tuple[str, int, int]]] = None) -> None:
    """Serialize one segment atomically: header, one line per request, then completion stamps."""
    hdr = dict(header)
    hdr["index"] = "runs-v1"
    hdr["since"] = int(since)
    hdr["up_to"] = int(up_to)
    if seq is not None:
        hdr["seq"] = int(seq)
    if final is not None:
        hdr["final"] = bool(final)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__header__": hdr}) + "\n")
        for rid in sorted(reqs):
            r = reqs[rid]
            f.write(json.dumps({"r": r.req_id, "l": list(r.layers),
                                "n": r.n_rows,
                                "runs": [[b, s, c] for b, s, c in r.runs]}) + "\n")
        for rid, wm_r, n_total in sorted(stamps or ()):
            f.write(json.dumps({"r": rid, "c": 1, "w": int(wm_r),
                                "n_total": int(n_total)}) + "\n")
    os.replace(tmp, path)


def read_run_index(path: str, *, with_stamps: bool = False):
    """Read one segment as ``(header, {req_id: ReqRuns})``, plus stamps with ``with_stamps``."""
    header: Optional[dict] = None
    out: Dict[str, ReqRuns] = {}
    stamps: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0:
                header = obj["__header__"]
                continue
            if "runs" in obj:
                out[obj["r"]] = ReqRuns(req_id=obj["r"], layers=tuple(int(x) for x in obj["l"]),
                                        runs=[(int(b), int(s), int(c)) for b, s, c in obj["runs"]],
                                        n_rows=int(obj["n"]))
            elif obj.get("c"):
                stamps[obj["r"]] = {"w": int(obj["w"]), "n_total": int(obj["n_total"])}
            else:
                raise RunIndexError(
                    f"{path} line {i}: not a run record and not a completion stamp ({sorted(obj)})")
    if header is None:
        raise ValueError(f"{path}: no index header")
    return (header, out, stamps) if with_stamps else (header, out)


class ChainCursor:
    """Walks a segment chain in ``seq`` order, refusing any chain a gather would misread."""

    def __init__(self, *, expect_run_id: Optional[str] = None) -> None:
        self.seq = 0
        self.lo = 0
        self.rows: Dict[str, int] = {}
        self.complete: Dict[str, dict] = {}
        self.expect_run_id = None if expect_run_id is None else str(expect_run_id)
        self._run_id: Optional[str] = self.expect_run_id

    @property
    def run_id(self) -> Optional[str]:
        """The run this chain belongs to (expected, or latched from its first segment)."""
        return self._run_id

    def _check_run(self, path: str, hdr: dict) -> None:
        got = hdr.get(RUN_ID_KEY)
        if self._run_id is None:
            self._run_id = None if got is None else str(got)
            return
        if got is None:
            raise RunIndexError(
                f"{path}: this segment carries no run id, and this reader is reading run "
                f"{self._run_id!r}. A chain written before run ids existed, or by a launch that did "
                f"not stamp one, is exactly as stale as one stamped with another run's id -- it is "
                f"refused rather than assumed to be this run's. Wipe the aperture directory between "
                f"engine launches, or give each launch its own.")
        if str(got) != self._run_id:
            raise RunIndexError(
                f"{path}: this segment belongs to run {str(got)!r}, but this reader is reading run "
                f"{self._run_id!r}. A run index is a CHAIN OF FILES ON DISK and every launch "
                f"writes into the same aperture directory, so a previous launch's chain is "
                f"still there when the next one truncates the layer files -- reading it would name "
                f"rows that this run's files do not contain. Refusing rather than delivering a short "
                f"or zero-padded artifact.")

    def feed(self, path: str, hdr: dict, reqs: Dict[str, ReqRuns],
             stamps: Dict[str, dict]) -> List[Tuple[str, dict]]:
        """Admit one segment; return the stamps it newly completes, in id order."""
        self._check_run(path, hdr)
        if int(hdr.get("seq", -1)) != self.seq:
            raise RunIndexError(
                f"{path}: segment seq {hdr.get('seq')} read at position {self.seq} -- a segment is "
                f"MISSING, so some rows are named by no segment at all")
        if int(hdr["since"]) != self.lo:
            raise RunIndexError(
                f"{path}: window starts at {hdr['since']} but the previous segment ended at "
                f"{self.lo} -- the segment chain must be a contiguous, disjoint cover (a gap drops "
                f"rows, an overlap delivers them twice)")
        if int(hdr["up_to"]) < int(hdr["since"]):
            raise RunIndexError(f"{path}: window {hdr['since']}..{hdr['up_to']} runs backwards")
        up_to = int(hdr["up_to"])
        for rid in stamps:
            if rid in self.complete:
                raise RunIndexError(f"{path}: {rid!r} is stamped complete twice")
        rows = dict(self.rows)
        for rid, r in reqs.items():
            rows[rid] = rows.get(rid, 0) + r.n_rows
        out: List[Tuple[str, dict]] = []
        for rid in sorted(stamps):
            st = stamps[rid]
            if int(st["w"]) > up_to:
                raise RunIndexError(
                    f"{path}: {rid!r} is stamped complete at watermark {st['w']} but this segment "
                    f"only covers rows below {up_to} -- the stamp precedes its own rows")
            got = rows.get(rid, 0)
            if got != int(st["n_total"]):
                raise RunIndexError(
                    f"{path}: {rid!r} is stamped complete with {st['n_total']} rows but the segments "
                    f"read so far hold {got} -- the artifact would be SHORT, refusing")
            out.append((rid, st))
        self.rows = rows
        self.lo = up_to
        self.seq += 1
        for rid, st in out:
            self.complete[rid] = st
        return out


def read_run_segments(run_dir: str, *, expect_run_id: Optional[str] = None
                      ) -> Tuple[dict, Dict[str, ReqRuns], Dict[str, dict]]:
    """Read the whole segment chain as ``(header, {req_id: ReqRuns}, {req_id: stamp})``."""
    paths = segment_paths(run_dir)
    if not paths:
        raise RunIndexError(
            f"{run_dir}: no index segments ({SEGMENT_FMT % 0}...). Either the run was not captured "
            f"with {FLUSH_MS_ENV} set, or it wrote only the close-time {INDEX_NAME}.")
    header: Optional[dict] = None
    reqs: Dict[str, ReqRuns] = {}
    cur = ChainCursor(expect_run_id=expect_run_id)
    for p in paths:
        hdr, seg, stamps = read_run_index(p, with_stamps=True)
        cur.feed(p, hdr, seg, stamps)
        if header is None:
            header = hdr
        for rid, r in seg.items():
            prev = reqs.get(rid)
            if prev is None:
                reqs[rid] = r
            else:
                reqs[rid] = ReqRuns(req_id=rid, layers=prev.layers,
                                    runs=list(prev.runs) + list(r.runs),
                                    n_rows=prev.n_rows + r.n_rows)
    return header, reqs, cur.complete


class RunIndexFlusher:
    """Publishes a :class:`RunIndex` mid-run as a chain of atomically replaced segments."""

    def __init__(self, index: RunIndex, run_dir: str, header: dict, interval_ms: int,
                 *, name: str = "hs run index", run_id: Optional[str] = None):
        self._index = index
        self._run_dir = run_dir
        self._header = dict(header)
        self.run_id = None if run_id is None else str(run_id)
        if self.run_id is not None:
            self._header[RUN_ID_KEY] = self.run_id
        self._interval_s = max(0.001, float(interval_ms) / 1000.0)
        self._name = name
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._since = 0
        self._seq = 0
        self._block_lo = 0
        self._rows: Dict[str, int] = {}
        self._stopped = False
        self.flushes = 0
        self.skipped_stamps = 0
        self._log: List[dict] = []

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="mia-run-index-flush",
                                            daemon=True)
            self._thread.start()

    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def stop(self) -> None:
        """Join the cadence thread, then publish one final segment and close the chain."""
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=30.0)
            if t.is_alive():
                logger.error("%s flusher thread did not stop within 30 s; publishing the final "
                             "segment anyway", self._name)
        if not self._stopped:
            self._stopped = True
            try:
                self.flush_once(final=True)
            except Exception:  # noqa: BLE001
                logger.exception("%s: the final index segment could not be written", self._name)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.flush_once()
            except Exception:  # noqa: BLE001
                logger.exception("%s: an incremental index flush failed; the chain stops here",
                                 self._name)
                return

    def flush_once(self, *, up_to: Optional[int] = None, final: bool = False) -> Optional[str]:
        """Publish the next segment; return its path, or None when there was nothing to publish."""
        with self._lock:
            snap_wm, blocks = self._index.snapshot(self._block_lo)
            wm = snap_wm if up_to is None else min(int(up_to), snap_wm)
            if wm < self._since:
                raise RunIndexError(
                    f"{self._name}: a segment at watermark {wm} would go backwards from the chain's "
                    f"{self._since}")
            if wm == self._since and not final and not self._index.has_finished(wm):
                return None
            t0 = time.perf_counter()
            cpu0 = time.thread_time()
            reqs = self._index.encode(up_to=wm, since=self._since, blocks=blocks)
            enc_ms = (time.perf_counter() - t0) * 1e3
            for rid, r in reqs.items():
                self._rows[rid] = self._rows.get(rid, 0) + r.n_rows
            stamps: List[Tuple[str, int, int]] = []
            for rid, wm_r in self._index.take_finished(wm):
                n_total = self._rows.pop(rid, None)
                if self._index.pop_asked(rid) and n_total is None:
                    n_total = 0
                if n_total is None:
                    self.skipped_stamps += 1
                    continue
                stamps.append((rid, wm_r, n_total))
            if wm == self._since and not final and not stamps:
                return None
            path = os.path.join(self._run_dir, segment_name(self._seq))
            tw = time.perf_counter()
            write_segment(path, self._header, reqs, since=self._since, up_to=wm, seq=self._seq,
                          final=bool(final), stamps=stamps)
            write_ms = (time.perf_counter() - tw) * 1e3
            j = 0
            while j + 1 < len(blocks) and int(blocks[j + 1][1]) <= wm:
                j += 1
            self._block_lo += j
            self._since = wm
            self._seq += 1
            self.flushes += 1
            self._log.append({
                "seq": self._seq - 1, "since": int(self._since), "up_to": int(wm),
                "n_reqs": len(reqs), "n_rows": int(sum(r.n_rows for r in reqs.values())),
                "n_stamps": len(stamps), "n_blocks": len(blocks), "final": bool(final),
                "encode_ms": enc_ms, "write_ms": write_ms,
                "wall_ms": (time.perf_counter() - t0) * 1e3,
                "cpu_ms": (time.thread_time() - cpu0) * 1e3,
            })
            return path

    def stats(self) -> dict:
        return {"flushes": list(self._log), "n_flushes": self.flushes, "since": self._since,
                "seq": self._seq, "skipped_stamps": self.skipped_stamps,
                "unstamped_live": len(self._rows)}

    def summary(self) -> dict:
        """What every flush of this chain cost, as percentiles."""
        log = list(self._log)
        out = {"n": len(log), "seq": self._seq, "skipped_stamps": self.skipped_stamps,
               "unstamped_live": len(self._rows)}
        if not log:
            return out
        for k in ("wall_ms", "cpu_ms", "encode_ms", "write_ms"):
            v = sorted(float(x.get(k, 0.0)) for x in log)
            n = len(v)
            out[k] = {"p50": v[n // 2], "p99": v[min(n - 1, int(n * 0.99))], "max": v[-1],
                      "sum": sum(v)}
        rows = sorted(int(x.get("n_rows", 0)) for x in log)
        reqs = sorted(int(x.get("n_reqs", 0)) for x in log)
        out["rows_per_flush"] = {"p50": rows[len(rows) // 2], "max": rows[-1], "sum": sum(rows)}
        out["reqs_per_flush"] = {"p50": reqs[len(reqs) // 2], "max": reqs[-1]}
        return out

    def summary_line(self, run_id=None) -> str:
        """:meth:`summary` as one log line."""
        d = self.summary()
        if not d.get("n"):
            return (f"[aperture-index] run {run_id or '?'} flusher: 0 flushes published "
                    f"(nothing to summarise)")
        w, c = d["wall_ms"], d["cpu_ms"]
        return (f"[aperture-index] run {run_id or '?'} flusher: {d['n']} flush(es), "
                f"wall p50/p99/max {w['p50']:.3f}/{w['p99']:.3f}/{w['max']:.3f} ms, "
                f"GIL-HELD (cpu) p50/p99/max {c['p50']:.3f}/{c['p99']:.3f}/{c['max']:.3f} ms, "
                f"encode p99 {d['encode_ms']['p99']:.3f}, write p99 {d['write_ms']['p99']:.3f}, "
                f"rows/flush p50/max {d['rows_per_flush']['p50']}/{d['rows_per_flush']['max']}, "
                f"total GIL-held {c['sum'] / 1e3:.2f} s of wall {w['sum'] / 1e3:.2f} s "
                f"-- wall includes GIL WAIT, cpu is what it takes FROM the engine")


def run_slices(runs: Sequence[Tuple[int, int, int]]):
    """``(start, stop, step)`` memmap slices, one per run."""
    for base, stride, n in runs:
        st = int(stride) if int(stride) > 0 else 1
        yield int(base), int(base) + st * int(n), st
