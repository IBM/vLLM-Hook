"""Reclaims the shared layer files behind the gather cursor (``MIA_APERTURE_GATHER_TRIM``)."""
from __future__ import annotations

import ctypes
import errno
import glob
import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

TRIM_ENV = "MIA_APERTURE_GATHER_TRIM"
CHUNK_ENV = "MIA_APERTURE_GATHER_TRIM_CHUNK_BYTES"
LAG_ENV = "MIA_APERTURE_GATHER_TRIM_LAG_BYTES"
ALIGN_ENV = "MIA_APERTURE_GATHER_TRIM_ALIGN"

MARKER_FMT = "hs_trim.w%02dof%02d.json"
MARKER_GLOB = "hs_trim.w*of*.json"
MARKER_FORMAT = "hs-trim-floor-v1"

FALLOC_FL_KEEP_SIZE = 0x01
FALLOC_FL_PUNCH_HOLE = 0x02

DEFAULT_CHUNK_BYTES = 64 << 20

DEFAULT_LAG_BYTES = 64 << 20

MIN_ALIGN = 4096


class TrimError(RuntimeError):
    """The trim is misconfigured, unsupported by the filesystem, or a trim marker is damaged."""


class TrimmedRegionError(RuntimeError):
    """A reader asked for rows the trim has already reclaimed from the shared layer files."""

    def __init__(self, message: str, *, run_dir: str = "", layer: Optional[int] = None,
                 first_row: Optional[int] = None, floor_row: Optional[int] = None):
        super().__init__(message)
        self.run_dir = run_dir
        self.layer = layer
        self.first_row = first_row
        self.floor_row = floor_row


def trim_explicit() -> bool:
    """Whether ``MIA_APERTURE_GATHER_TRIM`` was set, as opposed to defaulted."""
    return os.environ.get(TRIM_ENV) not in (None, "")


def trim_enabled() -> bool:
    """``MIA_APERTURE_GATHER_TRIM``: default on; ``"1"`` and ``"0"`` are the only legal values."""
    raw = os.environ.get(TRIM_ENV)
    if raw is None or raw.strip() == "":
        return True
    v = raw.strip()
    if v not in ("0", "1"):
        raise TrimError(
            f"{TRIM_ENV}={raw!r} is neither '1' nor '0'. This flag decides whether the hybrid frees "
            f"the shared layer files it has already delivered: read the wrong way it either keeps a "
            f"SECOND FULL COPY of every captured byte or DISCARDS rows a reader wanted, and neither "
            f"is visible from outside the run. So a spelling that is not exactly '1' or '0' is "
            f"refused rather than guessed. It is ON by default; set {TRIM_ENV}=0 to keep the shared "
            f"layer files whole.")
    return v == "1"


def _positive_int(env: str, default: int, *, allow_zero: bool = False) -> int:
    raw = os.environ.get(env)
    if raw is None or raw.strip() == "":
        return default
    try:
        n = int(raw.strip())
    except ValueError:
        raise TrimError(
            f"{env}={raw!r} is not a whole number of bytes. Unset it for the default "
            f"({default}).") from None
    if n < 0 or (n == 0 and not allow_zero):
        raise TrimError(f"{env}={raw!r} must be "
                        f"{'non-negative' if allow_zero else 'positive'}")
    return n


def trim_chunk_bytes() -> int:
    """``MIA_APERTURE_GATHER_TRIM_CHUNK_BYTES``, see :data:`DEFAULT_CHUNK_BYTES`."""
    return _positive_int(CHUNK_ENV, DEFAULT_CHUNK_BYTES)


def trim_lag_bytes() -> int:
    """``MIA_APERTURE_GATHER_TRIM_LAG_BYTES``, see :data:`DEFAULT_LAG_BYTES`."""
    return _positive_int(LAG_ENV, DEFAULT_LAG_BYTES, allow_zero=True)


def trim_align(path: str) -> int:
    """Punch alignment: ``MIA_APERTURE_GATHER_TRIM_ALIGN``, else the filesystem block size."""
    raw = os.environ.get(ALIGN_ENV)
    if raw is not None and raw.strip() != "":
        try:
            n = int(raw.strip())
        except ValueError:
            raise TrimError(f"{ALIGN_ENV}={raw!r} is not a whole number of bytes") from None
        if n < MIN_ALIGN or n % MIN_ALIGN != 0:
            raise TrimError(
                f"{ALIGN_ENV}={raw!r} must be a multiple of {MIN_ALIGN}: fallocate frees only whole "
                f"filesystem blocks fully inside the punched region, so a smaller or ragged "
                f"alignment silently frees less than the offsets say it did")
        return n
    try:
        bs = int(os.statvfs(path).f_bsize)
    except OSError:
        bs = MIN_ALIGN
    if bs < MIN_ALIGN:
        bs = MIN_ALIGN
    if bs % MIN_ALIGN:
        bs = ((bs + MIN_ALIGN - 1) // MIN_ALIGN) * MIN_ALIGN
    return bs


def align_down(n: int, align: int) -> int:
    """``n`` rounded down to a multiple of ``align``."""
    a = int(align)
    if a <= 0:
        raise TrimError(f"alignment {align!r} must be positive")
    return (int(n) // a) * a


def floor_row_for_bytes(punched_bytes: int, row_bytes: int) -> int:
    """The first row a reader may still trust, given ``[0, punched_bytes)`` is a hole."""
    rb = int(row_bytes)
    if rb <= 0:
        raise TrimError(f"row_bytes {row_bytes!r} must be positive")
    return -(-int(punched_bytes) // rb)


_libc: Optional[ctypes.CDLL] = None


def _fallocate():
    global _libc
    if _libc is None:
        lib = ctypes.CDLL(None, use_errno=True)
        try:
            fn = lib.fallocate
        except AttributeError:
            raise TrimError(
                "this libc exposes no fallocate(2), so a hole cannot be punched. Unset "
                f"{TRIM_ENV}.") from None
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
        fn.restype = ctypes.c_int
        _libc = lib
    return _libc.fallocate


def punch_hole(fd: int, offset: int, length: int) -> None:
    """``fallocate(fd, PUNCH_HOLE | KEEP_SIZE, offset, length)``."""
    off, ln = int(offset), int(length)
    if ln <= 0:
        return
    if off < 0:
        raise TrimError(f"punch offset {off} is negative")
    ctypes.set_errno(0)
    rc = _fallocate()(int(fd), FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE, off, ln)
    if rc != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"fallocate(PUNCH_HOLE|KEEP_SIZE, {off}, {ln}): {os.strerror(e)}")


def punch_supported(directory: str) -> Tuple[bool, str]:
    """Whether files in ``directory`` can be hole-punched."""
    try:
        fd, path = tempfile.mkstemp(dir=directory, prefix=".mia-trim-probe-")
    except OSError as e:
        return False, f"cannot create a probe file in {directory}: {e}"
    try:
        os.write(fd, b"\0" * (MIN_ALIGN * 2))
        try:
            punch_hole(fd, 0, MIN_ALIGN)
        except OSError as e:
            if e.errno in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
                return False, (f"fallocate(PUNCH_HOLE) on {directory} failed with "
                               f"{errno.errorcode.get(e.errno, e.errno)}: {e.strerror or e}")
            return False, f"fallocate(PUNCH_HOLE) on {directory} failed: {e}"
        except TrimError as e:
            return False, str(e)
        if os.fstat(fd).st_size != MIN_ALIGN * 2:
            return False, f"{directory}: a PUNCH_HOLE|KEEP_SIZE changed the file's logical size"
        return True, f"fallocate(PUNCH_HOLE|KEEP_SIZE) works in {directory}"
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


def blocks_bytes(path: str) -> int:
    """Bytes the filesystem has allocated to ``path`` (``st_blocks`` x 512)."""
    return int(os.stat(path).st_blocks) * 512


def first_hole(path: str) -> Optional[int]:
    """The offset of the first hole strictly inside ``path``, or None."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        size = os.fstat(fd).st_size
        if size <= 0:
            return None
        try:
            off = os.lseek(fd, 0, os.SEEK_HOLE)
        except (OSError, AttributeError, ValueError):
            return None
        return None if int(off) >= int(size) else int(off)
    finally:
        os.close(fd)


def marker_path(run_dir: str, worker: int, n_workers: int) -> str:
    return os.path.join(run_dir, MARKER_FMT % (int(worker), int(n_workers)))


def marker_paths(run_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(run_dir, MARKER_GLOB)))


class TrimLog:
    """One gather worker's trim marker: the rows it has punched, or is about to."""

    def __init__(self, run_dir: str, worker: int, n_workers: int):
        self.run_dir = str(run_dir)
        self.worker = int(worker)
        self.n_workers = int(n_workers)
        self.path = marker_path(self.run_dir, self.worker, self.n_workers)
        self._floor: Dict[int, int] = {}
        self._at_row = 0

    @property
    def floor_rows(self) -> Dict[int, int]:
        return dict(self._floor)

    @property
    def at_row(self) -> int:
        return self._at_row

    def publish(self, floor_rows: Dict[int, int], *, row_bytes: int,
                at_row: Optional[int] = None) -> Dict[int, int]:
        """Record each layer's floor and this worker's consumed watermark; return the merged floor."""
        changed = False
        for layer, row in floor_rows.items():
            L, r = int(layer), int(row)
            if r > self._floor.get(L, 0):
                self._floor[L] = r
                changed = True
        if at_row is not None and int(at_row) > self._at_row:
            self._at_row = int(at_row)
            changed = True
        if not changed:
            return dict(self._floor)
        body = {"format": MARKER_FORMAT, "worker": self.worker, "n_workers": self.n_workers,
                "row_bytes": int(row_bytes), "at_row": int(self._at_row),
                "floor_rows": {str(k): int(v) for k, v in sorted(self._floor.items())},
                "why": (f"these rows of the shared hs_layer_*.raw files were freed with "
                        f"fallocate(PUNCH_HOLE) after the gather delivered them ({TRIM_ENV}=1). "
                        f"The files are still their full LENGTH and the region reads back as "
                        f"ZEROS, so the shared files are no longer a readable artifact for these "
                        f"rows -- the per-request delivery under 'delivered/' is.")}
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(body, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        return dict(self._floor)


def published_cursors(run_dir: str, n_workers: int) -> Dict[int, int]:
    """``{worker: consumed row watermark}`` from this partition's markers."""
    out: Dict[int, int] = {}
    for p in marker_paths(run_dir):
        try:
            with open(p, "r", encoding="utf-8") as f:
                body = json.loads(f.read())
            if int(body.get("n_workers", -1)) != int(n_workers):
                continue
            w = int(body["worker"])
            if not (0 <= w < int(n_workers)):
                continue
            out[w] = int(body.get("at_row") or 0)
        except Exception:  # noqa: BLE001
            continue
    return out


def trimmed_floor_rows(run_dir: str) -> Dict[int, int]:
    """``{layer: first row no longer readable}`` over every worker's marker in ``run_dir``."""
    out: Dict[int, int] = {}
    for p in marker_paths(run_dir):
        try:
            with open(p, "r", encoding="utf-8") as f:
                body = json.loads(f.read())
            floors = body["floor_rows"]
            items = {int(k): int(v) for k, v in floors.items()}
        except Exception as e:  # noqa: BLE001
            raise TrimError(
                f"{p}: a trim marker is present but cannot be read ({e!r}). Its presence means "
                f"rows of the shared layer files were PUNCHED, and a punched region reads back as "
                f"zeros rather than failing -- so without a floor to compare against, NO row of "
                f"this run can be trusted from the shared files. This is the instrument being "
                f"broken, not the run being trimmed; read the per-request delivery under "
                f"'delivered/' instead.") from None
        for L, r in items.items():
            if r > out.get(L, 0):
                out[L] = r
    return out


def is_reclaimed(layer: int, first_row: int, floors: Dict[int, int]) -> bool:
    """Whether row ``first_row`` of ``layer`` is inside a punched region."""
    floor = int(floors.get(int(layer), 0))
    return floor > 0 and int(first_row) < floor


def refuse_trimmed_rows(run_dir: str, layer: int, first_row: int,
                        floors: Optional[Dict[int, int]] = None) -> None:
    """Raise :class:`TrimmedRegionError` if row ``first_row`` of ``layer`` was reclaimed."""
    f = trimmed_floor_rows(run_dir) if floors is None else floors
    if not is_reclaimed(layer, first_row, f):
        return
    floor = int(f.get(int(layer), 0))
    raise TrimmedRegionError(
        f"{run_dir}: row {int(first_row)} of layer {int(layer)} was RECLAIMED -- it is below this "
        f"layer's trim floor of {floor} rows. The hybrid gather freed those rows with "
        f"fallocate(PUNCH_HOLE) after delivering them ({TRIM_ENV} is ON by default). The file is "
        f"still its full length and the region reads back as ZEROS, so returning it would hand back "
        f"a plausible, wrong tensor; rows at or above {floor} are unaffected and byte-identical. "
        f"This request's data is its own artifact under '{run_dir}/delivered'. Pass "
        f"skip_trimmed=True to read what remains, or re-run with {TRIM_ENV}=0 to keep the shared "
        f"layer files whole.",
        run_dir=str(run_dir), layer=int(layer), first_row=int(first_row), floor_row=floor)


def trim_status(run_dir: str) -> Dict[str, object]:
    """What a trim took from ``run_dir``, as plain data."""
    floors = trimmed_floor_rows(run_dir)
    row_bytes = 0
    for p in marker_paths(run_dir):
        try:
            with open(p, "r", encoding="utf-8") as f:
                row_bytes = max(row_bytes, int(json.loads(f.read()).get("row_bytes") or 0))
        except Exception:  # noqa: BLE001
            pass
    holes = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "hs_layer_*.raw"))):
        m = re.search(r"hs_layer_(\d+)\.raw$", p)
        if m:
            holes[int(m.group(1))] = first_hole(p)
    return {"trimmed": bool(floors), "floor_rows": dict(floors), "row_bytes": row_bytes,
            "floor_bytes": {L: r * row_bytes for L, r in floors.items()} if row_bytes else {},
            "markers": [os.path.basename(p) for p in marker_paths(run_dir)], "holes": holes}


__all__ = ["ALIGN_ENV", "CHUNK_ENV", "DEFAULT_CHUNK_BYTES", "DEFAULT_LAG_BYTES",
           "FALLOC_FL_KEEP_SIZE", "FALLOC_FL_PUNCH_HOLE", "LAG_ENV", "MARKER_FORMAT",
           "MARKER_GLOB", "MIN_ALIGN", "TRIM_ENV", "TrimError", "TrimLog", "TrimmedRegionError",
           "align_down", "blocks_bytes", "first_hole", "floor_row_for_bytes", "is_reclaimed",
           "marker_path", "marker_paths", "published_cursors", "punch_hole", "punch_supported",
           "refuse_trimmed_rows",
           "trim_align", "trim_chunk_bytes", "trim_enabled", "trim_explicit", "trim_lag_bytes",
           "trim_status", "trimmed_floor_rows"]
