"""Per-step capture sidecar mapping aperture rows back to (req_id, layer, tokens)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class LayerEntry:
    """One (req_id, layer) row-block."""
    req_id: str
    layer: int
    logical_start: int
    n_rows: int
    hs_mode: str
    file_row: int = -1

    def __post_init__(self):
        if self.file_row < 0:
            self.file_row = self.logical_start


@dataclass
class StepMeta:
    entries: list = field(default_factory=list)


@dataclass(slots=True)
class ReqCaptureRecord:
    """One capturing request's per-step HS capture footprint."""
    req_id: str
    logical_start: int
    n_rows: int
    hs_mode: str
    layers: list


def expand_records(records) -> list:
    """Expand per-request records into the flat per-(req, layer) LayerEntry list, in record order."""
    out: list = []
    for rec in records:
        layers = getattr(rec, "layers", None)
        if layers is None:
            out.append(rec)
            continue
        for layer in layers:
            out.append(LayerEntry(rec.req_id, layer, rec.logical_start, rec.n_rows, rec.hs_mode))
    return out


@dataclass
class QKStepEntry:
    req_id: str
    layer: int
    k_start: int
    k_rows: int
    q_start: int
    q_rows: int
    prefix_end: int
    num_computed: int


@dataclass(slots=True)
class QKReqCaptureRecord:
    """One capturing request's per-step QK capture footprint."""
    req_id: str
    k_start: int
    k_rows: int
    q_start: int
    q_rows: int
    prefix_end: int
    num_computed: int
    layers: list


def expand_qk_records(records) -> list:
    """Expand per-request QK records into the flat per-(req, layer) QKStepEntry list."""
    out: list = []
    for rec in records:
        layers = getattr(rec, "layers", None)
        if layers is None:
            out.append(rec)
            continue
        for layer in layers:
            out.append(QKStepEntry(
                req_id=rec.req_id, layer=layer,
                k_start=rec.k_start, k_rows=rec.k_rows,
                q_start=rec.q_start, q_rows=rec.q_rows,
                prefix_end=rec.prefix_end, num_computed=rec.num_computed))
    return out


def write_qk_sidecar(path: str, steps: list, header: dict) -> None:
    """Write the QK sidecar header plus one JSON line per QKStepEntry."""
    header = _normalize_json_native(header)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__header__": header}) + "\n")
        for step_idx, step in enumerate(steps):
            for e in step.entries:
                f.write(json.dumps({
                    "s": step_idx, "r": e.req_id, "l": e.layer,
                    "ks": e.k_start, "kn": e.k_rows,
                    "qs": e.q_start, "qn": e.q_rows,
                    "pe": e.prefix_end, "nc": e.num_computed,
                }) + "\n")


def read_qk_sidecar(path: str):
    """Read a QK sidecar into (header, ordered QKStepEntry list)."""
    header = None
    entries: list = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if header is None and "__header__" in obj:
                header = obj["__header__"]
                continue
            entries.append(QKStepEntry(
                req_id=obj["r"], layer=obj["l"],
                k_start=obj["ks"], k_rows=obj["kn"],
                q_start=obj["qs"], q_rows=obj["qn"],
                prefix_end=obj["pe"], num_computed=obj["nc"]))
    return header, entries


def _normalize_json_native(value):
    if isinstance(value, tuple):
        return [_normalize_json_native(v) for v in value]
    if isinstance(value, list):
        return [_normalize_json_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_json_native(v) for k, v in value.items()}
    return value


def write_sidecar(path: str, steps: list, header: dict) -> None:
    """Write the header + one JSON line per `LayerEntry` across `steps`."""
    header = _normalize_json_native(header)
    with open(path, "w") as f:
        f.write(json.dumps({"__header__": header}) + "\n")
        for step_idx, step in enumerate(steps):
            for e in step.entries:
                row = {
                    "s": step_idx,
                    "r": e.req_id,
                    "l": e.layer,
                    "o": e.logical_start,
                    "n": e.n_rows,
                    "m": e.hs_mode,
                    "fr": e.file_row,
                }
                f.write(json.dumps(row) + "\n")


class _NotPlain(Exception):
    """A record the array fast path cannot represent exactly."""


class _Interner:
    """value -> small int, remembering ``json.dumps(value)``."""
    __slots__ = ("_idx", "json")

    def __init__(self):
        self._idx: dict = {}
        self.json: list = []

    def __call__(self, v) -> int:
        key = (type(v), v)
        try:
            i = self._idx.get(key)
        except TypeError:
            raise _NotPlain(f"unhashable value {v!r}") from None
        if i is None:
            i = self._idx[key] = len(self.json)
            self.json.append(json.dumps(v))
        return i


class _LayerLists:
    """Interns a record's layers list, remembering each layer's JSON spelling and position."""
    __slots__ = ("_idx", "lists", "json", "lens")

    def __init__(self):
        self._idx: dict = {}
        self.lists: list = []
        self.json: list = []
        self.lens: list = []

    def __call__(self, layers) -> int:
        t = tuple(layers)
        i = self._idx.get(t)
        if i is None:
            if not all(type(x) is int for x in t):
                raise _NotPlain(f"non-int layer in {t!r}")
            i = self._idx[t] = len(self.lists)
            self.lists.append(t)
            self.json.append([str(x) for x in t])
            self.lens.append(len(t))
        return i


def _plain_rows(fields: list, n_cols: int):
    import numpy as np
    try:
        arr = np.array(fields)
    except (ValueError, TypeError, OverflowError):
        raise _NotPlain("ragged or non-numeric record fields") from None
    if arr.ndim != 2 or arr.shape[1] != n_cols or arr.dtype.kind not in "iu":
        raise _NotPlain(f"record fields are not plain ints (dtype {arr.dtype})")
    return arr.astype(np.int64, copy=False)


class HsSidecarLog:
    """Per-step HS sidecar records for ``hs_aperture_meta.jsonl``, kept as arrays."""

    def __init__(self, layer_nums, legacy_stamp):
        self._lidx = {int(ln): i for i, ln in enumerate(layer_nums)}
        self._layer_nums = [int(ln) for ln in layer_nums]
        self._stamp = legacy_stamp
        self._rid = _Interner()
        self._mode = _Interner()
        self._lay = _LayerLists()
        self._lay_pos: list = []
        self.blocks: list = []
        self.n_entries = 0

    def has_entries(self) -> bool:
        return self.n_entries > 0

    def _cursor_dict(self, cursor) -> dict:
        if isinstance(cursor, int):
            return {ln: cursor for ln in self._layer_nums}
        return {ln: int(cursor[i]) for i, ln in enumerate(self._layer_nums) if int(cursor[i]) >= 0}

    def prepare(self, records, step_start: int, cursor, plans=None):
        """One step's block, or None when it expands to no entries."""
        if not records:
            return None
        try:
            rid, mode, lay = self._rid, self._mode, self._lay
            rows = [(rid(r.req_id), r.logical_start, r.n_rows, mode(r.hs_mode), lay(r.layers))
                    for r in records]
            arr = _plain_rows(rows, 5)
        except (_NotPlain, AttributeError):
            entries = expand_records(records)
            if not entries:
                return None
            self._stamp(entries, self._cursor_dict(cursor), int(step_start), plans)
            return ("entries", entries, len(entries))
        lens = self._lay.lens
        n = sum(lens[i] for i in arr[:, 4].tolist())
        if n == 0:
            return None
        if not isinstance(cursor, int):
            import numpy as np
            cursor = np.array(cursor, dtype=np.int64, copy=True)
        return ("rows", int(step_start), arr, cursor, plans, n)

    def commit(self, block) -> None:
        if block is None:
            return
        self.blocks.append(block)
        self.n_entries += block[-1]

    def _positions(self, li: int):
        while len(self._lay_pos) <= li:
            t = self._lay.lists[len(self._lay_pos)]
            pos = [self._lidx.get(x, -1) for x in t]
            self._lay_pos.append((pos, -1 not in pos))
        return self._lay_pos[li]

    def _lines(self, block, s: int) -> list:
        if block[0] == "entries":
            return [json.dumps({"s": s, "r": e.req_id, "l": e.layer, "o": e.logical_start,
                                "n": e.n_rows, "m": e.hs_mode, "fr": e.file_row}) + "\n"
                    for e in block[1]]
        _, step_start, arr, cur, plans, _n = block
        rid_json, mode_json = self._rid.json, self._mode.json
        lay_json, lay_lists = self._lay.json, self._lay.lists
        uniform = isinstance(cur, int)
        cur_l = None if uniform else cur.tolist()
        out: list = []
        for ri, o, n, mi, li in arr.tolist():
            pre = '{"s": %d, "r": %s, "l": ' % (s, rid_json[ri])
            mid = ', "o": %d, "n": %d, "m": %s, "fr": ' % (o, n, mode_json[mi])
            ljs = lay_json[li]
            off = o - step_start
            pos, all_installed = self._positions(li)
            if plans is None or n <= 0:
                if uniform and all_installed:
                    tail = mid + "%d}\n" % (cur + off)
                    out.extend([pre + lj + tail for lj in ljs])
                    continue
                for lj, p in zip(ljs, pos):
                    base = 0 if p < 0 else (cur if uniform else max(cur_l[p], 0))
                    out.append(pre + lj + mid + "%d}\n" % (base + off))
                continue
            for lj, p, l in zip(ljs, pos, lay_lists[li]):
                base = 0 if p < 0 else (cur if uniform else max(cur_l[p], 0))
                plan = plans.get(l)
                fr = base + (off if plan is None else plan.row_offset(o))
                out.append(pre + lj + mid + "%d}\n" % fr)
        return out

    def write(self, path: str, header: dict) -> None:
        """Write the sidecar, byte-identical to ``write_sidecar``."""
        header = _normalize_json_native(header)
        with open(path, "w") as f:
            f.write(json.dumps({"__header__": header}) + "\n")
            for s, block in enumerate(list(self.blocks)):
                f.write("".join(self._lines(block, s)))


class QkSidecarLog:
    """Per-step QK sidecar records for ``qk_aperture_meta.jsonl``, kept as arrays."""

    def __init__(self):
        self._rid = _Interner()
        self._lay = _LayerLists()
        self.blocks: list = []
        self.n_entries = 0

    def has_entries(self) -> bool:
        return self.n_entries > 0

    def prepare(self, records):
        if not records:
            return None
        try:
            rid, lay = self._rid, self._lay
            rows = [(rid(r.req_id), r.k_start, r.k_rows, r.q_start, r.q_rows, r.prefix_end,
                     r.num_computed, lay(r.layers)) for r in records]
            arr = _plain_rows(rows, 8)
        except (_NotPlain, AttributeError):
            entries = expand_qk_records(records)
            return ("entries", entries, len(entries)) if entries else None
        lens = self._lay.lens
        n = sum(lens[i] for i in arr[:, 7].tolist())
        return ("rows", arr, n) if n else None

    def commit(self, block) -> None:
        if block is None:
            return
        self.blocks.append(block)
        self.n_entries += block[-1]

    def _lines(self, block, s: int) -> list:
        if block[0] == "entries":
            return [json.dumps({"s": s, "r": e.req_id, "l": e.layer, "ks": e.k_start,
                                "kn": e.k_rows, "qs": e.q_start, "qn": e.q_rows,
                                "pe": e.prefix_end, "nc": e.num_computed}) + "\n"
                    for e in block[1]]
        arr = block[1]
        rid_json, lay_json = self._rid.json, self._lay.json
        out: list = []
        for ri, ks, kn, qs, qn, pe, nc, li in arr.tolist():
            pre = '{"s": %d, "r": %s, "l": ' % (s, rid_json[ri])
            tail = (', "ks": %d, "kn": %d, "qs": %d, "qn": %d, "pe": %d, "nc": %d}\n'
                    % (ks, kn, qs, qn, pe, nc))
            out.extend([pre + lj + tail for lj in lay_json[li]])
        return out

    def write(self, path: str, header: dict) -> None:
        """Write the sidecar, byte-identical to ``write_qk_sidecar``."""
        header = _normalize_json_native(header)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"__header__": header}) + "\n")
            for s, block in enumerate(list(self.blocks)):
                f.write("".join(self._lines(block, s)))


def read_sidecar(path: str):
    """Read a sidecar written by `write_sidecar` and return `(header, steps)`."""
    header = None
    steps_by_idx = {}
    order = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0:
                header = obj["__header__"]
                continue
            s = obj["s"]
            if s not in steps_by_idx:
                steps_by_idx[s] = StepMeta([])
                order.append(s)
            steps_by_idx[s].entries.append(
                LayerEntry(obj["r"], obj["l"], obj["o"], obj["n"], obj["m"],
                           file_row=obj.get("fr", obj["o"]))
            )
    steps = [steps_by_idx[s] for s in order]
    return header, steps

