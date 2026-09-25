"""Per-step capture metadata sidecar: maps aperture row-ranges back to (req_id, layer, tokens) so the
read-side reconstruction and the drain can rebuild each request's tensors. Pure host-side JSON-lines
format — small, no torch dependency. Line 0 is the header (`{"__header__": ...}`); every subsequent
line is one `LayerEntry`, tagged with its originating step index so `read_sidecar` can regroup them
into the same `StepMeta` ordering that `write_sidecar` was given.

Contract for downstream consumers (HS shared-file path; see `LayerEntry` for the two-key split):
the entries recovered by `read_sidecar` form a FLAT, ORDERED list. Reconstruction must key on each
entry's `file_row` — the row offset into THAT LAYER's own raw file — never on `logical_start` (the
aperture-wide reservation position, which only coincides with `file_row` while every layer receives every
row) nor on list position or step count. The `StepMeta` grouping is organizational only — step index
is NOT preserved (a step with zero entries is elided entirely from the returned list, so list position
never maps back to an absolute step). `read_sidecar` always populates `file_row`: it reads the "fr" key
when present, and falls back to "o" (`logical_start`) for a sidecar written before this field existed
— exactly correct there, since every pre-existing artifact is all-layers/all-rows."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class LayerEntry:
    """One (req_id, layer) row-block. Carries TWO distinct row offsets that coincide today but are
    NOT the same thing:

      * `logical_start` — the row's position in the SHARED aperture's global logical cursor (the aperture-wide
        reservation the request was given). Aperture-scoped, not layer-scoped.
      * `file_row`       — the row's offset within THIS LAYER's own raw file on disk. File-scoped:
        this is what `aperture_reader` must index with.

    They are equal only when every drained layer receives every row of every step (a full drain
    appends the full step range to every installed layer's file, unconditionally). Selective drain
    breaks that coincidence by skipping a layer's append on a step where no live request wants it, so
    `file_row` stops tracking `logical_start` one-for-one.

    `file_row` defaults to `logical_start` (`__post_init__`) so a caller that never learns the real
    per-layer file cursor (a direct-construction test, or the per-request disk-staging path, which
    already relabels `logical_start` to a per-request-local file offset by construction — see
    `_PerRequestDiskStaging`) still gets the historically-correct value. The shared-file drain
    (`MultiLayerApertureDrain` / `OffLoopApertureDrain`) OVERWRITES this default with the real tracked
    per-layer cursor value when it appends each step's rows — "stamped when the rows are appended,
    not when the entry is created"."""
    req_id: str
    layer: int
    logical_start: int
    n_rows: int
    hs_mode: str
    file_row: int = -1   # sentinel "unstamped"; __post_init__ resolves it to logical_start

    def __post_init__(self):
        if self.file_row < 0:
            self.file_row = self.logical_start


@dataclass
class StepMeta:
    entries: list = field(default_factory=list)


@dataclass(slots=True)
class ReqCaptureRecord:
    """ONE capturing request's per-step HS capture footprint — the "LayerEntry collapse" (HS only).

    The on-loop routing build (`_build_routing_hs`) used to fan out ONE `LayerEntry` per
    (request x captured layer) directly on the engine loop — an O(reqs x layers) dataclass
    allocation. This record collapses that to ONE object per request: `layers` holds this request's
    OWN 1-based artifact-layer list (`[L+1 for L in rows_layers]`, in the exact order the fan-out
    used), so heterogeneous per-request layer sets are carried verbatim (request A can hold
    `[1,2,3]`, request B `[5,10]` in the same step). `expand_records` fans it back out into the
    identical flat `LayerEntry` list OFF the loop (in the consumer thread / the sync drain), so the
    sidecar bytes, reader, and demux are byte-for-byte unchanged — only WHERE the `LayerEntry`
    objects are built moves off the engine loop."""
    req_id: str
    logical_start: int
    n_rows: int
    hs_mode: str
    layers: list          # this request's OWN 1-based artifact layers, in fan-out order


def expand_records(records) -> list:
    """Expand per-request `ReqCaptureRecord`s into the FLAT per-(req, layer) `LayerEntry` list the
    sidecar / demux consume — record order, then each record's `layers` order (identical to the old
    on-loop `for L in rows_layers` fan-out). This is the OFF-LOOP counterpart of that fan-out: moving
    it here takes the O(reqs x layers) `LayerEntry` construction off the engine loop while keeping
    every downstream byte identical.

    An element that is NOT a record (has no `layers` attribute) — an already-expanded `LayerEntry`,
    the shape a direct-drain caller / unit test may hand in — passes through UNCHANGED. So a caller
    holding either records or flat entries is handled; production always holds records (the collapse
    is the sole builder of `_hs_step_entries`)."""
    out: list = []
    for rec in records:
        layers = getattr(rec, "layers", None)
        if layers is None:
            out.append(rec)          # already a LayerEntry (or pre-expanded) -> pass through
            continue
        for layer in layers:
            # file_row is left at its __post_init__ default (== logical_start) here — this is
            # "entry creation" time, before any layer's file has actually been appended to. The
            # shared-file drain (MultiLayerApertureDrain.drain_once / OffLoopApertureDrain._drain_item)
            # overwrites it with the real per-layer file cursor right before these entries are
            # pushed into the sidecar's StepMeta list.
            out.append(LayerEntry(rec.req_id, layer, rec.logical_start, rec.n_rows, rec.hs_mode))
    return out


# --------------------------------------------------------------------------- #
# QK capture-aperture metadata (additive; HS format above is untouched).
#
# QK is TWO parallel per-layer apertures (q + k) sharing ONE logical cursor: every step scatters this
# step's q AND k to the SAME aperture slots. K is recorded EVERY step (so the concatenated k-file rows
# are the full key history `k_full`); q is recorded only on `emit_q` steps (all_tokens: every step;
# last_token: the final prefill chunk + each decode step). One QKStepEntry is emitted per
# (req, layer, step). The reader groups by (req, layer), sorts by `k_start` (== logical order, the
# monotonic cursor), and rebuilds:
#   k_full        = cat(k_file[e.k_start : e.k_start+e.k_rows]  for e in entries)
#   q             = cat(q_file[e.q_start : e.q_start+e.q_rows]  for e in entries if e.q_rows > 0)
#   k_prefix_ends = [e.prefix_end for e in entries if e.prefix_end >= 0]
#   k_all         = [k_full[:L] for L in k_prefix_ends]   (the COMPACT_KALL reconstruction)
# `num_computed` is the request's cached-prefix length at its FIRST capture step (0 for a fresh
# prefill). v1 does NOT reconstruct a trimmed prefix from paged KV, so the reader FAILS LOUD when
# the first-step `num_computed > 0` rather than returning a short k_all.
# --------------------------------------------------------------------------- #
@dataclass
class QKStepEntry:
    req_id: str
    layer: int            # 0-based (== the eager qkv_hook's match_attn layer_num)
    k_start: int          # aperture slot where this step's k rows begin (shared cursor)
    k_rows: int           # this step's forwarded key count (qlen); ALWAYS recorded
    q_start: int          # aperture slot of the emitted q rows; -1 when this step is not emit_q
    q_rows: int           # emitted q row count (0 when not emit_q)
    prefix_end: int       # abs_end paired with the emitted q (-1 when not emit_q)
    num_computed: int     # this step's pre-step processed count (cached prefix at the first step)


@dataclass(slots=True)
class QKReqCaptureRecord:
    """ONE capturing request's per-step QK capture footprint — the "LayerEntry collapse" (QK path).

    The on-loop routing build (`_build_routing`) used to fan out ONE `QKStepEntry` per
    (request x captured layer) directly on the engine loop — an O(reqs x layers) dataclass
    allocation, and `QKStepEntry` carries EIGHT fields, so the fan-out is heavier than the HS one.
    Every field except `layer` is computed ONCE per request (shared across all of that request's
    layers): `k_start`/`k_rows` from the shared aperture reserve, `q_start`/`q_rows`/`prefix_end` from the
    emit_q block, `num_computed` from the input batch. This record collapses the fan-out to ONE
    object per request: `layers` holds this request's OWN 0-based layer list (`req_layers`, in the
    exact order the fan-out used), so heterogeneous per-request layer sets are carried verbatim
    (request A can hold `[0,1,2]`, request B `[5,10]` in the same step). `expand_qk_records` fans it
    back out into the identical flat `QKStepEntry` list OFF the loop (in the consumer thread / the
    sync drain), so the sidecar bytes, reader, and demux are byte-for-byte unchanged — only WHERE the
    `QKStepEntry` objects are built moves off the engine loop."""
    req_id: str
    k_start: int
    k_rows: int
    q_start: int
    q_rows: int
    prefix_end: int
    num_computed: int
    layers: list          # this request's OWN 0-based layers, in req_layers (fan-out) order


def expand_qk_records(records) -> list:
    """Expand per-request `QKReqCaptureRecord`s into the FLAT per-(req, layer) `QKStepEntry` list the
    sidecar / demux consume — record order, then each record's `layers` order (identical to the old
    on-loop `for L in req_layers` fan-out). This is the OFF-LOOP counterpart of that fan-out: moving
    it here takes the O(reqs x layers) `QKStepEntry` construction off the engine loop while keeping
    every downstream byte identical.

    An element that is NOT a record (has no `layers` attribute) — an already-expanded `QKStepEntry`,
    the shape a direct-drain caller / unit test may hand in — passes through UNCHANGED (`QKStepEntry`
    carries a singular `layer`, never `layers`). So a caller holding either records or flat entries is
    handled; production always holds records (the collapse is the sole builder of
    `_qk_step_entries`)."""
    out: list = []
    for rec in records:
        layers = getattr(rec, "layers", None)
        if layers is None:
            out.append(rec)          # already a QKStepEntry (or pre-expanded) -> pass through
            continue
        for layer in layers:
            out.append(QKStepEntry(
                req_id=rec.req_id, layer=layer,
                k_start=rec.k_start, k_rows=rec.k_rows,
                q_start=rec.q_start, q_rows=rec.q_rows,
                prefix_end=rec.prefix_end, num_computed=rec.num_computed))
    return out


def write_qk_sidecar(path: str, steps: list, header: dict) -> None:
    """Write the header + one JSON line per :class:`QKStepEntry` across ``steps`` (a list of
    ``StepMeta``-shaped objects whose ``.entries`` are QKStepEntry). Header tuples are normalized to
    lists (see :func:`write_sidecar`)."""
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
    """Read a QK sidecar written by :func:`write_qk_sidecar` -> ``(header, entries)`` where
    ``entries`` is the FLAT, ORDERED list of :class:`QKStepEntry`. Downstream keys reconstruction on
    ``(req_id, layer)`` + ``k_start`` (the absolute row offset into the raw files), NEVER on list
    position (same contract as :func:`read_sidecar`)."""
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
    """Recursively coerce `tuple` values to `list` (dict/list/tuple nesting) so a header round-trips
    deterministically through JSON, which has no tuple type. Header values must otherwise already be
    JSON-native (str/int/float/bool/None/dict/list/tuple)."""
    if isinstance(value, tuple):
        return [_normalize_json_native(v) for v in value]
    if isinstance(value, list):
        return [_normalize_json_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_json_native(v) for k, v in value.items()}
    return value


def write_sidecar(path: str, steps: list, header: dict) -> None:
    """Write the header + one JSON line per `LayerEntry` across `steps`.

    `header` values must be JSON-native; any `tuple` (e.g. `row_shape=(4096,)`) is normalized to a
    `list` recursively before serialization, since JSON has no tuple type and a raw tuple would
    otherwise silently round-trip as a list anyway on read — normalizing on write keeps the on-disk
    JSON and the in-memory pre-write value consistent.
    """
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


# --------------------------------------------------------------------------- #
# Array-based sidecar accumulators (the drains' non-legacy write modes).
#
# The legacy drain fans every step out into one `LayerEntry` / `QKStepEntry` object per
# (request, layer), keeps all of them for the whole run, and json.dumps each at close: ~3.3 ms of a
# 15.9 ms 128-request 8B step, and an EngineCore that grew to 10 GB. These logs keep ONE small int64
# array per step instead -- a row per request: interned req-id / mode / layer-list indices plus the
# record's ints -- and derive every per-(request, layer) field only when the sidecar is written.
# The file they write is BYTE-IDENTICAL to `write_sidecar` / `write_qk_sidecar` over the entries the
# legacy drain would have built (same line order, same `"s"` numbering, same json.dumps spelling);
# the readers are untouched. A step whose records are not the builders' plain records (a flat,
# already-expanded entry; a float, unhashable or non-int field) is kept as legacy entries, so those
# stay exact too. The one input it does not reproduce is a `bool` standing in for an int field
# (numpy folds it to 0/1 where json.dumps spelled `true`); no builder produces one.
# --------------------------------------------------------------------------- #

class _NotPlain(Exception):
    """A record the array fast path does not represent exactly -> keep this step as entries."""


class _Interner:
    """value -> small int, remembering ``json.dumps(value)``. Keyed on ``(type, value)`` so values
    that compare equal but serialize differently (``1`` / ``True`` / ``1.0``) never share a slot."""
    __slots__ = ("_idx", "json")

    def __init__(self):
        self._idx: dict = {}
        self.json: list = []

    def __call__(self, v) -> int:
        key = (type(v), v)
        try:
            i = self._idx.get(key)
        except TypeError:                       # unhashable -> not a plain record
            raise _NotPlain(f"unhashable value {v!r}") from None
        if i is None:
            i = self._idx[key] = len(self.json)
            self.json.append(json.dumps(v))
        return i


class _LayerLists:
    """A record's ``layers`` list -> small int; remembers each layer's json spelling, its length,
    and (for HS) each layer's position among the installed layers. Accepts only lists/tuples of
    exact ``int``, checked once per distinct list."""
    __slots__ = ("_idx", "lists", "json", "lens")

    def __init__(self):
        self._idx: dict = {}
        self.lists: list = []     # tuple of ints
        self.json: list = []      # [str(l), ...] == json.dumps(l) for an int
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
    """``fields`` (a list of equal-length tuples) as an int64 ``(R, n_cols)`` array, refusing
    anything numpy would have to coerce (a float, an overflowing or non-int value)."""
    import numpy as np
    try:
        arr = np.array(fields)
    except (ValueError, TypeError, OverflowError):
        raise _NotPlain("ragged or non-numeric record fields") from None
    if arr.ndim != 2 or arr.shape[1] != n_cols or arr.dtype.kind not in "iu":
        raise _NotPlain(f"record fields are not plain ints (dtype {arr.dtype})")
    return arr.astype(np.int64, copy=False)


class HsSidecarLog:
    """Per-step HS sidecar records for ``hs_aperture_meta.jsonl``, kept as arrays.

    ``prepare(records, step_start, cursor, plans)`` builds one step's block WITHOUT committing it;
    the drain calls ``commit(block)`` only after every raw-file write of that step succeeded, so --
    exactly like the legacy drain, which appended a step's ``StepMeta`` only after its appends --
    the sidecar never names rows that did not reach the files.

    ``cursor`` is the per-layer file-row cursor BEFORE this step: an ``int`` when every installed
    layer was appended this step and all stood at that value, else an int64 array over
    ``layer_nums`` with ``-1`` for a layer not appended this step. ``plans`` is the per-layer
    ``LayerCopyPlan`` dict of a COMPACTING (selective) step, else None. ``file_row`` is derived at
    write time with the legacy ``_stamp_file_row`` arithmetic: base = that layer's cursor if it was
    appended this step else 0, plus ``logical_start - step_start``, or plus the row's offset in the
    layer's compacted copy (``plan.row_offset``) on a compacting step. The plans are built from the
    same records, so ``row_offset`` cannot miss here.

    ``legacy_stamp(entries, cursor_dict, step_start, plans)`` is the drain's own stamping function,
    used for a step kept as entries."""

    def __init__(self, layer_nums, legacy_stamp):
        self._lidx = {int(ln): i for i, ln in enumerate(layer_nums)}
        self._layer_nums = [int(ln) for ln in layer_nums]
        self._stamp = legacy_stamp
        self._rid = _Interner()
        self._mode = _Interner()
        self._lay = _LayerLists()
        self._lay_pos: list = []     # per layer list: ([installed index or -1, ...], all_installed)
        self.blocks: list = []
        self.n_entries = 0

    def has_entries(self) -> bool:
        return self.n_entries > 0

    def _cursor_dict(self, cursor) -> dict:
        if isinstance(cursor, int):
            return {ln: cursor for ln in self._layer_nums}
        return {ln: int(cursor[i]) for i, ln in enumerate(self._layer_nums) if int(cursor[i]) >= 0}

    def prepare(self, records, step_start: int, cursor, plans=None):
        """One step's block, or None when it expands to no entries (the legacy drain appended no
        ``StepMeta`` for such a step, so it takes no ``"s"`` index)."""
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
        """Write the sidecar -- byte-identical to ``write_sidecar`` over the legacy entries."""
        header = _normalize_json_native(header)
        with open(path, "w") as f:
            f.write(json.dumps({"__header__": header}) + "\n")
            for s, block in enumerate(list(self.blocks)):
                f.write("".join(self._lines(block, s)))


class QkSidecarLog:
    """Per-step QK sidecar records for ``qk_aperture_meta.jsonl``, kept as arrays: one int64 row
    per request (interned req id and layer list plus the record's six ints). Every field of a
    ``QKStepEntry`` but ``layer`` is per request, so each (request, layer) line is derived at write
    time. Byte-identical to ``write_qk_sidecar`` over the legacy entries. Same ``prepare`` /
    ``commit`` contract as :class:`HsSidecarLog`."""

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
        """Write the sidecar -- byte-identical to ``write_qk_sidecar`` over the legacy entries."""
        header = _normalize_json_native(header)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"__header__": header}) + "\n")
            for s, block in enumerate(list(self.blocks)):
                f.write("".join(self._lines(block, s)))


def read_sidecar(path: str):
    """Read a sidecar written by `write_sidecar` and return `(header, steps)`.

    `steps` is a list of `StepMeta`, one per DISTINCT step index that had at least one entry — a step
    with zero entries is elided (never appears as an empty `StepMeta`). Flattening `steps` in order
    (`[e for s in steps for e in s.entries]`) yields the FLAT, ORDERED list of `LayerEntry` records in
    the order they were written. Downstream consumers (the aperture reader + drain) must key reconstruction
    on each entry's `file_row` (the row offset into THAT LAYER's own raw file) — NEVER on
    `logical_start` (the aperture-wide reservation position — see `LayerEntry`), list position, or
    `len(steps)` — since step index is not preserved and elided steps shift positions.

    COMPATIBILITY (new reader, old sidecar): a sidecar written before the "fr" key existed has no
    per-entry file-row field. `obj.get("fr", obj["o"])` falls back to "o" (`logical_start`) in that
    case — not defensive padding, but EXACTLY correct: every artifact on disk before this change (and
    every all-layers/all-rows artifact forever) was written with every layer receiving every row, so
    its aperture-wide `logical_start` IS its per-layer file offset.

    That equivalence held only while the shared-file drain always copied every installed layer's
    whole span. Selective drain (`MIA_DRAIN_SELECTIVE`, default ON) breaks it: a sidecar it
    writes can carry entries where a layer's rows were compacted, so `o` no longer equals `file_row`
    for those entries. This reader is unaffected (it always prefers the "fr" key); the exposure is a
    DIFFERENT, hypothetical reader that predates that key and falls back to "o" the way this one does
    above — such a reader would silently reconstruct a compacted sidecar at the wrong offsets, and
    the header carries no version marker that would let it detect the mismatch. Contained today
    because no such reader exists (`aperture_reader` is the sole consumer, in this same repo, always
    current); worth revisiting if a second, independent parser is ever written against this format.
    """
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
