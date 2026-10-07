"""Delivered HS and graph-mode Q/K data in the shapes an eager run returns."""
from __future__ import annotations

import copy
import copyreg
import ctypes
import json
import math
import mmap
import platform
import re
import struct
import sys
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional

import torch
from safetensors.torch import load, save
from torch.nn.utils.rnn import pad_sequence

from mia._profiler import PROF
from mia.errors import MiaDeliveryError
from mia.core.aperture.aperture_gather import DeliveryTimeoutError
from mia.workers.qk_capture_worker import _use_compact_kall

ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
SUFFIX_RE = r"-[0-9a-f]{8}"
WIRE_FORMAT = "mia-delivered-v1"
SPARSE_PAD_MIN_BYTES = 1 << 20


class DeliveryReadTimeout(MiaDeliveryError, DeliveryTimeoutError):
    """A delivery read gave up; both a ``MiaDeliveryError`` and a ``DeliveryTimeoutError``."""


def trim_probes(probes: dict, key: str, expected_len: int) -> None:
    """Trim probe tensors to expected_len along the sequence dimension."""
    for entry in probes.get(key, {}).values():
        for tkey in ("hidden_states", "q", "k_all"):
            t = entry.get(tkey)
            if t is None or isinstance(t, list):
                continue
            if t.dim() == 3 and t.shape[1] > expected_len:
                PROF.incr("trim.event")
                entry[tkey] = t[:, :expected_len, :]


def serialize_probes(probes: dict) -> dict:
    """Serialize probe tensors to lists for JSON transport."""
    PROF.incr("serve.serialize_probes.calls")
    with PROF.timed("serve.serialize_probes"):
        result = {}
        n_tensors = 0
        n_elems = 0
        for key, cache in probes.items():
            if key == "config" and isinstance(cache, dict):
                result[key] = cache
                continue
            if not isinstance(cache, dict):
                continue
            result[key] = {}
            for mod_name, entry in cache.items():
                new_entry = {}
                for k, v in entry.items():
                    if isinstance(v, torch.Tensor):
                        n_tensors += 1
                        n_elems += v.numel()
                        new_entry[k] = v.tolist()
                    elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                        n_tensors += len(v)
                        n_elems += sum(t.numel() for t in v)
                        new_entry[k] = [t.tolist() for t in v]
                    else:
                        new_entry[k] = v
                result[key][mod_name] = new_entry
        PROF.gauge("serve.serialize_probes.tensors", n_tensors)
        PROF.gauge("serve.serialize_probes.elements", n_elems)
    return result


def pass_sizes(n_rows: int, meta: dict) -> List[int]:
    """Rows per forward pass, as the eager hooks would have split one request's rows."""
    n_rows = int(n_rows)
    if n_rows <= 0:
        return []
    if meta["hs_mode"] == "last_token":
        return [1] * n_rows
    hooks = meta["hooks_on"]
    if hooks == "prefill":
        return [n_rows]
    if hooks == "decode":
        return [1] * n_rows
    # Prompt pass: tokens this sample computed (<= rows - n_gen + 1); each later pass is one row.
    bound = n_rows - max(int(meta["n_gen"]) - 1, 0)
    cached = meta.get("n_cached")
    p = bound if cached is None else int(meta["n_prompt"]) - int(cached)
    if not 0 < p <= bound:
        if cached is not None:
            _warn_split(n_rows, meta, bound)
        p = bound
    if not 0 < p <= n_rows:
        raise MiaDeliveryError(
            f"cannot split {n_rows} delivered rows into passes (n_prompt={meta.get('n_prompt')}, "
            f"n_gen={meta.get('n_gen')}, n_cached={meta.get('n_cached')})")
    return [p] + [1] * (n_rows - p)


def _warn_split(n_rows: int, meta: dict, bound: int) -> None:
    PROF.incr("delivered.split_mismatch")
    print(f"[mia] WARNING: delivered rows do not match the prompt (n_rows={n_rows}, "
          f"n_prompt={meta.get('n_prompt')}, n_gen={meta.get('n_gen')}, "
          f"n_cached={meta.get('n_cached')}); the prompt pass is taken as {bound} rows",
          flush=True)


_MAP_NORESERVE = getattr(mmap, "MAP_NORESERVE", 0x4000 if sys.platform.startswith("linux")
                         and platform.machine() in ("x86_64", "aarch64") else 0)


def _vm(name: str) -> str:
    try:
        with open(f"/proc/sys/vm/{name}") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def _zero_pages(nbytes: int):
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE
    try:
        mm = mmap.mmap(-1, nbytes, flags=flags)
    except OSError as e:
        raise MiaDeliveryError(
            f"cannot map {nbytes} bytes ({nbytes / 2**30:.1f} GiB) for one output's padded probes "
            f"({e}): either the kernel refused the size (vm.overcommit_memory="
            f"{_vm('overcommit_memory')}) or the process holds vm.max_map_count "
            f"({_vm('max_map_count')}) mappings, one per read output; drop probes no longer "
            f"needed") from e
    if hasattr(mmap, "MADV_NOHUGEPAGE"):
        mm.madvise(mmap.MADV_NOHUGEPAGE)
    return mm


def _fill(out, passes) -> None:
    single = [i for i, p in enumerate(passes) if p.shape[0] == 1]
    if single:
        out[torch.tensor(single), 0] = torch.cat([passes[i] for i in single])
    for i, p in enumerate(passes):
        if p.shape[0] != 1:
            out[i, :p.shape[0]] = p


def _sparse(padded: int, real: int) -> bool:
    return padded >= max(SPARSE_PAD_MIN_BYTES, 2 * real)


def _zero_page_tensors(shapes: Dict[int, tuple], dtype) -> Dict[int, Any]:
    esize = torch.empty((), dtype=dtype).element_size()
    offsets, total = {}, 0
    for L, sh in shapes.items():
        offsets[L] = total
        total += -(-(math.prod(sh) * esize) // mmap.PAGESIZE) * mmap.PAGESIZE
    mm = _zero_pages(total)
    return {L: torch.frombuffer(mm, dtype=dtype, count=math.prod(sh), offset=offsets[L]).view(sh)
            for L, sh in shapes.items()}


def pad_layers(passes_by_layer: Dict[int, list]) -> Dict[int, Any]:
    """``pad_sequence(passes, batch_first=True)`` per layer."""
    shapes = {L: (len(ps), max(int(p.shape[0]) for p in ps)) + tuple(ps[0].shape[1:])
              for L, ps in passes_by_layer.items()}
    padded = sum(math.prod(sh) * passes_by_layer[L][0].element_size() for L, sh in shapes.items())
    real = sum(p.numel() * p.element_size() for ps in passes_by_layer.values() for p in ps)
    if not _sparse(padded, real):
        return {L: pad_sequence(ps, batch_first=True) for L, ps in passes_by_layer.items()}
    out = _zero_page_tensors(shapes, next(iter(passes_by_layer.values()))[0].dtype)
    for L, t in out.items():
        _fill(t, passes_by_layer[L])
    return out


def padded_reader(meta: dict) -> Optional[Callable]:
    """``into`` for ``load_delivered``: read into the all_tokens padded layout; None otherwise."""
    if meta["hs_mode"] != "all_tokens":
        return None

    def into(layers, n_rows, row_shape, dtype_name):
        sizes = pass_sizes(n_rows, meta)
        dtype = getattr(torch, dtype_name)
        esize = torch.empty((), dtype=dtype).element_size()
        row = math.prod(row_shape) * esize
        shape = (len(sizes), max(sizes)) + tuple(row_shape)
        shapes = {int(L): shape for L in layers}
        if _sparse(len(shapes) * math.prod(shape) * esize, len(shapes) * n_rows * row):
            targets = _zero_page_tensors(shapes, dtype)
        else:
            targets = {L: torch.zeros(sh, dtype=dtype) for L, sh in shapes.items()}
        step = shape[1] * row
        out = {}
        for L, t in targets.items():
            # ctypes, not .numpy(): a numpy export makes the storage non-resizable.
            flat = memoryview((ctypes.c_char * (t.numel() * esize)).from_address(
                t.data_ptr())).cast("B")
            out[L] = (t, [flat[i * step:i * step + n * row] for i, n in enumerate(sizes)])
        return out
    return into


def padded_offline_probes(padded_by_layer: Dict[int, Any], meta, module_names) -> dict:
    """:func:`offline_probes` of a delivery read through :func:`padded_reader`."""
    cache: dict = {}
    for layer in sorted(int(L) for L in padded_by_layer):
        if module_names.get(layer) is None:
            raise MiaDeliveryError(f"no module name for delivered layer {layer}")
        cache[module_names[layer]] = {"hidden_states": padded_by_layer[layer], "layer_num": layer,
                                      "hs_mode": meta["hs_mode"]}
    probes = {"hs_cache": cache, "config": dict(meta["config"])}
    trim_probes(probes, "hs_cache", int(meta["n_prompt"]) + int(meta["n_gen"]) - 1)
    return probes


def hs_probes(rows_by_layer: Dict[int, Any], meta: dict, module_names: Dict[int, str], *,
              layout: str = "rpc") -> dict:
    """One request's delivered rows ``{layer: (rows, hidden)}`` in an eager HS shape."""
    if layout not in ("rpc", "disk"):
        raise ValueError(f"layout must be 'rpc' or 'disk', not {layout!r}")
    mode = meta["hs_mode"]
    split: dict = {}
    for layer in sorted(int(L) for L in rows_by_layer):
        t = rows_by_layer[layer]
        if module_names.get(layer) is None:
            raise MiaDeliveryError(f"no module name for delivered layer {layer}")
        sizes = pass_sizes(t.shape[0], meta)
        split[layer] = [t[i] for i in range(t.shape[0])] if mode == "last_token" else list(
            torch.split(t, sizes))
    if layout == "rpc":
        stacked = ({L: torch.stack(ps) for L, ps in split.items()} if mode == "last_token"
                   else pad_layers(split))
    cache: dict = {}
    for layer, passes in split.items():
        hs = stacked[layer] if layout == "rpc" else [p.clone() for p in passes]
        cache[module_names[layer]] = {"hidden_states": hs, "layer_num": layer, "hs_mode": mode}
    conf = dict(meta["config"])
    if layout == "rpc":
        return {"hs_cache": cache, "config": conf}
    return {"config": conf, "hs_cache": cache}


def offline_probes(rows_by_layer, meta, module_names) -> dict:
    """``output.probes`` of an eager offline run (rpc layout, trimmed)."""
    probes = hs_probes(rows_by_layer, meta, module_names, layout="rpc")
    trim_probes(probes, "hs_cache", int(meta["n_prompt"]) + int(meta["n_gen"]) - 1)
    return probes


def first_pass_probes(rows_by_layer, meta, module_names) -> dict:
    """:func:`offline_probes` with each ``hidden_states`` cut to its first pass."""
    mode = meta["hs_mode"]
    end = int(meta["n_prompt"]) + int(meta["n_gen"]) - 1
    cache: dict = {}
    for layer in sorted(int(L) for L in rows_by_layer):
        t = rows_by_layer[layer]
        name = module_names.get(layer)
        if name is None:
            raise MiaDeliveryError(f"no module name for delivered layer {layer}")
        sizes = pass_sizes(t.shape[0], meta)
        if mode == "last_token":
            first = t[:1].clone()
        else:
            width = max(sizes)
            width = end if width > end else width
            first = t.new_zeros((1, width) + tuple(t.shape[1:]))
            n = min(sizes[0], width)
            first[0, :n] = t[:n]
        cache[name] = {"hidden_states": first, "layer_num": layer, "hs_mode": mode}
    return {"hs_cache": cache, "config": dict(meta["config"])}


def client_probes(rows_by_layer, meta, module_names) -> dict:
    """``response.probes`` as an eager client sees it (the server's serialized form)."""
    return serialize_probes(offline_probes(rows_by_layer, meta, module_names))


_DISK_LISTS = {"hs_cache": ("hidden_states",), "qk_cache": ("q", "k_full", "k_prefix_ends")}


def merge_disk(into: Optional[dict], one: dict) -> dict:
    """Append one request's disk-layout probes to a run's cache, as ``flush_disk`` merges them."""
    key = "qk_cache" if "qk_cache" in one else "hs_cache"
    lists = _DISK_LISTS[key]
    if into is None:
        into = {"config": dict(one["config"]), key: {}}
    for name, entry in one[key].items():
        have = into[key].get(name)
        if have is None:
            into[key][name] = {**entry, **{k: list(entry[k]) for k in lists}}
        else:
            for k in lists:
                have[k].extend(entry[k])
    return into


def qk_pass_rows(n_q: int, prefix_ends: List[int], mode: str) -> List[int]:
    """q rows per forward pass: one per pass (last_token), else this pass's new tokens."""
    if mode == "last_token":
        return [1] * len(prefix_ends)
    ends = [int(L) for L in prefix_ends]
    if not ends:
        return []
    first = int(n_q) - (ends[-1] - ends[0])
    return [first] + [b - a for a, b in zip(ends[:-1], ends[1:])]


def qk_probes(payload: dict, *, n_prompt: int, n_gen: int, hookq_mode: Optional[str] = None,
              hooks_on: Optional[str] = None, layout: str = "rpc") -> Optional[dict]:
    """One request's delivered Q/K in an eager shape, or None when it captured nothing."""
    if layout not in ("rpc", "disk"):
        raise ValueError(f"layout must be 'rpc' or 'disk', not {layout!r}")
    per_layer = payload.get("qk_cache") or {}
    if not any(rec.get("k_all") for rec in per_layer.values()):
        return None
    names = {int(L): n for L, n in (payload.get("module_names") or {}).items()}
    conf = dict(payload.get("config") or {})
    cache: dict = {}
    for layer in sorted(int(L) for L in per_layer):
        rec = per_layer[layer]
        name = names.get(layer)
        if name is None:
            raise MiaDeliveryError(f"no module name for delivered Q/K layer {layer}")
        mode = hookq_mode or rec.get("hookq_mode") or "all_tokens"
        k_all = list(rec.get("k_all") or [])
        ends = [int(k.shape[0]) for k in k_all]
        full = k_all[-1] if k_all else None
        kf = rec.get("k_full")
        if not ends or (kf is not None and int(kf.shape[0]) != ends[-1]):
            raise MiaDeliveryError(f"layer {layer}: key rows do not match the history it reports "
                                   f"(its prefill ran again after preemption)")
        q = rec["q"]
        sizes = qk_pass_rows(q.shape[0], ends, mode)
        first_ok = (hooks_on == "decode" or hooks_on is None or mode == "last_token"
                    or sizes[0] == ends[0])
        if not first_ok or any(n <= 0 for n in sizes):
            raise MiaDeliveryError(f"layer {layer}: {q.shape[0]} q rows do not split into "
                                   f"{len(ends)} passes ending at {ends[:4]}")
        passes = [q[i] for i in range(q.shape[0])] if mode == "last_token" else list(
            torch.split(q, sizes))
        if layout == "disk":
            cache[name] = {"q": [p.clone() for p in passes], "layer_num": layer,
                           "hookq_mode": mode, "k_full": [full.clone()], "k_prefix_ends": [ends]}
            continue
        q_st = (pad_sequence(passes, batch_first=True) if mode == "all_tokens"
                else torch.stack(passes))
        if _use_compact_kall({"k_prefix_ends": ends}):
            cache[name] = {"q": q_st, "k_full": full, "k_prefix_ends": ends, "layer_num": layer,
                           "hookq_mode": mode}
        else:
            cache[name] = {"q": q_st, "k_all": pad_sequence(k_all, batch_first=True),
                           "layer_num": layer, "hookq_mode": mode}
    if layout == "disk":
        return {"config": conf, "qk_cache": cache}
    # lazy: import cycle (mia._plugin imports this module)
    from mia._plugin import _reconstruct_compact_qk

    probes = {"qk_cache": cache, "config": conf}
    _reconstruct_compact_qk(probes)
    trim_probes(probes, "qk_cache", int(n_prompt) + int(n_gen) - 1)
    return probes


def check_id(rid: str) -> str:
    if not isinstance(rid, str) or not ID_RE.match(rid):
        raise ValueError("response id is not readable back (allowed: 1-256 of A-Za-z0-9._:-)")
    return rid


def external_id(rid: str, kind: str, item: int) -> str:
    """The engine request id vLLM gave one item of response ``rid``."""
    check_id(rid)
    if kind == "chat":
        if int(item) != 0:
            raise ValueError("a chat response has one item")
        return rid
    if kind == "completion":
        return f"{rid}-{int(item)}"
    raise ValueError(f"kind must be 'chat' or 'completion', not {kind!r}")


def response_id(engine_id: str) -> str:
    """The response an engine request id belongs to: ``cmpl-<id>-<i>`` -> ``cmpl-<id>``."""
    m = re.match(r"^(cmpl-.+)-\d+$", str(engine_id))
    return m.group(1) if m else str(engine_id)


def key_pattern(ext: str, n: int, *, suffix: bool = True):
    """Delivered keys of an engine request: ``ext-<8 hex>``, samples ``<j>_ext-<8 hex>``."""
    s = SUFFIX_RE if suffix else ""
    e = re.escape(ext)
    if int(n) == 1:
        return re.compile(rf"^{e}{s}$")
    return re.compile(rf"^(\d+)_{e}{s}$")


class AmbiguousDelivery(MiaDeliveryError):
    """More delivered keys match a response than it has samples (``names``: for logs only)."""

    def __init__(self, msg: str, count: int, names=()):
        super().__init__(msg)
        self.count = int(count)
        self.names = list(names)


def match_keys(names, ext: str, n: int, *, suffix: bool = True) -> Dict[int, str]:
    """``{sample: key}`` among ``names``; raises ``AmbiguousDelivery`` on surplus matches."""
    pat = key_pattern(ext, n, suffix=suffix)
    found: Dict[int, str] = {}
    extra: List[str] = []
    for k in sorted(set(names)):
        m = pat.match(k)
        if not m:
            continue
        j = 0 if int(n) == 1 else int(m.group(1))
        if j >= int(n) or j in found:
            extra.append(k)
            continue
        found[j] = k
    if extra:
        k = len(found) + len(extra)
        raise AmbiguousDelivery(f"ambiguous: {k} deliveries match", k,
                                sorted(found.values()) + extra)
    return found


def encode_delivery(samples: List[Dict[int, Any]], *, keys: List[str],
                    names: Dict[int, str], config: dict) -> bytes:
    tensors = {f"{j}/{int(L)}": t.contiguous() for j, s in enumerate(samples)
               for L, t in s.items()}
    meta = {"format": WIRE_FORMAT, "keys": json.dumps(list(keys)),
            "names": json.dumps({str(int(L)): str(v) for L, v in names.items()}),
            "config": json.dumps(dict(config))}
    return save(tensors, metadata=meta)


def decode_delivery(data: bytes):
    """``(samples, keys, names, config)`` from :func:`encode_delivery` bytes."""
    n = struct.unpack("<Q", data[:8])[0]
    meta = json.loads(data[8:8 + n].decode("utf-8")).get("__metadata__") or {}
    if meta.get("format") != WIRE_FORMAT:
        raise MiaDeliveryError(f"not a {WIRE_FORMAT} payload")
    keys = json.loads(meta["keys"])
    names = {int(L): v for L, v in json.loads(meta["names"]).items()}
    config = json.loads(meta["config"])
    samples: List[Dict[int, Any]] = [{} for _ in keys]
    for name, t in load(data).items():
        j, L = name.split("/")
        samples[int(j)][int(L)] = t
    return samples, keys, names, config


class _Pending:
    __slots__ = ("fn", "first", "lock", "fin")

    def __init__(self, fn, first=None):
        self.fn, self.first = fn, first
        self.lock = threading.Lock()
        self.fin = None


_PENDING: Dict[int, _Pending] = {}
_LAZY_TYPES: Dict[type, type] = {}
_BASES: Dict[type, type] = {}
_PLAIN: set = set()


def _extra(obj) -> dict:
    if type(obj) in _PLAIN:
        return object.__getattribute__(obj, "__dict__")
    ex = object.__getattribute__(obj, "__pydantic_extra__")
    if ex is None:
        ex = {}
        object.__setattr__(obj, "__pydantic_extra__", ex)
    return ex


def resolve(obj) -> None:
    """Run ``obj``'s pending loader once; failures surface as ``MiaDeliveryError``."""
    p = _PENDING.get(id(obj))
    if p is None:
        return
    with p.lock:
        if _PENDING.get(id(obj)) is not p:
            return
        try:
            value = p.fn()
        except MiaDeliveryError:
            raise
        except Exception as e:  # noqa: BLE001
            raise MiaDeliveryError(f"reading delivered probes failed: {e!r}") from e
        if value is not None:
            _extra(obj)["probes"] = value
        _PENDING.pop(id(obj), None)
        if p.fin is not None:
            p.fin.detach()


def _resolve_tree(obj) -> None:
    resolve(obj)
    d = object.__getattribute__(obj, "__dict__")
    for v in d.values():
        for x in (v if isinstance(v, list) else (v,)):
            if id(x) in _PENDING:
                resolve(x)


def _as_base(obj):
    base = _BASES.get(type(obj))
    if base is None:
        return obj
    m = base.__new__(base)
    keys = ("__dict__",) if type(obj) in _PLAIN else (
        "__dict__", "__pydantic_extra__", "__pydantic_fields_set__", "__pydantic_private__")
    for k in keys:
        object.__setattr__(m, k, copy.copy(object.__getattribute__(obj, k)))
    return m


def _lazy_type(base: type) -> type:
    t = _LAZY_TYPES.get(base)
    if t is not None:
        return t

    def _get(self):
        resolve(self)
        ex = _extra(self)
        if "probes" not in ex:
            raise AttributeError("probes")
        return ex["probes"]

    def _set(self, value):
        _PENDING.pop(id(self), None)
        _extra(self)["probes"] = value

    def _del(self):
        _PENDING.pop(id(self), None)
        _extra(self).pop("probes", None)

    def model_dump(self, *a, **kw):
        _resolve_tree(self)
        return base.model_dump(self, *a, **kw)

    def model_dump_json(self, *a, **kw):
        _resolve_tree(self)
        return base.model_dump_json(self, *a, **kw)

    def to_dict(self, *a, **kw):
        _resolve_tree(self)
        return base.to_dict(self, *a, **kw)

    def to_json(self, *a, **kw):
        _resolve_tree(self)
        return base.to_json(self, *a, **kw)

    def __iter__(self):
        _resolve_tree(self)
        return base.__iter__(self)

    def _model_extra(self):
        _resolve_tree(self)
        return object.__getattribute__(self, "__pydantic_extra__")

    def __repr_args__(self):
        _resolve_tree(self)
        return base.__repr_args__(self)

    def __eq__(self, other):
        _resolve_tree(self)
        if id(other) in _PENDING or type(other) in _BASES:
            _resolve_tree(other)
            other = _as_base(other)
        return _as_base(self) == other

    def __reduce_ex__(self, protocol):
        _resolve_tree(self)
        return (copyreg._reconstructor, (base, object, None), _as_base(self).__getstate__())

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __copy__(self):
        _resolve_tree(self)
        return _as_base(self)

    def __deepcopy__(self, memo=None):
        _resolve_tree(self)
        return copy.deepcopy(_as_base(self), memo)

    ns = {"__module__": base.__module__, "__qualname__": base.__qualname__,
          "probes": property(_get, _set, _del), "model_dump": model_dump,
          "model_dump_json": model_dump_json, "__iter__": __iter__,
          "model_extra": property(_model_extra), "__repr_args__": __repr_args__,
          "__eq__": __eq__, "__hash__": None, "__reduce_ex__": __reduce_ex__,
          "__reduce__": __reduce__, "__copy__": __copy__, "__deepcopy__": __deepcopy__}
    if hasattr(base, "to_dict"):
        ns["to_dict"] = to_dict
    if hasattr(base, "to_json"):
        ns["to_json"] = to_json
    t = type(base)(base.__name__, (base,), ns)
    _LAZY_TYPES[base] = t
    _BASES[t] = base
    return t


def _plain_lazy_type(base: type) -> type:
    t = _LAZY_TYPES.get(base)
    if t is not None:
        return t

    def _get(self):
        resolve(self)
        d = object.__getattribute__(self, "__dict__")
        if "probes" not in d:
            raise AttributeError("probes")
        return d["probes"]

    def _set(self, value):
        _PENDING.pop(id(self), None)
        object.__getattribute__(self, "__dict__")["probes"] = value

    def _del(self):
        _PENDING.pop(id(self), None)
        object.__getattribute__(self, "__dict__").pop("probes", None)

    def __reduce_ex__(self, protocol):
        resolve(self)
        return (copyreg._reconstructor, (base, object, None),
                dict(object.__getattribute__(self, "__dict__")))

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __copy__(self):
        resolve(self)
        return _as_base(self)

    def __deepcopy__(self, memo=None):
        resolve(self)
        return copy.deepcopy(_as_base(self), memo)

    ns = {"__module__": base.__module__, "__qualname__": base.__qualname__,
          "probes": property(_get, _set, _del), "__reduce_ex__": __reduce_ex__,
          "__reduce__": __reduce__, "__copy__": __copy__, "__deepcopy__": __deepcopy__}
    if base.__eq__ is not object.__eq__:
        def __eq__(self, other):
            return _as_base(self) == (_as_base(other) if type(other) in _BASES else other)
        ns["__eq__"], ns["__hash__"] = __eq__, base.__hash__
    t = type(base)(base.__name__, (base,), ns)
    _LAZY_TYPES[base] = t
    _BASES[t] = base
    _PLAIN.add(t)
    return t


def attach_lazy(obj, loader: Callable[[], Any], *, first: Optional[Callable] = None) -> None:
    """Make ``obj.probes`` lazy: ``loader()`` runs on first use."""
    # lazy: optional dependency (pydantic)
    import pydantic

    base = _BASES.get(type(obj), type(obj))
    weakref.ref(obj)
    t = _lazy_type(base) if isinstance(obj, pydantic.BaseModel) else _plain_lazy_type(base)
    if type(obj) is not t:
        object.__setattr__(obj, "__class__", t)
    p = _Pending(loader, first)
    _PENDING[id(obj)] = p
    p.fin = weakref.finalize(obj, _PENDING.pop, id(obj), None)


def pending(obj) -> bool:
    """True while ``obj.probes`` has not been read."""
    return id(obj) in _PENDING


def merge_source(obj) -> Optional[Callable[[], Any]]:
    """What a batch merge reads of ``obj``, without reading it now."""
    p = _PENDING.get(id(obj))
    if p is not None:
        fn = p.fn
        return p.first if p.first is not None else (lambda: fn() or {})
    v = getattr(obj, "probes", None)
    return None if v is None else (lambda: v)
