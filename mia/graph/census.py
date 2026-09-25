"""Measurement instrumentation for GPU->host offload cost attribution.

Splits the flush D2H (today's ``[t.cpu() for t in tensors]``) into allocation cost
(``torch.empty``) versus copy cost (``.copy_``), gated by ``MIA_CAPTURE_CENSUS=1``
(default OFF -> zero cost; every gated call site falls through to the plain ``.cpu()`` list
comprehension otherwise).

* :func:`census_bucket` -- a pure STRUCTURAL walk of a popped request bucket (tensor
  count/bytes/dtype/histogram), read BEFORE any ``.cpu()`` runs -- only tensor metadata, no
  device values, syncs, or copies.
* :func:`cpu_list_measured` -- the same D2H, decomposed and timed with
  ``time.perf_counter()``, accumulated ONCE per request (not per tensor -- see the warning
  on that function about perturbing the signal being measured).

:func:`census_record` merges one request's :func:`census_bucket` result with its
:func:`cpu_list_measured` accumulator into the JSON object :func:`census_emit` appends to
``MIA_CAPTURE_CENSUS_OUT`` (default ``census.jsonl`` under ``MIA_PROFILE_DIR``,
else the CWD). Diagnostic only -- it never changes a captured value.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

# Read once at import, matching how the rest of this package gates levers (e.g.
# qk_capture_worker._COMPACT_KALL_ENV, _profiler._ENABLED).
_CENSUS_ON = os.environ.get("MIA_CAPTURE_CENSUS") == "1"

# The four artifact families the two probe workers actually produce, and the only keys this
# experiment censuses. Scale/qmeta metadata lists (quantized capture) are intentionally left
# out -- they are not the O(tensor-count) trajectory this experiment targets.
_KNOWN_KEYS = ("q", "k_all", "hidden_states", "scores")

# Count of census_emit() calls that raised and were swallowed. Introspection only; never
# raised to the caller.
_emit_failures = 0


def census_enabled() -> bool:
    """Whether MIA_CAPTURE_CENSUS=1 was set when this process started."""
    return _CENSUS_ON


def _size_bucket(nbytes: int) -> str:
    """Coarse power-of-two byte-size bucket label (e.g. a 5000-byte tensor -> '4096-8192')."""
    if nbytes <= 0:
        return "0"
    lo = 1
    while lo * 2 <= nbytes:
        lo *= 2
    return f"{lo}-{lo * 2}"


def census_bucket(layer_dict: Dict[str, dict]) -> Dict[str, Any]:
    """Pure structural census of ONE popped request bucket.

    ``layer_dict`` is ``{module_name: entry}`` exactly as popped from ``_captured_states`` /
    ``_disk_states`` -- BEFORE any ``.cpu()`` conversion. Reads only ``.numel()`` /
    ``.element_size()`` / ``.dtype`` / ``.is_cuda`` (tensor metadata -- never a device value,
    never a sync, never a copy), so it is safe to call unconditionally under the gate.

    Returns total tensor count/bytes, per-key (q/k_all/hidden_states/scores) counts and
    bytes, a coarse size histogram, the dtype set, a cuda/cpu tensor split, the number of
    distinct modules (layers), and the append count per module -- the last is what makes the
    prefill-vs-decode split visible (chunked prefill appends multiple times before decode).
    """
    total_tensors = 0
    total_bytes = 0
    per_key_count: Dict[str, int] = {}
    per_key_bytes: Dict[str, int] = {}
    size_histogram: Dict[str, int] = {}
    dtypes = set()
    device_counts = {"cuda": 0, "cpu": 0}
    appends_per_module: Dict[str, int] = {}

    for mod_name, entry in layer_dict.items():
        module_appends = 0
        for key in _KNOWN_KEYS:
            values = entry.get(key)
            if not values:
                continue
            if module_appends == 0:
                module_appends = len(values)
            for t in values:
                if not hasattr(t, "numel"):
                    continue
                nbytes = int(t.numel()) * int(t.element_size())
                total_tensors += 1
                total_bytes += nbytes
                per_key_count[key] = per_key_count.get(key, 0) + 1
                per_key_bytes[key] = per_key_bytes.get(key, 0) + nbytes
                dtypes.add(str(t.dtype))
                bucket = _size_bucket(nbytes)
                size_histogram[bucket] = size_histogram.get(bucket, 0) + 1
                if getattr(t, "is_cuda", False):
                    device_counts["cuda"] += 1
                else:
                    device_counts["cpu"] += 1
        appends_per_module[mod_name] = module_appends

    return {
        "total_tensors": total_tensors,
        "total_bytes": total_bytes,
        "per_key_count": per_key_count,
        "per_key_bytes": per_key_bytes,
        "size_histogram": size_histogram,
        "dtypes": sorted(dtypes),
        "device_counts": device_counts,
        "num_layers": len(layer_dict),
        "appends_per_module": appends_per_module,
    }


def new_accumulator() -> Dict[str, float]:
    """A fresh per-request accumulator dict for :func:`cpu_list_measured`."""
    return {"n": 0, "bytes": 0, "alloc_s": 0.0, "copy_s": 0.0}


def cpu_list_measured(tensors, acc: Dict[str, float]) -> List[torch.Tensor]:
    """``[t.cpu() for t in tensors]``, decomposed into timed allocation + copy.

    Splits each ``.cpu()`` into exactly what it does internally --
    ``torch.empty(t.shape, dtype=t.dtype, device="cpu")`` then ``dst.copy_(t)`` -- so the
    return value equals ``[t.cpu() for t in tensors]`` elementwise. Timed with
    ``time.perf_counter()`` only, accumulated into the mutable per-request ``acc`` dict
    (keys ``n`` / ``bytes`` / ``alloc_s`` / ``copy_s``) ONCE after the loop.

    Deliberately NOT a ``PROF.timed`` context per tensor: at ~6,400 tensors/request a
    per-tensor context manager (lock acquire + list append) would perturb the ~18us signal
    this is trying to measure. ``perf_counter`` alone is ~20ns.
    """
    out: List[torch.Tensor] = []
    n = 0
    nbytes = 0
    alloc_s = 0.0
    copy_s = 0.0
    for t in tensors:
        t0 = time.perf_counter()
        dst = torch.empty(t.shape, dtype=t.dtype, device="cpu")
        t1 = time.perf_counter()
        dst.copy_(t)
        t2 = time.perf_counter()
        out.append(dst)
        n += 1
        nbytes += int(t.numel()) * int(t.element_size())
        alloc_s += t1 - t0
        copy_s += t2 - t1
    acc["n"] = acc.get("n", 0) + n
    acc["bytes"] = acc.get("bytes", 0) + nbytes
    acc["alloc_s"] = acc.get("alloc_s", 0.0) + alloc_s
    acc["copy_s"] = acc.get("copy_s", 0.0) + copy_s
    return out


def census_record(*, worker: str, sink: str, req_id: str, bucket: Dict[str, Any],
                   acc: Dict[str, float]) -> Dict[str, Any]:
    """Merge one request's :func:`census_bucket` + :func:`cpu_list_measured` accumulator
    into the single JSON record :func:`census_emit` writes.

    Computes the two derived numbers this whole experiment is for: ``alloc_frac`` (the
    allocation share of the flush D2H) and ``bandwidth_gbps`` (bytes moved / copy time --
    the effective bandwidth once allocation is excluded). ``n_tensors`` / ``total_bytes``
    come from the structural census (ground truth, pre-conversion); ``measured_n_tensors`` /
    ``measured_bytes`` come from what actually passed through :func:`cpu_list_measured` --
    the two should agree closely (the self-consistency check the spec asks for) whenever the
    native, non-quantized, non-score capture path is what fired.
    """
    alloc_s = float(acc.get("alloc_s", 0.0))
    copy_s = float(acc.get("copy_s", 0.0))
    total_s = alloc_s + copy_s
    alloc_frac = (alloc_s / total_s) if total_s > 0 else None
    measured_bytes = int(acc.get("bytes", 0))
    bandwidth_gbps = (measured_bytes / copy_s / 1e9) if copy_s > 0 else None

    record: Dict[str, Any] = {
        "worker": worker,
        "sink": sink,
        "req_id": req_id,
        "n_tensors": bucket.get("total_tensors", 0),
        "total_bytes": bucket.get("total_bytes", 0),
        "measured_n_tensors": int(acc.get("n", 0)),
        "measured_bytes": measured_bytes,
        "alloc_s": alloc_s,
        "copy_s": copy_s,
        "alloc_frac": alloc_frac,
        "bandwidth_gbps": bandwidth_gbps,
    }
    for k in ("per_key_count", "per_key_bytes", "size_histogram", "dtypes",
              "device_counts", "num_layers", "appends_per_module"):
        record[k] = bucket.get(k)
    return record


def census_emit(record: Dict[str, Any]) -> None:
    """Append one JSON object for ``record`` to ``MIA_CAPTURE_CENSUS_OUT``.

    Default path: ``census.jsonl`` inside ``MIA_PROFILE_DIR`` (the CWD if that is also
    unset). Opens in append mode per call and flushes -- robustness over speed; this runs
    once per finished request, not per tensor. Never raises into the caller: any failure is
    swallowed and counted in the module-level failure counter (see
    :func:`census_emit_failures`).
    """
    global _emit_failures
    try:
        out_path = os.environ.get("MIA_CAPTURE_CENSUS_OUT")
        if not out_path:
            profile_dir = os.environ.get("MIA_PROFILE_DIR") or "."
            out_path = os.path.join(profile_dir, "census.jsonl")
        with open(out_path, "a") as f:
            f.write(json.dumps(record))
            f.write("\n")
            f.flush()
    except Exception:
        _emit_failures += 1


def census_emit_failures() -> int:
    """Count of :func:`census_emit` calls that raised and were swallowed. Diagnostic only."""
    return _emit_failures
