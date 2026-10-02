"""Opt-in instrumentation attributing GPU-to-host offload cost (MIA_CAPTURE_CENSUS)."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

_CENSUS_ON = os.environ.get("MIA_CAPTURE_CENSUS") == "1"

_KNOWN_KEYS = ("q", "k_all", "hidden_states", "scores")

_emit_failures = 0


def census_enabled() -> bool:
    """Whether MIA_CAPTURE_CENSUS=1 was set when this process started."""
    return _CENSUS_ON


def _size_bucket(nbytes: int) -> str:
    if nbytes <= 0:
        return "0"
    lo = 1
    while lo * 2 <= nbytes:
        lo *= 2
    return f"{lo}-{lo * 2}"


def census_bucket(layer_dict: Dict[str, dict]) -> Dict[str, Any]:
    """Pure structural census of ONE popped request bucket."""
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
    """``[t.cpu() for t in tensors]``, decomposed into timed allocation + copy."""
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
    """Merge one request's census bucket and copy timings into a single JSON record."""
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
    """Append one JSON object for ``record`` to ``MIA_CAPTURE_CENSUS_OUT``."""
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
    """Count of :func:`census_emit` calls that raised and were swallowed."""
    return _emit_failures

