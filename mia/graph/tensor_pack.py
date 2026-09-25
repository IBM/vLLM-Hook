"""Pack an arbitrary tensor tree into ONE contiguous uint8 buffer + a pure-Python manifest.

torch.multiprocessing shares one shared-memory mapping (VMA) per distinct tensor STORAGE.
A writer-process flush hands O(batch x layers x steps) separately-allocated CPU tensors ->
that many VMAs -> exceeds the per-process vm.max_map_count ceiling and aborts with
"mmap ... Cannot allocate memory" (a COUNT limit, not a byte limit). Coalescing every
tensor into one buffer makes the handoff share exactly ONE storage -> one mapping.

Dedup is BY STORAGE (data_ptr): the QK k_all growing prefixes are views into one `full`
tensor; packing each view's bytes would re-expand to O(seq^2). Packing each unique storage
ONCE keeps it O(seq) and byte-exact. Reconstructed leaves are zero-copy strided views into
the buffer; tensors that shared a storage in the original share it again after unpack.
"""
from __future__ import annotations

import torch

_ALIGN = 8  # byte alignment per storage so every dtype (<=8B) reinterprets cleanly


def _dtype_str(dt: torch.dtype) -> str:
    return str(dt).split(".", 1)[1]  # "torch.float16" -> "float16"


def _storage_u8(t: torch.Tensor) -> torch.Tensor:
    """A 1-D uint8 view over the tensor's WHOLE untyped storage (host tensors only)."""
    s = t.untyped_storage()
    out = torch.empty(0, dtype=torch.uint8)
    out.set_(s, storage_offset=0, size=(s.nbytes(),), stride=(1,))
    return out


def pack_tensor_tree(obj):
    """Return (buffer_uint8_1d, manifest). manifest carries the storage table + a structural
    mirror of obj with every tensor replaced by a descriptor. Contains no tensors."""
    storages: dict[int, dict] = {}      # data_ptr -> record
    order: list[dict] = []              # unique storages in placement order
    cursor = 0

    def intern(t: torch.Tensor) -> int:
        nonlocal cursor
        s = t.untyped_storage()
        ptr = s.data_ptr()
        rec = storages.get(ptr)
        if rec is None:
            nbytes = s.nbytes()
            boff = cursor
            cursor += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
            rec = {"boff": boff, "nbytes": nbytes, "src": _storage_u8(t),
                   "slot": len(order)}
            storages[ptr] = rec
            order.append(rec)
        return rec["slot"]

    def walk(o):
        if isinstance(o, torch.Tensor):
            slot = intern(o)
            return {"__t": "tensor", "slot": slot,
                    "off": int(o.storage_offset()),
                    "shape": list(o.shape), "stride": list(o.stride()),
                    "dtype": _dtype_str(o.dtype)}
        if isinstance(o, dict):
            return {"__t": "dict", "items": [[k, walk(v)] for k, v in o.items()]}
        if isinstance(o, (list, tuple)):
            return {"__t": "tuple" if isinstance(o, tuple) else "list",
                    "items": [walk(v) for v in o]}
        return {"__t": "raw", "v": o}

    tree = walk(obj)
    buffer = torch.empty(cursor, dtype=torch.uint8)
    if cursor:
        buf_np = buffer.numpy()
        for rec in order:
            b, n = rec["boff"], rec["nbytes"]
            if n:
                buf_np[b:b + n] = rec["src"].numpy()   # GIL-released memcpy
    table = [{"boff": r["boff"], "nbytes": r["nbytes"]} for r in order]
    return buffer, {"__pack__": 1, "table": table, "tree": tree}


def unpack_tensor_tree(buffer: torch.Tensor, manifest: dict):
    """Inverse of pack_tensor_tree. Tensors are zero-copy strided views into buffer."""
    table = manifest["table"]
    storage = buffer.untyped_storage()

    def view(desc):
        dt = getattr(torch, desc["dtype"])
        rec = table[desc["slot"]]
        elems_off = rec["boff"] // dt.itemsize + desc["off"]
        t = torch.empty(0, dtype=dt)
        t.set_(storage, storage_offset=elems_off,
               size=tuple(desc["shape"]), stride=tuple(desc["stride"]))
        return t

    def build(node):
        k = node["__t"]
        if k == "tensor":
            return view(node)
        if k == "dict":
            return {kk: build(vv) for kk, vv in node["items"]}
        if k == "list":
            return [build(vv) for vv in node["items"]]
        if k == "tuple":
            return tuple(build(vv) for vv in node["items"])
        return node["v"]

    return build(manifest["tree"])
