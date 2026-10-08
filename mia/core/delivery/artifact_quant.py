"""On-GPU quantization of captured artifacts (QK q/k_all, hidden-states, scores)."""

import os

import torch


_INT_BITS = {"int2": 2, "int4": 4, "int8": 8}
_FLOAT_CAST = {
    "fp8_e4m3": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None,
    "fp8_e5m2": torch.float8_e5m2 if hasattr(torch, "float8_e5m2") else None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_ALIASES = {
    "int2": "int2", "i2": "int2",
    "int4": "int4", "i4": "int4",
    "int8": "int8", "i8": "int8",
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2", "e5m2": "fp8_e5m2",
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16", "half": "fp16",
    "fp32": "fp32", "float32": "fp32", "float": "fp32",
}
_OFF = {"", "none", "native", "off", "0", "false", "no"}

_FLOAT8_DTYPES = tuple(d for d in (getattr(torch, "float8_e4m3fn", None),
                                   getattr(torch, "float8_e5m2", None)) if d is not None)

_SCALE_SUFFIX = "_scale"
_ZP_SUFFIX = "_zp"
_QMETA_SUFFIX = "_qmeta"
SIBLING_TENSOR_SUFFIXES = (_SCALE_SUFFIX, _ZP_SUFFIX)


def scale_key(key):
    return key + _SCALE_SUFFIX


def zp_key(key):
    return key + _ZP_SUFFIX


def qmeta_key(key):
    return key + _QMETA_SUFFIX


def sibling_tensor_keys(base_keys):
    """The per-request tensor sibling keys for a set of primary artifact keys."""
    return tuple(k + s for k in base_keys for s in SIBLING_TENSOR_SUFFIXES)


def parse_artifact_dtype(value):
    """Normalize a dtype string to a canonical tag, or None for native."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in _OFF:
        return None
    tag = _ALIASES.get(v)
    if tag is None:
        raise ValueError(
            f"MIA_ARTIFACT_DTYPE={value!r} not recognized; expected one of "
            "int2,int4,int8,fp8_e4m3,fp8_e5m2,bf16,fp16,fp32 (or none/off)"
        )
    if tag in ("fp8_e4m3", "fp8_e5m2") and _FLOAT_CAST[tag] is None:
        raise ValueError(f"{tag} requires a torch build with the matching float8 dtype")
    return tag


_ARTIFACT_ENV = {
    "qk": "MIA_ARTIFACT_DTYPE_QK",
    "hs": "MIA_ARTIFACT_DTYPE_HS",
    "score": "MIA_ARTIFACT_DTYPE_SCORE",
}


def resolve_dtype(artifact="qk"):
    """Resolve an artifact family's dtype tag: per-artifact env override, else MIA_ARTIFACT_DTYPE."""
    v = os.environ.get(_ARTIFACT_ENV.get(artifact, ""))
    if v is None:
        v = os.environ.get("MIA_ARTIFACT_DTYPE")
    return parse_artifact_dtype(v)


def resolve_granularity():
    g = (os.environ.get("MIA_ARTIFACT_QUANT_GRAN", "per_token") or "per_token").strip().lower()
    return g if g in ("per_tensor", "per_token", "group") else "per_token"


def resolve_group_size():
    try:
        return max(1, int(os.environ.get("MIA_ARTIFACT_QUANT_GROUP_SIZE", "128")))
    except (TypeError, ValueError):
        return 128


def _pack_lowbit(qi_u8, bits):
    per_byte = 8 // bits
    L = qi_u8.shape[-1]
    pad = (-L) % per_byte
    if pad:
        qi_u8 = torch.nn.functional.pad(qi_u8, (0, pad))
    qi_u8 = qi_u8.contiguous()
    packed = qi_u8[..., 0::per_byte]
    for i in range(1, per_byte):
        packed = packed | (qi_u8[..., i::per_byte] << (i * bits))
    return packed.contiguous()


def _unpack_lowbit(packed, bits, orig_last):
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    fields = [(packed >> (i * bits)) & mask for i in range(per_byte)]
    out = torch.stack(fields, dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * per_byte)
    return out[..., :orig_last]


def quantize(x, tag, gran="per_token", group_size=128):
    """Quantize ``x`` to ``tag``."""
    if tag is None:
        return x, None, None, None

    if tag in _FLOAT_CAST:
        packed = x.to(_FLOAT_CAST[tag])
        if _FLOAT8_DTYPES and packed.dtype in _FLOAT8_DTYPES:
            packed = packed.view(torch.uint8)
        return packed, None, None, {"tag": tag, "gran": "none", "orig_dtype": str(x.dtype)}

    bits = _INT_BITS[tag]
    qmax = (1 << (bits - 1)) - 1
    orig_shape = tuple(x.shape)
    orig_dtype = str(x.dtype)
    xf = x.detach().to(torch.float32)

    if xf.numel() == 0:
        packed = xf.to(torch.int8) if bits == 8 else _pack_lowbit(
            torch.zeros_like(xf, dtype=torch.uint8), bits)
        scale = torch.ones(1, dtype=torch.float32, device=x.device)
        return packed, scale, None, {"tag": tag, "gran": "per_tensor", "bits": bits,
                                     "orig_shape": orig_shape, "orig_dtype": orig_dtype}

    use_gran = "per_tensor" if xf.dim() < 2 else gran
    qmeta = {"tag": tag, "gran": use_gran, "bits": bits,
             "orig_shape": orig_shape, "orig_dtype": orig_dtype}

    if use_gran == "per_tensor":
        scale = (xf.abs().amax() / qmax).clamp_min(1e-12)
        q = torch.round(xf / scale).clamp(-qmax, qmax)
        scale = scale.reshape(1).to(torch.float32)
    elif use_gran == "group":
        D = xf.shape[-1]
        gs = max(1, int(group_size))
        ng = (D + gs - 1) // gs
        pad = ng * gs - D
        xg = torch.nn.functional.pad(xf, (0, pad)) if pad else xf
        xg = xg.reshape(*xf.shape[:-1], ng, gs)
        amax = xg.abs().amax(dim=-1, keepdim=True)
        scale_b = (amax / qmax).clamp_min(1e-12)
        qg = torch.round(xg / scale_b).clamp(-qmax, qmax)
        q = qg.reshape(*xf.shape[:-1], ng * gs)[..., :D]
        scale = scale_b.reshape(*xf.shape[:-1], ng).to(torch.float32)
        qmeta["group_size"] = gs
    else:
        red = tuple(range(1, xf.dim()))
        amax = xf.abs().amax(dim=red, keepdim=True)
        scale_b = (amax / qmax).clamp_min(1e-12)
        q = torch.round(xf / scale_b).clamp(-qmax, qmax)
        scale = scale_b.reshape(xf.shape[0]).to(torch.float32)

    if bits == 8:
        packed = q.to(torch.int8)
    else:
        packed = _pack_lowbit((q + qmax).to(torch.uint8), bits)

    return packed, scale, None, qmeta


def _torch_dtype(name):
    return getattr(torch, name.split(".")[-1])


def dequantize(packed, scale, zp, qmeta):
    """Reconstruct a float tensor from :func:`quantize` output."""
    if qmeta is None:
        return packed
    tag = qmeta["tag"]
    if tag in _FLOAT_CAST:
        ft = _FLOAT_CAST[tag]
        if _FLOAT8_DTYPES and ft in _FLOAT8_DTYPES and packed.dtype == torch.uint8:
            packed = packed.view(ft)
        return packed.to(_torch_dtype(qmeta["orig_dtype"]))

    bits = qmeta["bits"]
    qmax = (1 << (bits - 1)) - 1
    orig_shape = qmeta["orig_shape"]
    if bits == 8:
        qi = packed.to(torch.float32)
    else:
        qi = _unpack_lowbit(packed, bits, orig_shape[-1]).to(torch.float32) - qmax

    sc = scale.to(torch.float32)
    if qmeta["gran"] == "per_tensor":
        x = qi * sc.reshape(())
    elif qmeta["gran"] == "group":
        gs = qmeta["group_size"]
        sc_full = sc.repeat_interleave(gs, dim=-1)[..., :orig_shape[-1]]
        x = qi * sc_full
    else:
        x = qi * sc.reshape(sc.shape[0], *([1] * (qi.dim() - 1)))
    return x.to(_torch_dtype(qmeta["orig_dtype"]))


def quant_nbytes(*tensors):
    """Total resident bytes of a quantized artifact = packed + scale (+ zp)."""
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def quantize_into(entry, key, x, tag, gran="per_token", group_size=128):
    """Quantize ``x`` and write ``entry[key]`` (+ sibling scale/zp/qmeta keys)."""
    packed, scale, zp, qmeta = quantize(x, tag, gran, group_size)
    entry[key] = packed
    if qmeta is not None:
        entry[scale_key(key)] = scale
        entry[zp_key(key)] = zp
        entry[qmeta_key(key)] = qmeta
    return packed


def dequantize_entry(entry, key):
    """Return ``entry[key]`` dequantized to float, or unchanged if not quantized."""
    qmeta = entry.get(qmeta_key(key))
    v = entry.get(key)
    if qmeta is None or v is None:
        return v
    return dequantize(v, entry.get(scale_key(key)), entry.get(zp_key(key)), qmeta)


def dequantize_cache_inplace(modules, keys):
    """Dequantize an assembled cache section ``{module_name: entry}`` IN PLACE."""
    for entry in modules.values():
        if not isinstance(entry, dict):
            continue
        for key in keys:
            qm = entry.get(qmeta_key(key))
            if qm is None:
                continue
            vals = entry.get(key)
            sc = entry.get(scale_key(key))
            if isinstance(vals, torch.Tensor):
                vals = list(vals.unbind(0))
                sc = list(sc.unbind(0)) if isinstance(sc, torch.Tensor) else sc
            if vals is not None:
                entry[key] = [
                    None if t is None else
                    dequantize(t, (sc[i] if (sc is not None and i < len(sc)) else None), None, qm)
                    for i, t in enumerate(vals)
                ]
            entry.pop(scale_key(key), None)
            entry.pop(zp_key(key), None)
            entry.pop(qmeta_key(key), None)

