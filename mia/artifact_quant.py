"""On-GPU quantization of captured artifacts (QK q/k_all, hidden-states, scores).

Reduces the precision of captured tensors so the D2H copy, PCIe transfer, GPU residency,
host RAM, and disk footprint shrink under high concurrency. The design:

  * quantize on-GPU **at egress** (off the cudagraph replay path),
  * keep the artifact quantized end-to-end through host + disk / RPC,
  * dequantize lazily only inside the analyzer math.

Representation: the artifact stays a plain ``torch.Tensor`` under its existing probe key
(native ``int8`` / ``float8`` / float, or ``uint8``-bit-packed for sub-byte int4/int2);
sibling ``<key>_scale`` / ``<key>_zp`` tensors + a small non-tensor ``<key>_qmeta`` dict
carry the reconstruction params. NEVER nest a dict under the primary key — that breaks the
merge / serialize paths (``llm``/``client``).

Fixed operator knob ``MIA_ARTIFACT_DTYPE`` in
``{int2,int4,int8,fp8_e4m3,fp8_e5m2,bf16,fp16,fp32}``. Default = native (off,
byte-identical to today). Steering is out of scope (it has no artifacts).

Integer quant is symmetric-signed (zero-point unused / ``None``) — correct for the roughly
zero-centred q/k/hidden-state activations. Scores (``[0,1]``) are better left native or
quantized asymmetric; that path is deferred.
"""

import os

import torch

# ---- dtype tags --------------------------------------------------------------

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

# float8 tensors do NOT survive pickle / torch.load in this torch (the storage reducer raises
# "UntypedStorage has no attribute 'dtype'"), which breaks the RPC (pickle) and .pt (torch.save)
# hops. So a float8 packed tensor is transported as its uint8 bit-pattern (same 1 byte, fully
# serializable) and bitcast back in dequantize. Same-size, value-preserving.
_FLOAT8_DTYPES = tuple(d for d in (getattr(torch, "float8_e4m3fn", None),
                                   getattr(torch, "float8_e5m2", None)) if d is not None)

_SCALE_SUFFIX = "_scale"
_ZP_SUFFIX = "_zp"
_QMETA_SUFFIX = "_qmeta"
# tensor-valued sibling suffixes (added to llm TENSOR_KEYS so they merge per-request);
# _qmeta is a small non-tensor dict and stays plain metadata.
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


# ---- knob resolution ---------------------------------------------------------

def parse_artifact_dtype(value):
    """Normalize an operator-supplied dtype string to a canonical tag, or ``None`` (native/off).

    Raises ``ValueError`` on an unrecognized tag or an unavailable fp8 build."""
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
    """Resolve the active tag for an artifact family from env: per-artifact override,
    else the global ``MIA_ARTIFACT_DTYPE``. Returns a canonical tag or ``None``."""
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


# ---- bit packing (sub-byte int4/int2 along the LAST dim) ---------------------

def _pack_lowbit(qi_u8, bits):
    """Pack an unsigned uint8 tensor (values in ``[0, 2**bits-1]``) along the last dim,
    ``8//bits`` values per output byte. Pads the last dim; caller records the original
    length in qmeta for unpack.

    Fused form: OR the ``per_byte`` **strided sub-lanes** (``qi[..., i::per_byte]``), each
    shifted into its bit-field, instead of a reshape + zero-init + per-field accumulate loop.
    This is a per-layer, per-step op on the capture critical path, so kernel-launch count
    matters: one shift + one OR per field, uint8 throughout (no int32/int64 promotion, no
    zero-init, no group reshape). Byte-identical to the reshape/loop form. The plugin
    runs the eager capture path with TORCHDYNAMO_DISABLE=1, so torch.compile can't fuse this
    further; a Triton kernel is the path to a single-launch pack."""
    per_byte = 8 // bits
    L = qi_u8.shape[-1]
    pad = (-L) % per_byte
    if pad:
        qi_u8 = torch.nn.functional.pad(qi_u8, (0, pad))
    qi_u8 = qi_u8.contiguous()
    packed = qi_u8[..., 0::per_byte]  # lane 0 = low bit-field (no shift)
    for i in range(1, per_byte):
        packed = packed | (qi_u8[..., i::per_byte] << (i * bits))
    return packed.contiguous()


def _unpack_lowbit(packed, bits, orig_last):
    """Inverse of :func:`_pack_lowbit`; returns a uint8 tensor of values in ``[0, 2**bits-1]``
    trimmed back to ``orig_last`` on the last dim."""
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    fields = [(packed >> (i * bits)) & mask for i in range(per_byte)]
    out = torch.stack(fields, dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * per_byte)
    return out[..., :orig_last]


# ---- quantize / dequantize ---------------------------------------------------

def quantize(x, tag, gran="per_token", group_size=128):
    """Quantize ``x`` to ``tag``. Returns ``(packed, scale, zp, qmeta)``.

    ``tag=None`` is a no-op passthrough. Float tags cast only (``scale=zp=None``). Integer
    tags return a packed tensor + a float32 scale (``zp=None``, symmetric). ``qmeta`` is a
    small non-tensor dict for the disk/metadata path."""
    if tag is None:
        return x, None, None, None

    if tag in _FLOAT_CAST:
        packed = x.to(_FLOAT_CAST[tag])
        if _FLOAT8_DTYPES and packed.dtype in _FLOAT8_DTYPES:
            packed = packed.view(torch.uint8)  # transport-safe (float8 fails pickle/torch.load)
        return packed, None, None, {"tag": tag, "gran": "none", "orig_dtype": str(x.dtype)}

    bits = _INT_BITS[tag]
    qmax = (1 << (bits - 1)) - 1  # symmetric: int8->127, int4->7, int2->1
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
    elif use_gran == "group":  # one scale per contiguous group along the LAST dim
        D = xf.shape[-1]
        gs = max(1, int(group_size))
        ng = (D + gs - 1) // gs
        pad = ng * gs - D
        xg = torch.nn.functional.pad(xf, (0, pad)) if pad else xf
        xg = xg.reshape(*xf.shape[:-1], ng, gs)
        amax = xg.abs().amax(dim=-1, keepdim=True)          # [..., ng, 1]
        scale_b = (amax / qmax).clamp_min(1e-12)
        qg = torch.round(xg / scale_b).clamp(-qmax, qmax)   # [..., ng, gs]
        q = qg.reshape(*xf.shape[:-1], ng * gs)[..., :D]    # trim the group pad
        scale = scale_b.reshape(*xf.shape[:-1], ng).to(torch.float32)
        qmeta["group_size"] = gs
    else:  # per_token: one scale per row (dim 0)
        red = tuple(range(1, xf.dim()))
        amax = xf.abs().amax(dim=red, keepdim=True)
        scale_b = (amax / qmax).clamp_min(1e-12)
        q = torch.round(xf / scale_b).clamp(-qmax, qmax)
        scale = scale_b.reshape(xf.shape[0]).to(torch.float32)

    if bits == 8:
        packed = q.to(torch.int8)
    else:
        packed = _pack_lowbit((q + qmax).to(torch.uint8), bits)  # offset to unsigned

    return packed, scale, None, qmeta


def _torch_dtype(name):
    return getattr(torch, name.split(".")[-1])


def dequantize(packed, scale, zp, qmeta):
    """Reconstruct a float tensor from :func:`quantize` output. No-op if ``qmeta is None``
    (native/unquantized). Returns a tensor in the original (model) dtype."""
    if qmeta is None:
        return packed
    tag = qmeta["tag"]
    if tag in _FLOAT_CAST:
        ft = _FLOAT_CAST[tag]
        if _FLOAT8_DTYPES and ft in _FLOAT8_DTYPES and packed.dtype == torch.uint8:
            packed = packed.view(ft)  # bitcast the transported uint8 back to float8
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
    elif qmeta["gran"] == "group":  # sc is [..., ng]; expand each group over its columns
        gs = qmeta["group_size"]
        sc_full = sc.repeat_interleave(gs, dim=-1)[..., :orig_shape[-1]]
        x = qi * sc_full
    else:  # per_token
        x = qi * sc.reshape(sc.shape[0], *([1] * (qi.dim() - 1)))
    return x.to(_torch_dtype(qmeta["orig_dtype"]))


def quant_nbytes(*tensors):
    """Total resident bytes of a quantized artifact = packed + scale (+ zp). Use for the
    drain / residency / gauge accounting so quantized sizes are tracked exactly."""
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


# ---- dict helpers (used by workers at store-time and analyzers at read-time) ---

def quantize_into(entry, key, x, tag, gran="per_token", group_size=128):
    """Quantize ``x`` and write ``entry[key]`` (+ sibling scale/zp/qmeta keys). Returns the
    packed tensor. With ``tag=None`` this stores ``x`` unchanged (native)."""
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
    """Dequantize an assembled cache section ``{module_name: entry}`` IN PLACE.

    This is the analysis-side dequant boundary: the worker hands artifacts off quantized
    (per-pass LIST of packed tensors + parallel ``key_scale`` list + ``key_qmeta``) across the
    RPC/disk hop, and the driver calls this right before the analyzer consumes so the artifact
    only expands to float at analysis time. For each ``entry`` and each ``key`` in ``keys``
    (e.g. ``("q","k_all")`` or ``("hidden_states",)``) that carries ``key_qmeta``, replace
    ``entry[key]`` with a float list (dequantizing each element by its aligned scale) and drop
    the ``key_scale``/``key_zp``/``key_qmeta`` siblings. A stacked tensor is unbound first
    (defensive; the eager-quant worker emits lists). No-op for native (unquantized) entries, so
    calling it on a fully-native cache is byte-identical to not calling it."""
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
                    dequantize(t, (sc[i] if (sc is not None and i < len(sc)) else None), None, qm)
                    for i, t in enumerate(vals)
                ]
            entry.pop(scale_key(key), None)
            entry.pop(zp_key(key), None)
            entry.pop(qmeta_key(key), None)
