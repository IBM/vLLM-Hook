"""CPU round-trip + accounting tests for artifact_quant (no GPU / no engine).

Per-precision error gates: fp32 exact; fp16/bf16/fp8/int8 tight; int4/int2 advisory-coarse.
Run: pytest tests/use_cases/test_artifact_quant.py -vv
"""

import math

import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from mia.artifact_quant import (
    dequantize,
    dequantize_entry,
    parse_artifact_dtype,
    quant_nbytes,
    quantize,
    quantize_into,
    resolve_dtype,
    sibling_tensor_keys,
)

# max |Δ| / global-amax bound per tag (round error <= scale = amax/qmax; float tags relative)
_ERR = {
    "fp32": 0.0,
    "fp16": 2e-3,
    "bf16": 1e-2,
    "fp8_e4m3": 0.08,
    "fp8_e5m2": 0.30,
    "int8": 1.0 / 127 + 1e-4,
    "int4": 1.0 / 7 + 1e-4,
    "int2": 1.0 + 1e-4,
}


def _rand(shape, dtype=torch.float16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * 3.0).to(dtype)


@pytest.mark.parametrize("tag", list(_ERR))
@pytest.mark.parametrize("shape", [(16, 64), (7, 65), (1, 128), (33, 96)])
def test_roundtrip_error(tag, shape):
    x = _rand(shape)
    packed, scale, zp, qmeta = quantize(x, tag, gran="per_token")
    xr = dequantize(packed, scale, zp, qmeta)
    assert xr.shape == x.shape
    amax = x.abs().float().amax().item()
    err = (xr.float() - x.float()).abs().amax().item()
    assert err <= _ERR[tag] * amax + 1e-3, f"{tag} {shape}: err={err} amax={amax}"


@pytest.mark.parametrize("tag", ["int8", "int4", "int2"])
def test_per_tensor_roundtrip(tag):
    x = _rand((20, 48), seed=3)
    packed, scale, zp, qmeta = quantize(x, tag, gran="per_tensor")
    assert scale.numel() == 1
    xr = dequantize(packed, scale, zp, qmeta)
    amax = x.abs().float().amax().item()
    err = (xr.float() - x.float()).abs().amax().item()
    assert err <= _ERR[tag] * amax + 1e-3


def test_native_passthrough():
    x = _rand((8, 8))
    packed, scale, zp, qmeta = quantize(x, None)
    assert qmeta is None and scale is None
    assert torch.equal(packed, x)
    assert torch.equal(dequantize(packed, scale, zp, qmeta), x)


def test_subbyte_packing_shapes_and_dtype():
    x = _rand((5, 64))
    for tag, per_byte in (("int4", 2), ("int2", 4)):
        packed, scale, zp, qmeta = quantize(x, tag, gran="per_token")
        assert packed.dtype == torch.uint8
        assert packed.shape[0] == 5  # seq/row dim preserved (pad_sequence / _trim safe)
        assert packed.shape[-1] == math.ceil(64 / per_byte)
        assert scale.shape == (5,)


def _reference_pack(qi_u8, bits):
    """The pre-fusion reshape/zero-init/loop pack — the byte-for-byte oracle."""
    per_byte = 8 // bits
    L = qi_u8.shape[-1]
    pad = (-L) % per_byte
    if pad:
        qi_u8 = torch.nn.functional.pad(qi_u8, (0, pad))
    grp = qi_u8.reshape(*qi_u8.shape[:-1], qi_u8.shape[-1] // per_byte, per_byte)
    packed = torch.zeros(grp.shape[:-1], dtype=torch.uint8)
    for i in range(per_byte):
        packed |= grp[..., i] << (i * bits)
    return packed


@pytest.mark.parametrize("bits,vmax", [(4, 15), (2, 3)])
@pytest.mark.parametrize("shape", [(5, 64), (3, 65), (7, 66), (1, 128), (33, 96), (8, 4, 40)])
def test_fused_pack_byte_identical(bits, shape, vmax):
    from mia.artifact_quant import _pack_lowbit
    g = torch.Generator().manual_seed(5)
    qi = torch.randint(0, vmax + 1, shape, generator=g).to(torch.uint8)
    a = _reference_pack(qi.clone(), bits)
    b = _pack_lowbit(qi.clone(), bits)
    assert b.shape == a.shape and torch.equal(a, b)
    assert b.dtype == torch.uint8 and b.is_contiguous()


def test_odd_hidden_dim_pack_trim():
    # odd last dim exercises the pad-then-trim path
    x = _rand((3, 65))
    packed, scale, zp, qmeta = quantize(x, "int4", gran="per_token")
    xr = dequantize(packed, scale, zp, qmeta)
    assert xr.shape == (3, 65)


def test_quant_nbytes_shrinks():
    x = _rand((64, 128))  # fp16 = 64*128*2 = 16384 B
    native = x.numel() * x.element_size()
    for tag, factor in (("int8", 2), ("int4", 4), ("int2", 8)):
        packed, scale, zp, qmeta = quantize(x, tag, gran="per_token")
        qb = quant_nbytes(packed, scale, zp)
        # packed ~ native/factor; scale adds 64*4 B; still well under native
        assert qb < native
        assert packed.numel() * packed.element_size() <= native / factor + 1


def test_fp8_and_float_casts_preserve_shape():
    x = _rand((10, 32))
    for tag in ("fp8_e4m3", "fp8_e5m2", "bf16", "fp16", "fp32"):
        packed, scale, zp, qmeta = quantize(x, tag)
        assert scale is None and packed.shape == x.shape
        assert dequantize(packed, scale, zp, qmeta).shape == x.shape


def test_dequantize_entry_dict():
    x = _rand((12, 40), seed=7)
    entry = {"layer_num": 3}
    quantize_into(entry, "q", x, "int8", gran="per_token")
    assert "q_scale" in entry and "q_qmeta" in entry
    xr = dequantize_entry(entry, "q")
    amax = x.abs().float().amax().item()
    assert (xr.float() - x.float()).abs().amax().item() <= _ERR["int8"] * amax + 1e-3
    # native entry (no qmeta) round-trips unchanged
    entry2 = {"q": x}
    assert torch.equal(dequantize_entry(entry2, "q"), x)


@pytest.mark.parametrize("tag", ["int8", "int4", "int2"])
@pytest.mark.parametrize("shape,gs", [
    ((16, 256), 128),   # 2 groups, exact
    ((16, 256), 64),    # 4 groups, exact
    ((7, 130), 128),    # 2 groups, last group padded (2 real cols)
    ((5, 96), 128),     # 1 group (gs > D)  -> equivalent to per_tensor over the row's last dim
    ((8, 4, 64), 32),   # 3-D: group the head_dim (2 groups per head)
])
def test_group_roundtrip_error(tag, shape, gs):
    x = _rand(shape, seed=11)
    packed, scale, zp, qmeta = quantize(x, tag, gran="group", group_size=gs)
    assert qmeta["gran"] == "group" and qmeta["group_size"] == gs
    xr = dequantize(packed, scale, zp, qmeta)
    assert xr.shape == x.shape
    amax = x.abs().float().amax().item()
    err = (xr.float() - x.float()).abs().amax().item()
    assert err <= _ERR[tag] * amax + 1e-3, f"{tag} {shape} gs={gs}: err={err} amax={amax}"


def test_group_scale_shape_and_count():
    # [S, hidden] with gs=128: one scale per (token, group) -> [S, ceil(hidden/gs)]
    x = _rand((10, 320))
    _p, scale, _z, qmeta = quantize(x, "int4", gran="group", group_size=128)
    assert scale.shape == (10, 3)  # ceil(320/128) = 3 groups
    # 3-D artifact [S, H, d]: group the last dim -> [S, H, ceil(d/gs)]
    x3 = _rand((6, 4, 64))
    _p, scale3, _z, _q = quantize(x3, "int8", gran="group", group_size=32)
    assert scale3.shape == (6, 4, 2)


def test_group_at_least_as_accurate_as_per_token():
    # group subdivides each row, so its max error must not exceed per_token's.
    x = _rand((12, 512), seed=13)
    for tag in ("int8", "int4", "int2"):
        pt = dequantize(*quantize(x, tag, gran="per_token"))
        gr = dequantize(*quantize(x, tag, gran="group", group_size=64))
        e_pt = (pt.float() - x.float()).abs().amax().item()
        e_gr = (gr.float() - x.float()).abs().amax().item()
        assert e_gr <= e_pt + 1e-4, f"{tag}: group err {e_gr} > per_token {e_pt}"


def test_group_resolve_env(monkeypatch):
    from mia.artifact_quant import resolve_granularity, resolve_group_size
    monkeypatch.setenv("MIA_ARTIFACT_QUANT_GRAN", "group")
    monkeypatch.setenv("MIA_ARTIFACT_QUANT_GROUP_SIZE", "64")
    assert resolve_granularity() == "group"
    assert resolve_group_size() == 64


def test_group_cache_inplace_roundtrip():
    """1b-full path with group scales (multi-dim, per-pass list)."""
    from mia.artifact_quant import dequantize_cache_inplace
    xs = [_rand((6, 256), seed=1), _rand((9, 256), seed=2)]
    packed, scales, qmeta = [], [], None
    for x in xs:
        p, s, _z, qmeta = quantize(x, "int4", gran="group", group_size=128)
        packed.append(p)
        scales.append(s)
    modules = {"m0": {"hidden_states": packed, "hidden_states_scale": scales,
                      "hidden_states_qmeta": qmeta, "layer_num": 0}}
    dequantize_cache_inplace(modules, ("hidden_states",))
    assert "hidden_states_qmeta" not in modules["m0"]
    for xr, x in zip(modules["m0"]["hidden_states"], xs):
        amax = x.abs().float().amax().item()
        assert (xr.float() - x.float()).abs().amax().item() <= _ERR["int4"] * amax + 1e-3


def test_parse_and_resolve(monkeypatch):
    assert parse_artifact_dtype("i8") == "int8"
    assert parse_artifact_dtype("half") == "fp16"
    assert parse_artifact_dtype("none") is None
    assert parse_artifact_dtype(None) is None
    with pytest.raises(ValueError):
        parse_artifact_dtype("int3")
    monkeypatch.setenv("MIA_ARTIFACT_DTYPE", "int4")
    assert resolve_dtype("hs") == "int4"
    monkeypatch.setenv("MIA_ARTIFACT_DTYPE_HS", "int8")
    assert resolve_dtype("hs") == "int8"  # per-artifact override wins
    assert resolve_dtype("qk") == "int4"


def test_sibling_tensor_keys():
    ks = sibling_tensor_keys(("q", "k_all"))
    assert set(ks) == {"q_scale", "q_zp", "k_all_scale", "k_all_zp"}


def test_dequantize_cache_inplace_pt_roundtrip(tmp_path):
    """1b-full disk path: a quantized cache section (per-pass LIST of packed tensors +
    scale list + qmeta, as the worker emits) survives torch.save/load and dequantizes at
    the read boundary within the dtype bound; siblings are stripped."""
    import torch
    from mia.artifact_quant import quantize, dequantize_cache_inplace
    xs = [_rand((6, 48), seed=1), _rand((9, 48), seed=2)]  # two per-pass tensors
    packed, scales, qmeta = [], [], None
    for x in xs:
        p, s, _z, qmeta = quantize(x, "int4", gran="per_token")
        packed.append(p)
        scales.append(s)
    modules = {"m0": {"q": packed, "q_scale": scales, "q_qmeta": qmeta, "layer_num": 0}}
    path = str(tmp_path / "qk.pt")
    torch.save(modules, path)
    loaded = torch.load(path, map_location="cpu")
    dequantize_cache_inplace(loaded, ("q",))
    assert "q_qmeta" not in loaded["m0"] and "q_scale" not in loaded["m0"]
    for xr, x in zip(loaded["m0"]["q"], xs):
        amax = x.abs().float().amax().item()
        assert (xr.float() - x.float()).abs().amax().item() <= _ERR["int4"] * amax + 1e-3
    # native (no qmeta) is untouched
    native = {"m0": {"q": [xs[0]], "layer_num": 0}}
    dequantize_cache_inplace(native, ("q",))
    assert torch.equal(native["m0"]["q"][0], xs[0])
