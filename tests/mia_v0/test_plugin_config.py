"""Regression tests for the C3 engine-config policy: MIA requires vLLM's V2 model
runner and rejects any CUDA-graph mode it has not validated (PIECEWISE, and the
FULL_AND_PIECEWISE default vLLM 0.29 resolves to on an out-of-the-box engine).

See ``.superpowers/sdd/2026-09-16-mia-v2sup-port/task-C3-brief.md``. MEASURED on a GPU
node (LSF 1696603, MIA not loaded so vLLM alone decided): vLLM 0.29 selects
``use_v2_model_runner=True`` by default, and ``cudagraph_mode`` resolves to
``FULL_AND_PIECEWISE`` by default -- not ``FULL``. So the FULL_AND_PIECEWISE case is the
first thing most users hit and gets its own regression test below.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from mia._plugin import UnsupportedGraphModeError, validate_graph_mode


def test_piecewise_is_rejected_not_silently_downgraded():
    with pytest.raises(UnsupportedGraphModeError, match="PIECEWISE"):
        validate_graph_mode("PIECEWISE")


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_supported_modes_pass(mode):
    validate_graph_mode(mode)


def test_mia_never_forces_the_v1_runner():
    """The 0.21-era V1 forcing must be gone: MIA targets V2 only."""
    import mia._plugin as plugin
    source = open(plugin.__file__).read()
    assert 'VLLM_USE_V2_MODEL_RUNNER"] = "0"' not in source
    assert '"VLLM_USE_V2_MODEL_RUNNER"] = "0"' not in source


def test_default_full_and_piecewise_is_rejected_with_actionable_message():
    """MEASURED (LSF 1696603): vLLM 0.29's default resolved mode is FULL_AND_PIECEWISE,
    not FULL -- so this is the very first thing a default engine hits. The error must
    name the fix (FULL, and enforce_eager for the eager path), not just the complaint."""
    with pytest.raises(UnsupportedGraphModeError) as excinfo:
        validate_graph_mode("FULL_AND_PIECEWISE")
    message = str(excinfo.value)
    assert "FULL" in message
    assert "enforce_eager" in message


# ---------------------------------------------------------------------------
# Task D4 (GPU-measured, LSF 1701974): MIA's baked op is invisible to vLLM's
# compile-cache key.
#
# MIA installs its capture/steer op by class-wrapping the traced forward, and
# WHICH wrapper gets installed is decided by `worker_extension_cls` -- which vLLM
# 0.29 deliberately lists in `ParallelConfig.compute_hash`'s `ignored_factors`.
# So an HS graph run and a QK graph run of the same model hash to the same key
# while tracing different code, and the second one loads the first one's compiled
# artifact. Observed symptom: `KeyError: '_mia_hs_host'` raised from inside
# `self.aot_compiled_fn` during `profile_run` of a QK engine, on a machine where
# an HS engine had compiled earlier. `additional_config` IS a hash factor, so the
# worker kind is stamped there.
# ---------------------------------------------------------------------------


class _FakeEngineArgs:
    """Just the attributes the stamp touches: the config it writes into, and the TP degree
    the graph LAYOUT is resolved against."""

    def __init__(self, additional_config=None, tensor_parallel_size=1):
        self.additional_config = additional_config
        self.tensor_parallel_size = tensor_parallel_size


def test_compile_cache_stamp_distinguishes_worker_kinds():
    from mia._plugin import stamp_compile_cache_key

    stamps = {}
    for kind in ("hidden_states", "qk", "steer"):
        args = _FakeEngineArgs()
        stamp_compile_cache_key(args, kind)
        stamps[kind] = json.dumps(args.additional_config, sort_keys=True)

    assert len(set(stamps.values())) == 3, (
        f"every MIA worker kind must hash to a DIFFERENT compile-cache key, got {stamps}")


def test_compile_cache_stamp_is_json_serializable_and_idempotent():
    from mia._plugin import stamp_compile_cache_key

    args = _FakeEngineArgs()
    stamp_compile_cache_key(args, "qk")
    once = json.dumps(args.additional_config, sort_keys=True)  # the hash factor's own encoding
    stamp_compile_cache_key(args, "qk")
    assert json.dumps(args.additional_config, sort_keys=True) == once


def test_compile_cache_stamp_preserves_a_callers_additional_config():
    from mia._plugin import stamp_compile_cache_key

    original = {"user_key": {"a": 1}}
    args = _FakeEngineArgs(original)
    stamp_compile_cache_key(args, "steer")
    assert args.additional_config["user_key"] == {"a": 1}
    assert "mia_graph_capture" in args.additional_config
    # The caller's OWN dict is never mutated: they may still hold a reference to it, or
    # reuse it to build a second engine, and finding MIA's key in it would be a surprise.
    assert original == {"user_key": {"a": 1}}, "the caller's dict was mutated in place"
    assert args.additional_config is not original


def test_compile_cache_stamp_keys_on_mia_source_not_only_worker_kind():
    """Editing MIA changes the traced forward but moves nothing vLLM hashes.

    Without a MIA-source component the stale compiled artifact is silently reused -- a trap
    for profiling, where the numbers would then describe the previously compiled code.
    """
    from mia._plugin import _mia_source_id, stamp_compile_cache_key

    args = _FakeEngineArgs()
    stamp_compile_cache_key(args, "qk")
    stamp = args.additional_config["mia_graph_capture"]
    assert stamp["mia"], "the stamp carries no MIA source identifier"
    assert stamp["mia"] == _mia_source_id()
    # and it has to actually reach the hash factor's encoding
    assert stamp["mia"] in json.dumps(args.additional_config, sort_keys=True)


def test_compile_cache_stamp_warns_rather_than_mutating_a_non_dict(capsys):
    """A SupportsHash additional_config belongs to the caller. Silently sharing one
    cache entry across worker kinds would resurface as a KeyError from inside a
    compiled artifact, so the refusal has to be audible."""
    from mia._plugin import stamp_compile_cache_key

    sentinel = object()
    args = _FakeEngineArgs(sentinel)
    stamp_compile_cache_key(args, "qk")
    assert args.additional_config is sentinel
    assert "cannot stamp the compile-cache key" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The graph LAYOUT half of the stamp (mia/_plugin.py::mia_graph_layout).
# ---------------------------------------------------------------------------
# The worker kind is necessary and was not sufficient. MIA's capture buffers are
# operands of the baked op, so their SHAPES live in the compiled artifact -- and at one
# model, one TP degree and one worker kind, MIA env alone decides those shapes. GPU,
# LSF 1794720 (model-scale W8): a TP=4 `hs` leg compiled with 8 owned layers per rank
# (R=65536, hs_buf (65537, hidden), the other 24 layers 1-row sinks), then the
# `hs_replicas` leg (MIA_HS_CAPTURE_ALL_RANKS=1, 32 owned layers, R=16384) hashed to the
# SAME key, loaded that artifact, and died inside determine_available_memory() with
# inductor's `expected size 16385==1` and `16385==65537`.
# ---------------------------------------------------------------------------

_HS_LAYOUT_ENV = ("MIA_HS_TP_SHARD", "MIA_HS_CAPTURE_ALL_RANKS", "MIA_HS_TP_SYMMETRIC",
                  "MIA_HS_CAPTURE", "MIA_APERTURE_GPU_BYTES", "MIA_STEER_VMAX",
                  "MIA_STEER_MODE", "MIA_QK_CAPTURE")


@pytest.fixture
def clean_layout_env(monkeypatch):
    """No MIA layout variable inherited from the caller's shell."""
    for name in _HS_LAYOUT_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _stamp(tp=1, kind="hidden_states"):
    from mia._plugin import stamp_compile_cache_key
    args = _FakeEngineArgs(tensor_parallel_size=tp)
    stamp_compile_cache_key(args, kind)
    return json.dumps(args.additional_config, sort_keys=True)


def test_compile_cache_stamp_separates_the_hs_capture_layouts(clean_layout_env):
    """THE W8 REGRESSION. Four HS layouts at one model / TP / worker kind bake four
    different buffer geometries, so they must be four different cache keys."""
    layouts = {}
    layouts["round_robin"] = _stamp(tp=4)                           # 8 of 32 layers per rank
    clean_layout_env.setenv("MIA_HS_CAPTURE_ALL_RANKS", "1")
    layouts["all_ranks"] = _stamp(tp=4)                             # 32 of 32 on every rank
    clean_layout_env.delenv("MIA_HS_CAPTURE_ALL_RANKS")
    clean_layout_env.setenv("MIA_HS_TP_SHARD", "0")
    layouts["rank0"] = _stamp(tp=4)                                 # 32 on rank 0, 0 elsewhere
    clean_layout_env.delenv("MIA_HS_TP_SHARD")
    clean_layout_env.setenv("MIA_HS_TP_SYMMETRIC", "0")
    layouts["no_sinks"] = _stamp(tp=4)                              # unowned layers bake NO op

    assert len(set(layouts.values())) == 4, (
        "HS capture layouts that bake different buffers must not share a compile-cache key; "
        f"collided: {layouts}")


def test_compile_cache_stamp_moves_with_the_aperture_budget(clean_layout_env):
    """R = aperture_bytes // (owned_layers * width * elem) -- the aperture budget is a direct
    factor of every capture buffer's row count, in both capture paths, and vLLM hashes no
    part of it."""
    for kind in ("hidden_states", "qk"):
        auto = _stamp(tp=4, kind=kind)
        clean_layout_env.setenv("MIA_APERTURE_GPU_BYTES", str(4 * (1 << 30)))
        four_gib = _stamp(tp=4, kind=kind)
        clean_layout_env.setenv("MIA_APERTURE_GPU_BYTES", str(8 * (1 << 30)))
        eight_gib = _stamp(tp=4, kind=kind)
        clean_layout_env.delenv("MIA_APERTURE_GPU_BYTES")
        assert len({auto, four_gib, eight_gib}) == 3, (
            f"{kind}: apertures of different sizes give different R, so different buffers")


def test_compile_cache_stamp_moves_with_steer_v_max(clean_layout_env):
    """vec_table is (V_max, hidden) and avg_proj_table is (V_max,); both are steer_buffer
    operands, so V_max is a shape in the compiled graph."""
    default = _stamp(kind="steer")
    clean_layout_env.setenv("MIA_STEER_VMAX", "64")
    assert _stamp(kind="steer") != default


def test_compile_cache_stamp_is_stable_for_an_unchanged_layout(clean_layout_env):
    """A stamp that moves when nothing about the geometry moved costs a cold recompile on
    every launch. Unrelated MIA state -- where artifacts are written, which kernel the op
    body dispatches to, how long backpressure waits -- must leave the key alone."""
    before = _stamp(tp=4)
    assert _stamp(tp=4) == before, "the same configuration stamped twice must agree"
    for name, value in (("MIA_APERTURE_DIR", "/tmp/somewhere-else"),
                        ("MIA_CAPTURE_FUSED", "0"),
                        ("MIA_STEER_FUSED", "0"),
                        ("MIA_APERTURE_BACKPRESSURE_TIMEOUT_S", "70"),
                        ("MIA_WRITER_PROCESS", "0"),
                        ("MIA_APERTURE_WRITE_MODE", "auto"),
                        ("MIA_ROUTE_VECTORIZED", "1")):
        clean_layout_env.setenv(name, value)
        assert _stamp(tp=4) == before, f"{name} does not shape the graph; it must not rekey it"
        clean_layout_env.delenv(name)


def test_graph_layout_refuses_a_near_miss_hs_flag(clean_layout_env):
    """The layout is resolved through resolve_hs_shard_mode, so a flag spelling MIA refuses
    is refused HERE too -- at engine construction -- rather than quietly stamping one layout
    and then running another."""
    from mia._plugin import mia_graph_layout
    from mia.errors import MiaConfigurationError

    clean_layout_env.setenv("MIA_HS_CAPTURE_ALL_RANKS", "true")
    with pytest.raises(MiaConfigurationError, match="MIA_HS_CAPTURE_ALL_RANKS"):
        mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=4), "hidden_states")


def test_graph_layout_names_the_owned_layer_rule_and_the_sinks(clean_layout_env):
    """What the layout entry has to SAY, so a reader of a stamped key can tell which
    geometry compiled: the sharding rule, this rank's owned-layer set, the sink geometry,
    and the aperture budget that sets R."""
    from mia._plugin import mia_graph_layout

    layout = mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=4), "hidden_states")
    assert layout["hs_layout"] == "round_robin"
    assert layout["hs_layer_shard_rule"] == "round_robin"
    assert layout["hs_owned_layers"] == "layers i where i % 4 == tp_rank"
    assert layout["hs_sinks"] is True
    assert layout["aperture_gpu_bytes"] == "auto"
    assert layout["tp_size"] == 4
    # TP = 1 owns every layer and bakes no sink at all.
    single = mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=1), "hidden_states")
    assert single["hs_layout"] == "single" and single["hs_sinks"] is False


def test_graph_layout_is_json_native(clean_layout_env):
    """VllmConfig.compute_hash encodes additional_config with json.dumps(sort_keys=True); a
    value it cannot encode would raise inside vLLM's hashing, at engine init."""
    from mia._plugin import mia_graph_layout

    for kind in ("hidden_states", "qk", "steer"):
        layout = mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=4), kind)
        assert json.loads(json.dumps(layout, sort_keys=True)) == layout
