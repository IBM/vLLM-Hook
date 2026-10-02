"""Set-aware aperture sizing (Task E3).

`resolve_aperture_bytes` layers a SET of enabled subsystems on top of the existing fixed
byte budget. The load-bearing claim is that this is a *generalization, not a retune*: with
one capture subsystem the number must be bit-for-bit what the single-kind path produced,
and `resolve_aperture_bytes_auto` -- which resolves the budget itself -- must be untouched.
Both are pinned below so a future tweak to the split cannot quietly move the single-subsystem
number, which every GPU-validated run to date was sized with.

Hermetic: pure arithmetic, no GPU, no engine.
"""
from __future__ import annotations

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from mia.graph.aperture_sizing import (
    CAPTURE_SUBSYSTEMS,
    DEFAULT_APERTURE_GPU_BYTES,
    KNOWN_SUBSYSTEMS,
    per_layer_token_bytes_hs,
    per_layer_token_bytes_qk,
    per_token_row_bytes,
    resolve_aperture_bytes,
    resolve_aperture_bytes_auto,
    resolve_aperture_bytes_fixed,
)

# Phi-3-mini-ish shape: the dims the split needs, in the brief's spelling.
DIMS = {"hidden": 3072, "q": 3072, "k": 1024, "dtype_bytes": 2, "layers": 32}

GIB = 1 << 30
BUDGET = 4 * GIB


# --------------------------------------------------------------------------------------
# The generalization contract: one subsystem sizes EXACTLY as before.
# --------------------------------------------------------------------------------------

def test_single_subsystem_matches_the_legacy_number():
    """One subsystem must size exactly as the pre-amendment code did.

    The legacy path was `resolve_aperture_bytes_auto` -> the whole fixed budget went to the
    process's single worker kind. So the pinned expectation is the budget itself, verbatim.
    """
    only_hs = resolve_aperture_bytes({"hs"}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
    assert set(only_hs) == {"hs"}
    assert only_hs["hs"] == BUDGET


def test_single_subsystem_is_the_budget_verbatim_for_every_capture_kind():
    for s in CAPTURE_SUBSYSTEMS:
        sizes = resolve_aperture_bytes({s}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
        assert sizes == {s: BUDGET}, s


def test_single_subsystem_needs_no_model_dims_at_all():
    """Nothing about the single-subsystem number is derived from the model -- the aperture is
    a fixed byte budget. Passing no dims must therefore still work, which is the sharpest
    possible statement that no retune crept in."""
    assert resolve_aperture_bytes({"qk"}, gpu_bytes_budget=BUDGET) == {"qk": BUDGET}


@pytest.mark.parametrize("budget", [1, 4096, DEFAULT_APERTURE_GPU_BYTES, 7 * GIB, 12345678901])
def test_single_subsystem_passes_any_budget_through_unchanged(budget):
    assert resolve_aperture_bytes(["hs"], gpu_bytes_budget=budget) == {"hs": budget}


def test_end_to_end_single_subsystem_equals_the_legacy_auto_number(monkeypatch):
    """The real composition the autocap guard performs: resolve the budget with the UNCHANGED
    `resolve_aperture_bytes_auto`, then split it over the enabled set. For one subsystem the
    two must be the same integer."""
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)
    total, util = 80 * GIB, 0.85
    legacy = resolve_aperture_bytes_auto(total, util)
    sizes = resolve_aperture_bytes({"hs"}, gpu_bytes_budget=legacy, model_dims=DIMS)
    assert sizes["hs"] == legacy == DEFAULT_APERTURE_GPU_BYTES


def test_resolve_aperture_bytes_auto_behaviour_is_unchanged(monkeypatch):
    """Pin the existing resolver itself: env override wins, default is 4 GiB, fit check raises."""
    total, util = 80 * GIB, 0.85
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)
    assert resolve_aperture_bytes_auto(total, util) == DEFAULT_APERTURE_GPU_BYTES
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(2 * GIB))
    assert resolve_aperture_bytes_auto(total, util) == 2 * GIB
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(60 * GIB))
    with pytest.raises(ValueError, match="no room"):
        resolve_aperture_bytes_auto(total, util)
    # and the fit check is still the one in _fixed
    assert resolve_aperture_bytes_fixed(total, 4 * GIB, util) == 4 * GIB


# --------------------------------------------------------------------------------------
# Several subsystems split one budget.
# --------------------------------------------------------------------------------------

def test_two_subsystems_split_one_budget_without_exceeding_it():
    both = resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
    assert set(both) == {"hs", "qk"}
    assert sum(both.values()) <= BUDGET
    assert all(v > 0 for v in both.values())


def test_the_split_is_in_the_ratio_of_per_token_row_bytes():
    """HS rows are hidden x dtype = 6144 B; QK rows are (q+k) x dtype = 8192 B. So the budget
    splits 3:4 -- the proportion each subsystem will actually consume."""
    both = resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
    assert both["hs"] == (BUDGET * 6144) // 14336
    assert both["qk"] == (BUDGET * 8192) // 14336
    assert both["qk"] > both["hs"]


def test_the_split_does_not_depend_on_set_iteration_order():
    a = resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
    b = resolve_aperture_bytes(["qk", "hs"], gpu_bytes_budget=BUDGET, model_dims=DIMS)
    c = resolve_aperture_bytes(("qk", "hs", "qk"), gpu_bytes_budget=BUDGET, model_dims=DIMS)
    assert a == b == c


def test_split_raises_rather_than_handing_a_subsystem_zero_bytes():
    """A 2-byte budget floors HS's 3/7 share to zero. A zero-size aperture would be a
    capture path that silently stores nothing, so it raises instead."""
    with pytest.raises(ValueError, match="too small"):
        resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=2, model_dims=DIMS)


def test_split_needs_dims_and_says_which(monkeypatch):
    with pytest.raises(ValueError, match="hidden"):
        resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=BUDGET,
                               model_dims={"q": 3072, "k": 1024, "dtype_bytes": 2})
    with pytest.raises(ValueError, match="n_q_heads|q'/'k"):
        resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=BUDGET,
                               model_dims={"hidden": 3072, "dtype_bytes": 2})


# --------------------------------------------------------------------------------------
# Steering owns no aperture.
# --------------------------------------------------------------------------------------

def test_steer_needs_no_aperture():
    sizes = resolve_aperture_bytes({"hs", "steer"}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
    assert "steer" not in sizes          # steering mutates the residual; it captures nothing
    assert sizes["hs"] == BUDGET


def test_steer_only_install_reserves_nothing():
    assert resolve_aperture_bytes({"steer"}, gpu_bytes_budget=BUDGET, model_dims=DIMS) == {}


def test_steer_does_not_dilute_a_two_capture_split():
    with_steer = resolve_aperture_bytes({"hs", "qk", "steer"}, gpu_bytes_budget=BUDGET,
                                        model_dims=DIMS)
    without = resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
    assert with_steer == without


def test_per_token_row_bytes_refuses_steer():
    with pytest.raises(KeyError):
        per_token_row_bytes("steer", DIMS)


# --------------------------------------------------------------------------------------
# Row-shape math is the EXISTING per-kind math, reached by subsystem name.
# --------------------------------------------------------------------------------------

def test_row_bytes_delegate_to_the_existing_per_kind_helpers():
    assert per_token_row_bytes("hs", DIMS) == per_layer_token_bytes_hs(3072, 2)
    assert per_token_row_bytes("qk", DIMS) == per_layer_token_bytes_qk(3072, 1024, 1, 2)


def test_qk_row_bytes_accept_the_head_count_form_the_plugin_reads():
    """_plugin reads head counts off hf_text_config; the brief's tests pass pre-multiplied
    widths. Both spellings must give the same number or the split would depend on the caller."""
    heads = {"hidden": 3072, "n_q_heads": 32, "n_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2}
    widths = {"hidden": 3072, "q": 32 * 128, "k": 8 * 128, "dtype_bytes": 2}
    assert per_token_row_bytes("qk", heads) == per_token_row_bytes("qk", widths)
    assert per_token_row_bytes("hs", heads) == per_token_row_bytes("hs", widths)


def test_dtype_bytes_default_is_two():
    assert per_token_row_bytes("hs", {"hidden": 100}) == 200


# --------------------------------------------------------------------------------------
# Fail loud on bad input.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["hidden_states", "HS", "attn", "", "steering"])
def test_unknown_subsystem_raises_and_names_the_known_ones(bad):
    with pytest.raises(ValueError) as e:
        resolve_aperture_bytes({bad}, gpu_bytes_budget=BUDGET, model_dims=DIMS)
    assert "unknown subsystem" in str(e.value)
    for known in KNOWN_SUBSYSTEMS:
        assert known in str(e.value)


@pytest.mark.parametrize("budget", [0, -1, -(1 << 30)])
def test_non_positive_budget_raises(budget):
    with pytest.raises(ValueError, match="positive"):
        resolve_aperture_bytes({"hs"}, gpu_bytes_budget=budget, model_dims=DIMS)


def test_empty_subsystem_set_reserves_nothing():
    assert resolve_aperture_bytes(set(), gpu_bytes_budget=BUDGET, model_dims=DIMS) == {}


# --------------------------------------------------------------------------------------
# The real consumer: the autocap guard in _plugin, driven set-aware.
#
# Ruling E-7 -- a set-aware resolver that nothing calls is dead code. These pin that the
# guard's SINGLE-KIND answer is bit-for-bit the number the pre-amendment single-kind code
# produced (the legacy formula is recomputed inline here from the same primitives), and
# that the set shape actually does something when more than one subsystem is enabled.
# --------------------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from mia._plugin import (  # noqa: E402
    _derive_safe_max_batched_tokens,
    _enabled_subsystems,
    _maybe_autocap_max_batched_tokens,
    _model_dims,
)
from mia.graph.aperture_sizing import (  # noqa: E402
    DEFAULT_AUTOCAP_HEADROOM_BYTES,
    DEFAULT_AUTOCAP_SAFETY,
    compute_safe_max_batched_tokens,
)

TOTAL_GPU = 80 * GIB
GPU_UTIL = 0.85


def _fake_config(max_batched=8192):
    """A vLLM engine config with just the fields the autocap guard reads (Phi-3-mini shape)."""
    text = SimpleNamespace(num_hidden_layers=32, hidden_size=3072,
                           num_attention_heads=32, num_key_value_heads=32, head_dim=96)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text, dtype="torch.bfloat16"),
        cache_config=SimpleNamespace(gpu_memory_utilization=GPU_UTIL),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_batched),
    )


@pytest.fixture
def nvml(monkeypatch):
    """Stub the NVML device-total read so this stays hermetic (no CUDA context)."""
    monkeypatch.setattr("vllm.platforms.current_platform",
                        SimpleNamespace(get_device_total_memory=lambda i: TOTAL_GPU),
                        raising=False)
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)
    monkeypatch.delenv("MIA_APERTURE_AUTOCAP_SAFETY", raising=False)
    monkeypatch.delenv("MIA_APERTURE_AUTOCAP_HEADROOM_BYTES", raising=False)


def _legacy_cap(per_layer_bytes):
    """The pre-amendment single-kind derivation, recomputed from the same primitives."""
    return compute_safe_max_batched_tokens(
        TOTAL_GPU, GPU_UTIL, DEFAULT_APERTURE_GPU_BYTES, 32, per_layer_bytes,
        safety=DEFAULT_AUTOCAP_SAFETY, headroom_bytes=DEFAULT_AUTOCAP_HEADROOM_BYTES)


def test_autocap_single_hs_is_bit_for_bit_the_legacy_number(nvml):
    got = _derive_safe_max_batched_tokens(_fake_config(), ["hidden_states"])
    assert got == _legacy_cap(per_layer_token_bytes_hs(3072, 2))
    assert got is not None and got > 0


def test_autocap_single_qk_is_bit_for_bit_the_legacy_number(nvml):
    got = _derive_safe_max_batched_tokens(_fake_config(), ["qk"])
    assert got == _legacy_cap(per_layer_token_bytes_qk(32, 32, 96, 2))
    assert got is not None and got > 0


def test_autocap_accepts_a_bare_string_exactly_as_it_did(nvml):
    """The call site now passes a list, but a bare worker kind must still size identically --
    nothing about the single-kind answer may depend on the container it arrives in."""
    cfg = _fake_config()
    assert (_derive_safe_max_batched_tokens(cfg, "qk")
            == _derive_safe_max_batched_tokens(cfg, ["qk"])
            == _derive_safe_max_batched_tokens(cfg, {"qk"}))


def test_autocap_steer_only_derives_nothing(nvml):
    assert _derive_safe_max_batched_tokens(_fake_config(), ["steer"]) is None


def test_autocap_mixed_set_is_tighter_than_either_alone(nvml):
    """A mixed install's per-step transient is the SUM of both row shapes, so the safe token
    budget must be strictly smaller than either subsystem's own. This is the whole reason the
    guard had to become set-aware rather than picking one kind."""
    cfg = _fake_config()
    hs = _derive_safe_max_batched_tokens(cfg, ["hidden_states"])
    qk = _derive_safe_max_batched_tokens(cfg, ["qk"])
    both = _derive_safe_max_batched_tokens(cfg, ["hidden_states", "qk"])
    assert both < min(hs, qk)


def test_autocap_guard_is_min_only_and_steer_is_a_no_op(nvml, monkeypatch):
    """End-to-end through the guard itself: it lowers a too-large budget for a capture set,
    leaves a already-small one alone, and never fires for steer."""
    monkeypatch.setenv("MIA_APERTURE_MAX_BATCHED_TOKENS", "auto")
    derived = _derive_safe_max_batched_tokens(_fake_config(), ["hidden_states"])

    big = _fake_config(max_batched=derived * 4)
    _maybe_autocap_max_batched_tokens(big, ["hidden_states"])
    assert big.scheduler_config.max_num_batched_tokens == derived

    small = _fake_config(max_batched=16)
    _maybe_autocap_max_batched_tokens(small, ["hidden_states"])
    assert small.scheduler_config.max_num_batched_tokens == 16      # min-only: untouched

    steer = _fake_config(max_batched=derived * 4)
    _maybe_autocap_max_batched_tokens(steer, ["steer"])
    assert steer.scheduler_config.max_num_batched_tokens == derived * 4


def test_autocap_guard_off_by_default(nvml, monkeypatch):
    monkeypatch.delenv("MIA_APERTURE_MAX_BATCHED_TOKENS", raising=False)
    cfg = _fake_config(max_batched=1 << 20)
    _maybe_autocap_max_batched_tokens(cfg, ["hidden_states"])
    assert cfg.scheduler_config.max_num_batched_tokens == 1 << 20


def test_worker_kinds_map_onto_the_registry_subsystem_vocabulary():
    assert _enabled_subsystems("hidden_states") == {"hs"}
    assert _enabled_subsystems("qk") == {"qk"}
    assert _enabled_subsystems("steer") == {"steer"}
    assert _enabled_subsystems(["hidden_states", "qk"]) == {"hs", "qk"}
    assert _enabled_subsystems([]) == set()
    assert _enabled_subsystems(None) == set()
    assert set(_enabled_subsystems(["hidden_states", "qk", "steer"])) <= set(KNOWN_SUBSYSTEMS)


# --------------------------------------------------------------------------------------
# Fix round 1, item 1: the guard must not FAIL OPEN on a config without attention heads.
# --------------------------------------------------------------------------------------

def _fake_config_without(*missing, max_batched=8192):
    """`_fake_config` with some hf_text_config attributes genuinely absent."""
    cfg = _fake_config(max_batched=max_batched)
    for name in missing:
        delattr(cfg.model_config.hf_text_config, name)
    return cfg


def test_hs_cap_is_still_derived_when_the_config_has_no_attention_heads(nvml):
    """An HS run needs `hidden_size` and nothing about attention. The pre-E3 code read
    `num_attention_heads` only inside its `if worker_kind == "qk"` branch; reading it
    unconditionally makes an HS run raise into `_maybe_autocap_max_batched_tokens`'s own
    `except Exception`, which prints a reason and leaves max_num_batched_tokens ALONE --
    the OOM guard silently stops guarding, on exactly the runs it exists to protect."""
    cfg = _fake_config_without("num_attention_heads", "num_key_value_heads", "head_dim")
    got = _derive_safe_max_batched_tokens(cfg, ["hidden_states"])
    assert got == _legacy_cap(per_layer_token_bytes_hs(3072, 2))


def test_the_guard_actually_lowers_the_budget_without_attention_heads(nvml, monkeypatch):
    """End-to-end through the guard, so a regression cannot hide in the except-clause."""
    monkeypatch.setenv("MIA_APERTURE_MAX_BATCHED_TOKENS", "auto")
    derived = _legacy_cap(per_layer_token_bytes_hs(3072, 2))
    cfg = _fake_config_without("num_attention_heads", "num_key_value_heads", "head_dim",
                               max_batched=1 << 20)
    _maybe_autocap_max_batched_tokens(cfg, ["hidden_states"])
    assert cfg.scheduler_config.max_num_batched_tokens == derived, (
        "guard failed open: max_num_batched_tokens left untouched")


def test_a_null_kv_head_count_does_not_fail_the_guard_open(nvml):
    """Same defect class one line down: `int(getattr(tc, 'num_key_value_heads', h_q))`
    raises TypeError on a config that carries the attribute set to None (an MHA model that
    never filled it in). Fall back to the query head count, as a missing attribute does."""
    cfg = _fake_config()
    cfg.model_config.hf_text_config.num_key_value_heads = None
    assert (_derive_safe_max_batched_tokens(cfg, ["qk"])
            == _legacy_cap(per_layer_token_bytes_qk(32, 32, 96, 2)))


def test_qk_without_attention_heads_still_refuses_loudly(nvml):
    """Making the dims lazy must not make QK sizing silently wrong: QK genuinely cannot be
    sized without head counts, so `per_token_row_bytes` must still raise -- and the guard
    then leaves the config alone rather than inventing a row shape."""
    cfg = _fake_config_without("num_attention_heads", "num_key_value_heads", "head_dim")
    with pytest.raises(ValueError, match="n_q_heads|q'/'k"):
        per_token_row_bytes("qk", _model_dims(cfg))


def test_the_pinned_cap_has_an_external_literal_anchor(nvml):
    """`_legacy_cap` shares `compute_safe_max_batched_tokens` with the implementation, so a
    future edit to that shared primitive would move BOTH sides of the differential pins above
    together and they would keep passing. These literals are the external anchor.

    Hand-computed from the fixture: 80 GiB total, gpu_memory_utilization 0.85, fixed 4 GiB
    aperture, 1 GiB headroom, safety 3, 32 layers, hidden 3072, 32 q + 32 kv heads x 96,
    2 bytes/element.

        free margin      = round(0.15 * 80 GiB) - 4 GiB - 1 GiB  = 7_516_192_768 B
        HS bytes/token   = 32 * (3072 * 2)                       =       196_608 B
        HS cap           = 7_516_192_768 // (196_608 * 3)        =        12_743
        QK bytes/token   = 32 * ((32 + 32) * 96 * 2)             =       393_216 B
        QK cap           = 7_516_192_768 // (393_216 * 3)        =         6_371

    If one of these numbers moves, that is a real change to how much GPU memory MIA reserves
    per step -- say why in the commit message. It is not a test-maintenance detail."""
    cfg = _fake_config()
    assert _derive_safe_max_batched_tokens(cfg, ["hidden_states"]) == 12_743
    assert _derive_safe_max_batched_tokens(cfg, ["qk"]) == 6_371
    # the same anchor through the resolver, so the budget itself is pinned to a literal too
    assert resolve_aperture_bytes({"hs"}, gpu_bytes_budget=4 * GIB) == {"hs": 4_294_967_296}
    assert resolve_aperture_bytes({"hs", "qk"}, gpu_bytes_budget=4 * GIB,
                                  model_dims=_model_dims(cfg)) == {"hs": 1_431_655_765,
                                                                   "qk": 2_863_311_530}
