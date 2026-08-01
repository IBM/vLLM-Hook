# tests/core/test_schema.py
"""Accept/reject tables for spec parsing; E_* codes and JSON paths are
asserted verbatim — they are part of the wire contract.
"""
import pytest

from vllm_hook_plugins.core.kinds import BASE_GATE_KINDS, GATE_KINDS
from vllm_hook_plugins.core.schema import (
    MAX_SPEC_BYTES,
    SpecError,
    parse_capture,
    parse_intervention_spec,
    parse_processor_spec,
)

VEC = "sha256:" + "ab" * 32
PROBE = "sha256:" + "cd" * 32

NUM_LAYERS = 16


def _op(**overrides):
    op = {
        "layers": [3],
        "transform": {"kind": "additive", "strength": 2.0, "artifact": VEC},
        "scope": {"kind": "all"},
        "gate": None,
    }
    op.update(overrides)
    return op


def _parse(spec_obj, allowed_gates=GATE_KINDS):
    return parse_intervention_spec(spec_obj, num_layers=NUM_LAYERS, allowed_gates=allowed_gates)


def _reject(spec_obj, code, path, allowed_gates=GATE_KINDS):
    with pytest.raises(SpecError) as excinfo:
        _parse(spec_obj, allowed_gates=allowed_gates)
    assert excinfo.value.code == code
    assert excinfo.value.path == path
    return excinfo.value


# ---------------------------------------------------------------------------
# Accepts
# ---------------------------------------------------------------------------


def test_accepts_full_spec():
    spec = _parse({
        "ops": [
            _op(),
            {
                "layers": [9, 10],
                "transform": {
                    "kind": "rotation",
                    "angle": 0.35,
                    "modifiers": [{"kind": "alignment_adaptive", "artifact": PROBE}],
                    "artifact": PROBE,
                },
                "scope": {"kind": "after_prompt"},
                "gate": {
                    "kind": "cache_once",
                    "inner": {
                        "kind": "multi_key_threshold",
                        "threshold": 0.7,
                        "condition_layers": [6],
                        "artifact": PROBE,
                    },
                },
            },
        ]
    })
    assert spec.ops[0].transform_kind == "additive"
    assert spec.ops[0].transform_params == {"strength": 2.0}
    assert spec.ops[1].layers == (9, 10)
    assert spec.ops[1].gate.kind == "cache_once"
    assert spec.ops[1].gate.inner.kind == "multi_key_threshold"
    assert spec.artifact_ids() == (VEC, PROBE)
    assert spec.layers() == frozenset({3, 9, 10})
    assert spec.condition_layers() == frozenset({6})


def test_accepts_all_scope_kinds():
    for scope in (
        {"kind": "all"},
        {"kind": "after_prompt"},
        {"kind": "last_k", "k": 4},
        {"kind": "from_position", "position": 0},
    ):
        spec = _parse({"ops": [_op(scope=scope)]})
        assert spec.ops[0].scope.kind == scope["kind"]


def test_null_gate_object_means_no_gate():
    spec = _parse({"ops": [_op(gate={"kind": "null"})]})
    assert spec.ops[0].gate is None


def test_int_accepted_for_float_params():
    spec = _parse({"ops": [_op(transform={"kind": "additive", "strength": 2, "artifact": VEC})]})
    assert spec.ops[0].transform_params["strength"] == 2.0


def test_empty_ops_is_valid():
    assert _parse({"ops": []}).ops == ()


# ---------------------------------------------------------------------------
# Rejects: kinds and fields
# ---------------------------------------------------------------------------


def test_unknown_transform_kind():
    _reject(
        {"ops": [_op(transform={"kind": "multiply", "artifact": VEC})]},
        "E_UNKNOWN_KIND", "ops[0].transform.kind",
    )


def test_unknown_scope_kind():
    _reject({"ops": [_op(scope={"kind": "everywhere"})]}, "E_UNKNOWN_KIND", "ops[0].scope.kind")


def test_unknown_gate_kind():
    _reject({"ops": [_op(gate={"kind": "always"})]}, "E_UNKNOWN_KIND", "ops[0].gate.kind")


def test_unknown_modifier_kind():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 1.0, "artifact": VEC,
                                "modifiers": [{"kind": "clamp"}]})]},
        "E_UNKNOWN_KIND", "ops[0].transform.modifiers[0].kind",
    )


def test_unknown_top_level_field():
    _reject({"ops": [], "version": 2}, "E_UNKNOWN_FIELD", "version")


def test_unknown_op_field():
    _reject({"ops": [_op(when="always")]}, "E_UNKNOWN_FIELD", "ops[0].when")


def test_unknown_transform_param():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 1.0, "alpha": 2.0, "artifact": VEC})]},
        "E_UNKNOWN_FIELD", "ops[0].transform.alpha",
    )


def test_gate_kind_not_served_without_conditional_handler():
    err = _reject(
        {"ops": [_op(gate={"kind": "probe_sum", "threshold": 0.5, "condition_layers": [2],
                           "artifact": PROBE})]},
        "E_UNKNOWN_KIND", "ops[0].gate.kind",
        allowed_gates=BASE_GATE_KINDS,
    )
    assert "conditional handler" in err.msg


# ---------------------------------------------------------------------------
# Rejects: params
# ---------------------------------------------------------------------------


def test_missing_required_param():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "artifact": VEC})]},
        "E_BAD_PARAM", "ops[0].transform.strength",
    )


def test_wrong_param_type():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": "big", "artifact": VEC})]},
        "E_BAD_PARAM", "ops[0].transform.strength",
    )


def test_huge_int_param_rejected_not_overflowed():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 10 ** 400, "artifact": VEC})]},
        "E_BAD_PARAM", "ops[0].transform.strength",
    )


def test_non_finite_param_rejected():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": float("inf"), "artifact": VEC})]},
        "E_BAD_PARAM", "ops[0].transform.strength",
    )


def test_unhashable_kind_values_reject_cleanly():
    _reject(
        {"ops": [_op(transform={"kind": ["additive"], "artifact": VEC})]},
        "E_UNKNOWN_KIND", "ops[0].transform.kind",
    )
    _reject({"ops": [_op(scope={"kind": {"k": 1}})]}, "E_UNKNOWN_KIND", "ops[0].scope.kind")
    with pytest.raises(SpecError) as excinfo:
        parse_capture({"layers": "all", "mode": ["all_tokens"]}, num_layers=NUM_LAYERS)
    assert excinfo.value.code == "E_BAD_PARAM"


def test_bool_rejected_for_numeric_param():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": True, "artifact": VEC})]},
        "E_BAD_PARAM", "ops[0].transform.strength",
    )


def test_last_k_requires_positive_k():
    _reject({"ops": [_op(scope={"kind": "last_k", "k": 0})]}, "E_BAD_PARAM", "ops[0].scope.k")


def test_from_position_requires_nonnegative_position():
    _reject(
        {"ops": [_op(scope={"kind": "from_position", "position": -1})]},
        "E_BAD_PARAM", "ops[0].scope.position",
    )


def test_missing_op_fields():
    _reject({"ops": [{"transform": {"kind": "additive", "strength": 1.0, "artifact": VEC},
                      "scope": {"kind": "all"}}]},
            "E_BAD_PARAM", "ops[0].layers")


# ---------------------------------------------------------------------------
# Rejects: layers
# ---------------------------------------------------------------------------


def test_layer_out_of_range():
    _reject({"ops": [_op(layers=[NUM_LAYERS])]}, "E_LAYER_RANGE", "ops[0].layers[0]")
    _reject({"ops": [_op(layers=[-1])]}, "E_LAYER_RANGE", "ops[0].layers[0]")


def test_condition_layer_out_of_range():
    _reject(
        {"ops": [_op(gate={"kind": "probe_sum", "threshold": 0.5,
                           "condition_layers": [NUM_LAYERS], "artifact": PROBE})]},
        "E_LAYER_RANGE", "ops[0].gate.condition_layers[0]",
    )


def test_duplicate_layers_rejected():
    _reject({"ops": [_op(layers=[3, 3])]}, "E_BAD_PARAM", "ops[0].layers[1]")


def test_empty_layers_rejected():
    _reject({"ops": [_op(layers=[])]}, "E_BAD_PARAM", "ops[0].layers")


# ---------------------------------------------------------------------------
# Rejects: artifacts
# ---------------------------------------------------------------------------


def test_malformed_artifact_id():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 1.0, "artifact": "sha256:zz"})]},
        "E_BAD_PARAM", "ops[0].transform.artifact",
    )
    # $ would accept a trailing newline; the pattern must not
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 1.0, "artifact": VEC + "\n"})]},
        "E_BAD_PARAM", "ops[0].transform.artifact",
    )


def test_missing_required_artifact():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 1.0})]},
        "E_BAD_PARAM", "ops[0].transform.artifact",
    )


def test_artifact_on_artifactless_kind():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 1.0, "artifact": VEC,
                                "modifiers": [{"kind": "norm_preserving", "artifact": VEC}]})]},
        "E_BAD_PARAM", "ops[0].transform.modifiers[0].artifact",
    )


# ---------------------------------------------------------------------------
# Rejects: gates
# ---------------------------------------------------------------------------


def test_cache_once_requires_inner():
    _reject({"ops": [_op(gate={"kind": "cache_once"})]}, "E_BAD_PARAM", "ops[0].gate.inner")


def test_cache_once_rejects_null_inner():
    _reject(
        {"ops": [_op(gate={"kind": "cache_once", "inner": {"kind": "null"}})]},
        "E_BAD_PARAM", "ops[0].gate.inner",
    )


def test_cache_once_cannot_nest():
    _reject(
        {"ops": [_op(gate={"kind": "cache_once",
                           "inner": {"kind": "cache_once",
                                     "inner": {"kind": "null"}}})]},
        "E_BAD_PARAM", "ops[0].gate.inner.kind",
    )


# ---------------------------------------------------------------------------
# Rejects: size
# ---------------------------------------------------------------------------


def test_spec_too_large():
    from vllm_hook_plugins.core.canonical import canonical_bytes

    ops = []
    while len(canonical_bytes({"ops": ops})) <= MAX_SPEC_BYTES:
        ops.append(_op())
    with pytest.raises(SpecError) as excinfo:
        _parse({"ops": ops})
    assert excinfo.value.code == "E_SPEC_TOO_LARGE"


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def test_capture_accepts_layers_list_and_all():
    spec = parse_capture({"layers": [0, 5], "mode": "last_token", "location": "layer_input"},
                         num_layers=NUM_LAYERS)
    assert spec.layers == (0, 5)
    assert spec.mode == "last_token"
    assert spec.location == "layer_input"
    assert spec.kind == "residual"

    spec = parse_capture({"layers": "all"}, num_layers=NUM_LAYERS)
    assert spec.layers is None
    assert spec.mode == "all_tokens"
    assert spec.location == "layer_output"


def test_capture_rejections():
    with pytest.raises(SpecError) as excinfo:
        parse_capture({"layers": "all", "granularity": "head"}, num_layers=NUM_LAYERS)
    assert (excinfo.value.code, excinfo.value.path) == ("E_UNKNOWN_FIELD", "granularity")

    with pytest.raises(SpecError) as excinfo:
        parse_capture({"layers": "all", "mode": "first_token"}, num_layers=NUM_LAYERS)
    assert (excinfo.value.code, excinfo.value.path) == ("E_BAD_PARAM", "mode")

    with pytest.raises(SpecError) as excinfo:
        parse_capture({"layers": "all", "location": "logits"}, num_layers=NUM_LAYERS)
    assert (excinfo.value.code, excinfo.value.path) == ("E_BAD_PARAM", "location")

    with pytest.raises(SpecError) as excinfo:
        parse_capture({"layers": [NUM_LAYERS]}, num_layers=NUM_LAYERS)
    assert (excinfo.value.code, excinfo.value.path) == ("E_LAYER_RANGE", "layers[0]")

    with pytest.raises(SpecError) as excinfo:
        parse_capture({"mode": "all_tokens"}, num_layers=NUM_LAYERS)
    assert (excinfo.value.code, excinfo.value.path) == ("E_BAD_PARAM", "layers")


# ---------------------------------------------------------------------------
# processor_spec
# ---------------------------------------------------------------------------


def test_processor_spec_rejected_until_handler_lands():
    with pytest.raises(SpecError) as excinfo:
        parse_processor_spec({"kind": "constraint"}, allowed_processors=frozenset())
    assert (excinfo.value.code, excinfo.value.path) == ("E_UNKNOWN_KIND", "kind")


# ---------------------------------------------------------------------------
# error payloads
# ---------------------------------------------------------------------------


def test_spec_error_payload_and_str():
    err = _reject({"ops": [_op(layers=[NUM_LAYERS])]}, "E_LAYER_RANGE", "ops[0].layers[0]")
    assert err.payload() == {"code": err.code, "path": err.path, "msg": err.msg}
    assert "E_LAYER_RANGE" in str(err)
    assert "ops[0].layers[0]" in str(err)
