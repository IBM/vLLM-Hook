# tests/core/test_schema.py
"""Accept/reject tables for spec parsing; E_* codes and JSON paths are
asserted verbatim — they are part of the wire contract.
"""
import pytest

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


def _gate(**overrides):
    gate = {
        "layers": [6],
        "pooling": "mean",
        "readout": {"kind": "affine", "artifact": PROBE},
        "rule": {"kind": "sum_threshold", "bias": -1.0},
    }
    gate.update(overrides)
    return gate


def _parse(spec_obj):
    return parse_intervention_spec(spec_obj, num_layers=NUM_LAYERS)


def _reject(spec_obj, code, path):
    with pytest.raises(SpecError) as excinfo:
        _parse(spec_obj)
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
                    "mode": "target",
                    "modifiers": [{
                        "kind": "alignment_adaptive",
                        "threshold": 0.0,
                        "use_cosine": False,
                        "artifact": PROBE,
                    }],
                    "artifact": PROBE,
                },
                "scope": {"kind": "after_prompt"},
                "gate": _gate(
                    layers=[6, 7],
                    pooling="last",
                    readout={"kind": "cosine", "artifact": PROBE},
                    rule={"kind": "per_key_threshold", "threshold": 0.7,
                          "comparator": "ge", "aggregate": "all"},
                ),
            },
        ]
    })
    assert spec.ops[0].transform_kind == "additive"
    assert spec.ops[0].transform_params == {"strength": 2.0}
    assert spec.ops[0].gate is None
    assert spec.ops[1].layers == (9, 10)
    gate = spec.ops[1].gate
    assert gate.layers == (6, 7)
    assert gate.pooling == "last"
    assert gate.readout.kind == "cosine"
    assert gate.readout.artifact == PROBE
    assert gate.rule.kind == "per_key_threshold"
    assert gate.rule.params == {"threshold": 0.7, "comparator": "ge", "aggregate": "all"}
    assert spec.artifact_ids() == (VEC, PROBE)
    assert spec.layers() == frozenset({3, 9, 10})
    assert spec.condition_layers() == frozenset({6, 7})


def test_accepts_all_scope_kinds():
    for scope in (
        {"kind": "all"},
        {"kind": "after_prompt"},
        {"kind": "last_k", "k": 4},
        {"kind": "from_position", "position": 0},
    ):
        spec = _parse({"ops": [_op(scope=scope)]})
        assert spec.ops[0].scope.kind == scope["kind"]


def test_accepts_every_readout_kind():
    for readout in (
        {"kind": "affine", "artifact": PROBE},
        {"kind": "cosine", "artifact": PROBE},
        {"kind": "projected_cosine", "artifact": PROBE},
    ):
        spec = _parse({"ops": [_op(gate=_gate(readout=readout))]})
        assert spec.ops[0].gate.readout.kind == readout["kind"]
        assert spec.ops[0].gate.readout.artifact == PROBE


def test_absent_gate_means_ungated():
    spec = _parse({"ops": [_op(gate=None)]})
    assert spec.ops[0].gate is None
    # A gate key that is simply omitted is equivalent to an explicit None.
    op = _op()
    del op["gate"]
    assert _parse({"ops": [op]}).ops[0].gate is None


def test_int_accepted_for_float_params():
    spec = _parse({"ops": [_op(transform={"kind": "additive", "strength": 2, "artifact": VEC})]})
    assert spec.ops[0].transform_params["strength"] == 2.0


def test_int_accepted_for_rule_float_param():
    spec = _parse({"ops": [_op(gate=_gate(rule={"kind": "sum_threshold", "bias": -3}))]})
    assert spec.ops[0].gate.rule.params["bias"] == -3.0


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


def test_unknown_readout_kind():
    _reject(
        {"ops": [_op(gate=_gate(readout={"kind": "logistic", "artifact": PROBE}))]},
        "E_UNKNOWN_KIND", "ops[0].gate.readout.kind",
    )


def test_unknown_rule_kind():
    _reject(
        {"ops": [_op(gate=_gate(rule={"kind": "majority_vote"}))]},
        "E_UNKNOWN_KIND", "ops[0].gate.rule.kind",
    )


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


def test_unknown_gate_field():
    _reject({"ops": [_op(gate=_gate(foo=1))]}, "E_UNKNOWN_FIELD", "ops[0].gate.foo")


def test_unknown_transform_param():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": 1.0, "alpha": 2.0, "artifact": VEC})]},
        "E_UNKNOWN_FIELD", "ops[0].transform.alpha",
    )


# ---------------------------------------------------------------------------
# Rejects: params
# ---------------------------------------------------------------------------


def test_missing_required_param():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "artifact": VEC})]},
        "E_BAD_PARAM", "ops[0].transform.strength",
    )


def test_missing_required_rule_param():
    _reject(
        {"ops": [_op(gate=_gate(rule={"kind": "sum_threshold"}))]},
        "E_BAD_PARAM", "ops[0].gate.rule.bias",
    )


def test_wrong_param_type():
    _reject(
        {"ops": [_op(transform={"kind": "additive", "strength": "big", "artifact": VEC})]},
        "E_BAD_PARAM", "ops[0].transform.strength",
    )


def test_wrong_rule_param_type():
    _reject(
        {"ops": [_op(gate=_gate(rule={"kind": "sum_threshold", "bias": "low"}))]},
        "E_BAD_PARAM", "ops[0].gate.rule.bias",
    )
    _reject(
        {"ops": [_op(gate=_gate(rule={"kind": "per_key_threshold", "threshold": "hi",
                                      "comparator": "ge", "aggregate": "any"}))]},
        "E_BAD_PARAM", "ops[0].gate.rule.threshold",
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


def test_rule_comparator_and_aggregate_validated():
    _reject(
        {"ops": [_op(gate=_gate(rule={"kind": "per_key_threshold", "threshold": 0.1,
                                      "comparator": "gt", "aggregate": "any"}))]},
        "E_BAD_PARAM", "ops[0].gate.rule.comparator",
    )
    _reject(
        {"ops": [_op(gate=_gate(rule={"kind": "per_key_threshold", "threshold": 0.1,
                                      "comparator": "ge", "aggregate": "some"}))]},
        "E_BAD_PARAM", "ops[0].gate.rule.aggregate",
    )


# ---------------------------------------------------------------------------
# Rejects: gate structure
# ---------------------------------------------------------------------------


def test_gate_missing_required_field():
    for missing in ("layers", "pooling", "readout", "rule"):
        gate = _gate()
        del gate[missing]
        _reject({"ops": [_op(gate=gate)]}, "E_BAD_PARAM", f"ops[0].gate.{missing}")


def test_gate_bad_pooling():
    _reject({"ops": [_op(gate=_gate(pooling="max"))]}, "E_BAD_PARAM", "ops[0].gate.pooling")


def test_gate_empty_layers_rejected():
    _reject({"ops": [_op(gate=_gate(layers=[]))]}, "E_BAD_PARAM", "ops[0].gate.layers")


def test_gate_layer_out_of_range():
    _reject(
        {"ops": [_op(gate=_gate(layers=[NUM_LAYERS]))]},
        "E_LAYER_RANGE", "ops[0].gate.layers[0]",
    )


def test_readout_missing_artifact():
    _reject(
        {"ops": [_op(gate=_gate(readout={"kind": "affine"}))]},
        "E_BAD_PARAM", "ops[0].gate.readout.artifact",
    )


def test_rule_may_not_carry_artifact():
    _reject(
        {"ops": [_op(gate=_gate(rule={"kind": "sum_threshold", "bias": 0.0, "artifact": PROBE}))]},
        "E_BAD_PARAM", "ops[0].gate.rule.artifact",
    )


# ---------------------------------------------------------------------------
# Rejects: layers
# ---------------------------------------------------------------------------


def test_layer_out_of_range():
    _reject({"ops": [_op(layers=[NUM_LAYERS])]}, "E_LAYER_RANGE", "ops[0].layers[0]")
    _reject({"ops": [_op(layers=[-1])]}, "E_LAYER_RANGE", "ops[0].layers[0]")


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
