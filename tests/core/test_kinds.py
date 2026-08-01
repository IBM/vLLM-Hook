# tests/core/test_kinds.py
"""The kind registries are closed sets with permanent names; these tests
pin the exact membership so any change is a deliberate, reviewed addition.
"""
from vllm_hook_plugins.core import kinds


def test_registries_are_frozen_and_closed():
    assert kinds.TRANSFORM_KINDS == frozenset(
        {"additive", "directional_ablation", "rotation", "head_additive"}
    )
    assert kinds.MODIFIER_KINDS == frozenset({"norm_preserving", "alignment_adaptive"})
    assert kinds.SCOPE_KINDS == frozenset({"all", "after_prompt", "last_k", "from_position"})
    assert kinds.GATE_KINDS == frozenset({"null", "cache_once", "probe_sum", "multi_key_threshold"})
    assert kinds.BASE_GATE_KINDS == frozenset({"null"})
    assert kinds.PROCESSOR_KINDS == frozenset({"constraint"})
    assert kinds.CAPTURE_KINDS == frozenset({"residual"})
    assert kinds.CAPTURE_LOCATIONS == frozenset({"layer_output", "layer_input"})
    assert kinds.CAPTURE_MODES == frozenset({"all_tokens", "last_token"})
    for registry in (
        kinds.TRANSFORM_KINDS,
        kinds.MODIFIER_KINDS,
        kinds.SCOPE_KINDS,
        kinds.GATE_KINDS,
        kinds.BASE_GATE_KINDS,
        kinds.PROCESSOR_KINDS,
        kinds.CAPTURE_KINDS,
        kinds.CAPTURE_LOCATIONS,
        kinds.CAPTURE_MODES,
    ):
        assert isinstance(registry, frozenset)


def test_kind_params_covers_every_parameterized_kind():
    all_kinds = (
        kinds.TRANSFORM_KINDS | kinds.MODIFIER_KINDS | kinds.SCOPE_KINDS | kinds.GATE_KINDS
    )
    # Every KIND_PARAMS entry names a registered kind.
    for kind in kinds.KIND_PARAMS:
        assert kind in all_kinds, f"KIND_PARAMS has unregistered kind {kind!r}"
    # Every kind that takes parameters is present; parameterless kinds may
    # appear with an empty table but must not be missing when they need one.
    assert kinds.KIND_PARAMS["additive"] == {"strength": float}
    assert kinds.KIND_PARAMS["directional_ablation"] == {}
    assert kinds.KIND_PARAMS["rotation"] == {"angle": float, "mode": str}
    assert kinds.KIND_PARAMS["head_additive"] == {"strength": float}
    assert kinds.KIND_PARAMS["norm_preserving"] == {}
    assert kinds.KIND_PARAMS["alignment_adaptive"] == {"threshold": float, "use_cosine": bool}
    assert kinds.KIND_PARAMS["last_k"] == {"k": int}
    assert kinds.KIND_PARAMS["from_position"] == {"position": int}
    assert kinds.KIND_PARAMS["probe_sum"] == {"condition_layers": list, "pooling": str}
    assert kinds.KIND_PARAMS["multi_key_threshold"] == {"threshold": float, "condition_layers": list}
    assert kinds.STRING_PARAM_VALUES == {
        ("rotation", "mode"): ("target", "offset"),
        ("probe_sum", "pooling"): ("mean", "last"),
    }


def test_artifact_tensors_and_constraints_name_registered_kinds():
    all_kinds = (
        kinds.TRANSFORM_KINDS | kinds.MODIFIER_KINDS | kinds.SCOPE_KINDS | kinds.GATE_KINDS
    )
    for kind in kinds.ARTIFACT_TENSORS:
        assert kind in all_kinds
    for kind in kinds.CONSTRAINTS:
        assert kind in all_kinds
    assert kinds.CONSTRAINTS == {"head_additive": "tensor_parallel_size==1"}
