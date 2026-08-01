"""Parse and validate ``intervention_spec`` / ``processor_spec`` / ``capture``.

Validation is strict and closed: unknown fields, unknown kinds, and bad
parameters raise ``SpecError`` with a stable ``E_*`` code and the JSON
path of the offending element. A parsed spec is immutable and complete —
``parse_*`` never returns a partially valid object.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from vllm_hook_plugins.core.canonical import canonical_bytes
from vllm_hook_plugins.core.kinds import (
    ARTIFACT_TENSORS,
    CAPTURE_KINDS,
    CAPTURE_LOCATIONS,
    CAPTURE_MODES,
    GATE_KINDS,
    KIND_PARAMS,
    MODIFIER_KINDS,
    SCOPE_KINDS,
    STRING_PARAM_VALUES,
    TRANSFORM_KINDS,
)

E_UNKNOWN_KIND = "E_UNKNOWN_KIND"
E_UNKNOWN_FIELD = "E_UNKNOWN_FIELD"
E_BAD_PARAM = "E_BAD_PARAM"
E_LAYER_RANGE = "E_LAYER_RANGE"
E_ARTIFACT_MISSING = "E_ARTIFACT_MISSING"
E_ARTIFACT_HASH = "E_ARTIFACT_HASH"
E_SALT_REQUIRED = "E_SALT_REQUIRED"
E_CONSTRAINT = "E_CONSTRAINT"
E_SPEC_TOO_LARGE = "E_SPEC_TOO_LARGE"
E_LEGACY_KEY = "E_LEGACY_KEY"

# Ceiling on the canonical-JSON size of any single spec object. Large tensor
# payloads belong in the artifact registry, never inline in a spec.
MAX_SPEC_BYTES = 64 * 1024

# \Z, not $: $ would also match before a trailing newline.
_ARTIFACT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")


class SpecError(ValueError):
    """``code`` is one of the ``E_*`` constants; ``path`` locates the
    offending element (e.g. ``ops[1].gate.kind``); ``str(err)`` is the
    user-facing message including the fix hint.
    """

    def __init__(self, code: str, path: str, msg: str):
        self.code = code
        self.path = path
        self.msg = msg
        super().__init__(f"{code} at {path or '<spec>'}: {msg}")

    def payload(self) -> dict:
        """JSON-safe form carried over collective_rpc and HTTP errors."""
        return {"code": self.code, "path": self.path, "msg": self.msg}


@dataclass(frozen=True)
class ScopeSpec:
    kind: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GateSpec:
    kind: str
    params: dict = field(default_factory=dict)
    artifact: str | None = None
    inner: "GateSpec | None" = None


@dataclass(frozen=True)
class ModifierSpec:
    kind: str
    params: dict = field(default_factory=dict)
    artifact: str | None = None


@dataclass(frozen=True)
class OpSpec:
    layers: tuple[int, ...]
    transform_kind: str
    transform_params: dict
    artifact: str | None
    modifiers: tuple[ModifierSpec, ...]
    scope: ScopeSpec
    gate: GateSpec | None


@dataclass(frozen=True)
class InterventionSpec:
    ops: tuple[OpSpec, ...]

    def artifact_ids(self) -> tuple[str, ...]:
        """All artifact ids the spec references, deduplicated in first-seen
        order: transform, then modifiers, then gate (outermost first).
        """
        seen: list[str] = []

        def _add(artifact_id: str | None) -> None:
            if artifact_id is not None and artifact_id not in seen:
                seen.append(artifact_id)

        for op in self.ops:
            _add(op.artifact)
            for modifier in op.modifiers:
                _add(modifier.artifact)
            gate = op.gate
            while gate is not None:
                _add(gate.artifact)
                gate = gate.inner
        return tuple(seen)

    def layers(self) -> frozenset[int]:
        """Every decoder layer any op applies at (condition layers excluded)."""
        return frozenset(layer for op in self.ops for layer in op.layers)

    def condition_layers(self) -> frozenset[int]:
        """Every condition layer any gate reads at the layer_input boundary."""
        layers: set[int] = set()
        for op in self.ops:
            gate = op.gate
            while gate is not None:
                layers.update(gate.params.get("condition_layers", ()))
                gate = gate.inner
        return frozenset(layers)


@dataclass(frozen=True)
class CaptureSpec:
    layers: tuple[int, ...] | None  # None means every decoder layer
    location: str
    mode: str
    kind: str = "residual"
    save_dir: str | None = None


# ---------------------------------------------------------------------------
# Field-level helpers
# ---------------------------------------------------------------------------


def _require_dict(obj, path: str, what: str) -> dict:
    if not isinstance(obj, dict):
        raise SpecError(E_BAD_PARAM, path, f"{what} must be a JSON object, got {type(obj).__name__}")
    return obj


def _reject_unknown_fields(obj: dict, allowed: set, path: str) -> None:
    for key in obj:
        if key not in allowed:
            raise SpecError(
                E_UNKNOWN_FIELD,
                f"{path}.{key}" if path else key,
                f"unknown field {key!r}; allowed fields: {sorted(allowed)}",
            )


def _check_size(obj, path: str) -> None:
    try:
        size = len(canonical_bytes(obj))
    except (TypeError, ValueError) as exc:
        raise SpecError(E_BAD_PARAM, path, f"spec is not JSON-serializable: {exc}")
    if size > MAX_SPEC_BYTES:
        raise SpecError(
            E_SPEC_TOO_LARGE,
            path,
            f"spec is {size} bytes, limit is {MAX_SPEC_BYTES}; move tensor data into artifacts",
        )


def _parse_scalar_param(kind: str, name: str, value, expected: type, path: str):
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SpecError(E_BAD_PARAM, path, f"{name!r} for kind {kind!r} must be a number")
        try:
            value = float(value)
        except OverflowError:
            raise SpecError(E_BAD_PARAM, path, f"{name!r} for kind {kind!r} is too large")
        if not math.isfinite(value):
            raise SpecError(E_BAD_PARAM, path, f"{name!r} for kind {kind!r} must be finite")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SpecError(E_BAD_PARAM, path, f"{name!r} for kind {kind!r} must be an integer")
        return int(value)
    if expected is bool:
        if not isinstance(value, bool):
            raise SpecError(E_BAD_PARAM, path, f"{name!r} for kind {kind!r} must be a boolean")
        return value
    if expected is str:
        if not isinstance(value, str):
            raise SpecError(E_BAD_PARAM, path, f"{name!r} for kind {kind!r} must be a string")
        allowed = STRING_PARAM_VALUES.get((kind, name))
        if allowed is not None and value not in allowed:
            raise SpecError(
                E_BAD_PARAM,
                path,
                f"{name!r} for kind {kind!r} must be one of {sorted(allowed)}; got {value!r}",
            )
        return value
    if expected is list:
        if not isinstance(value, list):
            raise SpecError(E_BAD_PARAM, path, f"{name!r} for kind {kind!r} must be a list")
        return list(value)
    raise SpecError(E_BAD_PARAM, path, f"unsupported parameter type for {name!r}")


def _parse_kind_params(kind: str, obj: dict, path: str, *, extra_fields: set) -> dict:
    """Validate the inline parameters of a kind-bearing object.

    ``obj`` carries the params as top-level fields next to ``kind`` (and
    the structural fields listed in ``extra_fields``). Every parameter in
    ``KIND_PARAMS[kind]`` is required; anything else is unknown.
    """
    param_table = KIND_PARAMS.get(kind, {})
    _reject_unknown_fields(obj, {"kind", *param_table, *extra_fields}, path)
    params = {}
    for name, expected in param_table.items():
        if name not in obj:
            raise SpecError(E_BAD_PARAM, f"{path}.{name}", f"missing required parameter {name!r} for kind {kind!r}")
        params[name] = _parse_scalar_param(kind, name, obj[name], expected, f"{path}.{name}")
    return params


def _parse_artifact_field(kind: str, obj: dict, path: str) -> str | None:
    """Validate the ``artifact`` field against the kind's requirement.

    Kinds listed in ``ARTIFACT_TENSORS`` require exactly one artifact id;
    all other kinds must not carry one.
    """
    artifact = obj.get("artifact")
    if kind in ARTIFACT_TENSORS:
        if artifact is None:
            raise SpecError(E_BAD_PARAM, f"{path}.artifact", f"kind {kind!r} requires an artifact id")
        if not isinstance(artifact, str) or not _ARTIFACT_ID_RE.match(artifact):
            raise SpecError(
                E_BAD_PARAM,
                f"{path}.artifact",
                "artifact id must be 'sha256:' followed by 64 lowercase hex chars",
            )
        return artifact
    if artifact is not None:
        raise SpecError(E_BAD_PARAM, f"{path}.artifact", f"kind {kind!r} does not take an artifact")
    return None


def _parse_layers(value, num_layers: int, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SpecError(E_BAD_PARAM, path, "must be a non-empty list of decoder-layer indices")
    layers = []
    for j, layer in enumerate(value):
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise SpecError(E_BAD_PARAM, f"{path}[{j}]", "layer index must be an integer")
        if not 0 <= layer < num_layers:
            raise SpecError(
                E_LAYER_RANGE,
                f"{path}[{j}]",
                f"layer {layer} out of range; model has layers 0..{num_layers - 1}",
            )
        if layer in layers:
            raise SpecError(E_BAD_PARAM, f"{path}[{j}]", f"duplicate layer {layer}")
        layers.append(layer)
    return tuple(layers)


# ---------------------------------------------------------------------------
# intervention_spec
# ---------------------------------------------------------------------------


def _parse_scope(obj, num_layers: int, path: str) -> ScopeSpec:
    obj = _require_dict(obj, path, "scope")
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in SCOPE_KINDS:
        raise SpecError(E_UNKNOWN_KIND, f"{path}.kind", f"unknown scope kind {kind!r}; expected one of {sorted(SCOPE_KINDS)}")
    params = _parse_kind_params(kind, obj, path, extra_fields=set())
    if kind == "last_k" and params["k"] < 1:
        raise SpecError(E_BAD_PARAM, f"{path}.k", "'k' must be >= 1")
    if kind == "from_position" and params["position"] < 0:
        raise SpecError(E_BAD_PARAM, f"{path}.position", "'position' must be >= 0")
    return ScopeSpec(kind=kind, params=params)


def _parse_gate(obj, num_layers: int, allowed_gates: frozenset[str], path: str, *, is_inner: bool = False) -> GateSpec | None:
    if obj is None:
        return None
    obj = _require_dict(obj, path, "gate")
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in GATE_KINDS:
        raise SpecError(E_UNKNOWN_KIND, f"{path}.kind", f"unknown gate kind {kind!r}; expected one of {sorted(GATE_KINDS)}")
    if kind not in allowed_gates:
        raise SpecError(
            E_UNKNOWN_KIND,
            f"{path}.kind",
            f"gate kind {kind!r} is not served by this worker (conditional handler inactive); allowed: {sorted(allowed_gates)}",
        )
    if kind == "null":
        _reject_unknown_fields(obj, {"kind"}, path)
        return None
    if kind == "cache_once":
        if is_inner:
            raise SpecError(E_BAD_PARAM, f"{path}.kind", "'cache_once' cannot wrap another 'cache_once'")
        _reject_unknown_fields(obj, {"kind", "inner"}, path)
        if "inner" not in obj or obj["inner"] is None:
            raise SpecError(E_BAD_PARAM, f"{path}.inner", "'cache_once' requires an 'inner' gate")
        inner = _parse_gate(obj["inner"], num_layers, allowed_gates, f"{path}.inner", is_inner=True)
        if inner is None:
            raise SpecError(E_BAD_PARAM, f"{path}.inner", "'cache_once' requires a conditional 'inner' gate, not 'null'")
        return GateSpec(kind=kind, params={}, artifact=None, inner=inner)
    params = _parse_kind_params(kind, obj, path, extra_fields={"artifact"})
    condition_layers = params.get("condition_layers")
    if condition_layers is not None:
        params["condition_layers"] = list(_parse_layers(condition_layers, num_layers, f"{path}.condition_layers"))
    artifact = _parse_artifact_field(kind, obj, path)
    return GateSpec(kind=kind, params=params, artifact=artifact, inner=None)


def _parse_modifier(obj, path: str) -> ModifierSpec:
    obj = _require_dict(obj, path, "modifier")
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in MODIFIER_KINDS:
        raise SpecError(E_UNKNOWN_KIND, f"{path}.kind", f"unknown modifier kind {kind!r}; expected one of {sorted(MODIFIER_KINDS)}")
    params = _parse_kind_params(kind, obj, path, extra_fields={"artifact"})
    artifact = _parse_artifact_field(kind, obj, path)
    return ModifierSpec(kind=kind, params=params, artifact=artifact)


def _parse_op(obj, num_layers: int, allowed_gates: frozenset[str], path: str) -> OpSpec:
    obj = _require_dict(obj, path, "op")
    _reject_unknown_fields(obj, {"layers", "transform", "scope", "gate"}, path)
    for required in ("layers", "transform", "scope"):
        if required not in obj:
            raise SpecError(E_BAD_PARAM, f"{path}.{required}", f"op requires a {required!r} field")

    layers = _parse_layers(obj["layers"], num_layers, f"{path}.layers")

    transform = _require_dict(obj["transform"], f"{path}.transform", "transform")
    kind = transform.get("kind")
    if not isinstance(kind, str) or kind not in TRANSFORM_KINDS:
        raise SpecError(
            E_UNKNOWN_KIND,
            f"{path}.transform.kind",
            f"unknown transform kind {kind!r}; expected one of {sorted(TRANSFORM_KINDS)}",
        )
    params = _parse_kind_params(kind, transform, f"{path}.transform", extra_fields={"modifiers", "artifact"})
    artifact = _parse_artifact_field(kind, transform, f"{path}.transform")

    raw_modifiers = transform.get("modifiers", [])
    if not isinstance(raw_modifiers, list):
        raise SpecError(E_BAD_PARAM, f"{path}.transform.modifiers", "'modifiers' must be a list")
    modifiers = tuple(
        _parse_modifier(m, f"{path}.transform.modifiers[{j}]") for j, m in enumerate(raw_modifiers)
    )

    scope = _parse_scope(obj["scope"], num_layers, f"{path}.scope")
    gate = _parse_gate(obj.get("gate"), num_layers, allowed_gates, f"{path}.gate")

    return OpSpec(
        layers=layers,
        transform_kind=kind,
        transform_params=params,
        artifact=artifact,
        modifiers=modifiers,
        scope=scope,
        gate=gate,
    )


def parse_intervention_spec(obj: dict, *, num_layers: int, allowed_gates: frozenset[str] = GATE_KINDS) -> InterventionSpec:
    """Validate ``obj`` against the kind registries, ``KIND_PARAMS``, and
    ``num_layers``. ``allowed_gates`` is ``GATE_KINDS`` when the
    conditional handler is active, else ``BASE_GATE_KINDS``.
    """
    obj = _require_dict(obj, "", "intervention_spec")
    _check_size(obj, "")
    _reject_unknown_fields(obj, {"ops"}, "")
    if "ops" not in obj or not isinstance(obj["ops"], list):
        raise SpecError(E_BAD_PARAM, "ops", "intervention_spec requires an 'ops' list")
    ops = tuple(
        _parse_op(op, num_layers, allowed_gates, f"ops[{i}]") for i, op in enumerate(obj["ops"])
    )
    return InterventionSpec(ops=ops)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def parse_capture(obj: dict, *, num_layers: int) -> CaptureSpec:
    obj = _require_dict(obj, "", "capture")
    _check_size(obj, "")
    _reject_unknown_fields(obj, {"layers", "mode", "location", "kind", "save_dir"}, "")

    if "layers" not in obj:
        raise SpecError(E_BAD_PARAM, "layers", "capture requires a 'layers' field ('all' or a list of layer indices)")
    raw_layers = obj["layers"]
    if raw_layers == "all":
        layers = None
    else:
        layers = _parse_layers(raw_layers, num_layers, "layers")

    mode = obj.get("mode", "all_tokens")
    if not isinstance(mode, str) or mode not in CAPTURE_MODES:
        raise SpecError(E_BAD_PARAM, "mode", f"unknown capture mode {mode!r}; expected one of {sorted(CAPTURE_MODES)}")

    location = obj.get("location", "layer_output")
    if not isinstance(location, str) or location not in CAPTURE_LOCATIONS:
        raise SpecError(E_BAD_PARAM, "location", f"unknown capture location {location!r}; expected one of {sorted(CAPTURE_LOCATIONS)}")

    kind = obj.get("kind", "residual")
    if not isinstance(kind, str) or kind not in CAPTURE_KINDS:
        raise SpecError(E_UNKNOWN_KIND, "kind", f"unknown capture kind {kind!r}; expected one of {sorted(CAPTURE_KINDS)}")

    save_dir = obj.get("save_dir")
    if save_dir is not None and not isinstance(save_dir, str):
        raise SpecError(E_BAD_PARAM, "save_dir", "'save_dir' must be a string path")

    return CaptureSpec(layers=layers, location=location, mode=mode, kind=kind, save_dir=save_dir)


# ---------------------------------------------------------------------------
# processor_spec
# ---------------------------------------------------------------------------


def parse_processor_spec(obj: dict, *, allowed_processors: frozenset[str] = frozenset()):
    """Validate a ``processor_spec``. ``allowed_processors`` stays empty
    until the adapter logits processor is active, so every kind rejects
    with ``E_UNKNOWN_KIND`` on workers that do not serve processors.
    """
    obj = _require_dict(obj, "", "processor_spec")
    _check_size(obj, "")
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in allowed_processors:
        raise SpecError(
            E_UNKNOWN_KIND,
            "kind",
            f"processor kind {kind!r} is not served by this worker; allowed: {sorted(allowed_processors)}",
        )
    raise SpecError(E_UNKNOWN_KIND, "kind", f"processor kind {kind!r} has no handler yet")
