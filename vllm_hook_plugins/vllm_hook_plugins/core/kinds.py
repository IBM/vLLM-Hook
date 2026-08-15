"""Kind-name registries for the declarative surface.

Kind names are permanent: meanings never change and new behavior is a new
kind. ``KIND_PARAMS`` is the authoritative per-kind scalar-parameter table
used by ``schema`` for validation; ``CONSTRAINTS`` maps a kind to the
engine condition under which it is admissible.

A gate is built from a readout (``READOUT_KINDS``: how pooled hidden rows
become one value per condition layer) and a rule (``RULE_KINDS``: the
boolean decision over those values). The pre-redesign gate kinds and the
``directional_ablation`` transform name were retired in a single wire
break, not repurposed; the permanence invariant applies to every name
still listed here.
"""

TRANSFORM_KINDS = frozenset({"additive", "projection", "rotation", "head_additive"})
MODIFIER_KINDS = frozenset({"norm_preserving", "alignment_adaptive"})
SCOPE_KINDS = frozenset({"all", "after_prompt", "last_k", "from_position"})
READOUT_KINDS = frozenset({"affine", "cosine", "projected_cosine"})
RULE_KINDS = frozenset({"per_key_threshold", "sum_threshold"})
PROCESSOR_KINDS = frozenset({"constraint"})
CAPTURE_KINDS = frozenset({"residual"})
CAPTURE_LOCATIONS = frozenset({"layer_output", "layer_input"})
CAPTURE_MODES = frozenset({"all_tokens", "last_token"})

CONSTRAINTS = {"head_additive": "tensor_parallel_size==1"}

# kind -> {param_name: type}; seeded here, extended only additively.
KIND_PARAMS = {
    "additive": {"strength": float},
    "projection": {},
    "rotation": {"angle": float, "mode": str},
    "head_additive": {"strength": float},
    "norm_preserving": {},
    "alignment_adaptive": {"threshold": float, "use_cosine": bool},
    "last_k": {"k": int},
    "from_position": {"position": int},
    "affine": {},
    "cosine": {},
    "projected_cosine": {},
    "sum_threshold": {"bias": float},
    "per_key_threshold": {"threshold": float, "comparator": str, "aggregate": str},
}

# (kind, param) -> allowed values for string-typed params. The ("gate",
# "pooling") entry validates a gate's pooling field, which is structural
# rather than a kind parameter.
STRING_PARAM_VALUES = {
    ("rotation", "mode"): ("target", "offset"),
    ("per_key_threshold", "comparator"): ("ge", "le"),
    ("per_key_threshold", "aggregate"): ("any", "all"),
    ("gate", "pooling"): ("mean", "last"),
}

# kind -> tensor names its artifact must carry; a kind listed here requires
# exactly one artifact, a kind absent here must not carry one. Extended only
# additively, like KIND_PARAMS.
ARTIFACT_TENSORS = {
    "additive": ("vector",),
    "projection": ("vector",),
    "rotation": ("basis",),
    "head_additive": ("vector",),
    "alignment_adaptive": ("vector",),
    "affine": ("weights",),
    "cosine": ("directions",),
    "projected_cosine": ("directions",),
}
