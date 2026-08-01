"""Kind-name registries for the declarative surface.

Kind names are permanent: meanings never change and new behavior is a new
kind. ``KIND_PARAMS`` is the authoritative per-kind scalar-parameter table
used by ``schema`` for validation; ``CONSTRAINTS`` maps a kind to the
engine condition under which it is admissible.
"""

TRANSFORM_KINDS = frozenset({"additive", "directional_ablation", "rotation", "head_additive"})
MODIFIER_KINDS = frozenset({"norm_preserving", "alignment_adaptive"})
SCOPE_KINDS = frozenset({"all", "after_prompt", "last_k", "from_position"})
GATE_KINDS = frozenset({"null", "cache_once", "probe_sum", "multi_key_threshold"})
BASE_GATE_KINDS = frozenset({"null"})  # served without the conditional handler
PROCESSOR_KINDS = frozenset({"constraint"})
CAPTURE_KINDS = frozenset({"residual"})
CAPTURE_LOCATIONS = frozenset({"layer_output", "layer_input"})
CAPTURE_MODES = frozenset({"all_tokens", "last_token"})

CONSTRAINTS = {"head_additive": "tensor_parallel_size==1"}

# kind -> {param_name: type}; seeded here, extended only additively.
KIND_PARAMS = {
    "additive": {"strength": float},
    "directional_ablation": {},
    "rotation": {"angle": float},
    "head_additive": {"strength": float},
    "norm_preserving": {},
    "alignment_adaptive": {},  # artifact-carried; scalars added additively
    "last_k": {"k": int},
    "from_position": {"position": int},
    "multi_key_threshold": {"threshold": float, "condition_layers": list},
    "probe_sum": {"threshold": float, "condition_layers": list},
}

# kind -> tensor names its artifact must carry; a kind listed here requires
# exactly one artifact, a kind absent here must not carry one. Extended only
# additively, like KIND_PARAMS.
ARTIFACT_TENSORS = {
    "additive": ("vector",),
    "directional_ablation": ("vector",),
    "rotation": ("basis",),
    "head_additive": ("vector",),
    "alignment_adaptive": ("vector",),
    "probe_sum": ("weight",),
    "multi_key_threshold": ("weights",),
}
