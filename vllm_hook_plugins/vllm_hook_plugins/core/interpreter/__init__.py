"""Reference math for the declarative kinds.

Pure tensor functions: no vLLM imports, no engine state, no in-place
mutation of inputs. ``stream`` is (rows, hidden) for residual transforms
and (rows, num_heads, head_dim) for ``head_additive``. The worker and any
external implementation call the same entry point, so equivalence is a
property of one function, not two.
"""
from __future__ import annotations

import torch

from vllm_hook_plugins.core.interpreter.gates import READOUTS, RULES, GateState, build_gate
from vllm_hook_plugins.core.interpreter.modifiers import MODIFIERS
from vllm_hook_plugins.core.interpreter.scopes import scope_rows
from vllm_hook_plugins.core.interpreter.transforms import TRANSFORMS
from vllm_hook_plugins.core.kinds import ARTIFACT_TENSORS

__all__ = [
    "GateState",
    "MODIFIERS",
    "READOUTS",
    "RULES",
    "TRANSFORMS",
    "apply_op",
    "build_gate",
    "scope_rows",
]


def _tensor_kwargs(kind: str, artifact_id: str | None, artifacts: dict, stream: torch.Tensor) -> dict:
    """Look up the tensors ``kind`` expects from its artifact, cast to the
    stream's device/dtype so the math runs in the model's precision.
    """
    names = ARTIFACT_TENSORS.get(kind)
    if not names:
        return {}
    tensors = artifacts[artifact_id]
    return {
        name: tensors[name].to(device=stream.device, dtype=stream.dtype) for name in names
    }


def apply_op(op, stream: torch.Tensor, artifacts: dict) -> torch.Tensor:
    """Apply ``op``'s transform + modifier chain to ``stream`` using the
    resolved ``artifacts`` (id -> tensor dict). Returns a new tensor.
    """
    transform = TRANSFORMS[op.transform_kind]
    transform_kwargs = dict(op.transform_params)
    transform_kwargs.update(_tensor_kwargs(op.transform_kind, op.artifact, artifacts, stream))

    def call(rows: torch.Tensor) -> torch.Tensor:
        return transform(rows, **transform_kwargs)

    for modifier in op.modifiers:
        factory = MODIFIERS[modifier.kind]
        modifier_kwargs = dict(modifier.params)
        modifier_kwargs.update(_tensor_kwargs(modifier.kind, modifier.artifact, artifacts, stream))
        call = factory(call, **modifier_kwargs)

    return call(stream)
