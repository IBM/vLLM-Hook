"""Defines what MIA's hidden-state worker would capture: decoder-layer outputs (default) or MLP neurons"""

from __future__ import annotations
import os
import re

from mia.errors import MiaConfigurationError

HS_TARGET_ENV = "MIA_HS_TARGET"
HS_TARGET_LAYER = "layer"
HS_TARGET_MLP = "mlp"
HS_TARGETS = (HS_TARGET_LAYER, HS_TARGET_MLP)

MLP_ACT_PATTERNS = [
        re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.act_fn$")
        ]

def hs_target() -> str:
    value = os.environ.get(HS_TARGET_ENV, "").strip().lower() or HS_TARGET_LAYER
    if value not in HS_TARGETS:
        raise MiaConfigurationError(
                 f"{HS_TARGET_ENV}={value!r} is not a capture target; use one of {HS_TARGETS}"
                 )
    return value

def match_mlp_act(name: str):
    for pat in MLP_ACT_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None

def hs_match_fn(match_layer):
    return match_mlp_act if hs_target() == HS_TARGET_MLP else match_layer

def hs_capture_width(text_cfg, tp_size: int = 1) -> int:
    if hs_target() != HS_TARGET_MLP:
        return int(getattr(text_cfg, "hidden_size"))
    if int(tp_size) != 1:
        raise MiaConfigurationError(
                f"{HS_TARGET_ENV}=mlp needs tensor_parallel_size=1: under tensor parallelism each "
            f"rank holds only a shard of the MLP neurons")
    width = getattr(text_cfg, "intermediate_size", None)
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise MiaConfigurationError(
            f"{HS_TARGET_ENV}=mlp needs a dense MLP with one integer intermediate_size; "
            f"this model has intermediate_size={width!r}")
    return width
