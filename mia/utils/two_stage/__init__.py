"""Two-stage detection: a cheap gate plus a second scorer that only sees its flags.

See ``mia.utils.two_stage.cascade`` for the calibration rules and
``mia.analyzers.two_stage_probe_analyzer`` for the hidden-state analyzer.
"""

from mia.utils.two_stage.cascade import (
    CascadeResult,
    OperatingPoint,
    TwoStageDetector,
    calibrate_cascade,
    calibrate_cascade_for_fpr,
    cascade_flags,
    evaluate_cascade,
    save_cascade_config,
    threshold_for_fpr,
    threshold_for_recall,
)
from mia.utils.two_stage.tool_call import find_tool_call, tool_call_window

__all__ = [
    "CascadeResult",
    "OperatingPoint",
    "TwoStageDetector",
    "calibrate_cascade",
    "calibrate_cascade_for_fpr",
    "cascade_flags",
    "evaluate_cascade",
    "find_tool_call",
    "save_cascade_config",
    "threshold_for_fpr",
    "threshold_for_recall",
    "tool_call_window",
]
