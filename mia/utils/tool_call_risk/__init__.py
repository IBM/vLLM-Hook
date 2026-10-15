"""Tool-call risk probe, inference side for vLLM-Hook.

Scores the hidden state at the last prompt token of an agent step to predict
whether the model is about to emit an unsafe tool call. Training and
evaluation code:

    https://github.com/rishabhsinha17/latent-state-auditing
"""

from mia.utils.tool_call_risk.score import (
    ToolCallProbeArtifact,
    ToolCallRiskProbe,
    ToolCallRiskScore,
)

__all__ = ["ToolCallProbeArtifact", "ToolCallRiskProbe", "ToolCallRiskScore"]
