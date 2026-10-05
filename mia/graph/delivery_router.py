"""Per-request delivery router: picks the transport (RPC or disk) and where to analyze."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RouteDecision:
    transport: str
    analyze_where: str

def decide_route(predicted_bytes: int, reducible: bool, t_rpc: int, t_analyze: int) -> RouteDecision:
    if reducible:
        return (RouteDecision("rpc", "inflight") if predicted_bytes <= t_analyze
                else RouteDecision("disk", "from_disk"))
    return (RouteDecision("rpc", "none") if predicted_bytes <= t_rpc
            else RouteDecision("disk", "none"))

