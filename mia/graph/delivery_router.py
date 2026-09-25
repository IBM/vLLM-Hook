"""Per-request delivery router: given the PREDICTED artifact bytes and whether the chosen analyzer is
REDUCIBLE server-side, pick the transport (rpc vs disk) and where to analyze (inflight/from_disk/none).
Pure logic — no torch/GPU.

WHERE THE THRESHOLDS COME FROM. This module does not own them; it is handed them.
``_plugin._aperture_route_thresholds(kind)`` DERIVES both, per worker kind, by solving that kind's
two on-loop cost models against each other (``run_utils.rpc_disk_crossover_kb``): RPC's blocking
ship ``5.0 + slope*KB`` (slope 0.03 HS / 0.157 QK, measured on granite-3.1-8b in the cb campaigns
that established the HS-last -> RPC optimum) against the disk route's ``20.0 + slope*KB``
(slope 0.0022 HS / 0.0078 QK, measured job 1802981 from the per-request staging's own writes).
That puts the shipped crossover at 539.6 KB for HS and 100.5 KB for QK; ``MIA_ROUTER_T_RPC`` /
``MIA_ROUTER_T_ANALYZE`` override it outright, and every coefficient has its own env override.

ONE of those coefficients is NOT measured, and says so where it lives: the 20.0 ms disk HANDOFF
(``run_utils._DISK_HANDOFF_MS``), an engine-loop cost no bench in this repo has timed. It dominates
the crossover, so read the defaults as "derived from a model with one estimated term", not as
calibration. (Until 2026-09-20 this docstring claimed the thresholds were "CALIBRATED from a
profile, never guessed". There was no such profile in the repo, and both kinds shared one
hardcoded 512 KiB.)"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RouteDecision:
    transport: str        # 'rpc' | 'disk'
    analyze_where: str    # 'none' (deliver raw) | 'inflight' (host buf) | 'from_disk'

def decide_route(predicted_bytes: int, reducible: bool, t_rpc: int, t_analyze: int) -> RouteDecision:
    if reducible:
        # ship the small RESULT: analyze in-flight if the artifact is small enough to hold, else
        # stream to disk and analyze from disk (avoid holding a huge artifact in host RAM).
        return (RouteDecision("rpc", "inflight") if predicted_bytes <= t_analyze
                else RouteDecision("disk", "from_disk"))
    # deliver the RAW artifact: RPC if small, else disk + offload.
    return (RouteDecision("rpc", "none") if predicted_bytes <= t_rpc
            else RouteDecision("disk", "none"))
