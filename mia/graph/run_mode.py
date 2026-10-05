"""The environment half of a run's mode: which worker, and graph or eager."""
from __future__ import annotations

from typing import Mapping, Optional

from mia.errors import MiaConfigurationError, MiaRefusal

GRAPH_ENV = "MIA_ALLOW_CUDAGRAPH"
WORKER_ENV = "MIA_WORKER"

MIA_WORKER_VALUES = ("hidden_states", "qk", "steer")
DEFAULT_MIA_WORKER = "hidden_states"


class UnknownMiaWorkerError(MiaRefusal, ValueError):
    """MIA_WORKER was set to something that is not one of MIA_WORKER_VALUES."""


def parse_mia_worker_env(raw):
    """Parse a raw MIA_WORKER value into 'hidden_states' | 'qk' | 'steer', or None."""
    if raw is None or raw == "":
        return None
    if raw in MIA_WORKER_VALUES:
        return raw
    raise UnknownMiaWorkerError(
        f"MIA_WORKER={raw!r} is not a MIA worker. Accepted values (exact, no aliases): "
        f"{', '.join(MIA_WORKER_VALUES)}; unset means {DEFAULT_MIA_WORKER!r}. "
        f"This is refused on BOTH paths. Under `vllm serve` an unrecognized value used to "
        f"install the hidden-states worker, so a run asking for QK captured hidden states "
        f"and reported success. Offline (MiaLLM(worker_name=...)) the variable is read too "
        f"-- first and authoritatively, ahead of the worker class -- so a stale value there "
        f"used to mis-size the capture-aperture OOM guard: HS row shape for a QK run, or no "
        f"guard at all for 'steer'. Fix the spelling or unset the variable; MIA will not "
        f"guess which of the two you meant."
    )


def capture_mode_from_env(env: Mapping[str, str]) -> Optional[bool]:
    """``MIA_ALLOW_CUDAGRAPH``: True ("1"), False ("0"), None (unset or empty)."""
    raw = env.get(GRAPH_ENV)
    if raw is None or raw == "":
        return None
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise MiaConfigurationError(
        f"{GRAPH_ENV}={raw!r} is not '1' or '0'. Unset means CUDA-graph capture unless "
        f"enforce_eager=True; '1' forces graph capture, '0' forces eager.")


__all__ = ["DEFAULT_MIA_WORKER", "GRAPH_ENV", "MIA_WORKER_VALUES", "UnknownMiaWorkerError",
           "WORKER_ENV", "capture_mode_from_env", "parse_mia_worker_env"]
