"""Coalesces the per-request ``flush_disk`` RPC when the aperture owns the capture."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SKIP_ENV = "MIA_APERTURE_SKIP_EMPTY_DISK_FLUSH"
PROBE_ENV = "MIA_APERTURE_DISK_FLUSH_PROBE"
DEFAULT_PROBE = 256


class DiskFlushProbeError(RuntimeError):
    """The probe interval is not a usable number."""


def skip_enabled() -> bool:
    """``MIA_APERTURE_SKIP_EMPTY_DISK_FLUSH``: default on; ``"0"`` is the only off-spelling."""
    return os.environ.get(SKIP_ENV, "1") != "0"


def probe_interval() -> int:
    """``MIA_APERTURE_DISK_FLUSH_PROBE``: one flush per this many finished requests of a class."""
    raw = os.environ.get(PROBE_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_PROBE
    try:
        n = int(raw.strip())
    except ValueError:
        raise DiskFlushProbeError(
            f"{PROBE_ENV}={raw!r} is not a whole number of requests. Unset it for "
            f"{DEFAULT_PROBE}, or set 1 for the per-request behaviour this replaces.") from None
    if n < 1:
        raise DiskFlushProbeError(
            f"{PROBE_ENV}={raw!r} must be at least 1 (1 = flush every request, which is exactly "
            f"today's behaviour). Use {SKIP_ENV}=0 to turn the skip off by name.")
    return n


def flush_class(*, graph: bool, wants_hs: bool, wants_qk: bool, wants_steer: bool,
                sink: str, per_request: bool, durable_wait: bool,
                run_id: str = "", hook_dir: str = "") -> Optional[str]:
    """The class key for a request whose ``flush_disk`` may be held back, or None."""
    if not (graph and wants_hs and not wants_qk and not wants_steer
            and sink == "disk" and not per_request and not durable_wait):
        return None
    # Keyed by hook_dir: run_id is per request, so keying on it would never arm the skip.
    _ = run_id
    return f"hook\x00{hook_dir}"


@dataclass
class FlushDecision:
    """What the caller must do with this request: flush ``ids`` now, or nothing."""
    flush: bool
    ids: List[str] = field(default_factory=list)
    is_probe: bool = False


@dataclass
class _Class:
    seen: int = 0
    pending: List[tuple] = field(default_factory=list)


class DiskFlushProbe:
    """Per-engine state: finished requests per class, and which ids are held back."""

    def __init__(self, probe: int = DEFAULT_PROBE):
        if int(probe) < 1:
            raise DiskFlushProbeError(f"probe={probe!r} must be at least 1")
        self.probe = int(probe)
        self.disarmed = False
        self.nonempty = 0
        self.probes = 0
        self.skipped = 0
        self._classes: Dict[str, _Class] = {}

    def decide(self, key: str, req_id: str = "<the caller's id goes here>", run_id: str = "",
               hook_dir: str = "") -> FlushDecision:
        """Whether this finished request's flush happens now, and with which ids."""
        if self.disarmed:
            return FlushDecision(flush=True, ids=[str(req_id)], is_probe=False)
        c = self._classes.setdefault(str(key), _Class())
        is_probe = (c.seen % self.probe) == 0
        c.seen += 1
        if is_probe:
            self.probes += 1
            return FlushDecision(flush=True, ids=[str(req_id)], is_probe=True)
        c.pending.append((str(req_id), str(run_id), str(hook_dir)))
        self.skipped += 1
        return FlushDecision(flush=False, ids=[], is_probe=False)

    def note_result(self, key: str, result) -> bool:
        """Feed a probe's ``flush_disk`` result back; True when it disarmed the skip."""
        if self.disarmed:
            return False
        try:
            wrote = any(bool(x) for x in (result or ()))
        except TypeError:
            wrote = bool(result)
        if not wrote:
            return False
        self.disarmed = True
        self.nonempty += 1
        return True

    def pending(self, key: str) -> List[tuple]:
        c = self._classes.get(str(key))
        return list(c.pending) if c else []

    def drain_all(self) -> Dict[str, List[tuple]]:
        """Take every held-back ``(request_id, run_id, hook_dir)``, per class."""
        out = {k: list(c.pending) for k, c in self._classes.items() if c.pending}
        for c in self._classes.values():
            c.pending = []
        return out

    def stats(self) -> dict:
        return {"probe": self.probe, "probes": self.probes, "skipped": self.skipped,
                "nonempty": self.nonempty, "disarmed": self.disarmed,
                "classes": len(self._classes),
                "pending": sum(len(c.pending) for c in self._classes.values())}


def announce(*, probe: int, enabled: bool) -> str:
    """The install line's clause."""
    if not enabled:
        return f"flush_disk coalescing: OFF ({SKIP_ENV}=0) -- one collective_rpc per finished request"
    return (f"flush_disk coalescing: ON (default; {SKIP_ENV}=0 restores one RPC per "
            f"request) -- one collective_rpc per {probe} finished requests of a class whose capture "
            f"the APERTURE owns. A probe that finds a rank DID write disarms this for the rest of "
            f"the run and says so. {PROBE_ENV}={probe}")


__all__ = ["SKIP_ENV", "PROBE_ENV", "DEFAULT_PROBE", "DiskFlushProbeError", "DiskFlushProbe",
           "FlushDecision", "announce", "flush_class", "probe_interval", "skip_enabled"]
