"""Which per-request delivery path a run gets; makes the hybrid the default."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Tuple

from mia.core.hooks.run_mode import GRAPH_ENV, WORKER_ENV, capture_mode_from_env, parse_mia_worker_env

logger = logging.getLogger(__name__)

MODE_ENV = "MIA_APERTURE_DELIVERY"
STAMP_ENV = "MIA_APERTURE_DELIVERY_RESOLVED"

PER_REQUEST_ENV = "MIA_APERTURE_PER_REQUEST"
DP_SIZE_ENV = "MIA_DP_SIZE"
PROFILE_ENV = "MIA_PROFILE_MODE"
SINK_ENV = "MIA_SINK"
GATHER_ENV = "MIA_APERTURE_GATHER"
DELIVER_ENV = "MIA_APERTURE_GATHER_DELIVER"
FLUSH_MS_ENV = "MIA_APERTURE_GATHER_FLUSH_MS"
TRIM_ENV = "MIA_APERTURE_GATHER_TRIM"
MMAP_ENV = "MIA_APERTURE_MMAP"
WRITE_MODE_ENV = "MIA_APERTURE_WRITE_MODE"

# A literal, not an import: mia._plugin is mid-import when apply() runs.
HS_WORKER = "hidden_states"

DEFAULT_FLUSH_MS = 200

MODES = ("auto", "hybrid", "drain", "off")
_RESOLVED_MODES = ("hybrid", "drain", "off")
_SOURCES = ("default", "explicit", "scope")


class DeliveryConfigError(RuntimeError):
    """The delivery configuration cannot be resolved into one path, and will not be guessed."""


@dataclass(frozen=True)
class Selection:
    """The resolved delivery path."""
    mode: str
    source: str
    reason: str
    note: str
    env_set: Tuple[Tuple[str, str], ...] = ()
    env_unset: Tuple[str, ...] = ()

    @property
    def stamp(self) -> str:
        return f"{self.mode}:{self.source}"


def _get(env: Mapping[str, str], key: str) -> Optional[str]:
    v = env.get(key)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def _is_on(env: Mapping[str, str], key: str) -> bool:
    return _get(env, key) == "1"


def _mode_env(env: Mapping[str, str]) -> str:
    raw = _get(env, MODE_ENV)
    if raw is None:
        return "auto"
    m = raw.lower()
    if m not in MODES:
        raise DeliveryConfigError(
            f"{MODE_ENV}={raw!r} is not one of {', '.join(map(repr, MODES))}; unset it for the "
            f"default.")
    return m


def _read_stamp(env: Mapping[str, str]) -> Optional[Tuple[str, str]]:
    raw = _get(env, STAMP_ENV)
    if raw is None:
        return None
    mode, sep, source = raw.partition(":")
    if not sep or mode not in _RESOLVED_MODES or source not in _SOURCES:
        raise DeliveryConfigError(f"{STAMP_ENV}={raw!r} is set by MIA itself; unset it.")
    return mode, source


def _refuse_if_hybrid_impossible(env: Mapping[str, str], source: str) -> None:
    if source == "explicit":
        asked = f"{MODE_ENV}=hybrid"
    elif _is_on(env, PER_REQUEST_ENV):
        asked = f"{PER_REQUEST_ENV}=1"
    else:
        asked = "hidden-states capture under CUDA graphs"
    mmap = _get(env, MMAP_ENV)
    if mmap is not None and mmap != "0":
        raise DeliveryConfigError(
            f"{MMAP_ENV}={mmap!r} cannot be used with {asked}; unset {MMAP_ENV}.")
    wm = _get(env, WRITE_MODE_ENV)
    if wm is not None and wm.lower() == "legacy":
        raise DeliveryConfigError(
            f"{WRITE_MODE_ENV}=legacy cannot be used with {asked}; use auto (the default), direct "
            f"or buffered.")


_HYBRID_SET = (
    f"{GATHER_ENV}=1 (the run-encoded per-request row index into the shared layer files), "
    f"{DELIVER_ENV}=1 (the gather child that scatters them), "
    f"{FLUSH_MS_ENV}={DEFAULT_FLUSH_MS} (the publication cadence)"
)
_HYBRID_RETRIEVAL = (
    "Each request's activations land as files under the delivery dir "
    "(aperture_gather.load_delivered); output.probes, MiaClient's response.probes and "
    "save_to_disk artifacts are read from them."
)


def _hybrid(env: Mapping[str, str], source: str, reason: str, *, asked: bool = False) -> Selection:
    _refuse_if_hybrid_impossible(env, source)
    sets = [(GATHER_ENV, "1"), (DELIVER_ENV, "1")]
    cadence = _get(env, FLUSH_MS_ENV)
    if cadence is None:
        sets.append((FLUSH_MS_ENV, str(DEFAULT_FLUSH_MS)))
    # The trim is not written: setting it would flip trim_explicit() for the same value.
    if source == "default" and asked:
        how = (f"chosen BY DEFAULT: {PER_REQUEST_ENV}=1 asks for per-request delivery, and the "
               f"HYBRID is how MIA delivers it ({MODE_ENV}=drain takes the in-drain per-request "
               f"writer instead)")
    elif source == "default":
        how = (f"chosen BY DEFAULT for hidden-states capture ({MODE_ENV}=drain takes the in-drain "
               f"per-request writer, {MODE_ENV}=off the shared files only)")
    else:
        how = f"chosen explicitly ({MODE_ENV}=hybrid)"
    note = (f"per-request delivery -> HYBRID (shared-file drain + streaming gather + trim), {how}. "
            f"Arming {_HYBRID_SET}; the trim keeps its own default (ON). {_HYBRID_RETRIEVAL}")
    return Selection(mode="hybrid", source=source, reason=reason, note=note,
                     env_set=tuple(sets), env_unset=(PER_REQUEST_ENV,))


def _drain(source: str, reason: str, *, hybrid_hint: bool = True) -> Selection:
    tail = f": {MODE_ENV}=hybrid selects it." if hybrid_hint else "."
    note = (f"per-request delivery -> the IN-DRAIN per-request writer ({reason}). Delivered through "
            f"the response (output.probes / the HTTP body), block-until-held within "
            f"MIA_APERTURE_DELIVER_TIMEOUT_S. The hybrid gather is NOT running{tail}")
    return Selection(mode="drain", source=source, reason=reason, note=note,
                     env_set=((PER_REQUEST_ENV, "1"),))


def _off(reason: str, source: str = "default") -> Selection:
    return Selection(mode="off", source=source, reason=reason,
                     note=f"no per-request delivery ({reason}); nothing changed")


def _conflict(a: str, b: str) -> DeliveryConfigError:
    return DeliveryConfigError(f"{a} and {b} cannot be set together; unset one of them.")


def _refuse_contradictions(env: Mapping[str, str], mode_env: str) -> None:
    per_request = _is_on(env, PER_REQUEST_ENV)
    gather_on = _is_on(env, GATHER_ENV)
    deliver_on = _is_on(env, DELIVER_ENV)
    if mode_env == "drain" and deliver_on:
        raise _conflict(f"{MODE_ENV}=drain", f"{DELIVER_ENV}=1")
    if mode_env == "off" and (per_request or deliver_on):
        flag = PER_REQUEST_ENV if per_request else DELIVER_ENV
        raise _conflict(f"{MODE_ENV}=off", f"{flag}=1")
    if mode_env == "auto" and per_request and (gather_on or deliver_on):
        flag = GATHER_ENV if gather_on else DELIVER_ENV
        raise _conflict(f"{PER_REQUEST_ENV}=1", f"{flag}=1")


def validate(env: Mapping[str, str]) -> None:
    """Syntax-only check (a bad mode, a malformed stamp, a contradictory pair)."""
    _read_stamp(env)
    _refuse_contradictions(env, _mode_env(env))


def resolve(env: Mapping[str, str], *, worker_kind: Optional[str] = None,
            graph: Optional[bool] = None) -> Selection:
    """Which delivery path this environment means; pure, mutates nothing."""
    stamp = _read_stamp(env)
    if stamp is not None:
        mode, source = stamp
        reason = f"carried in {STAMP_ENV} (resolution stamp)"
        if mode == "hybrid":
            return _hybrid(env, source, reason)
        if mode == "drain":
            return _drain(source, reason)
        return _off(reason)

    mode_env = _mode_env(env)
    _refuse_contradictions(env, mode_env)
    kind = worker_kind if worker_kind is not None else (
        parse_mia_worker_env(env.get(WORKER_ENV)) or HS_WORKER)
    if graph is None:
        graph = capture_mode_from_env(env) is not False
    per_request = _is_on(env, PER_REQUEST_ENV)
    gather_on = _is_on(env, GATHER_ENV)
    deliver_on = _is_on(env, DELIVER_ENV)

    if mode_env == "hybrid":
        if not graph:
            raise DeliveryConfigError(
                f"{MODE_ENV}=hybrid needs CUDA graph mode and this engine runs eager; unset "
                f"{MODE_ENV}, or drop enforce_eager=True (vllm serve: --enforce-eager).")
        if kind != HS_WORKER:
            raise DeliveryConfigError(
                f"{MODE_ENV}=hybrid applies to hidden-states capture only, not the {kind!r} "
                f"worker; unset {MODE_ENV}.")
        return _hybrid(env, "explicit", f"{MODE_ENV}=hybrid")
    if mode_env == "drain":
        return _drain("explicit", f"{MODE_ENV}=drain")
    if mode_env == "off":
        return _off(f"{MODE_ENV}=off: the shared files only", "explicit")

    if deliver_on or gather_on:
        return Selection(
            mode="hybrid" if deliver_on else "off", source="explicit",
            reason=f"{DELIVER_ENV}/{GATHER_ENV} set by hand",
            note=(f"the hybrid gather is armed explicitly ({GATHER_ENV}/{DELIVER_ENV} set by "
                  f"hand); the selector changed nothing. {_HYBRID_RETRIEVAL}"
                  if deliver_on else
                  f"{GATHER_ENV}=1 with no {DELIVER_ENV}: the run-encoded index is published and "
                  f"nothing is delivered; the selector changed nothing"))
    if not per_request:
        return _no_ask_default(env, kind, graph)
    if not graph:
        return _off(f"{PER_REQUEST_ENV}=1 on an eager engine, which is inert -- the capture "
                    f"aperture only exists under graph mode")

    if kind != HS_WORKER:
        return _drain("scope",
                      f"the hybrid gather reads hs_layer_*.raw and is HIDDEN-STATES ONLY; this run "
                      f"is {WORKER_ENV}={kind}, whose per-request delivery is the drain's",
                      hybrid_hint=False)
    if _is_on(env, PROFILE_ENV):
        return _drain("scope",
                      f"{PROFILE_ENV}=1 measures the capture pipeline only and disables delivery; "
                      f"the hybrid's gather delivers from another process and would not observe it, "
                      f"so the default does not arm it here ({MODE_ENV}=hybrid overrides)")
    return _hybrid(env, "default", f"{PER_REQUEST_ENV}=1 under {GRAPH_ENV}=1, HS, no override",
                   asked=True)


def _dp_size(env: Mapping[str, str]) -> int:
    raw = _get(env, DP_SIZE_ENV)
    if raw is None:
        return 1
    try:
        return int(raw)
    except ValueError:
        raise DeliveryConfigError(
            f"{DP_SIZE_ENV}={raw!r} is not an integer. MIA writes it from data_parallel_size; "
            f"unset it.") from None


def _no_ask_default(env: Mapping[str, str], kind: str, graph: bool) -> Selection:
    if not graph:
        return _off("eager -- no aperture", "scope")
    if kind not in (HS_WORKER, "qk"):
        return _off(f"the {kind!r} worker has no capture artifact", "scope")
    if _is_on(env, PROFILE_ENV):
        return _off(f"{PROFILE_ENV}=1 measures capture only", "scope")
    if (_get(env, SINK_ENV) or "").lower() == "drop":
        return _off(f"{SINK_ENV}=drop stores nothing", "scope")
    if kind == "qk":
        if _dp_size(env) > 1:
            return _off(f"{DP_SIZE_ENV}={_dp_size(env)}: QK per-request delivery cannot be collected "
                        f"across DP engines", "scope")
        return _drain("default", "QK capture under graph mode", hybrid_hint=False)
    mmap = _get(env, MMAP_ENV)
    if mmap is not None and mmap != "0":
        return _off(f"{MMAP_ENV}={mmap} pre-sizes the layer files the hybrid would read", "scope")
    wm = _get(env, WRITE_MODE_ENV)
    if wm is not None and wm.lower() == "legacy":
        return _off(f"{WRITE_MODE_ENV}=legacy keeps no per-step index for the hybrid", "scope")
    return _hybrid(env, "default", "plain HS capture under graph mode")


_ANNOUNCED: set = set()


def apply(env: Optional[MutableMapping[str, str]] = None, *, announce: bool = True,
          worker_kind: Optional[str] = None, graph: Optional[bool] = None) -> Selection:
    """Resolve, write the answer into ``env`` (default ``os.environ``), stamp it, announce once."""
    if env is None:
        env = os.environ
    sel = resolve(env, worker_kind=worker_kind, graph=graph)
    if sel.mode != "off":
        for k, v in sel.env_set:
            env[k] = v
        for k in sel.env_unset:
            env.pop(k, None)
        env[STAMP_ENV] = sel.stamp
    if announce:
        key = (sel.stamp, sel.reason, id(env))
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            if sel.mode != "off" or sel.source != "default":
                print(f"[mia/delivery] {sel.note}", flush=True)
            logger.info("delivery selector: %s (%s) -- %s", sel.mode, sel.source, sel.reason)
    return sel


WRITES = (GATHER_ENV, DELIVER_ENV, FLUSH_MS_ENV, PER_REQUEST_ENV, STAMP_ENV)

__all__ = ["DEFAULT_FLUSH_MS", "DeliveryConfigError", "MODES", "MODE_ENV", "STAMP_ENV",
           "Selection", "WRITES", "apply", "resolve", "validate"]
