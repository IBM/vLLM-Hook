"""
vLLM general plugin that exposes hidden-state and QK probe outputs via the
OpenAI-compatible API when ``output_hidden_states`` or ``output_qk`` is
passed in ``SamplingParams.extra_args``.

Installed automatically via the ``vllm.general_plugins`` entry point
(configured in setup.py). Patches ``EngineArgs.create_engine_config``
to inject the worker extension and eager mode, and patches
``AsyncLLM.generate`` and ``LLM.generate`` to retrieve per-request probe
outputs for both online (async) and offline (sync) usage.

For ``vllm serve``, also patches the OpenAI response builders so probe
outputs are included in HTTP responses as ``response.probes``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from collections.abc import AsyncIterator, Callable
from typing import Any

import zstandard as zstd

from mia._profiler import PROF
from mia.errors import MiaConfigurationError, MiaRefusal

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

# Populated by register() with the original unpatched methods.
_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None

_WORKER_EXT_HS = "mia.workers.hs_capture_worker.HSCaptureWorker"
_WORKER_EXT_QK = "mia.workers.qk_capture_worker.QKCaptureWorker"
_WORKER_EXT_STEER = "mia.workers.steer_worker.SteerWorker"

# ---------------------------------------------------------------------------
# MIA_WORKER — ONE parser, two readers.
# ---------------------------------------------------------------------------
# MIA_WORKER selects the subsystem for `vllm serve` (offline MiaLLM sets
# worker_extension_cls directly instead). It is read in two places:
#   1. _patched_create_engine_config -- picks which worker extension to install;
#   2. _worker_kind                   -- sizes the autocap OOM guard.
# Those two used to parse it DIFFERENTLY: (2) did .strip().lower(), (1) compared raw
# and fell through to HS for anything it did not recognize. So MIA_WORKER=probe_qk
# (which the profiling harness really shipped, in both templates and its README) ran
# the HS worker while reporting a QK campaign, and MIA_WORKER=QK sized the guard for
# QK while running HS. Both failures are silent: artifacts are written, numbers look
# clean, and they describe the wrong subsystem.
#
# The fix is this module-level parser, which BOTH sites call, and which raises on any
# value it does not recognize -- the same "fails loud (never a silent degrade)"
# contract graph/aperture_sizing.py states. Spellings are exact and there are NO
# aliases (this branch chose a full rename, no back-compat spellings): a near-miss
# like `probe_qk`, `hs`, `hidden-states`, `QK` or a stray trailing space is precisely
# the shape of the bug, so normalizing it away would re-open the hole one notch down.
MIA_WORKER_VALUES = ("hidden_states", "qk", "steer")
#: The worker extension each accepted MIA_WORKER value installs. Keys are exactly
#: MIA_WORKER_VALUES -- a value with no entry here would be an accepted-but-unroutable
#: worker, so tests assert the two stay in lockstep.
_WORKER_EXT_BY_KIND = {
    "hidden_states": _WORKER_EXT_HS,
    "qk": _WORKER_EXT_QK,
    "steer": _WORKER_EXT_STEER,
}
#: What an unset MIA_WORKER means. Documented default, not a fallback for bad input.
DEFAULT_MIA_WORKER = "hidden_states"


class UnknownMiaWorkerError(MiaRefusal, ValueError):
    """MIA_WORKER was set to something that is not one of MIA_WORKER_VALUES."""


class MiaWorkerConflictError(MiaConfigurationError):
    """MIA_WORKER names one subsystem and the engine was handed another one's worker class."""


def _kind_from_extension(worker_ext):
    """The worker kind a MIA worker EXTENSION CLASS names, or None if it is not one of MIA's
    own worker classes.

    The None case is load-bearing, and it is the reason MIA_WORKER has precedence at all: the
    profiler replaces ``worker_extension_cls`` with its OWN mixin while stashing the real
    worker elsewhere, so the class in hand names no subsystem and only the env knows. That is
    distinguishable from an explicitly-passed MIA worker class, which names one exactly --
    and where a disagreeing MIA_WORKER is a contradiction, not a default to be overridden.

    Matched against the three real class paths (dotted and bare), not by substring: the old
    ``"qk" in s`` test also matched strings that merely mention qk, and its ``else ->
    hidden_states`` fallback made a foreign mixin indistinguishable from the HS worker."""
    s = worker_ext if isinstance(worker_ext, str) else getattr(
        worker_ext, "__name__", str(worker_ext))
    sl = s.lower()
    for kind, dotted in _WORKER_EXT_BY_KIND.items():
        if dotted.lower() in sl or dotted.rsplit(".", 1)[-1].lower() in sl:
            return kind
    return None


def parse_mia_worker_env(raw):
    """Parse a raw MIA_WORKER value into 'hidden_states' | 'qk' | 'steer', or None.

    ``None`` means "not specified" -- the variable was unset or empty -- and lets each
    caller apply its own documented default (the engine-config seam installs
    ``DEFAULT_MIA_WORKER``; ``_worker_kind`` falls back to the extension-class string).
    Anything else that is not an exact match raises ``UnknownMiaWorkerError``.
    """
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

# Default hook_dir for save_to_disk requests when extra_args["hook_dir"] is not
# set. /dev/shm/mia is a RAM tmpfs on Linux — fast and ephemeral, matching
# MiaClient's default.
_DEFAULT_HOOK_DIR = "/dev/shm/mia"


def _graph_mode() -> bool:
    """True when the CUDA-graph QK capture path is armed for this process.

    Armed by ``_patched_create_engine_config`` when MIA_ALLOW_CUDAGRAPH==1.
    In graph mode the worker installs capture at load_model, so the generate
    patches must NOT also issue the lazy ``install_hooks`` collective_rpc.
    """
    from mia.graph.install import graph_mode_enabled
    return graph_mode_enabled()


def _decompress(data: bytes) -> Any:
    PROF.gauge("rpc.payload_bytes", len(data))
    with PROF.timed("rpc.decompress"):
        if data[:4] == _ZSTD_MAGIC:
            return pickle.loads(_ZSTD_DECOMPRESSOR.decompress(data))
        return pickle.loads(data)


def _aperture_per_request_mode() -> bool:
    """True when the off-loop HS capture-aperture PER-REQUEST delivery path is armed
    (``MIA_APERTURE_PER_REQUEST=1`` under graph mode). Additive gate: when False every existing
    response path (``get_captured_states`` RPC, disk flush, bank) is byte-identical -- the driver
    only reroutes an HS request's RPC retrieval to ``get_aperture_per_request`` when this is True."""
    import os
    return (os.environ.get("MIA_APERTURE_PER_REQUEST") == "1"
            and os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1")


def _aperture_per_request_kind(extra: dict, wants_hs: bool, wants_qk: bool,
                               wants_steer: bool):
    """Which per-request aperture delivery path owns this request: ``'hs'``, ``'qk'`` or None.

    The KIND matters as much as the boolean. HS and QK have SEPARATE finalize branches (one
    returns `hs_cache`, the other `qk_cache`), so a request must be routed to its own: sending a
    QK capture into the HS branch would return hidden states and silently drop `qk_cache`. A
    request that wants BOTH is owned by neither and falls through to `get_captured_states`,
    which can return both.

    Everything else -- the arming gates, ``drop``, profile mode, steering -- is shared, and
    :func:`_takes_aperture_per_request` is this function reduced to "does SOME aperture path own
    it", which is what the ``sink == "disk"`` branch needs to know in order to decline.
    """
    if not (_aperture_per_request_mode() and not _profile_mode()) or wants_steer:
        return None
    if _resolve_sink(extra) == "drop":
        return None
    if wants_hs and not wants_qk:
        return "hs"
    if wants_qk and not wants_hs:
        return "qk"
    return None


def _takes_aperture_per_request(extra: dict, wants_hs: bool, wants_qk: bool,
                                wants_steer: bool) -> bool:
    """Whether this request's capture is DELIVERED by an aperture per-request path (HS or QK).

    ONE predicate for the two sites that must agree: the request-start guard that arms the route
    (and crosses a disk transport to the drain) and the finalize gate that delivers it. They used
    to encode this condition separately, they disagreed, and that silently destroyed every
    disk-routed capture:

      * the drain builds NO shared-file sink in per-request mode
        (``aperture_drain_hs.py``: ``setup_sink=not per_request``) -- the rows are meant to be
        demuxed per request instead;
      * the request-start guard refused to arm the route when the sink was ``disk``, so nothing
        was ever staged for such a request;
      * finalize's ``sink == "disk"`` branch ran FIRST and called ``flush_disk``, which flushes
        the EAGER worker's ``_disk_states`` -- empty, because the capture went to the aperture.

    Net: rows captured, copied to host, dropped. Measured on an H100 (W13, 2026-09-21): **47 GB
    captured, 0 bytes written**, no error raised, and every harness evidence check passing --
    including "the drained bytes took the promised write path", which read direct=0 buffered=0
    legacy=0 and passed vacuously. It hit BOTH ways of asking for disk: an explicit
    ``save_to_disk: true`` AND MIA's own storage router, which writes
    ``extra["save_to_disk"] = True`` when it picks disk (``_patched_generate``) and so tripped the
    very same guard. The automatic data path's disk half delivered nothing.

    ``disk`` is therefore NOT an exclusion here -- it is the transport this path exists to serve,
    and `CAPTURE_IO.md` 8.3 ("how a disk-routed request is staged") always described that as the
    per-request staging writer, never ``flush_disk``. Only ``drop`` is excluded: a dropped request
    stores nothing by definition, and staging it would orphan NVMe files until shutdown.

    APPLIES TO QK TOO (2026-09-21, second pass). The first fix covered HS only, because W13 is an
    HS study and the QK sibling guard was left excluding ``disk`` with the note that "whether QK
    loses bytes the same way is NOT established". It does, by the same three-way coincidence:
    ``aperture_drain_qk.py`` also builds no shared sink under ``per_request``
    (``setup_sink=not per_request``), an HS-only predicate is False for a QK request so the
    ``sink == "disk"`` branch did not decline, and ``flush_disk`` then flushed empty eager
    buffers. Steering is still excluded (it stores nothing), and so is a request that wants both
    HS and QK -- see :func:`_aperture_per_request_kind`.
    """
    return _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) is not None


def _reconstruct_compact_qk(probes: dict) -> None:
    """Rebuild the padded ``k_all`` that a compact-transfer worker deferred
    (``MIA_QK_COMPACT_KALL``): the worker sends ``k_full`` + ``k_prefix_ends`` to avoid an
    O(seq^2) pad on its engine loop; here in the driver we rebuild
    ``pad_sequence([full[:L] for L in ends])``, byte-identical to the old ``k_stacked``. No-op
    when the worker sent a normal ``k_all``."""
    qk = probes.get("qk_cache") if isinstance(probes, dict) else None
    if not isinstance(qk, dict):
        return
    from torch.nn.utils.rnn import pad_sequence
    for entry in qk.values():
        if not isinstance(entry, dict):
            continue
        full = entry.pop("k_full", None)
        ends = entry.pop("k_prefix_ends", None)
        if full is not None and ends is not None:
            entry["k_all"] = pad_sequence([full[:int(L)] for L in ends], batch_first=True)


def _trim_probes(probes: dict, key: str, expected_len: int) -> None:
    """Trim probe tensors to expected_len along the sequence dimension.

    The vLLM v1 scheduler may run one extra forward pass after EOS is hit; vLLM discards the
    extra output token but capture hooks still fire, so this trims the surplus. Only tensors
    with a sequence dim (all_tokens mode) need trimming -- last_token tensors are (bs, hidden).
    """
    for entry in probes.get(key, {}).values():
        for tkey in ("hidden_states", "q", "k_all"):
            t = entry.get(tkey)
            # A quantized entry hands off a per-pass list (packed + scale + qmeta), already
            # unpadded per pass -- nothing to seq-trim; dequant happens downstream.
            if t is None or isinstance(t, list):
                continue
            # 3D = (bs, seq, hidden) -- trim seq (dim 1)
            if t.dim() == 3 and t.shape[1] > expected_len:
                PROF.incr("trim.event")
                entry[tkey] = t[:, :expected_len, :]


# ---------------------------------------------------------------------------
# Engine config patch — inject worker extension + eager mode
# ---------------------------------------------------------------------------


def _stable_artifact_files(run_dir: str) -> list:
    """Non-empty, non-``.tmp`` files under ``run_dir`` (recursive), sorted.

    Disk writes are atomic (``.tmp`` + ``os.rename``), so a visible non-``.tmp`` file is
    complete. Empty list when nothing has landed (or on OSError)."""
    import glob
    import os
    try:
        return sorted(
            f for f in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)
            if os.path.isfile(f) and not f.endswith(".tmp") and os.path.getsize(f) > 0
        )
    except OSError:
        return []


# save_to_disk=True must return with a durable artifact FILE already on disk. The writer runs
# in a non-blocking child process, so flush_disk can return before the file lands; these
# barriers poll until the artifact's file set is STABLE across one interval (so a multi-file
# artifact -- safetensors + JSON sidecar -- is fully present) before generate() returns. Runs
# in the caller's thread, never the engine loop. Bounded; on timeout the "no artifact" outcome
# stands.
#
# TENSOR PARALLELISM. The writer process runs on every TP rank, so at TP > 1 each QK rank's shard
# lands on its OWN writer's schedule. "The run dir is non-empty and stable" is then true as soon as
# the FIRST rank lands, and the caller would read 1/tp of the heads. So the barrier waits for every
# rank dir the flush_disk collective NAMED (``_flushed_rank_dirs``): each must hold a non-empty
# file set, and the union must be stable across one interval. At TP=1 that is tp_rank_0 alone --
# the same wait as before. A collective that names no dir keeps the old whole-run-dir wait.
_ARTIFACT_WAIT_S = 10.0
_ARTIFACT_POLL_S = 0.005


def _flushed_rank_dirs(results, run_id: str, hook_dir: str):
    """The rank dirs a ``flush_disk`` collective wrote, as seen from THIS (driver) process.

    ``results`` is the collective's per-worker list. A HS/QK worker returns its
    ``<hook_dir>/<run_id>/tp_rank_<r>`` when it wrote (or handed to its writer child) an artifact,
    False when it captured nothing. Each dir is re-rooted under the driver's own
    ``<hook_dir>/<run_id>`` by its rank name, so a worker that resolved ``hook_dir`` differently
    (relative path, other cwd) is still found. Sorted, de-duplicated.

    None -- the caller keeps the old whole-run-dir wait -- when no result names a dir (nothing
    captured, or not a per-worker list) or when any worker answers a bare ``True``: that worker
    says it wrote but not where. The Token Highlighter worker does this; it writes
    ``tp_rank_0`` itself, synchronously, whatever its rank, so its dir cannot be inferred from the
    worker's position."""
    import os
    if not isinstance(results, (list, tuple)) or any(r is True for r in results):
        return None
    from mia.graph.tp_shard import parse_rank_dir, rank_dir_name
    run_dir = os.path.join(hook_dir, run_id)
    dirs = set()
    for r in results:
        if isinstance(r, str) and r:
            rank = parse_rank_dir(r)
            dirs.add(os.path.join(run_dir, rank_dir_name(rank)) if rank is not None else r)
    return sorted(dirs) or None


def _artifact_barrier_state(run_dir: str, rank_dirs) -> "tuple[list, bool]":
    """``(files, complete)`` for one barrier poll. Without ``rank_dirs``: every artifact file under
    ``run_dir``, complete when there is one (the TP=1-era rule). With them: the files under those
    dirs, complete when EVERY named dir holds at least one."""
    if not rank_dirs:
        files = _stable_artifact_files(run_dir)
        return files, bool(files)
    per_rank = [_stable_artifact_files(d) for d in rank_dirs]
    return [f for fs in per_rank for f in fs], all(per_rank)


def _log_barrier_timeout(run_id: str, rank_dirs) -> None:
    """LOUD when a TP > 1 barrier gives up with some ranks' shards missing: the caller now holds a
    partial artifact, which the QK loader refuses (``TPShardError``) rather than merging."""
    import os
    if not rank_dirs or len(rank_dirs) < 2:
        return
    missing = [os.path.basename(d) for d in rank_dirs if not _stable_artifact_files(d)]
    if missing:
        print(f"[hookplugin/disk] durability barrier TIMEOUT after {_ARTIFACT_WAIT_S:.0f}s for "
              f"run_id {run_id!r}: {len(rank_dirs) - len(missing)}/{len(rank_dirs)} rank "
              f"artifact(s) landed, missing {missing}. The run's artifact is INCOMPLETE; the QK "
              f"loader will refuse it until every rank's shard is on disk.", flush=True)


def _resolve_sink(extra: dict) -> str:
    """Return this request's sink: 'disk' | 'rpc' | 'drop'.

    Precedence: MIA_SINK=drop is a global override; an explicit per-request
    save_to_disk is honored next; MIA_SINK=disk|rpc is the default when the request
    states no preference; otherwise falls back to save_to_disk semantics."""
    import os
    env = os.environ.get("MIA_SINK", "").lower()
    if env == "drop":
        return "drop"
    if "save_to_disk" in extra:
        return "disk" if bool(extra["save_to_disk"]) else "rpc"
    if env in ("disk", "rpc"):
        return env
    return "disk" if bool(extra.get("save_to_disk")) else "rpc"


async def _await_disk_artifact(run_id: str, hook_dir: str, rank_dirs=None) -> bool:
    """Serve-path barrier (``durable_wait``). ``rank_dirs``: see ``_flushed_rank_dirs``."""
    import asyncio
    import os
    run_dir = os.path.join(hook_dir, run_id)
    prev = None
    for _ in range(max(2, int(_ARTIFACT_WAIT_S / _ARTIFACT_POLL_S))):
        files, complete = _artifact_barrier_state(run_dir, rank_dirs)
        if complete and files == prev:
            return True
        prev = files
        await asyncio.sleep(_ARTIFACT_POLL_S)
    complete = _artifact_barrier_state(run_dir, rank_dirs)[1]
    if not complete:
        _log_barrier_timeout(run_id, rank_dirs)
    return complete


def _wait_disk_artifact(run_id: str, hook_dir: str, rank_dirs=None) -> bool:
    """Offline barrier (``LLM.generate``). ``rank_dirs``: see ``_flushed_rank_dirs``."""
    import os
    import time
    run_dir = os.path.join(hook_dir, run_id)
    prev = None
    for _ in range(max(2, int(_ARTIFACT_WAIT_S / _ARTIFACT_POLL_S))):
        files, complete = _artifact_barrier_state(run_dir, rank_dirs)
        if complete and files == prev:
            return True
        prev = files
        time.sleep(_ARTIFACT_POLL_S)
    complete = _artifact_barrier_state(run_dir, rank_dirs)[1]
    if not complete:
        _log_barrier_timeout(run_id, rank_dirs)
    return complete


# ---------------------------------------------------------------------------
# Per-request aperture delivery: BLOCK-UNTIL-HELD. The client contract is "the response does not
# return until the artifact/result is HELD". Polled from the async frontend (never the engine
# loop) with asyncio.sleep between polls, so EngineCore keeps decoding other requests while
# each worker-side check stays non-blocking. Bounded + loud on timeout, never an unbounded hang.
# ---------------------------------------------------------------------------
_APERTURE_DELIVER_POLL_S = 0.005


def _aperture_deliver_timeout_s() -> float:
    """Total wall-clock budget for a single request's block-until-held / disk-confirm poll, seconds.
    Env-overridable (MIA_APERTURE_DELIVER_TIMEOUT_S); a sane 30 s default (well above a healthy
    off-loop drain's per-request latency, low enough that a wedged consumer fails loud, not forever)."""
    import os
    try:
        return max(0.1, float(os.environ.get("MIA_APERTURE_DELIVER_TIMEOUT_S", "30") or "30"))
    except (TypeError, ValueError):
        return 30.0


async def _await_aperture_per_request(engine, request_id, hs_layers=None):
    """BLOCK-UNTIL-HELD for the host-buffer RPC route: poll get_aperture_per_request until it returns this
    request's marshaled probes (its off-loop finish has been processed) or the deliver timeout
    elapses. Returns the decompressed probes dict, or None on timeout (LOUD) -- the caller leaves
    output.probes unset rather than hanging.

    ``hs_layers`` is an HS request's ``output_hidden_states`` value: under the TP layer shard only
    the ranks that own one of its layers deliver a part, and this says which ones to wait for."""
    import asyncio
    import time
    timeout = _aperture_deliver_timeout_s()
    deadline = time.monotonic() + timeout
    # Each worker delivers a request EXACTLY ONCE (its stash entry is popped), and under tensor
    # parallelism every QK rank delivers its own shard on its own schedule -- so parts are
    # accumulated per worker index across polls and merged only once every rank has delivered.
    collected: dict = {}
    while True:
        with PROF.timed("rpc.get_aperture_per_request"):
            states = await engine.collective_rpc("get_aperture_per_request", args=(request_id,))
        for i, s in enumerate(states):
            if s is not None and i not in collected:
                collected[i] = _decompress(s)
        if collected:
            parts = [collected[i] for i in sorted(collected)]
            if len(parts) >= _expected_probe_parts(parts, hs_layers):
                return merge_probe_parts(parts, hs_layers)
        if time.monotonic() >= deadline:
            print(f"[hookplugin/aperture] BLOCK-UNTIL-HELD TIMEOUT after {timeout:.1f}s waiting for RPC "
                  f"per-request delivery of {request_id!r} ({len(collected)} rank part(s) held); "
                  f"leaving probes unset (raise MIA_APERTURE_DELIVER_TIMEOUT_S if the off-loop "
                  f"consumer is merely slow, else it is a bug)", flush=True)
            return None
        await asyncio.sleep(_APERTURE_DELIVER_POLL_S)


def _collect_aperture_per_request_sync(rpc, request_id, hs_layers=None):
    """The OFFLINE (``LLM.generate``) sibling of ``_await_aperture_per_request``: ONE merged probes
    dict for ``request_id``, or None when it is not (yet) deliverable. ``rpc`` is the engine's
    ``collective_rpc``.

    WHY THIS IS NOT ``merge_probe_parts([p for p in one_round if p])``. Every rank delivers a
    request EXACTLY ONCE -- ``get_aperture_per_request`` POPS its stash entry -- and under the HS
    TP layer shard each owning rank's off-loop consumer processes that request's ``_Finish`` on
    its OWN schedule. So a single RPC round can legitimately hand back 1..tp-1 of the tp parts,
    and a PARTIAL set is neither answer: merging it raises ``TPShardError`` ("missing tp_rank(s)")
    straight out of ``generate()``, while dropping it discards parts this round has already popped
    -- they are gone. Partial rounds therefore ACCUMULATE by worker index across further rounds,
    exactly as the serve path accumulates across its ``asyncio.sleep`` polls, until every expected
    rank has delivered or the deliver timeout elapses (LOUD, probes left unset).

    Two states answer on the FIRST round and never sleep, so TP = 1 keeps the pre-shard
    behaviour exactly (``_expected_probe_parts`` is 1 there, and the shard-less payloads of the
    rank-0-only / all-ranks layouts likewise):

      * a COMPLETE set -> merged and returned (at TP = 1: the same single part, same merge);
      * NOTHING delivered -> None, at once. This path is not block-until-held (that is serve-only)
        and an empty round has consumed nothing, so there is nothing to lose by answering "not
        ready" -- which is what the caller has always done with it.
    """
    import time
    collected: dict = {}
    deadline = None
    while True:
        with PROF.timed("rpc.get_aperture_per_request"):
            states = rpc("get_aperture_per_request", args=(request_id,))
        for i, s in enumerate(states):
            if s is not None and i not in collected:
                collected[i] = _decompress(s)
        if not collected:
            return None
        parts = [collected[i] for i in sorted(collected)]
        if len(parts) >= _expected_probe_parts(parts, hs_layers):
            return merge_probe_parts(parts, hs_layers)
        if deadline is None:
            deadline = time.monotonic() + _aperture_deliver_timeout_s()
        elif time.monotonic() >= deadline:
            print(f"[hookplugin/aperture] PER-REQUEST DELIVERY INCOMPLETE for {request_id!r} after "
                  f"{_aperture_deliver_timeout_s():.1f}s: {len(parts)} of "
                  f"{_expected_probe_parts(parts, hs_layers)} rank part(s) held "
                  f"(tp_rank(s) {sorted(collected)}); leaving probes unset -- the held part(s) are "
                  f"DISCARDED, each rank delivers a request only once. Raise "
                  f"MIA_APERTURE_DELIVER_TIMEOUT_S if the off-loop consumer is merely slow, else "
                  f"it is a bug", flush=True)
            return None
        time.sleep(_APERTURE_DELIVER_POLL_S)


def _expected_probe_parts(parts, hs_layers=None) -> int:
    """How many per-rank probe parts one request's delivery consists of: ``tp_size`` when the
    parts are QK shards (every TP rank captures its own heads); for HS layer shards (the TP layer
    shard, ``"hs_shard"``) the number of ranks that own one of the request's layers
    (``hs_layers``, its ``output_hidden_states`` value; None = every layer); else 1 (TP = 1, and
    HS captured on tp_rank 0 only)."""
    from mia.graph.tp_shard import (
        HS_SHARD_KEY, TP_SHARD_KEY, hs_expected_ranks, hs_requested_layers)
    for p in parts:
        shard = p.get(TP_SHARD_KEY) if isinstance(p, dict) else None
        if isinstance(shard, dict) and "tp_size" in shard:
            return int(shard["tp_size"])
    for p in parts:
        shard = p.get(HS_SHARD_KEY) if isinstance(p, dict) else None
        if isinstance(shard, dict) and "tp_size" in shard and "num_layers" in shard:
            layers = hs_requested_layers(hs_layers, int(shard["num_layers"]))
            return max(1, len(hs_expected_ranks(layers, int(shard["tp_size"]))))
    return 1


def merge_probe_parts(parts, hs_layers=None):
    """ONE probes dict from the per-worker results of a retrieval ``collective_rpc``.

    ``parts`` are the decompressed non-None results, in worker (= TP rank) order.

      * one part without TP geometry -> returned unchanged (TP=1: byte-identical).
      * QK parts carrying ``"tp_shard"`` -> ``graph/tp_shard.merge_qk_payloads``: q / k_all /
        k_full concatenated in GLOBAL head order, replicated KV heads de-duplicated, the
        ``tp_shard`` key dropped. Every rank must be present -- a missing shard raises
        ``TPShardError`` rather than handing the analyzer ``H_q / tp`` heads.
      * HS parts carrying ``"hs_shard"`` (the TP LAYER shard: each rank delivers its own layers)
        -> ``graph/tp_shard.merge_hs_payloads``: the union of the ranks' layers, ascending, the
        ``hs_shard`` key dropped. ``hs_layers`` (the request's ``output_hidden_states``) names the
        ranks that must be present and the layers the union must hold; a missing rank or layer, a
        duplicate, or a layer delivered by a rank that does not own it raises ``TPShardError``.
      * other HS parts -> the first (tp_rank 0's): without the layer shard the residual is
        captured on tp_rank 0 only, so several such parts can only be the
        ``MIA_HS_CAPTURE_ALL_RANKS`` diagnostic's copies.

    This replaces the old ``parts[0]``, which at TP > 1 silently returned rank 0's slice of the
    heads as if it were the whole layer."""
    from mia.graph.tp_shard import (
        HS_SHARD_KEY, TP_SHARD_KEY, TPShardError, merge_hs_payloads, merge_qk_payloads)
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1 and not (isinstance(parts[0], dict)
                                and (TP_SHARD_KEY in parts[0] or HS_SHARD_KEY in parts[0])):
        return parts[0]
    if all(isinstance(p, dict) and "qk_cache" in p for p in parts):
        return merge_qk_payloads(parts)
    if all(isinstance(p, dict) and "hs_cache" in p and "qk_cache" not in p for p in parts):
        if any(HS_SHARD_KEY in p for p in parts):
            return merge_hs_payloads(parts, hs_layers)
        return parts[0]
    raise TPShardError(
        f"{len(parts)} workers returned probes for one request, but they are neither all QK "
        f"shards nor all HS replicas; refusing to pick one")


async def _await_aperture_disk_confirm(engine, request_id) -> bool:
    """DISK-route CONFIRM: block until the per-request offload has landed the file at the client dest
    (confirm_aperture_delivery -> OffloadProcess.wait). Polled NON-BLOCKING (worker-side timeout 0.0 so
    the RPC never stalls the forward) with asyncio.sleep between polls. Returns True on confirmed
    delivery, False on timeout (LOUD). output.probes stays unset either way (the client reads the
    delivered file, as with save_to_disk)."""
    import asyncio
    import time
    timeout = _aperture_deliver_timeout_s()
    deadline = time.monotonic() + timeout
    while True:
        with PROF.timed("rpc.confirm_aperture_delivery"):
            res = await engine.collective_rpc("confirm_aperture_delivery", args=(request_id, 0.0))
        # collective_rpc returns one result per worker (TP=1 -> one). A worker with no per-request
        # drain answers None (an HS non-capturing TP rank); every worker that DOES stage the
        # request must have landed it -- under TP every QK rank offloads its own head shard, so
        # "any rank landed" would return with the other shards still in flight.
        staged = [r for r in res if r is not None]
        if staged and all(r is True for r in staged):
            return True
        if time.monotonic() >= deadline:
            print(f"[hookplugin/aperture] DISK-CONFIRM TIMEOUT after {timeout:.1f}s waiting for offload "
                  f"delivery of {request_id!r}; the client dest file may be incomplete (raise "
                  f"MIA_APERTURE_DELIVER_TIMEOUT_S, or check the offload worker)", flush=True)
            return False
        await asyncio.sleep(_APERTURE_DELIVER_POLL_S)


def _warn_profile_aperture_conflict() -> None:
    """Once-per-process warning: MIA_PROFILE_MODE=1 with MIA_APERTURE_PER_REQUEST=1 is a
    misconfiguration. Profile mode disables Component 2, but the host-buffer route then demuxes
    rows into an index nothing retrieves (a host-RAM leak) and misreports the capture->NVMe
    boundary. Warns rather than refuses, so a run started this way is never silently wrong."""
    import os
    if getattr(_warn_profile_aperture_conflict, "_warned", False):
        return
    if (os.environ.get("MIA_PROFILE_MODE") == "1"
            and os.environ.get("MIA_APERTURE_PER_REQUEST") == "1"):
        _warn_profile_aperture_conflict._warned = True
        print("[hookplugin/aperture] WARNING: MIA_PROFILE_MODE=1 AND MIA_APERTURE_PER_REQUEST=1 "
              "are BOTH set -- this is a misconfiguration. Profile mode must pair with the "
              "shared-file / disk Component-1 drain, NOT the host-buffer per-request route (which "
              "leaks host RAM -- rows demux into an index nothing retrieves -- and misreports the "
              "capture->NVMe boundary). Unset MIA_APERTURE_PER_REQUEST for profile-mode "
              "Component-1 measurement.", flush=True)


# vLLM version advisory: printed at most once per process (see _note_vllm_version).
_VLLM_VER_NOTED = False


def _note_vllm_version() -> None:
    """Advisory only, never a gate: warn when the running vLLM is not the pinned 0.29.x
    this branch targets and is GPU-validated against (``setup.py`` pins ``vllm==0.29.0``).
    The real gate is the V2-runner requirement -- see ``mia.runner.require_v2_runner`` and
    the ``VLLM_USE_V2_MODEL_RUNNER=0`` refusal in ``_patched_create_engine_config``; this
    function only notes an off-pin *version*. Compares with PEP440 ``Version``, not strings
    (``"0.29.10" > "0.29.9"`` is true numerically but not lexically as strings).
    Deliberately tolerant of an unparseable/absent version -- must never break engine
    construction over a log line."""
    global _VLLM_VER_NOTED
    if _VLLM_VER_NOTED:
        return
    _VLLM_VER_NOTED = True
    try:
        import vllm
        from packaging.version import Version
        raw = vllm.__version__
        ver = Version(raw)
        if ver.release[:2] == (0, 29):
            return                                  # the validated env — say nothing
        print(f"[mia] NOTE: this branch is developed and GPU-validated on vLLM 0.29.x "
              f"with the V2 model runner; found {raw} — UNTESTED on this branch.",
              flush=True)
    except Exception:  # noqa: BLE001 — an advisory must never break engine init
        return


def _worker_kind(worker_ext) -> str:
    """Resolve 'hidden_states' | 'qk' | 'steer' for autocap sizing (each worker's per-token
    transient shape differs; 'steer' has no capture aperture).

    MIA_WORKER is checked first and is authoritative: ``vllm serve`` selects the worker
    with it, and the profiler replaces ``worker_extension_cls`` with its own mixin while stashing
    the real worker here -- so keying off the extension class alone would mis-size a QK/steer run
    as HS. Falls back to the extension-class string when the env is unset (offline MiaLLM sets
    the class directly).

    The env parse is ``parse_mia_worker_env`` -- the SAME function
    ``_patched_create_engine_config`` uses to choose the worker, so the sizing here can no
    longer disagree with what actually runs. A set MIA_WORKER that is not a MIA worker
    raises here too, for the same reason it raises there."""
    import os
    env_w = parse_mia_worker_env(os.environ.get("MIA_WORKER"))
    cls_kind = _kind_from_extension(worker_ext)
    if env_w is not None:
        if cls_kind is not None and cls_kind != env_w:
            raise MiaWorkerConflictError(
                f"MIA_WORKER={env_w!r} contradicts the worker class this engine was given, "
                f"{_WORKER_EXT_BY_KIND[cls_kind]} ({cls_kind!r}). MIA used to let the env win "
                f"silently, which ran the {cls_kind!r} worker while stamping the compile-cache "
                f"key and sizing the aperture OOM guard for {env_w!r} -- defeating the very "
                f"guard that exists to stop a cross-worker compiled artifact being reused "
                f"(the loud 'KeyError: _mia_hs_host'). Set MIA_WORKER={cls_kind!r}, unset it, "
                f"or pass the {env_w!r} worker class; MIA will not guess which you meant."
            )
        return env_w
    return cls_kind if cls_kind is not None else DEFAULT_MIA_WORKER


_DTYPE_BYTES = {
    "torch.float32": 4, "torch.float": 4, "torch.float16": 2, "torch.half": 2,
    "torch.bfloat16": 2, "torch.float64": 8, "torch.double": 8,
    "torch.int8": 1, "torch.uint8": 1, "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
}


def _dtype_element_size(dt) -> int:
    """Bytes per element for a torch dtype, without importing torch (str map + the .itemsize
    attribute torch>=2.1 exposes). Defaults to 2 (bf16) if unknown."""
    itemsize = getattr(dt, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize
    return _DTYPE_BYTES.get(str(dt), 2)


def _autocap_setting():
    """Resolve the tri-state MIA_APERTURE_MAX_BATCHED_TOKENS knob (off/auto/explicit-int)."""
    import os
    from mia.graph.aperture_sizing import parse_autocap_setting
    return parse_autocap_setting(os.environ.get("MIA_APERTURE_MAX_BATCHED_TOKENS"))


#: Worker-kind vocabulary -> registry/aperture subsystem vocabulary. `steer` maps to a
#: subsystem name too, but it owns no aperture, so resolve_aperture_bytes drops it.
_WORKER_KIND_TO_SUBSYSTEM = {"hidden_states": "hs", "qk": "qk", "steer": "steer"}


def _enabled_subsystems(worker_kinds) -> set:
    """Normalize one worker kind, or an iterable of them, into the registry's subsystem
    vocabulary ("hs" | "qk" | "steer").

    Today a process runs exactly ONE subsystem, so callers pass a single string and this
    returns a one-element set. It is set-shaped because Plan 2 serves a mixed HS+QK batch
    from one engine -- the plumbing below must not re-acquire the single-kind assumption
    that this task exists to remove. Unknown kinds are dropped rather than raised on: this
    feeds a best-effort guard, and `_worker_kind` is the place that refuses bad input.

    PLAN 2 MUST REVISIT THE DROP. Dropping is safe only while there is exactly one caller
    passing exactly one already-vetted kind. The moment a SET arrives, dropping permits a
    PARTIAL drop: ``["hidden_states", "bogus"]`` yields ``{"hs"}``, which sizes the autocap
    for HS alone, under-reserves for whatever "bogus" was, and hands back a cap that is too
    HIGH -- a guard that looks armed and protects less than it claims. When a mixed install
    becomes expressible, this must BAIL OUT ENTIRELY on an unrecognized member (return
    nothing, so the guard makes no change) rather than silently sizing for the subset it
    happened to recognize. Unreachable today; it is reachable the day Plan 2 lands."""
    kinds = [worker_kinds] if isinstance(worker_kinds, str) else list(worker_kinds or ())
    return {_WORKER_KIND_TO_SUBSYSTEM[k] for k in kinds if k in _WORKER_KIND_TO_SUBSYSTEM}


def _model_dims(config) -> dict:
    """The row-shape dims aperture_sizing needs, read off the resolved vLLM config.

    ONLY the HS keys are unconditional. The attention-head keys are filled in lazily,
    because reading them for an HS-only run is how this guard FAILS OPEN: the pre-E3 code
    read ``num_attention_heads`` inside its ``if worker_kind == "qk"`` branch, and hoisting
    it out meant a config without that attribute raised into
    ``_maybe_autocap_max_batched_tokens``'s own ``except Exception``. That prints
    ``[mia] autocap SKIPPED (non-fatal): AttributeError(...)`` and leaves
    ``max_num_batched_tokens`` exactly where it was -- the OOM guard quietly stops guarding,
    on precisely the run it exists to protect, and says so only in a log line nobody reads.
    (Same reason ``num_key_value_heads`` tolerates a present-but-None value rather than
    letting ``int(None)`` do the same thing one line down.)

    Omitting the QK keys is not a silent degrade: ``per_token_row_bytes("qk", dims)`` raises
    for dims it cannot size, so the refusal still happens -- but only when QK is actually
    one of the enabled subsystems, which is where it belongs."""
    mc = config.model_config
    tc = mc.hf_text_config
    hidden = int(getattr(tc, "hidden_size"))
    dims = {
        "hidden": hidden,
        "dtype_bytes": _dtype_element_size(getattr(mc, "dtype", None)),
        "layers": int(getattr(tc, "num_hidden_layers")),
    }
    h_q = getattr(tc, "num_attention_heads", None)
    if h_q:
        h_q = int(h_q)
        h_kv = int(getattr(tc, "num_key_value_heads", None) or h_q)
        head_dim = int(getattr(tc, "head_dim", 0) or (hidden // h_q))
        # SHARDED head counts: under tensor parallelism every rank captures only its own heads
        # into its own aperture (graph/tp_shard.py), so the per-rank row -- which is what this
        # per-rank memory guard bounds -- is H_q/tp query heads + max(1, H_kv/tp) KV heads. The
        # full counts over-reserved by tp x. At TP=1 (or a config with no parallel_config) the
        # shard IS the whole layer, so the number is unchanged.
        tp = int(getattr(getattr(config, "parallel_config", None),
                         "tensor_parallel_size", 1) or 1)
        from mia.graph.tp_shard import qk_shard
        shard = qk_shard(0, tp, h_q, h_kv, head_dim)
        dims["n_q_heads"] = shard.num_local_q_heads
        dims["n_kv_heads"] = shard.num_local_kv_heads
        dims["head_dim"] = head_dim
        dims["tp"] = tp
    return dims


def _hs_layers_per_rank(config, n_layers: int) -> int:
    """The most HS layers one rank captures under the configured TP layout (``n_layers`` at TP = 1
    and without the layer shard; ``ceil(n_layers / tp)`` under it)."""
    from mia.graph.tp_shard import hs_max_owned_layers, resolve_hs_shard_mode
    tp = int(getattr(getattr(config, "parallel_config", None), "tensor_parallel_size", 1) or 1)
    return hs_max_owned_layers(int(n_layers), tp, resolve_hs_shard_mode(tp))


def _derive_safe_max_batched_tokens(config, worker_kinds):
    """Safe max_num_batched_tokens for THIS model + GPU + aperture (int, or None if inapplicable).

    ``worker_kinds`` is one worker kind or an iterable of them. The enabled *capture*
    subsystems come from ``resolve_aperture_bytes``, which also splits the fixed aperture
    budget between them; steer is absent from that dict because it has no aperture, which
    is the generalization of the old ``worker_kind not in ("hidden_states", "qk")`` guard.

    The per-step transient of a mixed install is the SUM of its subsystems' per-token row
    bytes (each writes its own egress gather + pinned staging), and the reservation they
    must fit around is the sum of their aperture slices -- so both the keys and the values
    of that dict are load-bearing here. With one capture subsystem the slice IS the whole
    budget and the sum IS that subsystem's row bytes, so the number is bit-for-bit the one
    the single-kind code produced (pinned by tests/test_aperture_sizing.py).

    Reads the GPU total via ``current_platform.get_device_total_memory`` (NVML) so it does NOT
    initialize a CUDA context in the driver process. Worst-case assumption (all_tokens, all layers)
    is deliberate — the min-only rule at the call site keeps that harmless when the real workload is
    lighter."""
    import os
    from vllm.platforms import current_platform
    from mia.graph.aperture_sizing import (
        resolve_aperture_bytes_auto, resolve_aperture_bytes, per_token_row_bytes,
        compute_safe_max_batched_tokens,
        DEFAULT_AUTOCAP_SAFETY, DEFAULT_AUTOCAP_HEADROOM_BYTES,
    )
    from mia.graph.aperture_sizing import (
        aperture_bytes_is_explicit, safe_cap_with_model_sized_aperture)
    dims = _model_dims(config)
    n_layers = dims["layers"]
    gpu_util = float(config.cache_config.gpu_memory_utilization)
    total_gpu = int(current_platform.get_device_total_memory(0))
    budget = resolve_aperture_bytes_auto(total_gpu, gpu_util)
    slices = resolve_aperture_bytes(
        _enabled_subsystems(worker_kinds), gpu_bytes_budget=budget, model_dims=dims)
    if not slices:                       # steer-only (or nothing): no aperture, no cap
        return None
    if set(slices) == {"hs"}:
        # HS under the TP LAYER shard (MIA_HS_TP_SHARD, default at TP > 1): each rank captures only
        # its round-robin share of the layers, so the per-rank HS transient -- what this per-rank
        # guard bounds -- is ceil(L / tp) layers wide (rank 0's share), not L. At TP = 1, and in
        # the rank-0-only / all-ranks layouts, it is every layer, as before. (A mixed HS + QK
        # install is not expressible yet; it keeps the full count.)
        n_layers = _hs_layers_per_rank(config, n_layers)
    aperture_bytes = sum(slices.values())
    plt = sum(per_token_row_bytes(s, dims) for s in slices)
    safety = int(os.environ.get("MIA_APERTURE_AUTOCAP_SAFETY") or DEFAULT_AUTOCAP_SAFETY)
    headroom = int(os.environ.get("MIA_APERTURE_AUTOCAP_HEADROOM_BYTES") or DEFAULT_AUTOCAP_HEADROOM_BYTES)
    cap = compute_safe_max_batched_tokens(
        total_gpu, gpu_util, aperture_bytes, n_layers, plt, safety=safety, headroom_bytes=headroom)
    if (cap is not None and len(slices) == 1 and not aperture_bytes_is_explicit()
            and cap * n_layers * plt > aperture_bytes):
        # The install grows a DEFAULT aperture to one max-token step
        # (aperture_sizing.resolve_aperture_bytes_auto(rows_needed=...)), so at this cap the
        # real aperture would exceed the fixed budget assumed above. Re-solve with the aperture
        # sized to the cap itself. Unreachable while one cap-sized step fits in 4 GiB -- every
        # pinned number (tests/test_aperture_sizing.py) stays bit-for-bit.
        cap = safe_cap_with_model_sized_aperture(
            total_gpu, gpu_util, aperture_bytes, n_layers, plt,
            safety=safety, headroom_bytes=headroom)
    return cap


def _maybe_autocap_max_batched_tokens(config, worker_kinds) -> None:
    """Min-only OOM guard: lower ``config.scheduler_config.max_num_batched_tokens`` so a heavy
    full-graph capture's per-step transient fits the free GPU margin.

    Applied after the config is built (against vLLM's resolved budget), so it only ever lowers,
    never raises. Fires only when armed (MIA_APERTURE_MAX_BATCHED_TOKENS) and only when at least
    one enabled subsystem actually captures (steer has no aperture). ``worker_kinds`` is one
    worker kind or an iterable of them. Best-effort: any failure leaves the config byte-identical."""
    try:
        mode, explicit = _autocap_setting()
        if mode == "off":
            return
        subsystems = _enabled_subsystems(worker_kinds)
        from mia.graph.aperture_sizing import CAPTURE_SUBSYSTEMS
        if not (subsystems & set(CAPTURE_SUBSYSTEMS)):
            return                       # steer-only: nothing captures, nothing to bound
        if mode == "explicit":
            safe = int(explicit)
        else:
            safe = _derive_safe_max_batched_tokens(config, worker_kinds)
        if safe is None:
            return
        from mia.graph.aperture_sizing import apply_min_only
        sc = config.scheduler_config
        current = getattr(sc, "max_num_batched_tokens", None)
        new = apply_min_only(current, safe)
        if new is not None and new != current:
            sc.max_num_batched_tokens = int(new)
            print(f"[mia] autocap: max_num_batched_tokens {current} -> {new} "
                  f"(subsystems={','.join(sorted(subsystems))}, mode={mode}); "
                  f"bounds full-graph capture per-step transient", flush=True)
    except Exception as e:  # noqa: BLE001 -- must never break engine init
        print(f"[mia] autocap SKIPPED (non-fatal): {e!r}", flush=True)


# ---------------------------------------------------------------------------
# torch.compile / AOT cache key — MIA's baked op is INVISIBLE to it by default
# ---------------------------------------------------------------------------
# GPU-measured (LSF 1701974): a QK graph run that booted on a machine where an HS
# graph run had already compiled died in `profile_run` with
# `KeyError: '_mia_hs_host'` raised from inside `self.aot_compiled_fn` -- vLLM had
# loaded the HS process's compiled artifact into the QK process.
#
# Why. MIA installs its capture/steer op by class-wrapping the traced forward
# (`install_hs.py::_wrap_layer_class` wraps the DECODER LAYER and reads
# `_mia_hs_host`; `install.py::_wrap_attn_class` wraps `Attention` and reads
# `_mia_qk_host`; `install_steer.py` wraps the layer for its own host). Which
# wrapper is installed is decided by `worker_extension_cls` -- and vLLM 0.29 lists
# `worker_extension_cls` in `ParallelConfig.compute_hash`'s `ignored_factors`
# (vllm/config/parallel.py), so it is EXCLUDED from the compile-cache key on
# purpose. Every MIA worker kind therefore hashes to the same key while tracing
# different code, and whichever kind compiled first wins the cache for all of them.
#
# 0.29 makes this reachable by default rather than exotic: `VLLM_USE_AOT_COMPILE`
# is declared `False` in envs.py but its resolver returns "1" whenever torch >=
# 2.10 and the compile cache is enabled -- which is this environment (torch 2.13).
#
# `additional_config` IS a hash factor (vllm/config/vllm.py hashes
# `json.dumps(additional_config, sort_keys=True)`), so stamping the worker kind
# there gives each kind its own cache entry. Graph mode only: eager never compiles
# the wrapped forward, and leaving the config untouched there keeps the eager path
# byte-identical to a no-plugin engine in every respect.
#
# The worker kind is necessary and NOT sufficient. The same kind can bake buffers of
# different SHAPES -- two HS capture layouts at the same model and TP differ by 8 vs 32
# owned layers per rank, hence by aperture rows R, hence by every hs_buf's (R+1, hidden).
# GPU-measured (LSF 1794720): an all-ranks HS engine loaded a sharded HS engine's artifact
# and died on `expected size 16385==1 / 16385==65537`. mia_graph_layout() below is the
# third component of the stamp and carries that geometry.


_COMPILE_CACHE_STAMP_KEY = "mia_graph_capture"


def _mia_source_id() -> str:
    """A token that changes whenever MIA's OWN code changes.

    The worker kind alone is not enough. MIA's baked op is part of the traced forward, so
    EDITING MIA changes the compiled artifact -- but nothing vLLM hashes moves, and the
    stale artifact is silently reused. Harmless-looking in a demo, and a trap for profiling:
    the measurement would be of the previously compiled code.

    Git SHA when the tree is a checkout, else the package version. An editable install
    imports the WORKING TREE, not the commit, so an uncommitted edit also has to move the
    token -- a bare `+dirty` marker would let two different uncommitted states share one
    cache entry, so the working-tree state itself is hashed.

    That hash covers TRACKED changes (`git diff HEAD`) *and* UNTRACKED files under `mia/`.
    Untracked matters and is not hypothetical: adding a NEW module that the baked op imports
    is the ordinary way this package grows, and `git diff` does not see it -- the token would
    not move and the stale compiled artifact would be reused, which in the profiling phase
    means measuring the previously compiled code.

    Cached per process: `create_engine_config` is not hot, but this shells out to git and
    there is no reason to pay it on every engine boot in a multi-engine driver.
    """
    import os
    import subprocess

    cached = getattr(_mia_source_id, "_cached", None)
    if cached is not None:
        return cached

    version = "0.6.0"
    try:
        from importlib.metadata import version as _v
        version = _v("mia")
    except Exception:  # noqa: BLE001
        pass
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode == 0:
            import hashlib

            state = hashlib.sha256()
            diff = subprocess.run(["git", "-C", root, "diff", "HEAD", "--", "mia"],
                                  capture_output=True, text=True, timeout=10)
            if diff.returncode == 0:
                state.update(diff.stdout.encode())
            dirty = bool(diff.returncode == 0 and diff.stdout.strip())
            # UNTRACKED files under mia/, contents included -- a new module the baked op
            # imports is invisible to `git diff`.
            untracked = subprocess.run(
                ["git", "-C", root, "ls-files", "--others", "--exclude-standard", "--", "mia"],
                capture_output=True, text=True, timeout=10)
            if untracked.returncode == 0:
                for rel in sorted(untracked.stdout.split()):
                    state.update(rel.encode())
                    try:
                        state.update(Path(root, rel).read_bytes())
                    except OSError:
                        state.update(b"<unreadable>")
                    dirty = True
            suffix = "+" + state.hexdigest()[:8] if dirty else ""
            result = f"{version}-{sha.stdout.strip()}{suffix}"
            _mia_source_id._cached = result
            return result
    except Exception:  # noqa: BLE001
        pass
    _mia_source_id._cached = version
    return version


def mia_graph_layout(engine_args, worker_kind: str) -> dict:
    """The MIA state that SHAPES the compiled graph and that vLLM's own cache key cannot see.

    WHY THIS EXISTS (GPU-measured, LSF 1794720, model-scale W8). A TP=4 parity leg ran the
    ``hs`` workload and then the ``hs_replicas`` workload (``MIA_HS_CAPTURE_ALL_RANKS=1``)
    against ONE ``VLLM_CACHE_ROOT``. Same model, same TP, same worker kind, same MIA source --
    so the ``{worker, mia}`` stamp was identical and vLLM handed the second engine the FIRST
    one's ``torch_aot_compile`` artifact. It died inside ``determine_available_memory()``,
    before CUDA-graph capture, with inductor assertions::

        expected size 16385==1, stride ...
        expected size 16385==65537, stride ...

    Those three numbers are exactly the three HS buffer geometries of Llama-3.1-8B at TP=4:

    * ``65537`` -- an OWNED layer under the round-robin layer shard. Each rank owns 8 of the 32
      layers, so ``R = aperture_bytes // (8 * hidden * elem) = 65536`` rows, and each owned
      layer's ``hs_buf`` is ``(R + 1, hidden)`` (row ``R`` is the aperture SENTINEL).
    * ``1`` -- the 1-row SINK an unowned layer bakes ``capture_hs`` against so the graph stays
      symmetric across ranks (``graph/install_hs.py``).
    * ``16385`` -- the replication layout: every rank owns all 32 layers, so ``R = 16384``.

    MIA's buffers are op INPUTS baked into the traced forward, so their shapes are part of the
    compiled artifact -- yet every quantity above is decided by MIA env, of which vLLM hashes
    nothing. This helper is that missing half of the key.

    WHAT IS ALREADY COVERED, and therefore deliberately NOT repeated here:

    * ``num_layers``, ``hidden_size``, ``head_dim``, the q/kv head counts, the dtype --
      ``ModelConfig.compute_hash()``.
    * ``tensor_parallel_size`` -- ``ParallelConfig.compute_hash()`` (it is not in that method's
      ``ignored_factors``; ``worker_extension_cls`` IS, which is the older half of this stamp).
      It is recorded below anyway, because the HS shard EXPRESSION only reads alongside it.
    * ``max_num_batched_tokens`` / ``max_num_seqs`` -- ``SchedulerConfig.compute_hash()``. That
      is the routing slabs' ``cap``, and it also covers ``MIA_APERTURE_MAX_BATCHED_TOKENS``:
      the autocap LOWERS ``config.scheduler_config.max_num_batched_tokens`` in place, and vLLM
      computes the key later, in the worker (``compilation/caching.py``), off the lowered value.
    * ``cudagraph_capture_sizes`` / ``max_cudagraph_capture_size`` --
      ``CompilationConfig.compute_hash()``, which also covers ``MIA_CUDAGRAPH_SIZES`` and
      ``MIA_CUDAGRAPH_MAX_CAPTURE`` (both are applied to ``compilation_config``).
    * THIS RANK's identity: vLLM writes the artifact under ``<hash>/rank_{rank}_{dp_rank}/``
      (``compilation/decorators.py``), so two ranks never share an entry. That is why the HS
      entry below names the rank's owned-layer SET as the RULE ``i % tp_size == tp_rank``
      rather than as a count: the rule, plus ``num_layers`` (vLLM's key), plus the rank (the
      directory) pin that count exactly -- and the expression stays a pure function of the
      configuration. Deriving the count here instead would mean reading an HF config a second
      time, which can fail on one boot and succeed on the next; a stamp that flaps costs a
      cold recompile per launch, and this one must never flap.

    ``gpu_memory_utilization`` IS in ``CacheConfig.compute_hash``'s ``ignored_factors``, and
    that is safe for us: in the AUTO aperture path it only GATES (``MiaSizingError`` when the
    aperture does not fit the free margin) and never changes the resolved byte budget, so it
    cannot move ``R``.

    DELIBERATELY LEFT OUT -- MIA state that was checked and does NOT shape the graph. Each of
    these would only ever cost false cache MISSES, so leaving them out is a judgement, not an
    oversight:

    * ``MIA_CAPTURE_FUSED`` / ``MIA_STEER_FUSED``. They pick which kernel the op BODY
      dispatches to, at call time (``graph/ops.py``). The traced graph holds a call to
      ``mia::capture_hs`` / ``capture_qk`` / ``steer_buffer``, never the body, and the two
      paths are byte-identical by construction.
    * The routing builders: ``MIA_ROUTE_VECTORIZED``, ``MIA_ROUTE_DECODE_CACHE``,
      ``MIA_STEER_GPU_ROUTING``, ``MIA_INCREMENTAL_ROUTING``, ``MIA_STEER_ROUTE_FASTPATH``.
      They change how the SAME ``capture_index`` / ``coeff`` / ``vec_id`` slabs are FILLED on
      the host between steps. Those slabs are shaped by ``cap`` and ``num_layers``, which vLLM
      hashes; ``MIA_STEER_GPU_ROUTING`` additionally allocates ``slot_*`` mirrors, and those
      are the router's, never operands of the baked op.
    * Everything downstream of the graph: ``MIA_APERTURE_DIR``, ``MIA_APERTURE_SYNC_DRAIN``,
      ``MIA_APERTURE_PER_REQUEST``, ``MIA_APERTURE_WRITE_MODE``, the writer-thread count,
      ``MIA_APERTURE_BACKPRESSURE_TIMEOUT_S`` / ``_POLL_S``, ``MIA_DRAIN_SELECTIVE``,
      ``MIA_WRITER_PROCESS``, ``MIA_BATCHED_EGRESS``. The drain reads the aperture after the
      step; none of it appears in the compiled forward.
    * ``MIA_APERTURE_AUTOCAP_SAFETY`` / ``_HEADROOM_BYTES``: they only move the autocapped
      ``max_num_batched_tokens``, which vLLM hashes (above).
    * ``MIA_QK_SCORE`` (this path refuses it at install), ``MIA_QK_PREFIXK_STASH`` and
      ``MIA_QK_NO_PREFIXK`` (prefix-K reconstruction happens at drain/retrieval, outside the
      graph), ``MIA_STEER_CONFIG`` (WHICH vectors, not how many rows the table has --
      ``MIA_STEER_VMAX`` sizes it, and that IS below).
    * ``hf_overrides``. vLLM lists it in ``ModelConfig.compute_hash``'s ``ignored_factors``, so
      an override of ``num_hidden_layers`` moves neither vLLM's key nor ours. That is a vLLM
      gap, not a MIA one -- it mis-keys the BASE graph identically, with or without MIA -- and
      papering over it here would hide it.

    Returns a JSON-native dict; ``VllmConfig.compute_hash`` encodes ``additional_config`` with
    ``json.dumps(..., sort_keys=True)``.
    """
    import os

    kind = str(worker_kind)
    tp_size = int(getattr(engine_args, "tensor_parallel_size", 1) or 1)
    # R = aperture_bytes // (owned_layers * width * elem) in BOTH capture paths, so the byte
    # budget is a direct factor of every capture buffer's row count. An explicit value always
    # wins (aperture_sizing.resolve_aperture_bytes_auto); "auto" is the 4 GiB default grown to
    # one max-token step, which is a function of hashed quantities only.
    raw_aperture = os.environ.get("MIA_APERTURE_GPU_BYTES")
    if raw_aperture is None or not raw_aperture.strip():
        aperture = "auto"
    else:
        try:
            aperture = str(int(raw_aperture))
        except ValueError:
            aperture = raw_aperture.strip()   # the install refuses it; keep the key distinct
    layout: dict = {"kind": kind, "tp_size": tp_size, "aperture_gpu_bytes": aperture}

    if kind == "hidden_states":
        from mia.graph.tp_shard import (
            HS_ALL_RANKS_ENV, HS_LAYER_SHARD_RULE, HS_MODE_ALL_RANKS, HS_MODE_RANK0,
            HS_MODE_ROUND_ROBIN, HS_SHARD_ENV, resolve_hs_shard_mode)
        mode = resolve_hs_shard_mode(tp_size)
        # The owned-layer SET, as the rule that produces it. Every layer in it is an
        # (R + 1, hidden) buffer; every layer outside it is a 1-row sink when the graph is
        # symmetric, and carries no op at all when it is not.
        owned = {
            HS_MODE_ROUND_ROBIN: f"layers i where i % {tp_size} == tp_rank",
            HS_MODE_RANK0: "every layer on tp_rank 0, none on any other rank",
            HS_MODE_ALL_RANKS: "every layer on every rank",
        }.get(mode, "every layer (single rank)")
        layout.update({
            "hs_layout": mode,
            # Redundant with hs_layout, and kept on purpose: if resolve_hs_shard_mode's
            # precedence ever changes, the RAW flags keep this key honest. The cost is a false
            # cache MISS (setting a flag that this tp_size ignores recompiles), never a false
            # hit -- and a miss is the safe direction.
            "hs_shard_env": os.environ.get(HS_SHARD_ENV) or "",
            "hs_all_ranks_env": os.environ.get(HS_ALL_RANKS_ENV) or "",
            "hs_layer_shard_rule": HS_LAYER_SHARD_RULE if mode == HS_MODE_ROUND_ROBIN else "none",
            "hs_owned_layers": owned,
            # MIA_HS_TP_SYMMETRIC=0 drops the sinks: the unowned layers then bake NO op at all,
            # which is a different traced forward, not merely a different shape.
            "hs_sinks": bool(tp_size > 1 and os.environ.get("MIA_HS_TP_SYMMETRIC", "1") != "0"),
            "hs_capture_mode": (os.environ.get("MIA_HS_CAPTURE", "buffer") or "").strip().lower(),
        })
    elif kind == "qk":
        # QK bakes (R + 1, q_dim) and (R + 1, k_dim) per layer, one pair on EVERY rank -- there
        # is no QK layer shard, each rank captures its own heads. q_dim/k_dim are the model's
        # head widths over tp_size, both of which vLLM hashes, so the only MIA-side factor of
        # R is the aperture budget above.
        layout.update({
            "qk_capture_mode": (os.environ.get("MIA_QK_CAPTURE", "buffer") or "").strip().lower(),
            "qk_head_shard": f"1/{tp_size} of the q and kv heads on every rank",
        })
    elif kind == "steer":
        # vec_table is (V_max, hidden) and avg_proj_table is (V_max,); both are steer_buffer
        # operands, so V_max is a shape in the graph (graph/install_steer.py::SteerRegistry).
        raw_vmax = os.environ.get("MIA_STEER_VMAX", "16") or "16"
        try:
            v_max: object = int(raw_vmax)
        except ValueError:
            v_max = raw_vmax   # the install refuses it; keep the key distinct
        layout.update({
            "steer_mode": (os.environ.get("MIA_STEER_MODE", "buffer") or "").strip().lower(),
            "steer_v_max": v_max,
        })
    return layout


def stamp_compile_cache_key(engine_args, worker_kind: str) -> None:
    """Make MIA's baked-op variant part of vLLM's compile-cache key. Idempotent.

    THREE components: the worker KIND (which class the op is baked into), the MIA SOURCE (what
    that code is) and the graph LAYOUT (what shape MIA's baked buffers are --
    ``mia_graph_layout``, added after the W8 crash where two HS layouts shared one artifact).

    Must run BEFORE `create_engine_config`, because the key is computed from the config that
    call builds. The caller's dict is never mutated -- a COPY is assigned, so an
    `additional_config` the caller still holds a reference to (or reuses for a second engine)
    is left exactly as they passed it.
    """
    stamp = {"worker": str(worker_kind), "mia": _mia_source_id(),
             "layout": mia_graph_layout(engine_args, worker_kind)}
    current = getattr(engine_args, "additional_config", None)
    if isinstance(current, dict):
        if current.get(_COMPILE_CACHE_STAMP_KEY) == stamp:
            return
        engine_args.additional_config = {**current, _COMPILE_CACHE_STAMP_KEY: stamp}
    elif current is None:
        engine_args.additional_config = {_COMPILE_CACHE_STAMP_KEY: stamp}
    else:
        # A non-dict additional_config (a SupportsHash object) is the caller's, and
        # mutating it is not safe. Say so instead of silently sharing a cache entry
        # across worker kinds -- the failure that would follow is a KeyError from
        # deep inside a compiled artifact, which reads like a MIA bug.
        print(f"[mia] WARNING: additional_config is a {type(current).__name__}, not a dict; "
              f"cannot stamp the compile-cache key with worker={worker_kind}. If you run more "
              f"than one MIA worker kind on this machine, set VLLM_DISABLE_COMPILE_CACHE=1.",
              flush=True)
        return
    import json as _json
    print(f"[mia] compile-cache key stamped with worker={worker_kind} mia={stamp['mia']} "
          f"layout={_json.dumps(stamp['layout'], sort_keys=True)} "
          f"(worker_extension_cls is excluded from vLLM's own key)", flush=True)


class UnsupportedGraphModeError(RuntimeError):
    """MIA supports eager (NONE) and FULL CUDA graphs. PIECEWISE is out of scope."""


def validate_graph_mode(mode_name: str) -> None:
    """Accept only the two modes MIA's capture path is validated for.

    MEASURED on a GPU node at vLLM 0.29 (LSF 1696603): the DEFAULT resolved mode is
    `FULL_AND_PIECEWISE`, not FULL — so this check fires on an out-of-the-box engine, and
    its message is the first thing most users will see. It must name the fix, not just the
    problem. FULL_AND_PIECEWISE is refused rather than accepted because it dispatches some
    batch shapes through PIECEWISE graphs, where MIA's in-graph capture op is unvalidated;
    silently accepting it would put capture on an unproven path.
    """
    if str(mode_name).upper() not in {"NONE", "FULL"}:
        raise UnsupportedGraphModeError(
            f"MIA supports cudagraph_mode NONE (eager) and FULL; got {mode_name}. "
            f"vLLM 0.29 defaults to FULL_AND_PIECEWISE, so this is expected on a default "
            f"engine: pass compilation_config={{'cudagraph_mode': 'FULL'}} for graph capture, "
            f"or enforce_eager=True for the eager path."
        )


def validate_v2_runner_selected(config) -> None:
    """Refuse a config that will build vLLM's V1 model runner. MIA is V2-only.

    THE EARLIER OF TWO GATES, and the better one: it fires at engine-config time, before a
    model is loaded, which is where a configuration error belongs. The later gate is
    ``require_v2_runner`` at install, which now propagates out of
    ``graph/install.py::patch_worker_load_model`` instead of being swallowed.

    Why this was needed at all. ``_patched_create_engine_config`` already refuses an explicit
    ``VLLM_USE_V2_MODEL_RUNNER=0``, but that is only ONE of the ways 0.29 lands on V1.
    ``VllmConfig.use_v2_model_runner`` (vllm/config/vllm.py) also returns False, with the env
    var UNSET, for: ngram speculative decode, sequence parallelism at TP>1,
    ``STOCK_TORCH_COMPILE``, pipeline parallelism with the external launcher, some ROCm
    architectures, and a missing Triton. On every one of those the env check passes, MIA
    installs against V1, and — in graph mode, where the install-time refusal used to be
    swallowed — the engine reported success and captured nothing.

    Reads the property DEFENSIVELY and refuses only on a definite False. The attribute is a
    computed property that imports ``vllm.platforms``; if a future vLLM removes it or it
    raises, this gate declines to judge rather than inventing a verdict, and the install-time
    ``require_v2_runner`` remains as the backstop. Silence here is never a pass on its own.
    """
    try:
        selected = getattr(config, "use_v2_model_runner")
    except Exception:  # noqa: BLE001 — cannot read it; the install-time gate still applies
        return
    if selected is False:
        from mia.runner import UnsupportedRunnerError
        raise UnsupportedRunnerError(
            "MIA requires vLLM's V2 model runner, but this engine config resolves to V1 "
            "(VllmConfig.use_v2_model_runner is False) with VLLM_USE_V2_MODEL_RUNNER unset. "
            "vLLM 0.29 falls back to V1 for ngram speculative decode, sequence parallelism at "
            "TP>1, STOCK_TORCH_COMPILE, pipeline parallelism with the external launcher, some "
            "ROCm architectures, and a missing Triton — check vLLM's own "
            "'using the V1 model runner instead' log line for which one applies here. MIA "
            "refuses at config time rather than installing against V1 and capturing nothing."
        )


def _cudagraph_mode_name(cudagraph_mode) -> str:
    """Extract a checkable name from ``config.compilation_config.cudagraph_mode``.

    VERIFIED against the installed 0.29 source (vllm/config/compilation.py's
    ``CUDAGraphMode`` enum + field validator, and vllm/config/vllm.py's
    ``VllmConfig.__post_init__``): by the time ``_original_create_engine_config`` returns,
    this field is always a resolved ``CUDAGraphMode`` enum member, never a bare string or
    ``None``. The optimization-level preset seeds it (``CUDAGraphMode.FULL_AND_PIECEWISE``
    by default), and ``__post_init__`` always reassigns a concrete member afterwards (e.g.
    to ``NONE`` under ``enforce_eager``, or ``PIECEWISE``/``FULL_DECODE_ONLY`` for pooling /
    encoder-decoder models) before returning.

    Handled defensively anyway, but deliberately WITHOUT a blind
    ``getattr(mode, "name", "NONE")`` fallback: that would silently wave an unrecognized
    shape through disguised as the safe eager case, which is worse than raising. An enum
    member uses ``.name``; a raw string (a test double, or a future vLLM version that hands
    one back) is used as-is; anything else — including ``None`` — raises loudly instead of
    guessing "NONE".
    """
    if isinstance(cudagraph_mode, str):
        return cudagraph_mode
    name = getattr(cudagraph_mode, "name", None)
    if name is not None:
        return name
    raise UnsupportedGraphModeError(
        f"MIA could not read a cudagraph_mode name from vLLM's engine config (got "
        f"{cudagraph_mode!r} of type {type(cudagraph_mode).__name__}); refusing rather "
        f"than silently treating an unrecognized shape as NONE."
    )


def _patched_create_engine_config(self, *args, **kwargs):
    """Inject worker extension and (legacy) force eager mode before VllmConfig.

    When ``MIA_ALLOW_CUDAGRAPH=1`` the caller's ``enforce_eager`` stands and the
    CUDA-graph QK capture path is armed (graph/install.py); otherwise eager mode is forced
    so the legacy ``register_forward_hook`` path works."""
    import os
    if not self.worker_extension_cls:
        # Default to the hidden-states worker; MIA_WORKER overrides. parse_mia_worker_env
        # RAISES on an unrecognized value instead of falling through to HS -- see the
        # comment on that function: the old catch-all ran the wrong subsystem in silence.
        worker_type = parse_mia_worker_env(os.environ.get("MIA_WORKER")) or DEFAULT_MIA_WORKER
        self.worker_extension_cls = _WORKER_EXT_BY_KIND[worker_type]
    # Worker kind (hidden_states | qk | steer), from the extension whether set by the caller
    # (offline MiaLLM) or defaulted above (serve). Feeds the autocap guard below.
    _wkind = _worker_kind(self.worker_extension_cls)

    # MIA runs on vLLM's V2 model runner ONLY. 0.29 selects V2 by default; an explicit
    # VLLM_USE_V2_MODEL_RUNNER=0 would hand us a runner whose internals MIA no longer
    # reads, so refuse it here rather than capture nothing later. The worker-side
    # require_v2_runner() is the belt to this braces.
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0":
        raise RuntimeError(
            "MIA requires vLLM's V2 model runner, but VLLM_USE_V2_MODEL_RUNNER=0 is set. "
            "Unset it (0.29 defaults to V2) or set it to 1."
        )

    # Pipeline parallelism is refused LOUDLY, here, before any model is built: under PP each rank
    # owns only its stage's layers and the rest are PPMissingLayer identities that MIA's layer
    # matcher still hooks -- capture would zero-fill them and steering would reach one stage.
    from mia.graph.tp_shard import refuse_pipeline_parallel, resolve_hs_shard_mode
    refuse_pipeline_parallel(getattr(self, "pipeline_parallel_size", 1), "engine arguments")

    # MIA_HS_TP_SHARD: refuse a value that is neither 0 nor 1 HERE, before the model is built,
    # rather than at the HS install in a worker subprocess (the same reason MIA_WORKER is parsed
    # here). The value itself is applied per rank in graph/install_hs.py.
    resolve_hs_shard_mode(int(getattr(self, "tensor_parallel_size", 1) or 1))

    # Advisory only (never a gate): note when we're off the pinned 0.29 vLLM version.
    _note_vllm_version()

    graph_mode = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"
    if graph_mode:
        # Graph mode: leave enforce_eager as the caller set it, and arm the
        # load_model install path for the worker subprocess(es).
        from mia.graph.install import set_graph_mode
        set_graph_mode(True)
        # Graph mode is the only mode that compiles the wrapped forward, and each
        # worker kind wraps a different class and reads a different host attribute.
        # See stamp_compile_cache_key above.
        stamp_compile_cache_key(self, _wkind)
    else:
        # Eager mode is mandatory for the forward-hook capture path.
        self.enforce_eager = True

    # Opt-in: densify the FULL decode cudagraph capture sizes past vLLM's default cap of
    # min(max_num_seqs*2, 512). Above that the saturated batch can no longer replay the decode
    # graph and falls back to eager. Setting max_cudagraph_capture_size before the config is
    # built lets FULL capture decode graphs up to the saturated batch, at the cost of more
    # warmup + cudagraph pool memory (why this is opt-in). MIA_CUDAGRAPH_SIZES overrides
    # the whole list; _MAX_CAPTURE sets the ceiling and lets vLLM regenerate the fine list.
    # Graph mode only.
    if graph_mode:
        _max_cap = os.environ.get("MIA_CUDAGRAPH_MAX_CAPTURE")
        _sizes = os.environ.get("MIA_CUDAGRAPH_SIZES")
        if _max_cap or _sizes:
            try:
                cc = self.compilation_config
                size_list = ([int(x) for x in _sizes.split(",") if x.strip()]
                             if _sizes else None)
                max_n = int(_max_cap) if _max_cap else (max(size_list) if size_list else None)
                def _apply(setter):
                    if size_list is not None:
                        setter("cudagraph_capture_sizes", sorted(set(size_list)))
                    else:
                        setter("cudagraph_capture_sizes", None)  # regenerate up to max_n
                    if max_n is not None:
                        setter("max_cudagraph_capture_size", max_n)
                if isinstance(cc, dict):
                    _apply(lambda k, v: cc.__setitem__(k, v))
                elif cc is not None:
                    _apply(lambda k, v: setattr(cc, k, v))
                else:
                    self.compilation_config = {
                        **({"cudagraph_capture_sizes": sorted(set(size_list))}
                           if size_list is not None else {}),
                        **({"max_cudagraph_capture_size": max_n} if max_n is not None else {}),
                    }
                print(f"[mia] Tier 3: cudagraph capture densified "
                      f"(max={max_n}, sizes={size_list or 'auto'})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[mia] Tier 3 cudagraph densify FAILED, default sizes: {e!r}",
                      flush=True)

    assert _original_create_engine_config is not None
    config = _original_create_engine_config(self, *args, **kwargs)

    # Reject any CUDA-graph mode MIA has not validated. Read AFTER the real
    # create_engine_config call (not before): cudagraph_mode is only resolved to a
    # concrete CUDAGraphMode member once VllmConfig.__post_init__ has run (see
    # _cudagraph_mode_name) — self.compilation_config beforehand may still be an
    # unvalidated dict, a string, or None.
    validate_graph_mode(_cudagraph_mode_name(config.compilation_config.cudagraph_mode))
    validate_v2_runner_selected(config)
    # The resolved config is the authority (a future vLLM could derive PP from something other
    # than the engine argument checked above).
    refuse_pipeline_parallel(
        getattr(getattr(config, "parallel_config", None), "pipeline_parallel_size", 1),
        "resolved engine config")

    # Graph mode only: min-only auto-cap of max_num_batched_tokens so heavy full-graph capture's
    # per-step transient cannot OOM at high batch. Applied post-build against vLLM's resolved
    # budget, so it only ever lowers. Armed by MIA_APERTURE_MAX_BATCHED_TOKENS (default off).
    # `[_wkind]` not `_wkind`: one process runs one subsystem TODAY, but the guard is sized
    # over a SET so Plan 2's mixed HS+QK engine has somewhere to stand (Task E3).
    if graph_mode:
        _maybe_autocap_max_batched_tokens(config, [_wkind])

    # Buffer mode declares no splitting op: the capture/steer kernel must be absorbed into the
    # FULL decode cudagraph, and declaring a splitting op under FULL would force a fallback to
    # FULL_AND_PIECEWISE (reintroducing an eager seam). Nothing to declare here.

    return config


# ---------------------------------------------------------------------------
# Generate patch — install hooks and attach probes to output
# ---------------------------------------------------------------------------


def _prompt_token_len(prompt):
    """Best-effort prompt token length for the serve-path size model."""
    try:
        toks = getattr(prompt, "prompt_token_ids", None)
        if toks is not None:
            return len(toks)
        if isinstance(prompt, dict):
            t = prompt.get("prompt_token_ids")
            if t is not None:
                return len(t)
    except Exception:  # noqa: BLE001
        return None
    return None


def _qk_model_dims(engine):
    """(H_q, H_kv, head_dim) for the serve-path size model, cached on the engine.

    Full/unsharded counts from the engine's text config; head_dim = hidden // H_q to
    match the worker ``_conf`` (and MiaLLM's offline dims). Cached so it is read once.
    """
    cached = getattr(engine, "_mia_qk_dims", "missing")
    if cached != "missing":
        return cached
    dims = None
    try:
        tc = engine.model_config.hf_text_config
        H_q = int(getattr(tc, "num_attention_heads"))
        H_kv = int(getattr(tc, "num_key_value_heads", H_q))
        hidden = int(getattr(tc, "hidden_size"))
        dims = (H_q, H_kv, hidden // H_q)
    except Exception:  # noqa: BLE001
        dims = None
    try:
        engine._mia_qk_dims = dims
    except Exception:  # noqa: BLE001
        pass
    return dims


def _hs_num_layers(engine) -> int | None:
    """Total decoder layers (for HS 'all layers' capture). Cached on the engine."""
    cached = getattr(engine, "_mia_hs_nlayers", "missing")
    if cached != "missing":
        return cached
    n = None
    try:
        n = int(getattr(engine.model_config.hf_text_config, "num_hidden_layers"))
    except Exception:  # noqa: BLE001
        n = None
    try:
        engine._mia_hs_nlayers = n
    except Exception:  # noqa: BLE001
        pass
    return n


def _emit_capture_evidence(engine, output, extra, wants_hs, wants_qk, gen_tokens) -> None:
    """Emit capture evidence (hook.fire.<w> + captured.bytes.<w>) in the driver process.

    Emitted here, not in the worker: under ``vllm serve`` the worker process's profiler dump is
    lost at teardown (SIGTERM kills the child before atexit runs), so evidence emitted in the
    worker never reaches the harvest even though the aperture captured and persisted the artifact.
    The driver's dump is collected, and the aperture path reaches this finalize per finished
    request, so evidence is reported from this request's actual output.

    Byte-exact for the all_tokens workload: captured bytes = n_layers x tokens x
    per-token-per-layer bytes, matching exactly what the aperture drain writes to NVMe (buffers hold
    all heads; head selection happens at analysis). ``hook.fire.<w>`` += n_layers so
    ``hook_fire_count / n_layers`` recovers the capturing-request count. Best-effort: a failure
    here must never perturb the finalize. (The worker-side emission stays for the offline path,
    whose dump is collected via ``dump_profiler``; the two paths never both land, so there is no
    double-count.)
    """
    try:
        n_prompt = len(getattr(output, "prompt_token_ids", None) or [])
        # gen_tokens is accumulated across the generate loop by the caller: the serve path streams
        # in delta mode, so the final output's outputs[].token_ids holds only the last delta --
        # summing per-yield deltas recovers the true generated length.
        n_gen = int(gen_tokens or 0)
        hooks_on = extra.get("hooks_on", "prefill")
        prefill_tok = n_prompt if hooks_on in ("prefill", "both") else 0
        decode_tok = n_gen if hooks_on in ("decode", "both") else 0
        tc = engine.model_config.hf_text_config
        elt = int(getattr(engine.model_config.dtype, "itemsize", 2) or 2)
        if wants_hs:
            oh = extra.get("output_hidden_states")
            n_layers = (len(oh) if isinstance(oh, (list, tuple)) and oh
                        else (_hs_num_layers(engine) or 0))
            mode = extra.get("hs_mode", "all_tokens")
            tokens = (prefill_tok + decode_tok) if mode == "all_tokens" \
                else ((1 if prefill_tok else 0) + decode_tok)
            bpt = int(getattr(tc, "hidden_size")) * elt
            if n_layers > 0:
                PROF.incr("hook.fire.hs", n_layers)
                if tokens > 0:
                    PROF.gauge("captured.bytes.hs", float(n_layers) * tokens * bpt)
        elif wants_qk:
            oq = extra.get("output_qk")
            n_layers = (len(oq) if isinstance(oq, (dict, list, tuple)) and oq
                        else (_hs_num_layers(engine) or 0))
            mode = extra.get("hookq_mode", "all_tokens")
            tokens = (prefill_tok + decode_tok) if mode == "all_tokens" \
                else ((1 if prefill_tok else 0) + decode_tok)
            num_h = int(getattr(tc, "num_attention_heads"))
            num_kv = int(getattr(tc, "num_key_value_heads", num_h) or num_h)
            head_dim = int(getattr(tc, "head_dim", 0) or (int(getattr(tc, "hidden_size")) // num_h))
            bpt = (num_h + num_kv) * head_dim * elt
            if n_layers > 0:
                PROF.incr("hook.fire.qk", n_layers)
                if tokens > 0:
                    PROF.gauge("captured.bytes.qk", float(n_layers) * tokens * bpt)
    except Exception:  # noqa: BLE001 — evidence is best-effort; never perturb the finalize
        pass


def _maybe_storage_route(engine, prompt, extra, max_tokens) -> bool | None:
    """Per-request storage router (MIA_STORAGE_ROUTER=1): predict this
    request's artifact size from its prompt length + captured layers/heads and
    return the min-tax storage choice (True=disk, False=RPC), or None to leave
    the caller's save_to_disk untouched (router off / can't decide / steer).

    Decided at request-START (before capture) because the worker routes each
    request's egress into the disk vs RPC bucket from save_to_disk at admit
    time -- a finish-time flip would arrive after the data already landed.
    """
    import os
    # DEFAULT ON: serve-only (this patch is the AsyncLLM path; offline LLM.generate never calls
    # it), and it reproduces the proven optimum HS-last->RPC / QK+HS-all->disk. `=0` to disable.
    # The default lives in optimizations.PUBLIC_LEVERS -- one source of truth, not a copy here.
    from mia.optimizations import env_is_on
    if not env_is_on("storage_router"):
        return None
    _dbg = os.environ.get("MIA_ROUTER_DEBUG") == "1"

    def _log(msg):
        if not _dbg:
            return
        n = getattr(_maybe_storage_route, "_dbg_n", 0)
        if n < 20:
            _maybe_storage_route._dbg_n = n + 1
            print(f"[hookplugin/router] {msg}", flush=True)

    wants_qk = extra.get("output_qk") is not None
    wants_hs = extra.get("output_hidden_states") is not None
    if not (wants_qk or wants_hs):
        return None  # steer / nothing to store
    P = _prompt_token_len(prompt)
    if not P:
        _log(f"P=None (prompt type={type(prompt).__name__}) -> no route")
        return None
    hooks_on = extra.get("hooks_on", "both")
    # An unpinned max_tokens is an UNKNOWN generation length, not a zero-token one:
    # estimate_gen_len keeps the prediction on the right side of the crossover.
    from mia.run_utils import estimate_gen_len
    gen_len = estimate_gen_len(max_tokens)
    try:
        from mia.run_utils import predict_artifact_kb, route_to_disk, predicted_rpc_ms
        if wants_qk:
            dims = _qk_model_dims(engine)
            if not dims:
                _log("qk dims=None -> no route")
                return None
            H_q, _H_kv, head_dim = dims
            oqk = extra.get("output_qk") or {}
            # total (layer,head) pairs captured -> fold into n_layers with H=1
            head_layers = sum(len(v) for v in oqk.values()) if isinstance(oqk, dict) else 0
            if head_layers <= 0:
                return None
            gran = extra.get("hookq_mode", "all_tokens")
            kb = predict_artifact_kb("qk", gran, P, head_layers, 1, head_dim,
                                     H_q * head_dim, 2, gen_len, hooks_on)
            d = route_to_disk("qk", kb)
            _log(f"qk P={P} hl={head_layers} gran={gran} kb={kb:.0f} "
                 f"rpc_ms={predicted_rpc_ms('qk', kb):.0f} -> {'DISK' if d else 'RPC'}")
            return d
        else:  # HS
            dims = _qk_model_dims(engine)
            hidden = dims[0] * dims[2] if dims else None
            if not hidden:
                return None
            layers = extra.get("output_hidden_states")
            n_layers = len(layers) if isinstance(layers, (list, tuple)) and layers \
                else _hs_num_layers(engine)
            if not n_layers:
                return None
            gran = extra.get("hs_mode", "last_token")
            kb = predict_artifact_kb("hs", gran, P, n_layers, 1, dims[2],
                                     hidden, 2, gen_len, hooks_on)
            d = route_to_disk("hs", kb)
            _log(f"hs P={P} L={n_layers} gran={gran} kb={kb:.0f} "
                 f"rpc_ms={predicted_rpc_ms('hs', kb):.0f} -> {'DISK' if d else 'RPC'}")
            return d
    except Exception as e:  # noqa: BLE001
        _log(f"exception {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-request delivery router -- the off-loop HS capture-aperture path.
#
# At request-start, predict the request's raw artifact bytes, read the chosen analyzer's
# reducibility, and pick transport (rpc/disk) + analyze_where (none/inflight/from_disk) via
# graph/delivery_router.decide_route. The disk transport is crossed to the worker drain now
# (before the first forward) via the route_aperture_to_disk RPC, so the drain stages that request
# to its own NVMe file; analyze_where stays driver-side and drives finalize.
#
# Additive + gated: every function below is inert unless MIA_APERTURE_PER_REQUEST is armed
# (graph mode), so the storage_router + all existing response paths stay byte-identical when
# it is off.
# ---------------------------------------------------------------------------

# RPC-vs-disk crossover thresholds, in bytes. DERIVED per worker kind, at every call, by solving
# run_utils' two on-loop cost models against each other (`rpc_disk_crossover_kb`) -- so a retuned
# coefficient moves the threshold with it and HS and QK get their own. Until 2026-09-20 both were
# one hardcoded 512 KiB, which fitted HS's crossover and was five times too high for QK.


def _profile_mode() -> bool:
    """Component-1-only profiling (MIA_PROFILE_MODE=1): only the capture pipeline (GPU
    forward + in-graph scatter + off-loop drain -> server NVMe) is measured; delivery (analyzer
    + offload/RPC response) is disabled -- the finalize stamps request_done at the data-prepared
    boundary and ships nothing. Default off = full serving runs both stages."""
    import os
    return os.environ.get("MIA_PROFILE_MODE") == "1"


def _aperture_route_thresholds(worker_kind: str = "hs") -> "tuple[int, int]":
    """(T_rpc, T_analyze) in bytes for decide_route, for ONE worker kind ("hs" / "qk").

    The default is DERIVED, not written down: `run_utils.rpc_disk_crossover_kb(kind)` solves that
    kind's RPC model against its disk model, so both thresholds carry the measured basis of those
    coefficients (and the one un-measured handoff constant, flagged there). At the shipped values:
    HS 539.6 KB, QK 100.5 KB.

    MIA_ROUTER_T_RPC / MIA_ROUTER_T_ANALYZE override outright, both kinds at once (they are the
    operator's "I want the crossover HERE"); read every call so either can be retuned without a
    restart.

    T_ANALYZE takes the same number for the same reason it always did -- a reducible analyzer
    holds the raw artifact in host RAM while it reduces, so it should stop doing that where
    shipping it stops being worthwhile. Its true basis is host-RAM residency, which nothing here
    measures; see docs/configs.md."""
    import os
    from mia.run_utils import rpc_disk_crossover_kb
    derived = int(rpc_disk_crossover_kb(worker_kind) * 1024)
    t_rpc = int(os.environ.get("MIA_ROUTER_T_RPC", derived))
    t_analyze = int(os.environ.get("MIA_ROUTER_T_ANALYZE", derived))
    return t_rpc, t_analyze


def _analyzer_reducible(analyzer_name, analyzer_spec) -> bool:
    """Is the request's chosen analyzer reducible server-side (ships a small result), or does
    it need the raw tensors delivered whole to the client?

      * ACCEPTS == "qk"    (core_reranker, two-pass)  -> needs raw          -> False
      * ACCEPTS == "score" (attn_tracker)             -> reduces to a score -> True
      * hidden_states (no ACCEPTS): reducible ONLY when a reduce is requested
        (analyzer_spec["reduce"] in {mean, norm}); reduce=none / absent      -> False

    A missing/unknown analyzer name falls back to the reduce-driven rule (a bare hidden_states-style
    spec), and an unresolvable one defaults to NOT reducible = RAW delivery — today's behavior, which
    never loses data. Pure logic over the registry capability + the spec; no engine/GPU."""
    reduce = (analyzer_spec or {}).get("reduce", "none") if isinstance(analyzer_spec, dict) else "none"
    reducible_reduce = reduce in ("mean", "norm")
    if not analyzer_name:
        return reducible_reduce
    entry = None
    try:
        from mia.registry import PluginRegistry
        entry = PluginRegistry.get_analyzer(analyzer_name)
        if entry is None:
            from mia import register_plugins
            register_plugins()
            entry = PluginRegistry.get_analyzer(analyzer_name)
    except Exception:  # noqa: BLE001
        entry = None
    accepts = getattr(entry.analyzer, "ACCEPTS", None) if entry is not None else None
    if accepts == "qk":
        return False
    if accepts == "score":
        return True
    return reducible_reduce


def _explicit_disk_route(extra: dict):
    """A ``RouteDecision`` when the sink is ALREADY settled as disk, else None.

    This is what each router returns INSTEAD OF ``None`` when it cannot price a request -- the
    bail-outs for a missing prompt length, missing model dims, or an unreadable capture spec.
    Those bail-outs used to return ``None``, which means the RPC default, so a request whose size
    could not be predicted silently overrode an explicit ``save_to_disk: true``. Measured on an
    H100 (W13 QK leg, 2026-09-21): the forced-DISK QK cell delivered 580 artifacts over RPC and
    zero to disk, and PASSED, because delivery happened -- just not by the route the caller
    required. An explicit ask is a requirement, not a hint (CAPTURE_IO.md 8.1), and a requirement
    must not be contingent on a cost model being able to price the request.

    ``analyze_where`` is ``'none'`` here and ONLY here, which is why this is a fallback rather
    than a short-circuit: reducibility is a size question, so a request this function answers is
    one whose size is unknown. When the size IS predictable the router runs its model and forces
    only the TRANSPORT, keeping the ``analyze_where`` the model chose -- see the call sites. (An
    earlier revision of this fix consulted the helper first, which was simpler and quietly threw
    away ``analyze_from_disk`` for every reducible disk-sink capture. Inert today, since
    ServerAnalyzeProcess is deferred, but wrong.)
    """
    if _resolve_sink(extra) != "disk":
        return None
    from mia.graph.delivery_router import RouteDecision
    return RouteDecision("disk", "none")


def _decide_aperture_route(engine, prompt, extra, max_tokens):
    """RouteDecision for the off-loop HS capture-aperture per-request path, decided at REQUEST-START, or
    None when it does not apply (not HS-only / cannot predict) so the caller falls back to the
    default host-buffer path.

    HS-ONLY: the capture-aperture per-request path is HS-only (get_aperture_per_request is HS-only), so a
    request that also wants QK is left to the general path. Mirrors _maybe_storage_route's HS
    prediction (predict_artifact_kb) so both routers agree on size."""
    wants_qk = extra.get("output_qk") is not None
    wants_hs = extra.get("output_hidden_states") is not None
    if not (wants_hs and not wants_qk):
        return None
    P = _prompt_token_len(prompt)
    if not P:
        return _explicit_disk_route(extra)
    dims = _qk_model_dims(engine)
    hidden = dims[0] * dims[2] if dims else None
    if not hidden:
        return _explicit_disk_route(extra)
    layers = extra.get("output_hidden_states")
    n_layers = len(layers) if isinstance(layers, (list, tuple)) and layers \
        else _hs_num_layers(engine)
    if not n_layers:
        return _explicit_disk_route(extra)
    gran = extra.get("hs_mode", "last_token")
    hooks_on = extra.get("hooks_on", "both")
    # An unpinned max_tokens is an UNKNOWN generation length, not a zero-token one:
    # estimate_gen_len keeps the prediction on the right side of the crossover.
    from mia.run_utils import estimate_gen_len
    gen_len = estimate_gen_len(max_tokens)
    try:
        from mia.run_utils import predict_artifact_kb
        from mia.graph.delivery_router import decide_route
        kb = predict_artifact_kb("hs", gran, P, n_layers, 1, dims[2], hidden, 2, gen_len, hooks_on)
        predicted_bytes = int(kb * 1024)
        reducible = _analyzer_reducible(extra.get("analyzer"), extra.get("analyzer_spec"))
        t_rpc, t_analyze = _aperture_route_thresholds("hs")
        decision = decide_route(predicted_bytes, reducible, t_rpc, t_analyze)
        # AN EXPLICIT ASK IS A REQUIREMENT, NOT A HINT (CAPTURE_IO.md 8.1). When the sink is
        # already settled as `disk` -- by the caller's own save_to_disk, or by MIA's storage
        # router, which writes that key when it picks disk -- the size model must not send the
        # artifact over RPC instead: the caller asked for a FILE and would get none. Only the
        # TRANSPORT is forced; analyze_where is left as decided, since a reducible capture still
        # reduces.
        if _resolve_sink(extra) == "disk" and decision.transport != "disk":
            from mia.graph.delivery_router import RouteDecision
            return RouteDecision("disk", decision.analyze_where)
        return decision
    except Exception:  # noqa: BLE001
        return _explicit_disk_route(extra)


def _decide_aperture_route_qk(engine, prompt, extra, max_tokens):
    """RouteDecision for the off-loop QK capture-aperture per-request path, decided at REQUEST-START, or
    None when it does not apply (not QK-only / cannot predict) so the caller falls back to the default
    host-buffer path. QK-ONLY (the QK aperture path is separate from the HS one). Mirrors
    ``_maybe_storage_route``'s QK prediction so both routers agree on size. A capture spec it cannot
    price -- a non-dict ``output_qk`` (whole-model capture), no prompt length, no model dims --
    yields ``_explicit_disk_route``: the host-buffer RPC default UNLESS the caller pinned a disk
    sink, which is a requirement rather than a preference (CAPTURE_IO.md 8.1)."""
    wants_qk = extra.get("output_qk") is not None
    wants_hs = extra.get("output_hidden_states") is not None
    if not (wants_qk and not wants_hs):
        return None
    P = _prompt_token_len(prompt)
    if not P:
        return _explicit_disk_route(extra)
    dims = _qk_model_dims(engine)
    if not dims:
        return _explicit_disk_route(extra)
    H_q, _H_kv, head_dim = dims
    oqk = extra.get("output_qk") or {}
    head_layers = sum(len(v) for v in oqk.values()) if isinstance(oqk, dict) else 0
    if head_layers <= 0:
        return _explicit_disk_route(extra)
    gran = extra.get("hookq_mode", "all_tokens")
    hooks_on = extra.get("hooks_on", "both")
    # An unpinned max_tokens is an UNKNOWN generation length, not a zero-token one:
    # estimate_gen_len keeps the prediction on the right side of the crossover.
    from mia.run_utils import estimate_gen_len
    gen_len = estimate_gen_len(max_tokens)
    try:
        from mia.run_utils import predict_artifact_kb
        from mia.graph.delivery_router import decide_route
        kb = predict_artifact_kb("qk", gran, P, head_layers, 1, head_dim,
                                 H_q * head_dim, 2, gen_len, hooks_on)
        predicted_bytes = int(kb * 1024)
        reducible = _analyzer_reducible(extra.get("analyzer"), extra.get("analyzer_spec"))
        t_rpc, t_analyze = _aperture_route_thresholds("qk")
        decision = decide_route(predicted_bytes, reducible, t_rpc, t_analyze)
        # Same rule as the HS sibling: an explicit ask is a requirement, not a hint
        # (CAPTURE_IO.md 8.1). QK crosses its own 100.5 KB threshold at almost any real size, so
        # this bites mainly on a tiny capture -- which is exactly the case where the size model
        # would say RPC and a caller who asked for a FILE would get none.
        if _resolve_sink(extra) == "disk" and decision.transport != "disk":
            from mia.graph.delivery_router import RouteDecision
            return RouteDecision("disk", decision.analyze_where)
        return decision
    except Exception:  # noqa: BLE001
        return _explicit_disk_route(extra)


def _aperture_finalize_action(route, profile_mode: bool) -> str:
    """Finalize action for a aperture-per-request HS request. Pure decision (no engine/GPU) so the
    no-GPU test drives it directly:

      * profile_mode           -> 'profile_stamp'      (Component-1-only: stamp request_done, run
                                    NO Component 2 — no analyzer, delivery, offload, or RPC response)
      * route is None          -> 'rpc_raw'            (default host-buffer path — back-compat)
      * analyze_where inflight  -> 'analyze_inflight'   (ServerAnalyzeProcess — see the caller)
      * analyze_where from_disk -> 'analyze_from_disk'  (ServerAnalyzeProcess — see the caller)
      * transport disk, none    -> 'disk_raw'           (delivered via the offloaded per-request file)
      * transport rpc,  none    -> 'rpc_raw'            (host-buffer RPC — get_aperture_per_request)
    """
    if profile_mode:
        return "profile_stamp"
    if route is None:
        return "rpc_raw"
    if route.analyze_where == "inflight":
        return "analyze_inflight"
    if route.analyze_where == "from_disk":
        return "analyze_from_disk"
    return "disk_raw" if route.transport == "disk" else "rpc_raw"


def _log_analyze_deferred(action: str) -> None:
    """Log once that server-side CPU analyze (ServerAnalyzeProcess) is deferred, so the
    fall-back to raw delivery is never silent."""
    n = getattr(_log_analyze_deferred, "_n", 0)
    if n < 4:
        _log_analyze_deferred._n = n + 1
        print(f"[hookplugin/aperture-router] analyze_where={action!r}: server-side CPU analyze is "
              f"deferred to Task 12; delivering RAW (client analyzes client-side)", flush=True)


def _engine_tp_size(engine) -> int:
    """``tensor_parallel_size`` of an ``AsyncLLM`` / ``LLM`` (1 when unreadable). Cached."""
    cached = getattr(engine, "_mia_tp_size", None)
    if cached is not None:
        return int(cached)
    tp = 1
    for path in (("vllm_config",), ("llm_engine", "vllm_config"), ("engine", "vllm_config")):
        obj = engine
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        pc = getattr(obj, "parallel_config", None) if obj is not None else None
        if pc is not None:
            try:
                tp = int(getattr(pc, "tensor_parallel_size", 1) or 1)
                break
            except (TypeError, ValueError):
                continue
    try:
        engine._mia_tp_size = tp
    except Exception:  # noqa: BLE001
        pass
    return tp


def _refuse_unsupported_tp_request(engine, extra) -> None:
    """Refuse, before submission, a request whose capture cannot be correct at TP > 1.

    Attention-SCORE capture (``qk_capture="score"``) computes a per-head softmax over the heads
    ONE rank holds, labelled with GLOBAL head indices, and the per-rank scores cannot be merged.
    Refused at the driver so the caller gets an error for THIS request instead of a worker that
    skips it (the worker-side check is only a backstop)."""
    if not isinstance(extra, dict) or extra.get("qk_capture") != "score":
        return
    tp = _engine_tp_size(engine)
    if tp > 1:
        raise MiaConfigurationError(
            f"qk_capture='score' is not supported at tensor_parallel_size={tp}: every TP rank "
            f"holds only its own attention heads, so a per-head score is computed over one "
            f"rank's slice. Capture raw Q/K (qk_capture='qk', merged across ranks by MIA) or run "
            f"at tensor_parallel_size=1.")


async def _patched_generate(
    self,
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    """Wrap AsyncLLM.generate to install hooks and attach probes on finish."""
    # In vLLM v1, the chat endpoint clones SamplingParams into EngineCoreRequest
    # before calling generate(). We must read/modify the clone so our changes take effect.
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:
            effective_params = prompt.sampling_params
    except ImportError:
        pass

    extra = dict(effective_params.extra_args or {})
    # vllm_xargs only allows scalar values, so MiaClient JSON-encodes nested
    # structures. Decode them back here before the worker reads extra_args.
    import json as _json
    for _k in ("output_qk", "output_hidden_states", "steer"):
        if isinstance(extra.get(_k), str):
            try:
                _decoded = _json.loads(extra[_k])
            except (ValueError, TypeError):
                # NOT valid JSON. Before giving up, try a PYTHON LITERAL -- `str(dict)` produces
                # single quotes, which json.loads rejects, and a producer that builds the value
                # with repr() rather than json.dumps() would otherwise slip through as a string.
                #
                # WHY THIS IS NOT MERELY TIDY (measured, W13 QK leg 2026-09-21). The capture
                # WORKER parses these forms itself, so a string here captures correctly and
                # nothing looks wrong -- but the ROUTERS type-check (`isinstance(oqk, dict)`) and
                # bail, so the request silently takes the RPC default whatever its size. A QK
                # campaign sending `{'0': [0, 8]}` recorded ZERO router decisions and shipped a
                # ~78 MB artifact over RPC. HS escaped only by luck: `[1, 2, 3]` happens to be
                # valid JSON, so its list decoded and its routing worked.
                try:
                    import ast as _ast
                    _decoded = _ast.literal_eval(extra[_k])
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    # Plain non-literal strings (e.g. legacy boolean-like) pass through.
                    continue
                if not isinstance(_decoded, (dict, list, tuple, bool, int, float)):
                    continue        # a bare word like "all" is not a structure -- leave it alone
            # output_qk comes back as {str_key: list} — restore int keys
            if _k == "output_qk" and isinstance(_decoded, dict):
                _decoded = {int(k): v for k, v in _decoded.items()}
            extra[_k] = _decoded
    effective_params.extra_args = extra

    wants_hs = extra.get("output_hidden_states") is not None
    wants_qk = extra.get("output_qk") is not None
    wants_steer = isinstance(extra.get("steer"), dict)
    needs_hooks = wants_hs or wants_qk or wants_steer
    save_to_disk = bool(extra.get("save_to_disk"))
    # Before ANY per-request state is created on a worker (routes, stashes): a request that
    # cannot be captured correctly at this TP size fails here, alone.
    _refuse_unsupported_tp_request(self, extra)

    # Per-request storage router: pick the min-tax path (disk vs RPC) from this request's
    # predicted artifact size. Decided here at request-start so the worker routes egress into
    # the right bucket.
    #
    # ONLY when the caller expressed NO preference. save_to_disk is not just a perf knob -- it
    # is how you ask for a durable artifact FILE. The router prices HS last_token at ~192 KB and
    # picks RPC, so overriding an explicit save_to_disk=True would silently write nothing for a
    # caller who needs the file on disk, with no way to express the requirement. An explicit
    # value is a requirement; an absent one is "you choose".
    if "save_to_disk" not in extra:
        _routed = _maybe_storage_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _routed is not None and _routed != save_to_disk:
            save_to_disk = _routed
            extra["save_to_disk"] = _routed
            effective_params.extra_args = extra

    # Per-request delivery router: for the off-loop HS capture-aperture per-request path, decide
    # this request's transport (rpc/disk) + analyze_where from its predicted artifact size and
    # the chosen analyzer's reducibility, at request-start. Cross the disk transport to the
    # worker drain now -- before the request's first forward -- via route_aperture_to_disk, so the
    # drain stages this request to its own NVMe file; analyze_where stays here and drives
    # finalize below. Additive + gated: no-op unless aperture per-request mode is armed, so the
    # storage router + every response path stay byte-identical when it is off. Not run in
    # profile mode (delivery is disabled there -- no per-request delivery to route).
    _warn_profile_aperture_conflict()  # guard (b): PROFILE_MODE + APERTURE_PER_REQUEST is a misconfig
    _aperture_route = None
    # Guard (a): only arm the aperture route when the request will actually take the aperture
    # finalize branch. That condition is now ONE predicate shared with finalize
    # (_takes_aperture_per_request) -- when the two sites spelled it out separately they drifted
    # apart on `disk` and silently dropped every disk-routed capture; the predicate's docstring
    # has the measurement.
    if _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "hs":
        _aperture_route = _decide_aperture_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _aperture_route is not None and _aperture_route.transport == "disk":
            import os as _os_route
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            dest = _os_route.path.join(hook_dir, str(run_id))
            with PROF.timed("rpc.route_aperture_to_disk"):
                await self.collective_rpc("route_aperture_to_disk", args=(request_id, dest))

    # QK sibling of the HS aperture route above: the off-loop QK capture-aperture per-request path.
    # Decide this request's transport (rpc/disk) at request-start and cross the disk route to
    # the QK drain now (before the first forward). QK-only (gated on `not wants_hs`), same
    # guards as the HS block. route_aperture_to_disk resolves to the QK worker's method.
    _aperture_route_qk = None
    # Symmetric with the HS guard above, and on the SAME predicate. It was asymmetric for one
    # round -- QK kept excluding `disk` while HS stopped -- on the reasoning that finalize had no
    # QK aperture branch to take precedence over the `sink == "disk"` flush. That reasoning was
    # wrong in the same way the original bug was: the QK drain also builds no shared sink under
    # per_request, so excluding `disk` did not preserve a working path, it preserved the DROP.
    if _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "qk":
        _aperture_route_qk = _decide_aperture_route_qk(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _aperture_route_qk is not None and _aperture_route_qk.transport == "disk":
            import os as _os_route
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            dest = _os_route.path.join(hook_dir, str(run_id))
            with PROF.timed("rpc.route_aperture_to_disk"):
                await self.collective_rpc("route_aperture_to_disk", args=(request_id, dest))

    # Serve-path QK auto-select. vllm serve goes through this patch, not MiaLLM.generate, so
    # the offline admission cannot run here. When opted in (MIA_QK_AUTO_SELECT=1 -- stands
    # in for the offline analyzer-accepts gate, since the analyzer runs client-side in serve) and
    # the client did not pin qk_capture, pick the smaller representation from the prompt length +
    # this request's output_qk head set + the model dims. Decided once, immutable.
    if wants_qk and "qk_capture" not in extra and _engine_tp_size(self) <= 1:
        # (No auto-select at TP > 1: score capture is TP=1-only -- see
        # _refuse_unsupported_tp_request.)
        import os as _os
        if _os.environ.get("MIA_QK_AUTO_SELECT") == "1":
            try:
                _dims = _qk_model_dims(self)
                _plen = _prompt_token_len(prompt)
                _oqk = extra.get("output_qk")
                if _dims and _plen and isinstance(_oqk, dict):
                    from mia.run_utils import qk_score_size_select
                    _pick = qk_score_size_select(
                        _plen, extra.get("hookq_mode", "all_tokens"), _oqk, *_dims)
                    extra["qk_capture"] = _pick
                    if _pick == "score":
                        extra.setdefault("score_head", 0)
                    effective_params.extra_args = extra
                    _n = getattr(_patched_generate, "_d2_log_n", 0)
                    if _n < 8:
                        _patched_generate._d2_log_n = _n + 1
                        print(f"[hookplugin/D2] serve auto-select qk_capture={_pick} "
                              f"(S={_plen} mode={extra.get('hookq_mode','all_tokens')})",
                              flush=True)
            except Exception:  # noqa: BLE001
                pass

    # In graph mode the QK capture path is already installed in the worker at
    # load_model (graph/install.py), so the lazy forward-hook install would only
    # double-capture — skip it. The legacy eager path still installs lazily.
    if (
        needs_hooks
        and not getattr(self, "_mia_installed", False)
        and not _graph_mode()
    ):
        PROF.incr("rpc.install_hooks")
        with PROF.timed("rpc.install_hooks"):
            await self.collective_rpc("install_hooks")
        setattr(self, "_mia_installed", True)

    assert _original_generate is not None
    _hook_gen_toks = 0
    _prof_capture = needs_hooks and not wants_steer and _profile_mode()
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if _prof_capture:
                # Accumulate generated tokens across yields: serve streams in DELTA mode, so summing
                # per-yield deltas recovers the full generated length for the evidence byte count.
                for _o in (getattr(output, "outputs", None) or []):
                    _hook_gen_toks += len(getattr(_o, "token_ids", None) or [])
            if output.finished and needs_hooks and not wants_steer and _profile_mode():
                # Profile mode (Component-1-only): the capture pipeline already persisted this
                # request's data to server NVMe (the off-loop drain). Run no delivery -- no
                # analyzer, offload, or RPC response -- and stamp request_done at the
                # data-prepared boundary. output.probes stays unset; the serving product (profile
                # mode off) runs both stages.
                PROF.incr("request_done")
                PROF.event("request_done",
                           {"req_id": str(request_id), "boundary": "data_prepared"})
                # Capture evidence in the driver dump (the worker's is lost at serve teardown)
                # so the harvest sees artifact_kb / hook_fire for this profile.
                _emit_capture_evidence(self, output, extra, wants_hs, wants_qk, _hook_gen_toks)
            elif output.finished and wants_steer:
                # Steer evidence in the driver dump (the worker's steer.fire counter is lost at
                # serve teardown): confirm steering fired. Under FULL cudagraph the baked steer op
                # runs on every forward of a steered request, so a finished steer request was
                # steered. steer.fire feeds hook_fire_count; artifact_kb stays 0 (steer has no
                # artifact).
                PROF.incr("steer.fire")
            elif output.finished and needs_hooks and not wants_steer:
                sink = _resolve_sink(extra)
                if sink == "drop":
                    pass  # drop sink: nothing on the finish path; finally clears the bucket
                elif (sink == "disk"
                      and not _takes_aperture_per_request(extra, wants_hs, wants_qk, wants_steer)):
                    # The disk branch DECLINES when the aperture per-request path owns this
                    # request. Under per-request delivery a disk-routed request is staged by the
                    # per-request writer (CAPTURE_IO.md 8.3), NOT by flush_disk -- which flushes
                    # the eager worker's `_disk_states`, empty on this path because the capture
                    # went to the aperture. This branch winning is what dropped 47 GB in W13
                    # without raising anything; see _takes_aperture_per_request.
                    run_id = extra.get("run_id") or request_id
                    hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                    with PROF.timed("rpc.flush_disk"):
                        flushed = await self.collective_rpc(
                            "flush_disk", args=([request_id], run_id, hook_dir))
                    # Durability is NOT waited for here (fire-and-forget) — the read side
                    # (analyze) and teardown drain guarantee it. Opt-in per-request durable_wait
                    # preserves the read-immediately contract for callers that need it: it waits
                    # for EVERY rank that wrote (at TP > 1 each QK rank lands its own shard).
                    if extra.get("durable_wait"):
                        with PROF.timed("disk.await_artifact"):
                            await _await_disk_artifact(
                                run_id, hook_dir, _flushed_rank_dirs(flushed, run_id, hook_dir))
                    # Leave output.probes unset — caller reads artifacts from disk.
                elif _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "hs":
                    # Aperture per-request delivery is HS-only. Gate on `not wants_qk` so a request
                    # that also wants QK is not swallowed by this branch (which returns hs_cache
                    # only, silently dropping qk_cache); it falls through to get_captured_states
                    # instead.
                    #
                    # Dispatch on the request-start RouteDecision. _aperture_route is None (no route
                    # decided) => 'rpc_raw', identical to the original default path.
                    action = _aperture_finalize_action(_aperture_route, False)
                    if action in ("analyze_inflight", "analyze_from_disk"):
                        # Server-side CPU analyze (ServerAnalyzeProcess) is deferred -- it needs
                        # the analyzer name/spec on the server plus a blocking analyze RPC. Falls
                        # back to raw delivery of the same transport, byte-safe (client reduces
                        # client-side); just missing the server-side reduce. Logged once.
                        _log_analyze_deferred(action)
                        action = ("disk_raw" if (_aperture_route is not None
                                                 and _aperture_route.transport == "disk")
                                  else "rpc_raw")
                    if action == "disk_raw":
                        # Disk transport: the per-request file streamed to NVMe and, on the
                        # worker drain's finish, was handed to the OffloadProcess -> delivered to
                        # the client dest. Block until the offload confirms the file landed, then
                        # leave output.probes unset -- the client reads the delivered file, as
                        # with save_to_disk. On confirm the worker also unlinks the server-side
                        # staging source. Bounded + loud on timeout.
                        with PROF.timed("aperture.await_disk_confirm"):
                            await _await_aperture_disk_confirm(self, request_id)
                    else:  # rpc_raw — host-buffer path
                        # Off-loop HS capture-aperture per-request delivery: the worker demuxes
                        # drained rows by req_id into a PerRequestIndex off-loop and assembles
                        # this request on finish. Block-until-held: poll get_aperture_per_request
                        # until it returns this request's marshaled probes or the deliver timeout
                        # elapses (loud; probes left unset, never an unbounded hang). Under the TP
                        # layer shard it waits for (and unions) every rank owning a requested layer.
                        probes = await _await_aperture_per_request(
                            self, request_id, extra.get("output_hidden_states"))
                        if probes is not None:
                            output.probes = probes
                elif _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "qk":
                    # Off-loop QK capture-aperture per-request delivery. QK-only: gate on `not
                    # wants_hs` so a request that also wants HS is not swallowed here (it falls
                    # through to get_captured_states). Dispatch on the request-start
                    # RouteDecision (_aperture_route_qk); None => 'rpc_raw' (the default path).
                    action = _aperture_finalize_action(_aperture_route_qk, False)
                    if action in ("analyze_inflight", "analyze_from_disk"):
                        # Server-side CPU analyze is deferred (as with HS) -> RAW delivery of the SAME
                        # transport, byte-safe (the client reduces client-side). Logged once.
                        _log_analyze_deferred(action)
                        action = ("disk_raw" if (_aperture_route_qk is not None
                                                 and _aperture_route_qk.transport == "disk")
                                  else "rpc_raw")
                    if action == "disk_raw":
                        # Disk transport: the per-request q/k files streamed to NVMe and, on
                        # finish, were handed to the OffloadProcess -> delivered to the client
                        # dest. Block until the offload confirms, then leave output.probes unset
                        # (the client reads the delivered file, as with save_to_disk).
                        with PROF.timed("aperture.await_disk_confirm"):
                            await _await_aperture_disk_confirm(self, request_id)
                    else:  # rpc_raw — host-buffer path
                        # BLOCK-UNTIL-HELD: poll get_aperture_per_request (QK worker) until it returns this
                        # request's marshaled qk_cache bytes or the deliver timeout elapses (LOUD).
                        probes = await _await_aperture_per_request(self, request_id)
                        if probes is not None:
                            output.probes = probes
                else:  # rpc
                    with PROF.timed("rpc.get_states"):
                        states = await self.collective_rpc(
                            "get_captured_states", args=(request_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    if parts:
                        for _p in parts:
                            _reconstruct_compact_qk(_p)   # per rank, before the head merge
                        probes = merge_probe_parts(parts)
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes
            yield output
    finally:
        # Cleanup on abort/disconnect. Runs for both paths: on normal completion the bucket was
        # already popped (get_captured_states / flush_disk), so this clear is a no-op; on an
        # abort before output.finished it releases the orphan bucket -- which for save_to_disk
        # also releases the resident-byte counter (else an aborted disk-mode request leaks it
        # and can wedge the admission ceiling).
        if needs_hooks and not wants_steer:
            await self.collective_rpc("clear_captured_states", args=(request_id,))
            # clear_captured_states clears only the bank/eager buckets, which the aperture path
            # never uses. On an abort before finish, free the request's aperture state too -- the
            # host-buffer PerRequestIndex entry/stash and the disk staging -- so
            # disk_residency() and the host index return to 0 (no leak). No-op when the request
            # has no aperture state. The QK-only branch resolves clear_aperture_request to the QK
            # worker's method (one worker per process).
            if _aperture_per_request_mode() and ((wants_hs and not wants_qk)
                                             or (wants_qk and not wants_hs)):
                await self.collective_rpc("clear_aperture_request", args=(request_id,))


# ---------------------------------------------------------------------------
# Offline (sync) LLM.generate patch
# ---------------------------------------------------------------------------


def _patched_llm_generate(self, prompts: Any, sampling_params: Any = None, **kwargs) -> list:
    """Wrap LLM.generate to install hooks and dispatch post-generation.

    Each request is dispatched based on its own extra_args:
    - save_to_disk=True -> collective_rpc("flush_disk"); output.probes unset.
    - otherwise        -> collective_rpc("get_captured_states"); attach to output.probes.
    """
    if isinstance(sampling_params, (list, tuple)):
        params_list = list(sampling_params)
    elif sampling_params is not None:
        params_list = [sampling_params]
    else:
        params_list = []

    needs_hooks = any(
        (sp.extra_args or {}).get("output_hidden_states") is not None
        or (sp.extra_args or {}).get("output_qk") is not None
        or bool((sp.extra_args or {}).get("steer"))
        for sp in params_list
    )
    for _sp in params_list:
        _refuse_unsupported_tp_request(self, _sp.extra_args or {})

    # The storage router is serve-only (async engine): its decision must precede
    # capture, and the offline path's per-request prompt<->params alignment across
    # every input shape is a separate ergonomics problem. Warn once rather than
    # silently no-op (the flip below would land AFTER the bucket is chosen).
    import os as _os
    if (needs_hooks and _os.environ.get("MIA_STORAGE_ROUTER") == "1"
            and not getattr(_patched_llm_generate, "_router_warned", False)):
        _patched_llm_generate._router_warned = True
        print("[hookplugin] MIA_STORAGE_ROUTER is serve-only; the offline "
              "LLM.generate path honors each request's explicit save_to_disk.",
              flush=True)

    # Graph mode installs the QK path in the worker at load_model; skip the lazy
    # forward-hook install (it would double-capture). Eager path unchanged.
    if (
        needs_hooks
        and not getattr(self, "_mia_installed", False)
        and not _graph_mode()
    ):
        PROF.incr("rpc.install_hooks")
        with PROF.timed("rpc.install_hooks"):
            self.collective_rpc("install_hooks")
        self._mia_installed = True

    assert _original_llm_generate is not None
    outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)

    if needs_hooks:
        import os

        # First pass: handle RPC (in-memory) requests immediately, and collect
        # disk-save requests grouped by run_id so all requests sharing the same
        # run_id are flushed together — preventing the second flush from
        # overwriting the first when a batch shares one run_id.
        disk_by_run: dict = {}  # run_id -> [(req_id, hook_dir)]

        for idx, output in enumerate(outputs):
            req_id = output.request_id
            sp = params_list[idx] if idx < len(params_list) else params_list[0] if params_list else None
            extra = (sp.extra_args if sp is not None else None) or {}

            wants_artifacts = extra.get("output_hidden_states") is not None or extra.get("output_qk") is not None
            if extra.get("save_to_disk"):
                run_id = extra.get("run_id") or req_id
                hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                disk_by_run.setdefault(run_id, []).append((req_id, hook_dir))
            elif wants_artifacts:
                if (_aperture_per_request_mode()
                        and extra.get("output_hidden_states") is not None
                        and extra.get("output_qk") is None):
                    # HS-only gate (see the serve path): a combined HS+QK request falls through
                    # to get_captured_states rather than being swallowed by this HS-only aperture
                    # branch. Off-loop HS capture-aperture per-request delivery: retrieve this
                    # request's marshaled probes. None until its off-loop finish has been
                    # processed (this path does not block-until-held; that's serve-only) -- and
                    # under the HS TP layer shard a COMPLETE rank set is what gets merged, so a
                    # round that returned only some of the owning ranks' parts waits for the rest
                    # instead of raising TPShardError out of generate().
                    probes = _collect_aperture_per_request_sync(
                        self.collective_rpc, req_id, extra.get("output_hidden_states"))
                    if probes is not None:
                        output.probes = probes
                else:
                    with PROF.timed("rpc.get_states"):
                        states = self.collective_rpc("get_captured_states", args=(req_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    if parts:
                        for _p in parts:
                            _reconstruct_compact_qk(_p)  # rebuild deferred k_all before trim/merge
                        probes = merge_probe_parts(parts)
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes

        # Second pass: finalize disk-save requests.
        #
        # graph+buffer (capture_aperture) mode: no per-generate finalize here. Durable capture
        # streams GPU-aperture -> off-loop drain -> APERTURE_DIR continuously during generate, so the
        # persist cost is already in gen_lat. Two reasons not to flush here: (1) the legacy
        # flush_disk host buckets (`_disk_states`) are never filled in graph+buffer mode, so
        # flush_disk would write nothing and the durability barrier would stall on a phantom
        # artifact; (2) flush_aperture would join+close the consumer thread, breaking the next rep's
        # enqueue in a multi-rep offline run. The aperture's atexit backstop (registered at install)
        # writes the reader sidecar at engine teardown; artifact_kb comes from the PROF
        # captured.bytes gauge. Mirrors the server PROFILE_MODE lifecycle (ship nothing at
        # finish; the aperture already persisted).
        if disk_by_run and not _graph_mode():
            # Eager path unchanged: register_forward_hook fills _disk_states, so flush_disk writes
            # the artifact and the barrier waits for the writer child to land it. All req_ids sharing
            # a run_id flush together so the second flush doesn't overwrite the first.
            # The barrier waits for EVERY rank dir each flush named: at TP > 1 each QK rank's
            # writer lands its own shard on its own schedule (_flushed_rank_dirs).
            flushed_by_run: dict = {}
            for run_id, req_list in disk_by_run.items():
                req_ids = [r for r, _ in req_list]
                _, hook_dir = req_list[0]
                with PROF.timed("rpc.flush_disk"):
                    flushed_by_run[run_id] = self.collective_rpc(
                        "flush_disk", args=(req_ids, run_id, hook_dir))
            for run_id, req_list in disk_by_run.items():
                _, hook_dir = req_list[0]
                with PROF.timed("disk.await_artifact"):
                    _wait_disk_artifact(run_id, hook_dir, _flushed_rank_dirs(
                        flushed_by_run.get(run_id), run_id, hook_dir))

    return outputs


# ---------------------------------------------------------------------------
# Response builder patches for vllm serve (OpenAI-compatible API)
# ---------------------------------------------------------------------------


def _serialize_probes(probes: dict) -> dict:
    """Serialize probe tensors to lists for JSON transport."""
    import torch
    PROF.incr("serve.serialize_probes.calls")
    with PROF.timed("serve.serialize_probes"):
        result = {}
        n_tensors = 0
        n_elems = 0
        for key, cache in probes.items():
            # config is a flat dict of scalars — pass through as-is.
            if key == "config" and isinstance(cache, dict):
                result[key] = cache
                continue
            if not isinstance(cache, dict):
                continue
            result[key] = {}
            for mod_name, entry in cache.items():
                new_entry = {}
                for k, v in entry.items():
                    if isinstance(v, torch.Tensor):
                        n_tensors += 1
                        n_elems += v.numel()
                        new_entry[k] = v.tolist()
                    elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                        # "scores": a per-pass list of [S_q,S_k] tensors.
                        n_tensors += len(v)
                        n_elems += sum(t.numel() for t in v)
                        new_entry[k] = [t.tolist() for t in v]
                    else:
                        new_entry[k] = v
                result[key][mod_name] = new_entry
        PROF.gauge("serve.serialize_probes.tensors", n_tensors)
        PROF.gauge("serve.serialize_probes.elements", n_elems)
        # Approximate JSON wire size via element count -- encoding to bytes here would double
        # the work. The harness computes the realized response_bytes from the HTTP response.
    return result


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    """Inject serialized probes into completion responses."""
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        probes = getattr(res, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)
            break
    return response


async def _patched_chat_full_generator(self, request, result_generator, *args, **kwargs):
    """Inject serialized probes into chat completion responses."""
    assert _original_chat_full_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    response = await _original_chat_full_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    )

    if last_output is not None and hasattr(response, "model_dump"):
        probes = getattr(last_output, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)

    return response


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup.

    Patches EngineArgs, AsyncLLM.generate, LLM.generate, and the OpenAI
    response builders to enable activation capture via extra_args.

    Usage:
        # Hidden states
        SamplingParams(extra_args={"output_hidden_states": True})
        SamplingParams(extra_args={"output_hidden_states": [layer1, layer2]})

        # QK weights
        SamplingParams(extra_args={"output_qk": True})
        SamplingParams(extra_args={"output_qk": [layer1, layer2]})

    Probe outputs are returned in output.probes and, when using
    vllm serve, injected into the HTTP response body as response.probes.
    """
    global _original_create_engine_config
    global _original_generate, _original_llm_generate
    global _original_completion_response, _original_chat_full_generator

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    # Arm the CUDA-graph QK install path: monkey-patch Worker.load_model so it installs the
    # static-buffer hosts + execute_model wrapper after the model is built (graph/install.py).
    # The patch is a strict no-op unless graph mode is enabled and the worker is the QK worker,
    # so the eager path and the HS / steer workers are untouched.
    #
    # Guarded: importing the graph stack must never be able to disable the eager plugin. If the
    # graph module fails to import for any reason, log and continue -- the eager path (forced
    # when MIA_ALLOW_CUDAGRAPH != "1") is entirely independent of this patch.
    try:
        from mia.graph.install import patch_worker_load_model
        patch_worker_load_model()
    except Exception as e:  # noqa: BLE001
        print(f"[mia] graph load_model patch unavailable ({e}); "
              f"eager path unaffected.")

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    # Patch OpenAI-compatible response builders (only available with vllm serve).
    # Module paths differ across vLLM versions; try all known locations.
    for _completion_module in (
        "vllm.entrypoints.openai.completion.serving",   # <0.12
        "vllm.entrypoints.openai.serving_completion",   # ≥0.12
    ):
        try:
            import importlib as _il
            _mod = _il.import_module(_completion_module)
            _cls = _mod.OpenAIServingCompletion
            _original_completion_response = (
                _cls.request_output_to_completion_response
            )
            _cls.request_output_to_completion_response = _patched_completion_response
            break
        except Exception:
            pass

    for _chat_module in (
        "vllm.entrypoints.openai.chat_completion.serving",  # <0.12
        "vllm.entrypoints.openai.serving_chat",             # ≥0.12
    ):
        try:
            import importlib as _il
            _mod = _il.import_module(_chat_module)
            _cls = _mod.OpenAIServingChat
            _original_chat_full_generator = _cls.chat_completion_full_generator
            _cls.chat_completion_full_generator = _patched_chat_full_generator
            break
        except Exception:
            pass
