"""vLLM plugin entry point: patches the engine, runner and serve path to arm MIA's hooks."""

from __future__ import annotations

import ast as _ast
import asyncio
import contextlib
import contextvars
import copy
import glob
import hashlib
import importlib as _il
import json as _json
import os
import pickle
import shutil
import subprocess
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import quote

import torch._dynamo
import vllm
import zstandard as zstd
from torch.nn.utils.rnn import pad_sequence
from vllm import LLM
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine import EngineCoreRequest, output_processor as op
from vllm.v1.engine.core_client import InprocClient

from mia import register_plugins
from mia._profiler import PROF
from mia.errors import MiaConfigurationError, MiaDeliveryError
from mia.graph import (
    aperture_gather as ag,
    delivery_selector as ds,
    disk_flush_probe as _dfp,
    run_artifact,
)
from mia.graph.aperture_reader import load_qk_aperture_tp
from mia.graph.aperture_sizing import (
    CAPTURE_SUBSYSTEMS,
    DEFAULT_AUTOCAP_HEADROOM_BYTES,
    DEFAULT_AUTOCAP_SAFETY,
    aperture_bytes_is_explicit,
    apply_min_only,
    compute_safe_max_batched_tokens,
    parse_autocap_setting,
    per_token_row_bytes,
    resolve_aperture_bytes,
    resolve_aperture_bytes_auto,
    safe_cap_with_model_sized_aperture,
)
from mia.graph.delivered_probes import (
    DeliveryReadTimeout,
    qk_probes,
    response_id,
    serialize_probes as _serialize_probes,
    trim_probes as _trim_probes,
)
from mia.graph.delivery_route import (
    _WRITE_POOL,
    await_turn,
    end_turn,
    engine_info,
    merge_info,
    patch_app_builder,
    route_enabled,
    run_filter,
)
from mia.graph.delivery_router import RouteDecision, decide_route
from mia.graph.delivery_selector import DP_SIZE_ENV, STAMP_ENV
from mia.graph.run_artifact import artifact_wait_s
from mia.graph.run_mode import DEFAULT_MIA_WORKER, capture_mode_from_env, parse_mia_worker_env
from mia.graph.run_mode import MIA_WORKER_VALUES, UnknownMiaWorkerError  # noqa: F401  (re-exported)
from mia.graph.tp_shard import (
    HS_ALL_RANKS_ENV,
    HS_LAYER_SHARD_RULE,
    HS_MODE_ALL_RANKS,
    HS_MODE_RANK0,
    HS_MODE_ROUND_ROBIN,
    HS_SHARD_ENV,
    HS_SHARD_KEY,
    TP_SHARD_KEY,
    TPShardError,
    hs_expected_ranks,
    hs_max_owned_layers,
    hs_requested_layers,
    merge_hs_payloads,
    merge_qk_payloads,
    parse_rank_dir,
    qk_shard,
    rank_dir_name,
    refuse_pipeline_parallel,
    resolve_hs_shard_mode,
)
from mia.optimizations import env_is_on
from mia.registry import PluginRegistry
from mia.run_utils import (
    estimate_gen_len,
    predict_artifact_kb,
    predicted_rpc_ms,
    qk_score_size_select,
    read_refused_qk,
    route_to_disk,
    rpc_disk_crossover_kb,
)
from mia.runner import UnsupportedRunnerError
from mia.workers._common import match_internal_ids
from mia.workers.steer_worker import (
    _effective_key,
    _load_steering_vector,
    _parse_steer_layers,
    _resolve_steer_config,
    resolve_steer_modes,
)

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_add_request: Callable | None = None
_original_process_inputs: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None

_ADDED_IDS: contextvars.ContextVar = contextvars.ContextVar("mia_added_ids", default=None)

_WORKER_EXT_HS = "mia.workers.hs_capture_worker.HSCaptureWorker"
_WORKER_EXT_QK = "mia.workers.qk_capture_worker.QKCaptureWorker"
_WORKER_EXT_STEER = "mia.workers.steer_worker.SteerWorker"

_WORKER_EXT_BY_KIND = {
    "hidden_states": _WORKER_EXT_HS,
    "qk": _WORKER_EXT_QK,
    "steer": _WORKER_EXT_STEER,
}


class MiaWorkerConflictError(MiaConfigurationError):
    """MIA_WORKER names one subsystem but the engine was given another subsystem's worker class."""


def _kind_from_extension(worker_ext):
    s = worker_ext if isinstance(worker_ext, str) else getattr(
        worker_ext, "__name__", str(worker_ext))
    sl = s.lower()
    for kind, dotted in _WORKER_EXT_BY_KIND.items():
        if dotted.lower() in sl or dotted.rsplit(".", 1)[-1].lower() in sl:
            return kind
    return None


_DEFAULT_HOOK_DIR = "/dev/shm/mia"


def _graph_mode() -> bool:
    # lazy: graph install code (reads env at import) stays out of import mia._plugin
    from mia.graph.install import graph_mode_enabled
    return graph_mode_enabled()


def _decompress(data: bytes) -> Any:
    PROF.gauge("rpc.payload_bytes", len(data))
    with PROF.timed("rpc.decompress"):
        if data[:4] == _ZSTD_MAGIC:
            return pickle.loads(_ZSTD_DECOMPRESSOR.decompress(data))
        return pickle.loads(data)


def _aperture_per_request_mode(engine=None) -> bool:
    graph = (_engine_graph(engine) if engine is not None
             else os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1" or _graph_mode())
    return os.environ.get("MIA_APERTURE_PER_REQUEST") == "1" and graph


def _aperture_per_request_kind(extra: dict, wants_hs: bool, wants_qk: bool,
                               wants_steer: bool, engine=None):
    if not (_aperture_per_request_mode(engine) and not _profile_mode()) or wants_steer:
        return None
    if _resolve_sink(extra) == "drop":
        return None
    if wants_hs and not wants_qk:
        return "hs"
    if wants_qk and not wants_hs:
        return "qk"
    return None


def _takes_aperture_per_request(extra: dict, wants_hs: bool, wants_qk: bool,
                                wants_steer: bool, engine=None) -> bool:
    return _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer, engine) is not None


def _hybrid_serves(extra: dict, wants_hs: bool, wants_qk: bool, wants_steer: bool,
                   engine=None) -> bool:
    stamp = os.environ.get(STAMP_ENV) or ""
    graph = _engine_graph(engine) if engine is not None else _graph_mode()
    return (stamp.partition(":")[0] == "hybrid" and graph
            and wants_hs and not wants_qk and not wants_steer
            and _resolve_sink(extra) != "drop" and not _profile_mode())


def _hs_ask_is_empty(extra: dict, num_layers) -> bool:
    layers = extra.get("output_hidden_states")
    if not isinstance(layers, (list, tuple)) or not num_layers:
        return False
    try:
        return not any(1 <= int(L) <= int(num_layers) for L in layers)
    except (TypeError, ValueError):
        return False


#: A sample's own prefix-cache count, set on its CompletionOutput by ``_patch_sample_cached``.
_SAMPLE_CACHED_ATTR = "_mia_n_cached"


def _patch_sample_cached() -> None:
    """Record every sample's own prefix-cache count on its output (vLLM keeps one per parent)."""
    orig = op.RequestState._new_completion_output
    if getattr(orig, "_mia", False):
        return

    def _new_completion_output(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            setattr(out, _SAMPLE_CACHED_ATTR, int(self.num_cached_tokens or 0))
        except Exception:  # noqa: BLE001
            pass
        return out

    _new_completion_output._mia = True
    op.RequestState._new_completion_output = _new_completion_output


def _sample_cached(output, sample, n: int):
    """A sample's prefix-cache count: its own if recorded, the request's when n == 1, else None."""
    v = getattr(sample, _SAMPLE_CACHED_ATTR, None)
    if v is not None:
        return int(v)
    return int(getattr(output, "num_cached_tokens", 0) or 0) if n == 1 else None


def _hybrid_marker(output, extra: dict, explicit_save, gen_counts: dict, nonce=None,
                   key=None, cached_counts=None) -> dict:
    # lazy: graph install code (reads env at import) stays out of import mia._plugin
    from mia.graph.install_hs import DEFAULT_HOOKS_ON, DEFAULT_HS_MODE
    outs = list(getattr(output, "outputs", None) or [])
    n = max(len(outs), 1)
    n_gen = [int(gen_counts.get(getattr(o, "index", j), len(o.token_ids or [])))
             for j, o in enumerate(outs)] or [0]
    cached_counts = cached_counts or {}
    n_cached = [cached_counts.get(getattr(o, "index", j), _sample_cached(output, o, n))
                for j, o in enumerate(outs)] or [_sample_cached(output, None, 1)]
    layers = extra.get("output_hidden_states")
    return {"delivery": "hybrid", "n": n,
            "layers": list(layers) if isinstance(layers, (list, tuple)) else None,
            "hs_mode": extra.get("hs_mode", DEFAULT_HS_MODE),
            "hooks_on": extra.get("hooks_on", DEFAULT_HOOKS_ON),
            "n_prompt": len(getattr(output, "prompt_token_ids", None) or []),
            "n_gen": n_gen, "n_cached": int(getattr(output, "num_cached_tokens", 0) or 0),
            "n_cached_samples": n_cached,
            "save_to_disk": explicit_save, "run_id": extra.get("run_id"),
            "hook_dir": extra.get("hook_dir"), "nonce": nonce, "key": key}


def _sample_metas(mark: dict) -> list:
    per = mark.get("n_cached_samples")
    return [{"hs_mode": mark["hs_mode"], "hooks_on": mark["hooks_on"],
             "n_prompt": mark["n_prompt"], "n_gen": g,
             "n_cached": per[j] if isinstance(per, list) else mark.get("n_cached", 0)}
            for j, g in enumerate(mark["n_gen"])]


def _spawn_hybrid_writer(engine, request_id, mark: dict, extra: dict, internal=None,
                         start=None) -> None:
    # lazy: spawn_writer looked up at call time so it can be patched
    from mia.graph.delivery_route import spawn_writer
    spawn_writer(engine, ext=str(request_id), n=mark["n"], layers=mark["layers"],
                 metas=_sample_metas(mark),
                 hook_dir=extra.get("hook_dir") or _DEFAULT_HOOK_DIR,
                 run_id=str(extra.get("run_id") or request_id),
                 unit=response_id(str(request_id)), stamp=time.time_ns(),
                 nonce=mark.get("nonce"),
                 keys=_sample_keys(str(internal), mark["n"]) if internal else None, start=start)


def _end_completion_turn(extra: dict, request_id) -> None:
    end_turn(extra.get("hook_dir") or _DEFAULT_HOOK_DIR, str(extra.get("run_id") or request_id),
             str(request_id))


def _sample_keys(base: str, n: int) -> list:
    return [base] if n <= 1 else [f"{j}_{base}" for j in range(n)]


def _qk_dest(hook_dir: str, run_id, key: str) -> str:
    return os.path.join(hook_dir, str(run_id), ".mia_qk", quote(str(key), safe=""))


def _qk_ask_is_empty(extra: dict, num_layers) -> bool:
    spec = extra.get("output_qk")
    if not isinstance(spec, (dict, list, tuple)) or not num_layers:
        return False
    try:
        return not any(0 <= int(L) < int(num_layers) for L in spec)
    except (TypeError, ValueError):
        return False


def _outputs_by_index(output) -> list:
    outs = list(getattr(output, "outputs", None) or [])
    return sorted(outs, key=lambda o: getattr(o, "index", 0))


def _raise_delivery_errors(request_id, parts) -> None:
    for p in parts:
        if isinstance(p, dict) and "mia_error" in p:
            raise MiaDeliveryError(
                f"Q/K capture of request {request_id!r} cannot be delivered: {p['mia_error']}")


def _qk_mode(extra: dict) -> str:
    return extra.get("hookq_mode") or "all_tokens"


def _qk_hooks(extra: dict) -> str:
    return extra.get("hooks_on") or "prefill"


async def _qk_serve_rpc(engine, output, keys, mode: str, hooks: str) -> None:
    outs = _outputs_by_index(output)
    n_prompt = len(getattr(output, "prompt_token_ids", None) or [])
    probes = []
    for j, key in enumerate(keys):
        p = await _await_aperture_per_request(engine, key)
        n_gen = len(outs[j].token_ids or []) if j < len(outs) else 0
        probes.append(None if p is None else qk_probes(p, n_prompt=n_prompt, n_gen=n_gen,
                                                       hookq_mode=mode, hooks_on=hooks))
    if probes and probes[0] is not None:
        output.probes = probes[0]
    if len(keys) > 1:
        for c, p in zip(outs, probes):
            if p is not None:
                c.probes = p


def _drop_qk_staging(hook_dir: str, run_id, keys) -> None:
    for key in keys:
        shutil.rmtree(_qk_dest(hook_dir, run_id, key), ignore_errors=True)
    try:
        os.rmdir(os.path.join(hook_dir, str(run_id), ".mia_qk"))
    except OSError:
        pass


def _write_qk_run(output, keys, info, hook_dir: str, run_id: str, mode: str, hooks: str,
                  unit=None, stamp=None, nonce=None, start=None) -> None:
    outs = _outputs_by_index(output)
    n_prompt = len(getattr(output, "prompt_token_ids", None) or [])
    try:
        items = []
        for j, key in enumerate(keys):
            try:
                got = load_qk_aperture_tp(_qk_dest(hook_dir, run_id, key))
            except NotImplementedError as e:
                raise MiaDeliveryError(
                    f"Q/K capture of request {key!r} cannot be delivered: {e}") from None
            if len(got) != 1:
                raise MiaDeliveryError(f"Q/K delivery of {key!r} holds {len(got)} requests, not 1")
            payload = {"qk_cache": next(iter(got.values())), "config": info["config"],
                       "module_names": info["names"]}
            n_gen = len(outs[j].token_ids or []) if j < len(outs) else 0
            items.append((key, qk_probes(payload, n_prompt=n_prompt, n_gen=n_gen,
                                         hookq_mode=mode, hooks_on=hooks, layout="disk")))
        run_artifact.append(hook_dir, run_id, kind="qk", items=items, unit=unit, stamp=stamp,
                            nonce=nonce, start=start)
    finally:
        _drop_qk_staging(hook_dir, run_id, keys)


async def _qk_serve_disk(engine, output, extra, keys, request_id, start=None) -> None:
    stamp, nonce = time.time_ns(), uuid.uuid4().hex
    run_id = str(extra.get("run_id") or request_id)
    hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
    loop = asyncio.get_running_loop()
    try:
        for key in keys:
            with PROF.timed("aperture.await_disk_confirm"):
                ok = await _await_aperture_disk_confirm(engine, key)
            if ok is not True:
                await loop.run_in_executor(_WRITE_POOL, _drop_qk_staging, hook_dir, run_id, keys)
                raise MiaDeliveryError(f"Q/K capture of request {key!r} did not land under "
                                       f"{hook_dir} (MIA_APERTURE_DELIVER_TIMEOUT_S)")
        info = await engine_info(engine)
        await await_turn(hook_dir, run_id, str(request_id))
        with PROF.timed("aperture.qk_run_write"):
            await loop.run_in_executor(_WRITE_POOL, _write_qk_run, output, keys, info, hook_dir,
                                       run_id, _qk_mode(extra), _qk_hooks(extra),
                                       response_id(str(request_id)), stamp, nonce, start)
    finally:
        end_turn(hook_dir, run_id, str(request_id))


def _reconstruct_compact_qk(probes: dict) -> None:
    qk = probes.get("qk_cache") if isinstance(probes, dict) else None
    if not isinstance(qk, dict):
        return
    for entry in qk.values():
        if not isinstance(entry, dict):
            continue
        full = entry.pop("k_full", None)
        ends = entry.pop("k_prefix_ends", None)
        if full is not None and ends is not None:
            entry["k_all"] = pad_sequence([full[:int(L)] for L in ends], batch_first=True)


def _stable_artifact_files(run_dir: str) -> list:
    try:
        return sorted(
            f for f in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)
            if os.path.isfile(f) and not f.endswith(".tmp") and os.path.getsize(f) > 0
        )
    except OSError:
        return []


_ARTIFACT_WAIT_S = 10.0
_ARTIFACT_POLL_S = 0.005


def _artifact_wait_s() -> float:
    if os.environ.get("MIA_ARTIFACT_WAIT_S"):
        return artifact_wait_s()
    return _ARTIFACT_WAIT_S


def _flushed_rank_dirs(results, run_id: str, hook_dir: str):
    if not isinstance(results, (list, tuple)) or any(r is True for r in results):
        return None
    run_dir = os.path.join(hook_dir, run_id)
    dirs = set()
    for r in results:
        if isinstance(r, str) and r:
            rank = parse_rank_dir(r)
            dirs.add(os.path.join(run_dir, rank_dir_name(rank)) if rank is not None else r)
    return sorted(dirs) or None


_ARTIFACT_NAMES = frozenset({"qk.pt", "qk.safetensors",
                             "hidden_states.pt", "hidden_states.safetensors"})


def _landed(files: list) -> bool:
    """A qk/hidden_states artifact is present, and every ``.safetensors`` has its ``.json``."""
    have = set(files)
    return (any(os.path.basename(f) in _ARTIFACT_NAMES for f in files)
            and all(f[:-len(".safetensors")] + ".json" in have
                    for f in files if f.endswith(".safetensors")))


def _all_refused(hook_dir: str, run_id, req_ids) -> bool:
    """Every request's Q/K capture was refused, so its flush wrote no artifact to wait for."""
    gone = read_refused_qk(hook_dir, str(run_id))
    return bool(gone) and all(match_internal_ids(gone, str(r)) for r in req_ids)


def _artifact_barrier_state(run_dir: str, rank_dirs) -> "tuple[list, bool]":
    if not rank_dirs:
        files = _stable_artifact_files(run_dir)
        return files, _landed(files)
    per_rank = [_stable_artifact_files(d) for d in rank_dirs]
    return [f for fs in per_rank for f in fs], all(_landed(fs) for fs in per_rank)


def _log_barrier_timeout(run_id: str, rank_dirs) -> None:
    if not rank_dirs:
        print(f"[mia/disk] durability barrier TIMEOUT after {_artifact_wait_s():.0f}s for "
              f"run_id {run_id!r}: no artifact landed. A loader will raise FileNotFoundError "
              f"for this run_id -- the write did not finish, the run_id is not wrong.", flush=True)
        return
    missing = [os.path.basename(d) for d in rank_dirs if not _landed(_stable_artifact_files(d))]
    if missing:
        print(f"[mia/disk] durability barrier TIMEOUT after {_artifact_wait_s():.0f}s for "
              f"run_id {run_id!r}: {len(rank_dirs) - len(missing)}/{len(rank_dirs)} rank "
              f"artifact(s) landed, missing {missing}. The run's artifact is INCOMPLETE; the QK "
              f"loader will refuse it until every rank's shard is on disk.", flush=True)


def _resolve_sink(extra: dict) -> str:
    env = os.environ.get("MIA_SINK", "").lower()
    if env == "drop":
        return "drop"
    if "save_to_disk" in extra:
        return "disk" if bool(extra["save_to_disk"]) else "rpc"
    if env in ("disk", "rpc"):
        return env
    return "disk" if bool(extra.get("save_to_disk")) else "rpc"


async def _await_disk_artifact(run_id: str, hook_dir: str, rank_dirs=None) -> bool:
    run_dir = os.path.join(hook_dir, run_id)
    prev = None
    for _ in range(max(2, int(_artifact_wait_s() / _ARTIFACT_POLL_S))):
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
    run_dir = os.path.join(hook_dir, run_id)
    prev = None
    for _ in range(max(2, int(_artifact_wait_s() / _ARTIFACT_POLL_S))):
        files, complete = _artifact_barrier_state(run_dir, rank_dirs)
        if complete and files == prev:
            return True
        prev = files
        time.sleep(_ARTIFACT_POLL_S)
    complete = _artifact_barrier_state(run_dir, rank_dirs)[1]
    if not complete:
        _log_barrier_timeout(run_id, rank_dirs)
    return complete


_APERTURE_DELIVER_POLL_S = 0.005


def _aperture_deliver_timeout_s() -> float:
    try:
        return max(0.1, float(os.environ.get("MIA_APERTURE_DELIVER_TIMEOUT_S", "30") or "30"))
    except (TypeError, ValueError):
        return 30.0


async def _await_aperture_per_request(engine, request_id, hs_layers=None):
    timeout = _aperture_deliver_timeout_s()
    deadline = time.monotonic() + timeout
    collected: dict = {}
    while True:
        with PROF.timed("rpc.get_aperture_per_request"):
            states = await engine.collective_rpc("get_aperture_per_request", args=(request_id,))
        for i, s in enumerate(states):
            if s is not None and i not in collected:
                collected[i] = _decompress(s)
        _raise_delivery_errors(request_id, collected.values())
        if collected:
            parts = [collected[i] for i in sorted(collected)]
            if len(parts) >= _expected_probe_parts(parts, hs_layers):
                return merge_probe_parts(parts, hs_layers)
        if time.monotonic() >= deadline:
            print(f"[mia/aperture] BLOCK-UNTIL-HELD TIMEOUT after {timeout:.1f}s waiting for RPC "
                  f"per-request delivery of {request_id!r} ({len(collected)} rank part(s) held); "
                  f"leaving probes unset (raise MIA_APERTURE_DELIVER_TIMEOUT_S if the off-loop "
                  f"consumer is merely slow, else it is a bug)", flush=True)
            return None
        await asyncio.sleep(_APERTURE_DELIVER_POLL_S)


def _collect_aperture_per_request_sync(rpc, request_id, hs_layers=None, *, wait=False):
    collected: dict = {}
    deadline = None
    while True:
        with PROF.timed("rpc.get_aperture_per_request"):
            states = rpc("get_aperture_per_request", args=(request_id,))
        for i, s in enumerate(states):
            if s is not None and i not in collected:
                collected[i] = _decompress(s)
        _raise_delivery_errors(request_id, collected.values())
        if not collected and not wait:
            return None
        parts = [collected[i] for i in sorted(collected)]
        if parts and len(parts) >= _expected_probe_parts(parts, hs_layers):
            return merge_probe_parts(parts, hs_layers)
        if deadline is None:
            deadline = time.monotonic() + _aperture_deliver_timeout_s()
        elif time.monotonic() >= deadline and wait:
            raise MiaDeliveryError(
                f"Q/K capture of request {request_id!r} was not delivered within "
                f"{_aperture_deliver_timeout_s():.1f}s ({len(parts)} rank part(s) held; raise "
                f"MIA_APERTURE_DELIVER_TIMEOUT_S if the drain is merely slow)")
        elif time.monotonic() >= deadline:
            print(f"[mia/aperture] PER-REQUEST DELIVERY INCOMPLETE for {request_id!r} after "
                  f"{_aperture_deliver_timeout_s():.1f}s: {len(parts)} of "
                  f"{_expected_probe_parts(parts, hs_layers)} rank part(s) held "
                  f"(tp_rank(s) {sorted(collected)}); leaving probes unset -- the held part(s) are "
                  f"DISCARDED, each rank delivers a request only once. Raise "
                  f"MIA_APERTURE_DELIVER_TIMEOUT_S if the off-loop consumer is merely slow, else "
                  f"it is a bug", flush=True)
            return None
        time.sleep(_APERTURE_DELIVER_POLL_S)


def _expected_probe_parts(parts, hs_layers=None) -> int:
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
    """Merge per-worker probe results from a retrieval collective_rpc into one dict."""
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
    timeout = _aperture_deliver_timeout_s()
    deadline = time.monotonic() + timeout
    while True:
        with PROF.timed("rpc.confirm_aperture_delivery"):
            res = await engine.collective_rpc("confirm_aperture_delivery", args=(request_id, 0.0))
        staged = [r for r in res if r is not None]
        if staged and all(r is True for r in staged):
            return True
        if time.monotonic() >= deadline:
            print(f"[mia/aperture] DISK-CONFIRM TIMEOUT after {timeout:.1f}s waiting for offload "
                  f"delivery of {request_id!r}; the client dest file may be incomplete (raise "
                  f"MIA_APERTURE_DELIVER_TIMEOUT_S, or check the offload worker)", flush=True)
            return False
        await asyncio.sleep(_APERTURE_DELIVER_POLL_S)


def _apply_delivery_selector():
    global _STARTED_STAMPED
    if os.environ.get(ds.STAMP_ENV):
        _STARTED_STAMPED = True
        return ds.apply()
    ds.validate(os.environ)
    return None


_STARTED_STAMPED = False
_MIA_WRITES: list = []
_MIA_DYNAMO_WRITE = None


def _mia_setenv(key: str, value) -> None:
    prev = os.environ.get(key)
    if prev == value:
        return
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    _MIA_WRITES.append((key, prev, value))


def _drop_inherited_delivery() -> None:
    """A process that inherited a delivery stamp and builds an engine resolves that engine itself."""
    global _STARTED_STAMPED
    mode = (os.environ.pop(ds.STAMP_ENV, None) or "").partition(":")[0]
    implied = {"hybrid": (ds.GATHER_ENV, ds.DELIVER_ENV), "drain": (ds.PER_REQUEST_ENV,)}
    for key in implied.get(mode, ()):
        if os.environ.get(key) == "1":
            os.environ.pop(key)
    os.environ.pop(ds.DP_SIZE_ENV, None)
    _STARTED_STAMPED = False


def _revert_mia_writes() -> None:
    global _MIA_DYNAMO_WRITE
    for key, prev, written in reversed(_MIA_WRITES):
        if os.environ.get(key) == written:
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
    _MIA_WRITES.clear()
    if _MIA_DYNAMO_WRITE is not None:
        prev, written = _MIA_DYNAMO_WRITE
        if torch._dynamo.config.disable == written:
            torch._dynamo.config.disable = prev
        _MIA_DYNAMO_WRITE = None


def _apply_engine_delivery(worker_kind: str, graph: bool):
    before = {k: os.environ.get(k) for k in ds.WRITES}
    sel = ds.apply(os.environ, worker_kind=worker_kind, graph=graph)
    for k in ds.WRITES:
        after = os.environ.get(k)
        if after != before[k]:
            _MIA_WRITES.append((k, before[k], after))
    return sel


def _warn_profile_aperture_conflict() -> None:
    if getattr(_warn_profile_aperture_conflict, "_warned", False):
        return
    if (os.environ.get("MIA_PROFILE_MODE") == "1"
            and os.environ.get("MIA_APERTURE_PER_REQUEST") == "1"):
        _warn_profile_aperture_conflict._warned = True
        print("[mia/aperture] WARNING: MIA_PROFILE_MODE=1 AND MIA_APERTURE_PER_REQUEST=1 "
              "are BOTH set -- this is a misconfiguration. Profile mode must pair with the "
              "shared-file / disk Component-1 drain, NOT the host-buffer per-request route (which "
              "leaks host RAM -- rows demux into an index nothing retrieves -- and misreports the "
              "capture->NVMe boundary). Unset MIA_APERTURE_PER_REQUEST for profile-mode "
              "Component-1 measurement.", flush=True)


_VLLM_VER_NOTED = False


def _note_vllm_version() -> None:
    global _VLLM_VER_NOTED
    if _VLLM_VER_NOTED:
        return
    _VLLM_VER_NOTED = True
    try:
        # lazy: optional dependency (packaging)
        from packaging.version import Version
        raw = vllm.__version__
        ver = Version(raw)
        if ver.release[:2] == (0, 29):
            return
        print(f"[mia] NOTE: this branch is developed and GPU-validated on vLLM 0.29.x "
              f"with the V2 model runner; found {raw} — UNTESTED on this branch.",
              flush=True)
    except Exception:  # noqa: BLE001
        return


def _worker_kind(worker_ext) -> str:
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
    itemsize = getattr(dt, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize
    return _DTYPE_BYTES.get(str(dt), 2)


def _autocap_setting():
    return parse_autocap_setting(os.environ.get("MIA_APERTURE_MAX_BATCHED_TOKENS"))


_WORKER_KIND_TO_SUBSYSTEM = {"hidden_states": "hs", "qk": "qk", "steer": "steer"}


def _enabled_subsystems(worker_kinds) -> set:
    kinds = [worker_kinds] if isinstance(worker_kinds, str) else list(worker_kinds or ())
    return {_WORKER_KIND_TO_SUBSYSTEM[k] for k in kinds if k in _WORKER_KIND_TO_SUBSYSTEM}


def _model_dims(config) -> dict:
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
        tp = int(getattr(getattr(config, "parallel_config", None),
                         "tensor_parallel_size", 1) or 1)
        shard = qk_shard(0, tp, h_q, h_kv, head_dim)
        dims["n_q_heads"] = shard.num_local_q_heads
        dims["n_kv_heads"] = shard.num_local_kv_heads
        dims["head_dim"] = head_dim
        dims["tp"] = tp
    return dims


def _hs_layers_per_rank(config, n_layers: int) -> int:
    tp = int(getattr(getattr(config, "parallel_config", None), "tensor_parallel_size", 1) or 1)
    return hs_max_owned_layers(int(n_layers), tp, resolve_hs_shard_mode(tp))


def _derive_safe_max_batched_tokens(config, worker_kinds):
    # lazy: current_platform looked up at call time so it can be patched
    from vllm.platforms import current_platform
    dims = _model_dims(config)
    n_layers = dims["layers"]
    gpu_util = float(config.cache_config.gpu_memory_utilization)
    total_gpu = int(current_platform.get_device_total_memory(0))
    budget = resolve_aperture_bytes_auto(total_gpu, gpu_util)
    slices = resolve_aperture_bytes(
        _enabled_subsystems(worker_kinds), gpu_bytes_budget=budget, model_dims=dims)
    if not slices:
        return None
    if set(slices) == {"hs"}:
        n_layers = _hs_layers_per_rank(config, n_layers)
    aperture_bytes = sum(slices.values())
    plt = sum(per_token_row_bytes(s, dims) for s in slices)
    safety = int(os.environ.get("MIA_APERTURE_AUTOCAP_SAFETY") or DEFAULT_AUTOCAP_SAFETY)
    headroom = int(os.environ.get("MIA_APERTURE_AUTOCAP_HEADROOM_BYTES") or DEFAULT_AUTOCAP_HEADROOM_BYTES)
    cap = compute_safe_max_batched_tokens(
        total_gpu, gpu_util, aperture_bytes, n_layers, plt, safety=safety, headroom_bytes=headroom)
    if (cap is not None and len(slices) == 1 and not aperture_bytes_is_explicit()
            and cap * n_layers * plt > aperture_bytes):
        cap = safe_cap_with_model_sized_aperture(
            total_gpu, gpu_util, aperture_bytes, n_layers, plt,
            safety=safety, headroom_bytes=headroom)
    return cap


def _maybe_autocap_max_batched_tokens(config, worker_kinds) -> None:
    try:
        mode, explicit = _autocap_setting()
        if mode == "off":
            return
        subsystems = _enabled_subsystems(worker_kinds)
        if not (subsystems & set(CAPTURE_SUBSYSTEMS)):
            return
        if mode == "explicit":
            safe = int(explicit)
        else:
            safe = _derive_safe_max_batched_tokens(config, worker_kinds)
        if safe is None:
            return
        sc = config.scheduler_config
        current = getattr(sc, "max_num_batched_tokens", None)
        new = apply_min_only(current, safe)
        if new is not None and new != current:
            sc.max_num_batched_tokens = int(new)
            print(f"[mia] autocap: max_num_batched_tokens {current} -> {new} "
                  f"(subsystems={','.join(sorted(subsystems))}, mode={mode}); "
                  f"bounds full-graph capture per-step transient", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[mia] autocap SKIPPED (non-fatal): {e!r}", flush=True)


_COMPILE_CACHE_STAMP_KEY = "mia_graph_capture"


def _mia_source_id() -> str:
    cached = getattr(_mia_source_id, "_cached", None)
    if cached is not None:
        return cached

    version = "0.6.0"
    try:
        # lazy: as _v it would clash with the _v locals elsewhere in this module
        from importlib.metadata import version as _v
        version = _v("mia")
    except Exception:  # noqa: BLE001
        pass
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode == 0:
            state = hashlib.sha256()
            diff = subprocess.run(["git", "-C", root, "diff", "HEAD", "--", "mia"],
                                  capture_output=True, text=True, timeout=10)
            if diff.returncode == 0:
                state.update(diff.stdout.encode())
            dirty = bool(diff.returncode == 0 and diff.stdout.strip())
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
    """The MIA state that shapes the compiled graph but is invisible to vLLM's cache key."""

    kind = str(worker_kind)
    tp_size = int(getattr(engine_args, "tensor_parallel_size", 1) or 1)
    raw_aperture = os.environ.get("MIA_APERTURE_GPU_BYTES")
    if raw_aperture is None or not raw_aperture.strip():
        aperture = "auto"
    else:
        try:
            aperture = str(int(raw_aperture))
        except ValueError:
            aperture = raw_aperture.strip()
    layout: dict = {"kind": kind, "tp_size": tp_size, "aperture_gpu_bytes": aperture}

    if kind == "hidden_states":
        mode = resolve_hs_shard_mode(tp_size)
        owned = {
            HS_MODE_ROUND_ROBIN: f"layers i where i % {tp_size} == tp_rank",
            HS_MODE_RANK0: "every layer on tp_rank 0, none on any other rank",
            HS_MODE_ALL_RANKS: "every layer on every rank",
        }.get(mode, "every layer (single rank)")
        layout.update({
            "hs_layout": mode,
            "hs_shard_env": os.environ.get(HS_SHARD_ENV) or "",
            "hs_all_ranks_env": os.environ.get(HS_ALL_RANKS_ENV) or "",
            "hs_layer_shard_rule": HS_LAYER_SHARD_RULE if mode == HS_MODE_ROUND_ROBIN else "none",
            "hs_owned_layers": owned,
            "hs_sinks": bool(tp_size > 1 and os.environ.get("MIA_HS_TP_SYMMETRIC", "1") != "0"),
            "hs_capture_mode": (os.environ.get("MIA_HS_CAPTURE", "buffer") or "").strip().lower(),
        })
    elif kind == "qk":
        layout.update({
            "qk_capture_mode": (os.environ.get("MIA_QK_CAPTURE", "buffer") or "").strip().lower(),
            "qk_head_shard": f"1/{tp_size} of the q and kv heads on every rank",
        })
    elif kind == "steer":
        raw_vmax = os.environ.get("MIA_STEER_VMAX", "16") or "16"
        try:
            v_max: object = int(raw_vmax)
        except ValueError:
            v_max = raw_vmax
        layout.update({
            "steer_mode": (os.environ.get("MIA_STEER_MODE", "buffer") or "").strip().lower(),
            "steer_v_max": v_max,
        })
    return layout


def stamp_compile_cache_key(engine_args, worker_kind: str) -> None:
    """Make MIA's baked-op variant part of vLLM's compile-cache key."""
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
        print(f"[mia] WARNING: additional_config is a {type(current).__name__}, not a dict; "
              f"cannot stamp the compile-cache key with worker={worker_kind}. If you run more "
              f"than one MIA worker kind on this machine, set VLLM_DISABLE_COMPILE_CACHE=1.",
              flush=True)
        return
    print(f"[mia] compile-cache key stamped with worker={worker_kind} mia={stamp['mia']} "
          f"layout={_json.dumps(stamp['layout'], sort_keys=True)} "
          f"(worker_extension_cls is excluded from vLLM's own key)", flush=True)


class UnsupportedGraphModeError(RuntimeError):
    """A cudagraph_mode outside NONE and ``_GRAPH_MODES`` (e.g. PIECEWISE alone)."""


DEFAULT_CUDAGRAPH_MODE = "FULL_AND_PIECEWISE"

_GRAPH_MODES = ("FULL_AND_PIECEWISE", "FULL_DECODE_ONLY", "FULL")

_FULL_MODE_WARNING = (
    "[mia] WARNING: cudagraph_mode=FULL replays mixed prefill/decode batches in FULL CUDA graphs, "
    "which can compute wrong attention (FlashAttention 3, vLLM 0.29); leave cudagraph_mode unset "
    f"for MIA's default {DEFAULT_CUDAGRAPH_MODE}.")


def validate_graph_mode(mode_name: str) -> None:
    """Accept only the modes MIA's capture path is validated for."""
    if str(mode_name).upper() not in {"NONE", *_GRAPH_MODES}:
        raise UnsupportedGraphModeError(
            f"MIA supports cudagraph_mode {', '.join(_GRAPH_MODES)} and NONE (eager); got "
            f"{mode_name}. Leave cudagraph_mode unset for MIA's default {DEFAULT_CUDAGRAPH_MODE}, "
            f"or pass enforce_eager=True for the eager path."
        )


def validate_v2_runner_selected(config) -> None:
    """Refuse a config that will build vLLM's V1 model runner."""
    try:
        selected = getattr(config, "use_v2_model_runner")
    except Exception:  # noqa: BLE001
        return
    if selected is False:
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


_ENGINE_HINTS: contextvars.ContextVar = contextvars.ContextVar("mia_engine_hints", default={})


@contextlib.contextmanager
def engine_hints(**hints):
    """Hints for engines built inside this block, in this thread (``qk_score=True``)."""
    token = _ENGINE_HINTS.set({**_ENGINE_HINTS.get(), **hints})
    try:
        yield
    finally:
        _ENGINE_HINTS.reset(token)


@dataclass(frozen=True)
class CaptureMode:
    """One engine's capture mode."""
    graph: bool
    engine_eager: bool
    source: str
    reason: str
    prefix_caching_off: bool = False


def _qk_prefix_unset(engine_args, kind: str) -> bool:
    return kind == "qk" and getattr(engine_args, "enable_prefix_caching", None) is None


def _is_o0(engine_args) -> bool:
    lvl = getattr(engine_args, "optimization_level", None)
    if isinstance(lvl, str):
        lvl = lvl.strip().upper().lstrip("O")
    try:
        return lvl is not None and int(lvl) == 0
    except (TypeError, ValueError):
        return False


def resolve_capture_mode(engine_args, env, hints=None, *, kind: str) -> CaptureMode:
    """Graph or eager for this engine, and who chose."""
    hints = hints or {}
    explicit = capture_mode_from_env(env)
    eager_arg = bool(getattr(engine_args, "enforce_eager", False))
    o0 = _is_o0(engine_args)
    if explicit is True:
        why = ("MIA_ALLOW_CUDAGRAPH=1" + (", enforce_eager=True" if eager_arg else "")
               + (", optimization_level=O0" if o0 else ""))
        return CaptureMode(True, eager_arg or o0, "explicit", why,
                           prefix_caching_off=_qk_prefix_unset(engine_args, kind))
    if explicit is False:
        return CaptureMode(False, True, "explicit", "MIA_ALLOW_CUDAGRAPH=0")
    if eager_arg:
        return CaptureMode(False, True, "explicit", "enforce_eager=True")
    if o0:
        return CaptureMode(False, True, "explicit", "optimization_level=O0")
    if kind == "qk":
        if (env.get("MIA_QK_SCORE") == "1" or env.get("MIA_QK_AUTO_SELECT") == "1"
                or hints.get("qk_score")):
            return CaptureMode(False, True, "scope", "score capture has no graph path")
        if getattr(engine_args, "enable_prefix_caching", None) is True:
            return CaptureMode(False, True, "scope", "QK with prefix caching has no graph path")
        if int(getattr(engine_args, "data_parallel_size", 1) or 1) > 1:
            return CaptureMode(False, True, "scope",
                               "QK per-request delivery cannot be collected across DP engines")
    return CaptureMode(True, False, "default", "unset",
                       prefix_caching_off=_qk_prefix_unset(engine_args, kind))


def _cc_mode_name(cc):
    m = cc.get("cudagraph_mode") if isinstance(cc, dict) else getattr(cc, "cudagraph_mode", None)
    if m is None:
        return None
    return m if isinstance(m, str) else getattr(m, "name", str(m))


def _compile_mode_name(cc):
    m = cc.get("mode") if isinstance(cc, dict) else getattr(cc, "mode", None)
    if m is None:
        return None
    try:
        if isinstance(m, str) and not m.isdigit():
            return CompilationMode[m.upper()].name
        return CompilationMode(int(m)).name
    except (KeyError, ValueError, TypeError):
        return str(m)


def _default_cudagraph_mode(cc):
    name = _compile_mode_name(cc)
    if name is not None and name != "VLLM_COMPILE":
        return "FULL_DECODE_ONLY", f"compilation mode {name}"
    if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        return "FULL_DECODE_ONLY", "TORCH_COMPILE_DISABLE=1"
    return DEFAULT_CUDAGRAPH_MODE, None


def _with_default_cudagraph(cc, default=DEFAULT_CUDAGRAPH_MODE):
    if cc is None:
        return CompilationConfig(cudagraph_mode=CUDAGraphMode[default])
    if isinstance(cc, dict):
        return (cc if cc.get("cudagraph_mode") is not None
                else {**cc, "cudagraph_mode": default})
    if getattr(cc, "cudagraph_mode", "missing") is None:
        new = copy.deepcopy(cc)
        new.cudagraph_mode = CUDAGraphMode[default]
        return new
    return cc


def _disable_dynamo_for_eager_hooks() -> None:
    global _MIA_DYNAMO_WRITE
    if os.environ.get("TORCHDYNAMO_DISABLE") is not None:
        return
    _mia_setenv("TORCHDYNAMO_DISABLE", "1")
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    _MIA_DYNAMO_WRITE = (prev, True)


_MIA_PKG_DIR = str(Path(__file__).resolve().parent)


def _capture_mode_line(mode: CaptureMode, engine_args, cg_explicit: bool = False,
                       cg_why=None) -> str:
    if mode.graph and not mode.engine_eager:
        cg = str(_cc_mode_name(getattr(engine_args, "compilation_config", None))).upper()
        who = "chosen explicitly" if cg_explicit else "chosen by default"
        what = (f"{cg} CUDA graph ({who})" if cg in _GRAPH_MODES
                else f"CUDA graph (cudagraph_mode={cg})")
    elif mode.graph:
        what = "graph capture op on an eager engine"
    else:
        what = "eager"
    if mode.source == "default":
        why = "chosen by default (enforce_eager=True or MIA_ALLOW_CUDAGRAPH=0 selects eager)"
    elif mode.source == "explicit":
        why = f"chosen explicitly ({mode.reason})"
    else:
        why = mode.reason
    if mode.graph and not mode.engine_eager:
        why = "graph mode " + why
    if cg_why:
        why += f"; FULL_DECODE_ONLY because {cg_why} has no piecewise CUDA graphs"
    if mode.prefix_caching_off:
        why += "; prefix caching off for QK capture (default)"
    return f"[mia] capture mode: {what} -- {why} mia={_MIA_PKG_DIR}"


def _patched_create_engine_config(self, *args, **kwargs):
    if not self.worker_extension_cls:
        worker_type = parse_mia_worker_env(os.environ.get("MIA_WORKER")) or DEFAULT_MIA_WORKER
        self.worker_extension_cls = _WORKER_EXT_BY_KIND[worker_type]
    _wkind = _worker_kind(self.worker_extension_cls)

    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0":
        raise RuntimeError(
            "MIA requires vLLM's V2 model runner, but VLLM_USE_V2_MODEL_RUNNER=0 is set. "
            "Unset it (0.29 defaults to V2) or set it to 1."
        )

    refuse_pipeline_parallel(getattr(self, "pipeline_parallel_size", 1), "engine arguments")

    resolve_hs_shard_mode(int(getattr(self, "tensor_parallel_size", 1) or 1))

    _note_vllm_version()

    if not _STARTED_STAMPED:
        _revert_mia_writes()
    else:
        _drop_inherited_delivery()
    mode = resolve_capture_mode(self, os.environ, _ENGINE_HINTS.get(), kind=_wkind)
    graph_mode = mode.graph
    # lazy: graph install code (reads env at import) stays out of import mia._plugin
    from mia.graph.install import set_graph_mode
    set_graph_mode(graph_mode)
    if mode.engine_eager:
        self.enforce_eager = True
    cg_explicit, cg_why = False, None
    if graph_mode:
        stamp_compile_cache_key(self, _wkind)
        if not mode.engine_eager:
            cc = getattr(self, "compilation_config", None)
            cg_explicit = _cc_mode_name(cc) is not None
            default, cg_why = _default_cudagraph_mode(cc)
            if cg_explicit:
                cg_why = None
            self.compilation_config = _with_default_cudagraph(cc, default)
        if mode.prefix_caching_off:
            self.enable_prefix_caching = False
    else:
        _disable_dynamo_for_eager_hooks()
    dp_size = int(getattr(self, "data_parallel_size", 1) or 1)
    if dp_size > 1:
        _mia_setenv(DP_SIZE_ENV, str(dp_size))
    print(_capture_mode_line(mode, self, cg_explicit, cg_why), flush=True)
    # vLLM may resolve the mode in place on this object; keep what MIA handed it.
    cg_asked = (str(_cc_mode_name(self.compilation_config)).upper()
                if graph_mode and not mode.engine_eager else None)
    if (graph_mode and not mode.engine_eager
            and str(_cc_mode_name(self.compilation_config)).upper() == "FULL"):
        print(_FULL_MODE_WARNING, flush=True)
    _apply_engine_delivery(_wkind, graph_mode)

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
                        setter("cudagraph_capture_sizes", None)
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

    resolved = _cudagraph_mode_name(config.compilation_config.cudagraph_mode)
    validate_graph_mode(resolved)
    if cg_asked is not None and cg_asked != str(resolved).upper():
        print(f"[mia] cudagraph_mode resolved by vLLM: {str(resolved).upper()} (asked {cg_asked})",
              flush=True)
    validate_v2_runner_selected(config)
    refuse_pipeline_parallel(
        getattr(getattr(config, "parallel_config", None), "pipeline_parallel_size", 1),
        "resolved engine config")

    if graph_mode:
        _maybe_autocap_max_batched_tokens(config, [_wkind])


    return config


def _prompt_token_len(prompt):
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
    try:
        n_prompt = len(getattr(output, "prompt_token_ids", None) or [])
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
    except Exception:  # noqa: BLE001
        pass


def _maybe_storage_route(engine, prompt, extra, max_tokens) -> bool | None:
    if not env_is_on("storage_router"):
        return None
    _dbg = os.environ.get("MIA_ROUTER_DEBUG") == "1"

    def _log(msg):
        if not _dbg:
            return
        n = getattr(_maybe_storage_route, "_dbg_n", 0)
        if n < 20:
            _maybe_storage_route._dbg_n = n + 1
            print(f"[mia/router] {msg}", flush=True)

    wants_qk = extra.get("output_qk") is not None
    wants_hs = extra.get("output_hidden_states") is not None
    if not (wants_qk or wants_hs):
        return None
    P = _prompt_token_len(prompt)
    if not P:
        _log(f"P=None (prompt type={type(prompt).__name__}) -> no route")
        return None
    hooks_on = extra.get("hooks_on", "both")
    gen_len = estimate_gen_len(max_tokens)
    try:
        if wants_qk:
            dims = _qk_model_dims(engine)
            if not dims:
                _log("qk dims=None -> no route")
                return None
            H_q, _H_kv, head_dim = dims
            oqk = extra.get("output_qk") or {}
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
        else:
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


_FLUSH_PROBE_ATTR = "_mia_flush_probe"
_FLUSH_ANNOUNCED: set = set()


async def _flush_disk_coalesced(engine, *, request_id, run_id, hook_dir, wants_hs, wants_qk,
                                wants_steer, sink, per_request, durable_wait):
    cls = None
    if _dfp.skip_enabled():
        cls = _dfp.flush_class(graph=_engine_graph(engine), wants_hs=bool(wants_hs),
                               wants_qk=bool(wants_qk), wants_steer=bool(wants_steer),
                               sink=str(sink), per_request=bool(per_request),
                               durable_wait=bool(durable_wait),
                               run_id=str(run_id), hook_dir=str(hook_dir))
    if cls is None:
        with PROF.timed("rpc.flush_disk"):
            return await engine.collective_rpc("flush_disk", args=([request_id], run_id, hook_dir))

    probe = getattr(engine, _FLUSH_PROBE_ATTR, None)
    if probe is None:
        probe = _dfp.DiskFlushProbe(probe=_dfp.probe_interval())
        setattr(engine, _FLUSH_PROBE_ATTR, probe)
        key = id(engine)
        if key not in _FLUSH_ANNOUNCED:
            _FLUSH_ANNOUNCED.add(key)
            print(f"[mia/aperture] {_dfp.announce(probe=probe.probe, enabled=True)}",
                  flush=True)
    d = probe.decide(cls, request_id, run_id=str(run_id), hook_dir=str(hook_dir))
    if not d.flush:
        PROF.incr("rpc.flush_disk.held")
        return None
    with PROF.timed("rpc.flush_disk"):
        res = await engine.collective_rpc("flush_disk", args=(d.ids, run_id, hook_dir))
    PROF.incr("rpc.flush_disk.probe" if d.is_probe else "rpc.flush_disk")
    if d.is_probe and probe.note_result(cls, res):
        PROF.incr("rpc.flush_disk.nonempty")
        left = probe.drain_all()
        print(f"[mia/aperture] flush_disk coalescing DISARMED: a probe found that a rank DID "
              f"write, so the eager _disk_states bucket is NOT empty on this path. Flushing "
              f"{sum(len(v) for v in left.values())} held-back request(s) now, EACH UNDER ITS OWN "
              f"run_id, and issuing one RPC per request for the rest of the run. {probe.stats()}",
              flush=True)
        for triples in left.values():
            for r_id_, run_, hook_ in triples:
                with PROF.timed("rpc.flush_disk"):
                    await engine.collective_rpc("flush_disk", args=([r_id_], run_, hook_))
    return res


def _profile_mode() -> bool:
    return os.environ.get("MIA_PROFILE_MODE") == "1"


def _aperture_route_thresholds(worker_kind: str = "hs") -> "tuple[int, int]":
    derived = int(rpc_disk_crossover_kb(worker_kind) * 1024)
    t_rpc = int(os.environ.get("MIA_ROUTER_T_RPC", derived))
    t_analyze = int(os.environ.get("MIA_ROUTER_T_ANALYZE", derived))
    return t_rpc, t_analyze


def _analyzer_reducible(analyzer_name, analyzer_spec) -> bool:
    reduce = (analyzer_spec or {}).get("reduce", "none") if isinstance(analyzer_spec, dict) else "none"
    reducible_reduce = reduce in ("mean", "norm")
    if not analyzer_name:
        return reducible_reduce
    entry = None
    try:
        entry = PluginRegistry.get_analyzer(analyzer_name)
        if entry is None:
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
    if _resolve_sink(extra) != "disk":
        return None
    return RouteDecision("disk", "none")


def _decide_aperture_route(engine, prompt, extra, max_tokens):
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
    gen_len = estimate_gen_len(max_tokens)
    try:
        kb = predict_artifact_kb("hs", gran, P, n_layers, 1, dims[2], hidden, 2, gen_len, hooks_on)
        predicted_bytes = int(kb * 1024)
        reducible = _analyzer_reducible(extra.get("analyzer"), extra.get("analyzer_spec"))
        t_rpc, t_analyze = _aperture_route_thresholds("hs")
        decision = decide_route(predicted_bytes, reducible, t_rpc, t_analyze)
        if _resolve_sink(extra) == "disk" and decision.transport != "disk":
            return RouteDecision("disk", decision.analyze_where)
        return decision
    except Exception:  # noqa: BLE001
        return _explicit_disk_route(extra)


def _decide_aperture_route_qk(engine, prompt, extra, max_tokens):
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
    gen_len = estimate_gen_len(max_tokens)
    try:
        kb = predict_artifact_kb("qk", gran, P, head_layers, 1, head_dim,
                                 H_q * head_dim, 2, gen_len, hooks_on)
        predicted_bytes = int(kb * 1024)
        reducible = _analyzer_reducible(extra.get("analyzer"), extra.get("analyzer_spec"))
        t_rpc, t_analyze = _aperture_route_thresholds("qk")
        decision = decide_route(predicted_bytes, reducible, t_rpc, t_analyze)
        if _resolve_sink(extra) == "disk" and decision.transport != "disk":
            return RouteDecision("disk", decision.analyze_where)
        if ("save_to_disk" in extra and _resolve_sink(extra) == "rpc"
                and decision.transport != "rpc"):
            return RouteDecision("rpc", decision.analyze_where)
        return decision
    except Exception:  # noqa: BLE001
        return _explicit_disk_route(extra)


def _aperture_finalize_action(route, profile_mode: bool) -> str:
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
    n = getattr(_log_analyze_deferred, "_n", 0)
    if n < 4:
        _log_analyze_deferred._n = n + 1
        print(f"[mia/aperture-router] analyze_where={action!r}: server-side CPU analyze is "
              f"deferred to Task 12; delivering RAW (client analyzes client-side)", flush=True)


def _engine_tp_size(engine) -> int:
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
    if not isinstance(extra, dict) or extra.get("qk_capture") != "score":
        return
    tp = _engine_tp_size(engine)
    if tp > 1:
        raise MiaConfigurationError(
            f"qk_capture='score' is not supported at tensor_parallel_size={tp}: every TP rank "
            f"holds only its own attention heads, so a per-head score is computed over one "
            f"rank's slice. Capture raw Q/K (qk_capture='qk', merged across ranks by MIA) or run "
            f"at tensor_parallel_size=1.")


def _engine_graph(engine) -> bool:
    for obj in (getattr(engine, "vllm_config", None),
                getattr(getattr(engine, "llm_engine", None), "vllm_config", None)):
        ac = getattr(obj, "additional_config", None) if obj is not None else None
        if isinstance(ac, dict):
            return _COMPILE_CACHE_STAMP_KEY in ac
    return _graph_mode()


def _engine_worker_kind(engine) -> str:
    """Worker kind ``engine`` was built with: cache stamp, else MIA class, else ``_worker_kind``."""
    vc = getattr(engine, "vllm_config", None)
    ac = getattr(vc, "additional_config", None)
    stamp = ac.get(_COMPILE_CACHE_STAMP_KEY) if isinstance(ac, dict) else None
    if isinstance(stamp, dict) and stamp.get("worker") in _WORKER_EXT_BY_KIND:
        return stamp["worker"]
    ext = getattr(getattr(vc, "parallel_config", None), "worker_extension_cls", None) or ""
    return _kind_from_extension(ext) or _worker_kind(ext)


def _engine_attr_dict(engine, name: str) -> dict:
    d = getattr(engine, name, None)
    if d is None:
        d = {}
        try:
            setattr(engine, name, d)
        except Exception:  # noqa: BLE001
            pass
    return d


def _steer_target(engine, steer_arg):
    cfg = _resolve_steer_config(steer_arg, os.environ.get("MIA_STEER_CONFIG"))
    if not isinstance(cfg, dict):
        return None
    path, method = cfg.get("vector_path"), cfg.get("method", "adjust_rs")
    n_layers = _hs_num_layers(getattr(engine, "llm_engine", engine))
    if not path or method not in ("add_vector", "adjust_rs") or (
            n_layers and not _parse_steer_layers(cfg.get("optimal_layer", -1), n_layers)):
        return None
    return str(path), method


def _load_steer_info(path: str):
    try:
        raw = _load_steering_vector(path)
        if "dir" not in raw:
            raise KeyError("dir")
    except Exception as e:  # noqa: BLE001
        return repr(e)
    return "avg_proj" in raw


def _steer_info(engine, path, info=None):
    loaded = _engine_attr_dict(engine, "_mia_steer_loaded")
    if info is None:
        info = loaded.get(path)
    if info is None:
        info = _load_steer_info(path)
    if not isinstance(info, str):
        loaded[path] = info
    return info


def _refuse_on_graph(engine, extra, pending=(), info=None, count=True):
    if not isinstance(extra, dict) or not (extra.get("qk_capture") == "score"
                                           or extra.get("steer")):
        return None
    if not _engine_graph(engine):
        return None
    if extra.get("qk_capture") == "score":
        raise MiaConfigurationError(
            "qk_capture='score' needs an eager engine and this one runs CUDA graphs; start it "
            "with enforce_eager=True (vllm serve: --enforce-eager).")
    target = _steer_target(engine, extra.get("steer"))
    if target is None:
        return None
    path, method = target
    info = _steer_info(engine, path, info)
    if isinstance(info, str):
        raise MiaConfigurationError(
            f"steering vector {path!r} cannot be loaded ({info}); the request would run "
            f"unsteered.")
    if method == "adjust_rs" and not info:
        raise MiaConfigurationError(
            f"adjust_rs needs avg_proj, which steering vector {path!r} lacks; the request would "
            f"run unsteered. Use method='add_vector' or a vector saved with avg_proj.")
    if path in _engine_attr_dict(engine, "_mia_steer_interned") or path in pending:
        return None
    if count:
        _check_steer_vmax(engine, path, pending)
    return path


def _check_steer_vmax(engine, path, pending=()) -> None:
    counted = (set(_engine_attr_dict(engine, "_mia_steer_interned"))
               | set(_engine_attr_dict(engine, "_mia_steer_inflight")) | set(pending))
    v_max = int(os.environ.get("MIA_STEER_VMAX", "16"))
    if path not in counted and len(counted) >= v_max:
        raise MiaConfigurationError(
            f"steering vector {path!r} would be distinct vector {len(counted) + 1} on this "
            f"engine, past MIA_STEER_VMAX={v_max}; the request would run unsteered. Start the "
            f"engine with a larger MIA_STEER_VMAX.")


async def _refuse_on_graph_serve(engine, extra):
    info = None
    if isinstance(extra, dict) and extra.get("steer") and _engine_graph(engine):
        target = _steer_target(engine, extra.get("steer"))
        if target is not None and target[0] not in _engine_attr_dict(engine,
                                                                     "_mia_steer_loaded"):
            info = await asyncio.get_running_loop().run_in_executor(
                None, _load_steer_info, target[0])
    return _refuse_on_graph(engine, extra, info=info, count=False)


def _hold_steer(engine, path):
    if path is None or path in _engine_attr_dict(engine, "_mia_steer_interned"):
        return None
    _check_steer_vmax(engine, path)
    inflight = _engine_attr_dict(engine, "_mia_steer_inflight")
    inflight[path] = inflight.get(path, 0) + 1
    return path


def _release_steer(engine, path, admitted: bool) -> None:
    if path is None:
        return
    if admitted:
        _admit_steer(engine, [path])
    inflight = _engine_attr_dict(engine, "_mia_steer_inflight")
    n = inflight.get(path, 0) - 1
    if n > 0:
        inflight[path] = n
    else:
        inflight.pop(path, None)


def _admit_steer(engine, paths) -> None:
    interned = _engine_attr_dict(engine, "_mia_steer_interned")
    for p in paths:
        interned[p] = True


async def _patched_generate(
    self,
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    effective_params = sampling_params
    if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:
        effective_params = prompt.sampling_params

    extra = dict(effective_params.extra_args or {})
    for _k in ("output_qk", "output_hidden_states", "steer"):
        if isinstance(extra.get(_k), str):
            try:
                _decoded = _json.loads(extra[_k])
            except (ValueError, TypeError):
                try:
                    _decoded = _ast.literal_eval(extra[_k])
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    continue
                if not isinstance(_decoded, (dict, list, tuple, bool, int, float)):
                    continue
            if _k == "output_qk" and isinstance(_decoded, dict):
                _decoded = {int(k): v for k, v in _decoded.items()}
            extra[_k] = _decoded
    effective_params.extra_args = extra

    wants_hs = extra.get("output_hidden_states") is not None
    wants_qk = extra.get("output_qk") is not None
    wants_steer = isinstance(extra.get("steer"), dict)
    needs_hooks = wants_hs or wants_qk or wants_steer
    save_to_disk = bool(extra.get("save_to_disk"))
    _explicit_std = bool(extra["save_to_disk"]) if "save_to_disk" in extra else None
    _refuse_unsupported_tp_request(self, extra)
    _steer_new = await _refuse_on_graph_serve(self, extra)
    _qk_keys = _sample_keys(str(request_id), int(getattr(effective_params, "n", 1) or 1))

    if "save_to_disk" not in extra:
        _routed = _maybe_storage_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _routed is not None and _routed != save_to_disk:
            save_to_disk = _routed
            extra["save_to_disk"] = _routed
            effective_params.extra_args = extra

    _warn_profile_aperture_conflict()
    _aperture_route = None
    if _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer, self) == "hs":
        _aperture_route = _decide_aperture_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _aperture_route is not None and _aperture_route.transport == "disk":
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            dest = os.path.join(hook_dir, str(run_id))
            with PROF.timed("rpc.route_aperture_to_disk"):
                await self.collective_rpc("route_aperture_to_disk", args=(request_id, dest))

    _aperture_route_qk = None
    if _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer, self) == "qk":
        _aperture_route_qk = _decide_aperture_route_qk(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _aperture_route_qk is not None and _aperture_route_qk.transport == "disk":
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            for _key in _qk_keys:
                with PROF.timed("rpc.route_aperture_to_disk"):
                    await self.collective_rpc("route_aperture_to_disk",
                                              args=(_key, _qk_dest(hook_dir, run_id, _key)))

    if (wants_qk and "qk_capture" not in extra and _engine_tp_size(self) <= 1
            and not _engine_graph(self)):
        if os.environ.get("MIA_QK_AUTO_SELECT") == "1":
            try:
                _dims = _qk_model_dims(self)
                _plen = _prompt_token_len(prompt)
                _oqk = extra.get("output_qk")
                if _dims and _plen and isinstance(_oqk, dict):
                    _pick = qk_score_size_select(
                        _plen, extra.get("hookq_mode", "all_tokens"), _oqk, *_dims)
                    extra["qk_capture"] = _pick
                    if _pick == "score":
                        extra.setdefault("score_head", 0)
                    effective_params.extra_args = extra
                    _n = getattr(_patched_generate, "_d2_log_n", 0)
                    if _n < 8:
                        _patched_generate._d2_log_n = _n + 1
                        print(f"[mia/qk-select] serve auto-select qk_capture={_pick} "
                              f"(S={_plen} mode={extra.get('hookq_mode','all_tokens')})",
                              flush=True)
            except Exception:  # noqa: BLE001
                pass

    if (
        needs_hooks
        and not getattr(self, "_mia_installed", False)
        and not _engine_graph(self)
    ):
        PROF.incr("rpc.install_hooks")
        with PROF.timed("rpc.install_hooks"):
            await self.collective_rpc("install_hooks")
        setattr(self, "_mia_installed", True)

    assert _original_generate is not None
    _hybrid = _hybrid_serves(extra, wants_hs, wants_qk, wants_steer, self)
    _gen_counts: dict = {}
    _cached_counts: dict = {}
    _delta = getattr(getattr(effective_params, "output_kind", None), "name", "") == "DELTA"
    _hook_gen_toks = 0
    _prof_capture = needs_hooks and not wants_steer and _profile_mode()
    _steer_new = _hold_steer(self, _steer_new)
    _added: list = []
    _ADDED_IDS.set(_added)
    _nonce, _start = uuid.uuid4().hex, time.time_ns()
    _writer_ends_turn = False
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            _ADDED_IDS.set(None)
            if _steer_new:
                _release_steer(self, _steer_new, admitted=True)
                _steer_new = None
            if _hybrid:
                for _o in (getattr(output, "outputs", None) or []):
                    _j, _k = getattr(_o, "index", 0), len(getattr(_o, "token_ids", None) or [])
                    _gen_counts[_j] = _gen_counts.get(_j, 0) + _k if _delta else _k
                    if getattr(_o, _SAMPLE_CACHED_ATTR, None) is not None:
                        _cached_counts[_j] = int(getattr(_o, _SAMPLE_CACHED_ATTR))
            if _prof_capture:
                for _o in (getattr(output, "outputs", None) or []):
                    _hook_gen_toks += len(getattr(_o, "token_ids", None) or [])
            if output.finished and needs_hooks and not wants_steer and _profile_mode():
                PROF.incr("request_done")
                PROF.event("request_done",
                           {"req_id": str(request_id), "boundary": "data_prepared"})
                _emit_capture_evidence(self, output, extra, wants_hs, wants_qk, _hook_gen_toks)
            elif output.finished and wants_steer:
                PROF.incr("steer.fire")
            elif output.finished and needs_hooks and not wants_steer:
                sink = _resolve_sink(extra)
                if sink == "drop":
                    pass
                elif (sink == "disk"
                      and not _takes_aperture_per_request(extra, wants_hs, wants_qk, wants_steer,
                                                        self)):
                    run_id = extra.get("run_id") or request_id
                    hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                    flushed = await _flush_disk_coalesced(
                        self, request_id=request_id, run_id=run_id, hook_dir=hook_dir,
                        wants_hs=wants_hs, wants_qk=wants_qk, wants_steer=wants_steer,
                        sink=sink, per_request=False,
                        durable_wait=bool(extra.get("durable_wait")) and not _hybrid)
                    if (extra.get("durable_wait") and not _hybrid
                            and not _all_refused(hook_dir, run_id, [request_id])):
                        with PROF.timed("disk.await_artifact"):
                            await _await_disk_artifact(
                                run_id, hook_dir, _flushed_rank_dirs(flushed, run_id, hook_dir))
                elif _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer,
                                                self) == "hs":
                    action = _aperture_finalize_action(_aperture_route, False)
                    if action in ("analyze_inflight", "analyze_from_disk"):
                        _log_analyze_deferred(action)
                        action = ("disk_raw" if (_aperture_route is not None
                                                 and _aperture_route.transport == "disk")
                                  else "rpc_raw")
                    if action == "disk_raw":
                        with PROF.timed("aperture.await_disk_confirm"):
                            await _await_aperture_disk_confirm(self, request_id)
                    else:
                        probes = await _await_aperture_per_request(
                            self, request_id, extra.get("output_hidden_states"))
                        if probes is not None:
                            output.probes = probes
                elif _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer,
                                                self) == "qk":
                    action = _aperture_finalize_action(_aperture_route_qk, False)
                    if action in ("analyze_inflight", "analyze_from_disk"):
                        _log_analyze_deferred(action)
                        action = ("disk_raw" if (_aperture_route_qk is not None
                                                 and _aperture_route_qk.transport == "disk")
                                  else "rpc_raw")
                    if _qk_ask_is_empty(extra, _hs_num_layers(self)):
                        pass
                    elif action == "disk_raw":
                        await _qk_serve_disk(self, output, extra, _qk_keys, request_id, _start)
                    else:
                        await _qk_serve_rpc(self, output, _qk_keys, _qk_mode(extra),
                                            _qk_hooks(extra))
                elif _hybrid:
                    pass
                else:
                    with PROF.timed("rpc.get_states"):
                        states = await self.collective_rpc(
                            "get_captured_states", args=(request_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    _raise_delivery_errors(request_id, parts)
                    if parts:
                        for _p in parts:
                            _reconstruct_compact_qk(_p)
                        probes = merge_probe_parts(parts)
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes
                if _hybrid and not _hs_ask_is_empty(extra, _hs_num_layers(self)):
                    _mark = _hybrid_marker(output, extra, _explicit_std, _gen_counts, _nonce,
                                           _added[0] if _added else None, _cached_counts)
                    if route_enabled():
                        output.mia = _mark
                    if _explicit_std:
                        _spawn_hybrid_writer(self, request_id, _mark, extra,
                                             _added[0] if _added else None, _start)
                        _writer_ends_turn = True
            yield output
    finally:
        if needs_hooks and not _writer_ends_turn:
            _end_completion_turn(extra, request_id)    # frees the next prompt's run write
        _release_steer(self, _steer_new, admitted=False)
        if needs_hooks and not wants_steer and not _hybrid:
            await self.collective_rpc("clear_captured_states", args=(request_id,))
            if _aperture_per_request_mode(self) and ((wants_hs and not wants_qk)
                                             or (wants_qk and not wants_hs)):
                for _key in (_qk_keys if wants_qk else [request_id]):
                    await self.collective_rpc("clear_aperture_request", args=(_key,))


async def _patched_add_request(self, request_id, *args, **kwargs):
    q = await _original_add_request(self, request_id, *args, **kwargs)
    added = _ADDED_IDS.get()
    if added is not None:
        added.append(str(getattr(q, "request_id", request_id)))
    return q


def _steer_cache_salt(extra) -> str | None:
    """Prefix-cache salt of a steered request, one per distinct steering; else None."""
    if not isinstance(extra, dict) or not extra.get("steer"):
        return None
    cfg = _resolve_steer_config(extra["steer"], os.environ.get("MIA_STEER_CONFIG"))
    if not isinstance(cfg, dict) or not cfg.get("vector_path"):
        return None
    try:
        st = os.stat(cfg["vector_path"])
        ident = [st.st_size, st.st_mtime_ns]      # a file rewritten under the same path is new
    except (OSError, TypeError, ValueError):
        ident = None
    try:
        key = [*_effective_key(cfg), cfg.get("optimal_layer", -1), *resolve_steer_modes(cfg), ident]
    except (TypeError, ValueError):
        key = [cfg, ident]                        # a config the worker cannot parse: salt it whole
    digest = hashlib.sha256(_json.dumps(key, sort_keys=True, default=repr).encode()).hexdigest()
    return f"mia-steer-{digest[:32]}"


def _patched_process_inputs(self, *args, **kwargs):
    """Wrap InputProcessor.process_inputs: salt a steered request, after any caller salt."""
    request = _original_process_inputs(self, *args, **kwargs)
    salt = _steer_cache_salt(getattr(getattr(request, "sampling_params", None), "extra_args", None))
    if salt is not None:
        request.cache_salt = f"{request.cache_salt}|{salt}" if request.cache_salt else salt
    return request


@contextlib.contextmanager
def _engine_request_ids(llm):
    eng = getattr(llm, "llm_engine", None)
    seen: dict = {}
    if eng is None or not callable(getattr(eng, "add_request", None)):
        yield seen
        return
    had = "add_request" in vars(eng)
    orig = eng.add_request

    def add_request(request_id, prompt, params, *args, **kwargs):
        internal = orig(request_id, prompt, params, *args, **kwargs)
        seen[str(request_id)] = (str(internal), params)
        return internal

    eng.add_request = add_request
    try:
        yield seen
    finally:
        if had:
            eng.add_request = orig
        else:
            del eng.add_request


def _offline_delivery_info(llm) -> dict:
    info = getattr(llm, "_mia_delivery_info", None)
    if info is not None:
        return info
    eng = getattr(llm, "llm_engine", None)
    info = {}
    if eng is not None and _engine_worker_kind(eng) == "hidden_states":
        info = merge_info(llm.collective_rpc("mia_delivery_info"))
    llm._mia_delivery_info = info
    return info


def _offline_hybrid_request(extra: dict) -> bool:
    wants_hs = extra.get("output_hidden_states") is not None
    return (wants_hs and extra.get("output_qk") is None and not extra.get("steer")
            and _resolve_sink(extra) != "drop" and not _profile_mode())


def _step_inproc_core(llm) -> None:
    eng = llm.llm_engine
    if isinstance(getattr(eng, "engine_core", None), InprocClient):
        eng.step()


def _deliver_offline(llm, outputs, engine_ids: dict) -> set:
    reqs = []
    for out in outputs:
        rec = engine_ids.get(str(out.request_id))
        if rec is None:
            continue
        internal, params = rec
        extra = getattr(params, "extra_args", None) or {}
        if _offline_hybrid_request(extra):
            reqs.append((out, internal, extra))
    if not reqs:
        return set()
    info = _offline_delivery_info(llm)
    if not info.get("roots"):
        return set()
    # lazy: graph install code stays out of import mia._plugin; readers patchable at call time
    from mia.graph.aperture_gather import GatherError, load_delivered, wait_delivered
    from mia.graph.delivered_probes import hs_probes
    from mia.graph.install_hs import DEFAULT_HOOKS_ON, DEFAULT_HS_MODE
    _step_inproc_core(llm)
    plan, exp, done = [], {}, set()
    for out, internal, extra in reqs:
        if _hs_ask_is_empty(extra, info["config"].get("num_layers")):
            done.add(str(out.request_id))
            continue
        n = max(len(out.outputs), 1)
        keys = [internal] if n == 1 else [f"{j}_{internal}" for j in range(n)]
        layers = extra.get("output_hidden_states")
        if isinstance(layers, (list, tuple)):
            exp.update({k: list(layers) for k in keys})
        plan.append((out, extra, keys))
    run_ids = run_filter(info)
    to_save = [k for _o, e, ks in plan if e.get("save_to_disk") for k in ks]
    with PROF.timed("hybrid.offline_wait"):
        rows = wait_delivered(info["roots"], to_save, expected_layers=exp or None,
                              run_ids=run_ids, load=False) if to_save else {}
    saved = [k for k in to_save if rows[k]]
    got = load_delivered(info["roots"], expected_layers=exp or None, req_ids=saved,
                         run_ids=run_ids) if saved else {}
    lost = [k for k in saved if k not in got]
    if lost:
        raise GatherError(f"requests {lost} were complete under {info['roots']} but did not read "
                          f"back")
    disk: dict = {}
    lazy: list = []
    for out, extra, keys in plan:
        metas = [{"hs_mode": extra.get("hs_mode", DEFAULT_HS_MODE),
                  "hooks_on": extra.get("hooks_on", DEFAULT_HOOKS_ON),
                  "n_prompt": len(out.prompt_token_ids or []), "n_gen": len(c.token_ids or []),
                  "n_cached": _sample_cached(out, c, len(keys)),
                  "config": info["config"]} for c in out.outputs]
        if extra.get("save_to_disk"):
            run_id = str(extra.get("run_id") or out.request_id)
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            disk.setdefault((hook_dir, run_id), []).extend(
                (k, hs_probes(got[k], m, info["names"], layout="disk") if rows[k] else None)
                for k, m in zip(keys, metas))
            continue
        targets = [(out, keys[0], metas[0])]
        if len(keys) > 1:
            targets += list(zip(out.outputs, keys, metas))
        lazy.extend(targets)
    if lazy:
        state = _defer_flush(llm, info, run_ids)
        for obj, k, m in lazy:
            _track(state, k, exp.get(k))
            _attach_delivered(obj, info, k, m, exp.get(k), run_ids, state)
    if disk:
        _save_offline_runs(disk, "hs", "hybrid.offline_save")
    return done | {str(out.request_id) for out, _e, _k in plan}


def _attach_delivered(obj, info: dict, key: str, meta: dict, layers, run_ids,
                      state=None) -> None:
    # lazy: padded_offline_probes looked up at call time so it can be patched
    from mia.graph.delivered_probes import (attach_lazy, first_pass_probes, offline_probes,
                                            padded_offline_probes, padded_reader)

    roots, names = info["roots"], info["names"]
    ask = {key: layers} if layers else None

    def rows(into=None):
        try:
            n = ag.wait_delivered(roots, [key], expected_layers=ask, run_ids=run_ids,
                                  load=False)[key]
        except ag.DeliveryTimeoutError as e:
            raise DeliveryReadTimeout(str(e), missing=e.missing, roots=e.roots) from None
        if not n:
            if state is not None:
                state["keys"].pop(key, None)
            return None
        got = ag.load_delivered(roots, expected_layers=ask, req_ids=[key], run_ids=run_ids,
                                into=into)
        if key not in got:
            raise MiaDeliveryError(f"the delivery of {key!r} is no longer under {roots}")
        if state is not None:
            state["keys"].pop(key, None)
        return got[key]

    def probes():
        into = padded_reader(meta)
        r = rows(into)
        if r is None:
            return None
        return offline_probes(r, meta, names) if into is None else padded_offline_probes(
            r, meta, names)

    def first():
        r = rows()
        return None if r is None else first_pass_probes(r, meta, names)

    try:
        attach_lazy(obj, probes, first=first)
    except TypeError:
        value = probes()
        if value is not None:
            obj.probes = value


_TRACKED_KEYS = 1 << 16


def _defer_flush(llm, info: dict, run_ids) -> dict:
    fin = getattr(llm, "_mia_finish", None)
    if fin is not None and fin.alive:
        return fin.peek()[2][1]
    state = {"roots": info["roots"], "run_ids": run_ids, "keys": {}}
    engine = getattr(llm, "llm_engine", None)
    if engine is not None:
        try:
            # Runs before vLLM's engine finalizer, which kills the gather 5 s after SIGTERM.
            llm._mia_finish = weakref.finalize(llm, _finish_deliveries, engine, state)
        except TypeError:
            pass
    return state


def _track(state: dict, key: str, layers) -> None:
    keys = state["keys"]
    keys[key] = layers
    while len(keys) > _TRACKED_KEYS:
        keys.pop(next(iter(keys)))


def _engine_dead(engine) -> bool:
    res = getattr(getattr(engine, "engine_core", None), "resources", None)
    if res is None:
        return False
    if getattr(res, "engine_dead", False):
        return True
    procs = getattr(getattr(res, "engine_manager", None), "processes", None) or ()
    return any(callable(getattr(p, "is_alive", None)) and not p.is_alive() for p in procs)


def _backlog(state: dict):
    # lazy: delivery_backlog looked up at call time so it can be patched
    from mia.graph.aperture_gather import delivery_backlog
    keys = state["keys"]
    items = list(keys.items())                   # lazy reads pop keys from other threads
    if not items:
        return [], ()
    exp = {k: v for k, v in items if v}
    pending, mark = delivery_backlog(state["roots"], [k for k, _v in items],
                                     expected_layers=exp or None, run_ids=state["run_ids"])
    for k, _v in items:
        if k not in pending:
            keys.pop(k, None)
    return pending, mark


def _abandoned(state: dict, why: str) -> None:
    try:
        pending = list(_backlog(state)[0])
    except Exception as e:  # noqa: BLE001
        pending = [f"<unknown: {e!r}>"]
    if pending:
        print(f"[mia] {why}: {len(pending)} request(s) not delivered under {state['roots']} were "
              f"abandoned: {pending[:8]}{' ...' if len(pending) > 8 else ''}", flush=True)


def _finish_deliveries(engine, state: dict) -> None:
    if threading.current_thread() is threading.main_thread():
        _finish_bounded(engine, state)
    else:
        threading.Thread(target=_finish_bounded, args=(engine, state),
                         name="mia-finish-deliveries").start()


def _finish_bounded(engine, state: dict) -> None:
    rpc = getattr(engine, "collective_rpc", None)
    if rpc is None:
        return
    if _engine_dead(engine):
        _abandoned(state, "the engine is gone at teardown")
        return
    box: dict = {}

    def call():
        try:
            rpc("flush_aperture")
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=call, name="mia-finish-rpc", daemon=True)
    th.start()
    idle = ag.delivery_timeout_s()
    start = last = time.monotonic()
    seen, why = None, None
    while True:
        th.join(1.0)
        if not th.is_alive():
            break
        now = time.monotonic()
        if _engine_dead(engine):
            why = "the engine died while finishing deliveries"
            break
        try:
            mark = _backlog(state)
        except Exception:  # noqa: BLE001
            mark = seen
        if mark != seen:
            seen, last = mark, now
        if now - start >= ag.DELIVERY_HARD_CAP_S:
            why = f"gave up finishing deliveries at the {ag.DELIVERY_HARD_CAP_S:g} s hard cap"
            break
        if now - last >= idle:
            why = f"gave up finishing deliveries after {idle:g} s with no new delivery"
            break
    if why is not None:
        _abandoned(state, why)
        return
    e = box.get("error")
    if e is None:
        return
    if type(e).__name__ == "EngineDeadError":
        _abandoned(state, "the engine is gone at teardown")
    else:
        print(f"[mia] finishing deliveries at engine teardown failed: {e!r}", flush=True)


def _save_offline_runs(disk: dict, kind: str, timer: str) -> None:
    unit, stamp = f"generate-{uuid.uuid4().hex}", time.time_ns()
    for (hook_dir, run_id), items in disk.items():
        with PROF.timed(timer):
            run_artifact.append(hook_dir, run_id, kind=kind, items=items, unit=unit, stamp=stamp)


def _offline_qk_info(llm) -> dict:
    info = getattr(llm, "_mia_qk_info", None)
    if info is not None:
        return info
    eng = getattr(llm, "llm_engine", None)
    info = {}
    if eng is not None and _engine_worker_kind(eng) == "qk":
        res = [r for r in (llm.collective_rpc("mia_delivery_info") or []) if isinstance(r, dict)]
        info = {"per_request": bool(res) and all(r.get("per_request") for r in res)}
    llm._mia_qk_info = info
    return info


def _offline_qk_request(extra: dict) -> bool:
    return (extra.get("output_qk") is not None and extra.get("output_hidden_states") is None
            and not extra.get("steer") and _resolve_sink(extra) != "drop" and not _profile_mode())


def _deliver_offline_qk(llm, outputs, engine_ids: dict) -> set:
    reqs = []
    for out in outputs:
        rec = engine_ids.get(str(out.request_id))
        if rec is None:
            continue
        internal, params = rec
        extra = getattr(params, "extra_args", None) or {}
        if _offline_qk_request(extra):
            reqs.append((out, internal, extra))
    if not reqs or not _offline_qk_info(llm).get("per_request"):
        return set()
    _step_inproc_core(llm)
    n_layers = _hs_num_layers(llm.llm_engine)
    disk: dict = {}
    refused: dict = {}
    for out, internal, extra in reqs:
        if _qk_ask_is_empty(extra, n_layers):
            continue
        outs = _outputs_by_index(out)
        keys = _sample_keys(internal, len(outs))
        n_prompt = len(out.prompt_token_ids or [])
        got, bad = [], []
        with PROF.timed("qk.offline_wait"):
            for k in keys:
                try:
                    got.append(_collect_aperture_per_request_sync(llm.collective_rpc, k,
                                                                  wait=True))
                except MiaDeliveryError as e:
                    got.append(None)
                    bad.append((k, str(e)))
        if bad:
            for k, _why in bad:
                llm.collective_rpc("clear_aperture_request", args=(k,))
            refused[str(out.request_id)] = "; ".join(why for _k, why in bad)
            continue
        layout = "disk" if extra.get("save_to_disk") else "rpc"
        probes = [qk_probes(p, n_prompt=n_prompt, n_gen=len(c.token_ids or []),
                            hookq_mode=_qk_mode(extra), hooks_on=_qk_hooks(extra), layout=layout)
                  for p, c in zip(got, outs)]
        if layout == "disk":
            run_id = str(extra.get("run_id") or out.request_id)
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            disk.setdefault((hook_dir, run_id), []).extend(zip(keys, probes))
            continue
        if probes and probes[0] is not None:
            out.probes = probes[0]
        if len(keys) > 1:
            for c, p in zip(outs, probes):
                if p is not None:
                    c.probes = p
    if disk:
        _save_offline_runs(disk, "qk", "qk.offline_save")
    _raise_refused(refused)
    return {str(out.request_id) for out, _i, _e in reqs}


def _raise_refused(refused: dict) -> None:
    if refused:
        raise MiaDeliveryError(
            f"Q/K capture refused for {len(refused)} request(s): "
            + "; ".join(f"{r!r}: {why}" for r, why in refused.items()))


def _patched_llm_generate(self, prompts: Any, sampling_params: Any = None, **kwargs) -> list:
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
    _steer_new: list = []
    for _sp in params_list:
        _refuse_unsupported_tp_request(self, _sp.extra_args or {})
        _v = _refuse_on_graph(self, _sp.extra_args or {}, _steer_new)
        if _v:
            _steer_new.append(_v)

    if (needs_hooks and os.environ.get("MIA_STORAGE_ROUTER") == "1"
            and not getattr(_patched_llm_generate, "_router_warned", False)):
        _patched_llm_generate._router_warned = True
        print("[mia] MIA_STORAGE_ROUTER is serve-only; the offline "
              "LLM.generate path honors each request's explicit save_to_disk.",
              flush=True)

    if (
        needs_hooks
        and not getattr(self, "_mia_installed", False)
        and not _engine_graph(self)
    ):
        PROF.incr("rpc.install_hooks")
        with PROF.timed("rpc.install_hooks"):
            self.collective_rpc("install_hooks")
        self._mia_installed = True

    assert _original_llm_generate is not None
    with _engine_request_ids(self) as _engine_ids:
        outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)
    _admit_steer(self, _steer_new)

    if needs_hooks:

        _hybrid_done = _deliver_offline(self, outputs, _engine_ids)
        _hybrid_done |= _deliver_offline_qk(self, outputs, _engine_ids)

        disk_by_run: dict = {}
        refused: dict = {}

        for idx, output in enumerate(outputs):
            req_id = output.request_id
            if str(req_id) in _hybrid_done:
                continue
            sp = params_list[idx] if idx < len(params_list) else params_list[0] if params_list else None
            extra = (sp.extra_args if sp is not None else None) or {}

            wants_artifacts = extra.get("output_hidden_states") is not None or extra.get("output_qk") is not None
            if extra.get("save_to_disk"):
                run_id = extra.get("run_id") or req_id
                hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                disk_by_run.setdefault(run_id, []).append((req_id, hook_dir))
            elif wants_artifacts:
                if (_aperture_per_request_mode(self)
                        and extra.get("output_hidden_states") is not None
                        and extra.get("output_qk") is None):
                    probes = _collect_aperture_per_request_sync(
                        self.collective_rpc, req_id, extra.get("output_hidden_states"))
                    if probes is not None:
                        output.probes = probes
                else:
                    with PROF.timed("rpc.get_states"):
                        states = self.collective_rpc("get_captured_states", args=(req_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    errs = [p["mia_error"] for p in parts
                            if isinstance(p, dict) and "mia_error" in p]
                    if errs:
                        refused[str(req_id)] = "; ".join(errs)
                        continue
                    if parts:
                        for _p in parts:
                            _reconstruct_compact_qk(_p)
                        probes = merge_probe_parts(parts)
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes

        _graph = _engine_graph(self)
        if disk_by_run and _graph:
            _warn_graph_save_unwritten(disk_by_run)
        if disk_by_run and not _graph:
            flushed_by_run: dict = {}
            for run_id, req_list in disk_by_run.items():
                req_ids = [r for r, _ in req_list]
                _, hook_dir = req_list[0]
                with PROF.timed("rpc.flush_disk"):
                    flushed_by_run[run_id] = self.collective_rpc(
                        "flush_disk", args=(req_ids, run_id, hook_dir))
            for run_id, req_list in disk_by_run.items():
                _, hook_dir = req_list[0]
                if not _all_refused(hook_dir, run_id, [r for r, _h in req_list]):
                    with PROF.timed("disk.await_artifact"):
                        landed = _wait_disk_artifact(run_id, hook_dir, _flushed_rank_dirs(
                            flushed_by_run.get(run_id), run_id, hook_dir))
                    if not landed:
                        PROF.incr("disk.await_artifact.timeout")
                gone = read_refused_qk(hook_dir, str(run_id))
                for r, _h in req_list:
                    hits = match_internal_ids(gone, str(r))
                    if hits:
                        refused[str(r)] = "; ".join(gone[h] for h in hits)
        _raise_refused(refused)

    return outputs


def _warn_graph_save_unwritten(runs) -> None:
    """One warning per call: save_to_disk on a graph engine that no delivery path writes."""
    names = ", ".join(repr(str(r)) for r in runs)
    for key, off in ((ds.MODE_ENV, lambda v: v.lower() in ("drain", "off")),
                     (ds.PROFILE_ENV, lambda v: v == "1"),
                     (ds.MMAP_ENV, lambda v: v != "0"),
                     (ds.WRITE_MODE_ENV, lambda v: v.lower() == "legacy"),
                     (ds.SINK_ENV, lambda v: v.lower() == "drop")):
        v = (os.environ.get(key) or "").strip()
        if v and off(v):
            why = f"{key}={v} turns its delivery off; unset it, or start the engine"
            break
    else:
        why = "delivers nothing for these requests; start the engine"
    print(f"[mia] WARNING: save_to_disk=True wrote nothing for run {names}: this engine runs CUDA "
          f"graphs and {why} with enforce_eager=True.", flush=True)


def _attach_choice_probes(response, outs) -> None:
    choices = list(getattr(response, "choices", None) or [])
    if len(choices) != len(outs):
        return
    for ch, o in zip(choices, outs):
        p = getattr(o, "probes", None)
        if p is not None:
            setattr(ch, "probes", _serialize_probes(p))


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        probes = getattr(res, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)
            break
    _attach_choice_probes(response, [o for res in final_res_batch or ()
                                     for o in (getattr(res, "outputs", None) or [])])
    marks = [getattr(res, "mia", None) for res in final_res_batch or ()]
    if marks and all(m is not None for m in marks) and hasattr(response, "model_dump"):
        response.mia = {"delivery": "hybrid", "kind": "completion",
                        "items": [{**m, "item": i} for i, m in enumerate(marks)]}
    return response


async def _patched_chat_full_generator(self, request, result_generator, *args, **kwargs):
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
        _attach_choice_probes(response, getattr(last_output, "outputs", None) or [])
        mark = getattr(last_output, "mia", None)
        if mark is not None:
            response.mia = {"delivery": "hybrid", "kind": "chat", "items": [{**mark, "item": 0}]}

    return response


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup."""
    global _original_create_engine_config
    global _original_generate, _original_add_request, _original_llm_generate
    global _original_process_inputs
    global _original_completion_response, _original_chat_full_generator

    _apply_delivery_selector()

    # lazy: AsyncLLM, InputProcessor and graph install code load only when register() runs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.input_processor import InputProcessor

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    try:
        from mia.graph.install import patch_worker_load_model
        patch_worker_load_model()
    except Exception as e:  # noqa: BLE001
        print(f"[mia] graph load_model patch unavailable ({e}); "
              f"eager path unaffected.")

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate
    _original_add_request = AsyncLLM.add_request
    AsyncLLM.add_request = _patched_add_request
    _original_process_inputs = InputProcessor.process_inputs
    InputProcessor.process_inputs = _patched_process_inputs

    try:
        _patch_sample_cached()
    except Exception as e:  # noqa: BLE001
        print(f"[mia] per-sample prefix-cache counts unavailable ({e!r}); n>1 hidden-state "
              f"passes are split by row count.", flush=True)

    try:
        patch_app_builder()
    except Exception as e:  # noqa: BLE001
        print(f"[mia] delivered-data read route unavailable ({e!r}); MiaClient cannot read "
              f"hybrid captures from this server.", flush=True)

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    for _completion_module in (
        "vllm.entrypoints.openai.completion.serving",
        "vllm.entrypoints.openai.serving_completion",
    ):
        try:
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
        "vllm.entrypoints.openai.chat_completion.serving",
        "vllm.entrypoints.openai.serving_chat",
    ):
        try:
            _mod = _il.import_module(_chat_module)
            _cls = _mod.OpenAIServingChat
            _original_chat_full_generator = _cls.chat_completion_full_generator
            _cls.chat_completion_full_generator = _patched_chat_full_generator
            break
        except Exception:
            pass

