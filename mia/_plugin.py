"""vLLM plugin entry point: patches the engine, runner and serve path to arm MIA's hooks."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from collections.abc import AsyncIterator, Callable
from typing import Any

import zstandard as zstd

from mia._profiler import PROF
from mia.errors import MiaConfigurationError, MiaRefusal

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None

_WORKER_EXT_HS = "mia.workers.hs_capture_worker.HSCaptureWorker"
_WORKER_EXT_QK = "mia.workers.qk_capture_worker.QKCaptureWorker"
_WORKER_EXT_STEER = "mia.workers.steer_worker.SteerWorker"

MIA_WORKER_VALUES = ("hidden_states", "qk", "steer")
_WORKER_EXT_BY_KIND = {
    "hidden_states": _WORKER_EXT_HS,
    "qk": _WORKER_EXT_QK,
    "steer": _WORKER_EXT_STEER,
}
DEFAULT_MIA_WORKER = "hidden_states"


class UnknownMiaWorkerError(MiaRefusal, ValueError):
    """MIA_WORKER was set to something that is not one of MIA_WORKER_VALUES."""


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

_DEFAULT_HOOK_DIR = "/dev/shm/mia"


def _graph_mode() -> bool:
    from mia.graph.install import graph_mode_enabled
    return graph_mode_enabled()


def _decompress(data: bytes) -> Any:
    PROF.gauge("rpc.payload_bytes", len(data))
    with PROF.timed("rpc.decompress"):
        if data[:4] == _ZSTD_MAGIC:
            return pickle.loads(_ZSTD_DECOMPRESSOR.decompress(data))
        return pickle.loads(data)


def _aperture_per_request_mode() -> bool:
    return (os.environ.get("MIA_APERTURE_PER_REQUEST") == "1"
            and os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1")


def _aperture_per_request_kind(extra: dict, wants_hs: bool, wants_qk: bool,
                               wants_steer: bool):
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
    return _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) is not None


def _reconstruct_compact_qk(probes: dict) -> None:
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
    for entry in probes.get(key, {}).values():
        for tkey in ("hidden_states", "q", "k_all"):
            t = entry.get(tkey)
            if t is None or isinstance(t, list):
                continue
            if t.dim() == 3 and t.shape[1] > expected_len:
                PROF.incr("trim.event")
                entry[tkey] = t[:, :expected_len, :]


def _stable_artifact_files(run_dir: str) -> list:
    import glob
    try:
        return sorted(
            f for f in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)
            if os.path.isfile(f) and not f.endswith(".tmp") and os.path.getsize(f) > 0
        )
    except OSError:
        return []


_ARTIFACT_WAIT_S = 10.0
_ARTIFACT_POLL_S = 0.005


def _flushed_rank_dirs(results, run_id: str, hook_dir: str):
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
    if not rank_dirs:
        files = _stable_artifact_files(run_dir)
        return files, bool(files)
    per_rank = [_stable_artifact_files(d) for d in rank_dirs]
    return [f for fs in per_rank for f in fs], all(per_rank)


def _log_barrier_timeout(run_id: str, rank_dirs) -> None:
    if not rank_dirs:
        print(f"[mia/disk] durability barrier TIMEOUT after {_ARTIFACT_WAIT_S:.0f}s for "
              f"run_id {run_id!r}: no artifact landed. A loader will raise FileNotFoundError "
              f"for this run_id -- the write did not finish, the run_id is not wrong.", flush=True)
        return
    missing = [os.path.basename(d) for d in rank_dirs if not _stable_artifact_files(d)]
    if missing:
        print(f"[mia/disk] durability barrier TIMEOUT after {_ARTIFACT_WAIT_S:.0f}s for "
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
    import asyncio
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


_APERTURE_DELIVER_POLL_S = 0.005


def _aperture_deliver_timeout_s() -> float:
    try:
        return max(0.1, float(os.environ.get("MIA_APERTURE_DELIVER_TIMEOUT_S", "30") or "30"))
    except (TypeError, ValueError):
        return 30.0


async def _await_aperture_per_request(engine, request_id, hs_layers=None):
    import asyncio
    import time
    timeout = _aperture_deliver_timeout_s()
    deadline = time.monotonic() + timeout
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
            print(f"[mia/aperture] BLOCK-UNTIL-HELD TIMEOUT after {timeout:.1f}s waiting for RPC "
                  f"per-request delivery of {request_id!r} ({len(collected)} rank part(s) held); "
                  f"leaving probes unset (raise MIA_APERTURE_DELIVER_TIMEOUT_S if the off-loop "
                  f"consumer is merely slow, else it is a bug)", flush=True)
            return None
        await asyncio.sleep(_APERTURE_DELIVER_POLL_S)


def _collect_aperture_per_request_sync(rpc, request_id, hs_layers=None):
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
    """Merge per-worker probe results from a retrieval collective_rpc into one dict."""
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
    import asyncio
    import time
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
        import vllm
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
    from mia.graph.aperture_sizing import parse_autocap_setting
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
        from mia.graph.tp_shard import qk_shard
        shard = qk_shard(0, tp, h_q, h_kv, head_dim)
        dims["n_q_heads"] = shard.num_local_q_heads
        dims["n_kv_heads"] = shard.num_local_kv_heads
        dims["head_dim"] = head_dim
        dims["tp"] = tp
    return dims


def _hs_layers_per_rank(config, n_layers: int) -> int:
    from mia.graph.tp_shard import hs_max_owned_layers, resolve_hs_shard_mode
    tp = int(getattr(getattr(config, "parallel_config", None), "tensor_parallel_size", 1) or 1)
    return hs_max_owned_layers(int(n_layers), tp, resolve_hs_shard_mode(tp))


def _derive_safe_max_batched_tokens(config, worker_kinds):
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
        from mia.graph.aperture_sizing import CAPTURE_SUBSYSTEMS
        if not (subsystems & set(CAPTURE_SUBSYSTEMS)):
            return
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
    except Exception as e:  # noqa: BLE001
        print(f"[mia] autocap SKIPPED (non-fatal): {e!r}", flush=True)


_COMPILE_CACHE_STAMP_KEY = "mia_graph_capture"


def _mia_source_id() -> str:
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
        from mia.graph.tp_shard import (
            HS_ALL_RANKS_ENV, HS_LAYER_SHARD_RULE, HS_MODE_ALL_RANKS, HS_MODE_RANK0,
            HS_MODE_ROUND_ROBIN, HS_SHARD_ENV, resolve_hs_shard_mode)
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
    import json as _json
    print(f"[mia] compile-cache key stamped with worker={worker_kind} mia={stamp['mia']} "
          f"layout={_json.dumps(stamp['layout'], sort_keys=True)} "
          f"(worker_extension_cls is excluded from vLLM's own key)", flush=True)


class UnsupportedGraphModeError(RuntimeError):
    """MIA supports eager (NONE) and FULL CUDA graphs."""


def validate_graph_mode(mode_name: str) -> None:
    """Accept only the two modes MIA's capture path is validated for."""
    if str(mode_name).upper() not in {"NONE", "FULL"}:
        raise UnsupportedGraphModeError(
            f"MIA supports cudagraph_mode NONE (eager) and FULL; got {mode_name}. "
            f"vLLM 0.29 defaults to FULL_AND_PIECEWISE, so this is expected on a default "
            f"engine: pass compilation_config={{'cudagraph_mode': 'FULL'}} for graph capture, "
            f"or enforce_eager=True for the eager path."
        )


def validate_v2_runner_selected(config) -> None:
    """Refuse a config that will build vLLM's V1 model runner."""
    try:
        selected = getattr(config, "use_v2_model_runner")
    except Exception:  # noqa: BLE001
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
    if not self.worker_extension_cls:
        worker_type = parse_mia_worker_env(os.environ.get("MIA_WORKER")) or DEFAULT_MIA_WORKER
        self.worker_extension_cls = _WORKER_EXT_BY_KIND[worker_type]
    _wkind = _worker_kind(self.worker_extension_cls)

    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0":
        raise RuntimeError(
            "MIA requires vLLM's V2 model runner, but VLLM_USE_V2_MODEL_RUNNER=0 is set. "
            "Unset it (0.29 defaults to V2) or set it to 1."
        )

    from mia.graph.tp_shard import refuse_pipeline_parallel, resolve_hs_shard_mode
    refuse_pipeline_parallel(getattr(self, "pipeline_parallel_size", 1), "engine arguments")

    resolve_hs_shard_mode(int(getattr(self, "tensor_parallel_size", 1) or 1))

    _note_vllm_version()

    graph_mode = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"
    if graph_mode:
        from mia.graph.install import set_graph_mode
        set_graph_mode(True)
        stamp_compile_cache_key(self, _wkind)
    else:
        self.enforce_eager = True

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

    validate_graph_mode(_cudagraph_mode_name(config.compilation_config.cudagraph_mode))
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


def _profile_mode() -> bool:
    return os.environ.get("MIA_PROFILE_MODE") == "1"


def _aperture_route_thresholds(worker_kind: str = "hs") -> "tuple[int, int]":
    from mia.run_utils import rpc_disk_crossover_kb
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
    if _resolve_sink(extra) != "disk":
        return None
    from mia.graph.delivery_router import RouteDecision
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
        if _resolve_sink(extra) == "disk" and decision.transport != "disk":
            from mia.graph.delivery_router import RouteDecision
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
        if _resolve_sink(extra) == "disk" and decision.transport != "disk":
            from mia.graph.delivery_router import RouteDecision
            return RouteDecision("disk", decision.analyze_where)
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


async def _patched_generate(
    self,
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:
            effective_params = prompt.sampling_params
    except ImportError:
        pass

    extra = dict(effective_params.extra_args or {})
    import json as _json
    for _k in ("output_qk", "output_hidden_states", "steer"):
        if isinstance(extra.get(_k), str):
            try:
                _decoded = _json.loads(extra[_k])
            except (ValueError, TypeError):
                try:
                    import ast as _ast
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
    _refuse_unsupported_tp_request(self, extra)

    if "save_to_disk" not in extra:
        _routed = _maybe_storage_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _routed is not None and _routed != save_to_disk:
            save_to_disk = _routed
            extra["save_to_disk"] = _routed
            effective_params.extra_args = extra

    _warn_profile_aperture_conflict()
    _aperture_route = None
    if _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "hs":
        _aperture_route = _decide_aperture_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _aperture_route is not None and _aperture_route.transport == "disk":
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            dest = os.path.join(hook_dir, str(run_id))
            with PROF.timed("rpc.route_aperture_to_disk"):
                await self.collective_rpc("route_aperture_to_disk", args=(request_id, dest))

    _aperture_route_qk = None
    if _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "qk":
        _aperture_route_qk = _decide_aperture_route_qk(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _aperture_route_qk is not None and _aperture_route_qk.transport == "disk":
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            dest = os.path.join(hook_dir, str(run_id))
            with PROF.timed("rpc.route_aperture_to_disk"):
                await self.collective_rpc("route_aperture_to_disk", args=(request_id, dest))

    if wants_qk and "qk_capture" not in extra and _engine_tp_size(self) <= 1:
        if os.environ.get("MIA_QK_AUTO_SELECT") == "1":
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
                        print(f"[mia/qk-select] serve auto-select qk_capture={_pick} "
                              f"(S={_plen} mode={extra.get('hookq_mode','all_tokens')})",
                              flush=True)
            except Exception:  # noqa: BLE001
                pass

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
                      and not _takes_aperture_per_request(extra, wants_hs, wants_qk, wants_steer)):
                    run_id = extra.get("run_id") or request_id
                    hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                    with PROF.timed("rpc.flush_disk"):
                        flushed = await self.collective_rpc(
                            "flush_disk", args=([request_id], run_id, hook_dir))
                    if extra.get("durable_wait"):
                        with PROF.timed("disk.await_artifact"):
                            await _await_disk_artifact(
                                run_id, hook_dir, _flushed_rank_dirs(flushed, run_id, hook_dir))
                elif _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "hs":
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
                elif _aperture_per_request_kind(extra, wants_hs, wants_qk, wants_steer) == "qk":
                    action = _aperture_finalize_action(_aperture_route_qk, False)
                    if action in ("analyze_inflight", "analyze_from_disk"):
                        _log_analyze_deferred(action)
                        action = ("disk_raw" if (_aperture_route_qk is not None
                                                 and _aperture_route_qk.transport == "disk")
                                  else "rpc_raw")
                    if action == "disk_raw":
                        with PROF.timed("aperture.await_disk_confirm"):
                            await _await_aperture_disk_confirm(self, request_id)
                    else:
                        probes = await _await_aperture_per_request(self, request_id)
                        if probes is not None:
                            output.probes = probes
                else:
                    with PROF.timed("rpc.get_states"):
                        states = await self.collective_rpc(
                            "get_captured_states", args=(request_id,))
                    parts = [_decompress(s) for s in states if s is not None]
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
            yield output
    finally:
        if needs_hooks and not wants_steer:
            await self.collective_rpc("clear_captured_states", args=(request_id,))
            if _aperture_per_request_mode() and ((wants_hs and not wants_qk)
                                             or (wants_qk and not wants_hs)):
                await self.collective_rpc("clear_aperture_request", args=(request_id,))


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
    for _sp in params_list:
        _refuse_unsupported_tp_request(self, _sp.extra_args or {})

    if (needs_hooks and os.environ.get("MIA_STORAGE_ROUTER") == "1"
            and not getattr(_patched_llm_generate, "_router_warned", False)):
        _patched_llm_generate._router_warned = True
        print("[mia] MIA_STORAGE_ROUTER is serve-only; the offline "
              "LLM.generate path honors each request's explicit save_to_disk.",
              flush=True)

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

        disk_by_run: dict = {}

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
                            _reconstruct_compact_qk(_p)
                        probes = merge_probe_parts(parts)
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes

        if disk_by_run and not _graph_mode():
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
                    landed = _wait_disk_artifact(run_id, hook_dir, _flushed_rank_dirs(
                        flushed_by_run.get(run_id), run_id, hook_dir))
                if not landed:
                    PROF.incr("disk.await_artifact.timeout")

    return outputs


def _serialize_probes(probes: dict) -> dict:
    import torch
    PROF.incr("serve.serialize_probes.calls")
    with PROF.timed("serve.serialize_probes"):
        result = {}
        n_tensors = 0
        n_elems = 0
        for key, cache in probes.items():
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
                        n_tensors += len(v)
                        n_elems += sum(t.numel() for t in v)
                        new_entry[k] = [t.tolist() for t in v]
                    else:
                        new_entry[k] = v
                result[key][mod_name] = new_entry
        PROF.gauge("serve.serialize_probes.tensors", n_tensors)
        PROF.gauge("serve.serialize_probes.elements", n_elems)
    return result


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        probes = getattr(res, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)
            break
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

    return response


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup."""
    global _original_create_engine_config
    global _original_generate, _original_llm_generate
    global _original_completion_response, _original_chat_full_generator

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

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

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    for _completion_module in (
        "vllm.entrypoints.openai.completion.serving",
        "vllm.entrypoints.openai.serving_completion",
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
        "vllm.entrypoints.openai.chat_completion.serving",
        "vllm.entrypoints.openai.serving_chat",
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

