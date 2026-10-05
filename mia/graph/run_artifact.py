"""Eager-format run artifacts written from delivered data."""
from __future__ import annotations

import contextlib
import fcntl
import glob
import json
import logging
import math
import os
import shutil
import socket
import tempfile
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open

from mia.errors import MiaDeliveryError
from mia.graph.artifact_writer import write_artifact
from mia.graph.delivered_probes import merge_disk
from mia.graph.tp_shard import rank_dir_name
from mia.run_utils import QK_REFUSED_FILE, load_and_merge_hs_cache

RUN_FORMAT = "mia-run-v1"
RUN_MANIFEST = "mia_run.json"
LOCK_NAME = ".mia_run.lock"
WAIT_ENV = "MIA_ARTIFACT_WAIT_S"
DEFAULT_WAIT_S = 10.0
_KINDS = {"hs": ("hidden_states.pt", "hidden_states.safetensors"),
          "qk": ("qk.pt", "qk.safetensors")}
_MODE_KEY = {"hs": ("hs_cache", "hs_mode", "last_token"),
             "qk": ("qk_cache", "hookq_mode", "all_tokens")}

_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
logger = logging.getLogger(__name__)


class RunArtifactError(MiaDeliveryError):
    """A run artifact cannot be extended or read as asked."""


def run_dir(hook_dir: str, run_id: str) -> str:
    return os.path.join(hook_dir, str(run_id), rank_dir_name(0))


def artifact_wait_s() -> float:
    v = os.environ.get(WAIT_ENV)
    if v in (None, ""):
        return DEFAULT_WAIT_S
    try:
        s = float(v)
    except ValueError:
        s = math.nan
    if math.isnan(s):
        raise RunArtifactError(f"{WAIT_ENV}={v!r} is not a number of seconds")
    return max(s, 0.0)  # <= 0: no wait


def read_manifest(hook_dir: str, run_id: str) -> Optional[dict]:
    p = os.path.join(run_dir(hook_dir, run_id), RUN_MANIFEST)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _proc_lock(path: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


def _write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _use_safetensors() -> bool:
    return os.environ.get("MIA_USE_SAFETENSORS", "0") == "1"


def _load_qk_compact(d: str) -> dict:
    pt = os.path.join(d, _KINDS["qk"][0])
    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu")
    with open(os.path.join(d, "qk.json"), encoding="utf-8") as f:
        meta = json.load(f)
    out: dict = {"config": meta["config"], "qk_cache": {}}

    def split(t, lens):
        return list(torch.split(t, [int(n) for n in lens]))

    with safe_open(os.path.join(d, _KINDS["qk"][1]), framework="pt", device="cpu") as sf:
        for item in meta["layer_order"]:
            mode = item.get("hookq_mode", meta.get("hookq_mode", "all_tokens"))
            t_q, t_k = sf.get_tensor(item["key_q"]), sf.get_tensor(item["key_k"])
            if meta.get("compact_all_tokens") and mode == "all_tokens":
                q = split(t_q, meta["q_seq_lens"])
                k_full = split(t_k, meta["k_full_lens"])
                ends = [[int(L) for L in e] for e in meta["k_prefix_ends"]]
            else:
                k_lens = meta.get("k_seq_lens") or [t_k.shape[1]] * t_k.shape[0]
                k_full = [t_k[i, :int(n)] for i, n in enumerate(k_lens)]
                ends = [[int(n)] for n in k_lens]
                q_lens = meta.get("q_seq_lens")
                q = ([t_q[i, :int(n)] for i, n in enumerate(q_lens)]
                     if mode == "all_tokens" and q_lens else [t_q[i] for i in range(t_q.shape[0])])
            out["qk_cache"][item["module_name"]] = {
                "q": q, "layer_num": item["layer_num"], "hookq_mode": mode,
                "k_full": k_full, "k_prefix_ends": ends}
    return out


def _load_current(kind: str, hook_dir: str, run_id: str) -> dict:
    if kind == "hs":
        return load_and_merge_hs_cache(hook_dir, str(run_id))
    if kind == "qk":
        return _load_qk_compact(run_dir(hook_dir, run_id))
    raise RunArtifactError(f"no run artifact for kind {kind!r}")


def _write(kind: str, cache: dict, d: str, mode: str) -> None:
    if kind in _KINDS:
        write_artifact(kind, cache, d, mode, 0, _use_safetensors(), False, _KINDS[kind][0])
        return
    raise RunArtifactError(f"no run artifact for kind {kind!r}")


def _kind_files(hook_dir: str, run_id: str, kind: str) -> List[str]:
    base = _KINDS[kind][0].rsplit(".", 1)[0]
    names = [f"{base}.pt", f"{base}.safetensors", f"{base}.json"]
    if kind == "qk":
        names.append(QK_REFUSED_FILE)
    root = os.path.join(hook_dir, str(run_id))
    return sorted(p for n in names for p in glob.glob(os.path.join(root, "**", n), recursive=True))


def _ident(path: str):
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns)


def append(hook_dir: str, run_id: str, *, kind: str, items: Iterable[Tuple[str, dict]],
           unit: Optional[str] = None, stamp: Optional[int] = None,
           nonce: Optional[str] = None, start: Optional[int] = None) -> List[str]:
    """Add ``items`` to the run in order; returns its keys."""
    if kind not in _KINDS:
        raise RunArtifactError(f"no run artifact for kind {kind!r}")
    items = list(items)
    d = run_dir(hook_dir, run_id)
    os.makedirs(d, exist_ok=True)
    man_path = os.path.join(d, RUN_MANIFEST)
    with _proc_lock(os.path.realpath(d)):
        with open(os.path.join(d, LOCK_NAME), "a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                man = read_manifest(hook_dir, run_id)
                present = [n for n in _KINDS[kind] if os.path.exists(os.path.join(d, n))]
                was = (man or {}).get("nonces") or {}
                reused = nonce is not None and any(was.get(k, nonce) != nonce for k, _ in items)
                same = man is not None and man.get("kind") == kind and man.get("unit") == unit
                if unit is not None and (reused or not same):
                    newer = man.get("stamp") if man and man.get("kind") == kind else None
                    if stamp is not None and newer is not None and int(newer) > int(stamp):
                        logger.warning("run %r under %s: a newer write replaced it; %s is not "
                                       "written", run_id, hook_dir, [k for k, _ in items])
                        return list(man.get("requests") or [])
                    man, present = None, []
                    stale = {p: _ident(p) for p in _kind_files(hook_dir, run_id, kind)}
                elif (unit is not None and stamp is not None and man.get("start") is not None
                      and int(stamp) < int(man["start"])):
                    logger.warning("run %r under %s: %s finished before the run's requests "
                                   "started (an older use of a reused id); not written", run_id,
                                   hook_dir, [k for k, _ in items])
                    return list(man.get("requests") or [])
                else:
                    stale = {}
                if man is None and unit is not None:
                    man = {"format": RUN_FORMAT, "kind": kind, "requests": []}
                elif man is None:
                    if present:
                        raise RunArtifactError(
                            f"{os.path.join(d, present[0])} was not written by a delivery read "
                            f"(no {RUN_MANIFEST}); refusing to extend it")
                    man = {"format": RUN_FORMAT, "kind": kind, "requests": []}
                elif man.get("format") != RUN_FORMAT or man.get("kind") != kind:
                    raise RunArtifactError(f"{man_path} is not a {kind!r} {RUN_FORMAT} manifest")
                keys = list(man["requests"])
                empty = list(man.get("empty") or [])
                held = [k for k in keys if k not in set(empty)]
                if held and not present:
                    raise RunArtifactError(
                        f"run {run_id!r} lists request(s) {held} but its artifact is gone from "
                        f"{d}; refusing to restart it")
                dup = sorted({k for k, _ in items} & set(keys))
                seen: set = set()
                for k, _ in items:
                    if k in seen:
                        dup.append(k)
                    seen.add(k)
                if dup:
                    raise RunArtifactError(f"run {run_id!r} already holds request(s) {dup}")
                data = [p for _k, p in items if p is not None]
                cache = _load_current(kind, hook_dir, run_id) if data and present else None
                ckey, mkey, mode0 = _MODE_KEY[kind]
                mode = None
                for k, probes in items:
                    keys.append(str(k))
                    if probes is None:
                        empty.append(str(k))
                        continue
                    cache = merge_disk(cache, probes)
                    for e in probes[ckey].values():
                        mode = mode or e.get(mkey)
                if data:
                    _write(kind, cache, d, mode or mode0)
                for p, ident in stale.items():
                    if _ident(p) == ident:
                        os.remove(p)
                man["requests"] = keys
                if empty:
                    man["empty"] = empty
                if unit is not None:
                    man["unit"] = str(unit)
                    if stamp is not None:
                        man["stamp"] = max(int(stamp), int(man.get("stamp") or 0))
                    began = start if start is not None else stamp
                    if began is not None:
                        man["start"] = min(int(began), int(man.get("start") or began))
                if nonce is not None:
                    man["nonces"] = {**(man.get("nonces") or {}),
                                     **{str(k): str(nonce) for k, _ in items}}
                _write_json(man_path, man)
                return keys
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def read_lock(hook_dir: str, run_ids: Iterable[str]):
    """Shared lock on each run while it is read."""
    fds = []
    try:
        for rid in sorted(set(str(r) for r in run_ids)):
            try:
                fd = os.open(os.path.join(run_dir(hook_dir, rid), LOCK_NAME),
                             os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            except FileNotFoundError:
                continue
            fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        for fd in fds:
            os.close(fd)


SNAPSHOT_PREFIX = ".mia_read_"
_IN_PLACE_NOTED: set = set()


def _link_file(src: str, dst: str) -> None:
    os.link(src, dst)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sweep_snapshots(hook_dir: str, prefix: str) -> None:
    try:
        names = os.listdir(hook_dir)
    except OSError:
        return
    for n in names:
        if n.startswith(prefix):
            pid = n[len(prefix):].split("_", 1)[0]
            if pid.isdigit() and int(pid) != os.getpid() and not _alive(int(pid)):
                shutil.rmtree(os.path.join(hook_dir, n), ignore_errors=True)


def _note_in_place(hook_dir: str, why) -> None:
    if hook_dir in _IN_PLACE_NOTED:
        return
    _IN_PLACE_NOTED.add(hook_dir)
    print(f"[mia] {hook_dir}: no snapshot ({why}); runs are analysed in place under their "
          f"shared lock", flush=True)


def snapshot(hook_dir: str, run_ids: Iterable[str]) -> Optional[str]:
    """A hook dir of hard links to each run's artifact files, taken under :func:`read_lock`."""
    prefix = f"{SNAPSHOT_PREFIX}{socket.gethostname().split('.', 1)[0]}_"
    _sweep_snapshots(hook_dir, prefix)
    try:
        snap = tempfile.mkdtemp(prefix=f"{prefix}{os.getpid()}_", dir=hook_dir)
    except OSError as e:
        _note_in_place(hook_dir, e)
        return None
    try:
        for rid in set(str(r) for r in run_ids):
            for d, dirs, files in os.walk(os.path.join(hook_dir, rid)):
                dirs[:] = [x for x in dirs if not x.startswith(".")]
                rel = os.path.relpath(d, hook_dir)
                os.makedirs(os.path.join(snap, rel), exist_ok=True)
                for f in files:
                    if not f.startswith("."):
                        _link_file(os.path.join(d, f), os.path.join(snap, rel, f))
    except OSError as e:
        shutil.rmtree(snap, ignore_errors=True)
        _note_in_place(hook_dir, e)
        return None
    except BaseException:
        shutil.rmtree(snap, ignore_errors=True)
        raise
    return snap


def holds(man: Optional[dict], keys: Iterable[str],
          nonces: Optional[Dict[str, str]] = None) -> bool:
    """The manifest lists ``keys``; with ``nonces``, exactly those keys, each under its nonce."""
    want = set(str(k) for k in keys)
    have = set((man or {}).get("requests") or [])
    if nonces is None:
        return want <= have
    held = (man or {}).get("nonces") or {}
    return want == have and all(held.get(k) == nonces.get(k) for k in want)


def wait_run(hook_dir: str, run_id: str, keys: Iterable[str], *,
             timeout_s: Optional[float] = None, nonces: Optional[Dict[str, str]] = None) -> dict:
    """Wait until the run's manifest lists ``keys``."""
    want = set(str(k) for k in keys)
    idle = artifact_wait_s() if timeout_s is None else float(timeout_s)
    last_n, deadline = -1, time.monotonic() + idle
    while True:
        man = read_manifest(hook_dir, run_id) or {}
        have = set(man.get("requests") or [])
        if holds(man, want, nonces):
            return man
        if len(have) != last_n:
            last_n, deadline = len(have), time.monotonic() + idle
        if time.monotonic() >= deadline:
            raise RunArtifactError(
                f"run {run_id!r} under {hook_dir}: request(s) {sorted(want - have) or sorted(want)}"
                f" not written by this response after {idle:g} s ({WAIT_ENV})")
        time.sleep(0.05)
