"""API-server side of delivered HS data: the read route and the save_to_disk writer."""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from urllib.parse import unquote

import vllm.envs as envs

from . import run_artifact
from .aperture_gather import (DELIVERY_HARD_CAP_S, DeliveryTimeoutError, GatherError,
                              NoDeliveryError, delivery_base, delivery_root, delivery_timeout_s,
                              discover_delivery_ranks, poll_delivered)
from .delivered_probes import (AmbiguousDelivery, check_id, encode_delivery, external_id,
                               hs_probes, key_pattern, match_keys, response_id)
from .delivery_selector import STAMP_ENV
from .tp_shard import parse_rank_dir
from mia.errors import MiaConfigurationError

logger = logging.getLogger(__name__)

ROUTE_ENV = "MIA_DELIVERY_ROUTE"
ROUTE_PATH = "/v1/mia/delivered"
MAX_N = 1024
_READ_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mia-read")
_WRITE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mia-write")
_TASKS: set = set()
_INFO_ATTR = "_mia_delivery_info_task"
_POLL_S = 0.25


def route_enabled() -> bool:
    v = os.environ.get(ROUTE_ENV, "")
    if v not in ("", "0", "1"):
        raise MiaConfigurationError(f"{ROUTE_ENV}={v!r}: use 1 (default) or 0")
    return v != "0"


def _suffix() -> bool:
    return not bool(envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION)


def run_filter(info: dict):
    """The engine's run ids when request ids can repeat across launches, else None."""
    return None if _suffix() else (info or {}).get("run_ids")


def merge_info(results) -> dict:
    """One engine's ``mia_delivery_info`` results as names, config, roots and run ids."""
    names: Dict[int, str] = {}
    config: dict = {}
    roots: List[str] = []
    run_ids: Optional[set] = set()
    for r in results or ():
        if not isinstance(r, dict):
            continue
        for ln, n in r.get("names") or ():
            names.setdefault(int(ln), str(n))
        config = config or dict(r.get("config") or {})
        d = r.get("delivery_dir")
        if d:
            root = os.path.dirname(d) if parse_rank_dir(os.path.basename(d)) is not None else d
            if root not in roots:
                roots.append(root)
            if run_ids is not None and r.get("run_id"):
                run_ids.add(str(r["run_id"]))
            else:
                run_ids = None
    return {"names": names, "config": config, "roots": roots,
            "run_ids": sorted(run_ids) if run_ids else None}


async def engine_info(engine) -> dict:
    """Names and config of ``engine``'s model: one ``collective_rpc`` per engine, cached."""
    got = getattr(engine, _INFO_ATTR, None)
    if isinstance(got, dict):
        return got
    task = got
    if task is None or task.get_loop() is not asyncio.get_running_loop():
        task = asyncio.ensure_future(engine.collective_rpc("mia_delivery_info"))
        setattr(engine, _INFO_ATTR, task)
    try:
        info = merge_info(await task)
    except Exception:
        if getattr(engine, _INFO_ATTR, None) is task:
            setattr(engine, _INFO_ATTR, None)
        raise
    setattr(engine, _INFO_ATTR, info)
    return info


def _key_names(roots) -> List[str]:
    out: List[str] = []
    for root in roots:
        dirs = [d for _r, d in discover_delivery_ranks(root)] or [root]
        for d in dirs:
            try:
                with os.scandir(d) as it:
                    out.extend(unquote(e.name) for e in it if e.is_dir())
            except OSError:
                continue
    return out


def _scan_keys(ext: str, n: int, suffix: bool):
    try:
        found = match_keys(_key_names(delivery_root()), ext, n, suffix=suffix)
    except NoDeliveryError:
        found = {}
    if len(found) == int(n):
        return [found[j] for j in range(int(n))], len(found)
    return None, len(found)


async def await_delivery(ext: str, n: int, layers, *, timeout_s: Optional[float] = None,
                         pool=None, run_ids=None, keys=None):
    """Delivered keys and per-sample layers of one engine request."""
    loop = asyncio.get_running_loop()
    pool = _READ_POOL if pool is None else pool
    idle = delivery_timeout_s() if timeout_s is None else float(timeout_s)
    suffix = _suffix()
    start = last = loop.time()
    seen = None
    while keys is None:
        keys, count = await loop.run_in_executor(pool, _scan_keys, ext, n, suffix)
        if keys is not None:
            break
        now = loop.time()
        if count != seen:
            seen, last = count, now
        if now - last >= idle or now - start >= DELIVERY_HARD_CAP_S:
            raise DeliveryTimeoutError(
                f"{ext!r}: {count} of {n} sample deliveries after {now - start:.1f} s",
                missing={ext: f"{count} of {n} sample deliveries present"})
        await asyncio.sleep(_POLL_S)
    exp = {k: list(layers) for k in keys} if isinstance(layers, list) else None
    base = delivery_base()
    seen, last = None, loop.time()
    while True:
        got, missing, mark = await loop.run_in_executor(pool, poll_delivered, base, keys, exp,
                                                        run_ids)
        if got is not None:
            return keys, [got[k] for k in keys]
        now = loop.time()
        if mark != seen:
            seen, last = mark, now
        if now - last >= idle or now - start >= DELIVERY_HARD_CAP_S:
            why = "; ".join(sorted(set(missing.values()))) or "not delivered"
            raise DeliveryTimeoutError(f"{ext!r}: {why}", missing={ext: why})
        await asyncio.sleep(_POLL_S)


def _parse_layers(s: str):
    s = (s or "").strip()
    if not s:
        return None
    vals = [int(x) for x in s.split(",")]
    if len(vals) > 4096 or any(v < 0 for v in vals):
        raise ValueError("bad layers")
    return vals


def sample_keys(key: str, n: int) -> List[str]:
    """Delivered keys of one engine request: itself, or ``<j>_<key>`` per sample when n > 1."""
    return [key] if int(n) <= 1 else [f"{j}_{key}" for j in range(int(n))]


def _encode(samples, keys, names, config) -> bytes:
    return encode_delivery(samples, keys=keys, names=names, config=config)


def _not_ready(rid: str, e: DeliveryTimeoutError) -> str:
    why = "; ".join(sorted(set(str(v) for v in (e.missing or {}).values()))) or "not delivered"
    return f"delivery for {rid!r} not ready: {why}"


def build_router():
    # lazy: optional dependency (fastapi)
    from fastapi import APIRouter, Query, Request
    from fastapi.responses import JSONResponse, Response

    router = APIRouter()

    @router.get(ROUTE_PATH)
    async def mia_delivered(request: Request, rid: str = Query(..., alias="id"),
                            item: int = Query(0, ge=0, le=MAX_N),
                            n: int = Query(1, ge=1, le=MAX_N),
                            kind: str = Query("chat"), layers: str = Query(""),
                            key: str = Query(""),
                            timeout_s: Optional[float] = Query(None, gt=0,
                                                               le=DELIVERY_HARD_CAP_S)):
        try:
            check_id(rid)
            lay = _parse_layers(layers)
            ext = external_id(rid, kind, item)
            if key and not key_pattern(ext, 1, suffix=_suffix()).match(key):
                raise ValueError("key does not belong to this response")
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if (os.environ.get(STAMP_ENV) or "").partition(":")[0] != "hybrid":
            return JSONResponse({"error": "no capture for this id"}, status_code=404)
        loop = asyncio.get_running_loop()
        try:
            info = await engine_info(request.app.state.engine_client)
            keys, samples = await await_delivery(ext, n, lay, timeout_s=timeout_s,
                                                 run_ids=run_filter(info),
                                                 keys=sample_keys(key, n) if key else None)
            data = await loop.run_in_executor(_READ_POOL, _encode, samples, keys,
                                              info["names"], info["config"])
        except AmbiguousDelivery as e:
            logger.warning("mia delivered read %r: %s: %s", rid, e, e.names)
            return JSONResponse({"error": f"ambiguous: {e.count} deliveries match this id"},
                                status_code=409)
        except DeliveryTimeoutError as e:
            logger.warning("mia delivered read timed out: %s", e)
            return JSONResponse({"error": _not_ready(rid, e)}, status_code=504)
        except (GatherError, OSError) as e:
            logger.error("mia delivered read %r failed: %s", rid, e)
            return JSONResponse({"error": "delivery for this id is unreadable"},
                                status_code=500)
        except Exception:  # noqa: BLE001
            logger.exception("mia delivered read %r failed", rid)
            return JSONResponse({"error": "delivery read failed"}, status_code=500)
        return Response(content=data, media_type="application/octet-stream")

    return router


def attach(app) -> None:
    """Add the read route to a vLLM API app (unless ``MIA_DELIVERY_ROUTE=0``)."""
    if not route_enabled():
        print(f"[mia/delivery] read route off ({ROUTE_ENV}=0)", flush=True)
        return
    app.include_router(build_router())


def patch_app_builder() -> None:
    """Wrap vLLM's ``attach_endpoint_plugins`` (called once by ``build_app``) to add the route."""
    # lazy: vLLM's API server app loads only when the patch runs
    import vllm.entrypoints.launchers.app as app_mod

    orig = app_mod.attach_endpoint_plugins
    if getattr(orig, "_mia_route", False):
        return

    def attach_endpoint_plugins(app, supported_tasks):
        orig(app, supported_tasks)
        attach(app)

    attach_endpoint_plugins._mia_route = True
    app_mod.attach_endpoint_plugins = attach_endpoint_plugins


def _write_items(ext: str, n: int, samples, metas: List[dict], names, config,
                 hook_dir: str, run_id: str, unit=None, stamp=None, nonce=None,
                 start=None) -> List[str]:
    items = []
    for j, rows in enumerate(samples):
        key = ext if int(n) == 1 else f"{j}_{ext}"
        items.append((key, hs_probes(rows, {**metas[j], "config": config}, names, layout="disk")
                      if rows else None))
    return run_artifact.append(hook_dir, run_id, kind="hs", items=items, unit=unit, stamp=stamp,
                               nonce=nonce, start=start)


# The longest a completion's prompt waits for an earlier prompt's run write.
ORDER_WAIT_S = DELIVERY_HARD_CAP_S
SWEEP_S = 10.0
_TURNS: Dict[tuple, dict] = {}
_SWEPT = [0.0]


def prompt_index(request_id: str) -> Optional[int]:
    """The prompt index of a completion's engine request (``cmpl-<id>-<i>``), else None."""
    unit = response_id(str(request_id))
    tail = str(request_id)[len(unit) + 1:]
    return int(tail) if unit != str(request_id) and tail.isdigit() else None


def _shared_index(run_id: str, request_id: str) -> Optional[int]:
    if str(run_id) == str(request_id):           # no run_id given: the prompt's own run
        return None
    return prompt_index(request_id)


def _sweep(now: float) -> None:
    for k in [k for k, v in _TURNS.items() if now - v["t"] > ORDER_WAIT_S]:
        del _TURNS[k]


def _turns(hook_dir: str, run_id: str, request_id: str) -> dict:
    now = time.monotonic()
    if now - _SWEPT[0] >= SWEEP_S:
        _SWEPT[0] = now
        _sweep(now)
    key = (str(hook_dir), str(run_id), response_id(str(request_id)))
    turns = _TURNS.setdefault(key, {"t": now, "ended": set(), "waits": {}})
    turns["t"] = now
    return turns


async def await_turn(hook_dir: str, run_id: str, request_id: str) -> None:
    """Return once every earlier prompt of this completion ended its turn on the shared run."""
    i = _shared_index(run_id, request_id)
    if not i:
        return
    turns = _turns(hook_dir, run_id, request_id)
    missing = [j for j in range(i) if j not in turns["ended"]]
    if not missing:
        return
    waits = [turns["waits"].setdefault(j, asyncio.Event()) for j in missing]
    try:
        await asyncio.wait_for(asyncio.gather(*(w.wait() for w in waits)), ORDER_WAIT_S)
    except asyncio.TimeoutError:
        unit = response_id(str(request_id))
        late = [f"{unit}-{j}" for j in missing if j not in turns["ended"]]
        print(f"[mia/delivery] save_to_disk: {late} never wrote run {run_id!r} after "
              f"{ORDER_WAIT_S:g} s; writing {request_id!r} out of prompt order", flush=True)


def end_turn(hook_dir: str, run_id: str, request_id: str) -> None:
    """This completion prompt is done with the shared run (written, skipped, failed or no write)."""
    i = _shared_index(run_id, request_id)
    if i is None:
        return
    turns = _turns(hook_dir, run_id, request_id)
    turns["ended"].add(i)
    w = turns["waits"].get(i)
    if w is not None:
        w.set()


async def _write_task(engine, ext, n, layers, metas, hook_dir, run_id, unit=None,
                      stamp=None, nonce=None, keys=None, start=None) -> None:
    try:
        info = await engine_info(engine)
        _keys, samples = await await_delivery(ext, n, layers, pool=_WRITE_POOL,
                                              run_ids=run_filter(info), keys=keys)
        await await_turn(hook_dir, run_id, ext)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_WRITE_POOL, _write_items, ext, n, samples, metas,
                                   info["names"], info["config"], hook_dir, run_id, unit, stamp,
                                   nonce, start)
    finally:
        end_turn(hook_dir, run_id, ext)


def _done(task) -> None:
    _TASKS.discard(task)
    if task.cancelled():
        return
    e = task.exception()
    if e is not None:
        print(f"[mia/delivery] save_to_disk write FAILED for {task.get_name()}: {e!r}",
              flush=True)


def spawn_writer(engine, *, ext: str, n: int, layers, metas: List[dict], hook_dir: str,
                 run_id: str, unit=None, stamp=None, nonce=None, keys=None, start=None):
    """Write one finished request's delivery as an eager run artifact, in prompt order."""
    task = asyncio.get_running_loop().create_task(
        _write_task(engine, ext, n, layers, metas, hook_dir, run_id, unit, stamp, nonce, keys,
                    start),
        name=f"mia-save:{ext}")
    _TASKS.add(task)
    task.add_done_callback(_done)
    return task
