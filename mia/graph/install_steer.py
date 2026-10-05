"""CUDA-graph activation-steering install (buffer mode)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from mia._profiler import PROF
from mia.graph import register_graph_ops
from mia.graph.hosts import SteerHost
from mia.graph.install import (
    _IDLE_ROUTE_KEY,
    _resolve_max_num_batched_tokens,
    install_prepare_inputs_routing,
)
from mia.graph.registry import PinnedMirror, set_registry
from mia.graph.steer_routing_gpu import scatter_routing
from mia.graph.tp_shard import refuse_pipeline_parallel
from mia.runner import StepView
from mia.workers._common import iter_matched_modules
from mia.workers.hs_capture_worker import match_layer
from mia.workers.steer_worker import (
    _load_steering_vector,
    _resolve_steer_config,
    _parse_steer_layers,
    STEER_COL_ALL,
    STEER_COL_NONE,
    is_default_steer_modes,
    resolve_steer_modes,
    steer_col_for,
    steer_span,
)

_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
_STEER_HOST_ATTR = "_mia_steer_host"

_ACTIVE_WORKER_STEER = None


def _require_buffer_mode_steer() -> None:
    mode = os.environ.get("MIA_STEER_MODE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise RuntimeError(
            f"MIA_STEER_MODE={mode!r} is no longer supported: the PIECEWISE steer_residual "
            "mode was removed. Buffer mode is the only FULL-cudagraph steering path; "
            "unset MIA_STEER_MODE or set it to 'buffer'.")
_STEER_ROUTE_FASTPATH = os.environ.get("MIA_STEER_ROUTE_FASTPATH", "0") == "1"


def _wrap_layer_class(cls: type) -> None:
    if cls in _WRAPPED_LAYER_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_LAYER_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            out = orig_fwd(self, *args, **kwargs)
            host = getattr(self, _STEER_HOST_ATTR, None)
            if host is not None and host.do_steer:
                if (isinstance(out, tuple) and len(out) >= 2
                        and isinstance(out[1], torch.Tensor)):
                    host.steer(out[1])
                elif isinstance(out, torch.Tensor):
                    host.steer(out)
            return out
        return wrapped

    cls.forward = make_wrapped(orig_forward)


def install_steer_hosts(worker) -> None:
    """Install the buffer-mode steering path via decoder-layer wrap."""
    global _ACTIVE_WORKER_STEER

    _require_buffer_mode_steer()
    refuse_pipeline_parallel(
        getattr(getattr(worker, "parallel_config", None), "pipeline_parallel_size", 1),
        "steer graph install")

    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install_steer] no model on model_runner; skip steer install")
        return

    register_graph_ops()
    _ACTIVE_WORKER_STEER = worker

    if not getattr(worker, "_vector_cache", None):
        worker._vector_cache = {}
    if not hasattr(worker, "_env_config_path"):
        worker._env_config_path = os.environ.get("MIA_STEER_CONFIG")

    matched = list(iter_matched_modules(model, match_layer))
    if not matched:
        print("[graph/install_steer] no decoder layers matched LAYER_PATTERNS; "
              "steering inactive")
        return

    device_t = next(model.parameters()).device

    _install_steer_buffer(worker, model, matched, device_t)


class SteerRegistry:
    """Per-worker owner of the steering routing slabs and the resident vector table."""

    def __init__(self, num_layers, cap, hidden, v_max, device, dtype):
        self.num_layers = int(num_layers)
        self.cap = int(cap)
        self.hidden = int(hidden)
        self.v_max = int(v_max)
        self.device = torch.device(device)
        self.should_capture = True

        self.coeff_all = torch.zeros(num_layers, cap, dtype=torch.float32, device=device)
        self.vec_id_all = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        self.mode_all = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        self.vec_table = torch.zeros(v_max, hidden, dtype=dtype, device=device)
        self.avg_proj_table = torch.zeros(v_max, dtype=torch.float32, device=device)

        pin = self.device.type == "cuda"
        self._aperture = PinnedMirror(
            [
                ("coeff", (num_layers, cap), torch.float32),
                ("vec_id", (num_layers, cap), torch.int64),
                ("mode", (num_layers, cap), torch.int64),
            ],
            pin=pin,
        )

        self.hosts: Dict[int, SteerHost] = {}
        self.vec_paths: Dict[str, int] = {}
        self.vec_has_avgproj: set = set()
        self._next_vec_id = 0
        self.steer_skipped = 0
        self._skip_warned: set = set()
        self._pending_plans: list = []

        self.incremental_enabled = (
            os.environ.get("MIA_INCREMENTAL_ROUTING", "1") != "0"
            and self.device.type == "cuda"
            and self.should_capture)
        self._col_state: Optional[np.ndarray] = None
        self._inc_coeff: Optional[torch.Tensor] = None
        self._inc_vecid: Optional[torch.Tensor] = None
        self._inc_mode: Optional[torch.Tensor] = None
        self._inc_event: Optional[torch.cuda.Event] = None
        self._inc_force_full = False
        self._st_intern: Dict[tuple, int] = {}
        self._st_val: Dict[int, tuple] = {}
        self._st_next = 1
        self._pending_assignments: list = []

        self._any_gated = False
        self._step_cols: Optional[list] = None

        self._gpu_routing_env = os.environ.get("MIA_STEER_GPU_ROUTING", "1") != "0"
        self.gpu_routing = self._gpu_routing_env and self.device.type == "cuda"
        self.slot_vid = self.slot_mode = self.slot_coeff = self.slot_layer_mask = None
        self._slot_req_key: Optional[tuple] = None
        if self._gpu_routing_env:
            pin = self.device.type == "cuda"
            self.slot_vid = torch.zeros(cap, dtype=torch.int64, device=device)
            self.slot_mode = torch.zeros(cap, dtype=torch.int64, device=device)
            self.slot_coeff = torch.zeros(cap, dtype=torch.float32, device=device)
            self.slot_layer_mask = torch.zeros(cap, num_layers, dtype=torch.bool, device=device)
            self._slot_vid_h = torch.zeros(cap, dtype=torch.int64, pin_memory=pin)
            self._slot_mode_h = torch.zeros(cap, dtype=torch.int64, pin_memory=pin)
            self._slot_coeff_h = torch.zeros(cap, dtype=torch.float32, pin_memory=pin)
            self._slot_mask_h = torch.zeros(cap, num_layers, dtype=torch.bool, pin_memory=pin)
            self._slot_vid_np = self._slot_vid_h.numpy()
            self._slot_mode_np = self._slot_mode_h.numpy()
            self._slot_coeff_np = self._slot_coeff_h.numpy()
            self._slot_mask_np = self._slot_mask_h.numpy()
            self.slot_col = torch.full((cap,), STEER_COL_ALL, dtype=torch.int64, device=device)
            self._slot_col_h = torch.full((cap,), STEER_COL_ALL, dtype=torch.int64,
                                          pin_memory=pin)

    def register_host(self, host: SteerHost) -> None:
        self.hosts[host.layer_num] = host

    def note_skip(self, path: str, why: str) -> None:
        """A steer request this table leaves unsteered: counted, and warned once per vector."""
        self.steer_skipped += 1
        PROF.incr("steer.skipped")
        if path not in self._skip_warned:
            self._skip_warned.add(path)
            print(f"[mia/steer] WARNING: request(s) run UNSTEERED: {why} ({path})", flush=True)

    def assign_views(self) -> None:
        for layer_num, host in self.hosts.items():
            host.bind_views(self.coeff_all[layer_num], self.vec_id_all[layer_num],
                            self.mode_all[layer_num], self.vec_table,
                            self.avg_proj_table)

    def begin_step(self) -> None:
        pass

    @property
    def coeff_pinned(self) -> torch.Tensor:
        return self._aperture.cur("coeff")

    @property
    def vec_id_pinned(self) -> torch.Tensor:
        return self._aperture.cur("vec_id")

    @property
    def mode_pinned(self) -> torch.Tensor:
        return self._aperture.cur("mode")

    def _resolve_step_cols(self, step: StepView) -> list:
        req_ids = step.req_ids
        if not req_ids:
            return []
        worker = _ACTIVE_WORKER_STEER
        env_path = getattr(worker, "_env_config_path", None) if worker else None
        qsl = step.query_start_loc_np
        bs = step.num_reqs
        cols = []
        for i in range(bs):
            extra = step.extra_args_for(i)
            steer_arg = (extra or {}).get("steer")
            cfg = _resolve_steer_config(steer_arg, env_path)
            if cfg is None:
                cols.append(STEER_COL_NONE)
                continue
            try:
                phase, positions = resolve_steer_modes(cfg)
            except ValueError:
                cols.append(STEER_COL_NONE)
                continue
            if not is_default_steer_modes(phase, positions):
                self._any_gated = True
            start = int(qsl[i])
            end = min(int(qsl[i + 1]), self.cap)
            is_prefill = bool(step.is_prefilling_np[i])
            is_final = True
            if is_prefill and positions == "last_token":
                is_final = (int(step.num_computed_tokens_np[i]) + (end - start)) \
                    >= int(step.prompt_len_np[i])
            cols.append(steer_col_for(phase, positions, is_prefill, is_final, start, end))
        return cols

    def routing_key(self, step: StepView) -> Optional[tuple]:
        """Invalidation signature for the routing wrapper."""
        req_ids = step.req_ids
        if not req_ids:
            return None
        base = (tuple(req_ids), tuple(int(x) for x in step.query_start_loc_np))
        if not self._any_gated:
            self._step_cols = None
            return base
        cols = self._resolve_step_cols(step)
        self._step_cols = cols
        return base + (tuple(cols),)

    def reset_pinned(self, width: Optional[int] = None) -> None:
        self._aperture.wait_current()
        coeff = self._aperture.cur("coeff")
        vec_id = self._aperture.cur("vec_id")
        mode = self._aperture.cur("mode")
        if width is None:
            coeff.zero_()
            vec_id.zero_()
            mode.zero_()
        else:
            w = max(1, min(int(width), self.cap))
            coeff[:, :w].zero_()
            vec_id[:, :w].zero_()
            mode[:, :w].zero_()

    def upload(self, width: Optional[int] = None) -> None:
        coeff = self._aperture.cur("coeff")
        vec_id = self._aperture.cur("vec_id")
        mode = self._aperture.cur("mode")
        if width is None:
            self.coeff_all.copy_(coeff, non_blocking=True)
            self.vec_id_all.copy_(vec_id, non_blocking=True)
            self.mode_all.copy_(mode, non_blocking=True)
        else:
            w = max(1, min(int(width), self.cap))
            self.coeff_all[:, :w].copy_(coeff[:, :w], non_blocking=True)
            self.vec_id_all[:, :w].copy_(vec_id[:, :w], non_blocking=True)
            self.mode_all[:, :w].copy_(mode[:, :w], non_blocking=True)
        self._aperture.record_advance()

    def force_full_routing(self) -> None:
        """Force the next ``apply_incremental_routing`` to rewrite [0,width) in full."""
        self._inc_force_full = True
        self._slot_req_key = None

    def _intern_state(self, layer: int, coeff: float, vid: int, mode: int) -> int:
        key = (int(layer), float(coeff), int(vid), int(mode))
        sid = self._st_intern.get(key)
        if sid is None:
            sid = self._st_next
            self._st_next += 1
            self._st_intern[key] = sid
            self._st_val[sid] = key
        return sid

    def apply_incremental_routing(self, assignments: list, width: int) -> bool:
        """Write and upload only the (layer, column) routing cells whose steer state changed."""
        cap = self.cap
        nL = self.num_layers
        w = max(1, min(int(width), cap))
        if self._col_state is None:
            self._col_state = np.zeros((nL, cap), dtype=np.int64)
            pin = self.device.type == "cuda"
            self._inc_coeff = torch.zeros(nL, cap, dtype=torch.float32, pin_memory=pin)
            self._inc_vecid = torch.zeros(nL, cap, dtype=torch.int64, pin_memory=pin)
            self._inc_mode = torch.zeros(nL, cap, dtype=torch.int64, pin_memory=pin)
            self._inc_event = torch.cuda.Event() if pin else None

        new_state = np.zeros((nL, cap), dtype=np.int64)
        for (start, end, layer, coeff, vid, mode) in assignments:
            if end <= start:
                continue
            new_state[layer, start:end] = self._intern_state(layer, coeff, vid, mode)

        old = self._col_state
        if self._inc_force_full:
            changed_mask = np.ones((nL, w), dtype=bool)
            self._inc_force_full = False
        else:
            changed_mask = new_state[:, :w] != old[:, :w]
        rows_idx, cols_idx = np.nonzero(changed_mask)
        if rows_idx.size == 0:
            return False

        rows = np.unique(rows_idx)
        lo = int(cols_idx.min())
        hi = int(cols_idx.max()) + 1

        if self._inc_event is not None:
            self._inc_event.synchronize()
        rr = torch.from_numpy(rows_idx)
        cc = torch.from_numpy(cols_idx)
        self._inc_coeff[rr, cc] = 0
        self._inc_vecid[rr, cc] = 0
        self._inc_mode[rr, cc] = 0
        new_changed = new_state[rows_idx, cols_idx]
        for sid in np.unique(new_changed):
            if sid == 0:
                continue
            m = new_changed == sid
            _layer, coeff, vid, mode = self._st_val[int(sid)]
            rr_s = torch.from_numpy(rows_idx[m])
            cc_s = torch.from_numpy(cols_idx[m])
            self._inc_coeff[rr_s, cc_s] = coeff
            self._inc_vecid[rr_s, cc_s] = vid
            self._inc_mode[rr_s, cc_s] = mode

        for L in rows.tolist():
            self.coeff_all[L, lo:hi].copy_(self._inc_coeff[L, lo:hi], non_blocking=True)
            self.vec_id_all[L, lo:hi].copy_(self._inc_vecid[L, lo:hi], non_blocking=True)
            self.mode_all[L, lo:hi].copy_(self._inc_mode[L, lo:hi], non_blocking=True)
        if self._inc_event is not None:
            self._inc_event.record()
        old[:, :w] = new_state[:, :w]
        return True

    def refresh_slot_config(self, step: StepView) -> bool:
        """Rebuild the per-slot config table only if the batch composition or order changed."""
        req_ids = step.req_ids
        key = tuple(req_ids)
        if key == self._slot_req_key:
            return False
        worker = _ACTIVE_WORKER_STEER
        env_path = getattr(worker, "_env_config_path", None) if worker else None
        bs = len(req_ids)
        prev = self._slot_req_key
        if prev is None:
            changed = range(bs)
        else:
            n_prev = len(prev)
            changed = [i for i in range(bs) if i >= n_prev or prev[i] != req_ids[i]]
        vid_np, mode_np = self._slot_vid_np, self._slot_mode_np
        coeff_np, mask_np = self._slot_coeff_np, self._slot_mask_np
        for i in changed:
            vid_np[i] = 0
            mode_np[i] = 0
            coeff_np[i] = 0.0
            mask_np[i] = False
            extra = step.extra_args_for(i)
            steer_arg = (extra or {}).get("steer")
            cfg = _resolve_steer_config(steer_arg, env_path)
            if cfg is None:
                continue
            layers = _parse_steer_layers(cfg.get("optimal_layer", -1), self.num_layers)
            method = cfg.get("method", "adjust_rs")
            vector_path = cfg.get("vector_path")
            if not layers or method not in ("add_vector", "adjust_rs") or not vector_path:
                continue
            vid = self.vec_id_for_path(vector_path, worker)
            if vid is None:
                continue
            if method == "adjust_rs" and vid not in self.vec_has_avgproj:
                self.note_skip(vector_path, "adjust_rs vector has no avg_proj")
                continue
            vid_np[i] = vid
            mode_np[i] = 1 if method == "adjust_rs" else 0
            coeff_np[i] = 0.0 if method == "adjust_rs" else float(
                cfg.get("coefficient", 0.0))
            mask_np[i, np.asarray(layers, dtype=np.intp)] = True
            PROF.incr("steer.fire")
        self.slot_vid[:bs].copy_(self._slot_vid_h[:bs], non_blocking=True)
        self.slot_mode[:bs].copy_(self._slot_mode_h[:bs], non_blocking=True)
        self.slot_coeff[:bs].copy_(self._slot_coeff_h[:bs], non_blocking=True)
        self.slot_layer_mask[:bs].copy_(self._slot_mask_h[:bs], non_blocking=True)
        self._slot_req_key = key
        return True

    def _qsl_device(self, step: StepView):
        return step.query_start_loc

    def build_and_upload_gpu(self, step: StepView, width, build_routing_fn=None) -> list:
        """GPU per-step routing: host slot refresh on req_ids change, then a GPU scatter into the slabs."""
        composition_changed = self.refresh_slot_config(step)
        qsl_np = step.query_start_loc_np
        real_n = int(qsl_np[-1]) if qsl_np.size else 0
        qsl_dev = self._qsl_device(step)
        slot_col = None
        cols = self._step_cols
        if cols is None and composition_changed:
            cols = self._resolve_step_cols(step)
        if self._any_gated and cols:
            bs = min(len(cols), self.cap)
            self._slot_col_h[:bs].copy_(torch.as_tensor(cols[:bs], dtype=torch.int64))
            self.slot_col[:bs].copy_(self._slot_col_h[:bs], non_blocking=True)
            slot_col = self.slot_col
        self._step_cols = None
        scatter_routing(qsl_dev, self.slot_vid, self.slot_mode, self.slot_coeff,
                        self.slot_layer_mask, self.coeff_all, self.vec_id_all, self.mode_all,
                        real_n, width, slot_col=slot_col)
        return []

    def vec_id_for_path(self, path: str, worker) -> Optional[int]:
        """Resolve ``vector_path`` to a vec_table row, loading and caching it on first use."""
        vid = self.vec_paths.get(path)
        if vid is not None:
            return vid
        cache = getattr(worker, "_vector_cache", None) if worker is not None else None
        if cache is None and worker is not None:
            worker._vector_cache = cache = {}
        data = cache.get(path) if cache is not None else None
        if data is None:
            try:
                raw = _load_steering_vector(path)
            except Exception as e:  # noqa: BLE001
                self.note_skip(path, f"steering vector could not be loaded: {e!r}")
                return None
            d = raw["dir"]
            data = {"dir": d if torch.is_tensor(d) else torch.tensor(d)}
            if "avg_proj" in raw:
                ap = raw["avg_proj"]
                data["avg_proj"] = float(ap.item()) if torch.is_tensor(ap) else float(ap)
            if cache is not None:
                cache[path] = data
        if self._next_vec_id >= self.v_max:
            self.note_skip(path, f"steering vector table is full (MIA_STEER_VMAX={self.v_max} "
                                 f"distinct vectors per engine)")
            return None
        vid = self._next_vec_id
        self._next_vec_id += 1
        self.vec_table[vid].copy_(
            data["dir"].to(self.vec_table.device, dtype=self.vec_table.dtype).view(-1))
        if "avg_proj" in data:
            self.avg_proj_table[vid] = float(data["avg_proj"])
            self.vec_has_avgproj.add(vid)
        self.vec_paths[path] = vid
        return vid


def _install_steer_buffer(worker, model, matched, device) -> SteerRegistry:
    cap = _resolve_max_num_batched_tokens(worker)
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = int(getattr(text_cfg, "hidden_size"))
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0)) or (
        max(ln for _, _, ln in matched) + 1)
    v_max = int(os.environ.get("MIA_STEER_VMAX", "16"))
    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    registry = SteerRegistry(num_layers, cap, hidden, v_max, device, buf_dtype)

    n_wired = 0
    for name, module, layer_num0 in matched:
        if 0 <= layer_num0 < num_layers:
            host = SteerHost(module_name=name, layer_num=layer_num0, cap=cap,
                             do_steer=True)
            setattr(module, _STEER_HOST_ATTR, host)
            registry.register_host(host)
            n_wired += 1
        _wrap_layer_class(type(module))
    registry.assign_views()

    set_registry(worker, "steer", registry)
    install_prepare_inputs_routing(
        worker.model_runner, worker, _build_routing_steer, label="steer",
        routing_key_fn=_steer_idle_key if _STEER_ROUTE_FASTPATH else None)
    print(f"[graph/install_steer] buffer-mode steering wired: {n_wired} layer(s) over "
          f"{num_layers} slots; cap={cap} hidden={hidden} V_max={v_max}; "
          f"NO splitting op (rides decode cudagraph)")
    return registry


def _steer_idle_key(step: StepView, registry):
    req_ids = step.req_ids
    if not req_ids:
        return None
    worker = _ACTIVE_WORKER_STEER
    if worker is not None and getattr(worker, "_env_config_path", None):
        return registry.routing_key(step)
    for i in range(step.num_reqs):
        extra = step.extra_args_for(i)
        if extra and extra.get("steer"):
            return registry.routing_key(step)
    return _IDLE_ROUTE_KEY


def _build_routing_steer(step: StepView, registry: SteerRegistry) -> list:
    worker = _ACTIVE_WORKER_STEER
    req_ids = step.req_ids
    bs = step.num_reqs
    qsl = step.query_start_loc_np
    inc = registry.incremental_enabled
    if not inc:
        coeff_pinned = registry.coeff_pinned
        vecid_pinned = registry.vec_id_pinned
        mode_pinned = registry.mode_pinned
    cap = registry.cap
    env_path = getattr(worker, "_env_config_path", None) if worker else None

    plans: list = []
    assignments: list = []
    for i in range(bs):
        req_id = req_ids[i]
        extra = step.extra_args_for(i)
        steer_arg = (extra or {}).get("steer")
        cfg = _resolve_steer_config(steer_arg, env_path)
        if cfg is None:
            continue
        layers = _parse_steer_layers(cfg.get("optimal_layer", -1), registry.num_layers)
        if not layers:
            continue
        method = cfg.get("method", "adjust_rs")
        if method not in ("add_vector", "adjust_rs"):
            continue
        vector_path = cfg.get("vector_path")
        if not vector_path:
            continue
        vid = registry.vec_id_for_path(vector_path, worker)
        if vid is None:
            continue
        if method == "adjust_rs" and vid not in registry.vec_has_avgproj:
            registry.note_skip(vector_path, "adjust_rs vector has no avg_proj")
            continue
        mode_val = 1 if method == "adjust_rs" else 0
        coefficient = 0.0 if method == "adjust_rs" else float(cfg.get("coefficient", 0.0))

        start = int(qsl[i])
        end = min(int(qsl[i + 1]), cap)
        if end <= start:
            continue

        try:
            phase, positions = resolve_steer_modes(cfg)
        except ValueError:
            continue
        if not is_default_steer_modes(phase, positions):
            registry._any_gated = True
        is_prefill = bool(step.is_prefilling_np[i])
        is_final = True
        if is_prefill and positions == "last_token":
            n_computed = int(step.num_computed_tokens_np[i])
            n_prompt = int(step.prompt_len_np[i])
            is_final = (n_computed + (end - start)) >= n_prompt
        span = steer_span(phase, positions, is_prefill, is_final, start, end)
        if span is None:
            continue
        row_lo, row_hi = span

        for layer in layers:
            if inc:
                assignments.append((row_lo, row_hi, layer, coefficient, vid, mode_val))
            else:
                coeff_pinned[layer, row_lo:row_hi] = coefficient
                vecid_pinned[layer, row_lo:row_hi] = vid
                mode_pinned[layer, row_lo:row_hi] = mode_val
            plans.append({"req_idx": i, "layer": layer, "method": method, "vid": vid})

    if inc:
        registry._pending_assignments = assignments
    return plans


__all__ = ["install_steer_hosts"]

