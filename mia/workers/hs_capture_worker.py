"""Hidden-state capture worker (capture_hs): eager hooks and the CUDA-graph aperture path."""
import contextlib
import os
import pickle
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context

from mia._profiler import PROF
from mia.runner import StepView, install_request_arg_stash, require_v2_runner, step_view
from mia.workers._common import (
    capture_bytes,
    clear_states_for_req,
    compact_page_backed_cache,
    get_query_metadata,
    iter_matched_modules,
    iter_matching_req_ids,
    match_layer,
    quant_clone,
    resolve_capture_quant,
    save_pt_atomic,
    save_safetensors_atomic,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _aperture_disk_dbg(msg: str) -> None:
    if os.environ.get("MIA_APERTURE_DEBUG") == "1":
        print(f"[mia/aperture-disk] {msg}", flush=True)

_CENSUS_ON = os.environ.get("MIA_CAPTURE_CENSUS") == "1"

_BATCHED_FLUSH = os.environ.get("MIA_BATCHED_FLUSH") == "1"


def _cpu_list(tensors, acc: dict | None):
    if acc is not None:
        from mia.graph.census import cpu_list_measured
        return cpu_list_measured(tensors, acc)
    if _BATCHED_FLUSH:
        from mia.workers._common import cpu_list_batched
        return cpu_list_batched(tensors)
    return [t.cpu() for t in tensors]


def _worker_tp_rank(worker) -> int:
    r = getattr(worker, "_tp_rank", None)
    if r is not None:
        return int(r)
    from mia.graph.tp_shard import resolve_tp_coords
    return resolve_tp_coords(worker)[0]


def _hs_layer_shard(worker):
    from mia.graph.tp_shard import HS_MODE_ROUND_ROBIN, HSShard
    if getattr(worker, "_hs_shard_mode", None) != HS_MODE_ROUND_ROBIN:
        return None
    conf = getattr(worker, "_conf", None) or {}
    return HSShard.of(int(getattr(worker, "_tp_rank", 0) or 0),
                      int(getattr(worker, "_hs_tp_size", 1) or 1),
                      int(conf.get("num_layers", 0) or 0))


def _marshal_perreq_hs(per_layer: dict, conf, shard=None) -> bytes:
    hs_cache = {}
    for layer, t in per_layer.items():
        layer = int(layer)
        hs_cache[layer] = {
            "hidden_states": t.detach().to(torch.float32).cpu(),
            "layer_num": layer,
        }
    payload = {"hs_cache": hs_cache, "config": conf}
    if shard is not None:
        from mia.graph.tp_shard import HS_SHARD_KEY
        payload[HS_SHARD_KEY] = shard.as_header()
    return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))


class HSCaptureWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls."""

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    _default_hooks_on: str = "prefill"

    _captured_states: dict = {}
    _hooks_installed: bool = False
    _step: "StepView | None" = None

    def install_hooks(self):
        """Install forward hooks on all target decoder layers."""
        if self._hooks_installed:
            return
        self._hooks_installed = True
        runner = self.model_runner
        require_v2_runner(runner)
        stash = install_request_arg_stash(runner)
        self._step = None

        original_prepare = runner.prepare_inputs

        def prepare_inputs(*args, **kwargs):
            input_batch = original_prepare(*args, **kwargs)
            self._step = step_view(runner, input_batch, stash)
            return input_batch

        runner.prepare_inputs = prepare_inputs

        self._captured_states = {}
        self._disk_states = {}
        model = getattr(runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hs_mode = "last_token"

        self._shm = None
        if os.environ.get("MIA_USE_SHM", "0") == "1":
            try:
                from multiprocessing.shared_memory import SharedMemory
                shm_name = os.environ["MIA_SHM_NAME"]
                self._shm = SharedMemory(create=False, name=shm_name)
                self._shm_hidden_size = int(os.environ["MIA_SHM_HIDDEN_SIZE"])
                self._shm_num_layers = int(os.environ["MIA_SHM_NUM_LAYERS"])
                self._shm_max_batch = int(os.environ["MIA_SHM_MAX_BATCH"])
                self._shm_ready_flag = os.environ["MIA_SHM_READY_FLAG"]
                layer_order_str = os.environ.get("MIA_SHM_LAYER_ORDER", "")
                self._shm_layer_order = [int(x) for x in layer_order_str.split(";") if x]
            except Exception as e:
                print(f"SHM attach failed: {e} — falling back to disk path")
                self._shm = None

        from mia.graph.tp_shard import refuse_pipeline_parallel, resolve_tp_coords
        refuse_pipeline_parallel(getattr(self.parallel_config, "pipeline_parallel_size", 1),
                                 "HS install_hooks")
        tp_rank, _tp_size = resolve_tp_coords(self)
        self._tp_rank = tp_rank
        self._should_capture = tp_rank == 0

        from mia.graph.writer_process import init_writer_process, mark_no_writer
        if self._should_capture:
            init_writer_process(self)
        else:
            mark_no_writer(self, "HS sink rank: captures nothing")

        self._artifact_tag, self._artifact_gran, self._artifact_gsize = resolve_capture_quant("hs")

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        hidden_size = int(getattr(text_cfg, "hidden_size"))
        num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
        self._conf = {"hidden_size": hidden_size, "num_layers": num_layers}

        self._batch_cache_key = None
        self._batch_cache_entries = None

        def _resolve_batch(step: StepView, bs, query_start_loc):
            entries = []
            for i in range(bs):
                req_id = step.req_ids[i]
                extra = step.extra_args_for(i)
                if not extra or extra.get("output_hidden_states") is None:
                    continue
                output_layers = extra.get("output_hidden_states")
                hooks_on = extra.get("hooks_on", self._default_hooks_on)
                is_prefill = bool(step.is_prefilling_np[i])
                if hooks_on != "both":
                    if hooks_on == "prefill" and not is_prefill:
                        continue
                    if hooks_on == "decode" and is_prefill:
                        continue

                req_mode = extra.get("hs_mode", self.hs_mode)
                if is_prefill and req_mode == "last_token":
                    chunk_len = int(query_start_loc[i + 1].item() - query_start_loc[i].item())
                    if int(step.num_computed_tokens_np[i]) + chunk_len < int(step.prompt_len_np[i]):
                        continue

                entries.append({
                    "i": i,
                    "req_id": req_id,
                    "output_layers": output_layers,
                    "hs_mode": req_mode,
                    "save_to_disk": bool(extra.get("save_to_disk")),
                })
            return entries

        def hs_hook(output, module_name, layer_num):
            if not self._should_capture:
                return None

            step = self._step
            if step is None:
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)

            if metadata is None:
                return
            if torch.cuda.is_current_stream_capturing():
                return None

            cache_key = id(metadata)
            if self._batch_cache_key != cache_key:
                query_start_loc, _seq_lens = get_query_metadata(metadata)
                if query_start_loc is None:
                    return
                bs = len(query_start_loc) - 1
                self._batch_cache_key = cache_key
                self._batch_cache_entries = _resolve_batch(step, bs, query_start_loc)
                self._batch_cache_qsl = query_start_loc

            entries = self._batch_cache_entries
            if not entries:
                return

            wanted = []
            for e in entries:
                ol = e["output_layers"]
                if isinstance(ol, list) and (layer_num not in ol):
                    continue
                wanted.append(e)
            if not wanted:
                return

            last_indices = self._batch_cache_qsl

            if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], torch.Tensor):
                hidden = output[0] + output[1]
            elif isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            for e in wanted:
                i = e["i"]
                req_id = e["req_id"]
                req_mode = e["hs_mode"]

                start = int(last_indices[i].item())
                end = int(last_indices[i + 1].item())

                if req_mode == "last_token":
                    activation = hidden[end - 1].detach().clone()
                else:
                    activation = hidden[start:end].detach().clone()

                activation, hs_scale, hs_qmeta = quant_clone(
                    activation, self._artifact_tag, self._artifact_gran, self._artifact_gsize)

                PROF.incr("hook.fire.hs")
                PROF.gauge("captured.bytes.hs", capture_bytes(activation, hs_scale))

                bucket = self._disk_states if e["save_to_disk"] else self._captured_states
                if req_id not in bucket:
                    bucket[req_id] = {}
                layer_states = bucket[req_id]
                if module_name not in layer_states:
                    layer_states[module_name] = {"hidden_states": [], "layer_num": layer_num, "hs_mode": req_mode}
                    if hs_qmeta is not None:
                        layer_states[module_name].update(_hs_scale=[], _hs_qmeta=hs_qmeta)
                ls = layer_states[module_name]
                ls["hidden_states"].append(activation)
                if hs_qmeta is not None:
                    ls["_hs_scale"].append(hs_scale)

        self._hooks = []
        matched = []
        for name, module, layer_num in iter_matched_modules(model, match_layer):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name, ln=layer_num+1: hs_hook(o, n, ln)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} hidden-state hooks on layers: {matched}")


    def graph_install(self):
        """Install the CUDA-graph hidden-state capture path (buffer mode)."""
        if not getattr(self, "_captured_states", None):
            self._captured_states = {}
        if not getattr(self, "_disk_states", None):
            self._disk_states = {}
        from mia.graph.install_hs import (
            install_execute_model_wrapper_hs,
            install_hs_hosts,
        )
        install_hs_hosts(self)
        install_execute_model_wrapper_hs(self.model_runner, self)

    def flush_aperture(self) -> str | None:
        """Final drain and sidecar write; return this worker's run_dir, or None if not installed."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None:
            return None
        stop = getattr(drain, "stop", None)
        if callable(stop):
            drain.stop()
        else:
            drain.drain_once()
        drain.close()
        from mia.graph.tp_shard import drain_holds_data
        if not drain_holds_data(drain):
            return None
        return getattr(self, "_hs_run_dir", None)

    def get_drain_row_counts(self) -> dict:
        """Read-only diagnostic: aperture rows this worker's HS drain copied and skipped."""
        from mia.graph.install_hs import get_drain_row_counts
        return get_drain_row_counts(self)

    def flush_aperture_per_request(self):
        """Drive end-of-run per-request delivery and return every deliverable capture."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        if index is None:
            return None
        stop = getattr(drain, "stop", None)
        if callable(stop):
            drain.stop()
        lock = getattr(drain, "_index_lock", None) or contextlib.nullcontext()
        with lock:
            popped = index.pop_deliverable()
            for req_id, _ in popped:
                index.free(req_id)
            residency_after = len(index.live_req_ids())
        deliverables: dict = {}
        for req_id, per_layer in popped:
            deliverables[str(req_id)] = {
                int(layer): t.detach().to(torch.float32).cpu()
                for layer, t in per_layer.items()
            }
        raw = pickle.dumps((deliverables, residency_after))
        return _ZSTD_COMPRESSOR.compress(raw)

    def get_aperture_per_request(self, external_req_id: str) -> bytes | None:
        """Per-request retrieval for the off-loop HS aperture delivery path."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        if index is None:
            return None
        stash = self._drain_aperture_into_stash(drain, index)
        match = next(iter(iter_matching_req_ids(stash, external_req_id)), None)
        if match is None:
            return None
        return stash.pop(match)

    def _drain_aperture_into_stash(self, drain, index, free_external: str | None = None) -> dict:
        stash = getattr(self, "_aperture_perreq_stash", None)
        if stash is None:
            stash = {}
            self._aperture_perreq_stash = stash
        conf = getattr(self, "_conf", {})
        dbg = os.environ.get("MIA_APERTURE_DEBUG") == "1"
        lock = getattr(drain, "_index_lock", None) or contextlib.nullcontext()
        with lock:
            live_before = len(index.live_req_ids()) if dbg else 0
            popped = index.pop_deliverable()
            for req_id, _ in popped:
                index.free(req_id)
            freed_target = []
            if free_external is not None:
                for rid in list(iter_matching_req_ids(index.live_req_ids(), free_external)):
                    index.free(rid)
                    freed_target.append(rid)
            if dbg:
                _aperture_disk_dbg(
                    f"host-free: popped={[r for r, _ in popped]} "
                    f"free_external={free_external!r}->{freed_target} "
                    f"live_before={live_before} live_after={len(index.live_req_ids())}")
        shard = _hs_layer_shard(self)
        for req_id, per_layer in popped:
            stash[str(req_id)] = _marshal_perreq_hs(per_layer, conf, shard)
        return stash

    def clear_aperture_request(self, external_req_id: str) -> None:
        """Abort cleanup: free all of an aborted request's HS aperture state, host and disk."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        clear_disk = getattr(drain, "clear_request_disk", None)
        if callable(clear_disk):
            clear_disk(external_req_id)
        index = getattr(drain, "index", None)
        if index is not None:
            mark_host = getattr(drain, "mark_host_aborted", None)
            if callable(mark_host):
                mark_host(external_req_id)
            self._drain_aperture_into_stash(drain, index, free_external=external_req_id)
            stash = getattr(self, "_aperture_perreq_stash", None)
            if stash:
                for rid in list(iter_matching_req_ids(stash, external_req_id)):
                    stash.pop(rid, None)
        return None

    def route_aperture_to_disk(self, req_id: str, dest: str) -> bool:
        """Route ``req_id`` to per-request disk staging, offloaded to ``dest`` when it finishes."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return False
        route = getattr(drain, "route_to_disk", None)
        if not callable(route):
            return False
        shard = _hs_layer_shard(self)
        if shard is not None:
            from mia.graph.tp_shard import rank_dir_name
            dest = os.path.join(str(dest), rank_dir_name(shard.tp_rank))
        _aperture_disk_dbg(f"worker.route_aperture_to_disk: req_id={req_id!r} (EXTERNAL) dest={dest!r}")
        route(str(req_id), str(dest))
        return True

    def confirm_aperture_delivery(self, req_id: str, timeout_s: float | None = None) -> bool | None:
        """Block until a disk-routed request's file has landed at the client ``dest``."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        offload = getattr(drain, "_offload", None)
        if offload is None:
            return None
        unstaged = getattr(drain, "finished_unstaged", None)
        if callable(unstaged) and unstaged(str(req_id)):
            return None
        ok = bool(offload.wait(str(req_id), timeout=timeout_s))
        if ok:
            unlink = getattr(drain, "unlink_delivered_source", None)
            if callable(unlink):
                unlink(str(req_id))
        return ok

    def aperture_residency(self):
        """Read-only residency query for the off-loop HS per-request path."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        lock = getattr(drain, "_index_lock", None) or contextlib.nullcontext()
        with lock:
            host_live = len(index.live_req_ids()) if index is not None else 0
        disk_fn = getattr(drain, "disk_residency", None)
        disk = int(disk_fn()) if callable(disk_fn) else 0
        return (int(host_live), int(disk))


    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve and remove captured hidden states for a completed request."""
        from mia.graph.drain import drain_barrier
        drain_barrier(self)
        consumer = getattr(self, "_capture_consumer", None)
        if consumer is not None:
            consumer.drain_writer_done(self)
        for req_id in iter_matching_req_ids(self._captured_states, external_req_id):
            layer_dict = self._captured_states.pop(req_id)
            if consumer is not None:
                consumer.on_pop(req_id, layer_dict)
            _census_bucket = None
            _census_acc = None
            if _CENSUS_ON:
                from mia.graph.census import census_bucket, new_accumulator
                _census_bucket = census_bucket(layer_dict)
                _census_acc = new_accumulator()
            cpu_dict = {}
            with PROF.timed("worker.cpu_transfer.hs"):
                for mod_name, entry in layer_dict.items():
                    mode = entry.get("hs_mode", self.hs_mode)
                    hs_qmeta = entry.get("_hs_qmeta")
                    if hs_qmeta is None:
                        with PROF.timed("cpu_transfer.hs.d2h"):
                            tensors = _cpu_list(entry["hidden_states"], _census_acc)
                        with PROF.timed("cpu_transfer.hs.pad"):
                            if mode == "last_token":
                                stacked = torch.stack(tensors)
                            else:
                                from torch.nn.utils.rnn import pad_sequence
                                stacked = pad_sequence(tensors, batch_first=True)
                        cpu_dict[mod_name] = {"hidden_states": stacked,
                                              "layer_num": entry["layer_num"], "hs_mode": mode}
                    else:
                        cpu_dict[mod_name] = {
                            "hidden_states": _cpu_list(entry["hidden_states"], _census_acc),
                            "hidden_states_scale": [s.cpu() if s is not None else None
                                                    for s in entry.get("_hs_scale", [])],
                            "hidden_states_qmeta": hs_qmeta,
                            "layer_num": entry["layer_num"], "hs_mode": mode}
            if _census_bucket is not None:
                from mia.graph.census import census_emit, census_record
                census_emit(census_record(worker="hs", sink="rpc", req_id=req_id,
                                           bucket=_census_bucket, acc=_census_acc))
            payload = {"hs_cache": cpu_dict, "config": self._conf}
            with PROF.timed("worker.compress.hs"):
                with PROF.timed("compress.hs.pickle"):
                    raw = pickle.dumps(payload)
                PROF.gauge("worker.raw_bytes.hs", len(raw))
                with PROF.timed("compress.hs.zstd"):
                    compressed = _ZSTD_COMPRESSOR.compress(raw)
            PROF.gauge("worker.compressed_bytes.hs", len(compressed))
            return compressed
        return None

    def dump_profiler(self) -> str | None:
        """Dump this worker's profiler snapshot to MIA_PROFILE_DIR; return the path or None."""
        from mia._profiler import PROF
        return PROF.dump(role="worker-rpc")

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured states without returning them (cleanup on abort/disconnect)."""
        consumer = getattr(self, "_capture_consumer", None)
        if consumer is None:
            clear_states_for_req(self._captured_states, external_req_id)
            clear_states_for_req(self._disk_states, external_req_id)
            return
        for bucket in (self._captured_states, self._disk_states):
            for req_id in iter_matching_req_ids(bucket, external_req_id):
                consumer.on_pop(req_id, bucket.pop(req_id))

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """Write captured hidden states for all requests in the batch to one artifact."""
        from mia.graph.drain import drain_barrier
        drain_barrier(self)
        consumer = getattr(self, "_capture_consumer", None)
        cpu_cache: dict = {"config": self._conf, "hs_cache": {}}
        found_any = False
        flushed_ids: list = []

        with PROF.timed("worker.cpu_transfer.hs"):
            for external_req_id in external_req_ids:
                for req_id in iter_matching_req_ids(self._disk_states, external_req_id):
                    layer_dict = self._disk_states.pop(req_id)
                    if consumer is not None:
                        consumer.on_pop(req_id, layer_dict, release_pages=False)
                    flushed_ids.append(req_id)
                    if not layer_dict:
                        continue
                    found_any = True
                    _census_bucket = None
                    _census_acc = None
                    if _CENSUS_ON:
                        from mia.graph.census import census_bucket, new_accumulator
                        _census_bucket = census_bucket(layer_dict)
                        _census_acc = new_accumulator()
                    for mod_name, entry in layer_dict.items():
                        hs_qmeta = entry.get("_hs_qmeta")
                        with PROF.timed("cpu_transfer.hs.d2h"):
                            _hs_hostlist = _cpu_list(entry["hidden_states"], _census_acc)
                        cpu_entry = {
                            "hidden_states": _hs_hostlist,
                            "layer_num": entry["layer_num"],
                            "hs_mode": entry.get("hs_mode", self.hs_mode),
                        }
                        if hs_qmeta is not None:
                            cpu_entry["hidden_states_scale"] = [
                                s.cpu() if s is not None else None
                                for s in entry.get("_hs_scale", [])]
                            cpu_entry["hidden_states_qmeta"] = hs_qmeta
                        if mod_name in cpu_cache["hs_cache"]:
                            existing = cpu_cache["hs_cache"][mod_name]
                            existing["hidden_states"].extend(cpu_entry["hidden_states"])
                            if hs_qmeta is not None:
                                existing.setdefault("hidden_states_scale", []).extend(
                                    cpu_entry["hidden_states_scale"])
                                existing.setdefault("hidden_states_qmeta", hs_qmeta)
                        else:
                            cpu_cache["hs_cache"][mod_name] = cpu_entry
                    if _census_bucket is not None:
                        from mia.graph.census import census_emit, census_record
                        census_emit(census_record(worker="hs", sink="disk", req_id=req_id,
                                                   bucket=_census_bucket, acc=_census_acc))

        if not found_any:
            if consumer is not None:
                consumer.drain_writer_done(self)
            return False

        from mia.graph.tp_shard import rank_dir_name
        tp_rank = _worker_tp_rank(self)
        run_dir = os.path.join(hook_dir, run_id, rank_dir_name(tp_rank))
        os.makedirs(run_dir, exist_ok=True)

        quant_on = any("hidden_states_qmeta" in e for e in cpu_cache["hs_cache"].values())
        wp = getattr(self, "_writer_process", None)
        use_st = os.environ.get("MIA_USE_SAFETENSORS", "0") == "1"
        if wp is not None:
            with PROF.timed("worker.queue_put"):
                submitted = wp.submit("hs", cpu_cache, run_dir, self.hs_mode, tp_rank,
                                      use_st, quant_on, "hidden_states.pt",
                                      req_ids=flushed_ids, block=True)
            if not submitted:
                from mia.graph.writer_process import note_submit_refused
                note_submit_refused(self, wp)
                compact_page_backed_cache(cpu_cache)
                if use_st and not quant_on:
                    self._save_safetensors(cpu_cache, run_dir)
                else:
                    save_pt_atomic(cpu_cache, os.path.join(run_dir, "hidden_states.pt"))
                if consumer is not None:
                    consumer.release_req_pages(flushed_ids)
        else:
            compact_page_backed_cache(cpu_cache)
            if use_st and not quant_on:
                self._save_safetensors(cpu_cache, run_dir)
            else:
                save_pt_atomic(cpu_cache, os.path.join(run_dir, "hidden_states.pt"))
            if consumer is not None:
                consumer.release_req_pages(flushed_ids)

        if consumer is not None:
            consumer.drain_writer_done(self)
        return run_dir

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        from mia.graph.artifact_writer import save_hs_cache_safetensors
        save_hs_cache_safetensors(cpu_cache, run_dir, self.hs_mode, _worker_tp_rank(self))

