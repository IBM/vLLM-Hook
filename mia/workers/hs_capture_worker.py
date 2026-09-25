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
    """Gated disk-pipeline debug logging (MIA_APERTURE_DEBUG=1). Off by default -> no perf impact.
    Read at call time so a per-worker env set before spawn takes effect."""
    if os.environ.get("MIA_APERTURE_DEBUG") == "1":
        print(f"[hookplugin/aperture-disk] {msg}", flush=True)

# Offload cost-attribution census: decomposes the flush D2H into allocation vs copy and
# censuses the popped bucket's tensor shape, BEFORE any .cpu() runs. Diagnostic only --
# never changes a captured value. Default OFF -> every call site below runs the plain
# `[t.cpu() for t in ...]` verbatim (see _cpu_list).
_CENSUS_ON = os.environ.get("MIA_CAPTURE_CENSUS") == "1"

# Batch the flush D2H (cat -> one pinned copy -> split) instead of one .cpu() per
# accumulated step-tensor. Default OFF -> _cpu_list runs the plain comprehension verbatim.
_BATCHED_FLUSH = os.environ.get("MIA_BATCHED_FLUSH") == "1"


def _cpu_list(tensors, acc: dict | None):
    """``[t.cpu() for t in tensors]``, routed through the census allocation/copy
    decomposition when ``acc`` is not None (MIA_CAPTURE_CENSUS=1). ``acc`` is None on
    the default path -> this is exactly ``[t.cpu() for t in tensors]``, unchanged."""
    if acc is not None:                       # census baseline: per-tensor decomposition
        from mia.graph.census import cpu_list_measured
        return cpu_list_measured(tensors, acc)
    if _BATCHED_FLUSH:                         # Tier 1: one pinned D2H for the whole list
        from mia.workers._common import cpu_list_batched
        return cpu_list_batched(tensors)
    return [t.cpu() for t in tensors]          # default: verbatim


def _worker_tp_rank(worker) -> int:
    """This worker's TENSOR-parallel rank (the ``tp_rank_<r>`` dir name), for eager and graph."""
    r = getattr(worker, "_tp_rank", None)
    if r is not None:
        return int(r)
    from mia.graph.tp_shard import resolve_tp_coords
    return resolve_tp_coords(worker)[0]


def _hs_layer_shard(worker):
    """This rank's ``HSShard`` when HS runs the round-robin TP layer shard, else None (TP = 1, the
    rank-0-only and all-ranks layouts, and the eager path, whose payloads stay unchanged)."""
    from mia.graph.tp_shard import HS_MODE_ROUND_ROBIN, HSShard
    if getattr(worker, "_hs_shard_mode", None) != HS_MODE_ROUND_ROBIN:
        return None
    conf = getattr(worker, "_conf", None) or {}
    return HSShard.of(int(getattr(worker, "_tp_rank", 0) or 0),
                      int(getattr(worker, "_hs_tp_size", 1) or 1),
                      int(conf.get("num_layers", 0) or 0))


def _marshal_perreq_hs(per_layer: dict, conf, shard=None) -> bytes:
    """Marshal ONE finished request's assembled per-layer tensors (the
    ``PerRequestIndex.pop_deliverable`` value: ``{layer_num(1-based): tensor}``) into the same
    driver-attachable payload ``get_captured_states`` returns, serialized to ZSTD-PICKLE BYTES.

    ``collective_rpc`` does NOT round-trip raw torch tensors (they arrive on the driver as plain
    Python lists), so every tensor payload MUST be bytes -- the same convention as
    ``get_captured_states`` / ``flush_aperture_per_request``.

    Payload SHAPE == ``get_captured_states``: ``{"hs_cache": {layer_num: {"hidden_states": tensor,
    "layer_num": layer_num}}, "config": conf}``. The value dict mirrors the eager path's
    ``{"hidden_states": tensor, "layer_num": int}`` (the aperture has no module names, so the outer key
    is ``layer_num`` -- the eager path's ``L+1``, the number the analyzer/oracle read off
    ``entry["layer_num"]``, so it lines up with NO remap). Each tensor is the FLAT per-token layout
    ``pop_deliverable`` produces (``torch.cat`` of the request's per-step demuxed row slices, in
    step order), moved to CPU float32.

    Under the TP layer shard (``shard`` given) each rank delivers only ITS layers of the request, so
    the payload also carries ``"hs_shard": {tp_rank, tp_size, num_layers, layer_shard,
    owned_layers}`` and the driver unions the ranks' parts (``mia._plugin.merge_probe_parts`` ->
    ``graph/tp_shard.merge_hs_payloads``). Without a shard the payload is unchanged."""
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
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (HSCaptureWorker,) at runtime,
    so self is the Worker instance. Methods are callable via collective_rpc.
    """

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    # Default capture phase — matches the old hooks_on=(True, False) registry entry.
    # Can be overridden per-request via extra_args["hooks_on"].
    _default_hooks_on: str = "prefill"

    # Per-request captured hidden states (API serving path):
    # internal_req_id -> {module_name -> {"hidden_states": [...], "layer_num": int}}
    _captured_states: dict = {}
    _hooks_installed: bool = False
    _step: "StepView | None" = None

    def install_hooks(self):
        """Install forward hooks on all target decoder layers. Idempotent.

        Callable via collective_rpc("install_hooks") — the plugin calls this
        lazily on the first request that sets output_hidden_states in extra_args.

        Fails loud (UnsupportedRunnerError) if the live runner is not vLLM's V2
        GPUModelRunner, BEFORE anything else runs — the failure mode this replaces is
        silent: a V1 runner used to capture nothing while reporting success.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        runner = self.model_runner
        require_v2_runner(runner)
        stash = install_request_arg_stash(runner)
        self._step = None

        # Snapshot the transient per-step InputBatch into an immutable StepView the
        # instant it's produced. Forward hooks read self._step; they never touch the
        # runner directly (V2's InputBatch is not stored on the runner).
        original_prepare = runner.prepare_inputs

        def prepare_inputs(*args, **kwargs):
            input_batch = original_prepare(*args, **kwargs)
            self._step = step_view(runner, input_batch, stash)
            return input_batch

        runner.prepare_inputs = prepare_inputs

        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}  # RPC path: req_id -> {module: {hidden_states, layer_num}}
        self._disk_states = {}      # disk path: req_id -> {module: {hidden_states, layer_num}} + {"_meta": {run_id, hook_dir}}
        model = getattr(runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # Worker-wide fallback when extra_args["hs_mode"] is missing.
        self.hs_mode = "last_token"

        # SHM path is a specialized same-machine transport, independent of the
        # per-request RPC/disk paths. Kept as-is for backward compat.
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

        # Only TP rank 0 captures: the residual stream is REPLICATED across TP ranks (every
        # row-parallel output is all-reduced before it is added back), so every rank holds the
        # same hidden state and one copy is the whole answer. (Contrast QK, whose q/k are
        # SHARDED by head and captured on every rank -- see mia/graph/tp_shard.py.) The TP LAYER
        # SHARD (MIA_HS_TP_SHARD) is a FULL-graph aperture feature (graph/install_hs.py): this
        # eager forward-hook path stays rank-0-only at every setting -- it has no aperture and no
        # drain to split, and the model-scale study runs FULL graphs only.
        from mia.graph.tp_shard import refuse_pipeline_parallel, resolve_tp_coords
        refuse_pipeline_parallel(getattr(self.parallel_config, "pipeline_parallel_size", 1),
                                 "HS install_hooks")
        tp_rank, _tp_size = resolve_tp_coords(self)
        self._tp_rank = tp_rank
        self._should_capture = tp_rank == 0

        # Writer PROCESS -- only on the capturing rank: a rank that never writes an artifact
        # never starts one (init_writer_process is idempotent on the attribute).
        from mia.graph.writer_process import init_writer_process, mark_no_writer
        if self._should_capture:
            init_writer_process(self)
        else:
            mark_no_writer(self, "HS sink rank: captures nothing")

        # Artifact quantization (opt-in): quantize the residual clone on GPU at
        # capture, dequantize in the worker at retrieval/flush. Default off = no-op.
        self._artifact_tag, self._artifact_gran, self._artifact_gsize = resolve_capture_quant("hs")

        cfg = model.config
        # Multimodal models (e.g. Qwen3.5) nest text config under text_config.
        text_cfg = getattr(cfg, "text_config", cfg)
        hidden_size = int(getattr(text_cfg, "hidden_size"))
        num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
        self._conf = {"hidden_size": hidden_size, "num_layers": num_layers}

        # Per-batch resolved request decisions, keyed by id(attn_metadata).
        # Each forward step produces a fresh attn_metadata, so the id is a
        # cheap fingerprint for "are we still in the same batch?" The previous
        # entry is dropped when a new batch arrives so the dict stays bounded.
        self._batch_cache_key = None
        self._batch_cache_entries = None  # list[dict] of resolved per-request info

        def _resolve_batch(step: StepView, bs, query_start_loc):
            """Build the per-batch list of capturing requests. Called once per
            forward step (on the first hook fire that gets here)."""
            entries = []
            for i in range(bs):
                req_id = step.req_ids[i]
                extra = step.extra_args_for(i)
                if not extra or extra.get("output_hidden_states") is None:
                    continue
                output_layers = extra.get("output_hidden_states")
                # Resolve hooks_on once. is_prefill check only matters when
                # hooks_on != "both"; is_prefilling_np is stable for the
                # duration of one forward step.
                hooks_on = extra.get("hooks_on", self._default_hooks_on)
                is_prefill = bool(step.is_prefilling_np[i])
                if hooks_on != "both":
                    if hooks_on == "prefill" and not is_prefill:
                        continue
                    if hooks_on == "decode" and is_prefill:
                        continue

                # With chunked-prefill, in last_token mode, only capture on the final chunk of the
                # prefill, i.e. when computed-after-step reaches the PROMPT length (prompt_len_np,
                # not prefill_len_np -- prefill_len can exceed the prompt after preemption-resume).
                req_mode = extra.get("hs_mode", self.hs_mode)
                if is_prefill and req_mode == "last_token":
                    chunk_len = int(query_start_loc[i + 1].item() - query_start_loc[i].item())
                    if int(step.num_computed_tokens_np[i]) + chunk_len < int(step.prompt_len_np[i]):
                        # Mid-prefill chunk doesn't need capture
                        continue

                entries.append({
                    "i": i,
                    "req_id": req_id,
                    "output_layers": output_layers,  # list or None ("all layers")
                    "hs_mode": req_mode,
                    "save_to_disk": bool(extra.get("save_to_disk")),
                })
            return entries

        def hs_hook(output, module_name, layer_num):
            # Fast-path: only rank 0 captures (RPC and disk paths both need it).
            if not self._should_capture:
                return None

            step = self._step
            if step is None:
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)

            # Warmup or non-attention passes: nothing to do
            if metadata is None:
                return
            if torch.cuda.is_current_stream_capturing():
                return None

            # Reuse the resolved-request list across all hook fires within
            # the same forward step. id(metadata) is a stable fingerprint
            # because vLLM allocates fresh attn_metadata per step.
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

            # Layer filter: any request want THIS layer?
            wanted = []
            for e in entries:
                ol = e["output_layers"]
                if isinstance(ol, list) and (layer_num not in ol):
                    continue
                wanted.append(e)
            if not wanted:
                return

            last_indices = self._batch_cache_qsl

            # vLLM uses a fused residual pattern: transformer blocks return
            # (hidden_states, residual) where the residual has not yet been added. 
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

                # Accumulate GPU tensors — clone() copies data immediately so we
                # own the buffer; .cpu() is deferred to the retrieval/flush call.
                if req_mode == "last_token":
                    activation = hidden[end - 1].detach().clone()
                else:
                    activation = hidden[start:end].detach().clone()

                # Optional on-GPU quantization (default off → packed is the clone unchanged).
                activation, hs_scale, hs_qmeta = quant_clone(
                    activation, self._artifact_tag, self._artifact_gran, self._artifact_gsize)

                PROF.incr("hook.fire.hs")
                PROF.gauge("captured.bytes.hs", capture_bytes(activation, hs_scale))

                # Route to disk or RPC bucket based on save_to_disk flag.
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

        # Hook every decoder layer. Per-request layer filtering via
        # extra_args['output_hidden_states'] happens inside the hook closure.
        # Note: layer_num returned by match_layer is the 0-based PyTorch index
        # (model.layers.N-1); we expose 1-based numbers (HuggingFace/Eagle
        # convention: layer N = output after the Nth transformer block) by adding 1.
        self._hooks = []
        matched = []
        for name, module, layer_num in iter_matched_modules(model, match_layer):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name, ln=layer_num+1: hs_hook(o, n, ln)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} hidden-state hooks on layers: {matched}")

    # ------------------------------------------------------------------
    # v0.3.0 CUDA-graph capture install (graph mode only)
    # ------------------------------------------------------------------

    def graph_install(self):
        """Install the CUDA-graph hidden-state capture path (buffer mode).

        Thin delegating entry called by the Worker.load_model monkey-patch
        (graph/install.py:patch_worker_load_model) AFTER the model is built but
        BEFORE warm-up/compile/capture. No-op on the eager path: only reached
        when graph mode is armed.

        Seeds the egress buckets as per-instance dicts so the graph capture body
        can populate them and get_captured_states / flush_disk / _save_safetensors
        consume them UNCHANGED — REUSE CONTRACT: the graph installers write the SAME
        worker buckets the eager path writes, which is why retrieval/flush behave
        identically in both modes. Delegates to graph.install_hs, which builds the
        per-layer static buffers and wraps the decoder-layer class to emit the
        capture_hs scatter op (absorbed into the decode cudagraph).
        """
        if not getattr(self, "_captured_states", None):
            self._captured_states = {}
        if not getattr(self, "_disk_states", None):
            self._disk_states = {}
        from mia.graph.install_hs import (
            install_execute_model_wrapper_hs,
            install_hs_hosts,
        )
        # Capture-aperture path: scatter -> aperture -> drain -> disk. Builds self._capture_aperture +
        # self._hs_drain; retrieval/flush go through the aperture path below (no separate bank).
        # Retrieval is off-loop, reading durable files via
        # aperture_reader.load_multilayer_aperture_artifact.
        install_hs_hosts(self)
        install_execute_model_wrapper_hs(self.model_runner, self)

    def flush_aperture(self) -> str | None:
        """collective_rpc-callable: final drain + write the capture-aperture metadata sidecar; return
        the per-worker run_dir (None if the aperture path is not installed).

        The capture-aperture path writes durable per-layer raw files continuously (per step); this
        flushes any last pending rows and writes the shared sidecar so the reader can reconstruct.
        Call once after all requests finish (the worker process is often killed rather than joined,
        so the atexit backstop is not reliable — this RPC is the durable-flush contract).

        Handles both drains: the off-loop consumer thread (``stop()`` drains its queue + joins,
        surfacing any consumer error) and the synchronous drain (``drain_once``). ``stop()`` is
        idempotent, so a duplicate flush is safe."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None:
            return None          # a non-capturing TP rank: no drain, no dir (see install_hs)
        stop = getattr(drain, "stop", None)
        if callable(stop):
            drain.stop()         # off-loop: drain the queue + join the consumer thread (raises on error)
        else:
            drain.drain_once()   # synchronous path: flush any last pending rows on the loop
        drain.close()            # write the shared sidecar (idempotent)
        # Only a dir that HOLDS captured rows is returned, so
        # `[d for d in collective_rpc("flush_aperture") if d]` is exactly the dirs to read.
        from mia.graph.tp_shard import drain_holds_data
        if not drain_holds_data(drain):
            return None
        return getattr(self, "_hs_run_dir", None)

    def get_drain_row_counts(self) -> dict:
        """collective_rpc-callable read-only diagnostic: how many aperture rows THIS worker's HS drain
        actually copied (`hs.drain.rows_copied`) vs. how many an unconditional full drain of the
        same steps would have copied but this one did not (`hs.drain.rows_skipped`), cumulative
        since install, plus whether selective drain (`MIA_DRAIN_SELECTIVE`, default ON,
        `=0` to disable) is in force and why it was refused if armed.

        `hs.drain.degenerate_steps`: steps where every installed layer was wanted over the whole
        span, so the armed lever took the flag-off fast path -- the only field that separates
        "fast path fired" from "never fired" on an all-layers workload, where `rows_skipped`
        reads 0 either way. `rows_copied` is counted where the copies are issued, so it can't
        agree with a plan that was never followed -- stronger than a byte-compare alone, which a
        subset run that silently copied every layer anyway would still pass. All zeros when no
        aperture drain is installed."""
        from mia.graph.install_hs import get_drain_row_counts
        return get_drain_row_counts(self)

    def flush_aperture_per_request(self):
        """collective_rpc-callable TEST read-hook for the per-request aperture-delivery parity oracle.
        SEPARATE from ``flush_aperture`` (the shared-file durable path) so that path stays
        byte-identical — never runs unless per-request delivery is armed.

        Drives end-of-run delivery on the OFF-LOOP per-request drain: ``drain.stop()`` drains the
        queue and joins the consumer, then ``finalize_all()`` marks every still-live request
        finished so last-step stragglers become deliverable; ``index.pop_deliverable()`` pops each
        finished request's per-layer assembled tensors (``torch.cat`` of its per-step demuxed row
        slices, in step order == the eager path's flat per-token layout), moved to CPU float32;
        then ``index.free(req_id)`` removes each delivered request so residency can reach 0.

        Returns ZSTD-COMPRESSED PICKLE BYTES of ``(deliverables, residency_after)`` (mirrors
        ``get_captured_states`` — ``collective_rpc`` does not round-trip raw tensors, so any tensor
        payload must be serialized to bytes). ``deliverables`` = ``{req_id: {layer_num(1-based):
        cpu_f32_tensor}}``, lining up with the eager probes with NO remap. ``residency_after`` MUST
        be 0 — the oracle asserts no request is left un-delivered / un-freed. A second call after
        everything is popped+freed serializes ``({}, 0)`` (idempotent). Under the TP layer shard
        each rank returns ITS layers of each request; union them across ranks with
        ``mia.graph.tp_shard.merge_hs_layer_maps`` (the production RPC path does that itself, in
        ``_plugin.merge_probe_parts``).

        Strict no-op -> ``None`` when the capture-aperture path is not installed OR per-request mode is
        off, so it can never perturb the shared-file / QK / eager paths. TP=1 in scope."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        if index is None:
            return None
        stop = getattr(drain, "stop", None)
        if callable(stop):
            # off-loop: drain the queue + join the consumer + finalize_all (last-step stragglers
            # become deliverable); raises if the consumer thread died.
            drain.stop()
        # pop + free + residency under the drain's lock as ONE section (uniformity — this path is
        # already race-free because stop() joined the consumer, so it is uncontended here, but the
        # index-access discipline is the same on every side). Marshal to cpu/f32 OUTSIDE the lock.
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
        # Serialize to bytes exactly like get_captured_states -- collective_rpc drops raw torch
        # tensors (they arrive on the driver as lists). The driver unpickles the tuple back.
        raw = pickle.dumps((deliverables, residency_after))
        return _ZSTD_COMPRESSOR.compress(raw)

    def get_aperture_per_request(self, external_req_id: str) -> bytes | None:
        """collective_rpc-callable PRODUCTION per-request retrieval for the off-loop HS
        capture-aperture demux path. Returns this request's marshaled probes as ZSTD-PICKLE BYTES
        (mirroring ``get_captured_states`` -- ``collective_rpc`` drops raw tensors, so tensor
        payloads must be serialized), or ``None`` when the request has not been delivered yet
        (still generating / its off-loop finish not yet processed).

        BULK-POP-INTO-STASH: ``pop_deliverable`` is a BULK drain -- it returns EVERY
        currently-finished request AND clears the deliverable list, so a single call cannot
        fetch just the asked-for one without discarding the rest. Instead, drain ALL
        currently-finished requests ONCE, marshal each into a per-req_id STASH of bytes,
        ``index.free`` each so residency drops, then return + remove the asked-for request's
        stashed bytes -- the stash keeps the others for their own later calls, so each request
        is delivered exactly once. A request appears here only once its assembly is complete
        (the consumer thread marks it finished after all its rows are noted).

        Strict no-op -> ``None`` when the capture-aperture path is not installed OR per-request mode is
        off, so it NEVER perturbs the bank / shared-file / QK / eager paths, nor
        ``get_captured_states`` / ``flush_aperture`` / ``flush_aperture_per_request``. TP=1 in scope."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        if index is None:
            return None
        stash = self._drain_aperture_into_stash(drain, index)
        # Serve + remove the asked-for request (exact match or "{external}-{suffix}", the v1/legacy
        # id rule); None if it has not been delivered into the stash yet.
        match = next(iter(iter_matching_req_ids(stash, external_req_id)), None)
        if match is None:
            return None
        return stash.pop(match)

    def _drain_aperture_into_stash(self, drain, index, free_external: str | None = None) -> dict:
        """Bulk-drain EVERY currently-finished aperture request into ``self._aperture_perreq_stash`` (bytes),
        returning the stash. Shared by ``get_aperture_per_request`` (retrieval) and ``clear_aperture_request``
        (abort cleanup) so BOTH drain the index identically and race-safe. Still-generating requests
        stay in the index for a later call.

        POP + FREE-POPPED + the optional FREE-TARGET (``free_external`` — the abort case's target
        request) run under the DRAIN'S ``_index_lock`` as ONE atomic section, because this runs on the
        engine/retrieval (or abort) thread while the off-loop consumer thread concurrently mutates the
        SAME PerRequestIndex (note_rows / mark_finished). Splitting the target-free into a separate
        lock section would let a concurrent ``mark_finished`` land in the gap and either (a) strand a
        request that was mid-delivery, never re-entering ``_deliverable`` (lost, client hangs), or (b)
        leave an aborting request finished in ``_entries`` after its target-free already removed it,
        so the next ``pop_deliverable`` KeyErrors and wedges the whole host RPC delivery path. Folding
        both into one section closes the gap: whichever side wins, the request ends up freed exactly
        once and consistently. A later ``_handle_finish(R)`` for a freed R is a no-op — it guards on
        ``live_req_ids()`` — so a ``mark_finished`` that lands after this section has already freed R
        cannot resurrect it.

        Under the lock, pop_deliverable's per-request torch.cat DOES run (deliberate and safe — these
        are CPU-only cats of ALREADY-CLONED host tensors: no D2H, no GPU sync, no I/O — and the cat
        must stay atomic with the pop+free for the same reason above). Only the heavier MARSHAL
        (compress/pickle) runs AFTER releasing the lock, so the critical section never blocks on
        serialization."""
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
        """collective_rpc-callable ABORT/disconnect cleanup for the off-loop HS capture-aperture
        per-request path: free ALL of an aborted request's aperture state so residency eventually
        returns to 0 -- the host-buffer ``PerRequestIndex`` entry + any stashed bytes AND the disk
        staging (+ a delivered-but-unconfirmed source dir). ``clear_captured_states`` clears only
        the bank / eager buckets, which the aperture path never uses, so this is the aperture path's own
        cleanup and the driver calls BOTH on abort.

        Strict no-op -> None when the aperture per-request path is not installed. TP=1, internal
        req_id == external, matched via ``iter_matching_req_ids`` (exact + legacy ``{external}-``
        suffix), except the disk maps, which ``clear_request_disk`` keys exactly.

        DISK route runs on the ENGINE thread and must NOT delete a staging dir the off-loop
        consumer may still be demuxing into (a ``rmtree``-vs-``open`` race killed the consumer
        once). SINGLE-OWNER lifecycle: this call only MARKS the request aborted and drops its
        route; the CONSUMER thread alone creates, writes, and deletes the staging dir, skipping a
        marked request's remaining writes and discarding the dir on that request's finish (or at
        ``finalize_all``). A delivered-but-unconfirmed source dir (finish already ran) is safe to
        reclaim here off-lock, since by FIFO the consumer is done writing it.

        HOST route: the shared ``_drain_aperture_into_stash`` drains AND frees this request's
        still-live entry under ONE ``_index_lock`` hold (``free_external``), so a concurrent
        ``mark_finished`` can never strand it in ``_deliverable`` while it is freed from
        ``_entries``."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        # DISK route: MARK aborted (single-owner: the consumer discards the staging dir on the _Finish)
        # + reclaim any delivered-but-unconfirmed source. No engine-thread rmtree of a live staging dir.
        # No-op if it was never disk-routed / already fully reclaimed.
        clear_disk = getattr(drain, "clear_request_disk", None)
        if callable(clear_disk):
            clear_disk(external_req_id)
        # HOST route: bulk-drain deliverables into the stash AND free this request's still-live entry
        # in ONE atomic _index_lock section (free_external). A concurrent consumer mark_finished for
        # this request can no longer land between the drain and the target free -> no stranded
        # _deliverable / KeyError wedge. Marshal + stash pop run off-lock.
        index = getattr(drain, "index", None)
        if index is not None:
            # Mark the request HOST-aborted BEFORE freeing it: once marked, the off-loop consumer
            # skips (re-)staging any backlogged/in-flight drained rows into the PerRequestIndex, so
            # free_external below removes the entry for good instead of it being re-note_rows'd and
            # stranded. mark_host_aborted only marks a request with LIVE host state, so a
            # normally-completed request (already delivered) is not marked -> the set stays
            # bounded; the request's _Finish drops the mark.
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
        """collective_rpc-callable SEAM for the router: mark ``req_id`` for the per-request DISK
        route on the off-loop drain — its rows stream to their own NVMe run_dir and, on finish, the
        file is offloaded to ``dest`` — instead of the host-buffer RPC path. Must be called at
        request-start (before the request's rows are drained). Returns True when the route was
        registered, False when the aperture per-request path is not installed (strict no-op — never
        perturbs the shared-file / QK / eager paths). TP=1 in scope."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return False
        route = getattr(drain, "route_to_disk", None)
        if not callable(route):
            return False
        # Under the TP layer shard every rank stages and offloads ITS layers of the request; each
        # lands in its own `dest/tp_rank_<r>/` (as QK's head shards do) so the ranks never overwrite
        # each other's sidecar, and `aperture_reader.load_hs_aperture_tp(dest, expected_layers=...)`
        # unions them. Every other layout keeps `dest` itself, unchanged.
        shard = _hs_layer_shard(self)
        if shard is not None:
            from mia.graph.tp_shard import rank_dir_name
            dest = os.path.join(str(dest), rank_dir_name(shard.tp_rank))
        _aperture_disk_dbg(f"worker.route_aperture_to_disk: req_id={req_id!r} (EXTERNAL) dest={dest!r}")
        route(str(req_id), str(dest))
        return True

    def confirm_aperture_delivery(self, req_id: str, timeout_s: float | None = None) -> bool | None:
        """collective_rpc-callable CONFIRM for a disk-routed request: block until its
        per-request file has actually landed at the client ``dest`` via the OffloadProcess, so the
        client can read it. Returns True on delivery, False on timeout / retry-exhausted give-up,
        None when the aperture per-request path is not installed. ``bool``/``None`` round-trip fine over
        ``collective_rpc`` (no tensor payload). ``req_id`` is the EXTERNAL ``request_id`` (the driver's
        ``_await_aperture_disk_confirm`` passes the request's external id) -- the SAME key the offload job
        was submitted under (``route_to_disk``/``_handle_finish`` register + submit on the external id),
        so ``offload.wait(req_id)`` matches. The row-level internal->external divergence is resolved
        earlier, inside the drain (``_match_disk_route``); it never reaches this confirm."""
        drain = getattr(self, "_hs_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        offload = getattr(drain, "_offload", None)
        if offload is None:
            return None
        # TP layer shard: a request that wanted none of THIS rank's layers finished here with nothing
        # staged, so there is nothing of it for this rank to deliver -- answer None ("not mine"), as
        # a rank without the path does, never False ("not landed"), which the driver would wait out.
        unstaged = getattr(drain, "finished_unstaged", None)
        if callable(unstaged) and unstaged(str(req_id)):
            return None
        ok = bool(offload.wait(str(req_id), timeout=timeout_s))
        if ok:
            # Delivered: reclaim the SERVER-side staging SOURCE dir -- the durable CLIENT dest
            # copy is kept. Runs on the worker's RPC thread, off the engine forward.
            unlink = getattr(drain, "unlink_delivered_source", None)
            if callable(unlink):
                unlink(str(req_id))
        return ok

    def aperture_residency(self):
        """collective_rpc-callable READ-ONLY residency query for the off-loop HS capture-aperture
        per-request path. Returns ``(host_live_count, disk_residency)`` -- the number of requests
        still holding a host-buffer ``PerRequestIndex`` entry and the number still holding
        per-request DISK staging -- WITHOUT stopping the drain, popping, or freeing anything, so
        it can be polled MID-serving.

        ``len(index.live_req_ids())`` is read under the drain's ``_index_lock``; ``disk_residency()``
        acquires that SAME lock itself and ``threading.Lock`` is non-reentrant, so it is called
        OUTSIDE the hold -- two short reads, never a nested acquire (nesting would deadlock the
        drain). The pair is a monitoring snapshot, not one atomic transaction, but at quiescence
        both read 0 regardless of interleaving.

        Strict no-op -> ``None`` when the capture-aperture path is not installed OR per-request mode is
        off. TP=1 in scope."""
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

    # ------------------------------------------------------------------
    # API serving: collective_rpc-callable artifact retrieval
    # ------------------------------------------------------------------

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve and remove captured hidden states for a completed request.

        Matches either by exact equality (vLLM v0.12+ uses the same id internally)
        or by "{external_req_id}-" prefix (older versions append a random suffix).

        CPU transfer happens here (once per request, not per hook).
        Returns zstd-compressed pickle, or None if nothing was captured.
        """
        from mia.graph.drain import drain_barrier
        drain_barrier(self)  # wait for any pending streaming drain before reading buckets
        consumer = getattr(self, "_capture_consumer", None)
        if consumer is not None:
            consumer.drain_writer_done(self)  # recycle any disk-path pages the feeder packed
        for req_id in iter_matching_req_ids(self._captured_states, external_req_id):
            layer_dict = self._captured_states.pop(req_id)
            # Release the request's resident bytes (counted at egress/stream) on pop.
            if consumer is not None:
                consumer.on_pop(req_id, layer_dict)
            # Offload cost-attribution measurement (MIA_CAPTURE_CENSUS=1): a pure
            # structural census of the bucket BEFORE any .cpu() below, plus a shared
            # per-request accumulator that _cpu_list() feeds when not None. Both stay None
            # on the default path -> zero-cost, verbatim.
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
                        # native: stack to the RPC format (unchanged behaviour).
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
                        # Hand off QUANTIZED; the driver dequantizes at analysis.
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
        """collective_rpc-callable: dump this WORKER process's PROF snapshot to
        MIA_PROFILE_DIR and return the path (None if profiling is off). The
        routing/egress timers live in the worker, not the driver, so the offline driver
        cannot read them otherwise. Read-only introspection, string-name callable."""
        from mia._profiler import PROF
        return PROF.dump(role="worker-rpc")

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured states without returning them (cleanup on abort/disconnect)."""
        consumer = getattr(self, "_capture_consumer", None)
        if consumer is None:
            clear_states_for_req(self._captured_states, external_req_id)
            clear_states_for_req(self._disk_states, external_req_id)
            return
        # Release resident bytes before dropping each bucket so aborts don't leak
        # residency (which would eventually starve admission). BOTH buckets: a
        # disk-mode abort must release too, else residency ratchets up monotonically.
        for bucket in (self._captured_states, self._disk_states):
            for req_id in iter_matching_req_ids(bucket, external_req_id):
                consumer.on_pop(req_id, bucket.pop(req_id))

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """Write captured hidden states for all requests in the batch to one artifact.

        Accepts a list of external_req_ids so all requests sharing a run_id
        are merged into one cpu_cache before writing — matching the old
        execute_model() behavior where the full batch was saved atomically.

        Returns this rank's artifact dir (``<hook_dir>/<run_id>/tp_rank_<r>``, truthy) when it wrote
        an artifact or handed one to its writer child, False when this rank captured nothing. The
        driver's durability barrier waits for EVERY dir the collective returns: with the writer
        process on every TP rank each shard lands on its own writer's schedule, so "the run dir is
        non-empty" means only that the FIRST rank landed (``_plugin._flushed_rank_dirs``).

        REUSE-AFTER-FREE: this pops each request's bucket (and its aperture pages, if any)
        BEFORE the writer feeder has copied the bytes out, so ``on_pop`` is called with
        ``release_pages=False`` here — pages are released later, once the feeder signals it has
        copied them (see ``consumer.drain_writer_done``), never immediately.
        """
        from mia.graph.drain import drain_barrier
        drain_barrier(self)  # wait for any pending streaming drain before reading buckets
        consumer = getattr(self, "_capture_consumer", None)
        cpu_cache: dict = {"config": self._conf, "hs_cache": {}}
        found_any = False
        flushed_ids: list = []  # req_ids popped this flush -> whose pages we deferred releasing

        with PROF.timed("worker.cpu_transfer.hs"):
            for external_req_id in external_req_ids:
                for req_id in iter_matching_req_ids(self._disk_states, external_req_id):
                    layer_dict = self._disk_states.pop(req_id)
                    # Disk path: defer page release until the writer feeder copies the bytes
                    # out (release_pages=False) — releasing here would let a later stream()
                    # overwrite pages the feeder still reads.
                    if consumer is not None:
                        consumer.on_pop(req_id, layer_dict, release_pages=False)
                    flushed_ids.append(req_id)
                    if not layer_dict:
                        continue
                    found_any = True
                    # Offload cost-attribution measurement (MIA_CAPTURE_CENSUS=1): see
                    # the matching comment in get_captured_states. Both stay None (verbatim,
                    # zero cost) on the default path.
                    _census_bucket = None
                    _census_acc = None
                    if _CENSUS_ON:
                        from mia.graph.census import census_bucket, new_accumulator
                        _census_bucket = census_bucket(layer_dict)
                        _census_acc = new_accumulator()
                    for mod_name, entry in layer_dict.items():
                        # Keep quantized onto disk; the disk loader dequantizes at read. Native
                        # (hs_qmeta None) stores float lists exactly as before.
                        hs_qmeta = entry.get("_hs_qmeta")
                        # Split the on-loop flush cost into D2H (_cpu_list) vs the rest (marshal
                        # ~= dict/extend): marshal.hs ~= cpu_transfer.hs - cpu_transfer.hs.d2h.
                        # Byte-identical; inert when profiling is off. Mirrors the
                        # get_captured_states RPC-path split.
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
                consumer.drain_writer_done(self)  # recycle any already-packed pages anyway
            return False

        from mia.graph.tp_shard import rank_dir_name
        tp_rank = _worker_tp_rank(self)
        run_dir = os.path.join(hook_dir, run_id, rank_dir_name(tp_rank))
        os.makedirs(run_dir, exist_ok=True)

        # Quantized cache -> .pt (packed uint8 + scale + qmeta don't fit fixed-shape safetensors).
        quant_on = any("hidden_states_qmeta" in e for e in cpu_cache["hs_cache"].values())
        # Hand serialize+write to a separate PROCESS (off the engine GIL) when armed. submit() is
        # NON-BLOCKING and returns False if the child is gone or the queue is full -> fall through
        # to the SAME thread/inline ladder below, so a dead/backed-up child never hangs the loop
        # or silently loses the artifact.
        wp = getattr(self, "_writer_process", None)
        use_st = os.environ.get("MIA_USE_SAFETENSORS", "0") == "1"
        if wp is not None:
            with PROF.timed("worker.queue_put"):
                submitted = wp.submit("hs", cpu_cache, run_dir, self.hs_mode, tp_rank,
                                      use_st, quant_on, "hidden_states.pt",
                                      req_ids=flushed_ids, block=True)
            if not submitted:  # child dead / bounded wait timed out -> data-safety inline (rare)
                from mia.graph.writer_process import note_submit_refused
                note_submit_refused(self, wp)   # LOUD: never a silent switch to inline saves
                # The inline fallback serializes cpu_cache directly (unlike the writer-process
                # pack, which torch.cat's into fresh storage) -- compact page-backed views to
                # owned storage first so pickle/torch.save doesn't re-serialize a whole aperture
                # page per narrow view.
                compact_page_backed_cache(cpu_cache)
                if use_st and not quant_on:
                    self._save_safetensors(cpu_cache, run_dir)
                else:
                    save_pt_atomic(cpu_cache, os.path.join(run_dir, "hidden_states.pt"))
                if consumer is not None:
                    consumer.release_req_pages(flushed_ids)  # cloned + written -> safe now
            # else: submitted -> pages released later by the writer's pack-done signal.
        else:
            # writer process OFF (MIA_WRITER_PROCESS=0): inline save. Same compaction
            # rationale as the fallback above -- this path never runs off-loop.
            compact_page_backed_cache(cpu_cache)
            if use_st and not quant_on:
                self._save_safetensors(cpu_cache, run_dir)
            else:
                save_pt_atomic(cpu_cache, os.path.join(run_dir, "hidden_states.pt"))
            if consumer is not None:
                consumer.release_req_pages(flushed_ids)

        if consumer is not None:
            consumer.drain_writer_done(self)  # recycle any pages the feeder already packed
        return run_dir

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        """Write cpu_cache as safetensors + JSON sidecar. The serialize body is a PURE
        function (graph/artifact_writer) so the writer PROCESS and this inline path
        serialize byte-identically; this wrapper supplies self.hs_mode + tp_rank."""
        from mia.graph.artifact_writer import save_hs_cache_safetensors
        save_hs_cache_safetensors(cpu_cache, run_dir, self.hs_mode, _worker_tp_rank(self))

