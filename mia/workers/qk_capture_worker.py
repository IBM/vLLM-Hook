"""Q/K capture worker (capture_qk): eager hooks and the CUDA-graph aperture path."""
import contextlib
import os
import math
import pickle
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd
from torch.nn.utils.rnn import pad_sequence
from vllm.forward_context import get_forward_context

from mia._profiler import PROF
from mia.errors import MiaConfigurationError
from mia.artifacts import write_refused_qk
from mia.core.runner import StepView, install_request_arg_stash, require_v2_runner, step_view
from mia.workers._common import (
    capture_bytes,
    clear_rank_artifact,
    clear_states_for_req,
    compact_page_backed_cache,
    cpu_list_batched,
    get_query_metadata,
    iter_matched_modules,
    iter_matching_req_ids,
    match_attn,
    match_internal_ids,
    quant_clone,
    resolve_capture_quant,
    save_pt_atomic,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)

_COMPACT_KALL_ENV = os.environ.get("MIA_QK_COMPACT_KALL")


def _use_compact_kall(entry: dict) -> bool:
    if _COMPACT_KALL_ENV == "0":
        return False
    if _COMPACT_KALL_ENV == "1":
        return True
    pe = entry.get("k_prefix_ends")
    return bool(pe) and len(pe) >= 2


_CENSUS_ON = os.environ.get("MIA_CAPTURE_CENSUS") == "1"

_BATCHED_FLUSH = os.environ.get("MIA_BATCHED_FLUSH") == "1"


def _cpu_list(tensors, acc: dict | None):
    if acc is not None:
        # lazy: keep mia.core.runtime (reads env at import) out of import mia
        from mia.core.runtime.census import cpu_list_measured
        return cpu_list_measured(tensors, acc)
    if _BATCHED_FLUSH:
        return cpu_list_batched(tensors)
    return [t.cpu() for t in tensors]


class CachedKeysUnavailable(RuntimeError):
    """A request's cached prefix keys cannot be read from the KV cache."""


def key_cache_from_layer_kv(kv_cache, num_kv_heads=None, head_size=None):
    """Return the KEY cache as ``[num_blocks, block_size, num_kv_heads, head_size]``."""
    kv = kv_cache
    if isinstance(kv, (list, tuple)) and len(kv) == 1 and hasattr(kv[0], "ndim"):
        kv = kv[0]
    if isinstance(kv, (list, tuple)) and len(kv) == 2 and hasattr(kv[0], "ndim"):
        return kv[0]
    if not hasattr(kv, "ndim"):
        raise CachedKeysUnavailable(f"the layer's kv_cache is a {type(kv).__name__}")
    if kv.ndim == 5:
        if kv.shape[1] == 2:
            return kv[:, 0]
        if kv.shape[0] == 2:
            return kv[0]
    if kv.ndim == 4:
        if num_kv_heads is None or head_size is None:
            raise CachedKeysUnavailable("the attention layer has no num_kv_heads / head_size")
        h, d = int(num_kv_heads), int(head_size)
        if kv.shape[1] == h and kv.shape[3] > d:
            return kv[..., :d].transpose(1, 2)
        if kv.shape[2] == h and kv.shape[3] == d:
            return kv
    raise CachedKeysUnavailable(f"unknown KV cache layout {tuple(kv.shape)} for "
                                f"num_kv_heads={num_kv_heads} head_size={head_size}")


def _read_cached_keys(
    module_name,
    attn_metadata,
    req_idx: int,
    num_cached: int,
    total_len: int,
):
    try:
        ctx = get_forward_context()
        layer = ctx.no_compile_layers[module_name]
        kv_cache = layer.kv_cache
        key_cache = key_cache_from_layer_kv(kv_cache, getattr(layer, "num_kv_heads", None),
                                            getattr(layer, "head_size", None))

        num_blocks   = key_cache.shape[0]
        block_size   = key_cache.shape[1]
        row_width    = math.prod(key_cache.shape[2:])

        block_table = attn_metadata.block_table
        num_blocks_needed = math.ceil(total_len / block_size)
        block_ids = block_table[req_idx, :num_blocks_needed]

        if block_ids.numel() == 0 or int(block_ids.max()) >= num_blocks \
                or int(block_ids.min()) < 0:
            raise CachedKeysUnavailable(f"block ids out of range for {num_blocks} blocks")

        prefix_keys = key_cache[block_ids].reshape(-1, row_width)

        return prefix_keys[:num_cached].detach()
    except CachedKeysUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        raise CachedKeysUnavailable(repr(e)) from None


def _prepend_cached_keys(module_name, metadata, i: int, k, seq_lens, start: int, end: int,
                         num_computed: int = 0):
    if seq_lens is None:
        if num_computed > 0:
            raise CachedKeysUnavailable("the attention metadata has no seq_lens")
        return k
    try:
        total_len = int(seq_lens[i].item()) if hasattr(seq_lens[i], "item") else int(seq_lens[i])
    except Exception as e:  # noqa: BLE001
        raise CachedKeysUnavailable(f"sequence length unreadable ({e!r})") from None
    num_cached = total_len - (end - start)
    if num_cached <= 0:
        return k
    PROF.incr("kv.prefix_recon")
    md = metadata.get(module_name) if isinstance(metadata, dict) else metadata
    if md is None:
        raise CachedKeysUnavailable("no attention metadata for this layer")
    with PROF.timed("kv.prefix_recon"):
        prefix_k = _read_cached_keys(module_name, md, i, num_cached, total_len)
    if tuple(prefix_k.shape) != (num_cached, k.shape[-1]):
        raise CachedKeysUnavailable(f"{tuple(prefix_k.shape)} cached key rows, expected "
                                    f"({num_cached}, {k.shape[-1]})")
    return torch.cat([prefix_k.to(k.device, dtype=k.dtype), k], dim=0)


def _k_all_cpu_list(entry: dict, _census_acc: dict | None = None) -> list:
    prefix_ends = entry.get("k_prefix_ends")
    parts = _cpu_list(entry["k_all"], _census_acc)
    if not prefix_ends:
        return parts
    full = torch.cat(parts, dim=0)
    return [full[:int(L)] for L in prefix_ends]


def _k_all_compact(entry: dict, _census_acc: dict | None = None):
    prefix_ends = entry.get("k_prefix_ends")
    if not prefix_ends:
        return None
    full = torch.cat(_cpu_list(entry["k_all"], _census_acc), dim=0)
    return full, [int(L) for L in prefix_ends]


def new_qk_entry(layer_num: int, mode: str, q_qmeta=None, k_qmeta=None) -> dict:
    """A fresh per-(request, layer) QK capture entry for the eager hook."""
    entry = {"q": [], "k_all": [], "layer_num": layer_num, "hookq_mode": mode}
    if q_qmeta is not None:
        entry.update(_q_scale=[], _k_all_scale=[], _q_qmeta=q_qmeta, _k_all_qmeta=k_qmeta)
    else:
        entry["k_prefix_ends"] = []
    return entry


def append_k_prefix(entry: dict, k_tok: torch.Tensor) -> None:
    """Record one eager pass's full key prefix ``k_tok`` in compact delta form."""
    ends = entry.get("k_prefix_ends")
    if ends is None:
        entry["k_all"].append(k_tok)
        return
    prev = ends[-1] if ends else 0
    cur = int(k_tok.shape[0])
    if cur < prev:
        full = torch.cat(entry["k_all"], dim=0)
        entry["k_all"] = [full[:L].clone() for L in ends]
        entry.pop("k_prefix_ends")
        entry["k_all"].append(k_tok)
        return
    entry["k_all"].append(k_tok[prev:].clone() if prev else k_tok)
    ends.append(cur)


def _resolve_score_heads(output_spec, layer_num: int, default_head: int) -> list:
    if isinstance(output_spec, dict):
        for k, v in output_spec.items():
            if int(k) == int(layer_num):
                return [int(h) for h in v] if v else [int(default_head)]
    return [int(default_head)]


def compute_head_scores(q_flat: torch.Tensor, k_flat: torch.Tensor, heads: list,
                        conf: dict, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Per-head causal attention scores for a SET of heads: ``[n, S_q, S_k]``."""
    H_q = int(conf["num_attention_heads"])
    H_kv = int(conf["num_key_value_heads"])
    d = int(conf["head_dim"])
    mult = float(conf["attention_multiplier"])
    S_q = q_flat.shape[0]
    S_k = k_flat.shape[0]
    g = max(1, H_q // H_kv)
    hq = torch.tensor([int(h) % H_q for h in heads], device=q_flat.device, dtype=torch.long)
    hkv = hq // g
    q = q_flat.view(S_q, H_q, d).float()
    k = k_flat.view(S_k, H_kv, d).float()
    q_sel = q.index_select(1, hq).permute(1, 0, 2)
    k_sel = k.index_select(1, hkv).permute(1, 0, 2)
    s = torch.bmm(q_sel, k_sel.transpose(1, 2)) * mult
    offset = S_k - S_q
    if S_q > 1 or offset < 0:
        mask = torch.ones(S_q, S_k, dtype=torch.bool, device=s.device).tril(diagonal=offset)
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1).to(dtype)


def compute_head_score(q_flat: torch.Tensor, k_flat: torch.Tensor, head: int,
                       conf: dict, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """One head's causal attention score: softmax((q_h k_h^T) * mult + causal), [S_q, S_k]."""
    return compute_head_scores(q_flat, k_flat, [head], conf, dtype)[0]


def _scores_from_qk_entry(entry: dict, conf: dict, dtype: torch.dtype = torch.float16) -> list:
    heads = entry.get("heads") or [int(entry.get("head", 0))]
    q_list = _cpu_list(entry["q"], None)
    prefix_ends = entry.get("k_prefix_ends")
    if prefix_ends:
        full_k = torch.cat(_cpu_list(entry["k_all"], None), dim=0)
        k_list = [full_k[:int(L)] for L in prefix_ends]
    else:
        k_list = _cpu_list(entry["k_all"], None)
    out = []
    for q_p, k_p in zip(q_list, k_list):
        q_p = q_p if q_p.dim() == 2 else q_p.unsqueeze(0)
        out.append(compute_head_scores(q_p, k_p, heads, conf, dtype))
    return out


def _aperture_disk_dbg(msg: str) -> None:
    if os.environ.get("MIA_APERTURE_DEBUG") == "1":
        print(f"[mia/aperture-disk] {msg}", flush=True)


def _worker_tp_rank(worker) -> int:
    r = getattr(worker, "_tp_rank", None)
    if r is not None:
        return int(r)
    # lazy: keep mia.core.runtime out of import mia
    from mia.core.runtime.tp_shard import resolve_tp_coords
    return resolve_tp_coords(worker)[0]


def _attach_tp_shard(payload: dict, worker) -> dict:
    shard = getattr(worker, "_qk_shard", None)
    if shard is not None and shard.tp_size > 1:
        # lazy: keep mia.core.runtime out of import mia
        from mia.core.runtime.tp_shard import TP_SHARD_KEY
        payload[TP_SHARD_KEY] = shard.as_header()
    return payload


def _marshal_qk_error(msg: str) -> bytes:
    return _ZSTD_COMPRESSOR.compress(pickle.dumps({"mia_error": str(msg)}))


def _marshal_perreq_qk(per_layer: dict, conf: dict, hookq_mode: str, shard=None,
                       module_names=None) -> bytes:
    qk_cache = {}
    for layer, rec in per_layer.items():
        layer = int(layer)
        q = rec["q"]
        k_all = rec.get("k_all") or []
        qk_cache[layer] = {
            "q": q.detach().cpu().contiguous(),
            "k_all": [k.detach().cpu().contiguous() for k in k_all],
            "layer_num": layer,
            "hookq_mode": hookq_mode,
        }
    payload = {"qk_cache": qk_cache, "config": conf}
    if module_names:
        payload["module_names"] = {int(L): str(n) for L, n in module_names.items()}
    if shard is not None and shard.tp_size > 1:
        # lazy: keep mia.core.runtime out of import mia
        from mia.core.runtime.tp_shard import TP_SHARD_KEY
        payload[TP_SHARD_KEY] = shard.as_header()
    return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))


class QKCaptureWorker:
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
        """Install forward hooks on all target attention modules."""
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

        self.hookq_mode = "all_tokens"

        self._score_mode_default = os.environ.get("MIA_QK_SCORE", "0") == "1"
        self._score_head_default = int(os.environ.get("MIA_QK_SCORE_HEAD", "0"))
        self._score_dtype = torch.float16

        self._artifact_tag, self._artifact_gran, self._artifact_gsize = resolve_capture_quant("qk")

        # lazy: keep mia.core.delivery out of import mia
        from mia.core.delivery.writer_process import init_writer_process
        init_writer_process(self)

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        num_h = int(getattr(text_cfg, "num_attention_heads"))
        num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
        hidden = int(getattr(text_cfg, "hidden_size"))
        from mia.core.runtime.tp_shard import check_attn_modules_match_shard, qk_conf_head_dim
        head_dim = qk_conf_head_dim(text_cfg)
        attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=attn_mult,
        )

        from mia.core.runtime.tp_shard import qk_shard, refuse_pipeline_parallel, resolve_tp_coords
        refuse_pipeline_parallel(getattr(self.parallel_config, "pipeline_parallel_size", 1),
                                 "QK install_hooks")
        tp_rank, tp_size = resolve_tp_coords(self)
        self._tp_rank = tp_rank
        self._qk_shard = qk_shard(tp_rank, tp_size, num_h, num_kv, head_dim)
        self._should_capture = True
        if tp_size > 1 and self._score_mode_default:
            raise MiaConfigurationError(
                "MIA_QK_SCORE=1 (attention-score capture) is not supported at "
                f"tensor_parallel_size={tp_size}: each rank holds only its own heads, so a "
                "per-head score is computed on one rank and cannot be merged. Capture raw Q/K "
                "(unset MIA_QK_SCORE) or run at tensor_parallel_size=1.")

        def qkv_hook(input, module_name, attn_module=None):
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

            query_start_loc, seq_lens = get_query_metadata(metadata)
            if query_start_loc is None:
                return

            bs = len(query_start_loc) - 1
            last_indices = query_start_loc

            layer_num = match_attn(module_name)

            refused = getattr(self, "_qk_refused", None)
            n_comp = getattr(step, "num_computed_tokens_np", None)
            for i in range(bs):
                req_id = step.req_ids[i]
                extra = step.extra_args_for(i)
                if not extra or extra.get("output_qk") is None:
                    continue
                if refused and req_id in refused:
                    continue
                output_spec = extra.get("output_qk")
                if isinstance(output_spec, dict):
                    layer_set = {int(k) for k in output_spec.keys()}
                    if layer_num not in layer_set:
                        continue
                elif isinstance(output_spec, list):
                    if layer_num not in output_spec:
                        continue

                hooks_on = extra.get("hooks_on", self._default_hooks_on)
                is_prefill = bool(step.is_prefilling_np[i])
                if hooks_on != "both":
                    if hooks_on == "prefill" and not is_prefill:
                        continue
                    if hooks_on == "decode" and is_prefill:
                        continue

                req_mode = extra.get("hookq_mode", self.hookq_mode)
                cap_mode = extra.get("qk_capture",
                                     "score" if self._score_mode_default else "qk")

                start = int(last_indices[i].item())
                end = int(last_indices[i + 1].item())
                n_done = int(n_comp[i]) if n_comp is not None else 0

                if is_prefill and req_mode == "last_token":
                    chunk_len = end - start
                    if int(step.num_computed_tokens_np[i]) + chunk_len < int(step.prompt_len_np[i]):
                        continue

                if cap_mode == "score" and self._qk_shard.tp_size > 1:
                    PROF.incr("qk.score_unsupported_tp")
                    if not getattr(self, "_score_tp_warned", False):
                        self._score_tp_warned = True
                        print("[mia/qk] WARNING: qk_capture='score' is unsupported at "
                              f"tensor_parallel_size={self._qk_shard.tp_size}; request NOT "
                              "captured.", flush=True)
                    continue
                if cap_mode == "score":
                    score_heads = _resolve_score_heads(
                        output_spec, layer_num, self._score_head_default)
                    if req_mode == "last_token":
                        q_view = input[0][end - 1:end, :].detach()
                    else:
                        q_view = input[0][start:end, :].detach()
                    k_view = input[1][start:end, :].detach()
                    try:
                        k_view = _prepend_cached_keys(module_name, metadata, i, k_view,
                                                      seq_lens, start, end, n_done)
                    except CachedKeysUnavailable as e:
                        self._refuse_qk_request(req_id, layer_num, e)
                        continue
                    score = compute_head_scores(q_view, k_view, score_heads, self._conf, self._score_dtype)
                    PROF.incr("hook.fire.qk")
                    PROF.gauge("captured.bytes.qk", score.numel() * score.element_size())
                    bucket = self._disk_states if extra.get("save_to_disk") else self._captured_states
                    if req_id not in bucket:
                        bucket[req_id] = {}
                    layer_states = bucket[req_id]
                    if module_name not in layer_states:
                        layer_states[module_name] = {"scores": [], "heads": score_heads, "layer_num": layer_num, "hookq_mode": req_mode, "capture": "score"}
                    layer_states[module_name]["scores"].append(score)
                    continue

                if req_mode == "all_tokens":
                    q_tok = input[0][start:end, :].detach().clone()
                else:
                    q_tok = input[0][end - 1, :].detach().clone()
                k_tok = input[1][start:end, :].detach().clone()

                try:
                    k_tok = _prepend_cached_keys(module_name, metadata, i, k_tok, seq_lens,
                                                 start, end, n_done)
                except CachedKeysUnavailable as e:
                    self._refuse_qk_request(req_id, layer_num, e)
                    continue

                q_tok, q_scale, q_qmeta = quant_clone(
                    q_tok, self._artifact_tag, self._artifact_gran, self._artifact_gsize)
                k_tok, k_scale, k_qmeta = quant_clone(
                    k_tok, self._artifact_tag, self._artifact_gran, self._artifact_gsize)

                PROF.incr("hook.fire.qk")
                PROF.gauge("captured.bytes.qk",
                           capture_bytes(q_tok, q_scale) + capture_bytes(k_tok, k_scale))

                bucket = self._disk_states if extra.get("save_to_disk") else self._captured_states
                if req_id not in bucket:
                    bucket[req_id] = {}
                layer_states = bucket[req_id]
                if module_name not in layer_states:
                    layer_states[module_name] = new_qk_entry(
                        layer_num, req_mode, q_qmeta=q_qmeta, k_qmeta=k_qmeta)
                ls = layer_states[module_name]
                ls["q"].append(q_tok)
                append_k_prefix(ls, k_tok)
                if q_qmeta is not None:
                    ls["_q_scale"].append(q_scale)
                    ls["_k_all_scale"].append(k_scale)

        check_attn_modules_match_shard(list(iter_matched_modules(model, match_attn)),
                                       self._qk_shard)
        self._hooks = []
        matched = []
        for name, module, _ in iter_matched_modules(model, match_attn):
            hook = module.register_forward_hook(
                lambda _m, i, _o, n=name: qkv_hook(i, n, _m)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} hooks on layers: {matched}")


    def _refuse_qk_request(self, req_id, layer_num, err) -> None:
        refused = self.__dict__.setdefault("_qk_refused", {})
        if req_id not in refused:
            refused[req_id] = f"layer {layer_num}: its cached keys could not be read ({err})"
            PROF.incr("kv.prefix_recon.errors")
            print(f"[mia/qk] request {req_id!r} refused: {refused[req_id]}", flush=True)

    def graph_install(self):
        """Install the CUDA-graph QK capture path (static buffers + wrap)."""
        if not getattr(self, "_captured_states", None):
            self._captured_states = {}
        if not getattr(self, "_disk_states", None):
            self._disk_states = {}

        # lazy: keep mia.core.hooks (reads env at import) out of import mia
        from mia.core.hooks.install import (
            install_execute_model_wrapper,
            install_qk_hosts,
        )
        install_qk_hosts(self)
        install_execute_model_wrapper(self.model_runner, self)

    def flush_aperture(self) -> str | None:
        """Final drain and QK sidecar write; return this rank's run_dir, or None if nothing was captured."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None:
            return None
        stop = getattr(drain, "stop", None)
        if callable(stop):
            drain.stop()
        else:
            drain.drain_once()
        drain.close()
        # lazy: keep mia.core.runtime out of import mia
        from mia.core.runtime.tp_shard import drain_holds_data
        if not drain_holds_data(drain):
            return None
        return getattr(self, "_qk_run_dir", None)


    def flush_aperture_per_request(self):
        """Drive end-of-run per-request QK delivery and return every deliverable capture."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        if index is None:
            return None
        stop = getattr(drain, "stop", None)
        if callable(stop):
            drain.stop()
        mode = self._aperture_hookq_mode(drain)
        lock = getattr(drain, "_index_lock", None) or contextlib.nullcontext()
        with lock:
            popped = index.pop_deliverable_qk()
            for req_id, _ in popped:
                index.free(req_id)
            residency_after = len(index.live_req_ids())
        deliverables: dict = {}
        for req_id, per_layer in popped:
            if isinstance(per_layer, Exception):
                deliverables[str(req_id)] = {"mia_error": str(per_layer)}
                continue
            deliverables[str(req_id)] = {
                int(layer): {
                    "q": rec["q"].detach().cpu().contiguous(),
                    "k_all": [k.detach().cpu().contiguous() for k in (rec.get("k_all") or [])],
                    "layer_num": int(layer),
                    "hookq_mode": mode,
                }
                for layer, rec in per_layer.items()
            }
        raw = pickle.dumps((deliverables, residency_after))
        return _ZSTD_COMPRESSOR.compress(raw)

    def _aperture_hookq_mode(self, drain) -> str:
        header = getattr(drain, "header", None) or {}
        return header.get("hookq_mode") or getattr(self, "hookq_mode", "all_tokens")

    def get_aperture_per_request(self, external_req_id: str) -> bytes | None:
        """Per-request retrieval for the off-loop QK aperture delivery path."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        if index is None:
            return None
        stash = self._drain_aperture_into_stash(drain, index)
        matches = match_internal_ids(list(stash), external_req_id)
        if len(matches) > 1:
            return _marshal_qk_error(f"{len(matches)} captured requests match {external_req_id!r}")
        if not matches:
            return None
        return stash.pop(matches[0])

    def _drain_aperture_into_stash(self, drain, index, free_external: str | None = None) -> dict:
        stash = getattr(self, "_aperture_perreq_stash", None)
        if stash is None:
            stash = {}
            self._aperture_perreq_stash = stash
        conf = getattr(self, "_conf", {})
        mode = self._aperture_hookq_mode(drain)
        lock = getattr(drain, "_index_lock", None) or contextlib.nullcontext()
        with lock:
            popped = index.pop_deliverable_qk()
            for req_id, _ in popped:
                index.free(req_id)
            if free_external is not None:
                for rid in match_internal_ids(list(index.live_req_ids()), free_external):
                    index.free(rid)
        names = getattr(self, "_qk_module_names", None)
        for req_id, per_layer in popped:
            if isinstance(per_layer, Exception):
                stash[str(req_id)] = _marshal_qk_error(str(per_layer))
                continue
            stash[str(req_id)] = _marshal_perreq_qk(per_layer, conf, mode,
                                                    getattr(self, "_qk_shard", None), names)
        return stash

    def clear_aperture_request(self, external_req_id: str) -> None:
        """Abort cleanup: free all of an aborted request's QK aperture state, host and disk."""
        drain = getattr(self, "_qk_drain", None)
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
                for rid in match_internal_ids(list(stash), external_req_id):
                    stash.pop(rid, None)
        return None

    def route_aperture_to_disk(self, req_id: str, dest: str) -> bool:
        """Route ``req_id`` to per-request disk staging, offloaded to ``dest`` when it finishes."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return False
        route = getattr(drain, "route_to_disk", None)
        if not callable(route):
            return False
        shard = getattr(self, "_qk_shard", None)
        if shard is not None and shard.tp_size > 1:
            # lazy: keep mia.core.runtime out of import mia
            from mia.core.runtime.tp_shard import rank_dir_name
            dest = os.path.join(str(dest), rank_dir_name(shard.tp_rank))
        _aperture_disk_dbg(f"worker.route_aperture_to_disk(qk): req_id={req_id!r} (EXTERNAL) dest={dest!r}")
        route(str(req_id), str(dest))
        return True

    def confirm_aperture_delivery(self, req_id: str, timeout_s: float | None = None) -> bool | None:
        """Block until a disk-routed QK request's file has landed at the client ``dest``."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        offload = getattr(drain, "_offload", None)
        if offload is None:
            return None
        ok = bool(offload.wait(str(req_id), timeout=timeout_s))
        if ok:
            unlink = getattr(drain, "unlink_delivered_source", None)
            if callable(unlink):
                unlink(str(req_id))
        return ok

    def mia_delivery_info(self) -> dict:
        """collective_rpc-callable: module names, config and whether the per-request drain is on."""
        drain = getattr(self, "_qk_drain", None)
        names = getattr(self, "_qk_module_names", None) or {}
        return {"names": [[int(L), str(n)] for L, n in sorted(names.items())],
                "config": dict(getattr(self, "_conf", {}) or {}),
                "per_request": bool(drain is not None and getattr(drain, "per_request", False)),
                "delivery_dir": None}

    def aperture_residency(self):
        """Read-only (host_live_count, disk_residency) query for the off-loop QK per-request path."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        lock = getattr(drain, "_index_lock", None) or contextlib.nullcontext()
        with lock:
            host_live = len(index.live_req_ids()) if index is not None else 0
        disk_fn = getattr(drain, "disk_residency", None)
        disk = int(disk_fn()) if callable(disk_fn) else 0
        return (int(host_live), int(disk))

    def _prune_seen(self, external):
        seen = getattr(self, "_capture_seen", None)
        if seen:
            self._capture_seen = {(r, m) for (r, m) in seen
                                  if not (r == external or r.startswith(f"{external}-"))}


    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve and remove captured QK states for a completed request."""
        # lazy: keep mia.core.hooks out of import mia
        from mia.core.hooks.drain import drain_barrier
        drain_barrier(self)
        consumer = getattr(self, "_capture_consumer", None)
        if consumer is not None:
            consumer.drain_writer_done(self)
        refused = getattr(self, "_qk_refused", None)
        hit = list(iter_matching_req_ids(refused, external_req_id)) if refused else []
        if hit:
            why = "; ".join(refused.pop(r) for r in hit)
            for r in hit:
                dropped = self._captured_states.pop(r, None)
                if dropped is not None and consumer is not None:
                    consumer.on_pop(r, dropped)
            return _marshal_qk_error(why)
        for req_id in iter_matching_req_ids(self._captured_states, external_req_id):
            layer_dict = self._captured_states.pop(req_id)
            if consumer is not None:
                consumer.on_pop(req_id, layer_dict)
            _census_bucket = None
            _census_acc = None
            if _CENSUS_ON:
                from mia.core.runtime.census import census_bucket, new_accumulator
                _census_bucket = census_bucket(layer_dict)
                _census_acc = new_accumulator()
            cpu_dict = {}
            with PROF.timed("worker.cpu_transfer.qk"):
                for mod_name, entry in layer_dict.items():
                    if "scores" in entry or entry.get("capture") == "score":
                        scores = entry["scores"] if "scores" in entry \
                            else _scores_from_qk_entry(entry, self._conf,
                                                       getattr(self, "_score_dtype", torch.float16))
                        cpu_dict[mod_name] = {
                            "scores": _cpu_list(scores, _census_acc),
                            "heads": entry.get("heads") or [entry.get("head", 0)],
                            "layer_num": entry["layer_num"],
                            "hookq_mode": entry.get("hookq_mode", "all_tokens"),
                            "capture": "score",
                        }
                        continue
                    mode = entry.get("hookq_mode", self.hookq_mode)
                    q_qmeta = entry.get("_q_qmeta")
                    if q_qmeta is None:
                        with PROF.timed("cpu_transfer.qk.d2h"):
                            q_cpu = _cpu_list(entry["q"], _census_acc)
                        if _use_compact_kall(entry):
                            with PROF.timed("cpu_transfer.qk.kall"):
                                compact = _k_all_compact(entry, _census_acc)
                        else:
                            compact = None
                        if compact is not None:
                            full_k, prefix_ends = compact
                            with PROF.timed("cpu_transfer.qk.pad"):
                                q_stacked = (pad_sequence(q_cpu, batch_first=True)
                                             if mode == "all_tokens" else torch.stack(q_cpu))
                            cpu_dict[mod_name] = {
                                "q": q_stacked, "k_full": full_k, "k_prefix_ends": prefix_ends,
                                "layer_num": entry["layer_num"], "hookq_mode": mode}
                        else:
                            with PROF.timed("cpu_transfer.qk.kall"):
                                k_cpu = _k_all_cpu_list(entry, _census_acc)
                            with PROF.timed("cpu_transfer.qk.pad"):
                                if mode == "all_tokens":
                                    q_stacked = pad_sequence(q_cpu, batch_first=True)
                                else:
                                    q_stacked = torch.stack(q_cpu)
                                k_stacked = pad_sequence(k_cpu, batch_first=True)
                            cpu_dict[mod_name] = {"q": q_stacked, "k_all": k_stacked,
                                                  "layer_num": entry["layer_num"], "hookq_mode": mode}
                    else:
                        cpu_dict[mod_name] = {
                            "q": _cpu_list(entry["q"], _census_acc),
                            "k_all": _k_all_cpu_list(entry, _census_acc),
                            "q_scale": [s.cpu() if s is not None else None
                                        for s in entry.get("_q_scale", [])],
                            "q_qmeta": q_qmeta,
                            "k_all_scale": [s.cpu() if s is not None else None
                                            for s in entry.get("_k_all_scale", [])],
                            "k_all_qmeta": entry.get("_k_all_qmeta"),
                            "layer_num": entry["layer_num"], "hookq_mode": mode}
            if _census_bucket is not None:
                from mia.core.runtime.census import census_emit, census_record
                census_emit(census_record(worker="qk", sink="rpc", req_id=req_id,
                                           bucket=_census_bucket, acc=_census_acc))
            payload = {"qk_cache": cpu_dict, "config": self._conf}
            _attach_tp_shard(payload, self)
            with PROF.timed("worker.compress.qk"):
                with PROF.timed("compress.qk.pickle"):
                    raw = pickle.dumps(payload)
                PROF.gauge("worker.raw_bytes.qk", len(raw))
                with PROF.timed("compress.qk.zstd"):
                    compressed = _ZSTD_COMPRESSOR.compress(raw)
            PROF.gauge("worker.compressed_bytes.qk", len(compressed))
            return compressed
        return None

    def reset_capture_peak_mem(self) -> int:
        """Reset the CUDA peak-allocated mark and return current allocated bytes."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return int(torch.cuda.memory_allocated())
        return 0

    def capture_drain_stats(self) -> dict:
        """Snapshot drain residency, backpressure and capture counters plus CUDA memory high-water."""
        out = {
            "consumer_present": False, "moves": 0, "bp_stalls": 0,
            "gpu_resident_bytes": 0, "gpu_budget_bytes": 0,
            "host_resident_bytes": 0, "host_budget_bytes": 0,
            "amber": 0, "red": 0, "refused": 0,
            "prof_throttled": 0, "prof_hookfire": 0,
            "max_mem_allocated": 0, "mem_allocated": 0, "total_gpu": 0,
        }
        consumer = getattr(self, "_capture_consumer", None)
        if consumer is not None:
            out["consumer_present"] = True
            out["moves"] = int(consumer._moves)
            out["bp_stalls"] = int(consumer._bp_stalls)
            out["gpu_resident_bytes"] = int(consumer.gpu_sensor.resident)
            out["gpu_budget_bytes"] = int(consumer.gpu_sensor.budget_bytes)
            out["host_resident_bytes"] = int(consumer.host_sensor.resident)
            out["host_budget_bytes"] = int(consumer.host_sensor.budget_bytes)
            out["amber"] = int(consumer.bp.stats.get("amber", 0))
            out["red"] = int(consumer.bp.stats.get("red", 0))
            out["refused"] = int(consumer.bp.stats.get("refused", 0))
        try:
            counters = PROF.snapshot().get("counters", {})
            out["prof_throttled"] = int(counters.get("capture.throttled", 0))
            out["prof_hookfire"] = int(counters.get("hook.fire.qk", 0))
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            out["max_mem_allocated"] = int(torch.cuda.max_memory_allocated())
            out["mem_allocated"] = int(torch.cuda.memory_allocated())
            try:
                _free, _total = torch.cuda.mem_get_info()
                out["total_gpu"] = int(_total)
            except Exception:  # noqa: BLE001
                pass
        return out

    def dump_profiler(self) -> str | None:
        """Dump this worker's profiler snapshot to MIA_PROFILE_DIR; return the path or None."""
        return PROF.dump(role="worker-rpc")

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured states without returning them (cleanup on abort/disconnect)."""
        consumer = getattr(self, "_capture_consumer", None)
        refused = getattr(self, "_qk_refused", None)
        if refused:
            clear_states_for_req(refused, external_req_id)
        if consumer is None:
            clear_states_for_req(self._captured_states, external_req_id)
            clear_states_for_req(self._disk_states, external_req_id)
            return
        for bucket in (self._captured_states, self._disk_states):
            for req_id in iter_matching_req_ids(bucket, external_req_id):
                consumer.on_pop(req_id, bucket.pop(req_id))

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """Write captured Q/K for all requests in the batch to one artifact."""
        # lazy: keep mia.core.hooks out of import mia
        from mia.core.hooks.drain import drain_barrier
        drain_barrier(self)
        consumer = getattr(self, "_capture_consumer", None)
        cpu_cache: dict = {"config": self._conf, "qk_cache": {}}
        found_any = False
        flushed_ids: list = []
        refused = getattr(self, "_qk_refused", None) or {}
        refused_here: dict = {}

        with PROF.timed("worker.cpu_transfer.qk"):
            for external_req_id in external_req_ids:
                for req_id in list(iter_matching_req_ids(refused, external_req_id)):
                    refused_here[req_id] = refused.pop(req_id)
                    dropped = self._disk_states.pop(req_id, None)
                    if dropped is not None and consumer is not None:
                        consumer.on_pop(req_id, dropped)
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
                        from mia.core.runtime.census import census_bucket, new_accumulator
                        _census_bucket = census_bucket(layer_dict)
                        _census_acc = new_accumulator()
                    for mod_name, entry in layer_dict.items():
                        if "scores" in entry or entry.get("capture") == "score":
                            scores = entry["scores"] if "scores" in entry \
                                else _scores_from_qk_entry(entry, self._conf,
                                                           getattr(self, "_score_dtype", torch.float16))
                            cpu_entry = {
                                "scores": _cpu_list(scores, _census_acc),
                                "heads": entry.get("heads") or [entry.get("head", 0)],
                                "layer_num": entry["layer_num"],
                                "hookq_mode": entry.get("hookq_mode", "all_tokens"),
                                "capture": "score",
                            }
                            existing = cpu_cache["qk_cache"].get(mod_name)
                            if existing is not None and "scores" in existing:
                                existing["scores"].extend(cpu_entry["scores"])
                            elif existing is not None:
                                existing.update({k: v for k, v in cpu_entry.items()
                                                 if k not in ("layer_num", "hookq_mode")})
                            else:
                                cpu_cache["qk_cache"][mod_name] = cpu_entry
                            continue
                        q_qmeta = entry.get("_q_qmeta")
                        cpu_entry = {
                            "q": _cpu_list(entry["q"], _census_acc),
                            "layer_num": entry["layer_num"],
                            "hookq_mode": entry.get("hookq_mode", self.hookq_mode),
                        }
                        kc = _k_all_compact(entry, _census_acc) if q_qmeta is None else None
                        if kc is not None:
                            cpu_entry["k_full"] = [kc[0]]
                            cpu_entry["k_prefix_ends"] = [kc[1]]
                        else:
                            cpu_entry["k_all"] = _k_all_cpu_list(entry, _census_acc)
                        if q_qmeta is not None:
                            cpu_entry["q_scale"] = [s.cpu() if s is not None else None
                                                    for s in entry.get("_q_scale", [])]
                            cpu_entry["q_qmeta"] = q_qmeta
                            cpu_entry["k_all_scale"] = [s.cpu() if s is not None else None
                                                        for s in entry.get("_k_all_scale", [])]
                            cpu_entry["k_all_qmeta"] = entry.get("_k_all_qmeta")
                        existing = cpu_cache["qk_cache"].get(mod_name)
                        if existing is not None and "q" in existing:
                            existing["q"].extend(cpu_entry["q"])
                            if "k_full" in cpu_entry and "k_full" in existing:
                                existing["k_full"].extend(cpu_entry["k_full"])
                                existing["k_prefix_ends"].extend(cpu_entry["k_prefix_ends"])
                            else:
                                existing.setdefault("k_all", []).extend(cpu_entry.get("k_all", []))
                            if q_qmeta is not None:
                                existing.setdefault("q_scale", []).extend(cpu_entry["q_scale"])
                                existing.setdefault("k_all_scale", []).extend(cpu_entry["k_all_scale"])
                                existing.setdefault("q_qmeta", q_qmeta)
                                existing.setdefault("k_all_qmeta", cpu_entry["k_all_qmeta"])
                        elif existing is not None:
                            for _k, _v in cpu_entry.items():
                                existing.setdefault(_k, _v)
                        else:
                            cpu_cache["qk_cache"][mod_name] = cpu_entry
                    if _census_bucket is not None:
                        from mia.core.runtime.census import census_emit, census_record
                        census_emit(census_record(worker="qk", sink="disk", req_id=req_id,
                                                   bucket=_census_bucket, acc=_census_acc))

        from mia.core.runtime.tp_shard import rank_dir_name
        tp_rank = _worker_tp_rank(self)
        run_dir = os.path.join(hook_dir, run_id, rank_dir_name(tp_rank))
        if refused_here or found_any:
            write_refused_qk(run_dir, refused_here)

        if not found_any:
            if consumer is not None:
                consumer.drain_writer_done(self)
            if refused_here:
                clear_rank_artifact(run_dir, "qk")
            return run_dir if refused_here else False

        os.makedirs(run_dir, exist_ok=True)
        # A reused run_id: no barrier or reader may take the previous artifact for this one.
        clear_rank_artifact(run_dir, "qk")
        _attach_tp_shard(cpu_cache, self)

        has_scores = any("scores" in e for e in cpu_cache["qk_cache"].values())
        quant_on = any("q_qmeta" in e for e in cpu_cache["qk_cache"].values())

        wp = getattr(self, "_writer_process", None)
        use_st = os.environ.get("MIA_USE_SAFETENSORS", "0") == "1"
        if wp is not None:
            with PROF.timed("worker.queue_put"):
                submitted = wp.submit("qk", cpu_cache, run_dir, self.hookq_mode, tp_rank,
                                      use_st, has_scores or quant_on, "qk.pt",
                                      req_ids=flushed_ids, block=True)
            if not submitted:
                from mia.core.delivery.writer_process import note_submit_refused
                note_submit_refused(self, wp)
                compact_page_backed_cache(cpu_cache)
                if use_st and not has_scores and not quant_on:
                    self._save_safetensors(cpu_cache, run_dir)
                else:
                    save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
                if consumer is not None:
                    consumer.release_req_pages(flushed_ids)
        else:
            compact_page_backed_cache(cpu_cache)
            if use_st and not has_scores and not quant_on:
                self._save_safetensors(cpu_cache, run_dir)
            else:
                save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
            if consumer is not None:
                consumer.release_req_pages(flushed_ids)

        if consumer is not None:
            consumer.drain_writer_done(self)
        return run_dir

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        # lazy: keep mia.core.delivery out of import mia
        from mia.core.delivery.artifact_writer import save_qk_cache_safetensors
        save_qk_cache_safetensors(cpu_cache, run_dir, self.hookq_mode, _worker_tp_rank(self))

