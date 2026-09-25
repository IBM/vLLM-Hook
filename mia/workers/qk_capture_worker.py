import contextlib
import os
import math
import pickle
import torch
from typing import TYPE_CHECKING, Any, Dict, List
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
    match_attn,
    quant_clone,
    resolve_capture_quant,
    save_pt_atomic,
    save_safetensors_atomic,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)

# Lossless compact k_all transfer: send full + prefix_ends (O(seq)) instead of the
# pad_sequence-expanded O(seq^2) growing-prefix tensor; the driver rebuilds the padded k_all
# at the analysis boundary (byte-identical reconstruction in _plugin.py). Tri-state
# MIA_QK_COMPACT_KALL:
#   "1"   -> always compact (opt-in, unchanged),
#   "0"   -> never compact,
#   unset -> compact only when a request accumulated >=2 growing-prefix rows (hooks_on=both /
#            all_tokens), where the pad is quadratic. A single-snapshot request (last_token +
#            prefill) has 1 row -> no pad waste -> the normal path, unchanged wire format.
_COMPACT_KALL_ENV = os.environ.get("MIA_QK_COMPACT_KALL")  # None (default) / "1" / "0"


def _use_compact_kall(entry: dict) -> bool:
    """Whether to ship this QK entry's k_all in compact form (see _COMPACT_KALL_ENV)."""
    if _COMPACT_KALL_ENV == "0":
        return False
    if _COMPACT_KALL_ENV == "1":
        return True
    pe = entry.get("k_prefix_ends")
    return bool(pe) and len(pe) >= 2


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


def key_cache_from_layer_kv(kv_cache):
    """Return the KEY cache as ``[num_blocks, block_size, num_kv_heads, head_size]``.

    The KV cache layout differs across vLLM versions / backends, and getting it
    wrong makes ``key_cache[block_ids]`` gather along the WRONG dimension — an
    out-of-bounds index into a size-2 (key/value) axis that triggers an
    unrecoverable device-side assert. vLLM v1 stores ONE tensor shaped
    ``[num_blocks, 2, block_size, num_kv_heads, head_size]`` and splits it with
    ``kv_cache.unbind(1)`` (confirmed in the TRITON/FLASH backends), so the key
    cache is ``kv_cache[:, 0]`` — NOT ``kv_cache[0]`` (which is block 0).

    Handles, in order:
      * a per-virtual-engine list/tuple wrapping ONE kv tensor -> unwrap it;
      * a (key, value) pair already split -> take element 0;
      * tensor ``[num_blocks, 2, block_size, H, D]`` (current vLLM) -> ``[:, 0]``;
      * tensor ``[2, num_blocks, block_size, H, D]`` (legacy) -> ``[0]``;
      * an already-key-only 4-D tensor -> as-is.
    """
    kv = kv_cache
    # Unwrap a single-element per-virtual-engine list -> the kv tensor.
    if isinstance(kv, (list, tuple)) and len(kv) == 1 and hasattr(kv[0], "ndim"):
        kv = kv[0]
    # Already-split (key, value) pair.
    if isinstance(kv, (list, tuple)) and len(kv) == 2 and hasattr(kv[0], "ndim"):
        return kv[0]
    if not hasattr(kv, "ndim"):
        return kv
    if kv.ndim == 5:
        if kv.shape[1] == 2:      # [num_blocks, 2, block_size, H, D] — vLLM unbind(1)
            return kv[:, 0]
        if kv.shape[0] == 2:      # [2, num_blocks, block_size, H, D] — legacy
            return kv[0]
    return kv                     # 4-D: already key-only


def _read_cached_keys(
    module_name,
    attn_metadata,
    req_idx: int,
    num_cached: int,
    total_len: int,
):
    """Read cached prefix keys from vLLM's paged KV cache.

    When prefix caching is active, the hook only fires for non-cached tokens.
    This function reconstructs the missing prefix keys by reading directly from
    vLLM's KV cache blocks, keyed by the block_table entry for this request.

    Returns a tensor of shape (num_cached, num_kv_heads * head_size) on the
    same device as the KV cache, or None on any error (caller falls back to
    new-tokens-only capture).
    """
    try:
        ctx = get_forward_context()
        # kv_cache is bound to the vLLM Attention wrapper in the forward context,
        # not to the PyTorch module. Access via no_compile_layers[layer_name].kv_cache.
        kv_cache = ctx.no_compile_layers[module_name].kv_cache
        # Correct key cache: [num_blocks, block_size, num_kv_heads, head_size]
        # (see key_cache_from_layer_kv for the layout lesson — using kv_cache[0]
        # here gathers a size-2 axis with real block ids -> device-side assert).
        key_cache = key_cache_from_layer_kv(kv_cache)

        num_blocks   = key_cache.shape[0]
        block_size   = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size    = key_cache.shape[3]

        # block_table: [batch_size, max_blocks_per_seq]
        block_table = attn_metadata.block_table
        num_blocks_needed = math.ceil(total_len / block_size)
        block_ids = block_table[req_idx, :num_blocks_needed]  # [num_blocks_needed]

        # Bounds guard: an out-of-range block id makes the gather below trigger an
        # UNRECOVERABLE device-side assert (kills the whole engine). If anything is
        # off (wrong layout, stale block_table), skip prefix-K rather than crash —
        # the caller then keeps the new-tokens-only keys, a safe degradation.
        if block_ids.numel() == 0 or int(block_ids.max()) >= num_blocks \
                or int(block_ids.min()) < 0:
            return None

        # Gather and flatten: [num_blocks_needed * block_size, kv_hidden]
        prefix_keys = key_cache[block_ids].reshape(-1, num_kv_heads * head_size)

        # Trim to exact cached token count (last block may be partially filled)
        return prefix_keys[:num_cached].detach()
    except Exception:
        return None


def _k_all_cpu_list(entry: dict, _census_acc: dict | None = None) -> list:
    """Return the per-step growing-prefix ``k_all`` as CPU tensors.

    Graph buffer-mode egress stores only each step's NEW key rows in
    ``entry["k_all"]`` plus the per-step prefix length in ``entry["k_prefix_ends"]``, so
    the per-step GPU clone is O(1) instead of the O(seq_len) full-prefix clone. Here we
    rebuild the growing prefixes (``full[:L]`` for each recorded ``L``) so the captured
    artifact is byte-identical to the eager path, which stores the full prefixes directly
    (no ``k_prefix_ends`` -> passed through unchanged). ``.cpu()`` first so a drained
    (pinned) + GPU mix concatenates on one device; the per-step slices are CPU views, and
    the downstream pad_sequence/stack copies them exactly as before.

    ``_census_acc`` is None on the default path (verbatim ``[t.cpu() for t in ...]`` via
    :func:`_cpu_list`); when the offload-census gate is on it is the caller's shared
    per-request accumulator (see the measurement spec).
    """
    prefix_ends = entry.get("k_prefix_ends")
    parts = _cpu_list(entry["k_all"], _census_acc)
    if not prefix_ends:
        return parts
    full = torch.cat(parts, dim=0)
    return [full[:int(L)] for L in prefix_ends]


def _k_all_compact(entry: dict, _census_acc: dict | None = None):
    """COMPACT form of the growing-prefix k_all: ``(full, prefix_ends)`` where
    ``full`` is the O(seq) unique key rows and ``[full[:L] for L in prefix_ends]`` is
    byte-identical to :func:`_k_all_cpu_list`. Sending this (O(seq)) instead of the
    ``pad_sequence``-expanded ``[num_steps, max_len, k_dim]`` tensor (O(seq^2), ~93% zeros)
    moves the quadratic pad + its pickle/zstd off the worker engine loop; the driver rebuilds
    the padded k_all at the analysis boundary (``MIA_QK_COMPACT_KALL``). Returns None
    when the entry has no ``k_prefix_ends`` (eager path) -> caller keeps the normal pad.

    ``_census_acc`` -- see :func:`_k_all_cpu_list`.
    """
    prefix_ends = entry.get("k_prefix_ends")
    if not prefix_ends:
        return None
    full = torch.cat(_cpu_list(entry["k_all"], _census_acc), dim=0)
    return full, [int(L) for L in prefix_ends]


def new_qk_entry(layer_num: int, mode: str, q_qmeta=None, k_qmeta=None) -> dict:
    """A fresh per-(request, layer) QK capture entry for the eager hook.

    Native entries carry an empty ``k_prefix_ends``, which arms the compact delta
    accumulation in :func:`append_k_prefix`. Quantized entries carry the scale/qmeta
    channels instead and stay on the legacy full-prefix append (packed rows can't be
    sliced by token), exactly as before.
    """
    entry = {"q": [], "k_all": [], "layer_num": layer_num, "hookq_mode": mode}
    if q_qmeta is not None:
        entry.update(_q_scale=[], _k_all_scale=[], _q_qmeta=q_qmeta, _k_all_qmeta=k_qmeta)
    else:
        entry["k_prefix_ends"] = []
    return entry


def append_k_prefix(entry: dict, k_tok: torch.Tensor) -> None:
    """Record one eager pass's FULL key prefix ``k_tok`` in COMPACT delta form.

    The eager hook sees the whole prefix each pass (prefill ``[P,Kd]``, then ``[P+1,Kd]``,
    ``[P+2,Kd]``, ... one per decode step). Appending those verbatim makes the capture
    bucket — and every artifact downstream of it — O(seq^2). Store only this pass's NEW
    rows in ``k_all`` plus the prefix length in ``k_prefix_ends`` (O(seq)) instead;
    ``_k_all_cpu_list`` rebuilds the exact growing prefixes, so captured values are
    unchanged.

    The delta is CLONED: a view would keep its pass's whole prefix storage alive and
    there would be no residency win.

    Two entries stay on the legacy verbatim append: quantized ones (no ``k_prefix_ends``;
    packed rows aren't sliceable by token), and any request whose prefix length DROPS —
    a preempted+recomputed request re-prefills, which no delta chain can encode. That
    case rebuilds the full prefixes it already holds and abandons delta mode for the
    entry, so its values are byte-identical either way.
    """
    ends = entry.get("k_prefix_ends")
    if ends is None:
        entry["k_all"].append(k_tok)          # quantized / already fell back
        return
    prev = ends[-1] if ends else 0
    cur = int(k_tok.shape[0])
    if cur < prev:                            # preemption + recompute: not a growing prefix
        full = torch.cat(entry["k_all"], dim=0)
        entry["k_all"] = [full[:L].clone() for L in ends]
        entry.pop("k_prefix_ends")
        entry["k_all"].append(k_tok)
        return
    entry["k_all"].append(k_tok[prev:].clone() if prev else k_tok)
    ends.append(cur)


def _resolve_score_heads(output_spec, layer_num: int, default_head: int) -> list:
    """The q-head set to score for a layer (v0.5.7 D2 multi-head score).

    When ``output_qk`` is a ``{layer: [heads]}`` dict (the analyzer's ``layer_to_heads``),
    score exactly the analyzer's important heads for this layer; otherwise fall back to a
    single ``[default_head]`` (the v0.6.0 single-head behaviour). Order is preserved so the
    captured ``scores[k]`` aligns with ``heads[k]``.
    """
    if isinstance(output_spec, dict):
        for k, v in output_spec.items():
            if int(k) == int(layer_num):
                return [int(h) for h in v] if v else [int(default_head)]
    return [int(default_head)]


def compute_head_scores(q_flat: torch.Tensor, k_flat: torch.Tensor, heads: list,
                        conf: dict, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Per-head causal attention scores for a SET of heads: ``[n, S_q, S_k]``.

    Vectorized generalization of :func:`compute_head_score` over ``heads`` (a list of
    q-head indices). Each head ``h`` reads KV head ``h // (H_q//H_kv)`` under GQA. Same
    math (matmul/softmax in fp32, stored in ``dtype``) so each captured score == the
    analyzer recompute for that head. ``heads`` order is preserved (``out[k] <-> heads[k]``).
    """
    H_q = int(conf["num_attention_heads"])
    H_kv = int(conf["num_key_value_heads"])
    d = int(conf["head_dim"])
    mult = float(conf["attention_multiplier"])
    S_q = q_flat.shape[0]
    S_k = k_flat.shape[0]
    g = max(1, H_q // H_kv)
    hq = torch.tensor([int(h) % H_q for h in heads], device=q_flat.device, dtype=torch.long)
    hkv = hq // g
    q = q_flat.view(S_q, H_q, d).float()                  # [S_q, H_q, d]
    k = k_flat.view(S_k, H_kv, d).float()                 # [S_k, H_kv, d]
    q_sel = q.index_select(1, hq).permute(1, 0, 2)        # [n, S_q, d]
    k_sel = k.index_select(1, hkv).permute(1, 0, 2)       # [n, S_k, d]
    s = torch.bmm(q_sel, k_sel.transpose(1, 2)) * mult    # [n, S_q, S_k] fp32
    offset = S_k - S_q
    if S_q > 1 or offset < 0:
        # Mask non-causal entries (broadcast [S_q,S_k] over the n-head dim). A single query
        # row (decode / last token) at offset==S_k-1 attends every key -> mask is a no-op.
        mask = torch.ones(S_q, S_k, dtype=torch.bool, device=s.device).tril(diagonal=offset)
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1).to(dtype)


def compute_head_score(q_flat: torch.Tensor, k_flat: torch.Tensor, head: int,
                       conf: dict, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """One head's causal attention score: ``softmax((q_h·k_hᵀ)·mult + causal) -> [S_q, S_k]``.

    Single-head wrapper over :func:`compute_head_scores` (value-identical, ``[0]`` of the
    stacked result) — kept for callers/oracles that want one head.

    The v0.6.0 GPU-side score capture: instead of cloning q/k (all heads) and recomputing
    in the analyzer, compute the score for ONE head here and flush only that. Matches the
    analyzer math exactly (``AttntrackerAnalyzer.compute_attention_from_qk`` /
    ``CorerAnalyzer.get_attn_all``) so the captured score == analyzer recompute for the same
    head:

      * ``q_flat`` [S_q, H_q·d] — this pass's post-RoPE queries (all_tokens slice).
      * ``k_flat`` [S_k, H_kv·d] — the FULL key history (prefix reconstructed upstream); S_k ≥ S_q.
      * ``head``   q-head index in [0, H_q); its KV head under GQA is ``head // (H_q//H_kv)``.
      * causal: query row r (absolute pos ``offset+r``, ``offset = S_k - S_q`` = prefix-cached
        count) attends keys ``[0, offset+r]`` — ``tril(diagonal=offset)`` == the analyzer's
        ``triu(diagonal=-(S_k-S_q))``.

    Matmul/softmax in fp32 (accuracy), stored in ``dtype`` (default fp16: [0,1] values fit
    fp16's mantissa better than bf16, same 2 bytes). ``conf`` is the worker ``_conf``.
    """
    return compute_head_scores(q_flat, k_flat, [head], conf, dtype)[0]


def _scores_from_qk_entry(entry: dict, conf: dict, dtype: torch.dtype = torch.float16) -> list:
    """Compute per-pass head scores from an accumulated Q/K entry (buffer-mode score).

    Buffer-mode egress stages Q/K exactly like QK capture (so it reuses the routing +
    growing-``k_all`` reconstruction + prefix-K), then marks the entry ``capture="score"``.
    Here, at retrieval, we recompute the one-head score ON GPU from those staged tensors and
    flush only the score — the analyzer's work, moved to the worker. Mirrors
    ``_k_all_cpu_list``'s prefix reconstruction (``full[:L]`` per recorded ``k_prefix_ends``)
    so each pass's keys match the eager path; aligns 1:1 with the per-pass ``q`` list (both
    appended on ``emit_q`` steps). Returns CPU score tensors.
    """
    heads = entry.get("heads") or [int(entry.get("head", 0))]
    # .cpu() every staged tensor BEFORE the cat/compute: the streaming drain
    # (graph/drain.py) replaces drained list elements in place with pinned-HOST tensors
    # while post-drain clones stay on GPU, so entry["k_all"]/["q"] can be a CUDA/CPU MIX
    # — torch.cat over mixed devices raises. Same fix _k_all_cpu_list uses; computing the
    # score on host at retrieval is value-identical (buffer mode is not perf-profiled).
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
        out.append(compute_head_scores(q_p, k_p, heads, conf, dtype))  # [n, S_q, S_k]
    return out


def _aperture_disk_dbg(msg: str) -> None:
    """Gated disk-pipeline debug logging (MIA_APERTURE_DEBUG=1). Off by default -> no perf impact.
    Read at call time so a per-worker env set before spawn takes effect."""
    if os.environ.get("MIA_APERTURE_DEBUG") == "1":
        print(f"[hookplugin/aperture-disk] {msg}", flush=True)


def _worker_tp_rank(worker) -> int:
    """This worker's TENSOR-parallel rank (the ``tp_rank_<r>`` dir name), for eager and graph."""
    r = getattr(worker, "_tp_rank", None)
    if r is not None:
        return int(r)
    from mia.graph.tp_shard import resolve_tp_coords
    return resolve_tp_coords(worker)[0]


def _attach_tp_shard(payload: dict, worker) -> dict:
    """Stamp this rank's QK shard onto a payload / cpu_cache under ``"tp_shard"`` when TP > 1.

    TP=1 payloads are left byte-identical (no key). At TP > 1 every rank's payload carries its
    shard, so ``graph/tp_shard.merge_qk_payloads`` can put the heads back in global order."""
    shard = getattr(worker, "_qk_shard", None)
    if shard is not None and shard.tp_size > 1:
        from mia.graph.tp_shard import TP_SHARD_KEY
        payload[TP_SHARD_KEY] = shard.as_header()
    return payload


def _marshal_perreq_qk(per_layer: dict, conf: dict, hookq_mode: str, shard=None) -> bytes:
    """Marshal ONE finished request's assembled per-layer QK capture (the
    ``PerRequestIndex.pop_deliverable_qk`` value: ``{layer(0-based): {"q": tensor, "k_all":
    [tensor, ...]}}``) into the same driver-attachable payload the QK RPC path returns, serialized to
    ZSTD-PICKLE BYTES.

    ``collective_rpc`` does NOT round-trip raw torch tensors (they arrive on the driver as plain Python
    lists), so every tensor payload MUST be bytes -- the same convention as ``get_captured_states`` /
    ``flush_aperture_per_request``.

    Payload SHAPE == the eager QK path's ``qk_cache`` per-entry dict, BUT ``k_all`` is the RAW
    growing-prefix LIST ``[k_full[:L] for L in prefix_ends]`` that ``assemble_qk`` (and the disk reader
    ``load_multilayer_qk_aperture_artifact``) produce -- NOT the eager RPC path's ``pad_sequence`` tensor.
    The two aperture routes (host RPC here, disk via the reader) therefore deliver the IDENTICAL list shape.
    The outer key is ``layer`` (0-based == the eager ``match_attn`` layer_num) with ``layer_num`` set to
    the same, so it lines up with the eager probes with NO remap. Tensors keep their native dtype (the
    eager QK path does not force float32)."""
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
    if shard is not None and shard.tp_size > 1:
        from mia.graph.tp_shard import TP_SHARD_KEY
        payload[TP_SHARD_KEY] = shard.as_header()
    return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))


class QKCaptureWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (QKCaptureWorker,) at runtime,
    so self is the Worker instance. Methods are callable via collective_rpc.
    """

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    # Default capture phase — matches the old hooks_on=(True, False) registry entry.
    # Can be overridden per-request via extra_args["hooks_on"].
    _default_hooks_on: str = "prefill"

    # Per-request captured QK states (API serving path):
    # internal_req_id -> {module_name -> {"q": [...], "k_all": [...], "layer_num": int}}
    _captured_states: dict = {}
    _hooks_installed: bool = False
    _step: "StepView | None" = None

    def install_hooks(self):
        """Install forward hooks on all target attention modules. Idempotent.

        Callable via collective_rpc("install_hooks") — the plugin calls this
        lazily on the first request that sets output_qk in extra_args.

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
        self._captured_states = {}  # RPC path
        self._disk_states = {}      # disk path: same shape, written via flush_disk()
        model = getattr(runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # Worker-wide fallback when extra_args["hookq_mode"] is missing.
        self.hookq_mode = "all_tokens"

        # v0.6.0 score capture: worker-wide defaults (per-request extra_args override).
        # MIA_QK_SCORE=1 flips the worker to flush per-head attention scores
        # instead of Q/K; MIA_QK_SCORE_HEAD picks the (single) head per layer.
        self._score_mode_default = os.environ.get("MIA_QK_SCORE", "0") == "1"
        self._score_head_default = int(os.environ.get("MIA_QK_SCORE_HEAD", "0"))
        self._score_dtype = torch.float16

        # Artifact quantization (opt-in): quantize q/k clones on GPU at capture,
        # dequantize in the worker at retrieval/flush. Default off = byte-identical.
        self._artifact_tag, self._artifact_gran, self._artifact_gsize = resolve_capture_quant("qk")

        # Writer PROCESS (no-op unless MIA_WRITER_PROCESS=1).
        from mia.graph.writer_process import init_writer_process
        init_writer_process(self)

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        num_h = int(getattr(text_cfg, "num_attention_heads"))
        num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
        hidden = int(getattr(text_cfg, "hidden_size"))
        from mia.graph.tp_shard import check_attn_modules_match_shard, qk_conf_head_dim
        head_dim = qk_conf_head_dim(text_cfg)
        attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
        # _conf describes the MERGED (global) layout the analyzers consume -- full head counts at
        # every TP size. One rank's shard of it is self._qk_shard.
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=attn_mult,
        )

        # EVERY TP rank captures, each its OWN heads. Q and K are NOT replicated across TP ranks:
        # they are the output of the column-parallel qkv_proj and are read here BEFORE attention,
        # so no collective has gathered them -- rank r's input[0]/input[1] hold only its shard of
        # the heads (mia/graph/tp_shard.py). The payload of every rank carries that shard under
        # "tp_shard" (TP > 1 only; TP=1 payloads are unchanged) and the driver merges the ranks
        # into the global head order (_plugin.merge_probe_parts / run_utils disk loaders).
        from mia.graph.tp_shard import qk_shard, refuse_pipeline_parallel, resolve_tp_coords
        refuse_pipeline_parallel(getattr(self.parallel_config, "pipeline_parallel_size", 1),
                                 "QK install_hooks")
        tp_rank, tp_size = resolve_tp_coords(self)
        self._tp_rank = tp_rank
        self._qk_shard = qk_shard(tp_rank, tp_size, num_h, num_kv, head_dim)
        self._should_capture = True
        if tp_size > 1 and self._score_mode_default:
            from mia.errors import MiaConfigurationError
            raise MiaConfigurationError(
                "MIA_QK_SCORE=1 (attention-score capture) is not supported at "
                f"tensor_parallel_size={tp_size}: each rank holds only its own heads, so a "
                "per-head score is computed on one rank and cannot be merged. Capture raw Q/K "
                "(unset MIA_QK_SCORE) or run at tensor_parallel_size=1.")

        def qkv_hook(input, module_name, attn_module=None):
            # Every TP rank captures its own heads (see the comment at _should_capture).
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

            query_start_loc, seq_lens = get_query_metadata(metadata)
            if query_start_loc is None:
                return

            bs = len(query_start_loc) - 1
            last_indices = query_start_loc

            layer_num = match_attn(module_name)

            # Per-request capture. Each request in the batch may route to
            # either _captured_states (RPC path) or _disk_states (disk path)
            # based on its extra_args.
            for i in range(bs):
                req_id = step.req_ids[i]
                extra = step.extra_args_for(i)
                if not extra or extra.get("output_qk") is None:
                    continue
                # output_qk accepts three forms, matching the old layer_to_heads config:
                #   True             -> capture all layers
                #   [layer_ids]      -> capture specific layers
                #   {layer: [heads]} -> capture specific layers (heads used downstream by analyzer)
                # The worker only uses the keys for layer filtering — head info is
                # forwarded to the analyzer by the caller, same as the old env-var flow.
                output_spec = extra.get("output_qk")
                if isinstance(output_spec, dict):
                    layer_set = {int(k) for k in output_spec.keys()}
                    if layer_num not in layer_set:
                        continue
                elif isinstance(output_spec, list):
                    if layer_num not in output_spec:
                        continue

                # hooks_on: "prefill" (default) | "decode" | "both"
                # is_prefilling_np detects the first (prefill) pass -- V2 computes it as
                # num_computed_prefill_tokens < prefill_len, the same boundary the old
                # output_token_ids == [] check drew, robust to prefix caching where
                # query_len < seq_len even on the first pass.
                hooks_on = extra.get("hooks_on", self._default_hooks_on)
                is_prefill = bool(step.is_prefilling_np[i])
                if hooks_on != "both":
                    if hooks_on == "prefill" and not is_prefill:
                        continue
                    if hooks_on == "decode" and is_prefill:
                        continue

                # Per-request mode: extra_args["hookq_mode"] overrides the worker default.
                req_mode = extra.get("hookq_mode", self.hookq_mode)
                # v0.6.0: per-request capture mode — "qk" (default) clones Q/K; "score"
                # computes one head's attention score on-GPU and flushes only that.
                cap_mode = extra.get("qk_capture",
                                     "score" if self._score_mode_default else "qk")

                start = int(last_indices[i].item())
                end = int(last_indices[i + 1].item())

                # With chunked-prefill, in last_token mode, only capture on the final chunk of the
                # prefill (when computed-after-step reaches the PROMPT length -- prompt_len_np,
                # not prefill_len_np, which can exceed the prompt after preemption-resume).
                # Applies to BOTH qk and score capture: last_token score flushes only the final
                # query row [1, S_k], so mid-prefill chunks are skipped exactly like qk mode.
                # all_tokens (either capture) never enters this gate (req_mode check) and
                # captures every chunk.
                if is_prefill and req_mode == "last_token":
                    chunk_len = end - start
                    if int(step.num_computed_tokens_np[i]) + chunk_len < int(step.prompt_len_np[i]):
                        # Mid-prefill chunk doesn't need capture
                        continue

                # ---- v0.6.0 score mode: flush one head's score, not Q/K ----
                # all_tokens -> the full causal [S_q, S_k] matrix; last_token -> only the final
                # query row [1, S_k] (the last token's attention over the whole context). Decode
                # passes are [1, S_k] in either mode. k is always the FULL history (every key).
                if cap_mode == "score" and self._qk_shard.tp_size > 1:
                    # A per-head score at TP > 1 would be computed over the heads THIS rank holds
                    # under GLOBAL head indices -- wrong on every rank. The driver refuses such a
                    # request before it is submitted (_plugin._refuse_unsupported_tp_request);
                    # this is the backstop for a caller that bypassed it. Never captured as a
                    # silently-wrong score; counted and warned once.
                    PROF.incr("qk.score_unsupported_tp")
                    if not getattr(self, "_score_tp_warned", False):
                        self._score_tp_warned = True
                        print("[mia/qk] WARNING: qk_capture='score' is unsupported at "
                              f"tensor_parallel_size={self._qk_shard.tp_size}; request NOT "
                              "captured.", flush=True)
                    continue
                if cap_mode == "score":
                    # v0.5.7 D2: score the analyzer's head SET for this layer (from the
                    # output_qk {layer:[heads]} dict), not a single head; fall back to the
                    # single score_head when no per-layer head info is present.
                    score_heads = _resolve_score_heads(
                        output_spec, layer_num, self._score_head_default)
                    # q/k are views consumed immediately by the matmul — no clone needed.
                    if req_mode == "last_token":
                        q_view = input[0][end - 1:end, :].detach()
                    else:
                        q_view = input[0][start:end, :].detach()
                    k_view = input[1][start:end, :].detach()
                    # Reconstruct the prefix-cache-trimmed keys so k is the FULL history
                    # (same path as QK mode below); the score needs every key.
                    if seq_lens is not None and attn_module is not None:
                        try:
                            total_len = int(seq_lens[i].item()) if hasattr(seq_lens[i], 'item') else int(seq_lens[i])
                            num_cached = total_len - (end - start)
                            if num_cached > 0:
                                PROF.incr("kv.prefix_recon")
                                with PROF.timed("kv.prefix_recon"):
                                    prefix_k = _read_cached_keys(module_name, metadata if not isinstance(metadata, dict) else next(iter(metadata.values())), i, num_cached, total_len)
                                if prefix_k is not None:
                                    k_view = torch.cat([prefix_k.to(k_view.device, dtype=k_view.dtype), k_view], dim=0)
                        except Exception:
                            PROF.incr("kv.prefix_recon.errors")
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

                # Accumulate GPU tensors — clone() copies data immediately so we
                # own the buffer; .cpu() is deferred to retrieval/flush.
                if req_mode == "all_tokens":
                    q_tok = input[0][start:end, :].detach().clone()
                else:
                    q_tok = input[0][end - 1, :].detach().clone()
                k_tok = input[1][start:end, :].detach().clone()

                # Reconstruct full k_all when prefix caching is active.
                # seq_lens[i] = total sequence length (cached + new tokens).
                # query_len = new tokens only (what the hook captured above).
                # If num_cached > 0, read the missing prefix keys directly from
                # vLLM's paged KV cache and prepend them to k_tok.
                if seq_lens is not None and attn_module is not None:
                    try:
                        total_len = int(seq_lens[i].item()) if hasattr(seq_lens[i], 'item') else int(seq_lens[i])
                        query_len = end - start
                        num_cached = total_len - query_len
                        if num_cached > 0:
                            PROF.incr("kv.prefix_recon")
                            with PROF.timed("kv.prefix_recon"):
                                prefix_k = _read_cached_keys(module_name, metadata if not isinstance(metadata, dict) else next(iter(metadata.values())), i, num_cached, total_len)
                            if prefix_k is not None:
                                k_tok = torch.cat([prefix_k.to(k_tok.device, dtype=k_tok.dtype), k_tok], dim=0)
                    except Exception:
                        PROF.incr("kv.prefix_recon.errors")

                # Optional on-GPU quantization (default off → packed is the clone unchanged).
                q_tok, q_scale, q_qmeta = quant_clone(
                    q_tok, self._artifact_tag, self._artifact_gran, self._artifact_gsize)
                k_tok, k_scale, k_qmeta = quant_clone(
                    k_tok, self._artifact_tag, self._artifact_gran, self._artifact_gsize)

                PROF.incr("hook.fire.qk")
                PROF.gauge("captured.bytes.qk",
                           capture_bytes(q_tok, q_scale) + capture_bytes(k_tok, k_scale))

                # Route to disk or RPC bucket based on save_to_disk flag.
                bucket = self._disk_states if extra.get("save_to_disk") else self._captured_states
                if req_id not in bucket:
                    bucket[req_id] = {}
                layer_states = bucket[req_id]
                if module_name not in layer_states:
                    layer_states[module_name] = new_qk_entry(
                        layer_num, req_mode, q_qmeta=q_qmeta, k_qmeta=k_qmeta)
                ls = layer_states[module_name]
                ls["q"].append(q_tok)
                # COMPACT: store this pass's NEW key rows + the prefix length, not the
                # whole growing prefix (which is O(seq^2) across a trajectory).
                append_k_prefix(ls, k_tok)
                if q_qmeta is not None:
                    ls["_q_scale"].append(q_scale)
                    ls["_k_all_scale"].append(k_scale)

        # Hook every attention module. Per-request layer filtering via
        # extra_args['output_qk'] happens inside the hook closure.
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

    # ------------------------------------------------------------------
    # v0.3.0 CUDA-graph capture install (graph mode only)
    # ------------------------------------------------------------------

    def graph_install(self):
        """Install the CUDA-graph QK capture path (static buffers + wrap).

        Thin delegating entry called by the Worker.load_model monkey-patch
        (graph/install.py:patch_worker_load_model) AFTER the model is built but
        BEFORE warm-up/compile/capture. It is a strict no-op for the eager
        v0.2.0 path: the load_model patch only reaches here when graph mode is
        armed (MIA_ALLOW_CUDAGRAPH==1) AND this worker is the QK worker.

        The heavy lifting (per-layer q/k aperture hosts, class-level Attention.forward
        wrap, execute_model routing + aperture drain wrapper) lives in graph/install.py.
        CAPTURE-APERTURE PATH: scatter -> shared GPU aperture -> off-loop drain -> durable
        per-layer q/k raw files. It builds self._capture_aperture + self._qk_drain;
        retrieval/flush go through the aperture path below (there is no separate bank).
        Retrieval on the aperture path is off-loop, reading the durable files (the caller flushes
        with flush_aperture()). EVERY TP rank captures its own heads into its own tp_rank_<r>/, so
        read them with aperture_reader.merge_qk_aperture_ranks(<every rank's dir>) -- one rank's
        dir (load_multilayer_qk_aperture_artifact) is only that rank's slice of the heads.

        Idempotent: graph.install's installers are themselves guarded, and the bucket init
        below only seeds dicts that are missing.
        """
        # Inert on the aperture path (the bank is never built) but seeded so the eager-path RPC
        # methods that read them never see the shared class-level default dicts.
        if not getattr(self, "_captured_states", None):
            self._captured_states = {}  # RPC path (inert on the aperture path)
        if not getattr(self, "_disk_states", None):
            self._disk_states = {}      # disk path (inert on the aperture path)

        # graph.install builds the hosts, populates self._conf / self.hookq_mode /
        # self._should_capture, wires the wrap, and installs the execute_model
        # wrapper. Import lazily so the eager path never imports the graph stack.
        from mia.graph.install import (
            install_execute_model_wrapper,
            install_qk_hosts,
        )
        install_qk_hosts(self)
        install_execute_model_wrapper(self.model_runner, self)

    def flush_aperture(self) -> str | None:
        """collective_rpc-callable: final drain + write the QK capture-aperture metadata sidecar; return
        this rank's run_dir (``<MIA_APERTURE_DIR>/tp_rank_<r>``), or None if the aperture path is not
        installed or the dir holds no captured rows.

        The QK capture-aperture path (graph/install.py::install_execute_model_wrapper) writes durable
        per-layer q + k raw files continuously; this flushes any last pending rows and writes the
        shared sidecar so aperture_reader.merge_qk_aperture_ranks can reconstruct every rank's heads
        into the full layer (at TP=1 the single dir is the whole layer). Call once
        after all requests finish (the worker is often killed rather than joined, so the atexit
        backstop is unreliable — this RPC is the durable-flush contract).

        Handles both drains: the off-loop consumer (``stop()`` drains its queue + joins, surfacing
        any consumer error) and the synchronous drain (``drain_once``). ``stop()`` is idempotent,
        so a duplicate flush is safe."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None:
            return None
        stop = getattr(drain, "stop", None)
        if callable(stop):
            drain.stop()         # off-loop: drain the queue + join (raises on consumer error)
        else:
            drain.drain_once()   # synchronous path: flush any last pending rows on the loop
        drain.close()            # write the shared sidecar (idempotent)
        # Only a dir that HOLDS captured rows is returned: a caller collecting
        # `[d for d in collective_rpc("flush_aperture") if d]` then gets exactly the rank dirs
        # to merge (every TP rank's, for QK), never an empty one.
        from mia.graph.tp_shard import drain_holds_data
        if not drain_holds_data(drain):
            return None
        return getattr(self, "_qk_run_dir", None)

    # ------------------------------------------------------------------
    # Off-loop QK capture-aperture PER-REQUEST delivery (the QK port of the HS worker's per-request
    # delivery methods). Every method is a STRICT NO-OP -> None/False when the aperture per-request
    # path is not installed OR per_request mode is off, so it NEVER perturbs the eager /
    # shared-file / bank paths. TP=1 in scope. ``collective_rpc`` drops raw tensors, so every tensor
    # payload is serialized to ZSTD-PICKLE BYTES.
    # ------------------------------------------------------------------

    def flush_aperture_per_request(self):
        """collective_rpc-callable TEST read-hook for the per-request QK aperture-delivery parity oracle
        (mirrors the HS ``flush_aperture_per_request``): drive end-of-run delivery on the OFF-LOOP QK drain
        -- ``stop()`` (drain queue + join consumer + ``finalize_all`` for last-step stragglers), then
        ``index.pop_deliverable_qk()`` (per-layer ``{"q", "k_all"}`` via ``assemble_qk``) + ``free``
        each. Returns ZSTD-PICKLE BYTES of ``(deliverables, residency_after)`` where ``deliverables`` =
        ``{req_id: {layer(0-based): {"q": tensor, "k_all": [tensor,...], "layer_num": int}}}`` and
        ``residency_after`` MUST be 0. A second call after everything is freed serializes ``({}, 0)``.
        Strict no-op -> None unless per_request mode is on."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return None
        index = getattr(drain, "index", None)
        if index is None:
            return None
        stop = getattr(drain, "stop", None)
        if callable(stop):
            drain.stop()   # drain queue + join consumer + finalize_all; raises on consumer error
        mode = self._aperture_hookq_mode(drain)
        lock = getattr(drain, "_index_lock", None) or contextlib.nullcontext()
        with lock:
            popped = index.pop_deliverable_qk()
            for req_id, _ in popped:
                index.free(req_id)
            residency_after = len(index.live_req_ids())
        deliverables: dict = {}
        for req_id, per_layer in popped:
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
        """The captured request's granularity for the marshal ``hookq_mode`` -- from the drain header
        (set at install), falling back to the worker default."""
        header = getattr(drain, "header", None) or {}
        return header.get("hookq_mode") or getattr(self, "hookq_mode", "all_tokens")

    def get_aperture_per_request(self, external_req_id: str) -> bytes | None:
        """collective_rpc-callable PRODUCTION per-request retrieval for the off-loop QK capture-aperture
        demux path. Returns this request's marshaled QK probes as ZSTD-PICKLE BYTES, or None when it
        has not been delivered yet (still generating / off-loop finish not yet processed).

        BULK-POP-INTO-STASH: ``pop_deliverable_qk`` is a BULK drain, so drain ALL currently-finished
        requests ONCE into a per-req_id STASH of bytes, ``free`` each so residency drops, then return +
        remove the asked-for request's stashed bytes. Each request is delivered exactly once. Strict
        no-op -> None unless per_request mode is on."""
        drain = getattr(self, "_qk_drain", None)
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
        """Bulk-drain EVERY currently-finished QK aperture request into ``self._aperture_perreq_stash`` (bytes),
        returning the stash. Shared by ``get_aperture_per_request`` (retrieval) and ``clear_aperture_request``
        (abort cleanup) so BOTH drain the index identically and race-safe. POP + FREE-POPPED + the
        optional FREE-TARGET run under the DRAIN'S ``_index_lock`` as ONE atomic critical section --
        the same race the HS path's ``_drain_aperture_into_stash`` closes: a concurrent consumer
        ``mark_finished`` can never strand a request in ``_deliverable`` while it is freed from
        ``_entries``, and a later ``_handle_finish`` for an already-freed request is a no-op (it
        guards on ``live_req_ids()``).

        ``pop_deliverable_qk``'s per-request ``assemble_qk`` (its ``torch.cat`` of the q/k streams) DOES
        run under the lock -- deliberately: those are CPU-only cats of ALREADY-CLONED host tensors (no
        D2H, no GPU sync, no I/O, no deadlock), and the cat must stay ATOMIC with the pop+free for the
        same reason. Only the heavier MARSHAL (compress/pickle) runs OFF the lock, from the popped
        python data."""
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
                for rid in list(iter_matching_req_ids(index.live_req_ids(), free_external)):
                    index.free(rid)
        for req_id, per_layer in popped:
            stash[str(req_id)] = _marshal_perreq_qk(per_layer, conf, mode,
                                                    getattr(self, "_qk_shard", None))
        return stash

    def clear_aperture_request(self, external_req_id: str) -> None:
        """collective_rpc-callable ABORT/disconnect cleanup for the off-loop QK capture-aperture
        per-request path: free ALL of an aborted request's aperture state so residency returns to 0 -- the
        host-buffer ``PerRequestIndex`` entry + any stashed bytes AND the disk staging (+ a delivered-
        but-unconfirmed source dir). SINGLE-OWNER lifecycle: for the DISK route ``clear_request_disk``
        only MARKS the request aborted (the consumer discards the staging dir on the ``_Finish``); for
        the HOST route ``mark_host_aborted`` + the shared ``_drain_aperture_into_stash`` (``free_external``)
        drain AND free the live entry in ONE ``_index_lock`` hold. Strict no-op unless per_request."""
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
                mark_host(external_req_id)   # mark BEFORE the free so a racing note is suppressed
            self._drain_aperture_into_stash(drain, index, free_external=external_req_id)
            stash = getattr(self, "_aperture_perreq_stash", None)
            if stash:
                for rid in list(iter_matching_req_ids(stash, external_req_id)):
                    stash.pop(rid, None)
        return None

    def route_aperture_to_disk(self, req_id: str, dest: str) -> bool:
        """collective_rpc-callable SEAM for the router: mark ``req_id`` for the per-request DISK route
        on the off-loop QK drain -- its q + k rows stream to their own NVMe run_dir and, on finish, the
        file is offloaded to ``dest``. Must be called at request-start. Returns True when registered,
        False when the aperture per-request path is not installed."""
        drain = getattr(self, "_qk_drain", None)
        if drain is None or not getattr(drain, "per_request", False):
            return False
        route = getattr(drain, "route_to_disk", None)
        if not callable(route):
            return False
        # At TP > 1 every rank stages and offloads its OWN head shard of the request; each lands in
        # its own `dest/tp_rank_<r>/` (TP=1 keeps `dest` itself, unchanged) so the shards never
        # overwrite each other, and `aperture_reader.load_qk_aperture_tp(dest)` merges them.
        shard = getattr(self, "_qk_shard", None)
        if shard is not None and shard.tp_size > 1:
            from mia.graph.tp_shard import rank_dir_name
            dest = os.path.join(str(dest), rank_dir_name(shard.tp_rank))
        _aperture_disk_dbg(f"worker.route_aperture_to_disk(qk): req_id={req_id!r} (EXTERNAL) dest={dest!r}")
        route(str(req_id), str(dest))
        return True

    def confirm_aperture_delivery(self, req_id: str, timeout_s: float | None = None) -> bool | None:
        """collective_rpc-callable CONFIRM for a disk-routed QK request: block until its per-request
        file has landed at the client ``dest`` via the OffloadProcess. Returns True on delivery, False
        on timeout, None when the aperture per-request path is not installed. On confirm, also unlink the
        server-side staging source (bounded live NVMe). ``req_id`` is the EXTERNAL ``request_id`` (the
        driver passes the request's external id) -- the SAME key the offload job was submitted under,
        so ``offload.wait(req_id)`` matches; the internal->external divergence is resolved earlier in
        the drain (``_match_disk_route``) and never reaches here."""
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

    def aperture_residency(self):
        """collective_rpc-callable READ-ONLY residency query for the off-loop QK capture-aperture
        per-request path: returns ``(host_live_count, disk_residency)`` WITHOUT stopping the drain /
        popping / freeing (pollable mid-serving). Strict no-op -> None unless per_request mode is on."""
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
        """Drop this request's entries from the persistent ``_capture_seen`` gate once its
        capture has ended (finish/abort) — engine-thread-only (egress + collective_rpc
        retrieval run on one worker thread), so no lock needed. A bounded-by-total-requests
        leak otherwise, since the set is never cleared per-key elsewhere. ``getattr``-guarded
        so the eager path (no ``_capture_seen``, buffer mode only) is a no-op.
        """
        seen = getattr(self, "_capture_seen", None)
        if seen:
            self._capture_seen = {(r, m) for (r, m) in seen
                                  if not (r == external or r.startswith(f"{external}-"))}

    # ------------------------------------------------------------------
    # API serving: collective_rpc-callable artifact retrieval
    # ------------------------------------------------------------------

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve and remove captured QK states for a completed request.

        Matches by "{external_req_id}-" prefix because vLLM internally
        transforms the user-provided request_id into "{request_id}-{random_suffix}".

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
            # per-request accumulator that _cpu_list()/_k_all_cpu_list()/_k_all_compact()
            # feed when not None. Both stay None on the default path -> zero-cost, verbatim.
            _census_bucket = None
            _census_acc = None
            if _CENSUS_ON:
                from mia.graph.census import census_bucket, new_accumulator
                _census_bucket = census_bucket(layer_dict)
                _census_acc = new_accumulator()
            cpu_dict = {}
            with PROF.timed("worker.cpu_transfer.qk"):
                for mod_name, entry in layer_dict.items():
                    from torch.nn.utils.rnn import pad_sequence
                    # v0.6.0 score entry. Two producers: eager/op store "scores" directly;
                    # buffer mode stages Q/K + marks capture=="score", so recompute here.
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
                        # native: stack to the RPC format (unchanged behaviour).
                        with PROF.timed("cpu_transfer.qk.d2h"):
                            q_cpu = _cpu_list(entry["q"], _census_acc)
                        if _use_compact_kall(entry):
                            with PROF.timed("cpu_transfer.qk.kall"):
                                compact = _k_all_compact(entry, _census_acc)
                        else:
                            compact = None
                        if compact is not None:
                            # COMPACT: send full + prefix_ends (O(seq)); skip the O(seq^2) k pad
                            # and its pickle/zstd. The driver rebuilds the padded k_all.
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
                        # Hand off QUANTIZED (packed per-pass lists + scales + qmeta); the
                        # driver (MiaLLM.generate) dequantizes at the analysis boundary.
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
                from mia.graph.census import census_emit, census_record
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
        """collective_rpc-callable: reset the CUDA peak-allocated high-water mark and return
        current allocated bytes. Read-only test/monitoring introspection (the CB-OOM oracle
        measures peak GPU residency this way; worker-internal CUDA stats have no other
        channel to the driver). Called by string name so no function payload is serialized."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return int(torch.cuda.memory_allocated())
        return 0

    def capture_drain_stats(self) -> dict:
        """collective_rpc-callable: snapshot the streaming consumer's two residency sensors
        + backpressure counters, PROF capture counters, and CUDA mem high-water. Read-only
        introspection for the CB-OOM oracle. Called by string name (no cloudpickled function
        payload)."""
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
            from mia._profiler import PROF
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
        """Write captured Q/K for all requests in the batch to one artifact.

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
        cpu_cache: dict = {"config": self._conf, "qk_cache": {}}
        found_any = False
        flushed_ids: list = []  # req_ids popped this flush -> whose pages we deferred releasing

        with PROF.timed("worker.cpu_transfer.qk"):
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
                        # v0.6.0 score entry (eager/op store "scores"; buffer stages Q/K
                        # + marks capture=="score" -> recompute). Merge per-pass lists.
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
                            # v0.5.7 D2: a mixed score+qk batch (auto-select) can hit one module
                            # with both representations. Coexist in the per-module dict (score
                            # passes under "scores", qk under "q"/"k_all") instead of crashing —
                            # best-effort; a single analyzer reads its own representation.
                            existing = cpu_cache["qk_cache"].get(mod_name)
                            if existing is not None and "scores" in existing:
                                existing["scores"].extend(cpu_entry["scores"])
                            elif existing is not None:
                                existing.update({k: v for k, v in cpu_entry.items()
                                                 if k not in ("layer_num", "hookq_mode")})
                            else:
                                cpu_cache["qk_cache"][mod_name] = cpu_entry
                            continue
                        # Keep artifacts QUANTIZED onto disk (packed lists + scales + qmeta);
                        # the analyzer's disk loader dequantizes at read. Native (q_qmeta
                        # None) stores the float lists exactly as before.
                        q_qmeta = entry.get("_q_qmeta")
                        cpu_entry = {
                            "q": _cpu_list(entry["q"], _census_acc),
                            "layer_num": entry["layer_num"],
                            "hookq_mode": entry.get("hookq_mode", self.hookq_mode),
                        }
                        # k_all COMPACT (O(seq)): store the per-request unique keys + per-step
                        # prefix lengths; the reader rebuilds the growing prefixes
                        # ([full[:L] for L in ends]) byte-identically. Materializing the growing
                        # prefixes here is O(seq^2). Skip for the quant path (packs the expanded
                        # rows) and the eager path (no k_prefix_ends -> _k_all_compact returns
                        # None).
                        kc = _k_all_compact(entry, _census_acc) if q_qmeta is None else None
                        if kc is not None:
                            cpu_entry["k_full"] = [kc[0]]          # per-request list (merge-appended)
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
                        from mia.graph.census import census_emit, census_record
                        census_emit(census_record(worker="qk", sink="disk", req_id=req_id,
                                                   bucket=_census_bucket, acc=_census_acc))

        if not found_any:
            if consumer is not None:
                consumer.drain_writer_done(self)  # recycle any already-packed pages anyway
            return False

        from mia.graph.tp_shard import rank_dir_name
        tp_rank = _worker_tp_rank(self)
        run_dir = os.path.join(hook_dir, run_id, rank_dir_name(tp_rank))
        os.makedirs(run_dir, exist_ok=True)
        _attach_tp_shard(cpu_cache, self)

        # v0.6.0 score artifacts are ragged [S_q,S_k] lists — the fixed-shape safetensors
        # format doesn't fit them, so a score cache always saves as .pt (pickle handles
        # the lists). The RPC/probes path is unaffected.
        has_scores = any("scores" in e for e in cpu_cache["qk_cache"].values())
        # A quantized cache carries packed uint8 + per-token scale + qmeta that the fixed-shape
        # safetensors format doesn't batch -> save .pt (pickle holds the struct), same fallback
        # the score cache uses. Async path inherits via _save_safetensors, which repeats this gate.
        quant_on = any("q_qmeta" in e for e in cpu_cache["qk_cache"].values())

        # Hand serialize+write to a separate PROCESS (off the engine GIL) when armed. submit() is
        # NON-BLOCKING and returns False if the child is gone or the queue is full -> we fall
        # through to the SAME thread/inline ladder below, so a dead/backed-up child never hangs
        # the loop or silently loses the artifact.
        wp = getattr(self, "_writer_process", None)
        use_st = os.environ.get("MIA_USE_SAFETENSORS", "0") == "1"
        if wp is not None:
            with PROF.timed("worker.queue_put"):
                submitted = wp.submit("qk", cpu_cache, run_dir, self.hookq_mode, tp_rank,
                                      use_st, has_scores or quant_on, "qk.pt",
                                      req_ids=flushed_ids, block=True)
            if not submitted:  # child dead / bounded wait timed out -> data-safety inline (rare)
                from mia.graph.writer_process import note_submit_refused
                note_submit_refused(self, wp)   # LOUD: never a silent switch to inline saves
                # The inline fallback serializes cpu_cache directly (unlike the writer-process
                # pack, which torch.cat's into fresh storage) -- compact page-backed views to
                # owned storage first so pickle/torch.save doesn't re-serialize a whole aperture
                # page per narrow view.
                compact_page_backed_cache(cpu_cache)
                if use_st and not has_scores and not quant_on:
                    self._save_safetensors(cpu_cache, run_dir)
                else:
                    save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
                if consumer is not None:
                    consumer.release_req_pages(flushed_ids)  # cloned + written -> safe now
            # else: submitted -> pages released later by the writer's pack-done signal.
        else:
            # writer process OFF (MIA_WRITER_PROCESS=0): inline save. Same compaction
            # rationale as the fallback above -- this path never runs off-loop.
            compact_page_backed_cache(cpu_cache)
            if use_st and not has_scores and not quant_on:
                self._save_safetensors(cpu_cache, run_dir)
            else:
                save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
            if consumer is not None:
                consumer.release_req_pages(flushed_ids)

        if consumer is not None:
            consumer.drain_writer_done(self)  # recycle any pages the feeder already packed
        return run_dir

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        # The serialize body is a PURE function (graph/artifact_writer) so the writer PROCESS
        # and this inline path serialize byte-identically. This wrapper just supplies
        # self.hookq_mode + tp_rank.
        from mia.graph.artifact_writer import save_qk_cache_safetensors
        save_qk_cache_safetensors(cpu_cache, run_dir, self.hookq_mode, _worker_tp_rank(self))

