"""Stateless helpers for TokenHighlighter worker / analyzer plumbing.

Grouped like ``workers/_common.py``: model graph lookup, driver selection, trace I/O,
extended/suffix teacher prefill, and last-layer Q/K/V capture (probe_hookqk-style).
"""

from __future__ import annotations

import glob
import logging
import time
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.utils.hooks
from vllm.forward_context import get_forward_context

from vllm_hook_plugins.workers._common import get_query_metadata

# pyright: reportOperatorIssue=false, reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportOptionalCall=false, reportAttributeAccessIssue=false

_REQUIRED_CAPTURE_KEYS = ("Q", "K", "V", "h_L")
_OPTIONAL_CAPTURE_KEYS = ("h_input", "h_mid")
_CAPTURE_KEYS = _REQUIRED_CAPTURE_KEYS + _OPTIONAL_CAPTURE_KEYS


# ---------------------------------------------------------------------------
# Model graph
# ---------------------------------------------------------------------------


def _last_transformer_block(model: torch.nn.Module) -> torch.nn.Module:
    for path in ("model.layers", "transformer.h", "layers"):
        obj: Any = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            if hasattr(obj, "__len__") and len(obj) > 0:
                block = obj[-1]
                if isinstance(block, torch.nn.Module):
                    return block
    raise RuntimeError("Could not locate last transformer block.")


def _attention_on_block(layer: torch.nn.Module) -> torch.nn.Module:
    for name in ("self_attn", "attn", "attention"):
        if (attn := getattr(layer, name, None)) is not None:
            return attn
    raise RuntimeError("Could not locate last self-attention module.")


def locate_last_attention(model: torch.nn.Module) -> torch.nn.Module:
    return _attention_on_block(_last_transformer_block(model))


# Dot-paths for pre-lm_head norm on common HF / vLLM decoder-only causal LMs.
_FINAL_NORM_PATHS = (
    "model.norm",                     # LLaMA, Mistral, Qwen
    "norm",
    "transformer.ln_f",               # GPT-2
    "model.decoder.final_layer_norm", # OPT
    "decoder.final_layer_norm",
    "model.gpt_neox.final_layer_norm",# GPT-NeoX
    "gpt_neox.final_layer_norm",
    "model.final_layernorm",
    "final_layernorm",
)

# Attribute names for the attention input norm on a single transformer block
_INPUT_NORM_ATTRS = (
    "input_layernorm",      # LLaMA, Qwen, Mistral, Gemma
    "ln_1",                 # GPT-2, GPT-NeoX
    "self_attn_layer_norm", # OPT
    "attention_norm",       # Custom Triton wrappers
)

# Attribute names for the FFN/pre-MLP norm on a single transformer block
_FFN_NORM_ATTRS = (
    "post_attention_layernorm",  # LLaMA, Qwen, Mistral, Gemma
    "ln_2",                      # GPT-2, GPT-NeoX
    "final_layer_norm",          # OPT-style block
    "ffn_norm",                  # custom wrappers
)

def _get_submodule(model: torch.nn.Module, path: str) -> torch.nn.Module | None:
    obj: Any = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, torch.nn.Module) else None


def locate_final_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    """Find the last hidden-state norm before ``lm_head`` (architecture-agnostic lookup)."""
    seen: set[int] = set()
    for path in _FINAL_NORM_PATHS:
        mod = _get_submodule(model, path)
        if mod is None or id(mod) in seen:
            continue
        if getattr(mod, "weight", None) is None and getattr(mod, "bias", None) is None:
            continue
        seen.add(id(mod))
        return mod
    return None

def locate_input_norm(
    model: torch.nn.Module
) -> torch.nn.Module | None:
    """Find the attention input norm for the final transformer block.
    
    Traverses the model to the last block and extracts the input norm layer 
    used to normalize the residual stream before the attention QKV projections.
    """
    try:
        block = _last_transformer_block(model)
    except RuntimeError:
        return None
        
    for attr in _INPUT_NORM_ATTRS:
        if (norm := getattr(block, attr, None)) is not None:
            # Verify it's a module with learnable parameters
            if getattr(norm, "weight", None) is not None or getattr(norm, "bias", None) is not None:
                return norm
    return None


def locate_ffn_input_norm(
    model: torch.nn.Module,
) -> torch.nn.Module | None:
    """Find the FFN input norm (norm between attention and MLP) for the final block."""
    try:
        block = _last_transformer_block(model)
    except RuntimeError:
        return None
    for attr in _FFN_NORM_ATTRS:
        if (norm := getattr(block, attr, None)) is None:
            continue
        if getattr(norm, "weight", None) is not None or getattr(norm, "bias", None) is not None:
            return norm
    return None

# Try to capture common norm families: rmsnorm, layernorm, gemma_rms
# WARNING: Scoring procedure and approximation may not be accurate for custom or otherwise specialized norms 
def infer_norm_kind(norm: torch.nn.Module) -> str:
    """``rmsnorm`` | ``layernorm`` | ``gemma_rms`` for analytic forward / Jacobian."""
    name = type(norm).__name__.lower()
    if "gemma" in name:
        return "gemma_rms"
    if getattr(norm, "bias", None) is not None:
        return "layernorm"
    if "layernorm" in name.replace("_", ""):
        return "layernorm"
    return "rmsnorm"


def export_norm_spec(norm: torch.nn.Module | None, norm_placement) -> dict[str, Any] | None:
    """CPU snapshot of norm weights for ``highlighter_activations.pt``
    either before final attention layer ("attn") or before lm_head ("final")."""
    if norm is None:
        return None
    weight = getattr(norm, "weight", None)
    bias = getattr(norm, "bias", None)
    if weight is None and bias is None:
        return None
    return {
        "norm_placement": norm_placement,
        "norm_kind": infer_norm_kind(norm),
        "weight": weight.detach().cpu() if weight is not None else None,
        "bias": bias.detach().cpu() if bias is not None else None,
        "eps": float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))),
        "has_bias": bias is not None,
    }

def export_final_norm_spec(norm: torch.nn.Module | None) -> dict[str, Any] | None:
    """Wrapper to export the bottleneck LM-head norm."""
    return export_norm_spec(norm, norm_placement="final")


def export_input_norm_spec(norm: torch.nn.Module | None) -> dict[str, Any] | None:
    """Wrapper to export the attention input norm."""
    return export_norm_spec(norm, norm_placement="attn")


def linear_weight(
    module: torch.nn.Module,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return unquantized weights in ``[out_features, in_features]`` layout.

    vLLM fp16 layers expose ``.weight``; AWQ/GPTQ-style layers use ``qweight`` and
    require dequantization before the analyzer can run matmul-based influence math.
    """
    weight = getattr(module, "weight", None)
    if weight is not None:
        w = cast(torch.Tensor, weight)
        return w if out_dtype is None else w.to(dtype=out_dtype)

    try:
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_and_maybe_dequant_weights,
        )

        if isinstance(module, LinearBase):
            dtype = out_dtype or torch.float16
            return get_and_maybe_dequant_weights(module, out_dtype=dtype)
    except Exception:
        pass

    qweight = getattr(module, "qweight", None)
    scales = getattr(module, "scales", None)
    qzeros = getattr(module, "qzeros", None)
    if qweight is not None and scales is not None and qzeros is not None:
        from vllm import _custom_ops as ops

        w = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
        return w if out_dtype is None else w.to(dtype=out_dtype)

    raise AttributeError(
        f"Could not resolve weight for {type(module).__name__} "
        f"(expected .weight or quantized qweight/scales/qzeros)."
    )


def lm_head_weight(model: torch.nn.Module) -> torch.Tensor:
    """Return unembedding / ``lm_head`` weights for affirmation loss gradients."""
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        out = getter()
        if isinstance(out, torch.nn.Module):
            try:
                return linear_weight(out)
            except AttributeError:
                pass
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, torch.nn.Module) and type(lm_head).__name__ != "PPMissingLayer":
        try:
            return linear_weight(lm_head)
        except AttributeError:
            pass
    raise RuntimeError(f"Could not resolve output embeddings for {type(model).__name__}.")



def get_attn_key_value_weights(
    last_attn: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(W_O, W_V, W_K)`` for forward-attribution key/value paths.

    Handles fused ``qkv_proj`` (vLLM) vs separate projections (HF-style layouts).
    """
    attn = cast(Any, last_attn)
    w_o = linear_weight(attn.o_proj)
    # Fused QKV proj: common with vLLM models
    if hasattr(last_attn, "qkv_proj"):
        qkv_w = linear_weight(attn.qkv_proj)
        q_size, kv_size = int(attn.q_size), int(attn.kv_size)
        return w_o, qkv_w[q_size + kv_size : q_size + 2 * kv_size], qkv_w[q_size : q_size + kv_size]
    
    # Separate QKV projections: common with HF models
    return w_o, linear_weight(attn.v_proj), linear_weight(attn.k_proj)


def get_attn_query_weight(last_attn: torch.nn.Module) -> torch.Tensor:
    """Return ``W_Q`` for the last attention module."""
    attn = cast(Any, last_attn)
    if hasattr(last_attn, "qkv_proj"):
        qkv_w = linear_weight(attn.qkv_proj)
        q_size = int(attn.q_size)
        return qkv_w[:q_size]
    return linear_weight(attn.q_proj)


# ---------------------------------------------------------------------------
# Driver selection / soft removal
# ---------------------------------------------------------------------------


def flag_driver_tokens(
    scores: list[float],
    *,
    threshold_k: float,
    mode: str | None = None,
    alpha: float | None = None,
) -> list[int]:
    """Return prompt token indices selected as drivers (mean+std or top-α).

    ``mean_std``: score > mean + k·std. ``top_percentage`` (default): top α fraction.
    Used for mitigate embedding scaling and trace metadata.
    """
    if not scores:
        return []
    mode = mode or "top_percentage"
    s = torch.tensor(scores, dtype=torch.float32)
    if mode == "top_percentage":
        if alpha is None:
            alpha = 0.25
        k = max(1, int(len(scores) * alpha))
        return sorted(torch.argsort(s, descending=True)[:k].tolist())
    threshold = s.mean() + threshold_k * s.std(unbiased=False)
    return [i for i, v in enumerate(scores) if float(v) > float(threshold)]


# ---------------------------------------------------------------------------
# Scheduler / trace I/O
# ---------------------------------------------------------------------------


def iter_prompt_batches(request: Any) -> list[list[int]]:
    """Collect ``prompt_token_ids`` from scheduler output (new + cached metadata)."""
    batches: list[list[int]] = []
    for attr in ("seq_group_metadata_list", "scheduled_new_reqs"):
        for obj in getattr(request, attr, []) or []:
            if prompt_ids := getattr(obj, "prompt_token_ids", None):
                batches.append(list(prompt_ids))
    return batches


def prompt_prefill_done(model_runner: Any, prompt_ids: list[int]) -> bool:
    """True when a matching in-flight request has ``num_computed_tokens >= len(prompt)``.

    Walks ``input_batch`` and compares the request prefix of ``prompt_token_ids`` to
    ``prompt_ids``. Used to know when capture can finish and suffix teacher can run.
    """
    plen = len(prompt_ids)
    requests = getattr(model_runner, "requests", None)
    input_batch = getattr(model_runner, "input_batch", None)
    if requests is None or input_batch is None or input_batch.num_reqs == 0:
        return False
    for req_id in list(input_batch.req_ids)[: input_batch.num_reqs]:
        req = requests.get(req_id)
        if req is None or req.prompt_token_ids is None:
            continue
        if len(req.prompt_token_ids) < plen or list(req.prompt_token_ids[:plen]) != prompt_ids:
            continue
        if int(req.num_computed_tokens) < plen:
            return False
        return True
    return False


def _highlighter_run_dir(hook_dir: str | None, run_id_file: str | None) -> str | None:
    if not (hook_dir and run_id_file and os.path.exists(run_id_file)):
        return None
    with open(run_id_file, "r") as f:
        ids = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    if not ids:
        return None
    run_dir = os.path.join(hook_dir, ids[-1], "tp_rank_0")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_highlighter_file(
    hook_dir: str | None,
    run_id_file: str | None,
    sequences: list[dict],
    *,
    filename: str,
    extra: dict[str, Any] | None = None,
) -> None:
    run_dir = _highlighter_run_dir(hook_dir, run_id_file)
    if not run_dir or not sequences:
        return
    run_id = os.path.basename(os.path.dirname(run_dir))
    payload: dict[str, Any] = {"run_id": run_id, "sequences": sequences}
    if extra:
        payload.update(extra)
    torch.save(payload, os.path.join(run_dir, filename))


def export_forward_attr_weights(
    model: torch.nn.Module,
    *,
    runtime_rope_probe: dict[str, Any] | None = None,
    include_unembedding: bool = False,
) -> dict[str, Any]:
    """Snapshot tensors needed for ``compute_grad_influences`` within analyzer without 
    reloading the model with `from_pretrained` during analysis.

    Saved alongside ``highlighter_activations.pt`` so the analyzer can run pure math on
    CPU/GPU using captures + this bundle (vLLM worker weights, including fused QKV layouts).

    The unembedding ``W_U`` (``[vocab, d_model]``) is large (hundreds of MB) and is only
    used to form the affirmation-loss gradient ``g`` at the few generation positions. The
    worker precomputes that tiny ``g`` per sequence (see ``_finish_capture``), so ``W_U`` is
    omitted by default. Set ``include_unembedding=True`` for legacy/standalone bundles whose
    analyzer recomputes ``g`` from ``W_U`` + ``target_ids``.
    """
    last_attn = locate_last_attention(model)
    w_o, w_v, w_k = get_attn_key_value_weights(last_attn)
    w_q = get_attn_query_weight(last_attn)
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("export_forward_attr_weights: model has no config")
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))

    bundle: dict[str, Any] = {
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "W_O": w_o.detach().cpu(),
        "W_Q": w_q.detach().cpu(),
        "W_V": w_v.detach().cpu(),
        "W_K": w_k.detach().cpu(),
    }
    if include_unembedding:
        bundle["W_U"] = lm_head_weight(model).detach().cpu()

    bundle["input_norm"] = export_norm_spec(locate_input_norm(model), "attn")
    bundle["final_norm"] = export_norm_spec(locate_final_norm(model), "final")
    # RoPE metadata for inverse VJP in analyzer:
    # only apply inverse rotation when this attention path actually uses RoPE.
    rope_spec: dict[str, Any] = {
        "has_rope": False,
        "rotate_q": None,
        "rotate_k": None,
        "rotate_v": None,
    }
    rope_theta = getattr(cfg, "rope_theta", None)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    # Some configs (e.g. vLLM's Qwen2) expose the RoPE base only inside ``rope_scaling``.
    if rope_theta is None and isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta")
    if rope_theta is not None:
        rope_spec["has_rope"] = True
        rope_spec["rope_theta"] = float(rope_theta)
    partial_rotary = getattr(cfg, "partial_rotary_factor", None)
    if partial_rotary is not None:
        rope_spec["partial_rotary_factor"] = float(partial_rotary)
    if rope_scaling is not None:
        rope_spec["has_rope"] = True
        rope_spec["rope_scaling"] = (
            dict(rope_scaling) if isinstance(rope_scaling, dict) else rope_scaling
        )
    for mod in (last_attn, model, getattr(model, "model", None)):
        if mod is None:
            continue
        rotary = getattr(mod, "rotary_emb", None)
        inv = getattr(rotary, "inv_freq", None) if rotary is not None else None
        if inv is not None:
            rope_spec["has_rope"] = True
            rope_spec["inv_freq"] = inv.detach().cpu()
            break
    if runtime_rope_probe:
        for key in ("has_rope", "rotate_q", "rotate_k", "rotate_v", "probe_rel_tol"):
            if key in runtime_rope_probe:
                rope_spec[key] = runtime_rope_probe[key]
    bundle["rope_spec"] = rope_spec
    return bundle


def save_highlighter_sequences(
    hook_dir: str | None, run_id_file: str | None, sequences: list[dict]
) -> None:
    """Write ``highlighter.pt`` (scores + soft indices) under the latest run id."""
    _save_highlighter_file(hook_dir, run_id_file, sequences, filename="highlighter.pt")
    logging.info(f"Saved {len(sequences)} sequences to {hook_dir}/{run_id_file}/highlighter.pt")


def save_highlighter_activations(
    hook_dir: str | None,
    run_id_file: str | None,
    sequences: list[dict],
    *,
    weight_bundle: dict[str, Any] | None = None,
) -> None:
    """Write ``highlighter_activations.pt`` (captures + optional weight bundle) under the run id."""
    extra = {"weight_bundle": weight_bundle} if weight_bundle else None
    _save_highlighter_file(
        hook_dir,
        run_id_file,
        sequences,
        filename="highlighter_activations.pt",
        extra=extra,
    )


def load_highlighter_artifact(
    hook_dir: str,
    run_id: str,
    filename: str,
    *,
    wait_seconds: float = 0.0,
    poll_interval: float = 0.05,
    quiet: bool = False,
) -> dict | None:
    """Load newest ``filename`` under ``hook_dir/{run_id}/**`` with optional short wait.

    Some vLLM capture flows finish writing the artifact just after ``generate`` returns.
    ``wait_seconds`` allows a bounded poll before declaring the file missing.
    """
    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        paths = glob.glob(os.path.join(hook_dir, run_id, "**", filename), recursive=True)
        if paths:
            return torch.load(max(paths, key=os.path.getmtime), map_location="cpu")
        if time.time() >= deadline:
            if not quiet:
                logging.warning(f"No {filename} found under {hook_dir}/{run_id}/**")
            return None
        time.sleep(max(0.01, poll_interval))


def build_highlighter_record(
    prompt_ids: list[int],
    score_list: list[float],
    *,
    tokenizer: Any,
    soft_beta: float,
    threshold_k: float,
    mode: str | None = None,
    alpha: float | None = None,
) -> dict:
    """Build one sequence entry for ``highlighter.pt`` (analyzer output artifact)."""
    mode = mode or "top_percentage"
    applied_alpha = alpha if alpha is not None else 0.25
    record = {
        "token_ids": list(prompt_ids),
        "token_scores": score_list,
        "soft_indices": flag_driver_tokens(
            score_list, threshold_k=threshold_k, mode=mode, alpha=applied_alpha
        ),
        "soft_beta": soft_beta,
        "applied_mode": mode,
        "applied_threshold_k": threshold_k,
        "applied_alpha": applied_alpha,
    }
    if tokenizer is not None:
        record["tokens"] = [
            tokenizer.decode([tid], skip_special_tokens=False) for tid in prompt_ids
        ]
    return record


# ---------------------------------------------------------------------------
# Forward-attr capture state
# ---------------------------------------------------------------------------


@dataclass
class ForwardAttrCapture:
    """Per-worker hook buffers for one capture phase (see ``HighlighterWorker._cap``)."""

    active: bool = False  # gates hooks during prefill forwards only
    live: dict[str, torch.Tensor] = field(default_factory=dict)  # Q, K, V, h_L, h_input
    meta: dict[str, Any] = field(default_factory=dict)  # query_start_loc, input_ids


def capture_ready(live: dict[str, torch.Tensor]) -> bool:
    """True when required last-layer tensors were stored by hooks."""
    return all(key in live and live[key].numel() > 0 for key in _REQUIRED_CAPTURE_KEYS)


def teacher_forced_token_ids(prompt_ids: list[int], target_ids: list[int]) -> list[int]:
    """Input token list for causal teacher forcing: ``prompt + target[:-1]``."""
    return list(prompt_ids) + list(target_ids[:-1])


def capture_ready_for_teacher(
    prompt_ids: list[int], target_ids: list[int], live: dict[str, torch.Tensor]
) -> bool:
    """True when extended prefill captured at least ``len(prompt) + len(target) - 1`` rows."""
    if not capture_ready(live):
        return False
    return int(live["h_L"].size(0)) >= len(teacher_forced_token_ids(prompt_ids, target_ids))


def _slice_capture(
    live: dict[str, torch.Tensor],
    meta: dict[str, Any],
    token_ids: list[int],
    *,
    missing_msg: str,
    match_msg: str,
) -> dict[str, torch.Tensor]:
    """Slice flat-batch Q/K/V/h_L/(optional h_input) by ``query_start_loc`` (probe-style).

    vLLM lays the batch out as one concatenated sequence; ``meta`` maps a request's
    token ids to ``[start:end)`` indices. Without meta, assumes a single-request batch.
    """
    if not capture_ready(live):
        raise RuntimeError(missing_msg)
    n = len(token_ids)
    if not meta:
        # Single-request path: hooks filled rows 0..n-1 in order.
        if live["h_L"].size(0) < n:
            raise RuntimeError(
                f"forward_attr: capture length {live['h_L'].size(0)} < expected {n}."
            )
        out = {
            k: live[k][:n].detach()
            for k in _REQUIRED_CAPTURE_KEYS
        }
        for key in _OPTIONAL_CAPTURE_KEYS:
            if key in live:
                out[key] = live[key][:n].detach()
        return out

    qsl, flat_ids = meta["query_start_loc"], meta["input_ids"]
    for r in range(int(qsl.numel()) - 1):
        start = int(qsl[r].item())
        seg = flat_ids[start : int(qsl[r + 1].item())].tolist()
        # Match this batch row to the expected token prefix.
        if len(seg) >= n and seg[:n] == token_ids:
            out = {
                k: live[k][start : start + n].detach()
                for k in _REQUIRED_CAPTURE_KEYS
            }
            for key in _OPTIONAL_CAPTURE_KEYS:
                if key in live:
                    out[key] = live[key][start : start + n].detach()
            return out
    raise RuntimeError(match_msg)


def slice_teacher_prefill_capture(
    prompt_ids: list[int],
    target_ids: list[int],
    live: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Slice activations for ``prompt + target[:-1]`` after extended prefill."""
    full_ids = teacher_forced_token_ids(prompt_ids, target_ids)
    return _slice_capture(
        live,
        meta,
        full_ids,
        missing_msg="forward_attr: extended prefill did not capture activations.",
        match_msg=(
            f"forward_attr: could not match extended capture (teacher len={len(full_ids)})."
        ),
    )


def slice_real_prefill_capture(
    prompt_ids: list[int], live: dict[str, torch.Tensor], meta: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Slice activations for the prompt segment from a normal prefill forward."""
    n = len(prompt_ids)
    return _slice_capture(
        live,
        meta,
        prompt_ids,
        missing_msg="forward_attr: hooks did not capture during prefill.",
        match_msg=f"forward_attr: could not match prefill capture (prompt len={n}).",
    )


# ---------------------------------------------------------------------------
# Extended teacher prefill (one scheduler step)
# ---------------------------------------------------------------------------


@dataclass
class TeacherPrefillExtendPlan:
    """Per-request state for extended prefill (scheduler bump + post-forward restore)."""

    req_id: str
    prompt_len: int
    suffix_ids: list[int]  # target[:-1]
    saved_prompt_row: Any = None  # prompt slice of token_ids_cpu before staging


def _prompt_matches(req_prompt: list[int] | None, prompt_ids: list[int]) -> bool:
    plen = len(prompt_ids)
    return req_prompt is not None and list(req_prompt[:plen]) == prompt_ids


def prefill_chunked_for_prompt(
    scheduler_output: Any, model_runner: Any, prompt_ids: list[int]
) -> bool:
    """True when the first prefill step schedules fewer tokens than ``len(prompt_ids)``."""
    if scheduler_output is None:
        return False
    plen = len(prompt_ids)

    def _chunked(computed: int, scheduled: int) -> bool:
        return computed == 0 and 0 < scheduled < plen

    # Check for newly scheduled requests within this step that match the prompt
    for nrd in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        if _prompt_matches(nrd.prompt_token_ids, prompt_ids):
            scheduled = int(scheduler_output.num_scheduled_tokens.get(nrd.req_id, 0))
            if _chunked(int(nrd.num_computed_tokens), scheduled):
                return True
    # Check for in-flight requests that match the prompt
    for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
        req = model_runner.requests.get(req_id)
        if _prompt_matches(getattr(req, "prompt_token_ids", None), prompt_ids):
            if _chunked(int(req.num_computed_tokens), int(scheduled)):
                return True
    return False


def plan_teacher_prefill_extend(
    scheduler_output: Any,
    model_runner: Any,
    pending_prompts: list[list[int]],
    target_ids: list[int],
    already_extended: set[str],
) -> list[TeacherPrefillExtendPlan]:
    """Bump ``num_scheduled_tokens`` so one forward runs ``prompt + target[:-1]``.

    Only applies on the first prefill step when the full prompt fits in that step
    (``computed == 0`` and ``scheduled >= plen``). Chunked prompts skip bump here.
    """
    suffix_ids = list(target_ids[:-1])
    if not suffix_ids:
        return []
    extra = len(suffix_ids)
    plans: list[TeacherPrefillExtendPlan] = []
    seen: set[str] = set()

    def _bump(req_id: str, prompt_ids: list[int], computed: int, scheduled: int) -> None:
        if req_id in already_extended or req_id in seen or scheduled <= 0:
            return
        plen = len(prompt_ids)
        # First chunk must cover entire prompt; otherwise use suffix capture path.
        if computed != 0 or scheduled < plen:
            return
        scheduler_output.num_scheduled_tokens[req_id] = scheduled + extra
        scheduler_output.total_num_scheduled_tokens = int(
            getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
        ) + extra
        seen.add(req_id)
        plans.append(TeacherPrefillExtendPlan(req_id, plen, suffix_ids))

    for prompt_ids in pending_prompts:
        for nrd in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            if _prompt_matches(nrd.prompt_token_ids, prompt_ids):
                _bump(
                    nrd.req_id,
                    prompt_ids,
                    int(nrd.num_computed_tokens),
                    int(scheduler_output.num_scheduled_tokens.get(nrd.req_id, 0)),
                )
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            req = model_runner.requests.get(req_id)
            if _prompt_matches(getattr(req, "prompt_token_ids", None), prompt_ids):
                _bump(req_id, prompt_ids, int(req.num_computed_tokens), int(scheduled))
    return plans


def stage_teacher_prefill_tokens(
    model_runner: Any, plans: list[TeacherPrefillExtendPlan]
) -> None:
    """Write ``target[:-1]`` into ``input_batch`` after ``_update_states``.

    vLLM fills prompt slots from ``prompt_token_ids``; this overwrites the tail of
    the row so the upcoming forward sees teacher-forced token ids.
    """
    batch = model_runner.input_batch
    for plan in plans:
        idx = batch.req_id_to_index[plan.req_id]
        plan.saved_prompt_row = batch.token_ids_cpu[idx, : plan.prompt_len].copy()
        batch.token_ids_cpu[idx, plan.prompt_len : plan.prompt_len + len(plan.suffix_ids)] = (
            plan.suffix_ids
        )


def restore_teacher_prefill_state(
    model_runner: Any, plans: list[TeacherPrefillExtendPlan]
) -> None:
    """Reset ``num_computed_tokens`` ("cursor") to ``prompt_len`` so decode ignores teacher tokens.

    KV for teacher positions may remain in the paged cache, but the scheduler cursor
    and token row are rewound so generation continues from the prompt boundary only.
    """
    batch = model_runner.input_batch
    for plan in plans:
        plen, m = plan.prompt_len, len(plan.suffix_ids)
        if (req := model_runner.requests.get(plan.req_id)) is not None:
            req.num_computed_tokens = plen
        idx = batch.req_id_to_index.get(plan.req_id)
        if idx is None:
            continue
        # Mirror cursor reset on batch tensors the runner reads on the next step.
        batch.num_computed_tokens_cpu[idx] = plen
        batch.num_computed_tokens_cpu_tensor[idx] = plen
        model_runner.num_computed_tokens[idx] = plen
        if plan.saved_prompt_row is not None:
            batch.token_ids_cpu[idx, :plen] = plan.saved_prompt_row
            batch.token_ids_cpu[idx, plen : plen + m] = 0


def find_request_index_for_prompt(model_runner: Any, prompt_ids: list[int]) -> int:
    """``input_batch`` row index with completed prefill matching ``prompt_ids``."""
    batch, plen = model_runner.input_batch, len(prompt_ids)
    for req_index in range(batch.num_reqs):
        req = model_runner.requests.get(batch.req_ids[req_index])
        if req is None or not _prompt_matches(req.prompt_token_ids, prompt_ids):
            continue
        if int(batch.num_computed_tokens_cpu[req_index]) >= plen:
            return req_index
    raise RuntimeError(
        f"forward_attr: no in-flight request with completed prefill (prompt len={plen})."
    )


def merge_real_and_teacher_captures(
    prompt_len: int,
    real_capture: dict[str, torch.Tensor],
    teacher_capture: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Prompt rows from scheduler prefill; suffix rows from teacher pass (suffix mode).

    ``teacher_capture`` is usually padded via ``expand_suffix_capture``; prompt
    positions are overwritten with activations from the real prefill hooks.
    """
    if not capture_ready(real_capture):
        raise RuntimeError("forward_attr: real prefill capture required for prompt merge.")
    merged: dict[str, torch.Tensor] = {}
    for key, t in teacher_capture.items():
        if torch.is_tensor(t):
            merged[key] = t.clone()
    for key in real_capture:
        if key not in merged and torch.is_tensor(real_capture[key]):
            merged[key] = real_capture[key].clone()
    for key in tuple(merged.keys()):
        if key not in real_capture:
            continue
        end = min(prompt_len, real_capture[key].size(0), merged[key].size(0))
        merged[key][:end] = real_capture[key][:end]  # prompt slice from scheduler path
    return merged


# ---------------------------------------------------------------------------
# Suffix teacher pass (chunked prefill fallback)
# ---------------------------------------------------------------------------


def expand_suffix_capture(
    prompt_len: int,
    suffix_capture: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Pad suffix-only hook tensors to full teacher-forced length ``prompt_len + |target|-1``.

    The suffix forward runs ``target[:-1]`` only, so hooks see ``m`` rows. Forward attribution
    expects ``prompt_len + m`` positions (prompt slots + teacher slots). Prefix zeros stand in
    for prompt positions; ``merge_real_and_teacher_captures`` overwrites ``[:prompt_len]`` with
    real prefill activations.
    """
    if not capture_ready(suffix_capture):
        raise RuntimeError("forward_attr: suffix capture missing Q/K/V/h_L.")
    out: dict[str, torch.Tensor] = {}
    for key, t in suffix_capture.items():
        if not torch.is_tensor(t):
            continue
        # [0:prompt_len) placeholder; [prompt_len:] filled from suffix-only forward.
        full = torch.zeros(
            (prompt_len + t.size(0),) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device
        )
        full[prompt_len:] = t
        out[key] = full
    return out


def _run_runner_forward(
    model_runner: Any,
    model: torch.nn.Module,
    *,
    tokens_per_req: dict[str, int],
) -> None:
    """Run one vLLM model forward outside the normal scheduler step.

    vLLM V1 does not expose ``model.forward(kv_caches=...)``; this is the minimal
    ``GPUModelRunner.execute_model`` path: ``_prepare_inputs`` → attention metadata →
    ``set_forward_context`` → ``model(...)``. Used by ``vllm_teacher_suffix_capture`` after
    prompt KV is already in the paged cache.
    """
    import numpy as np
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices

    batch = model_runner.input_batch
    # Per-request token counts for this ad-hoc forward (one hot request for suffix pass).
    sched_np = np.array(
        [int(tokens_per_req.get(batch.req_ids[i], 0)) for i in range(batch.num_reqs)],
        dtype=np.int32,
    )
    n_tok = int(sched_np.sum())
    sched = SchedulerOutput.make_empty()
    sched.total_num_scheduled_tokens = n_tok
    sched.num_scheduled_tokens = {batch.req_ids[i]: int(sched_np[i]) for i in range(batch.num_reqs)}

    with model_runner.synchronize_input_prep():
        model_runner._prepare_inputs(sched, sched_np)
        max_q = int(sched_np.max()) if n_tok else 0
        _, batch_desc, should_ubatch, num_across_dp, _ = (
            model_runner._determine_batch_execution_and_padding(
                num_tokens=n_tok,
                num_reqs=batch.num_reqs,
                num_scheduled_tokens_np=sched_np,
                max_num_scheduled_tokens=max_q,
                use_cascade_attn=False,
                force_eager=True,
            )
        )
        n_pad = batch_desc.num_tokens
        n_reqs_pad = batch_desc.num_reqs or batch.num_reqs
        ubatch_slices, ubatch_slices_pad = maybe_create_ubatch_slices(
            should_ubatch, sched_np, n_pad, n_reqs_pad, model_runner.parallel_config.num_ubatches
        )
        # Get slot mappings into KV cache for the padded batch.
        slots_gid, slots_layer = model_runner._get_slot_mappings(
            num_tokens_padded=n_tok,
            num_reqs_padded=batch.num_reqs,
            num_tokens_unpadded=n_tok,
            ubatch_slices=ubatch_slices_pad,
        )
        attn_md, _ = model_runner._build_attention_metadata(
            num_tokens=n_tok,
            num_reqs=batch.num_reqs,
            max_query_len=max_q,
            ubatch_slices=ubatch_slices,
            num_scheduled_tokens=sched.num_scheduled_tokens,
            slot_mappings=slots_gid,
        )
        ids, embeds, pos, inter, model_kw, _ = model_runner._preprocess(sched, n_pad, None)

    # Attention kernels read slot_mapping / metadata from forward context.
    with set_forward_context(
        attn_md,
        model_runner.vllm_config,
        num_tokens=n_pad,
        num_tokens_across_dp=num_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=batch_desc,
        ubatch_slices=ubatch_slices_pad,
        slot_mapping=slots_layer,
    ):
        with torch.no_grad():
            # Run the model forward.
            model(
                input_ids=ids,
                positions=pos,
                intermediate_tensors=inter,
                inputs_embeds=embeds,
                **model_kw,
            )


def vllm_teacher_suffix_capture(
    model_runner: Any,
    model: torch.nn.Module,
    prompt_ids: list[int],
    target_ids: list[int],
) -> None:
    """Forward ``target[:-1]`` on existing prompt KV, then restore prompt-only decode boundary.

    Used when extended prefill is unavailable (chunked prompt). Cursor stays at
    ``prompt_len`` so queries attend to prompt KV only and the target sequence
    is not mistaken for part of the prompt; hooks run inside the worker's
    temporary ``register_forward_attr_hooks`` call.
    """
    suffix = list(target_ids[:-1])
    if not suffix:
        raise RuntimeError("forward_attr: need at least two target tokens.")
    if getattr(model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("forward_attr: suffix capture needs idle model_runner state.")

    plen = len(prompt_ids)
    req_idx = find_request_index_for_prompt(model_runner, prompt_ids)
    batch = model_runner.input_batch
    req_id = batch.req_ids[req_idx]
    req = model_runner.requests[req_id]
    saved_row = batch.token_ids_cpu[req_idx, :plen].copy()

    # Stage teacher ids; cursor (num_computed_tokens) at prompt end (not after teacher tokens).
    batch.token_ids_cpu[req_idx, plen : plen + len(suffix)] = suffix
    batch.num_computed_tokens_cpu[req_idx] = plen
    batch.num_computed_tokens_cpu_tensor[req_idx] = plen
    req.num_computed_tokens = plen
    try:
        # Retrieve counts of suffix tokens for each request in the batch.
        counts = {
            batch.req_ids[i]: (len(suffix) if i == req_idx else 0)
            for i in range(batch.num_reqs)
        }
        # Run ad-hoc forward pass (simulates execute_model() path).
        _run_runner_forward(model_runner, model, tokens_per_req=counts)
    finally:
        # Restore cursor (num_computed_tokens) to prompt end and reset teacher tokens.
        restore_teacher_prefill_state(
            model_runner, [TeacherPrefillExtendPlan(req_id, plen, suffix, saved_row)]
        )


# ---------------------------------------------------------------------------
# Last-layer hooks (ProbeHookQKWorker-style)
# ---------------------------------------------------------------------------


def _decoder_layer_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        h0 = cast(torch.Tensor, output[0])
        if len(output) == 2 and isinstance(output[1], torch.Tensor):
            return h0 + cast(torch.Tensor, output[1])
        return h0
    return cast(torch.Tensor, output)


def _decoder_layer_input(inputs: Any) -> torch.Tensor:
    """Return residual-stream input to the decoder block (pre input-norm source).

    vLLM decoder layers are called as ``forward(positions, hidden_states, residual)``
    with a fused add+norm, so the value actually normalized by the input layernorm is
    ``hidden_states + residual`` (residual is ``None`` only on the first block). HF-style
    layers are called as ``forward(hidden_states, ...)`` with the residual already folded
    in, so a single 2-D tensor is present. Selecting ``inputs[0]`` blindly would grab the
    1-D ``positions`` tensor under vLLM and corrupt the input-norm Jacobian.
    """
    if isinstance(inputs, (tuple, list)):
        mats = [x for x in inputs if isinstance(x, torch.Tensor) and x.dim() >= 2]
        if len(mats) >= 2:
            return mats[0] + mats[1]  # vLLM fused residual: hidden_states + residual
        if mats:
            return mats[0]
        if inputs:
            return cast(torch.Tensor, inputs[0])
    return cast(torch.Tensor, inputs)


def _forward_context_att_metadata() -> Any | None:
    """Return ``attn_metadata`` from the active forward context, or ``None`` if unset."""
    try:
        ctx = get_forward_context()
    except AssertionError:
        return None
    return getattr(ctx, "attn_metadata", None)


def _skip_forward_capture() -> bool:
    """Skip CUDA-graph capture passes only.

    Do not gate on ``attn_metadata`` being unset: hooks run inside ``model()`` but
    ``record_batch_meta`` already tolerates missing metadata. Treating missing
    metadata as "skip capture" leaves ``cap.live`` empty on short prompts while
    prefill still completes (misleading ``max_num_batched_tokens`` errors).
    """
    return bool(
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    )


def _attn_core_module(attn: torch.nn.Module) -> torch.nn.Module | None:
    inner = getattr(attn, "attn", None)
    return inner if isinstance(inner, torch.nn.Module) else None


def infer_qkv_rotation_probe(
    *,
    q_pre: torch.Tensor | None,
    k_pre: torch.Tensor | None,
    v_pre: torch.Tensor | None,
    q_post: torch.Tensor | None,
    k_post: torch.Tensor | None,
    v_post: torch.Tensor | None,
    rel_tol: float = 5e-3,
) -> dict[str, Any]:
    """Infer which streams were rotated between projection output and attn-core input."""

    def _flatten_rows(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None or t.dim() == 0:
            return None
        return t.detach().reshape(-1, t.shape[-1]).float()

    def _rotated(pre: torch.Tensor | None, post: torch.Tensor | None) -> bool | None:
        pre_f = _flatten_rows(pre)
        post_f = _flatten_rows(post)
        if pre_f is None or post_f is None:
            return None
        n = min(pre_f.size(0), post_f.size(0))
        # Position 0 is often identity even when RoPE is enabled.
        if n <= 1:
            return None
        pre_f = pre_f[:n][1:]
        post_f = post_f[:n][1:]
        denom = pre_f.norm().clamp(min=1e-8)
        rel = (post_f - pre_f).norm() / denom
        return bool(rel > rel_tol)

    rotate_q = _rotated(q_pre, q_post)
    rotate_k = _rotated(k_pre, k_post)
    rotate_v = _rotated(v_pre, v_post)
    return {
        "has_rope": any(x is True for x in (rotate_q, rotate_k, rotate_v)),
        "rotate_q": rotate_q,
        "rotate_k": rotate_k,
        "rotate_v": rotate_v,
        "probe_rel_tol": rel_tol,
    }


def register_runtime_rope_probe_hooks(
    model: torch.nn.Module,
    *,
    meta: dict[str, Any],
    enabled: Callable[[], bool],
    rel_tol: float = 5e-3,
    require_attn_metadata: bool = False,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install one-shot runtime probe hooks to infer Q/K/V rotation status.

    Compares pre-projection outputs (qkv/q/k/v proj hooks) with post-attn-core inputs
    from the same forward pass and stores result under ``meta['rope_runtime_probe']``.
    """
    last_attn = _attention_on_block(_last_transformer_block(model))
    attn = cast(Any, last_attn)
    state: dict[str, Any] = {
        "done": False,
        "Q_pre": None,
        "K_pre": None,
        "V_pre": None,
        "Q_post": None,
        "K_post": None,
        "V_post": None,
    }

    def inactive() -> bool:
        return not enabled() or _skip_forward_capture()

    def maybe_finalize() -> None:
        if state["done"]:
            return
        out = infer_qkv_rotation_probe(
            q_pre=state["Q_pre"],
            k_pre=state["K_pre"],
            v_pre=state["V_pre"],
            q_post=state["Q_post"],
            k_post=state["K_post"],
            v_post=state["V_post"],
            rel_tol=rel_tol,
        )
        if all(out.get(k) is None for k in ("rotate_q", "rotate_k", "rotate_v")):
            return
        meta["rope_runtime_probe"] = out
        state["done"] = True

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    if (core := _attn_core_module(last_attn)) is not None:

        def post_hook(_m, inputs: Any, _output: Any) -> None:
            if state["done"] or inactive() or not inputs or len(inputs) < 3:
                return
            if require_attn_metadata and _forward_context_att_metadata() is None:
                return
            # Capture inputs to last attention module (after RoPe)
            state["Q_post"] = cast(torch.Tensor, inputs[0]).detach().clone()
            state["K_post"] = cast(torch.Tensor, inputs[1]).detach().clone()
            state["V_post"] = cast(torch.Tensor, inputs[2]).detach().clone()
            maybe_finalize()

        hooks.append(core.register_forward_hook(post_hook))

    # Fused QKV projection (qkv_proj) hook.
    if hasattr(last_attn, "qkv_proj"):
        q_size, kv_size = int(attn.q_size), int(attn.kv_size)

        def pre_fused_hook(_m, _inputs, output: Any) -> None:
            if state["done"] or inactive():
                return
            qkv = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(qkv):
                return
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            # Hooks capture outputs of QKV projection (before RoPe)
            state["Q_pre"] = q.detach().clone()
            state["K_pre"] = k.detach().clone()
            state["V_pre"] = v.detach().clone()
            maybe_finalize()

        # Register single QKV hook for fused projection (qkv_proj)
        hooks.append(attn.qkv_proj.register_forward_hook(pre_fused_hook))
        return hooks

    # Separate Q/K/V projection hooks.
    if all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
        tmp: dict[str, torch.Tensor] = {}

        def pre_hook(name: str):
            def _hook(_m, _inputs, output: Any) -> None:
                if state["done"] or inactive():
                    return
                t = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(t):
                    return
                tmp[name] = t.detach().clone()
                if len(tmp) == 3:
                    state["Q_pre"] = tmp["Q"]
                    state["K_pre"] = tmp["K"]
                    state["V_pre"] = tmp["V"]
                    maybe_finalize()

            return _hook

        # Register separate hooks for each projection module (q_proj, k_proj, v_proj)
        hooks.append(attn.q_proj.register_forward_hook(pre_hook("Q")))
        hooks.append(attn.k_proj.register_forward_hook(pre_hook("K")))
        hooks.append(attn.v_proj.register_forward_hook(pre_hook("V")))
    return hooks


def record_batch_meta(model_runner: Any, meta: dict[str, Any]) -> None:
    """Save flat ``input_ids`` and ``query_start_loc`` for per-request slicing.

    Populated after each capture forward so ``_slice_capture`` can find the right
    rows when multiple requests share one batch. Falls back to runner buffers if needed.
    """
    metadata = _forward_context_att_metadata()
    if metadata is None:
        return

    qsl, _ = get_query_metadata(metadata)
    if qsl is not None:
        n = int(qsl[-1].item())
        try:
            ctx = get_forward_context()
            n = int(getattr(ctx, "num_tokens", 0) or 0) or n
        except AssertionError:
            pass
        if n > 0:
            meta.clear()
            meta.update(
                {
                    "query_start_loc": qsl.detach().clone(),
                    "input_ids": model_runner.input_ids.gpu[:n].detach().clone(),
                }
            )
        return

    batch = getattr(model_runner, "input_batch", None)
    if batch is None or batch.num_reqs == 0:
        return
    if buf := getattr(model_runner, "query_start_loc", None):
        if not hasattr(buf, "gpu"):
            return
        qsl = buf.gpu[: batch.num_reqs + 1].detach().clone()
    elif hasattr(batch, "query_start_loc_np"):
        qsl = torch.tensor(
            batch.query_start_loc_np[: batch.num_reqs + 1],
            dtype=torch.int32,
            device=model_runner.input_ids.gpu.device,
        )
    else:
        return
    n = int(qsl[-1].item())
    if n <= 0:
        return
    meta.clear()
    meta.update(
        {
            "query_start_loc": qsl,
            "input_ids": model_runner.input_ids.gpu[:n].detach().clone(),
        }
    )


def _register_qkv_hooks(
    last_attn: torch.nn.Module,
    dest: dict[str, torch.Tensor],
    *,
    enabled: Callable[[], bool],
    on_qkv: Callable[[], None] | None = None,
    require_post_rope: bool = True,
    require_attn_metadata: bool = False,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture last-layer Q/K/V, preferring post-RoPE tensors from ``.attn`` inputs.

    Default behavior enforces post-RoPE capture (`require_post_rope=True`), which
    matches the online worker path and avoids analyzer mismatch from pre-RoPE Q/K.
    Set ``require_post_rope=False`` only as a compatibility fallback for models
    without a hookable inner attention core.
    """

    def store_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        dest["Q"], dest["K"], dest["V"] = q.detach().clone(), k.detach().clone(), v.detach().clone()
        if on_qkv:
            on_qkv()

    def inactive() -> bool:
        return not enabled() or _skip_forward_capture()

    if (core := _attn_core_module(last_attn)) is not None:

        def attn_input_hook(_m, inputs: Any, _output: Any) -> None:
            if inactive() or not inputs or len(inputs) < 3:
                return
            if require_attn_metadata and _forward_context_att_metadata() is None:
                return
            store_qkv(
                cast(torch.Tensor, inputs[0]),
                cast(torch.Tensor, inputs[1]),
                cast(torch.Tensor, inputs[2]),
            )

        return [core.register_forward_hook(attn_input_hook)]

    if require_post_rope:
        raise RuntimeError(
            "forward_attr: could not locate inner attention core (.attn) for post-RoPE "
            "Q/K/V capture. Set highlighter.allow_prerope_fallback=true in config to allow "
            "legacy pre-RoPE projection hooks (approximation may degrade)."
        )

    attn = cast(Any, last_attn)
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    if hasattr(last_attn, "qkv_proj"):
        q_size, kv_size = int(attn.q_size), int(attn.kv_size)

        def fused_hook(_m, _i, output: Any) -> None:
            if inactive():
                return
            qkv = cast(torch.Tensor, output[0] if isinstance(output, tuple) else output)
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            store_qkv(q, k, v)

        hooks.append(attn.qkv_proj.register_forward_hook(fused_hook))
        return hooks

    captured: dict[str, torch.Tensor] = {}

    def proj_hook(key: str):
        def hook(_m, _i, output: Any) -> None:
            if inactive():
                return
            captured[key] = cast(torch.Tensor, output).detach().clone()
            if key == "V" and len(captured) == 3:
                store_qkv(captured["Q"], captured["K"], captured["V"])

        return hook

    for key, attr in (("Q", "q_proj"), ("K", "k_proj"), ("V", "v_proj")):
        hooks.append(getattr(attn, attr).register_forward_hook(proj_hook(key)))
    return hooks


def register_forward_attr_hooks(
    model: torch.nn.Module,
    dest: dict[str, torch.Tensor],
    *,
    enabled: Callable[[], bool],
    model_runner: Any | None = None,
    meta: dict[str, Any] | None = None,
    require_attn_metadata: bool = False,
    allow_prerope_fallback: bool = False,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install last-layer Q/K/V + ``h_L`` hooks for forward attribution.

    Hooks are gated by ``enabled()`` (worker sets ``_cap.active`` during prefill).
    ``on_qkv`` records batch layout for multi-request slicing after the forward.

    The ``require_attn_metadata`` flag controls whether hooks are installed only
    during a scheduler-driven attention forward with populated context metadata (True),
    or whether they run as long as enabled() is set and CUDA-graph capture is not active (False).
    """
    # Locate last transformer block and last attention module
    last_layer = _last_transformer_block(model)
    last_attn = _attention_on_block(last_layer)
    ffn_input_norm = locate_ffn_input_norm(model)

    # Record batch metadata for multi-request slicing
    def on_qkv() -> None:
        if model_runner is not None and meta is not None and enabled():
            record_batch_meta(model_runner, meta)

    # Register Q/K/V hooks for last-layer attention module in vLLM and HF-format serving
    # models, and collect QKV activations immediately after projection
    # and immediately before the attention module to determine whether RoPE was applied.
    hooks = _register_qkv_hooks(
        last_attn,
        dest,
        enabled=enabled,
        on_qkv=on_qkv,
        require_post_rope=not allow_prerope_fallback,
        require_attn_metadata=require_attn_metadata,
    )

    def layer_input_prehook(_m, _i) -> None:
        if not enabled() or _skip_forward_capture():
            return
        if require_attn_metadata and _forward_context_att_metadata() is None:
            return
        # Capture the block's residual-stream input BEFORE the layer runs. vLLM's fused
        # add+norm rewrites the ``residual`` tensor in-place during forward, so reading the
        # layer inputs in a post-forward hook observes a mutated (wrong) residual stream and
        # corrupts the Norm1 base used by the attention VJP. A pre-hook clones the true input.
        dest["h_input"] = _decoder_layer_input(_i).detach().clone()

    def layer_hook(_m, _i, output: Any) -> None:
        if not enabled() or _skip_forward_capture():
            return
        if require_attn_metadata and _forward_context_att_metadata() is None:
            return
        dest["h_L"] = _decoder_layer_hidden(output).detach().clone()
        on_qkv()

    hooks.append(last_layer.register_forward_pre_hook(layer_input_prehook))  # block input
    hooks.append(last_layer.register_forward_hook(layer_hook))  # residual stream at layer L
    if ffn_input_norm is not None:
        def ffn_norm_prehook(_m, inputs: Any) -> None:
            if not enabled() or _skip_forward_capture():
                return
            if require_attn_metadata and _forward_context_att_metadata() is None:
                return
            if not inputs:
                return
            # vLLM's fused add+norm passes (attn_out, residual); the true MLP-norm input is
            # their sum. _decoder_layer_input sums the >=2-D tensors (and is a no-op for the
            # single-tensor HF call), matching the residual stream h_mid we backprop through.
            dest["h_mid"] = _decoder_layer_input(inputs).detach().clone()
            on_qkv()

        hooks.append(ffn_input_norm.register_forward_pre_hook(ffn_norm_prehook))
    return hooks


# ---------------------------------------------------------------------------
# User-facing generation wrappers
# ---------------------------------------------------------------------------


def load_highlighter_config(config_file: str) -> dict:
    """Load the ``highlighter`` section from a model config JSON file."""
    import json

    with open(config_file, "r", encoding="utf-8") as f:
        return dict(json.load(f).get("highlighter", {}))


def generate_with_highlighter(
    llm,
    prompts,
    mode: str = "capture",
    highlighter_config: dict | None = None,
    sampling_params=None,
    run_id: str | None = None,
    scores_run_id: str | None = None,
    save_to_disk: bool | None = None,
    **kwargs,
):
    """Generate with Token Highlighter capture or mitigation.

    Works with ``HookLLM`` configured with ``worker_name="token_highlighter"``.
    Installs worker hooks, sets per-request ``extra_args``, runs ``HookLLM.generate``,
    and flushes disk artifacts when ``save_to_disk`` is enabled (capture mode).
    """
    import copy
    import os
    import uuid

    from vllm import SamplingParams

    if isinstance(prompts, str):
        prompts = [prompts]

    mode = str(mode).strip().lower()
    if save_to_disk is None:
        save_to_disk = mode == "capture"

    if run_id is None:
        if mode == "mitigate" and getattr(llm, "_last_run_id", None):
            run_id = llm._last_run_id
        else:
            run_id = str(uuid.uuid4())

    hook_dir = getattr(llm, "_hook_dir", None)
    if not hook_dir:
        raise ValueError("llm._hook_dir missing: pass a HookLLM instance")

    run_id_file = os.path.join(hook_dir, "RUN_ID.txt")
    base_params = sampling_params or SamplingParams(**kwargs)
    if isinstance(base_params, list):
        if len(base_params) != len(prompts):
            raise ValueError(
                f"sampling_params list length ({len(base_params)}) "
                f"must match prompts length ({len(prompts)})"
            )
        sp_list = list(base_params)
    else:
        sp_list = [base_params] * len(prompts)

    new_sp_list = []
    for sp in sp_list:
        sp = copy.copy(sp)
        extra = dict(sp.extra_args or {})
        if highlighter_config:
            extra["highlighter"] = dict(highlighter_config)
        extra["highlighter_mode"] = mode
        extra["hook_dir"] = hook_dir
        extra["run_id"] = run_id
        extra["run_id_file"] = run_id_file
        if save_to_disk:
            extra["save_to_disk"] = True
        if mode == "mitigate":
            extra.setdefault("scores_run_id", scores_run_id or run_id)
        sp.extra_args = extra
        new_sp_list.append(sp)

    sp_arg = new_sp_list[0] if len(new_sp_list) == 1 else new_sp_list

    engine = getattr(llm, "llm", llm)
    if not getattr(engine, "_vllm_hook_installed", False):
        install_hooks = getattr(engine, "collective_rpc", None)
        if callable(install_hooks):
            print("Installing hooks via collective_rpc")
            install_hooks("install_hooks")
            setattr(engine, "_vllm_hook_installed", True)

    outputs = llm.generate(
        prompts,
        sampling_params=sp_arg,
        use_hook=True,
        save_to_disk=save_to_disk,
        run_id=run_id,
    )
    if not isinstance(outputs, list):
        outputs = list(outputs)

    if save_to_disk:
        req_ids = [output.request_id for output in outputs]
        if req_ids:
            flush_disk = getattr(engine, "collective_rpc", None)
            if callable(flush_disk):
                print("Flushing disk via collective_rpc")
                flush_disk("flush_disk", args=(req_ids, run_id, hook_dir))

    return outputs


def analyze_with_highlighter(
    llm,
    analyzer_spec: dict | None = None,
    *,
    highlighter_config: dict | None = None,
    run_id: str | None = None,
    run_ids: list[str] | None = None,
    probes: dict | None = None,
):
    """Run the Token Highlighter analyzer with optional config defaults."""
    spec = dict(analyzer_spec or {})
    if highlighter_config:
        spec.setdefault("mode", highlighter_config.get("mode"))
        spec.setdefault("alpha", highlighter_config.get("alpha"))
        spec.setdefault("threshold_k", highlighter_config.get("threshold_k"))
        spec.setdefault("soft_beta", highlighter_config.get("beta"))
    return llm.analyze(
        analyzer_spec=spec,
        probes=probes,
        run_id=run_id,
        run_ids=run_ids,
    )
