"""Forward-attribution influence scores from last-layer Q/K/V, h_L and weight captures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import logging

import torch

from mia.utils.TokenHighlighter.utils import (
    _last_transformer_block,
    export_final_norm_spec,
    export_input_norm_spec,
    get_attn_query_weight,
    get_attn_key_value_weights,
    linear_weight,
    locate_ffn_input_norm,
    locate_final_norm,
    locate_input_norm,
    lm_head_weight,
)


LossGradFn = Callable[[torch.Tensor, int, torch.Tensor, torch.device], torch.Tensor]


def _silu_prime(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def _gelu_prime(x: torch.Tensor) -> torch.Tensor:
    c = 0.044715
    k = torch.sqrt(2/torch.pi)
    x3 = x * x * x
    u = k * (x + c * x3)
    t = torch.tanh(u)
    sech2 = 1.0 - t * t
    du = k * (1.0 + 3.0 * c * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * du


def _activation_prime(ffn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    act = getattr(ffn, "act_fn", None)
    name = type(act).__name__.lower() if act is not None else ""
    if "silu" in name or "swish" in name:
        return _silu_prime(x)
    if "relu" in name:
        return (x > 0).to(dtype=x.dtype)
    return _gelu_prime(x)


def _activation_value(ffn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    act = getattr(ffn, "act_fn", None)
    name = type(act).__name__.lower() if act is not None else ""
    if "silu" in name or "swish" in name:
        return torch.nn.functional.silu(x)
    if "relu" in name:
        return torch.nn.functional.relu(x)
    logging.warning(f"Assuming GELU activation in final FFN since {name} is not supported")
    return torch.nn.functional.gelu(x)


def _linear_forward(module: torch.nn.Module, x: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    w = linear_weight(module, out_dtype=x.dtype).to(device=device)
    y = x @ w.T
    b = getattr(module, "bias", None)
    if b is not None:
        y = y + b.to(device=device, dtype=x.dtype)
    return y


def _resolve_final_ffn(module: torch.nn.Module | None) -> torch.nn.Module | None:
    if module is None:
        return None
    try:
        block = _last_transformer_block(module)
    except RuntimeError:
        return None
    for name in ("mlp", "feed_forward", "ffn"):
        ffn = getattr(block, name, None)
        if isinstance(ffn, torch.nn.Module):
            return ffn
    return None


def _ffn_jacobian_vjp(
    ffn: torch.nn.Module,
    x: torch.Tensor,
    g_out: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    def w(mod: torch.nn.Module) -> torch.Tensor:
        return linear_weight(mod, out_dtype=x.dtype).to(device=device)

    down = next((getattr(ffn, n) for n in ("down_proj", "c_proj", "fc2", "w2") if hasattr(ffn, n)), None)
    if down is None:
        return None
    d_h = g_out @ w(down)

    if hasattr(ffn, "gate_proj") and hasattr(ffn, "up_proj"):
        gate_pre = _linear_forward(ffn.gate_proj, x, device=device)
        up_val = _linear_forward(ffn.up_proj, x, device=device)
        d_gate = d_h * up_val * _activation_prime(ffn, gate_pre)
        d_up = d_h * _activation_value(ffn, gate_pre)
        return d_gate @ w(ffn.gate_proj) + d_up @ w(ffn.up_proj)

    if hasattr(ffn, "gate_up_proj"):
        pre = _linear_forward(ffn.gate_up_proj, x, device=device)
        w_gu = w(ffn.gate_up_proj)
        half = w_gu.size(0) // 2
        gate_pre, up_val = pre[..., :half], pre[..., half:]
        d_gate = d_h * up_val * _activation_prime(ffn, gate_pre)
        d_up = d_h * _activation_value(ffn, gate_pre)
        return d_gate @ w_gu[:half] + d_up @ w_gu[half:]

    in_proj = next((getattr(ffn, n) for n in ("c_fc", "fc1", "up_proj", "w1") if hasattr(ffn, n)), None)
    if in_proj is None:
        return None
    in_pre = _linear_forward(in_proj, x, device=device)
    return (d_h * _activation_prime(ffn, in_pre)) @ w(in_proj)


def compute_mid_boundary_gradient(
    captured: dict[str, torch.Tensor],
    prompt_len: int,
    g_out: torch.Tensor,
    *,
    model: torch.nn.Module | None,
    device: torch.device,
) -> torch.Tensor:
    """Compute g_mid = dL/dh_mid = g_out + J_norm2^T J_ffn^T g_out at generation positions."""
    if model is None:
        return g_out
    h_mid_full = captured.get("h_mid")
    if h_mid_full is None:
        return g_out

    n_gen_toks = g_out.size(0)
    start = prompt_len - 1
    h_mid = h_mid_full.squeeze(0).to(device=device)[start : start + n_gen_toks]
    if h_mid.size(0) != n_gen_toks:
        return g_out

    ffn = _resolve_final_ffn(model)
    norm2_spec = _materialize_norm_spec(
        export_input_norm_spec(locate_ffn_input_norm(model)), device, g_out.dtype
    )
    if ffn is None or norm2_spec is None:
        return g_out

    x = _apply_final_norm_hidden(h_mid, norm2_spec)
    j_ffn_t_g = _ffn_jacobian_vjp(ffn, x, g_out, device=device)
    if j_ffn_t_g is None:
        return g_out
    g_ffn_input = _project_g_through_norm(j_ffn_t_g, h_mid, norm2_spec, device, x.dtype)
    return g_out + g_ffn_input


def _has_rope(rope_spec: dict[str, Any]) -> bool:
    if "has_rope" in rope_spec:
        return bool(rope_spec["has_rope"])
    return (
        rope_spec.get("inv_freq") is not None
        or rope_spec.get("rope_scaling") is not None
        or rope_spec.get("rope_theta") is not None
    )


def _stream_rotated(rope_spec: dict[str, Any], key: str) -> bool:
    flag = rope_spec.get(key)
    if isinstance(flag, bool):
        return flag
    return False if key == "rotate_v" else _has_rope(rope_spec)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(
    rope_spec: dict[str, Any],
    seq_len: int,
    d_head: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    partial = float(rope_spec.get("partial_rotary_factor", 1.0))
    rotary_dim = min(int(d_head * partial), d_head)
    if rotary_dim % 2:
        rotary_dim -= 1

    inv_freq_cpu = rope_spec.get("inv_freq")
    if inv_freq_cpu is not None:
        inv_freq = inv_freq_cpu.to(device=device, dtype=dtype)
    else:
        theta = rope_spec.get("rope_theta")
        scaling = rope_spec.get("rope_scaling")
        if theta is None and isinstance(scaling, dict):
            theta = scaling.get("rope_theta")
        theta = float(theta) if theta is not None else 10000.0
        half = rotary_dim // 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, half, device=device, dtype=dtype) / float(rotary_dim))
        )

    positions = torch.arange(seq_len, device=device, dtype=dtype)
    scaling = rope_spec.get("rope_scaling")
    if isinstance(scaling, dict) and str(scaling.get("type", "")).lower() == "linear":
        factor = float(scaling.get("factor", 1.0))
        if factor != 1.0:
            positions = positions / factor

    emb = torch.cat((torch.outer(positions, inv_freq), torch.outer(positions, inv_freq)), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if rotary_dim < d_head:
        pad = d_head - rotary_dim
        cos = torch.cat([cos, torch.ones(seq_len, pad, device=device, dtype=dtype)], dim=-1)
        sin = torch.cat([sin, torch.zeros(seq_len, pad, device=device, dtype=dtype)], dim=-1)
    return cos, sin


def _rope_inverse_on_prompt_heads(
    grad: torch.Tensor,
    rope_spec: dict[str, Any],
    prompt_len: int,
    *,
    enabled: bool = True,
) -> torch.Tensor:
    if prompt_len <= 0 or not enabled or not _has_rope(rope_spec):
        return grad
    device, dtype = grad.device, grad.dtype
    cos, sin = _rope_cos_sin(rope_spec, prompt_len, grad.size(-1), device, dtype)
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return grad * cos + _rotate_half(grad) * (-sin)


def _rope_spec_from_bundle(
    weights: dict[str, Any] | None,
    model: torch.nn.Module | None,
) -> dict[str, Any]:
    if weights is not None and "rope_spec" in weights:
        return dict(weights["rope_spec"])
    cfg = getattr(model, "config", None) if model is not None else None
    if cfg is not None:
        rope_theta = getattr(cfg, "rope_theta", None)
        rope_scaling = getattr(cfg, "rope_scaling", None)
        partial_rotary = getattr(cfg, "partial_rotary_factor", None)
        return {
            "has_rope": bool(rope_theta is not None or rope_scaling is not None),
            "rope_theta": float(rope_theta) if rope_theta is not None else 10000.0,
            "partial_rotary_factor": float(partial_rotary) if partial_rotary is not None else 1.0,
            "rotate_q": bool(rope_theta is not None or rope_scaling is not None),
            "rotate_k": bool(rope_theta is not None or rope_scaling is not None),
            "rotate_v": False,
        }
    return {
        "has_rope": False,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
        "rotate_q": False,
        "rotate_k": False,
        "rotate_v": False,
    }

def _effective_norm_kind(norm_spec: dict[str, Any]) -> str:
    kind = norm_spec.get("norm_kind")
    if kind:
        return str(kind)
    return "layernorm" if norm_spec.get("has_bias") else "rmsnorm"


def _materialize_norm_spec(
    norm_spec: dict[str, Any] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    if norm_spec is None or norm_spec.get("weight") is None:
        return None
    out = dict(norm_spec)
    out["weight"] = norm_spec["weight"].to(device=device, dtype=dtype)
    bias = norm_spec.get("bias")
    out["bias"] = bias.to(device=device, dtype=dtype) if bias is not None else None
    return out


def _apply_final_norm_hidden(
    h: torch.Tensor,
    norm_spec: dict[str, Any] | None,
) -> torch.Tensor:
    if norm_spec is None:
        return h
    weight = norm_spec.get("weight")
    if weight is None:
        return h
    gamma = weight.to(device=h.device, dtype=h.dtype)
    bias = norm_spec.get("bias")
    eps = float(norm_spec.get("eps", 1e-6))
    kind = _effective_norm_kind(norm_spec)
    if kind == "layernorm":
        return torch.nn.functional.layer_norm(
            h,
            (h.size(-1),),
            gamma,
            bias.to(device=h.device, dtype=h.dtype) if bias is not None else None,
            eps,
        )
    inv_std = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h_hat = h * inv_std
    if kind == "gemma_rms":
        return h_hat * (1.0 + gamma)
    return h_hat * gamma


def _project_g_through_norm(
    g: torch.Tensor,
    h_in: torch.Tensor,
    norm_spec: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = norm_spec.get("weight")
    gamma = (
        weight.to(device=device, dtype=dtype)
        if weight is not None
        else torch.ones(h_in.size(-1), device=device, dtype=dtype)
    )
    eps = float(norm_spec.get("eps", 1e-6))
    kind = _effective_norm_kind(norm_spec)
    has_bias = kind == "layernorm"

    mean = h_in.mean(-1, keepdim=True) if has_bias else torch.zeros(1, device=device, dtype=dtype)
    variance = (h_in - mean).pow(2).mean(-1, keepdim=True)
    inv_std = torch.rsqrt(variance + eps)
    h_hat = (h_in - mean) * inv_std

    scale = (1.0 + gamma) if kind == "gemma_rms" else gamma
    g_gamma = g * scale

    mu_grad = g_gamma.mean(-1, keepdim=True) if has_bias else torch.zeros(1, device=device, dtype=dtype)
    mu_stat = (g_gamma * h_hat).mean(-1, keepdim=True)

    return inv_std * (g_gamma - mu_grad - h_hat * mu_stat)


def project_g_through_norm(
    g: torch.Tensor,
    h_in: torch.Tensor,
    norm_spec: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Public wrapper for analytic norm VJP."""
    return _project_g_through_norm(g, h_in, norm_spec, device, dtype)


def _lm_head_input_slice(
    h_L: torch.Tensor,
    prompt_len: int,
    n_gen_toks: int,
    model: torch.nn.Module | None,
    *,
    final_norm: dict[str, Any] | None = None,
) -> torch.Tensor:
    start = prompt_len - 1
    end = start + n_gen_toks
    h = h_L[start:end]
    if final_norm is not None:
        return _apply_final_norm_hidden(h, final_norm)
    if model is None:
        return h
    norm = locate_final_norm(model)
    return norm(h) if norm is not None else h


def affirmation_loss_grad(
    h_L: torch.Tensor,
    prompt_len: int,
    target_ids: list[int],
    W_U: torch.Tensor,
    device: torch.device,
    *,
    model: torch.nn.Module | None = None,
    final_norm: dict[str, Any] | None = None,
) -> torch.Tensor:
    """g_j = dL/dh_j for teacher-forced affirmation CE (L = -log P(y|x))."""
    n_gen_toks = len(target_ids)
    h = _lm_head_input_slice(
        h_L, prompt_len, n_gen_toks, model, final_norm=final_norm
    )
    logits = h @ W_U.T
    p = torch.softmax(logits, dim=-1)
    target = torch.tensor(target_ids, device=device, dtype=torch.long)
    p[torch.arange(n_gen_toks, device=device), target] -= 1.0
    return p @ W_U


def _resolve_forward_attr_tensors(
    model: torch.nn.Module | None,
    last_attn: torch.nn.Module | None,
    weights: dict[str, Any] | None,
    device: torch.device,
) -> tuple[
    int,
    int,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    if weights is not None:
        n_heads = int(weights["num_attention_heads"])
        n_kv_heads = int(weights["num_key_value_heads"])
        dtype = weights["W_O"].dtype
        w_u_cpu = weights.get("W_U")
        w_u = w_u_cpu.to(device=device, dtype=dtype) if w_u_cpu is not None else None
        w_o = weights["W_O"].to(device=device, dtype=dtype)
        w_v = weights["W_V"].to(device=device, dtype=dtype)
        w_q_cpu = weights.get("W_Q")
        w_q = w_q_cpu.to(device=device, dtype=dtype) if w_q_cpu is not None else None
        w_k = weights["W_K"].to(device=device, dtype=dtype)
        input_norm_spec = _materialize_norm_spec(weights.get("input_norm"), device, dtype)
        final_norm_spec = _materialize_norm_spec(weights.get("final_norm"), device, dtype)
        return n_heads, n_kv_heads, w_u, w_o, w_v, w_q, w_k, input_norm_spec, final_norm_spec

    if model is None or last_attn is None:
        raise ValueError("compute_grad_influence_vectors requires weights= or (model, last_attn)")

    cfg: Any = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "num_attention_heads"):
        raise ValueError("model.config.num_attention_heads is required for grad influence")
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    w_u = lm_head_weight(model)
    w_q = get_attn_query_weight(last_attn)
    w_o, w_v, w_k = get_attn_key_value_weights(last_attn)

    raw_final_norm_spec = export_final_norm_spec(locate_final_norm(model))
    final_norm_spec = _materialize_norm_spec(raw_final_norm_spec, device, w_u.dtype)

    raw_input_norm_spec = export_input_norm_spec(locate_input_norm(model))
    input_norm_spec = _materialize_norm_spec(raw_input_norm_spec, device, w_u.dtype)
    return n_heads, n_kv_heads, w_u, w_o, w_v, w_q, w_k, input_norm_spec, final_norm_spec


def compute_grad_influence_vectors(
    captured: dict[str, torch.Tensor],
    prompt_len: int,
    last_attn: torch.nn.Module | None = None,
    model: torch.nn.Module | None = None,
    *,
    weights: dict[str, Any] | None = None,
    loss_grad_fn: LossGradFn | None = None,
    target_ids: list[int] | None = None,
    apply_final_norm_to_g: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Gradient of the loss w.r.t. pre-last-layer hidden states for each input token."""
    if device is None:
        if weights is not None:
            device = torch.device("cpu")
        elif model is not None:
            device = next(model.parameters()).device
        else:
            raise ValueError("device required when neither model nor weights set")

    n_heads, n_kv_heads, w_u, w_o, w_v, w_q, w_k, input_norm_spec, final_norm_spec = _resolve_forward_attr_tensors(
        model, last_attn, weights, device
    )

    rope_spec = _rope_spec_from_bundle(weights, model)

    h_L = captured["h_L"].squeeze(0).to(device=device)
    Q = captured["Q"].squeeze(0).to(device=device)
    K = captured["K"].squeeze(0).to(device=device)
    V = captured["V"].squeeze(0).to(device=device)
    S_seq, d_model = h_L.shape
    d_head = Q.size(-1) // n_heads
    sqrt_d = d_head**0.5

    W_U = w_u
    W_O, W_V, W_K = w_o, w_v, w_k

    if loss_grad_fn is None:
        if not target_ids:
            raise ValueError("target_ids required for default affirmation_loss_grad")
        if W_U is None:
            raise ValueError(
                "W_U (unembedding) is required to recompute the loss gradient when no "
                "loss_grad_fn / precomputed g is supplied. Provide loss_grad_fn or a "
                "bundle/sequence with g_loss, or export with include_unembedding=True."
            )
        g = affirmation_loss_grad(
            h_L,
            prompt_len,
            target_ids,
            W_U,
            device,
            model=model if final_norm_spec is None and model is not None else None,
            final_norm=final_norm_spec,
        )
    else:
        g = loss_grad_fn(h_L, prompt_len, W_U, device)

    n_gen_toks = g.size(0)
    start = prompt_len - 1

    if apply_final_norm_to_g and final_norm_spec is not None:
        h_gen = h_L[start : start + n_gen_toks]
        g = _project_g_through_norm(g, h_gen, final_norm_spec, device, h_L.dtype)

    g_mid_cap = captured.get("g_mid")
    if g_mid_cap is not None:
        g_attn = g_mid_cap.to(device=device, dtype=g.dtype)
    elif model is not None:
        logging.info("Computing g_mid gradient through final FFN")
        g_attn = compute_mid_boundary_gradient(
            captured, prompt_len, g, model=model, device=device
        )
    else:
        logging.info("No gradient through final FFN found and model unavailable: using g directly, which is an approximation.")
        g_attn = g
    if g_attn.size(0) != n_gen_toks:
        logging.error(f"g_mid gradient has {g_attn.size(0)} tokens but expected {n_gen_toks} tokens")
        g_attn = g

    Q = Q.view(S_seq, n_heads, d_head).transpose(0, 1).contiguous()
    K = K.view(S_seq, n_kv_heads, d_head).transpose(0, 1).contiguous()
    V = V.view(S_seq, n_kv_heads, d_head).transpose(0, 1).contiguous()
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        K = K.repeat_interleave(rep, dim=0)
        V = V.repeat_interleave(rep, dim=0)

    W_O = W_O.view(d_model, n_heads, d_head).transpose(0, 1).contiguous()
    W_Q = w_q.view(n_heads, d_head, d_model) if w_q is not None else None
    W_V = W_V.view(n_kv_heads, d_head, d_model)
    W_K = W_K.view(n_kv_heads, d_head, d_model)
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        W_V = W_V.repeat_interleave(rep, dim=0)
        W_K = W_K.repeat_interleave(rep, dim=0)

    Q_gen = Q[:, start : start + n_gen_toks, :]
    scores = (Q_gen @ K.transpose(-1, -2)) / sqrt_d
    gen_pos = start + torch.arange(n_gen_toks, device=device)
    all_pos = torch.arange(S_seq, device=device)
    causal = all_pos.unsqueeze(0) > gen_pos.unsqueeze(1)
    scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))
    A = torch.softmax(scores, dim=-1)
    A_in = A[:, :, :prompt_len]

    g_a = torch.einsum("md,hdj->hmj", g_attn, W_O)

    val_post = A_in.transpose(-1, -2) @ g_a
    val_pre = _rope_inverse_on_prompt_heads(
        val_post, rope_spec, prompt_len, enabled=_stream_rotated(rope_spec, "rotate_v")
    )
    value_heads = val_pre @ W_V

    context_heads = A @ V
    baseline = torch.sum(g_a * context_heads, dim=-1, keepdim=True)

    l_full_prompt = g_a @ V[:, :prompt_len, :].transpose(-1, -2)

    delta = A_in * (l_full_prompt - baseline)

    key_post = (delta.transpose(-1, -2) @ Q_gen) / sqrt_d
    key_pre = _rope_inverse_on_prompt_heads(
        key_post, rope_spec, prompt_len, enabled=_stream_rotated(rope_spec, "rotate_k")
    )
    key_heads = key_pre @ W_K

    total_grad_heads = value_heads + key_heads
    total_grad = total_grad_heads.sum(dim=0)

    if W_Q is not None and 0 <= start < prompt_len:
        delta0 = delta[:, 0, :]
        k_prompt = K[:, :prompt_len, :]
        q_post0 = torch.einsum("hi,hid->hd", delta0, k_prompt) / sqrt_d
        q_post_rows = torch.zeros(
            (n_heads, prompt_len, d_head), device=device, dtype=q_post0.dtype
        )
        q_post_rows[:, start, :] = q_post0
        q_pre_rows = _rope_inverse_on_prompt_heads(
            q_post_rows, rope_spec, prompt_len, enabled=_stream_rotated(rope_spec, "rotate_q")
        )
        q_pre0 = q_pre_rows[:, start, :]
        total_grad[start] = total_grad[start] + torch.einsum("hd,hde->e", q_pre0, W_Q)

    if input_norm_spec is not None:
        h_input = captured.get("h_input", None)
        if h_input is None:
            raise ValueError("h_input not found in captured activations -- required for computing gradient of last attention input norm.")
        h_prompt = h_input[:prompt_len]
        total_grad = _project_g_through_norm(total_grad, h_prompt, input_norm_spec, device, h_L.dtype)

    if 0 <= start < prompt_len:
        total_grad[start] = total_grad[start] + g_attn[0].to(
            device=device, dtype=total_grad.dtype
        )

    return total_grad


def compute_grad_influences(
    captured: dict[str, torch.Tensor],
    prompt_len: int,
    last_attn: torch.nn.Module | None = None,
    model: torch.nn.Module | None = None,
    *,
    weights: dict[str, Any] | None = None,
    loss_grad_fn: LossGradFn | None = None,
    target_ids: list[int] | None = None,
    apply_final_norm_to_g: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    """Per-prompt-token influence scores (L2 norm of value and key path gradients)."""
    vec = compute_grad_influence_vectors(
        captured,
        prompt_len,
        last_attn,
        model,
        weights=weights,
        loss_grad_fn=loss_grad_fn,
        target_ids=target_ids,
        apply_final_norm_to_g=apply_final_norm_to_g,
        device=device,
    )
    return vec.norm(p=2, dim=-1).detach().cpu().tolist()

