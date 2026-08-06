"""Adaptive Raven adapter: per-token exit inside ``iterate_forward`` only.

Model-specific details live here:
  - Raven ``core_block_forward`` / RoPE slicing
  - ``RowSliceCacheProxy`` (expand sliced writes; congruent fill from model config)
  - optional Huginn baseline exit criteria for A/B

``raven_modeling_minimal_llama.py`` is the original file from McLeish's retrofitted Llama repo.
The shared worker/analyzer only refer to ``[B,S]`` metrics and return ``AnalyzerDecision``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch.nn.attention.flex_attention import BlockMask

from .raven_baseline_exit import RavenBaselineConfig, RavenBaselineExit
from .raven_modeling_minimal_llama import RavenForCausalLM
from vllm_hook_plugins.protocols.exit_controller import ExitController
from vllm_hook_plugins.protocols.recurrent_config import RecurrentDepthConfig


def block_geometry_from_config(config: Any) -> tuple[int, int]:
    """``(recurrent_block_size, prelude_depth)`` from a Raven config."""
    r = int(getattr(config, "n_layers_in_recurrent_block", 4))
    p = int(getattr(config, "n_layers_in_prelude", 2))
    return r, p


@dataclass
class RavenAdapterConfig:
    """Raven-only runtime options (not part of the shared protocol)."""

    # Decode FLOP savings: slice inactive batch rows out of core_block_forward.
    # Prefill (S > 1) never slices — positions must stay mutually attendable.
    slice_decode: bool = True
    # None → use shared contraction analyzer; else Huginn baseline name.
    baseline_criterion: Optional[str] = None
    baseline_exit_threshold: Optional[float] = None
    # Passed through to stock HuginnDynamicCache (upstream default name).
    cache_lookup_strategy: str = "latest-m4"


def _slice_freqs(
    freqs_cis: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    rows: torch.Tensor,
):
    """Slice batch dim of RoPE tables when present; leave broadcastable tables alone."""
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
        if cos.dim() >= 1 and cos.shape[0] > 1:
            return cos[rows], sin[rows]
        return cos, sin
    if isinstance(freqs_cis, torch.Tensor) and freqs_cis.dim() >= 1 and freqs_cis.shape[0] > 1:
        return freqs_cis[rows]
    return freqs_cis


class RowSliceCacheProxy:
    """Expand sliced decode K/V writes into a full-batch ``HuginnDynamicCache``.

    Upstream Huginn stores ``cache[step][token_pos] → [B, H, D]``. A sliced
    forward would otherwise overwrite that entry with ``[B_act, H, D]``.

    Inactive rows are filled with the **latest congruent** KV using geometry
    from the **model config** (not original file's hardcoded ``% 4``):

        same physical core layer ⇔ ``step % R == step_idx % R``
        with ``R = n_layers_in_recurrent_block``, ``step >= n_layers_in_prelude``

    The proxy chooses what full-batch tensor to pass into ``inner.update`` based on
    the active rows and the latest matching/congruent KV.
    """

    def __init__(
        self,
        inner,
        *,
        recurrent_block_size: int = 4,
        prelude_depth: int = 2,
    ):
        self.inner = inner
        self.recurrent_block_size = int(recurrent_block_size)
        self.prelude_depth = int(prelude_depth)
        self._rows: Optional[torch.Tensor] = None
        self._B: Optional[int] = None

    def set_write_rows(self, rows: Optional[torch.Tensor], B: int):
        self._rows = rows
        self._B = B

    def clear_write_rows(self):
        self._rows = None
        self._B = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _lookup_latest_kv(self, step_idx: int, token_pos: int):
        """Return ``([B,H,D], [B,H,D])`` from the latest congruent prior write, or None."""
        key_cache = self.inner.key_cache
        value_cache = self.inner.value_cache
        r, p = self.recurrent_block_size, self.prelude_depth
        if step_idx >= p:
            valid = [s for s in range(step_idx) if token_pos in key_cache.get(s, ())]
            congruent = [s for s in valid if s >= p and s % r == step_idx % r]
            if congruent:
                max_step = max(congruent)
            elif valid:
                max_step = max(valid)
            else:
                return None
        else:
            if token_pos in key_cache.get(step_idx, ()):
                max_step = step_idx
            elif token_pos in key_cache.get(0, ()):
                max_step = 0
            else:
                return None
        return key_cache[max_step][token_pos], value_cache[max_step][token_pos]

    def update(self, key_states, value_states, step_idx_tensor, lookup_strategy=None):
        rows, B = self._rows, self._B
        if rows is None or B is None or rows.numel() == B:
            if lookup_strategy is None:
                return self.inner.update(key_states, value_states, step_idx_tensor)
            return self.inner.update(
                key_states, value_states, step_idx_tensor, lookup_strategy
            )

        step_idx = int(step_idx_tensor)
        _, H, S, D = key_states.shape
        device, dtype = key_states.device, key_states.dtype

        # Start from latest-congruent history for ALL rows, then overwrite active.
        # Huginn write index: after step-0 increments _seen_tokens, token_pos =
        #   _seen_tokens - S + idx. Before that increment, the same slots are
        #   seen + idx. Recurrent steps already include the decode token in seen.
        k_full = torch.zeros(B, H, S, D, device=device, dtype=dtype)
        v_full = torch.zeros(B, H, S, D, device=device, dtype=dtype)
        seen = int(getattr(self.inner, "_seen_tokens", 0))
        for idx in range(S):
            token_pos = (seen + idx) if step_idx == 0 else (seen - S + idx)
            prior = self._lookup_latest_kv(step_idx, token_pos)
            if prior is not None:
                pk, pv = prior  # [B, H, D]
                k_full[:, :, idx, :] = pk
                v_full[:, :, idx, :] = pv

        k_full[rows] = key_states
        v_full[rows] = value_states

        if lookup_strategy is None:
            k_out, v_out = self.inner.update(k_full, v_full, step_idx_tensor)
        else:
            k_out, v_out = self.inner.update(
                k_full, v_full, step_idx_tensor, lookup_strategy
            )
        return k_out[rows], v_out[rows]


class AdaptiveRavenForCausalLM(RavenForCausalLM):
    """Overrides ``iterate_forward`` only. Attach worker/analyzer via ``attach_hook_stack``."""

    def attach_hook_stack(
        self,
        worker,
        analyzer,
        cfg: Optional[RecurrentDepthConfig] = None,
        raven_cfg: Optional[RavenAdapterConfig] = None,
    ):
        self.worker = worker
        self.analyzer = analyzer
        self.recur_cfg = cfg or getattr(worker, "cfg", None) or RecurrentDepthConfig()
        self.raven_cfg = raven_cfg or RavenAdapterConfig()
        self._aux_inputs = None
        self._baseline: Optional[RavenBaselineExit] = None
        if self.raven_cfg.baseline_criterion is not None:
            self._baseline = RavenBaselineExit(
                RavenBaselineConfig(
                    criterion=self.raven_cfg.baseline_criterion,
                    exit_threshold=self.raven_cfg.baseline_exit_threshold,
                    min_steps=self.recur_cfg.min_steps,
                )
            )

    # Override the iterate_forward method to add the adaptive exit mechanism,
    # including worker metric computations, ConvergenceState building, and
    # passage to analyzer for decision making about exit/steering.
    @torch._dynamo.disable(recursive=False)
    def iterate_forward(
        self,
        input_embeds: torch.Tensor,
        input_states: torch.Tensor,
        freqs_cis,
        block_idx: torch.Tensor,
        mask: Optional[BlockMask],
        past_key_values=None,
        num_steps: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
    ):
        if not hasattr(self, "worker") or self.worker is None:
            return super().iterate_forward(
                input_embeds,
                input_states,
                freqs_cis,
                block_idx,
                mask,
                past_key_values,
                num_steps,
                init_scale,
            )

        x = xk = (
            self.initialize_state(input_embeds, scale=init_scale)
            if input_states is None
            else input_states.clone()
        )
        B, S, _ = x.shape
        if num_steps is None:
            max_steps = int(self.config.mean_recurrence)
        elif hasattr(num_steps, "__len__") and len(num_steps) > 1:
            max_steps = int(num_steps[0] + num_steps[1])
        else:
            max_steps = int(num_steps)

        raven_cfg: RavenAdapterConfig = self.raven_cfg
        ctrl = ExitController(B, S, x.device)
        self.worker.reset()
        if self._baseline is not None:
            self._baseline.reset()
        prev_x = x.clone()

        cache = past_key_values
        proxy = None
        if raven_cfg.slice_decode and past_key_values is not None:
            r, p = block_geometry_from_config(self.config)
            proxy = RowSliceCacheProxy(
                past_key_values, recurrent_block_size=r, prelude_depth=p
            )
            cache = proxy

        with torch.no_grad():
            for step in range(max_steps):
                if not ctrl.active.any():
                    break

                xk = x
                use_slice = (
                    raven_cfg.slice_decode and S == 1 and not ctrl.active.all()
                )

                if use_slice:
                    rows = ctrl.active[:, 0].nonzero(as_tuple=True)[0]
                    if rows.numel() == 0:
                        break
                    if proxy is not None:
                        proxy.set_write_rows(rows, B)
                    freqs_sub = _slice_freqs(freqs_cis, rows)
                    x_sub, block_idx = self.core_block_forward(
                        x[rows],
                        input_embeds[rows],
                        freqs_sub,
                        mask,
                        cache,
                        block_idx,
                        step,
                    )
                    if proxy is not None:
                        proxy.clear_write_rows()
                    x_new = x.clone()
                    x_new[rows] = x_sub
                else:
                    x_all, block_idx = self.core_block_forward(
                        x,
                        input_embeds,
                        freqs_cis,
                        mask,
                        cache if proxy is None else proxy.inner,
                        block_idx,
                        step,
                    )
                    keep = ctrl.active.unsqueeze(-1)
                    x_new = torch.where(keep, x_all, x)

                aux = self._aux_inputs
                state = self.worker.build_state(x_new, prev_x, step, aux)

                logits = None
                if self._baseline is not None and self._baseline.cfg.needs_logits:
                    if aux is None:
                        raise RuntimeError(
                            "Huginn baseline criteria that need logits require "
                            "model._aux_inputs (cache_position / attention_mask / …)"
                        )
                    logits = self.predict_from_latents(x_new, **aux).logits

                if self._baseline is not None:
                    decision = self._baseline.analyze(state, ctrl, logits=logits)
                else:
                    decision = self.analyzer.analyze(state, ctrl)

                prev_x = x_new.clone()
                x = self.worker.apply_steering(x_new, decision)
                ctrl.apply(decision, step)

        self.last_exit_iteration = ctrl.exit_iteration
        self.last_nonconverging = ctrl.nonconverging
        self.last_active = ctrl.active
        return x, max_steps, torch.tensor(0), xk.detach(), block_idx
