"""Adaptive Raven executor for vLLM + the vLLM-Hook adaptive-exit protocol.

This subclasses the vendored ``RavenModel`` / ``RavenForvLLM`` (defined in
``model_adapters/vllm/original_raven_vllm.py``, left untouched) and adds per-token adaptive early
exit inside the recurrence loop via :class:`RecurrentStepController`.

Design:
    * The vendored executor gives us the vLLM-native forward graph: flattened
      ``[T, D]`` hidden states, vLLM ``Attention`` / linear layers, weight load.
    * The Hook protocol (worker -> analyzer -> ExitController), reused as-is,
      decides *when* each token exits. The controller is the single object the
      loop talks to and handles the ``[T, D]`` <-> ``[B, S]`` bridging.

Only ``RavenModel.forward`` and ``RavenDecoderLayer.forward`` (for the core
recurrent block, to skip the MLP on converged tokens) are overridden.
Everything else (weights, logits, attention, RoPE, cache) is inherited from the
reference implementation.

Users reach this through ``HookLLM`` / ``vllm serve`` after
:func:`model_adapters.vllm.register_adaptive_raven`. The architecture name is
``AdaptiveRavenForvLLM`` (not seal-rg ``RavenForCausalLM``), so Huginn
checkpoints need ``hf_overrides["architectures"]``::

    from model_adapters.vllm import ADAPTIVE_RAVEN_ARCH, register_adaptive_raven

    register_adaptive_raven()
    HookLLM(
        model="tomg-group-umd/huginn-0125",
        trust_remote_code=True,
        enforce_eager=True,
        hf_overrides={
            "architectures": [ADAPTIVE_RAVEN_ARCH],
            "recurrent_depth": {"rho": 0.02, "min_steps": 2},
        },
    )

``rho = 0`` (default) never exits, reproducing the fixed-depth model exactly.
"""

from typing import Optional

import torch
from torch import nn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm_hook_plugins.workers._common import get_query_metadata  # pyright: ignore[reportMissingImports]
from vllm_hook_plugins.protocols.recurrent_step_controller import RecurrentStepController  # pyright: ignore[reportMissingImports]

from .original_raven_vllm import RavenDecoderLayer, RavenForvLLM, RavenModel


class AdaptiveRavenDecoderLayer(RavenDecoderLayer):
    """``RavenDecoderLayer`` that runs the MLP on active tokens only.

    Attention runs on all tokens (every position's K/V must stay correct), but
    the MLP - the bulk of the per-token FLOPs - is computed only for rows still
    active in the recurrence and scattered back. Converged rows keep their
    post-attention value. When all rows are active (e.g. the exact-match
    ``rho = 0`` oracle) the fast path is bit-identical to the base layer.

    ``active`` is the flattened ``[T]`` bool mask supplied by the model loop.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.self_attn(self.norm_1(hidden_states), freqs_cis)
        hidden_states = self.norm_2(attn_output + hidden_states)

        # Full MLP when everything is active — no gather/scatter overhead.
        if active is None or bool(active.all()):
            return self.norm_4(self.mlp(self.norm_3(hidden_states)) + hidden_states)

        active_tokens = active.nonzero(as_tuple=True)[0]
        h_sliced = hidden_states[active_tokens]
        mlp_final = self.norm_4(self.mlp(self.norm_3(h_sliced)) + h_sliced)
        return hidden_states.index_copy(0, active_tokens, mlp_final)


class AdaptiveRavenModel(RavenModel):
    """``RavenModel`` with per-token adaptive exit in the recurrent core."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # ``recurrent_depth`` arrives via HookLLM / hf_overrides on the HF config.
        # Import strings for ``worker`` / ``analyzer`` are resolved here so they
        # instantiate inside the GPU worker process (instances cannot cross fork).
        recur_dict = getattr(self.config, "recurrent_depth", None)
        self.controller = RecurrentStepController.from_config(self, recur_dict)
        # Sidecar for lm-eval / Pareto: mean effective recurrence per forward.
        self._exit_depth_samples: list[float] = []

        # Give the core recurrent-block layers the active-token MLP path. Class
        # reassignment swaps only ``forward`` (no new params / submodules), so
        # weights load unchanged and the vLLM Attention modules built by the
        # parent are reused as-is (no re-registration).
        core_start = self.config.n_layers_in_prelude
        for i in range(core_start, core_start + self.config.n_layers_in_recurrent_block):
            # Use subclassed decoder layer instead of the base class
            self.layers[i].__class__ = AdaptiveRavenDecoderLayer

    def reset_exit_depth_samples(self) -> None:
        self._exit_depth_samples.clear()

    def pop_exit_depth_samples(self) -> list[float]:
        out = list(self._exit_depth_samples)
        self._exit_depth_samples.clear()
        return out
    
    @staticmethod
    def _get_allow_exit(num_tokens: torch.Tensor, device = torch.device) -> torch.Tensor:

        allow_exit = torch.ones(num_tokens.shape[0], dtype=torch.bool, device=num_tokens.device)
        attn_metadata = getattr(get_forward_context(), "attn_metadata", None)
        if attn_metadata is None:
            return allow_exit

        qsl, seq_lens = get_query_metadata(attn_metadata)
        if qsl is None:
            return allow_exit

        qsl = qsl.to(device)
        for i in range(len(qsl) - 1):
            if seq_lens[i] > 1: 
                allow_exit[int(qsl[i]):int(qsl[i + 1])] = False
        return allow_exit

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embeddings ([T, D] flattened across the whole batch under vLLM).
        if inputs_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        else:
            input_embeds = inputs_embeds
        if self.embed_scale != 1.0:
            input_embeds = input_embeds * self.embed_scale

        freqs_cis = self.freqs_cis.index_select(0, positions)  # type: ignore

        # Prelude (non-recurrent).
        for i in range(self.config.n_layers_in_prelude):
            input_embeds = self.layers[i](input_embeds, freqs_cis)

        # Recurrent state + per-token exit bookkeeping.
        hidden_states = self.initialize_state(input_embeds)
        num_tokens = hidden_states.shape[0]
        active = self.controller.reset(num_tokens, hidden_states.device)
        allow_exit = _get_allow_exit(num_tokens, hidden_states.device)  # pyright: ignore[reportArgumentType]
        
        core_start = self.config.n_layers_in_prelude
        steps_run = 0
        max_steps = int(self.config.mean_recurrence)

        for recurrent_step in range(max_steps):
            prev = hidden_states  # start-of-step state (not mutated in place)

            # Inject embeddings, then run the weight-tied core block. Each core
            # layer runs attention on all rows but the MLP on active rows only,
            # so converged tokens stop consuming the bulk of the compute. The
            # ``torch.where`` guards keep converged rows pinned to their
            # start-of-step latent (``prev``) before and after every sublayer,
            # so once a token exits it does not drift via the adapter injection
            # or attention.
            h, _ = self.adapter(torch.cat([hidden_states, input_embeds], dim=-1))
            h = torch.where(active.unsqueeze(-1), h, prev)
            for i in range(self.config.n_layers_in_recurrent_block):
                h_next = self.layers[core_start + i](h, freqs_cis, active)
                h = torch.where(active.unsqueeze(-1), h_next, prev)

            # vLLM-Hook protocol call at the end of the step.
            decision = self.controller.step(h, prev, recurrent_step)
            # Only allow exit for tokens during decode
            decision.exit_mask &= allow_exit.unsqueeze(1)
            hidden_states = self.controller.steer(h, decision)  # no-op in Stage 1
            active = self.controller.apply(decision, recurrent_step)
            steps_run = recurrent_step + 1

            if not active.any():
                break  # all tokens converged; skip remaining recurrence

        # Diagnostics for demos, benchmarks, etc. (flattened [T]; mirrors HF adapter fields).
        self.last_exit_iteration = self.controller.exit_iteration
        self.last_nonconverging = self.controller.nonconverging
        self.last_active = self.controller.active
        self.last_recurrence_steps_run = steps_run
        exits = self.controller.exit_iteration
        if exits is not None:
            depths = exits.to(dtype=torch.long).clone()
            never = depths < 0
            depths = depths + 1
            depths[never] = max_steps
            self._exit_depth_samples.append(float(depths.float().mean().item()))

        hidden_states = self.ln_f(hidden_states)

        # Coda (non-recurrent).
        coda_start = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block
        for i in range(self.config.n_layers_in_coda):
            hidden_states = self.layers[coda_start + i](hidden_states, freqs_cis)

        return self.ln_f(hidden_states)


class AdaptiveRavenForvLLM(RavenForvLLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model = AdaptiveRavenModel(
            vllm_config=vllm_config,
            prefix="model" if prefix == "" else prefix,
        )
        if config.tie_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.n_embd,
                quant_config=vllm_config.quant_config,
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, config.vocab_size, 1.0)

    def reset_exit_depth_samples(self) -> None:
        self.model.reset_exit_depth_samples()

    def pop_exit_depth_samples(self) -> list[float]:
        return self.model.pop_exit_depth_samples()
