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

from typing import Callable, NamedTuple, Optional

import torch
from torch import nn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm_hook_plugins.workers._common import get_query_metadata  # pyright: ignore[reportMissingImports]
from vllm_hook_plugins.protocols.recurrent_policy import (  # pyright: ignore[reportMissingImports]
    Readout,
    RecurrentAdapterCapabilities,
)
from vllm_hook_plugins.protocols.recurrent_step_controller import RecurrentStepController  # pyright: ignore[reportMissingImports]

from .original_raven_vllm import RavenDecoderLayer, RavenForvLLM, RavenModel


RAVEN_RECURRENT_CAPABILITIES = RecurrentAdapterCapabilities(
    readouts=frozenset(
        {
            Readout.CONTRACTION,
            Readout.DISPLACEMENT,
            Readout.COSINE,
            Readout.ACCELERATION,
            Readout.ENTROPY,
            Readout.ENTROPY_DELTA,
            Readout.CONFIDENCE,
            Readout.MARGIN,
            Readout.TOP1_STABILITY,
            Readout.PREDICTIVE_KL,
        }
    )
)

class LatentPrediction(NamedTuple):
    """vLLM Raven output projected from an intermediate recurrent latent."""

    logits: torch.Tensor
    hidden_states: torch.Tensor


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

        # Full MLP when all tokens are active
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
        self._compute_logits: Optional[
            Callable[[torch.Tensor], Optional[torch.Tensor]]
        ] = None
        # ``recurrent_depth`` arrives via HookLLM / hf_overrides on the HF config.
        # Import strings for ``worker`` / ``analyzer`` are resolved here so they
        # instantiate inside the GPU worker process (instances cannot cross fork).
        recur_dict = getattr(self.config, "recurrent_depth", None)
        self.controller = RecurrentStepController.from_config(
            self,
            recur_dict,
            capabilities=RAVEN_RECURRENT_CAPABILITIES,
        )
        # Sidecar for lm-eval / Pareto: one compute record per forward.
        self._exit_depth_samples: list[dict] = []

        # Give the core recurrent-block layers the active-token MLP path. Class
        # reassignment swaps only ``forward`` (no new params / submodules), so
        # weights load unchanged and the vLLM Attention modules built by the
        # parent are reused as-is (no re-registration).
        core_start = self.config.n_layers_in_prelude
        for i in range(core_start, core_start + self.config.n_layers_in_recurrent_block):
            # Use subclassed decoder layer instead of the base class
            self.layers[i].__class__ = AdaptiveRavenDecoderLayer

    def bind_compute_logits(
        self,
        compute_logits: Callable[[torch.Tensor], Optional[torch.Tensor]],
    ) -> None:
        """Bind the outer vLLM head after it has been initialized."""
        self._compute_logits = compute_logits

    def hidden_from_latents(
        self,
        latents: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """Run Raven's norm/coda/norm tail on flattened ``[T, D]`` latents."""
        if latents.ndim != 2:
            raise ValueError(
                f"hidden_from_latents expects [T, D], got {tuple(latents.shape)}"
            )
        hidden_states = self.ln_f(latents)
        coda_start = (self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block)
        for i in range(self.config.n_layers_in_coda):
            hidden_states = self.layers[coda_start + i](hidden_states, freqs_cis)
        return self.ln_f(hidden_states)

    def predict_from_latents(
        self,
        latents: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
    ) -> LatentPrediction:
        """Project an intermediate latent through Raven's exact vLLM tail."""
        had_sequence_dim = latents.ndim == 3
        if had_sequence_dim:
            if latents.shape[1] != 1:
                raise ValueError(
                    "vLLM recurrent latents must have shape [T, 1, D]"
                )
            latents = latents.squeeze(1)

        hidden_states = self.hidden_from_latents(latents, freqs_cis)
        logits = self._compute_logits(hidden_states)  # pyright: ignore[reportOptionalCall]
        if logits is None:
            raise RuntimeError("vLLM logits processor returned no logits")
        if logits.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(
                "predict_from_latents requires one logits row per latent; "
                f"got {logits.shape[0]} logits rows for "
                f"{hidden_states.shape[0]} latents"
            )

        if had_sequence_dim:
            logits = logits.unsqueeze(1)
            hidden_states = hidden_states.unsqueeze(1)
        return LatentPrediction(logits=logits, hidden_states=hidden_states)

    def reset_exit_depth_samples(self) -> None:
        self._exit_depth_samples.clear()

    def pop_exit_depth_samples(self) -> list[dict]:
        out = list(self._exit_depth_samples)
        self._exit_depth_samples.clear()
        return out

    def reset_trajectory_samples(self) -> None:
        self.controller.reset_trajectory_samples()

    def pop_trajectory_samples(self) -> list[dict]:
        return self.controller.pop_trajectory_samples()

    @staticmethod
    def _get_allow_exit(num_tokens: int, device: torch.device) -> torch.Tensor:
        """Return an exact per-row decode mask, with conservative fallbacks."""
        allow_exit = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return allow_exit
        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if attn_metadata is None:
            return allow_exit

        qsl, _ = get_query_metadata(attn_metadata)
        if qsl is None:
            return allow_exit

        # Current vLLM metadata marks chunked-prefill requests explicitly. Walk
        # hybrid-attention dictionaries to find the same field when present.
        metadata_entries = (
            attn_metadata.values() if isinstance(attn_metadata, dict) else (attn_metadata,)
        )
        is_prefilling = None
        for entry in metadata_entries:
            is_prefilling = getattr(entry, "is_prefilling", None)
            if is_prefilling is not None:
                break

        boundaries = [int(value) for value in qsl.detach().cpu().tolist()]
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] > num_tokens
            or any(left > right for left, right in zip(boundaries, boundaries[1:]))
        ):
            return allow_exit

        prefill_flags = None
        if is_prefilling is not None:
            prefill_flags = [
                bool(value) for value in is_prefilling.detach().cpu().tolist()
            ]
            if len(prefill_flags) != len(boundaries) - 1:
                return allow_exit

        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            # Older metadata has no explicit phase flag. A multi-token query is
            # definitely prefill; a one-token query is treated as decode.
            is_prefill = (
                prefill_flags[i] if prefill_flags is not None else end - start > 1
            )
            if not is_prefill:
                allow_exit[start:end] = True
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
        allow_exit = self._get_allow_exit(num_tokens, hidden_states.device)

        core_start = self.config.n_layers_in_prelude
        steps_run = 0
        max_steps = int(self.config.mean_recurrence)
        core_layers = int(self.config.n_layers_in_recurrent_block)
        # Token-steps that actually reach the MLP; accumulated on GPU so the
        # loop stays sync-free and one transfer reports the whole forward.
        mlp_token_steps = torch.zeros((), dtype=torch.long, device=hidden_states.device)

        for recurrent_step in range(max_steps):
            prev = hidden_states  # start-of-step state (not mutated in place)
            mlp_token_steps += active.sum()

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
            decision = self.controller.step(
                h,
                prev,
                recurrent_step,
                prediction_inputs={"freqs_cis": freqs_cis},
                phase_mask=allow_exit,
            )
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
            self._exit_depth_samples.append(
                self._compute_record(
                    exits,
                    allow_exit,
                    mlp_token_steps,
                    num_tokens=num_tokens,
                    steps_run=steps_run,
                    max_steps=max_steps,
                    core_layers=core_layers,
                )
            )

        return self.hidden_from_latents(hidden_states, freqs_cis)

    @staticmethod
    def _compute_record(
        exit_iteration: torch.Tensor,
        allow_exit: torch.Tensor,
        mlp_token_steps: torch.Tensor,
        *,
        num_tokens: int,
        steps_run: int,
        max_steps: int,
        core_layers: int,
    ) -> dict:
        """Summarize one forward's recurrence cost, keeping decode rows separate.

        Prefill rows can never exit, so mixing them into a single mean hides the
        real decode-time savings. Everything is reduced on GPU and moved in one
        transfer.
        """
        depths = exit_iteration.to(dtype=torch.long)
        depths = torch.where(depths < 0, torch.full_like(depths, max_steps - 1), depths) + 1
        decode = allow_exit.to(dtype=torch.bool)
        sentinel_high = torch.where(decode, depths, torch.full_like(depths, max_steps))
        sentinel_low = torch.where(decode, depths, torch.zeros_like(depths))
        reduced = torch.stack(
            [
                depths.sum(),
                (depths * decode.to(dtype=depths.dtype)).sum(),
                decode.sum(),
                mlp_token_steps.to(dtype=depths.dtype),
                sentinel_high.min(),
                sentinel_low.max(),
            ]
        ).tolist()
        depth_sum, decode_depth_sum, n_decode, mlp_steps, decode_min, decode_max = reduced
        return {
            "n_tokens": int(num_tokens),
            "n_decode_tokens": int(n_decode),
            "steps_run": int(steps_run),
            "max_steps": int(max_steps),
            "depth_sum": int(depth_sum),
            "decode_depth_sum": int(decode_depth_sum),
            "decode_min_depth": int(decode_min) if n_decode else None,
            "decode_max_depth": int(decode_max) if n_decode else None,
            # Attention always runs on every row; only the MLP is skipped.
            "attn_token_steps": int(num_tokens) * int(steps_run) * core_layers,
            "mlp_token_steps": int(mlp_steps) * core_layers,
        }


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
        self.model.bind_compute_logits(self.compute_logits)

    def reset_exit_depth_samples(self) -> None:
        self.model.reset_exit_depth_samples()

    def pop_exit_depth_samples(self) -> list[dict]:
        return self.model.pop_exit_depth_samples()

    def reset_trajectory_samples(self) -> None:
        self.model.reset_trajectory_samples()

    def pop_trajectory_samples(self) -> list[dict]:
        return self.model.pop_trajectory_samples()
