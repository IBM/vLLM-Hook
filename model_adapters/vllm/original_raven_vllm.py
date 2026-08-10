# type: ignore
"""
original_raven_vllm.py:

This module contains the model architecture, class definitions,
and basic vLLM-compatible Huginn-3.5B model implementation as proposed by Geiping et al. (2025).

Source:
    Repository: https://github.com/seal-rg/recurrent-pretraining/blob/main
    Paper: "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach",
           Geiping et al., 2025 (https://arxiv.org/abs/2502.05171)
    Models: https://huggingface.co/tomg-group-umd/huginn-0125    
    License: Apache 2.0

This (original) implementation does not support adaptive exit/steering at test-time, 
a gap which is addressed in the adaptive_raven_vllm.py module that leverages the
vLLM-Hook protocol for this purpose.
"""

from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn

try:
    from vllm.model_executor.layers.attention import Attention
except ImportError:  # vLLM < 0.11
    from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.compilation.decorators import support_torch_compile  # noqa: F401


from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.sequence import IntermediateTensors

from transformers import PretrainedConfig
from math import sqrt

"""The HuggingFace-style model configuration, replicated for vllm plugin packaging"""

class RavenConfig(PretrainedConfig):
    model_type = "huginn_raven"
    keys_to_ignore_at_inference = [""]
    attribute_map = {"num_attention_heads": "n_heads", "hidden_size": "n_embd", "num_hidden_layers": "n_layers"}

    def __init__(
        self,
        n_embd: int = 5280,
        n_heads: int = 55,
        n_layers: int = 8,  # total of prelude + recurrent + coda
        block_size: int = 4096,
        vocab_size: int = 65536,
        padding_multiple: int = 4096,
        tie_embeddings: bool = True,
        intermediate_size: int = 17920,
        bias: bool = False,
        architecture_class_name: str = "RecurrentGPT",
        block_class_name: str = "SandwichBlock",
        norm_class_name: str = "RMSNorm_llama",
        norm_eps: float = 0.000001,
        mlp_class_name: str = "GatedMLP",
        nonlin_name: str = "SiLU",
        init_strategy: str = "takase",
        init_orthogonal: bool = False,
        state_init: str = "like-init",
        injection_type: str = "linear",
        n_layers_in_recurrent_block: int = 4,
        mean_recurrence: int = 32,
        sampling_scheme: str = "poisson-lognormal-filling",
        mean_backprop_depth: int = 8,
        n_layers_in_prelude: int = 2,
        n_layers_in_coda: int = 2,
        test_time_noise: float = 0.0,
        test_time_noise_type: str = "none",
        qk_bias: bool = True,
        activation_checkpoint_impl: str = "per-iteration",
        rope_base: float = 50_000,
        torch_dtype: str = "bfloat16",
        transformers_version: str = "4.47.1",
        **kwargs,
    ):
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.vocab_size = self.padded_vocab_size = vocab_size
        self.padding_multiple = padding_multiple
        self.tie_embeddings = tie_embeddings
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.architecture_class_name = architecture_class_name
        self.block_class_name = block_class_name
        self.norm_class_name = norm_class_name
        self.norm_eps = norm_eps
        self.mlp_class_name = mlp_class_name
        self.nonlin_name = nonlin_name
        self.init_strategy = init_strategy
        self.init_orthogonal = init_orthogonal
        self.state_init = state_init
        self.injection_type = injection_type
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.mean_recurrence = mean_recurrence
        self.sampling_scheme = sampling_scheme
        self.mean_backprop_depth = mean_backprop_depth
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_coda = n_layers_in_coda
        self.qk_bias = qk_bias
        self.activation_checkpoint_impl = activation_checkpoint_impl
        self.rope_base = rope_base
        self.torch_dtype = torch_dtype  # Added from JSON
        self.transformers_version = transformers_version  # Added from JSON
        # inference
        self.test_time_noise = test_time_noise
        self.test_time_noise_type = test_time_noise_type
        # Derived
        self.num_key_value_heads = n_heads
        self.num_attention_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.effective_expected_depth = (
            self.n_layers_in_prelude + self.n_layers_in_coda + self.n_layers_in_recurrent_block * self.mean_recurrence
        )
        self.init_values = {
            "std": sqrt(2 / (5 * self.n_embd)),
            "out_proj": sqrt(2 / (5 * self.n_embd)) / sqrt(2 * self.effective_expected_depth),
            "embedding": sqrt(2 / (5 * self.n_embd)),
            "embed_scale": sqrt(self.n_embd),
        }

        super().__init__(
            pad_token_id=65509,
            bos_token_id=65504,
            eos_token_id=[65505, 65508],
            tie_word_embeddings=tie_embeddings,
            **kwargs,
        )


class RavenAttention(nn.Module):
    def __init__(
        self,
        config: RavenConfig,
        vllm_config: Optional[VllmConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.total_num_heads

        # Tensor parallel setup
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Combined QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.qk_bias = nn.Parameter(torch.zeros(2, self.num_heads, self.head_dim))
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
            prefix=prefix,
        )
        # self.attn.use_direct_call = True

    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply QK bias and rotary on head dim
        total_tokens = hidden_states.shape[0]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)

        q_bias, k_bias = self.qk_bias.split(1, dim=0)
        q = (q + q_bias).to(q.dtype)
        k = (k + k_bias).to(q.dtype)
        q, k = self._apply_rotary_emb_complex_like(q, k, freqs_cis)
        # Flatten back for vllm attention
        q = q.view(total_tokens, -1)
        k = k.view(total_tokens, -1)
        attn_output = self.attn(q, k, v)
        # print(attn_output.flatten()[:5])
        output, _ = self.o_proj(attn_output)
        return output

    def _apply_rotary_emb_complex_like(
        self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # with torch.autocast("cuda", enabled=False):
        # Concatenate q and k on head dimension (dim=1)
        qk_concat = torch.cat([q, k], dim=1)
        qk_r2 = qk_concat.unflatten(dim=-1, sizes=(-1, 2)).float()  # cast to float32 for smooth skin
        rotated_qk_r2 = torch.stack(
            [
                qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
                qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        ).flatten(-2)
        q_rotated, k_rotated = torch.split(rotated_qk_r2.type_as(q), q.shape[1], dim=1)
        return q_rotated, k_rotated


class RavenMLP(nn.Module):
    def __init__(
        self,
        config: RavenConfig,
        vllm_config: Optional[VllmConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size

        # Gate and up projections combined
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        # Down projection
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        # Activation function
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class RavenDecoderLayer(nn.Module):
    """Single decoder layer with sandwich normalization."""

    def __init__(
        self,
        config: RavenConfig,
        layer_idx: int,
        vllm_config: Optional[VllmConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Sandwich normalization layers
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)

        # Attention and MLP
        self.self_attn = RavenAttention(config, vllm_config, quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = RavenMLP(config, vllm_config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.norm_1(hidden_states), freqs_cis)
        hidden_states = self.norm_2(attn_output + hidden_states)
        hidden_states = self.norm_4(self.mlp(self.norm_3(hidden_states)) + hidden_states)

        return hidden_states


@support_torch_compile
class RavenModel(nn.Module):
    """The Raven model consisting of prelude, adaptive core, and coda layers."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.n_embd,
            quant_config=quant_config,
        )

        # Embedding scale
        self.embed_scale = config.init_values["embed_scale"]

        # Create all layers in a flat structure
        total_layers = config.n_layers_in_prelude + config.n_layers_in_recurrent_block + config.n_layers_in_coda

        self.layers = nn.ModuleList(
            [
                RavenDecoderLayer(config, i, vllm_config, quant_config, prefix=f"{prefix}.layers.{i}")
                for i in range(total_layers)
            ]
        )
        # Adapter layer (concatenates embeddings with current state)
        self.adapter = RowParallelLinear(
            input_size=config.n_embd * 2,
            output_size=config.n_embd,
            bias=config.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.adapter",
        )

        # Final norm
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        # rope
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)

    def _precompute_freqs_cis(self):
        dim = self.config.n_embd // self.config.num_attention_heads
        end = self.config.block_size
        theta = self.config.rope_base
        # with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device)
        freqs = torch.outer(t, inv_freqs).float()
        return torch.stack([torch.cos(freqs)[:, None, :], torch.sin(freqs)[:, None, :]], dim=3)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def initialize_state(self, input_embeds, scale: float = 1.0):
        """Initialize adaptive state exactly like the reference implementation."""
        x = torch.randn_like(input_embeds)
        std = self.config.init_values["std"] * scale
        if std > 0:
            with torch.no_grad():
                torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
                if self.embed_scale != 1:
                    x = x * self.embed_scale
        else:
            x.zero_()
        return x

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get embeddings
        if inputs_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        if self.embed_scale != 1.0:
            input_embeds *= self.embed_scale
        # Get rope frequencies
        freqs_cis = self.freqs_cis.index_select(0, positions)

        # Prelude layers
        for i in range(self.config.n_layers_in_prelude):
            input_embeds = self.layers[i](input_embeds, freqs_cis)

        # Initialize recurrent state
        hidden_states = self.initialize_state(input_embeds)
        for recurrent_step in range(self.config.mean_recurrence):
            # Concatenate adaptive state with input embeddings (reference pattern)
            hidden_states, _ = self.adapter(torch.cat([hidden_states, input_embeds], dim=-1))

            for i in range(self.config.n_layers_in_recurrent_block):
                hidden_states = self.layers[self.config.n_layers_in_prelude + i](hidden_states, freqs_cis)

        # Apply final norm to core
        hidden_states = self.ln_f(hidden_states)

        # Coda layers
        coda_start = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block
        for i in range(self.config.n_layers_in_coda):
            layer = self.layers[coda_start + i]
            hidden_states = layer(hidden_states, freqs_cis)

        return self.ln_f(hidden_states)


class RavenForvLLM(nn.Module):
    """Raven model for causal language modeling with vLLM support."""

    _supports_attention_backend = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        # Main model
        self.model = RavenModel(vllm_config=vllm_config, prefix="model" if prefix == "" else prefix)

        # Language modeling head
        if config.tie_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.n_embd, quant_config=vllm_config.quant_config)

        # Logits processor and sampler
        self.logits_processor = LogitsProcessor(config.vocab_size, config.vocab_size, 1.0)
        # self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings for vLLM compatibility."""
        return self.model.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from a state dict."""
        params_dict = dict(self.named_parameters())

        missing_params = []
        loaded_params = []

        for name, loaded_weight in weights:
            if "freqs_cis" in name:
                continue  # is that in there? Will be recomputed

            # Handle parameter name mapping for compatibility
            if "transformer.wte" in name:
                name = name.replace("transformer.wte.weight", "model.embed_tokens.weight")
            elif "lm_head.weight" in name and self.config.tie_embeddings:
                # If weights are tied, lm_head shares weights with embeddings
                name = "model.embed_tokens.weight"
            elif "lm_head.weight" in name:
                name = "lm_head.weight"
            elif "transformer.prelude" in name:
                # Map prelude layers to flat layer structure
                parts = name.split(".")
                layer_idx = int(parts[2])  # transformer.prelude.X.rest
                rest = ".".join(parts[3:])
                name = f"model.layers.{layer_idx}.{rest}"
            elif "transformer.core_block" in name:
                # Map core layers to flat layer structure
                parts = name.split(".")
                layer_idx = int(parts[2])  # transformer.core_block.X.rest
                rest = ".".join(parts[3:])
                # Core layers start after prelude layers
                flat_idx = self.config.n_layers_in_prelude + layer_idx
                name = f"model.layers.{flat_idx}.{rest}"
            elif "transformer.coda" in name:
                # Map coda layers to flat layer structure
                parts = name.split(".")
                layer_idx = int(parts[2])  # transformer.coda.X.rest
                rest = ".".join(parts[3:])
                # Coda layers start after prelude + core layers
                flat_idx = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block + layer_idx
                name = f"model.layers.{flat_idx}.{rest}"
            elif "transformer.ln_f" in name:
                name = name.replace("transformer.ln_f", "model.ln_f")
            elif "transformer.adapter" in name:
                name = name.replace("transformer.adapter", "model.adapter")

            if "attn.Wqkv.weight" in name:
                name = name.replace("attn.Wqkv.weight", "self_attn.qkv_proj.weight")
            elif "attn.proj.weight" in name:
                name = name.replace("attn.proj.weight", "self_attn.o_proj.weight")
            elif "attn.qk_bias" in name:
                name = name.replace("attn.qk_bias", "self_attn.qk_bias")

            if "mlp.fc.weight" in name:
                # HF fc layer contains both gate and up weights concatenated
                name = name.replace("mlp.fc.weight", "mlp.gate_up_proj.weight")
            elif "mlp.proj.weight" in name:
                name = name.replace("mlp.proj.weight", "mlp.down_proj.weight")

            if name in params_dict:
                param = params_dict[name]
                # Handle special case for qk bias
                if "attn.qk_bias" in name:
                    default_weight_loader(param, loaded_weight.squeeze())
                else:
                    default_weight_loader(param, loaded_weight)
                loaded_params.append(name)
            else:
                missing_params.append(name)

        if missing_params:
            print(f"Missing parameters: {missing_params[:10]}...")

    @property
    def base_model_tp_plan(self):
        """Tensor parallel plan for the base model Not required for now."""
        return {}