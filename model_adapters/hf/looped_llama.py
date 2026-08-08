# type: ignore

"""
looped_llama.py:

This module contains the model architecture and class definitions for the
retrofitted recurrent depth Llama models, as proposed by McLeish et al. (2025).

Source:
    Repository: https://github.com/mcleish7/retrofitting-recurrence/blob/master/
    Paper: "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence",
           McLeish et al., 2025 (https://arxiv.org/pdf/2511.07384)
    Models: https://hf.co/collections/tomg-group-umd/retrofitting-recurrence
    License: Apache 2.0

This file contains MLP, decoder layer, and model class definitions 
for the looped Llama models. The architecture splices prelude, recurrent, 
and coda layer blocks from original Llama models and features a 
forward pass (modified from standard transformer architectures) 
that loops over the recurrent block. Imports are resolved to match
current package versions (from the original file).

The vLLM-Hook recurrent depth and steering adapter enables variable-depth
recurrence at runtime with gradient-based early exit and activation steering
for controlled generation.
"""

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP
)

from transformers.models.llama.configuration_llama import LlamaConfig

from torch import nn
import torch
from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
import random
from transformers.activations import ACT2FN
from vllm import logger

class LoopedLlamaMLP(nn.Module):
    """
    Adds option to have different input and output shapes to mlp
    """

    def __init__(self, hidden_size, intermediate_size, hidden_act, output_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.output_size = self.hidden_size if output_size is None else output_size
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LoopedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(self, *args, cache_layer_idx=None, **kwargs):
        if cache_layer_idx is not None:
            saved_cache_idx = self.self_attn.layer_idx
            self.self_attn.layer_idx = cache_layer_idx
        out = super().forward(*args, **kwargs)
        if cache_layer_idx is not None:
            self.self_attn.layer_idx = saved_cache_idx
        return out

class LoopConfig:
    num_rec = 2
    start_index = 4
    block_size = 2
    coda_size = None
    prelude_size = None
    remove_layers = "none"

    def __init__(self, config: dict = None):
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown config key: {key}")

    def __repr__(self):
        return str(
            {key: getattr(self, key) for key in vars(self) if not key.startswith("__")}
        )


class LoopedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LoopedLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def rec_post_init(self, args, extra_tensors={}):
        self.model.rec_post_init(args, extra_tensors)

    def set_num_rec(self, new_num_rec):
        self.model.loop_config.num_rec = new_num_rec

    def get_num_rec(self):
        return self.model.loop_config.num_rec

    def get_latest_rep(self):
        return self.model.latest_rep


class LoopedLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LoopedLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_split(self, num_rec, remove_layers, i, j):
        base_list = list(range(self.config.num_hidden_layers))
        rec_layers = base_list[i : i + j]
        prelude = base_list[:i]
        coda = base_list[i + j :]
        return prelude, rec_layers, coda

    def rec_post_init(self, args, extra_tensors):
        self.loop_config = LoopConfig(args)

        i, j = self.loop_config.start_index, self.loop_config.block_size
        prelude_ind, rec_layers_ind, coda_ind = self.get_split(
            self.loop_config.num_rec, self.loop_config.remove_layers, i, j
        )
        if self.loop_config.coda_size is not None:
            coda_ind = coda_ind[:self.loop_config.coda_size]
        if self.loop_config.prelude_size is not None:
            prelude_ind = prelude_ind[:self.loop_config.prelude_size]

        print(f"regular model: {list(range(self.config.num_hidden_layers))}")
        print(f"prelude: {prelude_ind}")
        print(f"rec block: {rec_layers_ind}")
        print(f"coda: {coda_ind}")
        self.prelude = nn.ModuleList([self.layers[idx] for idx in prelude_ind])
        rec_layers = [self.layers[idx] for idx in rec_layers_ind]

        self.rec_block = nn.ModuleList(rec_layers)
        self.coda = nn.ModuleList([self.layers[idx] for idx in coda_ind])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        hidden_states, all_hidden_states, all_self_attns = self.prelude_rec_coda(
            output_hidden_states,
            all_hidden_states,
            hidden_states,
            causal_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            flash_attn_kwargs,
            all_self_attns,
        )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prelude_rec_coda(
        self,
        output_hidden_states,
        all_hidden_states,
        hidden_states,
        causal_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings,
        flash_attn_kwargs,
        all_self_attns,
    ):
        og_input_to_block = None
        layer_count = 0
        for layers, reps, is_rec_block, is_pre in [
            (self.prelude, 1, False, True),
            (self.rec_block, self.loop_config.num_rec, True, False),
            (self.coda, 1, False, False),
        ]:
            keep_looping = True
            for rep in range(reps):
                this_use_cache = use_cache
                pkv_for_layer = past_key_values
                this_layer_count = None

                for idx, decoder_layer in enumerate(layers):

                    if output_hidden_states:
                        all_hidden_states += (hidden_states.clone().detach(),)

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=pkv_for_layer,
                        output_attentions=output_attentions,
                        use_cache=this_use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                        cache_layer_idx=this_layer_count,
                    )

                    hidden_states = layer_outputs[0]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)


        return hidden_states, all_hidden_states, all_self_attns