"""Spotlight attention-steering utilities: backend-agnostic tensor operations."""
import logging
from typing import List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from vllm import SamplingParams


def find_span(
    prompt: str,
    emph_string: str,
    offset_mapping: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    """Get start and end token indices of an emphasized text span within a prompt."""
    if emph_string not in prompt:
        raise ValueError(f'"{emph_string}" not found in "{prompt}"')

    start_idx = prompt.find(emph_string)
    end_idx = start_idx + len(emph_string)

    span_start, span_end = None, None
    for index, (tk_start, tk_end) in enumerate(offset_mapping):
        if span_start is None:
            if tk_start <= start_idx and tk_end >= start_idx:
                span_start = index
        if span_end is None:
            if tk_start <= end_idx and tk_end >= end_idx:
                span_end = index
                break

    assert (
        span_start is not None and span_end is not None and span_start <= span_end
    ), f"Could not locate span for '{emph_string}' in prompt"

    return (span_start, span_end + 1)


def get_span_ranges(
    prompts: List[str],
    emph_strings: List[str] | List[List[str]],
    offset_mappings: Sequence[Sequence[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Find start and end token indices for all emphasized spans across all prompts."""
    if isinstance(prompts, str):
        prompts = [prompts]

    if len(emph_strings) == 0:
        emph_strings = []
    elif isinstance(emph_strings[0], str):
        emph_strings = [[s] for s in emph_strings]

    assert len(prompts) == len(emph_strings), (
        f"Mismatch between prompts ({len(prompts)}) and emph_strings ({len(emph_strings)})"
    )

    span_ranges_per_sample = []
    for prompt, span_list, offsets in zip(prompts, emph_strings, offset_mappings):
        sample_ranges = []
        for span in span_list:
            if not span:
                continue
            if span not in prompt:
                raise ValueError(f'Span "{span}" not found in prompt')
            if prompt.count(span) != 1:
                raise ValueError(f'Ambiguous span "{span}" (appears {prompt.count(span)} times)')
            rng = find_span(prompt, span, offset_mapping=offsets)
            sample_ranges.append(rng)
        span_ranges_per_sample.append(sample_ranges)

    return span_ranges_per_sample


def compute_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute scaled dot-product attention scores and apply softmax."""
    logits = torch.matmul(query, key.transpose(-2, -1)) * scale

    return F.softmax(logits, dim=-1, dtype=torch.float32)


def apply_attention_weights(
    attn_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Apply attention weights to value states."""
    return torch.matmul(attn_weights, value)


def reshape_attn_output(
    output: torch.Tensor,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
) -> torch.Tensor:
    """Reshape attention output from multihead to single vector."""
    output = output.transpose(1, 2).contiguous()

    return output.view(batch_size, seq_len, hidden_size)

def compute_spotlight_bias(
    logits: torch.Tensor,
    span_ranges: List[List[Tuple[int, int]]],
    target_proportion: float,
) -> torch.Tensor:
    """Apply Spotlight attention biasing (matches reference implementation)."""
    attn_weights = F.softmax(logits, dim=-1)
    modified_weights = attn_weights.clone()

    for batch_idx, ranges in enumerate(span_ranges):
        if not ranges:
            continue

        union_mask = torch.zeros(
            modified_weights.size(-1),
            device=modified_weights.device,
            dtype=modified_weights.dtype,
        )
        for start, end in ranges:
            union_mask[start:end] = 1.0
        union_mask = union_mask.view(1, 1, -1)

        current_proportion = (
            modified_weights[batch_idx] * union_mask
        ).sum() / modified_weights[batch_idx].sum()

        logger.debug(f"current={current_proportion:.4f}, target={target_proportion:.4f}, steer={current_proportion < target_proportion}")

        if current_proportion < target_proportion:
            bias_value = torch.log(
                torch.tensor(
                    target_proportion / current_proportion,
                    device=modified_weights.device,
                    dtype=torch.float32,
                )
            )
            bias_mask = union_mask * bias_value

            attn_logits = logits[batch_idx].float()
            attn_logits = attn_logits + bias_mask
            modified_weights[batch_idx] = F.softmax(
                attn_logits, dim=-1, dtype=torch.float32
            ).to(modified_weights.dtype)

            if logger.isEnabledFor(logging.DEBUG):
                new_proportion = (
                    modified_weights[batch_idx] * union_mask
                ).sum() / modified_weights[batch_idx].sum()
                logger.debug(f"after steering: {new_proportion:.4f}")

    return modified_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match number of query heads (for grouped-query attention)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def generate_with_spotlight(
    llm,
    prompts: Union[str, List[str]],
    emph_strings: Union[str, List[str], List[List[str]]],
    alpha: float = 0.2,
    sampling_params: Optional[SamplingParams] = None,
    **kwargs
):
    """Generate with Spotlight attention steering toward emphasized spans."""
    import copy

    if isinstance(prompts, str):
        prompts = [prompts]

    if isinstance(emph_strings, str):
        emph_strings = [[emph_strings]] * len(prompts)
    elif isinstance(emph_strings[0], str):
        emph_strings = [emph_strings] * len(prompts)

    if hasattr(llm.tokenizer, 'chat_template') and llm.tokenizer.chat_template:
        templated_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            templated = llm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            templated_prompts.append(templated)
        prompts = templated_prompts

    tokenized = llm.tokenizer(
        prompts,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
    )
    offset_mappings = tokenized.pop("offset_mapping")

    span_ranges = get_span_ranges(prompts, emph_strings, offset_mappings)

    base_params = sampling_params or SamplingParams(**kwargs)
    sp_list = []
    for i in range(len(prompts)):
        sp = copy.copy(base_params)
        extra = dict(sp.extra_args or {})
        extra["spotlight"] = {
            "span_ranges": span_ranges[i],
            "alpha": alpha,
        }
        sp.extra_args = extra
        sp_list.append(sp)

    engine = getattr(llm, "llm", llm)
    if not getattr(engine, "_mia_installed", False):
        install_hooks = getattr(engine, "collective_rpc", None)
        if callable(install_hooks):
            print("Installing hooks via collective_rpc")
            install_hooks("install_hooks")
            setattr(engine, "_mia_installed", True)

    return llm.generate(
        prompts=prompts,
        sampling_params=sp_list,
        use_hook=True,
    )

