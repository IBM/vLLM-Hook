"""
Spotlight attention steering utilities - backend-agnostic tensor operations.
Extracted from agent-lifecycle-toolkit for reuse in vLLM Hook.

**Paper**: [Venakteswaran and Contractor, EACL 2026](https://aclanthology.org/2026.eacl-long.174/)

Implementation: Q/K capture with logit-direct bias application.
"""
import os
import json
from typing import List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn.functional as F

from vllm import SamplingParams


def find_span(
    prompt: str,
    emph_string: str,
    offset_mapping: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Get start and end token indices of an emphasized text span within a prompt.

    Args:
        prompt: The full text prompt
        emph_string: The text span to find (must be present in prompt)
        offset_mapping: Tokenizer offset mapping (from return_offsets_mapping=True)

    Returns:
        (token_start, token_end + 1) — indices suitable for slicing token lists

    Raises:
        ValueError: if emph_string not found in prompt
        AssertionError: if span indices could not be located
    """
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
    """
    Find start and end token indices for all emphasized spans across all prompts.

    Args:
        prompts: List of input prompts
        emph_strings: Either a flat list of spans (applied to all prompts)
                     or a list of lists (one list per prompt)
        offset_mappings: Tokenizer offset mappings for each prompt

    Returns:
        List of span-range lists: one list per prompt, each containing (start, end) tuples
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    if len(emph_strings) == 0:
        emph_strings = []
    elif isinstance(emph_strings[0], str):
        # Flat list: apply same spans to all prompts
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
    """
    Compute scaled dot-product attention scores and apply softmax.

    Args:
        query: Query tensor, shape [batch, num_heads, seq_len, head_dim]
        key: Key tensor, shape [batch, num_heads, seq_len, head_dim]
        scale: Scaling factor, typically 1.0 / sqrt(head_dim)

    Returns:
        Attention weights after softmax, shape [batch, num_heads, seq_len, seq_len]
    """
    # Compute logits: Q @ K^T * scale
    logits = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply softmax over the sequence dimension
    return F.softmax(logits, dim=-1, dtype=torch.float32)


def apply_attention_weights(
    attn_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Apply attention weights to value states.

    Args:
        attn_weights: Attention weights, shape [batch, num_heads, seq_len, seq_len]
        value: Value tensor, shape [batch, num_heads, seq_len, head_dim]

    Returns:
        Attention output, shape [batch, num_heads, seq_len, head_dim]
    """
    return torch.matmul(attn_weights, value)


def reshape_attn_output(
    output: torch.Tensor,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
) -> torch.Tensor:
    """
    Reshape attention output from multihead to single vector.

    Args:
        output: Attention output, shape [batch, num_heads, seq_len, head_dim]
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Total hidden size (num_heads * head_dim)

    Returns:
        Reshaped output, shape [batch, seq_len, hidden_size]
    """
    # Transpose from [batch, heads, seq, dim] to [batch, seq, heads, dim]
    output = output.transpose(1, 2).contiguous()

    # Reshape to [batch, seq, hidden_size]
    return output.view(batch_size, seq_len, hidden_size)

def compute_spotlight_bias(
    logits: torch.Tensor,
    span_ranges: List[List[Tuple[int, int]]],
    target_proportion: float,
) -> torch.Tensor:
    """
    Apply Spotlight attention biasing (matches reference implementation).

    Reference: agent-lifecycle-toolkit SpotLightComponent.create_attention_bias_hook

    Algorithm (per batch item):
    1. Compute attention weights via softmax(logits)
    2. Compute current_proportion = sum of attention on span / total
    3. If current_proportion < target_proportion:
       a. Convert weights to log-domain: log(weights + eps)
       b. Add bias = log(target / current) to span positions
       c. Re-normalize with softmax
    4. Only increases attention toward span (never decreases)

    Args:
        logits: Raw attention logits, shape [batch, num_heads, q_len, k_len]
        span_ranges: List of span-range lists (one per batch item)
        target_proportion: Target proportion of attention on span (0.0 to 1.0)

    Returns:
        Modified attention weights (post-softmax), same shape as logits
    """
    # Compute baseline attention weights
    attn_weights = F.softmax(logits, dim=-1)
    modified_weights = attn_weights.clone()

    for batch_idx, ranges in enumerate(span_ranges):
        if not ranges:
            continue

        # Create union mask for emphasized token positions
        union_mask = torch.zeros(
            modified_weights.size(-1),
            device=modified_weights.device,
            dtype=modified_weights.dtype,
        )
        for start, end in ranges:
            union_mask[start:end] = 1.0
        union_mask = union_mask.view(1, 1, -1)  # [1, 1, k_len]

        # Current proportion: global across all heads and query positions
        current_proportion = (
            modified_weights[batch_idx] * union_mask
        ).sum() / modified_weights[batch_idx].sum()

        print(f"[BIAS] current={current_proportion:.4f}, target={target_proportion:.4f}, steer={current_proportion < target_proportion}")

        # Only steer upward (matching reference implementation)
        if current_proportion < target_proportion:
            bias_value = torch.log(
                torch.tensor(
                    target_proportion / current_proportion,
                    device=modified_weights.device,
                    dtype=torch.float32,
                )
            )
            bias_mask = union_mask * bias_value

            # Apply bias in log-domain of attention weights, then re-softmax
            attn_logits = logits[batch_idx].float()  # Use original logits for bias application
            attn_logits = attn_logits + bias_mask
            modified_weights[batch_idx] = F.softmax(
                attn_logits, dim=-1, dtype=torch.float32
            ).to(modified_weights.dtype)

            # Diagnostic: verify new proportion
            new_proportion = (
                modified_weights[batch_idx] * union_mask
            ).sum() / modified_weights[batch_idx].sum()
            print(f"[BIAS] after steering: {new_proportion:.4f}")

    return modified_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match number of query heads (for grouped-query attention).

    Args:
        hidden_states: KV tensor, shape [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        Repeated tensor, shape [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================================
# User-facing convenience function
# ============================================================================

def generate_with_spotlight(
    llm,
    prompts: Union[str, List[str]],
    emph_strings: Union[str, List[str], List[List[str]]],
    alpha: float = 0.2,
    sampling_params: Optional[SamplingParams] = None,
    **kwargs
):
    """
    Generate with Spotlight attention steering toward emphasized spans.

    Standalone function that works with any HookLLM instance configured
    with worker_name="probe_spotlight".

    Args:
        llm: A HookLLM instance with worker_name="probe_spotlight"
        prompts: Input prompt(s)
        emph_strings: Text span(s) to emphasize:
            - Single string: applied to all prompts
            - List of strings: applied to all prompts
            - List of lists: one list per prompt
        alpha: Target attention proportion (0.0-1.0)
        sampling_params: vLLM SamplingParams
        **kwargs: Additional vLLM generate kwargs (max_tokens, temperature, etc.)

    Returns:
        vLLM RequestOutput objects
    """
    # Normalize inputs
    if isinstance(prompts, str):
        prompts = [prompts]

    if isinstance(emph_strings, str):
        emph_strings = [[emph_strings]] * len(prompts)
    elif isinstance(emph_strings[0], str):
        emph_strings = [emph_strings] * len(prompts)

    # Apply chat template if available (instruct models)
    if hasattr(llm.tokenizer, 'chat_template') and llm.tokenizer.chat_template:
        templated_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            templated = llm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            templated_prompts.append(templated)
        prompts = templated_prompts

    params_file = None
    try:
        # Tokenize prompts with offset mappings
        tokenized = llm.tokenizer(
            prompts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
        )
        offset_mappings = tokenized.pop("offset_mapping")

        # Convert text spans to token ranges
        span_ranges = get_span_ranges(prompts, emph_strings, offset_mappings)

        # Write parameters to file for worker to read (cross-process safe)
        params_file = os.path.join(llm._hook_dir, "spotlight_params.json")
        with open(params_file, 'w') as f:
            json.dump({
                "span_ranges": span_ranges,
                "alpha": alpha,
            }, f)

        # Generate with hooks enabled
        result = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_hook=True,
            **kwargs
        )
        return result

    finally:
        # Clean up parameters file after generation
        try:
            if params_file and os.path.exists(params_file):
                os.remove(params_file)
        except (OSError, NameError, TypeError):
            pass
