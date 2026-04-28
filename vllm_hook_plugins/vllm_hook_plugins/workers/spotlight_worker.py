"""
Spotlight Worker for vLLM Hook - implements attention steering via token span highlighting.
Based on agent-lifecycle-toolkit's Spotlight component.

**Paper**: [Venakteswaran and Contractor, EACL 2026](https://aclanthology.org/2026.eacl-long.174/)

Implementation: Q/K capture at .attn submodule level with logit-direct bias.
"""
import os
import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from vllm.v1.worker.gpu_worker import Worker as V1Worker
from vllm.forward_context import get_forward_context
import re
from vllm.distributed import parallel_state as ps

from vllm_hook_plugins.utils.spotlight.utils import (
    compute_spotlight_bias,
    compute_attention_scores,
    apply_attention_weights,
    reshape_attn_output,
    repeat_kv,
)

# ============================================================================
# Parameter Access (File-Based, Cross-Process Safe)
# ============================================================================
# Main process writes params to a file in the hook directory before prefill.
# Worker reads from file at hook-fire time (filesystem is shared across processes).

def get_spotlight_params_from_file(params_file: str) -> Optional[Dict]:
    """
    Read spotlight parameters from shared file (cross-process safe).

    Called by the hook at fire time. Main process writes this file before prefill.

    Args:
        params_file: Path to spotlight_params.json in VLLM_HOOK_DIR

    Returns:
        Dict with "span_ranges" and "alpha", or None if file doesn't exist
    """
    import json
    try:
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError):
        pass
    return None

# ============================================================================
# Attention Pattern Matching
# ============================================================================

ATTN_PATTERNS = [
    # GPT-2: transformer.h.<i>.attn.attn
    re.compile(r"^transformer\.h\.(\d+)\.attn\.attn$"),
    # OPT: model.decoder.layers.<i>.self_attn.attn
    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.attn$"),
    # Qwen/LLaMA: model.layers.<i>.self_attn.attn
    re.compile(r"^model\.layers\.(\d+)\.self_attn\.attn$"),
]


def match_attn(name: str):
    """Match attention module name and extract layer index."""
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


# ============================================================================
# Spotlight Worker Implementation
# ============================================================================

class SpotlightWorker(V1Worker):
    """
    vLLM Worker that implements Spotlight's attention steering during inference.

    For each emphasized span in the prompt, steers attention weights toward that span
    by applying a log-ratio bias directly to attention logits.

    Usage:
        from vllm_hook_plugins.workers.spotlight_worker import set_spotlight_params

        llm = HookLLM(model="...", worker_name="probe_spotlight")

        # Before generate(), set spotlight parameters
        set_spotlight_params(
            span_ranges=[[[0, 4]]],  # Emphasize tokens 0-4 in first prompt
            alpha=0.3                # Target 30% attention on span
        )

        outputs = llm.generate(["Answer in JSON. What is 2+2?"])
    """

    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)
        try:
            self._install_hooks()
            print("Spotlight hooks installed successfully")
        except Exception as e:
            print(f"Spotlight hook installation failed: {e}")
        return r

    def _install_hooks(self):
        """Install Spotlight hooks on .attn modules with proper Q/K/V access."""
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip spotlight hooks")
            return

        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        if not self.hook_flag:
            print("Missing VLLM_HOOK_FLAG environment variable")
            return

        # Path to spotlight params file
        hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.spotlight_params_file = os.path.join(hook_dir, "spotlight_params.json") if hook_dir else None

        cfg = model.config
        num_h = int(getattr(cfg, "num_attention_heads", 32))
        num_kv = int(getattr(cfg, "num_key_value_heads", num_h))
        hidden = int(getattr(cfg, "hidden_size", 4096))
        head_dim = hidden // num_h
        attn_scale = 1.0 / math.sqrt(head_dim)

        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attn_scale=attn_scale,
        )

        def spotlight_attention_hook(module, input_tuple, output, *, module_name=None):
            """
            Spotlight hook on attention modules with proper tensor reshaping.

            input_tuple[0] = Q [total_tokens, hidden_size]
            input_tuple[1] = K [total_tokens, kv_dim]
            input_tuple[2] = V [total_tokens, kv_dim]
            """
            if not os.path.exists(self.hook_flag):
                return None

            params = get_spotlight_params_from_file(self.spotlight_params_file)
            if not params:
                return None

            span_ranges = params.get("span_ranges", [])
            alpha = params.get("alpha", 0.2)

            if not span_ranges or not any(span_ranges):
                return None

            try:
                Q = input_tuple[0]  # [total_tokens, hidden_size]
                K = input_tuple[1]  # [total_tokens, kv_dim]
                V = input_tuple[2]  # [total_tokens, kv_dim]

                # Get attention metadata — it's a dict keyed by layer name
                ctx = get_forward_context()
                metadata = getattr(ctx, "attn_metadata", None)
                if metadata is None:
                    return None

                # metadata is a dict: metadata[layer_name].seq_lens
                layer_meta = metadata.get(module_name) if isinstance(metadata, dict) else metadata
                if layer_meta is None:
                    return None

                # seq_lens = total sequence length (including KV cache)
                # query_start_loc = cumulative query token offsets
                # Q/K/V only contain query_len tokens (new tokens), not seq_len
                seq_lens = getattr(layer_meta, "seq_lens", None)
                query_start_loc = getattr(layer_meta, "query_start_loc", None)
                if seq_lens is None or query_start_loc is None:
                    return None

                # Compute query lengths per request
                query_lens = query_start_loc[1:] - query_start_loc[:-1]
                batch_size = len(seq_lens)

                # Only apply during initial prefill where all tokens are new
                # (query_len == seq_len means no cached tokens, full prompt available)
                # During decode or partial prefill we can't recompute attention
                # without KV cache access
                is_initial_prefill = torch.all(query_lens == seq_lens).item()
                if not is_initial_prefill:
                    return None

                print(f"[HOOK] Prefill: alpha={alpha}, Q={Q.shape}, query_lens={query_lens}")

                # Process each batch item separately
                all_outputs = []
                for batch_idx in range(batch_size):
                    q_start = int(query_start_loc[batch_idx])
                    q_end = int(query_start_loc[batch_idx + 1])
                    qlen = q_end - q_start

                    # Extract this batch's Q, K, V (only query tokens)
                    Q_batch = Q[q_start:q_end]  # [qlen, hidden_size]
                    K_batch = K[q_start:q_end]  # [qlen, kv_dim]
                    V_batch = V[q_start:q_end]  # [qlen, kv_dim]

                    orig_dtype = Q_batch.dtype

                    # Reshape to [num_heads, qlen, head_dim] and upcast to float32
                    # (float16 overflows in Q @ K^T; fused kernels handle this internally)
                    Q_batch = Q_batch.reshape(qlen, num_h, head_dim).transpose(0, 1).float()
                    K_batch = K_batch.reshape(qlen, num_kv, head_dim).transpose(0, 1).float()
                    V_batch = V_batch.reshape(qlen, num_kv, head_dim).transpose(0, 1).float()

                    # Handle grouped-query attention
                    if num_h != num_kv:
                        n_rep = num_h // num_kv
                        K_batch = repeat_kv(K_batch.unsqueeze(0), n_rep).squeeze(0)
                        V_batch = repeat_kv(V_batch.unsqueeze(0), n_rep).squeeze(0)

                    # Compute attention logits: Q @ K^T * scale
                    logits = torch.matmul(Q_batch, K_batch.transpose(-1, -2)) * attn_scale

                    # Apply causal mask
                    causal_mask = torch.triu(
                        torch.full((qlen, qlen), float('-inf'), device=logits.device, dtype=torch.float32),
                        diagonal=1
                    )
                    logits = logits + causal_mask.unsqueeze(0)

                    # Apply Spotlight bias
                    if batch_idx < len(span_ranges):
                        batch_spans = [span_ranges[batch_idx]]
                        logits_4d = logits.unsqueeze(0)  # [1, heads, qlen, qlen]
                        modified_weights = compute_spotlight_bias(logits_4d, batch_spans, target_proportion=alpha)
                        modified_weights = modified_weights.squeeze(0)
                    else:
                        modified_weights = F.softmax(logits, dim=-1)

                    # Compute attention output: weights @ V
                    attn_out = torch.matmul(modified_weights, V_batch)  # [heads, qlen, head_dim]
                    attn_out = attn_out.transpose(0, 1).reshape(qlen, -1)  # [qlen, hidden_size]
                    all_outputs.append(attn_out.to(orig_dtype))

                output_tensor = torch.cat(all_outputs, dim=0)
                diff = (output_tensor.float() - output.float()).abs()
                print(f"[HOOK] output diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
                return output_tensor

            except Exception as e:
                print(f"[HOOK] Error: {e}")
                import traceback
                traceback.print_exc()
                return None

        # Install hooks on .attn modules
        self._hooks = []
        matched = []

        for name, module in model.named_modules():
            if match_attn(name) is None:
                continue

            hook = module.register_forward_hook(
                lambda m, i, o, n=name: spotlight_attention_hook(m, i, o, module_name=n)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"[INSTALL] Installed {len(self._hooks)} Spotlight hooks: {matched[:3]}...")

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)
