# Spotlight: Attention Steering for Instruction Following

**Paper**: [Venkateswaran and Contractor, EACL 2026](https://aclanthology.org/2026.eacl-long.174/)  
**Reference implementation**: [agent-lifecycle-toolkit SpotLightComponent](https://github.com/AgentToolkit/agent-lifecycle-toolkit/blob/main/altk/pre_llm/spotlight/spotlight.py)

Spotlight is an inference-time attention steering mechanism. It biases how a model attends to emphasized text spans (e.g. "Answer in JSON format") to improve instruction following, without any fine-tuning.

## Architecture Overview

Spotlight was implemented as a worker + utility function following the existing `ProbeHookQKWorker` + `HookLLM` pattern as the primary reference.

### Key files

| File | Role |
|---|---|
| `workers/spotlight_worker.py` | Worker that registers PyTorch forward hooks on every `.attn` module |
| `utils/spotlight_utils.py` | Tensor operations, span tokenization, attention bias computation, and `generate_with_spotlight()` convenience function |

### How it works

```
User calls generate_with_spotlight(llm, prompts, emph_strings, alpha)
│
├─ 1. generate_with_spotlight() tokenizes emph_strings → token span ranges
├─ 2. Writes {span_ranges, alpha} to spotlight_params.json (cross-process safe)
├─ 3. Creates EXTRACT.flag file to activate hooks
├─ 4. Calls llm.generate() — single pass with hooks active
│     │
│     └─ For each attention layer during prefill:
│          ├─ Hook reads Q, K, V from input_tuple (post-RoPE)
│          ├─ Reads span_ranges and alpha from params file
│          ├─ Accesses per-layer metadata via get_forward_context()
│          │   └─ metadata[layer_name].seq_lens, .query_start_loc
│          ├─ Upcasts to float32 (avoids fp16 overflow in Q @ K^T)
│          ├─ Recomputes attention: logits → causal mask → softmax → weights
│          ├─ Applies Spotlight bias in log-domain of weights
│          ├─ Computes biased output: modified_weights @ V
│          └─ Returns modified output (replaces fused kernel result)
│
├─ 5. Removes EXTRACT.flag and spotlight_params.json
└─ 6. Returns generation output
```

### Spotlight bias algorithm (from reference)

For each batch item, across all heads and query positions:

1. Compute attention weights: `softmax(logits)`
2. Compute `current_proportion` = fraction of attention mass on the emphasized span
3. If `current_proportion < target_proportion` (alpha):
   - `bias = log(alpha / current_proportion)`
   - Convert weights to log-domain: `log(weights + 1e-10)`
   - Add bias to span positions
   - Re-normalize with softmax
4. Recompute attention output with biased weights

The bias only steers attention **upward** (increases attention on the span), matching the reference implementation.

### Differences from the reference implementation

The reference hooks on HuggingFace's `self_attn` module with `output_attentions=True`, giving direct access to post-softmax attention weights. vLLM does not expose attention weights, so the vLLM implementation:

- Hooks on the `.attn` submodule (the `Attention` class that wraps the fused kernel)
- Recomputes attention from Q/K/V inputs in float32 with a causal mask
- Returns the modified output tensor from the post-forward hook, which replaces the fused kernel's output in the residual stream

### Cross-process parameter passing

vLLM V1 spawns worker processes via `multiprocessing.spawn`. The main process and worker share no memory. Following the pattern established by `ProbeHookQKWorker` (which uses `EXTRACT.flag` and `RUN_ID.txt` files), Spotlight passes parameters via the filesystem:

- **Main process** (`generate_with_spotlight()`): writes `spotlight_params.json` to `VLLM_HOOK_DIR`
- **Worker process** (`SpotlightWorker`): reads the file when the hook fires
- **Flag file** (`EXTRACT.flag`): controls whether hooks are active

### Reference: ProbeHookQKWorker

The Spotlight worker was built using `ProbeHookQKWorker` as the architectural reference for:

- Hook registration pattern: `register_forward_hook` on `.attn` modules matched by regex
- Metadata access: `get_forward_context().attn_metadata[layer_name].seq_lens`
- Batch splitting: `query_start_loc` to slice flat `[total_tokens, hidden]` tensors per request
- Worker lifecycle: `load_model()` → `_install_hooks()` with env var configuration

## Quick Start

```python
from vllm_hook_plugins import HookLLM, generate_with_spotlight, register_plugins
register_plugins()

llm = HookLLM(
    model="Qwen/Qwen2-1.5B-Instruct",
    worker_name="probe_spotlight",
    enforce_eager=True,  # Required — disables fused attention
)

# Baseline (no steering)
baseline = llm.generate(
    prompts=["Answer in JSON. What is 2+2?"],
    use_hook=False,
    max_tokens=50,
)

# With Spotlight
spotlight = generate_with_spotlight(
    llm,
    prompts=["Answer in JSON. What is 2+2?"],
    emph_strings=["Answer in JSON"],
    alpha=0.95,
    max_tokens=50,
)
```

> **Note**: `enforce_eager=True` is required. Flash Attention fuses Q/K/V into a single kernel, preventing hook access to the intermediate tensors needed for attention steering.
