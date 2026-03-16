# Status Update: Granite Metal Attention Boundary

## What We Confirmed

- The original non-Metal worker is built around the `self_attn.attn` boundary, not `q_proj` / `k_proj`.
- On Granite MLX, `self_attn.attn` is not exposed as a child module.
- The installed Granite implementation computes attention inline via `scaled_dot_product_attention(...)`.
- Probing that boundary shows meaningful attention-ready tensors:
  - `queries`: `(1, 32, seq, 128)`
  - `keys`: `(1, 8, seq, 128)`
  - `values`: `(1, 8, seq, 128)`
- Reconstructed attention at that boundary is sharp and non-degenerate.

## What Changed

- The Metal worker now captures Q/K at the Granite attention boundary instead of at `q_proj` / `k_proj`.
- The Metal worker now prefers the longest observed sequence length per run/layer so it keeps the full prompt-side capture instead of short internal passes.
- Captured cache entries are written under the original-style key:
  - `model.layers.<i>.self_attn.attn`

## What The Logs Now Show

For single-request runs:
- warmup/internal pass appears at `seq_len=3`
- full prompt pass appears at `seq_len=39` or `seq_len=48`
- the worker keeps the longer capture
- the analyzer now sees `seq_len=39` / `48` instead of `3`

This means the main attention-boundary alignment problem is largely fixed.

## Current Remaining Problems

### 1. Zero score is still real

For the currently selected Granite head:
- attention is concentrated almost entirely on token position `0`
- the scored instruction/data slices begin at token `3`
- therefore the configured slices sum to zero

So the zero score is no longer explained by bad Q/K capture.
It is now explained by a mismatch between:
- the tracked Granite head/layer
- and the instruction/data token windows used by the analyzer

### 2. Batch run still drops one sample

For the 2-prompt batch run:
- worker flush reports only one stored sample:
  - `q: 1`
  - `k_all: 1`
- analyzer infers batch size `1`
- example crashes when trying to print `score[1]`

So the current batch bug is:
- the Metal attention-boundary path is only preserving one batch item
- not both requests

## Short Version

- attention-boundary capture on Granite Metal is working much better
- the worker now captures the full prompt-side attention-ready Q/K
- zero scores are now more likely a head/window issue than a hook-boundary issue
- batch capture is still incomplete and only returns one analyzed sample
