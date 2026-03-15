# Metal Q/K Semantic Boundary Summary

Because they are probably not the same semantic boundary the original worker was using.

The original non-Metal worker does not hook:
- `q_proj`
- `k_proj`
- `v_proj`

It hooks:
- `self_attn.attn`

and then reads:
- `input[0]` as `q`
- `input[1]` as `k_all`

That means the analyzer was built around tensors that are already "attention-ready."

Why `q_proj` / `k_proj` may differ:
- they may be pre-reshape
- they may be pre-RoPE
- they may be pre-cache/layout transforms
- they may not reflect the exact grouped-query / KV repetition semantics seen at the actual attention call site
- they are just outputs of linear projections, not necessarily the final Q/K tensors consumed by attention

In the housing analogy:
- `q_proj` and `k_proj` are early transformation rooms
- `self_attn.attn` is the control room where the actual attention meeting happens
- the analyzer expects the documents handed to the control room, not just drafts produced in earlier rooms

The sandbox probe showed:
- the `q_proj` / `k_proj` outputs are meaningful
- they can reconstruct some plausible attention

But "plausible attention" is not enough if the analyzer was calibrated on a later, slightly different representation.

Short version:
- `q_proj` / `k_proj` are probably close, but not exact
- the missing differences are likely reshape, RoPE, KV grouping/repetition, or other internal attention prep
- that is why they can look real and still not match the original scoring path
