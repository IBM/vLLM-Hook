# Housing Analogy For The Current Metal Worker

This note describes the current Metal worker, not the older projection-house sketch.

The current worker wraps each `model.layers.<i>.self_attn` house, captures the raw visitor packet `x` at that door, and then computes the Q/K/V projection packets inside the recording room before the original house continues its normal work.

## Core Mapping

Use this vocabulary consistently:

- `address`
  - a module path like `model.layers.15.self_attn`
- `house`
  - the module object living at an address
- `outer house`
  - the wrapper we install at that address
- `inner house`
  - the original module stored inside the wrapper
- `occupants`
  - the static contents of a house
  - weights, child modules, rope logic, projection modules
- `visitors`
  - live tensors moving through the house during execution
- `door`
  - the call boundary where `module(...)` happens
- `recording room`
  - the hook logic that writes notes about visitors
- `notebook`
  - the in-memory run cache
- `archive room`
  - the on-disk `qkv.pt` artifact

## What House Is Wrapped

In the current Metal worker, the wrapped house is:

- `model.layers.<i>.self_attn`

That means:

- before wrapping:
  - the address points directly to the original `Attention` house
- after wrapping:
  - the address points to an outer wrapper house
  - the original `Attention` house still exists inside the wrapper as `self.module`

This is different from the earlier sketch that tried to stand at a deeper inner door.

## What Lives Inside The `self_attn` House

The `self_attn` house contains these important occupants:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `rope`
- metadata like `n_heads` and `n_kv_heads`

These are occupants.
They are the permanent furniture and residents of the house.

The live visitors are different:

- `x`
- projected `queries`
- projected `keys`
- projected `values`

## What The Wrapper Does

The wrapper is a new outer house placed at the `self_attn` address.

When visitors arrive:

1. they enter the wrapper house first
2. the wrapper sends a copy of the visit details into the recording room
3. the wrapper then forwards the real visitors into the original inner house

So the wrapper does not replace the real attention computation.
It only adds an observation point at the house door.

## Why The Current Worker Uses `self_attn`

In the installed Metal Granite runtime, there is no separate `self_attn.attn` house to wrap.

So the closest reliable door we can stand at from outside the model is:

- `self_attn.__call__`

That means the recording room sees:

- the raw hidden-state visitor `x`
- the optional `mask`
- the optional `cache`

Then the recording room computes the same internal projection routes the house itself uses:

- `q_proj(x)`
- `k_proj(x)`
- `v_proj(x)`
- reshape
- transpose
- rope

This gives us the projected packets we need without requiring a deeper house that does not exist in the runtime.

## Visitor Story

The easiest way to think about the visitor flow is this:

- one raw visitor packet `x` arrives at the `self_attn` house
- inside the recording room, we make three specialized copies of that visitor
  - one copy becomes Q
  - one copy becomes K
  - one copy becomes V
- the real original house then receives the same visitor and does its own actual work

So the recording room is not stealing the visitor.
It is writing parallel notes based on the same incoming packet.

## What The Recording Room Captures

For each tracked layer, the recording room writes down four packet types:

- `x`
  - the raw visitor as it enters `self_attn`
- `q`
  - the visitor after `q_proj`, reshape, transpose, and rope
- `k`
  - the visitor after `k_proj`, reshape, transpose, and rope
- `v`
  - the visitor after `v_proj`, reshape, and transpose

These notes are stored under addresses like:

- `model.layers.15.self_attn.attn.x`
- `model.layers.15.self_attn.attn.q`
- `model.layers.15.self_attn.attn.k`
- `model.layers.15.self_attn.attn.v`

The `.attn` segment in the saved notebook name is just a label for consistency with the analyzer.
It does not mean there is a real `self_attn.attn` house in the runtime.

## Why Rope Is Applied In The Recording Room

The real house applies rope before attention.

If the recording room only stored the raw projection outputs before rope, its notes would not match the packets that actually matter at the attention meeting.

So the current worker applies the same rope step to:

- `queries`
- `keys`

When cache exists:

- rope uses `cache.offset`

When cache does not exist:

- rope uses the default no-offset form

That means the notebook stores the same attention-ready directional versions of Q and K that the house is about to use.

## Why `v` Is Different

The `values` packet goes through:

- `v_proj`
- reshape
- transpose

But not rope.

That mirrors the actual house behavior.

So in the analogy:

- Q and K are visitors who must also pass through a position-checkpoint desk
- V is a visitor who does not go through that same desk

## What The Notebook Looks Like

The notebook for one run is `self._run_cache`.

Inside it:

- `config`
  - building-wide structural facts
  - head counts, widths, dimensions
- `qkv_cache`
  - the recorded packets
- `meta`
  - notebook-level labels like tensor-parallel rank and capture boundary

Each recorded entry stores:

- `layer_num`
- `proj_kind`
- `tokens`

And each batch item is stored as a separate packet in `tokens`.

So in housing terms:

- one run gets one notebook
- one layer gets a tab
- one projection kind gets a subsection
- one request in the batch gets its own filed packet

## Batch Handling Analogy

Suppose two request groups enter the building together.

The recording room should not throw them into one envelope.
It should file:

- packet 0 for request 0
- packet 1 for request 1

That is why the worker stores one notebook entry per sample in the batch.

This is the same idea behind:

- preserving both prompt packets
- making batch size visible to the analyzer

## `all_tokens` vs `last_token`

The worker supports two filing modes:

- `all_tokens`
  - store the full packet of Q tokens for each request
- `last_token`
  - store only the final Q token packet for each request

K and V are still stored as full packet histories for each request.

In housing terms:

- `all_tokens` means file every page in the packet
- `last_token` means keep only the final page from the Q packet

## Building Directory Analogy

`named_modules()` is the building directory.

It is not a door.

It tells us:

- which addresses exist
- which houses live at those addresses

The worker uses that directory to:

1. find every `*.self_attn` address
2. find which of those addresses belong to tracked layers
3. swap the original house for a wrapper outer house

So:

- `named_modules()` helps us find the house
- the wrapper `__call__` is still the actual door where visitors arrive

## Why Repeated Hooked Runs Now Work

Repeated hooked runs now work because teardown restores the original houses.

When hook teardown runs, the worker walks back through the installed wrappers and restores the original inner houses at their public addresses.

In housing terms:

- the temporary outer recording houses are removed
- the original houses are moved back to the front of the address book

That prevents the next hooked engine in the same Python process from inheriting stale wrapper houses by accident.

The important lifecycle detail is that this teardown does not happen just because the recording room closes.

Instead:

- `HookLLM` creates a temporary hooked engine for the capture run
- when that temporary engine is being disposed, `HookLLM` explicitly asks the worker to run `_uninstall_hooks()`
- only then are the original houses restored before engine shutdown

## The Real House Still Works

The wrapper does not change model semantics on purpose.

The intended sequence is:

1. wrapper sees the incoming visit
2. recording room computes note copies
3. wrapper calls the original inner house
4. model continues unchanged

So the notebook stores copies for inspection.
The real visitors still go through the real house and continue to generation normally.

After teardown, the temporary outer houses are removed and the original houses become public-facing again.

## Why This Is Closer To The Current Runtime Than `scaled_dot_product_attention`

The old fallback stood deeper inside the building at the attention desk itself.

That worked as a fallback, but it had drawbacks:

- it depended on a global function patch
- it had to infer which layer the visit belonged to
- it did not naturally expose the raw incoming visitor `x`

The current `self_attn` wrapper is cleaner for this runtime because:

- it is per-layer rather than global
- it sees the raw visitor packet
- it computes Q/K/V in a way that matches the installed Granite house logic

## What Is And Is Not Essential

Essential parts of the current design:

- wrapper outer house at `self_attn`
- capture gate using the hook flag
- run-id notebook separation
- computation of `q`, `k`, `v` from raw `x`
- rope on `q` and `k`
- batch-preserving packet storage
- archive write to `qkv.pt`

Non-essential or supporting parts:

- extra debug prints
- debug-only stage logging
- explicit restore-on-teardown logic for repeated runs

## One-Sentence Summary

The current Metal worker stands at the `self_attn` house door, records the raw visitor packet `x`, computes the same Q/K/V projection packets that the house is about to use, files those packets into a per-run notebook, and then lets the real visitor continue through the original attention house unchanged.
