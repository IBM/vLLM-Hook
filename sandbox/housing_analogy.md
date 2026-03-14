# Housing Analogy For Hooking

## Core Mapping

Use this mapping consistently:

- `address`
  - a module path like `model.layers.22.self_attn.q_proj`
- `house`
  - a module object living at an address
- `occupants`
  - the static contents of a house
  - weights, parameters, child modules, internal logic
- `visitors`
  - live tensors passing through a house during a forward call
- `door`
  - the call boundary of a module
  - in code terms: the point where `module(...)` happens
- `doorbell`
  - the mechanism that alerts you when visitors arrive
  - in PyTorch this is `register_forward_hook`
  - in the MLX wrapper approach, the wrapper provides the observable entrypoint
- `recording room`
  - the hook function itself
  - here, `qkv_hook`

## What A Layer Is

A layer is the house.

The path to the layer is the address.

Example:

- address: `model.layers.22.self_attn`
- house: the `Attention` module object living there

Before hooking:

- the address points directly to the original house

After hooking:

- the address points to a new outer house, the wrapper
- the original house still exists, but it is stored inside the wrapper as the inner house

## Outer House And Inner House

When a wrapper is installed:

- if you are talking about the exact wrapped address itself:
  - wrapper = outer house
  - original module = inner house

- if you are talking relative to a larger containing house like `self_attn`:
  - `self_attn` = outer house
  - wrapper at `q_proj` / `k_proj` / `v_proj` = inner house
  - original projection module = inner-inner house

That means:

- before wrapping:
  - `self_attn` is the outer house
  - `q_proj` is an inner house inside `self_attn`
- after wrapping:
  - `self_attn` is still the outer house
  - `model.layers.22.self_attn.q_proj` points to the wrapper inner house
  - the original `q_proj` house lives inside that wrapper as the inner-inner house `wrapped.module`

So the original module is not destroyed.
It is no longer the public-facing house at that address.

## What Lives In A House

The occupants of a house are the static things that belong to it:

- parameters
- weights
- child modules
- internal logic

For an `MLP` house, the occupants include:

- `up_proj`
- `down_proj`
- `gate_proj`

For a `self_attn` house, the occupants include:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `rope`

These occupants are not the same as the live tensors flowing through the model during execution.

## What Visitors Are

Visitors are the live tensors moving through the model during a forward pass.

This is the key distinction:

- occupants = static things already living in the house
- visitors = dynamic things entering and leaving during execution

For Q/K/V capture, we care about visitors, not occupants.

## PyTorch Version Of The Analogy

In the PyTorch worker:

- the house is an attention module selected from `model.named_modules()`
- `register_forward_hook` is like attaching a doorbell to that house
- when visitors arrive, the doorbell rings and PyTorch gives the callback access to:
  - the module
  - the input tuple
  - the output

So in the PyTorch version:

- doorbell = `register_forward_hook`
- recording room = `qkv_hook`
- visitors = the live input tuple `i` passed to the hook

The important thing is not the hook handle object itself.
The important thing is that the hook callback sees the live call data.

## MLX Wrapper Version Of The Analogy

In the MLX wrapper approach:

- there is no native doorbell attached to the existing house
- instead, we build a new outer house at the same address
- visitors reach the wrapper house first
- the wrapper sends visitors into the recording room
- then the wrapper forwards visitors into the original inner house

So:

- PyTorch = attach a doorbell to the existing house
- MLX wrapper = replace the house at that address with a new outer house

Same purpose:

- observe live visitors

Different mechanism:

- PyTorch uses a hook API
- MLX wrapper uses object replacement

## MLP House vs Self-Attention House

Wrapping `mlp` is useful as a proof that wrapper-based interception works.

But `mlp` is the wrong house if the goal is Q/K/V capture.

Why:

- `mlp` contains `up_proj`, `down_proj`, `gate_proj`
- those are feed-forward houses, not the Q/K/V-producing houses

The better house is `self_attn`.

Because `self_attn` contains:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `rope`

That makes `self_attn` the correct main house to inspect for Q/K/V work.

## Inner Houses Inside `self_attn`

Inside the `self_attn` house:

- `q_proj` is an inner house that produces Q
- `k_proj` is an inner house that produces K
- `v_proj` is an inner house that produces V
- `o_proj` is an inner house that projects the combined attention result back out

These are separate houses because they are callable modules, not just passive fields.

## Bilocating Visitor Analogy

An incoming hidden-state visitor does not walk through `q_proj`, then `k_proj`, then `v_proj` in a simple single-file line.

Better intuition:

- one incoming visitor is split into three related versions
- one version goes through `q_proj`
- one version goes through `k_proj`
- one version goes through `v_proj`

So it is like one visitor bilocating into three specialized copies.

After that:

- Q, K, and V are used together in attention
- the combined result later goes through `o_proj`

## Where The Recording Room Must Be

If the goal is to capture Q/K/V, the recording room must sit after `q_proj`, `k_proj`, and `v_proj` have produced those tensors.

Too early:

- recording the visitor before it enters `q_proj`
- that only captures the pre-projection hidden state

Correct place:

- after the visitor leaves `q_proj`, `k_proj`, or `v_proj`
- at that point the visitor has become Q, K, or V

Too late:

- after the attention result has already been merged and projected through `o_proj`

So the recording room must be placed:

- after Q/K/V exist
- before everything has collapsed into the final attention output

## Why `q_proj` / `k_proj` / `v_proj` Are Better Hook Targets

If you wrap the outer `self_attn` house, you may only see the visitor before Q/K/V are created.

If you wrap:

- `q_proj`
- `k_proj`
- `v_proj`

then each wrapped projection address becomes:

- `self_attn` = outer house
- wrapper = inner house
- original projection module = inner-inner house

and the wrapper can record the projection output itself.

That means the recording room sees:

- a Q visitor leaving `q_proj`
- a K visitor leaving `k_proj`
- a V visitor leaving `v_proj`

This is why the Metal worker sketch was refined to target projection houses directly.

## What The `qkv_hook` Recording Room Does

Inside the Metal worker sketch:

- the wrapper inner house sends the output of the original `q_proj`, `k_proj`, or `v_proj` inner-inner house into `qkv_hook`
- inside `qkv_hook`, the recording room labels that visitor with:
  - layer number
  - projection kind (`q`, `k`, or `v`)
  - request/run id
  - token grouping information
- then the recording room saves a copy to disk

Important:

- the recorded visitor is a copy for notes
- the real visitor continues on unchanged through the forward path

So the recording room does not stop the model from functioning.
It writes down what passed through and lets the computation continue.

## Step-By-Step `qkv_hook` Logic

This is what happens inside the recording room in plain terms.

### 1. Check whether the recording room is open

In code, `qkv_hook` first checks the hook flag file.

Analogy:

- if the recording room door is closed, the visitor just walks past and nothing is written down

### 2. Find the notebook for the current run

The function reads the run id file.

Analogy:

- each experiment run gets its own notebook
- visitors from different days should not be mixed into the same notes

### 3. Ask the building office how the visitor stream is grouped

The function reads `attn_metadata` and `seq_lens`.

Analogy:

- the building office tells the recording room which visitors belong to which party
- this matters because multiple requests may be combined into one stream

### 4. Confirm which inner house produced the visitor

The function matches the module name against:

- `q_proj`
- `k_proj`
- `v_proj`

Analogy:

- the recorder checks whether the visitor just came out of the Q house, the K house, or the V house

### 5. Convert the visitor into a format suitable for note-taking

The function converts the output tensor into a torch CPU value before saving.

Analogy:

- the recorder makes a paper copy before filing the note
- the original visitor is not detained

### 6. Split the combined visitor stream into request-sized chunks

Using `seq_lens`, the function computes boundaries and slices the tensor.

Analogy:

- if several groups entered the building together, the recorder separates the hallway footage by group before filing it

### 7. Put the visitor in the right notebook

The cache is organized by:

- run id
- module name
- projection kind (`q`, `k`, or `v`)
- layer number

Analogy:

- there is not one giant notebook
- there is a shelf of labeled binders:
  - run
  - layer
  - which house the visitor came from

### 8. Decide whether to record all visitors or only the last one

The code supports:

- `all_tokens`
- `last_token`

Analogy:

- either write down every visitor in the group
- or only write down the last visitor from each group

### 9. Save the notes to disk

The function writes the cache to `qkv.pt`.

Analogy:

- after the recorder writes down the visitor details, the notebook is filed in the archive room

### 10. Let the real visitor continue

The wrapper returns the original output unchanged.

Analogy:

- the recorded visitor in the notebook is only a copy
- the real visitor keeps walking into the rest of attention and eventually contributes to the output

## Directory Tools vs Doors

Methods like these are not doors:

- `named_modules`
- `leaf_modules`
- `apply_to_modules`
- `update_modules`

They are directory or management tools.

They help you:

- find houses
- inspect houses
- update houses

But they do not themselves expose the live visitors.

The door is the call boundary:

- `module(...)`

That is where visitors enter.

## What Was Missing In The Earlier Metal Sketch

Two different problems were mixed together:

1. explicit problem

- the wrapper only passed one visible argument into the recording room
- that was too narrow

2. deeper problem

- it was not clear whether the chosen house boundary actually exposed Q/K/V

So the deeper question was not:

- which attribute stores Q/K/V

It was:

- at which house boundary do Q/K/V become visible as visitors

The refined answer is:

- likely at `q_proj`, `k_proj`, and `v_proj`
- not at `mlp`
- and not necessarily at the outer `self_attn` boundary

## One-Sentence Summary

The wrapper approach works by placing a wrapper house at the `q_proj` / `k_proj` / `v_proj` addresses inside the `self_attn` outer house, and Q/K/V capture only works if the recording room sees visitors after those projection houses have produced Q, K, and V.
