# Metal Q/K/V Hooking Status Update

This narrowed it down further.

## What We Now Know

- the Metal worker installs correctly
- the projection hook is firing
- the hooked projection output is visible
- at least one concrete shape is valid:
  - `model.layers.19.self_attn.v_proj`
  - shape `(1, 39, 1024)`

So the old problem is gone.

The new problem is deeper:

- the engine dies after the hook starts running
- but before we get a more specific Python exception out of the worker

## Most Likely Causes

1. too much hooking

You are wrapping many projection houses at once.

That is a lot of interception inside a fragile execution path.

2. wrong projection set for the first experiment

The original worker is a Q/K worker.

We are also wrapping `v_proj`.

For isolation, we should probably start with only `q_proj` and `k_proj`.

3. side effect during the critical path

File writes inside every hooked projection call may be too heavy for the first pass.

## Best Next Refinement

- reduce scope to `q_proj` and `k_proj` only
- hook only one layer first
- optionally collect in memory first and defer disk writes

Concretely, I would change the Metal worker to:

- stop matching `v_proj`
- temporarily hook only the first important layer, or even one fixed layer
- keep the output-shape logging

That gives a much cleaner experiment:

- "can I safely intercept Q/K projection outputs on Metal?"

instead of:

- "can I intercept everything everywhere at once?"

## Short Version

- we have proved the visitor reaches the recording room
- now the problem is stability of the full building under heavy instrumentation
- the right next step is to narrow the experiment, not broaden it

## Proposed Immediate Patch

If needed next, patch the worker to:

1. hook only `q_proj` and `k_proj`
2. hook only one layer initially
3. print exactly which layer is being instrumented
