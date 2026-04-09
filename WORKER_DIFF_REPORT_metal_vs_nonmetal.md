# Worker Equivalence Audit: Metal vs Non-Metal Workers

Date: 2026-04-09

## Scope

---

- This report evaluates whether the Metal workers are intended to preserve the same observable behavior as the non-Metal workers.
- It is not a branch comparison.
- It is not a merge plan.
- It is an equivalence audit for these worker pairs:
  - [probe_hookqk_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py)
  - [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py)
  - [steer_activation_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py)
  - [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py)

## Verification Limits

---

- Statements below are based on direct inspection of the current worker source files.
- I did not run end-to-end worker executions in this turn.
- Metal conclusions are based on code paths and control flow, not live MLX tracing.

## Equivalence Goal

---

### Probe Workers

---

The Metal probe worker should preserve the same logical behavior as the non-Metal probe worker at the `HookLLM` level:

- target the same configured layers and heads
- respect the same hook enable/disable flow
- honor the same `hookq_mode` intent
- produce artifacts sufficient to drive the same analyzer semantics after normalization
- clean up temporary instrumentation after use

### Steering Workers

---

The Metal steering worker should preserve the same logical behavior as the non-Metal steering worker at the `HookLLM` level:

- target the same configured transformer layer
- honor the same steering config fields
- apply the same steering method intent
- preserve tuple/non-tuple output structure
- apply steering only during the intended generation window
- clean up temporary instrumentation after use

## At A Glance

---

| Concern | GPU Probe | Metal Probe | Equivalence Status |
|---|---|---|---|
| Target layer selection | Regex over several model naming schemes | Narrow MLX self-attn pattern plus capability checks | Partially equivalent |
| Hook activation | Flag file plus forward-hook timing | `_capture_active` plus flag file plus wrapper phase | Partially equivalent |
| `hookq_mode` intent | Directly stores query slices | Reconstructed query projection stored by mode | Equivalent in intent |
| Artifact contract | `qk.pt` legacy shape | `qkv.pt` richer shape | Equivalent after normalization |
| Cleanup | Remove hook handles | Restore wrapped modules | Equivalent |

| Concern | GPU Steer | Metal Steer | Equivalence Status |
|---|---|---|---|
| Target layer selection | One configured layer | Same configured layer | Equivalent |
| Steering methods | `add_vector`, `adjust_rs` | `add_vector`, `adjust_rs` | Intended equivalent |
| Output structure | Preserves tuple/non-tuple | Preserves tuple/non-tuple | Equivalent |
| Runtime gate | Flag file only | `_capture_active` plus flag file | Partially equivalent |
| Tensor backend | torch only | torch or MLX | Equivalent in intent |
| Cleanup | Remove hook handles | Restore wrapped modules | Equivalent |

## Shared Observable Contract

---

### Probe Workers

---

From the caller's perspective, the pair should satisfy this contract:

- the configured layer/head selection determines which attention modules are observed
- hook activation is controlled by the hook flag lifecycle
- `generate_with_encode_hook(...)` results in a persisted artifact for the active run
- analyzer code can derive attention scores from the resulting artifact set

### Steering Workers

---

From the caller's perspective, the pair should satisfy this contract:

- steering targets the configured layer
- steering applies only when hook execution is armed
- `add_vector` and `adjust_rs` express the same math at the residual-stream level
- the wrapped layer returns the same structural type it would have returned without steering

## Why Separate Implementations Exist

---

- The non-Metal workers rely on PyTorch `register_forward_hook`.
- The Metal workers operate on MLX-backed modules that do not expose the same hook interface.
- The Metal probe path cannot depend on the same directly surfaced Q/K tensors and therefore reconstructs projection data.
- The Metal steering path must handle both torch tensors and MLX arrays.

These are architectural differences, not a different product intent.

## Function Inventory

---

### `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py`

- `match_attn(name: str)`
- `_segment_bounds_from_metadata(metadata, module_name: str, device: torch.device)`

Class: `ProbeHookQKWorker`

- `load_model(self, *args, **kwargs)`
- `_install_hooks(self)`
- `_parse_layer_heads(self) -> Dict[int, List[int]]`
- `_uninstall_hooks(self)`
- `execute_model(self, *args, **kwargs)`

### `vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py`

- `match_attn(name: str)`

Class: `MLXHookWrapper`

- `__init__(self, module, name, hook_fn)`
- `__call__(self, *args, **kwargs)`

Class: `ProbeHookQKWorkerMetal`

- `_stage(self, message: str) -> None`
- `__init__(self, *args, **kwargs)`
- `init_device(self) -> None`
- `_init_device_single_process(self) -> None`
- `load_model(self, *args, **kwargs)`
- `_current_run_id(self) -> str | None`
- `_ensure_run_cache(self, run_id: str)`
- `_append_proj_tokens(...)`
- `_append_proj_token_list(...)`
- `_mx_offsets_to_int_list(self, value, batch_size: int) -> list[int]`
- `_flatten_attention_sample(self, sample) -> torch.Tensor`
- `_build_full_kv_without_mutation(self, cache, keys, values)`
- `_record_qkv(...)`
- `_capture_from_self_attn(...)`
- `_parse_layer_heads(self) -> Dict[int, List[int]]`
- `_install_hooks(self)`
- `_uninstall_hooks(self)`
- `_flush_run_cache(self) -> None`
- `execute_model(self, *args, **kwargs)`

### `vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py`

Class: `SteerHookActWorker`

- `load_model(self, *args, **kwargs)`
- `_install_hooks(self)`
- `_parse_steering_config(self) -> Dict`
- `_uninstall_hooks(self)`
- `execute_model(self, *args, **kwargs)`

### `vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py`

Class: `MLXSteeringWrapper`

- `__init__(self, module, name, hook_fn)`
- `__call__(self, *args, **kwargs)`

Class: `SteerHookActWorkerMetal`

- `__init__(self, *args, **kwargs)`
- `init_device(self) -> None`
- `_init_device_single_process(self) -> None`
- `load_model(self, *args, **kwargs)`
- `_install_hooks(self)`
- `_parse_steering_config(self) -> Dict`
- `_steering_enabled(self) -> bool`
- `_mlx_cast_like(self, value: mx.array, reference: mx.array) -> mx.array`
- `_apply_torch_steering(self, residuals: torch.Tensor) -> torch.Tensor`
- `_apply_mlx_steering(self, residuals: mx.array) -> mx.array`
- `_steering_hook(self, output, _module_name: str)`
- `_uninstall_hooks(self)`
- `execute_model(self, *args, **kwargs)`

## Equivalence Assessment

---

### Probe Workers

---

#### 1. Base Role

---

Non-Metal:

- `ProbeHookQKWorker` extends vLLM's GPU worker in [probe_hookqk_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py#L5).
- The relevant worker methods for this role are:
  - `load_model(...)`
  - `_install_hooks(...)`
  - `execute_model(...)`

Metal:

- `ProbeHookQKWorkerMetal` extends `MetalWorker` in [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L13) and adds its own device path in [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L123).
- The relevant worker methods for this role are:
  - `init_device(...)`
  - `_init_device_single_process(...)`
  - `load_model(...)`
  - `execute_model(...)`

Assessment:

- `Equivalent`

Reason:

- Same user-facing job, different runtime substrate.

#### 2. Layer Discovery

---

Non-Metal:

- Matches multiple naming conventions through regexes in [probe_hookqk_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py#L10).
- This behavior is implemented by:
  - `match_attn(...)`
  - `_install_hooks(...)`

Metal:

- Matches only `model.layers.<n>.self_attn` in [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L15).
- Requires MLX attention-module capabilities in [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L584).
- This behavior is implemented by:
  - `match_attn(...)`
  - `_install_hooks(...)`

Assessment:

- `Partially equivalent`

Reason:

- The Metal worker is equivalent only for architectures that match its narrower discovery assumptions.

#### 3. Hook Enable/Disable Semantics

---

Non-Metal:

- Forward hooks are present once installed; capture effectively depends on the flag file and normal forward execution in [probe_hookqk_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py#L208).
- This behavior is implemented by:
  - `_install_hooks(...)`
  - the inner `qkv_hook(...)`
  - `execute_model(...)`

Metal:

- Capture depends on `_capture_active`, the flag file, and wrapper phase in [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L227) and [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L629).
- This behavior is implemented by:
  - `MLXHookWrapper.__call__(...)`
  - `_current_run_id(...)`
  - `_install_hooks(...)`
  - `execute_model(...)`

Assessment:

- `Partially equivalent`

Reason:

- Both are intended to capture only during active runs, but the Metal path enforces that more explicitly.

#### 4. Captured Information

---

Non-Metal:

- Captures query and key data directly from the forward-hook inputs and stores `qk.pt` in [probe_hookqk_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py#L196).
- This behavior is implemented by:
  - `_segment_bounds_from_metadata(...)`
  - the inner `qkv_hook(...)`
  - `_install_hooks(...)`

Metal:

- Reconstructs projection data and stores `x`, `q`, `k`, and `v` in `qkv.pt` through [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L399) and [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L684).
- This behavior is implemented by:
  - `_capture_from_self_attn(...)`
  - `_build_full_kv_without_mutation(...)`
  - `_record_qkv(...)`
  - `_flush_run_cache(...)`

Assessment:

- `Equivalent after normalization`

Reason:

- The artifact shapes are different, but the Metal output is intended to preserve enough information to recover the same analyzer inputs.

#### 5. `hookq_mode` Behavior

---

Non-Metal:

- In `all_tokens`, stores all query slices; in `last_token`, stores the final query token in [probe_hookqk_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py#L240).
- This behavior is implemented by the inner `qkv_hook(...)` inside `_install_hooks(...)`.

Metal:

- In `all_tokens`, stores the full projected query tensor; in `last_token`, stores only the last query position in [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L430).
- This behavior is implemented by:
  - `_record_qkv(...)`
  - `_append_proj_tokens(...)`

Assessment:

- `Equivalent in intent`

Reason:

- The storage mechanism differs, but the mode meaning is aligned.

#### 6. Cleanup

---

Non-Metal:

- Removes registered hook handles in [probe_hookqk_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py#L303).
- This behavior is implemented by `_uninstall_hooks(...)`.

Metal:

- Restores original wrapped modules in [probe_hookqk_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L670).
- This behavior is implemented by `_uninstall_hooks(...)`.

Assessment:

- `Equivalent`

Reason:

- Same observable outcome: temporary instrumentation is removed.

### Steering Workers

---

#### 1. Base Role

---

Non-Metal:

- `SteerHookActWorker` extends vLLM's GPU worker in [steer_activation_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py#L5).
- The relevant worker methods for this role are:
  - `load_model(...)`
  - `_install_hooks(...)`
  - `execute_model(...)`

Metal:

- `SteerHookActWorkerMetal` extends `MetalWorker` in [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L12).
- The relevant worker methods for this role are:
  - `init_device(...)`
  - `_init_device_single_process(...)`
  - `load_model(...)`
  - `execute_model(...)`

Assessment:

- `Equivalent`

Reason:

- Same user-facing role with different backend assumptions.

#### 2. Target Layer Selection

---

Non-Metal:

- Targets `model.layers.{optimal_layer}` in [steer_activation_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py#L113).
- This behavior is implemented by:
  - `_parse_steering_config(...)`
  - `_install_hooks(...)`

Metal:

- Targets `model.layers.{optimal_layer}` via `TARGET_LAYER_TEMPLATE` in [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L14) and [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L187).
- This behavior is implemented by:
  - `_parse_steering_config(...)`
  - `_install_hooks(...)`

Assessment:

- `Equivalent`

#### 3. Steering Math

---

Non-Metal:

- Implements `add_vector` and `adjust_rs` inside the forward-hook closure in [steer_activation_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py#L86).
- This behavior is implemented by:
  - `_parse_steering_config(...)`
  - the inner `steering_hook(...)`

Metal:

- Implements the same two methods in `_apply_torch_steering(...)` and `_apply_mlx_steering(...)` in [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L281) and [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L309).
- This behavior is implemented by:
  - `_parse_steering_config(...)`
  - `_apply_torch_steering(...)`
  - `_apply_mlx_steering(...)`
  - `_steering_hook(...)`

Assessment:

- `Intended equivalent`

Reason:

- The method intent is the same, but the implementations diverge to support different tensor runtimes.

#### 4. Output Structure

---

Non-Metal:

- Preserves tuple versus non-tuple outputs in [steer_activation_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py#L77).
- This behavior is implemented by the inner `steering_hook(...)`.

Metal:

- Preserves tuple versus non-tuple outputs in [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L356).
- This behavior is implemented by `_steering_hook(...)`.

Assessment:

- `Equivalent`

#### 5. Runtime Gating

---

Non-Metal:

- Steering is gated by the hook flag file inside the hook closure in [steer_activation_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py#L75).
- This behavior is implemented by the inner `steering_hook(...)` and `execute_model(...)`.

Metal:

- Steering is gated by `_capture_active` plus the hook flag file in [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L250).
- `_capture_active` is only enabled during active execution in [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L387).
- This behavior is implemented by:
  - `_steering_enabled(...)`
  - `_steering_hook(...)`
  - `execute_model(...)`

Assessment:

- `Partially equivalent`

Reason:

- Same intended effect, stricter enforcement on Metal.

#### 6. Cleanup

---

Non-Metal:

- Removes hook handles in [steer_activation_worker.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py#L149).
- This behavior is implemented by `_uninstall_hooks(...)`.

Metal:

- Restores the original wrapped module in [steer_activation_worker_metal.py](/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py#L372).
- This behavior is implemented by `_uninstall_hooks(...)`.

Assessment:

- `Equivalent`

## Known Semantic Gaps

---

- Probe layer discovery is broader on the non-Metal worker than on the Metal worker.
- Probe capture timing is simpler on the non-Metal worker and more explicitly gated on the Metal worker.
- Probe artifacts are not identical and require normalization to be treated as equivalent by downstream analyzer code.
- Steering runtime gating is stricter on the Metal worker than on the non-Metal worker.
- Steering math is intended to match, but the Metal path must support two tensor representations rather than one.

## Practical Verdict

---

- The worker pairs are intended to be semantically equivalent at the `HookLLM` level.
- Probe equivalence currently depends on artifact normalization and on model architectures matching the Metal worker's narrower discovery assumptions.
- Steering equivalence is closer than probe equivalence; the main difference is stricter runtime gating on Metal, not a different steering objective.
- The main risk is semantic drift, not feature mismatch.

## Safe Assertions

---

- The Metal workers are not separate because they are meant to do different jobs.
- The Metal workers are separate because they need different interception and tensor-runtime mechanics to accomplish the same job.
- Probe equivalence is best described as `equivalent after normalization`.
- Steering equivalence is best described as `intended equivalent with stricter Metal runtime gating`.
