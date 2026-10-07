# mia

Capture and steer vLLM model internals, then analyze what was captured.

## Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Package Reference](#package-reference)
- [Dependency Direction](#dependency-direction)

## Overview

`mia` exposes two entry points:

| Entry point | Defined in | Mode | Description |
|---|---|---|---|
| `MiaLLM` | `llm.py` | Offline | Wraps `vllm.LLM`; arms capture or steering and runs analyzers. |
| `MiaClient` | `client.py` | Served | OpenAI-compatible client for `vllm serve`; probe capture and analysis. |

## Directory Structure

```text
mia/
├── __init__.py            # Public API and plugin registration
├── llm.py                 # MiaLLM (offline)
├── client.py              # MiaClient (served)
├── artifacts.py           # Read captured artifacts back
├── optimizations.py       # Public optimization levers
├── errors.py              # Deliberate refusal error
├── registry.py            # Worker / analyzer plugin registry
├── _profiler.py           # Process-local profiler
│
├── analyzers/             # Turn captured data into results
├── workers/               # vLLM worker extensions: capture and steer
├── utils/                 # Use-case-specific helpers
│   └── hnode/
└── core/                  # Capture and steering engine
    ├── runner.py          # Only module touching vLLM runner internals
    ├── _plugin.py         # vLLM plugin entry point
    ├── hooks/             # Arm the hooks, bake the in-graph ops
    ├── aperture/          # Fixed GPU capture aperture, drains, sinks
    ├── delivery/          # Get a finished artifact to the caller
    └── runtime/           # Helper processes, device binding, CPU budget, TP geometry
```

## Package Reference

### Top-level modules

| Module | Role |
|---|---|
| `__init__.py` | Public API and plugin registration. |
| `llm.py` | `MiaLLM`: arms capture or steering, runs analyzers. |
| `client.py` | `MiaClient`: probe capture and analysis against `vllm serve`. |
| `artifacts.py` | Reads captured artifacts back: unpack, merge TP shards, load a run, dispatch a disk analyze. |
| `optimizations.py` | Public optimization levers, set from env or a config file. |
| `errors.py` | Deliberate refusal error. |
| `registry.py` | Registry of worker and analyzer plugins by name. |
| `_profiler.py` | Process-local profiler. |

### `analyzers/`

| Module | Role |
|---|---|
| `attention_tracker_analyzer.py` | Attention Tracker: prompt-injection detection from captured attention. |
| `attnlink_analyzer.py` | AttnLink-U: ranks schema columns using one generation-anchor attention head. |
| `core_reranker_analyzer.py` | CoRe reranker: document relevance from captured Q/K attention. |
| `hidden_states_analyzer.py` | Loads captured hidden states and applies a reduction. |
| `hnode_hallucination_analyzer.py` | Hallucination detection with the H-Node probe. |
| `science_hallucination_analyzer.py` | Classifies captured hidden states with a trained probe to detect hallucination. |

### `workers/`

| Module | Role |
|---|---|
| `hs_capture_worker.py` | Hidden-state capture (`capture_hs`): eager hooks and the CUDA-graph aperture path. |
| `qk_capture_worker.py` | Q/K capture (`capture_qk`): eager hooks and the CUDA-graph aperture path. |
| `steer_worker.py` | Activation steering: eager hooks and the CUDA-graph buffer path. |
| `_common.py` | Stateless helpers shared by the capture workers. |

### `utils/`

Helpers tied to a single use case, not shared engine code. New use-case helpers get their own subfolder.

| Module | Role |
|---|---|
| `hnode/__init__.py` | H-Node hallucination detection: inference side for MIA. |
| `hnode/score.py` | Numpy-only scorer for a trained H-Node probe. |

### `core/`

The engine. Its own modules are listed first, followed by its four subpackages.

| Module | Role |
|---|---|
| `runner.py` | Adapter isolating every vLLM V2 model-runner access. |
| `_plugin.py` | vLLM plugin entry point: patches engine, runner and serve path. Registered in `setup.py` as `mia.core._plugin:register`. |

#### `core/hooks/`

| Module | Role |
|---|---|
| `ops.py` | Custom ops for CUDA-graph QK/HS capture and steering. |
| `capture_triton.py` | Triton-fused capture scatter: one kernel for `capture_hs`. |
| `steer_triton.py` | Triton-fused `steer_buffer` op: the FULL-mode buffer-steering kernel. |
| `install.py` | CUDA-graph QK capture: install, per-step routing and egress. |
| `install_hs.py` | CUDA-graph hidden-state capture install on the capture-aperture path. |
| `install_steer.py` | CUDA-graph activation-steering install (buffer mode). |
| `hosts.py` | Per-layer static-buffer hosts. |
| `registry.py` | Per-worker device routing slabs and host registry. |
| `steer_routing_gpu.py` | GPU scatter of the steer and capture routing slabs. |
| `drain.py` | Worker-flush barrier for CUDA-graph capture. |
| `run_mode.py` | Env half of a run's mode: which worker, graph or eager. |

#### `core/aperture/`

| Module | Role |
|---|---|
| `capture_aperture.py` | Fixed GPU aperture written in-graph at an advancing cursor, drained off-loop. |
| `aperture_drain_hs.py` | Multi-layer host drain for the HS capture aperture. |
| `aperture_drain_qk.py` | Multi-layer host drain for the QK capture aperture. |
| `aperture_sink.py` | Raw-file write path for the drains: persistent sinks, O_DIRECT, writer threads. |
| `aperture_reader.py` | Rebuilds each request's per-layer tensors from an aperture raw dump and its sidecar. |
| `aperture_metadata.py` | Per-step sidecar mapping aperture rows to (req_id, layer, tokens). |
| `aperture_gather.py` | Hybrid delivery gather: scatters the shared layer files into per-request artifacts. `load_delivered` lives here. |
| `aperture_run_index.py` | Run-encoded per-request row index into the shared HS layer files (`MIA_APERTURE_GATHER`). |
| `aperture_trim.py` | Reclaims the shared layer files behind the gather cursor (`MIA_APERTURE_GATHER_TRIM`). |
| `aperture_sizing.py` | Byte budgets and the safe `max_num_batched_tokens` cap. |

#### `core/delivery/`

| Module | Role |
|---|---|
| `delivery_selector.py` | Chooses which per-request delivery path a run gets; hybrid is the default. |
| `delivery_router.py` | Per-request delivery router: picks the transport (RPC or disk) and where to analyze. |
| `sizing.py` | Artifact-size prediction and the RPC-vs-disk routing decision. |
| `per_request_delivery.py` | Per-request demux, finish-tracking, assembly. |
| `offload_process.py` | Background worker that ships a finished request's file to the client destination. |
| `writer_process.py` | Separate process that serializes and writes artifacts off the engine GIL. |
| `server_analyze_process.py` | CPU process that runs a reducible analyzer's server-side reduce off the GPU path. |
| `artifact_writer.py` | Pure serialize-and-write functions for captured artifacts. |
| `run_artifact.py` | Eager-format run artifacts written from delivered data. |
| `artifact_quant.py` | On-GPU quantization of captured artifacts. |
| `tensor_pack.py` | Pack a tensor tree into one uint8 buffer plus manifest. |
| `delivered_probes.py` | Delivered HS and graph-mode Q/K data in eager shapes. |
| `delivery_route.py` | API-server read route and `save_to_disk` writer. |
| `disk_flush_probe.py` | Coalesces the per-request `flush_disk` RPC under the aperture. |

#### `core/runtime/`

| Module | Role |
|---|---|
| `child_process.py` | Start helper child processes, daemonic TP workers included. |
| `thread_device.py` | Binds every MIA thread that can reach CUDA to its device first. |
| `cpu_budget.py` | How many CPUs this process may actually use. |
| `tp_shard.py` | TP capture geometry, rank dirs, shard merging. |
| `census.py` | Opt-in GPU-to-host offload cost attribution. |

## Dependency Direction

- `llm` and `client` use `core`; `core` uses `vllm`.
- `workers` and `analyzers` sit beside `core`: `core/hooks`, `core/aperture` and `core/delivery` import `mia.workers`, and `core/delivery` imports `mia.artifacts`.
