# mia

Capture and steer vLLM model internals, then analyze what was captured.

Two entry points:

- `MiaLLM` (`llm.py`): offline, wraps `vllm.LLM`.
- `MiaClient` (`client.py`): served, OpenAI-compatible client for `vllm serve`.

## Tree

```
mia/
  __init__.py  llm.py  client.py  artifacts.py  optimizations.py
  errors.py  registry.py  _profiler.py
  analyzers/    turn captured data into results
  workers/      vLLM worker extensions: capture and steer
  utils/        use-case helpers
  core/         capture and steering engine
    runner.py   the only module touching vLLM runner internals
    _plugin.py  vLLM plugin entry point (`mia.core._plugin:register`)
    hooks/      arm the hooks, bake the in-graph ops
    aperture/   fixed GPU capture aperture, drains, sinks
    delivery/   get a finished artifact to the caller
    runtime/    helper processes, device binding, CPU budget, TP geometry
```

## Top level

| Module | Role |
|---|---|
| `__init__.py` | public API and plugin registration |
| `llm.py` | `MiaLLM`: arms capture or steering, runs analyzers |
| `client.py` | `MiaClient`: probe capture and analysis against `vllm serve` |
| `artifacts.py` | read captured artifacts back: unpack, merge TP shards, load a run, dispatch a disk analyze |
| `optimizations.py` | the public optimization levers, set from env or a config file |
| `errors.py` | deliberate refusal error |
| `registry.py` | registry of worker and analyzer plugins by name |
| `_profiler.py` | process-local profiler |

## analyzers/

| Module | Role |
|---|---|
| `attention_tracker_analyzer.py`, `attnlink_analyzer.py`, `core_reranker_analyzer.py` | attention-based: prompt-injection detection, schema-column ranking, document relevance |
| `hidden_states_analyzer.py` | load captured hidden states, apply a reduction |
| `hnode_hallucination_analyzer.py`, `science_hallucination_analyzer.py` | hallucination detection with trained probes |

## workers/

| Module | Role |
|---|---|
| `hs_capture_worker.py`, `qk_capture_worker.py` | hidden-state and Q/K capture: eager hooks and the CUDA-graph aperture path |
| `steer_worker.py` | activation steering: eager hooks and the CUDA-graph buffer path |
| `_common.py` | stateless helpers shared by the capture workers |

## utils/

`utils/` holds helpers tied to one use case, not shared engine code. It currently wraps `hnode/`; new use-case helpers get their own subfolder here.

| Module | Role |
|---|---|
| `hnode/__init__.py`, `hnode/score.py` | H-Node hallucination probe: numpy-only scorer for a trained probe |

## core/

| Module | Role |
|---|---|
| `runner.py` | adapter isolating every vLLM V2 model-runner access |
| `_plugin.py` | vLLM plugin entry point: patches engine, runner and serve path; registered in `setup.py` as `mia.core._plugin:register` |

## core/hooks/

| Module | Role |
|---|---|
| `ops.py` | custom ops for CUDA-graph QK/HS capture and steering |
| `capture_triton.py`, `steer_triton.py` | Triton-fused kernels: `capture_hs` scatter, `steer_buffer` |
| `install.py`, `install_hs.py`, `install_steer.py` | CUDA-graph installs: QK capture, HS capture, buffer-mode steering |
| `hosts.py` | per-layer static-buffer hosts |
| `registry.py` | per-worker device routing slabs and host registry |
| `steer_routing_gpu.py` | GPU scatter of the steer and capture routing slabs |
| `drain.py` | worker-flush barrier for CUDA-graph capture |
| `run_mode.py` | env half of a run's mode: which worker, graph or eager |

## core/aperture/

| Module | Role |
|---|---|
| `capture_aperture.py` | fixed GPU aperture written in-graph at an advancing cursor, drained off-loop |
| `aperture_drain_hs.py`, `aperture_drain_qk.py`, `aperture_sink.py`, `aperture_reader.py` | host drains (HS, QK), raw-file write path, read-back from dump and sidecar |
| `aperture_metadata.py` | per-step sidecar mapping aperture rows to (req_id, layer, tokens) |
| `aperture_gather.py`, `aperture_run_index.py`, `aperture_trim.py` | hybrid gather into per-request artifacts, its row index, reclaiming gathered files |
| `aperture_sizing.py` | byte budgets and the safe `max_num_batched_tokens` cap |

## core/delivery/

| Module | Role |
|---|---|
| `delivery_selector.py`, `delivery_router.py`, `sizing.py` | pick the delivery path (hybrid default), the transport (RPC or disk), and the size prediction behind it |
| `per_request_delivery.py` | per-request demux, finish-tracking, assembly |
| `offload_process.py`, `writer_process.py`, `server_analyze_process.py` | background processes: ship files to the client, write off the engine GIL, run server-side reduce |
| `artifact_writer.py`, `run_artifact.py` | serialize and write artifacts; eager-format run artifacts |
| `artifact_quant.py` | on-GPU quantization of captured artifacts |
| `tensor_pack.py` | pack a tensor tree into one uint8 buffer plus manifest |
| `delivered_probes.py` | delivered HS and graph-mode Q/K data in eager shapes |
| `delivery_route.py` | API-server read route and `save_to_disk` writer |
| `disk_flush_probe.py` | coalesces the per-request `flush_disk` RPC under the aperture |

`load_delivered` lives in `core/aperture/aperture_gather.py`.

## core/runtime/

| Module | Role |
|---|---|
| `child_process.py` | start helper child processes, daemonic TP workers included |
| `thread_device.py`, `cpu_budget.py` | bind threads to their device; CPUs the process may use |
| `tp_shard.py` | TP capture geometry, rank dirs, shard merging |
| `census.py` | opt-in GPU-to-host offload cost attribution |

## Dependency direction

- `llm` / `client` use `core`; `core` uses `vllm`.
- `workers` and `analyzers` sit beside `core`: `core/hooks`, `core/aperture` and `core/delivery` import `mia.workers`, and `core/delivery` imports `mia.artifacts`.
- `core/runner.py` is the only module that touches vLLM's V2 runner internals.

## Import-time rule

- Every `__init__.py` under `core/` is docstring-only, because modules under `core/hooks/` read `MIA_*` env at import, and a bare `import mia` must not reach `core/hooks/`.
- `core/_plugin.py` is loaded lazily (`llm.py`) or by vLLM through the `vllm.general_plugins` entry point, never by `import mia`.
- The entry point is `mia.core._plugin:register`. After pulling this layout, re-run `pip install -e . --no-deps`: an environment still holding the old entry point cannot load the plugin.
