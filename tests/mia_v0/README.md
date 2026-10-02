# MIA's tests

Model compatibility tests for the `mia` package. They validate that hooks, workers and
analyzers work correctly with vLLM models.

## Layout

```
tests/mia_v0/
├── conftest.py     shared fixtures, the `gpu` marker and `requires_gpu`
├── use_cases/      one test per use case — these boot a real engine
├── test_plugin_config.py     engine-config policy: V2 runner, cudagraph mode, no PP
├── test_runner_adapter.py    the V2 model-runner adapter contract
├── test_graph_routing.py     per-step routing under CUDA graphs
└── test_tp_install.py        what each rank installs at tensor_parallel_size > 1
```

`use_cases/` is where a new worker or analyzer belongs — add a test there alongside your
demo. The four modules beside it are deliberately few: they cover the parts of the 0.29 /
V2 port that have no use case of their own, and that would fail silently rather than
loudly if they regressed.

This directory is MIA's. [`../use_cases/`](../use_cases/) and [`../conftest.py`](../conftest.py)
belong to the existing library and exercise `vllm_hook_plugins`; running those needs that
package installed (`pip install -e vllm_hook_plugins`), just as it always did.

These tests are **resource-aware** and do assume enough access to GPU resources. To reduce contention on shared systems:
- tests use low `gpu_memory_utilization` values
- only small or mid-sized models are enabled by default

If the GPU is heavily loaded, model initialization may fail. Current tests assume enough compute to host a 7B model and have `gpu_memory_utilization=0.2~0.5`.

---
## Run Tests
From the project root:

```bash
pytest -vv
```

### The hermetic gate (no GPU needed)

Tests that boot a real engine carry the `gpu` marker. To run everything else — the gate
CI and code review use — select on the **marker**:

```bash
pytest tests/mia_v0 -q -m "not gpu"      # 90 passed, 11 deselected
```

Use `-m`, **never `-k "not gpu"`**. `-k` is a substring filter over test ids, so it has no
idea what a GPU test is and gets it wrong both ways: it lets the real-engine tests through
(they fail on a CPU-only node with `RuntimeError: Device string must not be empty`) and it
drops pure-CPU tests whose names merely contain "gpu", such as the GPU-*routing* checks,
which need no GPU at all. See the comment in `tests/mia_v0/conftest.py`.

Run only attention tracker tests:

```bash
pytest tests/mia_v0/use_cases/test_attntracker.py -vv
```

Run a single model:

```bash
pytest tests/mia_v0/use_cases/test_attntracker.py::test_attention_tracker[gpt2] -vv
```

---

## Common Failures

- **Installed 0 hooks**  
  Model architecture not matched or config contains no heads.
