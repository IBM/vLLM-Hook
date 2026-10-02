# Tests

This directory contains model compatibility tests for the `mia` package.
The tests validate that hooks, workers, and analyzers work correctly with vLLM models.

## Layout

```
tests/
├── conftest.py     shared fixtures, the `gpu` marker and `requires_gpu`
├── use_cases/      the per-use-case model compatibility tests
└── mia/            tests for the CUDA-graph capture/steering internals
    ├── parity/     the T0-T3 correctness oracle (vanilla-vLLM reference, invariants, cross-branch)
    └── perf/       drain and host-build benches (tools, not gated tests)
```

`use_cases/` holds one test per use case and is where a new worker or analyzer belongs.
`mia/` holds the tests for the capture aperture, routing, delivery, TP sharding and the
naming/gate policies — the machinery behind the use cases rather than a use case itself.

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
pytest tests/ -q -m "not gpu"
```

Use `-m`, **never `-k "not gpu"`**. `-k` is a substring filter over test ids, so it has no
idea what a GPU test is and gets it wrong both ways: it lets the real-engine tests through
(they fail on a CPU-only node with `RuntimeError: Device string must not be empty`) and it
drops pure-CPU tests whose names merely contain "gpu", such as the GPU-*routing* band
checks in `test_parity_band_discrimination.py` and the `[MIA_T2_GPU_ROUTING_BAND]`
parametrization in `test_parity_band_index.py`. See the comment in `tests/conftest.py`.

Run only attention tracker tests:

```bash
pytest tests/use_cases/test_attntracker.py -vv
```

Run a single model:

```bash
pytest tests/use_cases/test_attntracker.py::test_attention_tracker[gpt2] -vv
```

---

## Common Failures

- **Installed 0 hooks**  
  Model architecture not matched or config contains no heads.
