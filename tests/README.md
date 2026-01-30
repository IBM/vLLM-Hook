# Tests

This directory contains model compatibility tests for the `vllm_hook_plugins` package.
The tests validate that hooks, workers, and analyzers work correctly with vLLM models.

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
