# MIA's tests

Model compatibility tests for the `mia` package. They validate that hooks, workers and
analyzers work correctly with vLLM models.

## Layout

```
tests/
├── conftest.py     shared fixtures, the `gpu` marker and `requires_gpu`
└── use_cases/      one test per use case — these boot a real engine
```

`use_cases/` is where a new worker or analyzer belongs — add a test there alongside your
demo.

---
## Run Tests

Install pytest first (`pip install pytest`), and run from the project root.

Tests that boot a real engine carry the `gpu` marker:

```bash
pytest tests/use_cases -m gpu
```

- They boot one engine at a time (each test shuts its engine down) on small models: opt-125m,
  gpt2, Qwen2-1.5B, Phi-3-mini and Mistral-7B (gated: `hf auth login`), downloaded on first use.
- `pytest tests -q` also runs the AttnLink checks, which need no GPU. On a GPU in exclusive-process
  mode (common on clusters) no other process may hold the device, or the engine fails with
  `CUDA-capable device(s) is/are busy or unavailable`.
- They use `gpu_memory_utilization` 0.2–0.5 of the card, and write `hs_aperture_dump/` /
  `qk_aperture_dump/` in the working directory.
- Models without a shipped config get a random test config in pytest's temporary directory.

Run only the attention tracker tests, or one model:

```bash
pytest tests/use_cases/test_attntracker.py -m gpu -vv
pytest "tests/use_cases/test_attntracker.py::test_attention_tracker[gpt2]" -vv
```

---

## Common Failures

- **Nothing captured** (`probes` is `None`, analyzer output empty): the engine log prints
  `no decoder layers matched` / `no attention modules matched` when the model's module names are
  not ones MIA knows ([model compatibility](../docs/configs.md#model-compatibility)); in eager mode it
  prints `Installed 0 ... hooks`.
- **`CUDA-capable device(s) is/are busy or unavailable`**: another process holds the GPU; see
  above.
- **`Cannot find any model weights`**: the model was not downloaded (offline, or a gated model
  without `hf auth login`).
