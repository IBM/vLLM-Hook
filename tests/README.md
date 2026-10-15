# MIA's tests

Model compatibility tests for the `mia` package. They validate that hooks, workers and
analyzers work correctly with vLLM models.

## Layout

```
tests/
├── conftest.py        shared fixtures, markers, and `requires_gpu`
├── use_cases/         one GPU-backed test per use case
└── cpu_behavioral/    shared demo scenarios with automatic backend discovery
```

`use_cases/` is where a new worker or analyzer belongs — add a test there alongside your
demo.

`cpu_behavioral/` is for Apple Silicon vLLM CPU parity fixtures. These scripts also boot a
real vLLM engine, but they use the source-built vLLM CPU path instead of CUDA. Each fixture
must call `generate()` and map to a real example/use case, not just an internal hook probe.

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

## CPU Behavioral Fixtures

The CPU fixtures live in `tests/cpu_behavioral/`:

- `run_hidden_states_cpu.py`
- `run_attention_tracker_cpu.py`
- `run_activation_steer_cpu.py`
- `run_core_reranker_cpu.py`

They require the Apple Silicon vLLM CPU setup documented in
`tests/cpu_behavioral/docs/APPLE_SILICON_VLLM_CPU_INSTALL.md`.

Run one fixture directly:

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
OMP_NUM_THREADS=1 \
VLLM_CPU_OMP_THREADS_BIND=nobind \
TRITON_DEFAULT_BACKEND=cpu \
TRITON_CACHE_DIR=/private/tmp/triton-cache \
TRITON_DISABLE_OPENMP=1 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
MIA_WRITER_PROCESS=0 \
.venv-mac-cpu/bin/python tests/cpu_behavioral/run_hidden_states_cpu.py
```

Run the shared demo scenarios with automatic backend discovery (no opt-in flag):

```bash
python -m pytest tests/cpu_behavioral/test_vllm_cpu_fixtures.py -v -rs
```

Each `demo_scenario` (hidden states, Attention Tracker, activation steering, CoRe
reranking) is paired with CPU, CUDA, and Metal. Discovery checks the installed vLLM
platform, a device operation, and that scenario's MIA worker/analyzer. Compatible
cases run the original fixtures through real `generate()` calls in separate
processes. Unavailable cases skip with a reason; runtime errors and failed fixtures
fail. A candidate is only validated when its fixture passes.

The current MIA branch supports CPU and CUDA runners. Metal reports unavailable
because its MIA scenario adapter is not implemented; detecting Apple hardware alone
would not establish support. One vLLM installation selects one platform. To cover
CPU and CUDA, provision separate matching Python environments and run the same suite
in each. The suite does not install runtimes or switch a CUDA build into a CPU build.

CI can enforce coverage with `MIA_REQUIRED_BACKENDS=cpu` (or `cuda`, or a comma-separated
list). Every scenario for a required backend must execute successfully; unavailable
required cases fail. This sets coverage requirements, not which cases are discovered.

Optional settings:

- `MIA_BEHAVIOR_MODEL`: model ID or cached local model path; falls back to
  `MIA_CPU_MODEL`, then `facebook/opt-125m`. The steering fixture still assumes a
  768-dimensional model.
- `MIA_BEHAVIOR_TIMEOUT`: seconds per fixture (default 1800).

Pytest gives each case its own temporary hooks directory. Discovery and fixture logs
are stored in pytest's temporary directories; failures print the log path and tail.
The historical `run_*_cpu.py` paths remain usable directly on CPU. Their assertions,
configs, and prompts also supply the shared suite; no fixtures have been removed.
The steering fixture is a generation smoke test (zero coefficient), not proof of a
steering effect or numerical parity. CUDA and Metal parity require runs on those
backends. GitHub Actions provisioning is separate work.

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
- **CPU fixture cannot reach Hugging Face**: set `MIA_CPU_MODEL` to a local/cached model path
  or run once with network access so vLLM can resolve the model.
