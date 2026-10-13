# CPU Behavioral Testing

This folder contains CPU behavior tests and install notes for vLLM-Hook/MIA.

The test uses the real hidden-state worker:

```text
worker_name="capture_hs"
analyzer_name="hidden_states"
```

It calls `generate()`, captures hidden states, and checks the captured CPU tensors.
The other fixtures follow the same rule: each one boots vLLM CPU and calls
`generate()` for the corresponding example path.

## Current Status

Native Apple Silicon now has a passing hidden-state fixture for the MLR-50 CPU
parity path. The native setup is preserved in
`APPLE_SILICON_VLLM_CPU_INSTALL.md`.

## Requirements

For the native Apple Silicon path:

- macOS on Apple Silicon.
- Python 3.12.
- Homebrew `libomp`.
- Editable `triton-cpu`, vLLM `v0.29.0`, and this repo installed in
  `.venv-mac-cpu`.

## Run

For the native Apple Silicon fixture, follow the setup in
`APPLE_SILICON_VLLM_CPU_INSTALL.md`, then run:
Set `MIA_CPU_MODEL` if you want to use a different cached model; otherwise the
fixtures use `facebook/opt-125m`.

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

## Files

- `run_hidden_states_cpu.py`: the hidden-states generation test for CPU runners.
- `run_attention_tracker_cpu.py`: vLLM CPU generation with Q/K capture and the
  Attention Tracker analyzer.
- `run_activation_steer_cpu.py`: vLLM CPU generation with the activation
  steering worker.
- `run_core_reranker_cpu.py`: vLLM CPU Q/K capture runs for the CoRe reranker
  analyzer.
- `test_vllm_cpu_fixtures.py`: optional pytest wrapper that calls the fixture
  scripts when `MIA_RUN_VLLM_CPU_FIXTURES=1`.
- `APPLE_SILICON_VLLM_CPU_INSTALL.md`: native Apple Silicon vLLM CPU install and patch notes.
- `RESEARCH_NOTES.md`: native macOS findings and troubleshooting notes.

Run the optional pytest wrapper with:

```bash
MIA_RUN_VLLM_CPU_FIXTURES=1 \
.venv-mac-cpu/bin/python -m pytest \
  tests/cpu_behavioral/test_vllm_cpu_fixtures.py \
  -vv -s
```
