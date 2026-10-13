# CPU Behavioral Testing

This folder contains CPU behavior tests and install notes for vLLM-Hook/MIA.

The test uses the real hidden-state worker:

```text
worker_name="capture_hs"
analyzer_name="hidden_states"
```

It calls `generate()`, captures hidden states, and checks the captured CPU tensors.

## Current Status

Native Apple Silicon now has a passing hidden-state fixture for the MLR-50 CPU
parity path. The native setup is preserved in
`APPLE_SILICON_VLLM_CPU_INSTALL.md`.

Docker remains useful as a separate Linux ARM64 path, but it is no longer the
only path that reaches generation and capture.

## Requirements

For the native Apple Silicon path:

- macOS on Apple Silicon.
- Python 3.12.
- Homebrew `libomp`.
- Editable `triton-cpu`, vLLM `v0.29.0`, and this repo installed in
  `.venv-mac-cpu`.

For the Docker Linux ARM64 path:

- Docker Desktop or another Docker runtime.
- Apple Silicon or another host that can run `linux/arm64` containers.
- Network access for the first image build.

## Run

For the native Apple Silicon fixture, follow the setup in
`APPLE_SILICON_VLLM_CPU_INSTALL.md`, then run:

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

For the Docker Linux ARM64 fixture, run from the repo root:

```bash
tests/cpu_behavioral/run_hidden_states_docker_aarch64.sh
```

The first run can take a while. The Docker image builds:

- `triton-lang/triton-cpu`
- vLLM `v0.29.0` with CPU requirements
- this repo installed editable with `--no-deps`

Then it runs the atomic CPU behavior test:

```bash
python tests/cpu_behavioral/run_hidden_states_cpu.py
```

## Files

- `Dockerfile.aarch64`: Linux ARM64 image for the CPU test environment.
- `run_hidden_states_docker_aarch64.sh`: build-and-run wrapper.
- `run_hidden_states_cpu.py`: the hidden-states generation test for CPU runners.
- `APPLE_SILICON_VLLM_CPU_INSTALL.md`: native Apple Silicon vLLM CPU install and patch notes.
- `RESEARCH_NOTES.md`: native macOS findings and troubleshooting notes.

## Image Name

Override the local image tag with:

```bash
MIA_CPU_DOCKER_IMAGE=my-local-image \
tests/cpu_behavioral/run_hidden_states_docker_aarch64.sh
```

## Build Resources

The Dockerfile limits native compilation to two jobs by default. If Colima or Docker drops the build
connection while compiling `triton-cpu`, retry with one build job:

```bash
MIA_CPU_DOCKER_BUILD_JOBS=1 \
tests/cpu_behavioral/run_hidden_states_docker_aarch64.sh
```

## Build Args

To change source refs manually:

```bash
docker build \
  --platform linux/arm64 \
  --build-arg BUILD_JOBS=2 \
  --build-arg VLLM_REF=v0.29.0 \
  --build-arg TRITON_CPU_REF=main \
  -f tests/cpu_behavioral/Dockerfile.aarch64 \
  -t vllm-hook-mia-cpu-aarch64 \
  .
```
