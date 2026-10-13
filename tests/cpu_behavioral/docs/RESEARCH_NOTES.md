# CPU Behavioral Research Notes

These notes capture the local Apple Silicon CPU investigation.

## Target

The first CPU behavior target is the existing hidden-state worker:

```text
worker_name="capture_hs"
analyzer_name="hidden_states"
```

The atomic test must call `generate()`. Hook installation alone is not enough.

## Native Apple Silicon Path

Native macOS reached important milestones:

```text
vLLM CPU starts
Using V2 Model Runner
MIA injects HSCaptureWorker into CPUWorker
capture_hs installs hooks
```

Native macOS later reached a passing hidden-state fixture after local patches:

```text
model.decoder.layers.0: 1 tensors, shape=(768,)
model.decoder.layers.1: 1 tensors, shape=(768,)
model.decoder.layers.2: 1 tensors, shape=(768,)
model.decoder.layers.3: 1 tensors, shape=(768,)
CPU hidden-state behavior fixture passed.
```

Before that, full generation hit platform-toolchain problems:

```text
instruction requires: cssc
```

OpenMP experiments also conflicted with the process:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Preloading `libomp` with `DYLD_INSERT_LIBRARIES` caused a model-load segfault in local testing:

```text
TVMFFIEnvRegisterCAPI
```

## Native macOS Notes

Native macOS is useful for investigation and now has a local passing path for the
hidden-state fixture.

The recovered native install path is preserved separately in
`APPLE_SILICON_VLLM_CPU_INSTALL.md`.

## Failure Labels

When this fails, record the first failing stage:

- `install`: vLLM, Triton CPU, or MIA cannot build or install.
- `import`: `import vllm`, `import triton`, or `import mia` fails.
- `triton-backend`: Triton imports but vLLM reports `HAS_TRITON False`.
- `engine-start`: `MiaLLM(...)` cannot start a CPU engine.
- `hook-registration`: MIA worker/plugin registration fails.
- `generation`: `llm.generate(...)` fails before probes are returned.
- `capture`: generation runs but probes are missing or malformed.
- `analyzer`: captured data exists but `analyze(...)` fails.
