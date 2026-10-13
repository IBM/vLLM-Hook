# Apple Silicon vLLM CPU Install Notes

These notes preserve the native macOS setup that reached generation and hidden-state capture on
Apple Silicon. This path is still investigative because it depends on local Triton CPU patches and
an OpenMP duplicate-runtime workaround, but it now passes the MLR-50 hidden-state CPU fixture.

## Known Local Environment

```text
repo: vLLM-Hook checkout root
venv: .venv-mac-cpu
triton-cpu: 9da8af77fbeb660616c41ae89a82e2bba1be82fc
vLLM: v0.29.0 / 98dff2a81d747d1dba01a47f939f48c3526d4206
MIA: 456408e7e2cca28b80a86d06bfb2d0e096d5c5fd
```

## Create the venv

Run from the repo root:

```bash
export REPO_ROOT="$(pwd)"
python3.12 -m venv .venv-mac-cpu
source .venv-mac-cpu/bin/activate
python -m pip install -U pip uv setuptools wheel cmake ninja pytest ipykernel
python -m pip install setuptools-rust setuptools-scm zstandard
```

## Build Triton CPU

```bash
mkdir -p /private/tmp/triton-cpu-build
cd /private/tmp/triton-cpu-build
git clone https://github.com/triton-lang/triton-cpu.git
cd triton-cpu
git checkout 9da8af77fbeb660616c41ae89a82e2bba1be82fc

"$REPO_ROOT/.venv-mac-cpu/bin/python" -m pip install \
  -r python/requirements.txt \
  -r python/test-requirements.txt \
  -r python/tutorials/requirements.txt

"$REPO_ROOT/.venv-mac-cpu/bin/python" -m pip install -e . --no-build-isolation
```

## Check Triton CPU Visibility

```bash
cd "$REPO_ROOT"

TRITON_DEFAULT_BACKEND=cpu .venv-mac-cpu/bin/python - <<'PY'
import triton
print("triton", triton.__version__)

from triton.backends import backends
print("backends", list(backends))

try:
    from triton.runtime import driver
    driver.set_active_to_cpu()
    print("set_active_to_cpu ok")
except Exception as e:
    print("set_active_to_cpu failed", repr(e))

from vllm.triton_utils import HAS_TRITON
print("vllm HAS_TRITON", HAS_TRITON)
PY
```

## Build vLLM CPU

```bash
mkdir -p /private/tmp/vllm-build
cd /private/tmp/vllm-build
git clone --branch v0.29.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt

# Clean up any previously broken build artifacts.
rm -rf build/ dist/ *.egg-info

# Enforce native CPU target optimizations and build the editable package.
CFLAGS="-march=native" CXXFLAGS="-march=native" "$REPO_ROOT/.venv-mac-cpu/bin/python" -m pip install -e .
```

If the build fails with `fatal error: 'omp.h' file not found`, install `libomp`
and pass its include/library paths explicitly:

```bash
brew install libomp
export LIBOMP_PREFIX="$(brew --prefix libomp)"

rm -rf build/ dist/ *.egg-info

CFLAGS="-march=native -I${LIBOMP_PREFIX}/include" \
CXXFLAGS="-march=native -I${LIBOMP_PREFIX}/include -UPy_LIMITED_API" \
LDFLAGS="-L${LIBOMP_PREFIX}/lib -lomp" \
"$REPO_ROOT/.venv-mac-cpu/bin/python" -m pip install -e .
```

The `-UPy_LIMITED_API` flag is included because one observed Apple Clang build
used `-DPy_LIMITED_API=3`, which made PyTorch's C++ headers see `c10::IValue`
as incomplete and fail before or alongside the `omp.h` error.

## Install MIA Without Replacing vLLM or Triton

```bash
cd "$REPO_ROOT"
.venv-mac-cpu/bin/python -m pip install zstandard
.venv-mac-cpu/bin/python -m pip install -e . --no-deps
```

## Run the Hidden-State Fixture

```bash
export REPO_ROOT="$(pwd)"

rm -rf /private/tmp/triton-cache

cd "$REPO_ROOT"

unset TRITON_LOCAL_LIBOMP_PATH
unset DYLD_INSERT_LIBRARIES

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

Expected success ends with:

```text
model.decoder.layers.0: 1 tensors, shape=(768,)
model.decoder.layers.1: 1 tensors, shape=(768,)
model.decoder.layers.2: 1 tensors, shape=(768,)
model.decoder.layers.3: 1 tensors, shape=(768,)
CPU hidden-state behavior fixture passed.
```

## Apple Silicon Triton Patch

On Apple Silicon, the local `triton-cpu` checkout may need a temporary patch in:

```text
/private/tmp/triton-cpu-build/triton-cpu/third_party/cpu/backend/build.py
/private/tmp/triton-cpu-build/triton-cpu/python/triton/backends/cpu/build.py
```

Keep `-mcpu=native` for Linux ARM only:

```python
if system == "Linux" and machine in ("aarch64", "arm64"):
    cpu_flags.append("-mcpu=native")
```

The exact local patch was:

```diff
-        if system in ("Linux", "Darwin") and machine in ("aarch64", "arm64"):
+        if system == "Linux" and machine in ("aarch64", "arm64"):
```

Without this, Apple clang can reject generated assembly with:

```text
instruction requires: cssc
umin w9, w9, #1
```

If generation still fails with the same CSSC error, add an explicit Darwin
assembly flag in that same `source_kind == "assembly"` branch:

```python
if system == "Darwin" and machine in ("aarch64", "arm64"):
    cpu_flags.append("-march=armv8.7-a+cssc")
```

On the observed Apple Clang toolchain, `-march=armv8.7-a+cssc` accepted the
`umin` instruction, while `-mcpu=native` and `-mcpu=apple-m4` did not.

## Triton Patch Ledger

Track every local `triton-cpu` workaround here before and after retesting.

| Patch | File(s) | Change | Why | Test Status | Retest Command |
| --- | --- | --- | --- | --- | --- |
| `linux-only-mcpu-native` | `third_party/cpu/backend/build.py`, `python/triton/backends/cpu/build.py` | Only append `-mcpu=native` when `system == "Linux"` on ARM. | Apple Clang accepted the flag but still rejected generated assembly using CSSC. Keeping this flag Linux-only avoids implying it fixes Darwin. | Applied locally. Not sufficient by itself; generation still failed on `umin w9, w9, #1`. | Run hidden-state fixture with a clean `TRITON_CACHE_DIR`. |
| `darwin-cssc-assembly-flag` | `third_party/cpu/backend/build.py`, `python/triton/backends/cpu/build.py` | Append `-march=armv8.7-a+cssc` for Darwin ARM assembly builds. | Triton CPU generated `umin`, which Apple Clang rejects unless CSSC is enabled for standalone assembly. | Applied locally. Passed hidden-state fixture with OpenMP duplicate workaround and CPU-safe MIA hook guard. | Run hidden-state fixture with a clean `TRITON_CACHE_DIR`. |

## MIA CPU Guard

The native CPU run also needs MIA hooks to avoid CUDA-only stream checks on CPU.

Patch:

```python
if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
    return None
```

Known locations:

```text
mia/workers/hs_capture_worker.py
mia/workers/qk_capture_worker.py
```

Without this, CPU generation reaches the model forward and then fails with:

```text
RuntimeError: Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing
```

## OpenMP Attempts

This variant was tried:

```bash
brew install libomp

TRITON_DEFAULT_BACKEND=cpu \
TRITON_CACHE_DIR=/private/tmp/triton-cache \
TRITON_LOCAL_LIBOMP_PATH="$(brew --prefix libomp)" \
MIA_WRITER_PROCESS=0 \
.venv-mac-cpu/bin/python tests/cpu_behavioral/run_hidden_states_cpu.py
```

It hit duplicate OpenMP runtime initialization:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Preloading libomp was also tried:

```bash
DYLD_INSERT_LIBRARIES="$(brew --prefix libomp)/lib/libomp.dylib" \
TRITON_DEFAULT_BACKEND=cpu \
TRITON_CACHE_DIR=/private/tmp/triton-cache \
TRITON_LOCAL_LIBOMP_PATH="$(brew --prefix libomp)" \
MIA_WRITER_PROCESS=0 \
.venv-mac-cpu/bin/python tests/cpu_behavioral/run_hidden_states_cpu.py
```

That reached a different failure during model load:

```text
TVMFFIEnvRegisterCAPI
```
