# CPU setup for behavioral tests

Use this guide to prepare a CPU environment and run MIA's four behavioral tests.
The build steps below are for **macOS on Apple Silicon**. Linux and Windows
installation steps have not been validated here.

[pyproject.toml](../../pyproject.toml) lists optional dependencies for CPU, CUDA,
and Metal. Choose the option that matches your runtime. Use a separate Python
environment for each one: CPU and CUDA need different builds of the same libraries.

For CPU, the steps are:

1. Install the build tools and create a Python environment.
2. Build Triton CPU and vLLM CPU.
3. Install MIA and run the tests.

Installing the `cpu` option alone does not complete this setup. Follow the build
steps below before installing MIA.

## 1. Prerequisites and paths

You need Python 3.12 running natively on Apple Silicon, Homebrew, Git, and
Apple development tools. Install the build tools and check your setup:

```bash
uname -m
xcode-select -p
brew install python@3.12 libomp cmake ninja rust
python3.12 -c 'import platform; print(platform.machine())'
rustc --version
cargo --version
```

Both architecture checks should print `arm64`.

Next, set the paths used throughout this guide. Run these commands from inside
your vLLM-Hook checkout, and keep using the same terminal session:

```bash
export MIA_REPO="$(git rev-parse --show-toplevel)"
export MIA_SANDBOX="${MIA_SANDBOX:-$(dirname "$MIA_REPO")/_sandbox}"
mkdir -p "$MIA_SANDBOX"
export MIA_SANDBOX="$(cd "$MIA_SANDBOX" && pwd -P)"
export CPU_PY="$MIA_REPO/.venv-install-test-cpu/bin/python"
```

`MIA_REPO` is your vLLM-Hook checkout. `MIA_SANDBOX` is a sibling folder that
will hold the Triton CPU and vLLM CPU source code. Set `MIA_SANDBOX` before this
step if you want to use another folder. `CPU_PY` points to the Python environment
created in the next step.

If a source checkout already exists, skip its clone command and keep any local
edits. Reusing an existing Python environment does not verify a fresh install.

## 2. CPU environment and Triton CPU

```bash
python3.12 -m venv "$MIA_REPO/.venv-install-test-cpu"
"$CPU_PY" -m pip install --upgrade pip
"$CPU_PY" -m pip install setuptools wheel cmake ninja setuptools-rust setuptools-scm nanobind==2.10.2 pytest zstandard

git clone https://github.com/triton-lang/triton-cpu.git "$MIA_SANDBOX/triton-cpu"
git -C "$MIA_SANDBOX/triton-cpu" fetch origin main
git -C "$MIA_SANDBOX/triton-cpu" checkout c8e7cc932320d05d08f39ffc99bfda32dc4f22fa
```

This Triton revision includes the upstream Darwin assembly fix. No manual
Triton patch is needed. Stop if checkout reports conflicting edits.

```bash
(
  cd "$MIA_SANDBOX/triton-cpu" || exit 1
  "$CPU_PY" -m pip install -r python/requirements.txt -r python/test-requirements.txt -r python/tutorials/requirements.txt &&
  "$CPU_PY" -m pip install -e . --no-build-isolation
)
```

## 3. Build vLLM CPU and install MIA

```bash
git clone --branch v0.29.0 --depth 1 https://github.com/vllm-project/vllm.git "$MIA_SANDBOX/vllm-cpu"
git -C "$MIA_SANDBOX/vllm-cpu" describe --tags --exact-match

(
  cd "$MIA_SANDBOX/vllm-cpu" || exit 1
  MIA_LIBOMP_PREFIX="$(brew --prefix libomp)" || exit 1
  "$CPU_PY" -m pip install -r requirements/cpu.txt || exit 1
  VLLM_TARGET_DEVICE=cpu \
  CFLAGS="-march=native -I${MIA_LIBOMP_PREFIX}/include" \
  CXXFLAGS="-march=native -I${MIA_LIBOMP_PREFIX}/include -UPy_LIMITED_API" \
  LDFLAGS="-L${MIA_LIBOMP_PREFIX}/lib -lomp" \
  "$CPU_PY" -m pip install -e .
)

"$CPU_PY" -m pip install zstandard pytest
"$CPU_PY" -m pip install -e "${MIA_REPO}[cpu]" --no-deps
```

The version check should print `v0.29.0`. Verify backend selection:

```bash
GLOO_SOCKET_IFNAME=lo0 VLLM_TARGET_DEVICE=cpu TRITON_DEFAULT_BACKEND=cpu "$CPU_PY" - <<'CHECK'
import triton
from triton.backends import backends
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
print("Triton:", triton.__version__, triton.__file__)
print("Platform:", type(current_platform).__name__)
assert "cpu" in backends
assert HAS_TRITON
assert current_platform.is_cpu()
CHECK
```

## 4. Run the behavioral scenarios

From the MIA checkout, using the CPU environment created above:

```bash
cd "$MIA_REPO"
MIA_REQUIRED_BACKENDS=cpu "$CPU_PY" -m pytest \
  tests/cpu_behavioral/test_vllm_cpu_fixtures.py -v -rs
```

The four CPU scenarios exercise hidden-state capture, attention tracking,
activation steering, and CoRe reranking. CUDA and Metal cases skip when their
runtime is unavailable. Required CPU coverage fails instead of skipping.
Activation steering uses a zero coefficient; it checks execution rather than
a steering effect. These tests do not establish numerical CPU/GPU parity.

The default model is `facebook/opt-125m`. Set `MIA_BEHAVIOR_MODEL` to use an
already downloaded compatible model. Each case keeps its own log and hooks
inside pytest's temporary directory.

If vLLM reports insufficient free CPU memory at startup, its existing
`VLLM_CPU_KVCACHE_SPACE=1` setting requests a 1 GiB cache for these small scenarios.
The first run may need time to compile Triton CPU kernels.

## Validation limits

Earlier CPU behavioral runs passed on native Apple Silicon with vLLM 0.29.0
and a locally patched Triton CPU build. The pinned Triton revision above includes
the upstream Darwin assembly fix, but a fresh install of that revision and a
complete build from this guide have not been validated. Compiler and OpenMP
workarounds remain platform-specific. No GitHub workflow or runner setup is
included.

## How the dependency settings work

`setup.py` defines MIA's version, base dependencies, and vLLM plugin entries.
`pyproject.toml` adds the optional CPU, CUDA, and Metal dependencies. Its `test`
and `cpu-build` groups list the test and build tools.

The `tool.uv.sources` settings tell uv which dependency builds to use. They select
the official vLLM CPU wheel on Linux x86_64, pinned vLLM source on other platforms,
pinned Triton CPU source, and PyTorch's CPU index on supported Linux architectures.
`pip` ignores these settings, so the manual steps above prepare the CPU builds
before installing MIA with `--no-deps`.

A full install with `uv sync` has not been validated. Keep environments prepared
with the manual steps separate from `uv sync`. Neither the dependency settings
nor the `cpu` option installs system compilers or OpenMP.

The original wheel checksum remains in the project settings. The Colab download
helper mentioned in its comment is excluded from this PR. The stored checksum
alone does not make uv verify the download against it.
