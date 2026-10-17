# Cross-platform install instructions for testing

This guide includes the repository's default CUDA installation and the CPU/Metal
installation checkpoint on Apple Silicon. Each backend uses its own environment.
The CUDA procedure comes from the [root README](../README.md#-installation);
CUDA was not executed as part of the local Apple Silicon verification.

- **CPU and Metal on Apple Silicon:** follow steps 1–7 below.
- **CUDA on a host with a compatible NVIDIA GPU and driver:** follow step 8.
- **Parity:** see the remaining work at the end. Passing installation checks
  does not establish numerical parity.

The CPU build steps here are specifically for macOS/Apple Silicon. They are not
a general Linux or Windows CPU build recipe. CUDA requires a separate compatible
host; it cannot run on the Apple Silicon GPU.

**Jump to a section**

- [Checkpoint and limits](#checkpoint-and-limits)
- [Prerequisites and paths](#1-prerequisites-and-paths)
- [CPU environment and Triton CPU](#2-cpu-environment-and-triton-cpu)
- [Build vLLM CPU and install MIA](#3-build-vllm-cpu-and-install-mia)
- [Metal source and registration fix](#4-metal-source-and-registration-fix)
- [Install and verify Metal](#5-install-and-verify-metal)
- [Concurrent CPU/Metal verification](#6-concurrent-verification)
- [MIA packaging check in Metal's environment](#7-optional-mia-packaging-check-in-metals-environment)
- [CUDA installation and testing](#8-default-cuda-installation-and-testing)
- [Remaining parity work](#remaining-parity-work)
- [Sources](#sources)

## Checkpoint and limits

At this checkpoint, the user reports a verification run with no errors, failed
tests, or xfailed tests. CPU scenarios and standalone Metal generation have also
passed during this session. Optional backends and unfinished parity cases skip.
Skipped cases are not evidence of Metal hook support or numerical parity.

Metal generation works, but MIA's Metal adapter is not implemented in this
checkout. The apple-silicon branch uses the older `vllm_hook_plugins` integration.
This procedure does not port that integration. CPU uses `facebook/opt-125m`;
the Metal smoke check uses `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Different
models cannot establish parity.

These are manual runtime-provisioning instructions. Installing an extra alone
does not perform all native build steps. `pip` ignores `tool.uv.sources`, and
`--no-deps` intentionally preserves the runtimes provisioned below. Independent
`uv sync` resolution of all profiles is not verified. Do not sync over these
editable source environments.

## 1. Prerequisites and paths

Use native arm64 Python 3.12, Homebrew, Git, and Apple development tools.
The pinned Metal runtime requires macOS 15 or newer. Its source installer
requires macOS SDK 26.2 or newer; update/select Xcode if necessary.

```bash
uname -m
xcode-select -p
brew install python@3.12 libomp cmake ninja rust
python3.12 -c 'import platform; print(platform.machine())'
sw_vers -productVersion
xcrun --sdk macosx --show-sdk-version
rustc --version
cargo --version
```

Both architecture checks should print `arm64`. Run the following from anywhere
inside your vLLM-Hook checkout, and keep using the same shell. The repository
path is detected automatically. The sandbox defaults to a sibling directory;
set `MIA_SANDBOX` first if you want another location outside the checkout:

```bash
export MIA_REPO="$(git rev-parse --show-toplevel)"
export MIA_SANDBOX="${MIA_SANDBOX:-$(dirname "$MIA_REPO")/_sandbox}"
mkdir -p "$MIA_SANDBOX"
export MIA_SANDBOX="$(cd "$MIA_SANDBOX" && pwd -P)"
export CPU_PY="$MIA_REPO/.venv-install-test-cpu/bin/python"
export METAL_PY="$MIA_SANDBOX/vllm-metal/.venv-vllm-metal/bin/python"
```

These paths do not depend on a username, home directory, or checkout name.
Relative sandbox overrides are converted to absolute paths before builds run.

The three Git checkouts are siblings under the selected sandbox: `triton-cpu`,
`vllm-cpu`, and `vllm-metal`. No repository is cloned inside another.
Skip clone commands when a checkout already exists. Do not discard local edits.
Reusing an existing environment is not a clean-install test.

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

## 4. Metal source and registration fix

```bash
git clone --branch v0.29.0 --depth 1 https://github.com/vllm-project/vllm-metal.git "$MIA_SANDBOX/vllm-metal"
git -C "$MIA_SANDBOX/vllm-metal" describe --tags --exact-match
```

The bundled patch fixes registration against the installed Transformers stack.
It checks the config registry without constructing an invalid default Gemma
config. Apply it once to the pinned checkout:

```bash
git -C "$MIA_SANDBOX/vllm-metal" apply --check "$MIA_REPO/cross-platform-install/vllm-metal-registration.patch" &&
git -C "$MIA_SANDBOX/vllm-metal" apply "$MIA_REPO/cross-platform-install/vllm-metal-registration.patch"
```

If already applied, the check above fails without changing anything. Confirm
with this command; success means the patch is already present:

```bash
git -C "$MIA_SANDBOX/vllm-metal" apply --reverse --check "$MIA_REPO/cross-platform-install/vllm-metal-registration.patch"
```

If neither check succeeds, inspect the checkout instead of forcing the patch.
The local fix is not asserted to be released upstream.

## 5. Install and verify Metal

```bash
(
  cd "$MIA_SANDBOX/vllm-metal" || exit 1
  env -u VIRTUAL_ENV -u VLLM_TARGET_DEVICE -u TRITON_DEFAULT_BACKEND \
    -u TRITON_DISABLE_OPENMP -u TRITON_LOCAL_LIBOMP_PATH \
    -u DYLD_INSERT_LIBRARIES bash ./install.sh
)
```

The installer creates `.venv-vllm-metal` inside its checkout and builds native
components. Verify the interpreter exists before using it:

```bash
test -x "${METAL_PY:?Set the paths in step 1 first}"
env -u VLLM_TARGET_DEVICE -u TRITON_DEFAULT_BACKEND \
  -u TRITON_DISABLE_OPENMP -u TRITON_LOCAL_LIBOMP_PATH \
  -u DYLD_INSERT_LIBRARIES VLLM_USE_V2_MODEL_RUNNER=0 \
  "$METAL_PY" "$MIA_REPO/cross-platform-install/serve_metal_checked.py" --help
```

The output must include `Verified platform: MetalPlatform`. The bundled launcher
rejects CPU fallback; MLX availability alone does not prove Metal was selected.
The Python environment remains separate from the CPU environment.

## 6. Concurrent verification

Both models must be cached or downloadable. Port 8001 must be unused.
No environment activation or separate terminal is required.

```bash
python3 "$MIA_REPO/cross-platform-install/verify_backends.py" \
  --repo "$MIA_REPO" --sandbox "$MIA_SANDBOX"
```

The runner launches both processes, saves separate logs and a JSON summary under
`_sandbox/backend-run.*`, checks Metal generation, and cleans up its process
groups. It returns nonzero if either backend fails. CPU process communication
uses `GLOO_SOCKET_IFNAME=lo0` to avoid the observed local hostname lookup failure.

Expected: CPU exit status 0 (`4 passed, 16 skipped` for the current suite), Metal
request success, and `Verified platform: MetalPlatform` in the Metal log.
The runner checks generation output as well as HTTP status.

Both runtimes share system memory. The current CPU fixtures request 8 GiB per
engine, sequentially. A memory-budget startup error is not a parity failure.
Free memory before retrying; changing the interface setting does not change
memory requirements. Model downloads can exceed the default readiness timeout;
use `--startup-timeout 900` if required after inspecting the log.

## 7. Optional MIA packaging check in Metal's environment

After standalone Metal generation succeeds:

```bash
"$METAL_PY" -m pip install zstandard pytest
"$METAL_PY" -m pip install -e "${MIA_REPO}[metal]" --no-deps
"$METAL_PY" -m pip check
```

This checks packaging while preserving the runtime. It does not enable MIA Metal
hooks or prove independent dependency resolution. Any `pip check` failures must
be inspected, not silently ignored.

## 8. Default CUDA installation and testing

This preserves the installation sequence in the [root README](../README.md).
Use a compatible NVIDIA GPU host with Conda available. Run from this checkout's
root on that host; if cloning there, check out the branch under test first.
Do not install the CUDA requirements into either Apple Silicon environment.

The root README uses `conda activate vllm-hook-mia`. Here, `conda run` selects the
same dedicated environment explicitly, leaving the calling shell's active
environment unchanged.

```bash
# Run from the repository root on the CUDA host.
conda create -n vllm-hook-mia python=3.12 pip
conda run -n vllm-hook-mia python -m pip install -r requirement.txt
conda run -n vllm-hook-mia python -m pip uninstall -y torchcodec
conda run -n vllm-hook-mia python -m pip install -e . --no-deps
conda run -n vllm-hook-mia python -m pip install pytest
```

`requirement.txt` pins the existing stack, including vLLM 0.29.0, torch 2.13.0,
and CUDA Triton 3.7.1. Do not replace CUDA Triton with the Triton CPU fork: they
are alternative builds of the same Python distribution. The CUDA extra in
`pyproject.toml` mirrors these pins; the commands above retain the root README's
installation route.

MIA does not use `torchcodec`, so the default instructions remove it. The root
README documents that `pip check` then reports the missing dependency; this
specific report is expected. Investigate any other dependency failures.

Before testing, remove CPU-only settings from the CUDA test process and verify
that vLLM and PyTorch both select CUDA:

```bash
env -u VLLM_TARGET_DEVICE -u TRITON_DEFAULT_BACKEND \
  -u TRITON_DISABLE_OPENMP -u TRITON_LOCAL_LIBOMP_PATH \
  -u DYLD_INSERT_LIBRARIES -u GLOO_SOCKET_IFNAME \
  -u VLLM_USE_V2_MODEL_RUNNER \
  conda run -n vllm-hook-mia python -c 'import torch; from vllm.platforms import current_platform; print("Platform:", type(current_platform).__name__); print("CUDA available:", torch.cuda.is_available()); assert current_platform.is_cuda(); assert torch.cuda.is_available()'
```

Run the default installation checks, which boot real GPU engines:

```bash
env -u VLLM_TARGET_DEVICE -u TRITON_DEFAULT_BACKEND \
  -u TRITON_DISABLE_OPENMP -u TRITON_LOCAL_LIBOMP_PATH \
  -u DYLD_INSERT_LIBRARIES -u GLOO_SOCKET_IFNAME \
  -u VLLM_USE_V2_MODEL_RUNNER \
  conda run --no-capture-output -n vllm-hook-mia \
  python -m pytest tests/use_cases -m gpu
```

For the shared four-scenario behavioral suite, require CUDA explicitly:

```bash
env -u VLLM_TARGET_DEVICE -u TRITON_DEFAULT_BACKEND \
  -u TRITON_DISABLE_OPENMP -u TRITON_LOCAL_LIBOMP_PATH \
  -u DYLD_INSERT_LIBRARIES -u GLOO_SOCKET_IFNAME \
  -u VLLM_USE_V2_MODEL_RUNNER MIA_REQUIRED_BACKENDS=cuda \
  conda run --no-capture-output -n vllm-hook-mia \
  python -m pytest tests/cpu_behavioral/test_vllm_cpu_fixtures.py -vv -rs
```

CUDA scenarios should execute; CPU/Metal scenarios and unfinished parity cases
skip. Required CUDA unavailability fails instead of being silently skipped.
The existing `verify_backends.py` runner remains dedicated to the Apple Silicon
CPU/Metal check; it does not launch a remote CUDA job.

As documented in the root README, MIA registers as a vLLM plugin for every engine
in its environment. Without `MIA_WORKER`, it installs the hidden-state capture
worker, including its 4 GiB GPU buffer. Keep this environment dedicated to MIA.
For a separate stock-vLLM run, set `VLLM_PLUGINS=''` only on that command; do not
disable plugins when testing MIA. See [the test guide](../tests/README.md) for
model requirements and GPU execution details.

## Remaining parity work

Port the MIA Metal adapter; implement equivalent scenarios; use identical model
weights, tokenizer, prompts, and settings; export results; compare tensor shapes,
scores, rankings, and behavior with explicit numerical tolerances. Compare only
after both backend runs pass. Concurrent smoke checks alone do not establish
numerical parity.

## Sources

- [Default CUDA installation](../README.md) and [validated dependency pins](../requirement.txt)

- [Triton CPU pinned source](https://github.com/triton-lang/triton-cpu/tree/c8e7cc932320d05d08f39ffc99bfda32dc4f22fa)
- [vLLM CPU requirements](https://github.com/vllm-project/vllm/blob/v0.29.0/requirements/cpu.txt)
- [Metal pinned installer](https://github.com/vllm-project/vllm-metal/blob/v0.29.0/install.sh)
- [Metal build tooling](https://github.com/vllm-project/vllm-metal/blob/v0.29.0/scripts/lib.sh)
- [Older Apple Silicon hook integration](https://github.com/IBM/vLLM-Hook/tree/apple-silicon)
- Historical local CPU notes: `tests/cpu_behavioral/docs/APPLE_SILICON_VLLM_CPU_INSTALL.md` (its manual Triton patch and old pytest opt-in guidance are superseded here).
