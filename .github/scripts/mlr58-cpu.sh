#!/usr/bin/env bash
set -euo pipefail
phase=$1
root=${GITHUB_WORKSPACE:-$PWD}
out="$root/evidence"
mkdir -p "$out" "$root/wheelhouse"
exec > >(tee -a "$out/$phase.log") 2>&1
trap 'code=$?; echo "$code" > "$out/'"$phase"'.exit-code"' EXIT
case "$phase" in
host)
  uname -a
  python - <<'PYHOST'
import json, platform, sys, os
from pathlib import Path
Path(os.environ['GITHUB_WORKSPACE'], 'evidence', 'host.json').write_text(json.dumps(dict(platform=platform.platform(), machine=platform.machine(), python=sys.version), indent=2))
PYHOST
  if [[ "$CPU_PLATFORM" = linux ]]; then
    cat /etc/os-release; lscpu; free -h; ldd --version
  elif [[ "$CPU_PLATFORM" = macos ]]; then
    sw_vers; sysctl -n machdep.cpu.brand_string; sysctl hw.memsize; clang --version
  else
    powershell.exe -NoProfile -Command 'Get-CimInstance Win32_Processor | Format-List *; Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize'
  fi
  df -h
  gcc --version || true
  python --version
  git -C "$root/target" rev-parse HEAD | tee "$out/tested-commit.txt"
  printf '%s\n' "workflow=$GITHUB_SHA" "run=$GITHUB_RUN_ID" "attempt=$GITHUB_RUN_ATTEMPT" > "$out/run.txt"
  if [[ "$CPU_PLATFORM" = linux ]]; then
    test "$(uname -sm)" = 'Linux x86_64'
    grep -qw avx2 /proc/cpuinfo
  fi
  test "$(git -C "$root/target" rev-parse HEAD)" = "$TEST_SHA"
  ;;
setup)
  if [[ "$CPU_PLATFORM" = linux ]]; then
    sudo apt-get update
    sudo apt-get install -y build-essential libnuma-dev libomp-dev
  elif [[ "$CPU_PLATFORM" = macos ]]; then
    brew install libomp
  fi
  python -m venv "$root/.venv"
  if [[ "$CPU_PLATFORM" = windows ]]; then source "$root/.venv/Scripts/activate"; else source "$root/.venv/bin/activate"; fi
  python -m pip install --upgrade pip
  python -m pip install setuptools==77.0.3 wheel 'cmake>=3.26.1,<4' ninja nanobind==2.10.2 packaging jinja2 regex lit pytest
  if [[ "$CPU_PLATFORM" = windows ]]; then
    # Native Windows is an installation probe, never a claimed behavioral pass.
    echo 'Experimental native Windows dependency probe; scenarios require a working CPU vLLM.'
    python -m pip install 'vllm==0.29.0' 'torch==2.13.0'
  elif [[ "$CPU_PLATFORM" = macos ]]; then
    python -m pip install torch==2.13.0 torchvision torchaudio setuptools-rust setuptools-scm zstandard
  else
  curl -fL --retry 3 'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcpu-cp38-abi3-manylinux_2_34_x86_64.whl' -o "$out/vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl"
  echo '4d22dac8259e7e24dbc615531c73cbd1a6d4b4214a3329ccddb7854f26c9dac8  '"$out/vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl" | sha256sum -c -
  python -m pip install "$out/vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
  rm "$out/"*.whl
  fi
  python -m pip install --no-deps -e "$root/target"
  ;;
wheel)
  if [[ "$CPU_PLATFORM" = windows ]]; then source "$root/.venv/Scripts/activate"; else source "$root/.venv/bin/activate"; fi
  if compgen -G "$root/wheelhouse/triton-*.whl" > /dev/null; then
    echo 'reused compiled wheel' | tee "$out/wheel-source.txt"
    python "$root/.github/scripts/mlr58-wheel.py" check
  else
    echo 'built compiled wheel (cache miss)' | tee "$out/wheel-source.txt"
    git clone https://github.com/triton-lang/triton-cpu.git "$root/triton-cpu"
    git -C "$root/triton-cpu" checkout --detach "$TRITON_SHA"
    test "$(git -C "$root/triton-cpu" rev-parse HEAD)" = "$TRITON_SHA"
    python -m pip wheel --no-deps --no-build-isolation -v "$root/triton-cpu" -w "$root/wheelhouse"
    python "$root/.github/scripts/mlr58-wheel.py" write
  fi
  python -m pip install --no-deps --force-reinstall "$root/wheelhouse/"triton-*.whl
  cp "$root/wheelhouse/wheel.sha256" "$out/"
  ;;
verify|scenarios)
  if [[ "$CPU_PLATFORM" = windows ]]; then source "$root/.venv/Scripts/activate"; else source "$root/.venv/bin/activate"; fi
  # The release wheel needs Intel OpenMP loaded before torch/vLLM.
  if [[ "$CPU_PLATFORM" = macos ]]; then
    if [[ "$phase" = verify ]]; then
      git clone https://github.com/vllm-project/vllm.git "$root/vllm-source"
      git -C "$root/vllm-source" checkout --detach 98dff2a81d747d1dba01a47f939f48c3526d4206
      python -m pip install -r "$root/vllm-source/requirements/cpu.txt"
      python -m pip install "$root/vllm-source"
    fi
    export KMP_DUPLICATE_LIB_OK=TRUE TRITON_DISABLE_OPENMP=1
  elif [[ "$CPU_PLATFORM" = linux ]]; then
  iomp=$(find "$root/.venv" -name libiomp5.so -print -quit)
  test -n "$iomp"
  export LD_PRELOAD="$iomp${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  if [ "$phase" = verify ]; then
    python -m pip freeze | tee "$out/dependencies.txt"
    python -m pip check
    python - <<'PY'
import importlib.metadata as m
import torch, triton
from triton.backends import backends
from vllm.platforms import current_platform
print('torch:', torch.__version__, 'triton:', triton.__version__)
print('vllm:', m.version('vllm'), 'platform:', current_platform, 'backends:', backends)
assert m.version('vllm').split('+')[0] == '0.29.0'
assert torch.__version__.split('+')[0] == '2.13.0'
assert current_platform.is_cpu()
assert 'cpu' in backends
PY
  else
    cd "$root/target"
    # Keep the PR scenarios and assertions unchanged. Retain per-case child logs.
    MIA_REQUIRED_BACKENDS=cpu VLLM_CPU_KVCACHE_SPACE=1 python -m pytest tests/cpu_behavioral/test_vllm_cpu_fixtures.py -v -rs --basetemp="$out/pytest" --junitxml="$out/results.xml"
  fi
  ;;
*) exit 2 ;;
esac
