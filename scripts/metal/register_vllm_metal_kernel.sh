#!/usr/bin/env bash
set -euo pipefail

# Register a Jupyter kernel that points at the active vLLM Metal virtualenv.
VENV_PATH="${VLLM_METAL_VENV:-$HOME/.venv-vllm-metal}"
KERNEL_NAME="${VLLM_METAL_KERNEL_NAME:-vllm-metal-local}"
DISPLAY_NAME="${VLLM_METAL_KERNEL_DISPLAY_NAME:-Python (vllm-metal)}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Could not find vLLM Metal virtualenv at: $VENV_PATH" >&2
  echo "Set VLLM_METAL_VENV or install vllm-metal first." >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip jupyter ipykernel

# Remove the previous spec for this kernel name so reruns stay idempotent.
jupyter kernelspec remove -f "$KERNEL_NAME" >/dev/null 2>&1 || true

python -m ipykernel install \
  --user \
  --name "$KERNEL_NAME" \
  --display-name "$DISPLAY_NAME"

echo "Registered Jupyter kernel: $DISPLAY_NAME ($KERNEL_NAME)"
echo "Kernel python: $(python -c 'import sys; print(sys.executable)')"
