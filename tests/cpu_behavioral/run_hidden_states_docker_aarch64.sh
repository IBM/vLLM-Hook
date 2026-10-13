#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image_name="${MIA_CPU_DOCKER_IMAGE:-vllm-hook-mia-cpu-aarch64}"
build_jobs="${MIA_CPU_DOCKER_BUILD_JOBS:-2}"

docker build \
  --platform linux/arm64 \
  --build-arg BUILD_JOBS="${build_jobs}" \
  -f "${repo_root}/tests/cpu_behavioral/Dockerfile.aarch64" \
  -t "${image_name}" \
  "${repo_root}"

docker run --rm \
  --platform linux/arm64 \
  -e TRITON_DEFAULT_BACKEND=cpu \
  -e TRITON_CACHE_DIR=/tmp/triton-cache \
  -e MIA_WRITER_PROCESS=0 \
  -e VLLM_RPC_TIMEOUT=300 \
  "${image_name}"
