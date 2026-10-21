#!/usr/bin/env python3
"""Provision and run Linux CUDA checks, without Colab. Dependencies come from pyproject."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib

REPO = Path(__file__).resolve().parents[1]


def cache_key(project):
    profile = {key: project[key] for key in ('build-system', 'project', 'dependency-groups', 'tool')}
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:24]
    return 'triton-cpu-ubuntu22-cp312-v1-' + digest


def run(command, env=None):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=REPO, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['cache-key', 'prepare-wheels', 'cuda', 'parity'])
    parser.add_argument('--work-dir', type=Path, default=REPO / '.gpu-ci')
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1.1e-3)
    args = parser.parse_args()
    project = tomllib.loads((REPO / 'pyproject.toml').read_text())
    if args.action == 'cache-key':
        print(cache_key(project))
        return
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('Provisioning requires Linux x86_64 with Python 3.12 and uv installed')
    uv = shutil.which('uv')
    if not uv:
        parser.error('Install uv first: python3 -m pip install uv')
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    cpu = work / 'cpu'
    prepare = [sys.executable, REPO / 'cross-platform-install/prepare_parity_cpu.py',
               '--environment', cpu]
    if args.action == 'prepare-wheels':
        run(prepare + ['--wheel-only'])
        return
    env = dict(os.environ)
    for key in ('VLLM_TARGET_DEVICE', 'TRITON_DEFAULT_BACKEND', 'TRITON_DISABLE_OPENMP',
                'TRITON_LOCAL_LIBOMP_PATH', 'LD_PRELOAD', 'DYLD_INSERT_LIBRARIES',
                'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_PLUGINS'):
        env.pop(key, None)
    cuda = work / 'cuda'
    python = cuda / 'bin/python'
    if not python.exists():
        run([uv, 'venv', '--python', '3.12', cuda], env)
    # pip-compatible resolution uses the CUDA extra, not CPU-specific uv sources.
    run([uv, 'pip', 'install', '--python', python, '-e', str(REPO) + '[cuda]',
         '--group', str(REPO / 'pyproject.toml') + ':test',
         '--only-binary', 'vllm', '--only-binary', 'torch', '--only-binary', 'triton'], env)
    run([uv, 'pip', 'uninstall', '--python', python, 'torchcodec'], env)
    run([python, '-c',
         'import torch; from vllm.platforms import current_platform; '
         'assert torch.cuda.is_available() and current_platform.is_cuda(), "CUDA unavailable"; '
         'print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), '
         'torch.cuda.get_device_capability()); '
         'print((torch.ones(8, device="cuda") + 1).cpu())'], env)
    results = work / 'results'
    results.mkdir(exist_ok=True)
    if args.action == 'parity':
        run(prepare, env)
        run([python, REPO / 'cross-platform-install/run_parity.py',
             '--cpu-python', cpu / 'bin/python', '--cuda-python', python,
             '--output', results, '--atol', args.atol, '--rtol', args.rtol], env)
    else:
        env['MIA_REQUIRED_BACKENDS'] = 'cuda'
        env['MIA_BEHAVIOR_HOOK_DIR'] = str(work / 'hooks')
        run([python, '-m', 'pytest', 'tests/cpu_behavioral/test_vllm_cpu_fixtures.py',
             '-k', 'test_vllm_demo_scenario and cuda', '-vv', '-s',
             '--basetemp', results / 'pytest', '--junitxml', results / 'junit.xml'], env)


if __name__ == '__main__':
    main()
