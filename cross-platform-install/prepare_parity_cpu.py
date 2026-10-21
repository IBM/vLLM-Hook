#!/usr/bin/env python3
"""Install the pyproject CPU profile using prebuilt vLLM and a reusable Triton wheel."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request

from wheel_cache import export_chunks, sha256, validate_wheel


@contextmanager
def linux_cpu_resolution(repo, wheel, vllm_wheel=None):
    """Use a local Triton wheel for this Linux install; restore project settings."""
    path = repo / 'pyproject.toml'
    original = path.read_text()
    if 'environments' in tomllib.loads(original)['tool']['uv']:
        raise RuntimeError('Project already restricts uv environments; reconcile that setting first')
    scoped = original.replace('[tool.uv]\n', '[tool.uv]\n'
        'environments = ["sys_platform == \'linux\' and platform_machine == \'x86_64\' '
        'and python_version == \'3.12\'"]\n', 1)
    scoped, count = re.subn(r'(?ms)^triton = \[\n.*?^\]',
        lambda match: 'triton = [\n    { path = ' + json.dumps(str(wheel)) + ', extra = "cpu" },\n]',
        scoped, count=1)
    if count != 1 or scoped == original:
        raise RuntimeError('Could not locate project uv source settings')
    if vllm_wheel is not None:
        scoped, count = re.subn(r'(?ms)^vllm = \[\n.*?^\]',
            lambda match: 'vllm = [\n    { path = ' + json.dumps(str(vllm_wheel)) + ', extra = "cpu" },\n]',
            scoped, count=1)
        if count != 1:
            raise RuntimeError('Could not locate vLLM source settings')
    try:
        path.write_text(scoped)
        yield
    finally:
        path.write_text(original)


def release_wheel(project, destination):
    source = next(s for s in project['tool']['uv']['sources']['vllm'] if 'url' in s)
    url = source['url']
    if urllib.parse.urlsplit(url).fragment:
        raise ValueError('uv source URLs must not contain fragments')
    expected = project['tool']['mia']['colab']['vllm-cpu-wheel-sha256']
    destination.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name
    wheel = destination / filename
    if wheel.exists():
        if sha256(wheel) != expected:
            raise ValueError('Cached vLLM wheel checksum differs from pyproject')
        return wheel
    temporary = destination / (filename + '.partial')
    print('Downloading official prebuilt vLLM CPU wheel...', flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if sha256(temporary) != expected:
            raise ValueError('Downloaded vLLM wheel checksum differs from pyproject')
        temporary.replace(wheel)
    finally:
        temporary.unlink(missing_ok=True)
    print('Official vLLM wheel checksum verified.', flush=True)
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', default='/content/mlr55-cpu')
    parser.add_argument('--triton-wheel', type=Path)
    parser.add_argument('--wheel-only', action='store_true', help='Prepare/export Triton wheel; do not install vLLM or run tests')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('Automatic CPU provisioning supports Linux x86_64 only')
    repo = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repo / 'pyproject.toml').read_text())
    source = next(s for s in project['tool']['uv']['sources']['triton'] if s.get('extra') == 'cpu')
    revision = source['rev']
    cpu = Path(args.environment)
    wheels = cpu.parent / 'mlr55-wheel-cache' / revision
    wheels.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ('VIRTUAL_ENV', 'UV_SYSTEM', 'TRITON_DISABLE_OPENMP', 'TRITON_LOCAL_LIBOMP_PATH',
                'DYLD_INSERT_LIBRARIES', 'GLOO_SOCKET_IFNAME', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(UV_PROJECT_ENVIRONMENT=str(cpu), VLLM_TARGET_DEVICE='cpu',
               CUDA_VISIBLE_DEVICES='', TRITON_DEFAULT_BACKEND='cpu',
               MAX_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2')

    def run(command):
        print('CPU setup:', ' '.join(str(v) for v in command), flush=True)
        subprocess.run(command, cwd=repo, env=env, check=True)

    wheel = args.triton_wheel
    if wheel:
        validate_wheel(wheel, revision)
    else:
        for candidate in sorted(wheels.glob('*.whl')):
            validate_wheel(candidate, revision)
            wheel = candidate
            break
    # A restored wheel can be backed up without touching either Python runtime.
    if args.wheel_only and wheel:
        export_chunks(wheel, revision, cpu.parent / 'mlr55-wheel-transfer')
        print('Triton CPU wheel is ready for local backup.', flush=True)
        return

    uv = shutil.which('uv')
    if not uv:
        run([sys.executable, '-m', 'pip', 'install', 'uv'])
        uv = shutil.which('uv')
        if not uv:
            raise RuntimeError('uv installed but its executable is not on PATH')
    if not (cpu / 'bin/python').exists():
        run([uv, 'venv', '--python', '3.12', str(cpu)])
    python = str(cpu / 'bin/python')
    torch_requirement = next(r for r in project['project']['optional-dependencies']['cpu'] if r.startswith('torch=='))
    torch_source = next(s for s in project['tool']['uv']['sources']['torch'] if s.get('extra') == 'cpu')
    index = next(i['url'] for i in project['tool']['uv']['index'] if i['name'] == torch_source['index'])

    if not wheel:
        # Only Triton is compiled. Its Python build dependencies still come from
        # the project; vLLM is required to use the official binary CPU wheel.
        if not shutil.which('g++') or not shutil.which('git'):
            if os.geteuid() != 0 or not shutil.which('apt-get'):
                raise RuntimeError('Install g++ and git, then retry')
            run(['apt-get', 'update', '-qq'])
            run(['apt-get', 'install', '-y', 'g++', 'git'])
        run([uv, 'pip', 'install', '--python', python, *project['dependency-groups']['cpu-build']])
        run([uv, 'pip', 'install', '--python', python, '--index-url', index, torch_requirement])
        checkout = cpu.parent / ('mlr55-triton-' + revision)
        if not checkout.exists():
            run(['git', 'init', str(checkout)])
            run(['git', '-C', str(checkout), 'remote', 'add', 'origin', source['git']])
        run(['git', '-C', str(checkout), 'fetch', '--depth', '1', 'origin', revision])
        run(['git', '-C', str(checkout), 'checkout', '--detach', 'FETCH_HEAD'])
        print('Building Triton CPU once. vLLM will NOT be compiled.', flush=True)
        run([uv, 'build', '--wheel', '--python', python, '--no-build-isolation',
             '--out-dir', str(wheels), str(checkout)])
        candidates = list(wheels.glob('*.whl'))
        if len(candidates) != 1:
            raise RuntimeError('Expected exactly one Triton wheel')
        wheel = candidates[0]
        validate_wheel(wheel, revision)
    if args.wheel_only:
        export_chunks(wheel, revision, cpu.parent / 'mlr55-wheel-transfer')
        print('Triton CPU wheel built. Back it up before installing/running tests.', flush=True)
        return

    # Read the published vLLM wheel source from pyproject, never fall back to Git.
    vllm_sources = project['tool']['uv']['sources']['vllm']
    if not any('url' in s and 'x86_64.whl' in s['url'] for s in vllm_sources):
        raise RuntimeError('pyproject must select an official Linux x86_64 vLLM CPU wheel')
    vllm_wheel = release_wheel(project, cpu.parent / 'mlr55-vllm-wheel')
    run([uv, 'pip', 'install', '--python', python, *project['dependency-groups']['cpu-build']])
    run([uv, 'pip', 'install', '--python', python, '--index-url', index, torch_requirement])
    with linux_cpu_resolution(repo, wheel, vllm_wheel):
        run([uv, 'sync', '--project', str(repo), '--python', '3.12', '--extra', 'cpu',
             '--group', 'parity', '--group', 'cpu-build', '--no-build-isolation',
             '--no-build-package', 'vllm', '--no-build-package', 'triton'])
    run([uv, 'pip', 'uninstall', '--python', python, 'torchcodec'])
    libraries = list(cpu.glob('lib/libiomp5.so'))
    libraries += list(cpu.glob('lib/python*/site-packages/**/libiomp5.so'))
    if libraries:
        env['LD_PRELOAD'] = str(libraries[0])
    run([python, '-c', 'from vllm.platforms import current_platform; '
         'from triton.backends import backends; assert current_platform.is_cpu(); '
         'assert "cpu" in backends; print("CPU software is ready")'])


if __name__ == '__main__':
    main()
