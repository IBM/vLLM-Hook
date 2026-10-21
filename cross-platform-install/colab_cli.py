#!/usr/bin/env python3
"""Operate an existing Colab session from the local MLR55 checkout."""
import argparse
import json
import tomllib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from wheel_cache import export_chunks, restore_chunks, sha256, validate_wheel


def remote_code(action, atol=1e-4, rtol=1.1e-3, cached=None):
    common = '''
from pathlib import Path
import os
import subprocess
import sys

env = dict(os.environ)
for key in ('VLLM_TARGET_DEVICE', 'TRITON_DEFAULT_BACKEND',
            'TRITON_DISABLE_OPENMP', 'TRITON_LOCAL_LIBOMP_PATH',
            'DYLD_INSERT_LIBRARIES', 'GLOO_SOCKET_IFNAME',
            'VLLM_USE_V2_MODEL_RUNNER'):
    env.pop(key, None)

def run(args, cwd=None):
    process = subprocess.Popen(args, cwd=cwd, env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait():
        raise RuntimeError('Remote command failed; see output above')
'''
    if action == 'sync':
        return common + '''
import tarfile
import tempfile
repo = Path(tempfile.mkdtemp(prefix='mlr55-src-', dir='/content'))
with tarfile.open('/content/mlr55-source.tar.gz') as archive:
    archive.extractall(repo, filter='data')
run([sys.executable, '-m', 'pip', 'install', '-e', str(repo), '--no-deps'])
Path('/content/mlr55-source-path.txt').write_text(str(repo))
print('Local changes synced:', repo, flush=True)
'''
    if action == 'install':
        return common + '''
repo = Path('/content/mlr55-source-path.txt').read_text().strip()
run([sys.executable, '-m', 'pip', 'install', '-e', repo + '[cuda]'])
run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchcodec'])
import tomllib
project = tomllib.loads((Path(repo) / 'pyproject.toml').read_text())
run([sys.executable, '-m', 'pip', 'install', *project['dependency-groups']['test']])
run([sys.executable, '-m', 'pip', 'install', '-e', repo, '--no-deps'])
print('GPU software installed. Run the check command next.', flush=True)
'''
    if action == 'wheel-status':
        return common + """
import json
repo = Path('/content/mlr55-source-path.txt').read_text().strip()
sys.path.insert(0, repo + '/cross-platform-install')
from wheel_cache import sha256
cached = CACHED
wheel = Path('/content/mlr55-wheel-cache') / cached['revision'] / cached['filename']
present = wheel.is_file() and sha256(wheel) == cached['sha256']
Path('/content/mlr55-wheel-status.json').write_text(json.dumps({'present': present}))
""".replace('CACHED', repr(cached))
    if action == 'prepare-wheel':
        return common + """
import json
import platform
import tomllib
repo = Path('/content/mlr55-source-path.txt').read_text().strip()
sys.path.insert(0, repo + '/cross-platform-install')
from wheel_cache import restore_chunks, sha256
project = tomllib.loads((Path(repo) / 'pyproject.toml').read_text())
revision = project['tool']['uv']['sources']['triton'][0]['rev']
cached = CACHED
arguments = []
if cached:
    if cached['revision'] != revision:
        raise RuntimeError('Cached Triton revision differs from pyproject')
    old_libc = cached.get('libc', ['', ''])
    new_libc = platform.libc_ver()
    if old_libc[0] == new_libc[0] == 'glibc':
        if tuple(map(int, new_libc[1].split('.'))) < tuple(map(int, old_libc[1].split('.'))):
            raise RuntimeError('Cached wheel needs a newer Linux glibc; use a separate --cache-dir')
    if cached.get('existing'):
        wheel = Path('/content/mlr55-wheel-cache') / revision / cached['filename']
        if not wheel.is_file() or sha256(wheel) != cached['sha256']:
            raise RuntimeError('Remote cached wheel changed; retry preparation')
    else:
        wheel = restore_chunks(cached, [c['path'] for c in cached['chunks']],
                               Path('/content/mlr55-wheel-cache') / revision)
    arguments = ['--triton-wheel', str(wheel)]
run([sys.executable, '-u', repo + '/cross-platform-install/prepare_parity_cpu.py',
     '--wheel-only', *arguments])
""".replace('CACHED', repr(cached))
    if action == 'prepare-cpu':
        return common + """
import hashlib
import json
repo = Path('/content/mlr55-source-path.txt').read_text().strip()
manifest = json.loads(Path('/content/mlr55-wheel-transfer/manifest.json').read_text())
wheel = Path('/content/mlr55-wheel-cache') / manifest['revision'] / manifest['filename']
# A wheel freshly built by the preparer is already here; a restored one is too.
profile = hashlib.sha256(b''.join((Path(repo) / name).read_bytes() for name in
    ['pyproject.toml', 'cross-platform-install/prepare_parity_cpu.py',
     'cross-platform-install/wheel_cache.py'])).hexdigest()
ready = Path('/content/mlr55-cpu-profile.txt')
if not Path('/content/mlr55-cpu/bin/python').exists() or not ready.exists() or ready.read_text() != profile:
    print('Installing CPU software from wheels. No vLLM compilation.', flush=True)
    run([sys.executable, '-u', repo + '/cross-platform-install/prepare_parity_cpu.py',
         '--triton-wheel', str(wheel)])
    ready.write_text(profile)
else:
    print('Reusing CPU environment.', flush=True)
"""
    if action == 'prepare-gpu':
        install = remote_code('install')
        return common + """
from importlib.metadata import version, PackageNotFoundError
import tomllib
from packaging.requirements import Requirement
repo = Path('/content/mlr55-source-path.txt').read_text().strip()
project = tomllib.loads((Path(repo) / 'pyproject.toml').read_text())
needs_install = False
for item in project['project']['optional-dependencies']['cuda']:
    requirement = Requirement(item)
    try:
        if version(requirement.name) not in requirement.specifier:
            needs_install = True
    except PackageNotFoundError:
        needs_install = True
if needs_install:
    print('Installing the pyproject CUDA profile in this new session.', flush=True)
    exec(INSTALL)
else:
    print('Reusing CUDA software.', flush=True)
""".replace('INSTALL', repr(install))
    if action == 'parity':
        return common + """
import shutil
import tempfile
repo = Path('/content/mlr55-source-path.txt').read_text().strip()
cpu = Path('/content/mlr55-cpu/bin/python')
run([sys.executable, '-m', 'pip', 'install', '-e', repo, '--no-deps'])
run([shutil.which('uv'), 'pip', 'install', '--python', str(cpu), '-e', repo, '--no-deps'])
results = tempfile.mkdtemp(prefix='mlr55-parity-', dir='/content')
Path('/content/mlr55-last-parity-path.txt').write_text(results)
report = Path('/content/mlr55-parity-report.json')
report.unlink(missing_ok=True)
try:
    run([sys.executable, '-u', repo + '/cross-platform-install/run_parity.py',
         '--cpu-python', str(cpu), '--cuda-python', sys.executable,
         '--output', results, '--atol', ATOL, '--rtol', RTOL])
finally:
    if (Path(results) / 'report.json').exists():
        shutil.copyfile(Path(results) / 'report.json', report)
""".replace('ATOL', repr(str(atol))).replace('RTOL', repr(str(rtol)))
    if action == 'check':
        probe = '''
from importlib.metadata import version
import socket
import torch
from vllm.platforms import current_platform
import mia
print('Machine:', socket.gethostname())
for name in ('torch', 'vllm', 'triton', 'vllm-hook-mia'):
    print(name, version(name))
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert current_platform.is_cuda(), 'vLLM did not select CUDA'
print('GPU:', torch.cuda.get_device_name(0))
print('MIA source:', mia.__file__)
print('GPU check passed.')
'''
        return common + '\nrun([sys.executable, "-u", "-c", ' + repr(probe) + '])\n'
    return common + '''
import tempfile
repo = Path('/content/mlr55-source-path.txt').read_text().strip()
results = Path(tempfile.mkdtemp(prefix='mlr55-results-', dir='/content'))
env['MIA_REQUIRED_BACKENDS'] = 'cuda'
env['MIA_BEHAVIOR_MODEL'] = 'facebook/opt-125m'
env.pop('VLLM_PLUGINS', None)
print('Reports:', results, flush=True)
print('Running CUDA execution checks only. Use parity for comparisons.', flush=True)
run([sys.executable, '-u', '-m', 'pytest',
     'tests/cpu_behavioral/test_vllm_cpu_fixtures.py',
     '-k', 'test_vllm_demo_scenario', '-vv', '-rs',
     '--basetemp', str(results / 'tmp'),
     '--junitxml', str(results / 'cuda.xml')], cwd=repo)
'''


def completion_wrapper(code, status_file):
    # Some colab exec versions print a remote traceback but still exit zero.
    # A unique remote status file is the authority for each individual step.
    return ("__mlr55_state = {'ok': False}\n"
            "try:\n"
            "    exec(" + repr(code) + ")\n"
            "    __mlr55_state['ok'] = True\n"
            "except BaseException as __mlr55_error:\n"
            "    __mlr55_state['error'] = str(__mlr55_error)\n"
            "    raise\n"
            "finally:\n"
            "    __import__('pathlib').Path(" + repr(str(status_file)) + ").write_text("
            "__import__('json').dumps(__mlr55_state))\n")


def execute_remote(base, session, code, timeout):
    import uuid
    marker = '/content/mlr55-command-' + uuid.uuid4().hex + '.json'
    result = subprocess.run(base + ['exec', '-s', session, '--timeout', str(timeout)],
                            input=completion_wrapper(code, marker), text=True)
    with tempfile.TemporaryDirectory(prefix='mlr55-status-') as temp:
        status_file = Path(temp) / 'status.json'
        download = subprocess.run(base + ['download', '-s', session, marker, str(status_file)])
        if download.returncode or not status_file.exists():
            raise RuntimeError('Could not verify the remote step completed. Keep the session and inspect its output.')
        status = json.loads(status_file.read_text())
    if not status.get('ok'):
        raise RuntimeError('Remote step failed: ' + status.get('error', 'unknown error'))
    if result.returncode:
        print('Remote step completed; its final CLI reply failed, but completion was verified.', flush=True)


def backup_wheel(base, session, cache, revision):
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mlr55-backup-') as temp:
        manifest_path = Path(temp) / 'manifest.json'
        subprocess.run(base + ['download', '-s', session,
                       '/content/mlr55-wheel-transfer/manifest.json', str(manifest_path)], check=True)
        manifest = json.loads(manifest_path.read_text())
        if manifest['revision'] != revision:
            raise ValueError('Remote wheel revision differs from pyproject')
        parts = []
        for number, item in enumerate(manifest['chunks']):
            expected = f'/content/mlr55-wheel-transfer/triton-{number:04d}.part'
            if item['path'] != expected:
                raise ValueError('Unexpected wheel chunk path')
            part = Path(temp) / f'{number}.part'
            subprocess.run(base + ['download', '-s', session, expected, str(part)], check=True)
            parts.append(part)
        wheel = restore_chunks(manifest, parts, cache)
        (cache / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print('Triton CPU wheel safely saved on your Mac:', wheel, flush=True)


def upload_cached_wheel(base, session, cache, revision):
    manifest_path = cache / 'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest['revision'] != revision or Path(manifest['filename']).name != manifest['filename']:
        raise ValueError('Invalid cached wheel manifest')
    wheel = cache / manifest['filename']
    if sha256(wheel) != manifest['sha256']:
        raise ValueError('Local wheel backup is damaged; use another --cache-dir')
    validate_wheel(wheel, revision)
    with tempfile.TemporaryDirectory(prefix='mlr55-wheel-status-') as temp:
        execute_remote(base, session, remote_code('wheel-status', cached=manifest), 180)
        status_file = Path(temp) / 'status.json'
        subprocess.run(base + ['download', '-s', session,
                       '/content/mlr55-wheel-status.json', str(status_file)], check=True)
        if json.loads(status_file.read_text())['present']:
            print('Reusing the Triton CPU wheel already in this session.', flush=True)
            return dict(manifest, existing=True)
    print('Restoring the saved Triton CPU wheel. No native rebuild needed.', flush=True)
    with tempfile.TemporaryDirectory(prefix='mlr55-restore-') as temp:
        transfer = export_chunks(wheel, revision, Path(temp))
        transfer['libc'] = manifest.get('libc', ['', ''])
        for number, item in enumerate(transfer['chunks']):
            remote = f'/content/mlr55-upload-{number:04d}.part'
            subprocess.run(base + ['upload', '-s', session, item['path'], remote], check=True)
            item['path'] = remote
    return transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'sync', 'install', 'test', 'prepare', 'parity'),
                        help='prepare backs up CPU wheels before tests; parity prepares and compares CPU/CUDA')
    parser.add_argument('--session', default='mlr55')
    parser.add_argument('--cache-dir', type=Path, help='Local wheel backup folder; defaults to .colab-cache in the checkout')
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1.1e-3)
    args = parser.parse_args()
    import math
    if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error('Tolerances must be finite and nonnegative')
    if not shutil.which('colab'):
        parser.error('Colab CLI is missing. Run: uv tool install google-colab-cli')
    repo = Path(__file__).resolve().parents[1]
    base = ['colab', '--auth', 'oauth2']

    def execute(action, timeout, cached=None):
        execute_remote(base, args.session, remote_code(action, args.atol, args.rtol, cached), timeout)

    if args.action in ('sync', 'prepare', 'parity'):
        if not (repo / 'mia').is_dir():
            parser.error('Keep this script inside the checkout’s cross-platform-install folder')
        with tempfile.TemporaryDirectory(prefix='mlr55-upload-') as temp:
            bundle = Path(temp) / 'source.tar.gz'
            excluded = {'__pycache__', '.pytest_cache', '.hooks', '.git'}
            with tarfile.open(bundle, 'w:gz') as archive:
                for name in ('mia', 'tests', 'examples', 'model_configs',
                             'steering_vectors', 'cross-platform-install',
                             'setup.py', 'pyproject.toml', 'requirement.txt'):
                    path = repo / name
                    files = sorted(path.rglob('*')) if path.is_dir() else [path]
                    for item in files:
                        relative = item.relative_to(repo)
                        if item.is_file() and not excluded.intersection(relative.parts):
                            archive.add(item, arcname=str(relative), recursive=False)
            subprocess.run(base + ['upload', '-s', args.session, str(bundle),
                                   '/content/mlr55-source.tar.gz'], check=True)
        execute('sync', 600)
        if args.action == 'sync':
            return
    if args.action in ('prepare', 'parity'):
        if args.action == 'parity':
            execute('prepare-gpu', 3600)
            execute('check', 180)
        project = tomllib.loads((repo / 'pyproject.toml').read_text())
        revision = project['tool']['uv']['sources']['triton'][0]['rev']
        cache = (args.cache_dir or repo / '.colab-cache') / revision
        cached = upload_cached_wheel(base, args.session, cache, revision)
        execute('prepare-wheel', 14400, cached)
        # This must succeed before CPU installation or tests can begin.
        if cached is None:
            backup_wheel(base, args.session, cache, revision)
        else:
            print('Verified local CPU wheel backup:', cache, flush=True)
        execute('prepare-cpu', 3600)
        if args.action == 'prepare':
            print('CPU prepared and backed up. Run parity when ready.', flush=True)
            return
        try:
            execute('parity', 15000)
        finally:
            reports = cache.parent / 'reports'
            reports.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            report = reports / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-report.json')
            result = subprocess.run(base + ['download', '-s', args.session,
                            '/content/mlr55-parity-report.json', str(report)])
            if result.returncode:
                print('Could not save a local comparison report. The wheel backup is safe.', file=sys.stderr)
            else:
                print('Comparison report saved:', report, flush=True)
        return
    execute(args.action, {'check': 180, 'install': 3600, 'test': 7500}[args.action])


if __name__ == '__main__':
    try:
        main()
    except (subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        print('Stopped:', error, file=sys.stderr)
        print('Keep the session. Completed local wheel backups are preserved.', file=sys.stderr)
        sys.exit(getattr(error, 'returncode', 1) or 1)
