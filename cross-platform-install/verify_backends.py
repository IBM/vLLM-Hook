"""Run CPU scenarios and guarded Metal generation concurrently on one Mac."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def stop(process):
    if process is None:
        return
    # Children may remain even after the parent exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--sandbox', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--cpu-timeout', type=float, default=2400)
    args = parser.parse_args()
    repo, sandbox = args.repo.resolve(), args.sandbox.resolve()
    cpu_python = repo / '.venv-install-test-cpu/bin/python'
    metal_python = sandbox / 'vllm-metal/.venv-vllm-metal/bin/python'
    for path in (cpu_python, metal_python):
        if not path.is_file():
            parser.error(f'Missing interpreter: {path}')
    # Fail before launching either backend if another server owns the port.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    run = Path(tempfile.mkdtemp(prefix='backend-run.', dir=sandbox))
    (run / 'triton-cache').mkdir()
    print(f'Logs and results: {run}', flush=True)
    cpu_env = dict(os.environ)
    metal_env = dict(os.environ)
    for env in (cpu_env, metal_env):
        for key in ('TRITON_LOCAL_LIBOMP_PATH', 'DYLD_INSERT_LIBRARIES'):
            env.pop(key, None)
    cpu_env.update(GLOO_SOCKET_IFNAME='lo0', VLLM_TARGET_DEVICE='cpu',
                   MIA_REQUIRED_BACKENDS='cpu', VLLM_USE_V2_MODEL_RUNNER='1',
                   TRITON_CACHE_DIR=str(run / 'triton-cache'))
    for key in ('VLLM_TARGET_DEVICE', 'TRITON_DEFAULT_BACKEND', 'TRITON_DISABLE_OPENMP'):
        metal_env.pop(key, None)
    metal_env['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    cpu = metal = None
    summary = {'scope': 'concurrent backend smoke checks; not hook parity',
               'cpu_exit_status': None, 'metal_request_passed': False}
    started = time.monotonic()
    try:
        with (run / 'cpu.log').open('w') as cpu_log, (run / 'metal.log').open('w') as metal_log:
            cpu = subprocess.Popen(
                [str(cpu_python), '-m', 'pytest', 'tests/cpu_behavioral/test_vllm_cpu_fixtures.py',
                 f'--basetemp={run / "cpu-pytest"}', '-vv', '-rs'],
                cwd=repo, env=cpu_env, stdout=cpu_log, stderr=subprocess.STDOUT,
                start_new_session=True)
            metal = subprocess.Popen(
                [str(metal_python), str(repo / 'cross-platform-install/serve_metal_checked.py'),
                 'serve', 'mlx-community/Qwen2.5-0.5B-Instruct-4bit',
                 '--served-model-name', 'metal-smoke', '--max-model-len', '512',
                 '--host', '127.0.0.1', '--port', str(args.port)],
                cwd=sandbox / 'vllm-metal', env=metal_env, stdout=metal_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            base = f'http://127.0.0.1:{args.port}'
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if metal.poll() is not None:
                        raise RuntimeError(f'Metal exited {metal.returncode}; inspect metal.log')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Metal readiness timeout; inspect metal.log')
                    try:
                        with urllib.request.urlopen(base + '/health', timeout=2):
                            break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(1)
                payload = json.dumps(dict(model='metal-smoke', prompt='The capital of France is',
                                          max_tokens=16, temperature=0)).encode()
                request = urllib.request.Request(base + '/v1/completions', data=payload,
                                                  headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                (run / 'metal-response.json').write_text(json.dumps(result, indent=2) + '\n')
                text = result['choices'][0]['text']
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError('Metal returned no generated text')
                if 'Verified platform: MetalPlatform' not in (run / 'metal.log').read_text():
                    raise RuntimeError('Metal platform confirmation missing')
                summary['metal_request_passed'] = True
                print(f'Metal generated: {text!r}', flush=True)
            except Exception as exc:
                summary['metal_error'] = str(exc)
                print(f'Metal failed: {exc}', flush=True)
            # Keep the server alive while CPU scenarios finish, matching the
            # concurrent verification procedure, then clean up in finally.
            remaining = max(0, args.cpu_timeout - (time.monotonic() - started))
            try:
                summary['cpu_exit_status'] = cpu.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                summary['cpu_error'] = 'CPU suite timed out'
    finally:
        stop(cpu)
        stop(metal)
        (run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['cpu_exit_status'] == 0 and summary['metal_request_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
