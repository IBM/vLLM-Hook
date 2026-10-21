#!/usr/bin/env python3
"""Run CPU and CUDA on this machine, then compare their numerical observations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / 'tests' / 'cpu_behavioral'
sys.path.insert(0, str(CASES))
from parity_results import compare_results

SCENARIOS = ('hidden_states', 'attention_tracker', 'core_reranker', 'activation_steer')


def environment(backend, python):
    env = dict(os.environ)
    for key in ('VLLM_TARGET_DEVICE', 'TRITON_DEFAULT_BACKEND', 'TRITON_DISABLE_OPENMP',
                'TRITON_LOCAL_LIBOMP_PATH', 'DYLD_INSERT_LIBRARIES', 'GLOO_SOCKET_IFNAME',
                'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_PLUGINS', 'LD_PRELOAD', 'PYTHONPATH'):
        env.pop(key, None)
    env.update(MIA_BEHAVIOR_BACKEND=backend, MIA_WRITER_PROCESS='0',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONPATH=str(REPO),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if backend == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='', VLLM_TARGET_DEVICE='cpu',
                   TRITON_DEFAULT_BACKEND='cpu', OMP_NUM_THREADS='1',
                   VLLM_CPU_OMP_THREADS_BIND='nobind', VLLM_CPU_KVCACHE_SPACE='1')
        prefix = Path(python).absolute().parent.parent
        libraries = list(prefix.glob('lib/libiomp5.so'))
        libraries += list(prefix.glob('lib/python*/site-packages/**/libiomp5.so'))
        if libraries:
            env['LD_PRELOAD'] = str(libraries[0])
        if sys.platform == 'darwin':
            env['GLOO_SOCKET_IFNAME'] = 'lo0'
    return env


def run_logged(command, env, path, timeout):
    with path.open('w') as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        tail = '\n'.join(path.read_text(errors='replace').splitlines()[-30:])
        raise RuntimeError(f'Process failed ({code}); log: {path}\n{tail}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-python', required=True)
    parser.add_argument('--cuda-python', default=sys.executable)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1.1e-3)
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds per scenario')
    args = parser.parse_args()
    if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error('Tolerances must be finite and nonnegative')
    result_dir = args.output or Path(tempfile.mkdtemp(prefix='mlr55-parity-'))
    result_dir.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'atol': args.atol, 'rtol': args.rtol,
              'scope': 'OPT-125m, float32, eager mode; strict ranking equality',
              'tolerances_validated_on_hardware': False, 'scenarios': {}}
    report_path = result_dir / 'report.json'
    print('Parity reports:', result_dir, flush=True)
    print(f'Fixed tolerances: atol={args.atol}, rtol={args.rtol} (initial, uncalibrated)', flush=True)
    try:
        # Resolve/download once, using CUDA's working hub client. Both backends
        # subsequently load exactly this immutable, local model snapshot.
        model_file = result_dir / 'model.json'
        download = '''
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
model = 'facebook/opt-125m'
revision = HfApi().model_info(model).sha
path = snapshot_download(model, revision=revision,
                         allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors', '*.bin'],
                         ignore_patterns=['*optimizer*', '*training*'])
Path(OUTPUT).write_text(json.dumps({'model_id': model, 'model_revision': revision, 'model_path': path}))
'''.replace('OUTPUT', repr(str(model_file)))
        env = dict(os.environ)
        env.pop('HF_HUB_OFFLINE', None)
        env.pop('TRANSFORMERS_OFFLINE', None)
        run_logged([args.cuda_python, '-c', download], env, result_dir / 'download.log', args.timeout)
        manifest = json.loads(model_file.read_text())
        digest = hashlib.sha256()
        files = list((REPO / 'mia').rglob('*.py')) + list(CASES.glob('*.py'))
        for path in sorted(files):
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(path.read_bytes())
        manifest['source_sha256'] = digest.hexdigest()
        manifest['run_id'] = result_dir.resolve().name
        manifest_path = result_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        for scenario in SCENARIOS:
            for backend, python in (('cpu', args.cpu_python), ('cuda', args.cuda_python)):
                print(f'Running {scenario} on {backend}...', flush=True)
                output = result_dir / f'{backend}-{scenario}.json'
                if output.exists():
                    raise RuntimeError(f'Refusing stale results: {output}; choose a new output directory')
                run_logged([python, '-u', str(CASES / 'parity_case.py'),
                            '--scenario', scenario, '--backend', backend,
                            '--manifest', str(manifest_path), '--output', str(output)],
                           environment(backend, python), result_dir / f'{backend}-{scenario}.log',
                           args.timeout)
            cpu = json.loads((result_dir / f'cpu-{scenario}.json').read_text())
            cuda = json.loads((result_dir / f'cuda-{scenario}.json').read_text())
            comparison = compare_results(cpu, cuda, atol=args.atol, rtol=args.rtol)
            report['scenarios'][scenario] = comparison
            print(scenario + ': ' + ('PASS' if comparison['passed'] else 'FAIL'), flush=True)
            report_path.write_text(json.dumps(report, indent=2))
        report['passed'] = all(v['passed'] for v in report['scenarios'].values())
    except Exception as error:
        report['error'] = str(error)
        print(str(error), file=sys.stderr, flush=True)
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Comparison report:', report_path, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
