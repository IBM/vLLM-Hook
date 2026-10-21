"""Check provisioning routes without installing GPU packages on this host."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'cross-platform-install/gpu_ci.py'
spec = importlib.util.spec_from_file_location('gpu_ci', SCRIPT)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class GPUFlowTests(unittest.TestCase):
    def invoke(self, action, fail=None):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / 'pyproject.toml').write_text('[project]\nname="test"\n')
            calls = []
            def record(command, env=None):
                calls.append(([str(v) for v in command], env))
                if fail and fail(command):
                    raise RuntimeError('injected failure')
            with patch.object(gpu, 'REPO', repo), patch.object(gpu.platform, 'system', return_value='Linux'), \
                 patch.object(gpu.platform, 'machine', return_value='x86_64'), \
                 patch.object(gpu.shutil, 'which', return_value='/bin/uv'), \
                 patch.object(gpu, 'run', side_effect=record), \
                 patch('sys.argv', ['gpu_ci.py', action, '--work-dir', str(repo / '.gpu-ci')]):
                gpu.main()
            return calls

    def test_cuda_does_not_build_cpu_and_requires_cuda(self):
        calls = self.invoke('cuda')
        self.assertFalse(any('prepare_parity_cpu.py' in ' '.join(cmd) for cmd, _ in calls))
        cmd, env = calls[-1]
        self.assertIn('pytest', cmd)
        self.assertEqual(env['MIA_REQUIRED_BACKENDS'], 'cuda')
        self.assertIn('test_vllm_demo_scenario and cuda', cmd)
        install = next(cmd for cmd, _ in calls if 'install' in cmd)
        self.assertIn('--only-binary', install)
        self.assertTrue(any(arg.endswith('pyproject.toml:test') for arg in install))

    def test_parity_installs_cpu_before_comparison(self):
        calls = self.invoke('parity')
        self.assertIn('prepare_parity_cpu.py', ' '.join(calls[-2][0]))
        cmd = calls[-1][0]
        self.assertIn('run_parity.py', ' '.join(cmd))
        self.assertIn('0.0011', cmd)
        self.assertIn('--cpu-python', cmd)
        self.assertIn('--cuda-python', cmd)

    def test_wheel_stage_does_not_touch_cuda(self):
        calls = self.invoke('prepare-wheels')
        self.assertEqual(len(calls), 1)
        self.assertIn('--wheel-only', calls[0][0])

    def test_cuda_failure_stops_before_tests(self):
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.invoke('cuda', fail=lambda cmd: '-c' in cmd)

    def test_cache_key_changes_with_dependencies(self):
        profile = {'build-system': {}, 'project': {}, 'dependency-groups': {}, 'tool': {}}
        first = gpu.cache_key(profile)
        profile['tool']['revision'] = 'changed'
        self.assertNotEqual(first, gpu.cache_key(profile))


if __name__ == '__main__':
    unittest.main()
