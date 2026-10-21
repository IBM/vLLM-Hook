"""Runner/report integration checks using simulated child-process results."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'cross-platform-install' / 'run_parity.py'
SPEC = importlib.util.spec_from_file_location('parity_runner', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def run_case(self, directory, *, mismatch=False, child_error=False):
        def child(command, env, log, timeout):
            if '-c' in command:
                (directory / 'model.json').write_text(json.dumps({
                    'model_id': 'fixture', 'model_revision': 'abc', 'model_path': '/fixture'}))
                return
            if child_error:
                raise RuntimeError('CPU worker failed')
            backend = command[command.index('--backend') + 1]
            scenario = command[command.index('--scenario') + 1]
            output = Path(command[command.index('--output') + 1])
            output.write_text(json.dumps({'backend': backend,
                'metadata': {'scenario': scenario, 'model_revision': 'abc'},
                'values': {'capture': [2.0 if mismatch and backend == 'cuda' else 1.0]}}))

        args = ['run_parity.py', '--cpu-python', '/fixture/cpu/python',
                '--cuda-python', '/fixture/cuda/python', '--output', str(directory)]
        with patch.object(sys, 'argv', args), patch.object(runner, 'run_logged', child):
            status = runner.main()
        return status, json.loads((directory / 'report.json').read_text())

    def test_all_four_comparisons_required_for_success(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report = self.run_case(Path(directory))
            self.assertEqual(status, 0)
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['scenarios']), 4)

    def test_numerical_mismatch_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report = self.run_case(Path(directory), mismatch=True)
            self.assertEqual(status, 1)
            self.assertFalse(report['passed'])

    def test_worker_failure_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report = self.run_case(Path(directory), child_error=True)
            self.assertEqual(status, 1)
            self.assertFalse(report['passed'])
            self.assertIn('CPU worker failed', report['error'])

    def test_cpu_environment_does_not_retain_cuda_selection(self):
        with patch.dict('os.environ', {'VLLM_TARGET_DEVICE': 'cuda', 'TRITON_DEFAULT_BACKEND': 'nvidia'}):
            env = runner.environment('cpu', '/nonexistent/venv/bin/python')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['TRITON_DEFAULT_BACKEND'], 'cpu')
        self.assertEqual(env['VLLM_TARGET_DEVICE'], 'cpu')


if __name__ == '__main__':
    unittest.main()
