"""A local wheel backup is mandatory before installing CPU or running parity."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cross-platform-install'))
SPEC = importlib.util.spec_from_file_location('colab_helper', ROOT / 'cross-platform-install/colab_cli.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class CliTests(unittest.TestCase):
    def exercise(self, fail_backup=False):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'mia').mkdir()
            (repo / 'pyproject.toml').write_text((ROOT / 'pyproject.toml').read_text())

            def call(command, **kwargs):
                if 'download' in command and '/content/mlr55-command-' in command[-2]:
                    Path(command[-1]).write_text(json.dumps({'ok': True}))
                code = kwargs.get('input', '')
                if 'exec' in command:
                    if '--wheel-only' in code:
                        events.append('wheel')
                    elif 'Installing CPU software from wheels' in code:
                        events.append('install-cpu')
                    elif 'run_parity.py' in code:
                        events.append('tests')
                return SimpleNamespace(returncode=0)

            def backup(*args):
                events.append('backup')
                if fail_backup:
                    raise ValueError('Disk full')

            args = ['colab_cli.py', 'parity']
            with patch.object(sys, 'argv', args), \
                 patch.object(helper, '__file__', str(repo / 'cross-platform-install/colab_cli.py')), \
                 patch.object(helper.shutil, 'which', return_value='/bin/colab'), \
                 patch.object(helper.subprocess, 'run', side_effect=call), \
                 patch.object(helper, 'upload_cached_wheel', return_value=None), \
                 patch.object(helper, 'backup_wheel', side_effect=backup):
                if fail_backup:
                    with self.assertRaises(ValueError):
                        helper.main()
                else:
                    helper.main()
        return events

    def test_backup_happens_before_install_and_tests(self):
        self.assertEqual(self.exercise(), ['wheel', 'backup', 'install-cpu', 'tests'])

    def test_failed_backup_prevents_install_and_tests(self):
        self.assertEqual(self.exercise(fail_backup=True), ['wheel', 'backup'])

    def test_generated_commands_are_valid_python(self):
        for action in ('sync', 'check', 'install', 'test', 'wheel-status',
                       'prepare-wheel', 'prepare-cpu', 'prepare-gpu', 'parity'):
            compile(helper.remote_code(action), action, 'exec')

    def test_remote_error_is_detected_even_if_cli_exits_zero(self):
        def command(args, **kwargs):
            if 'download' in args:
                Path(args[-1]).write_text(json.dumps({'ok': False, 'error': 'uv failed'}))
            return SimpleNamespace(returncode=0)
        with patch.object(helper.subprocess, 'run', side_effect=command):
            with self.assertRaisesRegex(RuntimeError, 'uv failed'):
                helper.execute_remote(['colab'], 'mlr55', 'raise RuntimeError()', 10)

    def test_completion_wrapper_records_actual_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / 'status.json'
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                exec(helper.completion_wrapper("raise ValueError('fixture failure')", status), {})
            self.assertEqual(json.loads(status.read_text()), {'ok': False, 'error': 'fixture failure'})

    def test_success_can_be_verified_after_final_cli_reply_fails(self):
        def command(args, **kwargs):
            if 'download' in args:
                Path(args[-1]).write_text(json.dumps({'ok': True}))
                return SimpleNamespace(returncode=0)
            return SimpleNamespace(returncode=1)
        with patch.object(helper.subprocess, 'run', side_effect=command):
            helper.execute_remote(['colab'], 'mlr55', "print('done')", 10)


if __name__ == '__main__':
    unittest.main()
