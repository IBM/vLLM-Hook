"""Linux installs select binary vLLM and restore the unmodified project profile."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
import io
import hashlib
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cross-platform-install'))
SPEC = importlib.util.spec_from_file_location('cpu_setup', ROOT / 'cross-platform-install/prepare_parity_cpu.py')
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_scope_is_linux_and_wheels_replace_only_triton_source(self):
        original = (ROOT / 'pyproject.toml').read_text()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / 'pyproject.toml'
            path.write_text(original)
            with setup.linux_cpu_resolution(repo, Path('/content/triton.whl')):
                scoped = tomllib.loads(path.read_text())
                uv = scoped['tool']['uv']
                self.assertIn("sys_platform == 'linux'", uv['environments'][0])
                self.assertIn("python_version == '3.12'", uv['environments'][0])
                self.assertEqual(uv['sources']['triton'][0]['path'], '/content/triton.whl')
                self.assertEqual(uv['sources']['vllm'], tomllib.loads(original)['tool']['uv']['sources']['vllm'])
            self.assertEqual(path.read_text(), original)

    def test_failed_resolution_restores_original(self):
        original = (ROOT / 'pyproject.toml').read_text()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / 'pyproject.toml'
            path.write_text(original)
            with self.assertRaises(RuntimeError):
                with setup.linux_cpu_resolution(repo, Path('/content/triton.whl')):
                    raise RuntimeError('uv failed')
            self.assertEqual(path.read_text(), original)

    def test_vllm_wheel_is_pinned_with_release_checksum(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        sources = project['tool']['uv']['sources']['vllm']
        wheel = next(s for s in sources if 'url' in s)
        self.assertIn('/v0.29.0/', wheel['url'])
        self.assertIn('manylinux_2_34_x86_64.whl', wheel['url'])
        self.assertNotIn('#', wheel['url'])
        self.assertEqual(len(project['tool']['mia']['colab']['vllm-cpu-wheel-sha256']), 64)
        self.assertIn("sys_platform == 'linux'", wheel['marker'])
        self.assertTrue(any('git' in s and "sys_platform != 'linux'" in s['marker'] for s in sources))

    def test_install_with_wheel_never_builds_vllm_or_triton(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'cross-platform-install').mkdir()
            original = (ROOT / 'pyproject.toml').read_text()
            (repo / 'pyproject.toml').write_text(original)
            cpu = repo / 'cpu'
            (cpu / 'bin').mkdir(parents=True)
            (cpu / 'bin/python').touch()
            args = ['setup', '--environment', str(cpu), '--triton-wheel', '/content/triton.whl']
            with patch.object(sys, 'argv', args), \
                 patch.object(setup, '__file__', str(repo / 'cross-platform-install/prepare.py')), \
                 patch.object(setup.platform, 'system', return_value='Linux'), \
                 patch.object(setup.platform, 'machine', return_value='x86_64'), \
                 patch.object(setup, 'validate_wheel'), \
                 patch.object(setup, 'release_wheel', return_value=Path('/content/vllm.whl')), \
                 patch.object(setup.shutil, 'which', return_value='/bin/uv'), \
                 patch.object(setup.subprocess, 'run', side_effect=lambda command, **kw: calls.append(command)):
                setup.main()
            sync = next(c for c in calls if 'sync' in c)
            self.assertEqual(sync.count('--no-build-package'), 2)
            self.assertIn('vllm', sync)
            self.assertIn('triton', sync)
            self.assertFalse(any('build' in c or 'git' == c[0] for c in calls))
            self.assertEqual((repo / 'pyproject.toml').read_text(), original)

    def test_release_download_validates_checksum_and_reuses_file(self):
        data = b'prebuilt release wheel fixture'
        project = {'tool': {'uv': {'sources': {'vllm': [{'url': 'https://example.com/vllm.whl'}]}},
                            'mia': {'colab': {'vllm-cpu-wheel-sha256': hashlib.sha256(data).hexdigest()}}}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(data)) as fetch:
                wheel = setup.release_wheel(project, Path(directory))
                self.assertEqual(wheel.read_bytes(), data)
                self.assertEqual(setup.release_wheel(project, Path(directory)), wheel)
                self.assertEqual(fetch.call_count, 1)

    def test_release_checksum_failure_does_not_commit_download(self):
        project = {'tool': {'uv': {'sources': {'vllm': [{'url': 'https://example.com/vllm.whl'}]}},
                            'mia': {'colab': {'vllm-cpu-wheel-sha256': '0' * 64}}}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(b'corrupt')):
                with self.assertRaises(ValueError):
                    setup.release_wheel(project, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
