"""Wheel backup/restore integrity and revision checks without a network or compiler."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cross-platform-install'))
import wheel_cache as cache

REVISION = 'c8e7cc932320d05d08f39ffc99bfda32dc4f22fa'


def make_wheel(folder, version='3.8.0+cpu.gitc8e7cc93', tag='cp312-cp312-linux_x86_64'):
    wheel = folder / ('triton-' + version + '-' + tag + '.whl')
    with zipfile.ZipFile(wheel, 'w') as output:
        output.writestr('triton.dist-info/METADATA', 'Name: triton\nVersion: ' + version + '\n')
        output.writestr('triton.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: ' + tag + '\n')
        output.writestr('triton/native.so', b'fake native binary for transfer tests')
    return wheel


class WheelTests(unittest.TestCase):
    def test_backup_reassembles_exact_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root)
            with patch.object(cache, 'CHUNK_SIZE', 64):
                manifest = cache.export_chunks(wheel, REVISION, root / 'parts')
                result = cache.restore_chunks(manifest, [c['path'] for c in manifest['chunks']], root / 'backup')
            self.assertEqual(result.read_bytes(), wheel.read_bytes())
            self.assertGreater(len(manifest['chunks']), 1)

    def test_corruption_does_not_commit_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root)
            manifest = cache.export_chunks(wheel, REVISION, root / 'parts')
            Path(manifest['chunks'][0]['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                cache.restore_chunks(manifest, [c['path'] for c in manifest['chunks']], root / 'backup')
            self.assertFalse((root / 'backup' / wheel.name).exists())
            self.assertFalse(list((root / 'backup').glob('*.partial')))

    def test_wrong_revision_and_wrong_python_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root)
            with self.assertRaises(ValueError):
                cache.validate_wheel(wheel, '00000000')
            wheel = make_wheel(root, tag='cp313-cp313-linux_x86_64')
            with self.assertRaises(ValueError):
                cache.validate_wheel(wheel, REVISION)

    def test_unsafe_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                cache.restore_chunks({'filename': '../escape.whl'}, [], Path(directory))


if __name__ == '__main__':
    unittest.main()
