"""Write or check the compiled wheel's checksum on each operating system."""
import hashlib
from pathlib import Path
import os
import sys
root = Path(os.environ['GITHUB_WORKSPACE']) / 'wheelhouse'
manifest = root / 'wheel.sha256'
if sys.argv[1] == 'write':
    manifest.write_text(''.join(f'{hashlib.file_digest(p.open("rb"), "sha256").hexdigest()}  {p.name}\n' for p in root.glob('triton-*.whl')))
else:
    for line in manifest.read_text().splitlines():
        expected, name = line.split('  ', 1)
        with (root / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
