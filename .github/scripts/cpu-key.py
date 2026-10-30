"""Exact cache key: runtime, toolchain, source and build configuration."""
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import shutil

facts = dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
             image=os.getenv('ImageVersion'), env={k: v for k, v in os.environ.items()
             if k.startswith(('TRITON_', 'CMAKE_', 'MAX_JOBS', 'CC', 'CXX'))})
for compiler in ('gcc', 'clang', 'cmake'):
    if not shutil.which(compiler):
        facts[compiler] = 'unavailable'
        continue
    result = subprocess.run([compiler, '--version'], text=True, capture_output=True)
    facts[compiler] = result.stdout
facts['scripts'] = {str(p): p.read_text() for p in Path('.github/scripts').glob('cpu-*')}
identity = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
Path('evidence/wheel-compatibility.json').write_text(json.dumps(facts, indent=2))
with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
    out.write(f'key=cpu-v2-{platform.system()}-{platform.machine()}-{identity}\n')
