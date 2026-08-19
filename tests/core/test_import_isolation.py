# tests/core/test_import_isolation.py
"""The core package must import on a bare install — no vLLM anywhere on
the import path. The subprocess probe cannot be fooled by modules other
tests already imported; the source scan pins the rule statically.
"""
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "vllm_hook_plugins"
CORE_DIR = PACKAGE_DIR / "vllm_hook_plugins" / "core"

_PROBE = """
import sys
import vllm_hook_plugins
import vllm_hook_plugins.core
import vllm_hook_plugins.core.kinds
import vllm_hook_plugins.core.schema
import vllm_hook_plugins.core.canonical
import vllm_hook_plugins.core.interpreter
import vllm_hook_plugins.core.artifacts
import vllm_hook_plugins.core.fingerprints
import vllm_hook_plugins.workers.positions
assert "vllm" not in sys.modules, "importing core pulled in vllm"
print("ok")
"""


def test_core_imports_without_vllm():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE_DIR),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_core_sources_never_import_vllm():
    engine_import = re.compile(r"^\s*(import vllm\b|from vllm\.|from vllm import)", re.MULTILINE)
    offenders = [
        str(path)
        for path in CORE_DIR.rglob("*.py")
        if engine_import.search(path.read_text())
    ]
    assert not offenders, f"core modules import vllm: {offenders}"
