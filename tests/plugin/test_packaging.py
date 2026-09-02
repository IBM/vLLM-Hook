# tests/plugin/test_packaging.py
"""Everything the plugin imports at load time is a declared dependency.

The 0.3.0 release imported ``zstandard`` unconditionally but declared it
only in the optional ``[engine]`` extra, so an environment with vLLM
present and the extra absent failed to load the plugin at boot. Two
guards: a static audit of module-level imports against ``setup.py``'s
``install_requires`` (deterministic, engine-free), and a subprocess probe
that registers the plugin and accepts a missing engine but nothing else.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "vllm_hook_plugins"
SOURCE_DIR = PACKAGE_DIR / "vllm_hook_plugins"

# The engine and its transitive dependencies are the only modules a plugin
# may import without declaring them: they are present in every process the
# plugin is loaded into.
ENGINE_MODULES = {"vllm"}


def _declared_imports() -> set:
    """Import names of setup.py's install_requires (distribution names,
    with the pip normalisation that differs from the import name).
    """
    text = (PACKAGE_DIR / "setup.py").read_text()
    block = re.search(r"install_requires=\[(.*?)\]", text, re.S).group(1)
    names = re.findall(r'"([A-Za-z0-9_.\-]+?)(?:[<>=!~;\[].*?)?"', block)
    assert names, "install_requires not found in setup.py"
    return {name.replace("-", "_") for name in names}


def _module_level_imports(path: Path):
    """Top-level names imported unconditionally at module load — not those
    under ``if TYPE_CHECKING:``, ``try:`` guards, or inside functions.
    """
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_module_level_imports_are_declared_or_engine():
    allowed = (
        _declared_imports()
        | set(sys.stdlib_module_names)
        | ENGINE_MODULES
        | {"vllm_hook_plugins", "__future__"}
    )
    offenders = {}
    for path in sorted(SOURCE_DIR.rglob("*.py")):
        undeclared = sorted({name for name in _module_level_imports(path) if name not in allowed})
        if undeclared:
            offenders[str(path.relative_to(SOURCE_DIR))] = undeclared
    assert not offenders, f"module-level imports missing from install_requires: {offenders}"


_PROBE = """
import vllm_hook_plugins
try:
    vllm_hook_plugins.register_plugins()
except ModuleNotFoundError as exc:
    print("missing", exc.name)
    raise SystemExit(3)
print("ok")
"""


def test_register_plugins_needs_only_declared_deps_and_the_engine():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, cwd=str(PACKAGE_DIR)
    )
    if result.returncode == 0:
        assert result.stdout.strip() == "ok"
        return
    assert result.returncode == 3, result.stderr
    missing = result.stdout.split()[-1]
    assert missing.split(".")[0] in ENGINE_MODULES, (
        f"register_plugins() needs {missing!r}, which is neither the engine nor in "
        f"install_requires:\n{result.stderr}"
    )
