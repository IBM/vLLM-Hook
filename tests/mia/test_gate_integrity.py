"""The exit criterion must be incapable of reporting success without running anything.

`pytest tests/ -q -m "not gpu"` is this branch's exit criterion. Two ways it was able to
lie, both closed and both pinned here:

  * `tests/conftest.py` called `pytest.importorskip("vllm")` as its FIRST statement, so a
    missing, half-installed or wrong-interpreter vLLM aborted collection of the entire
    `tests/` tree before one test ran. A guard placed inside `tests/` cannot catch that --
    it is skipped along with everything else -- which is why the zero-collected check lives
    in the repo-root `conftest.py`, and why the checks below drive pytest in a SUBPROCESS
    against a throwaway tree rather than inspecting the running session.
  * Nothing asserted that a zero-collected session fails. Today pytest itself returns
    NO_TESTS_COLLECTED (5) for every shape measured here, so the root conftest hook is belt
    to pytest's braces rather than the thing currently doing the work -- see its docstring.
    That is exactly why the property is pinned END-TO-END, by exit code: it keeps holding
    whichever layer provides it, and it fails if a pytest upgrade, a plugin or a config
    change starts calling an empty run a pass.

These are pure subprocess + text checks: no vLLM, no GPU, no engine. They deliberately keep
running when vLLM is absent, because that is the scenario they exist to police.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFTEST = REPO_ROOT / "conftest.py"
TESTS_DIR = REPO_ROOT / "tests"


# ---------------------------------------------------------------------------------------
# End-to-end: a session that runs nothing must not exit 0.
# ---------------------------------------------------------------------------------------

def _tree(tmp_path: Path, sub_conftest: str) -> Path:
    """A throwaway repo shaped like this one: the real root conftest on top, a package
    directory below it whose conftest we control, and one passing test inside."""
    shutil.copy(ROOT_CONFTEST, tmp_path / "conftest.py")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "conftest.py").write_text(sub_conftest)
    (sub / "test_demo.py").write_text("def test_one():\n    assert True\n")
    return tmp_path


def _run(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=tree, capture_output=True, text=True, timeout=300)


def test_the_fixture_can_exit_zero(tmp_path):
    """CONTROL ARM. Without it the two assertions below could pass for the wrong reason --
    a tree that can never exit 0 proves nothing about trees that run nothing."""
    result = _run(_tree(tmp_path, ""), ".")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_a_collection_aborted_by_a_missing_dependency_does_not_exit_zero(tmp_path):
    """The historical shape, reproduced exactly: a conftest that skips at import time
    because an optional dependency is absent. The tree contains a test that would pass;
    the run must still not be reported as a success, because it never ran."""
    tree = _tree(tmp_path, 'import pytest\n'
                           'pytest.importorskip("a_module_that_is_definitely_not_installed")\n')
    result = _run(tree, ".")
    assert result.returncode != 0, (
        "a session that collected nothing reported success:\n" + result.stdout + result.stderr)


def test_an_empty_selection_does_not_exit_zero(tmp_path):
    """An intentionally empty selection fails too, and that is the right answer rather than
    a special case: you asked for no tests, no tests ran, and calling that a pass is how a
    typo'd selector in CI turns into a permanently green pipeline. pytest's own
    NO_TESTS_COLLECTED (5) says the same thing, and the exit code distinguishes it from a
    real failure (1) for anyone who wants to tell them apart."""
    result = _run(_tree(tmp_path, ""), ".", "-k", "zzz_this_matches_nothing")
    assert result.returncode != 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------------------
# Static: the vLLM dependency stays pushed down to the modules that need it.
# ---------------------------------------------------------------------------------------

def _module_level_calls(path: Path) -> list[str]:
    """Dotted names of every call made at module level in `path` (not inside a def/class)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            out.append(ast.unparse(node.value.func))
    return out


def test_the_root_tests_conftest_does_not_gate_the_whole_suite_on_vllm():
    """The regression proper. A module-level importorskip HERE takes the entire tests/ tree
    down with it, including any guard placed inside tests/."""
    calls = _module_level_calls(TESTS_DIR / "conftest.py")
    assert "pytest.importorskip" not in calls, (
        "tests/conftest.py gates the whole suite on an optional import again; push the "
        "guard down to the modules that actually need it")


def _first_mia_import_line(tree: ast.Module) -> int | None:
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(a.name == "mia" or a.name.startswith(("mia.", "vllm.")) or a.name == "vllm"
                   for a in node.names):
                return node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "mia" or node.module.startswith(("mia.", "vllm.")) \
                    or node.module == "vllm":
                return node.lineno
    return None


def test_every_module_importing_mia_guards_its_own_vllm_dependency():
    """`import mia` reaches `from vllm import LLM` through mia/llm.py, so any test module
    that imports mia at module level needs vLLM to be collectable at all. Each such module
    must say so itself with `pytest.importorskip("vllm")` BEFORE that import -- otherwise a
    missing vLLM turns into a collection ERROR, which fails the gate for an environment
    reason rather than skipping the handful of tests that genuinely need the dependency.

    This is the mechanical half of the fix: it stops the next test module that imports mia
    from silently re-coupling the whole suite to an optional dependency. One test over all
    modules rather than a parametrization, so the gate gains no skips.
    """
    checked, problems = [], []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        first_import = _first_mia_import_line(tree)
        if first_import is None:
            continue                       # pure Python: nothing to guard
        rel = path.relative_to(REPO_ROOT)
        checked.append(rel)
        guards = [node.lineno for node in tree.body
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and ast.unparse(node.value.func) == "pytest.importorskip"
                  and node.value.args
                  and getattr(node.value.args[0], "value", None) == "vllm"]
        if not guards:
            problems.append(f"{rel}: imports mia/vllm at line {first_import} but never calls "
                            f'pytest.importorskip("vllm") -- a missing vLLM would ERROR '
                            f"collection here")
        elif min(guards) > first_import:
            problems.append(f"{rel}: guards at line {min(guards)}, AFTER the import at line "
                            f"{first_import}; the guard must come first to have any effect")
    # Positive control: if the walk found nothing to check, the assertion below would pass
    # vacuously and this test would be worthless.
    assert checked, "found no test module importing mia -- the walk is broken, not the repo"
    assert not problems, "\n".join(problems)
