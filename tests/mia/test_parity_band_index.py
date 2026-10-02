"""Declared-vs-DOCUMENTED: every band in the code is justified in TOLERANCES.md, and back.

This suite already has two anti-vacuity mechanisms of this shape. `EXPECTED_GATES` asserts
the verdicts a judge EMITS against the ones it DECLARES, so a leg that stops running fails
instead of vanishing. `test_t2_module_integrity` asserts that every global a gate names
actually resolves, so a deleted callee fails loudly rather than mid-suite. Both exist because
the alternative — noticing in review — did not work.

This is the third: the bound a gate ENFORCES against the measurement that JUSTIFIES it.

The gap it closes was real and silent. At the time this file was written, 13 of the suite's
14 `_band()` declarations named no env var anywhere in `tests/mia/parity/TOLERANCES.md`; three of
them -- the real-replay pair and the fused-steer band, plus the steer budget CEILING that is
the only thing stopping two runtime-computed budgets from widening to match their own failure
-- had no section at all. A reader could not get from a constant in the code to the
measurement behind it, and nothing failed when a band was added without one.

BOTH DIRECTIONS MATTER, and for different reasons:

  code -> doc   a band with no entry is a bound nobody can justify. This is the direction
                that was actually broken.
  doc -> code   an entry for a band that no longer exists is its own kind of lie: it reads
                as an active, justified bound and is enforcing nothing.

The DEFAULTS are checked too, because the failure that matters most is neither of those: a
band whose value is widened in code while the documented rationale — the margins, the "23x
above a mis-route" arithmetic — silently goes on describing the old one. That turns the
document into a justification for a bound that is not being enforced.

The index is delimited by HTML comments and parsed as data rather than scraped out of prose.
Matching band names by regex over the whole document would pass on an incidental mention in
a sentence, which is exactly the "the word appears, therefore it is documented" reasoning
this test exists to replace.

SECOND REGISTRY: the constants that are NOT `_band()` declarations. The first version of this
file reached only `_band()`, and a whole-branch review then found the exact failure it was
built to prevent sitting just outside that reach: `STEER_ARM_POSITIVE_CONTROL_ATOL` enforces
`2.0 x 5.830579e-02` while TOLERANCES.md and the README both published `10 x 5.830579e-02` --
a floor that task D13 had already withdrawn as broken, because it sits ABOVE the weakest real
per-request signal and so could not be cleared by any correct implementation. A gate whose
mechanism has a blind spot next to it is not a mechanism; `CONSTANT_INDEX` closes this one.

For that registry the enforced value is obtained by IMPORTING the module and reading the
attribute, not by parsing the source text. `2.0 * 5.830579e-02` is then compared as the
number the gate actually uses, which is the only form of the check that could have caught the
defect above.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:          # `declared_constants` imports the modules it checks
    sys.path.insert(0, str(REPO_ROOT))
PARITY = REPO_ROOT / "tests" / "mia" / "parity"
TOLERANCES = PARITY / "TOLERANCES.md"

_BEGIN = "<!-- BAND-INDEX:BEGIN -->"
_END = "<!-- BAND-INDEX:END -->"
_CONST_BEGIN = "<!-- CONSTANT-INDEX:BEGIN -->"
_CONST_END = "<!-- CONSTANT-INDEX:END -->"

# `_band("NAME", "default", ...)`, allowing the call to wrap after the opening paren -- three
# of the declarations do, and a pattern that missed them would silently check fewer bands
# than exist, which is the failure mode of a completeness test that cannot count.
_DECLARATION = re.compile(r'_band\(\s*\n?\s*"([A-Z0-9_]+)"\s*,\s*\n?\s*"([^"]+)"')
_INDEX_ROW = re.compile(r"^\|\s*`([A-Z0-9_]+)`\s*\|\s*`([^`]+)`\s*\|")
# constant rows carry a module column between the name and the value
_CONST_ROW = re.compile(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")


def _band_modules() -> list[Path]:
    """Every parity module that DECLARES a band -- found, not listed.

    A hard-coded list would keep passing after someone adds bands in a new sibling module,
    which is the same bug this test exists to catch, one level up.
    """
    modules = sorted(p for p in PARITY.glob("*.py")
                     if _DECLARATION.search(p.read_text()))
    assert modules, (
        f"no `_band(...)` declaration found anywhere in {PARITY} -- either the declaration "
        f"helper was renamed (update _DECLARATION) or the bands stopped being declared, and "
        f"in both cases this test is no longer checking anything.")
    return modules


def declared_bands() -> dict[str, str]:
    """`{env var: documented default}` for every band declared in `tests/mia/parity/`."""
    found: dict[str, str] = {}
    for module in _band_modules():
        for env, default in _DECLARATION.findall(module.read_text()):
            assert env not in found or found[env] == default, (
                f"{env} is declared twice with different defaults: "
                f"{found[env]} and {default}")
            found[env] = default
    return found


def documented_bands() -> dict[str, str]:
    """`{env var: documented default}` from the TOLERANCES.md band index."""
    text = TOLERANCES.read_text()
    assert _BEGIN in text and _END in text, (
        f"{TOLERANCES} has no {_BEGIN} / {_END} block. That table is the machine-readable "
        f"half of this check; without it nothing asserts that a band is justified.")
    block = text[text.index(_BEGIN):text.index(_END)]
    rows: dict[str, str] = {}
    for line in block.splitlines():
        match = _INDEX_ROW.match(line)
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def test_bands_are_actually_declared_somewhere():
    """Guard the guard: a regex that matches nothing would make every assertion below pass."""
    assert len(declared_bands()) >= 14
    assert len(documented_bands()) >= 14


def test_every_declared_band_is_documented():
    """code -> doc. A bound nobody can justify is a bound nobody should be enforcing."""
    missing = sorted(set(declared_bands()) - set(documented_bands()))
    assert missing == [], (
        f"{len(missing)} band(s) are enforced by the suite but absent from the "
        f"{TOLERANCES.name} index: {missing}. Every band needs a section recording what it "
        f"gates, what was measured, on which LSF job and node, the date, model and dtype, "
        f"and the derivation with its margins -- then a row in the index naming the env var "
        f"verbatim, so a reader can get from the constant to its justification.")


def test_every_documented_band_still_exists():
    """doc -> code. A stale entry reads as an active bound and enforces nothing."""
    stale = sorted(set(documented_bands()) - set(declared_bands()))
    assert stale == [], (
        f"{TOLERANCES.name} documents {len(stale)} band(s) that no `_band(...)` call "
        f"declares any more: {stale}. Either the band was deleted (remove its row and its "
        f"section) or it was renamed (update both) -- an entry for a bound that is not "
        f"enforced is worse than no entry.")


@pytest.mark.parametrize("env", sorted(declared_bands()))
def test_the_documented_default_is_the_enforced_default(env):
    """The failure that matters most: a band widened in code while the doc still argues for
    the old value. The margins recorded in TOLERANCES.md ("23x above a mis-route", "5.1x
    below a real intervention") are arithmetic ON this number; if it moves and they do not,
    the document becomes a justification for a bound nobody is enforcing.
    """
    documented = documented_bands()
    assert env in documented, f"{env} is not in the {TOLERANCES.name} index"
    code, doc = declared_bands()[env], documented[env]
    assert float(code) == float(doc), (
        f"{env}: the code enforces {code} but {TOLERANCES.name} documents {doc}. Whichever "
        f"is right, the derivation and its margins in that section were computed against "
        f"{doc} and have to be re-derived against {code} before this passes again.")


@pytest.mark.parametrize("env", sorted(declared_bands()))
def test_every_band_has_prose_beyond_its_index_row(env):
    """A row in the index is a pointer, not a justification. Every band must also be named
    in the body of the document -- the section that records the measurement.
    """
    text = TOLERANCES.read_text()
    body = text[text.index(_END):]
    assert env in body, (
        f"{env} appears only in the index table. The index says WHERE the justification is; "
        f"it is not itself one. Add a section recording the measurement, the LSF job and "
        f"node, the date, model, dtype and the derivation, naming {env} in it.")


# ---------------------------------------------------------------------------
# The SECOND registry: enforced constants that are not `_band()` declarations
# ---------------------------------------------------------------------------
# Discovery is by AST over every module-level assignment whose value is a float (or an
# arithmetic expression of floats), NOT by a naming convention and NOT by a hard-coded list.
# A convention like "*_ATOL" would miss a constant someone names differently, and a list
# would miss one added later -- both are the failure this whole file exists to prevent, one
# level up. Every such constant in `tests/mia/parity/` is either declared through `_band()`
# (first registry) or must appear in the constant index.


def _is_float_valued(node: ast.AST) -> bool:
    """A literal float, or an expression built only from float literals and arithmetic.

    `2.0 * 5.830579e-02` has to count: writing a tolerance as the derivation that produced it
    is GOOD practice here -- it keeps the margin visible at the point of enforcement -- and a
    check that only recognised bare literals would silently skip exactly the constants whose
    value is easiest to get wrong in prose.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_float_valued(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        return _is_float_valued(node.left) and _is_float_valued(node.right)
    return False


def declared_constants() -> dict[str, tuple[str, float]]:
    """`{name: (module basename, ENFORCED value)}` for every non-`_band()` float constant.

    The value comes from importing the module and reading the attribute, so what is checked
    is the number the gate actually enforces rather than the text that produced it.
    """
    import importlib

    band_names = set(declared_bands())
    found: dict[str, tuple[str, float]] = {}
    for module_path in sorted(PARITY.glob("*.py")):
        tree = ast.parse(module_path.read_text())
        names = [node.targets[0].id
                 for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and _is_float_valued(node.value)]
        names = [n for n in names if n not in band_names]
        if not names:
            continue
        module = importlib.import_module(f"tests.mia.parity.{module_path.stem}")
        for name in names:
            found[name] = (module_path.name, float(getattr(module, name)))
    return found


def documented_constants() -> dict[str, tuple[str, str]]:
    """`{name: (module, documented value)}` from the TOLERANCES.md constant index."""
    text = TOLERANCES.read_text()
    assert _CONST_BEGIN in text and _CONST_END in text, (
        f"{TOLERANCES} has no {_CONST_BEGIN} / {_CONST_END} block. Without it, every enforced "
        f"constant that is not a `_band()` -- which is where the published-vs-enforced "
        f"mismatch of task E1's whole-branch review actually lived -- is unchecked again.")
    block = text[text.index(_CONST_BEGIN):text.index(_CONST_END)]
    rows: dict[str, tuple[str, str]] = {}
    for line in block.splitlines():
        match = _CONST_ROW.match(line)
        if match:
            rows[match.group(1)] = (match.group(2), match.group(3))
    return rows


def test_constants_are_actually_discovered():
    """Guard the guard, again: a discovery that finds nothing makes every check below pass."""
    discovered = declared_constants()
    assert len(discovered) >= 7, f"only discovered {sorted(discovered)}"
    assert "STEER_ARM_POSITIVE_CONTROL_ATOL" in discovered
    # the expression form must survive discovery -- it is the one that was misdocumented
    assert discovered["STEER_ARM_POSITIVE_CONTROL_ATOL"][1] == pytest.approx(0.1166116)


def test_every_enforced_constant_is_documented():
    """code -> doc, for the constants `_band()` does not reach."""
    missing = sorted(set(declared_constants()) - set(documented_constants()))
    assert missing == [], (
        f"{len(missing)} enforced constant(s) are absent from the {TOLERANCES.name} constant "
        f"index: {missing}. Every enforced tolerance needs a documented value and a section "
        f"recording what it gates and what was measured -- a constant outside both indexes "
        f"is exactly where this suite's published-vs-enforced mismatch hid.")


def test_every_documented_constant_still_exists():
    """doc -> code."""
    stale = sorted(set(documented_constants()) - set(declared_constants()))
    assert stale == [], (
        f"{TOLERANCES.name} documents {len(stale)} constant(s) that no longer exist in "
        f"`tests/mia/parity/`: {stale}.")


@pytest.mark.parametrize("name", sorted(declared_constants()))
def test_the_documented_constant_value_is_the_enforced_one(name):
    """THE check. `STEER_ARM_POSITIVE_CONTROL_ATOL` was published as `10 x 5.830579e-02 =
    5.830579e-01` in two documents while enforcing `2.0 x 5.830579e-02 = 1.166116e-01`, and
    the published number was one task D13 had already withdrawn as unpassable. Nothing
    failed, because nothing compared them.

    A relative tolerance is used because documented values are written to a fixed number of
    significant figures (`1.166116e-01` for an enforced `1.1661158e-01`); it is far tighter
    than any plausible edit and would not absorb so much as a changed final digit of a
    mantissa, let alone a factor of five.
    """
    documented = documented_constants()
    assert name in documented, f"{name} is not in the {TOLERANCES.name} constant index"
    module, doc_value = declared_constants()[name][0], documented[name][1]
    enforced = declared_constants()[name][1]
    assert float(doc_value) == pytest.approx(enforced, rel=1e-6), (
        f"{name} ({module}): the suite ENFORCES {enforced!r} but {TOLERANCES.name} publishes "
        f"{doc_value}. Whichever is right, the margins argued in that section were computed "
        f"against the documented value and must be re-derived before this passes -- and if "
        f"the documented one is also in README.md, fix it there too.")


@pytest.mark.parametrize("name", sorted(declared_constants()))
def test_the_documented_constant_names_its_module(name):
    """The module column is how a reader gets from the doc back to the definition."""
    documented = documented_constants()
    assert name in documented
    assert documented[name][0] == declared_constants()[name][0], (
        f"{name} is documented as living in {documented[name][0]} but is defined in "
        f"{declared_constants()[name][0]}")


@pytest.mark.parametrize("name", sorted(declared_constants()))
def test_every_constant_has_prose_beyond_its_index_row(name):
    """A row is a pointer; the justification is the section it points at."""
    text = TOLERANCES.read_text()
    body = text[text.index(_CONST_END):]
    assert name in body, (
        f"{name} appears only in the constant index. Add a section recording what it "
        f"enforces, what was measured, on which LSF job and node, and the margins -- naming "
        f"{name} in it.")
