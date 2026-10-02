"""Task D10 item 2: the steer A/B pass must be driven by ONE function, not two
independently-maintained copies.

`tests/mia/parity/run_steer_ab_leak_control.py`'s dedicated GPU run (LSF 1725670) measured
BIT-EXACT 0.0 on all 17 unarmed requests of a mixed-arm, width-33 steer batch: same engine,
same boot, same prompts, same order, same alternating arming mask, real steer vs an
all-zero-direction control. Before this task, `tests/mia/parity/t2_invariants.py`'s in-suite
`full_batch_mixed_arm` leg (`_drive_steer_mixed`) ran the mechanically-equivalent sequence
as its OWN hand-maintained copy -- reusing only `_make_zero_vector` from the dedicated
script. LSF 1731353 found the in-suite gate that copy feeds (`steer_small
per-request-arming`) moving 3 of 33 unarmed requests by 1.613855e-02 / 2.467152e-02 /
2.032498e-02 at that SAME configuration. Being "structurally the same code" was not the
same as being the SAME code.

The fix: `run_steer_ab_leak_control.run_steer_ab_pass` is now the ONLY place that drives the
A/B pass (prompts, mask, zero-vector config, the two `engine.generate()` calls); both
`_drive_ab_leak_control` (this script's own driver) and `t2_invariants._drive_steer_mixed`
(the in-suite leg driver) call it directly. These tests pin that there is exactly one
driver -- not that the GPU result changed, which no hermetic test can prove -- by
disassembling both call sites and asserting neither one calls `engine.generate` itself any
more, and that `t2_invariants.py` does not define a second implementation under the same or
a different name.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.mia.parity.run_steer_ab_leak_control as ab_mod  # noqa: E402
import tests.mia.parity.t2_invariants as t2mod  # noqa: E402


def _calls_named(func, name: str) -> int:
    """How many times `func`'s body calls something named `name`, as either a bare name
    (`name(...)`) or an attribute (`x.name(...)`)."""
    tree = ast.parse(inspect.getsource(func))
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == name)
             or (isinstance(node.func, ast.Attribute) and node.func.attr == name))
    )


def _imports_from(func, module: str, name: str) -> bool:
    """Does `func`'s body contain a real `from <module> import <name>` STATEMENT?

    Not `"from ... import ..." in source`: that substring is equally satisfied by a comment,
    a docstring, or a string literal, so the assertion it backs can be made to pass by
    writing a sentence about the import instead of performing it. An `ast.ImportFrom` node
    cannot be a comment.
    """
    tree = ast.parse(inspect.getsource(func))
    return any(
        isinstance(node, ast.ImportFrom) and node.module == module
        and any(alias.name == name for alias in node.names)
        for node in ast.walk(tree))


def _keywords_of_call(func, called: str) -> dict:
    """Keyword arguments passed to every `called(...)` in `func`, as {name: literal value}.

    Reads the CALL, so `label_a="generation"` counts only when it is actually being passed.
    """
    tree = ast.parse(inspect.getsource(func))
    out: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = (node.func.id if isinstance(node.func, ast.Name)
                  else getattr(node.func, "attr", None))
        if target != called:
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                out[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                out[kw.arg] = ast.unparse(kw.value)
    return out


def _string_constants(func) -> set:
    """Every string LITERAL in `func`'s body, docstring excluded.

    A comment mentioning `arming.safetensors` is not an `ast.Constant`; a path the function
    actually writes is.
    """
    tree = ast.parse(inspect.getsource(func))
    body = tree.body[0].body if tree.body and hasattr(tree.body[0], "body") else []
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]                      # drop the docstring
    found = set()
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
    return found


def _raises_with_message(func, exc: str, fragment: str) -> bool:
    """Does `func` RAISE `exc` with `fragment` in one of its message literals?"""
    tree = ast.parse(inspect.getsource(func))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        call = node.exc
        name = (call.func.id if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                else getattr(call, "id", None))
        if name != exc:
            continue
        pieces = {c.value for c in ast.walk(call)
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        if any(fragment in piece for piece in pieces):
            return True
    return False


def test_drive_steer_mixed_delegates_to_the_shared_ab_pass():
    """`_drive_steer_mixed` (the in-suite `full_batch_mixed_arm` leg driver) must call
    `run_steer_ab_pass` and must NOT itself call `.generate(` -- two hand-written copies of
    "run both arms in one boot" is what let them drift."""
    assert _calls_named(t2mod._drive_steer_mixed, "run_steer_ab_pass") >= 1
    assert _calls_named(t2mod._drive_steer_mixed, "generate") == 0


def test_drive_ab_leak_control_delegates_to_the_shared_ab_pass():
    """The standalone script's own driver must ALSO go through `run_steer_ab_pass` -- it is
    the canonical mechanics now, not a twin of the in-suite one."""
    assert _calls_named(ab_mod._drive_ab_leak_control, "run_steer_ab_pass") >= 1
    assert _calls_named(ab_mod._drive_ab_leak_control, "generate") == 0


def test_t2_invariants_has_no_second_ab_pass_implementation():
    """`t2_invariants.py` must not define its own `run_steer_ab_pass` -- the whole point of
    this task is that there is exactly ONE driver, imported, not reimplemented under a
    different name in a second module."""
    assert not hasattr(t2mod, "run_steer_ab_pass"), (
        "t2_invariants.py defines its own run_steer_ab_pass; it must import the one in "
        "run_steer_ab_leak_control.py instead")


def test_drive_steer_mixed_imports_run_steer_ab_pass_from_the_canonical_module():
    """A structural check that the import actually points at
    `tests.mia.parity.run_steer_ab_leak_control`, not some other same-named stand-in."""
    assert _imports_from(t2mod._drive_steer_mixed,
                         "tests.mia.parity.run_steer_ab_leak_control", "run_steer_ab_pass"), (
        "_drive_steer_mixed does not contain a real `from "
        "tests.mia.parity.run_steer_ab_leak_control import run_steer_ab_pass` statement")


def test_run_steer_ab_pass_requires_both_classes_present():
    """The safety guard `_drive_ab_leak_control` always had -- a "mixed-arm" batch that is
    not actually mixed proves nothing about either hypothesis -- must survive having been
    extracted into the shared function."""
    assert _raises_with_message(ab_mod.run_steer_ab_pass, "RuntimeError",
                                "mixed-arm batch is not mixed"), (
        "run_steer_ab_pass no longer RAISES RuntimeError for an unmixed batch -- a "
        "'mixed-arm' run that is not mixed proves nothing about either hypothesis")


def test_run_steer_ab_pass_signature_has_the_two_label_parameters():
    """`label_a`/`label_b` are what let ONE driver serve two different readers: `armA`/
    `armB` for this script's own in-process readout, `generation`/`control` for
    `_judge_steer_per_request_arming`."""
    sig = inspect.signature(ab_mod.run_steer_ab_pass)
    assert "label_a" in sig.parameters and "label_b" in sig.parameters
    assert sig.parameters["label_a"].default == "armA"
    assert sig.parameters["label_b"].default == "armB"


def test_drive_steer_mixed_and_ab_leak_control_pass_the_gate_specific_labels():
    """`_drive_steer_mixed` must ask for the labels `_judge_steer_per_request_arming` reads
    (`generation`/`control`); `_drive_ab_leak_control` must ask for the labels ITS OWN
    readout reads (`armA`/`armB`). A swap here would silently break one consumer while
    leaving the other looking fine."""
    mixed = _keywords_of_call(t2mod._drive_steer_mixed, "run_steer_ab_pass")
    assert mixed.get("label_a") == "generation" and mixed.get("label_b") == "control", (
        f"_drive_steer_mixed passes {mixed.get('label_a')!r}/{mixed.get('label_b')!r}; "
        f"_judge_steer_per_request_arming reads 'generation'/'control'")

    ab = _keywords_of_call(ab_mod._drive_ab_leak_control, "run_steer_ab_pass")
    assert ab.get("label_a") == "armA" and ab.get("label_b") == "armB", (
        f"_drive_ab_leak_control passes {ab.get('label_a')!r}/{ab.get('label_b')!r}; "
        f"its own readout reads 'armA'/'armB'")


def test_run_steer_ab_pass_still_writes_the_arming_marker():
    """Both readers (`_judge_steer_per_request_arming` and this script's own readout) key
    their per-request loop off `arming.safetensors` -- the shared function must keep writing
    it, or both callers break the same way at once."""
    assert "arming.safetensors" in _string_constants(ab_mod.run_steer_ab_pass), (
        "run_steer_ab_pass no longer contains 'arming.safetensors' as a string LITERAL -- "
        "both readers key their per-request loop off that marker file")
