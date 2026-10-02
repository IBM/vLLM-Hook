"""A NameError in `tests/mia/parity/t2_invariants.py` must be caught WITHOUT a GPU job.

Task D4d. Commit `26bb3f5` added `_compare_generation_band` and three call sites for it.
Commit `22482d7`, a 250-line refactor two commits later, deleted the function's DEFINITION
but left all three call sites in `judge_capture`. Nothing hermetic noticed: importing the
module never executes a function body, so `def _compare_generation_band(...)` being absent
only raises `NameError` the moment `judge_capture` actually runs -- which needs a real GPU
job. Every T2 run since `22482d7` crashed at that point, immediately AFTER the
routing-identity / op-identity / replay-band verdicts printed (which is why they looked
fine) and BEFORE the T0-graph and GPU-routing bands, which therefore never ran for
`hs_small` or `qk_small` across three commits and one full GPU job (1710884-adjacent).

Two layers close this hole:

1. `test_no_unresolved_module_level_globals` is a GENERAL hermetic check for the whole
   class of bug, not just this one function: it disassembles every function object the
   module defines (module-level AND nested/closures) and asserts every name it loads as a
   global actually resolves, either in the module's own namespace or in builtins. A
   function referencing a name nobody defined or imported is exactly a `NameError` waiting
   for the one code path that calls it -- caught here at collection time instead.
2. `test_compare_generation_band_*` exercise `_compare_generation_band` directly, so its
   OWN behaviour (banded logprobs, fatal bit-exact token ids, missing-arm handling) is
   pinned even if collection-time checks like (1) somehow don't apply.
"""
from __future__ import annotations

import builtins
import dis
import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.mia.parity.t2_invariants as t2mod  # noqa: E402
from tests.mia.parity.t2_invariants import (  # noqa: E402
    _T0_GRAPH_LOGPROB_BAND,
    _compare_generation_band,
)


def _iter_code_objects(code):
    """A function's own code object plus every nested one (closures, comprehensions)."""
    yield code
    for const in code.co_consts:
        if inspect.iscode(const):
            yield from _iter_code_objects(const)


def test_no_unresolved_module_level_globals():
    """Every name any function in this module loads as a GLOBAL must actually be bound.

    This is what a NameError at call time looks like BEFORE the call happens: disassemble
    each function, walk its `LOAD_GLOBAL` instructions (the ones a plain `import` never
    executes), and check each name against the module's namespace (functions, classes,
    imports -- everything `_compare_generation_band` needed and, until it was restored,
    did not have) union builtins. A silent-refactor deletion like `22482d7`'s shows up here
    without booting an engine.
    """
    unresolved = []
    seen_code_ids: set[int] = set()
    module_names = vars(t2mod)
    for _, func in vars(t2mod).items():
        if not inspect.isfunction(func) or func.__module__ != t2mod.__name__:
            continue
        for code in _iter_code_objects(func.__code__):
            if id(code) in seen_code_ids:
                continue
            seen_code_ids.add(id(code))
            for instr in dis.get_instructions(code):
                if instr.opname != "LOAD_GLOBAL":
                    continue
                name = instr.argval
                # Some interpreter versions pack (push_null, name) into argval; normalise.
                if isinstance(name, tuple):
                    name = name[0]
                if name is None:
                    continue
                if hasattr(builtins, name) or name in module_names:
                    continue
                unresolved.append(f"{func.__qualname__} (code {code.co_name!r}) -> {name!r}")
    assert not unresolved, (
        "these functions reference a name that resolves to NOTHING at call time -- a "
        "NameError waiting for whichever test/job first calls them:\n  "
        + "\n  ".join(unresolved)
    )


def test_all_generation_band_call_sites_resolve():
    """The call sites named in the bug report -- plus the `--compare-t1-batched-generation`
    CLI dispatch task D10 added -- must bind to the SAME function object the module defines
    -- not to some other `_compare_generation_band`-shaped stand-in, and not fail merely
    because the name exists but the signature moved under it."""
    import ast

    src = inspect.getsource(t2mod)
    tree = ast.parse(src)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_compare_generation_band"
    ]
    assert len(calls) == 4, f"expected 4 call sites (3 from task D4d + 1 from task D10's " \
                            f"--compare-t1-batched-generation), found {len(calls)}"
    sig = inspect.signature(t2mod._compare_generation_band)
    for call in calls:
        # Every call site is positional (label, a, b) + keyword verdicts/bound/note --
        # bind() raises TypeError on an argument-count or keyword-name mismatch, which is
        # exactly the "signature moved under a caller" failure mode task step 2 worries
        # about (a fix that restores the function with a different shape than its callers
        # expect just moves the crash).
        dummy_args = [object()] * len(call.args)
        dummy_kwargs = {kw.arg: object() for kw in call.keywords if kw.arg is not None}
        sig.bind(*dummy_args, **dummy_kwargs)


def _generation_tree(root: Path, *, token_ids, token_logprobs, cumulative_logprob=0.0,
                     req="req0", prompt_token_ids=(7, 8, 9), drop=()) -> Path:
    """One arm's generation tree.

    `prompt_token_ids` is part of the real artifact (`capture_workload.generation_tensors`
    writes it for every request) and is now part of the gate: the T0-graph comparison has
    to verify the two arms ran the same prompts. `drop` removes a channel, which is how the
    silent-vacuity tests build the "missing on one side" case.
    """
    from safetensors.torch import save_file

    d = root / req
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_token_ids": torch.tensor(list(prompt_token_ids), dtype=torch.int64),
        "token_ids": torch.tensor(token_ids, dtype=torch.int64),
        "token_logprobs": torch.tensor(token_logprobs, dtype=torch.float64),
        "cumulative_logprob": torch.tensor(cumulative_logprob, dtype=torch.float64),
    }
    for key in drop:
        payload.pop(key)
    save_file(payload, str(d / "generation.safetensors"))
    return root


def _band_verdict(a: Path, b: Path, bound: float = _T0_GRAPH_LOGPROB_BAND):
    verdicts: list = []
    _compare_generation_band("unit", a, b, verdicts=verdicts, bound=bound, note="unit test")
    assert len(verdicts) == 1
    return verdicts[0]


def test_generation_band_passes_matching_ids_within_band(tmp_path):
    a = _generation_tree(tmp_path / "a", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
    b = _generation_tree(tmp_path / "b", token_ids=[1, 2, 3],
                        token_logprobs=[-0.1 + 1e-4, -0.2, -0.3])
    label, ok, fatal = _band_verdict(a, b, bound=5e-2)
    assert ok and fatal


def test_generation_band_rejects_logprob_delta_outside_band(tmp_path):
    a = _generation_tree(tmp_path / "a", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
    b = _generation_tree(tmp_path / "b", token_ids=[1, 2, 3],
                        token_logprobs=[-0.1, -0.2, -0.3 - 1.0])
    label, ok, fatal = _band_verdict(a, b, bound=5e-2)
    assert not ok and fatal


def test_generation_band_is_fatal_on_a_token_id_mismatch_no_matter_the_logprob_gap():
    """The one thing this gate must NEVER tolerate: the observer contract is that token ids
    are bit-exact even when the logprobs that produced them are banded."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = _generation_tree(root / "a", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
        b = _generation_tree(root / "b", token_ids=[1, 2, 4],  # last token id moved
                            token_logprobs=[-0.1, -0.2, -0.3])
        label, ok, fatal = _band_verdict(a, b, bound=1.0)  # band wide open
        assert not ok, "a token-id mismatch must fail even under an arbitrarily wide band"
        assert fatal, "the token-id check must be FATAL"


def test_generation_band_fails_when_an_arm_did_not_run(tmp_path):
    label, ok, fatal = _band_verdict(None, tmp_path)
    assert not ok and fatal
    label, ok, fatal = _band_verdict(tmp_path, None)
    assert not ok and fatal


def test_generation_band_fails_when_side_a_has_no_generation_artifacts(tmp_path):
    a = tmp_path / "empty_a"
    a.mkdir()
    b = _generation_tree(tmp_path / "b", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
    label, ok, fatal = _band_verdict(a, b)
    assert not ok and fatal


# ---------------------------------------------------------------------------
# SILENT VACUITY (task D5 item 3): a skipped channel printed as perfect agreement
# ---------------------------------------------------------------------------
# `_compare_generation_band` used to `continue` past a logprob channel that was missing on
# one side or shape-mismatched, and then print `worst |d(logprob)|=0.000000e+00`. That reads
# as "the two arms agree exactly"; it means "nothing was compared". It also never looked at
# `prompt_token_ids`, so the T0-graph gate did not check that the two arms ran the same
# prompts at all.


def test_generation_band_fails_when_a_logprob_channel_is_missing_on_one_side(tmp_path):
    """The proven hole. Side B stops emitting `token_logprobs`; the old gate skipped it and
    reported a worst of 0.000000e+00 with a PASS."""
    a = _generation_tree(tmp_path / "a", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
    b = _generation_tree(tmp_path / "b", token_ids=[1, 2, 3],
                         token_logprobs=[-0.1, -0.2, -0.3], drop=("token_logprobs",))
    label, ok, fatal = _band_verdict(a, b)
    assert not ok and fatal


def test_generation_band_fails_when_a_logprob_channel_is_shape_mismatched(tmp_path):
    """Same hole through the other door: the channel exists on both sides but describes a
    different number of steps, so it was skipped and scored 0."""
    a = _generation_tree(tmp_path / "a", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
    b = _generation_tree(tmp_path / "b", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2])
    label, ok, fatal = _band_verdict(a, b)
    assert not ok and fatal


def test_generation_band_fails_when_both_sides_lost_every_float_channel(tmp_path):
    """Both arms drop the logprobs: the key sets still MATCH, so a key-set check alone is
    not enough -- the gate must notice it has nothing left to bound."""
    kw = dict(token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3],
              drop=("token_logprobs", "cumulative_logprob"))
    a = _generation_tree(tmp_path / "a", **kw)
    b = _generation_tree(tmp_path / "b", **kw)
    label, ok, fatal = _band_verdict(a, b)
    assert not ok and fatal


def test_generation_band_fails_when_the_two_arms_ran_different_prompts(tmp_path):
    """The T0-graph gate compares a capture-OFF arm against a capture-ON arm. If the two ran
    different prompts the comparison is meaningless, and until now nothing said so: the
    token ids and logprobs could agree perfectly by construction."""
    a = _generation_tree(tmp_path / "a", token_ids=[1, 2, 3],
                         token_logprobs=[-0.1, -0.2, -0.3], prompt_token_ids=(7, 8, 9))
    b = _generation_tree(tmp_path / "b", token_ids=[1, 2, 3],
                         token_logprobs=[-0.1, -0.2, -0.3], prompt_token_ids=(7, 8, 10))
    label, ok, fatal = _band_verdict(a, b, bound=1.0)   # band wide open
    assert not ok, "the two arms ran different prompts and the gate passed"
    assert fatal


def test_generation_band_fails_when_prompt_token_ids_are_absent_from_both_sides(tmp_path):
    """A gate that cannot see the prompts cannot certify that both arms ran the same ones."""
    kw = dict(token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3],
              drop=("prompt_token_ids",))
    a = _generation_tree(tmp_path / "a", **kw)
    b = _generation_tree(tmp_path / "b", **kw)
    label, ok, fatal = _band_verdict(a, b)
    assert not ok and fatal


def test_generation_band_reports_how_many_channels_it_actually_compared(tmp_path, capsys):
    """The verdict line carries the channel COUNT, so `worst |d| = 0.000000e+00` can never
    again be read without knowing whether anything was measured."""
    a = _generation_tree(tmp_path / "a", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
    b = _generation_tree(tmp_path / "b", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3])
    _band_verdict(a, b)
    assert "over 2 channels" in capsys.readouterr().out
