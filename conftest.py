"""Repo-root conftest: the exit criterion must not be able to pass having run nothing.

`pytest tests/ -q -m "not gpu"` is this branch's exit criterion. An exit criterion that
reports success without executing anything is a check that cannot fail -- the same defect
class as everything else this branch has been closing.

That is not hypothetical. `tests/conftest.py` used to call `pytest.importorskip("vllm")`
as its first statement, so a missing or half-installed vLLM, or simply the wrong
interpreter, aborted collection of the whole `tests/` tree before a single test ran.
Depending on pytest version and invocation shape that surfaces either as a raw traceback
or as a directory-level skip with a green exit 0. A guard placed inside `tests/` cannot
catch it, because the guard is skipped along with everything else -- which is precisely
why this file exists one level up, imports nothing beyond the standard library, and is
loaded whether or not `tests/conftest.py` survives.

The import guard itself has also been pushed down to the eleven modules that genuinely
need vLLM (they call `pytest.importorskip("vllm")` themselves), so a missing vLLM now
skips those and leaves the several hundred pure-Python tests running and honest.

HONEST STATUS OF THE HOOK BELOW. Measured, not assumed: on the pinned pytest (9.1.1)
every zero-collected shape reachable here -- conftest skipping at import as an initial
arg, the same skip discovered during collection, an empty directory, a selector matching
nothing -- ALREADY exits non-zero on its own (1, 5, 5, 5). So this hook does not fire
today; it is belt to pytest's braces, not the thing currently doing the work. It is kept
because pytest's exit-code behaviour is not a contract MIA controls: a version bump, a
plugin, or a CI wrapper that maps NO_TESTS_COLLECTED to success would silently restore
the hole, and this file is one narrow line of defence that costs nothing at runtime. The
property it protects is pinned END-TO-END by exit code, independently of which layer
provides it, in tests/test_gate_integrity.py -- that is the check that can actually fail.
"""

# pytest's own exit code for a session that collected no tests. Reused rather than
# invented so `echo $?` means the same thing it always did.
_NO_TESTS_COLLECTED = 5


def pytest_sessionfinish(session, exitstatus):
    """Refuse to exit 0 from a session that ran no tests.

    Deliberately NARROW. It fires only when pytest was about to report success, and only
    when the selected-test count is zero, so:

      * it can never mask a real failure -- a non-zero status is returned untouched, with
        its own reason intact;
      * it can never fire on a run that executed tests, however many were deselected or
        skipped;
      * an intentionally empty selection (`-k nothing_matches_this`) is NOT turned into a
        confusing new failure, because pytest already exits 5 there on its own and this
        hook returns early. It stays a "no tests collected" outcome and says so.

    `session.testscollected` is the post-deselection count (set by `perform_collect` after
    `pytest_collection_modifyitems`), which is the number that matters here: tests that
    actually ran, not tests that existed before filtering.
    """
    if int(exitstatus) != 0:
        return                                   # already failing; leave the real reason alone
    if getattr(session, "testscollected", 0) != 0:
        return                                   # tests ran; nothing to police
    print(
        "\n[conftest] REFUSING to exit 0: the session collected 0 tests. A green run that "
        "executed nothing is not a passing gate. Usual causes: collection was aborted "
        "(a conftest raised or skipped at import), the path/selector matched nothing, or "
        "the wrong interpreter was used. Exiting "
        f"{_NO_TESTS_COLLECTED} (pytest's NO_TESTS_COLLECTED)."
    )
    session.exitstatus = _NO_TESTS_COLLECTED
