"""Regression test for ``mia._plugin._worker_kind``.

This heuristic feeds the opt-in autocap OOM guard's per-token-byte sizing
(``_derive_safe_max_batched_tokens``): mis-detecting a QK worker as
"hidden_states" silently applies the wrong formula, with no error.

It broke once already: the extension-class fallback was written against the
OLD class path ``mia.workers.probe_hookqk_worker.ProbeHookQKWorker`` (whose
lowercased form contained the literal "hookqk"), and the A5 rename to
``mia.workers.qk_capture_worker.QKCaptureWorker`` silently broke it because
the new lowercased class path no longer contains that substring. This test
pins ``_worker_kind`` against the three REAL class paths so the next rename
cannot silently re-break dispatch.
"""
from __future__ import annotations

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from mia._plugin import MiaWorkerConflictError, _worker_kind

# The three real worker_extension_cls dotted paths, as of the A5 rename.
QK_PATH = "mia.workers.qk_capture_worker.QKCaptureWorker"
HS_PATH = "mia.workers.hs_capture_worker.HSCaptureWorker"
STEER_PATH = "mia.workers.steer_worker.SteerWorker"


@pytest.fixture(autouse=True)
def _clean_mia_worker_env(monkeypatch):
    """MIA_WORKER must be unset by default so the class-path fallback is exercised."""
    monkeypatch.delenv("MIA_WORKER", raising=False)


# --- extension-class string fallback (MIA_WORKER unset -- the offline MiaLLM path) ---

def test_qk_class_path_resolves_to_qk():
    assert _worker_kind(QK_PATH) == "qk"


def test_hs_class_path_resolves_to_hidden_states():
    assert _worker_kind(HS_PATH) == "hidden_states"


def test_steer_class_path_resolves_to_steer():
    assert _worker_kind(STEER_PATH) == "steer"


def test_bare_class_name_also_resolves(monkeypatch=None):
    # worker_ext can also arrive as a class object; _worker_kind falls back to
    # __name__, which has no module prefix and no underscores.
    assert _worker_kind("QKCaptureWorker") == "qk"
    assert _worker_kind("HSCaptureWorker") == "hidden_states"
    assert _worker_kind("SteerWorker") == "steer"


# --- MIA_WORKER env precedence (the vllm-serve path) ---

@pytest.mark.parametrize(
    "env_value,expected",
    [("qk", "qk"), ("steer", "steer"), ("hidden_states", "hidden_states")],
)
def test_mia_worker_env_values_resolve(monkeypatch, env_value, expected):
    monkeypatch.setenv("MIA_WORKER", env_value)
    # Pass an irrelevant/empty extension string -- env alone must decide.
    assert _worker_kind("") == expected


# --- MIA_WORKER vs an EXPLICIT MIA worker class: a contradiction, not a default ---
#
# This used to assert that the env silently won. It does not any more, and the old
# behaviour was the bug: MIA_WORKER=qk with MiaLLM(worker_name="capture_hs") RAN the HS
# worker while stamping the compile-cache key and sizing the autocap for QK -- defeating
# the stamp whose whole job is to stop a cross-worker compiled artifact being loaded, and
# producing the `KeyError: '_mia_hs_host'` it exists to prevent.

@pytest.mark.parametrize(
    "env_value,class_path",
    [("steer", QK_PATH), ("qk", HS_PATH), ("hidden_states", STEER_PATH)],
)
def test_env_disagreeing_with_an_explicit_mia_worker_class_raises(monkeypatch, env_value, class_path):
    monkeypatch.setenv("MIA_WORKER", env_value)
    with pytest.raises(MiaWorkerConflictError) as e:
        _worker_kind(class_path)
    assert env_value in str(e.value) and class_path in str(e.value), (
        "the error must name BOTH sides of the contradiction")


@pytest.mark.parametrize("value,path", [("qk", QK_PATH), ("hidden_states", HS_PATH),
                                        ("steer", STEER_PATH)])
def test_env_agreeing_with_the_class_is_not_a_conflict(monkeypatch, value, path):
    monkeypatch.setenv("MIA_WORKER", value)
    assert _worker_kind(path) == value


@pytest.mark.parametrize("env_value", ["qk", "steer", "hidden_states"])
def test_env_still_wins_over_a_NON_mia_class(monkeypatch, env_value):
    """The case the precedence was written for, and now the only case it covers: the
    profiler swaps in its OWN mixin and stashes the real worker elsewhere, so the class in
    hand names no subsystem and only the env knows which one is running."""
    monkeypatch.setenv("MIA_WORKER", env_value)
    for foreign in ("profiling_harness.mixins.ProfHarvestMixin",
                    "some.other.Thing", "", None):
        assert _worker_kind(foreign) == env_value


def test_a_non_mia_class_with_no_env_still_defaults_to_hidden_states():
    """Unchanged fallback: nothing names a subsystem, so the documented default applies."""
    assert _worker_kind("profiling_harness.mixins.ProfHarvestMixin") == "hidden_states"
    assert _worker_kind("") == "hidden_states"
