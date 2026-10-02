"""MIA_WORKER is parsed ONCE, by both of its readers, and it fails loud (Task E3, folded fix 1).

The bug this pins: `_patched_create_engine_config` used to compare MIA_WORKER raw and fall
through to the hidden-states worker for anything it did not recognize, while `_worker_kind`
-- which sizes the autocap guard -- lowercased and stripped it first. Two consequences, both
SILENT:

  * `MIA_WORKER=probe_qk` (which the profiling harness really shipped, in both campaign
    templates and its README, and whose record normalizer independently emitted
    `capture_qk`) ran the HS worker. The run captured hidden states, wrote artifacts and
    reported a clean QK campaign.
  * `MIA_WORKER=QK` sized the OOM guard for QK while running HS -- one env var, two answers.

Both readers now call `parse_mia_worker_env`, and it raises on anything that is not an exact
accepted spelling. The tests below cover each bad-value shape, and assert the two readers
agree on every accepted spelling.

Hermetic: no GPU, no engine. The engine-config seam is exercised for real by stubbing
vLLM's `create_engine_config` out from under it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

import mia._plugin as plugin
from mia._plugin import (
    DEFAULT_MIA_WORKER,
    MIA_WORKER_VALUES,
    UnknownMiaWorkerError,
    _WORKER_EXT_BY_KIND,
    _worker_kind,
    parse_mia_worker_env,
)

# Exactly the shapes that used to be swallowed: a plausible near-miss from the harness, a
# registry kind that is not an env value, a case variant, two abbreviations, a hyphen, and
# stray whitespace from a YAML scalar.
BAD_VALUES = [
    "probe_qk",
    "capture_qk",
    "capture_hs",
    "QK",
    "Hidden_States",
    "hs",
    "hidden-states",
    "qk ",
    " qk",
    "\tsteer\n",
    "hidden_states,qk",
    "none",
    "0",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MIA_WORKER", raising=False)
    monkeypatch.delenv("MIA_ALLOW_CUDAGRAPH", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)


# --------------------------------------------------------------------------------------
# The parser itself.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", MIA_WORKER_VALUES)
def test_accepted_values_parse_to_themselves(value):
    assert parse_mia_worker_env(value) == value


def test_unset_and_empty_mean_unspecified():
    """None, not a kind: each caller then applies its own documented default (the engine-config
    seam installs hidden_states; _worker_kind falls back to the extension-class string)."""
    assert parse_mia_worker_env(None) is None
    assert parse_mia_worker_env("") is None


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_unrecognized_values_raise_rather_than_defaulting_to_hs(bad):
    with pytest.raises(UnknownMiaWorkerError) as e:
        parse_mia_worker_env(bad)
    msg = str(e.value)
    assert repr(bad) in msg, "the message must show the value received, whitespace and all"
    for accepted in MIA_WORKER_VALUES:
        assert accepted in msg, "the message must name the accepted values"


def test_no_aliases_are_accepted():
    """This branch chose a full rename with no back-compat spellings. A parser that quietly
    normalized `QK` or `hs` would re-open the hole one notch down."""
    for alias in ("hs", "hidden", "HS", "QK", "Steer", "act_steer", "steer_hook_act"):
        with pytest.raises(UnknownMiaWorkerError):
            parse_mia_worker_env(alias)


def test_every_accepted_value_routes_to_a_worker_extension():
    assert set(_WORKER_EXT_BY_KIND) == set(MIA_WORKER_VALUES)
    assert len(set(_WORKER_EXT_BY_KIND.values())) == len(MIA_WORKER_VALUES)
    assert DEFAULT_MIA_WORKER in MIA_WORKER_VALUES


# --------------------------------------------------------------------------------------
# Reader 2: _worker_kind (sizes the autocap guard).
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", BAD_VALUES)
def test_worker_kind_raises_on_a_bad_env_value(monkeypatch, bad):
    monkeypatch.setenv("MIA_WORKER", bad)
    with pytest.raises(UnknownMiaWorkerError):
        _worker_kind("mia.workers.hs_capture_worker.HSCaptureWorker")


def test_worker_kind_still_falls_back_to_the_class_string_when_unset():
    assert _worker_kind("mia.workers.qk_capture_worker.QKCaptureWorker") == "qk"
    assert _worker_kind("mia.workers.steer_worker.SteerWorker") == "steer"
    assert _worker_kind("mia.workers.hs_capture_worker.HSCaptureWorker") == "hidden_states"


def test_worker_kind_treats_an_empty_env_value_as_unset(monkeypatch):
    monkeypatch.setenv("MIA_WORKER", "")
    assert _worker_kind("mia.workers.qk_capture_worker.QKCaptureWorker") == "qk"


# --------------------------------------------------------------------------------------
# Reader 1: the engine-config seam -- driven for real, with vLLM stubbed out.
# --------------------------------------------------------------------------------------

class _FakeArgs:
    """Stands in for vLLM's EngineArgs at the create_engine_config seam."""

    def __init__(self):
        self.worker_extension_cls = None
        self.enforce_eager = False
        self.compilation_config = None


@pytest.fixture
def seam(monkeypatch):
    """Run the REAL `_patched_create_engine_config` with vLLM's own method stubbed out, so
    the worker-selection block is exercised exactly as it runs in `vllm serve`."""
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode="NONE"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    monkeypatch.setattr(plugin, "_original_create_engine_config",
                        lambda self, *a, **k: config, raising=False)

    def run():
        args = _FakeArgs()
        plugin._patched_create_engine_config(args)
        return args.worker_extension_cls

    return run


@pytest.mark.parametrize("value", MIA_WORKER_VALUES)
def test_seam_installs_the_worker_the_env_names(monkeypatch, seam, value):
    monkeypatch.setenv("MIA_WORKER", value)
    assert seam() == _WORKER_EXT_BY_KIND[value]


def test_seam_defaults_to_hidden_states_when_unset(seam):
    assert seam() == _WORKER_EXT_BY_KIND[DEFAULT_MIA_WORKER]


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_seam_refuses_a_bad_env_value_instead_of_installing_hs(monkeypatch, seam, bad):
    """The regression proper: this used to silently install the HS worker and run a whole
    campaign for the wrong subsystem."""
    monkeypatch.setenv("MIA_WORKER", bad)
    with pytest.raises(UnknownMiaWorkerError):
        seam()


def _seam_with_explicit_class(monkeypatch, kind):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode="NONE"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    monkeypatch.setattr(plugin, "_original_create_engine_config",
                        lambda self, *a, **k: config, raising=False)
    args = _FakeArgs()
    args.worker_extension_cls = _WORKER_EXT_BY_KIND[kind]
    plugin._patched_create_engine_config(args)
    return args


def test_seam_does_not_override_an_explicitly_set_worker(monkeypatch, seam):
    """Offline MiaLLM sets worker_extension_cls itself; MIA_WORKER must not second-guess it.
    With the env AGREEING (or unset) the class is passed through untouched."""
    monkeypatch.setenv("MIA_WORKER", "qk")
    assert _seam_with_explicit_class(monkeypatch, "qk").worker_extension_cls \
        == _WORKER_EXT_BY_KIND["qk"]
    monkeypatch.delenv("MIA_WORKER", raising=False)
    assert _seam_with_explicit_class(monkeypatch, "qk").worker_extension_cls \
        == _WORKER_EXT_BY_KIND["qk"]


def test_seam_refuses_an_env_that_contradicts_the_explicit_worker_class(monkeypatch, seam):
    """MIA_WORKER=qk with MiaLLM(worker_name="capture_hs") used to RUN HS while stamping the
    compile-cache key and sizing the autocap for QK. Now it refuses at config time."""
    monkeypatch.setenv("MIA_WORKER", "qk")
    with pytest.raises(plugin.MiaWorkerConflictError):
        _seam_with_explicit_class(monkeypatch, "hidden_states")


# --------------------------------------------------------------------------------------
# The whole point: the two readers cannot drift apart again.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", MIA_WORKER_VALUES)
def test_both_readers_agree_on_every_accepted_spelling(monkeypatch, seam, value):
    """The worker that gets INSTALLED and the kind the autocap guard SIZES for must be the
    same subsystem, for every spelling MIA accepts."""
    monkeypatch.setenv("MIA_WORKER", value)
    installed = seam()
    sized = _worker_kind(installed)
    assert sized == value
    assert _WORKER_EXT_BY_KIND[sized] == installed


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_both_readers_reject_the_same_bad_values(monkeypatch, bad):
    """Neither reader may be the lenient one -- that asymmetry WAS the bug."""
    monkeypatch.setenv("MIA_WORKER", bad)
    with pytest.raises(UnknownMiaWorkerError):
        parse_mia_worker_env(bad)
    with pytest.raises(UnknownMiaWorkerError):
        _worker_kind("")
