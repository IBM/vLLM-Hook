"""MIA is V2-only, and in graph mode that guarantee was not real.

`mia/graph/install.py::patch_worker_load_model` wraps `graph_install()` in a blanket
`except Exception` whose stated intent is "Never let an install failure take down model
loading -- fall back to no capture rather than crashing." That handler predates this
branch. This branch then added `require_v2_runner()` into `graph_install()`'s call chain
(via `install_prepare_inputs_routing`), so the V2-only refusal -- which the plan requires to
RAISE, never degrade -- was swallowed: the engine booted, MIA loaded, NOTHING installed,
`generate()` succeeded, and nothing was captured. Reporting success while capturing nothing
is the single failure mode the whole parity suite exists to prevent.

The defect shape is why it survived review: an added RAISE underneath a PRE-EXISTING
handler. Auditing added handlers finds nothing.

It is closed at two depths, tested here at both:

  * CONFIG TIME (the better one -- before a model is ever loaded):
    `_plugin.validate_v2_runner_selected` refuses a config whose
    `VllmConfig.use_v2_model_runner` is False. That property is False, with
    VLLM_USE_V2_MODEL_RUNNER UNSET, for ngram spec-decode, sequence parallelism at TP>1,
    STOCK_TORCH_COMPILE, PP with the external launcher, some ROCm architectures and a
    missing Triton -- none of which the existing env-var check sees.
  * INSTALL TIME: deliberate refusals carry the `MiaRefusal` marker and the handler
    re-raises them, keeping its blanket fallback for what it was written for.

Hermetic: no GPU, no engine. The install-time arm drives the REAL handler by patching the
real `Worker.load_model` under `monkeypatch`, so it tests the shipped code path rather than
a re-implementation of it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM (mia/llm.py); skip, never error the whole collection

from types import SimpleNamespace

from mia._plugin import validate_v2_runner_selected
from mia.errors import MiaConfigurationError, MiaRefusal, MiaSizingError
from mia.graph import install as graph_install_mod
from mia.runner import UnsupportedRunnerError


# ---------------------------------------------------------------------------------------
# Config time.
# ---------------------------------------------------------------------------------------

def test_a_config_that_resolves_to_v1_is_refused():
    """The reviewer's breaking input: V1 selected by vLLM's own logic, with the env unset,
    so `_patched_create_engine_config`'s VLLM_USE_V2_MODEL_RUNNER check never fires."""
    with pytest.raises(UnsupportedRunnerError) as e:
        validate_v2_runner_selected(SimpleNamespace(use_v2_model_runner=False))
    assert "V2 model runner" in str(e.value)


def test_a_v2_config_passes():
    validate_v2_runner_selected(SimpleNamespace(use_v2_model_runner=True))


def test_an_unreadable_property_does_not_invent_a_verdict():
    """Declines to judge rather than guessing -- the install-time gate is the backstop. A
    future vLLM that removes or renames the property must not turn every engine into a
    refusal, and must not turn one into a silent pass either (install still checks)."""
    class Boom:
        @property
        def use_v2_model_runner(self):
            raise AttributeError("gone in some future vLLM")

    validate_v2_runner_selected(Boom())          # no raise
    validate_v2_runner_selected(SimpleNamespace())  # attribute absent: no raise


def test_the_refusal_is_a_mia_refusal_so_handlers_cannot_swallow_it():
    assert issubclass(UnsupportedRunnerError, MiaRefusal)
    assert issubclass(UnsupportedRunnerError, RuntimeError)   # unchanged for existing callers


# ---------------------------------------------------------------------------------------
# Install time -- the real handler, driven for real.
# ---------------------------------------------------------------------------------------

@pytest.fixture
def patched_worker(monkeypatch):
    """Install MIA's real load_model patch over a stub Worker.load_model, and hand back a
    callable that runs it with a chosen graph_install. monkeypatch restores both the class
    attribute and MIA's idempotence flag."""
    from vllm.v1.worker.gpu_worker import Worker

    monkeypatch.setattr(Worker, "load_model", lambda self, *a, **k: "model-loaded",
                        raising=False)
    monkeypatch.setattr(graph_install_mod, "_LOAD_MODEL_PATCHED", False, raising=False)
    monkeypatch.setenv("MIA_GRAPH_MODE", "1")          # graph mode armed
    graph_install_mod.patch_worker_load_model()
    patched = Worker.load_model

    def run(graph_install):
        return patched(SimpleNamespace(graph_install=graph_install))

    return run


def _raiser(exc):
    def graph_install():
        raise exc
    return graph_install


@pytest.mark.parametrize("exc", [
    UnsupportedRunnerError("live runner is V1"),
    MiaConfigurationError("MIA_QK_CAPTURE=op is no longer supported"),
    MiaSizingError("capture aperture too small: R<1"),
])
def test_a_deliberate_refusal_propagates_out_of_load_model(patched_worker, exc):
    """THE REGRESSION. Before the fix every one of these was swallowed and the engine
    carried on with no capture installed, reporting success for the rest of the run."""
    with pytest.raises(type(exc)):
        patched_worker(_raiser(exc))


def test_an_unexpected_install_failure_still_degrades(patched_worker, capsys):
    """CONTROL ARM. Without it, the test above would pass just as well if someone deleted
    the handler outright -- which would be a different regression, not a fix. The blanket
    fallback must still do the job it was written for."""
    result = patched_worker(_raiser(ZeroDivisionError("something genuinely unexpected")))
    assert result == "model-loaded"
    assert "graph install FAILED" in capsys.readouterr().out


def test_graph_install_is_not_called_when_graph_mode_is_off(monkeypatch, patched_worker):
    """The refusal must not fire on the eager path, which does not install the graph hooks
    at all -- otherwise this fix would break every eager run."""
    monkeypatch.setenv("MIA_GRAPH_MODE", "0")
    monkeypatch.setattr(graph_install_mod, "_graph_mode_enabled", False, raising=False)
    assert patched_worker(_raiser(UnsupportedRunnerError("would fire"))) == "model-loaded"


# ---------------------------------------------------------------------------------------
# The audit: every deliberate refusal reachable from graph_install() carries the marker.
# ---------------------------------------------------------------------------------------

def test_every_deliberate_refusal_class_carries_the_marker():
    """The fix is a CLASS of defect, not one site. These are the refusals reachable from
    graph_install(): the V2-runner requirement, the removed MIA_QK_CAPTURE/MIA_HS_CAPTURE
    modes, the unsupported worker-wide MIA_QK_SCORE default, an unresolvable
    max_num_batched_tokens, an aperture too small to hold one row, and the
    gpu_memory_utilization fit check. Each was documented as fail-loud and each was silently
    absorbed; all now inherit MiaRefusal."""
    for cls in (UnsupportedRunnerError, MiaConfigurationError, MiaSizingError):
        assert issubclass(cls, MiaRefusal)
    # ...and the sizing refusal stays a ValueError, so existing callers and tests that
    # catch ValueError keep working.
    assert issubclass(MiaSizingError, ValueError)
    assert issubclass(MiaConfigurationError, RuntimeError)


def test_the_aperture_fit_check_is_now_a_refusal():
    """The fit check's own module docstring says 'Fails loud (never a silent degrade)' -- it
    was reached from graph_install() and swallowed, which made that claim false in graph
    mode."""
    from mia.graph.aperture_sizing import resolve_aperture_bytes_fixed
    with pytest.raises(MiaRefusal):
        resolve_aperture_bytes_fixed(80 << 30, 60 << 30, 0.85)
    with pytest.raises(ValueError):          # unchanged for anyone catching the builtin
        resolve_aperture_bytes_fixed(80 << 30, 60 << 30, 0.85)


# ---------------------------------------------------------------------------------------
# The config-time gate's WIRING, not just the function.
# ---------------------------------------------------------------------------------------

def _drive_seam(monkeypatch, use_v2):
    """Run the real `_patched_create_engine_config` with vLLM's own method stubbed out, so
    the refusal is reached the way `vllm serve` reaches it."""
    import mia._plugin as plugin

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode="NONE"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        use_v2_model_runner=use_v2,
    )
    monkeypatch.setattr(plugin, "_original_create_engine_config",
                        lambda self, *a, **k: config, raising=False)
    monkeypatch.delenv("MIA_WORKER", raising=False)
    monkeypatch.delenv("MIA_ALLOW_CUDAGRAPH", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    args = SimpleNamespace(worker_extension_cls=None, enforce_eager=False,
                           compilation_config=None)
    return plugin._patched_create_engine_config(args)


def test_the_engine_config_seam_actually_calls_the_v2_gate(monkeypatch):
    """Pins the CALL SITE, not just the validator. Testing the function alone would keep
    passing if someone deleted the one line that invokes it -- which is precisely the shape
    of this whole item: a correct refusal that nothing reaches."""
    with pytest.raises(UnsupportedRunnerError):
        _drive_seam(monkeypatch, use_v2=False)


def test_the_seam_builds_a_v2_config_normally(monkeypatch):
    """CONTROL ARM: the same path with V2 selected returns the config untouched, so the
    test above is not passing because the seam raises for some unrelated reason."""
    config = _drive_seam(monkeypatch, use_v2=True)
    assert config.use_v2_model_runner is True
