"""MIA's deliberate refusals, and the marker that keeps them from being swallowed.

MIA installs itself into vLLM's internals, so its install and per-step paths are wrapped in
defensive handlers whose job is availability: an UNEXPECTED failure in an internal vLLM
surface should not take the engine down. That is a reasonable policy for surprises.

It is exactly the wrong policy for a refusal MIA raised ON PURPOSE. A deliberate refusal --
"this runner is V1 and MIA is V2-only", "this aperture cannot hold one row", "this removed
capture mode no longer exists" -- is a statement that continuing would produce a run that
reports success while capturing nothing. Swallowing it converts a loud, correct error into
precisely the silent failure the refusal was written to prevent.

MEASURED, not theoretical: `patch_worker_load_model`'s handler in graph/install.py was
written before this branch, with the comment "Never let an install failure take down model
loading -- fall back to no capture rather than crashing." This branch then added
`require_v2_runner()` into `graph_install()`'s call chain. The V2-only guarantee -- which
the plan requires to RAISE, never degrade -- was therefore swallowed in graph mode: the
engine booted, MIA loaded, nothing installed, `generate()` succeeded, nothing was captured.
The defect shape is an added RAISE inside a pre-existing handler, which is invisible to an
audit that only looks at added handlers.

So every deliberate refusal inherits `MiaRefusal`, and the defensive handlers re-raise it
before their blanket `except Exception`. Adding a new refusal is then one decision -- pick
the right base class -- rather than a reminder to go and edit every handler that might
someday sit above it.

The concrete classes also inherit the builtin their call sites and tests already expect
(`RuntimeError`, `ValueError`), so `except ValueError` and `pytest.raises(ValueError)`
keep working unchanged.
"""


class MiaRefusal(Exception):
    """Marker: MIA refused on purpose, and continuing would capture nothing in silence.

    Never catch this in a defensive/availability handler. Handlers that exist to absorb
    unexpected failures must re-raise it explicitly -- see
    ``graph/install.py::patch_worker_load_model``.
    """


class MiaConfigurationError(MiaRefusal, RuntimeError):
    """A configuration MIA does not support, detected at config or install time.

    Deterministic and known before any token is produced: the same inputs will refuse
    every time, so there is nothing to degrade gracefully into.
    """


class MiaSizingError(MiaRefusal, ValueError):
    """A capture aperture that cannot be sized as asked.

    ``ValueError`` as well, because the sizing helpers are pure functions whose callers
    and tests already treat a bad size as a ValueError.
    """
