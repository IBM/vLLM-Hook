"""MIA's deliberate refusal errors, marked so they are never swallowed."""


class MiaRefusal(Exception):
    """Marker for deliberate MIA refusals; continuing would silently capture nothing."""


class MiaConfigurationError(MiaRefusal, RuntimeError):
    """A configuration MIA does not support, detected at config or install time."""


class MiaSizingError(MiaRefusal, ValueError):
    """A capture aperture that cannot be sized as asked."""

