"""Scope -> row selection, shared by interventions and capture.

Pure selection over a pass view's absolute positions; no engine imports.
The view only needs ``positions`` (the absolute positions this pass's rows
occupy) and ``prompt_len``, so any object with those attributes works.
"""
from __future__ import annotations


def scope_rows(view, scope) -> slice | None:
    """Rows of ``view.row_slice`` selected by ``scope``, or None when the
    scope selects nothing this pass.

    The returned slice is relative to the request's own rows (0 = the
    first row of ``view.row_slice``). Selection semantics per kind:

    - ``all``: every row of the pass.
    - ``after_prompt``: rows at absolute positions >= ``prompt_len``.
    - ``from_position``: rows at absolute positions >= ``position``.
    - ``last_k``: the trailing ``k`` rows of the pass (during decode every
      pass has one row, so any k >= 1 selects it; during prefill this is
      the last ``k`` prompt rows of the chunk).
    """
    positions = view.positions
    num_rows = len(positions)
    if num_rows == 0:
        return None

    if scope.kind == "all":
        return slice(0, num_rows)
    if scope.kind == "after_prompt":
        first = max(0, view.prompt_len - positions.start)
    elif scope.kind == "from_position":
        first = max(0, scope.params["position"] - positions.start)
    elif scope.kind == "last_k":
        first = max(0, num_rows - scope.params["k"])
    else:
        raise KeyError(f"no row selection for scope kind {scope.kind!r}")

    if first >= num_rows:
        return None
    return slice(first, num_rows)
