"""Aperture sizing.

The shared GPU capture aperture is a FIXED size, resolved by ``resolve_aperture_bytes_auto``:
``MIA_APERTURE_GPU_BYTES`` (bytes) if set -- an explicit value always wins -- else the default
4 GiB, grown at install to ``max_num_batched_tokens x (one token across every captured layer)``
when that is larger (a 70B all-layer HS capture needs 10 GiB for one 8192-token step; 4 GiB
held a third of it). The install gate is a *fit check* — the aperture must fit the free margin
left by ``gpu_memory_utilization``; a model-sized default that does not fit refuses at install.

A fixed aperture makes the free margin a known constant — which is what the auto
``max_num_batched_tokens`` derivation (compute_safe_max_batched_tokens) sizes its per-step
transient budget against.

``resolve_aperture_bytes`` (bottom of this module) layers a *set* of enabled subsystems on
top of that budget: one capture subsystem gets it verbatim, several split it. It never
changes how the budget itself is resolved.

Fails loud (never a silent degrade)."""
import os
from typing import Optional, Tuple

from mia.errors import MiaSizingError

# Default fixed GPU capture-aperture size: 4 GiB.
DEFAULT_APERTURE_GPU_BYTES = 4 * (1 << 30)

# Auto-cap defaults. SAFETY covers the buffering copies (egress gather + pinned staging +
# not-yet-drained clones) plus any activation the startup profiling under-counts because it never
# runs capture; HEADROOM is fragmentation slack. Both overridable by env for tuning.
DEFAULT_AUTOCAP_SAFETY = 3
DEFAULT_AUTOCAP_HEADROOM_BYTES = 1 << 30   # 1 GiB

_TRUE = ("1", "true", "on", "yes", "auto")
_FALSE = ("", "0", "false", "off", "no")


def resolve_aperture_bytes_fixed(total_gpu_bytes: int, fixed_bytes: int, gpu_mem_util: float) -> int:
    """Fixed-size aperture path: return ``fixed_bytes`` verbatim after a fit check. The aperture must fit the
    free margin ``(1 - gpu_mem_util) x total`` left after vLLM's KV commitment; otherwise raise
    (spec: fail loud, never silent degrade)."""
    fixed_bytes = int(fixed_bytes)
    total_gpu_bytes = int(total_gpu_bytes)
    free_margin = (1.0 - gpu_mem_util) * total_gpu_bytes
    if fixed_bytes > free_margin:
        raise MiaSizingError(
            f"aperture {fixed_bytes/(1<<30):.2f} GiB + gpu_memory_utilization={gpu_mem_util} leaves no "
            f"room (free margin {free_margin/(1<<30):.2f} GiB): lower gpu_memory_utilization or "
            f"MIA_APERTURE_GPU_BYTES")
    return fixed_bytes


def resolve_aperture_bytes_auto(total_gpu_bytes: int, gpu_mem_util: float,
                                rows_needed: Optional[int] = None,
                                row_bytes: Optional[int] = None,
                                what: str = "capture") -> int:
    """Resolve the aperture byte budget: ``MIA_APERTURE_GPU_BYTES`` if set, else the default.
    Applies the fit-check gate either way.

    * EXPLICIT ``MIA_APERTURE_GPU_BYTES`` ALWAYS WINS. It is used verbatim (fit-checked), never
      grown or shrunk to the model -- an operator who sized it (the profiling harness computes
      ``rows x sum(captured width) x 2``) gets exactly that aperture. If it holds fewer rows than
      one max-token step the install WARNS; the risk is theirs to take.
    * DEFAULT: ``DEFAULT_APERTURE_GPU_BYTES`` (4 GiB), grown to ``rows_needed x row_bytes`` when
      the caller passes them (``rows_needed`` = the scheduler's ``max_num_batched_tokens``,
      ``row_bytes`` = one captured token across every captured layer of THIS rank). A fixed 4 GiB
      holds 16384 all-layer HS rows of Llama-3.1-8B but only 3276 of Llama-3.1-70B (80 layers x
      8192 hidden x 2 B = 1.25 MiB/row) -- fewer than one 8192-token prefill step, so the
      reserve could never succeed and ``ApertureBackpressureError`` fired mid-run after the
      backpressure timeout. With both args given the default can no longer be smaller than one
      step. If the grown default does not fit the free margin, this REFUSES at install with the
      numbers and the knob to turn (``MiaSizingError``), instead of booting a capture that will
      die on its first long prefill. Without the args the result is the legacy fixed default.
    """
    raw_fixed = os.environ.get("MIA_APERTURE_GPU_BYTES")
    if raw_fixed is not None and raw_fixed.strip() != "":
        return resolve_aperture_bytes_fixed(total_gpu_bytes, int(raw_fixed), gpu_mem_util)
    budget = DEFAULT_APERTURE_GPU_BYTES
    need = model_sized_aperture_bytes(rows_needed, row_bytes)
    if need is not None and need > budget:
        free_margin = (1.0 - gpu_mem_util) * int(total_gpu_bytes)
        if need > free_margin:
            raise MiaSizingError(
                f"{what} aperture cannot hold one max-token step by default: "
                f"{int(rows_needed)} rows (max_num_batched_tokens) x {int(row_bytes)} B/row "
                f"(every captured layer) = {need / (1 << 30):.2f} GiB, but "
                f"gpu_memory_utilization={gpu_mem_util} leaves {free_margin / (1 << 30):.2f} GiB. "
                f"Lower gpu_memory_utilization, lower max_num_batched_tokens, capture fewer "
                f"layers, or set MIA_APERTURE_GPU_BYTES explicitly (an explicit value always "
                f"wins; one smaller than {need} B risks ApertureBackpressureError on a "
                f"max-token step).")
        budget = need
    return resolve_aperture_bytes_fixed(total_gpu_bytes, budget, gpu_mem_util)


def model_sized_aperture_bytes(rows_needed: Optional[int], row_bytes: Optional[int]) -> Optional[int]:
    """``rows_needed x row_bytes`` -- the aperture one max-token step fills -- or None when either
    is unknown / non-positive (the caller then keeps the fixed default)."""
    try:
        rows, rb = int(rows_needed), int(row_bytes)
    except (TypeError, ValueError):
        return None
    if rows <= 0 or rb <= 0:
        return None
    return rows * rb


def aperture_bytes_is_explicit() -> bool:
    """True when ``MIA_APERTURE_GPU_BYTES`` is set (and therefore wins over any derived size)."""
    raw = os.environ.get("MIA_APERTURE_GPU_BYTES")
    return raw is not None and raw.strip() != ""


# ---------------------------------------------------------------------------
# Auto-derive max_num_batched_tokens (the OOM fix).
#
# Under offline lock-step at high batch, a large synchronized prefill's per-step transient (egress
# gather + pinned staging + undrained clones) scales with tokens-processed-per-step, which is ~batch.
# vLLM's startup profiling never runs capture, so this transient is un-budgeted and overflows the thin
# free margin. Capping the scheduler's per-step token budget chunks the prefill and bounds the
# transient INDEPENDENT of batch (chunked prefill is already proven capture-chunk-invariant). These
# pure helpers compute the safe cap; _plugin applies it MIN-ONLY at the create_engine_config seam.
# ---------------------------------------------------------------------------

def per_layer_token_bytes_hs(hidden_size: int, dtype_size: int) -> int:
    """HS: one captured token costs one residual-stream row per layer."""
    return int(hidden_size) * int(dtype_size)


def per_layer_token_bytes_qk(n_q_heads: int, n_kv_heads: int, head_dim: int, dtype_size: int) -> int:
    """QK: one captured token costs (H_q + H_kv) * head_dim elements per layer ON ONE RANK.

    Pass the rank's LOCAL head counts. Under tensor parallelism each rank's buffers hold only its
    own shard (``graph/tp_shard.qk_shard``: ``H_q / tp`` query heads, ``max(1, H_kv / tp)`` KV
    heads), and every rank captures, so the per-rank transient is the SHARDED width -- the full
    counts would over-reserve by ``tp`` x. ``_plugin._model_dims`` does the sharding; at TP=1
    local == global."""
    return (int(n_q_heads) + int(n_kv_heads)) * int(head_dim) * int(dtype_size)


def compute_safe_max_batched_tokens(
    total_gpu_bytes: int,
    gpu_mem_util: float,
    aperture_gpu_bytes: int,
    n_layers_captured: int,
    per_layer_token_bytes: int,
    safety: int = DEFAULT_AUTOCAP_SAFETY,
    headroom_bytes: int = DEFAULT_AUTOCAP_HEADROOM_BYTES,
) -> Optional[int]:
    """Largest per-step token budget whose worst-case (all-token, all-layer) capture transient fits
    the free GPU margin left after ``gpu_memory_utilization`` + the fixed aperture + a fragmentation
    headroom.

    ``free_margin = (1 - util) * total - aperture - headroom``; ``bytes_per_token = n_layers * per_layer``;
    ``cap = floor(free_margin / (bytes_per_token * safety))``. Returns None (caller makes NO change)
    when there is no usable margin or the inputs are degenerate — never raises (min-only contract)."""
    total_gpu_bytes = int(total_gpu_bytes)
    # Round the (1-util)*total product to an int before the integer subtraction: `1.0 - 0.9` is
    # 0.0999... in float, which would drop the exact 3 GiB margin to a hair under and cost the
    # canonical cap a spurious -1 (4096 -> 4095). The rounding error is < 1 byte, safety-neutral.
    free_margin = round((1.0 - gpu_mem_util) * total_gpu_bytes) - int(aperture_gpu_bytes) - int(headroom_bytes)
    bytes_per_token = int(n_layers_captured) * int(per_layer_token_bytes)
    if free_margin <= 0 or bytes_per_token <= 0 or int(safety) <= 0:
        return None
    cap = int(free_margin // (bytes_per_token * int(safety)))
    return cap if cap >= 1 else None


def safe_cap_with_model_sized_aperture(
    total_gpu_bytes: int,
    gpu_mem_util: float,
    default_aperture_bytes: int,
    n_layers_captured: int,
    per_layer_token_bytes: int,
    safety: int = DEFAULT_AUTOCAP_SAFETY,
    headroom_bytes: int = DEFAULT_AUTOCAP_HEADROOM_BYTES,
) -> Optional[int]:
    """The safe token cap when the install will size a DEFAULT aperture to one cap-sized step.

    ``resolve_aperture_bytes_auto(rows_needed=cap, row_bytes=b)`` installs
    ``max(default, cap * b)`` with ``b = n_layers * per_layer`` -- the same ``b`` the per-step
    transient costs per token. So the cap ``S`` must satisfy
    ``S * b * safety + max(default, S * b) <= margin`` where
    ``margin = (1 - util) * total - headroom``. On the ``S * b > default`` branch that is
    ``S <= margin / (b * (safety + 1))``; if that lands at or below ``default / b`` the aperture
    stays at ``default`` and the boundary ``floor(default / b)`` is the largest consistent cap
    (it is below the fixed-default cap, so it is safe). None when there is no usable margin."""
    b = int(n_layers_captured) * int(per_layer_token_bytes)
    margin = round((1.0 - gpu_mem_util) * int(total_gpu_bytes)) - int(headroom_bytes)
    if b <= 0 or margin <= 0 or int(safety) <= 0:
        return None
    grown = int(margin // (b * (int(safety) + 1)))
    if grown * b > int(default_aperture_bytes):
        return grown if grown >= 1 else None
    boundary = int(int(default_aperture_bytes) // b)
    return boundary if boundary >= 1 else None


def apply_min_only(current: Optional[int], safe: Optional[int]) -> Optional[int]:
    """The min-only decision: only ever LOWER ``max_num_batched_tokens``.

    ``current`` is vLLM's resolved value (or None if unresolved); ``safe`` is the derived cap (or
    None). Returns the new (lower) int to set, or None meaning "change nothing" — byte-identical.
    When ``safe >= current`` we leave vLLM's value untouched, which is the whole safety argument:
    the plugin bites only in the regime that would OOM."""
    if safe is None:
        return None
    if current is None:
        return int(safe)
    return int(safe) if int(safe) < int(current) else None


def parse_autocap_setting(raw: Optional[str]) -> Tuple[str, Optional[int]]:
    """Parse the tri-state MIA_APERTURE_MAX_BATCHED_TOKENS knob.

    Returns one of:
      ``("off", None)``      — disabled (unset is OFF: opt-in first, matching APERTURE_PER_REQUEST)
      ``("auto", None)``     — derive the cap automatically
      ``("explicit", int)``  — use this int exactly (still applied min-only)
    Truthy spellings (on/true/yes/1/auto) mean 'derive'; an unparseable value is treated as OFF so a
    typo never silently changes the per-step budget."""
    if raw is None:
        return ("off", None)
    s = raw.strip().lower()
    if s in _FALSE:
        return ("off", None)
    if s in _TRUE:
        return ("auto", None)
    try:
        return ("explicit", int(s))
    except ValueError:
        return ("off", None)


# ---------------------------------------------------------------------------
# Set-aware aperture sizing (Task E3).
#
# Until now a process ran EXACTLY ONE subsystem, so the fixed aperture budget was
# handed to it whole and nobody had to name which subsystem owned it. Plan 2 lets one
# engine serve a mixed HS+QK batch; this layer removes the single-kind assumption
# WITHOUT retuning anything -- with one capture subsystem the budget is returned
# verbatim (pinned bit-for-bit by tests/test_aperture_sizing.py), and only a mixed
# install reaches the split arithmetic.
#
# The subsystem vocabulary is the registry's -- "hs" | "qk" | "steer"
# (mia/graph/registry.py::set_registry) -- not the worker-kind vocabulary
# ("hidden_states" | "qk" | "steer"); _plugin.py maps between them.
#
# Steering is NOT in the capture set: it mutates the residual stream in place and
# captures nothing, so it has no aperture and never consumes budget.
# ---------------------------------------------------------------------------

#: Subsystems that own a slice of the GPU capture aperture, in the registry's vocabulary.
CAPTURE_SUBSYSTEMS = ("hs", "qk")
#: Every subsystem MIA knows, capture or not.
KNOWN_SUBSYSTEMS = ("hs", "qk", "steer")


def per_token_row_bytes(subsystem: str, model_dims) -> int:
    """Bytes ONE captured token costs ONE layer, for ``subsystem``.

    This is the existing per-kind row-shape math (``per_layer_token_bytes_hs`` /
    ``per_layer_token_bytes_qk``) reached through a subsystem name, so the set-aware
    split cannot drift from the sizing the single-kind path has always used.

    ``model_dims`` is a mapping. Recognized keys (first spelling found wins):
      * ``hidden`` / ``hidden_size``      — residual width (HS)
      * ``dtype_bytes`` / ``dtype_size``  — bytes per element (default 2, bf16/fp16)
      * QK, either form:
          - ``q`` / ``q_dim`` and ``k`` / ``k_dim``   — already head-multiplied widths
          - ``n_q_heads`` + ``n_kv_heads`` + ``head_dim``
    ``layers`` is accepted and ignored: every subsystem here captures the same layer
    count, so it cancels out of the ratio the split is computed from.

    Raises ``KeyError`` for a subsystem with no aperture (``steer``) or an unknown one,
    and ``ValueError`` when the dims needed for this subsystem are absent -- never a
    silent zero, which would poison the split ratio.
    """
    dims = dict(model_dims or {})
    dtype_size = int(_first(dims, ("dtype_bytes", "dtype_size"), default=2))
    if subsystem == "hs":
        hidden = _first(dims, ("hidden", "hidden_size"))
        if hidden is None:
            raise ValueError("model_dims needs 'hidden' (residual width) to size the HS aperture")
        return per_layer_token_bytes_hs(int(hidden), dtype_size)
    if subsystem == "qk":
        q_dim = _first(dims, ("q", "q_dim"))
        k_dim = _first(dims, ("k", "k_dim"))
        if q_dim is None or k_dim is None:
            h_q = _first(dims, ("n_q_heads", "num_attention_heads"))
            h_kv = _first(dims, ("n_kv_heads", "num_key_value_heads"), default=h_q)
            head_dim = _first(dims, ("head_dim",))
            if h_q is None or head_dim is None:
                raise ValueError(
                    "model_dims needs 'q'/'k' widths (or n_q_heads/n_kv_heads/head_dim) "
                    "to size the QK aperture")
            return per_layer_token_bytes_qk(int(h_q), int(h_kv), int(head_dim), dtype_size)
        # q/k arrive already multiplied by head_dim, so fold head_dim=1 into the same helper:
        # (H_q + H_kv) * head_dim * dtype == (q_dim + k_dim) * 1 * dtype.
        return per_layer_token_bytes_qk(int(q_dim), int(k_dim), 1, dtype_size)
    if subsystem == "steer":
        raise KeyError("steer has no capture aperture: it mutates the residual and captures nothing")
    raise KeyError(f"unknown subsystem {subsystem!r}; known: {', '.join(KNOWN_SUBSYSTEMS)}")


def _first(dims, names, default=None):
    """First present (non-None) value among ``names`` in ``dims``, else ``default``."""
    for n in names:
        v = dims.get(n)
        if v is not None:
            return v
    return default


def resolve_aperture_bytes(subsystems, *, gpu_bytes_budget: int, model_dims=None):
    """Split ONE fixed aperture budget across the enabled *capture* subsystems.

    ``subsystems`` is any iterable of registry subsystem names ("hs" | "qk" | "steer").
    Returns ``{subsystem: bytes}`` covering only the subsystems that own an aperture --
    ``steer`` is dropped, so a steer-only install gets ``{}`` and the caller knows there
    is nothing to reserve.

    Sizing rules, in order:
      * EXACTLY ONE capture subsystem -> it gets ``gpu_bytes_budget`` VERBATIM. No
        arithmetic touches it, so this refactor is provably a generalization and not a
        retune (tests/test_aperture_sizing.py pins the number bit-for-bit).
      * SEVERAL -> split in the ratio of their per-token row bytes (HS ``hidden x dtype``,
        QK ``(q_dim + k_dim) x dtype``), so a mixed install reserves capacity in the
        proportion it will actually consume. Floor division keeps the sum <= the budget.

    Plan 2 replaces the *split* with page-granular sharing of one aperture; this function
    exists so that work has a seam to stand on, not because a static split is the end state.

    Raises ``ValueError`` on a non-positive budget, an unknown subsystem name, or a split
    that would hand some subsystem zero bytes (fail loud, never a silent zero-size aperture).
    """
    budget = int(gpu_bytes_budget)
    if budget <= 0:
        raise ValueError(f"gpu_bytes_budget must be positive, got {gpu_bytes_budget!r}")
    requested = list(dict.fromkeys(subsystems))
    unknown = [s for s in requested if s not in KNOWN_SUBSYSTEMS]
    if unknown:
        raise ValueError(
            f"unknown subsystem(s) {unknown}; known: {', '.join(KNOWN_SUBSYSTEMS)}")
    # Sorted so the split is deterministic regardless of the caller's set iteration order
    # (a set's order varies with PYTHONHASHSEED for str members).
    capture = sorted(s for s in requested if s in CAPTURE_SUBSYSTEMS)
    if not capture:
        return {}
    if len(capture) == 1:
        return {capture[0]: budget}
    weights = {s: per_token_row_bytes(s, model_dims) for s in capture}
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError(f"per-token row bytes summed to {total_w} for {capture}; cannot split")
    sizes = {s: (budget * w) // total_w for s, w in weights.items()}
    zero = [s for s, b in sizes.items() if b <= 0]
    if zero:
        raise ValueError(
            f"aperture budget {budget} B is too small to give {zero} a non-empty slice across "
            f"{capture}: raise MIA_APERTURE_GPU_BYTES or enable fewer capture subsystems")
    return sizes
