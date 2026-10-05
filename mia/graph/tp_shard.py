"""Tensor-parallel capture geometry: per-rank heads and HS layers, rank dirs, and shard merging."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from vllm.distributed import parallel_state as ps

from mia.errors import MiaConfigurationError
from mia.graph.delivery_selector import DP_SIZE_ENV, _dp_size

RANK_DIR_PREFIX = "tp_rank_"
_RANK_DIR_RE = re.compile(r"^tp_rank_(\d+)$")

QK_SHARD_FIELDS = (
    "tp_rank",
    "tp_size",
    "q_head_start",
    "num_local_q_heads",
    "kv_head_start",
    "num_local_kv_heads",
    "num_kv_head_replicas",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)

TP_SHARD_KEY = "tp_shard"

HS_SHARD_ENV = "MIA_HS_TP_SHARD"
HS_ALL_RANKS_ENV = "MIA_HS_CAPTURE_ALL_RANKS"
HS_LAYER_SHARD_RULE = "round_robin"
HS_MODE_SINGLE = "single"
HS_MODE_ROUND_ROBIN = HS_LAYER_SHARD_RULE
HS_MODE_RANK0 = "rank0"
HS_MODE_ALL_RANKS = "all_ranks"
HS_SHARD_KEY = "hs_shard"
HS_SHARD_FIELDS = ("tp_rank", "tp_size", "num_layers", "owned_layers")


class TPShardError(ValueError):
    """A set of per-rank captures that cannot be merged into the global head layout."""


def rank_dir_name(tp_rank: int) -> str:
    """``tp_rank_{tp_rank}`` -- the one per-rank directory name."""
    return f"{RANK_DIR_PREFIX}{int(tp_rank)}"


def parse_rank_dir(name: str) -> Optional[int]:
    """The rank encoded in a ``tp_rank_<N>`` directory name, else None."""
    m = _RANK_DIR_RE.match(os.path.basename(os.path.normpath(str(name))))
    return int(m.group(1)) if m else None


DP_DIR_PREFIX = "dp_rank_"
_DP_DIR_RE = re.compile(r"^dp_rank_(\d+)$")


def dp_dir_name(dp_rank: int) -> str:
    return f"{DP_DIR_PREFIX}{int(dp_rank)}"


def parse_dp_dir(name: str) -> Optional[int]:
    """The DP rank encoded in a ``dp_rank_<N>`` directory name, else None."""
    m = _DP_DIR_RE.match(os.path.basename(os.path.normpath(str(name))))
    return int(m.group(1)) if m else None


def dp_layout(worker) -> dict:
    """``{"dp_rank": i, "dp_size": n}`` when ``MIA_DP_SIZE`` > 1, else ``{}``."""
    n = _dp_size(os.environ)
    if n <= 1:
        return {}
    pc = getattr(worker, "parallel_config", None)
    idx = getattr(pc, "data_parallel_index", None)
    if idx is None or not 0 <= int(idx) < n:
        raise MiaConfigurationError(
            f"{DP_SIZE_ENV}={n} but this worker's parallel_config.data_parallel_index is {idx!r}: "
            f"without it every DP engine would write the same run dir. Refusing.")
    return {"dp_rank": int(idx), "dp_size": n}


def dp_run_base(base: str, dp: dict) -> str:
    """``base`` itself, or ``base/dp_rank_<i>`` for a :func:`dp_layout` that names a DP engine."""
    return os.path.join(base, dp_dir_name(dp["dp_rank"])) if dp else base


def refuse_pipeline_parallel(pp_size, where: str = "") -> None:
    """Raise if pipeline parallelism is on."""
    try:
        pp = int(pp_size or 1)
    except (TypeError, ValueError):
        pp = 1
    if pp > 1:
        suffix = f" ({where})" if where else ""
        raise MiaConfigurationError(
            f"MIA does not support pipeline parallelism: pipeline_parallel_size={pp}{suffix}. "
            f"Under PP each rank holds only its own stage's decoder layers; the rest are "
            f"PPMissingLayer identities that MIA's layer matcher still hooks, so HS/QK capture "
            f"would record zero-filled layers and steering would reach only one stage -- a run "
            f"that reports success with wrong data. Use tensor parallelism "
            f"(tensor_parallel_size) with pipeline_parallel_size=1.")


def resolve_tp_coords(worker) -> Tuple[int, int]:
    """``(tp_rank, tp_size)`` for a vLLM worker."""
    pc = getattr(worker, "parallel_config", None)
    tp_size = int(getattr(pc, "tensor_parallel_size", 1) or 1)
    try:
        if ps.model_parallel_is_initialized():
            ws = int(ps.get_tensor_model_parallel_world_size())
            if ws == tp_size:
                return int(ps.get_tensor_model_parallel_rank()), tp_size
    except Exception:  # noqa: BLE001
        pass
    return int(getattr(worker, "rank", 0) or 0) % max(1, tp_size), tp_size


@dataclass(frozen=True)
class QKShard:
    """One TP rank's slice of a layer's post-RoPE Q and K."""
    tp_rank: int
    tp_size: int
    q_head_start: int
    num_local_q_heads: int
    kv_head_start: int
    num_local_kv_heads: int
    num_kv_head_replicas: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def q_width(self) -> int:
        return self.num_local_q_heads * self.head_dim

    @property
    def k_width(self) -> int:
        return self.num_local_kv_heads * self.head_dim

    @property
    def global_q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def global_k_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def as_header(self) -> dict:
        """The ``QK_SHARD_FIELDS`` dict (plain ints, JSON-native)."""
        return {k: int(getattr(self, k)) for k in QK_SHARD_FIELDS}


def qk_shard(tp_rank: int, tp_size: int, num_attention_heads: int,
             num_key_value_heads: int, head_dim: int) -> QKShard:
    """The QK shard vLLM gives ``tp_rank`` of ``tp_size`` (mirrors ``QKVParallelLinear``)."""
    tp_rank, tp_size = int(tp_rank), int(tp_size)
    h_q, h_kv, d = int(num_attention_heads), int(num_key_value_heads), int(head_dim)
    if tp_size < 1 or not (0 <= tp_rank < tp_size):
        raise MiaConfigurationError(f"invalid TP coordinates tp_rank={tp_rank} tp_size={tp_size}")
    if h_q < 1 or h_kv < 1 or d < 1:
        raise MiaConfigurationError(
            f"invalid attention geometry num_attention_heads={h_q} num_key_value_heads={h_kv} "
            f"head_dim={d}")
    if h_q % tp_size:
        raise MiaConfigurationError(
            f"num_attention_heads={h_q} is not divisible by tensor_parallel_size={tp_size}")
    if tp_size >= h_kv:
        if tp_size % h_kv:
            raise MiaConfigurationError(
                f"tensor_parallel_size={tp_size} >= num_key_value_heads={h_kv} but is not a "
                f"multiple of it; vLLM cannot replicate KV heads evenly")
        local_kv, replicas = 1, tp_size // h_kv
    else:
        if h_kv % tp_size:
            raise MiaConfigurationError(
                f"num_key_value_heads={h_kv} is not divisible by tensor_parallel_size={tp_size}")
        local_kv, replicas = h_kv // tp_size, 1
    local_q = h_q // tp_size
    return QKShard(
        tp_rank=tp_rank, tp_size=tp_size,
        q_head_start=tp_rank * local_q, num_local_q_heads=local_q,
        kv_head_start=(tp_rank // replicas) * local_kv, num_local_kv_heads=local_kv,
        num_kv_head_replicas=replicas,
        num_attention_heads=h_q, num_key_value_heads=h_kv, head_dim=d)


def qk_shard_from_header(header) -> Optional[QKShard]:
    """Parse a QK shard from a sidecar header or payload; None for a full-width TP=1 capture."""
    if not isinstance(header, dict):
        return None
    present = [k for k in QK_SHARD_FIELDS if k in header]
    if not present:
        return None
    missing = [k for k in QK_SHARD_FIELDS if k not in header]
    if missing:
        raise TPShardError(f"QK shard header is incomplete: missing {missing} (has {present})")
    try:
        recorded = QKShard(**{k: int(header[k]) for k in QK_SHARD_FIELDS})
        expected = qk_shard(recorded.tp_rank, recorded.tp_size, recorded.num_attention_heads,
                            recorded.num_key_value_heads, recorded.head_dim)
    except MiaConfigurationError as e:
        raise TPShardError(f"QK shard header describes an impossible geometry: {e}") from e
    if recorded != expected:
        raise TPShardError(
            f"QK shard header disagrees with vLLM's sharding for its own geometry: recorded "
            f"{recorded.as_header()} but tp_rank={recorded.tp_rank}/{recorded.tp_size} of "
            f"{recorded.num_attention_heads}/{recorded.num_key_value_heads} heads is "
            f"{expected.as_header()}")
    return recorded


def qk_conf_head_dim(text_cfg) -> int:
    """Per-head width of Q/K: ``config.head_dim`` if declared, else hidden_size // num_heads."""
    num_h = int(getattr(text_cfg, "num_attention_heads"))
    hidden = int(getattr(text_cfg, "hidden_size"))
    return int(getattr(text_cfg, "head_dim", None) or hidden // num_h)


def check_attn_modules_match_shard(matched, shard) -> None:
    """Refuse if a matched attention module's local head geometry contradicts ``shard``."""
    for name, module, _ in matched:
        got = {k: getattr(module, k, None) for k in ("num_heads", "num_kv_heads", "head_size")}
        want = {"num_heads": shard.num_local_q_heads, "num_kv_heads": shard.num_local_kv_heads,
                "head_size": shard.head_dim}
        bad = {k: (got[k], want[k]) for k in got
               if got[k] is not None and int(got[k]) != int(want[k])}
        if bad:
            raise MiaConfigurationError(
                f"QK TP shard geometry mismatch on {name}: module (got, expected) = {bad} for "
                f"tp_rank {shard.tp_rank}/{shard.tp_size}. MIA derives each rank's heads the way "
                f"vLLM's QKVParallelLinear shards them; this model shards differently, so its "
                f"per-rank Q/K cannot be labelled correctly. Refusing rather than writing shards "
                f"whose headers lie.")


def check_complete_shard_set(shards: Sequence[QKShard]) -> List[int]:
    """Validate one shard per rank ``0..tp_size-1``, all of one geometry."""
    if not shards:
        raise TPShardError("no QK shards to merge")
    first = shards[0]
    geom = (first.tp_size, first.num_attention_heads, first.num_key_value_heads, first.head_dim)
    for s in shards[1:]:
        g = (s.tp_size, s.num_attention_heads, s.num_key_value_heads, s.head_dim)
        if g != geom:
            raise TPShardError(
                f"QK shards disagree on the global geometry (tp_size, H_q, H_kv, head_dim): "
                f"{geom} vs {g}")
    ranks = [s.tp_rank for s in shards]
    dupes = sorted({r for r in ranks if ranks.count(r) > 1})
    if dupes:
        raise TPShardError(f"QK shards duplicate tp_rank(s) {dupes}")
    missing = sorted(set(range(first.tp_size)) - set(ranks))
    if missing:
        raise TPShardError(
            f"QK capture is missing tp_rank(s) {missing} of tp_size={first.tp_size}: only "
            f"{sorted(ranks)} captured. Every TP rank holds a DIFFERENT slice of the heads, so a "
            f"partial set cannot be merged into the full layer (a rank-0-only capture records "
            f"{first.num_local_q_heads} of {first.num_attention_heads} query heads).")
    return sorted(range(len(shards)), key=lambda i: shards[i].tp_rank)


def _width(t) -> int:
    return int(t.shape[-1])


def merge_head_tensors(kind: str, items: Sequence[Tuple[QKShard, "object"]],
                       check_replicas: bool = False):
    """Concatenate per-rank Q or K tensors in global head order, de-duplicating replicated KV heads."""
    if kind not in ("q", "k"):
        raise ValueError(f"kind must be 'q' or 'k', got {kind!r}")
    shards = [s for s, _ in items]
    order = check_complete_shard_set(shards)
    lead = None
    for s, t in items:
        want = s.q_width if kind == "q" else s.k_width
        if _width(t) != want:
            raise TPShardError(
                f"tp_rank {s.tp_rank}: {kind} width {_width(t)} != its shard's "
                f"{want} ({s.num_local_q_heads if kind == 'q' else s.num_local_kv_heads} heads "
                f"x head_dim {s.head_dim})")
        if lead is None:
            lead = tuple(t.shape[:-1])
        elif tuple(t.shape[:-1]) != lead:
            raise TPShardError(
                f"tp_rank {s.tp_rank}: {kind} leading shape {tuple(t.shape[:-1])} != rank "
                f"{shards[order[0]].tp_rank}'s {lead} -- the ranks captured different tokens")
    if kind == "q":
        parts = [items[i][1] for i in order]
        expect_w = shards[0].global_q_width
    else:
        by_start: dict = {}
        for i in order:
            s, t = items[i]
            if s.kv_head_start in by_start:
                if check_replicas and not torch.equal(by_start[s.kv_head_start], t):
                    raise TPShardError(
                        f"replicated KV head(s) starting at {s.kv_head_start} differ between "
                        f"ranks (tp_rank {s.tp_rank} vs the lowest replica)")
                continue
            by_start[s.kv_head_start] = t
        starts = sorted(by_start)
        step = shards[0].num_local_kv_heads
        if starts != list(range(0, shards[0].num_key_value_heads, step)):
            raise TPShardError(
                f"KV shards do not tile the {shards[0].num_key_value_heads} KV heads: starts "
                f"{starts} with {step} head(s) each")
        parts = [by_start[k] for k in starts]
        expect_w = shards[0].global_k_width
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    if _width(out) != expect_w:
        raise TPShardError(f"merged {kind} width {_width(out)} != global {expect_w}")
    return out


def _merge_field(kind: str, values: Sequence[Tuple[QKShard, "object"]], check_replicas: bool):
    first = values[0][1]
    if torch.is_tensor(first):
        return merge_head_tensors(kind, values, check_replicas)
    if isinstance(first, (list, tuple)):
        n = len(first)
        for s, v in values:
            if not isinstance(v, (list, tuple)) or len(v) != n:
                raise TPShardError(
                    f"tp_rank {s.tp_rank}: {kind} list has {len(v) if isinstance(v, (list, tuple)) else '?'} "
                    f"elements, rank-{values[0][0].tp_rank} has {n}")
        return [merge_head_tensors(kind, [(s, v[j]) for s, v in values], check_replicas)
                for j in range(n)]
    raise TPShardError(f"cannot merge a {type(first).__name__} {kind} field across ranks")


_Q_FIELDS = ("q",)
_K_FIELDS = ("k_all", "k_full")
_SAME_FIELDS = ("layer_num", "hookq_mode", "k_prefix_ends")


def merge_qk_entries(items: Sequence[Tuple[QKShard, dict]], check_replicas: bool = False) -> dict:
    """Merge one (request, layer) QK entry from every rank into the global layout."""
    order = check_complete_shard_set([s for s, _ in items])
    ranked = [items[i] for i in order]
    base = ranked[0][1]
    for s, e in ranked:
        if not isinstance(e, dict):
            raise TPShardError(f"tp_rank {s.tp_rank}: entry is a {type(e).__name__}, not a dict")
        if "scores" in e or e.get("capture") == "score":
            raise TPShardError(
                "attention-SCORE entries cannot be merged across TP ranks (each rank scores only "
                "its own heads); score capture is unsupported at tensor_parallel_size > 1")
        if any(k.endswith("_qmeta") and e.get(k) is not None for k in e):
            raise TPShardError(
                "quantized QK entries must be dequantized before a TP merge (packed rows cannot "
                "be concatenated along the head dimension)")
    for key in _SAME_FIELDS:
        vals = [e.get(key) for _, e in ranked]
        if any(v != vals[0] for v in vals[1:]):
            raise TPShardError(f"QK entries disagree on {key!r} across ranks: {vals}")
    out = {k: v for k, v in base.items()}
    for key in _Q_FIELDS:
        if key in base:
            out[key] = _merge_field("q", [(s, e[key]) for s, e in ranked], check_replicas)
    for key in _K_FIELDS:
        if key in base:
            out[key] = _merge_field("k", [(s, e[key]) for s, e in ranked], check_replicas)
    return out


def merge_qk_payloads(payloads: Sequence[dict], check_replicas: bool = False) -> dict:
    """Merge per-rank QK payloads into one payload with full-width entries."""
    payloads = [p for p in payloads if p is not None]
    if not payloads:
        raise TPShardError("no QK payloads to merge")
    shards = [qk_shard_from_header(p.get(TP_SHARD_KEY)) for p in payloads]
    if len(payloads) == 1 and shards[0] is None:
        return payloads[0]
    if any(s is None for s in shards):
        raise TPShardError(
            f"{len(payloads)} per-rank QK payloads but {sum(s is None for s in shards)} carry no "
            f"'{TP_SHARD_KEY}' geometry; refusing to guess the head order")
    if len(payloads) == 1 and shards[0].tp_size == 1:
        out = dict(payloads[0])
        out.pop(TP_SHARD_KEY, None)
        return out
    order = check_complete_shard_set(shards)
    ranked = [(shards[i], payloads[i]) for i in order]
    names: list = []
    for _, p in ranked:
        for n in (p.get("qk_cache") or {}):
            if n not in names:
                names.append(n)
    merged_cache = {}
    for name in names:
        entries = []
        for s, p in ranked:
            e = (p.get("qk_cache") or {}).get(name)
            if e is None:
                raise TPShardError(
                    f"layer entry {name!r} is present on some ranks but missing on tp_rank "
                    f"{s.tp_rank}")
            entries.append((s, e))
        merged_cache[name] = merge_qk_entries(entries, check_replicas)
    out = {k: v for k, v in ranked[0][1].items() if k != TP_SHARD_KEY}
    out["qk_cache"] = merged_cache
    return out


def _hs_layout_flag(env, name: str, meaning: str) -> str:
    raw = env.get(name)
    val = "" if raw is None else str(raw).strip()
    if val not in ("", "0", "1"):
        raise MiaConfigurationError(
            f"{name}={raw!r} is not a valid value: use 1 ({meaning}) or 0, or leave it unset. "
            f"Only the exact strings '0' and '1' are read -- 'true'/'yes'/'on' are REFUSED rather "
            f"than silently ignored, because this flag picks the HS capture LAYOUT.")
    return val


def resolve_hs_shard_mode(tp_size: int, environ=None) -> str:
    """Which HS capture layout this engine runs (one of the ``HS_MODE_*`` constants)."""
    env = os.environ if environ is None else environ
    val = _hs_layout_flag(
        env, HS_SHARD_ENV,
        "shard the HS layers round-robin across the TP ranks, the default at "
        "tensor_parallel_size > 1; 0 = tp_rank 0 captures every layer, the pre-shard layout, for "
        "A/B only")
    all_ranks = _hs_layout_flag(
        env, HS_ALL_RANKS_ENV,
        f"the replication diagnostic: every rank captures EVERY layer into its own dir; wins "
        f"over {HS_SHARD_ENV}")
    if int(tp_size or 1) <= 1:
        return HS_MODE_SINGLE
    if all_ranks == "1":
        return HS_MODE_ALL_RANKS
    return HS_MODE_RANK0 if val == "0" else HS_MODE_ROUND_ROBIN


def hs_layer_owner(layer0: int, tp_size: int) -> int:
    """The TP rank that captures 0-based decoder layer ``layer0`` under the round-robin shard."""
    return int(layer0) % max(1, int(tp_size))


def hs_owned_rows(num_layers: int, tp_size: int, tp_rank: int) -> List[int]:
    """0-based decoder-layer indices (registry rows) ``tp_rank`` captures, ascending."""
    num_layers, tp_size, tp_rank = int(num_layers), max(1, int(tp_size)), int(tp_rank)
    if not (0 <= tp_rank < tp_size):
        raise MiaConfigurationError(f"invalid TP coordinates tp_rank={tp_rank} tp_size={tp_size}")
    return list(range(tp_rank, num_layers, tp_size))


def hs_owned_layers(num_layers: int, tp_size: int, tp_rank: int) -> List[int]:
    """The same set as 1-based artifact layer numbers."""
    return [i + 1 for i in hs_owned_rows(num_layers, tp_size, tp_rank)]


def hs_rows_for_mode(mode: str, num_layers: int, tp_size: int, tp_rank: int) -> List[int]:
    """0-based rows a rank captures in HS layout ``mode``."""
    if mode == HS_MODE_ROUND_ROBIN:
        return hs_owned_rows(num_layers, tp_size, tp_rank)
    if mode == HS_MODE_RANK0:
        return list(range(int(num_layers))) if int(tp_rank) == 0 else []
    if mode in (HS_MODE_SINGLE, HS_MODE_ALL_RANKS):
        return list(range(int(num_layers)))
    raise ValueError(f"unknown HS shard mode {mode!r}")


def hs_max_owned_layers(num_layers: int, tp_size: int, mode: str) -> int:
    """Most layers any one rank captures under ``mode`` (the per-rank sizing bound)."""
    num_layers, tp_size = int(num_layers), max(1, int(tp_size))
    if mode == HS_MODE_ROUND_ROBIN:
        return -(-num_layers // tp_size)
    return num_layers


def hs_requested_layers(spec, num_layers: int) -> List[int]:
    """The 1-based layers an ``output_hidden_states`` value asks for."""
    num_layers = int(num_layers)
    if isinstance(spec, list):
        return sorted({int(x) for x in spec if 1 <= int(x) <= num_layers})
    return list(range(1, num_layers + 1))


def hs_expected_ranks(layers, tp_size: int) -> List[int]:
    """Ranks that capture any of ``layers`` (1-based) under the round-robin shard, ascending."""
    return sorted({hs_layer_owner(int(L) - 1, tp_size) for L in layers})


@dataclass(frozen=True)
class HSShard:
    """One TP rank's share of the HS layers under the round-robin shard (1-based)."""
    tp_rank: int
    tp_size: int
    num_layers: int
    owned_layers: Tuple[int, ...]

    @classmethod
    def of(cls, tp_rank: int, tp_size: int, num_layers: int) -> "HSShard":
        return cls(int(tp_rank), int(tp_size), int(num_layers),
                   tuple(hs_owned_layers(num_layers, tp_size, tp_rank)))

    def as_header(self) -> dict:
        """The sidecar-header / payload fields: ``HS_SHARD_FIELDS`` plus ``layer_shard``."""
        return {"tp_rank": self.tp_rank, "tp_size": self.tp_size, "num_layers": self.num_layers,
                "layer_shard": HS_LAYER_SHARD_RULE, "owned_layers": list(self.owned_layers)}


def hs_shard_from_header(header) -> Optional[HSShard]:
    """Parse a round-robin HS shard from a sidecar header or payload; None if unsharded."""
    if not isinstance(header, dict):
        return None
    if "owned_layers" not in header and "layer_shard" not in header:
        return None
    rule = header.get("layer_shard", HS_LAYER_SHARD_RULE)
    if rule != HS_LAYER_SHARD_RULE:
        raise TPShardError(f"HS layer shard rule {rule!r} is not {HS_LAYER_SHARD_RULE!r}")
    missing = [k for k in HS_SHARD_FIELDS if k not in header]
    if missing:
        raise TPShardError(f"HS shard header is incomplete: missing {missing}")
    try:
        tp_rank, tp_size = int(header["tp_rank"]), int(header["tp_size"])
        num_layers = int(header["num_layers"])
        owned = tuple(int(x) for x in header["owned_layers"])
    except (TypeError, ValueError) as e:
        raise TPShardError(f"HS shard header has a non-integer field: {e}") from e
    if tp_size < 1 or not (0 <= tp_rank < tp_size) or num_layers < 1:
        raise TPShardError(f"HS shard header describes an impossible geometry: tp_rank={tp_rank} "
                           f"tp_size={tp_size} num_layers={num_layers}")
    want = tuple(hs_owned_layers(num_layers, tp_size, tp_rank))
    if owned != want:
        raise TPShardError(
            f"HS shard header disagrees with the round-robin rule: tp_rank {tp_rank}/{tp_size} of "
            f"{num_layers} layers owns {list(want)[:8]}{'...' if len(want) > 8 else ''}, but the "
            f"header says {list(owned)[:8]}{'...' if len(owned) > 8 else ''}")
    return HSShard(tp_rank, tp_size, num_layers, owned)


def check_hs_shard_set(shards: Sequence[HSShard], expected_ranks=None) -> List[int]:
    """Validate the HS layer shards of one capture; return their indices sorted by rank."""
    if not shards:
        raise TPShardError("no HS layer shards to merge")
    first = shards[0]
    geom = (first.tp_size, first.num_layers)
    for s in shards[1:]:
        if (s.tp_size, s.num_layers) != geom:
            raise TPShardError(
                f"HS shards disagree on the geometry (tp_size, num_layers): {geom} vs "
                f"{(s.tp_size, s.num_layers)}")
    ranks = [s.tp_rank for s in shards]
    dupes = sorted({r for r in ranks if ranks.count(r) > 1})
    if dupes:
        raise TPShardError(f"HS shards duplicate tp_rank(s) {dupes}: two dirs claim the same "
                           f"layers")
    seen: dict = {}
    for s in shards:
        for L in s.owned_layers:
            if L in seen:
                raise TPShardError(f"HS layer {L} is claimed by tp_rank {seen[L]} and tp_rank "
                                   f"{s.tp_rank}")
            seen[L] = s.tp_rank
    tp, nl = geom
    if expected_ranks is None:
        expected = [r for r in range(tp) if hs_owned_rows(nl, tp, r)]
    else:
        expected = sorted({int(r) for r in expected_ranks})
    missing = sorted(set(expected) - set(ranks))
    if missing:
        lost = sorted(L for r in missing for L in hs_owned_layers(nl, tp, r))
        raise TPShardError(
            f"HS capture is missing tp_rank(s) {missing} of tp_size={tp}: only {sorted(ranks)} "
            f"are present, so layers {lost[:8]}{'...' if len(lost) > 8 else ''} ({len(lost)} of "
            f"{nl}) have no data. Every rank captures a DIFFERENT share of the layers; a partial "
            f"set is refused rather than returned without them. (A rank that captured nothing "
            f"still writes a header-only sidecar, which flush_aperture omits: pass every rank dir, "
            f"or read the run with load_hs_aperture_tp(MIA_APERTURE_DIR).)")
    return sorted(range(len(shards)), key=lambda i: shards[i].tp_rank)


def _rows_of(t) -> int:
    return int(t.shape[0]) if hasattr(t, "shape") and len(t.shape) else 0


def merge_hs_layer_maps(items: Sequence[Tuple[HSShard, dict]]) -> dict:
    """Union per-rank HS captures into one, layers ascending per request."""
    out: dict = {}
    where: dict = {}
    for shard, art in items:
        owned = set(shard.owned_layers)
        for req_id, per in art.items():
            dst = out.setdefault(req_id, {})
            for layer, t in per.items():
                L = int(layer)
                if L not in owned:
                    raise TPShardError(
                        f"tp_rank {shard.tp_rank} holds req {req_id!r} layer {L}, which it does "
                        f"not own (its layers: {list(shard.owned_layers)[:8]}...)")
                if L in dst:
                    raise TPShardError(f"req {req_id!r} layer {L} is present on tp_rank "
                                       f"{where[(req_id, L)]} and tp_rank {shard.tp_rank}")
                dst[L] = t
                where[(req_id, L)] = shard.tp_rank
    for req_id, per in out.items():
        rows = {L: _rows_of(t) for L, t in per.items()}
        if len(set(rows.values())) > 1:
            by_rank = {}
            for L, n in rows.items():
                by_rank.setdefault(where[(req_id, L)], set()).add(n)
            raise TPShardError(
                f"req {req_id!r}: its layers hold different row counts across ranks "
                f"(rank -> rows {dict(sorted((r, sorted(v)) for r, v in by_rank.items()))}); every "
                f"layer of a request captures the same tokens, so the ranks captured different "
                f"steps")
    return {req_id: dict(sorted(per.items())) for req_id, per in out.items()}


def merge_hs_payloads(payloads: Sequence[dict], requested_layers=None) -> dict:
    """Merge one request's per-rank HS payloads into one payload, layers ascending."""
    payloads = [p for p in payloads if p is not None]
    if not payloads:
        raise TPShardError("no HS payloads to merge")
    shards = []
    for p in payloads:
        raw = p.get(HS_SHARD_KEY) if isinstance(p, dict) else None
        s = hs_shard_from_header(raw) if isinstance(raw, dict) else None
        if s is None:
            raise TPShardError(f"{len(payloads)} HS payload(s), but one carries no "
                               f"'{HS_SHARD_KEY}' layer shard; refusing to guess which layers it "
                               f"holds")
        shards.append(s)
    tp, nl = shards[0].tp_size, shards[0].num_layers
    requested = None
    expected = None
    if requested_layers is not None:
        requested = hs_requested_layers(requested_layers, nl)
        expected = hs_expected_ranks(requested, tp)
    order = check_hs_shard_set(shards, expected)
    items = []
    for i in order:
        cache = payloads[i].get("hs_cache") or {}
        per = {}
        for key, entry in cache.items():
            L = int(entry.get("layer_num", key)) if isinstance(entry, dict) else int(key)
            per[L] = entry.get("hidden_states") if isinstance(entry, dict) else entry
        items.append((shards[i], {"_": per}))
    merged = merge_hs_layer_maps(items).get("_", {})
    if requested is not None and sorted(merged) != requested:
        missing = sorted(set(requested) - set(merged))
        extra = sorted(set(merged) - set(requested))
        raise TPShardError(
            f"the merged HS parts hold layers that are not the request's: missing "
            f"{missing[:8]} extra {extra[:8]} (requested {len(requested)} layer(s) over ranks "
            f"{expected})")
    out = {k: v for k, v in payloads[order[0]].items() if k != HS_SHARD_KEY}
    by_layer = {}
    for i in order:
        for key, entry in (payloads[i].get("hs_cache") or {}).items():
            L = int(entry.get("layer_num", key)) if isinstance(entry, dict) else int(key)
            by_layer[L] = (key, entry)
    out["hs_cache"] = {by_layer[L][0]: by_layer[L][1] for L in sorted(merged)}
    return out


def drain_holds_data(drain) -> bool:
    """Whether a shared-file aperture drain wrote any captured rows."""
    if drain is None:
        return False
    if getattr(drain, "per_request", False):
        return True
    has = getattr(drain, "has_sidecar_entries", None)
    if callable(has) and has():
        return True
    if getattr(drain, "_steps", None):
        return True
    paths: List[str] = []
    for attr in ("raw_paths", "q_raw_paths", "k_raw_paths"):
        paths.extend((getattr(drain, attr, None) or {}).values())
    for p in paths:
        try:
            if os.path.getsize(p) > 0:
                return True
        except OSError:
            continue
    return False


def discover_rank_dirs(base_dir: str, sidecar_name: str) -> List[Tuple[int, str]]:
    """``[(tp_rank, dir), ...]`` for each ``tp_rank_<N>`` subdir holding ``sidecar_name``."""
    out: List[Tuple[int, str]] = []
    try:
        names = os.listdir(base_dir)
    except OSError:
        names = []
    for n in names:
        r = parse_rank_dir(n)
        d = os.path.join(base_dir, n)
        if r is not None and os.path.isdir(d) and os.path.exists(os.path.join(d, sidecar_name)):
            out.append((r, d))
    if not out and os.path.exists(os.path.join(base_dir, sidecar_name)):
        r = parse_rank_dir(base_dir)
        out.append((0 if r is None else r, base_dir))
    return sorted(out)


__all__ = [
    "QK_SHARD_FIELDS", "TP_SHARD_KEY", "RANK_DIR_PREFIX", "TPShardError", "QKShard",
    "HS_SHARD_ENV", "HS_ALL_RANKS_ENV", "HS_LAYER_SHARD_RULE", "HS_SHARD_KEY", "HS_SHARD_FIELDS",
    "HS_MODE_SINGLE", "HS_MODE_ROUND_ROBIN", "HS_MODE_RANK0", "HS_MODE_ALL_RANKS", "HSShard",
    "resolve_hs_shard_mode", "hs_layer_owner", "hs_owned_rows", "hs_owned_layers",
    "hs_rows_for_mode", "hs_max_owned_layers", "hs_requested_layers", "hs_expected_ranks",
    "hs_shard_from_header", "check_hs_shard_set", "merge_hs_layer_maps", "merge_hs_payloads",
    "rank_dir_name", "parse_rank_dir", "refuse_pipeline_parallel", "resolve_tp_coords",
    "DP_DIR_PREFIX", "dp_dir_name", "parse_dp_dir", "dp_layout", "dp_run_base",
    "qk_shard", "qk_shard_from_header", "qk_conf_head_dim", "check_attn_modules_match_shard",
    "check_complete_shard_set", "merge_head_tensors",
    "merge_qk_entries", "merge_qk_payloads", "drain_holds_data", "discover_rank_dirs",
]

