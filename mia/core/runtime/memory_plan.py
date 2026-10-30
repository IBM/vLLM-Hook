"""Per-rank plan of MIA's GPU storage, sized to fit outside vLLM's ``gpu_memory_utilization``."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import math
import os

from mia.errors import MiaConfigurationError, MiaSizingError
from mia.core.aperture.aperture_sizing import DEFAULT_APERTURE_GPU_BYTES
from mia.core.runtime.tp_shard import hs_rows_for_mode, resolve_hs_shard_mode, qk_shard

_SIZES = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'int64': 8,
          'int32': 4, 'bool': 1, 'uint8': 1}


@dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    category: str
    offset: int = 0

    @property
    def element_size(self) -> int:
        return _SIZES[self.dtype]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.element_size


@dataclass(frozen=True)
class MiaMemoryPlan:
    model_bytes: int
    outside_bytes: int
    aperture_bytes: int
    auxiliary_bytes: int
    buffers: tuple[BufferSpec, ...]
    rows: int
    kind: str
    layers: int
    hidden: int
    cap: int
    dtype: str
    owned_layers: tuple[int, ...] = ()
    scratch_bytes: int = 0

    @property
    def storage_bytes(self) -> dict:
        """Bytes of each device storage; Q/K keeps k.* apart from q.* (one op mutates both)."""
        k = sum(b.nbytes for b in self.buffers if b.category == 'aperture_k')
        sizes = {'aperture': self.aperture_bytes - k, 'aperture_k': k,
                 'auxiliary': self.auxiliary_bytes, 'scratch': self.scratch_bytes}
        return {name: size for name, size in sizes.items() if size}

    @property
    def total_bytes(self) -> int:
        return self.aperture_bytes + self.auxiliary_bytes + self.scratch_bytes

    @property
    def remaining_bytes(self) -> int:
        return self.outside_bytes - self.total_bytes

    def buffer(self, name: str) -> BufferSpec:
        for spec in self.buffers:
            if spec.name == name:
                return spec
        raise MiaConfigurationError(f'MIA memory plan has no buffer {name!r}')

    def check_fit(self) -> None:
        if self.total_bytes > self.outside_bytes:
            raise MiaSizingError(
                f'MIA outside budget insufficient: aperture={self.aperture_bytes} B, '
                f'auxiliary={self.auxiliary_bytes} B, scratch={self.scratch_bytes} B; '
                f'outside={self.outside_bytes} B, model={self.model_bytes} B. '
                'Lower gpu_memory_utilization or MIA_APERTURE_GPU_BYTES/token/vector capacity.')


def build_memory_plan(config, worker_kind: str, *, total_bytes: int, tp_rank: int,
                      check_fit: bool = True) -> MiaMemoryPlan:
    """Resolve every owned device tensor before the model or its graph exists."""
    if worker_kind not in ('hidden_states', 'qk', 'steer'):
        raise MiaConfigurationError(f'Unknown MIA worker kind {worker_kind!r}')
    mc, pc = config.model_config, config.parallel_config
    if int(getattr(pc, 'pipeline_parallel_size', 1)) != 1:
        raise MiaConfigurationError('MIA memory planning requires pipeline_parallel_size=1')
    tc = mc.hf_text_config
    n, h = int(tc.num_hidden_layers), int(tc.hidden_size)
    cap = int(config.scheduler_config.max_num_batched_tokens)
    tp = int(pc.tensor_parallel_size)
    dtype = str(mc.dtype).removeprefix('torch.')
    dtype = {'half': 'float16', 'float': 'float32'}.get(dtype, dtype)
    util = float(config.cache_config.gpu_memory_utilization)
    if n <= 0 or h <= 0 or cap <= 0 or tp <= 0 or not 0 <= tp_rank < tp:
        raise MiaSizingError('MIA requires positive model/token dimensions and a valid TP rank')
    if dtype not in ('float16', 'bfloat16', 'float32'):
        raise MiaConfigurationError(f'Unsupported MIA buffer dtype {dtype!r}')
    if int(total_bytes) <= 0 or not math.isfinite(util) or not 0 < util <= 1:
        raise MiaSizingError('Invalid GPU total memory or gpu_memory_utilization')
    model_bytes = math.ceil(util * int(total_bytes))
    outside = int(total_bytes) - model_bytes
    specs: list[BufferSpec] = []
    offsets = {'aperture': 0, 'aperture_k': 0, 'auxiliary': 0}

    def add(name, shape, dt=dtype, category='auxiliary'):
        size = _SIZES[dt]
        offset = ((offsets[category] + size - 1) // size) * size
        b = BufferSpec(name, tuple(shape), dt, category, offset)
        specs.append(b)
        offsets[category] = offset + b.nbytes

    aperture, rows, owned = 0, 0, ()
    if worker_kind == 'hidden_states':
        owned = tuple(hs_rows_for_mode(resolve_hs_shard_mode(tp), n, tp, tp_rank))
        symmetric = tp > 1 and os.environ.get('MIA_HS_TP_SYMMETRIC', '1') != '0'
        widths = [('hs', h)]
        captured = owned
    elif worker_kind == 'qk':
        nq = int(tc.num_attention_heads)
        nk = int(getattr(tc, 'num_key_value_heads', None) or nq)
        hd = int(getattr(tc, 'head_dim', None) or h // nq)
        shard = qk_shard(tp_rank, tp, nq, nk, hd)
        widths = [('q', shard.q_width), ('k', shard.k_width)]
        captured = tuple(range(n))
        symmetric = False
    else:
        captured, widths, symmetric = (), [], False

    if captured:
        raw = os.environ.get('MIA_APERTURE_GPU_BYTES', '').strip()
        row_bytes = len(captured) * sum(w for _, w in widths) * _SIZES[dtype]
        try:
            aperture = int(raw) if raw else max(DEFAULT_APERTURE_GPU_BYTES, (cap + 1) * row_bytes)
        except ValueError as e:
            raise MiaSizingError('MIA_APERTURE_GPU_BYTES must be a positive byte count') from e
        if aperture <= 0:
            raise MiaSizingError('MIA_APERTURE_GPU_BYTES must be positive')
        rows = aperture // row_bytes - 1
        if rows < 1:
            raise MiaSizingError('Aperture must hold at least one capture row plus its sentinel')
        for layer in captured:
            for label, width in widths:
                add(f'{label}.{layer}', (rows + 1, width),
                    category='aperture_k' if label == 'k' else 'aperture')
        prefix = 'hs' if worker_kind == 'hidden_states' else 'qk'
        add(f'{prefix}.capture_index', (n, cap), 'int64')
        add(f'{prefix}.any_active', (n,), 'int32')
        if os.environ.get('MIA_CAPTURE_GPU_ROUTING', '0') == '1':
            add(f'{prefix}.slot_layer_mask', (cap, n), 'bool')
    if worker_kind == 'hidden_states' and symmetric and len(owned) < n:
        add('hs.sink_index', (n, cap), 'int64')
        add('hs.sink_active', (n,), 'int32')
        for layer in range(n):
            if layer not in owned:
                add(f'hs.sink.{layer}', (1, h))
    if worker_kind == 'steer':
        try:
            vmax = int(os.environ.get('MIA_STEER_VMAX', '16'))
        except ValueError as e:
            raise MiaSizingError('MIA_STEER_VMAX must be a positive integer') from e
        if vmax <= 0:
            raise MiaSizingError('MIA_STEER_VMAX must be positive')
        for name, dt in [('coeff', 'float32'), ('vec_id', 'int64'), ('mode', 'int64')]:
            add(f'steer.{name}', (n, cap), dt)
        add('steer.vec_table', (vmax, h))
        add('steer.avg_proj', (vmax,), 'float32')
        if os.environ.get('MIA_STEER_GPU_ROUTING', '1') != '0':
            for name, dt in [('vid', 'int64'), ('mode', 'int64'), ('coeff', 'float32'),
                             ('col', 'int64')]:
                add(f'steer.slot_{name}', (cap,), dt)
            add('steer.slot_layer_mask', (cap, n), 'bool')
    plan = MiaMemoryPlan(model_bytes, outside, aperture, offsets['auxiliary'],
                         tuple(specs), rows, worker_kind, n, h, cap, dtype, owned)
    if offsets['aperture'] + offsets['aperture_k'] > aperture:
        raise MiaSizingError('Capture layout exceeds its exact aperture storage')
    if check_fit:
        plan.check_fit()
    return plan


def safe_token_cap(config, worker_kind: str, *, total_bytes: int):
    """Largest ``max_num_batched_tokens`` whose plan fits on every TP rank; ``config`` is unchanged."""
    if worker_kind == 'steer':
        return None
    candidate = copy(config)
    candidate.scheduler_config = copy(config.scheduler_config)
    low, high, best = 1, int(config.scheduler_config.max_num_batched_tokens), None
    while low <= high:
        mid = (low + high) // 2
        candidate.scheduler_config.max_num_batched_tokens = mid
        try:
            plans = [build_memory_plan(candidate, worker_kind, total_bytes=total_bytes, tp_rank=r)
                     for r in range(int(config.parallel_config.tensor_parallel_size))]
            fits = all(p.rows == 0 or p.rows >= mid for p in plans)
        except MiaSizingError:
            fits = False
        if fits:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best
