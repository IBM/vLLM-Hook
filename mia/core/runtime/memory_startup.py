"""Reserve MIA's GPU storage before vLLM 0.32 takes its startup memory snapshot."""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import os

from vllm.logger import init_logger

import torch

from mia.errors import MiaConfigurationError, MiaSizingError
from .memory_owner import MiaMemoryOwner
from .memory_plan import build_memory_plan

_INITIALIZING = ContextVar('mia_initializing_worker', default=None)
_LOG = init_logger(__name__)
_KINDS = {'mia.workers.hs_capture_worker.HSCaptureWorker': 'hidden_states',
          'mia.workers.qk_capture_worker.QKCaptureWorker': 'qk',
          'mia.workers.steer_worker.SteerWorker': 'steer'}


def _kind(worker):
    config = worker.vllm_config
    if getattr(config.model_config, 'enforce_eager', False):
        return None
    return _KINDS.get(getattr(config.parallel_config, 'worker_extension_cls', None))


def reserve_before_model_baseline(worker, snapshot) -> None:
    from .tp_shard import resolve_tp_coords
    if getattr(worker, '_mia_memory_owner', None) is not None:
        raise MiaConfigurationError('MIA storage reservation reached twice for this worker')
    rank, _ = resolve_tp_coords(worker)
    plan = build_memory_plan(worker.vllm_config, _kind(worker),
                             total_bytes=int(snapshot.total_memory), tp_rank=rank)
    if snapshot.free_memory < plan.model_bytes + plan.total_bytes:
        raise MiaSizingError(f'MIA physical free memory {snapshot.free_memory} B cannot fit '
                             f'model budget {plan.model_bytes} B plus outside storage '
                             f'{plan.total_bytes} B')
    worker._mia_memory_owner = MiaMemoryOwner.allocate(plan, snapshot.device_)
    torch.cuda.synchronize(snapshot.device_)
    torch.cuda.empty_cache()


def memory_budget_info(worker) -> dict:
    owner = getattr(worker, '_mia_memory_owner', None)
    if owner is None:
        return {'enabled': False}
    return {'enabled': True, **owner.summary(),
            'baseline_free_bytes': int(worker.init_snapshot.free_memory),
            'requested_bytes': int(worker.requested_memory),
            'kv_bytes': getattr(worker, 'available_kv_cache_memory_bytes', None),
            'profiled_consumed_bytes': getattr(worker, 'total_consumed', None),
            'graph_estimate_bytes': getattr(worker, 'cudagraph_memory_estimate', None)}


def close_mia_consumers(worker) -> None:
    for name in ("_hs_drain", "_qk_drain"):
        drain = getattr(worker, name, None)
        if drain is not None:
            drain.close()
            thread = getattr(drain, "_thread", None)
            if thread is not None and thread.is_alive():
                raise RuntimeError("MIA consumer did not stop; retaining live device storage")
    writer = getattr(worker, "_writer_process", None)
    if writer is not None:
        writer.close()


def release_mia_storage(worker) -> None:
    # Called only after consumers and model graphs have stopped.
    for name in ("_hs_drain", "_qk_drain"):
        drain = getattr(worker, name, None)
        if drain is not None:
            drain.layers = []  # atexit may retain the closed drain
    runner = getattr(worker, "model_runner", None)
    model = getattr(runner, "model", None)
    if model is not None and callable(getattr(model, "modules", None)):
        for module in list(model.modules()):
            for name in ("_mia_hs_host", "_mia_qk_host", "_mia_steer_host", "_mia_qk_registry"):
                if hasattr(module, name):
                    delattr(module, name)
    registries = getattr(worker, "_mia_registries", {})
    for registry in registries.values():
        if registry is not None:
            registry.hosts.clear()
            for name, value in list(vars(registry).items()):
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    setattr(registry, name, None)
    registries.clear()
    for name in ("_hs_sink_index", "_hs_sink_active"):
        if hasattr(worker, name):
            delattr(worker, name)
    from mia.core.hooks import install_steer
    if install_steer._ACTIVE_WORKER_STEER is worker:
        install_steer._ACTIVE_WORKER_STEER = None
    owner = getattr(worker, "_mia_memory_owner", None)
    if owner is not None:
        owner.close()
        del worker._mia_memory_owner


def abort_mia_startup(worker) -> None:
    """Release failed-startup storage only after consumers and any graphs stop."""
    if getattr(worker, "_mia_memory_owner", None) is None:
        return
    close_mia_consumers(worker)
    runner = getattr(worker, "model_runner", None)
    if getattr(runner, "cudagraph_manager", None) is not None:
        runner.shutdown()
    release_mia_storage(worker)


def install_memory_startup_patch(module=None, *, version=None) -> None:
    if module is None:
        import vllm
        from vllm.v1.worker import gpu_worker as module
        version = vllm.__version__
    if str(version).split('+', 1)[0] != '0.32.0':
        raise MiaConfigurationError(f'MIA outside-buffer startup requires vLLM 0.32.0, got {version}')
    if getattr(module, '_mia_memory_startup_patched', False):
        return
    if not callable(getattr(module, 'request_memory', None)):
        raise MiaConfigurationError('vLLM 0.32 request_memory startup boundary is unavailable')
    original_init = module.Worker.init_device
    original_request = module.request_memory
    original_shutdown = getattr(module.Worker, "shutdown", None)

    @wraps(original_request)
    def request(snapshot, cache, external_weight_memory=0):
        worker = _INITIALIZING.get()
        if worker is None:
            return original_request(snapshot, cache, external_weight_memory)
        if getattr(worker, '_mia_memory_baseline_finalized', False):
            raise MiaConfigurationError('MIA initial memory baseline reached twice')
        if external_weight_memory:
            raise MiaConfigurationError('Externally held weights shrink the engine budget; '
                                        'unsupported with MIA outside-buffer budgeting')
        reserve_before_model_baseline(worker, snapshot)
        snapshot.measure()
        if snapshot.free_memory < worker.vllm_config.cache_config.gpu_memory_utilization * snapshot.total_memory:
            raise MiaSizingError('MIA allocation fits the nominal outside budget but leaves '
                                 'insufficient physical GPU memory for the model budget')
        worker._mia_memory_baseline_finalized = True
        result = original_request(snapshot, cache, external_weight_memory)
        owner = getattr(worker, '_mia_memory_owner', None)
        if owner is not None:
            _LOG.info('MIA outside GPU storage: %s; physical free after reservation=%d B',
                      owner.summary(), snapshot.free_memory)
        return result

    @wraps(original_init)
    def init(worker, *args, **kwargs):
        kind = _kind(worker)
        if kind is None:
            return original_init(worker, *args, **kwargs)
        flag = {"hidden_states": "MIA_CAPTURE_FUSED", "steer": "MIA_STEER_FUSED"}.get(kind)
        if flag and os.environ.get(flag, "1") != "1":
            raise MiaConfigurationError(f"MIA outside-buffer mode requires {flag}=1")
        if getattr(worker.vllm_config.cache_config, 'kv_cache_memory_bytes', None) is not None:
            raise MiaConfigurationError('Explicit kv_cache_memory_bytes bypasses utilization; '
                                        'unsupported with MIA outside-buffer budgeting')
        token = _INITIALIZING.set(worker)
        try:
            result = original_init(worker, *args, **kwargs)
            if not getattr(worker, '_mia_memory_baseline_finalized', False):
                raise MiaConfigurationError('vLLM did not reach the MIA initial memory boundary')
            return result
        except BaseException:
            worker._mia_memory_baseline_finalized = False
            owner = getattr(worker, '_mia_memory_owner', None)
            if owner is not None:
                owner.close()
                del worker._mia_memory_owner
            raise
        finally:
            _INITIALIZING.reset(token)

    if original_shutdown is not None:
        @wraps(original_shutdown)
        def shutdown(worker, *args, **kwargs):
            if getattr(worker, "_mia_memory_owner", None) is None:
                return original_shutdown(worker, *args, **kwargs)
            close_mia_consumers(worker)
            result = original_shutdown(worker, *args, **kwargs)
            release_mia_storage(worker)
            return result
        module.Worker.shutdown = shutdown

    def guard_startup(original):
        @wraps(original)
        def guarded(worker, *args, **kwargs):
            try:
                return original(worker, *args, **kwargs)
            except BaseException as failure:
                try:
                    abort_mia_startup(worker)
                except BaseException as cleanup_error:
                    # Retain storage if a consumer or graph could still access it.
                    if hasattr(failure, "add_note"):
                        failure.add_note(f"MIA startup cleanup also failed: {cleanup_error}")
                    _LOG.exception("MIA startup cleanup failed; device storage retained")
                raise
        return guarded

    for name in ("load_model", "determine_available_memory", "initialize_from_config",
                 "compile_or_warm_up_model"):
        original = getattr(module.Worker, name, None)
        if callable(original):
            setattr(module.Worker, name, guard_startup(original))
    module.Worker.init_device = init
    module.Worker.mia_memory_budget = memory_budget_info
    module.request_memory = request
    module._mia_memory_startup_patched = True
