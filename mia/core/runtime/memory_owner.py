"""Own MIA's device allocations; every installed tensor is a view of this storage."""
from __future__ import annotations

import torch

from mia.errors import MiaConfigurationError, MiaSizingError
from .memory_plan import MiaMemoryPlan


class MiaMemoryOwner:
    def __init__(self, plan: MiaMemoryPlan):
        self.plan = plan
        self._storage = {}
        self._views = {}
        self._closed = False

    @classmethod
    def allocate(cls, plan: MiaMemoryPlan, device) -> 'MiaMemoryOwner':
        plan.check_fit()
        owner = cls(plan)
        try:
            for category, size in [('aperture', plan.aperture_bytes),
                                   ('auxiliary', plan.auxiliary_bytes),
                                   ('scratch', plan.scratch_bytes)]:
                if size:
                    owner._storage[category] = torch.zeros(size, dtype=torch.uint8, device=device)
            for spec in plan.buffers:
                storage = owner._storage[spec.category]
                if spec.offset + spec.nbytes > storage.numel():
                    raise MiaConfigurationError(f'MIA buffer {spec.name} exceeds its storage')
                owner._views[spec.name] = storage.narrow(0, spec.offset, spec.nbytes).view(
                    getattr(torch, spec.dtype)).view(spec.shape)
        except Exception as e:
            owner.close()
            if isinstance(e, torch.OutOfMemoryError):
                raise MiaSizingError(f'MIA could not allocate its outside storage: '
                                     f'{plan.total_bytes} B requested') from e
            raise
        return owner

    def view(self, name: str) -> torch.Tensor:
        if self._closed:
            raise RuntimeError('MIA memory owner is closed')
        try:
            return self._views[name]
        except KeyError as e:
            raise MiaConfigurationError(f'MIA storage was not planned for {name!r}') from e

    def summary(self) -> dict:
        p = self.plan
        return dict(model_bytes=p.model_bytes, outside_bytes=p.outside_bytes,
                    aperture_bytes=p.aperture_bytes, auxiliary_bytes=p.auxiliary_bytes,
                    scratch_bytes=p.scratch_bytes, owned_bytes=p.total_bytes,
                    remaining_bytes=p.remaining_bytes,
                    storage={k: {'bytes': v.numel(), 'pointer': v.data_ptr()}
                             for k, v in self._storage.items()})

    def close(self) -> None:
        """Drop ownership after consumers/graphs release their views, or on init failure."""
        self._views.clear()
        self._storage.clear()
        self._closed = True


def device_buffer(owner, name: str, shape, dtype, device, *, fill=0):
    """The planned view ``name``, or a fresh tensor when there is no owner (eager)."""
    if owner is None:
        return torch.full(tuple(shape), fill, dtype=dtype, device=device)
    value = owner.view(name)
    if tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != torch.device(device):
        raise MiaConfigurationError(f'MIA planned buffer geometry differs for {name}: '
                                    f'{tuple(value.shape)}/{value.dtype}/{value.device} versus '
                                    f'{tuple(shape)}/{dtype}/{device}')
    if fill:
        value.fill_(fill)
    return value


def validate_layer_matches(owner, matched) -> None:
    """Refuse layer matches other than each planned layer exactly once, sinks included."""
    if owner is None:
        return
    actual = [int(layer) for _, _, layer in matched]
    expected = list(range(owner.plan.layers))
    if sorted(actual) != expected:
        raise MiaConfigurationError(
            f"MIA layer matches differ from the startup plan: expected {expected}, got {actual}")
