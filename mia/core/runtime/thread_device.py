"""Bind every MIA thread that can reach CUDA to its device first."""
from __future__ import annotations

import torch


def creator_cuda_device():
    """The calling thread's current CUDA device, or None when CUDA is not initialized."""
    try:
        if not torch.cuda.is_initialized():
            return None
        return torch.device("cuda", torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None


def bind_thread_to_device(device):
    """Make ``device`` the calling thread's current CUDA device and return it."""
    if device is None:
        return None
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    if dev.index is None:
        raise ValueError(
            f"bind_thread_to_device({device!r}): a CUDA device without an index names the calling "
            f"thread's current device, which in a new thread is cuda:0 -- pass the explicit device")
    torch.cuda.set_device(dev)
    return dev


__all__ = ["bind_thread_to_device", "creator_cuda_device"]

