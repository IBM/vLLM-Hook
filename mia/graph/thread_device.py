"""Every thread MIA starts that can reach CUDA selects its device FIRST.

The CUDA "current device" is per THREAD, and a new thread starts on device 0. vLLM runs tensor-
parallel rank ``r`` on ``cuda:r`` of a process that can still see every GPU of the node (it sets the
device with ``torch.cuda.set_device`` on the worker's main thread; ``CUDA_VISIBLE_DEVICES`` keeps the
whole node). So on any rank >= 1, a thread that never selects its device sends every implicit-device
CUDA call to ``cuda:0``, a GPU that another process (``Worker_TP0``) owns:

* under LSF's ``exclusive_process`` mode (every study job) the call raises
  ``cudaErrorDevicesUnavailable`` -- "CUDA-capable device(s) is/are busy or unavailable";
* without it, the thread silently opens a second context on GPU 0 and works on the wrong device.

Gate G1 (LSF 1777562) hit the first. The QK aperture drain consumer died on every rank >= 1 because
``torch.cuda.stream(copy_stream)`` recorded the thread's current stream on DEVICE 0 at ``__enter__``
and restored it at ``__exit__`` (``torch.cuda.set_stream`` -> ``cudaSetDevice(0)``). Rank 0 never
failed, because there device 0 is the right device, so TP=1 could not have caught it.

THE RULE: a thread body's first statement is ``bind_thread_to_device(<device>)``, where the device is
the one the thread's CUDA objects live on (a drain's copy stream), or the one its CREATOR thread was on
(``creator_cuda_device()``, captured in the constructor, which runs on the worker's main thread after
vLLM selected the device). A thread that cannot select its device raises instead of touching another.
"""
from __future__ import annotations

# torch is imported inside the functions: writer_process imports this module at its top level and
# promises its spawned child no module-level torch import (see that module's docstring).


def creator_cuda_device():
    """The CALLING thread's current CUDA device, or None when this process has not initialized CUDA.

    Never initializes CUDA itself (``torch.cuda.current_device()`` would), so a CPU-only process, the
    API server and a unit test stay CUDA-free and their threads bind nothing."""
    import torch
    try:
        if not torch.cuda.is_initialized():
            return None
        return torch.device("cuda", torch.cuda.current_device())
    except Exception:  # noqa: BLE001 -- "no CUDA here" is the answer, not an error
        return None


def bind_thread_to_device(device):
    """Make ``device`` the calling thread's current CUDA device and return it.

    ``None`` or a non-CUDA device is a no-op that returns None (the CPU paths the unit tests drive).
    A CUDA device without an index is refused: "cuda" means "the current device", which is exactly the
    per-thread ambiguity this function exists to remove. Anything ``torch.cuda.set_device`` raises
    propagates -- the caller's thread must fail loud rather than go on against another GPU."""
    if device is None:
        return None
    import torch
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
