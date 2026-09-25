"""Worker-flush barrier for CUDA-graph capture.

The pre-aperture off-loop drain subsystem this once served has been removed in favor of the
off-loop ``CaptureAperture`` + ``aperture_drain_hs``/``aperture_drain_qk`` mechanism. Only the
flush-time barrier survives here — it waits on whatever side-stream state a worker actually
holds (read via ``getattr`` so it is a no-op when the attribute is unset, which is the case
on every live path today).
"""


def drain_barrier(worker) -> None:
    """Worker-flush helper: wait for any pending egress gather + capture drain before
    reading buckets (both run on side streams; the flush reads the resulting tensors).
    Also recycles any disk-path pages the writer feeder finished copying out."""
    em = getattr(worker, "_egress_stream", None)
    if em is not None:
        em.synchronize()
    c = getattr(worker, "_capture_consumer", None)
    if c is not None:
        c.sync(worker)


__all__ = ["drain_barrier"]
