"""Worker-flush barrier for CUDA-graph capture."""


def drain_barrier(worker) -> None:
    """Wait for pending egress and capture drains before reading buckets."""
    em = getattr(worker, "_egress_stream", None)
    if em is not None:
        em.synchronize()
    c = getattr(worker, "_capture_consumer", None)
    if c is not None:
        c.sync(worker)


__all__ = ["drain_barrier"]

