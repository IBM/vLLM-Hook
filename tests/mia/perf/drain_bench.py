"""GPU micro-benchmark of the capture-aperture drains' write path (``MIA_APERTURE_WRITE_MODE``).

Drives the REAL off-loop drain classes (``OffLoopApertureDrain`` for HS, ``OffLoopQKApertureDrain``
for QK) with a synthetic GPU aperture: per-layer bf16 buffers on the GPU filled once with random
rows, one ``CaptureAperture`` cursor, and per step the engine's hand-off -- reserve the rows, record
a CUDA event on the current stream, ``enqueue`` the step's records. The drain does exactly what it
does in a serving run (D2H on its copy stream, the write path under test, the sidecar log,
``advance_drain``); nothing in the drain is stubbed.

Per step it reports the drain's own split (``drain.write_stats()["last"]``): ``d2h_s`` (D2H issue
until the last layer's rows landed), ``write_s`` (first write submitted until the last finished;
it overlaps ``d2h_s`` on the pooled path), ``write_tail_s`` (writes still running after D2H and
bookkeeping), ``bookkeeping_s`` (copy plans + sidecar records), ``step_s`` and the step's GB/s.
The first ``--warmup`` steps (host-buffer allocation: ~7 s of cudaHostAlloc for a 10 GiB 70B step)
are excluded from the summary.

Presets (``--preset``) are the shapes that matter; every field can be overridden:

  70b-hs-tp4     HS, 80 layers x 8192 hidden, 8192 rows/step (10.74 GB), rank 0 of TP4 with
                 MIA_HS_TP_SHARD=0 (the pre-shard layout: rank 0 drains every layer)
  70b-hs-tp4-shard  HS under the TP layer shard: rank 0's 20 of 80 layers (2.68 GB/step)
  70b-hs-tp8-shard  HS under the TP layer shard: rank 0's 10 of 80 layers (1.34 GB/step)
  8b-hs-tp4-shard   HS under the TP layer shard: rank 0's 8 of 32 layers x 4096 (0.54 GB/step)
  70b-qk-tp4     QK, 80 layers, q 2048 / k 256 per rank, 8192 rows/step
  70b-qk-tp8     QK, 80 layers, q 1024 / k 128 per rank, 8192 rows/step
  8b-hs-prefill  HS, 32 layers x 4096, 8192 rows/step (2.15 GB)
  8b-hs-decode   HS, 32 layers x 4096, 128 rows/step (one decode token x 128 requests)
  8b-qk-tp1      QK, 32 layers, q 4096 / k 1024, 8192 rows/step

Run on a GPU node, pointing ``--dir`` at the sink under test (node-local NVMe for the numbers that
matter); the raw files are deleted afterwards unless ``--keep``:

    python tests/mia/perf/drain_bench.py --preset 70b-hs-tp4 --mode auto --threads 2 \\
        --dir /opt/nvme/$USER/drain_bench --steps 4
    python tests/mia/perf/drain_bench.py --preset 70b-hs-tp4 --mode legacy --dir ... --steps 2

``--hs-shard`` (the ``*-shard`` presets set it) makes ``--layers`` the MODEL's layer count and
drains only rank ``--tp-rank``'s round-robin share of it (``mia.graph.tp_shard.hs_owned_layers``),
with the shard in the sidecar header, exactly as one TP rank does. One process benches one rank; the
node-level number for a TP-N run is N concurrent processes on N GPUs into the same sink.

``--pipeline`` enqueues steps back to back (the aperture holds ``--slots`` rows, default two steps)
to measure sustained drain throughput instead of isolated per-step latency. ``--verify`` reads
back the first and last measured step of the first and last layer and compares them with the GPU
rows. Output: one JSON document on stdout (and ``--out``).

As a pytest file it holds one ``gpu``-marked smoke test (tiny shapes, legacy vs auto, byte
compare); it is not collected by ``pytest tests/`` (the file name has no ``test_`` prefix).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time

PRESETS = {
    "70b-hs-tp4": dict(kind="hs", layers=80, hidden=8192, rows=8192, requests=1, tp_rank=0,
                       tp_size=4),
    "70b-qk-tp4": dict(kind="qk", layers=80, q_dim=2048, k_dim=256, rows=8192, requests=1,
                       tp_rank=0, tp_size=4),
    "70b-qk-tp8": dict(kind="qk", layers=80, q_dim=1024, k_dim=128, rows=8192, requests=1,
                       tp_rank=0, tp_size=8),
    "8b-hs-prefill": dict(kind="hs", layers=32, hidden=4096, rows=8192, requests=1, tp_rank=0,
                          tp_size=1),
    "8b-hs-decode": dict(kind="hs", layers=32, hidden=4096, rows=128, requests=128, tp_rank=0,
                         tp_size=1),
    "8b-qk-tp1": dict(kind="qk", layers=32, q_dim=4096, k_dim=1024, rows=8192, requests=1,
                      tp_rank=0, tp_size=1),
    "70b-hs-tp4-shard": dict(kind="hs", layers=80, hidden=8192, rows=8192, requests=1,
                             tp_rank=0, tp_size=4, hs_shard=True),
    "70b-hs-tp8-shard": dict(kind="hs", layers=80, hidden=8192, rows=8192, requests=1,
                             tp_rank=0, tp_size=8, hs_shard=True),
    "8b-hs-tp4-shard": dict(kind="hs", layers=32, hidden=4096, rows=8192, requests=1,
                            tp_rank=0, tp_size=4, hs_shard=True),
}


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--kind", choices=["hs", "qk"])
    p.add_argument("--layers", type=int)
    p.add_argument("--hidden", type=int, help="HS row width (elements)")
    p.add_argument("--dims", help="QK 'q_dim,k_dim' per rank (elements)")
    p.add_argument("--rows", type=int, help="rows per step (tokens captured per step)")
    p.add_argument("--requests", type=int, help="records per step (rows split evenly)")
    p.add_argument("--tp-size", type=int, help="TP degree recorded in the header (and --hs-shard)")
    p.add_argument("--tp-rank", type=int, help="TP rank recorded in the header (and --hs-shard)")
    p.add_argument("--hs-shard", action="store_true", default=None,
                   help="HS: --layers is the model's count; drain only --tp-rank's round-robin "
                        "share of it (the TP layer shard, MIA_HS_TP_SHARD)")
    p.add_argument("--steps", type=int, default=4, help="measured steps")
    p.add_argument("--warmup", type=int, default=1, help="unmeasured leading steps")
    p.add_argument("--mode", choices=["legacy", "buffered", "direct", "auto"], default="auto")
    p.add_argument("--threads", type=int, default=2, help="MIA_APERTURE_WRITE_THREADS")
    p.add_argument("--dir", required=True, help="sink directory (a run dir is created under it)")
    p.add_argument("--device", default="cuda:0",
                   help="cuda:N; 'cpu' is a GPU-free dry run of this script (CPU copy path)")
    p.add_argument("--slots", type=int, help="aperture rows (default: 2 steps)")
    p.add_argument("--pipeline", action="store_true",
                   help="enqueue steps back to back instead of one at a time")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--keep", action="store_true", help="keep the raw files")
    p.add_argument("--out", help="also write the JSON here")
    a = p.parse_args(argv)
    cfg = dict(PRESETS[a.preset]) if a.preset else {}
    for k in ("kind", "layers", "hidden", "rows", "requests", "tp_size", "tp_rank", "hs_shard"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    if a.dims:
        q, k = (int(x) for x in a.dims.split(","))
        cfg["q_dim"], cfg["k_dim"] = q, k
    cfg.setdefault("requests", 1)
    cfg.setdefault("tp_rank", 0)
    cfg.setdefault("tp_size", 1)
    missing = [k for k in ("kind", "layers", "rows") if k not in cfg]
    if cfg.get("kind") == "hs" and "hidden" not in cfg:
        missing.append("hidden")
    if cfg.get("kind") == "qk" and "q_dim" not in cfg:
        missing.append("dims")
    if missing:
        p.error(f"missing {missing} (give --preset or the fields)")
    if cfg.get("hs_shard") and cfg.get("kind") != "hs":
        p.error("--hs-shard applies to --kind hs only")
    if not (0 <= int(cfg["tp_rank"]) < int(cfg["tp_size"])):
        p.error(f"--tp-rank {cfg['tp_rank']} is not in [0, --tp-size {cfg['tp_size']})")
    return a, cfg


def run_bench(cfg: dict, *, mode: str, threads: int, base_dir: str, steps: int, warmup: int = 1,
              device: str = "cuda:0", slots=None, pipeline: bool = False, verify: bool = False,
              keep: bool = False) -> dict:
    """Run one configuration; returns the result dict (see module docstring)."""
    saved = {k: os.environ.get(k) for k in ("MIA_APERTURE_WRITE_MODE",
                                             "MIA_APERTURE_WRITE_THREADS")}
    os.environ["MIA_APERTURE_WRITE_MODE"] = mode
    os.environ["MIA_APERTURE_WRITE_THREADS"] = str(threads)
    try:
        return _run_bench(cfg, mode=mode, threads=threads, base_dir=base_dir, steps=steps,
                          warmup=warmup, device=device, slots=slots, pipeline=pipeline,
                          verify=verify, keep=keep)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_bench(cfg, *, mode, threads, base_dir, steps, warmup, device, slots, pipeline, verify,
               keep) -> dict:
    import torch

    from mia.graph.capture_aperture import CaptureAperture

    dev = torch.device(device)
    on_gpu = dev.type == "cuda"
    if on_gpu:
        torch.cuda.set_device(dev)
    kind, n_layers, rows = cfg["kind"], int(cfg["layers"]), int(cfg["rows"])
    n_req = max(1, min(int(cfg["requests"]), rows))
    n_slots = int(slots or 2 * rows)
    if n_slots < rows:
        raise ValueError(f"--slots {n_slots} < rows per step {rows}: a step could never be admitted")
    run_dir = os.path.join(base_dir, f"drain_bench_{kind}_{mode}_{os.getpid()}_{int(time.time())}")
    g = torch.Generator(device=dev).manual_seed(1234)
    t_setup = time.perf_counter()
    if kind == "hs":
        from mia.graph.aperture_drain_hs import OffLoopApertureDrain
        from mia.graph.aperture_metadata import ReqCaptureRecord
        hidden = int(cfg["hidden"])
        widths = [hidden]
        model_layers = n_layers
        header = {"dtype": "bfloat16", "row_shape": [hidden], "hidden": hidden,
                  "tp_rank": cfg["tp_rank"], "tp_size": cfg["tp_size"], "num_layers": model_layers,
                  "capture_all_ranks": False}
        if cfg.get("hs_shard"):
            # One TP rank's round-robin share of the layers, numbered and labelled as MIA does.
            from mia.graph.tp_shard import HSShard
            shard = HSShard.of(int(cfg["tp_rank"]), int(cfg["tp_size"]), model_layers)
            owned = list(shard.owned_layers)
            header.update(shard.as_header())
        else:
            owned = [L + 1 for L in range(model_layers)]
        n_layers = len(owned)
        ap = CaptureAperture(row_bytes=n_layers * hidden * 2, n_slots=n_slots, device=dev,
                             dtype=torch.bfloat16, row_shape=(hidden,))
        bufs = [(L, torch.randn(n_slots + 1, hidden, generator=g, device=dev,
                                dtype=torch.bfloat16)) for L in owned]
        drain = OffLoopApertureDrain(ap, bufs, run_dir, header)
        layer_ids = [L for L, _ in bufs]

        def records(start, sizes):
            out, s = [], start
            for i, n in enumerate(sizes):
                out.append(ReqCaptureRecord(req_id=f"bench-{i}", logical_start=s, n_rows=n,
                                            hs_mode="all_tokens", layers=layer_ids))
                s += n
            return out
    else:
        from mia.graph.aperture_drain_qk import OffLoopQKApertureDrain
        from mia.graph.aperture_metadata import QKReqCaptureRecord
        qd, kd = int(cfg["q_dim"]), int(cfg["k_dim"])
        widths = [qd, kd]
        ap = CaptureAperture(row_bytes=n_layers * (qd + kd) * 2, n_slots=n_slots, device=dev,
                             dtype=torch.bfloat16, row_shape=(kd,))
        bufs = [(L, torch.randn(n_slots + 1, qd, generator=g, device=dev, dtype=torch.bfloat16),
                 torch.randn(n_slots + 1, kd, generator=g, device=dev, dtype=torch.bfloat16))
                for L in range(n_layers)]
        header = {"dtype": "bfloat16", "q_row_shape": [qd], "k_row_shape": [kd], "q_dim": qd,
                  "k_dim": kd, "hookq_mode": "all_tokens", "tp_rank": cfg["tp_rank"],
                  "tp_size": cfg["tp_size"], "num_layers": n_layers}
        drain = OffLoopQKApertureDrain(ap, bufs, run_dir, header)
        layer_ids = [L for L, _, _ in bufs]
        prefix: dict = {}

        def records(start, sizes):
            out, s = [], start
            for i, n in enumerate(sizes):
                rid = f"bench-{i}"
                prefix[rid] = prefix.get(rid, 0) + n
                out.append(QKReqCaptureRecord(req_id=rid, k_start=s, k_rows=n, q_start=s, q_rows=n,
                                              prefix_end=prefix[rid], num_computed=0,
                                              layers=layer_ids))
                s += n
            return out
    if on_gpu:
        torch.cuda.synchronize()
    setup_s = time.perf_counter() - t_setup
    step_bytes = rows * n_layers * sum(widths) * 2
    sizes = [rows // n_req + (1 if i < rows % n_req else 0) for i in range(n_req)]
    drain.start()
    per_step, starts = [], []
    t_run = None
    try:
        for k in range(warmup + steps):
            if k == warmup:
                t_run = time.perf_counter()
            deadline = time.monotonic() + 600
            while True:
                start = ap.reserve(rows)
                if start is not None:
                    break
                if drain.error is not None:
                    raise RuntimeError("drain consumer died") from drain.error
                if time.monotonic() > deadline:
                    raise TimeoutError("aperture never freed rows")
                time.sleep(0.0002)
            starts.append(start)
            ev = None
            if on_gpu:
                ev = torch.cuda.Event()
                ev.record()                 # after the (synthetic) scatter, as the engine does
            drain.enqueue(records(start, sizes), start, rows, ev)
            if not pipeline:
                # The drain accounts a step after advance_drain; its `steps` counter moves last.
                while drain.write_stats()["steps"] < k + 1:
                    if drain.error is not None:
                        raise RuntimeError("drain consumer died") from drain.error
                    time.sleep(0.0002)
                if k >= warmup:
                    last = dict(drain.write_stats()["last"])
                    last["step"] = k - warmup
                    last["GBps"] = step_bytes / last["step_s"] / 1e9 if last["step_s"] else None
                    per_step.append(last)
        while ap._drain < ap._write:
            if drain.error is not None:
                raise RuntimeError("drain consumer died") from drain.error
            time.sleep(0.0002)
        run_s = time.perf_counter() - (t_run or time.perf_counter())
        summary_line = drain.write_path_summary()
        stats = drain.write_stats()
    finally:
        t_close = time.perf_counter()
        drain.close()
        close_s = time.perf_counter() - t_close

    result = {
        "config": dict(cfg, mode=mode, threads=threads, steps=steps, warmup=warmup,
                       slots=n_slots, pipeline=pipeline, device=str(dev), run_dir=run_dir,
                       drained_layers=len(bufs)),
        "write_path": summary_line,
        "step_bytes": step_bytes,
        "setup_s": setup_s,
        "close_s": close_s,
        "bytes_by_mode": {m: stats[f"bytes_{m}"] for m in ("direct", "buffered", "legacy")},
        "per_step": per_step,
    }
    if per_step:
        med = lambda key: statistics.median(s[key] for s in per_step)  # noqa: E731
        result["summary"] = {
            "step_s_median": med("step_s"),
            "GBps_median": step_bytes / med("step_s") / 1e9,
            "d2h_s_median": med("d2h_s"),
            "write_s_median": med("write_s"),
            "write_tail_s_median": med("write_tail_s"),
            "bookkeeping_s_median": med("bookkeeping_s"),
            "write_GBps_median": step_bytes / med("write_s") / 1e9 if med("write_s") else None,
        }
    if pipeline and steps:
        result["summary_pipeline"] = {"run_s": run_s, "GBps": steps * step_bytes / run_s / 1e9}
    if verify:
        result["verify"] = _verify(kind, run_dir, bufs, n_slots, rows, starts[warmup:] or starts,
                                   warmup_rows=warmup * rows)
    if not keep:
        shutil.rmtree(run_dir, ignore_errors=True)
    return result


def _verify(kind, run_dir, bufs, n_slots, rows, starts, warmup_rows) -> dict:
    """Compare the first and last measured step of the first and last layer with the GPU rows."""
    import numpy as np
    import torch

    checks = []
    for layer in (bufs[0], bufs[-1]):
        tensors = [("hs", layer[1])] if kind == "hs" else [("q", layer[1]), ("k", layer[2])]
        for tag, gbuf in tensors:
            name = (f"hs_layer_{layer[0]}.raw" if kind == "hs"
                    else f"qk_{tag}_layer_{layer[0]}.raw")
            path = os.path.join(run_dir, name)
            width = gbuf.shape[1]
            mm = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, width)
            for i in sorted({0, len(starts) - 1}):
                s = starts[i]
                file_row = warmup_rows + i * rows
                idx = [(s + j) % n_slots for j in range(rows)]
                want = gbuf[idx].cpu().view(torch.uint16).numpy()
                got = np.asarray(mm[file_row:file_row + rows])
                checks.append({"file": name, "step": i, "equal": bool(np.array_equal(want, got))})
    return {"ok": all(c["equal"] for c in checks), "checks": checks}


def main(argv=None) -> int:
    a, cfg = _parse(argv)
    os.makedirs(a.dir, exist_ok=True)
    res = run_bench(cfg, mode=a.mode, threads=a.threads, base_dir=a.dir, steps=a.steps,
                    warmup=a.warmup, device=a.device, slots=a.slots, pipeline=a.pipeline,
                    verify=a.verify, keep=a.keep)
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    return 0 if res.get("verify", {}).get("ok", True) else 1


# --------------------------------------------------------------------------------------------
# pytest: one GPU-marked smoke run (explicitly: pytest tests/mia/perf/drain_bench.py --gpu... -m gpu)
# --------------------------------------------------------------------------------------------

try:
    import pytest
except ImportError:  # pragma: no cover -- the CLI does not need pytest
    pytest = None

if pytest is not None:
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            return False

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_gpu(), reason="drives the drain from a real CUDA aperture")
    @pytest.mark.parametrize("kind", ["hs", "qk"])
    def test_drain_bench_smoke(tmp_path, kind):
        """Tiny shapes through the real drains on a GPU: legacy and auto produce the same files,
        --verify reads back the GPU rows, and the per-step split is populated."""
        pytest.importorskip("vllm")
        cfg = (dict(kind="hs", layers=4, hidden=256, rows=64, requests=4, tp_rank=0, tp_size=1)
               if kind == "hs" else
               dict(kind="qk", layers=4, q_dim=256, k_dim=64, rows=64, requests=4, tp_rank=0,
                    tp_size=1))
        out = {}
        for mode in ("legacy", "auto"):
            out[mode] = run_bench(cfg, mode=mode, threads=2, base_dir=str(tmp_path / mode),
                                  steps=3, warmup=1, verify=True, keep=True)
            assert out[mode]["verify"]["ok"], out[mode]["verify"]
            assert len(out[mode]["per_step"]) == 3
        a, b = out["legacy"]["config"]["run_dir"], out["auto"]["config"]["run_dir"]
        assert sorted(os.listdir(a)) == sorted(os.listdir(b))
        for f in os.listdir(a):
            with open(os.path.join(a, f), "rb") as fa, open(os.path.join(b, f), "rb") as fb:
                assert fa.read() == fb.read(), f


if __name__ == "__main__":
    sys.exit(main())
