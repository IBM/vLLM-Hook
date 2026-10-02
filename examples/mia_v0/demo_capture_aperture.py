"""FULL CUDA-graph hidden-state capture demo, with evidence for the optimization levers.
Runs in-process (`MiaLLM`) on purpose: this is the local FULL-CUDA-graph showcase, and
the determinism check below needs two generations against one engine. For the server
path see the demos that use `MiaClient`.
"""
import os
import multiprocessing as mp
import time
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("MIA_ALLOW_CUDAGRAPH", "1")
os.environ.setdefault("MIA_PROFILE", "1")

import vllm
from vllm import SamplingParams
from mia import MiaLLM
from _paths import config_path
from mia._profiler import PROF
from mia.optimizations import describe, PUBLIC_LEVERS


LEVER_NOTES = {
    "batched_egress": (
        "Read once at import (graph/install.py:85, `_BATCHED_EGRESS`); in this "
        "checkout it belongs to the QK capture/egress path, not the hidden-states "
        "path this demo captures, so there is nothing to flip here."
    ),
    "steer_fused": (
        "Read once at import (graph/ops.py:24, `_STEER_FUSED`); applies only to "
        "the steer worker (see examples/demo_actsteer.py), which this "
        "demo does not load."
    ),
    "compact_kall": (
        "QK-only (workers/qk_capture_worker.py:40). With the shipped 'auto' "
        "default it self-selects PER REQUEST -- compact only once a request's "
        "growing-prefix rows reach 2 (`_use_compact_kall`) -- there is no "
        "user-facing toggle to flip mid-engine even in principle. This demo "
        "captures hidden states, not QK, so it never exercises this path."
    ),
    "writer_process": (
        "Resolved once per worker, at worker init, and cached for the engine's "
        "lifetime (graph/writer_process.py:238 `WriterProcess.from_env`, called "
        "once via `init_writer_process`, itself idempotent -- see "
        "workers/hs_capture_worker.py:149-150). Comparing on vs off needs "
        "a second engine, which this script does not build. What IS checked above, "
        "in-process, on the one engine: that the shipped-on writer path reproduces "
        "the in-memory capture byte-for-byte -- the load-bearing half of the claim."
    ),
    "storage_router": (
        "Serve-only; _plugin.py itself warns it is inert for LLM.generate "
        "(offline), which is all MiaLLM ever calls. N/A to this demo."
    ),
    "artifact_dtype": (
        "The one PUBLIC_LEVERS entry that is deliberately LOSSY. Left at its "
        "'native' (off) default throughout so every comparison above is "
        "apples-to-apples; quantifying its error is a different demo."
    ),
    "aperture_mmap": (
        "A durable-sink scheduling choice for the disk path (mmap vs plain "
        "append) -- same bytes either way per its own docstring in "
        "optimizations.py. Not independently re-verified beyond the "
        "writer_process byte-identity check above, which exercises the disk "
        "path this lever also touches."
    ),
    "aperture_max_batched_tokens": (
        "Resolved once, in the DRIVER, while LLM(...) is still building the "
        "engine config -- before any worker exists (_plugin.py, "
        "`_maybe_autocap_max_batched_tokens`, called from "
        "`_patched_create_engine_config`). It only ever LOWERS the scheduler's "
        "token budget, and only matters to guard heavy full-graph capture's "
        "per-step transient at HIGH batch. This demo's batch is deliberately "
        "small (a handful of short prompts on a single modest GPU), so there is "
        "nothing for it to guard against here; left at its off default."
    ),
}


def _print_evidence(elapsed_s: float, n_tokens: int, label: str) -> None:
    per_step = (elapsed_s * 1000 / n_tokens) if n_tokens else float("nan")
    print(f"[evidence:{label}] generate: {elapsed_s * 1000:.1f} ms total, "
          f"{per_step:.2f} ms/decode-step over {n_tokens} tokens")


def _all_equal(tensors_a: dict, tensors_b: dict):
    mismatches = []
    for layer_name in sorted(set(tensors_a) | set(tensors_b)):
        list_a = tensors_a.get(layer_name)
        list_b = tensors_b.get(layer_name)
        if list_a is None or list_b is None or len(list_a) != len(list_b):
            mismatches.append(f"{layer_name}: missing or request-count mismatch "
                               f"({len(list_a) if list_a is not None else 'N/A'} vs "
                               f"{len(list_b) if list_b is not None else 'N/A'})")
            continue
        for i, (ta, tb) in enumerate(zip(list_a, list_b)):
            if ta.shape != tb.shape:
                mismatches.append(f"{layer_name}[{i}]: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
            elif not torch.equal(ta, tb):
                mismatches.append(f"{layer_name}[{i}]: same shape {tuple(ta.shape)}, values differ")
    return (len(mismatches) == 0), mismatches


def main() -> None:
    cache_dir = "./cache/"
    hook_dir = "/dev/shm/mia"
    model = os.environ.get("MIA_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
    config_file = os.environ.get(
        "MIA_CONFIG_FILE",
        config_path(f"hidden_states/{model.split('/')[-1]}.json"))

    graph_mode = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"
    print("=" * 70)
    print(f"[demo_capture_aperture] mode={'FULL CUDA-graph capture' if graph_mode else 'EAGER (fallback)'} "
          f"(MIA_ALLOW_CUDAGRAPH={'1' if graph_mode else '0'})")
    if not graph_mode:
        print("[demo_capture_aperture] NOTE: none of the graph-aperture claims below apply in eager mode; "
              "set MIA_ALLOW_CUDAGRAPH=1 to actually exercise the capture aperture.")
    print(f"[demo_capture_aperture] model={model}  config={config_file}")
    print("[demo_capture_aperture] shipped optimization levers:")
    print(describe())
    print("=" * 70)

    llm = MiaLLM(
        model=model,
        worker_name="capture_hs",
        analyzer_name="hidden_states",
        config_file=config_file,
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=torch.float16,
        enforce_eager=not graph_mode,
        compilation_config={"cudagraph_mode": "FULL"} if graph_mode else None,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    prompts = [
        "The capital of France is",
        "Quantum computing leverages",
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=32)

    t0 = time.perf_counter()
    out1 = llm.generate(prompts, sampling, save_to_disk=False)
    elapsed1 = time.perf_counter() - t0
    n_tokens1 = sum(len(o.outputs[0].token_ids) for o in out1)
    stats1 = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out1[0].probes)
    text1 = [o.outputs[0].text for o in out1]

    llm.llm_engine.reset_prefix_cache()

    t0 = time.perf_counter()
    out2 = llm.generate(prompts, sampling, save_to_disk=False)
    elapsed2 = time.perf_counter() - t0
    n_tokens2 = sum(len(o.outputs[0].token_ids) for o in out2)
    stats2 = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out2[0].probes)
    text2 = [o.outputs[0].text for o in out2]

    det_ok, det_mismatches = _all_equal(stats1["hidden_states"], stats2["hidden_states"])
    print("\n[check 1/2] capture-aperture determinism (same engine, same prompts, run twice)")
    _print_evidence(elapsed1, n_tokens1, "run1")
    _print_evidence(elapsed2, n_tokens2, "run2")
    print(f"[check 1/2] generated text identical: {text1 == text2}")
    print(f"[check 1/2] captured hidden states byte-identical across the two runs: {det_ok}")
    if not det_ok:
        print(f"[check 1/2] mismatches (first 5): {det_mismatches[:5]}")

    llm.llm_engine.reset_prefix_cache()
    run_id = "capture_aperture_demo_writer_process"
    t0 = time.perf_counter()
    out3 = llm.generate(prompts, sampling, save_to_disk=True, run_id=run_id)
    elapsed3 = time.perf_counter() - t0
    n_tokens3 = sum(len(o.outputs[0].token_ids) for o in out3)
    stats3 = llm.analyze(analyzer_spec={"reduce": "none"}, run_id=run_id)
    text3 = [o.outputs[0].text for o in out3]

    wp_ok, wp_mismatches = _all_equal(stats1["hidden_states"], stats3["hidden_states"])
    wp_on = os.environ.get("MIA_WRITER_PROCESS", "1") != "0"
    print(f"\n[check 2/2] writer_process={'on' if wp_on else 'off'} (shipped default): "
          f"disk-retrieved capture vs the in-memory capture from check 1")
    _print_evidence(elapsed3, n_tokens3, "disk-path")
    print(f"[check 2/2] generated text identical to check 1: {text1 == text3}")
    print(f"[check 2/2] disk-retrieved hidden states byte-identical to in-memory capture: {wp_ok}")
    if not wp_ok:
        print(f"[check 2/2] mismatches (first 5): {wp_mismatches[:5]}")
    print("[check 2/2] NOT checked here (would need a second engine): the writer_process=off "
          "timing comparison and its disk-SLO-knee claim -- see LEVER_NOTES['writer_process'].")

    snap = PROF.summary_only()
    print("\n[profiler] driver-process counters/timers "
          f"(MIA_PROFILE={'1' if snap['enabled'] else '0'}):")
    if snap["enabled"]:
        print(f"  counters: {snap['counters']}")
        hookllm_timers = {k: v for k, v in snap["timers"].items() if k.startswith("hookllm.")}
        for name, t in sorted(hookllm_timers.items()):
            print(f"  {name}: mean={t.get('mean', float('nan')):.3f}ms n={t.get('count', 0)}")
        print("[profiler] the capture aperture's own counters (qk.aperture.*, hs.aperture.*, graph.forward, "
              "graph.drain, captured.bytes.*) live in the WORKER subprocess, not here -- they are "
              "dumped at worker exit to "
              f"{os.environ.get('MIA_PROFILE_DIR', '/tmp/mia_profile')}/"
              "profile-worker-*.json (see mia/_profiler.py:_atexit_dump). Inspect "
              "that file after this script exits for the aperture-side numbers.")
    else:
        print("  disabled -- MIA_PROFILE was not '1' at import time.")

    print("\n[levers] why the remaining PUBLIC_LEVERS entries are not independently "
          "flipped on this one engine:")
    for key in PUBLIC_LEVERS:
        if key in ("writer_process",):
            continue
        print(f"  - {key}: {LEVER_NOTES.get(key, 'not discussed')}")

    if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "engine_core"):
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
    vllm.destroy_process_group()

