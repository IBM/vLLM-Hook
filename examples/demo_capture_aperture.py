"""CUDA-graph hidden-state capture: two runs byte-identical, and a disk run equal to in-memory.

Runs offline (`MiaLLM`): the determinism check needs two generations on one engine. Run with
``MIA_PROFILE=1`` to also print the profiler counters.
"""
import multiprocessing as mp
import os
import time

import torch
from vllm import SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

from mia import MiaLLM
from mia._profiler import PROF
from mia.optimizations import PUBLIC_LEVERS, describe
from _paths import config_path

#: Why each other lever is not flipped on this one engine.
LEVER_NOTES = {
    "steer_fused": "steering only (see demo_actsteer.py).",
    "compact_kall": "Q/K capture only; it selects itself per request.",
    "storage_router": "served requests only; inert offline.",
    "artifact_dtype": "lossy quantization; left off so the checks above compare exact bytes.",
    "aperture_mmap": "a disk-sink write path; the same bytes either way.",
    "aperture_max_batched_tokens": "guards heavy captures at high batch; this batch is small.",
}


def _print_evidence(elapsed_s: float, n_tokens: int, label: str) -> None:
    """Print a run's wall time and its time per decode step."""
    per_step = (elapsed_s * 1000 / n_tokens) if n_tokens else float("nan")
    print(f"[evidence:{label}] generate: {elapsed_s * 1000:.1f} ms total, "
          f"{per_step:.2f} ms/decode-step over {n_tokens} tokens")


def _all_equal(tensors_a: dict, tensors_b: dict):
    """``(all layers byte-identical, mismatch descriptions)`` for two analyzer results."""
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


def _probes(outputs, which: str) -> dict:
    """The captured states on the first output, or exit naming why there are none."""
    probes = getattr(outputs[0], "probes", None)
    if probes is None:
        raise SystemExit(
            f"[demo_capture_aperture] {which}: the engine returned no captured states.\n"
            f"  MIA_PROFILE_MODE routes capture straight to the sink; leave it unset.\n"
            f"  MIA_PROFILE_MODE={os.environ.get('MIA_PROFILE_MODE')!r}")
    return probes


def main() -> None:
    hook_dir = "/dev/shm/mia"
    model = os.environ.get("MIA_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
    config_file = os.environ.get(
        "MIA_CONFIG_FILE",
        config_path(f"hidden_states/{model.split('/')[-1]}.json"))

    print("=" * 70)
    print(f"[demo_capture_aperture] model={model}  config={config_file}")
    print("[demo_capture_aperture] shipped optimization levers:")
    print(describe())
    print("=" * 70)

    llm = MiaLLM(
        model=model,
        worker_name="capture_hs",
        analyzer_name="hidden_states",
        config_file=config_file,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=torch.float16,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    # CUDA graphs unless enforce_eager=True.
    vc = llm.llm_engine.vllm_config
    graph_mode = not vc.model_config.enforce_eager
    print(f"[demo_capture_aperture] mode={'CUDA-graph capture' if graph_mode else 'EAGER'} "
          f"(enforce_eager={vc.model_config.enforce_eager}, "
          f"cudagraph_mode={vc.compilation_config.cudagraph_mode.name})")

    prompts = [
        "The capital of France is",
        "Quantum computing leverages",
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=32)

    t0 = time.perf_counter()
    out1 = llm.generate(prompts, sampling, save_to_disk=False)
    elapsed1 = time.perf_counter() - t0
    n_tokens1 = sum(len(o.outputs[0].token_ids) for o in out1)
    stats1 = llm.analyze(analyzer_spec={"reduce": "none"}, probes=_probes(out1, "run 1"))
    text1 = [o.outputs[0].text for o in out1]

    llm.llm_engine.reset_prefix_cache()

    t0 = time.perf_counter()
    out2 = llm.generate(prompts, sampling, save_to_disk=False)
    elapsed2 = time.perf_counter() - t0
    n_tokens2 = sum(len(o.outputs[0].token_ids) for o in out2)
    stats2 = llm.analyze(analyzer_spec={"reduce": "none"}, probes=_probes(out2, "run 2"))
    text2 = [o.outputs[0].text for o in out2]

    det_ok, det_mismatches = _all_equal(stats1["hidden_states"], stats2["hidden_states"])
    print("\n[check 1/2] capture-aperture determinism (same engine, same prompts, run twice)")
    _print_evidence(elapsed1, n_tokens1, "run1")
    _print_evidence(elapsed2, n_tokens2, "run2")
    print(f"[check 1/2] generated text identical: {text1 == text2}")
    print(f"[check 1/2] captured hidden states byte-identical across the two runs: {det_ok}")
    if not det_ok:
        print(f"[check 1/2] mismatches (first 5): {det_mismatches[:5]}")

    wp_on = os.environ.get("MIA_WRITER_PROCESS", "1") != "0"
    llm.llm_engine.reset_prefix_cache()
    run_id = "capture_aperture_demo_writer_process"
    t0 = time.perf_counter()
    out3 = llm.generate(prompts, sampling, save_to_disk=True, run_id=run_id)
    elapsed3 = time.perf_counter() - t0
    n_tokens3 = sum(len(o.outputs[0].token_ids) for o in out3)
    stats3 = llm.analyze(analyzer_spec={"reduce": "none"}, run_id=run_id)
    text3 = [o.outputs[0].text for o in out3]

    wp_ok, wp_mismatches = _all_equal(stats1["hidden_states"], stats3["hidden_states"])
    print(f"\n[check 2/2] writer_process={'on' if wp_on else 'off'} (shipped default): "
          f"disk-retrieved capture vs the in-memory capture from check 1")
    _print_evidence(elapsed3, n_tokens3, "disk-path")
    print(f"[check 2/2] generated text identical to check 1: {text1 == text3}")
    print(f"[check 2/2] disk-retrieved hidden states byte-identical to in-memory "
          f"capture: {wp_ok}")
    if not wp_ok:
        print(f"[check 2/2] mismatches (first 5): {wp_mismatches[:5]}")
    print("[check 2/2] writer_process on vs off needs a second engine; not compared here.")

    snap = PROF.summary_only()
    print("\n[profiler] driver-process counters/timers "
          f"(MIA_PROFILE={'1' if snap['enabled'] else '0'}):")
    if snap["enabled"]:
        print(f"  counters: {snap['counters']}")
        miallm_timers = {k: v for k, v in snap["timers"].items() if k.startswith("miallm.")}
        for name, t in sorted(miallm_timers.items()):
            print(f"  {name}: mean={t.get('mean', float('nan')):.3f}ms n={t.get('count', 0)}")
        print("[profiler] the capture counters (hs.aperture.*, graph.*, captured.bytes.*) are the "
              "engine process's: at exit it writes them to "
              f"{os.environ.get('MIA_PROFILE_DIR', '/tmp/mia_profile')}/profile-*.json.")
    else:
        print("  disabled -- run with MIA_PROFILE=1 to collect them.")

    print("\n[levers] why the other PUBLIC_LEVERS entries are not flipped on this one engine:")
    for key in PUBLIC_LEVERS:
        if key == "writer_process":
            continue
        print(f"  - {key}: {LEVER_NOTES.get(key, 'not discussed')}")

    if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "engine_core"):
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    # Release anything of the distributed state the engine shutdown missed.
    destroy_model_parallel()
    destroy_distributed_environment()

