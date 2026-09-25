"""Hidden-state capture demo: capture layer activations and read them back."""
import os
import multiprocessing as mp
import time
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("MIA_USE_SAFETENSORS", "1")

from vllm import SamplingParams
from mia import MiaLLM


def _print_evidence(elapsed_s: float, n_tokens: int) -> None:
    per_step = (elapsed_s * 1000 / n_tokens) if n_tokens else float("nan")
    print(f"[evidence] generate: {elapsed_s * 1000:.1f} ms total, "
          f"{per_step:.2f} ms/decode-step over {n_tokens} tokens")

    from mia._profiler import PROF
    snap = PROF.summary_only()
    if snap["enabled"]:
        print(f"[evidence] profiler counters: {snap['counters']}")
    else:
        print("[evidence] profiler disabled -- set MIA_PROFILE=1 to see hook/aperture counters")

    from mia.optimizations import describe
    print("[evidence] active optimization levers:")
    print(describe())


if __name__ == "__main__":
    cache_dir = "./cache/"
    hook_dir  = "/dev/shm/mia"
    model = os.environ.get("MIA_DEMO_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    config_file = os.environ.get(
        "MIA_CONFIG_FILE",
        f"model_configs/hidden_states/{model.split('/')[-1]}.json")

    GRAPH_MODE = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"
    print(f"[demo_hiddenstate] mode={'FULL CUDA-graph capture' if GRAPH_MODE else 'eager'} "
          f"(MIA_ALLOW_CUDAGRAPH={'1' if GRAPH_MODE else '0'})")

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
        enforce_eager=not GRAPH_MODE,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    test_cases = [
        "The capital of France is",
        "Quantum computing leverages",
    ]

    print("=" * 50)
    for case in test_cases:
        t0 = time.time()
        output = llm.generate(case, SamplingParams(temperature=0.0, max_tokens=10), save_to_disk=True)
        elapsed = time.time() - t0
        stats = llm.analyze(analyzer_spec={"reduce": "none"})

        print(f"\nPrompt: '{case}'")
        print(f"Generated: '{output[0].outputs[0].text.strip()}'")
        for layer_name, tensors in sorted(stats["hidden_states"].items()):
            t = tensors[0]
            print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")
        _print_evidence(elapsed, len(output[0].outputs[0].token_ids))

        llm.llm_engine.reset_prefix_cache()

    print("=" * 50)
    print("Batch processing examples...")
    t0 = time.time()
    output = llm.generate(test_cases, SamplingParams(temperature=0.0, max_tokens=10), save_to_disk=True)
    elapsed = time.time() - t0
    stats = llm.analyze(analyzer_spec={"reduce": "norm"})

    for i, prompt in enumerate(test_cases):
        print(f"\nPrompt [{i}]: '{prompt}'")
        for layer_name, norms in sorted(stats["hidden_states"].items()):
            print(f"  {layer_name}: norm={norms[i]:.4f}")
    n_tokens = sum(len(o.outputs[0].token_ids) for o in output)
    _print_evidence(elapsed, n_tokens)

