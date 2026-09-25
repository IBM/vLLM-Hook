"""Long-decode hidden-state capture demo with capture on every decode step."""
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
    hook_dir = "/dev/shm/mia"
    model = os.environ.get("MIA_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
    config_file = os.environ.get(
        "MIA_CONFIG_FILE",
        f"model_configs/hidden_states/{model.split('/')[-1]}.json")
    max_tokens = int(os.environ.get("MIA_DEMO_MAX_TOKENS", "128"))
    hooks_on = os.environ.get("MIA_DEMO_HOOKS_ON", "both")

    GRAPH_MODE = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"

    print(f"[longdec-hs] model={model} config={config_file} "
          f"max_tokens={max_tokens} hooks_on={hooks_on} "
          f"mode={'FULL CUDA-graph capture' if GRAPH_MODE else 'eager'} "
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
    ]

    print("=" * 50)
    for case in test_cases:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        sp.extra_args = {"hooks_on": hooks_on}

        t0 = time.time()
        output = llm.generate(case, sp, save_to_disk=True)
        elapsed = time.time() - t0
        stats = llm.analyze(analyzer_spec={"reduce": "none"})

        print(f"\nPrompt: '{case}'  (decode={max_tokens}, hooks_on={hooks_on})")
        print(f"Generated: '{output[0].outputs[0].text.strip()[:160]}'")
        for layer_name, tensors in sorted(stats["hidden_states"].items()):
            t = tensors[0]
            print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")
        _print_evidence(elapsed, len(output[0].outputs[0].token_ids))
        llm.llm_engine.reset_prefix_cache()

