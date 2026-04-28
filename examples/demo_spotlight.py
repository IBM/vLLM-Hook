import os
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM, generate_with_spotlight, register_plugins

if __name__ == "__main__":

    register_plugins()

    MODEL = "Qwen/Qwen2-1.5B-Instruct"
    cache_dir = os.path.expanduser("~/.cache")

    print(f"Loading {MODEL}...")

    llm = HookLLM(
        model=MODEL,
        worker_name="probe_spotlight",
        download_dir=cache_dir,
        gpu_memory_utilization=0.8,
        dtype=torch.float16,
        enable_hook=True,
        enforce_eager=True,  # Required for Spotlight (disables Flash Attention)
        enable_prefix_caching=False,
    )

    print(f"Model loaded: {MODEL}")

    # ========================================================
    # Test parameters
    # ========================================================
    prompt = (
        "Return the response for the following as a JSON: "
        "Write a 400 word paragraph on France and its food specifically "
        "focussing on dishes. Include the paragraph and the list of dishes "
        "mentioned the paragraph in seperate fields."
    )
    emph_strings = ["Return the response for the following as a JSON:"]
    alpha = 0.1
    max_tokens = 500
    temperature = 0.0

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # ========================================================
    # Baseline (no steering)
    # ========================================================
    print("\n" + "=" * 70)
    print("Generating baseline (no attention steering)...")
    print("=" * 70)

    outputs_baseline = llm.generate(
        prompts=[prompt],
        sampling_params=sampling_params,
        use_hook=False,
    )

    baseline_text = outputs_baseline[0].outputs[0].text
    print(baseline_text)

    # ========================================================
    # With Spotlight
    # ========================================================
    print("\n" + "=" * 70)
    print(f"Generating with Spotlight (alpha={alpha})...")
    print("=" * 70)

    outputs_spotlight = generate_with_spotlight(
        llm,
        prompts=[prompt],
        emph_strings=emph_strings,
        alpha=alpha,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    spotlight_text = outputs_spotlight[0].outputs[0].text
    print(spotlight_text)

    # ========================================================
    # Comparison
    # ========================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nPrompt: {prompt}")
    print(f"Emphasized span(s): {emph_strings}")
    print(f"Alpha (target attention): {alpha}")

    print("\n" + "-" * 70)
    print("BASELINE (No Steering)")
    print("-" * 70)
    print(baseline_text)

    print("\n" + "-" * 70)
    print(f"WITH SPOTLIGHT (alpha={alpha})")
    print("-" * 70)
    print(spotlight_text)
