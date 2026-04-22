import os
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")

from vllm_hook_plugins import HookLLM

if __name__ == "__main__":

    cache_dir = "./cache/"
    hook_dir  = "/dev/shm/vllm_hook" # None
    model = "Qwen/Qwen2.5-3B-Instruct"

    llm = HookLLM(
        model=model,
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=f"model_configs/hidden_states/{model.split('/')[-1]}.json",
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=torch.float16,
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
        output = llm.generate(case, temperature=0.0, max_tokens=10)
        stats = llm.analyze(analyzer_spec={"reduce": "none"})

        print(f"\nPrompt: '{case}'")
        print(f"Generated: '{output[0].outputs[0].text.strip()}'")
        for layer_name, tensors in sorted(stats["hidden_states"].items()):
            t = tensors[0]
            print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")

        llm.llm_engine.reset_prefix_cache()

    print("=" * 50)
    print("Batch processing examples...")
    output = llm.generate(test_cases, temperature=0.0, max_tokens=10)
    stats = llm.analyze(analyzer_spec={"reduce": "norm"})

    for i, prompt in enumerate(test_cases):
        print(f"\nPrompt [{i}]: '{prompt}'")
        for layer_name, norms in sorted(stats["hidden_states"].items()):
            print(f"  {layer_name}: norm={norms[i]:.4f}")
