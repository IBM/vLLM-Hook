"""Long-decode Attention-Tracker (Q/K capture) demo with capture on every decode step."""
import os
import multiprocessing as mp
import torch
import time

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


def apply_chat_template_and_get_ranges(tokenizer, model_name: str, instruction: str, data: str):
    """Apply the chat template and return token ranges, following Attention-Tracker."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "Data: " + data},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    instruction_len = len(tokenizer.encode(instruction))
    data_len = len(tokenizer.encode(data))
    if "granite-3.1" in model_name:
        data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
    elif "Mistral-7B" in model_name:
        data_range = ((3, 3 + instruction_len), (-1 - data_len, -1))
    elif "Qwen2-1.5B" in model_name:
        data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
    else:
        raise NotImplementedError
    return text, data_range


if __name__ == "__main__":
    cache_dir = "./cache/"
    hook_dir = "/dev/shm/mia"
    model = os.environ.get("MIA_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
    config_file = os.environ.get(
        "MIA_CONFIG_FILE",
        f'model_configs/attention_tracker/{model.split("/")[-1]}.json')
    max_tokens = int(os.environ.get("MIA_DEMO_MAX_TOKENS", "128"))
    hooks_on = os.environ.get("MIA_DEMO_HOOKS_ON", "both")

    dtype_map = {
        "mistralai/Mistral-7B-Instruct-v0.3": torch.float16,
        "ibm-granite/granite-3.1-8b-instruct": torch.float16,
        "Qwen/Qwen2-1.5B-Instruct": torch.float,
    }

    GRAPH_MODE = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"

    print(f"[longdec-attn] model={model} config={config_file} "
          f"max_tokens={max_tokens} hooks_on={hooks_on} "
          f"mode={'FULL CUDA-graph capture' if GRAPH_MODE else 'eager'} "
          f"(MIA_ALLOW_CUDAGRAPH={'1' if GRAPH_MODE else '0'})")

    llm = MiaLLM(
        model=model,
        worker_name="capture_qk",
        analyzer_name="attn_tracker",
        config_file=config_file,
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=dtype_map.get(model, torch.float16),
        enforce_eager=not GRAPH_MODE,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    test_cases = [
        {
            "instruction": "Analyze and output the sentence attitude:",
            "data": "The weather is nice today, with a soft breeze and sunshine.",
        },
    ]

    for case in test_cases:
        instruction, data = case["instruction"], case["data"]
        print("=" * 50)
        print(f"Instruction: '{instruction}'\nData: '{data}'")
        text, input_range = apply_chat_template_and_get_ranges(llm.tokenizer, model, instruction, data)

        sp = SamplingParams(temperature=0.1, max_tokens=max_tokens)
        sp.extra_args = {"hooks_on": hooks_on}

        t0 = time.time()
        output = llm.generate(text, sp, save_to_disk=True)
        t1 = time.time()
        print(f"hook llm generation runtime: {(t1 - t0):.3f}s  (decode={max_tokens}, hooks_on={hooks_on})")

        stats = llm.analyze(
            analyzer_spec={"input_range": input_range, "attn_func": "sum_normalize"})
        print(f"hook llm analysis runtime: {(time.time() - t1):.3f}s")
        print(f"Generated: {output[0].outputs[0].text[:160]!r}")
        print(f"Attention tracker score: {stats['score'][0]:.3f}")
        _print_evidence(t1 - t0, len(output[0].outputs[0].token_ids))
        llm.llm_engine.reset_prefix_cache()

