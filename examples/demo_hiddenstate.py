"""Hidden-state capture: capture layer activations and read them back.

Runs offline with `MiaLLM`. The same demo over `vllm serve` is kept, commented out, at the end.
"""
import multiprocessing as mp
import os
import time

import torch
from vllm import SamplingParams

from mia import MiaLLM
from _paths import config_path

MODEL = os.environ.get("MIA_DEMO_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CONFIG = os.environ.get(
    "MIA_CONFIG_FILE", config_path(f"hidden_states/{MODEL.split('/')[-1]}.json"))

PROMPTS = [
    "The capital of France is",
    "Quantum computing leverages",
]


def chat_text(tokenizer, prompt: str) -> str:
    """The prompt under the model's chat template, as the server's chat endpoint renders it."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)


def main():
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="hidden_states",
                 config_file=CONFIG, hook_dir="/dev/shm/mia",
                 gpu_memory_utilization=0.7, max_model_len=2048)
    sp = SamplingParams(max_tokens=10, temperature=0.0)

    print("=" * 50)
    for prompt in PROMPTS:
        t0 = time.time()
        out = llm.generate(chat_text(llm.tokenizer, prompt), sp)
        elapsed = time.time() - t0
        stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out[0].probes)

        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{out[0].outputs[0].text.strip()}'")
        for layer_name, tensors in sorted(stats["hidden_states"].items()):
            t = tensors[0]
            print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")
        print(f"[evidence] {elapsed * 1000:.1f} ms for {len(out[0].outputs[0].token_ids)} tokens")

    print("=" * 50)
    print("Reducing to a norm per layer instead of the raw tensor...")
    llm.llm_engine.reset_prefix_cache()
    t0 = time.time()
    out = llm.generate(chat_text(llm.tokenizer, PROMPTS[0]), sp)
    elapsed = time.time() - t0
    stats = llm.analyze(analyzer_spec={"reduce": "norm"}, probes=out[0].probes)

    print(f"\nPrompt: '{PROMPTS[0]}'")
    for layer_name, norms in sorted(stats["hidden_states"].items()):
        print(f"  {layer_name}: norm={norms[0]:.4f}")
    print(f"[evidence] {elapsed * 1000:.1f} ms for {len(out[0].outputs[0].token_ids)} tokens")


# --- Server mode ---------------------------------------------------------------------------
# The same demo against `vllm serve`. Start the server in another terminal:
#
#   VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
#       vllm serve Qwen/Qwen2.5-3B-Instruct \
#       --max-model-len 2048 --port 8770 --gpu-memory-utilization 0.8
#
# then uncomment serve_main() and call it instead of main() at the bottom.
#
# def serve_main():
#     from mia import MiaClient
#     from _serve import (HS, chat, completion_text, completion_tokens, print_evidence,
#                         require_server)
#
#     url = require_server(MODEL, HS, model_env=True)
#     client = MiaClient(base_url=url, analyzer_name="hidden_states", config_file=CONFIG)
#
#     print("=" * 50)
#     for prompt in PROMPTS:
#         t0 = time.time()
#         response = client.generate(messages=chat(prompt), model=MODEL, max_tokens=10,
#                                    temperature=0.0)
#         elapsed = time.time() - t0
#         stats = client.analyze(analyzer_spec={"reduce": "none"})
#
#         print(f"\nPrompt: '{prompt}'")
#         print(f"Generated: '{completion_text(response).strip()}'")
#         for layer_name, tensors in sorted(stats["hidden_states"].items()):
#             t = tensors[0]
#             print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")
#         print_evidence(elapsed, completion_tokens(response))
#
#     print("=" * 50)
#     print("Reducing to a norm per layer instead of the raw tensor...")
#     t0 = time.time()
#     response = client.generate(messages=chat(PROMPTS[0]), model=MODEL, max_tokens=10,
#                                temperature=0.0)
#     elapsed = time.time() - t0
#     stats = client.analyze(analyzer_spec={"reduce": "norm"})
#
#     print(f"\nPrompt: '{PROMPTS[0]}'")
#     for layer_name, norms in sorted(stats["hidden_states"].items()):
#         print(f"  {layer_name}: norm={norms[0]:.4f}")
#     print_evidence(elapsed, completion_tokens(response))
# --- end of server mode ---


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    # serve_main()  # server mode: see the block above
