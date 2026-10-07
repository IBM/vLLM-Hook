"""Long-decode hidden-state capture, capturing on every decode step.

Runs offline with `MiaLLM`. The same demo over `vllm serve` is kept, commented out, at the end.
"""
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from vllm import SamplingParams

from mia import MiaLLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))  # _paths and _serve
from _paths import config_path  # noqa: E402

MODEL = os.environ.get("MIA_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
CONFIG = os.environ.get(
    "MIA_CONFIG_FILE", config_path(f"hidden_states/{MODEL.split('/')[-1]}.json"))
MAX_TOKENS = int(os.environ.get("MIA_DEMO_MAX_TOKENS", "128"))
HOOKS_ON = os.environ.get("MIA_DEMO_HOOKS_ON", "both")


def main():
    print(f"[longdec-hs] model={MODEL} config={CONFIG} "
          f"max_tokens={MAX_TOKENS} hooks_on={HOOKS_ON}")
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="hidden_states",
                 config_file=CONFIG, hook_dir="/dev/shm/mia",
                 gpu_memory_utilization=0.7, max_model_len=2048)
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0,
                        extra_args={"hooks_on": HOOKS_ON})

    print("=" * 50)
    for prompt in ["The capital of France is"]:
        text = llm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        out = llm.generate(text, sp, save_to_disk=True)
        elapsed = time.time() - t0
        stats = llm.analyze(analyzer_spec={"reduce": "norm"})

        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{out[0].outputs[0].text.strip()[:120]}...'")
        for layer_name, norms in sorted(stats["hidden_states"].items()):
            print(f"  {layer_name}: norm={norms[0]:.4f}")
        print(f"[evidence:longdec-hs] {elapsed * 1000:.1f} ms for "
              f"{len(out[0].outputs[0].token_ids)} tokens")


# --- Server mode ---------------------------------------------------------------------------
# The same demo against `vllm serve`. Start the server in another terminal:
#
#   VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
#       vllm serve ibm-granite/granite-3.1-8b-instruct \
#       --max-model-len 2048 --port 8770 --gpu-memory-utilization 0.8
#
# then uncomment serve_main() and call it instead of main() at the bottom.
#
# def serve_main():
#     from mia import MiaClient
#     from _serve import (HS, chat, completion_text, completion_tokens, print_evidence,
#                         require_server)
#
#     print(f"[longdec-hs] model={MODEL} config={CONFIG} "
#           f"max_tokens={MAX_TOKENS} hooks_on={HOOKS_ON}")
#     url = require_server(MODEL, HS, model_env=True)
#     client = MiaClient(base_url=url, analyzer_name="hidden_states", config_file=CONFIG)
#
#     print("=" * 50)
#     for prompt in ["The capital of France is"]:
#         t0 = time.time()
#         response = client.generate(messages=chat(prompt), model=MODEL,
#                                    max_tokens=MAX_TOKENS, temperature=0.0,
#                                    save_to_disk=True,
#                                    extra_xargs={"hooks_on": HOOKS_ON})
#         elapsed = time.time() - t0
#         stats = client.analyze(analyzer_spec={"reduce": "norm"})
#
#         print(f"\nPrompt: '{prompt}'")
#         print(f"Generated: '{completion_text(response).strip()[:120]}...'")
#         for layer_name, norms in sorted(stats["hidden_states"].items()):
#             print(f"  {layer_name}: norm={norms[0]:.4f}")
#         print_evidence(elapsed, completion_tokens(response), "longdec-hs")
# --- end of server mode ---


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    # serve_main()  # server mode: see the block above
