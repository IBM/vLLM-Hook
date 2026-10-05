"""Language steering: the same prompt answered in another language.

Runs offline with `MiaLLM`. Each request carries its own steer config in
``SamplingParams.extra_args["steer"]``, so one engine serves the unsteered answer and both steered
ones. The same demo over `vllm serve` is kept, commented out, at the end.
"""
import json
import multiprocessing as mp
import os
import time

from vllm import SamplingParams

from mia import MiaLLM
from _paths import config_path

MODEL = "microsoft/Phi-3-mini-4k-instruct"   # the language vectors are Phi-3's

LANGUAGE_CONFIGS = {
    "Chinese": "activation_steer/Phi-3-mini-4k-instruct-chinese.json",
    "Korean": "activation_steer/Phi-3-mini-4k-instruct-korean.json",
}

PROMPTS = [
    "If a tree is on the top of a mountain and the mountain is far from the see then is "
    "the tree close to the sea?",
    "What is the difference between HTML and JavaScript?",
    "Why might someone prefer to shop at a small, locally-owned business instead of a "
    "large chain store, even if the prices are higher?",
    "What's the permission that allows creating provisioning profiles in Apple Developer "
    "account is called?",
]


def steer_by_language():
    """Each language's steering section, from its config file."""
    out = {}
    for language, rel in LANGUAGE_CONFIGS.items():
        with open(config_path(rel)) as f:
            out[language] = json.load(f)["steering"]
    return out


def main():
    # Phi-3-mini-4k has a 4096-token context; 2048 output tokens do not fit under 2048.
    llm = MiaLLM(model=MODEL, worker_name="steer", gpu_memory_utilization=0.7,
                 max_model_len=4096)
    steers = steer_by_language()

    for prompt in PROMPTS:
        print("=" * 50)
        print(f"Original prompt: {prompt}")
        text = llm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True)

        for label, steer in [("unsteered", None)] + list(steers.items()):
            sp = SamplingParams(max_tokens=2048, temperature=0.0,
                                extra_args=None if steer is None else {"steer": steer})
            t0 = time.time()
            out = llm.generate(text, sp, use_hook=steer is not None)
            elapsed = time.time() - t0
            print(f"\n[{label}]")
            print(out[0].outputs[0].text)
            print(f"[evidence:{label}] {elapsed * 1000:.1f} ms for "
                  f"{len(out[0].outputs[0].token_ids)} tokens")


# --- Server mode ---------------------------------------------------------------------------
# The same demo against `vllm serve`. Start the server in another terminal:
#
#   VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=steer \
#       vllm serve microsoft/Phi-3-mini-4k-instruct \
#       --max-model-len 4096 --port 8770
#
# then uncomment serve_main() and call it instead of main() at the bottom. Each request carries
# its steer config in ``vllm_xargs["steer"]``, JSON-encoded because `vllm_xargs` only accepts
# scalars.
#
# def serve_main():
#     import openai
#
#     from _serve import STEER, base_url, print_evidence, require_server
#
#     require_server(MODEL, STEER, max_model_len=4096)
#     client = openai.OpenAI(base_url=base_url(), api_key="EMPTY")
#     steers = steer_by_language()
#
#     for prompt in PROMPTS:
#         print("=" * 50)
#         print(f"Original prompt: {prompt}")
#
#         for label, steer in [("unsteered", None)] + list(steers.items()):
#             extra_body = (None if steer is None
#                           else {"vllm_xargs": {"steer": json.dumps(steer)}})
#             t0 = time.time()
#             response = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=2048,
#                 temperature=0.0,
#                 extra_body=extra_body,
#             )
#             elapsed = time.time() - t0
#             print(f"\n[{label}]")
#             print(response.choices[0].message.content)
#             print_evidence(elapsed, response.usage.completion_tokens, label)
# --- end of server mode ---


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    # serve_main()  # server mode: see the block above
