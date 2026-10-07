"""Activation steering: the same prompt, steered and unsteered.

Runs offline with `MiaLLM`. Each steered request carries its steer config in
``SamplingParams.extra_args["steer"]``. The same demo over `vllm serve` is kept, commented out,
at the end.
"""
import json
import multiprocessing as mp
import os
import time

from vllm import SamplingParams

from mia import MiaLLM
from _paths import config_path

MODEL = os.environ.get("MIA_DEMO_MODEL", "microsoft/Phi-3-mini-4k-instruct")
CONFIG = os.environ.get(
    "MIA_CONFIG_FILE", config_path(f'activation_steer/{MODEL.split("/")[-1]}.json'))

PROMPTS = [
    "Write a dialogue between two people, one is dressed up in a ball gown and the other "
    "is dressed down in sweats. The two are going to a nightly event. Your answer must "
    "contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each "
    "bullet) such as:\n* This is the first point.\n* This is the second point.",
    "What is the difference between the 13 colonies and the other British colonies in "
    "North America? Your answer must contain exactly 6 bullet point in Markdown using the "
    "following format:\n* Bullet point one.\n* Bullet point two.\n...\n* Bullet point fix.",
]


def steer_config():
    """The config file's steering section, applied as add_vector with coefficient 10."""
    with open(CONFIG) as f:
        base_steer = json.load(f)["steering"]
    return {**base_steer, "method": "add_vector", "coefficient": 10}


def main():
    # Phi-3-mini-4k has a 4096-token context; 2048 output tokens do not fit under 2048.
    llm = MiaLLM(model=MODEL, worker_name="steer", config_file=CONFIG,
                 gpu_memory_utilization=0.7, max_model_len=4096)
    steer = steer_config()

    for prompt in PROMPTS:
        text = llm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True)
        for label, steered in (("unsteered", False), ("steered", True)):
            print("=" * 50)
            print(f"[{label}] {prompt[:70]}...")
            sp = SamplingParams(max_tokens=2048, temperature=0.0,
                                extra_args={"steer": steer} if steered else None)
            t0 = time.time()
            out = llm.generate(text, sp, use_hook=steered)
            elapsed = time.time() - t0
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
# then uncomment serve_main() and call it instead of main() at the bottom. Steering produces
# no artifact, so a plain OpenAI client is enough; the steer config goes in
# ``vllm_xargs["steer"]``, JSON-encoded because `vllm_xargs` only accepts scalars.
#
# def serve_main():
#     import openai
#
#     from _serve import STEER, base_url, print_evidence, require_server
#
#     require_server(MODEL, STEER, max_model_len=4096, model_env=True)
#     client = openai.OpenAI(base_url=base_url(), api_key="EMPTY")
#     steer = steer_config()
#
#     for prompt in PROMPTS:
#         for label, extra_body in (
#             ("unsteered", None),
#             ("steered", {"vllm_xargs": {"steer": json.dumps(steer)}}),
#         ):
#             print("=" * 50)
#             print(f"[{label}] {prompt[:70]}...")
#             t0 = time.time()
#             response = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=2048,
#                 temperature=0.0,
#                 extra_body=extra_body,
#             )
#             elapsed = time.time() - t0
#             print(response.choices[0].message.content)
#             print_evidence(elapsed, response.usage.completion_tokens, label)
# --- end of server mode ---


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    # serve_main()  # server mode: see the block above
