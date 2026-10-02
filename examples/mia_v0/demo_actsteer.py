"""Activation steering over `vllm serve`: the same prompt, steered and unsteered.

Steering produces no artifact, so there is nothing to analyze and nothing for `MiaClient`
to do — a plain OpenAI client is enough. Each request carries its own steer config under
``vllm_xargs["steer"]``, JSON-encoded because `vllm_xargs` only accepts scalars.
"""
import json
import os
import time

import openai

from _paths import config_path
from _serve import STEER, base_url, print_evidence, require_server

MODEL = os.environ.get("MIA_DEMO_MODEL", "microsoft/Phi-3-mini-4k-instruct")
CONFIG = os.environ.get(
    "MIA_CONFIG_FILE", config_path(f'activation_steer/{MODEL.split("/")[-1]}.json'))
GRAPH = os.environ.get("MIA_ALLOW_CUDAGRAPH") == "1"

PROMPTS = [
    "Write a dialogue between two people, one is dressed up in a ball gown and the other "
    "is dressed down in sweats. The two are going to a nightly event. Your answer must "
    "contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each "
    "bullet) such as:\n* This is the first point.\n* This is the second point.",
    "What is the difference between the 13 colonies and the other British colonies in "
    "North America? Your answer must contain exactly 6 bullet point in Markdown using the "
    "following format:\n* Bullet point one.\n* Bullet point two.\n...\n* Bullet point fix.",
]

if __name__ == "__main__":
    require_server(MODEL, STEER, graph=GRAPH)
    client = openai.OpenAI(base_url=base_url(), api_key="EMPTY")

    with open(CONFIG) as f:
        base_steer = json.load(f)["steering"]
    steer = {**base_steer, "method": "add_vector", "coefficient": 10}

    for prompt in PROMPTS:
        for label, extra_body in (
            ("unsteered", None),
            ("steered", {"vllm_xargs": {"steer": json.dumps(steer)}}),
        ):
            print("=" * 50)
            print(f"[{label}] {prompt[:70]}...")
            t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
                extra_body=extra_body,
            )
            elapsed = time.time() - t0
            print(response.choices[0].message.content)
            print_evidence(elapsed, response.usage.completion_tokens, label)
