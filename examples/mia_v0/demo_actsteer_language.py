"""Language steering over `vllm serve`: the same prompt answered in another language.

Each request carries its own steer config under ``vllm_xargs["steer"]``, so one server
serves the unsteered answer and both steered ones without restarting.
"""
import json
import os
import time

import openai

from _paths import config_path
from _serve import STEER, base_url, print_evidence, require_server

MODEL = os.environ.get("MIA_DEMO_MODEL", "microsoft/Phi-3-mini-4k-instruct")
GRAPH = os.environ.get("MIA_ALLOW_CUDAGRAPH", "1") != "0"

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

if __name__ == "__main__":
    require_server(MODEL, STEER, graph=GRAPH)
    client = openai.OpenAI(base_url=base_url(), api_key="EMPTY")

    steer_by_language = {}
    for language, rel in LANGUAGE_CONFIGS.items():
        with open(config_path(rel)) as f:
            steer_by_language[language] = json.load(f)["steering"]

    for prompt in PROMPTS:
        print("=" * 50)
        print(f"Original prompt: {prompt}")

        for label, steer in [("unsteered", None)] + list(steer_by_language.items()):
            extra_body = (None if steer is None
                          else {"vllm_xargs": {"steer": json.dumps(steer)}})
            t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
                extra_body=extra_body,
            )
            elapsed = time.time() - t0
            print(f"\n[{label}]")
            print(response.choices[0].message.content)
            print_evidence(elapsed, response.usage.completion_tokens, label)
