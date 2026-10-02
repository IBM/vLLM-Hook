"""Hidden-state capture over `vllm serve`: capture layer activations and read them back."""
import os
import time

import torch

from mia import MiaClient
from _paths import config_path
from _serve import (HS, chat, completion_text, completion_tokens, print_evidence,
                    require_server)

MODEL = os.environ.get("MIA_DEMO_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CONFIG = os.environ.get(
    "MIA_CONFIG_FILE", config_path(f"hidden_states/{MODEL.split('/')[-1]}.json"))
GRAPH = os.environ.get("MIA_ALLOW_CUDAGRAPH", "1") != "0"

PROMPTS = [
    "The capital of France is",
    "Quantum computing leverages",
]

if __name__ == "__main__":
    url = require_server(MODEL, HS, graph=GRAPH)
    client = MiaClient(base_url=url, analyzer_name="hidden_states", config_file=CONFIG)

    print("=" * 50)
    for prompt in PROMPTS:
        t0 = time.time()
        response = client.generate(messages=chat(prompt), model=MODEL, max_tokens=10,
                                   temperature=0.0, save_to_disk=True)
        elapsed = time.time() - t0
        stats = client.analyze(analyzer_spec={"reduce": "none"})

        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{completion_text(response).strip()}'")
        for layer_name, tensors in sorted(stats["hidden_states"].items()):
            t = tensors[0]
            print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")
        print_evidence(elapsed, completion_tokens(response))

    print("=" * 50)
    print("Reducing to a norm per layer instead of the raw tensor...")
    t0 = time.time()
    response = client.generate(messages=chat(PROMPTS[0]), model=MODEL, max_tokens=10,
                               temperature=0.0, save_to_disk=True)
    elapsed = time.time() - t0
    stats = client.analyze(analyzer_spec={"reduce": "norm"})

    print(f"\nPrompt: '{PROMPTS[0]}'")
    for layer_name, norms in sorted(stats["hidden_states"].items()):
        print(f"  {layer_name}: norm={norms[0]:.4f}")
    print_evidence(elapsed, completion_tokens(response))
