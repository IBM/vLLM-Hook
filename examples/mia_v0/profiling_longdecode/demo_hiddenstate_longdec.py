"""Long-decode hidden-state capture over `vllm serve`, capturing on every decode step."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mia import MiaClient                                             # noqa: E402
from _paths import config_path                                        # noqa: E402
from _serve import (HS, chat, completion_text, completion_tokens,     # noqa: E402
                    print_evidence, require_server)

MODEL = os.environ.get("MIA_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
CONFIG = os.environ.get(
    "MIA_CONFIG_FILE", config_path(f"hidden_states/{MODEL.split('/')[-1]}.json"))
MAX_TOKENS = int(os.environ.get("MIA_DEMO_MAX_TOKENS", "128"))
HOOKS_ON = os.environ.get("MIA_DEMO_HOOKS_ON", "both")
GRAPH = os.environ.get("MIA_ALLOW_CUDAGRAPH", "1") != "0"

if __name__ == "__main__":
    print(f"[longdec-hs] model={MODEL} config={CONFIG} "
          f"max_tokens={MAX_TOKENS} hooks_on={HOOKS_ON}")
    url = require_server(MODEL, HS, graph=GRAPH)
    client = MiaClient(base_url=url, analyzer_name="hidden_states", config_file=CONFIG)

    print("=" * 50)
    for prompt in ["The capital of France is"]:
        t0 = time.time()
        response = client.generate(messages=chat(prompt), model=MODEL,
                                   max_tokens=MAX_TOKENS, temperature=0.0,
                                   save_to_disk=True,
                                   extra_xargs={"hooks_on": HOOKS_ON})
        elapsed = time.time() - t0
        stats = client.analyze(analyzer_spec={"reduce": "norm"})

        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{completion_text(response).strip()[:120]}...'")
        for layer_name, norms in sorted(stats["hidden_states"].items()):
            print(f"  {layer_name}: norm={norms[0]:.4f}")
        print_evidence(elapsed, completion_tokens(response), "longdec-hs")
