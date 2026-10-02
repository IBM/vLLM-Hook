"""H-Node hallucination detection over `vllm serve` (inference only)."""
from __future__ import annotations

import os
import sys

from mia import MiaClient
from _paths import config_path
from _serve import HS, chat, require_server

MODEL = os.environ.get("MIA_DEMO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
INFER_CFG = config_path("hnode_hallucination/Qwen2.5-1.5B-Instruct.infer.json")
GRAPH = os.environ.get("MIA_ALLOW_CUDAGRAPH", "1") != "0"

PROBE_BASE_URL = (
    "https://raw.githubusercontent.com/Samarpit-bhatia/hnode-probe-builder/"
    "master/artifacts"
)
ART_DIR = "./cache/hnode_probe"
PROBE_PATH = os.path.join(ART_DIR, "probe.npz")

EXAMPLES = [
    "Q: What is the capital of France?\nA: Paris",
    "Q: What is the capital of France?\nA: London",
    "Q: Who wrote Hamlet?\nA: William Shakespeare",
    "Q: Who wrote Hamlet?\nA: Charles Dickens",
    "Q: What is 2 + 2?\nA: 4",
    "Q: What is 2 + 2?\nA: 5",
]


def ensure_probe():
    """Download probe.npz + probe.json into ART_DIR if not already cached."""
    import urllib.error
    import urllib.request

    os.makedirs(ART_DIR, exist_ok=True)
    for name in ("probe.npz", "probe.json"):
        dest = os.path.join(ART_DIR, name)
        if os.path.exists(dest):
            continue
        url = f"{PROBE_BASE_URL}/{name}"
        print(f"Downloading {name} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as exc:
            sys.exit(
                f"Could not download {name} ({exc}).\n"
                f"Download it manually from {url}\n"
                f"and place it in {ART_DIR}/."
            )


if __name__ == "__main__":
    ensure_probe()
    url = require_server(MODEL, HS, graph=GRAPH, max_model_len=1024)
    client = MiaClient(base_url=url, analyzer_name="hnode_hallucination",
                       config_file=INFER_CFG)

    print("Running detection on example prompts...\n")
    run_id = "halludetect_detect"
    for prompt in EXAMPLES:
        # One run_id for the whole set: the analyzer reduces over every artifact in it.
        client.generate(messages=chat(prompt), model=MODEL, max_tokens=1,
                        temperature=0.0, save_to_disk=True, run_id=run_id)

    result = client.analyze(
        analyzer_spec={"probe_path": PROBE_PATH, "threshold": 0.5}, run_id=run_id)

    print(f"Best layer: {result['best_layer']}  |  threshold: {result['threshold']}")
    print("-" * 78)
    for prompt, p, exc, verdict in zip(
        EXAMPLES, result["probabilities"], result["h_node_excess"], result["verdicts"]
    ):
        line = prompt.replace("\n", "  ")
        print(f"[{verdict:>12s}]  P(hall)={p:.3f}  H-excess={exc:.3f}  |  {line}")
