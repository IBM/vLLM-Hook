"""H-Node hallucination detection (inference only).

Runs offline with `MiaLLM`. The same demo over `vllm serve` is kept, commented out, at the end.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import urllib.error
import urllib.request

from vllm import SamplingParams

from mia import MiaLLM
from _paths import config_path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # the probe is trained on this model's layer 14
INFER_CFG = config_path("hnode_hallucination/Qwen2.5-1.5B-Instruct.infer.json")

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


def report(result):
    """Print the verdict, probability and H-Node excess per prompt."""
    print(f"Best layer: {result['best_layer']}  |  threshold: {result['threshold']}")
    print("-" * 78)
    for prompt, p, exc, verdict in zip(
        EXAMPLES, result["probabilities"], result["h_node_excess"], result["verdicts"]
    ):
        line = prompt.replace("\n", "  ")
        print(f"[{verdict:>12s}]  P(hall)={p:.3f}  H-excess={exc:.3f}  |  {line}")


def main():
    ensure_probe()
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="hnode_hallucination",
                 config_file=INFER_CFG, hook_dir="/dev/shm/mia",
                 gpu_memory_utilization=0.7, max_model_len=1024)

    print("Running detection on example prompts...\n")
    run_id = "halludetect_detect"
    llm.generate(EXAMPLES, SamplingParams(max_tokens=1, temperature=0.0),
                 save_to_disk=True, run_id=run_id)

    report(llm.analyze(
        analyzer_spec={"probe_path": PROBE_PATH, "threshold": 0.5}, run_id=run_id))


# --- Server mode ---------------------------------------------------------------------------
# The same demo against `vllm serve`. Start the server in another terminal:
#
#   VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
#       vllm serve Qwen/Qwen2.5-1.5B-Instruct \
#       --max-model-len 1024 --port 8770 --gpu-memory-utilization 0.8
#
# then uncomment serve_main() and call it instead of main() at the bottom.
#
# def serve_main():
#     from mia import MiaClient
#     from _serve import HS, require_server
#
#     ensure_probe()
#     url = require_server(MODEL, HS, max_model_len=1024)
#     client = MiaClient(base_url=url, analyzer_name="hnode_hallucination",
#                        config_file=INFER_CFG)
#
#     print("Running detection on example prompts...\n")
#     run_id = "halludetect_detect"
#     # One request for the whole set, as raw text: a run holds its last response's requests.
#     client.generate_text(EXAMPLES, model=MODEL, max_tokens=1, temperature=0.0,
#                          save_to_disk=True, run_id=run_id)
#
#     report(client.analyze(
#         analyzer_spec={"probe_path": PROBE_PATH, "threshold": 0.5}, run_id=run_id))
# --- end of server mode ---


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    # serve_main()  # server mode: see the block above
