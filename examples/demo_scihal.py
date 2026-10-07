"""Science hallucination demo: classify SciHal answers from captured hidden states.

Runs offline with `MiaLLM`. Both passes take exact token ids, and the second pass needs the
token ids the FIRST one generated, so the continuation is rebuilt from ids, never from
detokenized text (detokenize-then-retokenize is not an identity). The same demo over
`vllm serve` is kept, commented out, at the end.
"""
import json
import multiprocessing as mp
import os
import sys
from urllib.request import urlretrieve

from vllm import SamplingParams, TokensPrompt

from mia import MiaLLM
from _paths import config_path

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONFIG = config_path(f'hidden_states/{MODEL.split("/")[-1]}.json')
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
HOOK_DIR = "/dev/shm/mia"
N_TEST = 9
LABEL_NAMES = ["entailment", "contradiction", "unverifiable"]


PROMPT_TEMPLATE_PREFIX = (
    "\n"
    "You are a helpful assistant. Learn from the examples below and complete the task accordingly. \n"
    "\n"
    "### Task: Detect if the claims are well-supported by the references. Provide a justification and classify each example into three labels: entailment, contradiction, or unverifiable\n"
    "\n"
)

PROMPT_TEMPLATE_SUFFIX = (
    "\n"
    "\n"
    "### Now, apply the same pattern: \n"
    "\n"
    "Input: !INPUT!\n"
    "Output: \n"
)

SciHal_url = (
    "https://raw.githubusercontent.com/InfintyLab/SciHal-Challenge/"
    "main/data/dataset/"
)

def load_scihal_split(cache_dir: str, filename: str) -> list:
    """Download SciHal-Challenge dataset if no cache in local."""
    local = os.path.join(cache_dir, filename)
    if not os.path.exists(local):
        print(f"Downloading SciHal to {local}")
        urlretrieve(SciHal_url + filename, local)
    with open(local) as f:
        return json.load(f)


def build_few_shot_middle(train_dataset: list, count_target: int = 2, total_target: int = 6) -> str:
    """Build few-shot examples following the SciHal-Challenge reference implementation."""
    middle = ""
    count_dict = {"entailment": 0, "contradiction": 0, "unverifiable": 0}
    total = 0
    for x in train_dataset:
        label = x["label"]
        if count_dict.get(label, 0) >= count_target:
            continue
        my_input = "#Claim: " + x["claim"] + "\n #Reference: " + x["reference"]
        middle += (
            "Input: " + my_input + "\n\n"
            "Output:\n" + x["justification"] + "\n#Label: " + label + "\n\n"
        )
        count_dict[label] += 1
        total += 1
        if total == total_target:
            break
    return middle


def build_prompt_ids(tokenizer, few_shot_middle: str, claim: str, reference: str) -> list:
    """Build SciHal prompt token IDs with the authors' doubled-BOS convention."""
    my_input = "#Claim: " + claim + "\n #Reference: " + reference
    user_msg = few_shot_middle + PROMPT_TEMPLATE_SUFFIX.replace("!INPUT!", my_input)
    chat = [
        {"role": "system", "content": PROMPT_TEMPLATE_PREFIX},
        {"role": "user", "content": user_msg},
    ]
    message = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return tokenizer(message, add_special_tokens=True).input_ids


def load_prompts(tokenizer):
    """The test cases and their prompt ids."""
    train = load_scihal_split(CACHE_DIR, "subtask1_train_batch3.json")
    few_shot_middle = build_few_shot_middle(train)
    test_cases = load_scihal_split(CACHE_DIR, "subtask1_test.json")[:N_TEST]
    return test_cases, [build_prompt_ids(tokenizer, few_shot_middle, q["claim"], q["reference"])
                        for q in test_cases]


def classifier_spec():
    """The analyzer spec, or exit naming how to supply the classifier."""
    with open(CONFIG) as f:
        config_clf_path = json.load(f)["scihal"]["clf_path"]
    clf_path = os.environ.get("MIA_SCIHAL_CLF", config_clf_path)
    if not os.path.isfile(clf_path):
        print(
            f"[demo_scihal] SciHal classifier not found: {clf_path}\n"
            f"  Train one on the SciHal-Challenge data "
            f"(https://github.com/InfintyLab/SciHal-Challenge; recipe in examples/README.md), "
            f"then point this demo at it with either:\n"
            f"    export MIA_SCIHAL_CLF=/path/to/your_classifier.joblib\n"
            f"  or \"scihal.clf_path\" in {CONFIG}."
        )
        sys.exit(1)
    return {"label_names": LABEL_NAMES, "clf_path": clf_path, "model_id": MODEL}


def print_labels(test_cases, stats):
    """Print the classifier's label for each test case."""
    print("=" * 50)
    for case, label in zip(test_cases, stats["prediction_labels"]):
        print(f"classifier label: {label}")


def main():
    spec = classifier_spec()
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="science_hallucination",
                 config_file=CONFIG, hook_dir=HOOK_DIR,
                 gpu_memory_utilization=0.7, max_model_len=8192)
    test_cases, prompt_ids_list = load_prompts(llm.tokenizer)

    # Pass 1: plain generation, nothing armed.
    gen = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list],
                       SamplingParams(max_tokens=1024, temperature=0.0), use_hook=False)
    response_token_ids = [list(o.outputs[0].token_ids) for o in gen]

    # Pass 2: re-prompt on prompt+response and capture the hidden states at that point.
    capture_prompts = [TokensPrompt(prompt_token_ids=list(p) + list(r[:-2]))
                       for p, r in zip(prompt_ids_list, response_token_ids)]
    llm.generate(capture_prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 save_to_disk=True)

    print_labels(test_cases, llm.analyze(analyzer_spec=spec))


# --- Server mode ---------------------------------------------------------------------------
# The same demo against `vllm serve`. Start the server in another terminal:
#
#   VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
#       vllm serve meta-llama/Llama-3.1-8B-Instruct \
#       --max-model-len 8192 --port 8770 --gpu-memory-utilization 0.8
#
# then uncomment serve_main() and call it instead of main() at the bottom. Both passes go
# through /v1/completions with exact token ids; `return_token_ids` is how the server reports
# the ids the first pass generated.
#
# def serve_main():
#     from mia import MiaClient
#     from _serve import HS, require_server
#
#     url = require_server(MODEL, HS, max_model_len=8192)
#     spec = classifier_spec()
#     client = MiaClient(base_url=url, analyzer_name="science_hallucination",
#                        config_file=CONFIG, hook_dir=HOOK_DIR, tokenizer_for=MODEL)
#     test_cases, prompt_ids_list = load_prompts(client.tokenizer)
#
#     # Pass 1: plain generation, nothing armed (`capture=False` == offline use_hook=False).
#     gen = client.generate_tokens(
#         prompt_ids_list, model=MODEL, max_tokens=1024, temperature=0.0,
#         capture=False, extra_body={"return_token_ids": True},
#     )
#     by_index = sorted(gen.choices, key=lambda c: c.index)
#     response_token_ids = [list(c.token_ids or []) for c in by_index]
#     if len(response_token_ids) != len(prompt_ids_list) or not all(response_token_ids):
#         raise RuntimeError(
#             "the server returned no generated token ids; pass return_token_ids and check "
#             "this is vLLM 0.29, where completion choices carry `token_ids`.")
#
#     # Pass 2: re-prompt on prompt+response and capture the hidden states at that point.
#     capture_prompts = [list(p) + list(r[:-2])
#                        for p, r in zip(prompt_ids_list, response_token_ids)]
#     client.generate_tokens(capture_prompts, model=MODEL, max_tokens=1, temperature=0.0,
#                            save_to_disk=True)
#
#     print_labels(test_cases, client.analyze(analyzer_spec=spec))
# --- end of server mode ---


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    # serve_main()  # server mode: see the block above
