"""Science hallucination demo: classify SciHal answers from captured hidden states.
Runs over `vllm serve`. Both passes go through /v1/completions with exact token ids, and
the second pass needs the token ids the FIRST one generated -- `return_token_ids` is how
the server reports them, so the continuation is rebuilt from ids, never from detokenized
text (detokenize-then-retokenize is not an identity).
"""
import json
import os
import sys
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("MIA_USE_SAFETENSORS", "1")

from mia import MiaClient
from _serve import HS, require_server
from _paths import config_path

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
        from urllib.request import urlretrieve
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


if __name__ == "__main__":
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    hook_dir = "/dev/shm/mia"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    n_test = 9

    url = require_server(model, HS, max_model_len=8192)
    client = MiaClient(
        base_url=url,
        analyzer_name="science_hallucination",
        config_file=config_path(f'hidden_states/{model.split("/")[-1]}.json'),
        hook_dir=hook_dir,
        tokenizer_for=model,
    )

    tokenizer = client.tokenizer
    train = load_scihal_split(cache_dir, "subtask1_train_batch3.json")
    few_shot_middle = build_few_shot_middle(train)
    test_cases = load_scihal_split(cache_dir, "subtask1_test.json")[:n_test]
    prompt_ids_list = [build_prompt_ids(tokenizer, few_shot_middle, q["claim"], q["reference"]) for q in test_cases]

    # Pass 1: plain generation, nothing armed (`capture=False` == offline use_hook=False).
    gen = client.generate_tokens(
        prompt_ids_list, model=model, max_tokens=1024, temperature=0.0,
        capture=False, extra_body={"return_token_ids": True},
    )
    by_index = sorted(gen.choices, key=lambda c: c.index)
    response_token_ids = [list(c.token_ids or []) for c in by_index]
    if len(response_token_ids) != len(prompt_ids_list) or not all(response_token_ids):
        raise RuntimeError(
            "the server returned no generated token ids; pass return_token_ids and check "
            "this is vLLM 0.29, where completion choices carry `token_ids`.")

    # Pass 2: re-prompt on prompt+response and capture the hidden states at that point.
    capture_prompts = [list(p) + list(r[:-2])
                       for p, r in zip(prompt_ids_list, response_token_ids)]
    client.generate_tokens(capture_prompts, model=model, max_tokens=1, temperature=0.0,
                           save_to_disk=True)

    config_file = config_path(f"hidden_states/{model.split('/')[-1]}.json")
    with open(config_file) as f:
        config_clf_path = json.load(f)["scihal"]["clf_path"]
    clf_path = os.environ.get("MIA_SCIHAL_CLF", config_clf_path)
    if not os.path.isfile(clf_path):
        print(
            f"[demo_scihal] SciHal classifier not found: {clf_path}\n"
            f"  This joblib file is produced by the SciHal-Challenge repo linked "
            f"above (https://github.com/InfintyLab/SciHal-Challenge) -- train/export "
            f"a classifier there, then point this demo at it by either:\n"
            f"    export MIA_SCIHAL_CLF=/path/to/your_classifier.joblib\n"
            f"  or updating \"scihal.clf_path\" in {config_file}."
        )
        sys.exit(1)
    LABEL_NAMES = ["entailment", "contradiction", "unverifiable"]
    spec = {
        "label_names": LABEL_NAMES,
        "clf_path": clf_path,
        "model_id": model,
    }
    stats = client.analyze(analyzer_spec=spec)

    labels = stats["prediction_labels"]
    print("=" * 50)
    for case, label in zip(test_cases, labels):
        print(f"classifier label: {label}")

