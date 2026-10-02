"""Science hallucination demo: classify SciHal answers from captured hidden states.
Runs in-process (`MiaLLM`) rather than over `vllm serve`: it prompts with exact token
ids and checks the span alignment it depends on. The chat endpoint applies the model's
chat template server-side, which re-tokenizes and would invalidate those spans. Serving
it would need a pass-through chat template on the server.
"""
import json
import os
import sys
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("MIA_USE_SAFETENSORS", "1")

from vllm import SamplingParams, TokensPrompt
from mia import MiaLLM
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

    dtype_map = {
        'meta-llama/Llama-3.1-8B-Instruct': torch.float16
    }

    llm = MiaLLM(
        model=model,
        worker_name="capture_hs",
        analyzer_name="science_hallucination",
        config_file=config_path(f'hidden_states/{model.split("/")[-1]}.json'),
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        trust_remote_code=True,
        dtype=dtype_map[model],
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1
    )

    tokenizer = llm.tokenizer
    train = load_scihal_split(cache_dir, "subtask1_train_batch3.json")
    few_shot_middle = build_few_shot_middle(train)
    test_cases = load_scihal_split(cache_dir, "subtask1_test.json")[:n_test]
    prompt_ids_list = [build_prompt_ids(tokenizer, few_shot_middle, q["claim"], q["reference"]) for q in test_cases]

    gen_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list],
        SamplingParams(temperature=0.0, max_tokens=1024),
        use_hook=False,
    )
    response_token_ids = [list(gen_output.outputs[0].token_ids) for gen_output in gen_outputs]

    capture_prompts = [
        TokensPrompt(prompt_token_ids=list(p) + list(r[:-2]))
        for p, r in zip(prompt_ids_list, response_token_ids)
    ]
    output = llm.generate(capture_prompts, SamplingParams(temperature=0.0, max_tokens=1), save_to_disk=True)

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
    stats = llm.analyze(analyzer_spec=spec)

    labels = stats["prediction_labels"]
    print("=" * 50)
    for case, label in zip(test_cases, labels):
        print(f"classifier label: {label}")

