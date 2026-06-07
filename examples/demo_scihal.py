import json
import os
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")

from vllm import SamplingParams, TokensPrompt
from vllm_hook_plugins import HookLLM

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
    """Build few shot examples following https://github.com/InfintyLab/SciHal-Challenge/blob/97817c788bde330c61f8ed2fc750d83b19fa1590/llama3_finetunec3_just_task1.py#L141."""
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
    """Build SciHal prompt token IDs following the original authors' doubled-BOS practice: apply_chat_template returns a string already prefixed
    with <|begin_of_text|>, then tokenizer() with add_special_tokens=True prepends ANOTHER <|begin_of_text|>. The classifier was trained on hidden
    states extracted from this prefix."""
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
    hook_dir = "/dev/shm/vllm_hook" # None #
    model = "meta-llama/Llama-3.1-8B-Instruct"
    n_test = 9 # number of test samples

    dtype_map = {
        'meta-llama/Llama-3.1-8B-Instruct': torch.float16
    }

    llm = HookLLM(
        model=model,
        worker_name="probe_hidden_states",
        analyzer_name="science_hallucination",
        config_file=f'model_configs/hidden_states/{model.split("/")[-1]}.json',
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        trust_remote_code=True,
        dtype=dtype_map[model],
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1  # the number of gpus
    )

    tokenizer = llm.tokenizer
    train = load_scihal_split(cache_dir, "subtask1_train_batch3.json")
    few_shot_middle = build_few_shot_middle(train)
    test_cases = load_scihal_split(cache_dir, "subtask1_test.json")[:n_test]
    prompt_ids_list = [build_prompt_ids(tokenizer, few_shot_middle, q["claim"], q["reference"]) for q in test_cases]

    # normal generation pass 
    gen_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list],
        SamplingParams(temperature=0.0, max_tokens=1024),
        use_hook=False,
    )
    response_token_ids = [list(gen_output.outputs[0].token_ids) for gen_output in gen_outputs]

    # re-prefill (prompt_ids + response_ids[:-2]) and capture last-token hidden state
    # following https://github.com/InfintyLab/SciHal-Challenge/blob/97817c788bde330c61f8ed2fc750d83b19fa1590/llama3_finetunec3_just_task1.py#L262
    capture_prompts = [
        TokensPrompt(prompt_token_ids=list(p) + list(r[:-2]))
        for p, r in zip(prompt_ids_list, response_token_ids)
    ]
    output = llm.generate(capture_prompts, SamplingParams(temperature=0.0, max_tokens=1), save_to_disk=True)

    # classify with the science_hallucination analyzer
    config_file = f"model_configs/hidden_states/{model.split('/')[-1]}.json"
    with open(config_file) as f:
        clf_path = json.load(f)["scihal"]["clf_path"]
    LABEL_NAMES = ["entailment", "contradiction", "unverifiable"]
    spec = {
        "label_names": LABEL_NAMES,
        "clf_path": clf_path,
        "model_id": model,
    }
    stats = llm.analyze(probes=getattr(output[0], "probes", None), analyzer_spec=spec)
    
    labels = stats["prediction_labels"]
    print("=" * 50)
    for case, label in zip(test_cases, labels):
        print(f"classifier label: {label}")
