import os
import sys
import json
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_hook_plugins import HookLLM
from vllm import SamplingParams

if __name__ == "__main__":

    cache_dir = "./cache/"
    model = 'microsoft/Phi-3-mini-4k-instruct'
    
    dtype_map = {
        'microsoft/Phi-3-mini-4k-instruct': 'auto',
        'mistralai/Mistral-7B-Instruct-v0.3': torch.float16,
        'ibm-granite/granite-3.1-8b-instruct': torch.float16,
        'Qwen/Qwen2-1.5B-Instruct': torch.float
    }

    llm = HookLLM(
        model=model,
        worker_name="steer_hook_act",
        config_file=f'model_configs/activation_steer/{model.split("/")[-1]}.json',
        download_dir=cache_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=dtype_map[model],
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_hook=True, 
        tensor_parallel_size=1  # the number of gpus
    )
    
    test_cases = [
        "If a tree is on the top of a mountain and the mountain is far from the see then is the tree close to the sea?",
        "Create a short, concise summary of the paper based on its abstract.\n\nFew-shot learning (FSL) is one of the key future steps in machine learning and raises a lot of attention. In this paper, we focus on the FSL problem of dialogue understanding, which contains two closely related tasks: intent detection and slot filling. Dialogue understanding has been proven to benefit a lot from jointly learning the two sub-tasks. However, such joint learning becomes challenging in the few-shot scenarios: on the one hand, the sparsity of samples greatly magnifies the difficulty of modeling the connection between the two tasks; on the other hand, how to jointly learn multiple tasks in the few-shot setting is still less investigated. In response to this, we introduce FewJoint, the first FSL benchmark for joint dialogue understanding. FewJoint provides a new corpus with 59 different dialogue domains from real industrial API and a code platform to ease FSL experiment set-up, which are expected to advance the research of this field. Further, we find that insufficient performance of the few-shot setting often leads to noisy sharing between two sub-task and disturbs joint learning. To tackle this, we guide slot with explicit intent information and propose a novel trust gating mechanism that blocks low-confidence intent information to ensure high quality sharing. Besides, we introduce a Reptile-based meta-learning strategy to achieve better generalization in unseen few-shot domains. In the experiments, the proposed method brings significant improvements on two datasets and achieve new state-of-the-art performance.",
        "What is the difference between HTML and JavaScript?",
        "Why might someone prefer to shop at a small, locally-owned business instead of a large chain store, even if the prices are higher?",
        "What's the permission that allows creating provisioning profiles in Apple Developer account is called?",
    ]

    # Per-request steering: compare Chinese and Korean steering on each prompt
    config_paths = {
        "Chinese": "model_configs/activation_steer/Phi-3-mini-4k-instruct-chinese.json",
        "Korean": "model_configs/activation_steer/Phi-3-mini-4k-instruct-korean.json",
    }

    sampling_params_by_language = {}
    for language, config_path in config_paths.items():
        with open(config_path) as f:
            config = json.load(f)
        sampling_params_by_language[language] = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            stop_token_ids=[llm.tokenizer.eos_token_id, 32007],
            extra_args={"steer": config["steering"]},
        )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop_token_ids=[llm.tokenizer.eos_token_id, 32007],
    )

    for case in test_cases:
        print("=" * 50)
        prompt = case
        print(f"Original prompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        example = llm.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        for language, language_sampling_params in sampling_params_by_language.items():
            output = llm.generate(example, language_sampling_params)
            print(f"With {language} activation steering:")
            print(output[0].outputs[0].text)
            llm.llm_engine.reset_prefix_cache()
        
        llm.llm_engine.reset_prefix_cache()
        output = llm.generate(example, sampling_params, use_hook=False)
        print("Without activation steering:")
        print(output[0].outputs[0].text)
        llm.llm_engine.reset_prefix_cache()

