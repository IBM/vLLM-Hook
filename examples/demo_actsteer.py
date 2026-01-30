import os
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_hook_plugins import HookLLM
from vllm import SamplingParams

if __name__ == "__main__":

    cache_dir = "/dccstor/pyrite/irene/"
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
        "Write a dialogue between two people, one is dressed up in a ball gown and the other is dressed down in sweats. The two are going to a nightly event. Your answer must contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each bullet) such as:\n* This is the first point.\n* This is the second point.",
        "What is the difference between the 13 colonies and the other British colonies in North America? Your answer must contain exactly 6 bullet point in Markdown using the following format:\n* Bullet point one.\n* Bullet point two.\n...\n* Bullet point fix."
    ]
    
    for case in test_cases:
        print("=" * 50)
        prompt = case
        messages = [{"role": "user", "content": prompt}]
        example = llm.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        sampling_params = SamplingParams(
            temperature=0.0,                       
            max_tokens=2048,
            stop_token_ids=[llm.tokenizer.eos_token_id, 32007],  
        )

        output = llm.generate(example, sampling_params)
        print("With activation steering:")
        print(output[0].outputs[0].text)
        
        llm.llm_engine.reset_prefix_cache()
        output = llm.generate(example, sampling_params, use_hook=False)
        print("Without activation steering:")
        print(output[0].outputs[0].text)
        llm.llm_engine.reset_prefix_cache()
            
