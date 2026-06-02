"""Serve-path

Launch a `vllm serve` instance in a separate terminal:

    VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOOK_WORKER=steer \\
        vllm serve microsoft/Phi-3-mini-4k-instruct \\
        --enforce-eager --max-model-len 2048 --port 8770

Different requests can carry different steer configs via
``extra_body["vllm_xargs"]["steer"]`` — vllm_xargs requires scalar values,
so the per-request dict is JSON-encoded as a string. The plugin
(_hook_plugin._patched_generate) decodes it back to a dict before the
worker reads it.
"""
import json
import openai


if __name__ == "__main__":
    base_url = "http://localhost:8770/v1"
    model = "microsoft/Phi-3-mini-4k-instruct"
    config_path = f'model_configs/activation_steer/{model.split("/")[-1]}.json'

    with open(config_path) as f:
        config = json.load(f)
    default_config = config["steering"]

    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")

    test_case = [
        "Write a dialogue between two people, one is dressed up in a ball gown and the other is dressed down in sweats. The two are going to a nightly event. Your answer must contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each bullet) such as:\n* This is the first point.\n* This is the second point.",
    ]

    sp = dict(model=model, max_tokens=1024, temperature=0.0)

    print("=" * 50)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": test_case[0]}],
        extra_body={"vllm_xargs": {"steer": json.dumps(default_config)},
                    "stop_token_ids": [32007]},
        **sp,
    )
    print("With activation steering:")
    print(response.choices[0].message.content)

    print("=" * 50)
    override = {**default_config, "method": "add_vector", "coefficient": 10}
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": test_case[0]}],
        extra_body={"vllm_xargs": {"steer": json.dumps(override)},
                    "stop_token_ids": [32007]},
        **sp,
    )
    print("With activation steering and overwritten steering configs:")
    print(response.choices[0].message.content)

    print("=" * 50)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": test_case[0]}],
        extra_body={"vllm_xargs": {}, "stop_token_ids": [32007]},
        **sp,
    )
    print("Without activation steering:")
    print(response.choices[0].message.content)
