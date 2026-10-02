"""Activation steering over ``vllm serve``: per-request steer configs via vllm_xargs."""
import json
import openai

from _paths import config_path, vector_path
from _serve import STEER, base_url, require_server


if __name__ == "__main__":
    model = "microsoft/Phi-3-mini-4k-instruct"
    require_server(model, STEER)
    cfg_file = config_path(f'activation_steer/{model.split("/")[-1]}-chinese.json')

    with open(cfg_file) as f:
        config = json.load(f)
    default_config = config["steering"]

    client = openai.OpenAI(base_url=base_url(), api_key="EMPTY")

    test_case = [
        "Hello hello! Thank you for coming to our Expo talk today. I hope you enjoyed our talk so far. Do you have any questions?"
        ]

    sp = dict(model=model, max_tokens=100, temperature=0.0)

    print("=" * 50)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": test_case[0]}],
        extra_body={"vllm_xargs": {"steer": json.dumps(default_config)},
                    "stop_token_ids": [32007]},
        **sp,
    )
    print("With activation steering for Chinese:")
    print(response.choices[0].message.content)

    print("=" * 50)
    override = {**default_config, "method": "add_vector", "coefficient": 4, "optimal_layer": 0, "vector_path": vector_path("phi3_korean.pt")}
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": test_case[0]}],
        extra_body={"vllm_xargs": {"steer": json.dumps(override)},
                    "stop_token_ids": [32007]},
        **sp,
    )
    print("With activation steering and overwritten steering configs to generate Korea:")
    print(response.choices[0].message.content)

