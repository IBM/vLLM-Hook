import mlx.core as mx
from mlx_lm import load
import mlx_lm.models.granite as granite_mod


MODEL_PATH = "ibm-granite/granite-3.1-8b-instruct"
TARGET_LAYER = 6
PROMPT = "The quick brown fox"


captures = {}
original_sdpa = granite_mod.scaled_dot_product_attention
call_count = {"count": 0}


def probing_sdpa(queries, keys, values, cache=None, scale=None, mask=None):
    call_count["count"] += 1
    current_layer = call_count["count"] - 1

    if current_layer == TARGET_LAYER:
        captures["queries"] = queries
        captures["keys"] = keys
        captures["values"] = values
        captures["scale"] = scale
        print(f"[sdpa probe] layer={current_layer}")
        print(f"[sdpa probe] queries shape={queries.shape}")
        print(f"[sdpa probe] keys shape={keys.shape}")
        print(f"[sdpa probe] values shape={values.shape}")
        print(f"[sdpa probe] mask shape={getattr(mask, 'shape', None)}")
        print(f"[sdpa probe] cache type={type(cache).__name__}")

    return original_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask)


granite_mod.scaled_dot_product_attention = probing_sdpa

model, tokenizer = load(MODEL_PATH)
tokens = mx.array([tokenizer.encode(PROMPT)])
print(f"Loaded model: {MODEL_PATH}")
print(f"Prompt token count: {tokens.shape[-1]}")

_ = model(tokens)
mx.eval(captures)

queries = captures["queries"]
keys = captures["keys"]
values = captures["values"]
scale = captures["scale"]

print(
    f"\nqueries stats: min={mx.min(queries).item():.6f} "
    f"max={mx.max(queries).item():.6f} "
    f"mean={mx.mean(queries).item():.6f}"
)
print(
    f"keys stats: min={mx.min(keys).item():.6f} "
    f"max={mx.max(keys).item():.6f} "
    f"mean={mx.mean(keys).item():.6f}"
)
print(
    f"values stats: min={mx.min(values).item():.6f} "
    f"max={mx.max(values).item():.6f} "
    f"mean={mx.mean(values).item():.6f}"
)

queries_last = queries[0, :, -1, :]
keys_all = keys[0, :, :, :]
num_repeat = queries.shape[1] // keys.shape[1]
keys_all = mx.repeat(keys_all, repeats=num_repeat, axis=0)
scores = mx.matmul(queries_last[:, None, :], mx.transpose(keys_all, (0, 2, 1)))
attention_scores = mx.softmax(scores * scale, axis=-1)[:, 0, :]

print(
    f"attention stats: min={mx.min(attention_scores).item():.6f} "
    f"max={mx.max(attention_scores).item():.6f} "
    f"mean={mx.mean(attention_scores).item():.6f}"
)

head0 = attention_scores[0]
top_idx = mx.argmax(head0).item()
print(f"head0 top position: {top_idx}")
print(f"head0 first 10 values: {head0[:10]}")
print(f"queries_last head0 first 10 values: {queries_last[0, :10]}")
print(f"keys_all head0 last token first 10 values: {keys_all[0, -1, :10]}")
