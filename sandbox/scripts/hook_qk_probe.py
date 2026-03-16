import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


MODEL_PATH = "ibm-granite/granite-3.1-8b-instruct"
TARGET_LAYER = 6
PROMPT = "The quick brown fox"


model, tokenizer = load(MODEL_PATH)
attn = model.model.layers[TARGET_LAYER].self_attn

print(f"Loaded model: {MODEL_PATH}")
print(f"Inspecting layer: {TARGET_LAYER}")
print(f"self_attn type: {type(attn).__name__}")
print(f"has attr 'attn': {hasattr(attn, 'attn')}")
print(f"has attr 'q_proj': {hasattr(attn, 'q_proj')}")
print(f"has attr 'k_proj': {hasattr(attn, 'k_proj')}")
print(f"has attr 'v_proj': {hasattr(attn, 'v_proj')}")
print(f"has attr 'o_proj': {hasattr(attn, 'o_proj')}")

print("\nFirst 60 attribute names on self_attn:")
for attr_name in sorted(name for name in dir(attn) if not name.startswith("__"))[:60]:
    print(f"  - {attr_name}")

print("\nNamed modules under self_attn:")
for child_name, child_module in attn.named_modules():
    label = child_name if child_name else "<self_attn>"
    print(f"  - {label}: {type(child_module).__name__}")

print("\nLeaf modules under self_attn:")
for item in attn.leaf_modules():
    print(f"  - {type(item).__name__}: {item}")


class SelfAttnProbe(nn.Module):
    def __init__(self, original_layer, layer_name):
        super().__init__()
        self.layer = original_layer
        self.name = layer_name

    def __call__(self, *args, **kwargs):
        print(f"\n[self_attn probe] layer={self.name}")
        print(f"[self_attn probe] positional args={len(args)} keyword args={sorted(kwargs.keys())}")
        for idx, arg in enumerate(args):
            shape = getattr(arg, "shape", None)
            print(f"[self_attn probe] arg[{idx}] type={type(arg).__name__} shape={shape}")
        for key, value in kwargs.items():
            shape = getattr(value, "shape", None)
            print(f"[self_attn probe] kwarg[{key}] type={type(value).__name__} shape={shape}")

        out = self.layer(*args, **kwargs)
        print(f"[self_attn probe] output type={type(out).__name__} shape={getattr(out, 'shape', None)}")
        return out


class FakeAttnProbe(nn.Module):
    def __call__(self, *args, **kwargs):
        print("\n[fake attn probe] self_attn.attn was called")
        print(f"[fake attn probe] positional args={len(args)} keyword args={sorted(kwargs.keys())}")
        for idx, arg in enumerate(args):
            print(f"[fake attn probe] arg[{idx}] type={type(arg).__name__} shape={getattr(arg, 'shape', None)}")
        return args[0] if args else None


attn.attn = FakeAttnProbe()
print(f"\nInjected fake attr 'attn': {hasattr(attn, 'attn')}")
print(f"Updated named_modules under self_attn after injection:")
for child_name, child_module in attn.named_modules():
    label = child_name if child_name else "<self_attn>"
    print(f"  - {label}: {type(child_module).__name__}")

model.model.layers[TARGET_LAYER].self_attn = SelfAttnProbe(
    model.model.layers[TARGET_LAYER].self_attn,
    f"model.layers.{TARGET_LAYER}.self_attn",
)

tokens = mx.array([tokenizer.encode(PROMPT)])
print(f"\nPrompt token count: {tokens.shape[-1]}")
_ = model(tokens)
