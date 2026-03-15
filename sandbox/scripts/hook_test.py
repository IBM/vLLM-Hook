import mlx.core as mx 
import mlx.nn as nn
from mlx_lm import load 

#1. Load a tiny model for testing. 
model_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
model, tokenizer = load(model_path)

#2. Storage for our "hooked" Activations
activations = {}

#3. Define the Hook Wrapper
class SimpleHook(nn.Module): 
    def __init__(self, original_layer, layer_name): 
        super().__init__()
        self.layer = original_layer
        self.name = layer_name
            
    def __call__(self, x, *args, **kwargs): 
        #Pass data through original layer 
        out = self.layer(x, *args, **kwargs)
        #Store a copy of the output (intermediate activation)
        activations[self.name] = out
        return out
        
#4. Corrected Manual Hook Registration
max_debug = 3
hook_count = 0

for name, module in model.named_modules():
    #Only hook the self attention module itself. This prevents trying to hook its children
    #which would fail because we've already wrapped the parent
    if name.endswith(".self_attn") and "layers" in name: 
        parts = name.split(".")
        parent = model
        
        for part in parts[:-1]: 
            if part.isdigit(): 
                parent = parent[int(part)]
            else: 
                parent = getattr(parent, part)
        
        #Swap the layer 
        target_name = parts[-1]
        wrapped = SimpleHook(module, name)
        setattr(parent, target_name, wrapped)
        
        if hook_count < max_debug:
            print(f"\n[hook] selecting {name}")
            print(f"[hook] parent path: {'.'.join(parts[:-1])}")
            print(f"[hook] target attr: {target_name}")
            print(f"[hook] before: {type(module).__name__}")
            print(f"[hook] after: {type(getattr(parent, target_name)).__name__}")
            print(
                f"[hook] effect: model.{name} now dispatches through "
                "SimpleHook.__call__"
            )
        hook_count += 1
        
print(f"\nRegistration complete: wrapped {hook_count} modules")
print("Running inference...")

# House analogy inspection:
# - address: where the model looks up a layer
# - outer house: the wrapper currently living at that address
# - inner house: the original layer stored inside the wrapper
sample_layer_idx = 22
sample_address = f"model.model.layers.{sample_layer_idx}.self_attn"
outer_house = model.model.layers[sample_layer_idx].self_attn
inner_house = outer_house.layer

print("\nHouse Analogy Inspection")
print(f"[address] {sample_address}")
print(f"[outer house] object at address now: {type(outer_house).__name__}")
print(f"[inner house] original layer stored inside wrapper: {type(inner_house).__name__}")
print(f"[occupant] original layer object: {inner_house}")

print("\nOccupants of the original house")
print("[occupant] attribute names on original layer:")
for attr_name in sorted(
    name for name in dir(inner_house)
    if not name.startswith("__")
)[:40]:
    print(f"  - {attr_name}")

print("\nModule directory tools on the original house")
print("[named_modules] houses inside the original house:")
for child_name, child_module in inner_house.named_modules():
    label = child_name if child_name else "<original house itself>"
    print(f"  - {label}: {type(child_module).__name__}")

print("\n[leaf_modules] smallest houses inside the original house:")
for item in inner_house.leaf_modules():
    print(f"  - {type(item).__name__}: {item}")

print(
    "\n[analogy] these tools are house directories. "
    "They help you find houses inside the building, but they are not the door."
)

print("\n[occupant] original layer parameters:")
def print_parameter_tree(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            print_parameter_tree(value, next_prefix)
        return

    shape = getattr(node, "shape", None)
    if shape is not None:
        print(f"  - {prefix}: shape={shape}")
    else:
        print(f"  - {prefix}: type={type(node).__name__}")

print_parameter_tree(inner_house.parameters())

#5. Run a forward pass
prompt = "The quick brown fox"
tokens = mx.array([tokenizer.encode(prompt)])

# 1. Forward Pass
logits = model(tokens)

# 2. MLX is lazy - force the hooks to actually compute and save data
mx.eval(activations)


# 3. Check Results 
print(f"\nResults")
for name, tensor in activations.items(): 
    #Typically shape is (Batch, Sequence_Length, Hidden_Dim)
    print(f"{name} | Shape: {tensor.shape}")
    
# Optional: Print the actual values of the first layer's first token.
first_layer_key = list(activations.keys())[0]
print(f"\nSample data from {first_layer_key} (first five values):")
print(activations[first_layer_key][0, 0, :5])
