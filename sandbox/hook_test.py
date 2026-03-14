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
    #Only hook the MLP itself. This prevents trying to hook its children
    #which would fail because we've already wrapped the parent
    if name.endswith(".mlp") and "layers" in name: 
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
