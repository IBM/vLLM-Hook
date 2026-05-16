import os
import json
import torch
from typing import Dict, Optional
from vllm.v1.worker.gpu_worker import Worker as V1Worker
from vllm.forward_context import get_forward_context


def _last_token_indices(residuals: torch.Tensor) -> Optional[torch.Tensor]:
    """Return per-sequence last-query-token indices into the flat residuals.

    Returns None when no real forward is in flight (warmup, CUDA graph
    capture, or absent attn_metadata) so the caller can no-op.

    Hybrid-model note: in models like Qwen3.5, linear-attention layers may
    have no entry under their own key, so query_start_loc is taken from any
    available metadata entry (the query layout is shared across layers).
    """
    if torch.cuda.is_current_stream_capturing():
        return None
    ctx = get_forward_context()
    metadata = getattr(ctx, "attn_metadata", None)
    if metadata is None:
        return None

    query_start_loc = getattr(metadata, "query_start_loc", None)
    if query_start_loc is None and isinstance(metadata, dict):
        for entry in metadata.values():
            query_start_loc = getattr(entry, "query_start_loc", None)
            if query_start_loc is not None:
                break
    if query_start_loc is None:
        return None

    return query_start_loc[1:] - 1


class SteerHookActWorker(V1Worker):
    
    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)
        
        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")
        
        return r
    
    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return
        
        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        steering_config = self._parse_steering_config()   
        self.steering_method = steering_config["method"]
        self.optimal_layer = steering_config["optimal_layer"]
        self.coefficient = steering_config["coefficient"]
        self.apply_at_all_positions = steering_config["apply_at_all_positions"]

        vector_path = steering_config["vector_path"]
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Steering vector not found at: {vector_path}")
        steering_data = torch.load(vector_path)
        self.dir = torch.tensor(steering_data["dir"])
        if self.steering_method == "adjust_rs":
            self.avg_proj = steering_data["avg_proj"]
            self.unit_vector = self.dir # / torch.norm(self.dir)
        
        def steering_hook(input, output):
            
            if not os.path.exists(self.hook_flag):
                return output
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                hidden_states, residuals = output
            else:
                hidden_states = None
                residuals = output
                
            steering_vec = self.dir.to(residuals.device, dtype=residuals.dtype)
            
            if self.steering_method == "add_vector":
                if self.apply_at_all_positions:
                    steering_vec = steering_vec.view(1, -1)
                    residuals = residuals + self.coefficient * steering_vec
                else:
                    # Last-token-per-sequence steering. Mirrors the `last_token`
                    # idiom used by probe_hidden_states_worker: in vLLM v1 the
                    # decoder-block output is flattened to [total_tokens, hidden]
                    # and query_start_loc[i+1]-1 is the last query-token index
                    # of sequence i (works uniformly for prefill, decode, and
                    # mixed batches; matches HF-hook "score at final position"
                    # semantics used by many steering-vector papers).
                    last_indices = _last_token_indices(residuals)
                    if last_indices is None:
                        # Warmup / CUDA graph capture / non-attention pass.
                        return output
                    residuals = residuals.clone()
                    residuals[last_indices] = (
                        residuals[last_indices] + self.coefficient * steering_vec
                    )
                
            elif self.steering_method == "adjust_rs":
                unit_vec = self.unit_vector.to(residuals.device, dtype=residuals.dtype)
                avg_proj = self.avg_proj.to(residuals.device, dtype=residuals.dtype)
                
                current_projections = torch.matmul(residuals, unit_vec) 
                coeff = (avg_proj - current_projections).unsqueeze(-1)       
                unit_vec = unit_vec.view(1, -1)  
                
                residuals = residuals + coeff * unit_vec
            
            else:
                raise ValueError(f"Unknown steering method: {self.steering_method}")
            
            if is_tuple:
                return (hidden_states, residuals)
            else:
                return residuals

        # register hooks on attention modules 
        self._hooks = []
        target_layer_name = f"model.layers.{self.optimal_layer}"

        for name, module in model.named_modules():
            if name == target_layer_name:
                hook = module.register_forward_hook(
                    lambda m, i, o: steering_hook(i,o)
                    )
                self._hooks.append(hook)
                break

        print(f"Installed {len(self._hooks)} hooks on layers: {name}")
    
    def _parse_steering_config(self) -> Dict:
        config_path = os.environ.get("VLLM_ACTSTEER_CONFIG")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        steering_config = config.get("steering", {})
        return {
            "method": steering_config.get("method", "adjust_rs"),  # "add_vector" or "adjust_rs"
            "optimal_layer": int(steering_config.get("optimal_layer", 15)),
            "coefficient": float(steering_config.get("coefficient", 0)),  # for add_vector
            "vector_path": steering_config.get("vector_path"),
            "apply_at_all_positions": steering_config.get("apply_at_all_positions", True)
        }

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)