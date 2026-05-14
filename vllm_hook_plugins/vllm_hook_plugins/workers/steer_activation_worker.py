import os
import json
import torch
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


class SteerHookActWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (SteerHookActWorker,) at runtime,
    so self is the Worker instance. Methods are callable via collective_rpc.
    """

    if TYPE_CHECKING:
        model_runner: Any

    def install_hooks(self):
        """Install steering hook. Idempotent. Callable via collective_rpc."""
        if getattr(self, "_hooks_installed", False):
            return
        self._hooks_installed = True
        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")
    
    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return
        
        steering_config = self._parse_steering_config()
        self.steering_method = steering_config["method"]
        self.optimal_layer = steering_config["optimal_layer"]
        self.coefficient = steering_config["coefficient"]
        self.apply_at_all_positions = steering_config["apply_at_all_positions"]

        vector_path = steering_config["vector_path"]
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Steering vector not found at: {vector_path}")
        steering_data = torch.load(vector_path, weights_only=False)
        self.dir = torch.tensor(steering_data["dir"])
        if self.steering_method == "adjust_rs":
            self.avg_proj = steering_data["avg_proj"]
            self.unit_vector = self.dir # / torch.norm(self.dir)
        
        def steering_hook(input, output):
            # Apply only when at least one request in the batch has steer=True.
            req_ids = getattr(self.model_runner.input_batch, "req_ids", [])
            active = False
            for r in req_ids:
                req_state = self.model_runner.requests.get(r)
                if req_state and (req_state.sampling_params.extra_args or {}).get("steer"):
                    active = True
                    break
            if not active:
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
                else:
                    raise NotImplementedError("Only supports apply_at_all_positions=True for now.")
                residuals = residuals + self.coefficient * steering_vec
                
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

        print(f"Installed {len(self._hooks)} steering hooks on layer: {target_layer_name}")
    
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