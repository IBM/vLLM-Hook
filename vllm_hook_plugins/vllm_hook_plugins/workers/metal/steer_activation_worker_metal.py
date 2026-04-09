import json
import os
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.utils.torch_utils import set_random_seed
from vllm_metal.platform import MetalPlatform
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.utils import set_wired_limit
from vllm_metal.v1.worker import MetalWorker

TARGET_LAYER_TEMPLATE = "model.layers.{layer_num}"


class MLXSteeringWrapper(nn.Module):
    """
    Wrap an MLX layer and apply steering to its output.

    Args:
        None.

    Returns:
        None: Instances proxy calls to the wrapped MLX module.
    """

    def __init__(self, module, name, hook_fn):
        """
        Store the wrapped module metadata and steering callback.

        Args:
            module: MLX module being wrapped.
            name (str): Fully qualified module name.
            hook_fn: Callback invoked after wrapped module execution.

        Returns:
            None: Initializes wrapper state.
        """
        super().__init__()
        self.module = module
        self.name = name
        self.hook_fn = hook_fn

    def __call__(self, *args, **kwargs):
        """
        Run the wrapped module and post-process its output.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Any: Output produced by the wrapped module after steering.
        """
        output = self.module(*args, **kwargs)
        return self.hook_fn(output, self.name)


class SteerHookActWorkerMetal(MetalWorker):
    def __init__(self, *args, **kwargs):
        """
        Initialize steering state before Metal worker setup.

        Args:
            *args: Positional arguments forwarded to the base Metal worker.
            **kwargs: Keyword arguments forwarded to the base Metal worker.

        Returns:
            None: Worker state is initialized in-place.
        """
        self._capture_active = False
        super().__init__(*args, **kwargs)

    def init_device(self) -> None:
        """
        Initialize the Metal device, with a single-process fast path.

        Args:
            None.

        Returns:
            None: Device state is initialized in-place.
        """
        try:
            world_size = self.parallel_config.world_size
            if world_size == 1:
                # This branch remains because the Metal worker can bypass the
                # distributed setup used by the non-Metal worker when only one
                # process is active.
                self._init_device_single_process()
            else:
                super().init_device()
        except Exception:
            raise

    def _init_device_single_process(self) -> None:
        """
        Initialize MLX and the Metal model runner for world size one.

        Args:
            None.

        Returns:
            None: MLX and model-runner state are initialized in-place.
        """
        if self.metal_config.use_mlx:
            device_type = (
                mx.DeviceType.gpu
                if self.metal_config.mlx_device == "gpu"
                else mx.DeviceType.cpu
            )
            mx.set_default_device(mx.Device(device_type))
            set_wired_limit()

        self.device = MetalPlatform.get_torch_device(0)
        set_random_seed(self.model_config.seed)

        from vllm_metal.v1.model_runner import MetalModelRunner

        self.model_runner = MetalModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def load_model(self, *args, **kwargs):
        """
        Load the model and install the steering wrapper.

        Args:
            *args: Positional arguments forwarded to the base worker.
            **kwargs: Keyword arguments forwarded to the base worker.

        Returns:
            Any: Result returned by the base worker ``load_model`` call.
        """
        result = super().load_model(*args, **kwargs)

        try:
            self._install_hooks()
            print("Hooks installed successfully", flush=True)
        except Exception as exc:
            print(f"Hook installation failed: {exc}", flush=True)

        return result

    def _install_hooks(self):
        """
        Install the steering wrapper on the configured transformer layer.

        Args:
            None.

        Returns:
            None: Wrappers are installed in-place on the loaded model.
        """
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

        steering_data = torch.load(vector_path, map_location="cpu")
        self.dir = torch.as_tensor(steering_data["dir"]).detach().cpu()
        # These cached MLX copies remain because Metal layers may emit MLX
        # arrays, while the non-Metal worker only needs torch tensors.
        self._dir_mlx = torch_to_mlx(self.dir)

        if self.steering_method == "adjust_rs":
            self.avg_proj = torch.as_tensor(steering_data["avg_proj"]).detach().cpu()
            self.unit_vector = self.dir
            self._avg_proj_mlx = torch_to_mlx(self.avg_proj)
            self._unit_vector_mlx = self._dir_mlx

        self._hooks = []
        self._matched_hook_modules = []
        target_layer_name = TARGET_LAYER_TEMPLATE.format(layer_num=self.optimal_layer)
        named_modules = dict(model.named_modules())

        for name, module in named_modules.items():
            if name != target_layer_name:
                continue

            parent_name, target_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name)
            if parent is None:
                break

            # This remains a wrapper replacement instead of `register_forward_hook`
            # because the Metal MLX modules do not expose the same hook API as
            # the non-Metal PyTorch modules.
            wrapped_module = MLXSteeringWrapper(
                module=module,
                name=name,
                hook_fn=self._steering_hook,
            )
            setattr(parent, target_name, wrapped_module)
            self._hooks.append(
                {
                    "parent": parent,
                    "target_name": target_name,
                    "original_module": module,
                }
            )
            self._matched_hook_modules.append(name)
            break

        print(
            f"Installed {len(self._matched_hook_modules)} hooks on layers: "
            f"{self._matched_hook_modules}",
            flush=True,
        )

    def _parse_steering_config(self) -> Dict:
        """
        Load steering settings from `VLLM_ACTSTEER_CONFIG`.

        Args:
            None.

        Returns:
            Dict: Parsed steering configuration with normalized field types.
        """
        config_path = os.environ.get("VLLM_ACTSTEER_CONFIG")

        with open(config_path, "r") as f:
            config = json.load(f)

        steering_config = config.get("steering", {})
        return {
            "method": steering_config.get("method", "adjust_rs"),
            "optimal_layer": int(steering_config.get("optimal_layer", 15)),
            "coefficient": float(steering_config.get("coefficient", 0)),
            "vector_path": steering_config.get("vector_path"),
            "apply_at_all_positions": steering_config.get(
                "apply_at_all_positions", True
            ),
        }

    def _steering_enabled(self) -> bool:
        """
        Return whether steering is active for the current execution.

        Args:
            None.

        Returns:
            bool: ``True`` when the current execution should apply steering.
        """
        return bool(
            self._capture_active
            and self.hook_flag
            and os.path.exists(self.hook_flag)
        )

    def _mlx_cast_like(self, value: mx.array, reference: mx.array) -> mx.array:
        """
        Cast an MLX array to the dtype used by a reference tensor.

        Args:
            value (mx.array): MLX array to cast.
            reference (mx.array): Reference array providing the target dtype.

        Returns:
            mx.array: Array converted to the reference dtype when needed.
        """
        if value.dtype != reference.dtype:
            return value.astype(reference.dtype)
        return value

    def _apply_torch_steering(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Apply steering to a PyTorch residual tensor.

        Args:
            residuals (torch.Tensor): Residual stream tensor to modify.

        Returns:
            torch.Tensor: Residual tensor after steering is applied.
        """
        steering_vec = self.dir.to(residuals.device, dtype=residuals.dtype)

        if self.steering_method == "add_vector":
            if not self.apply_at_all_positions:
                raise NotImplementedError(
                    "Only supports apply_at_all_positions=True for now."
                )
            return residuals + self.coefficient * steering_vec.view(1, -1)

        if self.steering_method == "adjust_rs":
            unit_vec = self.unit_vector.to(residuals.device, dtype=residuals.dtype)
            avg_proj = self.avg_proj.to(residuals.device, dtype=residuals.dtype)
            current_projections = torch.matmul(residuals, unit_vec)
            coeff = (avg_proj - current_projections).unsqueeze(-1)
            return residuals + coeff * unit_vec.view(1, -1)

        raise ValueError(f"Unknown steering method: {self.steering_method}")

    def _apply_mlx_steering(self, residuals: mx.array) -> mx.array:
        """
        Apply steering to an MLX residual tensor.

        Args:
            residuals (mx.array): Residual stream array to modify.

        Returns:
            mx.array: Residual array after steering is applied.
        """
        # This separate MLX path remains because Metal layers may surface MLX
        # arrays instead of PyTorch tensors, while the non-Metal worker only
        # needs the torch implementation.
        steering_vec = self._mlx_cast_like(self._dir_mlx, residuals)

        if self.steering_method == "add_vector":
            if not self.apply_at_all_positions:
                raise NotImplementedError(
                    "Only supports apply_at_all_positions=True for now."
                )
            return residuals + (self.coefficient * mx.expand_dims(steering_vec, axis=0))

        if self.steering_method == "adjust_rs":
            unit_vec = self._mlx_cast_like(self._unit_vector_mlx, residuals)
            avg_proj = self._mlx_cast_like(self._avg_proj_mlx, residuals)
            current_projections = mx.matmul(
                residuals, mx.expand_dims(unit_vec, axis=-1)
            ).squeeze(-1)
            coeff = mx.expand_dims(avg_proj - current_projections, axis=-1)
            return residuals + coeff * mx.expand_dims(unit_vec, axis=0)

        raise ValueError(f"Unknown steering method: {self.steering_method}")

    def _steering_hook(self, output, _module_name: str):
        """
        Transform the hooked module output when steering is enabled.

        Args:
            output: Output produced by the wrapped transformer layer.
            _module_name (str): Hook-reported module name.

        Returns:
            Any: Original or steered output in the same structure as input.
        """
        if not self._steering_enabled():
            return output

        is_tuple = isinstance(output, tuple)
        if is_tuple:
            hidden_states, residuals = output
        else:
            hidden_states = None
            residuals = output

        if torch.is_tensor(residuals):
            residuals = self._apply_torch_steering(residuals)
        else:
            residuals = self._apply_mlx_steering(residuals)

        if is_tuple:
            return (hidden_states, residuals)
        return residuals

    def _uninstall_hooks(self):
        """
        Restore the original module after temporary wrapping.

        Args:
            None.

        Returns:
            None: Wrapped modules are restored in-place.
        """
        for entry in reversed(getattr(self, "_hooks", [])):
            setattr(entry["parent"], entry["target_name"], entry["original_module"])
        if hasattr(self, "_hooks"):
            self._hooks.clear()

    def execute_model(self, *args, **kwargs):
        """
        Run the model with steering enabled only during active execution.

        Args:
            *args: Positional arguments forwarded to the base worker.
            **kwargs: Keyword arguments forwarded to the base worker.

        Returns:
            Any: Result returned by the base worker ``execute_model`` call.
        """
        # This extra gate remains because the Metal runtime may invoke wrapped
        # modules during setup paths where steering should stay disabled.
        self._capture_active = True
        try:
            return super().execute_model(*args, **kwargs)
        finally:
            self._capture_active = False
