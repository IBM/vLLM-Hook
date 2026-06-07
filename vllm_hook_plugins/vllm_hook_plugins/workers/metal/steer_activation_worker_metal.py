import gc
import json
import os
import re
import subprocess
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.utils.torch_utils import set_random_seed
from vllm_metal.platform import MetalPlatform
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.utils import set_wired_limit

LAYER_PATTERNS = (
    re.compile(r"^model\.layers\.(\d+)$"),
    re.compile(r"^model\.model\.layers\.(\d+)$"),
    re.compile(r"^layers\.(\d+)$"),
)

MEMORY_GUARD_MIN_AVAILABLE_MB = int(
    float(os.environ.get("VLLM_HOOK_MIN_AVAILABLE_GB", "4")) * 1024
)
MEMORY_GUARD_MAX_RSS_MB = int(
    float(os.environ.get("VLLM_HOOK_MAX_RSS_GB", "24")) * 1024
)


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
        trace = os.environ.get(
            "VLLM_HOOK_METAL_STEER_TRACE",
            os.environ.get("HOOK_METAL_STEER_TRACE", "0"),
        ) == "1"
        if trace:
            print(f"[metal-steer-wrapper] enter {self.name}", flush=True)
        output = self.module(*args, **kwargs)
        if trace:
            print(f"[metal-steer-wrapper] steer {self.name}", flush=True)
        output = self.hook_fn(output, self.name)
        if trace:
            print(f"[metal-steer-wrapper] exit {self.name}", flush=True)
        return output


class SteerHookActWorkerMetal:
    @staticmethod
    def _match_layer(name: str) -> int | None:
        for pattern in LAYER_PATTERNS:
            match = pattern.match(name)
            if match:
                return int(match.group(1))
        return None

    def _ensure_extension_state(self) -> None:
        if getattr(self, "_metal_steer_extension_ready", False):
            return
        self._capture_active = False
        self._hooks_installed = False
        self._debug_hook = os.environ.get(
            "HOOK_DEBUG", os.environ.get("VLLM_HOOK_DEBUG", "")
        ) == "1"
        self._memory_guard_enabled = os.environ.get(
            "VLLM_HOOK_METAL_STEER_MEMORY_GUARD",
            os.environ.get("HOOK_METAL_STEER_MEMORY_GUARD", "0"),
        ) == "1"
        self._metal_steer_extension_ready = True

    @staticmethod
    def _format_mb(value_mb: float | None) -> str:
        if value_mb is None:
            return "n/a"
        return f"{value_mb / 1024:.2f} GB"

    def _memory_snapshot(self) -> Dict[str, float | None]:
        rss_mb = None
        available_mb = None
        total_mb = None

        try:
            import psutil

            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)
            total_mb = vm.total / (1024 * 1024)
        except Exception:
            try:
                import resource

                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_mb = float(rss_kb) / 1024.0
            except Exception:
                pass

            try:
                page_size = int(
                    subprocess.check_output(
                        ["sysctl", "-n", "hw.pagesize"], text=True
                    ).strip()
                )
                vm_stat = subprocess.check_output(["vm_stat"], text=True)
                page_counts = {}
                for line in vm_stat.splitlines():
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    value = value.strip().rstrip(".")
                    if value.endswith("pages"):
                        value = value[:-5].strip()
                    if value.endswith("."):
                        value = value[:-1]
                    try:
                        page_counts[key.strip()] = int(value.replace(".", ""))
                    except ValueError:
                        continue
                free_pages = page_counts.get("Pages free", 0)
                inactive_pages = page_counts.get("Pages inactive", 0)
                speculative_pages = page_counts.get("Pages speculative", 0)
                available_mb = (
                    (free_pages + inactive_pages + speculative_pages)
                    * page_size
                    / (1024 * 1024)
                )
                total_mb = float(
                    int(
                        subprocess.check_output(
                            ["sysctl", "-n", "hw.memsize"], text=True
                        ).strip()
                    )
                    / (1024 * 1024)
                )
            except Exception:
                pass

        return {
            "rss_mb": rss_mb,
            "available_mb": available_mb,
            "total_mb": total_mb,
        }

    def _log_memory(self, stage: str, module_name: str, residuals=None) -> None:
        snap = self._memory_snapshot()
        residual_desc = "n/a"
        if residuals is not None:
            dtype = getattr(residuals, "dtype", None)
            shape = getattr(residuals, "shape", None)
            residual_desc = f"type={type(residuals).__name__} shape={shape} dtype={dtype}"
        self._stage(
            f"{stage} module={module_name} "
            f"rss={self._format_mb(snap['rss_mb'])} "
            f"available={self._format_mb(snap['available_mb'])} "
            f"total={self._format_mb(snap['total_mb'])} "
            f"residuals={residual_desc}"
        )

    def _enforce_memory_guard(self, stage: str, module_name: str) -> None:
        snap = self._memory_snapshot()
        rss_mb = snap["rss_mb"]
        available_mb = snap["available_mb"]
        if available_mb is not None and available_mb < MEMORY_GUARD_MIN_AVAILABLE_MB:
            raise MemoryError(
                f"Metal memory guard triggered at {stage} for {module_name}: "
                f"available={self._format_mb(available_mb)} "
                f"threshold={self._format_mb(MEMORY_GUARD_MIN_AVAILABLE_MB)}"
            )
        if rss_mb is not None and rss_mb > MEMORY_GUARD_MAX_RSS_MB:
            raise MemoryError(
                f"Metal memory guard triggered at {stage} for {module_name}: "
                f"rss={self._format_mb(rss_mb)} "
                f"threshold={self._format_mb(MEMORY_GUARD_MAX_RSS_MB)}"
            )

    def _stage(self, message: str) -> None:
        self._ensure_extension_state()
        if not getattr(self, "_debug_hook", False):
            return
        pid = os.getpid()
        rank = getattr(self, "rank", "?")
        local_rank = getattr(self, "local_rank", "?")
        print(
            f"[metal-steer-worker pid={pid} rank={rank} local_rank={local_rank}] {message}",
            flush=True,
        )

    def __init__(self, *args, **kwargs):
        """
        Initialize steering state for the Metal mixin.

        Args:
            *args: Positional arguments forwarded to the base Metal worker.
            **kwargs: Keyword arguments forwarded to the base Metal worker.

        Returns:
            None: Worker state is initialized in-place.
        """
        self._ensure_extension_state()

    def install_hooks(self):
        """
        Install the steering wrapper on the configured transformer layer.

        Returns:
            None: Wrappers are installed in-place on the loaded model.
        """
        self._ensure_extension_state()
        if getattr(self, "_hooks_installed", False):
            return
        self._hooks_installed = True
        self._capture_active = True
        self._stage("install_hooks RPC start")
        try:
            self._install_hooks()
            print("Hooks installed successfully", flush=True)
        except Exception as exc:
            print(f"Hook installation failed: {exc}", flush=True)
        self._stage("install_hooks RPC complete")

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

        self.hook_flag = os.environ.get("HOOK_FLAG", os.environ.get("VLLM_HOOK_FLAG"))
        steering_config = self._parse_steering_config()
        self.steering_method = steering_config["method"]
        self.optimal_layer = steering_config["optimal_layer"]
        self.coefficient = steering_config["coefficient"]
        self.apply_at_all_positions = steering_config["apply_at_all_positions"]

        vector_path = steering_config["vector_path"]
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Steering vector not found at: {vector_path}")

        steering_data = torch.load(
            vector_path,
            map_location="cpu",
            weights_only=False,
        )
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
        named_modules = dict(model.named_modules())
        wrap_all_layers = os.environ.get(
            "VLLM_HOOK_METAL_STEER_ALL_LAYERS",
            os.environ.get("HOOK_METAL_STEER_ALL_LAYERS", "0"),
        ) == "1"

        def install_wrapper(name, module, parent, target_name) -> bool:
            if parent is None:
                return False
            if any(entry["original_module"] is module for entry in self._hooks):
                return False

            # This remains a wrapper replacement instead of `register_forward_hook`
            # because the Metal MLX modules do not expose the same hook API as
            # the non-Metal PyTorch modules.
            wrapped_module = MLXSteeringWrapper(
                module=module,
                name=name,
                hook_fn=self._steering_hook,
            )
            if isinstance(parent, list):
                parent[target_name] = wrapped_module
            else:
                setattr(parent, target_name, wrapped_module)
            self._hooks.append(
                {
                    "parent": parent,
                    "target_name": target_name,
                    "original_module": module,
                }
            )
            self._matched_hook_modules.append(name)
            return True

        for name, module in named_modules.items():
            layer_num = self._match_layer(name)
            if layer_num is None:
                continue
            if not wrap_all_layers and layer_num != self.optimal_layer:
                continue
            parent_name, target_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name)
            install_wrapper(name, module, parent, target_name)

        if not self._matched_hook_modules:
            try:
                from vllm_metal.paged_attention_common import find_layers
            except Exception:
                find_layers = None

            if find_layers is not None:
                layers = find_layers(model)
                for layer_num, module in enumerate(layers):
                    if not wrap_all_layers and layer_num != self.optimal_layer:
                        continue
                    installed = False
                    for name, candidate in named_modules.items():
                        if candidate is not module or "." not in name:
                            continue
                        parent_name, target_name = name.rsplit(".", 1)
                        parent = named_modules.get(parent_name)
                        if install_wrapper(name, module, parent, target_name):
                            installed = True
                            break
                    if not installed:
                        install_wrapper(
                            f"layers.{layer_num}",
                            module,
                            layers,
                            layer_num,
                        )

        print(
            f"Installed {len(self._matched_hook_modules)} hooks on layers: "
            f"{self._matched_hook_modules}",
            flush=True,
        )

    def _parse_steering_config(self) -> Dict:
        """
        Load steering settings from `ACTSTEER_CONFIG`.

        Args:
            None.

        Returns:
            Dict: Parsed steering configuration with normalized field types.
        """
        config_path = os.environ.get("ACTSTEER_CONFIG", os.environ.get("VLLM_ACTSTEER_CONFIG"))

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
            getattr(self, "_capture_active", False)
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
        self._ensure_extension_state()
        if not self._steering_enabled():
            return output

        is_tuple = isinstance(output, tuple)
        if is_tuple:
            hidden_states, residuals = output
        else:
            hidden_states = None
            residuals = output

        layer_num = self._match_layer(_module_name)
        if layer_num != self.optimal_layer:
            return output

        if getattr(self, "_memory_guard_enabled", False):
            self._log_memory("hook-entry", _module_name, residuals)
            self._enforce_memory_guard("hook-entry", _module_name)
        if torch.is_tensor(residuals):
            residuals = self._apply_torch_steering(residuals)
        else:
            residuals = self._apply_mlx_steering(residuals)
        if getattr(self, "_memory_guard_enabled", False):
            self._log_memory("hook-exit", _module_name, residuals)
            self._enforce_memory_guard("hook-exit", _module_name)

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
        self._ensure_extension_state()
        self._capture_active = False
        hooks = getattr(self, "_hooks", None)
        if not hooks:
            self._hooks_installed = False
            return

        for entry in reversed(hooks):
            parent = entry["parent"]
            target_name = entry["target_name"]
            original_module = entry["original_module"]
            try:
                if isinstance(parent, list):
                    parent[target_name] = original_module
                else:
                    setattr(parent, target_name, original_module)
            except Exception as exc:
                print(
                    f"Error restoring Metal steering hook {target_name}: {exc}",
                    flush=True,
                )
        hooks.clear()
        self._hooks_installed = False
