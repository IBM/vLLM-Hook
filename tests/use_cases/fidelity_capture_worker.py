"""Test-only vLLM worker for final-normalized hidden-state fidelity scoring."""

import torch

from vllm.forward_context import get_forward_context
from vllm_hook_plugins.workers._common import get_query_metadata
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker


class FidelityCaptureWorker(SteerHookActWorker):
    """Steer normally and capture requested post-final-RMSNorm states."""

    def _install_hooks(self):
        super()._install_hooks()
        final_norm = self.model_runner.model.model.norm
        self._fidelity_final_norm_states = []

        def capture_final_norm(_module, _inputs, output):
            assert isinstance(output, torch.Tensor)
            metadata = get_forward_context().attn_metadata
            query_start_loc, _ = get_query_metadata(metadata)
            assert query_start_loc is not None
            req_ids = self.model_runner.input_batch.req_ids
            for i, req_id in enumerate(req_ids):
                request = self.model_runner.requests[req_id]
                extra = request.sampling_params.extra_args or {}
                if extra.get("fidelity_capture"):
                    end = int(query_start_loc[i + 1].item())
                    self._fidelity_final_norm_states.append(
                        output[end - 1].detach().cpu()
                    )
            return output

        final_norm.register_forward_hook(capture_final_norm)

    def pop_final_norm_last_token(self):
        states = self._fidelity_final_norm_states
        self._fidelity_final_norm_states = []
        if len(states) != 1:
            raise RuntimeError(
                f"expected one final-norm state, captured {len(states)}"
            )
        return states[0].clone()
