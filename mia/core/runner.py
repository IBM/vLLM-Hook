"""Adapter that isolates every vLLM V2 model-runner access MIA makes."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import torch

from mia.errors import MiaConfigurationError

_V2_MODULE_PREFIX = "vllm.v1.worker.gpu."
_STASH_ATTR = "_mia_arg_stash"


class UnsupportedRunnerError(MiaConfigurationError):
    """Raised when MIA is installed against a runner it does not support."""


def is_v2_runner(runner) -> bool:
    """True iff `runner` is vLLM's V2 GPUModelRunner."""
    return type(runner).__module__.startswith(_V2_MODULE_PREFIX)


def require_v2_runner(runner) -> None:
    if not is_v2_runner(runner):
        raise UnsupportedRunnerError(
            "MIA requires vLLM's V2 model runner (vllm.v1.worker.gpu.model_runner), but "
            f"the live runner is {type(runner).__module__}.{type(runner).__name__}. "
            "Unset VLLM_USE_V2_MODEL_RUNNER (or set it to 1) and use vLLM 0.29.0."
        )


def install_request_arg_stash(runner) -> dict[str, dict]:
    """Keep `sampling_params.extra_args` alive past `add_requests`."""
    existing = getattr(runner, _STASH_ATTR, None)
    if existing is not None:
        return existing

    stash: dict[str, dict] = {}
    setattr(runner, _STASH_ATTR, stash)

    original_add = runner.add_requests
    original_finish = runner.finish_requests

    def add_requests(scheduler_output):
        for new_req in scheduler_output.scheduled_new_reqs:
            params = getattr(new_req, "sampling_params", None)
            extra = getattr(params, "extra_args", None) if params is not None else None
            if extra:
                stash[new_req.req_id] = extra
        return original_add(scheduler_output)

    def finish_requests(scheduler_output):
        for req_id in scheduler_output.finished_req_ids:
            stash.pop(req_id, None)
        return original_finish(scheduler_output)

    runner.add_requests = add_requests
    runner.finish_requests = finish_requests
    return stash


@dataclasses.dataclass(frozen=True)
class StepView:
    """Immutable per-step snapshot of everything MIA reads from the runner."""

    req_ids: list[str]
    num_reqs: int
    num_scheduled_tokens: np.ndarray
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    num_computed_tokens_np: np.ndarray
    prefill_len_np: np.ndarray
    prompt_len_np: np.ndarray
    is_prefilling_np: np.ndarray
    seq_lens: torch.Tensor
    block_tables: tuple[torch.Tensor, ...]
    extra_args: Mapping[str, dict]

    def extra_args_for(self, index: int) -> dict | None:
        """Stashed extra_args for batch row `index`, or None."""
        return self.extra_args.get(self.req_ids[index])


def step_view(runner, input_batch, stash: Mapping[str, dict]) -> StepView:
    """Snapshot the transient V2 `InputBatch` into a `StepView`."""
    n = int(input_batch.num_reqs)
    block_tables = getattr(getattr(runner, "block_tables", None), "input_block_tables", ())
    idx_mapping_np = input_batch.idx_mapping_np[:n]
    return StepView(
        req_ids=list(input_batch.req_ids),
        num_reqs=n,
        num_scheduled_tokens=input_batch.num_scheduled_tokens[:n],
        query_start_loc=input_batch.query_start_loc[: n + 1],
        query_start_loc_np=input_batch.query_start_loc_np[: n + 1],
        num_computed_tokens_np=input_batch.num_computed_tokens_np[:n],
        prefill_len_np=input_batch.prefill_len_np[:n],
        prompt_len_np=runner.req_states.prompt_len.np[idx_mapping_np],
        is_prefilling_np=input_batch.is_prefilling_np[:n],
        seq_lens=input_batch.seq_lens[:n],
        block_tables=tuple(bt[:n] for bt in block_tables),
        extra_args=stash,
    )

