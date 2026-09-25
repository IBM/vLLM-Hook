"""The one place MIA knows what a vLLM model runner looks like.

MIA targets vLLM 0.29's V2 model runner (`vllm.v1.worker.gpu.model_runner`) and nothing
else. Two V2 facts drive this module:

1. `InputBatch` is TRANSIENT — built and returned by `prepare_inputs`, never stored on
   the runner. So per-step state is captured once, at the wrapper, into an immutable
   `StepView` that is handed down explicitly. Nothing below this seam reaches into the
   runner.
2. `sampling_params.extra_args` — how a caller asks for capture or steering — reaches
   the runner only inside `add_requests(scheduler_output)` and is then dropped. MIA
   stashes it there, keyed by request id, and prunes on finish.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import torch

from mia.errors import MiaConfigurationError

_V2_MODULE_PREFIX = "vllm.v1.worker.gpu."
_STASH_ATTR = "_mia_arg_stash"


class UnsupportedRunnerError(MiaConfigurationError):
    """Raised when MIA is installed against a runner it does not support.

    Loud by design: the failure mode this replaces is silent — a V1 runner would
    capture nothing and steer nothing while reporting success.

    A ``MiaConfigurationError`` (hence a ``MiaRefusal``, and still a ``RuntimeError``) so the
    defensive handler in ``graph/install.py::patch_worker_load_model`` re-raises it instead of
    degrading to no-capture — which is what it did, silently, in graph mode. See mia/errors.py.
    """


def is_v2_runner(runner) -> bool:
    """True iff `runner` is vLLM's V2 GPUModelRunner.

    Discriminated by module path: V2 lives in the package `vllm.v1.worker.gpu.*`,
    V1 in the module `vllm.v1.worker.gpu_model_runner`. Both classes are named
    `GPUModelRunner`, so the name alone tells you nothing.
    """
    return type(runner).__module__.startswith(_V2_MODULE_PREFIX)


def require_v2_runner(runner) -> None:
    if not is_v2_runner(runner):
        raise UnsupportedRunnerError(
            "MIA requires vLLM's V2 model runner (vllm.v1.worker.gpu.model_runner), but "
            f"the live runner is {type(runner).__module__}.{type(runner).__name__}. "
            "Unset VLLM_USE_V2_MODEL_RUNNER (or set it to 1) and use vLLM 0.29.0."
        )


def install_request_arg_stash(runner) -> dict[str, dict]:
    """Keep `sampling_params.extra_args` alive past `add_requests`. Idempotent."""
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
    """Everything MIA needs about ONE step, read once, immutable thereafter.

    Index `i` is a batch row throughout: `req_ids[i]` owns `num_scheduled_tokens[i]`
    tokens starting at `query_start_loc_np[i]`.
    """

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
    """Snapshot the transient V2 `InputBatch` into a `StepView`.

    Every array is sliced to `num_reqs`: V2's buffers are allocated at `max_num_reqs`
    and the tail rows hold stale values from previous steps.

    `prompt_len_np` is NOT one of `InputBatch`'s own fields (its `prompt_lens` field is
    `None` outside R-SWA). It lives on the runner's persistent `req_states`, indexed by a
    per-request SLOT that is a different numbering than the batch row `i` -- `input_batch
    .idx_mapping_np[i]` is the slot for row `i` (the same indirection `model_runner.py`
    itself uses for `prefill_len_np`/`num_computed_prefill_tokens`), so row `i`'s prompt
    length is `runner.req_states.prompt_len.np[idx_mapping_np[i]]`.
    """
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
