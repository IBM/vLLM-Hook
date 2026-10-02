"""Spotlight worker: attention steering toward emphasized token spans."""
import logging
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context

from mia.runner import UnsupportedRunnerError
from mia.workers._common import (
    get_query_metadata,
    iter_matched_modules,
    match_attn,
)
from mia.utils.spotlight.utils import (
    compute_spotlight_bias,
    repeat_kv,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpotlightWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls."""

    if TYPE_CHECKING:
        model_runner: Any

    _default_hooks_on: str = "prefill"
    _spotlight_hooks_installed: bool = False

    def install_hooks(self):
        """Install Spotlight forward hooks on .attn modules."""
        if self._spotlight_hooks_installed:
            return
        self._spotlight_hooks_installed = True

        raise UnsupportedRunnerError(
            "SpotlightWorker is not supported on vLLM's V2 model runner. "
            "Supported worker kinds are: capture_hs, capture_qk, steer. "
            "Use MiaLLM(worker_name='capture_hs'), MiaLLM(worker_name='capture_qk'), or MiaLLM(worker_name='steer')."
        )

        model = getattr(self.model_runner, "model", None)
        if model is None:
            logger.warning("No model found; skipping Spotlight hook installation")
            return

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        num_h = int(getattr(text_cfg, "num_attention_heads", 32))
        num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
        hidden = int(getattr(text_cfg, "hidden_size", 4096))
        head_dim = hidden // num_h
        attn_scale = 1.0 / math.sqrt(head_dim)

        self._spotlight_conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attn_scale=attn_scale,
        )

        self._spotlight_batch_cache_key = None
        self._spotlight_batch_entries = None

        def _resolve_batch(req_ids, bs, query_start_loc, seq_lens):
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

            is_initial_prefill = torch.all(query_lens == seq_lens).item()
            if not is_initial_prefill:
                if not getattr(self, '_spotlight_chunked_warned', False):
                    if torch.any(query_lens > 1).item():
                        logger.warning(
                            "Spotlight steering disabled: chunked prefill detected. "
                            "Set enable_chunked_prefill=False for Spotlight to work."
                        )
                        self._spotlight_chunked_warned = True
                return []

            entries = []
            for i in range(bs):
                req_id = req_ids[i]
                req_state = self.model_runner.requests.get(req_id)
                if req_state is None or req_state.sampling_params is None:
                    entries.append(None)
                    continue
                extra = req_state.sampling_params.extra_args
                if not extra or "spotlight" not in extra:
                    entries.append(None)
                    continue
                spotlight_cfg = extra["spotlight"]
                span_ranges = spotlight_cfg.get("span_ranges", [])
                alpha = spotlight_cfg.get("alpha", 0.2)
                if not span_ranges:
                    entries.append(None)
                    continue
                entries.append({"span_ranges": span_ranges, "alpha": alpha})
            return entries

        def spotlight_attention_hook(module, input_tuple, output, *, module_name=None):
            """Forward hook that applies Spotlight bias during initial prefill."""
            if torch.cuda.is_current_stream_capturing():
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)
            if metadata is None:
                return None


            cache_key = id(metadata)
            if self._spotlight_batch_cache_key != cache_key:
                query_start_loc, seq_lens = get_query_metadata(metadata)
                if query_start_loc is None or seq_lens is None:
                    return None
                try:
                    req_ids = self.model_runner.input_batch.req_ids
                except Exception:
                    return None
                bs = len(query_start_loc) - 1
                self._spotlight_batch_cache_key = cache_key
                self._spotlight_batch_entries = _resolve_batch(
                    req_ids, bs, query_start_loc, seq_lens
                )
                self._spotlight_batch_qsl = query_start_loc

            entries = self._spotlight_batch_entries
            if not entries or not any(entries):
                return None

            try:
                Q = input_tuple[0]
                K = input_tuple[1]
                V = input_tuple[2]
                query_start_loc = self._spotlight_batch_qsl
                batch_size = len(entries)

                all_outputs = []
                for batch_idx in range(batch_size):
                    q_start = int(query_start_loc[batch_idx])
                    q_end = int(query_start_loc[batch_idx + 1])
                    qlen = q_end - q_start

                    params = entries[batch_idx]
                    if params is None:
                        all_outputs.append(output[q_start:q_end])
                        continue

                    span_ranges = params["span_ranges"]
                    alpha = params["alpha"]

                    Q_batch = Q[q_start:q_end]
                    K_batch = K[q_start:q_end]
                    V_batch = V[q_start:q_end]

                    orig_dtype = Q_batch.dtype

                    Q_batch = Q_batch.reshape(qlen, num_h, head_dim).transpose(0, 1).float()
                    K_batch = K_batch.reshape(qlen, num_kv, head_dim).transpose(0, 1).float()
                    V_batch = V_batch.reshape(qlen, num_kv, head_dim).transpose(0, 1).float()

                    if num_h != num_kv:
                        n_rep = num_h // num_kv
                        K_batch = repeat_kv(K_batch.unsqueeze(0), n_rep).squeeze(0)
                        V_batch = repeat_kv(V_batch.unsqueeze(0), n_rep).squeeze(0)

                    logits = torch.matmul(Q_batch, K_batch.transpose(-1, -2)) * attn_scale

                    causal_mask = torch.triu(
                        torch.full((qlen, qlen), float('-inf'),
                                   device=logits.device, dtype=torch.float32),
                        diagonal=1
                    )
                    logits = logits + causal_mask.unsqueeze(0)

                    logits_4d = logits.unsqueeze(0)
                    modified_weights = compute_spotlight_bias(
                        logits_4d, [span_ranges], target_proportion=alpha
                    )
                    modified_weights = modified_weights.squeeze(0)

                    attn_out = torch.matmul(modified_weights, V_batch)
                    attn_out = attn_out.transpose(0, 1).reshape(qlen, -1)
                    all_outputs.append(attn_out.to(orig_dtype))

                return torch.cat(all_outputs, dim=0)

            except Exception as e:
                logger.error(f"Spotlight hook error: {e}", exc_info=True)
                return None

        self._spotlight_hooks = []
        matched = []
        for name, module, _layer_num in iter_matched_modules(model, match_attn):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: spotlight_attention_hook(m, i, o, module_name=n)
            )
            self._spotlight_hooks.append(hook)
            matched.append(name)

        logger.info(f"Installed {len(self._spotlight_hooks)} Spotlight hooks on: {matched[:3]}...")

