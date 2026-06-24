"""Token Highlighter worker: capture (forward_attr) and mitigate in one class."""

from __future__ import annotations

import os
import json
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm_hook_plugins.utils.TokenHighlighter.utils import (
    ForwardAttrCapture,
    capture_ready,
    capture_ready_for_teacher,
    expand_suffix_capture,
    export_final_norm_spec,
    export_forward_attr_weights,
    flag_driver_tokens,
    iter_prompt_batches,
    lm_head_weight,
    locate_final_norm,
    load_highlighter_artifact,
    merge_real_and_teacher_captures,
    plan_teacher_prefill_extend,
    prefill_chunked_for_prompt,
    prompt_prefill_done,
    register_forward_attr_hooks,
    register_runtime_rope_probe_hooks,
    restore_teacher_prefill_state,
    save_highlighter_activations,
    slice_real_prefill_capture,
    slice_teacher_prefill_capture,
    stage_teacher_prefill_tokens,
    teacher_forced_token_ids,
    vllm_teacher_suffix_capture,
)

from vllm_hook_plugins.utils.TokenHighlighter.grad_influence import (
    affirmation_loss_grad,
    compute_mid_boundary_gradient,
    project_g_through_norm,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

# pyright: reportOperatorIssue=false, reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportOptionalCall=false


def _highlighter_active(scheduler_output: Any, model_runner: Any) -> bool:
    """True when any scheduled request sets ``extra_args['highlighter_mode']`` (probe-style gate)."""
    if scheduler_output is not None:
        for nrd in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            sp = getattr(nrd, "sampling_params", None)
            if sp and sp.extra_args and sp.extra_args.get("highlighter_mode"):
                return True
        for req_id in getattr(scheduler_output, "num_scheduled_tokens", {}) or {}:
            req = model_runner.requests.get(req_id)
            if req and req.sampling_params and req.sampling_params.extra_args:
                if req.sampling_params.extra_args.get("highlighter_mode"):
                    return True
    flag = os.environ.get("VLLM_HOOK_FLAG")
    if flag and os.path.exists(flag):
        return True
    return False


def _iter_highlighter_extras(
    scheduler_output: Any, model_runner: Any
):
    """Yield ``extra_args`` dicts for requests in the current scheduler step."""
    if scheduler_output is None:
        return
    for nrd in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        sp = getattr(nrd, "sampling_params", None)
        if sp and sp.extra_args:
            yield sp.extra_args
    for req_id in scheduler_output.num_scheduled_tokens:
        req = model_runner.requests.get(req_id)
        if req and req.sampling_params and req.sampling_params.extra_args:
            yield req.sampling_params.extra_args


def _apply_highlighter_config(worker: "HighlighterWorker", extra: dict) -> None:
    """Apply per-request highlighter settings from ``extra_args['highlighter']``."""
    hl = extra.get("highlighter")
    if isinstance(hl, str):
        try:
            hl = json.loads(hl)
        except (TypeError, ValueError):
            hl = None
    if not isinstance(hl, dict):
        return
    if "threshold_k" in hl:
        worker.threshold_k = float(hl["threshold_k"])
    if "beta" in hl:
        worker.soft_beta = float(hl["beta"])
    if "capture" in hl:
        worker._capture_style = str(hl["capture"]).strip().lower()
    if "mode" in hl:
        worker._driver_mode = str(hl["mode"])
    if "alpha" in hl:
        worker._driver_alpha = float(hl["alpha"])
    if "reselect_drivers" in hl:
        worker.reselect_drivers = bool(hl["reselect_drivers"])
    if "require_attn_metadata" in hl:
        worker._require_attn_metadata = bool(hl["require_attn_metadata"])
    if "allow_prerope_fallback" in hl:
        worker._allow_prerope_fallback = bool(hl["allow_prerope_fallback"])
    if ids := hl.get("target_token_ids"):
        worker.target_ids = [int(x) for x in ids]
        worker._target_tensor = None
    elif phrase := hl.get("target_phrase"):
        worker._target_phrase = str(phrase)
        worker.target_ids = []
        worker._target_tensor = None


def _sync_highlighter_paths(
    worker: "HighlighterWorker", scheduler_output: Any, model_runner: Any
) -> None:
    """Apply ``hook_dir`` / ``run_id`` / highlighter config from per-request ``extra_args``."""
    seen: set[tuple[str, str]] = set()
    for extra in _iter_highlighter_extras(scheduler_output, model_runner):
        _apply_highlighter_config(worker, extra)
        hook_dir = extra.get("hook_dir")
        run_id = extra.get("run_id")
        if not hook_dir and not run_id:
            continue
        key = (str(hook_dir), str(run_id))
        if key in seen:
            continue
        seen.add(key)
        if hook_dir:
            worker.hook_dir = hook_dir
        if hook_dir and run_id:
            run_id_file = extra.get("run_id_file") or os.path.join(hook_dir, "RUN_ID.txt")
            worker.run_id_file = run_id_file
            os.makedirs(hook_dir, exist_ok=True)
            with open(run_id_file, "a") as f:
                f.write(str(run_id) + "\n")


def _highlighter_mode(scheduler_output: Any, model_runner: Any) -> str:
    """``capture`` or ``mitigate`` from ``extra_args['highlighter_mode']`` (default ``capture``)."""
    for nrd in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        sp = getattr(nrd, "sampling_params", None)
        if sp and sp.extra_args:
            mode = sp.extra_args.get("highlighter_mode")
            if mode:
                return str(mode).strip().lower()
    if scheduler_output is not None:
        for req_id in scheduler_output.num_scheduled_tokens:
            req = model_runner.requests.get(req_id)
            if req and req.sampling_params and req.sampling_params.extra_args:
                mode = req.sampling_params.extra_args.get("highlighter_mode")
                if mode:
                    return str(mode).strip().lower()
    return "capture"


class HighlighterWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (HighlighterWorker,) at runtime, so ``self`` is
    the Worker instance. Registered as ``token_highlighter``.
    ``highlighter_mode`` in ``SamplingParams.extra_args`` indicates capture vs mitigate per request.

    **capture** — ``forward_attr`` (default): **extended** prefill schedules
    ``prompt + target[:-1]`` in one forward when the full prompt fits the first scheduler chunk.
    Suffix capture runs only when that is impossible (chunked prefill) or when a second
    teacher forward is required; each case emits an explicit warning with detailed reasoning. Q/K/V hooks follow
    ``ProbeHookQKWorker`` format (last-layer ``.attn`` inputs + ``get_query_metadata`` parsing).

    **mitigate** — β-scaled prompt embeddings on prefill (embedding hook), then decode.

    Callable via collective_rpc (``install_hooks``).
  """

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    _hooks_installed: bool = False
    _input_embeddings: nn.Module

    def install_hooks(self):
        """Wrap ``execute_model`` and install the mitigate embedding hook.

        Forward-attr + RoPE probe hooks are deferred to ``_ensure_capture_hooks``
        on the first capture request, once ``extra_args['highlighter']`` is available.
        This moves hook installation-specific parameters to the worker instance from the broader vLLM-Hook plugin.
        """
        if self._hooks_installed:
            return
        self._install_highlighter_state()
        if not getattr(self, "_highlighter_execute_wrapped", False):
            self._highlighter_execute_wrapped = True
            orig_execute = self.execute_model

                # Wrap the execute_model method with custom highlighter.
            def execute_model(*args, **kwargs):
                return self._highlighter_execute_model_step(orig_execute, *args, **kwargs)

            self.execute_model = execute_model
        self._hooks_installed = True

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """collective_rpc-compatible disk flush for save_to_disk requests.

        The generic hook plugin calls ``flush_disk`` after sync ``LLM.generate``
        when ``extra_args['save_to_disk']`` is set. Token Highlighter writes its
        own artifact in ``_finish_capture``; this method bridges that API by
        forcing a pending capture finalize against the caller-provided run path.
        """
        if hook_dir:
            self.hook_dir = hook_dir
        if self.hook_dir:
            os.makedirs(self.hook_dir, exist_ok=True)
        if run_id and self.hook_dir:
            self.run_id_file = os.path.join(self.hook_dir, "RUN_ID.txt")
            with open(self.run_id_file, "a") as f:
                f.write(str(run_id) + "\n")

        # If capture completion was deferred (common in transition from extended to suffix path),
        # finish now so the caller can immediately load highlighter_activations.pt.
        if self._pending and (self._capture_finish_pending or capture_ready(self._cap.live)):
            self._capture_finish_pending = False
            self._finish_capture()
            return True
        return False

    def _install_highlighter_state(self):
        """Attach persistent state and the mitigate embedding hook."""
        self.hook_dir = None
        self.run_id_file = None
        self._model = getattr(self.model_runner, "model", None)
        if self._model is None:
            raise RuntimeError("HighlighterWorker requires model.")

        self._device = next(self._model.parameters()).device
        # Default config values
        self.threshold_k = 2.0
        self.soft_beta = 0.4
        self._target_phrase: str | None = None
        self.reselect_drivers = False
        # Token Highlighter-specific hook installation parameters
        self._require_attn_metadata = False
        self._allow_prerope_fallback = False

        self._capture_hooks_installed = False
        self._pending_soft: list[dict] = []

        self._tokenizer = getattr(self.model_runner, "tokenizer", None)
        if self._tokenizer is None:
            tg = getattr(self.model_runner, "tokenizer_group", None)
            self._tokenizer = getattr(tg, "tokenizer", None) if tg else None

        self._target_tensor = None
        self.target_ids: list[int] = []
        self._cap = ForwardAttrCapture()
        self._pending: list[list[int]] = []
        self._scores_by_prompt: dict[tuple[int, ...], list[float]] = {}
        self._capture_finish_pending = False
        self._extended_req_ids: set[str] = set()
        self._capture_style = "extended"
        self._soft_indices_by_prompt: dict[tuple[int, ...], list[int]] = {}

        getter = getattr(self._model, "get_input_embeddings", None)
        emb: nn.Module | None
        if callable(getter):
            candidate = getter()
            emb = candidate if isinstance(candidate, nn.Module) else None
        else:
            raw = (
                getattr(getattr(self._model, "model", None), "embed_tokens", None)
                or getattr(self._model, "embed_tokens", None)
            )
            emb = raw if isinstance(raw, nn.Module) else None
        if emb is None:
            raise RuntimeError("Input embeddings not found in the model.")
        self._input_embeddings = emb
        # Register the embedding_soft_hook to the input embeddings. Note that
        # this is present even in "capture" mode but captured embeddings are only used in "mitigate" mode.
        self._input_embeddings.register_forward_hook(self._embedding_soft_hook)
        self._hooks: list = []

    def _ensure_capture_hooks(self) -> None:
        """Install forward-attr + RoPE probe hooks (once), using synced config."""
        if self._capture_hooks_installed:
            return
        cap = self._cap
        self._hooks.extend(
            register_forward_attr_hooks(
                self._model,
                cap.live,
                enabled=lambda: cap.active,
                model_runner=self.model_runner,
                meta=cap.meta,
                require_attn_metadata=self._require_attn_metadata,
                allow_prerope_fallback=self._allow_prerope_fallback,
            )
        )
        self._hooks.extend(
            register_runtime_rope_probe_hooks(
                self._model,
                meta=cap.meta,
                enabled=lambda: cap.active,
                require_attn_metadata=self._require_attn_metadata,
            )
        )
        self._capture_hooks_installed = True
        print("[highlighter] capture hooks installed (qkv + h + rope probe)")

    def _embedding_soft_hook(self, _module, _inputs, output):
        """Swap mitigated prompt embedding rows during mitigate prefill.

        Matches the incoming prefill token ids to a queued entry from ``_queue_soft_embeddings``.
        In-place ``copy_`` overwrites driver rows before layers write mitigated KV.
        """
        if not self._pending_soft or not torch.is_tensor(output):
            return output
        if not (_inputs and torch.is_tensor(_inputs[0])):
            return output
        ids = _inputs[0]
        token_ids = ids[0].tolist() if ids.dim() == 2 and ids.size(0) == 1 else ids.tolist()
        for idx, pending in enumerate(self._pending_soft):
            plen = pending["prompt_len"]
            if len(token_ids) < plen or token_ids[:plen] != pending["prompt_ids"]:
                continue
            seq_len = output.size(-2) if output.dim() == 3 else output.size(0)
            if output.dim() in (2, 3) and seq_len >= plen:
                # Copy beta-scaled driver rows to the output
                slot = output[:plen] if output.dim() == 2 else output[0, :plen]
                slot.copy_(pending["soft_prompt_embeds"])  # beta-scaled driver rows
                del self._pending_soft[idx]
                return output
        return output

    def _queue_soft_embeddings(
        self,
        prompt_ids: list[int],
        scores: list[float],
        *,
        soft_indices: list[int] | None = None,
    ) -> None:
        """Queue β-scaled driver embeddings for the next matching mitigate prefill.

        By default this uses analyzer-saved ``soft_indices`` from ``highlighter.pt``.
        Set ``highlighter.reselect_drivers`` in config to recompute from scores.
        """
        plen = len(prompt_ids)
        pt = torch.tensor(prompt_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        reselect = self.reselect_drivers
        if reselect or soft_indices is None:
            # Compute drivers from scores using the threshold_k parameter.
            drivers = flag_driver_tokens(scores, threshold_k=self.threshold_k)
        else:
            drivers = sorted({int(i) for i in soft_indices if 0 <= int(i) < plen})
        mode = "reselected" if (reselect or soft_indices is None) else "saved"

        rows = self._input_embeddings(pt).detach().squeeze(0).clone()
        if drivers: rows[drivers] = rows[drivers] * self.soft_beta
        soft = rows
        # Queue the beta-scaled driver embeddings for the next matching mitigate prefill.
        self._pending_soft.append(
            {"prompt_ids": list(prompt_ids), "prompt_len": plen, "soft_prompt_embeds": soft}
        )
        print(
            f"[highlighter] mitigate: soft-removal {len(drivers)}/{plen} "
            f"(beta={self.soft_beta}, drivers={mode})"
        )

    def _mitigate_scores_run_id(self) -> str | None:
        """Run id that owns ``highlighter.pt`` (from mitigate request ``extra_args``)."""
        for req in (getattr(self.model_runner, "requests", None) or {}).values():
            sp = getattr(req, "sampling_params", None)
            extra = sp.extra_args if sp else None
            if not extra or extra.get("highlighter_mode") != "mitigate":
                continue
            return extra.get("scores_run_id") or extra.get("run_id")
        return None

    def _load_scores(self) -> None:
        """Load ``highlighter.pt`` for mitigate (request run id, not latest RUN_ID.txt)."""
        if (self._scores_by_prompt or self._soft_indices_by_prompt) or not self.hook_dir:
            return
        run_id = self._mitigate_scores_run_id()
        if not run_id and self.run_id_file and os.path.exists(self.run_id_file):
            with open(self.run_id_file) as f:
                run_id = [ln.strip() for ln in f if ln.strip()][-1]
        if not run_id:
            return
        # Load the highlighter.pt artifact from the analyzer for the mitigate request.
        trace = load_highlighter_artifact(self.hook_dir, run_id, "highlighter.pt")
        if not trace:
            raise RuntimeError("No highlighter.pt found; run capture + analyze first.")
        # Load the token scores and soft indices from the highlighter.pt artifact.
        for seq in trace.get("sequences", []):
            key = tuple(seq.get("token_ids", []))
            if key and seq.get("token_scores"):
                self._scores_by_prompt[key] = seq["token_scores"]
                self._soft_indices_by_prompt[key] = list(seq.get("soft_indices", []))

    def _ensure_target_tensor(self) -> None:
        """Resolve affirmation ``target_ids`` from per-request ``highlighter`` config."""
        if self._target_tensor is not None:
            return
        if not self.target_ids:
            raw_ids = os.environ.get("VLLM_HIGHLIGHTER_TARGET_TOKEN_IDS", "")
            if raw_ids.strip():
                try:
                    self.target_ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
                except ValueError:
                    self.target_ids = []
        self._target_tensor = (
            torch.tensor(self.target_ids, dtype=torch.long, device=self._device).unsqueeze(0)
            if self.target_ids
            else None
        )

    def _reset_capture_style(self) -> None:
        """Use extended capture unless ``highlighter.capture`` requests suffix mode."""
        raw = str(getattr(self, "_capture_style", "extended")).strip().lower()
        if raw not in ("extended", "suffix"):
            warnings.warn(
                f"Unknown highlighter.capture={raw!r}; using extended capture.",
                UserWarning,
                stacklevel=2,
            )
            raw = "extended"
        if raw == "suffix":
            warnings.warn(
                "highlighter.capture=suffix: skipping extended prefill by request. "
                "Omit capture or set extended to use one-forward capture (recommended).",
                UserWarning,
                stacklevel=2,
            )
        self._capture_style = raw

    def _announce_suffix_fallback(self, reason: str, *, trigger: str) -> None:
        """Warn when suffix capture is required — always visible, never once-only."""
        banner = (
            "\n"
            "======== Token Highlighter: SUFFIX capture required ========\n"
            f"Trigger: {trigger}\n"
            f"Reason: {reason}\n"
            "Extended prefill (one forward on prompt + target[:-1]) was not possible.\n"
            "Fallback: scheduler prompt prefill, then a separate teacher forward on paged KV.\n"
            "To stay on extended: raise max_num_batched_tokens so the full prompt fits the\n"
            "first prefill chunk (see notebook capture section).\n"
            "==========================================================\n"
        )
        print(banner)
        warnings.warn(
            f"Token Highlighter suffix fallback ({trigger}): {reason}",
            UserWarning,
            stacklevel=3,
        )

    def _switch_to_suffix(self, reason: str, *, trigger: str) -> None:
        """Centralized style transition + warning for suffix fallback."""
        self._announce_suffix_fallback(reason, trigger=trigger)
        self._capture_style = "suffix"

    def _merge_with_suffix_capture(
        self, prompt_ids: list[int], real_capture: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run suffix teacher pass and merge with real prompt rows."""
        return merge_real_and_teacher_captures(
            len(prompt_ids), real_capture, self._suffix_teacher_capture(prompt_ids)
        )

    def _require_suffix_for_chunked_prefill(
        self, scheduler_output: Any, pending: list[list[int]]
    ) -> None:
        """Suffix is mandatory when vLLM chunks the first prefill (scheduled tokens < prompt len)."""
        if self._capture_style != "extended" or not self.target_ids:
            return
        for prompt_ids in pending:
            if not prefill_chunked_for_prompt(
                scheduler_output, self.model_runner, prompt_ids
            ):
                continue
            plen = len(prompt_ids)
            self._switch_to_suffix(
                f"Prompt length is {plen} tokens but the first prefill step schedules "
                "fewer tokens (vLLM chunked prefill / max_num_batched_tokens limit). "
                "Extended capture needs the full prompt in one scheduler step.",
                trigger="chunked_prefill",
            )
            return

    def _suffix_teacher_capture(self, prompt_ids: list[int]) -> dict[str, torch.Tensor]:
        """Run suffix teacher forward + hooks; return full-length tensors via ``expand_suffix_capture``.

        Temporary hooks avoid mixing with ``_cap.live`` during normal prefill. Prompt
        activations still come from ``slice_real_prefill_capture`` in ``_capture_for_prompt``.
        """
        captured: dict[str, torch.Tensor] = {}
        hooks = register_forward_attr_hooks(
            self._model,
            captured,
            enabled=lambda: True,
            require_attn_metadata=self._require_attn_metadata,
            allow_prerope_fallback=self._allow_prerope_fallback,
        )
        try:
            vllm_teacher_suffix_capture(
                self.model_runner, self._model, prompt_ids, self.target_ids
            )
        finally:
            for h in hooks:
                h.remove()
        if not capture_ready(captured):
            raise RuntimeError(
                "forward_attr: target suffix pass failed to capture Q/K/V/h_L."
            )
        return expand_suffix_capture(len(prompt_ids), captured)

    def _capture_for_prompt(self, prompt_ids: list[int]) -> dict[str, torch.Tensor]:
        """Build one ``prompt + target[:-1]`` activation dict for the analyzer.

        Extended: single slice from ``_cap.live``. Suffix: real prompt slice + teacher
        pass merged. Called from ``_finish_capture`` after prefill completes.
        """
        if (
            self.target_ids
            and self._capture_style == "extended"
            and capture_ready_for_teacher(prompt_ids, self.target_ids, self._cap.live)
        ):
            return slice_teacher_prefill_capture(
                prompt_ids, self.target_ids, self._cap.live, self._cap.meta
            )
        real = slice_real_prefill_capture(prompt_ids, self._cap.live, self._cap.meta)
        if not self.target_ids:
            return real
        if self._capture_style == "suffix":
            return self._merge_with_suffix_capture(prompt_ids, real)
        if self._capture_style == "extended" and capture_ready(self._cap.live):
            need = len(teacher_forced_token_ids(prompt_ids, self.target_ids))
            got = int(self._cap.live["h_L"].size(0))
            self._switch_to_suffix(
                f"Extended prefill captured {got} hook rows but need {need} "
                f"(prompt {len(prompt_ids)} + target {len(self.target_ids) - 1}). "
                "Running a second suffix teacher forward to complete activations.",
                trigger="incomplete_extended_sequence",
            )
            return self._merge_with_suffix_capture(prompt_ids, real)
        raise RuntimeError(
            "forward_attr: extended capture failed — hooks did not record Q/K/V during "
            "prefill. Ensure max_num_batched_tokens is large enough for "
            "len(prompt) + len(target) in one prefill step, or set "
            "highlighter.capture=suffix in the model config."
        )

    def _finish_capture(self) -> None:
        """Slice hooks, write ``highlighter_activations.pt``, clear pending state following capture.

        May run on the prefill step when extended capture is complete, or on the next
        idle step when ``_capture_finish_pending`` was set (suffix path).
        """
        self._ensure_target_tensor()
        # Precompute the affirmation-loss gradient g = dL/dh^{L} (post final-norm) at the
        # generation positions here, while the unembedding W_U is resident on-device. This
        # small [n_gen, d_model] tensor is all the analyzer needs from W_U, so we ship it in
        # the artifact instead of the ~hundreds-of-MB W_U matrix (avoids re-saving on every
        # capture and re-loading on every analyze, improving wall-clock time and efficiency).
        w_u = lm_head_weight(self._model)
        target_ids = list(self.target_ids)
        sequences = []
        for prompt_ids in self._pending:
            capture = self._capture_for_prompt(prompt_ids)
            cpu_capture = {k: v.detach().cpu() for k, v in capture.items()}
            if target_ids:
                g_loss = affirmation_loss_grad(
                    capture["h_L"],
                    len(prompt_ids),
                    target_ids,
                    w_u,
                    device=capture["h_L"].device,
                    model=self._model,
                )
                cpu_capture["g_loss"] = g_loss.detach().cpu()
                # FFN-aware boundary gradient at h_mid in the final decoder block.
                # This replaces naive g_out in the analyzer attention paths.
                start = len(prompt_ids) - 1

                # Compute the gradient at the boundary of the final FFN
                # Locate final norm module and project dL/dh_j through
                norm = locate_final_norm(self._model)
                g_out = g_loss
                if norm is not None:
                    norm_spec = export_final_norm_spec(norm)
                    if norm_spec is not None:
                        h_gen = capture["h_L"][start : start + len(target_ids)]
                        g_out = project_g_through_norm(
                            g_loss,
                            h_gen,
                            norm_spec,
                            capture["h_L"].device,
                            capture["h_L"].dtype,
                        )
                # Compute dL/dh_mid = dL/dh_out + J_norm2^T J_ffn^T dL/dh_out
                # where dL/dh_out is the gradient at the boundary of the final FFN
                # (affirmation loss gradient wrt direct last layer hidden states)
                g_mid = compute_mid_boundary_gradient(
                    capture,
                    len(prompt_ids),
                    g_out,
                    model=self._model,
                    device=capture["h_L"].device,
                )
                # Pass single gradient vector to analyzer via highlighter.pt for efficiency
                cpu_capture["g_mid"] = g_mid.detach().cpu()
            sequences.append({
                "token_ids": list(prompt_ids),
                "prompt_len": len(prompt_ids),
                "target_ids": target_ids,
                "capture": cpu_capture,
            })
        self._pending.clear()
        runtime_rope_probe = self._cap.meta.get("rope_runtime_probe")
        # Clear in place: the forward hooks captured references to these dicts at
        # install time. Rebinding to new dicts would orphan the hooks so they write
        # to a stale buffer while the worker reads an empty one.
        self._cap.live.clear()
        self._cap.meta.clear()
        weight_bundle = export_forward_attr_weights(
            self._model, runtime_rope_probe=runtime_rope_probe
        )
        save_highlighter_activations(
            self.hook_dir, self.run_id_file, sequences, weight_bundle=weight_bundle
        )

    def _highlighter_execute_model_step(
        self, super_execute_model, *args, **kwargs
    ):
        """Intercept scheduler steps for capture (extended/suffix) or mitigate (soft embeds).

        One vLLM scheduling step per call. Capture: bump scheduler for extended
        teacher prefill (default), run the worker forward, restore prompt boundary, finish when
        hooks hold a full teacher sequence. Mitigate: queue soft embeddings for mitigated prefill.
        """
        scheduler_output = args[0] if args else kwargs.get("execute_model_req")
        active = _highlighter_active(scheduler_output, self.model_runner)
        if active and scheduler_output is not None:
            _sync_highlighter_paths(self, scheduler_output, self.model_runner)
        n_tok = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        new_prompts = iter_prompt_batches(scheduler_output) if scheduler_output else []
        mode = (
            _highlighter_mode(scheduler_output, self.model_runner)
            if active and scheduler_output is not None
            else "mitigate"
        )

        # --- Mitigate: load scores, queue scaled embeddings, normal prefill + decode ---
        if active and mode == "mitigate":
            if new_prompts:
                self._load_scores()
                for prompt_ids in new_prompts:
                    scores = self._scores_by_prompt.get(tuple(prompt_ids))
                    if scores:
                        self._queue_soft_embeddings(
                            prompt_ids,
                            scores,
                            soft_indices=self._soft_indices_by_prompt.get(tuple(prompt_ids)),
                        )
            return super_execute_model(*args, **kwargs)

        # Deferred finish: some capture paths mark completion pending and need one
        # subsequent idle worker step to flush highlighter_activations.pt.
        #
        # Do not gate on mode=="capture": a common flow is capture -> analyze -> mitigate,
        # where the next worker step may already be in mitigate mode. If we only flush
        # in capture mode, the artifact can remain unwritten and analyzer sees no file.
        if (
            self._capture_finish_pending
            and getattr(self.model_runner, "execute_model_state", None) is None
        ):
            self._capture_finish_pending = False
            self._finish_capture()

        pending, cap = self._pending, self._cap
        runner = self.model_runner
        if active and mode == "capture":
            self._ensure_capture_hooks()
        if active and mode == "capture" and new_prompts:
            self._ensure_target_tensor()
            self._reset_capture_style()
            pending.extend(p for p in new_prompts if p not in pending)

        capturing = (
            active
            and mode == "capture"
            and n_tok > 0
            and (
                new_prompts
                or any(not prompt_prefill_done(runner, p) for p in pending)
            )
        )
        if capturing and new_prompts:
            # Clear in place (not rebind): hooks hold references to these dicts from
            # install time; rebinding orphans them and loses all captures.
            cap.live.clear()
            cap.meta.clear()

        if capturing and scheduler_output is not None:
            self._require_suffix_for_chunked_prefill(scheduler_output, pending)

        # --- Extended: patch scheduler + token row, then restore after forward ---
        extend_plans = []
        orig_update_states = None
        use_extended = capturing and self.target_ids and self._capture_style == "extended"
        if use_extended:
            extend_plans = plan_teacher_prefill_extend(
                scheduler_output,
                runner,
                pending,
                self.target_ids,
                self._extended_req_ids,
            )
            if extend_plans:
                orig_update_states = runner._update_states

                def _update_states_with_teacher_suffix(sched_out):
                    assert orig_update_states is not None
                    orig_update_states(sched_out)
                    stage_teacher_prefill_tokens(runner, extend_plans)

                runner._update_states = _update_states_with_teacher_suffix

        was_capturing = capturing
        cap.active = capturing
        try:
            out = super_execute_model(*args, **kwargs)
            # Batch meta is recorded from Q/K/V hooks during the forward (on_qkv).
        finally:
            cap.active = False
            if orig_update_states is not None:
                runner._update_states = orig_update_states
            if extend_plans:
                restore_teacher_prefill_state(runner, extend_plans)
                self._extended_req_ids.update(p.req_id for p in extend_plans)

        # --- Post-prefill: finish capture or defer suffix merge on next idle step ---
        if pending and mode == "capture":
            done = all(prompt_prefill_done(runner, p) for p in pending)
            if not done or n_tok == 0:
                return out
            if self.target_ids and all(
                capture_ready_for_teacher(p, self.target_ids, cap.live) for p in pending
            ):
                self._finish_capture()
            elif capture_ready(cap.live):
                # Extended path: prompt activations captured; teacher rows completed in
                # _capture_for_prompt via an announced suffix teacher forward if needed.
                self._capture_finish_pending = True
            elif was_capturing:
                raise RuntimeError(
                    "forward_attr: prefill ran with cap.active but last-layer Q/K/V hooks "
                    "recorded nothing (check [highlighter] capture hooks log at install)."
                )
        return out


HighlighterCaptureWorker = HighlighterWorker
HighlighterSoftWorker = HighlighterWorker
