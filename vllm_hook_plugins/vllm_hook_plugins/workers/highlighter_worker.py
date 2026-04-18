import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.v1.worker.gpu_worker import Worker as V1Worker
from transformers import AutoModelForCausalLM, AutoTokenizer


class HighlighterWorker(V1Worker):
    """
    Prefill-time worker that computes gradients with 
    respect to the affirmation loss for each prompt.

    Record schema produced per prompt:
    {
      "token_ids": [...],
      "token_scores": [...]  # gradient L2 norm per prompt token
    }
    """

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self._model = getattr(self.model_runner, "model", None)
        if self._model is None:
            raise RuntimeError("HighlighterWorker requires model.")

        self._tokenizer = getattr(self.model_runner, "tokenizer", None)
        if self._tokenizer is None:
            tokenizer_group = getattr(self.model_runner, "tokenizer_group", None)
            self._tokenizer = getattr(tokenizer_group, "tokenizer", None) if tokenizer_group else None

        self._device = next(self._model.parameters()).device
        self.target_phrase = os.environ.get(
            "VLLM_HIGHLIGHTER_TARGET_PHRASE",
            "Sure! I can help with that",
        )
        self.threshold_k = float(os.environ.get("VLLM_HIGHLIGHTER_THRESHOLD_K", "2.0"))
        self.soft_beta = float(os.environ.get("VLLM_HIGHLIGHTER_BETA", "0.1"))
        self._target_tensor = None
        self.target_ids = []
        self._pending_soft = []
        self._grad_model = None
        self._grad_tokenizer = None
        self._grad_device = os.environ.get("VLLM_HIGHLIGHTER_GRAD_DEVICE", "cpu")
        getter = getattr(self._model, "get_input_embeddings", None)
        if callable(getter):
            self._input_embeddings = getter()
        elif getattr(getattr(self._model, "model", None), "embed_tokens", None) is not None:
            self._input_embeddings = self._model.model.embed_tokens
        elif getattr(self._model, "embed_tokens", None) is not None:
            self._input_embeddings = self._model.embed_tokens
        else:
            raise RuntimeError("Input embeddings not found in the model.")
        self._soft_hook = self._input_embeddings.register_forward_hook(self._embedding_soft_hook)

        return result

    def _embedding_soft_hook(self, _module, _inputs, output):
        # Apply soft-removal on the matching prefill prompt only.
        if not self._pending_soft or not torch.is_tensor(output):
            return output

        token_ids = None
        if _inputs and torch.is_tensor(_inputs[0]):
            ids = _inputs[0]
            if ids.dim() == 1:
                token_ids = ids.tolist()
            elif ids.dim() == 2 and ids.size(0) == 1:
                token_ids = ids[0].tolist()
        if token_ids is None:
            return output

        for idx, pending in enumerate(self._pending_soft):
            prompt_ids = pending["prompt_ids"]
            prompt_len = pending["prompt_len"]
            if len(token_ids) < prompt_len or token_ids[:prompt_len] != prompt_ids:
                continue

            soft_prompt = pending["soft_prompt_embeds"]
            if output.dim() == 2 and output.size(0) >= prompt_len:
                out = output
                out[:prompt_len, :] = soft_prompt
                del self._pending_soft[idx]
                return out
            if output.dim() == 3 and output.size(1) >= prompt_len:
                out = output
                out[0, :prompt_len, :] = soft_prompt
                del self._pending_soft[idx]
                return out

        return output

    def _flag_driver_tokens(self, scores, mode="mean_std"):
        if not scores:
            return []
        s = torch.tensor(scores, dtype=torch.float32)
        if mode == "top_percentage":
            alpha = float(os.environ.get("VLLM_HIGHLIGHTER_ALPHA", "0.25"))
            k = max(1, int(len(scores) * alpha))
            order = torch.argsort(s, descending=True)[:k].tolist()
            return sorted(order)
        threshold = s.mean() + self.threshold_k * s.std(unbiased=False)
        return [i for i, v in enumerate(scores) if float(v) > float(threshold)]

    def _ensure_grad_model(self):
        if self._grad_model is not None:
            return
        model_cfg = getattr(getattr(self.model_runner, "vllm_config", None), "model_config", None)
        model_id = getattr(model_cfg, "model", None)
        if not model_id:
            raise RuntimeError("Unable to resolve model id for autograd scorer.")

        snapshot_path = model_id
        snapshots_dir = os.path.join(model_id, "snapshots") if isinstance(model_id, str) else ""
        if isinstance(model_id, str) and os.path.isdir(snapshots_dir):
            entries = sorted(os.listdir(snapshots_dir))
            if entries:
                snapshot_path = os.path.join(snapshots_dir, entries[-1])
        if not isinstance(snapshot_path, str) or not os.path.exists(snapshot_path):
            raise RuntimeError(f"Resolved snapshot path does not exist: {snapshot_path}")

        # Load scorer model and tokenizer from the same snapshot path as serving model.
        self._grad_model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(self._grad_device)
        self._grad_tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._grad_model.eval()
        for param in self._grad_model.parameters():
            param.requires_grad_(False)
        if self._tokenizer is None: self._tokenizer = self._grad_tokenizer

    def _score_prompt_tokens_autograd(self, prompt_ids):
        self._ensure_grad_model()
        target = self._target_tensor.to(self._grad_device)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self._grad_device).unsqueeze(0)
        prompt_len = prompt_tensor.size(1)
        target_prefix = target[:, :-1]
        input_ids = torch.cat([prompt_tensor, target_prefix], dim=1)

        with torch.enable_grad():
            embed_layer = self._grad_model.get_input_embeddings()
            all_embeds = embed_layer(input_ids).detach()
            prompt_embeds = all_embeds[:, :prompt_len, :].clone().requires_grad_(True)
            suffix_embeds = all_embeds[:, prompt_len:, :]
            inputs_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1)
            attention_mask = torch.ones(
                (1, inputs_embeds.size(1)), dtype=torch.long, device=self._grad_device
            )

            outputs = self._grad_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            start = prompt_len - 1
            end = start + target.size(1)
            target_logits = logits[:, start:end, :].contiguous()

            # Paper-equivalent affirmation loss: -log P(y|x).
            token_log_probs = F.log_softmax(target_logits, dim=-1)
            selected_log_probs = token_log_probs.gather(
                dim=-1, index=target.unsqueeze(-1)
            ).squeeze(-1)
            affirmation_loss = -selected_log_probs.sum()
            affirmation_loss.backward()

            grad = prompt_embeds.grad
            if grad is None:
                raise RuntimeError("Autograd failed: prompt embedding gradient is None.")
            gradient_norms = grad.norm(p=2, dim=-1).squeeze(0)
            return gradient_norms.detach().cpu().tolist()

    def _compute_affirmation_gradients(self, request) -> None:
        if self._target_tensor is None:
            target_ids_env = os.environ.get("VLLM_HIGHLIGHTER_TARGET_TOKEN_IDS", "").strip()
            if target_ids_env:
                self.target_ids = [int(tok.strip()) for tok in target_ids_env.split(",") if tok.strip()]
            else:
                self._ensure_grad_model()
                if self._grad_tokenizer is not None:
                    self.target_ids = self._grad_tokenizer.encode(self.target_phrase, add_special_tokens=False)
                elif self._tokenizer is not None:
                    self.target_ids = self._tokenizer.encode(self.target_phrase, add_special_tokens=False)
                else:
                    raise RuntimeError(
                        "Tokenizer is unavailable in this worker process and "
                        "VLLM_HIGHLIGHTER_TARGET_TOKEN_IDS was not provided."
                    )
            if not self.target_ids:
                raise ValueError(f"Target phrase '{self.target_phrase}' is empty or invalid.")
            self._target_tensor = torch.tensor(
                self.target_ids, dtype=torch.long, device=self._device
            ).unsqueeze(0)

        embed_layer = self._input_embeddings
        run_id = None
        if self.run_id_file and os.path.exists(self.run_id_file):
            with open(self.run_id_file, "r") as f:
                ids = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
            run_id = ids[-1] if ids else None

        sequences = []  # Store token ids and per-token gradient norms/scores.

        prompt_batches = []
        for metadata in getattr(request, "seq_group_metadata_list", []):
            prompt_ids = getattr(metadata, "prompt_token_ids", None)
            if prompt_ids:
                prompt_batches.append(prompt_ids)
        for new_req in getattr(request, "scheduled_new_reqs", []):
            prompt_ids = getattr(new_req, "prompt_token_ids", None)
            if prompt_ids:
                prompt_batches.append(prompt_ids)

        with torch.inference_mode(False), torch.enable_grad():
            for prompt_ids in prompt_batches:
                prompt_tensor = torch.tensor(
                    prompt_ids, dtype=torch.long, device=self._device
                ).unsqueeze(0)
                prompt_len = prompt_tensor.size(1)
                prompt_embeds = embed_layer(prompt_tensor).detach()

                # Use true autograd (via separate grad model) for prompt-token influence scores.
                score_list = self._score_prompt_tokens_autograd(prompt_ids)
                soft_indices = self._flag_driver_tokens(score_list, mode=os.environ.get("VLLM_HIGHLIGHTER_MODE", "mean_std"))
                soft_prompt_embeds = self._apply_soft_removal(prompt_embeds, score_list)
                self._pending_soft.append({
                    "prompt_ids": list(prompt_ids),
                    "prompt_len": prompt_len,
                    "soft_prompt_embeds": soft_prompt_embeds,
                })
                print(
                    f"[highlighter] soft-removal applied to {len(soft_indices)}/{prompt_len} prompt tokens "
                    f"(beta={self.soft_beta})"
                )

                record = {
                    "token_ids": list(prompt_ids),
                    "token_scores": score_list,
                    "soft_indices": soft_indices,
                    "soft_beta": self.soft_beta,
                    "applied_mode": os.environ.get("VLLM_HIGHLIGHTER_MODE", "top_percentage"),
                    "applied_threshold_k": self.threshold_k,
                    "applied_alpha": float(os.environ.get("VLLM_HIGHLIGHTER_ALPHA", "0.25")),
                }
                if self._tokenizer is not None:
                    record["tokens"] = [
                        self._tokenizer.decode([tid], skip_special_tokens=False)
                        for tid in prompt_ids
                    ]

                sequences.append(record)

        self._model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if run_id and self.hook_dir and sequences:
            run_dir = os.path.join(self.hook_dir, run_id, "tp_rank_0")
            os.makedirs(run_dir, exist_ok=True)
            torch.save(
                {"run_id": run_id, "sequences": sequences},
                os.path.join(run_dir, "highlighter.pt"),
            )
        
    def _apply_soft_removal(self, prompt_embeds, score_list):
        # prompt_embeds: [1, prompt_len, hidden]
        soft_indices = set(self._flag_driver_tokens(score_list, mode=os.environ.get("VLLM_HIGHLIGHTER_MODE", "mean_std")))
        detach_prompt = prompt_embeds.detach().squeeze(0).clone()
        prompt_len = detach_prompt.size(0)
        soft_rows = [
            detach_prompt[i] * self.soft_beta if i in soft_indices else detach_prompt[i]
            for i in range(prompt_len)
        ]
        return torch.stack(soft_rows, dim=0)

    def shutdown(self) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        super().shutdown()


    def execute_model(self, *args, **kwargs):
        request = args[0] if args else kwargs.get("execute_model_req")
        has_prefill_prompts = False
        if request is not None:
            has_prefill_prompts = any(
                getattr(metadata, "prompt_token_ids", None)
                for metadata in getattr(request, "seq_group_metadata_list", [])
            ) or any(
                getattr(new_req, "prompt_token_ids", None)
                for new_req in getattr(request, "scheduled_new_reqs", [])
            )

        if has_prefill_prompts and self.hook_flag and os.path.exists(self.hook_flag):
            self._compute_affirmation_gradients(request)

        # Execute the model
        return super().execute_model(*args, **kwargs)
