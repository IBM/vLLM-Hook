"""Unified worker: interventions, capture, and gates on one engine.

Mixin injected into vLLM's GPU Worker via worker_extension_cls; self is
the Worker instance and every public method is callable via
collective_rpc. Selected with VLLM_HOOK_WORKER=unified; the env default
remains ``hidden_states``. Legacy extra_args keys are not served here.

Layer indices on this surface are the 0-based decoder-layer indices in
model order (``model.layers.N`` is layer N), unlike the legacy
``output_hidden_states`` path which exposes 1-based numbers.
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from vllm.forward_context import get_forward_context

from vllm_hook_plugins.core import __version__ as _PLUGIN_VERSION
from vllm_hook_plugins.core.artifacts import ArtifactRegistry
from vllm_hook_plugins.core.canonical import canonical_bytes, spec_hash
from vllm_hook_plugins.core.fingerprints import (
    chat_template_fingerprint,
    config_fingerprint,
    tokenizer_fingerprint,
)
from vllm_hook_plugins.core.interpreter import apply_op, build_gate, scope_rows
from vllm_hook_plugins.core.kinds import (
    ARTIFACT_TENSORS,
    CAPTURE_KINDS,
    CAPTURE_LOCATIONS,
    CAPTURE_MODES,
    CONSTRAINTS,
    GATE_KINDS,
    MODIFIER_KINDS,
    SCOPE_KINDS,
    TRANSFORM_KINDS,
)
from vllm_hook_plugins.core.schema import (
    E_BAD_PARAM,
    E_CONSTRAINT,
    CaptureSpec,
    InterventionSpec,
    SpecError,
    parse_capture,
    parse_intervention_spec,
    parse_processor_spec,
)
from vllm_hook_plugins.workers._common import (
    iter_matched_modules,
    iter_matching_req_ids,
    match_layer,
    match_o_proj,
    save_safetensors_atomic,
)
from vllm_hook_plugins.workers.positions import PositionTracker, build_pass_views

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

logger = logging.getLogger("vllm_hook.unified_worker")

# The three extra_args keys this worker serves.
NEW_SURFACE_KEYS = ("intervention_spec", "processor_spec", "capture")

# Loaded-artifact memo size (distinct artifacts, not bytes; each entry is a
# small steering vector / probe, bounded by VLLM_HOOK_MAX_ARTIFACT_MB).
_ARTIFACT_CACHE_SIZE = 128


def _decode_spec_obj(value, key: str):
    """extra_args carries specs as JSON strings on serve and as dicts
    offline; accept both.
    """
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise SpecError(E_BAD_PARAM, key, f"{key} is not valid JSON: {exc}")
    raise SpecError(E_BAD_PARAM, key, f"{key} must be a JSON object or JSON-encoded string")


class _StagedRequest:
    """Everything the hooks need for one request: parsed specs, resolved
    artifact tensors, gate state machines, and capture buffers.
    """

    def __init__(self, spec: InterventionSpec | None, capture: CaptureSpec | None, artifacts: dict):
        self.spec = spec
        self.capture = capture
        self.artifacts = artifacts
        self.spec_hash = None
        self.key = None  # id this state is staged under
        self.gates = {}  # op index -> GateState
        self.ops_by_layer = {}  # layer -> [(op index, OpSpec)] for residual transforms
        self.head_ops_by_layer = {}  # layer -> [(op index, OpSpec)] for head_additive
        self.condition_layers = frozenset()
        self.capture_buffers = {}  # (layer, location) -> {absolute position: cpu row}
        self._reset_key = None

        if spec is not None:
            self.condition_layers = spec.condition_layers()
            for op_index, op in enumerate(spec.ops):
                if op.gate is not None:
                    self.gates[op_index] = build_gate(op.gate, artifacts)
                by_layer = self.head_ops_by_layer if op.transform_kind == "head_additive" else self.ops_by_layer
                for layer in op.layers:
                    by_layer.setdefault(layer, []).append((op_index, op))

    def ops_for(self, layer: int) -> list:
        return self.ops_by_layer.get(layer, ())

    def head_ops_for(self, layer: int) -> list:
        return self.head_ops_by_layer.get(layer, ())

    def feed_conditions(self, layer: int, view, stream_rows: torch.Tensor) -> None:
        """Ingest a condition-layer reading (materialized layer input) into
        every gate; each gate filters on its own condition_layers.
        """
        for gate in self.gates.values():
            gate.observe(layer, view.positions, stream_rows)

    def gate_open(self, op_index: int, op, view) -> bool:
        if op.gate is None:
            return True
        gate = self.gates[op_index]
        gate.note_pass(view.positions, view.prompt_len)
        return gate.decision() is True

    def maybe_reset(self, pass_key) -> None:
        """Reset gate evidence once per restarted pass. Hooks fire at every
        layer of the pass, so the reset is keyed to the pass and applied on
        the first hook that sees it. Capture buffers are position-keyed and
        overwrite on replay, so they are left intact — a recompute that
        skips prefix-cached positions must not lose rows already captured.
        """
        if self._reset_key == pass_key:
            return
        self._reset_key = pass_key
        for gate in self.gates.values():
            gate.reset()

    def wants_capture(self, layer: int, location: str) -> bool:
        if self.capture is None or self.capture.location != location:
            return False
        return self.capture.layers is None or layer in self.capture.layers

    def record_capture(self, layer: int, location: str, view, stream_rows: torch.Tensor) -> None:
        buffer = self.capture_buffers.setdefault((layer, location), {})
        if self.capture.mode == "last_token":
            buffer[view.positions[-1]] = stream_rows[-1].detach().to("cpu", copy=True)
            return
        rows = stream_rows.detach().to("cpu", copy=True)
        for idx, position in enumerate(view.positions):
            buffer[position] = rows[idx]


class UnifiedHookWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (UnifiedHookWorker,) at runtime,
    so self is the Worker instance. Methods are callable via collective_rpc.
    """

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    _hooks_installed: bool = False

    # -----------------------------------------------------------------
    # RPC surface
    # -----------------------------------------------------------------

    def install_hooks(self):
        """Install layer hooks (and o_proj pre-hooks when TP==1).
        Idempotent; called lazily by the plugin on first new-surface
        request.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        try:
            self._install_hooks()
            logger.info("unified hooks installed")
        except Exception as exc:
            logger.exception("unified hook installation failed: %s", exc)

    def prepare_requests(self, specs: dict) -> dict:
        """Validate and stage per-request state before scheduling.

        ``specs`` maps req_id -> {"intervention_spec": str|None,
        "capture": str|None} (JSON strings as carried in extra_args).
        Parses via ``core.schema``, resolves artifacts through the
        registry cache, builds gate states. Returns req_id -> None on
        success or the ``SpecError`` payload (code, path, msg) on
        failure; a failed request is not staged.
        """
        self.install_hooks()
        results = {}
        for req_id, entry in specs.items():
            try:
                self._stage_request(req_id, dict(entry or {}))
                results[req_id] = None
            except SpecError as err:
                results[req_id] = err.payload()
                logger.warning("rejected request %s: %s", req_id, err)
            except Exception as exc:
                # An unexpected validation error must reject this request,
                # never the batch it was prepared with.
                results[req_id] = SpecError(E_BAD_PARAM, "", f"unexpected error validating spec: {exc}").payload()
                logger.exception("unexpected error staging request %s", req_id)
        return results

    def get_capture(self, external_req_id: str, expected_len: int | None = None):
        """Return (manifest_json, safetensors_bytes) for a finished
        capture request, or None. Manifest: layers, location, mode,
        positions covered, dtype, shapes. Tensors are assembled from the
        position-keyed buffers, sorted by position, truncated to the
        request's reported token count.

        ``expected_len`` is that count (prompt + generated - 1, matching
        the number of forward passes with real tokens); the vLLM v1
        post-stop extra pass writes one surplus position that this drops.
        None skips truncation. When the capture spec carries ``save_dir``,
        the assembled artifact is written there as
        ``capture_{req_id}.safetensors`` instead and None is returned.
        """
        if not getattr(self, "_should_capture", False):
            return None
        for req_id in iter_matching_req_ids(self._staged, external_req_id):
            state = self._staged[req_id]
            if state.capture is None:
                return None
            capture = state.capture
            layers = list(capture.layers) if capture.layers is not None else sorted(self._hooked_layers)

            flat = {}
            positions_by_layer = {}
            shapes = {}
            dtype = None
            for layer in layers:
                buffer = state.capture_buffers.get((layer, capture.location), {})
                positions = sorted(p for p in buffer if expected_len is None or p < expected_len)
                if not positions:
                    continue
                stacked = torch.stack([buffer[p] for p in positions])
                flat[f"layer_{layer}"] = stacked
                positions_by_layer[str(layer)] = positions
                shapes[str(layer)] = list(stacked.shape)
                dtype = str(stacked.dtype).removeprefix("torch.")

            if not flat:
                # This rank captured nothing (e.g. a pipeline stage that owns
                # none of the requested layers); the plugin merges the ranks
                # that did.
                return None
            manifest = {
                "layers": layers,
                "location": capture.location,
                "mode": capture.mode,
                "positions": positions_by_layer,
                "dtype": dtype,
                "shapes": shapes,
            }
            if capture.save_dir:
                # Under pipeline parallelism each stage holds its own layers;
                # non-zero ranks get a suffixed basename so stages sharing a
                # filesystem never clobber each other.
                basename = f"capture_{external_req_id}"
                if self.rank != 0:
                    basename = f"{basename}.rank{self.rank}"
                os.makedirs(capture.save_dir, exist_ok=True)
                save_safetensors_atomic(flat, manifest, capture.save_dir, basename)
                return None
            import safetensors.torch
            return (json.dumps(manifest), safetensors.torch.save(flat))
        return None

    def clear_request(self, external_req_id: str):
        """Drop staged spec state, gate state, capture buffers, and the
        position tracker entry. Called from the plugin's finally block;
        an LRU cap (VLLM_HOOK_STATE_LRU, default 4096 requests) is the
        safety net for ids that never receive cleanup.
        """
        if not self._hooks_installed:
            return
        for req_id in iter_matching_req_ids(self._staged, external_req_id):
            del self._staged[req_id]
            self._tracker.drop(req_id)
        for alias, target in list(self._alias.items()):
            if target not in self._staged:
                del self._alias[alias]
        self._rejected.discard(external_req_id)

    def hook_capabilities(self) -> dict:
        """The discovery payload, rank 0."""
        if self.rank != 0:
            return None
        self.install_hooks()
        model_config = self.model_runner.model_config
        cache_config = getattr(self, "cache_config", None)
        return {
            "plugin_version": _PLUGIN_VERSION,
            "vllm_version": self._vllm_version(),
            "active_worker": "unified",
            "intervention_kinds": {
                "transforms": sorted(TRANSFORM_KINDS),
                "modifiers": sorted(MODIFIER_KINDS),
                "scopes": sorted(SCOPE_KINDS),
                "gates": sorted(GATE_KINDS),
                "constraints": dict(CONSTRAINTS),
            },
            "processor_kinds": {"processors": []},
            "capture_kinds": {
                "kinds": sorted(CAPTURE_KINDS),
                "locations": sorted(CAPTURE_LOCATIONS),
                "modes": sorted(CAPTURE_MODES),
            },
            "artifact_transports": ["shared_fs"],
            "engine": {
                "enforce_eager": bool(getattr(model_config, "enforce_eager", True)),
                "prefix_caching": bool(getattr(cache_config, "enable_prefix_caching", False)),
                "speculative_decoding": getattr(self, "speculative_config", None) is not None,
                "tensor_parallel_size": self.parallel_config.tensor_parallel_size,
                "pipeline_parallel_size": self.parallel_config.pipeline_parallel_size,
            },
            "model": {
                "id": model_config.model,
                **self._model_fingerprints(model_config),
                "num_layers": self._num_layers,
                "hidden_size": self._hidden_size,
                "num_attention_heads": self._num_heads,
                "head_dim": self._head_dim,
                "dtype": str(model_config.dtype).removeprefix("torch."),
            },
        }

    # -----------------------------------------------------------------
    # Staging
    # -----------------------------------------------------------------

    def _stage_request(self, req_id: str, entry: dict) -> None:
        intervention_obj = _decode_spec_obj(entry.get("intervention_spec"), "intervention_spec")
        capture_obj = _decode_spec_obj(entry.get("capture"), "capture")
        processor_obj = _decode_spec_obj(entry.get("processor_spec"), "processor_spec")

        if processor_obj is not None:
            # No processor kinds are served yet; this always rejects.
            parse_processor_spec(processor_obj, allowed_processors=frozenset())

        spec = None
        artifacts = {}
        if intervention_obj is not None:
            spec = parse_intervention_spec(intervention_obj, num_layers=self._num_layers, allowed_gates=GATE_KINDS)
            self._check_constraints(spec)
            artifacts = self._resolve_artifacts(spec)

        capture = None
        if capture_obj is not None:
            capture = parse_capture(capture_obj, num_layers=self._num_layers)

        if spec is None and capture is None:
            return

        state = _StagedRequest(spec, capture, artifacts)
        state.key = req_id
        if intervention_obj is not None:
            state.spec_hash = spec_hash(intervention_obj)
        self._staged[req_id] = state
        self._staged.move_to_end(req_id)
        while len(self._staged) > self._state_lru:
            evicted, _ = self._staged.popitem(last=False)
            self._tracker.drop(evicted)
            logger.warning("state LRU evicted request %s before cleanup", evicted)
        logger.info(
            "staged request %s spec=%s artifacts=%s capture=%s",
            req_id,
            state.spec_hash,
            list(artifacts),
            capture is not None,
        )

    def _check_constraints(self, spec: InterventionSpec) -> None:
        if self.parallel_config.tensor_parallel_size == 1:
            return
        for op_index, op in enumerate(spec.ops):
            if op.transform_kind in CONSTRAINTS:
                raise SpecError(
                    E_CONSTRAINT,
                    f"ops[{op_index}].transform.kind",
                    f"kind {op.transform_kind!r} requires {CONSTRAINTS[op.transform_kind]}",
                )

    def _resolve_artifacts(self, spec: InterventionSpec) -> dict:
        """Load every artifact the spec references, memoized worker-wide,
        and check per (kind, artifact) pair that it carries the tensors
        the kind expects, with the shapes this model's hooks will feed.
        A shared artifact is loaded once but checked once per kind.
        """
        artifacts = {}
        for op_index, op in enumerate(spec.ops):
            entries = [
                (op.transform_kind, op.transform_params, op.artifact, f"ops[{op_index}].transform.artifact")
            ]
            for m_index, modifier in enumerate(op.modifiers):
                entries.append(
                    (
                        modifier.kind,
                        modifier.params,
                        modifier.artifact,
                        f"ops[{op_index}].transform.modifiers[{m_index}].artifact",
                    )
                )
            gate = op.gate
            gate_path = f"ops[{op_index}].gate"
            while gate is not None:
                entries.append((gate.kind, gate.params, gate.artifact, f"{gate_path}.artifact"))
                gate = gate.inner
                gate_path += ".inner"
            for kind, params, artifact_id, path in entries:
                if artifact_id is None:
                    continue
                tensors = artifacts.get(artifact_id)
                if tensors is None:
                    tensors = self._load_artifact(artifact_id, path)
                self._check_artifact_tensors(kind, artifact_id, tensors, path, params)
                artifacts[artifact_id] = tensors
        return artifacts

    def _check_artifact_tensors(
        self, kind: str, artifact_id: str, tensors: dict, path: str, params: dict | None = None
    ) -> None:
        """Reject artifacts whose tensors would blow up inside a hook.

        Hooks run inside the forward pass; a shape mismatch there aborts
        the whole batch, so every tensor is checked against the model's
        dimensions at staging, where a failure rejects only this request.
        """

        def _fail(msg: str):
            raise SpecError(E_BAD_PARAM, path, f"artifact {artifact_id} {msg}")

        for name in ARTIFACT_TENSORS.get(kind, ()):
            if name not in tensors:
                _fail(f"lacks tensor {name!r} required by kind {kind!r}")
        for name, tensor in tensors.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                _fail(f"tensor {name!r} has non-finite values")

        hidden = self._hidden_size
        if kind in ("additive", "directional_ablation", "alignment_adaptive"):
            vector = tensors["vector"]
            if tuple(vector.shape) != (hidden,):
                _fail(f"tensor 'vector' has shape {tuple(vector.shape)}; kind {kind!r} needs ({hidden},)")
            if kind != "additive" and float(vector.float().norm()) == 0.0:
                _fail(f"tensor 'vector' is all zeros; kind {kind!r} needs a direction")
        elif kind == "rotation":
            basis = tensors["basis"]
            if tuple(basis.shape) != (2, hidden):
                _fail(f"tensor 'basis' has shape {tuple(basis.shape)}; rotation needs (2, {hidden})")
            b1 = basis[0].float()
            b2 = basis[1].float()
            if float(b1.norm()) == 0.0 or float(b2.norm()) == 0.0:
                _fail("tensor 'basis' has a zero row; rotation needs two plane-spanning vectors")
            cosine = float((b1 @ b2) / (b1.norm() * b2.norm()))
            if abs(cosine) > 1.0 - 1e-6:
                _fail("tensor 'basis' rows are colinear; rotation needs two plane-spanning vectors")
        elif kind == "head_additive":
            vector = tensors["vector"]
            allowed = {(self._num_heads, self._head_dim), (self._head_dim,)}
            if tuple(vector.shape) not in allowed:
                _fail(
                    f"tensor 'vector' has shape {tuple(vector.shape)}; head_additive needs "
                    f"({self._num_heads}, {self._head_dim}) or ({self._head_dim},)"
                )
        elif kind == "probe_sum":
            weights = tensors["weights"]
            num_condition_layers = len(params["condition_layers"]) if params else None
            if (
                weights.dim() != 2
                or weights.shape[1] != hidden
                or (num_condition_layers is not None and weights.shape[0] != num_condition_layers)
            ):
                _fail(
                    f"tensor 'weights' has shape {tuple(weights.shape)}; probe_sum needs "
                    f"({num_condition_layers if num_condition_layers is not None else 'num_condition_layers'}, "
                    f"{hidden}) row-aligned with condition_layers"
                )
            if "bias" in tensors and tensors["bias"].numel() != 1:
                _fail("tensor 'bias' must be a scalar")
        elif kind == "multi_key_threshold":
            weights = tensors["weights"]
            if weights.dim() != 2 or weights.shape[0] == 0 or weights.shape[1] != hidden:
                _fail(
                    f"tensor 'weights' has shape {tuple(weights.shape)}; "
                    f"multi_key_threshold needs (num_keys >= 1, {hidden})"
                )
            if "biases" in tensors and tuple(tensors["biases"].shape) != (weights.shape[0],):
                _fail(f"tensor 'biases' must have shape ({weights.shape[0]},)")

    def _load_artifact(self, artifact_id: str, path: str) -> dict:
        cached = self._artifact_cache.get(artifact_id)
        if cached is not None:
            self._artifact_cache.move_to_end(artifact_id)
            return cached
        tensors = self._registry.load(artifact_id, path=path)
        self._artifact_cache[artifact_id] = tensors
        while len(self._artifact_cache) > _ARTIFACT_CACHE_SIZE:
            self._artifact_cache.popitem(last=False)
        return tensors

    def _state_for(self, req_id: str) -> _StagedRequest | None:
        """Staged state for an internal request id.

        Matches exact ids first, then the legacy ``{external}-{suffix}``
        convention. When neither matches, falls back to staging from the
        request's own extra_args — the offline path can add requests whose
        ids were not known at prepare_requests time.
        """
        state = self._staged.get(req_id)
        if state is not None:
            return state
        alias = self._alias.get(req_id)
        if alias is not None:
            return self._staged.get(alias)
        for key in self._staged:
            if req_id.startswith(f"{key}-"):
                # Guard against a coincidental id extension: a suffixed
                # internal id belongs to the staged request, so its own
                # extra_args must carry the same new-surface keys. A live
                # request that carries none is a different request.
                if self._carries_new_surface(req_id) is False:
                    continue
                self._alias[req_id] = key
                return self._staged[key]
        return self._lazy_stage(req_id)

    def _carries_new_surface(self, req_id: str) -> bool | None:
        """Whether the live request's extra_args carry new-surface keys;
        None when the request state is not reachable.
        """
        try:
            req_state = self.model_runner.requests.get(req_id)
        except AttributeError:
            return None
        if req_state is None or req_state.sampling_params is None:
            return None
        extra = req_state.sampling_params.extra_args or {}
        return any(extra.get(key) is not None for key in NEW_SURFACE_KEYS)

    def _disable_request(self, state: _StagedRequest, exc: Exception, req_id: str | None = None) -> None:
        """Drop a request whose hook processing failed; the batch goes on.

        ``req_id`` is the internal id the failing view carried; marking it
        rejected too keeps the lazy-staging fallback from re-staging the
        request from its own extra_args when internal ids are suffixed.
        """
        if state.key is not None:
            self._staged.pop(state.key, None)
            self._mark_rejected(state.key)
        if req_id is not None and req_id != state.key:
            self._mark_rejected(req_id)
        logger.exception("disabled request %s after hook failure: %s", state.key, exc)

    def _mark_rejected(self, req_id: str) -> None:
        if len(self._rejected) > 8192:
            self._rejected.clear()
        self._rejected.add(req_id)

    def _lazy_stage(self, req_id: str) -> _StagedRequest | None:
        if req_id in self._plain or req_id in self._rejected:
            return None
        try:
            req_state = self.model_runner.requests.get(req_id)
        except AttributeError:
            return None
        if req_state is None or req_state.sampling_params is None:
            return None
        extra = req_state.sampling_params.extra_args or {}
        if not any(extra.get(key) is not None for key in NEW_SURFACE_KEYS):
            if len(self._plain) > 8192:
                self._plain.clear()
            self._plain.add(req_id)
            return None
        try:
            self._stage_request(req_id, {key: extra.get(key) for key in NEW_SURFACE_KEYS})
        except Exception as exc:
            # Admission validates before scheduling, so this is a fallback
            # path racing a rejection; never fail the batch from a hook.
            self._mark_rejected(req_id)
            logger.warning("lazy staging rejected request %s: %s", req_id, exc)
            return None
        state = self._staged.get(req_id)
        if state is None:
            # Keys present but decoded to null: nothing to stage, ever.
            self._mark_rejected(req_id)
        return state

    # -----------------------------------------------------------------
    # Hook installation
    # -----------------------------------------------------------------

    def _install_hooks(self):
        self._staged = OrderedDict()
        self._alias = {}
        self._plain = set()
        self._rejected = set()
        self._tracker = PositionTracker()
        self._registry = ArtifactRegistry()
        self._artifact_cache = OrderedDict()
        self._state_lru = int(os.environ.get("VLLM_HOOK_STATE_LRU", "4096"))
        self._hooks = []
        self._hooked_layers = set()

        model = getattr(self.model_runner, "model", None)
        if model is None:
            logger.warning("no model; skip hooks")
            return

        cfg = model.config
        # Multimodal models (e.g. Qwen3.5) nest text config under text_config.
        text_cfg = getattr(cfg, "text_config", cfg)
        self._hidden_size = int(getattr(text_cfg, "hidden_size"))
        self._num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
        self._num_heads = int(getattr(text_cfg, "num_attention_heads", 1))
        self._head_dim = int(getattr(text_cfg, "head_dim", None) or self._hidden_size // self._num_heads)

        # Only TP rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical. Interventions
        # run on every rank for the same reason: each rank must apply the
        # same edit to its replicated stream.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        matched = []
        for name, module, layer_num in iter_matched_modules(model, match_layer):
            self._hooked_layers.add(layer_num)
            self._hooks.append(
                module.register_forward_pre_hook(
                    lambda m, i, ln=layer_num: self._layer_pre_hook(m, i, ln)
                )
            )
            self._hooks.append(
                module.register_forward_hook(
                    lambda m, i, o, ln=layer_num: self._layer_hook(m, i, o, ln)
                )
            )
            matched.append(name)

        # head_additive edits per-head attention outputs before o_proj; the
        # heads are sharded across TP ranks, so the hook is TP==1 only
        # (prepare_requests rejects head_additive specs under TP).
        if tp_size == 1:
            for name, module, layer_num in iter_matched_modules(model, match_o_proj):
                self._hooks.append(
                    module.register_forward_pre_hook(
                        lambda m, i, ln=layer_num: self._o_proj_pre_hook(m, i, ln)
                    )
                )

        logger.info("installed unified hooks on %d layers: %s", len(matched), matched)

    def _pass_views(self):
        """Views for the current forward pass, or [] when hooks should be
        inert (warmup, CUDA-graph capture, no staged work possible).
        """
        ctx = get_forward_context()
        metadata = getattr(ctx, "attn_metadata", None)
        if metadata is None:
            return []
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return []
        return build_pass_views(self.model_runner, metadata, self._tracker)

    # -----------------------------------------------------------------
    # Hooks
    # -----------------------------------------------------------------

    def _layer_pre_hook(self, module, inputs, layer_num: int):
        """Condition-layer reads for gates and layer_input capture.

        The materialized layer input (hidden + residual when the previous
        layer returned a fused pair) is the layer-boundary value entering
        this layer.
        """
        views = self._pass_views()
        if not views:
            return None

        stream = None
        for view in views:
            state = self._state_for(view.req_id)
            if state is None:
                continue
            try:
                wants_conditions = layer_num in state.condition_layers
                wants_capture = state.wants_capture(layer_num, "layer_input") and self._should_capture
                if not wants_conditions and not wants_capture:
                    continue
                if view.is_restart:
                    state.maybe_reset(self._tracker.pass_key)
                if stream is None:
                    stream = self._materialize_input(inputs)
                    if stream is None:
                        return None
                if wants_conditions:
                    state.feed_conditions(layer_num, view, stream[view.row_slice])
                if wants_capture:
                    state.record_capture(layer_num, "layer_input", view, stream[view.row_slice])
            except Exception as exc:
                self._disable_request(state, exc, view.req_id)
        return None

    def _layer_hook(self, module, inputs, output, layer_num: int):
        views = self._pass_views()
        if not views:
            return output

        # Materialize only when some request acts at this layer.
        relevant = []
        for view in views:
            state = self._state_for(view.req_id)
            if state is None:
                continue
            if state.ops_for(layer_num) or (
                state.wants_capture(layer_num, "layer_output") and self._should_capture
            ):
                relevant.append((view, state))
        if not relevant:
            return output

        is_tuple = isinstance(output, tuple)
        if is_tuple:
            hidden, residual = output
        else:
            hidden = None
            residual = output
        # vLLM decoder layers return (hidden_states, residual) with the
        # residual add fused into the next layer's input layernorm; the true
        # layer-boundary value is their sum.
        stream = hidden + residual if is_tuple else residual  # (tokens, hidden)

        for view, state in relevant:
            try:
                if view.is_restart:
                    state.maybe_reset(self._tracker.pass_key)
                for op_index, op in state.ops_for(layer_num):
                    if not state.gate_open(op_index, op, view):
                        continue
                    rows = scope_rows(view, op.scope)
                    if rows is None:
                        continue
                    req_rows = stream[view.row_slice]
                    new_rows = apply_op(op, req_rows[rows], state.artifacts)
                    req_rows[rows] = new_rows
                    if is_tuple:
                        # Write back residual' = stream' - hidden for the
                        # steered rows only: co-batched requests must keep
                        # their residual bit-identical, and a whole-batch
                        # re-split would perturb them by low-precision
                        # rounding of (hidden + residual) - hidden.
                        residual[view.row_slice][rows] = new_rows - hidden[view.row_slice][rows]
                    # non-tuple layers: stream IS the output tensor, already
                    # updated in place by the req_rows write.
                if state.wants_capture(layer_num, "layer_output") and self._should_capture:
                    state.record_capture(layer_num, "layer_output", view, stream[view.row_slice])
            except Exception as exc:
                # A hook failure must cost this request its interventions,
                # never the forward pass of the whole batch.
                self._disable_request(state, exc, view.req_id)
        return output

    def _o_proj_pre_hook(self, module, inputs, layer_num: int):
        """head_additive: edit per-head attention outputs entering o_proj."""
        views = self._pass_views()
        if not views:
            return None

        x = inputs[0] if inputs else None
        if not isinstance(x, torch.Tensor):
            return None

        heads = None
        cloned = False
        for view in views:
            state = self._state_for(view.req_id)
            if state is None or not state.head_ops_for(layer_num):
                continue
            try:
                if view.is_restart:
                    state.maybe_reset(self._tracker.pass_key)
                if heads is None:
                    heads = x.reshape(-1, self._num_heads, self._head_dim)
                for op_index, op in state.head_ops_for(layer_num):
                    if not state.gate_open(op_index, op, view):
                        continue
                    rows = scope_rows(view, op.scope)
                    if rows is None:
                        continue
                    if not cloned:
                        heads = heads.clone()
                        cloned = True
                    req_rows = heads[view.row_slice]
                    req_rows[rows] = apply_op(op, req_rows[rows], state.artifacts)
            except Exception as exc:
                self._disable_request(state, exc, view.req_id)
        if not cloned:
            return None
        return (heads.reshape(x.shape),) + tuple(inputs[1:])

    @staticmethod
    def _materialize_input(inputs):
        """Layer-boundary value from a decoder layer's positional args.

        Position ids are integer tensors, so the floating 2-D args are the
        stream: a (hidden, residual) pair to sum, or a single tensor.
        """
        tensors = [
            t for t in inputs if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dim() == 2
        ]
        if not tensors:
            return None
        if len(tensors) == 2 and tensors[0].shape == tensors[1].shape:
            return tensors[0] + tensors[1]
        return tensors[0]

    # -----------------------------------------------------------------
    # Discovery helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _vllm_version() -> str:
        try:
            import vllm
            return vllm.__version__
        except Exception:
            return "unknown"

    def _model_fingerprints(self, model_config) -> dict:
        """Best-effort model fingerprints; each falls back to the recipe's
        empty-input value when the source cannot be resolved.
        """
        try:
            hf_config = model_config.hf_config.to_dict()
        except Exception:
            hf_config = {}
        try:
            config_fp = config_fingerprint(hf_config)
        except (TypeError, ValueError):
            config_fp = config_fingerprint({})

        tokenizer_files = self._tokenizer_files(model_config)
        tokenizer_fp = tokenizer_fingerprint(
            [p for p in tokenizer_files if not p.endswith("tokenizer_config.json")]
        )

        template = None
        for file_path in tokenizer_files:
            if file_path.endswith("tokenizer_config.json"):
                try:
                    with open(file_path) as f:
                        raw = json.load(f).get("chat_template")
                except (OSError, ValueError):
                    break
                if isinstance(raw, str) or raw is None:
                    template = raw
                else:
                    template = canonical_bytes(raw).decode("utf-8")
                break
        return {
            "config_fingerprint": config_fp,
            "tokenizer_fingerprint": tokenizer_fp,
            "chat_template_fingerprint": chat_template_fingerprint(template),
        }

    @staticmethod
    def _tokenizer_files(model_config) -> list:
        """Locate tokenizer files: ``tokenizer.json`` when present, else the
        vocab/merges file set, plus ``tokenizer_config.json`` for the chat
        template. Resolves local paths directly and hub ids via the local
        HF cache.
        """
        source = getattr(model_config, "tokenizer", None) or model_config.model
        candidates = ("tokenizer.json", "vocab.json", "merges.txt", "vocab.txt", "tokenizer_config.json")

        found = []
        if os.path.isdir(source):
            for name in candidates:
                file_path = os.path.join(source, name)
                if os.path.exists(file_path):
                    found.append(file_path)
        else:
            try:
                from huggingface_hub import try_to_load_from_cache
                for name in candidates:
                    cached = try_to_load_from_cache(source, name)
                    if isinstance(cached, str):
                        found.append(cached)
            except Exception:
                return []
        has_tokenizer_json = any(p.endswith("/tokenizer.json") or p.endswith("tokenizer.json") for p in found)
        if has_tokenizer_json:
            found = [p for p in found if not (p.endswith("vocab.json") or p.endswith("merges.txt") or p.endswith("vocab.txt"))]
        return found
