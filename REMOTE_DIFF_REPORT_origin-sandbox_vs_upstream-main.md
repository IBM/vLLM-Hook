# Remote Diff Report: `origin/sandbox` vs `upstream/main`

Date: 2026-04-08

## Exact Refs

- `origin/main` = `7f5fd04702091f83def96ae600dc87217a41141d`
- `origin/sandbox` = `5406bcb88e66b09e092802d8aa669afbe7a4e867`
- `upstream/main` = `d7e3758b6ae096082b355375aeafb482e9b58ddd`

## Scope

- `origin/main` vs `upstream/main`: only `README.md` differs.
- The substantial comparison is `origin/sandbox` vs `upstream/main`.

## Commit Delta

- `upstream/main` only: `d7e3758 Update README.md`
- `origin/sandbox` only: 60 commits, including backend routing, Metal worker support, GPU probe-worker changes, notebook/config additions, and demo defaults.

## Verification Limits

- I fetched current remote refs successfully.
- I could not run end-to-end `pytest` coverage because `tests/conftest.py` skips when `vllm` is missing in the local environment.
- Statements below are based on direct `git diff` inspection of the changed files and exact function bodies that were inspected locally.

## Changed File Inventory

Legend:

- `COSMETIC`: docs, text, formatting, comments, docstrings, metadata
- `DEMO`: example/notebook/test harness behavior changes
- `CONFIG`: config/dependency additions
- `RUNTIME`: core library behavior change
- `NEW`: file does not exist in `upstream/main`

| Path | Type | Classification | Notes |
|---|---|---|---|
| `.gitignore` | A | COSMETIC, NEW | Ignore rules only |
| `README.md` | M | COSMETIC | Upstream/main also changed README independently |
| `examples/demo_actsteer.py` | M | DEMO | Default model/cache/backend behavior changed |
| `examples/demo_attntracker.py` | M | DEMO | Debug input path, backend auto-detect, template logic broadened |
| `examples/demo_corer.py` | M | DEMO | Default model changed, batch example gated |
| `model_configs/activation_steer/granite-3.1-2b-instruct-quantized.w4a16.json` | A | CONFIG, NEW | Added model config |
| `model_configs/activation_steer/granite-4.0-micro.json` | A | CONFIG, NEW | Added model config |
| `model_configs/attention_tracker/granite-3.1-2b-instruct-quantized.w4a16.json` | A | CONFIG, NEW | Added model config |
| `model_configs/attention_tracker/granite-4.0-micro.json` | A | CONFIG, NEW | Added model config |
| `model_configs/core_reranker/granite-3.1-2b-instruct-quantized.w4a16.json` | A | CONFIG, NEW | Added model config |
| `model_configs/core_reranker/granite-4.0-h-tiny.json` | A | CONFIG, NEW | Added model config |
| `model_configs/core_reranker/granite-4.0-micro.json` | A | CONFIG, NEW | Added model config |
| `notebooks/COLAB.md` | A | COSMETIC, NEW | Colab instructions |
| `notebooks/README.md` | A | COSMETIC, NEW | Apple Silicon / local notebook instructions |
| `notebooks/demo_actsteer.ipynb` | M | DEMO | Notebook flow/defaults/setup changes |
| `notebooks/demo_actsteer_colab.ipynb` | A | DEMO, NEW | New Colab notebook |
| `notebooks/demo_attntracker.ipynb` | M | DEMO | Notebook flow/defaults/setup changes |
| `notebooks/demo_attntracker_colab.ipynb` | A | DEMO, NEW | New Colab notebook |
| `notebooks/demo_corer.ipynb` | M | DEMO | Notebook flow/defaults/setup changes |
| `notebooks/demo_corer_colab.ipynb` | A | DEMO, NEW | New Colab notebook |
| `requirement.txt` | M | CONFIG | Adds `nbformat>=5.10.4` |
| `tests/test_token_mode_sensitivity.py` | A | DEMO, NEW | New semantic tests for analyzer behavior |
| `tests/use_cases/test_corer.py` | M | DEMO | Test target model changed to Granite |
| `vllm_hook_plugins/vllm_hook_plugins/__init__.py` | M | RUNTIME | Backend-specific worker registration |
| `vllm_hook_plugins/vllm_hook_plugins/analyzers/attention_tracker_analyzer.py` | M | RUNTIME | Attention tracker contract and token semantics changed |
| `vllm_hook_plugins/vllm_hook_plugins/hook_llm.py` | M | RUNTIME | Backend resolution, hook-engine lifecycle, artifact assertions |
| `vllm_hook_plugins/vllm_hook_plugins/run_utils.py` | M | RUNTIME | Load/normalize `qkv.pt` in addition to `qk.pt` |
| `vllm_hook_plugins/vllm_hook_plugins/workers/metal/__init__.py` | A | RUNTIME, NEW | Exposes Metal workers |
| `vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py` | A | RUNTIME, NEW | New Metal probe worker |
| `vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py` | A | RUNTIME, NEW | New Metal steering worker |
| `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py` | M | RUNTIME | GPU probe-worker capture logic materially changed |
| `vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py` | M | RUNTIME | Mostly lifecycle cleanup and docstrings |

## File Tree View

```text
.
├── .gitignore                                      [NEW] COSMETIC
├── README.md                                       [MOD] COSMETIC
├── examples
│   ├── demo_actsteer.py                            [MOD] DEMO
│   ├── demo_attntracker.py                         [MOD] DEMO
│   └── demo_corer.py                               [MOD] DEMO
├── model_configs
│   ├── activation_steer
│   │   ├── granite-3.1-2b-instruct-quantized.w4a16.json [NEW] CONFIG
│   │   └── granite-4.0-micro.json                  [NEW] CONFIG
│   ├── attention_tracker
│   │   ├── granite-3.1-2b-instruct-quantized.w4a16.json [NEW] CONFIG
│   │   └── granite-4.0-micro.json                  [NEW] CONFIG
│   └── core_reranker
│       ├── granite-3.1-2b-instruct-quantized.w4a16.json [NEW] CONFIG
│       ├── granite-4.0-h-tiny.json                 [NEW] CONFIG
│       └── granite-4.0-micro.json                  [NEW] CONFIG
├── notebooks
│   ├── COLAB.md                                    [NEW] COSMETIC
│   ├── README.md                                   [NEW] COSMETIC
│   ├── demo_actsteer.ipynb                         [MOD] DEMO
│   ├── demo_actsteer_colab.ipynb                   [NEW] DEMO
│   ├── demo_attntracker.ipynb                      [MOD] DEMO
│   ├── demo_attntracker_colab.ipynb                [NEW] DEMO
│   ├── demo_corer.ipynb                            [MOD] DEMO
│   └── demo_corer_colab.ipynb                      [NEW] DEMO
├── requirement.txt                                 [MOD] CONFIG
├── tests
│   ├── test_token_mode_sensitivity.py              [NEW] DEMO
│   └── use_cases
│       └── test_corer.py                           [MOD] DEMO
└── vllm_hook_plugins
    └── vllm_hook_plugins
        ├── __init__.py                             [MOD] RUNTIME
        ├── analyzers
        │   └── attention_tracker_analyzer.py       [MOD] RUNTIME
        ├── hook_llm.py                             [MOD] RUNTIME
        ├── run_utils.py                            [MOD] RUNTIME
        └── workers
            ├── metal
            │   ├── __init__.py                     [NEW] RUNTIME
            │   ├── probe_hookqk_worker_metal.py    [NEW] RUNTIME
            │   └── steer_activation_worker_metal.py [NEW] RUNTIME
            ├── probe_hookqk_worker.py              [MOD] RUNTIME
            └── steer_activation_worker.py          [MOD] RUNTIME
```

## Function Inventory For Runtime Files

### `vllm_hook_plugins/vllm_hook_plugins/__init__.py`

- `_can_register_metal_worker() -> bool`
- `ensure_backend_workers_registered(backend: str) -> None`
- `register_plugins()`

### `vllm_hook_plugins/vllm_hook_plugins/hook_llm.py`

Class: `HookLLM`

- `__init__(...)`
- `_resolve_hook_dir(download_dir: str, hook_dir: Optional[str]) -> str`
- `_normalize_backend_name(name: Optional[str]) -> Optional[str]`
- `_resolve_backend(backend: Optional[str], vllm_kwargs: Dict) -> str`
- `_resolve_worker_name(PluginRegistry, worker_name: str, backend: str) -> str`
- `_should_use_hook_worker(self) -> bool`
- `_build_llm(self, use_hook_worker: bool) -> LLM`
- `_dispose_llm(self, llm: Optional[LLM]) -> None`
- `load_config(self, config_file: str)`
- `generate(...)`
- `generate_with_encode_hook(...)`
- `generate_with_decode_hook(...)`
- `analyze(...)`
- `_setup_hooks(self, cleanup)`
- `_cleanup_hooks(self)`
- `_assert_hook_artifacts_exist(self) -> None`

### `vllm_hook_plugins/vllm_hook_plugins/analyzers/attention_tracker_analyzer.py`

Class: `AttntrackerAnalyzer`

- `__init__(self, hook_dir: str, layer_to_heads: Dict[int, list])`
- `analyze(...)`
- `compute_attention_from_qk(self, run_id_file: str) -> Dict[str, Dict]`
- `attn2score(...) -> float`

### `vllm_hook_plugins/vllm_hook_plugins/run_utils.py`

- `read_run_ids(run_id_file: str) -> List[str]`
- `latest_run_id(run_id_file: str) -> str`
- `_artifact_glob(hook_dir: str, run_id: str) -> List[str]`
- `_normalize_qkv_cache(cache: Dict[str, Any]) -> Dict[str, Any]`
- `load_and_merge_qk_cache(hook_dir: str, run_id: str)`

### `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py`

- `match_attn(name: str)`
- `_segment_bounds_from_metadata(metadata, module_name: str, device: torch.device)`

Class: `ProbeHookQKWorker`

- `load_model(self, *args, **kwargs)`
- `_install_hooks(self)`
- `_parse_layer_heads(self) -> Dict[int, List[int]]`
- `_uninstall_hooks(self)`
- `execute_model(self, *args, **kwargs)`

### `vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py`

Class: `SteerHookActWorker`

- `load_model(self, *args, **kwargs)`
- `_install_hooks(self)`
- `_parse_steering_config(self) -> Dict`
- `_uninstall_hooks(self)`
- `execute_model(self, *args, **kwargs)`

### `vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py`

- `match_attn(name: str)`

Class: `MLXHookWrapper`

- `__init__(self, module, name, hook_fn)`
- `__call__(self, *args, **kwargs)`

Class: `ProbeHookQKWorkerMetal`

- `_stage(self, message: str) -> None`
- `__init__(self, *args, **kwargs)`
- `init_device(self) -> None`
- `_init_device_single_process(self) -> None`
- `load_model(self, *args, **kwargs)`
- `_current_run_id(self) -> str | None`
- `_ensure_run_cache(self, run_id: str)`
- `_append_proj_tokens(...)`
- `_append_proj_token_list(...)`
- `_mx_offsets_to_int_list(self, value, batch_size: int) -> list[int]`
- `_flatten_attention_sample(self, sample) -> torch.Tensor`
- `_build_full_kv_without_mutation(self, cache, keys, values)`
- `_record_qkv(...)`
- `_capture_from_self_attn(...)`
- `_parse_layer_heads(self) -> Dict[int, List[int]]`
- `_install_hooks(self)`
- `_uninstall_hooks(self)`
- `_flush_run_cache(self) -> None`
- `execute_model(self, *args, **kwargs)`

### `vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py`

Class: `MLXSteeringWrapper`

- `__init__(self, module, name, hook_fn)`
- `__call__(self, *args, **kwargs)`

Class: `SteerHookActWorkerMetal`

- `__init__(self, *args, **kwargs)`
- `init_device(self) -> None`
- `_init_device_single_process(self) -> None`
- `load_model(self, *args, **kwargs)`
- `_install_hooks(self)`
- `_parse_steering_config(self) -> Dict`
- `_steering_enabled(self) -> bool`
- `_mlx_cast_like(self, value: mx.array, reference: mx.array) -> mx.array`
- `_apply_torch_steering(self, residuals: torch.Tensor) -> torch.Tensor`
- `_apply_mlx_steering(self, residuals: mx.array) -> mx.array`
- `_steering_hook(self, output, _module_name: str)`
- `_uninstall_hooks(self)`
- `execute_model(self, *args, **kwargs)`

## Side-By-Side Report For Core Runtime Files

### `vllm_hook_plugins/vllm_hook_plugins/__init__.py`

Upstream behavior:

- Registers only default GPU workers:
  - `probe_hook_qk`
  - `steer_hook_act`
- Registers analyzers:
  - `attn_tracker`
  - `core_reranker`

Sandbox behavior:

- Adds `_can_register_metal_worker()`
- Adds `ensure_backend_workers_registered(backend)`
- Conditionally registers:
  - `probe_hook_qk_metal`
  - `steer_hook_act_metal`
- Adds analyzer alias:
  - `attention_tracker`

Overwrite risk:

- Medium.
- If `sandbox` overwrites upstream with this file, upstream must also take the Metal worker files and `HookLLM` backend-routing changes or backend-specific worker lookup will break.

### `vllm_hook_plugins/vllm_hook_plugins/hook_llm.py`

Upstream behavior:

- Single `LLM` instance created in `__init__`.
- Worker path is resolved only if `worker_name` is provided.
- Hook cache directory is fixed to `hook_dir` or `download_dir/_v1_qk_peeks`.
- No explicit backend resolution function.
- Hooked encode/decode paths run against `self.llm`.
- No explicit post-hook artifact assertion.
- `analyze()` does not guard against the previous `generate()` having run without hooks.

Sandbox behavior:

- `__init__` adds `backend` and computes backend with `_resolve_backend(...)`.
- `_resolve_hook_dir(...)` falls back across:
  - user-provided `hook_dir`
  - `download_dir/_v1_qk_peeks`
  - `~/.cache/_v1_qk_peeks`
  - temp dir
- Metal-specific env defaults are set in `__init__`.
- `_resolve_worker_name(...)` prefers backend-specific worker names like `probe_hook_qk_metal`.
- `_should_use_hook_worker(...)` disables Metal hook workers when `VLLM_DISABLE_METAL_HOOKS=1`.
- `_build_llm(...)` can build a hooked worker-backed engine or a plain engine.
- `_dispose_llm(...)` tears down temporary hooked engines.
- `generate(...)` tracks whether hooks were used via `_last_generate_used_hooks`.
- `generate_with_encode_hook(...)`:
  - sets up hooks
  - builds a temporary hooked engine
  - performs hooked prefill
  - asserts artifacts exist with `_assert_hook_artifacts_exist()`
  - cleans up hooks
  - disposes temporary engine
  - runs normal generation on base engine
- `generate_with_decode_hook(...)`:
  - prefills on base engine
  - sets up hooks
  - builds temporary hooked engine
  - runs hooked decode/generation
  - cleans up hooks
  - disposes temporary engine
- `_setup_hooks(...)` now removes both `qk.pt` and `qkv.pt`.
- `analyze(...)` raises if the last generate call did not produce hook artifacts.

Directly observed semantic differences by function:

- `_resolve_backend(...)`: new
- `_resolve_worker_name(...)`: new backend-specific worker resolution
- `_should_use_hook_worker(...)`: new Metal-specific disable switch
- `_build_llm(...)`: new hook-worker-aware engine construction
- `_dispose_llm(...)`: new lifecycle cleanup
- `generate(...)`: now branches based on actual hook-worker availability
- `generate_with_encode_hook(...)`: no longer reuses only `self.llm`
- `generate_with_decode_hook(...)`: same
- `analyze(...)`: now rejects stale/no-hook analysis calls

Overwrite risk:

- High.
- If `sandbox` overwrites upstream with this file, upstream will inherit backend selection, worker selection, hook lifecycle, analysis-validity checks, and Metal-specific behavior.
- The real merge risk is not loss inside `sandbox`; it is that upstream must also accept the coupled worker and analyzer changes for this file to remain correct.

### `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py`

Upstream behavior:

- Hooks only attention modules matching older patterns ending in `.attn`.
- Assumes Q and K are present in the hooked module input tuple.
- Cache write path is centered around `qk.pt`.

Sandbox behavior:

- `match_attn(...)` additionally supports `model.layers.<i>.self_attn`.
- `_segment_bounds_from_metadata(...)` is new and recovers boundaries from nested metadata fields such as:
  - `prefill`
  - `decode`
  - `_cached_prefill_metadata`
  - `_cached_decode_metadata`
  - per-module metadata mapping
- `_install_hooks(...)` now has two capture paths:
  - direct attention hook path via `qkv_hook(...)`
  - projection fallback via `proj_hook(...)` on `q_proj` and `k_proj`
- Fallback path exists specifically for wrapper modules whose forward-hook tuple no longer exposes both tensors.
- `_uninstall_hooks(...)` is new.

Directly observed semantic differences by function:

- `match_attn(...)`: broader module-name matching
- `_segment_bounds_from_metadata(...)`: new metadata decoding path
- `_install_hooks(...)`: handles Granite-style `self_attn` and projection-hook fallback
- `_uninstall_hooks(...)`: new cleanup path

Overwrite risk:

- High.
- If `sandbox` overwrites upstream with this file, upstream will inherit support for newer Granite-style attention layouts and nested metadata paths.
- The real risk is behavioral expansion: upstream GPU capture semantics will change for newer attention-module layouts and wrapper structures.

### `vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py`

Upstream behavior:

- Steering hook loads and applies the configured steering vector.
- No explicit `_uninstall_hooks()`.

Sandbox behavior:

- Same steering logic is still present.
- Adds `_uninstall_hooks()` lifecycle cleanup.
- Most other changes are docstrings and structure, not observed behavior changes.

Overwrite risk:

- Low to medium.
- If `sandbox` overwrites upstream with this file, the main effect is hook-cleanup symmetry plus docstring/structure churn.
- This is a comparatively low-risk overwrite relative to the probe worker and `HookLLM`.

### `vllm_hook_plugins/vllm_hook_plugins/run_utils.py`

Upstream behavior:

- `_artifact_glob(...)` only searches for `qk.pt`.
- `load_and_merge_qk_cache(...)` assumes input artifacts are already in `qk_cache` shape.

Sandbox behavior:

- `_artifact_glob(...)` searches both `qk.pt` and `qkv.pt`.
- `_normalize_qkv_cache(...)` converts Metal-style `qkv_cache` data into legacy `qk_cache`.
- `load_and_merge_qk_cache(...)` normalizes each artifact before merge.

Overwrite risk:

- High if you keep Metal workers.
- If `sandbox` overwrites upstream with this file, upstream will inherit compatibility with Metal `qkv.pt` artifacts.
- The real risk is coupling: upstream should not take this file without also taking the Metal worker path that produces those artifacts.

### `vllm_hook_plugins/vllm_hook_plugins/analyzers/attention_tracker_analyzer.py`

Upstream behavior:

- `analyze(...)` returns only:
  - `{"score": score}`
- `compute_attention_from_qk(...)` assumes each cached `q` item is already a single vector.
- `attn2score(...)` returns only batch scores.

Sandbox behavior:

- `analyze(...)` now defaults `analyzer_spec = analyzer_spec or {}`
- `analyze(...)` returns:
  - `score`
  - `per_head_scores`
- `compute_attention_from_qk(...)` checks `q_last.ndim == 2` and, if so, uses the last query token.
- `attn2score(...)` accumulates both mean score and per-head score entries.

Directly observed semantic differences by function:

- `compute_attention_from_qk(...)`: multi-token query tensors are reduced to last-token semantics
- `attn2score(...)`: contract changed to return `(batch_scores, batch_per_head_scores)`
- `analyze(...)`: output schema changed

Overwrite risk:

- High.
- If `sandbox` overwrites upstream with this file, upstream will inherit:
  - a changed analyzer output schema
  - last-query-token normalization for multi-token `q`
  - per-head score reporting
- This is a true contract change, so downstream callers on upstream may break if they expect the old score-only result shape.

### `vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py`

Upstream behavior:

- File does not exist.

Sandbox behavior:

- Adds an entirely new Metal probe worker.
- The file contains Metal-specific device init, MLX wrapper hooks, QKV recording, cache flush, and self-attention capture logic.
- The worker is reachable only through:
  - `ensure_backend_workers_registered("metal")`
  - `HookLLM._resolve_worker_name(...)`

Overwrite risk:

- High if you need Apple Silicon / `vllm_metal` support.
- If `sandbox` overwrites upstream with this file, upstream will gain a new Apple Silicon / `vllm_metal` probe-worker path.
- The real risk is maintenance and coupling with `HookLLM`, worker registration, and artifact loading.

### `vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py`

Upstream behavior:

- File does not exist.

Sandbox behavior:

- Adds an entirely new Metal steering worker.
- Includes MLX wrapper, Metal device init, torch and MLX steering implementations, and steering hook lifecycle.

Overwrite risk:

- High if you need Apple Silicon / `vllm_metal` steering support.
- If `sandbox` overwrites upstream with this file, upstream will gain a new Apple Silicon / `vllm_metal` steering-worker path.
- The real risk is maintenance and coupling with backend registration and Metal runtime assumptions.

## Demo/Test/Config Report

### `examples/demo_actsteer.py`

Observed changes:

- Imports `platform`
- Default cache dir changed to `~/.cache/vllm-hook`
- Default model changed to `ibm-granite/granite-3.1-8b-instruct`
- Auto-selects `backend="metal"` on Darwin arm64 if env var not set
- Passes `backend=backend` into `HookLLM`
- Adds dtype entry for `ibm-granite/granite-4.0-micro`

Risk:

- Low for library runtime.
- Medium for reproducing old demo outputs because model/backend defaults changed.

### `examples/demo_attntracker.py`

Observed changes:

- Adds `debug_token_layout(...)`
- Adds `argparse` and `--debug-input-json`
- Lowers log verbosity unless `VLLM_HOOK_DEBUG=1`
- Broadens template logic from exact model IDs to family matching
- Adds backend auto-detection for Metal
- Uses `config_basename` rather than direct model-split inline usage

Risk:

- Medium for demo reproducibility.
- Not a core-library compatibility risk by itself.

### `examples/demo_corer.py`

Observed changes:

- Adds `argparse` and `--debug-input-json`
- Changes default model to Granite
- Adds Metal backend auto-detection
- Adds `keep_only_case` and `run_batch_example`
- Batch example is now gated and disabled by default

Risk:

- Medium for demo reproducibility.

### `tests/test_token_mode_sensitivity.py`

New functions:

- `_load_analyzer_classes()`
- `_write_qk_cache(...)`
- `test_attention_tracker_ignores_non_last_query_token_changes(...)`
- `test_core_reranker_changes_when_earlier_query_tokens_change(...)`

Observed intent:

- Attention Tracker is expected to use last-token semantics.
- CoRe reranker is expected to remain sensitive to earlier query-token changes.

### `tests/use_cases/test_corer.py`

Observed changes:

- `TEST_MODELS` replaced Mistral coverage with Granite 4 micro coverage.
- `apply_chat_template_and_get_ranges(...)` remains broadly the same.

## Overwrite Guidance

If `sandbox` is going to substantially overwrite upstream code, treat these files as coupled and review them together:

- `vllm_hook_plugins/vllm_hook_plugins/hook_llm.py`
- `vllm_hook_plugins/vllm_hook_plugins/__init__.py`
- `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py`
- `vllm_hook_plugins/vllm_hook_plugins/run_utils.py`
- `vllm_hook_plugins/vllm_hook_plugins/analyzers/attention_tracker_analyzer.py`
- `vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py`
- `vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py`

Most likely breakpoints if `sandbox` overwrites upstream without carrying the full coupled set:

- backend-specific worker lookup
- Metal support disappearing silently
- analyzer expecting `qk.pt` only while Metal writes `qkv.pt`
- Attention Tracker output schema changing back to score-only
- Attention Tracker reverting from last-query-token semantics
- GPU probe worker losing Granite/self-attn/projection fallback behavior

## Minimal Truths That Are Safe To Assert

- `origin/main` and `upstream/main` differ only in `README.md`.
- `origin/sandbox` contains the substantive runtime changes.
- `hook_llm.py`, `__init__.py`, `probe_hookqk_worker.py`, `run_utils.py`, and `attention_tracker_analyzer.py` all have direct functional differences from `upstream/main`.
- Metal worker support exists only in `origin/sandbox`.
- `attention_tracker_analyzer.py` in `origin/sandbox` changes both analyzer output schema and query-token handling.
- `run_utils.py` in `origin/sandbox` adds explicit support for `qkv.pt`-style artifacts.
