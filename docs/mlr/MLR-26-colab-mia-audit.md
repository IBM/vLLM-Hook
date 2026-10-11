# MLR-26 Colab MIA Audit

This note compares the current MIA demo scripts on `upstream/main` with the Colab notebooks that still use the vLLM-Hook v0 API.

## Scope

Colab notebooks currently present on `upstream/main`:

- `notebooks/demo_actsteer_colab.ipynb`
- `notebooks/demo_attntracker_colab.ipynb`
- `notebooks/demo_corer_colab.ipynb`

Reference MIA demo scripts:

- `examples/demo_actsteer.py`
- `examples/demo_attntracker.py`
- `examples/demo_corer.py`

## Shared Changes Needed

All three Colab notebooks still install and import the old package:

- install path: `vllm_hook_plugins`
- import: `from vllm_hook_plugins import HookLLM`
- class: `HookLLM`

Current MIA demos use:

- repo-root install: `pip install -e . --no-deps`
- import: `from mia import MiaLLM`
- class: `MiaLLM`

The Colab setup cells also set `PKG_DIR = REPO_ROOT / "vllm_hook_plugins"` and manually validate the old package path. That setup should be replaced with repo-root package registration for MIA.

Each notebook should keep the Colab clone/restart flow, but after checkout it should install from the repo root, not from `vllm_hook_plugins`.

## Notebook Mapping

| Colab notebook | Reference script | Old worker | MIA worker | Analyzer |
|---|---|---:|---:|---|
| `demo_actsteer_colab.ipynb` | `examples/demo_actsteer.py` | `steer_hook_act` | `steer` | none |
| `demo_attntracker_colab.ipynb` | `examples/demo_attntracker.py` | `probe_hook_qk` | `capture_qk` | `attn_tracker` |
| `demo_corer_colab.ipynb` | `examples/demo_corer.py` | `probe_hook_qk` | `capture_qk` | `core_reranker` |

## ActSteer Colab

Current notebook pattern:

- `from vllm_hook_plugins import HookLLM`
- `llm = HookLLM(..., worker_name="steer_hook_act", config_file=json_path, ...)`
- steered and unsteered calls both use the same `SamplingParams`
- `use_hook=False` disables steering for the baseline

MIA script pattern:

- `from mia import MiaLLM`
- `llm = MiaLLM(..., worker_name="steer", config_file=CONFIG, ...)`
- steering config is passed per request in `SamplingParams(extra_args={"steer": steer})`
- `use_hook=steered` decides whether the steering worker runs

Changes needed:

- Replace `HookLLM` with `MiaLLM`.
- Change worker name from `steer_hook_act` to `steer`.
- Load the config file's `steering` section and pass it through `SamplingParams.extra_args`.
- Keep `use_hook=False` for the baseline and `use_hook=True` for the steered run.
- Use MIA's default `spawn` multiprocessing setup before constructing the engine.

## Attention Tracker Colab

Current notebook pattern:

- `from vllm_hook_plugins import HookLLM`
- `llm = HookLLM(..., worker_name="probe_hook_qk", analyzer_name="attn_tracker", ...)`
- generation uses `save_to_disk=True`
- analysis uses `llm.analyze(analyzer_spec=...)` without passing probes
- baseline calls `llm.generate(..., use_hook=False)`

MIA script pattern:

- `from mia import MiaLLM`
- `llm = MiaLLM(..., worker_name="capture_qk", analyzer_name="attn_tracker", hook_dir="/dev/shm/mia", ...)`
- generation returns probes on the output object
- analysis passes `probes=out[0].probes`

Changes needed:

- Replace `HookLLM` with `MiaLLM`.
- Change worker name from `probe_hook_qk` to `capture_qk`.
- Prefer in-memory probe flow from the script: `out = llm.generate(...)`, then `llm.analyze(..., probes=out[0].probes)`.
- Keep the model-specific token range helper, but update it to match the script's `messages_and_range` logic.
- If using disk instead of probes, keep `save_to_disk=True` and make sure `run_id` / `hook_dir` behavior matches MIA's artifact paths.

## CoRe Colab

Current notebook pattern:

- `from vllm_hook_plugins import HookLLM`
- `llm = HookLLM(..., worker_name="probe_hook_qk", analyzer_name="core_reranker", ...)`
- text prompts are generated with `SamplingParams`
- analysis uses `run_ids=["corer-doc", "corer-na"]`

MIA script pattern:

- `from mia import MiaLLM`
- `from vllm import SamplingParams, TokensPrompt`
- `llm = MiaLLM(..., worker_name="capture_qk", analyzer_name="core_reranker", hook_dir="/dev/shm/mia", ...)`
- prompts are passed as exact token IDs using `TokensPrompt(prompt_token_ids=ids)`
- analysis still uses `run_ids=["corer-doc", "corer-na"]`

Changes needed:

- Replace `HookLLM` with `MiaLLM`.
- Change worker name from `probe_hook_qk` to `capture_qk`.
- Import `TokensPrompt`.
- Use exact token IDs for query and N/A prompts, matching `examples/demo_corer.py`.
- Keep `save_to_disk=True` and stable run IDs, because the CoRe analyzer compares two captured runs.

## Deleted / Not Present As Colab

These MIA examples exist as scripts but do not currently have Colab notebooks on `upstream/main`:

- `examples/demo_hiddenstate.py`
- `examples/demo_halludetect.py`
- `examples/demo_scihal.py`
- `examples/demo_attnlink.py`
- `examples/demo_capture_aperture.py`
- `examples/demo_actsteer_language.py`
- `examples/demo_actsteer_serve.py`

The old Spotlight and Token Highlighter notebook paths were removed from `upstream/main`, so they are not part of the current Colab update set unless we decide to recreate them.
