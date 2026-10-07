# Building your own demo

A demo is one Python file: build a `MiaLLM` engine, generate, read back what was captured. Every
demo here runs that way, offline, under MIA's defaults, except `demo_actsteer_serve.py`, the server
example. Each other demo (bar `demo_capture_aperture.py`) keeps its `vllm serve` version as a
commented block ([Server mode](#8-server-mode)).

## Getting started

Install MIA as in the root [`README.md`](../README.md#-installation) (vLLM 0.29.0, torch 2.13.0),
then run a demo from the repository root:

```bash
python examples/demo_hiddenstate.py
```

It builds the engine, prints a shape and a norm per captured layer, and its engine log names the
mode MIA chose: `[mia] capture mode: FULL_AND_PIECEWISE CUDA graph (chosen by default) -- ...`.

### Where the captured data goes

| You call | Your data | Ready |
|---|---|---|
| `MiaLLM.generate(...)` | `out[i].probes` per prompt; with several prompts `out[0].probes` holds the whole batch, one entry per prompt, for `llm.analyze(probes=out[0].probes, ...)` | on first access to `probes`, which waits for that output (a timeout names the request) |
| `MiaClient.generate(...)` | `response.probes`; `client.analyze(...)` | shortly after the response; `response.probes` waits for it |
| either, with `save_to_disk=True, run_id=R` | `<hook_dir>/R/`; `analyze(run_id=R)`, or `mia.artifacts.load_and_merge_hs_cache` / `load_and_merge_qk_cache(hook_dir, R)` | offline: when `generate` returns; served: shortly after the response (`client.analyze()` waits for it) |
| hidden-state files (CUDA graphs) | `mia.core.aperture.aperture_gather.load_delivered(<capture dir>)` → `{request_id: {layer: Tensor}}` | lists only requests already delivered |

- **`hook_dir`** defaults to `/dev/shm/mia` for `MiaClient`, `~/.cache/mia` for `MiaLLM`.
  `/dev/shm/mia` and `/tmp/mia_profile` are shared by every user of a node: pass your own
  `hook_dir` (and `MIA_PROFILE_DIR`) there.
- **Capture dir** is `$MIA_APERTURE_DIR`, default `./hs_aperture_dump` (or `./qk_aperture_dump`)
  in the working directory; the engine log names it (`... aperture drain ON -> <dir>`). One live
  engine per capture dir and worker kind: a second is refused.
- **MIA cleans up neither directory**, and an all-layers, all-tokens run writes tens of GB: keep the
  capture dir until you have read your probes, then delete it.
- **Layer names differ by reader:** `analyze()` keys hidden states by module name
  (`model.layers.0` for config layer 1), `load_delivered` by the config's 1-based layer number.
- **Offline, `generate` can return before every output's data has landed.** The rest is finished
  when the `MiaLLM` is dropped or Python exits normally; killing the process right after
  `generate` can lose it.
  - It gives up after 60 s without progress (30 min cap); a hung engine holds GPU memory until exit.
  - A `MiaLLM` dropped off the main thread finishes in the background, so exit can wait as long.
  - One collected off the main thread during shutdown skips this; Python prints "Exception ignored".
  - Read `probes` before `llm.llm_engine.engine_core.shutdown()`: a shutdown loses outputs not yet
    delivered.
- A served capture sent without `save_to_disk` may land in `<hook_dir>/<run_id>/` instead of on the
  response; `client.analyze()` reads it either way.
- **Served disk reads need a shared filesystem:** the server writes to the client's `hook_dir`, so
  run the client on the server's host or on a shared mount; otherwise use the in-memory route
  (`response.probes`).
- A served completion with several prompts and `save_to_disk` rewrites run `R` once per prompt; for
  large runs give each prompt its own `run_id`.
- `load_delivered` keys start with the request's id (`response.id`, `output.request_id`); with
  `n > 1`, sample `j`'s key starts with `<j>_`.
- Holding any one layer of an output's `probes` keeps that whole output's data in memory.

### Did it actually capture anything?

A run that captured nothing looks like a fast run, so check rather than assume:

- **The data exists.** `probes` is not `None`, `ls <hook_dir>/<run_id>/` lists files on the disk
  route, or `load_delivered(<capture dir>)` lists your request.
- **The analyzer returned something.** `stats["hidden_states"]` empty is a failed capture, not
  an empty model.
- **The counters moved.** Run with `MIA_PROFILE=1` (on the server, in server mode). At exit each
  process writes `$MIA_PROFILE_DIR/profile-<role>-<pid>-<n>.json` (default dir `/tmp/mia_profile`;
  the log prints `[mia profiler] wrote <path>`). In the file the `(EngineCore pid=...)` line names,
  `gauges["captured.bytes.hs"]` (or `.qk`) above zero proves rows were captured. Do not use a
  counter that merely proves the request finished.

### Tensor parallelism

Pass `tensor_parallel_size=N` to `MiaLLM` (`--tensor-parallel-size N` on the server). Capture
shards across the ranks — hidden states by **layer** (rank `r` takes layers where `i % N == r`), Q/K
by **head** — and each rank writes its own `tp_rank_<r>/`, which the readers merge:

```python
import os

from mia.core.aperture.aperture_gather import load_delivered

# Merges every tp_rank_<r>/; refuses a missing or duplicate rank.
per_request = load_delivered(os.environ.get("MIA_APERTURE_DIR", "./hs_aperture_dump"))
```

Three things to know before you try it:

- **`MIA_APERTURE_GPU_BYTES` is per rank**, so TP × 4 claims four times that much GPU in total.
  Each rank checks its own budget against `gpu_memory_utilization`; see
  [`docs/configs.md`](../docs/configs.md#sizing-the-capture-aperture-and-the-gpu-memory-it-costs)
  for the constraint.
- **Q/K `score` capture requires TP = 1.** Raw Q/K shards fine; the rebuilt per-head scores do
  not.
- **Pipeline parallelism is refused**, not merely untested — under PP each rank holds only its
  own stage's layers and the rest are identity placeholders that the layer matcher would hook
  anyway, so capture would return zero-filled rows for every off-stage layer.

Supported for `capture_hs`, `capture_qk` and `steer`. In server mode, pass `tp=N` to
`_serve.require_server()` to print a command with `--tensor-parallel-size N`.

## 1. Skeleton

```python
import multiprocessing as mp
import os

from vllm import SamplingParams

from mia import MiaLLM
from _paths import config_path          # resolves MIA's configs

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = MiaLLM(
        model="Qwen/Qwen2-1.5B-Instruct",
        worker_name="capture_hs",                            # what to capture
        analyzer_name="hidden_states",                       # what to do with it
        config_file=config_path("hidden_states/Qwen2-1.5B-Instruct.json"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
    )

    out = llm.generate("The capital of France is",
                       SamplingParams(temperature=0.0, max_tokens=10))
    stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out[0].probes)

    print(out[0].outputs[0].text)
    for layer, tensors in sorted(stats["hidden_states"].items()):
        print(layer, tuple(tensors[0].shape))
    llm.llm_engine.engine_core.shutdown()                    # after reading probes: a clean exit
```

Save it as `examples/my_demo.py` and run it from the repo root: `python examples/my_demo.py`.

**Steering** produces no artifact: pass the config in `SamplingParams(extra_args={"steer": {...}})`
and generate with `use_hook=False` for the unsteered run. See `demo_actsteer.py`.

**Per-request knobs** that the config file does not cover go in `SamplingParams.extra_args`, e.g.
`{"hooks_on": "both"}` (see the [long-decode demos](../docs/numerical_analysis/README.md)); the full list is in
[`docs/configs.md`](../docs/configs.md#per-request-arguments).

### Token-exact prompts

`demo_corer.py`, `demo_attnlink.py` and `demo_scihal.py` score **token spans**, so they pass exact
ids (`TokensPrompt(prompt_token_ids=ids)`) rather than text that would be tokenized again.
`demo_attnlink.py` fails loudly if the ids the engine prompted on differ from the ones its spans
were aligned to.

`demo_scihal.py` is the one to copy if you need a **continuation**: it rebuilds the second prompt
from the token ids the first pass generated, never from the detokenized text, because
detokenize-then-retokenize is not an identity.

## 2. Pick a worker and analyzer

`worker_name` decides what is captured. `analyzer_name` decides what happens to it, and is
optional — omit it if you only want the raw tensors.

| `worker_name` | captures | config section | reference demo |
|---|---|---|---|
| `capture_hs` | hidden states | `hidden_states` | `demo_hiddenstate.py` |
| `capture_qk` | attention Q/K | `params.important_heads` and `hookq` | `demo_attntracker.py` |
| `steer` | — (steers instead) | `steering` | `demo_actsteer.py` |

| `analyzer_name` | reference demo |
|---|---|
| `hidden_states` | `demo_hiddenstate.py` |
| `attn_tracker` | `demo_attntracker.py` |
| `core_reranker` | `demo_corer.py` |
| `hnode_hallucination` | `demo_halludetect.py` |
| `science_hallucination` | `demo_scihal.py` |
| `attnlink` | `demo_attnlink.py` |

## 3. Write the config

One JSON under `model_configs/<use_case>/<model_name>.json`. Only the sections matching your
worker are read; every key is listed in the
[config reference](../docs/configs.md#config-reference).

Capture hidden states from layers 1–4, last token only:

```json
{
  "model_info": { "name": "Qwen/Qwen2-1.5B-Instruct" },
  "hidden_states": { "layers": [1, 2, 3, 4], "mode": "last_token" }
}
```

`layers` are 1-based hidden-state indices: layer `i` is the output of decoder block `i - 1`, and
`[]` means every layer. `mode` is `last_token` or `all_tokens`. For Q/K use
`"params": {"important_heads": [[layer, head], ...]}` (0-based) with
`"hookq": {"hookq_mode": "last_token"}`. For steering use `"steering": {"method": "add_vector",
"coefficient": 10, "optimal_layer": 8, "vector_path": "steering_vectors/phi3_format.pt"}`
(`optimal_layer` is 0-based).

Copy the closest existing file in `model_configs/` rather than writing one from scratch.

## 4. Get your data back

Two paths. **Pick one — mixing them raises an error.**

```python
# Disk: artifacts written, analyze() reads them back.
out   = llm.generate(prompt, sp, save_to_disk=True)
stats = llm.analyze(analyzer_spec={...})

# In-memory: captures ride back on the output object.
out   = llm.generate(prompt, sp, save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})
```

Under `save_to_disk=True`, `out[0].probes` is always `None`. When each is ready:
[Where the captured data goes](#where-the-captured-data-goes).

For raw tensors with no analysis, use the `hidden_states` analyzer with `reduce="none"` — it
returns what was captured, unchanged.

## 5. Graph mode

`MiaLLM` runs under CUDA graphs by default; pass `enforce_eager=True` for eager. Leave
`cudagraph_mode` unset. Q/K capture turns prefix caching off unless you set it.
`demo_capture_aperture.py` shows the graph-mode capture and its checks.

## 6. Gotchas

- Set the `mp.set_start_method` / env lines **before** building the engine (`MiaLLM(...)`).
- Run from the repo root: configs resolve from anywhere, but a config's `vector_path` and the
  demos' `./cache/` are relative to the working directory (the server's, over `vllm serve`).
- Graph mode is the default, offline and served; `enforce_eager=True` / `--enforce-eager` opts out.
- Call `llm.llm_engine.reset_prefix_cache()` between prompts if you capture the same prefix twice.
  Steering needs no reset: a steered request shares cached prefixes only with requests steered
  the same way.
- Profiler counters need `MIA_PROFILE=1`; without it they are no-ops.
- Performance levers: `from mia.optimizations import describe; print(describe())`.

## 7. Running the included demos

Run every demo from the repo root, e.g. `python examples/demo_hiddenstate.py`. A few need more:

- **`MIA_DEMO_MODEL` / `MIA_CONFIG_FILE`** swap the model and config of `demo_hiddenstate.py`,
  `demo_actsteer.py`, `demo_attntracker.py` (granite, Mistral-7B or Qwen2-1.5B: it needs a
  token-range recipe per chat template), `demo_capture_aperture.py` and the two
  long-decode demos in `docs/numerical_analysis/`. The other demos are tied to their model. `demo_actsteer.py` on
  another model needs your own steering config and vector: only Phi-3 ships a real one.
- **`demo_actsteer_serve.py`** is server-only: start the server it prints, then run it
  ([Server mode](#8-server-mode)).
- **`demo_capture_aperture.py`** checks graph-mode capture: two runs byte-identical, and a
  `save_to_disk` run identical to the in-memory one. Pick the model with `MIA_DEMO_MODEL`; add
  `MIA_PROFILE=1` to print the profiler counters:

  ```bash
  MIA_DEMO_MODEL=Qwen/Qwen2-1.5B-Instruct python examples/demo_capture_aperture.py
  ```
- **`demo_halludetect.py`** downloads a pre-built H-Node probe (~22 KB) into `./cache/hnode_probe/`
  on first run, from
  [hnode-probe-builder](https://github.com/Samarpit-bhatia/hnode-probe-builder/tree/master/artifacts).
  Method: *H-Node Attack and Defense in Large Language Models*, <https://arxiv.org/abs/2603.26045>.
- **`demo_scihal.py`** downloads the SciHal data on first run and needs a classifier, which is not
  shipped; it stops before loading the model when none is found.
  - Install `pip install joblib scikit-learn`.
  - Features: the demo's second pass, run on the
    [SciHal-Challenge](https://github.com/InfintyLab/SciHal-Challenge) train split
    (`subtask1_train_batch3.json`): the last token's hidden state at layer 32 of
    Llama-3.1-8B-Instruct, after the model's final RMSNorm (4096 values).
  - Labels: 0 entailment, 1 contradiction, 2 unverifiable. Fit a scikit-learn
    `LogisticRegression`, save it with `joblib.dump`, and point `MIA_SCIHAL_CLF` (or
    `scihal.clf_path` in `model_configs/hidden_states/Llama-3.1-8B-Instruct.json`) at the file.
- **Long-decode variants** of the Q/K and hidden-state demos are in
  [`docs/numerical_analysis/`](../docs/numerical_analysis/README.md).

## 8. Server mode

Over `vllm serve` the worker that captures or steers is the server's own. One server serves one
worker kind, chosen at launch with `MIA_WORKER` (`hidden_states` · `qk` · `steer` — exact, no
aliases), and runs under CUDA graphs by default — leave `cudagraph_mode` unset
([modes](../README.md#-supported-configurations)):

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
    vllm serve Qwen/Qwen2.5-3B-Instruct --max-model-len 2048 --port 8770 \
    --gpu-memory-utilization 0.8
```

- **Start it** from the repo root, in its own shell; it is ready when the log prints
  `Application startup complete.` (about a minute).
- **Run the client** in a second shell. **Stop the server** with Ctrl-C.
- **GPU memory:** a capture server needs room for the 4 GiB capture buffer outside
  `--gpu-memory-utilization`; the demos' commands pass 0.8. Go lower on a small card, or set
  `MIA_APERTURE_GPU_BYTES` ([sizing](../docs/configs.md#sizing-the-capture-aperture-and-the-gpu-memory-it-costs)).
  A steering server needs no such room.

Eager is the opt-out, and the reason to reach for it is bit-exactness — graph-mode capture can
move per-token logprobs slightly, eager cannot: add `--enforce-eager`.

**Each demo's server block.** At the end of every demo, a commented block holds the same demo over
`vllm serve`, with the exact server command to start. To run it, start that server, uncomment
`serve_main()`, and at the bottom call `serve_main()` instead of `main()`. With nothing listening,
it prints the command it needs and exits. Override the endpoint with `MIA_DEMO_BASE_URL`.

The client is `MiaClient`, the server-side counterpart of `MiaLLM`:

```python
from mia import MiaClient
from _paths import config_path
from _serve import HS, chat, require_server

url = require_server("Qwen/Qwen2.5-3B-Instruct", HS)       # or exit with the server command
client = MiaClient(base_url=url, analyzer_name="hidden_states",
                   config_file=config_path("hidden_states/Qwen2.5-3B-Instruct.json"))
response = client.generate(messages=chat("The capital of France is"),
                           model="Qwen/Qwen2.5-3B-Instruct", max_tokens=10, temperature=0.0)
stats = client.analyze(analyzer_spec={"reduce": "none"})
```

- **Steering needs no `MiaClient`**: use a plain `openai` client and put the config in
  `vllm_xargs["steer"]`, JSON-encoded (`vllm_xargs` takes scalars only). `demo_actsteer_serve.py`
  sends a different config per request.
- **Per-request knobs** go through `client.generate(..., extra_xargs={"hooks_on": "both"})` — the
  serve-path equivalent of `SamplingParams.extra_args`.
- **Token-exact prompts** use `generate_tokens()`, which goes through `/v1/completions` with the
  exact ids; `return_token_ids` makes the server report the ids it prompted on.
- **No prefix-cache reset over serve**: a Q/K server already runs without prefix caching; a
  hidden-state server that captures the same prefix twice needs `--no-enable-prefix-caching`.
- The server exposes captured activations at `/v1/mia/delivered`: protect it with `--api-key KEY`
  (and `MiaClient(..., api_key=KEY)`), or turn it off with `MIA_DELIVERY_ROUTE=0` (hidden states
  are then read from files only).
