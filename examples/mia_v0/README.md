# Building your own demo

MIA installs into the **server**. The worker that captures or steers is `vllm serve`'s own
worker, so a demo is a client: start a server, send requests, read back what was captured.

## Getting started

Install MIA from the repository root — [`docs/mia_v0/README.md`](../../docs/mia_v0/README.md)
has the full environment (vLLM 0.29.0, torch 2.13.0, Python 3.12):

```bash
pip install -e . --no-deps && pip install zstandard
```

Then, in four steps:

```bash
# 1. start a server for the worker you want (hidden_states | qk | steer)
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
    vllm serve Qwen/Qwen2-1.5B-Instruct --max-model-len 2048 --port 8770 --enforce-eager

# 2. confirm MIA loaded into it — this line comes from the server, not the client
#    [graph/install_hs] HS aperture drain ON -> ...        (graph mode)

# 3. run a demo against it, from the repository root
python examples/mia_v0/demo_hiddenstate.py

# 4. it prints a shape and a norm per captured layer
```

If step 1 is missing, step 3 does not fail obscurely: every demo probes the endpoint first and
prints the exact `vllm serve` line it needs, then exits.

### Where the captured data goes

Two routes, and they put bytes in different places. Knowing which one you are on is the
difference between finding your artifact and thinking nothing was captured.

| Route | Location | Read it back with |
|---|---|---|
| in host memory (default for small artifacts) | nowhere on disk — it rides back on the response | `client.analyze(...)` |
| disk (`save_to_disk=True`) | `<hook_dir>/<run_id>/`, default `hook_dir=/dev/shm/mia`; `tp_rank_<r>/` under TP | `client.analyze(...)`, which waits for the files |
| FULL CUDA-graph capture | `$MIA_APERTURE_DIR/tp_rank_<r>/` — per-layer `hs_layer_<N>.raw` plus an `hs_aperture_meta.jsonl` sidecar | `mia.graph.aperture_reader.load_multilayer_aperture_artifact(run_dir)`, or `load_hs_aperture_tp(dir)` to union the ranks |

**Set `MIA_APERTURE_DIR`.** Unset, it defaults to `./hs_aperture_dump` (or `./qk_aperture_dump`)
**relative to the process's working directory** — and since everything here says to run from the
repository root, that means inside your clone. An all-layers, all-tokens run writes tens of GB
there. Both names are gitignored, so nothing gets committed, but a home filesystem with a quota
will notice.

In graph mode `analyze()` is **not** the way back: use the reader above. Each capturing rank
logs its directory once at install (`... aperture drain ON -> <dir>`), so the server's own log
tells you where it is going.

### Did it actually capture anything?

A run that captured nothing looks like a fast run, so check rather than assume:

- **The artifact exists.** `ls <hook_dir>/<run_id>/` on the disk route, or
  `ls $MIA_APERTURE_DIR/tp_rank_0/` in graph mode — you want `hs_layer_*.raw` **and** the
  `.jsonl` sidecar. A sidecar with no entries means the drain never saw a row.
- **The analyzer returned something.** `stats["hidden_states"]` empty is a failed capture, not
  an empty model.
- **The counters moved.** Start the server with `MIA_PROFILE=1` and read its log at shutdown:
  `captured.bytes.hs` (or `.qk`) above zero is the only signal that proves rows were captured
  and handed to the drain. Do not use a counter that merely proves the request finished.

### Tensor parallelism

Add `--tensor-parallel-size N` to the server command. Capture shards across the ranks — hidden
states by **layer** (rank `r` takes layers where `i % N == r`), Q/K by **head** — and each rank
writes its own `tp_rank_<r>/`, which the reader unions:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
    vllm serve meta-llama/Llama-3.1-70B \
    --max-model-len 2048 --port 8770 --tensor-parallel-size 4 --enforce-eager
```

```python
from mia.graph.aperture_reader import load_hs_aperture_tp
per_request = load_hs_aperture_tp(os.environ["MIA_APERTURE_DIR"])   # refuses a gap or duplicate
```

Three things to know before you try it:

- **`MIA_APERTURE_GPU_BYTES` is per rank**, so TP × 4 claims four times that much GPU in total.
  Each rank checks its own budget against `gpu_memory_utilization`; see
  [`docs/mia_v0/configs.md`](../../docs/mia_v0/configs.md) for the constraint.
- **Q/K `score` capture requires TP = 1.** Raw Q/K shards fine; the rebuilt per-head scores do
  not.
- **Pipeline parallelism is refused**, not merely untested — under PP each rank holds only its
  own stage's layers and the rest are identity placeholders that the layer matcher would hook
  anyway, so capture would return zero-filled rows for every off-stage layer.

Supported for `capture_hs`, `capture_qk` and `steer`. `_serve.py`'s helper takes `tp=` so a
demo's printed command is runnable as-is.

## 1. Start a server

One server serves one worker kind, chosen at launch with `MIA_WORKER`
(`hidden_states` · `qk` · `steer` — exact, no aliases):

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
    vllm serve Qwen/Qwen2-1.5B-Instruct \
    --max-model-len 2048 --port 8770 --enforce-eager
```

For FULL CUDA graphs, add `MIA_ALLOW_CUDAGRAPH=1` and ask for the mode explicitly — 0.29
defaults to `FULL_AND_PIECEWISE`, which MIA refuses:

```bash
MIA_ALLOW_CUDAGRAPH=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
    vllm serve Qwen/Qwen2-1.5B-Instruct \
    --max-model-len 2048 --port 8770 \
    --compilation-config '{"cudagraph_mode": "FULL"}'
```

Every demo here prints the exact command it needs if nothing is listening.

## 2. Skeleton

```python
import os

from mia import MiaClient
from _paths import config_path          # resolves MIA's configs, then upstream's
from _serve import HS, chat, require_server

MODEL = "Qwen/Qwen2-1.5B-Instruct"

if __name__ == "__main__":
    url = require_server(MODEL, HS)      # or exit with the server command
    client = MiaClient(
        base_url=url,
        analyzer_name="hidden_states",                       # what to do with the capture
        config_file=config_path("hidden_states/Qwen2-1.5B-Instruct.json"),
    )

    response = client.generate(messages=chat("The capital of France is"), model=MODEL,
                               max_tokens=10, temperature=0.0, save_to_disk=True)
    stats = client.analyze(analyzer_spec={"reduce": "none"})

    print(response.choices[0].message.content)
    for layer, tensors in sorted(stats["hidden_states"].items()):
        print(layer, tuple(tensors[0].shape))
```

Run from the repo root: `python examples/mia_v0/my_demo.py`. Override the endpoint with
`MIA_DEMO_BASE_URL`.

**Steering needs no `MiaClient`** — it produces no artifact, so there is nothing to analyze.
Use a plain `openai` client and put the config in `vllm_xargs["steer"]`, JSON-encoded
(`vllm_xargs` takes scalars only). See `demo_actsteer.py`.

**Per-request knobs** that the config file does not cover go through
`client.generate(..., extra_xargs={"hooks_on": "both"})` — the serve-path equivalent of
`SamplingParams.extra_args`.

### Token-exact prompts

`demo_corer.py`, `demo_attnlink.py` and `demo_scihal.py` score **token spans**, so they
cannot let the server apply a chat template — that re-tokenizes and the spans stop meaning
anything. They use `generate_tokens()`, which goes through `/v1/completions` with the exact
ids, and they verify it: `return_token_ids` makes the server report the ids it actually
prompted on, and `demo_attnlink.py` fails loudly if they differ from the ones its spans were
aligned to.

`demo_scihal.py` is the one to copy if you need a **continuation**: it rebuilds the second
prompt from the token ids the first pass generated, never from the detokenized text, because
detokenize-then-retokenize is not an identity.

`demo_corer.py` captures the same prefix twice, so start its server with
`--no-enable-prefix-caching`: 0.29 exposes no endpoint to reset the prefix cache, and a
cached prefix means the second pass captures nothing for those tokens.


## 2. Pick a worker and analyzer

`worker_name` decides what is captured. `analyzer_name` decides what happens to it, and is
optional — omit it if you only want the raw tensors.

| `worker_name` | captures | config section | reference demo |
|---|---|---|---|
| `capture_hs` | hidden states | `hidden_states` | `demo_hiddenstate.py` |
| `capture_qk` | attention Q/K | `hookq` | `demo_attntracker.py` |
| `steer` | — (steers instead) | `steering` | `demo_actsteer.py` |
| `spotlight` | — (steers attention) | — | `demo_spotlight.py` (not supported on vLLM 0.29) |
| `token_highlighter` | gradient influence | — | `demo_token_highlighter.py` (not supported on vLLM 0.29) |

| `analyzer_name` | reference demo |
|---|---|
| `hidden_states` | `demo_hiddenstate.py` |
| `attn_tracker` | `demo_attntracker.py` |
| `core_reranker` | `demo_corer.py` |
| `hnode_hallucination` | `demo_halludetect.py` |
| `science_hallucination` | `demo_scihal.py` |
| `token_highlighter` | `demo_token_highlighter.py` (not supported on vLLM 0.29) |

## 3. Write the config

One JSON under `model_configs/<use_case>/<model_name>.json`. Only the section matching your
worker is read.

Capture hidden states from layers 1–4, last token only:

```json
{
  "model_info": { "name": "Qwen/Qwen2-1.5B-Instruct" },
  "hidden_states": { "layers": [1, 2, 3, 4], "mode": "last_token" }
}
```

`mode` is `last_token` or `all_tokens`. For QK use `"hookq": {"hookq_mode": "last_token"}`.
For steering use `"steering": {"method": "add_vector", "coefficient": 1.0, "optimal_layer": 14,
"vector_path": "steering_vectors/mia_v0/qwen2_dummy.pt"}`.

Copy the closest existing file in `model_configs/` rather than writing one from scratch.

## 4. Get your data back

Two paths. **Pick one — mixing them silently returns nothing.**

```python
# Disk: artifacts written, analyze() reads them back.
out   = llm.generate(prompt, sp, save_to_disk=True)
stats = llm.analyze(analyzer_spec={...})

# In-memory: captures ride back on the output object.
out   = llm.generate(prompt, sp, save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})
```

Under `save_to_disk=True`, `out[0].probes` is always `None`.

For raw tensors with no analysis, use the `hidden_states` analyzer with `reduce="none"` — it
returns what was captured, unchanged.

## 5. Optional: FULL CUDA-graph mode

Capture and steering run under CUDA graphs instead of eager. Off by default.

```bash
MIA_ALLOW_CUDAGRAPH=1 python examples/mia_v0/my_demo.py
```

Your demo must also pass `enforce_eager=False` and `compilation_config={"cudagraph_mode": "FULL"}`
(vLLM 0.29 defaults to `FULL_AND_PIECEWISE`, which MIA refuses); without the env var the plugin
forces eager regardless. See `demo_capture_aperture.py`.

## 6. Gotchas

- Set the `mp.set_start_method` / env lines **before** importing `vllm`.
- Run from the repo root — config and vector paths are relative to it.
- `enforce_eager=True` is required unless you enabled graph mode.
- Call `llm.llm_engine.reset_prefix_cache()` between prompts if you capture the same prefix twice.
- Profiler counters need `MIA_PROFILE=1`; without it they are no-ops.
- Performance levers: `from mia.optimizations import describe; print(describe())`.

## 7. Notebooks

`notebooks/` has the same demos in notebook form; see [notebooks/README.md](../../notebooks/README.md)
for the kernel setup.

## 8. Running the included demos

Run every demo from the repo root, e.g. `python examples/demo_hiddenstate.py`. A few need more:

- **`demo_actsteer_serve.py`** talks to a running server. Start it in another terminal first:

  ```bash
  VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=steer \
      vllm serve microsoft/Phi-3-mini-4k-instruct --enforce-eager --max-model-len 2048 --port 8770
  ```

  Each request carries its own steer config in `extra_body["vllm_xargs"]["steer"]`, JSON-encoded,
  because `vllm_xargs` only accepts scalar values.
- **`demo_capture_aperture.py`** runs hidden-state capture under FULL CUDA graphs (it sets
  `MIA_ALLOW_CUDAGRAPH=1` itself). Pick the model with `MIA_DEMO_MODEL`:

  ```bash
  MIA_DEMO_MODEL=Qwen/Qwen2-1.5B-Instruct python examples/mia_v0/demo_capture_aperture.py
  ```
- **`demo_halludetect.py`** downloads a pre-built H-Node probe (~22 KB) into `./cache/hnode_probe/`
  on first run, from
  [hnode-probe-builder](https://github.com/Samarpit-bhatia/hnode-probe-builder/tree/master/artifacts).
  Method: *H-Node Attack and Defense in Large Language Models*, <https://arxiv.org/abs/2603.26045>.
- **`profiling_longdecode/`** holds long-decode variants of the Q/K and hidden-state demos; see
  its [README](profiling_longdecode/README.md).
