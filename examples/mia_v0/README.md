# Building your own demo

MIA installs into the **server**. The worker that captures or steers is `vllm serve`'s own
worker, so a demo is a client: start a server, send requests, read back what was captured.

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

### Demos that stay in-process

Four do not use the server, and say why at the top of the file:

| Demo | Why |
|---|---|
| `demo_corer.py`, `demo_attnlink.py`, `demo_scihal.py` | they prompt with exact token ids and check span alignment; the chat endpoint re-templates server-side, which would invalidate the spans |
| `demo_capture_aperture.py` | the local FULL-graph showcase: its determinism check needs two generations against one engine |

`demo_spotlight.py` and `demo_token_highlighter.py` do not run on 0.29 at all — MIA raises
`UnsupportedRunnerError` on the V2 runner.

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
