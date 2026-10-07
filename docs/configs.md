# MIA supported configurations

This document lists the supported configurations, every config-file key and per-request argument,
and how to invoke each from user code.

---

## Configuration axes

| Axis | Values | How it's selected |
|---|---|---|
| **Execution path** | offline (`MiaLLM` in your process) · served (`vllm serve` + `MiaClient`) | which API you call |
| **Storage** | `rpc` (on the output: `.probes`) · `disk` (artifact under `<hook_dir>/<run_id>/`) | per-request `save_to_disk` |
| **Disk format** | `pt` (`torch.save`) · `st` (safetensors) | `MIA_USE_SAFETENSORS={0,1}` |

Every capture use case runs on both paths with either storage. CoRe is disk-only: its analyzer
compares two runs' artifacts. Steering writes no artifact, so storage and format do not apply.

Disk writes go through a writer process (`MIA_WRITER_PROCESS`, default on).

---

## Model compatibility

MIA does not rewrite model files: it finds decoder layers by module name, so it works with any model
whose vLLM implementation names them one of these ways:

| Module names | Examples | Workers |
|---|---|---|
| `model.layers.N` | Llama, Qwen, Mistral, Granite, Phi-3 and similar | all |
| `transformer.h.N` | GPT-2 | all |
| `model.decoder.layers.N` | OPT | all |
| `language_model.model.layers.N` | multimodal models wrapping one of the above | `capture_hs`, `steer` |

A model that matches none of these captures nothing; the engine log says so
(`no decoder layers matched` / `no attention modules matched`).

---

## Use cases

The same code shape covers every use case — only `worker_name` / `analyzer_name`, or
`MIA_WORKER` on the server, varies:

| Use case | `worker_name` / `MIA_WORKER` | `analyzer_name` | `analyzer_spec` | Returns |
|---|---|---|---|---|
| attention tracker | `capture_qk` / `qk` | `attn_tracker` | `input_range`, `attn_func` | `score` per prompt |
| CoRe reranker | `capture_qk` / `qk` | `core_reranker` | `query_spec`, `na_spec` (+ `run_ids`) | `scores`, `ranking` per case |
| hidden states | `capture_hs` / `hidden_states` | `hidden_states` | `reduce`: `none` · `mean` · `norm` | `hidden_states` per layer |
| H-Node detector | `capture_hs` / `hidden_states` | `hnode_hallucination` | `probe_path`, `threshold` | `probabilities`, `verdicts` |
| science hallucination | `capture_hs` / `hidden_states` | `science_hallucination` | `clf_path`, `model_id`, `label_names` | `predictions`, `prediction_labels` |
| AttnLink-U | `capture_qk` / `qk` | `attnlink` | `candidates`, `candidate_spans`, `prompt_length` | `scores`, `ranking`, `selected` |
| activation steering | `steer` / `steer` | (none — no artifacts) | — | — |

Papers and demos: [`docs/use_cases/`](use_cases/README.md).

---

## Config reference

A config is one JSON file, passed as `config_file=` to `MiaLLM` or `MiaClient`. Only the sections
your worker uses are read.

| Key | Values | Meaning |
|---|---|---|
| `hidden_states.layers` | list of ints; `[]` = every layer | **1-based** hidden-state indices: layer `i` is the output of decoder block `i - 1` (`[32]` is Llama-3.1-8B's last block); `analyze()` keys it `model.layers.<i-1>`, `load_delivered` keys it `i` |
| `hidden_states.mode` | `last_token` (default) · `all_tokens` | which token positions to capture |
| `params.important_heads` | list of `[layer, head]` | the Q/K capture targets, both **0-based** |
| `hookq.hookq_mode` | `all_tokens` (default) · `last_token` | which query positions to capture |
| `hookq.capture` | `qk` (default) · `score` | `score` captures the listed heads' attention scores instead of Q/K (eager only; TP = 1) |
| `steering.method` | `adjust_rs` (default) · `add_vector` | `add_vector` adds `coefficient × dir`; `adjust_rs` moves the projection on `dir` to `avg_proj` |
| `steering.coefficient` | float | the `add_vector` scale |
| `steering.optimal_layer` | int · list of ints · `"all"` | the decoder blocks to steer, **0-based** |
| `steering.phase` | `both` (default) · `prefill` · `decode` | which passes are steered |
| `steering.positions` | `all_tokens` (default) · `last_token` | which tokens of a pass are steered |
| `steering.apply_at_all_positions` | bool | older spelling of `positions` (`true` = `all_tokens`) |
| `steering.vector_path` | path, relative to the working directory | a `torch.save` dict: `"dir"` (hidden-size vector) and, for `adjust_rs`, `"avg_proj"` (scalar) |
| `optimizations` | `{lever: value}` | sets a [performance lever](#tuning) unless its env var is already set |
| `model_info` | any | descriptive only; MIA does not read it |

Analyzer-specific sections (e.g. `scihal.clf_path`) are read by their demo. `score` capture of a
layer with no listed heads uses the single head `MIA_QK_SCORE_HEAD` (env var, default 0).

---

## Selecting a configuration from user code

All hook activation is **per-request** via `extra_body["vllm_xargs"]` under `vllm serve`, or
`SamplingParams.extra_args` when driving an engine in-process. Different requests in the same
batch can use different configs.

### Driving an engine in-process (`MiaLLM`)

`MiaLLM` builds a vLLM engine in your own process, under CUDA graphs by default
(`enforce_eager=True` opts out) — the quickest way to run capture on one machine, not a
deployment path. For serving, use `vllm serve` with `MiaClient` below.

```python
import os

from mia import MiaLLM
from vllm import SamplingParams

llm = MiaLLM(
    model="ibm-granite/granite-3.1-8b-instruct",
    worker_name="capture_qk",
    analyzer_name="attn_tracker",
    config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json",
    hook_dir=os.path.expanduser("~/mia_runs"),  # your own dir for disk artifacts
)

# rpc (in-memory) path:
out   = llm.generate(text, SamplingParams(...), save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})

# disk path (artifact under ~/mia_runs/<run_id>/); reset the prefix cache when re-capturing a prompt:
llm.llm_engine.reset_prefix_cache()
out   = llm.generate(text, SamplingParams(...), save_to_disk=True, run_id="run-1")
stats = llm.analyze(analyzer_spec={...})  # uses the last run_id
```

Steering (`worker_name="steer"`) applies the config's `steering` section; `extra_args["steer"]` overrides it per request:

```python
import json

llm = MiaLLM(model="microsoft/Phi-3-mini-4k-instruct", worker_name="steer",
             config_file="model_configs/activation_steer/Phi-3-mini-4k-instruct.json")
with open("model_configs/activation_steer/Phi-3-mini-4k-instruct.json") as f:
    base = json.load(f)["steering"]

steer = {**base, "method": "add_vector", "coefficient": 10}
out_steered = llm.generate(text, SamplingParams(temperature=0.0, max_tokens=200, extra_args={"steer": steer}))
out_plain   = llm.generate(text, SamplingParams(temperature=0.0, max_tokens=200), use_hook=False)
```

A steered request gets a prefix-cache salt from its steering, so it shares cached prefixes only
with requests steered the same way.

Format is an env var on the driver process, set **before** `MiaLLM(...)` is constructed (the
worker subprocess inherits it at spawn):

```bash
MIA_USE_SAFETENSORS=1   # write .safetensors instead of .pt
```

`MIA_WORKER`, if set, is read offline too: a value that contradicts `worker_name` is refused.

### Serve (`vllm serve` + `MiaClient` / openai client)

Start the server with `MIA_WORKER` set to the worker that matches your use case, from the repo
root, in its own shell; it is ready when the log prints `Application startup complete.`, and
Ctrl-C stops it. It runs under CUDA graphs by default; add `--enforce-eager` only when you need
bit-exact logprobs:

```bash
# probes (attention tracker / CoRe):
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=qk \
  vllm serve ibm-granite/granite-3.1-8b-instruct --max-model-len 2048 --port 8770 \
  --gpu-memory-utilization 0.8

# activation steering:
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=steer \
  vllm serve microsoft/Phi-3-mini-4k-instruct --max-model-len 2048 --port 8770
```

A capture server needs room for the capture aperture outside `--gpu-memory-utilization`
([sizing](#sizing-the-capture-aperture-and-the-gpu-memory-it-costs)).

The server exposes captured activations at `/v1/mia/delivered`: protect it with `--api-key KEY`
(and `MiaClient(..., api_key=KEY)`), or turn it off with `MIA_DELIVERY_ROUTE=0` (hidden states
are then read from files only).

For probe use cases, `MiaClient` mirrors the in-process `MiaLLM` API:

```python
from mia import MiaClient

client = MiaClient(base_url="http://localhost:8770/v1",
                   analyzer_name="attn_tracker",
                   config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json")

# rpc path:
resp  = client.generate(model=MODEL, messages=msgs, max_tokens=10)
stats = client.analyze(analyzer_spec={...})

# disk path:
client.generate(model=MODEL, messages=msgs, save_to_disk=True, run_id="run-2", max_tokens=1)
stats = client.analyze(analyzer_spec={...})
```

The disk path needs a filesystem the client and server share: the server writes into the
client's `hook_dir` (default `/dev/shm/mia`). A client on another machine uses the rpc path.

#### Three ways to send a prompt

`generate` goes to `/v1/chat/completions`, so **the server applies the chat template** and the
token layout is the server's. An analyzer that scores token spans cannot use it — the spans were
computed against a different tokenization. Use the completions entry points for those:

| Call | Endpoint | Template | Use it when |
|---|---|---|---|
| `generate(messages=[...])` | chat | server applies it | ordinary capture; you only need the text back |
| `generate_tokens(ids)` | completions | none | an analyzer scores token spans, or you need a continuation |
| `generate_text(prompt)` | completions | none | you templated the prompt yourself |

Both completions calls take one sequence or a list of them; **a list shares one `run_id`**, the
way a list passed to the in-process `generate` does.

To check that the server prompted on the ids you meant — and to read back the ids it *generated*,
which is what a second pass needs — ask vLLM for them:

```python
resp = client.generate_tokens(ids, model=MODEL, max_tokens=1, save_to_disk=True,
                              extra_body={"return_token_ids": True})
assert list(resp.choices[0].prompt_token_ids) == list(ids)   # nothing re-tokenized
continuation = list(resp.choices[0].token_ids)               # ids, never detokenized text
```

Rebuild a continuation from those ids, not from `choices[0].text`: detokenizing and
re-tokenizing is not an identity, and the drift is silent.

#### Per-request arguments

| Argument | What it does | In-process equivalent |
|---|---|---|
| `save_to_disk=` | artifact to `hook_dir/<run_id>/` instead of riding back on the response | same |
| `run_id=` | names the artifact directory; a batch shares one | same |
| `steer=` | a steering config for this request alone | `extra_args["steer"]` |
| `extra_xargs=` | any other per-request knob, e.g. `{"hooks_on": "both"}` | `SamplingParams.extra_args` |
| `capture=False` | arm nothing — a plain request | `use_hook=False` |
| `extra_body=` | vLLM's own request extensions, merged with MIA's `vllm_xargs` | — |

Knobs for `extra_xargs` (served) or `SamplingParams.extra_args` (offline); the config file sets
the capture ones, a request overrides them:

| Knob | Values | Meaning |
|---|---|---|
| `hooks_on` | `prefill` (default) · `decode` · `both` | which passes capture |
| `hs_mode` | `last_token` · `all_tokens` | hidden-state positions (`hidden_states.mode`) |
| `hookq_mode` | `last_token` · `all_tokens` | Q/K positions (`hookq.hookq_mode`) |
| `qk_capture` | `qk` · `score` | Q/K or the listed heads' attention scores (`hookq.capture`) |
| `steer` | a steering dict (JSON string over serve) | offline: merged over the config's `steering`; served: the full steering config |

`extra_xargs` values must be **scalars** unless the key is one the plugin JSON-decodes
(`output_qk`, `output_hidden_states`, `steer`); a dict under any other key would reach the worker
as a string and do nothing, so the client refuses it instead. On a collision in `extra_body`,
MIA's own keys win — a caller cannot redirect `run_id` or `hook_dir`.

`analyze(probes=...)` analyzes a payload you already hold, and `.tokenizer` gives the served
model's tokenizer for computing spans (pass `tokenizer_for=<model id>` to the constructor).

#### No prefix-cache reset over serve

The in-process path calls `llm.llm_engine.reset_prefix_cache()` between captures of the same
prompt. **vLLM 0.29 exposes no endpoint for that.** A Q/K server already runs without prefix
caching (unless you enable it); a hidden-state server whose run captures the same prefix twice
needs `--no-enable-prefix-caching`. A cached prefix means the second pass captures nothing for
those tokens, and the run still looks like it worked.

For activation steering there's no artifact to analyze, so a plain openai client suffices. Each request carries its own steer config as a JSON-encoded string under `vllm_xargs["steer"]` (vllm_xargs only allows scalar values; the plugin decodes the string back to a dict before the worker reads it). Different requests can use different configs:

```python
import openai, json

with open("model_configs/activation_steer/Phi-3-mini-4k-instruct.json") as f:
    base = json.load(f)["steering"]

client = openai.OpenAI(base_url="http://localhost:8770/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="microsoft/Phi-3-mini-4k-instruct",
    messages=[...], max_tokens=200, temperature=0.0,
    extra_body={"vllm_xargs": {"steer": json.dumps({**base, "coefficient": 5})}},
)
```
See [`examples/demo_actsteer_serve.py`](../examples/demo_actsteer_serve.py) for a runnable example with requests using different steer configs.

Set `MIA_USE_SAFETENSORS` for both the server and the process that calls `analyze()`: the readers
look for `.safetensors` files only when it is set.

---

## Sizing the capture aperture, and the GPU memory it costs

The aperture is a **fixed** GPU allocation — fixed because a buffer that grows with traffic
OOMs under load. It is taken in addition to vLLM's own KV-cache budget, so the two have to fit
on the card together, and MIA checks that at engine start rather than letting the allocation
fail later.

| env var | default | effect |
|---|---|---|
| `MIA_APERTURE_GPU_BYTES` | **4 GiB** | the aperture's byte budget. **Per rank** at TP > 1, not per engine |
| `MIA_APERTURE_MAX_BATCHED_TOKENS` | off | `auto` lowers `max_num_batched_tokens` (an integer sets the cap) so a heavy capture step fits in GPU memory at high batch; it never raises vLLM's value |
| `MIA_APERTURE_BACKPRESSURE_TIMEOUT_S` | `10` (s) | how long a step waits for free capture space before `ApertureBackpressureError`. Capture blocks; it never silently drops rows. Raise it if a slow disk makes a heavy run hit it |

### The one rule: `gpu_memory_utilization` must leave room for the aperture

`gpu_memory_utilization` is vLLM's flag, not MIA's (`--gpu-memory-utilization` on the server,
default 0.9); it tells vLLM what fraction of the card to claim for weights and KV cache. **The
aperture lives entirely in the fraction vLLM does not claim**, and engine start fails unless:

```
MIA_APERTURE_GPU_BYTES  ≤  (1 − gpu_memory_utilization) × total GPU bytes
```

On an 80 GiB card, with the 4 GiB default:

| `gpu_memory_utilization` | left for the aperture | 4 GiB default |
|---|---|---|
| 0.80 | 16.0 GiB | fits easily |
| 0.90 | 8.0 GiB | fits |
| 0.95 | 4.0 GiB | exactly at the limit |
| 0.97 | 2.4 GiB | **refused** |

On a card of 40 GB or less, vLLM's default 0.9 leaves under 4 GiB, which is refused: pass 0.8 or
lower, or set a smaller `MIA_APERTURE_GPU_BYTES`.

Two things worth knowing about that check. It is computed from the *fraction*, not from a
measurement — so it does not know about anything else sharing the card; and it runs at engine
construction, so you get a named error instead of a CUDA OOM mid-run:

```
aperture 4.00 GiB + gpu_memory_utilization=0.97 leaves no room (free margin 2.40 GiB):
lower gpu_memory_utilization or MIA_APERTURE_GPU_BYTES
```

When you set nothing, MIA also checks the default against what **one max-token step** actually
needs (`max_num_batched_tokens × row bytes × captured layers`). If a step needs more than 4 GiB
it grows the budget to fit — but only within that same free margin. If it cannot, it refuses and
names all four ways out: lower `gpu_memory_utilization`, lower `max_num_batched_tokens`, capture
fewer layers, or set `MIA_APERTURE_GPU_BYTES` yourself.

An explicit `MIA_APERTURE_GPU_BYTES` always wins and skips that growth — including when it is
*smaller* than one step needs, which is legal and will surface as `ApertureBackpressureError`
when a max-token step cannot be admitted. The demos use `gpu_memory_utilization=0.7`, which
leaves 24 GiB on an 80 GiB card and never runs into this.

At TP > 1 the budget is per rank, which is cheaper than it sounds for hidden states: a step costs
`max_num_batched_tokens × ceil(L / tp) × hidden × 2` on each rank, so 2.5 GiB for
Llama-3.1-70B at TP4 rather than 10 GiB.

---

## Reading your data back

Where each call puts its data, how to read it and when it is ready:
[examples/README.md](../examples/README.md#where-the-captured-data-goes). In short:
`out[i].probes` / `llm.analyze()` offline, `response.probes` / `client.analyze()` served,
`<hook_dir>/<run_id>/` with `save_to_disk`, and `load_delivered(<capture dir>)` for the
hidden-state files. A served capture sent without `save_to_disk` may land in
`<hook_dir>/<run_id>/` instead of on the response; `client.analyze()` reads it either way. An
explicit `save_to_disk` is never overridden.

## Limits

- Graph-mode capture can move per-token logprobs slightly; `enforce_eager=True`
  (`--enforce-eager`) is bit-exact. `MIA_ALLOW_CUDAGRAPH=0` selects eager for every engine in the
  process, `vllm serve` included.
- Q/K capture turns prefix caching off unless you set it.
- Offline score capture, and Q/K with explicit prefix caching or DP > 1, run eager; a served score
  request on a graph engine is refused (start the server with `--enforce-eager`).
- Graph steering: at most `MIA_STEER_VMAX` (16) distinct vectors per engine; more are refused. The
  table is sized at engine start (`MIA_STEER_VMAX × hidden_size`, 128 KiB at 16 × 4096 in bf16), not
  from free GPU memory.
- CoRe batch reranking (several cases in one `analyze`) fails when the cases' prompts differ in
  length; rerank one case at a time, as `demo_corer.py` does.

## Tuning

Optional performance levers and their env vars (the defaults need no change):

```python
from mia.optimizations import describe; print(describe())
```
