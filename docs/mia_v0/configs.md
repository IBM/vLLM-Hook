# MIA supported configurations

This document enumerates the supported configs and how to invoke each from user code.

---

## Configuration axes

| Axis | Values | How it's selected |
|---|---|---|
| **Execution path** | `serve` (`vllm serve` + `MiaClient`) | — |
| **Storage** | `rpc` (in-memory via `collective_rpc`) · `disk` (artifact under `/dev/shm/mia/<run_id>/`) · `shm` (legacy shared memory, hidden states-only) | per-request `extra_args["save_to_disk"]` (SHM via `MIA_USE_SHM=1`) |
| **Disk format** | `pt` (`torch.save`) · `st` (safetensors ) | `MIA_USE_SAFETENSORS={0,1}` |

> **Async save note:** the old per-request `sync`/`async` save-mode axis (`MIA_ASYNC_SAVE`) has been removed. It is superseded by the **writer process** (`MIA_WRITER_PROCESS`, default **on**) — a persistent child process that serializes and writes disk artifacts off the engine GIL (see the `writer_process` lever in `optimizations.py`). Unlike the old knob, this isn't a per-request axis you opt into: it's a process-wide default that's already on, so it does not appear as a selectable dimension in the coverage matrices below. It runs on **every TP rank that writes artifacts**: vLLM's daemonic TP workers used to fall back to the in-process save. Each rank logs its mode. The child exits when its worker dies (`MIA_CHILD_PARENT_POLL_S`, default `1.0` s, is how often an idle child checks).

---

## Coverage matrix

### Attention tracker 

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `attn-serve-rpc-na`     | serve   | rpc  | —  |
| `attn-serve-disk-pt`    | serve   | disk | pt |
| `attn-serve-disk-st`    | serve   | disk | st |

### Hidden states 

The same 3 combinations as above.

### CoRer

CoRer is intrinsically two-pass and only uses the disk path (the analyzer needs both runs' artifacts on disk to compute the difference). No `rpc` cells.

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `corer-serve-disk-pt`   | serve   | disk | pt |
| `corer-serve-disk-st`   | serve   | disk | st |

### Activation steering 

Steering modifies the residual stream in-place and produces no artifacts, so storage/format/async axes don't apply. Per-request via `extra_args["steer"]`.

| Cell ID | Path |
|---|---|
| `actsteer-serve-na-na`   | serve   |

---

## Selecting a configuration from user code

All hook activation is **per-request** via `extra_body["vllm_xargs"]` under `vllm serve`, or
`SamplingParams.extra_args` when driving an engine in-process. Different requests in the same
batch can use different configs.

The same code shape covers all four use cases — only `worker_name` / `analyzer_name`, or
`MIA_WORKER` on the server, varies:

| Use case | `worker_name` / `MIA_WORKER` | `analyzer_name` |
|---|---|---|
| attention tracker | `capture_qk` / `qk` | `attn_tracker` |
| CoRer | `capture_qk` / `qk` | `core_reranker` |
| hidden states | `capture_hs` / `hidden_states` | `hidden_states` |
| activation steering | `steer` / `steer` | (none — no artifacts) |

### Driving an engine in-process (`MiaLLM`)

`MiaLLM` builds a vLLM engine in your own process. It is how every demo under
`examples/mia_v0/` runs, and the quickest way to exercise capture under FULL CUDA graphs on
one machine — not a deployment path. For serving, use `vllm serve` with `MiaClient` below.

```python
import torch
from mia import MiaLLM
from vllm import SamplingParams

llm = MiaLLM(
    model="ibm-granite/granite-3.1-8b-instruct",
    worker_name="capture_qk",
    analyzer_name="attn_tracker",
    config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json",
    hook_dir="/dev/shm/mia",  # where disk artifacts are written
    dtype=torch.float16,      # attn_tracker cannot read bfloat16
)

# rpc (in-memory) path:
out   = llm.generate(text, SamplingParams(...), save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})

# disk path (artifact under /dev/shm/mia/<run_id>/); reset the prefix cache when re-capturing a prompt:
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

Format/save-mode are env-vars on the driver process, set **before** `MiaLLM(...)` is constructed (the worker subprocess inherits them at spawn):

```bash
MIA_USE_SAFETENSORS=1   # write .safetensors instead of .pt
MIA_USE_SHM=1           # legacy shared-memory fast path (hidden states + last_token only)
```

### Serve (`vllm serve` + `MiaClient` / openai client)

Start the server with `MIA_WORKER` set to the worker that matches your use case. FULL CUDA
graphs are the expected setting; swap the two graph flags for `--enforce-eager` only when you
need bit-exact logprobs:

```bash
# probes (attention tracker / CoRer / hidden states):
MIA_ALLOW_CUDAGRAPH=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=qk \
  vllm serve ibm-granite/granite-3.1-8b-instruct \
    --max-model-len 2048 --port 8770 --compilation-config '{"cudagraph_mode": "FULL"}'

# activation steering:
MIA_ALLOW_CUDAGRAPH=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=steer \
  vllm serve microsoft/Phi-3-mini-4k-instruct \
    --max-model-len 2048 --port 8770 --compilation-config '{"cudagraph_mode": "FULL"}'
```

For probe use cases, `MiaClient` mirrors the in-process `MiaLLM` API:

```python
from mia import MiaClient

hook = MiaClient(base_url="http://localhost:8770/v1",
                  analyzer_name="attn_tracker",
                  config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json")

# rpc path:
resp  = hook.generate(model=MODEL, messages=msgs, max_tokens=10)
stats = hook.analyze(analyzer_spec={...})

# disk path:
hook.generate(model=MODEL, messages=msgs, save_to_disk=True, run_id="run-2", max_tokens=1)
stats = hook.analyze(analyzer_spec={...})
```

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
resp = hook.generate_tokens(ids, model=MODEL, max_tokens=1, save_to_disk=True,
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

`extra_xargs` values must be **scalars** unless the key is one the plugin JSON-decodes
(`output_qk`, `output_hidden_states`, `steer`); a dict under any other key would reach the worker
as a string and do nothing, so the client refuses it instead. On a collision in `extra_body`,
MIA's own keys win — a caller cannot redirect `run_id` or `hook_dir`.

`analyze(probes=...)` analyzes a payload you already hold, and `.tokenizer` gives the served
model's tokenizer for computing spans (pass `tokenizer_for=<model id>` to the constructor).

#### No prefix-cache reset over serve

The in-process path calls `llm.llm_engine.reset_prefix_cache()` between captures of the same
prompt. **vLLM 0.29 exposes no endpoint for that**, so if your run captures the same prefix twice
— CoRer does — start the server with `--no-enable-prefix-caching`. A cached prefix means the
second pass captures nothing for those tokens, and the run still looks like it worked.

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
See [`examples/mia_v0/demo_actsteer_serve.py`](../../examples/mia_v0/demo_actsteer_serve.py) for a runnable example with requests using different steer configs.

`MIA_USE_SAFETENSORS` is set when launching `vllm serve` (the server's worker process reads it at hook-fire time).

---

## You set nothing: what MIA decides about the capture data path

MIA picks the capture data path per request and per file, from the configuration it already has.
The defaults below are what a user gets without setting anything; each one names the measurement
behind it and the env var that overrides it. Nothing here changes what is captured or the bytes
that are written -- only which road they take.

**1. Where a request's artifact comes back: host memory (RPC) or disk.** The per-request storage
router (`MIA_STORAGE_ROUTER`, default **on**, serve only) predicts the artifact's size from the
prompt length, the captured layers/heads and the capture mode, then compares two on-loop costs:

| side | model (ms) | where the coefficients come from |
|---|---|---|
| RPC (`get_captured_states`, `save_to_disk=false`) | `5.0 + slope x KB`, slope **0.03** HS / **0.157** QK | measured on granite-3.1-8b (FULL CUDA graph) |
| disk (`save_to_disk=true`) | `20.0 + slope x KB`, slope **0.0022** HS / **0.0078** QK | the per-KB term is measured: the per-request staging's own writes, 0.0174 ms at 8 KiB and 0.0197 ms at 16 KiB, i.e. `0.0151 ms/write + 0.000288 ms/KB` over a row-wide write. **The 20.0 ms handoff is NOT measured** -- see below |

The threshold is **solved** from those two, per worker kind, rather than written down:
**HS 539.6 KB, QK 100.5 KB** (`run_utils.rpc_disk_crossover_kb`). QK crosses five times earlier
than HS because its RPC ship is five times dearer per KB. In practice: a `last_token` HS
capture (256 KB at Llama-3.1-8B, all 32 layers) comes back over **RPC**; an `all_tokens` capture,
and QK at almost any size, go to **disk**.

> **The one number that is not measured, stated plainly.** `MIA_ROUTER_DISK_HANDOFF_MS` (20.0) is
> the engine-loop cost of handing a request to the disk route. No bench in this repo has timed it,
> and it dominates the crossover. It is kept, not replaced by a guess. A GPU measurement would
> have to time the engine loop between a disk-routed request being admitted and the loop
> continuing, at a fixed artifact size, against the same request routed to RPC. If it turns out to
> be anywhere near the write cost above, the crossover collapses toward zero and essentially
> everything routes to disk.

**An explicit `save_to_disk` from the caller always wins.** The router fires only when the request
carries no `save_to_disk` at all: an explicit value is a requirement (`true` = "I need the artifact
FILE"), not a hint, and MIA never overrides it.

| env var | default | effect |
|---|---|---|
| `MIA_STORAGE_ROUTER` | `on` | `0` disables the router; the caller's `save_to_disk` (or `MIA_SINK`) then decides alone |
| `MIA_ROUTER_T_RPC` / `MIA_ROUTER_T_ANALYZE` | derived (HS 552517 B, QK 102949 B) | override the crossover outright, in bytes, for the per-request aperture delivery route. `T_ANALYZE` takes the same derived number because it always did; its true basis is different (how big an artifact a reducible analyzer should hold in host RAM while it reduces), and nothing here measures that |
| `MIA_ROUTER_RPC_INTERCEPT_MS` | `5.0` | the RPC model's fixed term |
| `MIA_ROUTER_RPC_SLOPE_MS_PER_KB_HS` / `_QK` | `0.03` / `0.157` | the RPC model's per-KB term |
| `MIA_ROUTER_DISK_HANDOFF_MS` | `20.0` | the disk model's fixed term (**not measured**, see above) |
| `MIA_ROUTER_DISK_SLOPE_MS_PER_KB_HS` / `_QK` | `0.0022` / `0.0078` | the disk model's per-KB term (measured) |
| `MIA_ROUTER_DEBUG` | off | `1` prints the first 20 routing decisions with their predicted sizes |

Every coefficient is read on each call, so any of them can be retuned without a restart, and
retuning one MOVES the threshold -- the threshold is solved from them, never stored beside them.

**2. How a file is written: O_DIRECT or buffered.** `MIA_APERTURE_WRITE_MODE=auto` (the default)
opens a raw file `O_DIRECT` only where it is legal (the row width is a multiple of the detected
block size) **and** where it pays (the predicted write is at least **64 KiB** -- the
low end of the band where the measurement can no longer tell the two apart; below it, at one
writer thread, buffered wins decisively). MIA predicts that size at install from the capture
configuration alone, as an UPPER BOUND: an `all_tokens` capture writes up to a whole step of tokens
per file and takes O_DIRECT; a `last_token` one writes at most a row per in-flight request, so it
takes the buffered path only at low concurrency (below 8 concurrent requests at 8B, 4 at 70B).
Every rank logs its mode, the predicted size and the reason, and a prediction that real traffic
contradicts is reported once. Override the size threshold with
`MIA_APERTURE_DIRECT_MIN_BYTES`.

**3. How a disk-routed request is staged.** The per-request staging writes zero-copy through the
same writer, one fd per file kept open for the request, always buffered (its 8-16 KiB writes go
inline on the drain thread with no writer pool, where O_DIRECT measures 1.65-1.74x slower per
write). No setting selects this; `MIA_APERTURE_WRITE_MODE=legacy`
reaches the old `tobytes` + open/append/close writer for an A/B.

---

## Sizing the capture aperture, and the GPU memory it costs

The aperture is a **fixed** GPU allocation — fixed because a buffer that grows with traffic
OOMs under load. It is taken in addition to vLLM's own KV-cache budget, so the two have to fit
on the card together, and MIA checks that at engine start rather than letting the allocation
fail later.

| env var | default | effect |
|---|---|---|
| `MIA_APERTURE_GPU_BYTES` | **4 GiB** | the aperture's byte budget. **Per rank** at TP > 1, not per engine |
| `MIA_APERTURE_MAX_BATCHED_TOKENS` | off | derive (`auto`) or pin `max_num_batched_tokens` so a heavy capture's per-step transient cannot OOM at high batch. MIN-ONLY: it never raises the budget, so it is byte-identical whenever the derived cap is the larger one |
| `MIA_APERTURE_BACKPRESSURE_TIMEOUT_S` | `10` (s) | how long a step waits for the drain to free rows before `ApertureBackpressureError`. Capture blocks; it never silently drops rows. Raise it if a slow sink makes a heavy run trip the barrier |

### The one rule: `gpu_memory_utilization` must leave room for the aperture

Yes — explicitly, and it is enforced. `gpu_memory_utilization` is vLLM's flag, not MIA's
(`--gpu-memory-utilization` on the server); it tells vLLM what fraction of the card to claim for
weights and KV cache. **The aperture lives entirely in the fraction vLLM does not claim**, and
engine start fails unless:

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

## FULL-graph capture aperture: how the raw files are written

In FULL-graph (buffer/aperture) mode the drain consumer thread of each capturing rank writes the
per-layer raw files and the sidecar itself; the writer process idles. Two env vars choose how:

| env var | default | values |
|---|---|---|
| `MIA_APERTURE_WRITE_MODE` | `auto` | `auto` (O_DIRECT for each file whose rows are a multiple of the detected direct-I/O block size **and whose predicted write reaches the crossover**, zero-copy buffered for the rest) · `direct` (O_DIRECT for every file, else refused at install; ignores the size) · `buffered` (zero-copy buffered everywhere) · `legacy` (the old `tobytes` + `open`/`write`/`close`-per-step path, for A/B validation only) |
| `MIA_APERTURE_WRITE_THREADS` | `2` | writer threads per off-loop drain, 1..64 |
| `MIA_APERTURE_DIRECT_MIN_BYTES` | `65536` | the O_DIRECT crossover, in bytes per write -- the low end of the measured undecidable band; `0` decides on alignment alone |

All four modes write the same bytes: the raw files and the sidecar are byte-identical, and the
readers are unchanged. The synchronous drain (`MIA_APERTURE_SYNC_DRAIN=1`) writes buffered only.
Per-request delivery writes no shared raw files, so the per-file decision does not apply to it --
its per-request DISK staging takes the same writer in buffered mode (`legacy` for an A/B; an
explicit `direct` is refused there). `MIA_APERTURE_MMAP=1` is accepted only with `legacy`. Each
capturing rank logs one `... aperture write path (tp_rank r): ...` line at install, naming the mode
per tensor kind, the predicted write size and why it went that way, the thread count and the block
size.

---

## FULL-graph HS capture under tensor parallelism: which rank captures which layer

The residual stream is replicated on every TP rank, so any rank's copy of a layer IS the layer and
HS shards by LAYER: rank `r` captures the 0-based decoder layers `i` with `i % tp_size == r` into
its own `tp_rank_<r>/`, and each rank's aperture, drain thread and writer cover only those layers.

| env var | default | values |
|---|---|---|
| `MIA_HS_TP_SHARD` | `1` | `1` = shard the HS layers round-robin across the ranks (at TP > 1); `0` = the pre-shard layout, `tp_rank 0` captures every layer and the others bake sinks (**A/B only**). Anything else is refused at engine construction. Ignored at TP = 1 |
| `MIA_HS_CAPTURE_ALL_RANKS` | off (unset / `0`) | `1` = every rank captures EVERY layer into its own dir (replicas of one residual). Diagnostic; wins over `MIA_HS_TP_SHARD`. Anything else is refused at engine construction too -- `true`/`yes`/`on` are NOT read as `1` |
| `MIA_HS_TP_SYMMETRIC` | `1` | `0` bakes `capture_hs` only on the layers a rank owns. Expected to hang at TP > 1; kept to reproduce that |

`MIA_APERTURE_GPU_BYTES` is a PER-RANK budget: one max-token step of HS now costs
`max_num_batched_tokens × ceil(L / tp) × hidden × 2` on a rank (2.5 GiB for Llama-3.1-70B at TP4,
not 10 GiB). Read a run back with `mia.graph.aperture_reader.load_hs_aperture_tp(MIA_APERTURE_DIR)`,
which unions the rank dirs and refuses a gap or a duplicate. TP = 1 is unchanged, byte for byte.
