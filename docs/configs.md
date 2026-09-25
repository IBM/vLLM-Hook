# MIA supported configurations

This document enumerates the supported configs and how to invoke each from user code.

---

## Configuration axes

| Axis | Values | How it's selected |
|---|---|---|
| **Execution path** | `offline` (in-process `MiaLLM`) · `serve` (`vllm serve` + `MiaClient`) | -|
| **Storage** | `rpc` (in-memory via `collective_rpc`) · `disk` (artifact under `/dev/shm/mia/<run_id>/`) · `shm` (legacy shared memory, hidden states-only) | per-request `extra_args["save_to_disk"]` (SHM via `MIA_USE_SHM=1`) |
| **Disk format** | `pt` (`torch.save`) · `st` (safetensors ) | `MIA_USE_SAFETENSORS={0,1}` |

> **Async save note:** the old per-request `sync`/`async` save-mode axis (`MIA_ASYNC_SAVE`) has been removed. It is superseded by the **writer process** (`MIA_WRITER_PROCESS`, default **on**) — a persistent child process that serializes and writes disk artifacts off the engine GIL (see the `writer_process` lever in `optimizations.py`). Unlike the old knob, this isn't a per-request axis you opt into: it's a process-wide default that's already on, so it does not appear as a selectable dimension in the coverage matrices below. It runs on **every TP rank that writes artifacts**: vLLM's daemonic TP workers used to fall back to the in-process save. Each rank logs its mode. The child exits when its worker dies (`MIA_CHILD_PARENT_POLL_S`, default `1.0` s, is how often an idle child checks).

---

## Coverage matrix

### Attention tracker 

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `attn-offline-rpc-na`   | offline | rpc  | —  |
| `attn-offline-disk-pt`  | offline | disk | pt |
| `attn-offline-disk-st`  | offline | disk | st |
| `attn-serve-rpc-na`     | serve   | rpc  | —  |
| `attn-serve-disk-pt`    | serve   | disk | pt |
| `attn-serve-disk-st`    | serve   | disk | st |

### Hidden states 

Same 6 axis combinations as above, plus the legacy SHM fast-path:

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `hs-offline-shm-na` | offline | shm  | —  |

SHM is gated by `MIA_USE_SHM=1` and only supports `capture_hs` in `last_token` mode (auto-disabled otherwise; see `shm_utils.py`).

### CoRer

CoRer is intrinsically two-pass and only uses the disk path (the analyzer needs both runs' artifacts on disk to compute the difference). No `rpc` cells.

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `corer-offline-disk-pt` | offline | disk | pt |
| `corer-offline-disk-st` | offline | disk | st |
| `corer-serve-disk-pt`   | serve   | disk | pt |
| `corer-serve-disk-st`   | serve   | disk | st |

### Activation steering 

Steering modifies the residual stream in-place and produces no artifacts, so storage/format/async axes don't apply. Per-request via `extra_args["steer"]`.

| Cell ID | Path |
|---|---|
| `actsteer-offline-na-na` | offline |
| `actsteer-serve-na-na`   | serve   |

---

## Selecting a configuration from user code

All hook activation is **per-request** via `SamplingParams.extra_args` (offline) or `extra_body["vllm_xargs"]` (serve). Different requests in the same batch can use different configs.

The two execution paths are documented below. For each path, the same code shape covers all four use cases — only `worker_name` / `analyzer_name` (offline) or `MIA_WORKER` (serve) varies:

| Use case | `worker_name` / `MIA_WORKER` | `analyzer_name` |
|---|---|---|
| attention tracker | `capture_qk` / `qk` | `attn_tracker` |
| CoRer | `capture_qk` / `qk` | `core_reranker` |
| hidden states | `capture_hs` / `hidden_states` | `hidden_states` |
| activation steering | `steer` / `steer` | (none — no artifacts) |

### Offline (`MiaLLM`)

```python
from mia import MiaLLM
from vllm import SamplingParams

llm = MiaLLM(
    model="ibm-granite/granite-3.1-8b-instruct",
    worker_name="capture_qk",
    analyzer_name="attn_tracker",
    config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json",
)

# rpc (in-memory) path:
out   = llm.generate(text, SamplingParams(...), save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})

# disk path (artifact under /dev/shm/mia/<run_id>/):
out   = llm.generate(text, SamplingParams(...), save_to_disk=True, run_id="run-1")
stats = llm.analyze(analyzer_spec={...})  # uses the last run_id

# activation steering (worker_name="steer", no analyzer): no save_to_disk, difference is observed by comparing against a use_hook=False baseline.
out_steered = llm.generate(text, SamplingParams(...))
out_plain   = llm.generate(text, SamplingParams(...), use_hook=False)
```

Format/save-mode are env-vars on the offline driver process, set **before** `MiaLLM(...)` is constructed (the worker subprocess inherits them at spawn):

```bash
MIA_USE_SAFETENSORS=1   # write .safetensors instead of .pt
MIA_USE_SHM=1           # legacy shared-memory fast path (hidden states + last_token only)
```

### Serve (`vllm serve` + `MiaClient` / openai client)

Start the server with `MIA_WORKER` set to the worker that matches your use case:

```bash
# probes (attention tracker / CoRer / hidden states):
VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=qk \
  vllm serve ibm-granite/granite-3.1-8b-instruct \
    --enforce-eager --max-model-len 2048 --port 8770

# activation steering:
VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=steer \
  vllm serve microsoft/Phi-3-mini-4k-instruct \
    --enforce-eager --max-model-len 2048 --port 8770
```

For probe use cases, `MiaClient` mirrors the offline `MiaLLM` API:

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

---

## Preliminary study regarding the storage variant choice

We have done a preliminary test regarding different storage variants using hidden-states extraction as an example. Numbers below are for `last_token` mode at 512-token prompts on Qwen2-1.5B-Instruct, averaged over 4 captured-layer counts {1, 4, 16, 28} (5 timed repetitions per cell after 5 warm-up runs that are discarded).

> **Historical note:** these numbers were measured under the old per-request `MIA_ASYNC_SAVE` background-save-thread path described above, which has since been removed and superseded by the writer process. The `-async` cell labels below are archival measurement IDs only — they are not a configuration you can select today; see the coverage matrices above for what's currently selectable.

| Variant | gen (ms) | total (ms) | analyze overhead (ms) |
|---|---:|---:|---:|
| **disk-st-async** | 40.0 | 41.8 | 1.8 |
| disk-pt-async | 41.0 | 49.8 | 8.7 |
| disk-pt | 47.3 | 53.8 | 6.5 |
| shm | 53.1 | 53.1 | 0.0 |
| rpc | 65.3 | 65.3 | 0.0 |

### Takeaways

- **disk-st (via the now-removed async-save path) was the fastest measured variant.** It minimized generate-side latency (async I/O off the critical path) and produced the smallest safetensors artifact. Today's recommended equivalent is `disk` + `st` with the writer process (on by default) — it wasn't re-benchmarked against this table, but it's the closest currently-selectable configuration to what these numbers measured.
- **rpc is the slowest path across the board** — `collective_rpc` serializes the tensor through Python/IPC, and the cost grows with the captured-layer count. Avoid rpc when artifacts are large; use disk.
- **`shm` is no longer competitive** post-refactor even at `last_token`. The legacy fast-path is kept for back-compat but the disk-st variant (measured under the old async-save path) beat it on every measured cell.
