# Task E2 — host-build bench results

Recorded run, reproduce with:

```
cd <repo root>
python tests/mia/perf/host_build_bench.py
```

Hermetic: no GPU, no vLLM engine, plugin SHA irrelevant (pure Python/numpy/torch host
code under `mia/graph/`, timed against a synthetic `StepView` — see `make_fake_step` in
`host_build_bench.py`). Environment: the pinned vLLM 0.29 environment (torch
2.13.0+cu130, numpy 2.3.5), 128-core login node. iters=200, repeats=5 (median of
repeats reported — see `_timed_ms`).

**Gotcha found and fixed in the script itself:** without pinning
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1` (+
`torch.set_num_threads(1)`), HS "vectorized" and "decode_cache" timings showed a ~20x
cliff jumping from width=64 to width=128 — an OpenBLAS/MKL thread-pool storm on this
128-core node once an array crossed some internal parallelization threshold, not a real
algorithmic effect. Matches this repo's own documented env-hygiene gotcha. The script
now sets these itself (`os.environ.setdefault`) before importing numpy/torch, so a
plain re-run reproduces the clean numbers below without external env setup.

## 1. Slope vs. concurrent capturing requests (num_layers=32)

| width | HS legacy (churn) | HS vectorized (churn) | HS PRODUCTION DEFAULT (decode, warm) | QK legacy (churn) | Steer legacy (churn) |
|---|---|---|---|---|---|
| 1   | 0.0116 | 0.0150 | 0.0133 | 0.0196 | 0.337 |
| 8   | 0.0864 | 0.0618 | 0.0462 | 0.1498 | 2.716 |
| 32  | 0.3440 | 0.2207 | 0.1551 | 0.5947 | 10.930 |
| 64  | 0.7036 | 0.4273 | 0.2995 | 1.1966 | 22.366 |
| 128 | 1.3733 | 0.8256 | 0.5877 | 2.4225 | 44.184 |
| 200 | 2.1462 | 1.2947 | 0.9126 | 3.7386 | 67.123 |

(ms/step; "churn" = fresh req ids every call / prefill-shaped; "decode, warm" = same
req ids reused, decode-cache pre-warmed once — models steady-state decode.)

Least-squares slope (`ms = a + b*width`), all R²=1.000 (perfectly linear, no noise
left once threads are pinned):

| Builder / config | slope (us per +1 concurrent req) |
|---|---|
| HS legacy (decode_cache=OFF, vectorized=OFF) | 10.72 |
| HS vectorized (decode_cache=OFF, vectorized=ON) | 6.41 |
| HS **production default** (decode_cache=ON, steady decode) | 4.51 |
| QK legacy (only builder QK has) | 18.75 |
| Steer legacy, `optimal_layer="all"` (only builder steer has on CPU — see caveat below) | 337.3 |

**Verdict on the claim "linear in concurrent capturing requests":** CONFIRMED for
every configuration measured, on V2, at R²=1.000. Vectorizing the HS scatter lowers the
slope (~40% below legacy) and the decode-cache fast path lowers it further (~58% below
legacy) — but none of them change the *shape*: it is linear before and after. No
configuration is flat or sublinear.

**Steer caveat:** `SteerRegistry.incremental_enabled` requires `device.type == "cuda"`,
so this hermetic CPU bench can only reach steer's *legacy* (non-incremental) branch,
which writes `coeff_pinned[layer, span] = ...` (three tensor slices) **per (request,
layer) pair** — genuinely O(reqs × layers), not O(reqs). That is exactly the cost
`MIA_INCREMENTAL_ROUTING` (default ON on GPU) exists to avoid; the number above is the
fallback path's cost, not production GPU behaviour. It is included because task E2
asked to bench all three builders, and it is a useful worst-case bound / regression
canary for the legacy branch.

## 2. Layers sweep at width=64 (churn), independent check of "flat in layers"

| num_layers | HS legacy | HS vectorized | QK legacy | Steer legacy |
|---|---|---|---|---|
| 1  | 0.4951 | 0.2959 | 0.9567 | 0.7785 |
| 8  | 0.5410 | 0.3204 | 1.0142 | 5.4540 |
| 16 | 0.5729 | 0.3556 | 1.0679 | 10.8303 |
| 32 | 0.6618 | 0.4018 | 1.1888 | 21.4181 |

| Builder | slope (us per +1 layer, at width=64) | R² |
|---|---|---|
| HS legacy | 5.27 | 0.996 |
| HS vectorized | 3.44 | 0.992 |
| QK legacy | 7.42 | 0.999 |
| Steer legacy | 665.8 | 1.000 |

**HS/QK: mostly flat, not perfectly flat.** Converting the per-layer slope to a
per-request-per-layer cost (divide by width=64): HS legacy ≈0.082 us/req/layer, QK
legacy ≈0.116 us/req/layer. Over 31 added layers that is ≈2.55 us/req (HS) / ≈3.6
us/req (QK) — compare against the *total* width-sweep slope at 32 layers (10.72 us/req
HS, 18.75 us/req QK): **layers explain only ~24% (HS) / ~19% (QK) of the per-request
routing-build cost; ~76-81% is layer-INDEPENDENT per-request overhead** (gating,
`extra_args_for` dict lookups, `ReqCaptureRecord`/`QKReqCaptureRecord` construction,
aperture reserve). So "flat in layers" is directionally true (the dominant cost is
per-request, not per-layer) but not literally flat — there is a small, real, linear
per-layer term neither builder eliminates.

**Steer: NOT flat in layers** on the CPU-reachable (non-incremental) branch — R²=1.000
linear in num_layers, as expected from its O(reqs × layers) design (see caveat above).
This is architecture, not a bug: it is exactly what `MIA_INCREMENTAL_ROUTING`'s GPU
default path exists to avoid, per that flag's own docstring in `registry.py` /
`install_steer.py`.

## 3. Decomposition at width=64, num_layers=32

| Component | ms/step |
|---|---|
| `ReqCaptureRecord` construction only (64 records) | 0.0240 |
| Per-request torch scatter only, legacy-style (64 individual `torch.tensor`+advanced-index writes) | 0.4644 |
| Batched numpy+torch scatter only, one-shot (idealized, no per-request Python loop at all) | 0.0325 |
| Full legacy builder | 0.6575 |
| Full vectorized builder | 0.4064 |
| legacy − vectorized (measured, from the real functions) | 0.2510 |
| isolated per-req-scatter − isolated batched-scatter (from the two microbenchmarks) | 0.4319 |
| `ReqCaptureRecord` construction as % of the full vectorized builder's total | 5.9% |

**This refines the docstring's attribution rather than simply confirming it.** The
`_route_vectorized_enabled` docstring says record construction — not the scatter — "is
the dominant O(N) term". Measured directly: record construction alone is only **5.9%**
of the vectorized builder's total, so it is not individually dominant. What *is* true,
and measured directly:

- The scatter is nearly eliminable in isolation (a clean, one-shot, fully-vectorized
  batched write costs 0.0325ms vs. 0.4644ms doing it per-request the legacy way — a
  93% reduction on that slice alone).
- Yet the *real* vectorized builder (which still loops per request in Python for
  gating + numpy-array bookkeeping, and only bathes the *final device write*) is only
  38% cheaper than the real legacy builder (0.4064 vs. 0.6575ms) — far short of the 93%
  the isolated scatter comparison would predict if scatter were the whole story.

The gap is because `_build_routing_hs_vectorized` still pays, per request, everything
the legacy loop pays **except** the individual torch scatter: `extra_args_for` dict
lookups, the `output_hidden_states`/`hooks_on`/`hs_mode` gating branches,
`list(range(num_layers))` (rebuilt fresh every request when there is no per-request
layer filter — a real O(layers) allocation per request), `np.asarray(rows_layers)`,
the `plans` dict literal, and the `ReqCaptureRecord` append. **That whole shared
per-request loop body — of which `ReqCaptureRecord` construction is one modest
(~6%) part — is the true dominant O(N) term, not the scatter and not
`ReqCaptureRecord` alone.** The docstring's *headline conclusion* ("no collapse in the
linear-in-N routing-build cost") is confirmed exactly; its specific attribution to
"constructing the ReqCaptureRecord dataclasses" is too narrow on this measurement —
the dominant cost is the whole per-request Python gating/bookkeeping loop.

## 4. A/B at width=64, num_layers=32, steady DECODE (warm)

| Config | ms/step | vs legacy |
|---|---|---|
| decode_cache=OFF, vectorized=OFF (legacy) | 0.6527 | — |
| decode_cache=OFF, vectorized=ON | 0.4081 | −37.5% |
| decode_cache=ON, vectorized=OFF (**production default**) | 0.2914 | −55.4% |
| decode_cache=ON, vectorized=ON | 0.2861 | −56.2% (matches the row above within noise) |

Confirms the code-read directly: `_build_routing_hs` checks `decode_cache` *before*
`vectorized` (`install_hs.py` ~line 470) and returns from
`_build_routing_hs_decode_cache` unconditionally when decode_cache is on — so
`vectorized` is **dead code under the current default configuration**
(`MIA_ROUTE_DECODE_CACHE` defaults to `"1"`). Rows 3 and 4 agree to within run-to-run
noise (0.2914 vs 0.2861ms), exactly as the dispatch order predicts.

## Verdict

**`MIA_ROUTE_VECTORIZED` stays OFF by default.** Three independent reasons, all
measured on V2 in this run:

1. It does not collapse the linear-in-N routing-build cost — every configuration
   measured is linear (R²=1.000) in concurrent capturing requests, on V2 exactly as
   the V1-era A/B found. The brief's premise ("V2 hands parallel numpy arrays ... so
   the per-request loop ... is now avoidable") is not supported: the per-request loop
   remains because per-request heterogeneity (hooks_on/hs_mode/layer_filter gating,
   prefill/decode phase, the shared aperture's ordered reserve) is not something the
   *existing* vectorized twin removes — it still loops in Python for all of that and
   only batches the final device-plane write.
2. Under the actual default configuration (`MIA_ROUTE_DECODE_CACHE=1`, itself already
   ON), the vectorized branch is provably unreachable — flipping its default changes
   nothing for anyone running defaults.
3. Where it *is* reachable (`MIA_ROUTE_DECODE_CACHE=0`), it is a real, consistent,
   reproducible ~37-40% ms/step reduction — but still linear, and decode_cache alone
   (already the default) already buys a larger reduction (~55%) than vectorized ever
   does on top of legacy.

No production code changed. No new implementation was written per Ruling E-4 — this
run reproduces the "no collapse" finding rather than refuting it, so per that ruling
the recorded measurement and this written finding are the deliverable.

---

## Follow-up round 2 (post-review): the `all_tokens` regime, and a standing drift risk

Review passed (spec PASS, quality approved; the reviewer independently re-ran the bench
twice and reproduced every headline number). Two non-blocking findings from that review
are addressed here. Ruling E-4 still binds: **no default changed.**

### 5. The `all_tokens` regime (a materially different code path, not just a knob)

Section 1's sweep used `hs_mode="last_token"` exclusively (the registry default, and a
reasonable place to start) — but `all_tokens` reserves and scatters the WHOLE
`tokens_per_req`-token span per request, not one row, so both the aperture reserve and
the plane-fill scatter scale with **tokens**, not just request count. This is the
regime this project's historical throughput knees actually live in (capture payload
scales by tokens). Re-ran the width sweep with `hs_mode="all_tokens"`,
`tokens_per_req=64`, `num_layers=32` (a larger `cap`/aperture sizing was needed so no
request gets truncated at the widest point: 200 reqs x 64 tokens = 12800 columns/rows —
see `CAP_ALL_TOKENS`/`TOKENS_PER_REQ_ALL_TOKENS` in the script):

| width | HS legacy (all_tokens) | HS vectorized (all_tokens) | HS PRODUCTION DEFAULT (all_tokens, churn) | QK legacy (tokens_per_req=64) |
|---|---|---|---|---|
| 1   | 0.0263 | 0.0399 | 0.0412 | 0.0266 |
| 8   | 0.2018 | 0.2419 | 0.2507 | 0.2140 |
| 32  | 0.8042 | 1.0159 | 1.0395 | 0.8499 |
| 64  | 1.6270 | 2.0174 | 2.0457 | 1.7050 |
| 128 | 3.2385 | 3.9745 | 4.0727 | 3.3953 |
| 200 | 5.0621 | 6.2768 | 6.3571 | 5.3134 |

(ms/step; full iters=200 x 5-repeat run, `python tests/mia/perf/host_build_bench.py`.)

| Series | slope (us/req) | R² |
|---|---|---|
| HS legacy, all_tokens | 25.31 | 1.000 |
| HS vectorized, all_tokens | 31.29 | 1.000 |
| HS production default, all_tokens (churn) | 31.75 | 1.000 |
| QK legacy, tokens_per_req=64 | 26.55 | 1.000 |

**Slope ratio, vectorized/legacy: 1.236 in the all_tokens regime, vs. 0.598 in the
last_token regime measured in round 1.** This is a full reversal, not just a smaller
win: vectorization is **~24% SLOWER** than the legacy loop under `all_tokens` with a
64-token span, where in `last_token` it was ~40% faster. Both fits are R²=1.000 —
clean, reproducible, not noise (confirmed on a full iters=200 x 5-repeat run after
first observing it on a 20-iter smoke run at the same ratio, ~1.25).

**Why (measured, not just theorized):** `_build_routing_hs_vectorized`'s `all_tokens`
branch (`install_hs.py:646-653` as of `4f9c2f0` — same three lines duplicated again at
`786-788` in the decode-cache slow path, itself evidence for the drift-risk finding
below) explicitly **materializes** each
request's flat `(layer, col, slot)` triples with `np.repeat(rows_np, n)` /
`np.tile(cols_span, nl)` / `np.tile(phys_np, nl)` — three new arrays of size
`num_layers * tokens_per_req` allocated and copied PER REQUEST, appended to Python
lists, then `np.concatenate`d and copied into torch tensors at the end. The legacy
loop's equivalent line is a single **broadcasted** torch write,
`capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]` — one ATen
dispatch that lets the advanced-indexing kernel handle the `(num_layers, n_tokens)`
outer-product shape without ever materializing the flattened index arrays. At
`tokens_per_req=1` (`last_token`) the materialization is trivial (arrays of length
`num_layers`) and vectorization's real win (fewer torch dispatches) dominates. At
`tokens_per_req=64` the per-request materialization cost (arrays of length
`num_layers * 64` = 2048, freshly allocated per request) outgrows what the batched
final write saves, and vectorization becomes a net loss.

**Verdict per regime:**
- `last_token` (round 1): vectorized is a real, consistent, but non-collapsing win
  (~40% lower ms/step, still linear) — reachable only when `MIA_ROUTE_DECODE_CACHE=0`.
- `all_tokens` (this round): vectorized is a real, consistent, but non-collapsing
  **loss** (~24% higher ms/step, still linear) — also only reachable when
  `MIA_ROUTE_DECODE_CACHE=0`, and additionally requires an all_tokens-mode capturing
  workload.
- **`MIA_ROUTE_VECTORIZED` staying OFF is now doubly justified, not just "not proven a
  win"**: flipping it on would actively regress the `all_tokens` regime — the regime
  the project's own historical profiling says the real throughput knees live in — while
  only helping the narrower `last_token` + `decode_cache=0` combination. This is a
  **finding and a recommendation for later work**, not a change made here: if
  `MIA_ROUTE_VECTORIZED` is ever revisited, the `all_tokens` regression should gate
  it, and the decode-cache path (which currently shadows both regimes under the
  default) would need to be involved in any such redesign and re-validated with the
  GPU parity suite — none of that is done here, per Ruling E-4.

Reproduce with `python tests/mia/perf/host_build_bench.py` (section 5 of the output); the
constants are `TOKENS_PER_REQ_ALL_TOKENS = 64` and `CAP_ALL_TOKENS = 16384` in
`host_build_bench.py`.

### Standing drift risk: three independent implementations of the same gating logic

The review traced every gate and confirmed a structural fact worth naming plainly:
**HS's routing build has three independent implementations of the same per-request
gating logic**, not one implementation with two optional accelerations layered on top:

1. **Legacy loop** — `_build_routing_hs`, `mia/graph/install_hs.py:490-558` (line
   numbers verified against the last commit to touch this file at the time of this
   round, `4f9c2f0`; another agent has uncommitted edits to this file in progress as of
   this writing — re-grep the three function names if the ranges have since moved).
2. **Vectorized loop** — `_build_routing_hs_vectorized`,
   `mia/graph/install_hs.py:591-675` (gating loop body 591-665, batched-write epilogue
   665-675).
3. **Decode-cache SLOW PATH** — `_build_routing_hs_decode_cache`,
   `mia/graph/install_hs.py:748-803`. This is a *textually separate* copy of the same
   gating (`output_hidden_states`/`hooks_on`/`hs_mode`/layer-filter/aperture-reserve)
   — it does not call into either of the other two.

All three are meant to compute byte-identical routing (the docstrings say so, and the
parity suite's builder-equivalence legs check pairs of them). The sharp edge: **the
copy that actually runs in production — the decode-cache slow path — is the one LEAST
likely to be exercised by a routing-builder A/B**, because:

- The parity suite's own pairwise coverage (`tests/mia/parity/t2_invariants.py:196-201`,
  `full_batch_legacy` / `full_batch_vec`) pins `MIA_ROUTE_DECODE_CACHE=0` explicitly on
  BOTH legs — so it validates legacy-vs-vectorized agreement, but never exercises the
  decode-cache path at all, let alone checks it against the other two under the SAME
  conditions a default deployment runs.
- This bench's own round-1 A/B (`### 4.` above) measured decode-cache's *steady-state
  fast path* (cache warm, `n_tokens==1`, not prefill) against legacy/vectorized — but
  the fast path reuses CACHED fields from a PRIOR slow-path call; it never times the
  slow path's own gating cost in isolation the way `### 1.`/`### 5.` time the legacy
  and vectorized loops. Round 2's `### 5.` "HS PRODUCTION DEFAULT, all_tokens, churn"
  row is the first number in either round that actually times the decode-cache slow
  path's cost directly (a churn/prefill workload never warms into the fast path), and
  it lands close to the vectorized line (31.75 vs 31.29 us/req) — consistent with the
  slow path sharing the same textual shape as the legacy/vectorized gating loops, but
  this is incidental agreement on THIS measurement, not a property either code path
  guarantees, precisely because the three are independent copies with no shared
  helper enforcing it.

**This is documentation of an existing hazard, not a new defect, and not something
this task restructures.** A future edit to the shared gating rules (a new
`hooks_on` mode, a new `output_hidden_states` filter shape, a cap-truncation edge
case) has three call sites to update, and the one most likely to be forgotten is the
one no dedicated A/B directly targets end-to-end against the other two under matching
conditions. Anyone touching HS routing gating should grep all three ranges above, not
just the one their diff happens to open.
