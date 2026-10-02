# Tolerances in the MIA parity suite

Byte-identity is the gate. Every comparison in `tests/mia/parity/` runs with
`--require-bit-exact` (`torch.equal`, no tolerance) **except the one recorded below**.
A tolerance is never widened to make a run pass: if a comparison exceeds the value here,
the finding is a real difference and belongs in the task report, not in this file.

## Every band, by its constant — the index

A band that exists in code but not in this file is a bound nobody can justify, and a band
documented here but deleted from the code is a different kind of lie. Both are now
mechanical failures rather than review findings: every bound in `tests/mia/parity/` is declared
through `t2_invariants._band(env_var, default, what)`, and
`tests/test_parity_band_index.py` asserts this table against those declarations **in both
directions, defaults included**. Naming each constant verbatim is also what lets a reader
jump from a number in the code to the measurement that justifies it, and back.

Only `STEER_ATOL` is a *tolerance* (a bit-exact gate run at a non-zero `atol`); the rest are
*bands* on comparisons that are known not to be bit-exact, each for a reason recorded below.

<!-- BAND-INDEX:BEGIN -->
| env var | default | gates | section |
|---|---|---|---|
| `STEER_ATOL` | `1e-5` | the suite's only tolerance: the bit-exact steer gates | “T1 steer” |
| `MIA_T2_REPLAY_REL_BAND` | `2e-2` | `replay-band-w1` / `-w32`, tensor-global relative | “Real-replay capture” |
| `MIA_T2_REPLAY_ROW_BAND` | `1.5e-1` | `replay-band-w1` / `-w32`, per-row relative | “Real-replay capture” |
| `MIA_T2_T0_GRAPH_BAND` | `5e-2` | `T0-graph capture-off-vs-on`, \|Δ(logprob)\| | “T0 under FULL CUDA graphs” |
| `MIA_T2_STEER_FUSED_BAND` | `5e-2` | `steer_small op-identity [BANDED]`, \|Δ(logprob)\| | “The fused steer kernel” |
| `MIA_T2_STEER_BUDGET_CEILING` | `1.5e-1` | the ceiling on `steer-gap` and `width`'s runtime-computed budgets | “The steer budget ceiling” |
| `MIA_T2_GPU_ROUTING_BAND` | `1e-2` | `gpu-routing vs host-routing`, relative | “`MIA_CAPTURE_GPU_ROUTING` under FULL graphs” |
| `MIA_T2_ROUTING_QK_REL_BAND` | `2e-2` | `qk_small routing-identity-w32` / `-w33-distinct`, tensor-global | “`qk_small routing-identity-w32`” |
| `MIA_T2_ROUTING_QK_ROW_BAND` | `1.5e-1` | the same gates, per-row relative | “`qk_small routing-identity-w32`” |
| `MIA_T2_BUILDER_REL_BAND` | `2e-2` | `hs_small` routing-builder equivalence, tensor-global | “`hs_small` routing-builder equivalence” |
| `MIA_T2_BUILDER_ROW_BAND` | `1.5e-1` | the same gate, per-row relative | “`hs_small` routing-builder equivalence” |
| `MIA_T1_BATCHED_REL_BAND` | `2e-2` | `T1-batched capture_hs` / `capture_qk`, tensor-global | “`T1-batched capture_hs` / `capture_qk`” |
| `MIA_T1_BATCHED_ROW_BAND` | `1.5e-1` | the same gates, per-row relative | “`T1-batched capture_hs` / `capture_qk`” |
| `MIA_T1_BATCHED_LOGPROB_BAND` | `5e-2` | `T1-batched ... GENERATION`, \|Δ(logprob)\| | “`T1-batched ...` GENERATION” |
<!-- BAND-INDEX:END -->

`_band()` refuses any environment value that WIDENS a documented default unless
`MIA_T2_ALLOW_BAND_OVERRIDE=1` is set, and every band prints its source line in the job log.

### ...and every enforced constant that is NOT a `_band()`

The table above reaches only what `_band()` declares, and three of the suite's enforced
tolerances deliberately do not go through it:

* `tests/mia/parity/capture_workload.py`'s first half is **arm-neutral by contract** — it names
  no MIA symbol and imports nothing from `t2_invariants`, which is what lets a driver written
  against the pre-rename package load the same file and makes the A/B a controlled
  experiment. Routing its constants through `_band()` would invert that dependency.
* `_band()` guards **one direction**: it refuses values that WIDEN a maximum. Two of these are
  **floors** — a minimum signal a positive control must exceed — where the dangerous direction
  is *downward*, so `_band()`'s refusal would guard the wrong way (the same reason
  `t2_invariants.py`'s steer-budget helper does not use it either).
* T3's ceilings are derived in-process from measurements taken in the same job and have no
  env var at all, so there is nothing to override.
* `tp_parity_probe.py` is a standalone GPU gate with its own CLI, not a `t2_invariants` leg;
  its band overrides are command-line flags that the verdict JSON flags as
  `bands_overridden: true`, rather than environment variables `_band()` would read.

They are registered here instead, and `tests/test_parity_band_index.py` applies **the same
four checks** — code→doc, doc→code, documented-value-vs-enforced-value, and prose beyond the
index row. The third is the one that matters: the enforced value is read by **importing the
module and evaluating the attribute**, not by scraping the source, so an expression like
`2.0 * 5.830579e-02` is compared as the number the gate actually uses. That check is what a
published `10 × 5.830579e-02` would have failed against an enforced `2.0 × 5.830579e-02`.

<!-- CONSTANT-INDEX:BEGIN -->
| constant | module | enforced value | what it enforces | section |
|---|---|---|---|---|
| `STEER_LIVENESS_ATOL` | `capture_workload.py` | `1e-3` | floor: a steered pass must differ from its own unsteered control by more than this, or the steer is a no-op | “T1 steer” and “`steer_small per-request-arming`” |
| `STEER_ARM_POSITIVE_CONTROL_ATOL` | `capture_workload.py` | `1.166116e-01` | floor: an ARMED request must move by more than this in the mixed-arm A/B positive control | “`steer_small per-request-arming`” |
| `STEER_UNARMED_LOGPROB_ENVELOPE` | `capture_workload.py` | `3.265401e-02` | INFO only: the measured pass-to-pass envelope for UNARMED-row logprob movement — never a gate | “`steer_small per-request-arming`” |
| `_MARGIN` | `t3_crossbranch.py` | `2.0` | the headroom T3's derived band takes over its measured control | “T3 cross-version” |
| `_REL_CEILING` | `t3_crossbranch.py` | `1.0e-1` | cap on T3's derived tensor-global relative band | “T3 cross-version” |
| `_ROW_CEILING` | `t3_crossbranch.py` | `5.0e-1` | cap on T3's derived per-row relative band | “T3 cross-version” |
| `_LOGPROB_CEILING` | `t3_crossbranch.py` | `1.0e-1` | cap on T3's derived absolute logprob band | “T3 cross-version” |
| `TP_HS_REL_BAND` | `tp_parity_probe.py` | `5e-2` | TP=N vs TP=1 HS, per (request, layer) tensor-global relative | “TP parity probe” |
| `TP_HS_ROW_BAND` | `tp_parity_probe.py` | `1.5e-1` | TP=N vs TP=1 HS, worst single row relative | “TP parity probe” |
| `TP_QK_REL_BAND` | `tp_parity_probe.py` | `1e-1` | TP=N vs TP=1 merged q / k_full, per (request, layer) tensor-global relative | “TP parity probe” |
| `TP_QK_HEAD_REL_BAND` | `tp_parity_probe.py` | `3e-1` | TP=N vs TP=1 merged q / k_full, worst single head's block relative | “TP parity probe” |
| `TP_STEER_DELTA_REL_BAND` | `tp_parity_probe.py` | `5e-2` | `‖δ_N − δ_1‖ / ‖δ_1‖` on the teacher-forced steering effect | “TP parity probe” |
| `TP_STEER_LIVENESS_FLOOR` | `tp_parity_probe.py` | `5e-1` | floor: `max|δ|` (nats) each side's steer must exceed | “TP parity probe” |
<!-- CONSTANT-INDEX:END -->

T3's bands themselves are not constants and appear in neither table: they are re-derived per
job from legs measured in that job. The ceilings above are what bound them, and their
measured values are in “T3 cross-version (`a21` vs `a29`)” below.

---

## T1 steer — `atol = 1e-5`

| | |
|---|---|
| Recorded | 2026-09-16 (task C5) |
| Applies to | `tests/mia/parity/run_parity.sh`, the `T1 steer` leg only |
| Compared | `req<N>/generation.safetensors` and `req<N>/control.safetensors` — generated token ids, per-token logprobs, cumulative logprob |
| Arm A | vanilla vLLM **0.29.0**, `VLLM_PLUGINS=""`, steering applied by `torch.nn.Module.register_forward_hook` (`tests/mia/parity/t1_reference.py::reference_steer`) |
| Arm B | MIA on vLLM **0.29.0**, V2 runner, eager, steering applied by `mia/workers/steer_worker.py` |
| Model | `microsoft/Phi-3-mini-4k-instruct` |
| dtype | `float16` weights/activations; logprobs serialized as `float64` |
| Intervention | `adjust_rs`, coefficient `1.0`, `optimal_layer=[15]` (0-based → `model.layers.15`), vector `steering_vectors/phi3_adjust_rs_test.pt` (`sha256:5848aba16c91dcc8…`), `phase="decode"`, `positions="all_tokens"` |
| Observed max\|Δ\| | **0.000e+00 — bit-exact** (LSF 1699423; every one of the 24 `DELTA` lines in the log is `0.000e+00`) |

### Why steering has a tolerance at all when capture does not

Capture is observed *directly*: MIA writes a tensor, the reference writes a tensor, and
they are the same tensor or the port is broken. Steering writes **nothing**. It mutates the
residual stream, so the only observable is the *effect* — what the mutation does to the
output distribution — and that effect is read back through the sampler as float logprobs.
The comparison is therefore numeric by nature, and a numeric comparison needs a stated
tolerance even when the expected answer is exact.

### Why 1e-5 specifically

* Both arms apply the **same op sequence to the same tensor**: MIA's
  `_steer_per_request` groups the step's requests by effective steer config, runs
  `_steer_rows` — `rows + (avg_proj − rows·dir)·dir` — ONCE per group on the block's whole
  residual output, and keeps only that group's rows (a masked select; skipped when one
  group owns every row, as T1's single request does); the reference does the same
  arithmetic, independently written, on the same rows. Verified bit-exact against
  `mia.workers.steer_worker._steer_rows` on CPU float16 before the GPU run. So the
  *expected* difference is zero, and the tolerance is insurance against a kernel picking
  a different reduction order for a differently-shaped slice — not a budget to spend.
* `1e-5` is roughly the float32 logprob floor: log-probabilities are accumulated in
  float32, where values of order 1–20 carry ~1e-6 of representational slack, so 1e-5
  admits a couple of ulps of accumulation jitter and nothing more.
* It sits **100× below** the steer-liveness floor (`STEER_LIVENESS_ATOL = 1e-3`, in
  `tests/mia/parity/capture_workload.py`), so "the two arms agree" and "the steer actually did
  something" can never be satisfied by the same difference. The measured liveness signal on
  this workload is **7.720511e-01** (identical in both arms), five orders of magnitude
  above the parity tolerance.

### What is NOT covered by this tolerance

The T1 capture legs (`capture_hs`, `capture_qk`), the **eager** T0 non-interference legs,
and the steer leg's **token ids** all still compare bit-exactly — a token flip is a delta of
≥ 1 and fails at any tolerance in this range. T0 under FULL CUDA graphs is a separate case
and is recorded below. Cross-*version* comparison (task E1) is a different question and, if
it needs a tolerance, gets its own entry here with its own measured justification.

---

## `steer_small per-request-arming` — RESOLVED: no leakage (the CONTROL was the defect)

**LSF 1725139 found 5 of 17 UNARMED requests, in a 33-request mixed-arm batch, moving by up
to 2.575421e-02 against the 1e-3 liveness floor.** This looked like a per-row steering leak
(one request's steering landing on its neighbour). It was not.

| | |
|---|---|
| Recorded | 2026-09-17 (task D6/D7) |
| Suspect finding | LSF 1725139, `steer_small per-request-arming` — 5/17 unarmed requests moved, worst 2.575421e-02 vs 1e-3 floor |
| Decisive experiment | LSF 1725670, `tests/mia/parity/run_steer_ab_leak_control.py` |
| Model / dtype | `microsoft/Phi-3-mini-4k-instruct`, float16, V2 runner, FULL cudagraph |

**Why the original finding was ambiguous.** The leg's control ran a SECOND pass over the
same 33 prompts with **nobody armed at all** — a batch composition that diverges from the
pass being judged the moment any of the 16 armed requests generates differently (a different
sampled token shifts what every later step in that request looks like, which can shift
per-step host work and padding/bucketing for the whole batch). A real per-row leak and an
artifact of comparing two differently-composed passes both predict "some unarmed rows move a
little" — the original design could not tell them apart.

**The decisive experiment held batch composition IDENTICAL.** One engine boot, the SAME 33
distinct-length prompts, SAME order, SAME alternating arming mask, SAME 16 rows armed in
**both** passes:

* **Arm A** — the 16 armed requests carry the REAL `adjust_rs` steer config.
* **Arm B** — the SAME 16 requests carry the SAME config, except `vector_path` points at an
  ALL-ZERO-`dir` clone of the vector (same `avg_proj`).

**Why not `coefficient: 0.0`.** `mia/graph/install_steer.py` and
`mia/workers/steer_worker.py` both force `coefficient = 0.0` for method `adjust_rs`
**unconditionally**, regardless of the host config — the per-token magnitude is computed
IN-KERNEL as `avg_proj − residual·unit`. So `coefficient: 0.0` on an `adjust_rs` config is
already the status quo and changes nothing: it would have been a worthless, silently-passing
control. An all-zero `dir` is the real zero-effect control instead, and it does **not** skip
the op: `current_projections = matmul(rows, unit=0) = 0`, `coeff = avg_proj − 0 = avg_proj`
(still nonzero, still computed), `rows + coeff·unit(=0) = rows` (the add still executes,
landing on an exact zero). The Triton kernel takes the identical `steer_mode=1` branch, the
identical 2-pass reduction+add, the identical per-row routing/masking as Arm A — only the
numeric content of `dir` differs.

**Result (LSF 1725670).**

| | worst delta |
|---|---|
| 17 UNARMED requests, Arm A vs Arm B | **0.000000e+00 — bit-exact, all 17** |
| 16 ARMED requests, Arm A vs Arm B (positive control) | **2.757723e-01** (weakest) |

**Conclusion: there is NO leakage.** With composition held identical, every unarmed request
is bit-exact, and the armed requests move by a whole intervention — confirming Arm B really
disabled the steering rather than skipping the op. The original gate's failure was an
artifact of comparing two differently-composed batches, not a routing defect.

**Gate.** `steer_small per-request-arming` (`tests/mia/parity/t2_invariants.py`,
`_judge_steer_per_request_arming` / `_drive_steer_mixed`) runs the A/B design — so the leg's
"control" is Arm B, not a fully-unarmed baseline. The unarmed check is **bit-exact and
FATAL** (`delta != 0.0`), STRONGER than the 1e-3-banded check it replaces, because there is
no real batch-composition noise for it to absorb. The armed check (`delta <=
STEER_LIVENESS_ATOL` fails the gate) is kept as a **positive control**: a future change that
silently disabled steering everywhere would show 0 armed movement and FAIL this gate, rather
than reading as "no leakage". Pinned by `tests/test_parity_steer_bands.py` (adds a test that
a movement of 1e-5 — two orders of magnitude under the OLD 1e-3 floor, and a value the
pre-D7 gate would have silently passed — now FAILS).

**Addendum (task D10): "reused, not reimplemented" was not yet true, and it mattered.**
Adopting the A/B design (above) was necessary but not sufficient. LSF 1731353 re-ran this
exact gate and still found 3 of 33 unarmed requests moving:

| request | max\|Δ(logprob)\| |
|---|---|
| req0 | 1.613855e-02 |
| req2 | 2.467152e-02 |
| req30 | 2.032498e-02 |

— each ~12x below the weakest ARMED effect (3.134444e-01 in that job) and the same order of
magnitude as every cross-boot logprob noise figure elsewhere in this document. The cause:
`_drive_steer_mixed` had reused only `run_steer_ab_leak_control._make_zero_vector`, and kept
its OWN hand-written copy of everything else (prompts, mask, `SamplingParams` construction,
the two `engine.generate()` calls) — a twin of `_drive_ab_leak_control`, not a call into it.
Two independently-maintained copies of "run both arms in one boot" is still two chances for
one to drift from the other, structural equivalence notwithstanding.

**Fix.** `run_steer_ab_leak_control.run_steer_ab_pass` is now the ONLY place that drives the
A/B pass; both `_drive_ab_leak_control` (this script's own driver, labels `armA`/`armB`) and
`t2_invariants._drive_steer_mixed` (the in-suite leg driver, labels `generation`/`control`)
call it directly. The unarmed check stays exactly as it was — bit-exact, fatal, not banded —
and the positive-control assertion is unchanged. Whether this closes LSF 1731353's residual
3-request finding is a question for the next GPU run, not a hermetic one: no test in this
repository can execute two CUDA kernel launches and check they agree bit-for-bit without a
GPU. What IS pinned hermetically (`tests/test_steer_ab_single_boot_reuse.py`, 8 tests) is
that the duplication is gone: neither driver calls `engine.generate` directly any more,
`t2_invariants.py` defines no second `run_steer_ab_pass`-shaped function, and each driver
passes the label pair its own reader expects.

**Addendum (task D12): the unarmed BIT-EXACT LOGPROB premise itself was false — moved the
gate to the deterministic channel.** Dedup (task D10) was necessary but did not close LSF
1731353's residual 3-request finding, and a fourth attempt (task D11) found a stronger
result: comparing three jobs' on-disk artifacts by SHA1 showed every unarmed row taking one
of a small number of bitwise-distinct, cross-job-reproducible values, and two runs of the
identical code flagging two nearly disjoint request sets. That is not a parameter difference
or a fixed mis-route — it is evidence the A/B design has no pass that differs from another
in NOTHING, so it cannot measure whether its own "unarmed rows are bit-exact" premise is
even true.

| | |
|---|---|
| Recorded | 2026-09-17 (task D12) |
| Decisive experiment | LSF 1733997, `tests/mia/parity/run_steer_ab_null_control.py` (kept in the tree as the evidence for this decision) |
| Design | ONE engine, ONE boot: P1 real, P2 zero, P3 real (NULL for P1), P4 zero (NULL for P2), P5 real vector BYTES at a different `vector_path` (PATH) |

**Result.**

```
GATE 1 (P1 real vs P2 zero):        unarmed worst=3.265401e-02  moved=9/17  token-id-flips=0 ; armed worst=1.216572e+01
GATE 2 (P3 real vs P4 zero):        unarmed worst=0.000000e+00  moved=0/17  token-id-flips=0 ; armed worst=1.216572e+01
PATH  (P1 real vs P5 same-bytes-different-path): unarmed worst=3.265401e-02 moved=9/17 token-id-flips=0 ; armed worst=5.830579e-02
VERDICT NULL: PASS-TO-PASS NOISE -- two passes differing in NOTHING disagree on unarmed rows, so the gate's bit-exact premise is FALSE
[NULL] token-id flips on unarmed rows, all pairs: 0
```

**GATE 1 and GATE 2 are the SAME comparison, run twice in one boot, with nothing else
different.** One draw measures 3.265401e-02 on the worst unarmed row; the other measures
EXACTLY 0.0. So "an honest unarmed delta is EXACTLY 0.0" was never a property of this
system — it was one lucky draw (LSF 1725670) from a distribution the same code samples
differently every time, and no wording of the gate on that channel can be made to hold,
because the channel itself is not reproducible pass-to-pass, even within a single engine
boot. The PATH pass (same real vector bytes, a different `vector_path`) also moved ARMED
rows by 5.830579e-02 — pure path noise, no steering difference at all — which is 58x above
the old positive-control floor (`STEER_LIVENESS_ATOL = 1e-3`), so that floor could not have
told a silently-disabled Arm B from a live one either.

**What IS reproducible: unarmed TOKEN IDS never flip.** 0 flips across every pair in the
job — both NULLs, the PATH pass, and both GATE draws. That is the deterministic channel.

**Fix.** `_judge_steer_per_request_arming` now adjudicates:

* **UNARMED rows — TOKEN IDS bit-exact between Arm A and Arm B (FATAL, unbanded).** Same
  standard as the pre-D12 logprob check ("any difference is a leak"), moved to the channel
  LSF 1733997 showed is actually reproducible.
* **ARMED rows — the positive control (FATAL, unbanded), floor raised.**
  `STEER_ARM_POSITIVE_CONTROL_ATOL` (`tests/mia/parity/capture_workload.py`) replaces
  `STEER_LIVENESS_ATOL` for this gate: **`2.0 × 5.830579e-02 = 1.166116e-01`**, which sits
  strictly between the two measured quantities it has to separate:

  | anchor | measured | margin |
  |---|---|---|
  | LOWER — PATH-noise ceiling it must EXCEED | `5.830579e-02` (LSF 1733997) | **2.00× above** |
  | (and the unarmed pass-to-pass envelope, `STEER_UNARMED_LOGPROB_ENVELOPE`) | `3.265401e-02` | 3.57× above |
  | UPPER — weakest genuine per-request armed effect it must NOT exceed | `2.757719e-01` (LSF 1737784) / `2.757723e-01` (LSF 1725670) | **2.36× below** |

  A future change that silently disabled steering everywhere still shows ~0 armed movement
  and FAILS.

  **The `10×` floor (`5.830579e-01`) this replaces was WITHDRAWN as broken (task D13), and
  why is the more instructive half.** It was anchored on `1.216572e+01` — the worst effect
  across a whole real-vs-zero *batch* comparison — and used as a bound on a *single request*.
  That put it **above** the weakest genuine per-request armed effect (`2.757719e-01`), so **no
  correct steering implementation could ever clear it.** LSF 1737784 req29 failed exactly
  there: `ARMED but did not clear the positive-control floor (2.757719e-01 <= 5.830579e-01)`.
  Its advertised "≈20.9× below the live-steering signal" was the same category error — the
  ratio was computed against the batch-worst effect, not against the per-request signal the
  gate actually sees.

  **A floor above the real signal is not a strict gate; it is a gate no correct
  implementation can pass.** It fails honest code and says nothing about broken code, which
  is strictly worse than no gate at all: a gate that always fails gets waived, and a waived
  gate protects nothing. The margins above are asserted hermetically by
  `tests/test_parity_steer_bands.py::test_arming_gate_atol_sits_between_noise_and_signal`
  off these same recorded constants, so a future edit that pushes the floor back above the
  real signal (or below the noise) fails a unit test rather than a GPU job an hour later.
* **UNARMED logprob movement — demoted to INFO, never fatal.** Printed against
  `STEER_UNARMED_LOGPROB_ENVELOPE = 3.265401e-02` (the worst pass-to-pass spread LSF 1733997
  measured on this exact quantity) so the number stays visible without ever failing a run on
  a channel proven irreproducible pass-to-pass.

**The no-leakage property established by this gate now rests on the token-id channel plus
the armed positive control — not on unarmed-logprob bit-exactness.** Unarmed-row logprobs
are real information (still printed, still worth watching for a change in the noise
envelope itself) but are not, and never were, a fact about the system precise enough to gate
on. Pinned hermetically (`tests/test_parity_steer_bands.py`): a synthetic unarmed token-id
flip still FAILS the gate; a synthetic silently-disabled steering (an armed request not
moving) still FAILS the gate, against the new floor; a synthetic unarmed logprob movement at
the full measured envelope (3.265401e-02), with token ids intact, does **NOT** fail the gate
— superseding the task D7 test that asserted the opposite.

---

## T0 under FULL CUDA graphs — a measured LIMITATION, not a tolerance

**Band constant:** `MIA_T2_T0_GRAPH_BAND` = `5e-2` (`t2_invariants.py:384`), gating
`<workload> T0-graph capture-off-vs-on`. Derivation at the end of this section.

This is not a budget anyone may spend; it is a property of the system that users of graph
mode need to know. **Under `cudagraph_mode=FULL`, arming MIA's capture leaves the generated
TOKEN IDS bit-identical but does NOT leave the per-token LOGPROBS bit-identical.**

| | |
|---|---|
| Recorded | 2026-09-16 (task D4) |
| Measured | LSF 1702981 |
| Control | vanilla vLLM 0.29, `VLLM_PLUGINS=""`, `cudagraph_mode=FULL`, no hooks anywhere |
| Arm | MIA on the same engine + mode, capture armed |
| Model / dtype | `microsoft/Phi-3-mini-4k-instruct`, float16 |

Observed `max|Δ(token_logprobs)|`, capture-OFF vs capture-ON, **both under FULL graphs**:

| request | `capture_hs` | `capture_qk` |
|---|---|---|
| req0 | **1.471710e-02** | 5.364418e-07 |
| req1 | 9.862185e-04 | 1.192093e-07 |
| req2 | 4.032373e-03 | 0.000000e+00 |

Token ids: **identical on every request, both workloads.**

**Cause.** MIA's capture is a baked op inside the traced forward, so arming it changes what
inductor is given to fuse. The model computes the same thing through slightly different
fused kernels, and fp16 rounding differs. In **eager** mode nothing is compiled, the op is a
plain call, and T0 is bit-exact (task C4) — so compilation is the only variable.

**The HS/QK asymmetry is the evidence for where it lives.** QK is 4–5 orders of magnitude
smaller than HS, and QK wraps `Attention.forward` while HS wraps the whole DECODER LAYER and
reads its residual pair. The locus is the HS decoder-layer residual path, not graph mode in
general.

**What this means for a user.** Token ids, and therefore generated text, are unaffected.
If you consume `logprobs` and need them bit-reproducible against a run without capture, use
eager (`enforce_eager=True`), where T0 is exact.

**Gate.** `tests/mia/parity/t2_invariants.py` bounds this rather than asserting equality:
`_T0_GRAPH_LOGPROB_BAND = 5e-2` on the logprobs (2.8x above the measured 1.815024e-02,
~15x below the 7.72e-01 scale of a real intervention), with the **token ids held bit-exact**
inside the same gate. A regression still trips it.

---

## Trajectory bifurcation under FULL graphs — the suite now DETECTS it instead of misreading it as corruption

**A near-tie greedy decode step can flip its argmax under a FULL-graph compile, and from
that step on the two arms generate — and capture — DIFFERENT TOKENS.** This follows directly
from the limitation above: compiling the model moves logprobs by a small, already-measured
amount (up to 1.8e-02 nats), and at a step whose top-1/top-2 candidates are within that same
margin of each other, the perturbation is enough to change which token is chosen. Every
`layer*.safetensors` row from that point on then holds the hidden state / Q / K of a
DIFFERENT sampled token, not a corrupted one.

| | |
|---|---|
| Recorded | 2026-09-17 (task D8 diagnosis, task D9 fix) |
| Diagnosed | LSF 1726855, `hs_small routing-identity-w33-distinct-replay`, req25 (`.superpowers/sdd/2026-09-16-mia-v2sup-port/task-D8-report.md`) |
| Divergence | decode index 3; `prompt_len=36` → first affected row 39, measured first bad row **39** (exact) |
| Cause | FULL arm `token_logprobs[3]` moved **1.15e-02 nats** vs the eager arm, at a step whose top probability was only ~9% (a genuine near-tie) — inside the T0-graph band above (worst measured 1.815024e-02) and the `MIA_CAPTURE_GPU_ROUTING` precedent (1.977980e-02) |
| Rows 0–38 (shared prefix) | max per-row relative **8.104e-04** — the ~1-ULP floor, i.e. correct |
| Rows 39–50 (post-divergence) | max per-row relative up to **9.248e-01** — the hidden state of a DIFFERENT token, not garbage: `full/req25` row39 matches OTHER requests' row39 that also fed token 1068 (4.76e-03 – 6.90e-03), and sits 2.11e-01 from `eager/req25` row39 (which fed token 14448) |
| MIA's capture | EXONERATED: the third arm, MIA's own baked-op + aperture capture on the SAME batch, is **bit-exact** to the eager forward-hook arm at all three layers (`maxabs = 0.000e+00`) |

**Why this used to read as corruption.** Every eager↔FULL replay leg
(`routing-identity-w33-distinct-replay`, `replay-band-w32`, `replay-band-w1`) fixes
`max_tokens`, so a bifurcation keeps ROW COUNTS EQUAL — it cannot surface as the shape
mismatch the `*_batch_distinct` design otherwise relies on to catch a whole-request problem
with no tolerance involved. It surfaces instead as an unabsorbable magnitude blowup on the
rows that consumed the divergent token (measured 4.53e-01 relative at layer 1 — within 0.4%
of the suite's own documented off-by-one-row FLOOR, 4.512e-01), which is what makes it look
exactly like a mis-route. `_compare_replay_band` never looked at `generation.safetensors` at
all, so nothing before this fix could tell the two apart.

**Gate.** `tests/mia/parity/t2_invariants.py::_compare_replay_band_checked` reads
`generation.safetensors` for every request FIRST, on all three eager↔FULL replay legs:

* a divergence in the generated token ids is reported EXPLICITLY, before any tensor
  `MISMATCH` line — naming the request, the decode index, both token ids, and the logprob
  delta that (should) explain it;
* the captured tensors are then compared only for the rows BEFORE the divergence (the
  shared prefix — still a live, banded comparison, exactly as if no divergence had
  happened); the post-divergence rows are excluded from every band, not merely tolerated by
  one, because they are not the same quantity on both sides;
* the verdict states how many rows were compared and how many were skipped.

**A bifurcation is a documented, non-fatal condition — NOT a blanket excuse.** It is the
same published FULL-graph logprob limitation as the T0-graph section above, so a bifurcation
with an explanatory logprob delta does not by itself fail the leg. It still fails when:

1. the pre-divergence rows disagree beyond the existing band (a bifurcation only excuses the
   rows it cannot explain), or
2. the bifurcation has **no** explanatory logprob delta — greedy sampling from bit-identical
   logits cannot produce two different argmaxes, so an unexplained flip is a DIFFERENT bug
   (a token-stream misalignment, a stale index read), not a tie-flip, and must fail loudly.

The "no explanatory delta" floor reuses `_STEER_ATOL` (`1e-5`, the suite's own bit-exact
logprob tolerance) rather than a new invented number. Pinned by
`tests/test_trajectory_bifurcation.py` (7 tests: a synthetic bifurcation is detected and
reported at the right row and does not by itself fail; corruption strictly before the
divergence still fails; a bifurcation with a zero logprob delta fails; a bifurcation with no
logprob channel at all fails; a non-diverging pair is judged identically to the un-checked
gate).

---

## `MIA_CAPTURE_GPU_ROUTING` under FULL graphs — a measured LIMITATION

**Band constant:** `MIA_T2_GPU_ROUTING_BAND` = `1e-2` (`t2_invariants.py:420`), gating
`<workload> gpu-routing vs host-routing [BANDED]` and its no-tail sibling.

**The lever is OFF by default.** With it ON, captured artifacts are not bit-reproducible
against the host-routing path under FULL CUDA graphs.

| | |
|---|---|
| Recorded | 2026-09-16 (task D4) |
| Measured | LSF 1702981, 1704450 |
| Compared | `full_batch` vs `full_batch_gpuroute`, same model, same mode, same width |

| width | relative delta (`max|a-b| / max|a|`) |
|---|---|
| 32 | 2.228e-03 (1702981) / **0.000000e+00** (1704450) — boot-dependent |
| 33 (no tail wave) | 1.876e-03, on all 33 requests |

**This is not a mis-route, and the evidence is generation, not argument.** Generation itself
differs between the two legs (33/33 requests, `max|Δ(logprob)|` 1.977980e-02, token ids
identical throughout). Capture cannot alter generation, so the two arms did not run the same
forward; the captured delta is downstream of an input-side difference. The magnitudes agree:
the delta grows with depth (1.33e-05 / 1.65e-05 / 1.88e-03 at layers 1 / 16 / 32), which is
accumulation, while a mis-route is full-scale — an off-by-one row measures 4.51e-01
tensor-global / 5.96e-01 per-row (the FLOOR over tensors, LSF 1705469) and a zeroed row 1.0
per-row, 240-530x larger.

Raising `max_num_seqs` from 32 to 33 to remove the scheduling tail did **not** remove the
difference; it made it global. That rules out a tail effect and fits a padding / bucketing
change driven by the lever's different per-step host work.

**Gate.** `_GPU_ROUTING_REL_BAND = 1e-2` (4.5x above the worst measurement, 45-100x below
the two mis-route signatures: an off-by-one row floors at 4.51e-01 and a zeroed row at 1.0
per-row), plus the per-row band and a bit-exact **token-id** check on generation. The
bit-exact artifact comparison is still recorded as an INFO line so a run that achieves 0
stays visible.

The one thing no value gate on the repeated-prompt workload can see is a whole-request
mis-route between two SAME-SHAPE requests: the same-shape pairs there are same-prompt
replicas, whose mutual difference is only 2.23e-03 (HS) / 1.11e-03 (QK) — under the band.
The `*_batch_distinct` legs remove the possibility by giving all 33 requests different
prompt lengths, so such a mis-route becomes a shape mismatch — fatal, no tolerance.

---

## `qk_small routing-identity-w32` — a measured LIMITATION (vLLM's nondeterminism, not MIA's)

**Band constants:** `MIA_T2_ROUTING_QK_REL_BAND` = `2e-2` (tensor-global relative) and
`MIA_T2_ROUTING_QK_ROW_BAND` = `1.5e-1` (per-row relative), `t2_invariants.py:506-510`.
They gate `qk_small routing-identity-w32`, `-w33-distinct` and `-w33-distinct-replay`.

**QK's `routing-identity-w32` gate is no longer bit-exact.** It compares MIA's two capture
mechanisms — the plain forward-hook path and the baked in-graph aperture op — over the same
33-request, width-32 batch. It passed bit-exact in LSF 1705469 and 1705794, then failed in
1710144 with **no code change** in between. `hs_small`'s `routing-identity-w32` is unaffected
and stays bit-exact.

| | |
|---|---|
| Recorded | 2026-09-17 (task D4c) |
| Diagnosed | LSF 1710884 — 3 boots per arm (A1/A2/A3 = forward-hook, B1/B2/B3 = aperture) |
| Compared | `full_batch`-width QK artifacts (`q`, `k_full`), same model, same mode, same code, same GPU model |

**Neither arm reproduces itself boot-to-boot, and the cross-arm delta is the same magnitude
as the within-arm delta** — bit-exact `compare_artifacts.compare` (absolute `max|d|`) over
the raw LSF 1710884 artifacts:

| pair | problems | max\|Δ\| | | pair | problems | max\|Δ\| |
|---|---|---|---|---|---|---|
| A1 vs A2 (same arm) | 24 | 3.281e-02 | | A1 vs B1 (cross-arm) | 12 | 3.507e-02 |
| A1 vs A3 (same arm) | 48 | 3.281e-02 | | A2 vs B2 (cross-arm) | 168 | 1.562e-02 |
| A2 vs A3 (same arm) | 24 | 3.281e-02 | | A3 vs B3 (cross-arm) | 48 | 3.281e-02 |
| B1 vs B2 (same arm) | 192 | 1.986e-02 | | | | |
| B1 vs B3 (same arm) | 12 | 3.507e-02 | | | | |
| B2 vs B3 (same arm) | 192 | 1.921e-02 | | | | |

Token ids: **identical on every request, both arms, all 3 boots (0/33 differ)** — generation
is stable, so this is sub-argmax noise.

**Not a MIA defect.** Arm A is the plain forward-hook path — it only *reads* what vLLM's
attention computed. If that varies boot-to-boot with no code change, the variation is in
vLLM's attention kernel, not in MIA's capture. This is consistent with attention-internal
reduction order (split-K / atomics / per-boot kernel autotuning). It is also **not** "QK is
always noisy": the `w33-distinct` legs passed **bit-exact for both `hs` and `qk`** in the
same LSF 1710144 job that failed this gate — the effect is width/composition-dependent, not
a property of QK capture in general.

**Why relative, not absolute.** An absolute `max|d|` of ~3.3e-02 on `q`/`k_full` does not by
itself separate noise from a mis-route — by rough estimate it is "roughly 1e-2" relative, and
an absolute band that loose would barely discriminate a real defect. Measuring the *actual*
relative and per-row metric (`_compare_replay_band`'s own metric, reused rather than
reimplemented) directly on the same LSF 1710884 artifacts, over all 3 same-arm-A, 3
same-arm-B and 3 matched-boot cross-arm pairs (198 float tensors per pair), gives a much
tighter picture:

| | worst tensor-global relative | worst per-row relative |
|---|---|---|
| measured (LSF 1710884, all 9 pairs) | 1.446e-03 | 1.880e-03 |

All 9 pairs land in `[1.216e-03, 1.880e-03]` on both metrics, same order whether same-arm or
cross-arm — the signature of a shared noise floor, not a mis-route.

**Gate.** `_ROUTING_IDENTITY_QK_REL_BAND = 2e-2` / `_ROUTING_IDENTITY_QK_ROW_BAND = 1.5e-1`
(`tests/mia/parity/t2_invariants.py`), the same magnitudes `_REPLAY_REL_BAND` /
`_REPLAY_ROW_BAND` already use for the real-replay leg, kept as separate constants so the
two gates can be retuned independently:

* rel band 2e-2: **13.8x above** the worst measured boot spread (1.446e-03), **22.6x below**
  the off-by-one-row floor measured elsewhere in this suite (4.512e-01 tensor-global, LSF
  1705469).
* row band 1.5e-1: **79.8x above** the worst measured boot per-row spread (1.880e-03),
  **4.0x below** the off-by-one-row per-row floor (5.959e-01, LSF 1705469) and **6.7x below**
  a zeroed/sentinel row (1.0 exactly). The 4.0x margin is the thinnest of the four — the
  same margin `_REPLAY_ROW_BAND` already runs on for its own gate, not a new risk, but the
  number to watch if a future off-by-one measurement on this workload comes in below 5.959e-01.

Integer metadata (`layer_num`, `k_prefix_ends`) and shape stay **bit-exact** inside this same
gate — `_compare_replay_band` never bands them, so a mis-route that changes row counts still
fails as a SHAPE mismatch, no tolerance involved. The bit-exact comparison is still recorded
as an INFO line so a run that achieves 0 stays visible. Pinned by
`tests/test_qk_routing_identity_band.py` (6 tests: accepts the measured boot spread, rejects
an off-by-one row, a zeroed row, a shape mismatch, and integer-metadata drift; asserts the
margins above numerically).

### `qk_small routing-identity-w33-distinct` — the SAME limitation, now also banded (task D9)

**Band constants:** the same `MIA_T2_ROUTING_QK_REL_BAND` / `MIA_T2_ROUTING_QK_ROW_BAND`
as the section above — one pair of bounds for one measured phenomenon at two widths.

**Separately diagnosed (task-D8-report.md) on the SAME job (LSF 1726855) that produced the
trajectory-bifurcation finding above, but unrelated to it.** `qk_small
routing-identity-w33-distinct` compares MIA's two capture mechanisms over the 33-distinct-
prompt-length batch — the same comparison as `routing-identity-w32` above, at a different
width — and it is not this leg that bifurcated: **generations were IDENTICAL on all 33
requests.**

| | |
|---|---|
| Recorded | 2026-09-17 (task D9) |
| Compared | `1726855/t2_qk_small/{eager,aperture}_batch_distinct`, 33 distinct-length requests |
| Shapes / integer metadata | bit-exact (`layer_num`, `k_prefix_ends`) |
| Float tensors differing | 156 of 396 |
| Worst tensor-global relative | **1.870e-03**, layer031 `q`/`k_full`, across ALL 33 requests |

That measurement lands **inside** the `[1.216e-03, 1.880e-03]` boot-spread cluster
`routing-identity-w32`'s band was built on (LSF 1710884) — **10.7x under** the `2e-2` band
and **241x under** the off-by-one-row floor (4.512e-01). The "~1e-2 to 2e-2" figures a raw
log shows for this leg are `_compare`'s **absolute** `max|d|`; `q`/`k_full` values run to
tens in magnitude, so that absolute number is not the relative one the band reads — the same
conversion the `routing-identity-w32` section above already calls out. Arm A only *reads*
what vLLM's attention computed, so boot-varying split-K / atomics / autotuning is vLLM's
variance here too, not MIA's.

**Gate.** Banded with the **identical** `_ROUTING_IDENTITY_QK_REL_BAND` /
`_ROUTING_IDENTITY_QK_ROW_BAND` constants as `routing-identity-w32` — not a new bound, since
it is the same noise source at a different width. Integer metadata and shape remain
bit-exact and fatal inside this gate, exactly as for `w32`. `hs_small
routing-identity-w33-distinct` is **UNCHANGED** — HS is bit-reproducible on this leg and
stays bit-exact; do not band it. Pinned by `tests/test_qk_w33_distinct_band.py` (the gate
label routing decision per workload, and that the measured 1.870e-03 delta — which fails a
bit-exact `_compare` — passes through `judge_capture`'s banded gate and still rejects an
off-by-one row).

---

## `hs_small` routing-builder equivalence — a measured LIMITATION (vLLM's nondeterminism, not a MIA lever)

**Band constants:** `MIA_T2_BUILDER_REL_BAND` = `2e-2` (tensor-global relative) and
`MIA_T2_BUILDER_ROW_BAND` = `1.5e-1` (per-row relative), `t2_invariants.py:563-566`,
gating `hs_small builder legacy vs default` and `hs_small builder vectorized vs default`.

**HS capture at the batch tail is not bit-reproducible across boots at width 32/33, and this
applies to ALL THREE interchangeable HS routing builders equally.** `hs_small builder legacy
vs default` and `hs_small builder vectorized vs default` compare MIA's three interchangeable
host-side routing-table builders — legacy (`_build_routing_hs`, per-request host scatter),
vectorized (`_build_routing_hs_vectorized`), and decode-cache (`_build_routing_hs_
decode_cache`, the **default**) — under real FULL-cudagraph prefill+decode at width 32/33.
These two gates used to require bit-exact agreement (carried item 2 of the task brief).

| | |
|---|---|
| Recorded | 2026-09-17 (task D4f) |
| Diagnosed | LSF 1722805 — 3 boots per builder (default, legacy, vectorized), 40 comparison pairs |
| Compared | `full_batch`-width HS artifacts (`hidden_states`), same model, same mode, same code, same GPU model |
| Do not re-run | the diagnosis is complete; this file encodes it |

**Every builder disagrees with ITSELF across boots, at the same tail requests** — the same
requests each time (roughly reqs 28–32, the second prefill wave at this batch width):

| builder (self, boot-pair) | problems |
|---|---|
| default | 3, 9, 9 (reqs 30–32) |
| legacy | 3, 6, 3 (reqs 28–29) |
| vectorized | 0, then 9, 9 (reqs 30–32) |
| default, width 33 | 6 (reqs 30–31) |

A bit-exact gate **between two different builders** was never achievable: the noise floor
already exceeds it **within** a single builder, boot to boot.

**The builders are genuinely equivalent — confirmed, not merely undisprovable.** Several
CROSS-builder pairs agree **bit-exactly**: `defaultboot3-vs-vectorizedboot1`,
`defaultboot3-vs-vectorizedboot2`, `legacyboot3-vs-vectorizedboot1`,
`legacyboot3-vs-vectorizedboot2`, and `defaultboot3-vs-legacyboot3`. If legacy and vectorized
computed different routing planes they could never land on identical bits on *any* pair —
this is positive evidence for carried item 2 (the builders are interchangeable), not a
weakening of the gate. `token_ids_differ=no` on all 40 pairs — generation is unaffected; the
noise is sub-argmax, the same character as the `qk_small routing-identity-w32` limitation
above, now observed on HS's own routing builders rather than on vLLM's attention kernel.

**Gate.** The existing relative + per-row band (`_compare_replay_band`, reused rather than
reimplemented) already passes on all 40 diagnosis pairs:

| | worst tensor-global relative | worst per-row relative |
|---|---|---|
| measured (LSF 1722805, all 40 pairs) | 2.230e-03 | 8.348e-03 |

`_BUILDER_EQUIV_REL_BAND = 2e-2` / `_BUILDER_EQUIV_ROW_BAND = 1.5e-1`
(`tests/mia/parity/t2_invariants.py`) — the same magnitudes `_REPLAY_REL_BAND` /
`_REPLAY_ROW_BAND` and `_ROUTING_IDENTITY_QK_REL_BAND` / `_ROUTING_IDENTITY_QK_ROW_BAND`
already use, kept as separate constants so this gate can be retuned independently:

* rel band 2e-2: **~9.0x above** the worst measured pair (2.230e-03), **~22.5x below** the
  off-by-one-row floor measured elsewhere in this suite (4.512e-01 tensor-global, LSF
  1705469).
* row band 1.5e-1: **~18.0x above** the worst measured per-row pair (8.348e-03), **~4.0x
  below** the off-by-one-row per-row floor (5.959e-01, LSF 1705469) and **~6.7x below** a
  zeroed/sentinel row (1.0 exactly). The 4.0x margin is the thinnest of the four — the same
  margin `_REPLAY_ROW_BAND` and `_ROUTING_IDENTITY_QK_ROW_BAND` already run on, not a new
  risk.

Integer metadata (`layer_num`) and shape stay **bit-exact and fatal** inside this same gate
— `_compare_replay_band` never bands them, so a mis-route that changes row counts still
fails as a SHAPE mismatch, no tolerance involved. The bit-exact comparison
(`RECORD {label} vs default (bit-exact)`) is still recorded as an INFO line so a run that
achieves 0 stays visible.

**What this gate now proves, and what it no longer proves.** It proves the three builders
agree to within run-to-run noise — the same standard every other value-based gate in this
file already holds vLLM's own kernels to. It no longer proves bit-identity: nothing at this
batch width *is* bit-identical across boots, including a builder against itself, so a
bit-exact gate here was never actually measuring the builders — it was measuring whichever
boot happened to land closest. Pinned by `tests/test_hs_builder_equivalence_band.py` (7
tests: accepts the measured 40-pair spread on both metrics, rejects an off-by-one row, a
zeroed row, a shape mismatch, and integer-metadata drift; asserts the margins above
numerically). HS's `routing-identity-w32` and both `w33-distinct` legs are **untouched** —
they stay bit-exact and fatal.

---

## `T1-batched capture_hs` / `capture_qk` — a measured LIMITATION (vLLM's nondeterminism, seen for the first time via an INDEPENDENT oracle)

**Band constants:** `MIA_T1_BATCHED_REL_BAND` = `2e-2` (tensor-global relative) and
`MIA_T1_BATCHED_ROW_BAND` = `1.5e-1` (per-row relative), `t2_invariants.py:704-711`,
gating `T1-batched capture_hs` and `T1-batched capture_qk`.

**Both `T1-batched` capture kinds are no longer bit-exact against the independent,
never-imports-MIA oracle at width 33.** LSF 1725139 failed both:

| | |
|---|---|
| Recorded | 2026-09-17 (task D7) |
| Diagnosed | LSF 1725139 (the failure) + LSF 1710884, all 15 pairwise boots (the re-derived floor, task D6) |
| Compared | `refb_capture_{hs,qk}` (vanilla vLLM, forward hooks, no MIA) vs `miab_capture_{hs,qk}` (MIA), width-33, 33 DISTINCT prompt lengths, prefill-wave `layer*.safetensors` |

| workload | worst relative (`max\|a-b\|/max\|a\|`) | note |
|---|---|---|
| `capture_qk` | **7.457e-04** | req0/layer031::q; absolute `max\|d\|`=7.812e-03 looks alarming only because q/k_full have a dominant outlier channel |
| `capture_hs` | **1.272e-03** | req0, growing with depth: 1.221e-04 → 3.125e-02 → 3.750e-01 ABSOLUTE at layers 1/16/32 — HS activations grow with depth, so the absolute figure alone overstates it; relative is what matters |

**Not a new regime.** Re-measured with `_compare_replay_band`'s own metric (reused, not
reimplemented) over **all 15** pairwise boots of LSF 1710884's `d4b/{A1,A2,A3,B1,B2,B3}` —
supersedes the 9-pair figure quoted for `qk_small routing-identity-w32` above and confirms it
as the true max, not a cherry-pick:

| | worst tensor-global relative | worst per-row relative |
|---|---|---|
| boot floor (LSF 1710884, all 15 pairs) | 1.446e-03 | 1.880e-03 |

Both T1-batched failures sit **at or below** that boot-to-boot floor — the same vLLM kernel
nondeterminism (split-K / atomics / per-boot autotuning) `qk_small routing-identity-w32`
already documents, now visible (a) in a comparison with **no MIA code on either side**, and
(b) in HS's batched capture, where it had not previously been measured. (This repository's
`3.5e-02` figure elsewhere is a **different, absolute** number — the same-arm QK boot spread
in LSF 1710884's own units — not this relative + per-row metric; it is not the floor being
compared against here.)

**Gate.** `layer*.safetensors` (the captured HS/QK payload) is banded with `_T1_BATCHED_REL_BAND =
2e-2` / `_T1_BATCHED_ROW_BAND = 1.5e-1` (`tests/mia/parity/t2_invariants.py`, via
`--compare-t1-batched`) — the **identical** magnitude as `_ROUTING_IDENTITY_QK_REL_BAND` /
`_ROUTING_IDENTITY_QK_ROW_BAND`, reused rather than loosened, because this is the same
category of comparison (matched/uncompiled-kernel capture vs an independent reader, gated on
vLLM's own boot-to-boot float noise):

* rel band 2e-2: **26.8x above** the worst measured QK value (7.457e-04), **15.7x above**
  the worst measured HS value (1.272e-03); **22.6x below** the off-by-one-row floor measured
  elsewhere in this suite (4.512e-01 tensor-global, LSF 1705469).
* row band 1.5e-1: same headroom shape as the QK T2 band above (see
  `tests/test_parity_t1_batched_band.py` for the pinned margins).

Integer metadata (`layer_num`, `k_prefix_ends`) and SHAPE stay **bit-exact and fatal** inside
this same comparison — `_compare_replay_band` never bands them, and the batched oracle's 33
DISTINCT prompt lengths mean a whole-request mis-route changes row COUNT, not just
magnitude, so it still fails as a SHAPE mismatch with no tolerance involved.

**What this gate still proves.** Multi-row batched ROUTING — 33 distinct-length requests in
ONE step, row assignment read from token ids alone by a reference that imports **nothing**
from `mia/runner.py` — agrees with that independent oracle to within run-to-run noise vLLM
itself already exhibits. That independence from `mia/runner.py` (the V1→V2 port surface) is
the entire reason this gate exists — it is the ONE comparison in the whole suite that is not
MIA-vs-MIA — and banding its VALUES does not remove that; it only concedes the bit-for-bit
requirement that a genuinely non-associative batched kernel replayed across two different
process boots can never promise.

**Width-1 `T1 capture_hs` / `T1 capture_qk` are UNCHANGED** — still bit-exact, still fatal,
and, being one request per pass with no batched reduction to reorder, immune to this effect
entirely: they remain the strongest evidence in this suite. Do not band those. Pinned by
`tests/test_parity_t1_batched_band.py` (9 tests: accepts the measured QK and HS spreads,
rejects an off-by-one row, a zeroed row, a shape mismatch, and integer-metadata drift;
asserts the band equals the QK T2 band exactly; asserts the margins above numerically;
asserts `run_parity.sh` bands the captured payload via `--compare-t1-batched` and never
touches the width-1 legs).

---

### `T1-batched capture_hs` / `capture_qk` GENERATION — the SAME cross-boot logprob noise, now on the sibling gate (task D10)

**Band constant:** `MIA_T1_BATCHED_LOGPROB_BAND` = `5e-2` (`t2_invariants.py:713-715`),
gating `T1-batched capture_hs generation` and `T1-batched capture_qk generation`. Token
ids stay bit-exact inside that gate; only the logprob channel is banded.

**The claim two paragraphs up — "`generation.safetensors` stays on the bit-exact path,
LSF 1725139 never showed it move" — did not hold in a later job.** LSF 1731353 failed
`T1-batched capture_hs generation` and `T1-batched capture_qk generation` identically:

| | |
|---|---|
| Recorded | 2026-09-17 (task D10) |
| Diagnosed | LSF 1731353 |
| Compared | `refb_capture_{hs,qk}/req0/generation.safetensors` (vanilla vLLM, no MIA) vs `miab_capture_{hs,qk}/req0/generation.safetensors` (MIA), width-33, 33 DISTINCT prompt lengths |

| channel | `capture_hs` max\|Δ\| | `capture_qk` max\|Δ\| |
|---|---|---|
| `cumulative_logprob` | **1.854e-02** | **1.854e-02** |
| `token_logprobs` | **1.269e-02** | **1.269e-02** |

`token_ids` and `prompt_token_ids` were **not** among the mismatches on either capture kind
— the two arms produced the identical sequence on all 33 requests; only the emitted
logprobs moved. **This is not the bifurcation `_bifurcation_report` detects** (that is a
token-id divergence) and **not a batched-routing defect** (agreeing token ids on 33/33
requests rules out a row mis-route) — it is the same cross-boot logprob noise measured
throughout this document (`_T0_GRAPH_LOGPROB_BAND`'s HS value, 1.815024e-02, is the same
order), now observed between the vanilla-vLLM T1 reference boot and the MIA boot instead of
between two MIA boots.

**Gate.** `--compare-t1-batched-generation` (`tests/mia/parity/t2_invariants.py`,
`_compare_generation_band`, reused via its existing `tag` parameter rather than
reimplemented) keeps `token_ids` and `prompt_token_ids` **bit-exact and fatal** — the
load-bearing claim this gate exists to make: MIA does not change what the model generates —
and bands only the two logprob channels at `_T1_BATCHED_LOGPROB_BAND = 5e-2`:

* Reuses `_T0_GRAPH_LOGPROB_BAND`'s **value** (5e-2) rather than inventing a new one, per
  instruction, kept as its own independently-retunable named constant because it gates a
  different comparison (reference-vs-MIA generation at width 33, not capture-off-vs-on at
  width 1).
* **2.7x above** the worst measured value here (1.854e-02) — consistent with the 2.8x
  margin `_T0_GRAPH_LOGPROB_BAND` already carries for the identical quantity — and **~15x
  below** the 7.632304e-01 scale of a real steering intervention.

`run_parity.sh`'s T1-batched section now runs ONE shared `--compare-t1-batched-generation`
call inside the `for kind in capture_hs capture_qk` loop; the old bare
`compare_artifacts.py --require-bit-exact --name generation.safetensors` call is gone.
Width-1 `T1 capture_hs` / `T1 capture_qk` remain untouched by this or any other flag — see
above. Pinned by `tests/test_parity_t1_batched_generation_band.py` (7 tests: accepts both
measured deltas, rejects an intervention-scale delta, keeps the token-id check fatal under
an arbitrarily wide band, asserts the band reuses `_T0_GRAPH_LOGPROB_BAND`'s value as its
own separate named constant, asserts the margins numerically, asserts `run_parity.sh` routes
both capture kinds through the one shared flag with no bare `--require-bit-exact` left in
that section).

---

## What the batched oracle cannot cover — an irreducible limit, not a tolerance

*Recorded 2026-09-17 (task D5 item 8). No tolerance is involved. This section exists
because the honest statement of what a suite does **not** cover is part of what its PASS
means.*

**The common mode.** Every T2 gate is MIA-vs-MIA. Both arms import `StepView` /
`step_view` from `mia/runner.py`, which **is** the V1→V2 port surface, so a bug there moves
both arms identically and every T2 gate still passes. Until task D5 the only oracle outside
MIA was T1 — vanilla vLLM with `VLLM_PLUGINS=""` and torch forward hooks — and it ran **one
request, eager**. So batching and graph replay, the two things the port actually changed,
were never checked against anything but MIA itself.

**What was added.** `t1_reference.reference_capture_batched` runs 33 distinct-length
requests in **one batch** and compares vanilla-vLLM forward hooks against MIA's capture
(`VERDICT T1 batched-oracle` in the job log). Generation (prompts, token ids, logprobs)
compares bit-exact; the captured HS/QK payload is BANDED as of task D7 — see
`T1-batched capture_hs` / `capture_qk` above for why (the same vLLM boot-to-boot kernel
noise `qk_small routing-identity-w32` already documents, now also visible through this
oracle). The reference decides which rows of a batched forward pass belong to which request
from the **token ids the model was fed** (`segment_batch_rows`), not from `query_start_loc`,
not from `idx_mapping`, and not from anything in `mia/runner.py` — which is what keeps it an
oracle rather than a second copy of the code under test. Multi-row routing is therefore
verified against something outside MIA for the first time.

**The limit: a Python forward hook cannot observe a CUDA-graph replay.** A replay is a
single graph launch on the device; no Python executes during it, so `register_forward_hook`
never fires. This is a property of CUDA graphs, not a gap in the implementation: **an
independent oracle for graph mode is not constructible this way** — not by a cleverer hook,
not by hooking a different module, not at all. The batched oracle is eager-only by nature.

**What graph-mode correctness therefore rests on, plainly.** MIA-vs-MIA plus the replay
band:

* `op-identity` and `routing-identity-w32` — the baked op against the forward-hook path with
  the kernels matched (`graph=True, cudagraph=False`), bit-exact;
* `routing-identity-w33-distinct` (QK: `[BANDED]`, task D9; HS: bit-exact) and
  `routing-identity-w33-distinct-replay [BANDED]` — 33 distinct prompt lengths, the second
  of them under **real** capture + replay and bifurcation-checked (see "Trajectory
  bifurcation under FULL graphs" above), where a whole-request mis-route is still a
  **shape** mismatch and fatal at any band width;
* `replay-band-w32` / `replay-band-w1` — forward hooks against a real FULL-cudagraph replay,
  on the relative + per-row bands recorded above.

Those are MIA-vs-MIA comparisons, and they can be fooled by a defect that moves both arms
the same way. The eager batched oracle narrows that exposure — a routing bug in
`mia/runner.py` that affects both eager and graph capture is now caught in eager, against
vanilla vLLM — but it does not eliminate it for defects that exist **only** on the
replay path.

---

## Real-replay capture (`replay-band-w1` / `-w32`) — the port's central graph-mode claim

**Date:** 2026-09-17 · **Bands:** `MIA_T2_REPLAY_REL_BAND` = `2e-2` (tensor-global relative)
and `MIA_T2_REPLAY_ROW_BAND` = `1.5e-1` (per-row relative), `t2_invariants.py:359-362` ·
**Gates:** `<workload> replay-band-w1` and `<workload> replay-band-w32`.

| | |
|---|---|
| Derived from | LSF 1705469, node `p5-r28-n2`, vLLM 0.29.0 / V2 runner |
| Last confirmed | LSF 1737784 and 1738651 (node `p5-r15-n3`), **identical values in both** |
| Model / dtype | `microsoft/Phi-3-mini-4k-instruct`, float16, greedy, `seed=0` |
| Compares | eager forward-hook capture vs capture under a **real FULL cudagraph replay** |

This is the one T2 leg that touches a real graph replay, and it is the port's central
graph-mode claim: what MIA captures from inside a baked, replayed graph is what the model
computed. It **cannot** be bit-exact — it inherits the compiled-vs-uncompiled confound
already recorded under "T0 under FULL CUDA graphs" — so it is a magnitude band. **Two**
metrics are applied, because either one alone has a hole.

**(1) Tensor-global relative**, `max|a−b| / max|a|` over the whole tensor:

| | HS | QK |
|---|---|---|
| observed, real replay (w1 / w32) | 6.152e-03 / 5.523e-03 | 2.121e-03 / 2.121e-03 |
| **band `2e-2`** | 3.3× / 3.6× above | 9.4× above |
| off-by-one row within the request, **floor** over tensors | 4.51e-01 (23×) | 4.51e-01 (23×) |
| a zeroed / sentinel row, **floor** over which row | **1.11e-02 — BELOW the band** | 2.70e-01 |

That HS figure is the hole, and it is the reason metric (2) exists: this metric normalises by
the tensor-global maximum, so zeroing a row whose own values are small relative to the
tensor's largest row produces a small number. An earlier revision of the comment beside these
constants quoted a "50× sentinel margin", which was the best case (zeroing the largest row),
not the floor.

**(2) Per-row relative**, `max` over rows `r` of `max|a[r]−b[r]| / max|a[r]|`. Normalising
inside the row makes a destroyed row cost the same whatever its magnitude:

| | HS | QK |
|---|---|---|
| observed, real replay (w1 / w32) | 2.769e-02 / 2.769e-02 | 2.715e-03 / 2.697e-03 |
| **band `1.5e-1`** | 5.4× above | 55× above |
| off-by-one row, **floor** | 5.96e-01 (4.0×) | 7.47e-01 (5.0×) |
| a zeroed / sentinel row | **1.0000 exactly, ANY row** (6.7×) | **1.0000 exactly** (6.7×) |

A zeroed row is now exactly `1.0` by construction rather than magnitude-dependent, so the
hazard this leg exists for — padding rows reading stale routing and scattering into live
aperture slots, or a re-allocated slab leaving the sentinel behind — is caught with a real
margin instead of an advertised one. Integer metadata (`layer_num`, `k_prefix_ends`) is held
**bit-exact** inside the same gate and the key sets must match, so an artifact present on one
side only is a failure rather than something the row-wise loop never visits.

**What neither metric excludes, measured rather than asserted.** A whole-request mis-route
between two requests whose captured tensors have the **same shape**. In this workload the
same-shape pairs are same-prompt replicas, and their mutual difference — which *is* the signal
such a mis-route would produce — measures only 2.23e-03 (HS) / 1.11e-03 (QK) tensor-global and
1.11e-02 / 1.35e-03 per-row (LSF 1705469). All four are **under** both bands, so no
value-based gate on this workload can see that swap. It is closed by construction rather than
left documented: the `*_batch_distinct` legs run 33 requests with 33 **distinct** prompt
lengths, so any whole-request mis-route changes the row count and lands as a **shape**
mismatch — fatal, with no tolerance involved.

---

## The fused steer kernel (`steer_small op-identity`) — a lever that is NOT byte-identical

**Date:** 2026-09-17 · **Band:** `MIA_T2_STEER_FUSED_BAND` = `5e-2`,
`t2_invariants.py:396` · **Gate:** `steer_small op-identity [BANDED]`.

| | |
|---|---|
| Measured | LSF 1738651, node `p5-r15-n3`, vLLM 0.29.0 / V2 runner |
| Model / dtype | `microsoft/Phi-3-mini-4k-instruct`, float16, greedy, `seed=0` |
| Compares | `MIA_STEER_FUSED=1` (the default Triton kernel) vs the eager forward-hook steer |
| Observed | **2.342296e-02** `max|Δ(logprob)|` — vs band `5.000e-02` |

**What it means that this lever is not byte-identical, stated plainly.** `PUBLIC_LEVERS`
previously advertised `MIA_STEER_FUSED` as byte-identical. It is not, and correcting that was
one of this branch's real findings. The fused Triton projection reduction computes the
steering projection in a different order from the reference path, and on fp16 that is not
associative — so the steered logprobs differ by ~2.3e-02 nats.

**What is NOT at fault, and how we know.** The same arm with `MIA_STEER_FUSED=0` reproduces
the eager steer **bit-exactly** — `op-identity-nofuse` is a FATAL bit-exact gate run at
`STEER_ATOL`, and it measured `0.000000e+00` in LSF 1738651 and every job before it. So MIA's
baked steer op, its V2 routing and the graph install are exact; **only the Triton reduction
differs.** The unsteered control between the same two arms is also exactly `0.000000e+00`,
which rules out ambient numerics: the delta appears only when the kernel actually steers.

**Band `5e-2`:** 2.1× above the measurement, and ~33× below the 7.632e-01..8.145e-01 that a
real `adjust_rs` steering intervention moves these logprobs (measured on every arm of seven
GPU jobs). So the gate still fails if the fused kernel ever starts steering *differently*
rather than merely *rounding* differently.

**Published for users** in `mia/optimizations.py::PUBLIC_LEVERS` and README "Known
Limitations". Users who need bit-reproducible steered logprobs set `MIA_STEER_FUSED=0`.

---

## The steer budget ceiling — the one thing stopping a gate from widening to match its own failure

**Date:** 2026-09-17 · **Band:** `MIA_T2_STEER_BUDGET_CEILING` = `1.5e-1`,
`t2_invariants.py:596` · **Gates it caps:** `steer_small steer-gap` and `steer_small width`.

| | |
|---|---|
| Measured over | 7 GPU jobs — LSF 1704450, 1705469, 1705794, 1710144, 1719790, 1720902, 1723978 |
| Last confirmed | LSF 1738651, node `p5-r15-n3`, vLLM 0.29.0 / V2 runner |
| Model / dtype | `microsoft/Phi-3-mini-4k-instruct`, float16, greedy, `seed=0` |

**Why a runtime-computed budget needs a ceiling at all.** Two steer gates size themselves
from a measured floor — `steer-gap` from the UNSTEERED eager-vs-FULL delta, `width` from the
UNSTEERED alone-vs-batch delta — because the confound each must tolerate (vLLM's
compiled-vs-eager numerics; the model's own batch-variance) is not a constant that can be
pinned. `max(floor, atol)` with **no upper bound is a gate that widens to match its own
failure**: if the floor blew up, so did the budget, and the gate passes whatever it is given.
That is the defect this constant closes, and it is the same defect class the T3 band's G1/G2
terms were exposed to in task E1's review.

**The two floors it caps:**

| floor | what it measures | observed |
|---|---|---|
| `steer-gap` | UNSTEERED eager-vs-FULL | **3.635375e-02 in all 7 jobs** (reproducible, not noise) |
| `width` control | UNSTEERED alone-vs-batch | 2.933863e-02 .. 5.865807e-02 |

**What a blown budget would let through:** a stale or mis-routed steer config applying — or
dropping — a whole intervention. An `adjust_rs` steer on this workload moves the logprobs
**7.632304e-01 .. 8.145643e-01**, measured on every arm of every one of those jobs, so that
scale is not an estimate.

**Ceiling `1.5e-1`:**

* **2.6× above** the worst floor ever measured (5.866e-02) — no observed boot trips it;
* **8.4×** above the worst STEERED width delta (1.779e-02) and **4.1×** above the worst
  STEERED gap delta (1.801e-02) — neither gate sits near its own cap. LSF 1738651 measured
  `steer-gap` at 1.801313e-02 against a budget of 3.635375e-02, and `width` at 1.656437e-02
  against 3.050631e-02;
* **5.1× below** the smallest steering intervention ever measured (7.632e-01) — a budget
  pinned *at* the ceiling still fails a real mis-steer by a factor of five.

**A floor that exceeds the ceiling is not a licence to widen — it is a FAILURE**, and both
gates report it as one.

---

## T3 cross-version (`a21` vs `a29`) — a DERIVED band, not a constant

**Date:** 2026-09-17 · **Task:** E1 · **Gate:** `<workload> G5 T3 capture a21-vs-a29 [BANDED
from G4]`, `<workload> G5 T3 generation a21-vs-a29 [BANDED]` and (steer only) `steer_small
G5 T3 control a21-vs-a29 [BANDED]` in `tests/mia/parity/t3_crossbranch.py` · **Job:** LSF
1749270, node `p3-r01-n1`, 2026-09-17, repo `4f9c2f0`, which records the derived value for
every workload before it enforces it.

| | |
|---|---|
| old world | MIA @ `9cfbef60f72a0e8429a56d19ed60c965705193f3` on **vLLM 0.21.0 / V1 runner**, torch 2.11.0+cu130 |
| new world | MIA @ this branch on **vLLM 0.29.0 / V2 runner**, torch 2.13.0+cu130 |
| model | `microsoft/Phi-3-mini-4k-instruct`, `dtype=float16`, greedy, `seed=0`, `enable_prefix_caching=False` |
| regime | eager (`enforce_eager=True`), `max_num_seqs=1`, 3 prompts, 16 tokens, `ignore_eos` |
| workloads | `hs_small` (layers 1/16/32), `qk_small` (layers 0/15/31), `steer_small` (layer 15) |

### MEASURED — LSF 1749270, node `p3-r01-n1`, 2026-09-17

The band is re-derived every job, so these are not the bound: they are the observation the
bound came from, and the thing a later run has to be compared against. A control that grows
from `0` to `4.9e-02` would more than double the gate's width with **no** anomaly line,
because it stays under the ceiling — only a recorded prior value makes that visible.

Every number below is `max |Δ|`, tensor-global relative unless the row says otherwise.

| leg | what it compares | `hs_small` | `qk_small` | `steer_small` |
|---|---|---|---|---|
| **G1** rel / per-row | `a21` vs `c21` — MIA inert on 0.21 | **0** / **0** | **0** / **0** | **0** / **0** |
| **G1** logprob (gen / control) | " | **0** | **0** | **0** / **0** |
| **G2** rel / per-row | `a29` vs `c29` — MIA inert on 0.29 | **0** / **0** | **0** / **0** | **0** / **0** |
| **G2** logprob (gen / control) | " | **0** | **0** | **0** / **0** |
| **G4 CONTROL** rel / per-row | `c21` vs `c29` — pure vLLM drift | **0** / **0** | **0** / **0** | 5.293000e-07 / **0** |
| **G4 CONTROL** logprob | " (steered `generation`) | 1.165667e-06 | 1.165667e-06 | 6.221235e-07 |
| **G4 CONTROL** control logprob | " (unsteered `control`) | — | — | 1.165667e-06 |
| **G3** rel / per-row / logprob | `ref` vs `a21r` — reference reproducibility | **0** / **0** / **0** | **0** / **0** / **0** | **0** / **0** / **0** |
| **WIDTH CONTROL** rel / per-row | `a21` vs `a21r` — width 1 vs default | 1.876000e-03 / 2.139000e-03 | 2.805000e-03 / 3.211000e-03 | 1.090000e-02 / **0** |
| **WIDTH CONTROL** logprob | " | 2.181029e-02 | 2.181029e-02 | 7.136625e-02 |
| **G5b** rel / logprob | `ref` vs `a29` (cross-job **and** cross-width) | 1.876000e-03 / 2.181005e-02 | 2.805000e-03 / 2.181005e-02 | 1.078000e-02 / 7.136563e-02 |

**Bands this produced, and what they then measured:**

| band | `hs_small` | `qk_small` | `steer_small` |
|---|---|---|---|
| relative `= 2 × (G1+G4+G2)` | **0.000000e+00** | **0.000000e+00** | 1.058600e-06 *(unused — steer has no layer tensors)* |
| per-row | **0.000000e+00** | **0.000000e+00** | 0.000000e+00 *(unused)* |
| logprob | 2.331333e-06 | 2.331333e-06 | 1.244247e-06 |
| control logprob | — | — | 2.331333e-06 |
| **G5 capture** observed | **0.000000e+00** ✓ | **0.000000e+00** ✓ | — |
| **G5 generation** observed | 1.165667e-06 ✓ | 1.165667e-06 ✓ | 6.221235e-07 ✓ |
| **G5 control** observed | — | — | 1.165667e-06 ✓ |

No `CONTROL ANOMALY` fired; no band was capped.

**The headline, stated plainly.** `G4 = 0` on the captured payload means vanilla vLLM 0.21
and vanilla vLLM 0.29 produced **bit-identical** hidden states and Q/K on this workload — two
vLLM versions, two torch versions, two attention backends. So the band the job actually
enforced on `G5 T3 capture` was `0.000000e+00`, i.e. **bit-exactness**, and it passed. With
`G1 = G2 = 0` as well, the transitive claim is not a tolerance at all: **MIA on 0.29/V2
captures the same bytes MIA on 0.21/V1 captured.** The only residue anywhere is
`1.165667e-06` on the logprob channel, which is vLLM's own (it is present with MIA absent
from both sides) and 1.9 × 10⁴ times smaller than a real steering intervention (7.632e-01).

**The WIDTH CONTROL row is why `a21` and `a21r` both exist.** `a21` vs `a21r` differ only in
running-batch width, in the same job on the same node in the same boot, and they differ by
`1.876e-03`–`1.090e-02` relative and up to `7.137e-02` on the logprobs — **three to four
orders of magnitude above the cross-version drift the FATAL gates are measuring.** Comparing
the width-default reference against a width-1 arm would have buried T3's actual signal under
batch variance. That is also why `G5b` (`ref` vs `a29`, the only width-crossing leg) tracks
the WIDTH CONTROL almost exactly rather than the near-zero `G5`.

### The band is not a number written here — it is a measurement taken in the same job

Every other entry in this file records a constant and the measurement that justifies it. T3
records a **rule**, because the thing it must tolerate — two vLLM versions' kernels — is
measurable directly, on the same node, in the same job, with MIA removed from both sides:

```
|a21 - a29|  <=  |a21 - c21|  +  |c21 - c29|  +  |c29 - a29|      (triangle inequality)
                    G1              G4              G2

band = _MARGIN * (G1 + G4 + G2),  capped at CEILING,  _MARGIN = 2.0
```

* `c21` / `c29` are **vanilla vLLM** + naive `register_forward_hook`, with MIA never
  imported (`t1_reference.assert_no_plugin()` proves it structurally, not by grepping a
  log). `G4` is therefore **pure vLLM 0.21 -> 0.29 drift**: attention backend, kernel
  selection, torch version.
* `G1` and `G2` are the inertness claims, gated **bit-exact and FATAL**. When they hold —
  the expected case, since T1 already passes bit-exact in this exact regime on 0.29 — the
  band collapses to `2 x G4`, and T3's result becomes a statement rather than a tolerance:
  *MIA contributes exactly zero on both vLLM versions, and every difference between the old
  world and the new one is vLLM's own.*
* If `G1` or `G2` FAILS, that is the finding. It is printed under a `HEADLINE T3` line, it
  is never absorbed into the band, and it invalidates the derivation above (which assumes
  inertness) — read it before reading the T3 verdict.

### The ceilings, and why a derived band still needs them

| bound | value | what it must still be able to catch |
|---|---|---|
| `_REL_CEILING` | `1.0e-1` | a whole-request off-by-one row: **4.512e-01** tensor-global (LSF 1705469) |
| `_ROW_CEILING` | `5.0e-1` | the same shift per-row: ~**1.0** |
| `_LOGPROB_CEILING` | `1.0e-1` | a real steering intervention on a logprob: **7.632e-01** |

A band wide enough to pass a mis-route catches nothing, so the derivation is capped. When a
control measures above `CEILING / MARGIN` the cap applies, a `CONTROL ANOMALY` line names it,
and G5 is allowed to fail — **a red T3 with a named cause beats a green one whose band was
widened to fit.** The ceilings are source constants with **no environment override**;
`t2_invariants._band()` is not used here because there is nothing external to defend against
— the band is computed in-process from measurements taken in the same job.

The measurement is read from the comparator's own `%.3e` rendering (four significant
figures), so it can understate the truth by at most 5e-04 relative. That is four orders of
magnitude below `MARGIN`, and `tests/test_parity_t3_crossbranch.py::
test_measure_agrees_with_the_gate_it_derives_from` pins the measurement and the gate to each
other so the two cannot drift apart.

### Discrimination — the part that makes the band mean something

`tests/test_parity_t3_crossbranch.py` proves the band can FAIL, which is the property this
suite has learned to demand of every bound it enforces:

* accepts a control-sized delta (1e-3) at a band derived from that same control;
* rejects a mis-route-sized delta (4.512e-01) at that band **and** at the capped ceiling;
* end-to-end through `judge`: a whole step scaled by `1 - 4.512e-01` in the `a29` arm fails
  **both** `G2` (MIA no longer inert on 0.29) and `G5`'s banded capture gate.

### Token ids come first on every cross-version leg

Across two vLLM versions a greedy argmax can legitimately flip at a near-tie, and every row
after such a flip holds the activations of a **different token** — not comparable under any
band. `_bifurcation_report` runs before any tensor comparison; only pre-divergence rows are
compared and the skipped ones are counted in the log. A bifurcation is not a blanket excuse:
the leg still fails if the pre-divergence rows disagree, or if the divergence carries no
explanatory logprob delta (greedy sampling from bit-identical logits cannot produce two
different argmaxes). For the generation channel the truncation re-derives
`cumulative_logprob` as the sum of the surviving `token_logprobs` on **both** sides, so that
channel stays a real comparison rather than a dropped key (vacuous) or a kept one (two
different sequences).

### `steer_small`'s unsteered `control` channel — banded for the same reason

`steer_small` writes no layer tensors, so its `*.safetensors` payload is `generation` +
`control`. Until task E1's review, the only value-banded cross-version gate walked
`generation.safetensors` only: `control.safetensors` was bit-exact **within** a version
(G1/G2) and structurally checked **across** versions, but a drift confined to the unsteered
control would have failed nothing — while the `*.safetensors` payload made it look covered.
It does drift: all six control files differ between 0.21 and 0.29. `G5 T3 control
a21-vs-a29 [BANDED]` now bounds it on its own three legs (G1/G4/G2 measured on that channel,
recorded in the table above: band `2.331333e-06`, observed `1.165667e-06`).

### What T3 does NOT cover

* **Graph mode.** Every T3 arm is eager. The vanilla-vLLM control arms could not be anything
  else — a Python forward hook cannot observe a CUDA-graph replay — and comparing MIA's
  graph path across two vLLM versions without a control would mix the port, two kernel
  generations and two capture mechanisms in one number. Graph-mode correctness rests on T2.
* **Width.** The A6 reference tree was minted with `MIA_PARITY_MAX_NUM_SEQS` unset (all three
  requests resident at once) while the vanilla oracle can only run at width 1, and **the
  model is not batch-invariant** — T2's own `CONTROL eager alone-vs-batch` fails
  bit-exactness on every workload in every job that has run it. Every FATAL T3 gate is
  therefore matched-width at width 1; the `a21r` arm reproduces the reference's width so the
  reference check is matched-width too; and the `WIDTH CONTROL a21-vs-a21r [INFO]` line
  measures what the width is worth on this workload (measured above: up to `1.090e-02`
  relative and `7.137e-02` on the logprobs — orders of magnitude above what the FATAL gates
  are measuring). **Exactly one leg crosses widths: `G5b ref-vs-a29`**, and it is INFO for
  that reason. `G3 ref-vs-a21r` is matched-width on both sides; what it crosses is a job, a
  node and a boot, which is a different limitation and the reason it is INFO too.
* **Cross-boot reproducibility at 0.21.** No measurement in this suite established it, so
  `G3 ref-vs-a21r` measures it and reports it rather than gating on it.

---

## TP parity probe (`tp_parity_probe.py`) — HS, QK and steer MEASURED (G1 1777562, its re-run 1780395, and the noise legs 1781918)

**Gate:** G1 of the model-scale study (`--compare <tp1> [<tp1_repeat>] <tpN>`) · **Model:**
`meta-llama/Llama-3.1-8B`, **bf16**, greedy, `ignore_eos`, prefix caching off,
`cudagraph_mode=FULL`, vLLM 0.29.0 + the V2 runner · **Measured:** LSF **1777562**, node
**p3-r06-n4** (4 × H100 80GB HBM3, `exclusive_process`), 2026-09-19, MIA pin `e32aa71`. Legs:
`tp1`, `tp1_repeat`, `tp2`, `tp4` (fusion "default"), `tp2_nofuse`, `tp4_nofuse`. Then LSF
**1780395**, node **p4-r19-n3** (GPUs 0-3, 4 × H100 80GB HBM3, `exclusive_process`), 2026-09-19,
MIA pin `b44da1e`: the same legs with `tp2`/`tp4` FUSED (`--fuse on`), QK measured for the first
time. Then LSF **1781918**, node **p6-r09-n1** (GPUs 1,5,6,7), 2026-09-19, MIA pin `6ae479c`
(plugin byte-identical to `b44da1e`): the same legs plus the NOISE legs (`tp1_repeat2`,
`tp2_repeat`), every leg judged against two TP=1 boots. The bands below
were first DERIVED (2026-09-18, values kept in the table for the record), re-derived from 1777562,
and checked against 1780395 and 1781918, which moved none of them. A band is never widened to make a run pass:
an overshoot is a finding, or a new measurement recorded in this section with its job.

**What the probe compares.** A TP=N run against a TP=1 run of the same prompts (or, with
`--repeat`, against a second boot of the same TP=N configuration): HS on every
layer and token (1-based layers 1..L), QK on every layer (0..L-1) merged across ranks by
`mia.graph.aperture_reader.merge_qk_aperture_ranks`, and the STEERING EFFECT
`δ = logprob_steered − logprob_unsteered` on teacher-forced prompt tokens. Everything
structural is exact and FATAL with no band involved: the captured layer set (all layers on
both sides), the captured widths (a rank-0-only QK capture is `H_q/tp × head_dim` wide and
fails on shape), the rank dirs (QK exactly `tp_rank_0..N-1`; HS, since the TP layer shard
(`MIA_HS_TP_SHARD`, default 1 at TP>1), every rank that owns a layer, each header declaring the
round-robin layers, the capture read as the UNION of the rank dirs by
`mia.graph.aperture_reader.load_hs_aperture_tp`; `tp_rank_0` alone at TP=1 or in a
`MIA_HS_TP_SHARD=0` leg -- both from `flush_aperture` AND from the on-disk listing), the row
structure (`n_prompt + n_gen − 1` rows, `k_prefix_ends`), prompt token ids. Rows are compared over
the common input prefix only (the greedy trajectories may flip at a near-tie; rows after the flip
describe different inputs).

**The replication leg (`hs_replicas`, TP>1) has NO band.** It captures every layer on every rank
(`MIA_HS_CAPTURE_ALL_RANKS=1`) and requires each rank's copy to be BITWISE equal to rank 0's, which
is what makes the layer shard legitimate: the HS measurements above were rank 0's copy of every
layer; under the shard layer `i` is rank `i % tp`'s copy, and those are the same bits exactly when
this leg passes. Rank 0's copy is then judged against TP=1 with the HS bands, unchanged.

**Why a band at all.** TP=N is a different numerical path, not a perturbed copy of TP=1:
every `o_proj` / `down_proj` output is reduced in per-rank K-slices, rounded to bf16, and
all-reduced (NCCL/custom all-reduce, or vLLM's fused all-reduce + RMSNorm under
`fuse_allreduce_rms`) — `2L` extra bf16 roundings of relative size ~`2^-9` fed into the
residual stream, plus whatever GEMM algorithm cuBLAS picks for the narrower per-rank shapes.

### MEASURED — LSF 1777562 (every leg UNFUSED)

`fuse_allreduce_rms` resolved **False on every leg**, the "default" ones included: vLLM 0.29's
DEFAULT fuses at TP>1 only when `has_flashinfer()`, and `mia_v029` had no `flashinfer-cubin` and no
`nvcc` on PATH. So every number below is the UNFUSED path. (An EXPLICIT `fuse_allreduce_rms=True`
does not need `has_flashinfer()`, only `flashinfer.comm` and its workspace: that is the probe's
`--fuse on`, first run in 1780395 below.)

**Noise floor.** `tp1_repeat` vs `tp1` is **bit-identical** on all three workloads (HS, QK, steer:
every relative error `0.0`; HS `capture.pt` and steer `generation.pt` md5-identical across the two
boots). All TP=N error below therefore comes from the TP numerical path.

**HS** (80 rows compared per leg; greedy tokens agree 8/8 on every request):

| leg | worst tensor-global rel (at) | worst row rel | median over layers |
|---|---|---|---|
| `tp2` | `1.4670e-02` (req1, **layer 32**) | `3.3961e-02` | `1.43e-03` |
| `tp2_nofuse` | `1.4670e-02` (req1, **layer 32**) — md5-identical capture to `tp2` | `3.3961e-02` | `1.43e-03` |
| `tp4` | `1.5670e-02` (req1, **layer 32**) | `2.9905e-02` | `1.53e-03` |
| `tp4_nofuse` | `1.6677e-02` (req1, **layer 32**) | `4.8124e-02` | `1.82e-03` |

The worst layer is **always the last**. The error grows through the residual stream: at `tp2`,
per-layer worst `2.78e-04` (L2), `7.30e-04` (L8), `1.25e-03` (L16), `2.64e-03` (L24), `6.68e-03`
(L31), `1.467e-02` (L32). Layer 32's own update is the largest in the model
(`‖h32 − h31‖ / ‖h32‖` = 1.6–1.8), which amplifies what 31 layers accumulated. The random-walk
estimate (`sqrt(64) × 2^-9 ≈ 1.6e-02`) predicted the worst layer. `tp4` and `tp4_nofuse` resolved
the same (unfused) config yet differ (`1.567e-02` vs `1.668e-02`), unlike the md5-identical `tp2`
pair: the TP4 path is not boot-to-boot deterministic, or the two requested configs compile
differently. No TP4 repeat exists to separate the two.

**Steer:**

| leg | `‖δ_N − δ_1‖ / ‖δ_1‖` | cand `max|δ|` (nats) | unsteered TP=N vs TP=1 `max|Δlogprob|` (nats) |
|---|---|---|---|
| `tp2` | `4.1706e-03` | `15.6804` | `0.1910` |
| `tp2_nofuse` | `3.2697e-03` | `15.6863` | `0.1010` |
| `tp4` | `3.8221e-03` | `15.7293` | `0.1046` |
| `tp4_nofuse` | `4.2957e-03` | `15.7078` | `0.1412` |

The ref (`tp1`) `max|δ|` is `15.6778`. 8 of 8 generated tokens differ steered vs unsteered on
every request. Unlike HS, the steer channel is **not** boot-to-boot deterministic at TP>1: `tp2` and
`tp2_nofuse` ran the same unfused configuration, and their HS captures are md5-identical, yet their
steer `generation.pt` differ (`δ` rel `4.17e-03` vs `3.27e-03`, unsteered `0.191` vs `0.101` nats).

**QK:** no TP>1 measurement. On every TP>1 leg the QK child crashed: the aperture drain consumer
thread on every rank >= 1 never selected its CUDA device and died with `cudaErrorDevicesUnavailable`
under `exclusive_process` (fixed in `mia/graph/thread_device.py`). The TP=1 repeat is
bit-identical.

### MEASURED — LSF 1780395 (the G1 re-run: `tp2`/`tp4` fused, QK for the first time)

Legs: `tp1`, `tp2` and `tp4` with `--fuse on`, `tp2_nofuse` and `tp4_nofuse` with `--fuse off`,
`tp1_repeat`. The explicit toggle resolved `fuse_allreduce_rms=True` on every fused child and False
on every nofuse one (each child's config dump). The gate FAILED on the probe's fusion check
alone: it scanned `gc.get_objects()` for the pass after vLLM 0.29 had frozen the worker heap
(`freeze_gc_heap()`), so every rank of every TP>1 leg, fused or not, reported no pass (fixed: see
"Fusion: ARMED vs APPLIED" below). Every capture and steer check, every structural check and every
writer / rank-dir check PASSED. The numbers below were re-computed GPU-free from that job's run
dirs with the current `--compare`, against BOTH TP=1 boots.

**The TP=1 HS noise floor is NOT zero.** This job's first `tp1` HS capture differs from `tp1_repeat`
in 118 of 128 (request, layer) tensors (max `|diff|` 0.3125): worst layer **`1.3184e-02`** (req1,
layer 32), worst row `3.3025e-02`. `tp1_repeat`'s HS capture is bit-identical to BOTH 1777562 TP=1
runs, so the first `tp1` boot is the outlier. QK and steer at TP=1 stayed bit-identical (the
`--compare` sha256 of both artifacts match). So TP=1 HS is not always boot-deterministic, and its
floor is about the size of the TP=N error: the worst TP=N layer is only 1.1–1.2× it.

**HS** (worst tensor-global rel, always req1 **layer 32**; worst row; rows compared):

| leg | vs `tp1` | row | vs `tp1_repeat` | row | rows |
|---|---|---|---|---|---|
| `tp2` (fused) | `1.6250e-02` | `3.7706e-02` | `1.6110e-02` | `3.3091e-02` | 80 |
| `tp2_nofuse` | `1.4475e-02` | `3.0562e-02` | `1.4122e-02` | `2.6848e-02` | 80 |
| `tp4` (fused) | `1.6086e-02` | `3.1007e-02` | `1.5802e-02` | `2.9820e-02` | 80 |
| `tp4_nofuse` | `1.6370e-02` | `3.8764e-02` | `1.5664e-02` | `3.1574e-02` | 77 |
| `tp1_repeat` (TP=1 floor) | `1.3184e-02` | `3.3025e-02` | — | — | 80 |

**QK**, merged across ranks at full width (q `32 × 128` = 4096, k_full `8 × 128` = 1024, layers
0..31). Worst always req1 **layer 30 `k_full`**. The same against both references (their QK is
bit-identical):

| leg | worst q rel | worst k_full rel | worst single head |
|---|---|---|---|
| `tp2` (fused) | `9.9503e-03` | `1.0112e-02` | `1.3609e-02` |
| `tp2_nofuse` | `8.8793e-03` | `8.7859e-03` | `1.0940e-02` |
| `tp4` (fused) | `1.0472e-02` | `1.0678e-02` | `1.3000e-02` |
| `tp4_nofuse` | `9.6611e-03` | `9.6269e-03` | `1.2241e-02` |
| `tp1_repeat` | `0` | `0` | `0` |

**Steer** (ref `max|δ|` `15.6778` nats; the same against both references):

| leg | `‖δ_N − δ_1‖ / ‖δ_1‖` | cand `max|δ|` (nats) | unsteered `max|Δlogprob|` (nats) |
|---|---|---|---|
| `tp2` (fused) | `4.2810e-03` | `15.7236` | `0.1018` |
| `tp2_nofuse` | `3.3768e-03` | `15.6885` | `0.1010` |
| `tp4` (fused) | `3.5012e-03` | `15.6687` | `0.1235` |
| `tp4_nofuse` | `3.5603e-03` | `15.7035` | `0.1046` |
| `tp1_repeat` | `0` | `15.6778` | `0` |

**Fused vs unfused.** Every fused number sits in the unfused range; fusion moved nothing near a
band. Evidence that the fusion actually engaged, independent of the probe: at TP=2 the fused
Inductor artifacts are 17–31 % smaller on both ranks (hs 4.52 vs 5.44 MB, qk 3.46 vs 5.01, steer
3.62 vs 4.94; same-config jitter ~48 KB), and the fused-vs-unfused HS median layer rel is 1.72e-3
(TP2) / 2.06e-3 (TP4) against 0.98e-3 / 0.89e-3 unfused boot to boot. At TP=4 the fusion applies
only to steps of at most 256 tokens: FlashInfer's fusion limit on H100 is 2 MB at TP4 (0.5 MB, 64
tokens, at TP8), so vLLM compiled two ranges, (1, 256) fused and (257, 8192) not.

### Noise legs, and which TP=1 reference

Because a TP=1 boot can be the outlier, "against TP=1" depends on which boot. From the next G1 run
on:
* **Every leg is judged against `tp1` AND `tp1_repeat`** (`--compare tp1 tp1_repeat <leg>`), and FAILS
  if it fails against either. The verdict records each reference (`refs`: label, dir, sha256 of each
  workload's artifact), whether they are bit-identical (`refs_identical`), and the numbers against
  each (`by_ref`). Report the worse of the two.
* **`tp1_repeat2`**, a third TP=1 boot (`--compare tp1 tp1_repeat tp1_repeat2`), gives three TP=1
  pairs. The **TP=1 noise floor** of a channel is the worst of the three, and an outlier boot shows
  as the one that differs from the other two.
* **`tp2_repeat`**, a second boot of `tp2`'s configuration, is judged for parity like any leg and also
  against `tp2` itself (`--repeat --compare tp2 tp2_repeat`). That is the **TP>1 boot-to-boot noise**
  of every channel. Steer is known not to be deterministic there (1777562: the unfused `tp2` pair
  differed by 1.28× on `δ` rel), and the TP4 HS pair differed too (1777562: `1.567e-02` vs
  `1.668e-02` under the same resolved config).

The bands do not move on these legs. If the measured floor of a channel comes within 2× of its
band, that is a finding for this section, not a reason to widen.

### MEASURED — LSF 1781918 (the noise legs: TP=1 and TP=2 boot-to-boot noise)

Legs `tp1`, `tp2` / `tp4` (`--fuse on`), `tp2_nofuse` / `tp4_nofuse` (`--fuse off`), `tp1_repeat`,
`tp1_repeat2`, `tp2_repeat` (`--fuse on`, a second boot of `tp2`). Every child compiled fresh into
its own leg's `VLLM_CACHE_ROOT` (no rank `loaded_from_disk`), so every repeat is a real boot, not a
cached-kernel replay. G1 PASSED: every verdict PASS against both references, no band overridden.
**These are the numbers any band tightening has to rest on; no band moves in this entry.**

**Boot-to-boot noise, per channel** (worst tensor-global rel; the worst of the pairs available):

| channel | TP=1 (three boots: `tp1`, `tp1_repeat`, `tp1_repeat2` = three pairs) | TP=2 (`tp2` vs `tp2_repeat`, both `--fuse on`) |
|---|---|---|
| **HS** | **`1.3193e-02`** worst layer (req1 layer 32), `3.2891e-02` worst row — `tp1_repeat2` against both `tp1` and `tp1_repeat`; `tp1` vs `tp1_repeat` `0` | **`1.4725e-02`** worst layer (req1 layer 32), `3.0576e-02` worst row |
| **QK** (q, k_full, head) | **`0` exactly** on all three pairs | **`0` exactly** (the two captures are md5-identical) |
| **steer** `δ` rel | `0` (all three bit-identical) | `1.4873e-03` (`max|δ|` `15.6991` vs `15.7008` nats); unsteered logprobs `0` |

**The TP=1 HS "noise" is two discrete states, not a spread.** Across the seven TP=1 boots of the
three G1 jobs, the HS `capture.pt` takes exactly two values. One (md5 `244aaacc7fca`) comes from
five boots: both 1777562 boots, 1780395's `tp1_repeat`, and 1781918's `tp1` and `tp1_repeat`. The
other (md5 `668957eeb5dc`) comes from two boots in two different jobs, on two different nodes:
1780395's `tp1` and 1781918's `tp1_repeat2`. They are bit-identical to each other. The distance
between the two states is the floor:
`‖B − A‖ / ‖A‖ = 1.3193e-02` (1781918) and `‖A − B‖ / ‖B‖ = 1.3184e-02` (1780395). QK and steer
stay in one state on all seven boots. TP=2 fused HS shows the same pattern across jobs: 1780395's
`tp2` and 1781918's `tp2_repeat` are md5-identical (`1ba4efb32a97`), and 1781918's `tp2` is the
other state. `tp2_nofuse` is md5-identical across 1780395 and 1781918 (`750f275f31d9`). So a
"boot" chooses between a small number of numerical paths for the HS child, and the choice
persists through the whole capture (it starts in one row at layers 2-7 and grows to layer 32; the
generated tokens stay identical). The root cause is NOT established. A candidate to test is a
timing-based kernel choice made while compiling, e.g. vLLM 0.29's Inductor config
`benchmark_combo_kernel: True`, which a fresh compile makes again at every boot.

**TP=N parity** (per leg, the worse of `by_ref.tp1` and `by_ref.tp1_repeat`; the two references
are bit-identical here, so the two agree):

| leg | HS worst layer (req1 L32) | HS worst row | QK q / k_full / head (req1 L30 `k_full`) | steer `δ` rel | unsteered `max|Δlogprob|` |
|---|---|---|---|---|---|
| `tp2` (fused) | `1.6268e-02` | `3.1020e-02` | `9.9503e-03` / `1.0112e-02` / `1.3609e-02` | `4.8980e-03` | `0.1419` |
| `tp4` (fused) | `1.5995e-02` | `5.8376e-02` | `1.0100e-02` / `1.0199e-02` / `1.2479e-02` | `3.2939e-03` | `0.1235` |
| `tp2_nofuse` | `1.4122e-02` | `2.6848e-02` | `8.8124e-03` / `8.6735e-03` / `1.0916e-02` | `3.2697e-03` | `0.1010` |
| `tp4_nofuse` | `1.6501e-02` | `4.7234e-02` | `9.6618e-03` / `9.7950e-03` / `1.2519e-02` | `4.2957e-03` | `0.1412` |
| `tp2_repeat` (fused) | `1.6110e-02` | `3.3091e-02` | `9.9503e-03` / `1.0112e-02` / `1.3609e-02` | `5.1193e-03` | `0.1419` |

Steer liveness `max|δ|` `15.67`–`15.71` nats on every leg (floor `0.5`).

**What this means for the bands (a record, not a change):**
* **HS cannot be tightened below boot noise.** The HS TP error (`1.41e-02`–`1.65e-02`) is not
  separable from HS boot noise at layer 32: TP=1 `1.319e-02`, TP=2 `1.472e-02`. `TP_HS_REL_BAND`
  (`5e-2`) is 3.4× the TP=2 boot noise and 3.8× the TP=1 floor, and it stays where it is.
* **QK boot noise is exactly zero at TP=1 (three boots) and at TP=2 (two boots)**, while the QK TP
  error is `8.67e-03`–`1.02e-02` (`k_full`) / `1.09e-02`–`1.36e-02` (head). So what separates
  TP=N from TP=1 on QK is the TP numerical path alone, and `TP_QK_REL_BAND` (`1e-1`, 9.8× the worst
  measured `1.0199e-02`) and `TP_QK_HEAD_REL_BAND` (`3e-1`, 22× `1.3609e-02`) have real headroom
  to tighten. **This entry does not tighten them.** A tightening is its own change, cites these
  numbers and 1780395's, and must still clear the TP4 value and whatever TP8 measures (QK TP error
  grew TP2 → TP4 by ×1.01–1.13 on `k_full` here).
* **Steer:** TP=2 boot-to-boot `δ` rel `1.49e-03` is 30 % of the worst TP=N value here
  (`5.12e-03`, `tp2_repeat`), inside `TP_STEER_DELTA_REL_BAND` (`5e-2`, 9.8× that worst value).
  Unchanged.

### The two regimes the bands separate

| | relative size | source |
|---|---|---|
| zero-filled layer (the PP / missing-shard failure) | `1.0` exactly | arithmetic |
| head or rank permutation in the merge | ~`1.41` | two independent heads of similar norm |
| whole-request one-row shift | `4.512e-01` tensor-global, ~`1.0` per-row | LSF 1705469 (see “T3”) |
| **a rank captures its PRE-all-reduce partial sum** (misses the other ranks' share of the layer's last row-parallel output) | up to the layer's own update `‖h_L − h_{L−1}‖ / ‖h_L‖`: `≈1.0` at L2, **1.6–1.8 at L32**, `1.0e-02` (L3) – `2.1e-01` (L31) in between; ≈`1/√2` of it at TP=2 with independent partials | the G1 `tp1` HS capture, max over its 4 requests |
| steer on one rank only / applied per rank and summed | `O(1)` of the effect (`1.0` for doubling) | arithmetic |
| **measured TP noise, HS** | `1.467e-02`–`1.668e-02` worst layer (always the last), `1.4e-03`–`1.8e-03` median layer; `2.99e-02`–`4.81e-02` worst row | LSF 1777562 |
| **measured TP noise, HS** (fused and unfused) | `1.448e-02`–`1.637e-02` worst layer vs `tp1`, `1.412e-02`–`1.611e-02` vs `tp1_repeat`; `2.68e-02`–`3.88e-02` worst row | LSF 1780395 |
| **measured TP=1 boot noise, HS** | `1.318e-02` worst layer, `3.30e-02` worst row (one outlier among the four TP=1 boots of the two jobs) | LSF 1780395 |
| **measured TP=1 boot noise, HS** (three boots) | `1.3193e-02` worst layer, `3.29e-02` worst row: two discrete states, 5 of 7 boots across 1777562 / 1780395 / 1781918 in one, 2 in the other (bit-identical to each other) | LSF 1781918 |
| **measured TP=2 boot-to-boot noise** (`tp2` vs `tp2_repeat`, fused) | HS `1.4725e-02` worst layer, `3.06e-02` worst row; QK `0`; steer `δ` rel `1.487e-03` | LSF 1781918 |
| **measured TP=1 boot noise, QK** | `0` exactly on three boots (three pairs); and `0` at TP=2 | LSF 1781918 |
| **measured TP noise, HS / QK / steer** (fused and unfused, vs two TP=1 refs) | HS `1.412e-02`–`1.650e-02` worst layer, `2.68e-02`–`5.84e-02` worst row; QK `8.67e-03`–`1.020e-02` q / k_full, `1.092e-02`–`1.361e-02` head; steer `3.27e-03`–`5.12e-03` `δ` rel, `0.101`–`0.142` nats unsteered | LSF 1781918 |
| **measured TP noise, QK** (fused and unfused) | `8.79e-03`–`1.068e-02` worst q / k_full layer; `1.094e-02`–`1.361e-02` worst head; TP=1 repeat `0` | LSF 1780395 |
| **measured TP noise, steer** | `3.27e-03`–`4.30e-03` `δ` rel; `0.101`–`0.191` nats unsteered logprob | LSF 1777562 |
| **measured TP noise, steer** (fused and unfused) | `3.38e-03`–`4.28e-03` `δ` rel; `0.101`–`0.124` nats unsteered logprob | LSF 1780395 |

The partial-sum row is the failure the TP port most plausibly produces (a capture op placed on
the wrong side of an all-reduce, e.g. once vLLM's fused all-reduce + RMSNorm moves the add). A
systematic one is caught at the worst layers — at L2 and L32 it is 14–36× the HS band (`1/√2` of
the update up to the whole update, over `5e-2`) — but a
partial-sum capture CONFINED to mid-depth layers (`1e-02`–`8e-02` there) would sit inside any
uniform band that also admits the last layer's measured noise. That is a limit of a per-layer
uniform band, recorded here rather than papered over.

### The bands

* `TP_HS_REL_BAND = 5e-2` (was the derived `1e-1`) — per (request, layer) `‖a−b‖_F / ‖b‖_F`:
  **3.0×** the worst measured (`1.6677e-02`, `tp4_nofuse` layer 32, 1777562; 1780395's worst,
  `1.6370e-02`, is inside it), 9× below a one-row shift, 20× below a zero-filled layer, and 14–36×
  below a systematic partial-sum capture at L2 / L32. The 3× covers the growth with TP degree seen
  so far (TP2 → TP4: ×1.07–1.14) and the fused path (1780395: fused and unfused alike). It is 3.8×
  the TP=1 boot noise 1780395 found (`1.318e-02`) and 1781918 confirmed (`1.319e-02`, three boots),
  and 3.4× the TP=2 boot-to-boot noise 1781918 measured (`1.472e-02`). 1781918's worst TP=N layer
  (`1.650e-02`, `tp4_nofuse`) and worst row (`5.84e-02`, `tp4` fused) are inside it. It cannot go
  below the boot noise. Unchanged.
* `TP_HS_ROW_BAND = 1.5e-1` (was `3e-1`) — the worst single row: **3.1×** the worst measured
  (`4.8124e-02`, 1777562; 1780395's worst is `3.8764e-02`); still 6.7× below the ~1.0 per-row
  shift. Unchanged.
* `TP_QK_REL_BAND = 1e-1` — merged `q` and `k_full` per (request, layer). Derived as tracking HS at
  the layer below plus the column-parallel `qkv_proj`'s own GEMM differences; **first MEASURED by
  1780395**: worst `1.0678e-02` (`tp4` fused, req1 layer 30 `k_full`), so the band is **9.4×** the
  measurement. **QK's own boot noise is now measured (1781918): exactly `0` at TP=1 over three
  boots and at TP=2 over two** (the HS child's two-state boot noise does not reach the QK child),
  and 1781918's worst QK TP error is `1.0199e-02` (`tp4`, `k_full`), 9.8× under the band. The
  headroom to tighten is real and recorded above; this band is NOT tightened here (a separate,
  cited change). Not widened, and nothing near it.
* `TP_QK_HEAD_REL_BAND = 3e-1` — the worst single head's column block (all compared rows): a head
  is 1/32 of a layer, so noisier; a swapped head is ~1.41, 4.7× above. **Measured by 1780395**:
  worst `1.3609e-02` (`tp2` fused), so the band is **22×** it; 1781918 measured the same worst
  (`tp2` and `tp2_repeat`, whose QK captures are md5-identical) and QK head boot noise `0` at TP=1
  and TP=2. Unchanged, for the same reason.
* `TP_STEER_DELTA_REL_BAND = 5e-2` (was `2.5e-1`) — `‖δ_N − δ_1‖ / ‖δ_1‖` over every prompt
  position: **11.6×** the worst measured (`4.2957e-03`, 1777562; 1780395's worst, fused `tp2`,
  `4.2810e-03`). The multiple is wider than HS's because this channel varies boot to boot at TP>1
  (the 1777562 `tp2` pair: 1.28× on `δ` rel, 1.9× on the unsteered logprobs; `tp2_repeat` measures
  it directly), and it had to cover the then-unmeasured fused path (1780395: fused `4.28e-03` /
  `3.50e-03`, unfused `3.38e-03` / `3.56e-03`). It is still 20× below one-rank or doubled steering
  (≥ 1), and it now also catches a steer scale error of a few percent, which `2.5e-1` did not.
  Unchanged.
* `TP_STEER_LIVENESS_FLOOR = 5e-1` (nats) — a FLOOR: `max|δ|` on each side must exceed it, so an
  inert steer cannot "agree" with another inert steer at zero. The noise it has to clear is the
  logprob movement that a numerically negligible, rounding-scale perturbation of the residual
  stream produces. G1 measured exactly that: the unsteered TP=N vs TP=1 teacher-forced
  `max|Δlogprob|`, up to **`0.191` nats** (`tp2`, 1777562; 1780395: `0.101`–`0.124`, fused legs
  included). The floor is **2.6×** above it, and **31×** below
  the live effect measured on every leg (`15.68`–`15.73` nats). (The pre-G1 rationale cited
  `7.1e-02`, the fp16 batch-width logprob noise from “T3”; the TP noise is 2.7× larger than that, so
  the old "7× the noise" margin was really 2.6×.) Not raised: `δ` is taken within ONE engine, where
  an inert steer gives exactly `0` (the TP=1 repeat is bit-identical), so the floor's job is to reject a
  steer doing no more than rounding-scale work, and 2.6× does that.

**If a fused leg exceeds a band re-derived here**, that is a FINDING: fusion moves MIA's TP
numerics by more than the stated margin. Record the measurement and the job in this section and
investigate (the partial-sum row above is the first suspect). Never silently restore a pre-G1
value. (1780395's fused legs did not: see its tables.)

### Fusion: ARMED vs APPLIED

**How the probe sees the pass.** Each rank's `AllReduceFusionPass` is reached through the compiled
model: `worker.get_model()`, its `torch.compile`'d submodule, the `VllmBackend` that compiled it
(`aot_compiled_fn._artifacts.compiled_fn.vllm_backend` under vLLM 0.29's default AOT compile), then
`pass_manager.passes`. Until 1780395 it scanned `gc.get_objects()`. That cannot work in vLLM 0.29:
`compile_or_warm_up_model` ends with `freeze_gc_heap()` (`gc.freeze()`), and CPython 3.12's
`gc.get_objects()` does not return the permanent generation. So the scan found no pass on any rank,
armed or not, and `_fusion_active` read that empty list as "not armed". The result was a false FAIL
on every fused leg and a vacuous PASS on every nofuse leg.

**ARMED** (the gate): every rank's configured post-grad pass manager holds an `AllReduceFusionPass`
that did not disable itself. False only on positive evidence: a disabled pass, or a configured pass
manager without one. **No evidence is None, and None FAILS** both a `--fuse on` leg and a
`--fuse off` leg. No evidence means no records, an error, a rank missing, or an empty list from a
rank where no configured pass manager was reached (e.g. a compile loaded from vLLM's compile cache,
where `vllm_backend` is None and no pass ran). An empty list alone is never "not armed". The
verdict recomputes this from the per-rank records, so re-judging 1780395's run dirs now FAILS its
fused AND nofuse legs for want of evidence (naming the pre-fix scan), not for an unarmed pass.

**APPLIED** (reported, never asserted): per rank, the number of all-reduce + RMSNorm pairs the
armed pass replaced in its LAST call. That is all vLLM 0.29 keeps. `AllReduceFusionPass.__call__`
OVERWRITES `matched_count` on every call (`allreduce_rms_fusion.py`). There is one call per
compiled graph piece and compile range, and FX-graph-cache hits skip it. Unlike vLLM's other
pattern passes, it does not add to `VllmPatternMatcherPass.match_table`. So a per-rank TOTAL is
unavailable (`matched_count_total: null`), and "fully" vs "partially" fused cannot be told from
it. For the **HS** workload a fused leg is expected to be only PARTIALLY fused.
`mia::capture_hs(out[0], out[1], ...)` (`mia/graph/install_hs.py`) is a second consumer of each
hooked layer's output, i.e. of the down_proj all-reduce. The pass replaces an all-reduce + RMSNorm
pair only when nothing else consumes the all-reduce output (`docs/tp_support.md` §2). With every
layer hooked, the o_proj all-reduces can fuse and the down_proj ones cannot. The probe records this
and never fails an HS leg for it. QK (q/k are read inside attention) is not expected to block the
pattern. Steer's `steer_buffer` mutates the residual in place between the down_proj all-reduce and
the next fused add + RMSNorm, which may block it. What fusion actually buys, and what MIA's ops take
back, is diagnostic D3/D4's question, not G1's. A fused HS leg's numbers describe a PARTIALLY fused
graph, and a fused-vs-unfused comparison of HS says nothing about the down_proj all-reduces.

**Discrimination is pinned hermetically** in `tests/test_tp_parity_probe.py`: identical runs
and in-band noise PASS, a diverging greedy trajectory PASSES on its common prefix, and a
rank-0-only QK capture, a recorded merge error, a missing layer, a stray HS rank dir (in
`flush_dirs` or on disk, empty or not), an HS leg that is not fully sharded (a missing rank dir, a
rank-0-only capture under the default layout, a header whose `owned_layers` break the rule, a
recorded `load_hs_aperture_tp` error), a replica differing from rank 0's by one ULP, an incomplete
replication leg, an expected rank dir holding only a sidecar header, a head
permutation, a zero-filled layer, an inert steer on either side, a doubled steer, a
requested-but-inactive fusion, a wrong child env, and a capturing rank whose writer process failed
to start, died, or was never recorded all FAIL. So do the fusion-evidence cases: a pass disabled on
one rank, a fused or unfused leg with no per-rank evidence (the pre-fix gc-scan record, a compile
loaded from cache), a candidate in band of one TP=1 reference but not the other, and a `--repeat`
leg of a different TP or fusion request. The locator is tested after `gc.freeze()`, on both the AOT
and the non-AOT route, shipped by value the way `collective_rpc` ships it, and its names are pinned
against the installed vLLM and torch sources.
