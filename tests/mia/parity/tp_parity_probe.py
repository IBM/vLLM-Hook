"""TP parity probe: MIA's HS / QK / steer at ``tensor_parallel_size=N`` against the same run at 1.

WHAT IT ANSWERS. Whether MIA's FULL-cudagraph capture and steering produce the SAME results at
TP=N as at TP=1 on vLLM 0.29 + the V2 model runner -- the gate (G1 of the model-scale study)
that has to pass before any large-model number counts:

  * HS  -- every layer (``output_hidden_states`` = 1..L), every token, prefill AND decode.
           At TP>1 MIA shards the layers ROUND-ROBIN across the ranks (``MIA_HS_TP_SHARD``,
           default 1: rank r captures the 0-based layers i % tp == r into ``tp_rank_<r>``), so
           every rank dir must hold data, each header must declare the rule's layers, and the
           probe reads the capture with MIA's own ``load_hs_aperture_tp`` (the UNION of the rank
           dirs; a gap or duplicate raises and is recorded as ``merge_error``) and compares that
           MERGED capture against TP=1. A leg run with ``MIA_HS_TP_SHARD=0`` (the pre-shard A/B
           layout) is judged by its own layout instead: exactly ``tp_rank_0``, sink ranks elsewhere.
  * hs_replicas -- the REPLICATION leg, TP>1 only: the HS workload with
           ``MIA_HS_CAPTURE_ALL_RANKS=1``, so every rank captures every layer. Each rank's copy
           must be BITWISE equal to rank 0's (the fact the layer shard rests on; FATAL, no band),
           MIA's own ``load_hs_aperture_tp(check_replicas=True)`` must agree, and rank 0's copy is
           then judged against TP=1 like the hs leg. At TP=1 the parent skips it and
           ``--compare`` reports it as INFO; a TP=1 reference contributes its hs leg.
  * QK  -- every layer (``output_qk`` = 0..L-1), all tokens, both phases. Every TP rank
           captures its own heads; the probe merges them with MIA's own
           ``mia.graph.aperture_reader.merge_qk_aperture_ranks`` and compares the FULL-WIDTH
           result (``num_attention_heads x head_dim`` for q, ``num_key_value_heads x head_dim``
           for k) against TP=1.
  * steer -- ``add_vector`` with a fixed unit vector on EVERY layer, steered and unsteered
           generations in one engine. The observable is the steering EFFECT on
           TEACHER-FORCED prompt logprobs (``delta = lp_steered - lp_unsteered`` per prompt
           position), which does not depend on where a greedy trajectory happens to diverge.

It runs Llama-3.1-8B from the local HF cache (``HF_HUB_OFFLINE=1``), bf16, greedy, fixed
prompts, ``ignore_eos``, prefix caching off, ``cudagraph_mode=FULL`` on the V2 runner. Each
workload boots its own engine in a child process (one MIA worker kind per engine).

WHY NOT BITWISE. TP=N changes the reduction order of every row-parallel matmul: each rank
reduces its K-slice in fp32, rounds to bf16, and the partial sums are then all-reduced (NCCL,
or vLLM's fused flashinfer all-reduce+RMSNorm, ``fuse_allreduce_rms``). vLLM 0.29's DEFAULT turns
that fusion on at TP>1 on Hopper/Blackwell only when ``has_flashinfer()`` (the ``flashinfer-cubin``
package or an ``nvcc`` on PATH), which is False in mia_v029, so a leg that asks nothing runs
unfused. An EXPLICIT ``pass_config.fuse_allreduce_rms=True`` (``--fuse on``) does not need
``has_flashinfer()``: ``AllReduceFusionPass`` needs only ``flashinfer.comm`` (with
``allreduce_fusion`` and ``create_allreduce_fusion_workspace``), a world size FlashInfer supports
(2/4/8/16, ``PassConfig.flashinfer_max_size``) and its workspace (``get_fi_ar_workspace``). That
is at least one extra bf16 rounding per ``o_proj``/``down_proj`` output, i.e.
2L extra roundings of relative size ~2^-9 fed into the residual stream, plus whatever GEMM
algorithm cuBLAS picks for the narrower per-rank shapes. So TP=N is a different, equally valid
numerical path, and the comparison is banded -- but only on VALUES. Everything structural is
exact and fatal, with no tolerance involved:

  * the set of layers each request captured (must be ALL layers on both sides),
  * the captured widths (HS ``hidden``; QK full ``H_q x d`` / ``H_kv x d`` -- a rank-0-only QK
    capture is ``H_q/tp x d`` wide and fails HERE, as does a merge error),
  * the rank dirs (HS: every rank under the layer shard, ``tp_rank_0`` at TP=1 or with
    ``MIA_HS_TP_SHARD=0``; QK and hs_replicas: exactly ``tp_rank_0..tp_rank_{N-1}``), and the
    HS headers' layer shard (``layer_shard`` / ``owned_layers`` recomputed from the rule),
  * replication: every hs_replicas rank bitwise equal to rank 0,
  * the row structure (``n_prompt + n_gen - 1`` rows; QK ``k_prefix_ends``),
  * prompt token ids.

Rows are compared only over the COMMON INPUT PREFIX: row ``r`` is the model's state for input
position ``r`` (prompt token, then generated tokens), so once the two greedy trajectories pick a
different token (a legitimate near-tie flip under a different numerical path) the rows after it
describe different inputs and are not comparable under any band. The probe reports how many
rows each request compared.

THE BANDS (module constants below, documented in ``tests/mia/parity/TOLERANCES.md``, section "TP
parity probe", with every measured value). HS and steer were MEASURED by gate G1 (LSF 1777562,
p3-r06-n4, 2026-09-19, TP 1/2/4, every leg UNFUSED -- ``fuse_allreduce_rms`` resolved False
throughout) and re-derived from it. The G1 re-run (LSF 1780395, p4-r19-n3, pin ``b44da1e``, fused
tp2/tp4 legs via ``--fuse on``) measured QK for the first time and every channel again; its numbers
sit inside every band below, so no band moved. Never widened to make a run pass.

  * failure signal: a zero-filled layer is relative error 1.0; a head or rank permutation is
    ~1.41 (two independent heads of similar norm); a whole-request one-row shift measured
    4.512e-01 tensor-global / ~1.0 per-row (LSF 1705469, TOLERANCES.md "T3"); a rank capturing its
    PRE-all-reduce partial sum misses the other ranks' share of the layer update, measured on the
    G1 TP=1 capture as ~1.0 at layer 2 and 1.6-1.8 at layer 32 (1.0e-2 - 2.1e-1 in between); a
    steer applied on one rank only, or applied once per rank and summed, changes the steering
    effect by O(1) of itself; a dead steer has ``max|delta| ~ 0``.
  * measured noise (G1): HS worst layer 1.467e-2 - 1.668e-2 (always the LAST layer; median over
    layers 1.4e-3 - 1.8e-3), worst row 2.99e-2 - 4.81e-2; steer delta rel 3.27e-3 - 4.30e-3;
    unsteered teacher-forced logprob max|diff| vs TP=1 0.101 - 0.191 nats; the TP=1 repeat is
    bit-identical. The random-walk estimate (2L extra bf16 roundings, sqrt(64) x 2^-9 ~ 1.6e-2)
    predicted the HS worst layer.
  * measured noise (G1 re-run, 1780395): HS worst layer 1.448e-2 - 1.637e-2 vs ``tp1`` (1.412e-2 -
    1.611e-2 vs ``tp1_repeat``), worst row 3.06e-2 - 3.88e-2; QK merged worst q 8.88e-3 - 1.047e-2,
    k_full 8.79e-3 - 1.068e-2, head 1.094e-2 - 1.361e-2; steer delta rel 3.38e-3 - 4.28e-3,
    unsteered 0.101 - 0.124 nats. NEW: the TP=1 HS noise floor is not zero -- that job's first
    ``tp1`` HS capture differs from ``tp1_repeat`` by 1.318e-2 (row 3.30e-2), about the size of the
    TP=N error, while ``tp1_repeat`` is bit-identical to both earlier TP=1 runs. The noise legs
    (``tp1_repeat2``, ``tp2_repeat``) and the multi-ref ``--compare`` below exist to measure this.

  ``TP_HS_REL_BAND = 5e-2``        per (request, layer), ``||a-b||_F / ||b||_F``: 3.0x the worst
                                    measured (1.6677e-2), 9x below a one-row shift, 20x below a
                                    zero-filled layer. (Was the derived 1e-1.)
  ``TP_HS_ROW_BAND = 1.5e-1``      worst single row ``||a_r-b_r|| / ||b_r||``: 3.1x the worst
                                    measured (4.8124e-2); 6.7x below the ~1.0 per-row shift. (Was
                                    3e-1.)
  ``TP_QK_REL_BAND = 1e-1``        q and k_full per (request, layer), tensor-global: derived by
                                    the same reasoning as HS (q/k are linear in the layer's input
                                    residual); first MEASURED by 1780395: 9.4x its worst
                                    (1.0678e-2). Not tightened until the noise legs measure QK's
                                    own boot noise.
  ``TP_QK_HEAD_REL_BAND = 3e-1``   worst single head's block (all compared rows): a head is 1/32
                                    of the data, so noisier; a swapped head is ~1.41, 4.7x above;
                                    22x the worst 1780395 measured (1.3609e-2).
  ``TP_STEER_DELTA_REL_BAND = 5e-2`` ``||delta_N - delta_1|| / ||delta_1||`` over every prompt
                                    position: 11.6x the worst measured (4.2957e-3) -- a wider
                                    multiple than HS because this channel is not boot-to-boot
                                    deterministic at TP>1 -- and 20x below a doubled or one-rank
                                    steer (>= 1). (Was 2.5e-1.)
  ``TP_STEER_LIVENESS_FLOOR = 5e-1`` nats; ``max|delta|`` on EACH side must exceed it, so an
                                    inert steer (at TP=1 or TP=N) fails instead of "agreeing"
                                    at zero. 2.6x the largest logprob movement a rounding-scale
                                    perturbation of the residual produced in G1 (the unsteered
                                    TP=N vs TP=1 max|diff|, 0.191 nats), and 31x below the live
                                    effect measured (15.68 nats).

CLI
---
Run one TP degree (writes ``<out>/``)::

    PYTHONPATH=<mia checkout> python tests/mia/parity/tp_parity_probe.py --tp 1 --out /x/tp1
    PYTHONPATH=<mia checkout> python tests/mia/parity/tp_parity_probe.py --tp 4 --out /x/tp4 --fuse on
    PYTHONPATH=<mia checkout> python tests/mia/parity/tp_parity_probe.py --tp 4 --out /x/tp4_nofuse \\
        --fuse off

Compare (GPU-free). ``--compare REF [REF ...] CAND``: the LAST dir is the candidate, every dir
before it a reference; the candidate is judged against EACH reference and the verdict is FAIL if
it fails against any of them::

    python tests/mia/parity/tp_parity_probe.py --compare /x/tp1 /x/tp4 [--json-out verdict.json]
    python tests/mia/parity/tp_parity_probe.py --compare /x/tp1 /x/tp1_repeat /x/tp4
    python tests/mia/parity/tp_parity_probe.py --repeat --compare /x/tp2 /x/tp2_repeat

Parity mode (the default) requires every reference to be TP=1. ``--repeat`` compares a leg with
a repeat of the SAME configuration instead (same TP, same ``--fuse`` request): a boot-to-boot
noise measurement, e.g. ``tp2`` vs ``tp2_repeat``; at TP>1 the fusion check then applies to both
sides.

NOISE LEGS. The G1 re-run found the TP=1 HS channel not boot-deterministic (the first ``tp1`` HS
capture 1.318e-2 away from ``tp1_repeat``, which matched both earlier TP=1 runs bit for bit), and
the TP>1 steer channel was already known not to be. So besides ``tp1``, ``tp2``, ``tp4``,
``tp2_nofuse``, ``tp4_nofuse`` and ``tp1_repeat`` a G1 job runs two NOISE legs, both ordinary probe
runs:

  * ``tp1_repeat2`` -- ``--tp 1`` a third time: three TP=1 captures, so an outlier TP=1 boot can be
    told from the others, and the TP=1 HS noise floor is the worst of the three pairs;
  * ``tp2_repeat`` -- ``--tp 2`` with ``tp2``'s own ``--fuse``: the TP>1 boot-to-boot noise of every
    channel (steer especially), read with ``--repeat --compare tp2 tp2_repeat``.

and compares every leg against BOTH TP=1 references, ``--compare tp1 tp1_repeat <leg>`` (the TP=1
repeats against the references before them: ``--compare tp1 tp1_repeat`` and ``--compare tp1
tp1_repeat tp1_repeat2``). Each verdict records which references it used (``refs``: label, dir,
and the sha256 of every workload's artifact, so a bit-identical TP=1 pair is visible at a glance)
and the result against each (``by_ref``).

Options: ``--workloads hs,qk,steer,hs_replicas`` (default all), ``--model`` (default
``meta-llama/Llama-3.1-8B``), ``--hf-cache`` (default: the first of the known local caches that
holds the model), ``--gpu-memory-utilization`` (0.85), ``--max-num-batched-tokens`` (8192),
``--max-model-len`` (2048), ``--max-tokens`` (8), ``--steer-coefficient`` (4.0),
``--scratch`` (default ``<out>/_scratch``: aperture dumps + vLLM compile cache),
``--require-mia-under DIR`` (refuse to run if ``import mia`` resolves elsewhere),
``--fuse on|off|default`` (vLLM 0.29's ``compilation_config.pass_config.fuse_allreduce_rms``:
``on`` = True, ``off`` = False, ``default`` = ask nothing and let vLLM decide; default
``default``; ``--no-fuse-allreduce-rms`` is an alias of ``--fuse off``). ``default`` resolves True
at TP>1 only when ``has_flashinfer()``; ``on`` needs only what ``AllReduceFusionPass`` needs
(``flashinfer.comm``, a supported world size, its workspace -- see WHY NOT BITWISE), and the pass
disables itself, logging a warning on local rank 0 only, when one is missing. So the resolved flag
is not proof, and each child records, besides the request and vLLM's RESOLVED value,
``fusion_env`` (``has_flashinfer``, ``flashinfer_cubin``, ``nvcc``) and every TP rank's
``AllReduceFusionPass`` state (``fusion_pass``, read by a ``collective_rpc`` callable).
Band overrides for ``--compare``: ``--hs-rel-band`` ``--hs-row-band`` ``--qk-rel-band``
``--qk-head-rel-band`` ``--steer-delta-rel-band`` ``--steer-liveness-floor``; any override is
flagged ``bands_overridden: true`` in the verdict.

FUSION EVIDENCE (``fusion_pass``, one record per TP rank). The pass is reached DIRECTLY, through
the compiled model: ``worker.get_model()`` -> every ``torch.compile``'d submodule (vLLM's
``@support_torch_compile`` classes) -> the ``VllmBackend`` that compiled it (AOT compile, vLLM
0.29's default with torch >= 2.10: ``aot_compiled_fn._artifacts.compiled_fn.vllm_backend``; else
the ``__compiled_fn_*`` callable dynamo installed in ``forward``'s module globals) ->
``pass_manager.passes``. (Until the G1 re-run it scanned ``gc.get_objects()``, which cannot see the
pass: vLLM 0.29's ``compile_or_warm_up_model`` ends with ``freeze_gc_heap()``, i.e.
``gc.freeze()``, and CPython 3.12's ``gc.get_objects()`` skips the permanent generation, so every
rank of every leg reported no pass.) Per rank: ``backends_configured`` (post-grad pass managers
found that were configured, i.e. that compiled in this process), ``compiled`` (each compiled
submodule, its route, ``loaded_from_disk`` and the pass classes of its pass manager) and
``passes`` (each ``AllReduceFusionPass``: ``disabled``, ``max_token_num``, ``tp_size``, ``called``,
``matched_count_last_call``). A per-rank TOTAL of replaced pairs is UNAVAILABLE in vLLM 0.29 (the
pass overwrites ``matched_count`` on every call and keeps no total): ``matched_count_total`` is
``null`` with the reason.

  * ARMED (``fusion_armed``, also ``fusion_active``): every rank's configured pass manager holds an
    ``AllReduceFusionPass`` that did not disable itself. False only on positive evidence (a
    configured pass manager without an armed pass, or a disabled pass). None = NO EVIDENCE: no
    rank records, a record with an ``error``, one record short, or an empty ``passes`` list with no
    configured pass manager seen (a compile loaded from the compile cache, or a pre-fix gc-scan
    record). An empty list is never read as "not armed".
  * APPLIED (``fusion_applied``): per rank, what the armed pass's LAST call replaced (``null``: it
    never ran on that rank). Reported, never asserted: the last call covers one compiled graph
    piece and compile range, so it cannot say "fully" or "partially" fused. A fused HS leg is
    EXPECTED to be only partially fused (``mia::capture_hs`` is a second consumer of each hooked
    layer's down_proj all-reduce output), and is never failed for it.

Child environment: ``VLLM_PLUGINS`` and ``MIA_APERTURE_PER_REQUEST`` are REMOVED (never inherited,
never defaulted -- ``DROPPED_ENV``), so MIA's plugin loads exactly as in a production plugin cell
and the shared-file drain is what is judged; ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` is set for the
fusion-state RPC. ``MIA_HS_CAPTURE_ALL_RANKS`` is removed for every leg but hs_replicas, which sets
it to 1; ``MIA_HS_TP_SHARD`` is inherited (export ``MIA_HS_TP_SHARD=0`` to run the rank-0-only A/B
leg), and ``--compare`` derives each leg's expected HS layout from the env its child RECORDED. Each child records these plus EVERY ``MIA_*`` / ``VLLM_*`` variable it saw
(``env``; credential-looking names redacted), whether MIA's plugin patched vLLM
(``mia_plugin_loaded``), and each TP rank's save mode after the workload (``writer``: MIA's
``worker._writer_mode``, and whether the writer child is alive, read by a ``collective_rpc``
callable).

Aperture listing: the parent wipes ``<scratch>/<wl>/aperture`` before each workload's child (so a
reused ``--out`` / ``--scratch`` cannot contribute stale rank dirs), and after the child exits it
writes ``<out>/<wl>/aperture_listing.json`` -- every ``tp_rank_*`` dir under that child's
``MIA_APERTURE_DIR`` with each file's name and size -- before anything can delete the scratch.

VERDICT (``--compare``): ONE JSON object printed as the LAST line of stdout (and written to
``--json-out``)::

    {"verdict": "PASS" | "FAIL",            # FAIL if the candidate fails against ANY reference
     "mode": "parity" | "repeat",
     "ref":  {"dir", "tp", "fuse_allreduce_rms", "requested_fuse_allreduce_rms",
              "fusion_active": {wl: bool | None}, "fusion_armed": {wl: bool | None},
              "fusion_applied": {wl: {...}}, "fusion_env", "mia", "mia_git", "model",
              "label", "artifact_sha256": {wl: hex | None}},     # the FIRST reference
     "refs": [{...same, one per reference, in the order given...}],
     "ref_labels": [label, ...],
     "refs_identical": {wl: bool | None} (two or more references; None = no reference has
                       that leg's artifact, e.g. hs_replicas against TP=1 references),
     "cand": {...same...},
     "bands": {name: value}, "bands_overridden": bool,
     "checks": [{"workload", "name", "status": "PASS"|"FAIL"|"INFO", "detail",
                 "value"?, "band"?, "ref": label}, ...],
     "measured": {"hs": {"worst_rel", "worst_row_rel", "worst_at", "rows_compared"},
                  "hs_replicas": {...as hs, rank 0's copy...},
                  "qk": {"worst_q_rel", "worst_k_rel", "worst_head_rel", "worst_at",
                         "rows_compared"},
                  "steer": {"delta_rel", "ref_max_abs_delta", "cand_max_abs_delta",
                            "unsteered_max_abs_logprob_diff"}},   # against the FIRST reference
     "by_ref": {label: {"dir", "verdict", "measured", "failures"}},
     "failures": [str, ...]}    # "[wl] (vs <label>) ..." when there are two or more references

Besides the capture checks, every workload adds, per side: ``<side> child env`` and ``<side> MIA
plugin loaded`` (FATAL), ``<side> MIA_HS_CAPTURE_ALL_RANKS`` for the HS legs (FATAL: unset for hs,
``1`` for hs_replicas), ``<side> inherited MIA_* env`` (INFO), ``<side> aperture dirs on disk``
(FATAL: exactly the HS layout's capturing ranks -- every rank under the layer shard,
``tp_rank_0`` at TP=1 / ``MIA_HS_TP_SHARD=0`` -- QK and hs_replicas ``tp_rank_0..N-1``, steer none,
each expected dir holding DATA, i.e. ``*.raw`` bytes, not a header-only sidecar; a stray dir,
empty or not, a missing one, or a missing listing FAILS), ``<side> writer process`` (FATAL for the
capture legs: every capturing rank -- QK and hs_replicas all, HS every rank that owns a layer --
runs ``"process"`` with its child alive after the workload, or ``"in-process
(MIA_WRITER_PROCESS=0)"`` when the recorded env says so; an HS rank that captures nothing (only in
the ``MIA_HS_TP_SHARD=0`` layout) ``"none (HS sink rank: captures nothing)"``; anything else, e.g.
"failed to start", or no record FAILS; steer INFO), and the fusion checks on the CANDIDATE (both
sides in ``--repeat`` mode):
``--fuse on`` at TP>1 FAILS with "fusion not active" unless vLLM resolved
``fuse_allreduce_rms=True`` AND the pass is ARMED on every rank (no evidence FAILS too, naming
what is missing), plus an INFO ``fusion applied`` line; ``--fuse off`` FAILS if fusion resolved or
armed anyway, or if there is no per-rank evidence that it is not armed; ``default`` is INFO. A TP=1
side is exempt (INFO): vLLM never fuses at TP=1.

Exit status: 0 = PASS, 1 = FAIL, 2 = the probe itself could not run (usage, missing files).
A run (not ``--compare``) exits 0 only if every requested workload's child succeeded.

Importable with no GPU and no vLLM: the heavy imports happen inside the child only. The
comparison is unit-tested in ``tests/test_tp_parity_probe.py`` against fabricated run dirs,
including a rank-0-only QK capture and a missing layer, both of which must FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
#: Extra read-only HF hub caches to search before falling back to the environment's own.
#: The first that has the model wins unless --hf-cache is given. Empty by default: set
#: MIA_EXTRA_HF_CACHES to a colon-separated list of directories on a machine where the
#: models live outside HF_HOME, rather than hard-coding one site's paths here.
KNOWN_HF_CACHES = tuple(
    p for p in os.environ.get("MIA_EXTRA_HF_CACHES", "").split(os.pathsep) if p
)

#: Fixed, greedy prompts. Different lengths (so no two requests are interchangeable) and
#: different first tokens.
PROMPTS = (
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Tensor parallelism splits every attention layer's heads across GPUs, so",
)

#: ``hs_replicas`` is the REPLICATION leg: the HS workload with ``MIA_HS_CAPTURE_ALL_RANKS=1``, so
#: every rank captures every layer into its own dir and the ranks' copies are compared BITWISE (the
#: residual is replicated; the layer shard of the ``hs`` workload relies on it). It needs TP > 1:
#: at TP = 1 the parent skips it and ``--compare`` reports it as INFO.
WORKLOADS = ("hs", "qk", "steer", "hs_replicas")
HS_WORKLOADS = ("hs", "hs_replicas")
WORKER_KIND = {"hs": "hidden_states", "qk": "qk", "steer": "steer",
               "hs_replicas": "hidden_states"}
WORKER_CLS = {
    "hs": "mia.workers.hs_capture_worker.HSCaptureWorker",
    "qk": "mia.workers.qk_capture_worker.QKCaptureWorker",
    "steer": "mia.workers.steer_worker.SteerWorker",
    "hs_replicas": "mia.workers.hs_capture_worker.HSCaptureWorker",
}
#: ``hs_replicas``: every rank's per-request per-layer capture, ``{"ranks": {r: [{layer:
#: Tensor}, ...]}}`` in PROMPTS order -- what ``--compare`` checks bitwise.
REPLICAS_FILE = "replicas.pt"
#: The HS layout env (mirrors ``mia.graph.tp_shard``; the compare side does not import mia).
HS_SHARD_ENV = "MIA_HS_TP_SHARD"
HS_ALL_RANKS_ENV = "MIA_HS_CAPTURE_ALL_RANKS"

# --------------------------------------------------------------------------------------
# Bands -- see the module docstring and TOLERANCES.md "TP parity probe". HS and steer re-derived
# from the G1 measurement (LSF 1777562) with the stated margins; QK derived, then first measured
# by the G1 re-run (LSF 1780395), which moved no band. Never widened to make a run pass.
# --------------------------------------------------------------------------------------
TP_HS_REL_BAND = 5e-2
TP_HS_ROW_BAND = 1.5e-1
TP_QK_REL_BAND = 1e-1
TP_QK_HEAD_REL_BAND = 3e-1
TP_STEER_DELTA_REL_BAND = 5e-2
TP_STEER_LIVENESS_FLOOR = 5e-1

DEFAULT_BANDS = {
    "hs_rel": TP_HS_REL_BAND,
    "hs_row": TP_HS_ROW_BAND,
    "qk_rel": TP_QK_REL_BAND,
    "qk_head_rel": TP_QK_HEAD_REL_BAND,
    "steer_delta_rel": TP_STEER_DELTA_REL_BAND,
    "steer_liveness_floor": TP_STEER_LIVENESS_FLOOR,
}

CHILD_MANIFEST = "child_manifest.json"
CAPTURE_FILE = "capture.pt"
STEER_FILE = "generation.pt"
#: Written by the PARENT into ``<out>/<wl>/`` after that workload's child has exited (every rank's
#: teardown has run) and before anything can delete the scratch: every ``tp_rank_*`` dir under the
#: child's ``MIA_APERTURE_DIR`` with each file's name and size. ``--compare`` judges it.
APERTURE_LISTING = "aperture_listing.json"
_RANK_DIR_RE = re.compile(r"^tp_rank_(\d+)$")

#: ``--fuse`` -> what the child asks vLLM 0.29 for in ``compilation_config.pass_config``.
#: ``default`` asks nothing (vLLM decides: on only at TP>1 on Hopper/Blackwell when
#: ``has_flashinfer()``, i.e. the ``flashinfer-cubin`` package or an ``nvcc`` on PATH). ``on`` is an
#: explicit True, which does NOT need ``has_flashinfer()``: ``AllReduceFusionPass`` needs only
#: ``flashinfer.comm``, a supported world size and its workspace (and disables itself without them).
FUSE_REQUEST = {"on": True, "off": False, "default": "vllm-default"}

#: How ``_fusion_pass_state`` reaches vLLM 0.29's ``AllReduceFusionPass`` on a worker (recorded in
#: every rank's record). NOT ``gc.get_objects()``: vLLM 0.29's ``compile_or_warm_up_model`` ends
#: with ``freeze_gc_heap()`` (``gc.freeze()``), and CPython 3.12's ``gc.get_objects()`` does not
#: return the permanent generation, so a gc scan finds no pass on any rank, armed or not.
FUSION_LOCATOR = ("worker.get_model() -> each torch.compile'd submodule -> its VllmBackend "
                  "(aot_compiled_fn._artifacts.compiled_fn.vllm_backend, or the __compiled_fn_* "
                  "global dynamo installed for forward) -> pass_manager.passes")
#: Why a rank record's ``matched_count_total`` is null.
FUSION_TOTAL_UNAVAILABLE = (
    "unavailable in vLLM 0.29: AllReduceFusionPass.__call__ overwrites matched_count on every "
    "call (one per compiled graph piece and compile range; FX-graph-cache hits skip the call) "
    "and, unlike vLLM's other pattern passes, does not add to VllmPatternMatcherPass.match_table; "
    "matched_count_last_call is the last call's count only")

#: Environment the probe child must NOT inherit (W1): ``VLLM_PLUGINS`` -- an inherited allowlist
#: ("" from a baseline cell, a stale name list) can silently keep MIA's plugin from loading; unset,
#: vLLM loads every installed ``vllm.general_plugins`` entry point, exactly as a production plugin
#: cell does. ``MIA_APERTURE_PER_REQUEST`` -- per-request delivery changes the capture layout and
#: flush_aperture's return; the probe judges the shared-file drain.
DROPPED_ENV = ("VLLM_PLUGINS", "MIA_APERTURE_PER_REQUEST")

#: Recorded in every child manifest's ``env`` (None when unset), on top of EVERY ``MIA_*`` and
#: ``VLLM_*`` variable the child actually saw: an inherited ``MIA_HS_CAPTURE_ALL_RANKS``,
#: ``MIA_HS_TP_SYMMETRIC``, ``MIA_APERTURE_GPU_BYTES`` or ``MIA_CAPTURE_DRAIN_APERTURE`` changes the
#: layout or the capture, and a verdict must be able to show it.
RECORDED_ENV = (*DROPPED_ENV, "VLLM_ALLOW_INSECURE_SERIALIZATION", "MIA_WORKER",
                "MIA_ALLOW_CUDAGRAPH", "CUDA_VISIBLE_DEVICES")
#: Values of variables whose NAME looks like a credential are recorded as "<redacted>".
_SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
#: What the probe itself sets in the child (not "inherited").
_PROBE_SET_ENV = ("MIA_ALLOW_CUDAGRAPH", "MIA_WORKER", "MIA_APERTURE_DIR",
                  "VLLM_ALLOW_INSECURE_SERIALIZATION", "VLLM_WORKER_MULTIPROC_METHOD",
                  "VLLM_CACHE_ROOT")


def _recorded_env(environ=None) -> dict:
    """The child manifest's ``env``: ``RECORDED_ENV`` (None when unset) plus every ``MIA_*`` /
    ``VLLM_*`` variable present, credential-looking names redacted."""
    env = os.environ if environ is None else environ
    names = set(RECORDED_ENV) | {k for k in env if k.startswith(("MIA_", "VLLM_"))}
    return {k: ("<redacted>" if env.get(k) is not None and _SECRET_NAME_RE.search(k)
                else env.get(k)) for k in sorted(names)}


# ======================================================================================
# RUN side (GPU)
# ======================================================================================

def _find_hf_cache(model: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    folder = "models--" + model.replace("/", "--")
    for c in KNOWN_HF_CACHES:
        if os.path.isdir(os.path.join(c, folder, "snapshots")):
            return c
    return None


def _child_env(args, wl: str, scratch: Path) -> dict:
    """The environment one WORKLOAD's child runs under.

    ``VLLM_CACHE_ROOT`` is per WORKLOAD (``<root>/<leg>/<workload>``), not per leg. MIA's
    compile-cache stamp (``mia/_plugin.py::mia_graph_layout``) now carries the baked-buffer
    geometry, so two workloads can no longer collide inside one cache; this split is defence
    in depth, and it is free here -- these legs compile cold anyway. It closes the exact
    failure of LSF 1794720, where ``hs`` and ``hs_replicas`` shared one root, the second
    engine loaded the first's ``torch_aot_compile`` artifact, and engine init died in
    ``determine_available_memory()`` on ``expected size 16385==1`` / ``16385==65537`` -- the
    two HS aperture geometries.

    An inherited ``VLLM_CACHE_ROOT`` is honoured as the ROOT (that is how a caller puts the
    cache on node-local disk) but never used verbatim -- using it verbatim is what let every
    workload share one cache. The leg name is the ``--out`` directory's, so legs that share a
    ``--scratch`` stay separate too.
    """
    env = dict(os.environ)
    env["MIA_ALLOW_CUDAGRAPH"] = "1"
    env["MIA_WORKER"] = WORKER_KIND[wl]
    env["MIA_APERTURE_DIR"] = str(scratch / wl / "aperture")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    cache = _find_hf_cache(args.model, args.hf_cache)
    if cache:
        env["HF_HUB_CACHE"] = cache
    for k in DROPPED_ENV:                   # see DROPPED_ENV: never inherited, never defaulted
        env.pop(k, None)
    # The HS replication diagnostic is the hs_replicas leg's whole point and never the hs leg's:
    # an inherited MIA_HS_CAPTURE_ALL_RANKS=1 would turn the hs leg's layer shard into replicas.
    # (MIA_HS_TP_SHARD IS inherited: exporting MIA_HS_TP_SHARD=0 runs the rank-0-only A/B leg, and
    # --compare derives the expected layout from the recorded env.)
    if wl == "hs_replicas":
        env[HS_ALL_RANKS_ENV] = "1"
    else:
        env.pop(HS_ALL_RANKS_ENV, None)
    # The child reads each TP worker's fusion-pass state with a collective_rpc CALLABLE
    # (_fusion_pass_state), which vLLM 0.29 only ships across the engine-core boundary with this.
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TORCHDYNAMO_DISABLE"] = "0"        # graph mode compiles; never inherit a stray "1"
    env.pop("VLLM_USE_V2_MODEL_RUNNER", None)  # 0.29 defaults to V2; MIA refuses V1 anyway
    cache_root = (os.environ.get("VLLM_CACHE_ROOT") or "").strip() or str(scratch / "vllm_cache")
    env["VLLM_CACHE_ROOT"] = str(Path(cache_root) / (Path(args.out).name or "leg") / wl)
    return env


def _child_argv(args, wl: str) -> list:
    argv = [sys.executable, str(Path(__file__).resolve()), "--child-workload", wl,
            "--tp", str(args.tp), "--out", str(args.out), "--model", args.model,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--max-model-len", str(args.max_model_len),
            "--max-tokens", str(args.max_tokens),
            "--steer-coefficient", str(args.steer_coefficient)]
    argv += ["--fuse", args.fuse]
    if args.require_mia_under:
        argv += ["--require-mia-under", args.require_mia_under]
    return argv


def run_parent(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else out / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    bad = [w for w in workloads if w not in WORKLOADS]
    if bad:
        print(f"[tp-probe] unknown workload(s) {bad}; known {WORKLOADS}", file=sys.stderr)
        return 2
    manifest = {"tp": args.tp, "model": args.model, "workloads": {}, "fuse": args.fuse,
                "requested_fuse_allreduce_rms": FUSE_REQUEST[args.fuse],
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ok_all = True
    for wl in workloads:
        if wl == "hs_replicas" and int(args.tp) <= 1:
            # One rank has no replicas to compare; --compare reports this leg as INFO at TP=1.
            manifest["workloads"][wl] = {"status": "skipped", "reason": "TP=1: one rank"}
            print(f"[tp-probe] tp={args.tp} {wl}: skipped (TP=1 has one rank)", flush=True)
            continue
        wl_dir = out / wl
        wl_dir.mkdir(parents=True, exist_ok=True)
        log = wl_dir / "child.log"
        t0 = time.monotonic()
        (wl_dir / CHILD_MANIFEST).unlink(missing_ok=True)      # never judge a stale run
        # ...nor a stale aperture: with --out / --scratch reused, an earlier run's tp_rank_* dirs
        # would be listed as this child's (a false "stray", or a "missing" one hidden).
        shutil.rmtree(scratch / wl / "aperture", ignore_errors=True)
        with open(log, "w") as fh:
            try:
                rc = subprocess.run(_child_argv(args, wl), env=_child_env(args, wl, scratch),
                                    stdout=fh, stderr=subprocess.STDOUT,
                                    timeout=args.child_timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        status = "ok" if rc == 0 and (wl_dir / CHILD_MANIFEST).exists() else "failed"
        ok_all &= status == "ok"
        # The child has exited, so every rank's teardown (and any file it creates) is done; the
        # scratch still exists (a caller deletes it only after this parent returns). Listed even
        # for a failed child: what a crashed rank left behind is evidence too.
        listing = list_aperture_dir(scratch / wl / "aperture")
        (wl_dir / APERTURE_LISTING).write_text(json.dumps(listing, indent=2))
        manifest["workloads"][wl] = {"status": status, "returncode": rc,
                                     "seconds": round(time.monotonic() - t0, 1),
                                     "log": str(log),
                                     "aperture_rank_dirs": {
                                         k: d["total_bytes"]
                                         for k, d in listing["rank_dirs"].items()}}
        print(f"[tp-probe] tp={args.tp} {wl}: {status} (rc={rc}, "
              f"{manifest['workloads'][wl]['seconds']}s) log={log}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0 if ok_all else 1


def list_aperture_dir(root) -> dict:
    """Everything a run left under ``MIA_APERTURE_DIR``: each ``tp_rank_*`` dir with every file
    in it (path relative to the rank dir, bytes) and its total, plus any other top-level entry.

    ``{"root", "exists", "rank_dirs": {"tp_rank_<r>": {"files": [{"name", "bytes"}],
    "total_bytes"}}, "other": [{"name", "is_dir", "bytes"}]}``. Pure filesystem, GPU-free."""
    root = Path(root)
    out: dict = {"root": str(root), "exists": root.is_dir(), "rank_dirs": {}, "other": [],
                 "listed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not out["exists"]:
        return out
    for e in sorted(root.iterdir()):
        if e.is_dir() and _RANK_DIR_RE.match(e.name):
            files = []
            for dirpath, _dirs, names in os.walk(e):
                for n in sorted(names):
                    fp = Path(dirpath) / n
                    try:
                        size = fp.stat().st_size
                    except OSError:
                        size = -1
                    files.append({"name": str(fp.relative_to(e)), "bytes": size})
            out["rank_dirs"][e.name] = {"files": files,
                                        "total_bytes": sum(max(0, f["bytes"]) for f in files)}
        else:
            try:
                size = e.stat().st_size
            except OSError:
                size = -1
            out["other"].append({"name": e.name, "is_dir": e.is_dir(), "bytes": size})
    return out


def _find_vllm_backend(obj, max_nodes: int = 64):
    """Follow a compiled callable to the ``VllmBackend`` that compiled it: the first object on the
    way whose ``__dict__`` holds ``vllm_backend`` (vLLM 0.29's ``VllmSerializableFunction``),
    unwrapping torch's wrappers on the way (``compiled_fn``, ``_torchdynamo_orig_callable``,
    ``__wrapped__``, closure cells; at most ``max_nodes`` objects).

    Returns ``(holder, backend)``. ``holder`` None: nothing with a ``vllm_backend`` was reached.
    ``backend`` None WITH a holder: the function was rebuilt from vLLM's compile cache
    (``reconstruct_serializable_fn_from_mega_artifact`` passes ``vllm_backend=None``), so no pass
    manager ran in this process and there is nothing to inspect."""
    seen: set = set()
    queue = [obj]
    while queue and len(seen) < max_nodes:
        o = queue.pop(0)
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict) and "vllm_backend" in d:
            return o, d["vllm_backend"]
        for attr in ("compiled_fn", "_torchdynamo_orig_callable", "__wrapped__"):
            try:
                nxt = getattr(o, attr, None)
            except Exception:  # noqa: BLE001 -- a hostile __getattr__ is just a dead end
                nxt = None
            if nxt is not None:
                queue.append(nxt)
        for cell in (getattr(o, "__closure__", None) or ()):
            try:
                queue.append(cell.cell_contents)
            except ValueError:                  # an empty cell
                pass
    return None, None


def _compiled_backends(model) -> list:
    """Every ``torch.compile``'d submodule of ``model`` -- vLLM's ``@support_torch_compile``
    classes, which carry ``_compiled_callable`` (and ``aot_compiled_fn`` once AOT-compiled) in
    their ``__dict__`` -- with the ``VllmBackend`` that compiled it: ``[(record, backend)]``.

    Routes, in order: the AOT artifact (vLLM 0.29's default with torch >= 2.10:
    ``aot_compiled_fn._artifacts.compiled_fn.vllm_backend``), else the ``__compiled_fn_*`` callable
    dynamo installed in the globals of the module defining ``forward`` (non-AOT compile).
    ``backend`` is None when neither reaches one; the record says why (``note``)."""
    found = []
    named = model.named_modules() if hasattr(model, "named_modules") else [("", model)]
    for name, mod in named:
        d = getattr(mod, "__dict__", None) or {}
        aot = d.get("aot_compiled_fn")
        if "_compiled_callable" not in d and aot is None:
            continue
        rec: dict = {"module": name or "<root>", "class": type(mod).__name__, "route": None,
                     "loaded_from_disk": d.get("was_aot_compile_fn_loaded_from_disk")}
        holder = backend = None
        if aot is not None:
            holder, backend = _find_vllm_backend(
                getattr(getattr(aot, "_artifacts", None), "compiled_fn", None))
            if holder is not None:
                rec["route"] = "aot_compiled_fn._artifacts.compiled_fn.vllm_backend"
        if holder is None:
            glb = getattr(getattr(type(mod), "forward", None), "__globals__", None) or {}
            for key in sorted(k for k in list(glb) if str(k).startswith("__compiled_fn")):
                h, b = _find_vllm_backend(glb[key])
                if h is not None:
                    holder, backend = h, b
                    rec["route"] = f"forward.__globals__[{key!r}] -> vllm_backend"
                    if b is not None:
                        break
        if holder is None:
            rec["note"] = "no compiled function carrying a vllm_backend was reached"
        elif backend is None:
            rec["note"] = ("vllm_backend is None: the compiled function was rebuilt from vLLM's "
                           "compile cache, so no pass manager ran in this process")
        found.append((rec, backend))
    return found


def _fusion_pass_state(worker) -> dict:
    """collective_rpc CALLABLE, run on EVERY TP worker: is vLLM's all-reduce + RMSNorm fusion pass
    really armed on this rank, and what did it replace? The resolved config flag alone is not
    proof: an explicit ``fuse_allreduce_rms=True`` stays True in the config while
    ``AllReduceFusionPass`` disables itself at construction (no flashinfer comm module, workspace
    init failed, unsupported world size) and only logs a warning, on local rank 0.

    The pass is reached through the compiled model (``FUSION_LOCATOR``), never by a
    ``gc.get_objects()`` scan: vLLM 0.29 freezes the worker heap (``gc.freeze()``) after warm-up,
    and a frozen object is invisible to that scan. ``backends_configured`` counts the post-grad pass
    managers reached that were configured, i.e. compiled in this process (``configure()`` sets
    ``pass_config``); with none, an empty ``passes`` list is NOT evidence of an unarmed pass.
    Shipped by value (cloudpickle), so it imports what it needs."""
    out: dict = {"rank": getattr(worker, "rank", None), "locator": FUSION_LOCATOR}
    try:
        out["config"] = worker.vllm_config.compilation_config.pass_config.fuse_allreduce_rms
    except Exception as e:  # noqa: BLE001
        out["config_error"] = repr(e)
    try:
        from vllm.compilation.passes.fusion import allreduce_rms_fusion as arf
        out["flashinfer_comm"] = arf.flashinfer_comm is not None
        compiled, passes, configured = [], [], 0
        for rec, backend in _compiled_backends(worker.get_model()):
            pm = getattr(backend, "pass_manager", None) if backend is not None else None
            rec["backend"] = backend is not None
            rec["pass_manager_configured"] = pm is not None and hasattr(pm, "pass_config")
            rec["pass_names"] = [type(p).__name__ for p in (getattr(pm, "passes", None) or [])]
            if rec["pass_manager_configured"]:
                configured += 1
                for p in pm.passes:
                    if isinstance(p, arf.AllReduceFusionPass):
                        pd = vars(p)
                        passes.append({
                            "module": rec["module"],
                            "disabled": bool(pd.get("disabled", True)),
                            "max_token_num": pd.get("max_token_num"),
                            "tp_size": pd.get("tp_size"),
                            # matched_count is a class attribute (0) until the first call
                            "called": "matched_count" in pd,
                            "matched_count_last_call": pd.get("matched_count")})
            compiled.append(rec)
        out.update(compiled=compiled, backends_configured=configured, passes=passes,
                   matched_count_total=None, matched_count_total_note=FUSION_TOTAL_UNAVAILABLE)
        if not compiled:
            out["note"] = "worker.get_model() has no torch.compile'd submodule"
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def _writer_state(worker) -> dict:
    """collective_rpc CALLABLE, run on EVERY TP worker: the save mode MIA recorded for this rank
    (``worker._writer_mode``: ``"process"``, ``"in-process (...)"`` or ``"none (HS sink rank:
    ...)"``) and whether its writer child is still alive. Shipped by value (cloudpickle)."""
    out: dict = {"rank": getattr(worker, "rank", None),
                 "writer_mode": getattr(worker, "_writer_mode", None)}
    wp = getattr(worker, "_writer_process", None)
    if wp is not None:
        try:
            out["alive"] = bool(wp.alive())
        except Exception as e:  # noqa: BLE001
            out["alive_error"] = repr(e)
        out["child_pids"] = [getattr(p, "pid", None) for p in (getattr(wp, "_procs", None) or [])]
        out["parent_daemonic"] = getattr(wp, "parent_daemonic", None)
    return out


def _rank_has_pass_evidence(r) -> bool:
    """A rank record says something about arming: it lists at least one AllReduceFusionPass, or it
    reached a CONFIGURED post-grad pass manager (whose ``passes`` then really hold no such pass)."""
    return (isinstance(r, dict) and not r.get("error") and isinstance(r.get("passes"), list)
            and bool(r["passes"] or r.get("backends_configured")))


def _fusion_active(per_rank) -> bool | None:
    """ARMED: True iff EVERY rank holds an AllReduceFusionPass that did not disable itself.

    False only on POSITIVE evidence that some rank has none armed: a disabled pass, or a configured
    pass manager holding no such pass. None = NO usable evidence, which FAILS a leg that requested
    fusion: no rank records, a record carrying an ``error``, or an empty ``passes`` list from a rank
    where no configured pass manager was reached (a compile loaded from vLLM's cache, or the old
    ``gc.get_objects()`` scan, which saw nothing after vLLM's ``gc.freeze()``). An empty list alone
    is never read as 'not armed'."""
    if not isinstance(per_rank, list) or not per_rank:
        return None
    if not all(_rank_has_pass_evidence(r) for r in per_rank):
        return None
    return all(any(not p.get("disabled", True) for p in r["passes"] if isinstance(p, dict))
               for r in per_rank)


def _fusion_applied(per_rank) -> dict:
    """APPLIED, as far as vLLM 0.29 lets anyone see it: per rank, how many all-reduce + RMSNorm
    pairs the ARMED pass replaced in its LAST call (None: no armed pass ran on that rank, or no
    record says). ``status``: ``applied`` (every rank's last call replaced >= 1), ``none`` (every
    rank's armed pass ran and replaced 0), ``mixed``, or ``unknown`` (a rank without a count).

    Reported, never asserted: one call covers one compiled graph piece and compile range, and no
    total is kept (``FUSION_TOTAL_UNAVAILABLE``), so it cannot tell a fully fused graph from a
    partially fused one -- and a fused HS graph IS expected to be partial."""
    by_rank: dict = {}
    for i, r in enumerate(per_rank if isinstance(per_rank, list) else []):
        if not isinstance(r, dict):
            continue
        rank = r.get("rank") if r.get("rank") is not None else i
        counts = [p["matched_count_last_call"] for p in (r.get("passes") or [])
                  if isinstance(p, dict) and not p.get("disabled", True) and p.get("called")
                  and type(p.get("matched_count_last_call")) is int]
        by_rank[str(rank)] = sum(counts) if counts else None
    vals = list(by_rank.values())
    if not vals or any(x is None for x in vals):
        status = "unknown"
    elif all(x > 0 for x in vals):
        status = "applied"
    elif all(x == 0 for x in vals):
        status = "none"
    else:
        status = "mixed"
    return {"status": status, "last_call_by_rank": by_rank, "total": None,
            "scope": "the armed pass's last call on each rank",
            "total_note": FUSION_TOTAL_UNAVAILABLE}


def _fusion_evidence_gaps(per_rank, tp: int) -> list:
    """Why there is no arming evidence, rank by rank (for the verdict's FAIL message)."""
    if not isinstance(per_rank, list) or not per_rank:
        return ["no per-rank fusion records"]
    gaps = [] if len(per_rank) == tp else [f"{len(per_rank)} rank record(s) for tp={tp}"]
    for i, r in enumerate(per_rank):
        if not isinstance(r, dict):
            gaps.append(f"record {i}: {r!r}")
            continue
        rank = r.get("rank") if r.get("rank") is not None else i
        if r.get("error"):
            gaps.append(f"rank {rank}: {r['error']}")
        elif not isinstance(r.get("passes"), list):
            gaps.append(f"rank {rank}: no passes recorded")
        elif not _rank_has_pass_evidence(r):
            if "compiled" not in r:
                why = ("a pre-fix record from the gc.get_objects() scan, which cannot see the pass "
                       "after vLLM's gc.freeze(): re-run the leg")
            else:
                notes = sorted({c.get("note") for c in r["compiled"]
                                if isinstance(c, dict) and c.get("note")})
                why = "; ".join(notes) or r.get("note") or "no configured post-grad pass manager"
            gaps.append(f"rank {rank}: no pass evidence ({why})")
    return gaps


def _fusion_armed(m) -> bool | None:
    """ARMED for one side, RECOMPUTED from its per-rank records -- never the stored
    ``fusion_active``, which the child judged with whatever rule it was built with. One record per
    TP rank is required."""
    fp = m.get("fusion_pass")
    if not isinstance(fp, list) or len(fp) != int(m.get("tp") or 1):
        return None
    return _fusion_active(fp)


def _generation_record(output, with_prompt_logprobs: bool = False) -> dict:
    import torch

    comp = output.outputs[0]
    rec = {
        "prompt_token_ids": torch.tensor(list(output.prompt_token_ids), dtype=torch.int64),
        "token_ids": torch.tensor(list(comp.token_ids), dtype=torch.int64),
    }
    steps = getattr(comp, "logprobs", None) or []
    vals = []
    for step, tid in zip(steps, comp.token_ids):
        try:
            vals.append(float(step[tid].logprob))
        except Exception:  # noqa: BLE001
            break
    if len(vals) == len(comp.token_ids):
        rec["token_logprobs"] = torch.tensor(vals, dtype=torch.float64)
    if with_prompt_logprobs:
        pl = getattr(output, "prompt_logprobs", None)
        ids = list(output.prompt_token_ids)
        lp = [float("nan")]                      # position 0 has no logprob
        for i in range(1, len(ids)):
            try:
                lp.append(float(pl[i][ids[i]].logprob))
            except Exception:  # noqa: BLE001
                lp.append(float("nan"))
        rec["prompt_logprobs"] = torch.tensor(lp, dtype=torch.float64)
    return rec


def _git_state(root: str) -> dict:
    """``{"sha", "dirty"}`` of the checkout ``mia`` was imported from (provenance only)."""
    out: dict = {}
    try:
        out["sha"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10).stdout.strip() or None
        st = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "mia"],
                            capture_output=True, text=True, timeout=10)
        out["dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except Exception:  # noqa: BLE001 -- provenance must never fail a run
        pass
    return out


def _match_req(art: dict, request_id: str):
    if request_id in art:
        return art[request_id]
    for key in art:
        if str(key).startswith(f"{request_id}-"):
            return art[key]
    raise RuntimeError(f"request {request_id!r} has no rows in the capture (keys "
                       f"{sorted(map(str, art))[:6]})")


def run_child(args) -> int:
    """One engine, one workload. Writes ``<out>/<wl>/{child_manifest.json, capture.pt|
    generation.pt}``. Raises (nonzero exit) on anything that would make the artifact
    meaningless."""
    wl = args.child_workload
    out = Path(args.out) / wl
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import mia
    from vllm import LLM, SamplingParams

    mia_file = os.path.abspath(mia.__file__)
    if args.require_mia_under:
        root = os.path.abspath(args.require_mia_under)
        if not (mia_file.startswith(root.rstrip("/") + "/")
                or os.path.realpath(mia_file).startswith(os.path.realpath(root).rstrip("/") + "/")):
            raise RuntimeError(f"import mia resolved to {mia_file}, not under {root}")

    compilation_config: dict = {"cudagraph_mode": "FULL"}
    requested_fuse = FUSE_REQUEST[args.fuse]
    if requested_fuse != "vllm-default":
        compilation_config["pass_config"] = {"fuse_allreduce_rms": requested_fuse}

    llm = LLM(
        model=args.model,
        tensor_parallel_size=int(args.tp),
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_num_batched_tokens=int(args.max_num_batched_tokens),
        enable_prefix_caching=False,
        enforce_eager=False,
        seed=0,
        compilation_config=compilation_config,
        worker_extension_cls=WORKER_CLS[wl],
    )
    vcfg = getattr(llm.llm_engine, "vllm_config", None)
    pc = getattr(vcfg, "parallel_config", None)
    cc = getattr(vcfg, "compilation_config", None)
    resolved = {
        "tensor_parallel_size": getattr(pc, "tensor_parallel_size", None),
        "pipeline_parallel_size": getattr(pc, "pipeline_parallel_size", None),
        "fuse_allreduce_rms": getattr(getattr(cc, "pass_config", None), "fuse_allreduce_rms", None),
        "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
        "use_v2_model_runner": getattr(vcfg, "use_v2_model_runner", None),
        "max_num_batched_tokens": getattr(getattr(vcfg, "scheduler_config", None),
                                          "max_num_batched_tokens", None),
    }
    if resolved["tensor_parallel_size"] != int(args.tp):
        raise RuntimeError(f"engine resolved tensor_parallel_size="
                           f"{resolved['tensor_parallel_size']}, asked {args.tp}")
    if resolved["use_v2_model_runner"] is False:
        raise RuntimeError("engine resolved to the V1 model runner; the probe is V2-only")

    tc = llm.llm_engine.model_config.hf_text_config
    n_layers = int(tc.num_hidden_layers)
    h_q = int(tc.num_attention_heads)
    h_kv = int(getattr(tc, "num_key_value_heads", None) or h_q)
    hidden = int(tc.hidden_size)
    head_dim = int(getattr(tc, "head_dim", None) or hidden // h_q)
    model_info = {"id": args.model, "num_hidden_layers": n_layers, "hidden_size": hidden,
                  "num_attention_heads": h_q, "num_key_value_heads": h_kv, "head_dim": head_dim}
    # Worker-side fusion evidence: one entry per TP rank (see _fusion_pass_state).
    try:
        fusion_pass = llm.collective_rpc(_fusion_pass_state)
    except Exception as e:  # noqa: BLE001 -- recorded; --compare fails a fused leg without it
        fusion_pass = [{"error": f"collective_rpc failed: {type(e).__name__}: {e}"}]
    print(f"[tp-probe] fusion evidence tp={args.tp}: armed={_fusion_active(fusion_pass)!r} "
          f"applied={_fusion_applied(fusion_pass)['last_call_by_rank']} (last pass call per rank); "
          + "; ".join(f"rank {r.get('rank')}: configured pass managers "
                      f"{r.get('backends_configured')}, passes {r.get('passes')}"
                      + (f", error {r['error']}" if r.get("error") else "")
                      for r in fusion_pass if isinstance(r, dict)), flush=True)
    fusion_env: dict = {"nvcc": shutil.which("nvcc")}
    try:
        from vllm.utils import flashinfer as _fi
        fusion_env["has_flashinfer"] = bool(_fi.has_flashinfer())
        fusion_env["flashinfer_cubin"] = bool(_fi.has_flashinfer_cubin())
    except Exception as e:  # noqa: BLE001
        fusion_env["error"] = repr(e)
    try:
        from vllm.engine.arg_utils import EngineArgs
        mia_plugin_loaded = (getattr(EngineArgs.create_engine_config, "__name__", "")
                             == "_patched_create_engine_config")
    except Exception:  # noqa: BLE001
        mia_plugin_loaded = None
    manifest = {"workload": wl, "tp": int(args.tp), "mia": mia_file,
                "mia_git": _git_state(os.path.dirname(os.path.dirname(mia_file))),
                "model": model_info,
                "resolved": resolved,
                "requested_fuse_allreduce_rms": requested_fuse,
                "fusion_pass": fusion_pass,
                "fusion_active": _fusion_active(fusion_pass),
                "fusion_applied": _fusion_applied(fusion_pass),
                "fusion_env": fusion_env,
                "mia_plugin_loaded": mia_plugin_loaded,
                "env": _recorded_env(),
                "aperture_dir": os.environ.get("MIA_APERTURE_DIR"),
                "prompts": list(PROMPTS), "max_tokens": int(args.max_tokens)}
    try:
        import vllm
        manifest["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass

    def sp(extra=None, prompt_logprobs=None):
        return SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(args.max_tokens),
                              ignore_eos=True, logprobs=1, prompt_logprobs=prompt_logprobs,
                              extra_args=extra)

    if wl in ("hs", "qk", "hs_replicas"):
        if wl in HS_WORKLOADS:
            extra = {"output_hidden_states": list(range(1, n_layers + 1)),
                     "hs_mode": "all_tokens", "hooks_on": "both"}
        else:
            extra = {"output_qk": list(range(n_layers)),
                     "hookq_mode": "all_tokens", "hooks_on": "both"}
        outputs = llm.generate(list(PROMPTS), sp(extra), use_tqdm=False)
        dirs = [d for d in llm.collective_rpc("flush_aperture") if d]
        manifest["flush_dirs"] = dirs
        from mia.graph.aperture_reader import (
            load_hs_aperture_tp, load_multilayer_aperture_artifact,
            load_multilayer_qk_aperture_artifact, merge_qk_aperture_ranks, read_sidecar_header)
        sidecar = "hs_aperture_meta.jsonl" if wl in HS_WORKLOADS else "qk_aperture_meta.jsonl"
        manifest["rank_headers"] = {d: read_sidecar_header(os.path.join(d, sidecar))
                                    for d in dirs}
        if not dirs:
            raise RuntimeError("flush_aperture returned no dir holding data: nothing captured")
        replicas = None
        if wl == "hs":
            # MIA's own reader: at TP>1 under the layer shard it UNIONS every rank's dir (refusing
            # a gap or a duplicate); at TP=1 / MIA_HS_TP_SHARD=0 it reads tp_rank_0. The compare
            # side then judges the MERGED full-layer capture against TP=1.
            try:
                art = load_hs_aperture_tp(os.environ["MIA_APERTURE_DIR"])
            except Exception as e:  # noqa: BLE001 -- recorded; compare fails on it
                manifest["merge_error"] = f"{type(e).__name__}: {e}"
                art = load_multilayer_aperture_artifact(dirs[0])
        elif wl == "hs_replicas":
            # Every rank captured every layer: keep each rank's copy for the bitwise check, judge
            # rank 0's against TP=1, and record MIA's own replica check.
            ranked = sorted((int(h.get("tp_rank", 0)), d)
                            for d, h in manifest["rank_headers"].items())
            per_rank = {r: load_multilayer_aperture_artifact(d) for r, d in ranked}
            try:
                load_hs_aperture_tp(os.environ["MIA_APERTURE_DIR"], check_replicas=True)
                manifest["replica_check"] = "ok"
            except Exception as e:  # noqa: BLE001 -- recorded; compare fails on it
                manifest["replica_check"] = f"{type(e).__name__}: {e}"
            art = per_rank.get(0) or per_rank[ranked[0][0]]
            replicas = {r: [] for r in per_rank}
        else:
            try:
                art = merge_qk_aperture_ranks(dirs)
            except Exception as e:  # noqa: BLE001 -- recorded; compare fails on it
                manifest["merge_error"] = f"{type(e).__name__}: {e}"
                art = load_multilayer_qk_aperture_artifact(dirs[0])
        requests = []
        for output in outputs:
            rec = _generation_record(output)
            per_layer = _match_req(art, output.request_id)
            if wl in HS_WORKLOADS:
                rec["layers"] = {int(L): t.contiguous() for L, t in per_layer.items()}
            else:
                rec["layers"] = {int(L): {"q": e["q"].contiguous(),
                                          "k_full": e["k_full"].contiguous(),
                                          "k_prefix_ends": torch.tensor(
                                              [int(x) for x in e["k_prefix_ends"]],
                                              dtype=torch.int64)}
                                 for L, e in per_layer.items()}
            requests.append(rec)
            if replicas is not None:
                for r, a in per_rank.items():
                    try:
                        mine = _match_req(a, output.request_id)
                    except RuntimeError:
                        mine = {}
                    replicas[r].append({int(L): t.contiguous() for L, t in mine.items()})
        torch.save({"workload": wl, "requests": requests}, out / CAPTURE_FILE)
        if replicas is not None:
            torch.save({"workload": wl, "ranks": replicas}, out / REPLICAS_FILE)
    else:
        g = torch.Generator().manual_seed(1234)
        vec = torch.randn(hidden, generator=g, dtype=torch.float32)
        vec = vec / vec.norm()
        vec_path = out / "steer_vector.pt"
        torch.save({"dir": vec}, vec_path)
        cfg = {"method": "add_vector", "coefficient": float(args.steer_coefficient),
               "optimal_layer": "all", "vector_path": str(vec_path),
               "phase": "both", "positions": "all_tokens"}
        manifest["steer"] = cfg
        steered = llm.generate(list(PROMPTS), sp({"steer": cfg}, prompt_logprobs=1),
                               use_tqdm=False)
        plain = llm.generate(list(PROMPTS), sp(None, prompt_logprobs=1), use_tqdm=False)
        torch.save({"workload": "steer",
                    "steered": [_generation_record(o, True) for o in steered],
                    "unsteered": [_generation_record(o, True) for o in plain]},
                   out / STEER_FILE)
    # Each rank's save mode AFTER the workload ran (a writer that died mid-run shows here).
    try:
        manifest["writer"] = llm.collective_rpc(_writer_state)
    except Exception as e:  # noqa: BLE001 -- recorded; --compare fails a capture leg without it
        manifest["writer"] = [{"error": f"collective_rpc failed: {type(e).__name__}: {e}"}]
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / CHILD_MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[tp-probe] child {wl} tp={args.tp} wrote {out}", flush=True)
    return 0


# ======================================================================================
# COMPARE side (GPU-free)
# ======================================================================================

class _Verdict:
    def __init__(self, bands: dict):
        self.checks: list = []
        self.failures: list = []
        self.bands = bands
        self.measured: dict = {}

    def add(self, workload: str, name: str, ok, detail: str = "", value=None, band=None):
        status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        c = {"workload": workload, "name": name, "status": status, "detail": detail}
        if value is not None:
            c["value"] = value
        if band is not None:
            c["band"] = band
        self.checks.append(c)
        if status == "FAIL":
            self.failures.append(f"[{workload}] {name}: {detail}")
        return ok


def _load_child(run_dir: Path, wl: str):
    m = run_dir / wl / CHILD_MANIFEST
    if not m.exists():
        return None, None
    manifest = json.loads(m.read_text())
    import torch
    fname = STEER_FILE if wl == "steer" else CAPTURE_FILE
    p = run_dir / wl / fname
    data = torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None
    rp = run_dir / wl / REPLICAS_FILE
    if wl == "hs_replicas" and isinstance(data, dict):
        data["ranks"] = (torch.load(rp, map_location="cpu", weights_only=False).get("ranks")
                         if rp.exists() else None)
    return manifest, data


def _run_tp(run_dir: Path):
    """The TP degree of a run dir: its parent manifest, else any child manifest's."""
    top = run_dir / "manifest.json"
    if top.exists():
        try:
            return int(json.loads(top.read_text()).get("tp") or 1)
        except (ValueError, TypeError):
            pass
    for wl in WORKLOADS:
        m = run_dir / wl / CHILD_MANIFEST
        if m.exists():
            return int(json.loads(m.read_text()).get("tp") or 1)
    return None


# --------------------------------------------------------------------------------------
# The HS layout a leg asked for (MIA's mia.graph.tp_shard, re-derived here: the compare side
# never imports mia). HS shards by LAYER at TP > 1 unless MIA_HS_TP_SHARD=0 (rank 0 captures every
# layer) or MIA_HS_CAPTURE_ALL_RANKS=1 (every rank every layer: the hs_replicas leg).
# --------------------------------------------------------------------------------------

def _leg_of(wl: str, m) -> str:
    """The workload a manifest actually ran (``hs`` for the TP=1 reference of ``hs_replicas``)."""
    return (m.get("workload") if isinstance(m, dict) else None) or wl


def _hs_layout(wl: str, m: dict) -> str:
    """``single`` (TP=1) | ``round_robin`` (the default at TP>1) | ``rank0`` | ``all_ranks``, from
    the leg's TP and its RECORDED env (what the child actually saw)."""
    tp = int(m.get("tp") or 1)
    env = m.get("env") or {}
    if tp <= 1:
        return "single"
    if wl == "hs_replicas" or str(env.get(HS_ALL_RANKS_ENV)) == "1":
        return "all_ranks"
    return "rank0" if str(env.get(HS_SHARD_ENV) or "").strip() == "0" else "round_robin"


def _hs_owned(num_layers: int, tp: int, rank: int) -> list:
    """1-based layers ``rank`` captures under the round-robin layer shard: ``(L-1) % tp == rank``."""
    return [i + 1 for i in range(int(num_layers)) if i % int(tp) == int(rank)]


def _hs_capturing_ranks(wl: str, m: dict) -> list:
    """The ranks that hold HS data in this leg's layout."""
    tp = int(m.get("tp") or 1)
    layout = _hs_layout(wl, m)
    if layout in ("single", "rank0"):
        return [0]
    if layout == "all_ranks":
        return list(range(tp))
    n = int((m.get("model") or {}).get("num_hidden_layers") or tp)
    return [r for r in range(tp) if _hs_owned(n, tp, r)]


def _common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    return n


def _rel(a, b) -> float:
    a, b = a.double(), b.double()
    den = float(b.norm())
    num = float((a - b).norm())
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


def _row_rel_max(a, b) -> float:
    a, b = a.double(), b.double()
    num = (a - b).norm(dim=-1)
    den = b.norm(dim=-1)
    worst = 0.0
    for n, d in zip(num.tolist(), den.tolist()):
        r = (0.0 if n == 0.0 else math.inf) if d == 0.0 else n / d
        worst = max(worst, r)
    return worst


def _head_rel_max(a, b, head_dim: int) -> float:
    worst = 0.0
    for h in range(a.shape[-1] // head_dim):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        worst = max(worst, _rel(a[..., sl], b[..., sl]))
    return worst


def _meta_summary(run_dir: Path, manifests: dict) -> dict:
    top = run_dir / "manifest.json"
    t = json.loads(top.read_text()) if top.exists() else {}
    any_m = next((m for m in manifests.values() if m), {}) or {}
    armed = {wl: (_fusion_armed(m) if m else None) for wl, m in manifests.items()}
    return {"dir": str(run_dir), "tp": any_m.get("tp", t.get("tp")),
            "fuse_allreduce_rms": (any_m.get("resolved") or {}).get("fuse_allreduce_rms"),
            "requested_fuse_allreduce_rms": any_m.get("requested_fuse_allreduce_rms"),
            # ARMED (recomputed from the per-rank records); "fusion_active" is its older name
            "fusion_active": armed,
            "fusion_armed": armed,
            "fusion_applied": {wl: _fusion_applied((m or {}).get("fusion_pass"))
                               for wl, m in manifests.items()},
            "fusion_env": any_m.get("fusion_env"),
            "mia": any_m.get("mia"), "mia_git": any_m.get("mia_git"),
            "model": any_m.get("model")}


# --------------------------------------------------------------------------------------
# Run hygiene: the child's environment, the aperture dirs left on disk, the fusion request
# --------------------------------------------------------------------------------------

def _check_child_env(v, wl, side, m) -> None:
    """The child ran with MIA's plugin loaded and without the env the probe drops (DROPPED_ENV).
    A manifest that does not record it FAILS: an unverifiable run is not a passing one."""
    env = m.get("env")
    if not isinstance(env, dict):
        v.add(wl, f"{side} child env", False, "not recorded (a pre-W2 probe run): re-run the leg")
        return
    bad = {k: env.get(k) for k in DROPPED_ENV if env.get(k) is not None}
    v.add(wl, f"{side} child env", not bad,
          "VLLM_PLUGINS and MIA_APERTURE_PER_REQUEST unset" if not bad else
          f"must be unset in the child, got {bad}")
    kind = _leg_of(wl, m)
    if kind in HS_WORKLOADS:
        want = "1" if kind == "hs_replicas" else None
        got = env.get(HS_ALL_RANKS_ENV)
        v.add(wl, f"{side} {HS_ALL_RANKS_ENV}", got == want,
              f"{got!r}; " + ("the replication leg sets it to '1'" if want else
                              "the hs leg must not inherit it (it would capture replicas)"))
    loaded = m.get("mia_plugin_loaded")
    v.add(wl, f"{side} MIA plugin loaded", loaded is True,
          "EngineArgs.create_engine_config is MIA's patch" if loaded else
          f"MIA's vllm.general_plugins entry point did not patch vLLM (recorded {loaded!r})")
    probe_set = _PROBE_SET_ENV + ((HS_ALL_RANKS_ENV,) if kind == "hs_replicas" else ())
    inherited = {k: val for k, val in env.items()
                 if k.startswith("MIA_") and val is not None and k not in probe_set}
    v.add(wl, f"{side} inherited MIA_* env", None,
          ", ".join(f"{k}={val}" for k, val in sorted(inherited.items())) or "none")


#: ``worker._writer_mode`` values (mia/graph/writer_process.py).
WRITER_ON = "process"
WRITER_OFF = "in-process (MIA_WRITER_PROCESS=0)"
WRITER_HS_SINK = "none (HS sink rank: captures nothing)"


def _check_writer(v, wl, side, m) -> None:
    """Every rank runs the save mode the contract says: the out-of-process writer on each
    capturing rank (QK: every rank; HS: tp_rank 0), ``none`` on an HS sink rank, and the writer
    still alive after the workload. ``MIA_WRITER_PROCESS=0`` in the recorded env expects the
    in-process mode instead. A TP>1 regression to "failed to start, falling back to in-process
    save" (gate G1's state) FAILS here instead of hiding in child.log. Steer writes nothing: INFO."""
    ranks = m.get("writer")
    name = f"{side} writer process"
    kind = _leg_of(wl, m)
    if kind == "steer":
        v.add(wl, name, None, f"steer writes no artifact; recorded {ranks!r}")
        return
    if not isinstance(ranks, list) or not ranks:
        v.add(wl, name, False, "not recorded (a pre-W3 probe run): re-run the leg")
        return
    tp = int(m.get("tp") or 1)
    on = (WRITER_OFF if str((m.get("env") or {}).get("MIA_WRITER_PROCESS")) == "0"
          else WRITER_ON)
    capturing = set(_hs_capturing_ranks(kind, m)) if kind in HS_WORKLOADS else None
    bad, seen = [], []
    for i, r in enumerate(ranks):
        if not isinstance(r, dict) or r.get("error"):
            bad.append(f"rank {i}: no evidence ({(r or {}).get('error') if isinstance(r, dict) else r!r})")
            continue
        rank = r.get("rank") if r.get("rank") is not None else i
        want = WRITER_HS_SINK if (capturing is not None and rank not in capturing) else on
        mode = r.get("writer_mode")
        seen.append(f"rank {rank}: {mode}")
        if mode != want:
            bad.append(f"rank {rank}: {mode!r}, expected {want!r}")
        elif mode == WRITER_ON and r.get("alive") is not True:
            bad.append(f"rank {rank}: writer child not alive after the workload "
                       f"({r.get('alive_error') or r.get('alive')!r})")
    if len(ranks) != tp:
        bad.append(f"{len(ranks)} rank record(s) for tp={tp}")
    v.add(wl, name, not bad, "; ".join(seen) if not bad else "; ".join(bad))


def _expected_rank_dirs(wl: str, tp: int, m: dict | None = None) -> set:
    if wl in HS_WORKLOADS:
        # HS: every rank that owns a layer under the round-robin layer shard (the default at
        # TP>1); tp_rank_0 alone at TP=1 or with MIA_HS_TP_SHARD=0; every rank for hs_replicas.
        return {f"tp_rank_{r}" for r in _hs_capturing_ranks(wl, dict(m or {}, tp=tp))}
    if wl == "qk":
        return {f"tp_rank_{r}" for r in range(tp)}  # every rank its own heads
    return set()                                    # steer writes nothing


def _rank_data_bytes(d: dict) -> int:
    """Bytes in a listed rank dir's DATA files, the per-layer ``*.raw`` files. The sidecar is not
    data: one holding only its header line makes ``total_bytes`` > 0 with nothing captured."""
    return sum(max(0, int(f.get("bytes", 0) or 0)) for f in (d.get("files") or [])
               if str(f.get("name", "")).endswith(".raw"))


def _check_aperture_listing(v, wl, side, run_dir: Path, tp: int, m: dict | None = None) -> None:
    """What the run LEFT on disk under MIA_APERTURE_DIR (listed by the parent after the child
    exited), not only what flush_aperture returned: exactly the expected ``tp_rank_*`` dirs, each
    holding DATA (``*.raw`` bytes, not a header-only sidecar). A stray rank dir -- an empty one
    especially, which flush_aperture filters out and so never shows in ``flush_dirs`` -- FAILS
    here."""
    kind = _leg_of(wl, m)
    f = run_dir / kind / APERTURE_LISTING
    name = f"{side} aperture dirs on disk"
    if not f.exists():
        v.add(wl, name, False, f"no {APERTURE_LISTING}: the stray-rank-dir check cannot run "
                               f"(a pre-W2 probe run, or the parent died): re-run the leg")
        return
    listing = json.loads(f.read_text())
    dirs = listing.get("rank_dirs") or {}
    got = {k: int(d.get("total_bytes", 0)) for k, d in dirs.items()}
    data = {k: _rank_data_bytes(d) for k, d in dirs.items()}
    want = _expected_rank_dirs(kind, tp, m)
    stray = {k: b for k, b in got.items() if k not in want}
    missing = sorted(want - set(got))
    empty = sorted(k for k in want & set(got) if data[k] <= 0)
    ok = not stray and not missing and not empty
    desc = (", ".join(f"{k} {b} B ({data[k]} B data)" for k, b in sorted(got.items()))
            or "no tp_rank_* dir")
    why = []
    if stray:
        why.append("STRAY " + ", ".join(f"{k} ({'EMPTY' if b <= 0 else f'{b} B'})"
                                        for k, b in sorted(stray.items())))
    if missing:
        why.append(f"missing {missing}")
    if empty:
        why.append(f"expected dirs holding no data {empty} (no *.raw bytes)")
    v.add(wl, name, ok, f"{desc}; expected {sorted(want) or 'none'}"
          + ("" if ok else " -- " + "; ".join(why)))
    if listing.get("other"):
        v.add(wl, f"{side} aperture root: other entries", None,
              ", ".join(e["name"] for e in listing["other"]))


#: What the verdict says about APPLIED for each workload (INFO; never a FAIL, see _fusion_applied).
_APPLIED_EXPECTATION = {
    "hs": ("PARTIAL fusion is EXPECTED and never failed: mia::capture_hs is a second consumer of "
           "each hooked layer's down_proj all-reduce output, so that all-reduce cannot fold into "
           "the next RMSNorm; the o_proj ones can"),
    "hs_replicas": ("as hs: mia::capture_hs is baked on every layer of every rank in every HS "
                    "layout, so fusion is as partial as in the hs leg"),
    "qk": ("q/k are read inside attention, before any all-reduce: capture_qk is not expected to "
           "block the pattern"),
    "steer": ("steer_buffer mutates each layer's residual in place between the down_proj "
              "all-reduce and the next fused add + RMSNorm, which may keep that pair from "
              "folding (what diagnostic D3 measures)"),
}


def _check_fusion(v, wl, ref_m, cand_m, repeat: bool = False) -> None:
    """The CANDIDATE must run with the fusion the leg asked for (``_check_side_fusion``). In
    parity mode the reference is TP=1 and exempt (vLLM never fuses at TP=1); in ``--repeat`` mode
    the reference is a TP=N leg of the same configuration and is held to the same rule."""
    if repeat:
        _check_side_fusion(v, wl, "ref", ref_m)
    else:
        v.add(wl, "ref fusion (TP=1: exempt)", None,
              f"requested {ref_m.get('requested_fuse_allreduce_rms')!r}, resolved "
              f"{(ref_m.get('resolved') or {}).get('fuse_allreduce_rms')!r}; "
              "vLLM never fuses at TP=1")
    _check_side_fusion(v, wl, "cand", cand_m)


def _check_side_fusion(v, wl, side, m) -> None:
    """One side ran with the fusion its leg asked for, judged on its PER-RANK evidence
    (``_fusion_armed``, recomputed here, never the child's stored verdict):

      * ``--fuse on`` at TP>1: FAIL ("fusion not active") unless vLLM resolved
        ``fuse_allreduce_rms=True`` AND the pass is ARMED on every rank -- no evidence FAILS too,
        naming what is missing per rank; plus an INFO ``fusion applied`` line (never asserted).
      * ``--fuse off``: FAIL if fusion resolved or is armed anyway, or if no per-rank evidence
        shows it unarmed (the resolved flag alone is not proof, and an empty pass list from a
        scan that could see nothing used to pass here vacuously).
      * ``default`` asks nothing: reported, not asserted.

    TP=1 is exempt: vLLM never fuses at TP=1."""
    tp = int(m.get("tp") or 1)
    req = m.get("requested_fuse_allreduce_rms")
    res = (m.get("resolved") or {}).get("fuse_allreduce_rms")
    if tp <= 1:
        v.add(wl, f"{side} fusion (TP=1: exempt)", None,
              f"requested {req!r}, resolved {res!r}; vLLM never fuses at TP=1")
        return
    fp = m.get("fusion_pass")
    armed = _fusion_armed(m)
    applied = _fusion_applied(fp)
    unarmed = [r.get("rank") for r in (fp if isinstance(fp, list) else [])
               if _rank_has_pass_evidence(r)
               and not any(not p.get("disabled", True) for p in r["passes"] if isinstance(p, dict))]
    gaps = _fusion_evidence_gaps(fp, tp) if armed is None else []
    evidence = (f"resolved fuse_allreduce_rms={res!r}; ARMED on every rank: {armed!r}"
                + (f"; unarmed on rank(s) {unarmed}" if unarmed else "")
                + (f"; no evidence: {' | '.join(gaps)}" if gaps else "")
                + f"; APPLIED (last pass call per rank): {applied['last_call_by_rank']}"
                + f"; env {m.get('fusion_env')}")
    if req is True:
        if res is not True:
            ok, why = False, f"fusion not active: requested on, vLLM resolved {res!r}"
        elif armed is None:
            ok, why = False, ("fusion not active: the config says True but there is no per-rank "
                              "evidence that the pass is armed")
        elif armed is not True:
            ok, why = False, ("fusion not active: the config says True but the pass is not armed "
                              "on every rank")
        else:
            ok, why = True, "fusion active on every rank (ARMED)"
        v.add(wl, f"{side} fusion active (requested on)", ok, f"{why} ({evidence})", value=res)
        v.add(wl, f"{side} fusion applied (requested on)", None,
              f"{applied['status']}: last pass call per rank {applied['last_call_by_rank']} "
              f"(a per-rank total is {FUSION_TOTAL_UNAVAILABLE}). {_APPLIED_EXPECTATION[wl]}.",
              value=applied["status"])
    elif req is False:
        if res is not False:
            ok, why = False, f"fusion requested OFF but vLLM resolved {res!r}"
        elif armed is True:
            ok, why = False, "fusion requested OFF but the pass is armed anyway"
        elif armed is None:
            ok, why = False, ("fusion requested OFF, resolved False, but there is no per-rank "
                              "evidence that the pass is not armed")
        else:
            ok, why = True, "off (no armed pass on any rank)"
        v.add(wl, f"{side} fusion off (requested off)", ok, f"{why} ({evidence})", value=res)
    else:
        v.add(wl, f"{side} fusion (vLLM default: not asserted)", None,
              f"requested {req!r} ({evidence})", value=res)


def _check_requests_align(v, wl, ref_reqs, cand_reqs) -> bool:
    if not v.add(wl, "request count", len(ref_reqs) == len(cand_reqs),
                 f"ref {len(ref_reqs)} cand {len(cand_reqs)}"):
        return False
    ok = True
    for i, (r, c) in enumerate(zip(ref_reqs, cand_reqs)):
        same = bool((r["prompt_token_ids"].shape == c["prompt_token_ids"].shape)
                    and bool((r["prompt_token_ids"] == c["prompt_token_ids"]).all()))
        ok &= bool(v.add(wl, f"req{i} prompt token ids", same,
                         "identical" if same else "DIFFER -- not the same inputs"))
    return ok


def _check_hs_layout(v, wl, side, m) -> None:
    """The rank dirs a side's flush returned and the layer shard their headers declare are the
    ones its layout requires: round-robin (the default at TP>1) -> every owning rank, each header
    ``layer_shard=round_robin`` with ``owned_layers`` = the rule; ``MIA_HS_TP_SHARD=0`` / TP=1 ->
    ``tp_rank_0`` only, no shard fields; hs_replicas -> every rank, ``capture_all_ranks``. A merge
    error recorded by the child (MIA's reader refused a gap / duplicate) FAILS."""
    tp = int(m.get("tp") or 1)
    layout = _hs_layout(wl, m)
    n = int((m.get("model") or {}).get("num_hidden_layers") or 0)
    names = sorted(os.path.basename(os.path.normpath(d)) for d in (m.get("flush_dirs") or []))
    want = sorted(_expected_rank_dirs(wl, tp, m))
    if tp > 1 or side == "cand":
        v.add(wl, f"{side} rank dirs", names == want,
              f"flush_aperture returned {names}; expected {want} (layout {layout}: "
              + {"round_robin": "each rank captures its round-robin share of the layers",
                 "rank0": "MIA_HS_TP_SHARD=0, tp_rank 0 captures every layer",
                 "all_ranks": "every rank captures every layer (replicas)",
                 "single": "TP=1"}[layout] + ")")
    headers = m.get("rank_headers")
    if isinstance(headers, dict) and headers:
        bad = []
        for d, h in sorted(headers.items()):
            r = int(h.get("tp_rank", -1))
            if layout == "round_robin":
                if (h.get("layer_shard") != "round_robin"
                        or list(h.get("owned_layers") or []) != _hs_owned(n, tp, r)):
                    bad.append(f"tp_rank {r}: layer_shard={h.get('layer_shard')!r} "
                               f"owned_layers={h.get('owned_layers')!r}, want round_robin "
                               f"{_hs_owned(n, tp, r)}")
            elif "owned_layers" in h or "layer_shard" in h:
                bad.append(f"tp_rank {r}: declares a layer shard in layout {layout}")
            if layout == "all_ranks" and tp > 1 and h.get("capture_all_ranks") is not True:
                bad.append(f"tp_rank {r}: capture_all_ranks={h.get('capture_all_ranks')!r}")
        v.add(wl, f"{side} HS layout headers", not bad,
              f"{len(headers)} header(s) match layout {layout}" if not bad else "; ".join(bad))
    elif tp > 1:
        v.add(wl, f"{side} HS layout headers", False, "no rank headers recorded")
    if wl == "hs":
        err = m.get("merge_error")
        v.add(wl, f"{side} TP merge", err is None,
              "merged" if err is None else f"load_hs_aperture_tp raised: {err}")


def _compare_hs(v, ref_m, ref, cand_m, cand, wl="hs"):
    model = ref_m["model"]
    L, hidden = int(model["num_hidden_layers"]), int(model["hidden_size"])
    want_layers = set(range(1, L + 1))
    for side, m in (("ref", ref_m), ("cand", cand_m)):
        if side == "ref" and ref_m.get("workload") != wl:
            continue                          # hs_replicas vs a TP=1 ref's "hs" leg
        _check_hs_layout(v, wl, side, m)
    if not _check_requests_align(v, wl, ref["requests"], cand["requests"]):
        return
    worst = {"rel": 0.0, "row": 0.0, "at": None, "rows": 0}
    for i, (r, c) in enumerate(zip(ref["requests"], cand["requests"])):
        for side, rec in (("ref", r), ("cand", c)):
            got = set(rec["layers"])
            v.add(wl, f"req{i} {side} layer set",
                  got == want_layers,
                  f"{len(got)} layers captured, want all {L}"
                  + ("" if got == want_layers else
                     f" (missing {sorted(want_layers - got)[:8]}, extra {sorted(got - want_layers)[:8]})"))
        n_p = int(r["prompt_token_ids"].numel())
        d = _common_prefix(r["token_ids"], c["token_ids"])
        for side, rec in (("ref", r), ("cand", c)):
            want_rows = n_p + int(rec["token_ids"].numel()) - 1
            bad = [L_ for L_, t in rec["layers"].items()
                   if tuple(t.shape) != (want_rows, hidden)]
            v.add(wl, f"req{i} {side} shapes", not bad,
                  f"every layer ({want_rows}, {hidden})" if not bad else
                  f"layers {sorted(bad)[:6]} have shape "
                  f"{tuple(rec['layers'][bad[0]].shape)}, want ({want_rows}, {hidden})")
        n_rows = n_p + min(d, int(r["token_ids"].numel()) - 1)
        v.add(wl, f"req{i} rows compared", None,
              f"{n_rows} (prompt {n_p} + common generated prefix {d})", value=n_rows)
        worst["rows"] += n_rows
        for L_ in sorted(want_layers & set(r["layers"]) & set(c["layers"])):
            a, b = c["layers"][L_], r["layers"][L_]
            if a.shape[-1] != hidden or b.shape[-1] != hidden:
                continue
            a, b = a[:n_rows], b[:n_rows]
            if a.shape[0] != n_rows or b.shape[0] != n_rows:
                continue
            rel, row = _rel(a, b), _row_rel_max(a, b)
            if rel > worst["rel"]:
                worst["rel"], worst["at"] = rel, f"req{i} layer {L_}"
            worst["row"] = max(worst["row"], row)
            if rel > v.bands["hs_rel"]:
                v.add(wl, f"req{i} layer {L_} rel", False, f"{rel:.3e} > band", rel,
                      v.bands["hs_rel"])
            if row > v.bands["hs_row"]:
                v.add(wl, f"req{i} layer {L_} worst row", False, f"{row:.3e} > band", row,
                      v.bands["hs_row"])
    v.add(wl, "worst tensor-global rel", worst["rel"] <= v.bands["hs_rel"],
          f"{worst['rel']:.3e} at {worst['at']}", worst["rel"], v.bands["hs_rel"])
    v.add(wl, "worst per-row rel", worst["row"] <= v.bands["hs_row"],
          f"{worst['row']:.3e}", worst["row"], v.bands["hs_row"])
    v.measured[wl] = {"worst_rel": worst["rel"], "worst_row_rel": worst["row"],
                      "worst_at": worst["at"], "rows_compared": worst["rows"]}


def _compare_qk(v, ref_m, ref, cand_m, cand):
    wl = "qk"
    model = ref_m["model"]
    L = int(model["num_hidden_layers"])
    d = int(model["head_dim"])
    q_w, k_w = int(model["num_attention_heads"]) * d, int(model["num_key_value_heads"]) * d
    want_layers = set(range(L))
    tp = int(cand_m.get("tp") or 1)
    for side, m in (("ref", ref_m), ("cand", cand_m)):
        err = m.get("merge_error")
        v.add(wl, f"{side} TP merge", err is None,
              "merged" if err is None else f"merge_qk_aperture_ranks raised: {err}")
    names = sorted(os.path.basename(os.path.normpath(x)) for x in (cand_m.get("flush_dirs") or []))
    want_names = sorted(f"tp_rank_{r}" for r in range(tp))
    v.add(wl, "cand rank dirs", names == want_names,
          f"flush_aperture returned {names}; expected {want_names} (every TP rank captures "
          f"its own heads)")
    if not _check_requests_align(v, wl, ref["requests"], cand["requests"]):
        return
    worst = {"q": 0.0, "k": 0.0, "head": 0.0, "at": None, "rows": 0}
    for i, (r, c) in enumerate(zip(ref["requests"], cand["requests"])):
        n_p = int(r["prompt_token_ids"].numel())
        for side, rec in (("ref", r), ("cand", c)):
            got = set(rec["layers"])
            v.add(wl, f"req{i} {side} layer set", got == want_layers,
                  f"{len(got)} layers captured, want all {L}"
                  + ("" if got == want_layers else
                     f" (missing {sorted(want_layers - got)[:8]})"))
            n_gen = int(rec["token_ids"].numel())
            want_rows = n_p + n_gen - 1
            ends = [n_p + j for j in range(n_gen)]
            bad = []
            for L_, e in rec["layers"].items():
                if (tuple(e["q"].shape) != (want_rows, q_w)
                        or tuple(e["k_full"].shape) != (ends[-1], k_w)
                        or [int(x) for x in e["k_prefix_ends"]] != ends):
                    bad.append(L_)
            v.add(wl, f"req{i} {side} full-width shapes", not bad,
                  f"q ({want_rows}, {q_w}) k_full ({ends[-1]}, {k_w})" if not bad else
                  f"layers {sorted(bad)[:6]}: q {tuple(rec['layers'][bad[0]]['q'].shape)} "
                  f"k_full {tuple(rec['layers'][bad[0]]['k_full'].shape)}; want q "
                  f"({want_rows}, {q_w}) k_full ({ends[-1]}, {k_w}) -- a narrower width is a "
                  f"partial-rank capture")
        dcom = _common_prefix(r["token_ids"], c["token_ids"])
        n_rows = n_p + min(dcom, int(r["token_ids"].numel()) - 1)
        v.add(wl, f"req{i} rows compared", None,
              f"{n_rows} (prompt {n_p} + common generated prefix {dcom})", value=n_rows)
        worst["rows"] += n_rows
        for L_ in sorted(want_layers & set(r["layers"]) & set(c["layers"])):
            for key, width in (("q", q_w), ("k_full", k_w)):
                a, b = c["layers"][L_][key], r["layers"][L_][key]
                if a.shape[-1] != width or b.shape[-1] != width:
                    continue
                a, b = a[:n_rows], b[:n_rows]
                if a.shape[0] != n_rows or b.shape[0] != n_rows:
                    continue
                rel, head = _rel(a, b), _head_rel_max(a, b, d)
                slot = "q" if key == "q" else "k"
                if rel > worst[slot]:
                    worst[slot] = rel
                    worst["at"] = f"req{i} layer {L_} {key}"
                worst["head"] = max(worst["head"], head)
                if rel > v.bands["qk_rel"]:
                    v.add(wl, f"req{i} layer {L_} {key} rel", False, f"{rel:.3e} > band", rel,
                          v.bands["qk_rel"])
                if head > v.bands["qk_head_rel"]:
                    v.add(wl, f"req{i} layer {L_} {key} worst head", False,
                          f"{head:.3e} > band", head, v.bands["qk_head_rel"])
    for slot, name in (("q", "worst q rel"), ("k", "worst k_full rel")):
        v.add(wl, name, worst[slot] <= v.bands["qk_rel"], f"{worst[slot]:.3e}", worst[slot],
              v.bands["qk_rel"])
    v.add(wl, "worst per-head rel", worst["head"] <= v.bands["qk_head_rel"],
          f"{worst['head']:.3e}", worst["head"], v.bands["qk_head_rel"])
    v.measured[wl] = {"worst_q_rel": worst["q"], "worst_k_rel": worst["k"],
                      "worst_head_rel": worst["head"], "worst_at": worst["at"],
                      "rows_compared": worst["rows"]}


def _compare_hs_replicas(v, ref_m, ref, cand_m, cand):
    """The REPLICATION leg (``MIA_HS_CAPTURE_ALL_RANKS=1``): every rank captured every layer of
    every request, and each rank's copy is BITWISE equal to rank 0's -- the fact the HS layer shard
    rests on (a rank's copy of a layer IS the layer). FATAL, no tolerance. MIA's own
    ``load_hs_aperture_tp(check_replicas=True)`` verdict, recorded by the child, must agree. Then
    rank 0's copy is judged against the reference like the ``hs`` leg (banded)."""
    import torch

    wl = "hs_replicas"
    tp = int(cand_m.get("tp") or 1)
    ranks = cand.get("ranks") if isinstance(cand, dict) else None
    if not isinstance(ranks, dict) or not ranks:
        v.add(wl, "cand replicas present", False, f"no {REPLICAS_FILE}: re-run the leg")
    else:
        got_ranks = sorted(int(r) for r in ranks)
        ok_ranks = v.add(wl, "cand replica ranks", got_ranks == list(range(tp)),
                         f"ranks {got_ranks}; expected 0..{tp - 1}")
        if ok_ranks:
            L = int(cand_m["model"]["num_hidden_layers"])
            base = ranks[0] if 0 in ranks else ranks["0"]
            diffs, compared = [], 0
            for r in got_ranks[1:]:
                other = ranks[r] if r in ranks else ranks[str(r)]
                if len(other) != len(base):
                    diffs.append(f"rank {r}: {len(other)} requests vs rank 0's {len(base)}")
                    continue
                for i, (a, b) in enumerate(zip(other, base)):
                    if set(a) != set(b) or set(b) != set(range(1, L + 1)):
                        diffs.append(f"rank {r} req{i}: layers {len(a)} vs rank 0's {len(b)} "
                                     f"(want all {L})")
                        continue
                    for layer in sorted(b):
                        compared += 1
                        x, y = a[layer], b[layer]
                        if x.shape != y.shape or not torch.equal(x, y):
                            md = (float((x.double() - y.double()).abs().max())
                                  if x.shape == y.shape else float("nan"))
                            diffs.append(f"rank {r} req{i} layer {layer}: max|diff| {md:.3e}")
            v.add(wl, "replicas bitwise equal across ranks", not diffs,
                  f"{compared} (rank, request, layer) tensors equal to rank 0's" if not diffs
                  else "; ".join(diffs[:6]) + (f" (+{len(diffs) - 6} more)"
                                               if len(diffs) > 6 else ""))
    rc = cand_m.get("replica_check")
    v.add(wl, "cand MIA replica check (load_hs_aperture_tp check_replicas)", rc == "ok",
          f"{rc!r}")
    _compare_hs(v, ref_m, ref, cand_m, cand, wl=wl)


def _prompt_lp(recs):
    import torch
    parts = [rec["prompt_logprobs"][1:] for rec in recs]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)


def _compare_steer(v, ref_m, ref, cand_m, cand):
    import torch

    wl = "steer"
    for key in ("steered", "unsteered"):
        if not _check_requests_align(v, wl, ref[key], cand[key]):
            return
    for side, data in (("ref", ref), ("cand", cand)):
        ok = all("prompt_logprobs" in r for key in ("steered", "unsteered") for r in data[key])
        if not v.add(wl, f"{side} prompt logprobs present", ok, "teacher-forced channel"):
            return
    rs, ru = _prompt_lp(ref["steered"]), _prompt_lp(ref["unsteered"])
    cs, cu = _prompt_lp(cand["steered"]), _prompt_lp(cand["unsteered"])
    finite = torch.isfinite(rs) & torch.isfinite(ru) & torch.isfinite(cs) & torch.isfinite(cu)
    v.add(wl, "finite prompt logprob positions", bool(finite.any()),
          f"{int(finite.sum())} of {finite.numel()}")
    if not finite.any():
        return
    d_ref = (rs - ru)[finite]
    d_cand = (cs - cu)[finite]
    live_ref = float(d_ref.abs().max())
    live_cand = float(d_cand.abs().max())
    floor = v.bands["steer_liveness_floor"]
    v.add(wl, "ref steer alive (max|delta|)", live_ref > floor,
          f"{live_ref:.3e} vs floor {floor:.1e} -- an inert steer cannot 'agree'", live_ref,
          floor)
    v.add(wl, "cand steer alive (max|delta|)", live_cand > floor,
          f"{live_cand:.3e} vs floor {floor:.1e}", live_cand, floor)
    den = float(d_ref.norm())
    rel = float((d_cand - d_ref).norm()) / den if den > 0 else math.inf
    v.add(wl, "steering effect matches TP=1 (rel)", rel <= v.bands["steer_delta_rel"],
          f"||delta_N - delta_1|| / ||delta_1|| = {rel:.3e}", rel, v.bands["steer_delta_rel"])
    un = float((cu - ru)[finite].abs().max())
    v.add(wl, "unsteered prompt logprob max|diff| (TP noise)", None, f"{un:.3e}", un)
    for key in ("steered", "unsteered"):
        pre = [_common_prefix(a["token_ids"], b["token_ids"])
               for a, b in zip(ref[key], cand[key])]
        v.add(wl, f"{key} generated common prefix", None, f"{pre}", pre)
    v.measured[wl] = {"delta_rel": rel, "ref_max_abs_delta": live_ref,
                      "cand_max_abs_delta": live_cand, "unsteered_max_abs_logprob_diff": un}


def compare_runs(ref_dir, cand_dir, workloads=WORKLOADS, bands=None, repeat: bool = False) -> dict:
    """Compare a candidate run dir against ONE reference run dir. Returns the verdict dict (see
    module docstring; ``compare_legs`` adds the multi-reference fields). Pure CPU.

    Parity mode (default): the reference must be TP=1 and the candidate is any TP. ``repeat``: the
    candidate repeats the reference's configuration (same TP, same fusion request) -- a
    boot-to-boot noise measurement -- and at TP>1 both sides' fusion is judged."""
    bands = dict(DEFAULT_BANDS, **(bands or {}))
    v = _Verdict(bands)
    ref_dir, cand_dir = Path(ref_dir), Path(cand_dir)
    ref_ms, cand_ms = {}, {}
    for wl in workloads:
        ref_m, ref = _load_child(ref_dir, wl)
        cand_m, cand = _load_child(cand_dir, wl)
        if wl == "hs_replicas":
            cand_tp = _run_tp(cand_dir)
            if cand_m is None and cand_tp is not None and cand_tp <= 1:
                # One rank: nothing to replicate. The parent skips the leg at TP=1.
                v.add(wl, "replication leg (TP=1: skipped)", None,
                      f"cand tp={cand_tp} has one rank; the leg runs at TP>1 only")
                ref_ms[wl], cand_ms[wl] = ref_m, cand_m
                continue
            if ref_m is None and (_run_tp(ref_dir) or 1) <= 1:
                # A TP=1 reference has no replicas leg: rank 0's copy is judged against its hs.
                ref_m, ref = _load_child(ref_dir, "hs")
                v.add(wl, "ref data", None, "the TP=1 reference's hs leg (TP=1 skips hs_replicas)")
        ref_ms[wl], cand_ms[wl] = ref_m, cand_m
        present = ref_m is not None and ref is not None and cand_m is not None and cand is not None
        v.add(wl, "both runs present", present,
              f"ref {'ok' if ref_m and ref is not None else 'MISSING'}, "
              f"cand {'ok' if cand_m and cand is not None else 'MISSING'}")
        if not present:
            continue
        if repeat:
            rt, ct = int(ref_m.get("tp") or 1), int(cand_m.get("tp") or 1)
            rq = ref_m.get("requested_fuse_allreduce_rms")
            cq = cand_m.get("requested_fuse_allreduce_rms")
            v.add(wl, "cand repeats ref (same TP and fusion request)", rt == ct and rq == cq,
                  f"ref tp={rt} fuse={rq!r}; cand tp={ct} fuse={cq!r}")
        else:
            v.add(wl, "ref is TP=1", int(ref_m.get("tp") or 1) == 1, f"ref tp={ref_m.get('tp')}")
        same_model = ref_m.get("model") == cand_m.get("model")
        if not v.add(wl, "same model", same_model,
                     f"ref {ref_m.get('model')} cand {cand_m.get('model')}"):
            continue
        for side, run, m in (("ref", ref_dir, ref_m), ("cand", cand_dir, cand_m)):
            # Checks are labelled with the leg being judged; each side is held to ITS OWN leg's
            # contract (a TP=1 reference of hs_replicas is its hs leg: _leg_of).
            _check_child_env(v, wl, side, m)
            _check_aperture_listing(v, wl, side, run, int(m.get("tp") or 1), m)
            _check_writer(v, wl, side, m)
        _check_fusion(v, wl, ref_m, cand_m, repeat=repeat)
        if wl == "hs":
            _compare_hs(v, ref_m, ref, cand_m, cand)
        elif wl == "hs_replicas":
            _compare_hs_replicas(v, ref_m, ref, cand_m, cand)
        elif wl == "qk":
            _compare_qk(v, ref_m, ref, cand_m, cand)
        else:
            _compare_steer(v, ref_m, ref, cand_m, cand)
    return {
        "verdict": "FAIL" if v.failures else "PASS",
        "mode": "repeat" if repeat else "parity",
        "ref": _meta_summary(ref_dir, ref_ms),
        "cand": _meta_summary(cand_dir, cand_ms),
        "bands": bands,
        "bands_overridden": bands != DEFAULT_BANDS,
        "checks": v.checks,
        "measured": v.measured,
        "failures": v.failures,
    }


def _artifact_sha256(run_dir, wl: str) -> str | None:
    """sha256 of a run's saved artifact for one workload (``capture.pt`` / ``generation.pt``), or
    None. Equal digests = bit-identical captures (``torch.save`` of equal tensors is byte-stable
    here: G1's TP=1 repeats matched md5 for md5); unequal digests are then read with the values."""
    p = Path(run_dir) / wl / (STEER_FILE if wl == "steer" else CAPTURE_FILE)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ref_labels(ref_dirs) -> list:
    """One label per reference: its dir name (``tp1``, ``tp1_repeat``), made unique with ``#i``."""
    labels: list = []
    for i, d in enumerate(ref_dirs):
        lab = Path(d).name or str(d)
        labels.append(lab if lab not in labels else f"{lab}#{i}")
    return labels


def compare_legs(ref_dirs, cand_dir, workloads=WORKLOADS, bands=None, repeat: bool = False) -> dict:
    """Judge ONE candidate against EVERY reference in ``ref_dirs`` (``--compare REF [REF ...]
    CAND``): the verdict is FAIL if it fails against any of them. With one reference this is
    ``compare_runs`` plus the fields that say which reference was used.

    Why more than one: the G1 re-run's first ``tp1`` HS capture was 1.318e-2 away from
    ``tp1_repeat`` (which matched both earlier TP=1 runs bit for bit), so "against TP=1" depends on
    WHICH TP=1 boot. Every TP=N leg is judged against ``tp1`` AND ``tp1_repeat``; the verdict
    records each reference (``refs``: label, dir, artifact sha256 per workload), whether the
    references are bit-identical (``refs_identical``), and the result against each (``by_ref``).
    ``ref``/``measured`` stay the FIRST reference's, as before."""
    ref_dirs = [Path(d) for d in ref_dirs]
    if not ref_dirs:
        raise ValueError("compare_legs needs at least one reference dir")
    workloads = list(workloads)
    labels = _ref_labels(ref_dirs)
    per = {lab: compare_runs(d, cand_dir, workloads, bands, repeat=repeat)
           for lab, d in zip(labels, ref_dirs)}
    first = per[labels[0]]
    refs = [dict(per[lab]["ref"], label=lab,
                 artifact_sha256={wl: _artifact_sha256(d, wl) for wl in workloads})
            for lab, d in zip(labels, ref_dirs)]
    cand = dict(first["cand"], label=Path(cand_dir).name,
                artifact_sha256={wl: _artifact_sha256(cand_dir, wl) for wl in workloads})
    multi = len(labels) > 1
    failures = []
    for lab in labels:
        for f in per[lab]["failures"]:
            head, sep, rest = f.partition("] ")
            failures.append(f"{head}] (vs {lab}) {rest}" if multi and sep else f)
    out = {
        "verdict": "FAIL" if any(per[lab]["verdict"] != "PASS" for lab in labels) else "PASS",
        "mode": first["mode"],
        "ref": refs[0],
        "refs": refs,
        "ref_labels": labels,
        "cand": cand,
        "bands": first["bands"],
        "bands_overridden": first["bands_overridden"],
        "checks": [dict(c, ref=lab) for lab in labels for c in per[lab]["checks"]],
        "measured": first["measured"],
        "by_ref": {lab: {"dir": str(d), "verdict": per[lab]["verdict"],
                         "measured": per[lab]["measured"], "failures": per[lab]["failures"]}
                   for lab, d in zip(labels, ref_dirs)},
        "failures": failures,
    }
    if multi:
        # None = no reference has this leg's artifact (hs_replicas: TP=1 references skip it).
        out["refs_identical"] = {
            wl: (None if all(r["artifact_sha256"][wl] is None for r in refs) else
                 refs[0]["artifact_sha256"][wl] is not None
                 and all(r["artifact_sha256"][wl] == refs[0]["artifact_sha256"][wl]
                         for r in refs))
            for wl in workloads}
    return out


# ======================================================================================
# CLI
# ======================================================================================

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--compare", nargs="+", metavar="DIR",
                    help="REF [REF ...] CAND: judge the run dir CAND (the last one) against EVERY "
                         "reference run dir before it (GPU-free); FAIL if it fails against any")
    ap.add_argument("--repeat", action="store_true",
                    help="--compare: CAND repeats each REF's configuration (same TP and --fuse) "
                         "-- a boot-to-boot noise leg, e.g. tp2 vs tp2_repeat -- instead of the "
                         "default parity mode, which requires every REF to be TP=1")
    ap.add_argument("--json-out", help="--compare: also write the verdict JSON here")
    ap.add_argument("--tp", type=int, help="tensor_parallel_size to run at")
    ap.add_argument("--out", help="run output dir")
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--steer-coefficient", type=float, default=4.0)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--require-mia-under", default=None)
    ap.add_argument("--fuse", choices=sorted(FUSE_REQUEST), default=None,
                    help="vLLM 0.29 compilation_config.pass_config.fuse_allreduce_rms: on -> True "
                         "(needs flashinfer.comm and its workspace, not has_flashinfer()), off -> "
                         "False, default -> ask nothing (vLLM decides: True at TP>1 only when "
                         "has_flashinfer()). --compare FAILS a TP>1 'on' leg unless the fusion "
                         "pass is armed on every rank")
    ap.add_argument("--no-fuse-allreduce-rms", action="store_true",
                    help="alias of --fuse off (kept for existing runners)")
    ap.add_argument("--child-timeout-s", type=float, default=3600.0)
    ap.add_argument("--child-workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    for flag, key in (("--hs-rel-band", "hs_rel"), ("--hs-row-band", "hs_row"),
                      ("--qk-rel-band", "qk_rel"), ("--qk-head-rel-band", "qk_head_rel"),
                      ("--steer-delta-rel-band", "steer_delta_rel"),
                      ("--steer-liveness-floor", "steer_liveness_floor")):
        ap.add_argument(flag, type=float, default=None, dest=f"band_{key}")
    return ap


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.no_fuse_allreduce_rms and args.fuse not in (None, "off"):
        print(f"[tp-probe] --no-fuse-allreduce-rms contradicts --fuse {args.fuse}", file=sys.stderr)
        return 2
    args.fuse = "off" if args.no_fuse_allreduce_rms else (args.fuse or "default")
    if args.repeat and not args.compare:
        print("[tp-probe] --repeat only applies to --compare", file=sys.stderr)
        return 2
    if args.compare:
        if len(args.compare) < 2:
            print(f"[tp-probe] --compare needs REF [REF ...] CAND (two run dirs at least), got "
                  f"{args.compare!r}", file=sys.stderr)
            return 2
        *refs, cand = args.compare
        missing = [d for d in (*refs, cand) if not Path(d).is_dir()]
        if missing:
            print(f"[tp-probe] --compare needs run dirs; not a dir: {missing!r}", file=sys.stderr)
            return 2
        bands = {k[len("band_"):]: v for k, v in vars(args).items()
                 if k.startswith("band_") and v is not None}
        workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
        verdict = compare_legs(refs, cand, workloads, bands, repeat=args.repeat)
        for f in verdict["failures"]:
            print(f"[tp-probe] FAIL {f}", file=sys.stderr)
        text = json.dumps(verdict, default=str)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(verdict, indent=2, default=str))
        print(text, flush=True)
        return 0 if verdict["verdict"] == "PASS" else 1
    if args.tp is None or not args.out:
        print("[tp-probe] a run needs --tp and --out (or use --compare REF [REF ...] CAND)",
              file=sys.stderr)
        return 2
    if args.child_workload:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
