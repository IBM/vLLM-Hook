"""Decisive experiment: is the `steer_small per-request-arming` UNARMED movement a real
per-row leak, or boot/composition noise? (task D6, ad-hoc diagnostic -- NOT a parity gate.)

LSF 1725139's `full_batch_mixed_arm` leg found 5 of 17 UNARMED requests moved by up to
2.575421e-02 (floor 1.0e-03), while the weakest ARMED effect was 2.757737e-01 -- about 10x
larger. That is suggestive but not decisive: the leg's own "control" pass runs a COMPLETELY
different batch composition (nobody armed at all), so a real per-row leak and an artifact of
comparing two differently-composed passes both predict "some unarmed rows move a little".

THE SEPARATING EXPERIMENT. Two passes, same engine, same boot, same 33 distinct-length
prompts, same order, same max_num_seqs=33, same seed, same ignore_eos+fixed max_tokens, same
alternating arming mask (`steer_arm_mask`, task D5 item 7) -- so batch composition is
IDENTICAL in both passes, row for row:

  Arm A -- the 16 armed requests carry the REAL steer config (method=adjust_rs,
           coefficient field is irrelevant for this method -- see below).
  Arm B -- the SAME 16 requests carry the SAME config, except `vector_path` points at an
           ALL-ZERO-DIRECTION clone of the same vector (same `avg_proj`). The unarmed 17
           are untouched in both arms (extra_args=None, exactly as in the real leg).

WHY NOT "coefficient: 0.0". This workload's method is `adjust_rs`, and
`mia/graph/install_steer.py` (`coefficient = 0.0 if method == "adjust_rs" else
float(cfg.get("coefficient", 0.0))`) and the eager path (`mia/workers/steer_worker.py`,
`_steer_rows`) BOTH ignore the host `coefficient` field for adjust_rs entirely -- the
per-token magnitude is computed IN-KERNEL as `avg_proj[vid] - (residual . unit)`, where
`unit` is the raw `dir` tensor used AS-IS (`unit_vec = steering_vec  # use dir as unit
vector (matches old behavior)`; the buffer-mode kernel does the identical thing:
`unit = vec_table[vec_id[t]]`, no runtime renormalization anywhere in the pipeline). So
`coefficient: 0.0` on an adjust_rs config is already the status quo and changes nothing --
exactly the "control is worthless" trap the task warned about.

An ALL-ZERO `dir` is the real zero-effect control instead, and it does NOT skip the op:
  * `current_projections = matmul(rows, unit=0) = 0`   (still computed, still a real matmul)
  * `coeff = avg_proj - 0 = avg_proj`                    (still computed, still nonzero)
  * `rows + coeff * unit(=0) = rows`                     (still executed: the ADD runs, adds
                                                            an exact zero, not skipped)
The Triton kernel takes the identical `steer_mode=1` (adjust_rs) branch, the identical
2-pass reduction+add, the identical per-row routing/masking machinery as Arm A -- the ONLY
difference is that `dir` is zero instead of the trained direction, which zeroes the
mathematical RESULT without touching which code path runs or which rows get an assignment.

THE READOUT, per unarmed request, A vs B:
  BIT-IDENTICAL  -> the steering arithmetic never touches unarmed rows; the movement in the
                     real gate is a composition-comparison artifact. Conclusion (b).
  DIFFERS        -> the only thing that changed between A and B is the numeric content
                     applied to OTHER (armed) rows, so it reached an unarmed row. Real bug,
                     conclusion (a) -- report which requests/layers/magnitude and stop.
Per armed request, A vs B is the positive control: it must be LARGE (Arm A's steer is live,
Arm B's is a genuine no-op), or Arm B failed to be a real control.

TASK D10: this module's drive logic (``run_steer_ab_pass``) is now the ONLY place the A/B
pass is driven -- ``tests/mia/parity/t2_invariants.py``'s in-suite `steer_small
per-request-arming` gate calls it directly instead of keeping its own copy. See
``run_steer_ab_pass``'s docstring for why (LSF 1731353 vs LSF 1725670).

Run:  python tests/mia/parity/run_steer_ab_leak_control.py <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402

os.environ.setdefault("MIA_PARITY_PROMPTS", "distinct")

from tests.mia.parity.capture_workload import (  # noqa: E402
    WORKLOADS,
    prompts_for,
    run_workload,
    sampling_params_for,
    steer_arm_mask,
    steer_config,
    steer_fingerprint,
    write_request_artifacts,
    assert_distinct_prompt_lengths,
    LOGPROB_KEYS,
    STEER_LIVENESS_ATOL,
)


def _make_zero_vector(base_path: Path, out_dir: Path) -> Path:
    """Clone the real steering vector with an all-zero `dir`, same `avg_proj`.

    `dir` is used AS-IS as the unit vector everywhere in the pipeline (no runtime
    renormalization) -- see the module docstring. A zero `dir` makes both the eager path
    (`_steer_rows`) and the buffer-mode Triton kernel compute a real, nonzero in-kernel
    `coeff`, then multiply it by a zero `unit`, landing on an EXACT zero addition without
    skipping the op or touching the routing/masking machinery.
    """
    import torch

    raw = torch.load(str(base_path), map_location="cpu", weights_only=False)
    d = raw["dir"]
    import numpy as np
    zero_dir = np.zeros_like(d) if isinstance(d, np.ndarray) else torch.zeros_like(d)
    zero = {"dir": zero_dir, "avg_proj": raw["avg_proj"]}
    out_path = out_dir / "phi3_adjust_rs_ZERO_control.pt"
    torch.save(zero, str(out_path))
    return out_path


def run_steer_ab_pass(engine, wl, out_dir: Path, *, batch: int = 1,
                      label_a: str = "armA", label_b: str = "armB") -> list[bool]:
    """Drive Arm A (real steer) then Arm B (all-zero-direction control) on the SAME engine,
    the SAME boot, back to back. THE ONE PLACE this experiment's mechanics live (task D10).

    Before task D10, ``tests/mia/parity/t2_invariants.py``'s ``_drive_steer_mixed`` kept its OWN
    copy of this exact sequence -- textually close to this one, independently maintained --
    reusing only ``_make_zero_vector`` from this module. LSF 1731353 (task D9's GPU run)
    found the in-suite gate that copy feeds (`steer_small per-request-arming`) moving 3 of
    33 UNARMED requests by 1.613855e-02 / 2.467152e-02 / 2.032498e-02, at the SAME
    configuration (same engine, same boot, same 33 distinct-length prompts, same alternating
    mask) where THIS script's own dedicated GPU run (LSF 1725670) measured bit-exact 0.0 on
    all 17 unarmed requests. Two hand-maintained copies of "run both arms in one boot" is
    still two chances for one of them to drift from the other; there is now exactly one
    driver, called by both this module's ``main()`` (via ``_drive_ab_leak_control`` below)
    and ``t2_invariants._drive_steer_mixed``.

    ``label_a``/``label_b`` name the two ``generate()`` passes' artifact files
    (``<label>.safetensors`` under each ``req<N>/``): this script's own in-process readout
    below reads ``armA``/``armB``; ``_judge_steer_per_request_arming`` reads ``generation``/
    ``control``. An ``arming.safetensors`` marker (bool, 1 per request) is written either
    way, so a reader never has to re-derive ``steer_arm_mask`` and risk disagreeing with
    what was actually driven.

    Returns the ``armed`` mask, in prompt order.
    """
    import torch
    from safetensors.torch import save_file

    from tests.mia.parity.capture_workload import STEER_VECTOR

    out_dir = Path(out_dir)
    prompts = prompts_for(wl, batch)
    armed = steer_arm_mask(len(prompts))
    lengths = assert_distinct_prompt_lengths(engine.get_tokenizer(), prompts)
    n_armed, n_unarmed = sum(armed), len(armed) - sum(armed)
    print(f"[AB] {n_armed} of {len(prompts)} requests armed (alternating), "
          f"prompt token counts {min(lengths)}..{max(lengths)}, all different", flush=True)
    if not n_armed or not n_unarmed:
        raise RuntimeError(f"mixed-arm batch is not mixed: {n_armed} armed, {n_unarmed} "
                           f"unarmed -- this experiment needs both classes")

    real_cfg = steer_config(wl)
    zero_vec_path = _make_zero_vector(STEER_VECTOR, out_dir)
    zero_cfg = dict(real_cfg, vector_path=str(zero_vec_path))
    print(f"[AB] Arm A fingerprint {steer_fingerprint(real_cfg)}", flush=True)
    print(f"[AB] Arm B fingerprint {steer_fingerprint(zero_cfg)}", flush=True)

    params_a = [sampling_params_for(wl, {"steer": real_cfg} if a else None) for a in armed]
    params_b = [sampling_params_for(wl, {"steer": zero_cfg} if a else None) for a in armed]

    print(f"[AB] running Arm A (real steer on armed rows) -> {label_a}.safetensors...",
          flush=True)
    out_a = engine.generate(prompts, params_a, use_tqdm=False)
    print(f"[AB] running Arm B (zero-direction steer on armed rows) -> {label_b}."
          f"safetensors...", flush=True)
    out_b = engine.generate(prompts, params_b, use_tqdm=False)

    written = write_request_artifacts(out_dir, out_a, label=label_a)
    written += write_request_artifacts(out_dir, out_b, label=label_b)
    for index, is_armed in enumerate(armed):
        save_file({"armed": torch.tensor(int(is_armed), dtype=torch.int64)},
                  str(Path(out_dir) / f"req{index}" / "arming.safetensors"))
        written += 1
    print(f"[AB] wrote {written} artifact files under {out_dir}", flush=True)
    if written == 0:
        raise RuntimeError(f"the steer A/B pass produced NOTHING under {out_dir}")
    return armed


def _drive_ab_leak_control(engine, wl, out_dir: Path, batch: int) -> Path:
    """Drive the A/B pass (``run_steer_ab_pass``, task D10) and do THIS script's OWN
    in-process readout, so the answer is visible in the log without a second pass over the
    artifact tree.
    """
    import torch
    from safetensors.torch import load_file

    armed = run_steer_ab_pass(engine, wl, Path(out_dir), batch=batch,
                              label_a="armA", label_b="armB")

    worst_unarmed, weakest_armed_delta = 0.0, float("inf")
    unarmed_bitexact = True
    problems = []
    for index, is_armed in enumerate(armed):
        gen_dir = Path(out_dir) / f"req{index}"
        a = load_file(str(gen_dir / "armA.safetensors"))
        b = load_file(str(gen_dir / "armB.safetensors"))
        shared = [k for k in LOGPROB_KEYS if k in a and k in b and a[k].shape == b[k].shape]
        if not shared:
            problems.append(f"req{index}: no comparable logprob channel A-vs-B")
            continue
        delta = max(float((a[k].double() - b[k].double()).abs().max().item())
                    for k in shared)
        # token ids too -- a leak that also flips the sampled token is the loudest possible
        # signal, and armA/armB token_ids differing on an UNARMED row is definitionally the
        # same finding as a logprob move (recorded, not required, since greedy decoding can
        # mask a small delta).
        tok_diff = not torch.equal(a["token_ids"], b["token_ids"])
        if is_armed:
            weakest_armed_delta = min(weakest_armed_delta, delta)
        else:
            if delta > 0.0:
                unarmed_bitexact = False
            worst_unarmed = max(worst_unarmed, delta)
            tag = "MOVED" if delta > STEER_LIVENESS_ATOL else (
                "bit-exact" if delta == 0.0 else "float-noise")
            print(f"[AB] req{index} UNARMED A-vs-B: max|d(logprob)|={delta:.6e} "
                  f"[{tag}]{' TOKEN-IDS-DIFFER' if tok_diff else ''}", flush=True)

    print(f"[AB] SUMMARY armed A-vs-B (positive control): weakest delta = "
          f"{0.0 if weakest_armed_delta == float('inf') else weakest_armed_delta:.6e} "
          f"(must be LARGE, ~2.76e-01, or Arm B failed to be a real control)", flush=True)
    print(f"[AB] SUMMARY unarmed A-vs-B (the actual test): worst delta = "
          f"{worst_unarmed:.6e}, bit-exact={unarmed_bitexact}", flush=True)
    verdict = "NO-LEAKAGE (b)" if unarmed_bitexact else "LEAK DETECTED (a)"
    print(f"VERDICT AB steer_leak_control: {verdict}", flush=True)
    for p in problems:
        print("MISMATCH", p, flush=True)

    return Path(out_dir)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ab_steer_leak_control")
    run_workload("steer_small", out_dir, graph=True, cudagraph=True, batch=1,
                drive=_drive_ab_leak_control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
