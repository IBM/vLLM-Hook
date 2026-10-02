"""The NULL control the steer A/B design never had (task D11, ad-hoc diagnostic -- NOT a gate).

WHAT THE ARTIFACTS ALREADY SAY. `run_steer_ab_leak_control.py` (LSF 1725670) measured all
17 UNARMED requests BIT-EXACT 0.0 between Arm A (real steer) and Arm B (all-zero-direction
control). The in-suite gate `steer_small per-request-arming` drives -- since task D10 --
THAT VERY FUNCTION (`run_steer_ab_pass`), with the same workload, the same 33
distinct-length prompts, the same alternating mask, the same `max_num_seqs=33`, the same
`graph=True, cudagraph=True`, the same fingerprints, and the same engine boot lines, and it
still reports failures: LSF 1731353 flagged {req0, req2, req30}, LSF 1732150 flagged
{req14, 16, 18, 20, 22, 24, 26, 28, 30, 32}. Two runs of the SAME CODE flagging two nearly
DISJOINT sets is not a parameter difference and not a fixed mis-route.

Comparing the three jobs' on-disk artifacts request by request (sha1 of `token_logprobs`)
shows what is actually going on:

  * every unarmed row takes one of a small number of BITWISE-DISTINCT values that recur
    across jobs and even across NODES (1725670 ran on p2-r20-n2, 1731353/1732150 on
    p3-r19-n1), so these are reproducible numeric regimes, not hardware noise;
  * in LSF 1732150 the rows the gate flagged are rows where the CONTROL pass (Arm B, whose
    steering is a mathematical no-op) reproduces LSF 1725670's values bit-exactly while the
    STEERED pass matches LSF 1731353's steered pass bit-exactly -- i.e. the pass that
    "moved" away from the reference is the one where steering does nothing;
  * LSF 1731353's and LSF 1732150's Arm A agree bit-exactly on req0..req22 and disagree on
    req23..req32, with no reference to arming at all.

So "unarmed rows are bit-exact between the two passes" is not known to be a PROPERTY of the
system; it is one draw (n=1) from a distribution the same code samples differently. Before
anyone changes the gate, that has to be measured, and the A/B design cannot measure it: it
has no pass that differs from another pass in NOTHING.

THIS EXPERIMENT adds the missing passes. ONE engine, ONE boot, same prompts, same mask,
same batch composition in every pass:

  P1 real      -- Arm A, exactly as the gate drives it
  P2 zero      -- Arm B, exactly as the gate drives it (all-zero `dir`, same avg_proj)
  P3 real      -- THE NULL: identical to P1 in every argument, run at a different position
  P4 zero      -- THE NULL for Arm B: identical to P2
  P5 real-copy -- identical vector BYTES to P1 at a DIFFERENT `vector_path`

and reads off, per request:

  P1 vs P3 / P2 vs P4 (NULL)   nonzero => two passes that differ in NOTHING do not agree,
                               so the gate's bit-exact premise is false and the movement it
                               reports is pass-to-pass engine noise, not a leak. The gate
                               has to be re-expressed on a channel that IS deterministic
                               (token ids), never banded.
  P1 vs P5 (PATH)              nonzero with the NULLs at zero => the CONTROL is the defect:
                               giving Arm B a different `vector_path` (a second entry in the
                               steer vector table) is itself a structural difference between
                               the passes, and the fix is to make Arm A carry the same kind
                               of difference rather than to touch the gate's threshold.
  P1 vs P2 / P3 vs P4 (GATE)   the gate's own comparison, sampled TWICE in one boot.

Run:  python tests/mia/parity/run_steer_ab_null_control.py <out_dir>
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
    LOGPROB_KEYS,
    STEER_VECTOR,
    assert_distinct_prompt_lengths,
    prompts_for,
    run_workload,
    sampling_params_for,
    steer_arm_mask,
    steer_config,
    steer_fingerprint,
    write_request_artifacts,
)
from tests.mia.parity.run_steer_ab_leak_control import _make_zero_vector  # noqa: E402


def _copy_vector(base_path: Path, out_dir: Path) -> Path:
    """The SAME vector bytes under a different filename.

    P5 exists to separate "Arm B steers differently" from "Arm B names a different file".
    `_make_zero_vector` changes BOTH at once, so the A/B design cannot tell them apart.
    """
    import shutil

    out_path = Path(out_dir) / "phi3_adjust_rs_test_COPY.pt"
    shutil.copyfile(str(base_path), str(out_path))
    return out_path


def _delta(a: dict, b: dict) -> tuple[float, bool]:
    """max |a-b| over the logprob channels, and whether the token ids agree."""
    import torch

    shared = [k for k in LOGPROB_KEYS if k in a and k in b and a[k].shape == b[k].shape]
    if not shared:
        raise RuntimeError("no comparable logprob channel between two passes of one boot")
    worst = max(float((a[k].double() - b[k].double()).abs().max().item()) for k in shared)
    return worst, bool(torch.equal(a["token_ids"], b["token_ids"]))


def _drive_null_control(engine, wl, out_dir: Path, batch: int) -> Path:
    from safetensors.torch import load_file, save_file
    import torch

    out_dir = Path(out_dir)
    prompts = prompts_for(wl, batch)
    armed = steer_arm_mask(len(prompts))
    lengths = assert_distinct_prompt_lengths(engine.get_tokenizer(), prompts)
    print(f"[NULL] {sum(armed)} of {len(prompts)} requests armed (alternating), prompt token "
          f"counts {min(lengths)}..{max(lengths)}, all different", flush=True)

    real_cfg = steer_config(wl)
    zero_cfg = dict(real_cfg, vector_path=str(_make_zero_vector(STEER_VECTOR, out_dir)))
    copy_cfg = dict(real_cfg, vector_path=str(_copy_vector(STEER_VECTOR, out_dir)))
    for name, cfg in (("real", real_cfg), ("zero", zero_cfg), ("real-copy", copy_cfg)):
        print(f"[NULL] {name:>9} fingerprint {steer_fingerprint(cfg)}", flush=True)

    passes = (("P1_real", real_cfg), ("P2_zero", zero_cfg), ("P3_real", real_cfg),
              ("P4_zero", zero_cfg), ("P5_realcopy", copy_cfg))
    for label, cfg in passes:
        params = [sampling_params_for(wl, {"steer": cfg} if a else None) for a in armed]
        print(f"[NULL] running {label} -> {label}.safetensors ...", flush=True)
        write_request_artifacts(out_dir, engine.generate(prompts, params, use_tqdm=False),
                                label=label)
    for index, is_armed in enumerate(armed):
        save_file({"armed": torch.tensor(int(is_armed), dtype=torch.int64)},
                  str(out_dir / f"req{index}" / "arming.safetensors"))

    pairs = (("NULL A/A  P1_real vs P3_real", "P1_real", "P3_real"),
             ("NULL B/B  P2_zero vs P4_zero", "P2_zero", "P4_zero"),
             ("PATH      P1_real vs P5_realcopy", "P1_real", "P5_realcopy"),
             ("GATE 1    P1_real vs P2_zero", "P1_real", "P2_zero"),
             ("GATE 2    P3_real vs P4_zero", "P3_real", "P4_zero"))
    summary: dict[str, tuple[float, float, int, int]] = {}
    for title, la, lb in pairs:
        worst_un = worst_arm = 0.0
        moved_un = tok_un = 0
        for index, is_armed in enumerate(armed):
            req = out_dir / f"req{index}"
            d, toks_eq = _delta(load_file(str(req / f"{la}.safetensors")),
                                load_file(str(req / f"{lb}.safetensors")))
            if is_armed:
                worst_arm = max(worst_arm, d)
                continue
            worst_un = max(worst_un, d)
            moved_un += int(d != 0.0)
            tok_un += int(not toks_eq)
            if d != 0.0:
                print(f"[NULL] {title}: req{index} UNARMED max|d(logprob)|={d:.6e}"
                      f"{' TOKEN-IDS-DIFFER' if not toks_eq else ''}", flush=True)
        summary[title] = (worst_un, worst_arm, moved_un, tok_un)
        print(f"[NULL] SUMMARY {title}: unarmed worst={worst_un:.6e} moved={moved_un}/"
              f"{len(armed) - sum(armed)} token-id-flips={tok_un}; armed worst="
              f"{worst_arm:.6e}", flush=True)

    null_a = summary["NULL A/A  P1_real vs P3_real"]
    null_b = summary["NULL B/B  P2_zero vs P4_zero"]
    path = summary["PATH      P1_real vs P5_realcopy"]
    nulls_exact = null_a[0] == 0.0 and null_b[0] == 0.0
    path_exact = path[0] == 0.0
    if not nulls_exact:
        verdict = ("PASS-TO-PASS NOISE -- two passes differing in NOTHING disagree on "
                   "unarmed rows, so the gate's bit-exact premise is FALSE")
    elif not path_exact:
        verdict = ("CONTROL ARTIFACT -- identical vector bytes at a different vector_path "
                   "move unarmed rows, so Arm B's path change, not steering, is the mover")
    else:
        verdict = ("BIT-EXACT NULLS -- pass repeatability and the path change are both "
                   "clean, so any A-vs-B unarmed movement is a REAL leak")
    print(f"VERDICT NULL steer_ab_null_control: {verdict}", flush=True)
    print(f"[NULL] token-id flips on unarmed rows, all pairs: "
          f"{sum(v[3] for v in summary.values())}", flush=True)
    return out_dir


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ab_steer_null_control")
    run_workload("steer_small", out_dir, graph=True, cudagraph=True, batch=1,
                 drive=_drive_null_control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
