"""The three round-numbered bands must reject a corruption, and must not be wideable.

Task D5 item 4. The structural review found that three of this suite's six bands --
`_T0_GRAPH_LOGPROB_BAND`, `_STEER_FUSED_LOGPROB_BAND`, `_GPU_ROUTING_REL_BAND` -- were
round numbers with NO discrimination test and a live environment override:

    MIA_T2_T0_GRAPH_BAND=10 MIA_T2_STEER_FUSED_BAND=10 MIA_T2_GPU_ROUTING_BAND=10 pytest
      -> 88 passed

and at `STEER_FUSED_BAND=10` a totally broken fused steer (intervention scale 7.7e-01)
passed. `run_parity.sh` neither pinned nor echoed any of them, so the log said nothing
about what was enforced.

Three things are pinned here. The fused-steer band lives in
`tests/test_parity_steer_bands.py` with the rest of the steer machinery; this file covers:

  * `_T0_GRAPH_LOGPROB_BAND` -- accept the measured 1.815024e-02, reject an
    intervention-scale 7.632304e-01;
  * `_GPU_ROUTING_REL_BAND` -- accept the measured 2.228e-03 relative, reject the two
    mis-route signatures (an off-by-one row, a zeroed row);
  * the registry itself -- a WIDENING override is refused at import, a tightening one is
    allowed, and every enforced bound is echoed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mia.parity.t2_invariants import (  # noqa: E402
    _GPU_ROUTING_REL_BAND,
    _REPLAY_ROW_BAND,
    _T0_GRAPH_LOGPROB_BAND,
    _compare_generation_band,
    _compare_replay_band,
)

# Measured on GPU; each appears beside the constant it justifies in t2_invariants.py.
T0_GRAPH_MEASURED = 1.815024e-02      # HS worst |d(logprob)|, LSF 1702981 AND 1704450
GPU_ROUTING_MEASURED = 2.228e-03      # worst relative, HS width 32, LSF 1702981
STEER_EFFECT = 7.632304e-01           # the smallest real steering intervention measured
OFF_BY_ONE_ROW = 4.512e-01            # tensor-global FLOOR of a one-row shift, LSF 1705469


# ---------------------------------------------------------------------------
# _T0_GRAPH_LOGPROB_BAND: "does arming capture perturb generation under FULL graphs?"
# ---------------------------------------------------------------------------

def _generation(root: Path, *, shift: float) -> Path:
    for i in range(3):
        d = root / f"req{i}"
        d.mkdir(parents=True, exist_ok=True)
        save_file({"prompt_token_ids": torch.tensor([5, 6, 7], dtype=torch.int64),
                   "token_ids": torch.tensor([11, 12, 13], dtype=torch.int64),
                   "token_logprobs": torch.tensor([-1.0, -2.0, -3.0],
                                                  dtype=torch.float64) + shift,
                   "cumulative_logprob": torch.tensor(-6.0 + shift, dtype=torch.float64)},
                  str(d / "generation.safetensors"))
    return root


def _t0_verdict(tmp_path: Path, delta: float) -> bool:
    a = _generation(tmp_path / "off", shift=0.0)
    b = _generation(tmp_path / "on", shift=delta)
    verdicts: list = []
    _compare_generation_band("unit", a, b, verdicts=verdicts,
                             bound=_T0_GRAPH_LOGPROB_BAND, note="unit test")
    return verdicts[0][1]


def test_t0_graph_band_accepts_the_measured_compile_fusion_delta(tmp_path):
    """1.815024e-02, the SAME value in LSF 1702981 and 1704450 -- reproducible, and a
    property of what inductor is given to fuse rather than of capture. Token ids were
    identical on every request in both jobs. The band exists to tolerate exactly this."""
    assert _t0_verdict(tmp_path, T0_GRAPH_MEASURED)


def test_t0_graph_band_rejects_an_intervention_scale_perturbation(tmp_path):
    """The reject side the band never had. If capture started genuinely perturbing
    generation, the scale to beat is a real intervention on these same logprobs: a steer
    moves them 7.632304e-01, 15x the band."""
    assert not _t0_verdict(tmp_path, STEER_EFFECT)


def test_t0_graph_band_sits_between_the_measurement_and_the_corruption():
    assert T0_GRAPH_MEASURED < _T0_GRAPH_LOGPROB_BAND < STEER_EFFECT
    assert _T0_GRAPH_LOGPROB_BAND / T0_GRAPH_MEASURED > 2.0
    assert STEER_EFFECT / _T0_GRAPH_LOGPROB_BAND > 10.0


# ---------------------------------------------------------------------------
# _GPU_ROUTING_REL_BAND: the MIA_CAPTURE_GPU_ROUTING lever against host routing
# ---------------------------------------------------------------------------

def _rows(n: int = 20, width: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, width, generator=g, dtype=torch.float32)
    x[:, 0] *= 500.0          # a residual-stream-like magnitude spread
    return x.to(torch.float16)


def _tree(root: Path, hidden: torch.Tensor, layer_num: int = 16) -> Path:
    d = root / "req0"
    d.mkdir(parents=True, exist_ok=True)
    save_file({"hidden_states": hidden.contiguous(),
               "layer_num": torch.tensor(layer_num, dtype=torch.int64)},
              str(d / f"layer{layer_num:03d}.safetensors"))
    return root


def _routing_verdict(a: Path, b: Path) -> bool:
    verdicts: list = []
    _compare_replay_band("unit", a, b, name="layer*.safetensors", verdicts=verdicts,
                         rel=_GPU_ROUTING_REL_BAND)
    return verdicts[0][1]


def test_gpu_routing_band_accepts_the_measured_lever_delta(tmp_path):
    """2.228e-03 relative at width 32 (LSF 1702981; 0.000000e+00 in 1704450 -- boot
    dependent). The two legs do not run the same forward: GENERATION itself differs between
    them, and capture cannot alter generation, so the capture delta is downstream of an
    input-side difference. Rounding is proportional to each value, so that is how it is
    modelled."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    b = _tree(tmp_path / "b", (x.float() * (1.0 + GPU_ROUTING_MEASURED)).to(torch.float16))
    assert _routing_verdict(a, b)


def test_gpu_routing_band_rejects_an_off_by_one_row(tmp_path):
    """The first mis-route signature, at the FLOOR measured over tensors: 4.512e-01
    tensor-global, 45x the band. An aperture off-by-one lands exactly here."""
    x = _rows()
    a = _tree(tmp_path / "a", x)
    shifted = torch.roll(x, shifts=1, dims=0)
    glob = (x.float() - shifted.float()).abs().max().item() / x.abs().max().item()
    assert glob > OFF_BY_ONE_ROW * 0.5, f"fixture is not a real off-by-one: {glob:.3e}"
    assert not _routing_verdict(a, _tree(tmp_path / "b", shifted))


def test_gpu_routing_band_rejects_a_zeroed_row(tmp_path):
    """The second signature: a padding row that read stale routing and scattered into a live
    aperture slot leaves a zeroed/sentinel row, which the per-row metric scores at exactly
    1.0 whatever its magnitude. This gate inherits `_REPLAY_ROW_BAND` for that metric."""
    x = _rows()
    x[3] *= 1e-3                      # small next to the tensor -- invisible to the global metric
    corrupt = x.clone()
    corrupt[3] = 0
    glob = (x.float() - corrupt.float()).abs().max().item() / x.abs().max().item()
    assert glob < _GPU_ROUTING_REL_BAND, (
        f"precondition: this corruption must be invisible to the global metric ({glob:.3e})")
    assert 1.0 > _REPLAY_ROW_BAND
    assert not _routing_verdict(_tree(tmp_path / "a", x), _tree(tmp_path / "b", corrupt))


def test_gpu_routing_band_sits_between_the_measurement_and_the_corruption():
    assert GPU_ROUTING_MEASURED < _GPU_ROUTING_REL_BAND < OFF_BY_ONE_ROW
    assert _GPU_ROUTING_REL_BAND / GPU_ROUTING_MEASURED > 4.0
    assert OFF_BY_ONE_ROW / _GPU_ROUTING_REL_BAND > 40.0


# ---------------------------------------------------------------------------
# The registry: a band that can be widened from the environment is not a band
# ---------------------------------------------------------------------------

def _import_with(env: dict) -> subprocess.CompletedProcess:
    child = dict(os.environ)
    child.pop("MIA_T2_ALLOW_BAND_OVERRIDE", None)
    child.update(env)
    child["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + child.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests" / "mia" / "parity" / "t2_invariants.py"),
         "--print-bands"],
        env=child, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)


def test_the_three_reviewed_overrides_are_refused():
    """The exact command line from the review. It must no longer start."""
    proc = _import_with({"MIA_T2_T0_GRAPH_BAND": "10",
                         "MIA_T2_STEER_FUSED_BAND": "10",
                         "MIA_T2_GPU_ROUTING_BAND": "10"})
    assert proc.returncode != 0, proc.stdout
    assert "REFUSING TO RUN" in proc.stderr + proc.stdout


def test_every_band_refuses_its_own_widening():
    """Not just the three the review happened to try: every registered bound."""
    listing = _import_with({})
    assert listing.returncode == 0, listing.stderr
    names = [line.split()[2] for line in listing.stdout.splitlines()
             if line.startswith("[T2] BAND ")]
    assert len(names) >= 10, names
    for name in names:
        proc = _import_with({name: "1e9"})
        assert proc.returncode != 0, f"{name} accepted a 1e9 widening:\n{proc.stdout}"


def test_a_tightening_override_is_allowed_and_labelled():
    """A stricter gate cannot manufacture a pass, so tightening stays available -- but the
    log has to say the run was not on the documented value."""
    proc = _import_with({"MIA_T2_T0_GRAPH_BAND": "1e-3"})
    assert proc.returncode == 0, proc.stderr
    assert "TIGHTENED" in proc.stdout


def test_a_deliberate_widening_is_possible_but_shouted():
    """The escape hatch exists -- a suite you cannot override is a suite people edit -- but
    it is explicit and every band line carries the fact."""
    proc = _import_with({"MIA_T2_T0_GRAPH_BAND": "10",
                         "MIA_T2_ALLOW_BAND_OVERRIDE": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "WIDENED" in proc.stdout and "MIA_T2_ALLOW_BAND_OVERRIDE=1" in proc.stdout


def test_print_bands_echoes_every_enforced_bound():
    """`run_parity.sh` calls this before any arm boots: a green run whose log does not say
    what it enforced is not evidence of anything."""
    proc = _import_with({})
    assert proc.returncode == 0, proc.stderr
    for name in ("MIA_T2_T0_GRAPH_BAND", "MIA_T2_STEER_FUSED_BAND", "MIA_T2_GPU_ROUTING_BAND",
                 "MIA_T2_REPLAY_REL_BAND", "MIA_T2_REPLAY_ROW_BAND",
                 "MIA_T2_ROUTING_QK_REL_BAND", "MIA_T2_ROUTING_QK_ROW_BAND",
                 "MIA_T2_BUILDER_REL_BAND", "MIA_T2_BUILDER_ROW_BAND",
                 "MIA_T2_STEER_BUDGET_CEILING", "STEER_ATOL"):
        assert f"BAND {name} =" in proc.stdout, f"{name} is not echoed"


def test_run_parity_refuses_to_start_on_a_widened_band():
    """The shell half: the job aborts before booting an engine, with a message that says
    why -- not three hours later with a PASS nobody can interpret."""
    script = (REPO_ROOT / "tests" / "mia" / "parity" / "run_parity.sh").read_text()
    assert "--print-bands" in script
    assert "a band override was refused" in script
