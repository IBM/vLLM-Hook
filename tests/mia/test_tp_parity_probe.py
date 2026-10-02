"""The TP parity probe's COMPARISON (tests/mia/parity/tp_parity_probe.py --compare), GPU-free.

The probe's GPU half can only run in gate G1; its judging half is what decides whether G1
passes, so it is pinned here against fabricated run dirs in the probe's own on-disk format.
The comparisons that MUST fail -- a rank-0-only QK capture, a missing layer, a head
permutation, a zero-filled layer, a stray HS rank dir, an inert or doubled steer -- are each a
test, next to the controls that must pass (the identical run, in-band noise, and a greedy
trajectory that diverges after a few tokens, whose later rows are legitimately different).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.mia.parity import tp_parity_probe as probe  # noqa: E402

L, HIDDEN, H_Q, H_KV, D = 4, 32, 4, 2, 8
MODEL = {"id": "fake/llama", "num_hidden_layers": L, "hidden_size": HIDDEN,
         "num_attention_heads": H_Q, "num_key_value_heads": H_KV, "head_dim": D}
PROMPT_LENS = (5, 7)
N_GEN = 4


def _gen(i, seed_tokens=None):
    p = torch.arange(PROMPT_LENS[i], dtype=torch.int64) + 100 * (i + 1)
    t = torch.arange(N_GEN, dtype=torch.int64) + 7 if seed_tokens is None else seed_tokens
    return {"prompt_token_ids": p, "token_ids": t.clone()}


def _hs_requests(seed=0):
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for i, P in enumerate(PROMPT_LENS):
        rec = _gen(i)
        rec["layers"] = {layer: torch.randn(P + N_GEN - 1, HIDDEN, generator=g)
                         for layer in range(1, L + 1)}
        reqs.append(rec)
    return reqs


def _qk_requests(seed=0, q_w=H_Q * D, k_w=H_KV * D):
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for i, P in enumerate(PROMPT_LENS):
        rec = _gen(i)
        rows = P + N_GEN - 1
        rec["layers"] = {layer: {"q": torch.randn(rows, q_w, generator=g),
                                 "k_full": torch.randn(rows, k_w, generator=g),
                                 "k_prefix_ends": torch.tensor([P + j for j in range(N_GEN)])}
                         for layer in range(L)}
        reqs.append(rec)
    return reqs


def _steer(seed=0, effect=2.0):
    g = torch.Generator().manual_seed(seed)
    un, st = [], []
    for i, P in enumerate(PROMPT_LENS):
        base = -torch.rand(P, generator=g, dtype=torch.float64) * 5
        base[0] = float("nan")
        delta = torch.randn(P, generator=g, dtype=torch.float64) * effect
        u, s = _gen(i), _gen(i)
        u["prompt_logprobs"], s["prompt_logprobs"] = base, base + delta
        un.append(u)
        st.append(s)
    return {"workload": "steer", "steered": st, "unsteered": un}


def _rank_rec(r, passes, configured=1):
    """One rank's fusion record as the child's _fusion_pass_state writes it: the pass reached
    through the compiled model's VllmBackend, ``configured`` post-grad pass managers seen."""
    return {"rank": r, "config": True, "flashinfer_comm": True, "locator": probe.FUSION_LOCATOR,
            "compiled": [{"module": "model", "class": "LlamaModel",
                          "route": "aot_compiled_fn._artifacts.compiled_fn.vllm_backend",
                          "loaded_from_disk": False, "backend": bool(configured),
                          "pass_manager_configured": bool(configured),
                          "pass_names": (["NoOpEliminationPass"]
                                         + (["AllReduceFusionPass"] if passes else []))}],
            "backends_configured": configured, "passes": passes, "matched_count_total": None,
            "matched_count_total_note": probe.FUSION_TOTAL_UNAVAILABLE}


def _pass(disabled=False, count=2):
    return {"module": "model", "disabled": disabled, "max_token_num": 1024, "tp_size": 4,
            "called": not disabled and count is not None,
            "matched_count_last_call": None if disabled else count}


def _armed(tp, disabled_ranks=(), counts=None):
    """Per-rank fusion evidence of a fused leg; ``counts`` = the last pass call's matches."""
    return [_rank_rec(r, [_pass(disabled=r in disabled_ranks, count=(counts or {}).get(r, 2))])
            for r in range(tp)]


def _unarmed(tp):
    """An unfused leg: every rank's pass manager was configured and holds no AllReduceFusionPass."""
    return [_rank_rec(r, []) for r in range(tp)]


def _w3_blind(tp):
    """What the gc.get_objects() scan recorded on EVERY rank of every TP>1 leg of the G1 re-run
    (LSF 1780395), fused or not: vLLM had frozen the heap, so the scan saw no pass."""
    return [{"rank": r, "config": True, "flashinfer_comm": True, "passes": []} for r in range(tp)]


def _listing(wl, tp, rank_bytes=None):
    """An aperture listing the way the parent writes it; default = exactly the expected dirs:
    HS every rank (the round-robin layer shard; L=4 layers cover TP<=4) or tp_rank_0 at TP=1,
    hs_replicas and QK every rank, steer none."""
    if rank_bytes is None:
        rank_bytes = ({f"tp_rank_{r}": 4096 for r in range(tp)} if wl in ("hs", "hs_replicas")
                      else {f"tp_rank_{r}": 2048 for r in range(tp)} if wl == "qk" else {})
    return {"root": f"/scratch/{wl}/aperture", "exists": bool(rank_bytes), "other": [],
            "rank_dirs": {k: {"files": ([{"name": f"{wl}_aperture_meta.jsonl", "bytes": 64},
                                         {"name": "layer_1.raw", "bytes": b}] if b else []),
                              "total_bytes": b + 64 if b else 0}
                          for k, b in rank_bytes.items()}}


def _writers(wl, tp, per_rank=None, hs_rank0_only=False):
    """Per-rank save modes as the child's _writer_state records them; default = the contract
    (every capturing rank runs its writer; HS sink ranks exist only in the MIA_HS_TP_SHARD=0
    rank-0-only layout)."""
    out = []
    for r in range(tp):
        if wl == "steer":
            rec = {"rank": r, "writer_mode": None}
        elif wl == "hs" and r > 0 and hs_rank0_only:
            rec = {"rank": r, "writer_mode": probe.WRITER_HS_SINK}
        else:
            rec = {"rank": r, "writer_mode": probe.WRITER_ON, "alive": True,
                   "child_pids": [1000 + r], "parent_daemonic": tp > 1}
        rec.update((per_rank or {}).get(r, {}))
        out.append(rec)
    return out


def _hs_header(r, tp, layout="round_robin"):
    """An HS sidecar header as MIA writes it: shard fields only under the TP layer shard."""
    h = {"dtype": "bfloat16", "row_shape": [HIDDEN], "hidden": HIDDEN, "tp_rank": r,
         "tp_size": tp, "num_layers": L, "capture_all_ranks": layout == "all_ranks"}
    if layout == "round_robin" and tp > 1:
        h.update({"layer_shard": "round_robin",
                  "owned_layers": [i + 1 for i in range(L) if i % tp == r]})
    return h


def _hs_manifest(wl, tp, layout=None):
    """flush_dirs + rank_headers of an HS leg in ``layout`` (default: the leg's own)."""
    layout = layout or ("all_ranks" if wl == "hs_replicas" else "round_robin")
    ranks = [0] if (tp == 1 or layout == "rank0") else list(range(tp))
    dirs = [f"/ap/{wl}/aperture/tp_rank_{r}" for r in ranks]
    return {"flush_dirs": dirs,
            "rank_headers": {d: _hs_header(r, tp, layout) for d, r in zip(dirs, ranks)}}


def _write(run: Path, wl, tp, data, listing="default", replicas="default", **manifest_extra):
    d = run / wl
    d.mkdir(parents=True, exist_ok=True)
    fused = tp > 1
    env = {"VLLM_PLUGINS": None, "MIA_APERTURE_PER_REQUEST": None,
           "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
    if wl == "hs_replicas":
        env[probe.HS_ALL_RANKS_ENV] = "1"
    m = {"workload": wl, "tp": tp, "model": MODEL, "mia": "/x/mia/__init__.py",
         "resolved": {"fuse_allreduce_rms": fused},
         "requested_fuse_allreduce_rms": True if fused else "vllm-default",
         "fusion_pass": _armed(tp) if fused else [],
         "fusion_active": True if fused else None,
         "mia_plugin_loaded": True,
         "env": env,
         "writer": _writers(wl, tp)}
    if wl in ("hs", "hs_replicas"):
        m.update(_hs_manifest(wl, tp))
    if wl == "hs_replicas":
        m["replica_check"] = "ok"
    if wl == "qk":
        m["flush_dirs"] = [f"/ap/qk/aperture/tp_rank_{r}" for r in range(tp)]
    m.update(manifest_extra)
    (d / probe.CHILD_MANIFEST).write_text(json.dumps(m))
    if listing == "default":
        listing = _listing(wl, tp)
    if listing is not None:
        (d / probe.APERTURE_LISTING).write_text(json.dumps(listing))
    fname = probe.STEER_FILE if wl == "steer" else probe.CAPTURE_FILE
    torch.save(data if wl == "steer" else {"workload": wl, "requests": data}, d / fname)
    if wl == "hs_replicas":
        if replicas == "default":        # every rank holds rank 0's copy, bit for bit
            replicas = {r: [{k: t.clone() for k, t in rec["layers"].items()} for rec in data]
                        for r in range(tp)}
        if replicas is not None:
            torch.save({"workload": wl, "ranks": replicas}, d / probe.REPLICAS_FILE)


def _noisy(reqs, scale=1e-3, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for rec in reqs:
        r = {k: v for k, v in rec.items() if k != "layers"}
        r["layers"] = {}
        for layer, v in rec["layers"].items():
            if isinstance(v, dict):
                r["layers"][layer] = {
                    "q": v["q"] + scale * torch.randn(v["q"].shape, generator=g),
                    "k_full": v["k_full"] + scale * torch.randn(v["k_full"].shape, generator=g),
                    "k_prefix_ends": v["k_prefix_ends"].clone()}
            else:
                r["layers"][layer] = v + scale * torch.randn(v.shape, generator=g)
        out.append(r)
    return out


@pytest.fixture
def ref(tmp_path):
    run = tmp_path / "tp1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"tp": 1}))
    _write(run, "hs", 1, _hs_requests())
    _write(run, "qk", 1, _qk_requests())
    _write(run, "steer", 1, _steer())
    return run


def _cand(tmp_path, tp=4, hs=None, qk=None, steer=None, hs_extra=None, qk_extra=None,
          steer_extra=None, listings=None, replicas=None, replicas_extra=None,
          replicas_ranks="default"):
    run = tmp_path / f"tp{tp}"
    listings = listings or {}
    if tp > 1:
        _write(run, "hs_replicas", tp, replicas if replicas is not None else _noisy(_hs_requests()),
               listing=listings.get("hs_replicas", "default"), replicas=replicas_ranks,
               **(replicas_extra or {}))
    (run).mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"tp": tp}))
    _write(run, "hs", tp, hs if hs is not None else _noisy(_hs_requests()),
           listing=listings.get("hs", "default"), **(hs_extra or {}))
    _write(run, "qk", tp, qk if qk is not None else _noisy(_qk_requests()),
           listing=listings.get("qk", "default"), **(qk_extra or {}))
    _write(run, "steer", tp, steer if steer is not None else _steer(),
           listing=listings.get("steer", "default"), **(steer_extra or {}))
    return run


def _failed(verdict, workload):
    return [c for c in verdict["checks"] if c["workload"] == workload and c["status"] == "FAIL"]


# --------------------------------------------------------------------------------------
# Controls that must PASS
# --------------------------------------------------------------------------------------

def test_identical_runs_pass(ref):
    v = probe.compare_runs(ref, ref)
    assert v["verdict"] == "PASS", v["failures"]
    assert v["measured"]["hs"]["worst_rel"] == 0.0 and v["measured"]["qk"]["worst_q_rel"] == 0.0


def test_in_band_tp_noise_passes_and_is_measured(ref, tmp_path):
    v = probe.compare_runs(ref, _cand(tmp_path))
    assert v["verdict"] == "PASS", v["failures"]
    assert 0 < v["measured"]["hs"]["worst_rel"] < probe.TP_HS_REL_BAND
    assert 0 < v["measured"]["qk"]["worst_head_rel"] < probe.TP_QK_HEAD_REL_BAND
    assert v["measured"]["steer"]["delta_rel"] == 0.0
    assert v["bands_overridden"] is False


def test_a_greedy_divergence_only_compares_the_common_prefix(ref, tmp_path):
    """After the trajectories pick different tokens the rows describe different inputs: the
    probe must compare only rows before the divergence, and say how many."""
    hs = _noisy(_hs_requests())
    for rec in hs:
        rec["token_ids"][2] += 1                       # diverge at generated token 2
        P = rec["prompt_token_ids"].numel()
        for layer in rec["layers"]:
            rec["layers"][layer][P + 2:] = 1e3         # rows after it: unrelated
    v = probe.compare_runs(ref, _cand(tmp_path, hs=hs), workloads=("hs",))
    assert v["verdict"] == "PASS", v["failures"]
    rows = [c["value"] for c in v["checks"] if c["name"].endswith("rows compared")]
    assert rows == [P + 2 for P in PROMPT_LENS]


# --------------------------------------------------------------------------------------
# Structural failures: exact, no tolerance involved
# --------------------------------------------------------------------------------------

def test_rank0_only_qk_capture_fails(ref, tmp_path):
    """THE regression the gate exists for: a TP4 capture holding rank 0's heads only."""
    narrow = _qk_requests(q_w=(H_Q // 4) * D, k_w=max(1, H_KV // 4) * D)
    cand = _cand(tmp_path, qk=narrow,
                 qk_extra={"flush_dirs": ["/ap/qk/aperture/tp_rank_0"]})
    v = probe.compare_runs(ref, cand, workloads=("qk",))
    assert v["verdict"] == "FAIL"
    names = {c["name"] for c in _failed(v, "qk")}
    assert "cand rank dirs" in names
    assert any("full-width shapes" in n for n in names)


def test_a_recorded_merge_error_fails(ref, tmp_path):
    cand = _cand(tmp_path, qk_extra={"merge_error": "TPShardError: missing tp_rank(s) [3]"})
    v = probe.compare_runs(ref, cand, workloads=("qk",))
    assert v["verdict"] == "FAIL"
    assert any(c["name"] == "cand TP merge" for c in _failed(v, "qk"))


@pytest.mark.parametrize("wl", ["hs", "qk"])
def test_a_missing_layer_fails(ref, tmp_path, wl):
    if wl == "hs":
        hs = _noisy(_hs_requests())
        del hs[1]["layers"][L]
        cand = _cand(tmp_path, hs=hs)
    else:
        qk = _noisy(_qk_requests())
        del qk[0]["layers"][0]
        cand = _cand(tmp_path, qk=qk)
    v = probe.compare_runs(ref, cand, workloads=(wl,))
    assert v["verdict"] == "FAIL"
    assert any("layer set" in c["name"] for c in _failed(v, wl))


def test_a_stray_hs_rank_dir_at_tp4_fails(ref, tmp_path):
    """Non-capturing HS ranks must write nothing: two data dirs means rank 1 wrote one."""
    cand = _cand(tmp_path, hs_extra={"flush_dirs": ["/a/tp_rank_0", "/a/tp_rank_1"]})
    v = probe.compare_runs(ref, cand, workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert any(c["name"] == "cand rank dirs" for c in _failed(v, "hs"))


def test_different_prompts_fail(ref, tmp_path):
    hs = _noisy(_hs_requests())
    hs[0]["prompt_token_ids"][1] += 1
    v = probe.compare_runs(ref, _cand(tmp_path, hs=hs), workloads=("hs",))
    assert v["verdict"] == "FAIL"


def test_a_tp2_reference_is_refused(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "hs", 2, _hs_requests())
    _write(b, "hs", 4, _hs_requests())
    v = probe.compare_runs(a, b, workloads=("hs",))
    assert v["verdict"] == "FAIL" and any(c["name"] == "ref is TP=1" for c in _failed(v, "hs"))


# --------------------------------------------------------------------------------------
# Value failures: outside the band
# --------------------------------------------------------------------------------------

def test_permuted_heads_fail_the_per_head_band(ref, tmp_path):
    """A rank-order or head-order mix-up keeps every shape right; only values can see it."""
    qk = _noisy(_qk_requests())
    e = qk[1]["layers"][2]
    e["q"] = torch.cat([e["q"][:, D:2 * D], e["q"][:, :D], e["q"][:, 2 * D:]], dim=-1)
    v = probe.compare_runs(ref, _cand(tmp_path, qk=qk), workloads=("qk",))
    assert v["verdict"] == "FAIL"
    assert v["measured"]["qk"]["worst_head_rel"] > probe.TP_QK_HEAD_REL_BAND


def test_a_zero_filled_layer_fails(ref, tmp_path):
    hs = _noisy(_hs_requests())
    hs[0]["layers"][3] = torch.zeros_like(hs[0]["layers"][3])
    v = probe.compare_runs(ref, _cand(tmp_path, hs=hs), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert v["measured"]["hs"]["worst_rel"] == pytest.approx(1.0)


def test_an_inert_steer_at_tp_n_fails(ref, tmp_path):
    dead = _steer(effect=0.0)
    v = probe.compare_runs(ref, _cand(tmp_path, steer=dead), workloads=("steer",))
    assert v["verdict"] == "FAIL"
    names = {c["name"] for c in _failed(v, "steer")}
    assert "cand steer alive (max|delta|)" in names


def test_an_inert_reference_steer_cannot_agree_at_zero(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "steer", 1, _steer(effect=0.0))
    _write(b, "steer", 4, _steer(effect=0.0))
    v = probe.compare_runs(a, b, workloads=("steer",))
    assert v["verdict"] == "FAIL"


def test_a_doubled_steer_fails(ref, tmp_path):
    """Steering applied once per rank and summed is the classic TP bug: the effect doubles."""
    s = _steer()
    for st, un in zip(s["steered"], s["unsteered"]):
        st["prompt_logprobs"] = un["prompt_logprobs"] + 2 * (st["prompt_logprobs"]
                                                             - un["prompt_logprobs"])
    v = probe.compare_runs(ref, _cand(tmp_path, steer=s), workloads=("steer",))
    assert v["verdict"] == "FAIL"
    assert v["measured"]["steer"]["delta_rel"] == pytest.approx(1.0)


def test_a_missing_workload_fails(ref, tmp_path):
    run = tmp_path / "tp4"
    _write(run, "hs", 4, _noisy(_hs_requests()))
    v = probe.compare_runs(ref, run)
    assert v["verdict"] == "FAIL"
    assert any(c["name"] == "both runs present" for c in _failed(v, "qk"))


# --------------------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------------------

def test_cli_prints_the_verdict_json_last_and_exits_by_verdict(ref, tmp_path, capsys):
    out = tmp_path / "verdict.json"
    rc = probe.main(["--compare", str(ref), str(_cand(tmp_path)), "--json-out", str(out)])
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 0 and json.loads(last)["verdict"] == "PASS"
    assert json.loads(out.read_text())["verdict"] == "PASS"
    bad = tmp_path / "bad"
    _write(bad, "hs", 4, [dict(r, layers={}) for r in _hs_requests()])
    rc = probe.main(["--compare", str(ref), str(bad), "--workloads", "hs"])
    assert rc == 1 and json.loads(capsys.readouterr().out.strip().splitlines()[-1])["verdict"] \
        == "FAIL"


def test_cli_usage_errors_exit_2(tmp_path):
    assert probe.main(["--compare", str(tmp_path / "nope"), str(tmp_path)]) == 2
    assert probe.main([]) == 2


def test_band_overrides_are_flagged(ref, tmp_path, capsys):
    probe.main(["--compare", str(ref), str(ref), "--hs-rel-band", "0.5"])
    v = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert v["bands_overridden"] is True and v["bands"]["hs_rel"] == 0.5


def test_the_fuse_knob_is_the_real_vllm_029_field():
    """--fuse must set a field vLLM 0.29's PassConfig actually has, to exactly True / False, and
    leave it unset for 'default'."""
    pytest.importorskip("vllm")
    from vllm.config.compilation import PassConfig
    assert "fuse_allreduce_rms" in PassConfig.__dataclass_fields__
    assert probe.FUSE_REQUEST == {"on": True, "off": False, "default": "vllm-default"}
    import inspect
    src = inspect.getsource(probe.run_child)
    assert 'compilation_config["pass_config"] = {"fuse_allreduce_rms": requested_fuse}' in src
    assert 'if requested_fuse != "vllm-default":' in src


def test_generation_record_reads_teacher_forced_prompt_logprobs():
    """The run side's extraction of the steer observable, against vLLM's output shapes: a list
    of per-position ``{token_id: Logprob}`` dicts with None at position 0."""
    from types import SimpleNamespace as NS

    def lp(v):
        return NS(logprob=v)

    out = NS(prompt_token_ids=[5, 6, 7], prompt_logprobs=[None, {6: lp(-1.5), 9: lp(-0.1)},
                                                          {7: lp(-2.0)}],
             request_id="0",
             outputs=[NS(token_ids=[11, 12], logprobs=[{11: lp(-0.3)}, {12: lp(-0.4)}])])
    rec = probe._generation_record(out, with_prompt_logprobs=True)
    assert rec["prompt_token_ids"].tolist() == [5, 6, 7]
    assert rec["token_ids"].tolist() == [11, 12]
    assert rec["token_logprobs"].tolist() == [-0.3, -0.4]
    pl = rec["prompt_logprobs"]
    assert pl[1:].tolist() == [-1.5, -2.0] and torch.isnan(pl[0])


def test_match_req_accepts_vllm_suffixed_ids_and_refuses_a_missing_request():
    art = {"0-abc": 1, "1": 2}
    assert probe._match_req(art, "0") == 1 and probe._match_req(art, "1") == 2
    with pytest.raises(RuntimeError, match="no rows"):
        probe._match_req(art, "2")


# --------------------------------------------------------------------------------------
# W2 (after gate G1): the aperture listing, the fusion request, the child environment
# --------------------------------------------------------------------------------------

def _failure_text(v, wl=None):
    return " | ".join(f for f in v["failures"] if wl is None or f.startswith(f"[{wl}]"))


def _rank0_only_hs(tp=4):
    """The manifest of an HS leg run with MIA_HS_TP_SHARD=0 (the pre-shard A/B layout)."""
    env = {"VLLM_PLUGINS": None, "MIA_APERTURE_PER_REQUEST": None,
           "VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "MIA_HS_TP_SHARD": "0"}
    return dict(_hs_manifest("hs", tp, "rank0"), env=env,
                writer=_writers("hs", tp, hs_rank0_only=True))


def test_a_stray_empty_hs_rank_dir_on_disk_fails_even_when_flush_hid_it(ref, tmp_path):
    """G1 could not check this: flush_aperture filters out dirs holding no rows, so an EMPTY
    tp_rank_1 a sink rank created never shows in flush_dirs. Only the on-disk listing sees it.
    (Sink ranks exist in the MIA_HS_TP_SHARD=0 layout.)"""
    cand = _cand(tmp_path, hs_extra=_rank0_only_hs(),
                 listings={"hs": _listing("hs", 4, {"tp_rank_0": 4096, "tp_rank_1": 0})})
    v = probe.compare_runs(ref, cand, workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert "cand aperture dirs on disk" in _failure_text(v, "hs")
    assert "STRAY tp_rank_1 (EMPTY)" in _failure_text(v, "hs")
    # ...and the flush-based check alone would have passed it:
    assert not any(c["name"] == "cand rank dirs" and c["status"] == "FAIL" for c in v["checks"])


@pytest.mark.parametrize("wl,rank_bytes,want", [
    ("qk", {"tp_rank_0": 10, "tp_rank_1": 10, "tp_rank_2": 10}, "missing ['tp_rank_3']"),
    ("qk", {"tp_rank_0": 10, "tp_rank_1": 10, "tp_rank_2": 0, "tp_rank_3": 10},
     "expected dirs holding no data ['tp_rank_2']"),
    ("steer", {"tp_rank_0": 0}, "STRAY tp_rank_0 (EMPTY)"),
    ("hs", {}, "missing ['tp_rank_0', 'tp_rank_1', 'tp_rank_2', 'tp_rank_3']"),
    ("hs", {"tp_rank_0": 10, "tp_rank_1": 10, "tp_rank_3": 10}, "missing ['tp_rank_2']"),
    ("hs", {f"tp_rank_{r}": 10 for r in range(5)}, "STRAY tp_rank_4 (74 B)"),
])
def test_the_on_disk_rank_dirs_must_be_exactly_the_expected_ones(ref, tmp_path, wl, rank_bytes,
                                                                 want):
    cand = _cand(tmp_path, listings={wl: _listing(wl, 4, rank_bytes)})
    v = probe.compare_runs(ref, cand, workloads=(wl,))
    assert v["verdict"] == "FAIL" and want in _failure_text(v, wl)


def test_a_run_without_an_aperture_listing_fails(ref, tmp_path):
    """No listing -> the stray-dir check cannot run -> FAIL, never a silent pass."""
    v = probe.compare_runs(ref, _cand(tmp_path, listings={"hs": None}), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert "no aperture_listing.json" in _failure_text(v, "hs")


def test_list_aperture_dir_records_every_rank_dir_file_and_size(tmp_path):
    root = tmp_path / "aperture"
    (root / "tp_rank_0").mkdir(parents=True)
    (root / "tp_rank_0" / "hs_layer_1.raw").write_bytes(b"x" * 40)
    (root / "tp_rank_0" / "hs_aperture_meta.jsonl").write_bytes(b"{}\n")
    (root / "tp_rank_1").mkdir()                    # the stray empty dir G1 could not look for
    (root / "stray.txt").write_bytes(b"ab")
    got = probe.list_aperture_dir(root)
    assert got["exists"] is True
    assert got["rank_dirs"]["tp_rank_0"]["total_bytes"] == 43
    assert {f["name"]: f["bytes"] for f in got["rank_dirs"]["tp_rank_0"]["files"]} == {
        "hs_layer_1.raw": 40, "hs_aperture_meta.jsonl": 3}
    assert got["rank_dirs"]["tp_rank_1"] == {"files": [], "total_bytes": 0}
    assert got["other"] == [{"name": "stray.txt", "is_dir": False, "bytes": 2}]
    assert probe.list_aperture_dir(tmp_path / "nope")["exists"] is False


def test_the_parent_lists_the_aperture_after_each_child_even_a_failed_one(tmp_path, monkeypatch):
    """The listing is written by the parent once the child has exited (every rank's teardown has
    run) and before any caller can delete the scratch -- also for a child that crashed."""
    calls = []

    def fake_run(argv, env, **kw):
        wl = argv[argv.index("--child-workload") + 1]
        calls.append((wl, env))
        d = Path(env["MIA_APERTURE_DIR"]) / "tp_rank_1"
        d.mkdir(parents=True, exist_ok=True)             # what a crashed rank left behind
        return type("R", (), {"returncode": 1})()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    out = tmp_path / "leg"
    args = probe._parser().parse_args(["--tp", "2", "--out", str(out), "--workloads", "hs,qk",
                                       "--fuse", "on"])
    args.fuse = "on"
    assert probe.run_parent(args) == 1
    for wl in ("hs", "qk"):
        listing = json.loads((out / wl / probe.APERTURE_LISTING).read_text())
        assert listing["rank_dirs"] == {"tp_rank_1": {"files": [], "total_bytes": 0}}
    top = json.loads((out / "manifest.json").read_text())
    assert top["fuse"] == "on" and top["requested_fuse_allreduce_rms"] is True
    assert top["workloads"]["hs"]["aperture_rank_dirs"] == {"tp_rank_1": 0}


# ---- fusion -------------------------------------------------------------------------------

def test_a_leg_that_requests_fusion_and_resolves_false_fails(ref, tmp_path):
    """G1's 'fusion default' legs resolved False on every rank and nothing noticed. A leg that
    ASKS for fusion must fail when it does not get it."""
    off = {"resolved": {"fuse_allreduce_rms": False}, "fusion_active": False,
           "fusion_pass": [{"rank": r, "passes": []} for r in range(4)]}
    cand = _cand(tmp_path, hs_extra=off, qk_extra=off, steer_extra=off)
    v = probe.compare_runs(ref, cand)
    assert v["verdict"] == "FAIL"
    for wl in ("hs", "qk", "steer"):
        assert "fusion not active" in _failure_text(v, wl)


def test_a_config_true_but_a_disarmed_rank_is_not_fusion(ref, tmp_path):
    """An explicit fuse_allreduce_rms=True stays True in the config while the pass disables itself
    (no flashinfer comm, workspace init failed). The per-rank pass state decides."""
    half = {"fusion_pass": _armed(4, disabled_ranks=(2,)), "fusion_active": False}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=half), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert "fusion not active" in _failure_text(v, "hs")
    assert "unarmed on rank(s) [2]" in _failure_text(v, "hs")


def test_requested_fusion_that_is_armed_on_every_rank_passes(ref, tmp_path):
    v = probe.compare_runs(ref, _cand(tmp_path))
    assert v["verdict"] == "PASS", v["failures"]
    c = [c for c in v["checks"] if c["name"] == "cand fusion active (requested on)"]
    assert len(c) == 4 and all(x["status"] == "PASS" for x in c)
    assert v["cand"]["fusion_active"] == {"hs": True, "qk": True, "steer": True,
                                          "hs_replicas": True}


@pytest.mark.parametrize("res,evidence,ok,want", [
    (False, "unarmed", True, None),
    (False, "w3_blind", False, "no per-rank evidence that the pass is not armed"),
    (True, "unarmed", False, "vLLM resolved True"),
    (False, "armed", False, "the pass is armed anyway"),
    (True, "armed", False, "vLLM resolved True"),
])
def test_a_nofuse_leg_must_really_be_unfused(ref, tmp_path, res, evidence, ok, want):
    """'off' needs POSITIVE per-rank evidence too: the G1 re-run's nofuse legs PASSED on a pass
    list the gc scan could not have filled either way (vacuous)."""
    fp = {"unarmed": _unarmed(4), "w3_blind": _w3_blind(4), "armed": _armed(4)}[evidence]
    m = {"requested_fuse_allreduce_rms": False, "resolved": {"fuse_allreduce_rms": res},
         "fusion_pass": fp, "fusion_active": False}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=m), workloads=("hs",))
    assert (v["verdict"] == "PASS") is ok, v["failures"]
    if want:
        assert want in _failure_text(v, "hs")


def test_the_vllm_default_leg_is_reported_not_asserted(ref, tmp_path):
    m = {"requested_fuse_allreduce_rms": "vllm-default", "resolved": {"fuse_allreduce_rms": False},
         "fusion_active": False}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=m), workloads=("hs",))
    assert v["verdict"] == "PASS", v["failures"]
    info = [c for c in v["checks"] if c["name"] == "cand fusion (vLLM default: not asserted)"]
    assert info and info[0]["status"] == "INFO" and info[0]["value"] is False


def test_tp1_is_exempt_on_either_side(tmp_path):
    """vLLM never fuses at TP=1: a TP=1 side asked for fusion is reported, never failed."""
    a, b = tmp_path / "a", tmp_path / "b"
    on_tp1 = {"requested_fuse_allreduce_rms": True, "resolved": {"fuse_allreduce_rms": True},
              "fusion_active": False, "fusion_pass": [{"rank": 0, "passes": [{"disabled": True}]}]}
    _write(a, "hs", 1, _hs_requests(), **on_tp1)
    _write(b, "hs", 1, _hs_requests(), **on_tp1)
    v = probe.compare_runs(a, b, workloads=("hs",))
    assert v["verdict"] == "PASS", v["failures"]
    names = {c["name"]: c["status"] for c in v["checks"]}
    assert names["ref fusion (TP=1: exempt)"] == "INFO"
    assert names["cand fusion (TP=1: exempt)"] == "INFO"


def test_fusion_active_needs_evidence_from_every_rank():
    assert probe._fusion_active(_armed(4)) is True
    assert probe._fusion_active(_armed(4, disabled_ranks=(3,))) is False
    assert probe._fusion_active(_unarmed(4)) is False        # configured pass manager, no pass
    assert probe._fusion_active([]) is None
    assert probe._fusion_active(None) is None
    assert probe._fusion_active([{"error": "collective_rpc failed"}]) is None
    assert probe._fusion_active(_armed(3) + [{"error": "x"}]) is None


def test_an_empty_pass_list_is_no_evidence_never_not_armed():
    """THE G1 re-run bug: the gc scan returned ``passes: []`` on every rank, fused or not, and
    _fusion_active read that as False -- "not armed" on the fused legs (a false FAIL) and "not
    armed" on the nofuse legs (a vacuous PASS). An empty list with no configured pass manager
    seen says nothing."""
    assert probe._fusion_active([{"rank": 0, "passes": []}]) is None
    assert probe._fusion_active(_w3_blind(2)) is None
    assert probe._fusion_active([_rank_rec(0, [], configured=0)]) is None
    assert probe._fusion_active(_unarmed(1) + _w3_blind(2)[1:]) is None


def test_a_w3_record_fails_a_fused_leg_for_want_of_evidence_not_for_being_unarmed(ref, tmp_path):
    """Re-judged under the fixed rule, the G1 re-run's fused legs FAIL for the right reason --
    no evidence, naming the blind scan -- and their stored ``fusion_active: false`` is ignored."""
    m = {"fusion_pass": _w3_blind(4), "fusion_active": False}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=m), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    text = _failure_text(v, "hs")
    assert "no per-rank evidence that the pass is armed" in text
    assert "pre-fix record from the gc.get_objects() scan" in text
    assert "unarmed on rank(s)" not in text
    assert v["cand"]["fusion_armed"] == {"hs": None} == v["cand"]["fusion_active"]


# ---- the fusion locator: reach the pass through the compiled model, never a gc scan --------

@pytest.fixture
def fake_arf(monkeypatch):
    """A stand-in for vllm.compilation.passes.fusion.allreduce_rms_fusion: the real module needs
    vLLM's compiled _C ops, which a GPU-free node cannot import. Its names are pinned against the
    installed source in test_the_locator_walks_vllm_029s_own_names."""
    import sys
    import types

    fake = types.ModuleType("vllm.compilation.passes.fusion.allreduce_rms_fusion")

    class AllReduceFusionPass:
        pass

    fake.AllReduceFusionPass, fake.flashinfer_comm = AllReduceFusionPass, object()
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    pkg = types.ModuleType("vllm.compilation.passes.fusion")
    pkg.allreduce_rms_fusion = fake
    monkeypatch.setitem(sys.modules, pkg.__name__, pkg)
    return AllReduceFusionPass


def _fake_pass(cls, disabled=False, matched=None, tp_size=4):
    """An AllReduceFusionPass as vLLM 0.29 leaves it: ``disabled`` set in __init__,
    ``max_token_num`` only when it armed, ``matched_count`` an instance attribute only once
    __call__ ran."""
    p = cls()
    p.disabled, p.tp_size = disabled, tp_size
    if not disabled:
        p.max_token_num = 2048
    if matched is not None:
        p.matched_count = matched
    return p


class _PassManager:
    """PostGradPassManager stand-in: ``passes`` from __init__, ``pass_config`` from configure()."""

    def __init__(self, passes, configured=True):
        self.passes = list(passes)
        if configured:
            self.pass_config = object()


class _NoOpEliminationPass:
    pass


class _LlamaModel(torch.nn.Module):
    def forward(self, x):
        return x


class _LlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LlamaModel()


def _compiled_model(backend, route="aot", loaded=False):
    """A model whose torch.compile'd submodule (``model``) was compiled by ``backend``, the way
    vLLM 0.29 leaves it: AOT (``aot_compiled_fn._artifacts.compiled_fn`` = the
    VllmSerializableFunction holding ``vllm_backend``) or not (the function only in the
    ``__compiled_fn_*`` global dynamo installed -- set up by the caller)."""
    from types import SimpleNamespace as NS

    top = _LlamaForCausalLM()
    top.model._compiled_callable = lambda *a, **k: None
    if route == "aot":
        top.model.aot_compiled_fn = NS(_artifacts=NS(compiled_fn=NS(vllm_backend=backend)))
        top.model.was_aot_compile_fn_loaded_from_disk = loaded
    return top


def _worker(rank, model, fuse=True):
    from types import SimpleNamespace as NS

    return NS(rank=rank, get_model=lambda: model, vllm_config=NS(compilation_config=NS(
        pass_config=NS(fuse_allreduce_rms=fuse))))


def _backend(passes, configured=True):
    from types import SimpleNamespace as NS

    return NS(pass_manager=_PassManager(passes, configured))


def test_the_pass_is_found_after_vllm_freezes_the_gc_heap(fake_arf):
    """THE G1 re-run bug. vLLM 0.29's compile_or_warm_up_model ends with freeze_gc_heap()
    (gc.freeze()), and CPython 3.12's gc.get_objects() skips frozen objects, so the old scan found
    no pass on any rank. The locator reaches it through the compiled model instead."""
    import gc

    armed = _fake_pass(fake_arf, matched=2)
    worker = _worker(0, _compiled_model(_backend([_NoOpEliminationPass(), armed])))
    gc.collect()
    gc.freeze()
    try:
        assert not [o for o in gc.get_objects() if isinstance(o, fake_arf)], \
            "premise: a frozen pass is invisible to gc.get_objects()"
        got = probe._fusion_pass_state(worker)
    finally:
        gc.unfreeze()
    assert got["backends_configured"] == 1 and got["locator"] == probe.FUSION_LOCATOR
    assert got["passes"] == [{"module": "model", "disabled": False, "max_token_num": 2048,
                              "tp_size": 4, "called": True, "matched_count_last_call": 2}]
    c = got["compiled"][0]
    assert c["route"] == "aot_compiled_fn._artifacts.compiled_fn.vllm_backend"
    assert c["pass_names"] == ["_NoOpEliminationPass", "AllReduceFusionPass"]
    assert c["loaded_from_disk"] is False and c["pass_manager_configured"] is True
    assert got["matched_count_total"] is None and "unavailable" in got["matched_count_total_note"]
    assert probe._fusion_active([got]) is True
    assert probe._fusion_applied([got])["last_call_by_rank"] == {"0": 2}


def test_a_disabled_pass_is_recorded_and_fails_a_fused_leg(fake_arf, ref, tmp_path):
    """An explicit fuse_allreduce_rms=True stays True in the config while the pass disables itself
    (no flashinfer comm, workspace init failed) -- here on rank 1 only, which vLLM's warning_once
    (local rank 0) would never have logged."""
    recs = [probe._fusion_pass_state(_worker(r, _compiled_model(_backend(
        [_fake_pass(fake_arf, disabled=(r == 1), matched=None if r == 1 else 2)]))))
        for r in range(4)]
    assert [x["passes"][0]["disabled"] for x in recs] == [False, True, False, False]
    assert recs[1]["passes"][0] == {"module": "model", "disabled": True, "max_token_num": None,
                                    "tp_size": 4, "called": False, "matched_count_last_call": None}
    assert probe._fusion_active(recs) is False
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra={"fusion_pass": recs}), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    text = _failure_text(v, "hs")
    assert "the pass is not armed on every rank" in text and "unarmed on rank(s) [1]" in text


def test_an_unfused_leg_is_positive_evidence_of_no_armed_pass(fake_arf, ref, tmp_path):
    """--fuse off: every rank's pass manager is configured and simply holds no AllReduceFusionPass
    -- evidence, unlike the gc scan's empty list."""
    recs = [probe._fusion_pass_state(_worker(r, _compiled_model(_backend([_NoOpEliminationPass()])),
                                             fuse=False)) for r in range(4)]
    assert all(x["passes"] == [] and x["backends_configured"] == 1 for x in recs)
    assert probe._fusion_active(recs) is False
    m = {"requested_fuse_allreduce_rms": False, "resolved": {"fuse_allreduce_rms": False},
         "fusion_pass": recs}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=m), workloads=("hs",))
    assert v["verdict"] == "PASS", v["failures"]


def test_a_compile_loaded_from_cache_is_no_evidence_not_an_unarmed_pass(fake_arf):
    """vLLM rebuilds a cached AOT artifact with vllm_backend=None: no pass manager ran in this
    process, so there is nothing to judge -- None, with the reason, never False."""
    got = probe._fusion_pass_state(_worker(0, _compiled_model(None, loaded=True)))
    assert got["backends_configured"] == 0 and got["passes"] == []
    assert "rebuilt from vLLM's compile cache" in got["compiled"][0]["note"]
    assert got["compiled"][0]["loaded_from_disk"] is True
    assert probe._fusion_active([got]) is None
    assert "compile cache" in " ".join(probe._fusion_evidence_gaps([got], 1))
    # a backend whose pass manager never ran configure() is no evidence either
    unconf = probe._fusion_pass_state(_worker(0, _compiled_model(_backend([], configured=False))))
    assert unconf["backends_configured"] == 0 and probe._fusion_active([unconf]) is None


def test_the_non_aot_route_reaches_the_backend_through_dynamos_global(fake_arf, monkeypatch):
    """With VLLM_USE_AOT_COMPILE=0 the compiled function lives only in the ``__compiled_fn_*``
    global dynamo installs in forward's module, wrapped by torch._dynamo.disable (which keeps it
    as ``_torchdynamo_orig_callable``)."""
    from types import SimpleNamespace as NS

    serializable = NS(vllm_backend=_backend([_fake_pass(fake_arf, matched=1)]))

    def disabled_wrapper(*a, **k):
        return None

    disabled_wrapper._torchdynamo_orig_callable = serializable
    monkeypatch.setitem(_LlamaModel.forward.__globals__, "__compiled_fn_1_0123abcd",
                        disabled_wrapper)
    got = probe._fusion_pass_state(_worker(0, _compiled_model(None, route="dynamo")))
    assert got["compiled"][0]["route"].startswith("forward.__globals__['__compiled_fn_1_0123abcd']")
    assert got["backends_configured"] == 1 and probe._fusion_active([got]) is True


def test_no_compiled_module_or_a_broken_worker_is_no_evidence(fake_arf):
    from types import SimpleNamespace as NS

    got = probe._fusion_pass_state(_worker(0, torch.nn.Linear(2, 2)))    # nothing compiled
    assert got["note"] == "worker.get_model() has no torch.compile'd submodule"
    assert got["compiled"] == [] and probe._fusion_active([got]) is None

    def boom():
        raise RuntimeError("no model")

    broken = probe._fusion_pass_state(NS(rank=0, get_model=boom, vllm_config=None))
    assert "no model" in broken["error"] and probe._fusion_active([broken]) is None


def test_the_locator_walks_vllm_029s_own_names():
    """Every attribute the locator follows, and every claim FUSION_LOCATOR and
    FUSION_TOTAL_UNAVAILABLE make, pinned against the INSTALLED vLLM and torch sources (read, not
    imported: the pass module needs vLLM's compiled _C ops)."""
    import ast
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None:
        pytest.skip("vLLM not installed")
    root = Path(spec.origin).parent
    src = {rel: (root / rel).read_text() for rel in (
        "compilation/decorators.py", "compilation/wrapper.py", "compilation/caching.py",
        "compilation/backends.py", "compilation/passes/pass_manager.py",
        "compilation/passes/fusion/allreduce_rms_fusion.py", "v1/worker/gpu_worker.py")}

    def cls_src(rel, name):
        tree = ast.parse(src[rel])
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        return ast.unparse(node)

    # the compiled submodule and the AOT artifact
    assert "self._compiled_callable = torch.compile(" in src["compilation/wrapper.py"]
    assert "self.aot_compiled_fn = self.aot_compile(" in src["compilation/decorators.py"]
    assert "self.was_aot_compile_fn_loaded_from_disk = True" in src["compilation/decorators.py"]
    # VllmSerializableFunction carries the backend; the mega-artifact reload passes None
    assert "self.vllm_backend = vllm_backend" in cls_src("compilation/caching.py",
                                                         "VllmSerializableFunction")
    assert "vllm_backend=None" in src["compilation/caching.py"]
    backend = cls_src("compilation/backends.py", "VllmBackend")
    assert "self.pass_manager = resolve_obj_by_qualname(" in backend
    assert "vllm_backend=self" in backend and "self.pass_manager.configure(self.vllm_config)" \
        in backend
    # configure() is what sets pass_config ("configured"), and adds the pass
    pm = cls_src("compilation/passes/pass_manager.py", "PostGradPassManager")
    assert "self.passes: list[InductorPass] = []" in pm
    assert "self.pass_config = config.compilation_config.pass_config" in pm
    assert "self.passes += [AllReduceFusionPass(config)]" in pm
    # the pass: armed/disabled, and a per-call count that is never totalled
    arf = cls_src("compilation/passes/fusion/allreduce_rms_fusion.py", "AllReduceFusionPass")
    for attr in ("self.disabled = True", "self.disabled = False", "self.max_token_num",
                 "self.tp_size", "self.matched_count = self.patterns.apply(graph)"):
        assert attr in arf, attr
    assert "match_table" not in arf, "AllReduceFusionPass now keeps a total: use it"
    tree = ast.parse(src["compilation/passes/fusion/allreduce_rms_fusion.py"])
    assert any(isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "flashinfer_comm"
               for n in tree.body)
    # why a gc scan cannot work, and how the worker hands out its model
    worker = cls_src("v1/worker/gpu_worker.py", "Worker")
    assert "freeze_gc_heap()" in worker and "def get_model(self)" in worker
    # torch's side: the AOT artifact and dynamo's installed global
    tdir = Path(torch.__file__).parent / "_dynamo"
    aot = (tdir / "aot_compile.py").read_text()
    assert "compiled_fn: SerializableCallable" in aot and "_artifacts: CompileArtifacts" in aot
    assert 'unique_id("__compiled_fn"' in (tdir / "output_graph.py").read_text()
    assert "_fn._torchdynamo_orig_callable = fn" in (tdir / "eval_frame.py").read_text()


# ---- APPLIED: reported apart from ARMED, never asserted -----------------------------------

def test_fusion_applied_statuses():
    assert probe._fusion_applied(_armed(2))["status"] == "applied"
    assert probe._fusion_applied(_armed(2, counts={0: 1, 1: 0}))["status"] == "mixed"
    assert probe._fusion_applied(_armed(2, counts={0: 0, 1: 0}))["status"] == "none"
    assert probe._fusion_applied(_armed(2, counts={1: None}))["status"] == "unknown"
    assert probe._fusion_applied(_w3_blind(2))["status"] == "unknown"
    assert probe._fusion_applied(None)["status"] == "unknown"
    got = probe._fusion_applied(_armed(2, disabled_ranks=(1,)))
    assert got["last_call_by_rank"] == {"0": 2, "1": None}
    assert got["total"] is None and got["total_note"] == probe.FUSION_TOTAL_UNAVAILABLE


@pytest.mark.parametrize("wl", ["hs", "qk", "steer"])
def test_applied_is_reported_apart_from_armed_and_never_fails(ref, tmp_path, wl):
    """ARMED is the gate; APPLIED (the last pass call's count per rank) is recorded next to it. A
    fused HS leg is EXPECTED to be partial (capture_hs consumes each down_proj all-reduce output):
    recorded, never failed."""
    m = {"fusion_pass": _armed(4, counts={0: 1, 1: 0, 2: 1, 3: 1})}
    v = probe.compare_runs(ref, _cand(tmp_path, **{f"{wl}_extra": m}), workloads=(wl,))
    assert v["verdict"] == "PASS", v["failures"]
    by = {c["name"]: c for c in v["checks"]}
    assert by["cand fusion active (requested on)"]["status"] == "PASS"
    applied = by["cand fusion applied (requested on)"]
    assert applied["status"] == "INFO" and applied["value"] == "mixed"
    assert v["cand"]["fusion_armed"] == {wl: True}
    assert v["cand"]["fusion_applied"][wl]["last_call_by_rank"] == {"0": 1, "1": 0, "2": 1, "3": 1}
    if wl == "hs":
        assert "PARTIAL fusion is EXPECTED and never failed" in applied["detail"]


# ---- noise legs: every leg against every TP=1 reference; the TP=2 repeat --------------------

def _repeat_of(src: Path, dst: Path, hs=None) -> Path:
    """A second boot of ``src``: bit-identical captures unless ``hs`` replaces the HS one."""
    import shutil

    shutil.copytree(src, dst)
    if hs is not None:
        torch.save({"workload": "hs", "requests": hs}, dst / "hs" / probe.CAPTURE_FILE)
    return dst


def test_a_leg_is_judged_against_every_reference_and_says_which(ref, tmp_path):
    """The G1 re-run's first tp1 HS capture was 1.318e-2 away from tp1_repeat: which TP=1 boot a
    leg is judged against matters. Here tp1_repeat is the outlier boot; the candidate is in band of
    tp1 and out of band of tp1_repeat, so it FAILS, and the verdict says against which."""
    rep = _repeat_of(ref, tmp_path / "tp1_repeat", hs=_noisy(_hs_requests(), scale=4e-2, seed=7))
    cand = _cand(tmp_path, hs=_noisy(_hs_requests(), scale=4e-2, seed=8))
    v = probe.compare_legs([ref, rep], cand)
    assert v["ref_labels"] == ["tp1", "tp1_repeat"]
    assert [r["label"] for r in v["refs"]] == ["tp1", "tp1_repeat"] and v["ref"]["label"] == "tp1"
    assert v["by_ref"]["tp1"]["verdict"] == "PASS"
    assert v["by_ref"]["tp1_repeat"]["verdict"] == "FAIL"
    assert v["verdict"] == "FAIL"
    assert v["failures"] and all(f.startswith("[hs] (vs tp1_repeat) ") for f in v["failures"])
    assert v["measured"] == v["by_ref"]["tp1"]["measured"]
    assert v["by_ref"]["tp1_repeat"]["measured"]["hs"]["worst_rel"] > probe.TP_HS_REL_BAND
    assert {c["ref"] for c in v["checks"]} == {"tp1", "tp1_repeat"}
    assert v["refs_identical"] == {"hs": False, "qk": True, "steer": True, "hs_replicas": None}
    shas = [r["artifact_sha256"] for r in v["refs"]]
    assert shas[0]["qk"] == shas[1]["qk"] and shas[0]["hs"] != shas[1]["hs"]
    assert v["cand"]["label"] == "tp4" and len(v["cand"]["artifact_sha256"]["hs"]) == 64


def test_one_reference_keeps_the_old_verdict_and_names_it(ref, tmp_path):
    v = probe.compare_legs([ref], _cand(tmp_path))
    assert v["verdict"] == "PASS" and v["ref_labels"] == ["tp1"] and "refs_identical" not in v
    assert not any("(vs " in f for f in v["failures"])
    assert v["mode"] == "parity"


def test_cli_takes_several_references(ref, tmp_path, capsys):
    rep = _repeat_of(ref, tmp_path / "tp1_repeat")
    out = tmp_path / "verdict.json"
    rc = probe.main(["--compare", str(ref), str(rep), str(_cand(tmp_path)), "--json-out", str(out)])
    v = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and v["verdict"] == "PASS" and v["ref_labels"] == ["tp1", "tp1_repeat"]
    assert v["refs_identical"] == {"hs": True, "qk": True, "steer": True, "hs_replicas": None}
    assert json.loads(out.read_text())["by_ref"]["tp1_repeat"]["verdict"] == "PASS"
    # the third TP=1 boot against the two before it
    rep2 = _repeat_of(ref, tmp_path / "tp1_repeat2")
    assert probe.main(["--compare", str(ref), str(rep), str(rep2)]) == 0
    assert probe.main(["--compare", str(ref)]) == 2                         # no candidate
    assert probe.main(["--repeat", "--tp", "1", "--out", str(tmp_path / "x")]) == 2


def test_repeat_mode_measures_tp2_boot_to_boot_noise(tmp_path):
    """tp2 vs tp2_repeat: the same configuration twice. Parity mode refuses a TP=2 reference;
    --repeat compares like with like and judges BOTH sides' fusion."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    tp2 = _cand(tmp_path / "a", tp=2)
    s = _steer()
    for st, un in zip(s["steered"], s["unsteered"]):          # a boot-to-boot wobble in steer
        st["prompt_logprobs"] = un["prompt_logprobs"] + 1.004 * (st["prompt_logprobs"]
                                                                 - un["prompt_logprobs"])
    rep = _cand(tmp_path / "b", tp=2, hs=_noisy(_hs_requests(), seed=5), steer=s)
    v = probe.compare_legs([tp2], rep, repeat=True)
    assert v["verdict"] == "PASS", v["failures"]
    assert v["mode"] == "repeat"
    assert v["measured"]["steer"]["delta_rel"] == pytest.approx(4e-3)
    st = {c["name"]: c["status"] for c in v["checks"] if c["workload"] == "steer"}
    assert st["cand repeats ref (same TP and fusion request)"] == "PASS"
    assert st["ref fusion active (requested on)"] == "PASS"
    assert st["cand fusion active (requested on)"] == "PASS"
    assert "ref is TP=1" not in st
    parity = probe.compare_legs([tp2], rep)
    assert parity["verdict"] == "FAIL" and "ref is TP=1" in " ".join(parity["failures"])
    assert probe.main(["--repeat", "--compare", str(tp2), str(rep)]) == 0


@pytest.mark.parametrize("cand_tp,cand_fuse", [(4, True), (2, False)])
def test_a_repeat_must_be_the_same_configuration(tmp_path, cand_tp, cand_fuse):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    tp2 = _cand(tmp_path / "a", tp=2)
    extra = {"requested_fuse_allreduce_rms": cand_fuse,
             "resolved": {"fuse_allreduce_rms": cand_fuse},
             "fusion_pass": _armed(cand_tp) if cand_fuse else _unarmed(cand_tp)}
    other = _cand(tmp_path / "b", tp=cand_tp, hs_extra=extra)
    v = probe.compare_legs([tp2], other, workloads=("hs",), repeat=True)
    assert v["verdict"] == "FAIL"
    assert "cand repeats ref (same TP and fusion request)" in _failure_text(v, "hs")


def test_repeat_mode_holds_the_reference_to_its_fusion_request_too(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    blind = _cand(tmp_path / "a", tp=2, hs_extra={"fusion_pass": _w3_blind(2)})
    rep = _cand(tmp_path / "b", tp=2)
    v = probe.compare_legs([blind], rep, workloads=("hs",), repeat=True)
    assert v["verdict"] == "FAIL"
    assert "ref fusion active (requested on)" in _failure_text(v, "hs")
    assert "cand fusion active (requested on)" not in _failure_text(v, "hs")


# ---- the child environment ----------------------------------------------------------------

def test_child_env_drops_vllm_plugins_and_per_request_mode(monkeypatch, tmp_path):
    """W1: the probe used to setdefault VLLM_PLUGINS (so an inherited "" could disarm MIA) and to
    inherit MIA_APERTURE_PER_REQUEST."""
    monkeypatch.setenv("VLLM_PLUGINS", "")
    monkeypatch.setenv("MIA_APERTURE_PER_REQUEST", "1")
    args = probe._parser().parse_args(["--tp", "2", "--out", str(tmp_path)])
    env = probe._child_env(args, "hs", tmp_path)
    assert "VLLM_PLUGINS" not in env and "MIA_APERTURE_PER_REQUEST" not in env
    assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    monkeypatch.delenv("VLLM_PLUGINS")
    assert "VLLM_PLUGINS" not in probe._child_env(args, "hs", tmp_path)   # never defaulted


def test_child_env_gives_every_workload_its_own_compile_cache(monkeypatch, tmp_path):
    """W8 (LSF 1794720): the probe gave each LEG one VLLM_CACHE_ROOT, so `hs` and
    `hs_replicas` shared it; the second engine loaded the first's torch_aot_compile artifact
    and died on `expected size 16385==1 / 16385==65537` -- the two HS aperture geometries.
    MIA's compile-cache stamp now carries that geometry; this is the defence in depth."""
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    args = probe._parser().parse_args(["--tp", "4", "--out", str(tmp_path / "tp4")])
    roots = {wl: probe._child_env(args, wl, tmp_path)["VLLM_CACHE_ROOT"]
             for wl in ("hs", "qk", "steer", "hs_replicas")}
    assert len(set(roots.values())) == 4, f"workloads share a compile cache: {roots}"
    for wl, root in roots.items():
        assert Path(root).name == wl and Path(root).parent.name == "tp4"

    # An inherited root is honoured as the ROOT (callers put the cache on node-local disk
    # that way) but never used verbatim -- verbatim is what let the workloads collide.
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "nvme"))
    inherited = {wl: probe._child_env(args, wl, tmp_path)["VLLM_CACHE_ROOT"]
                 for wl in ("hs", "hs_replicas")}
    assert len(set(inherited.values())) == 2
    for root in inherited.values():
        assert str(tmp_path / "nvme") in root

    # Two LEGS that share a --scratch still get separate caches: the leg name comes from --out.
    other = probe._parser().parse_args(["--tp", "1", "--out", str(tmp_path / "tp1")])
    assert (probe._child_env(other, "hs", tmp_path)["VLLM_CACHE_ROOT"]
            != probe._child_env(args, "hs", tmp_path)["VLLM_CACHE_ROOT"])


@pytest.mark.parametrize("extra,want", [
    ({"env": {"VLLM_PLUGINS": "", "MIA_APERTURE_PER_REQUEST": None}}, "must be unset"),
    ({"env": {"VLLM_PLUGINS": None, "MIA_APERTURE_PER_REQUEST": "1"}}, "must be unset"),
    ({"env": None}, "not recorded"),
    ({"mia_plugin_loaded": False}, "did not patch vLLM"),
])
def test_a_child_that_ran_with_the_wrong_env_fails(ref, tmp_path, extra, want):
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=extra), workloads=("hs",))
    assert v["verdict"] == "FAIL" and want in _failure_text(v, "hs")


# ---- CLI ----------------------------------------------------------------------------------

@pytest.mark.parametrize("cli,want", [([], "default"), (["--fuse", "on"], "on"),
                                      (["--fuse", "off"], "off"),
                                      (["--no-fuse-allreduce-rms"], "off")])
def test_the_fuse_flag_reaches_the_child(tmp_path, monkeypatch, cli, want):
    seen = []
    monkeypatch.setattr(probe, "run_parent", lambda a: seen.append(a.fuse) or 0)
    assert probe.main(["--tp", "2", "--out", str(tmp_path)] + cli) == 0
    assert seen == [want]
    args = probe._parser().parse_args(["--tp", "2", "--out", str(tmp_path)])
    args.fuse = want
    argv = probe._child_argv(args, "hs")
    assert argv[argv.index("--fuse") + 1] == want


def test_contradictory_fuse_flags_are_a_usage_error(tmp_path):
    assert probe.main(["--tp", "2", "--out", str(tmp_path), "--fuse", "on",
                       "--no-fuse-allreduce-rms"]) == 2


# ---- the bands re-derived from G1 (TOLERANCES.md "TP parity probe") ----------------------

def test_g1_sized_hs_noise_still_passes(ref, tmp_path):
    """The worst HS error G1 measured (1.6677e-2, tp4_nofuse layer 32) must pass with margin."""
    v = probe.compare_runs(ref, _cand(tmp_path, hs=_noisy(_hs_requests(), scale=1.6677e-2)),
                           workloads=("hs",))
    assert v["verdict"] == "PASS", v["failures"]


def test_an_hs_error_the_old_derived_band_admitted_now_fails(ref, tmp_path):
    """8e-2 relative is 4.8x G1's worst layer: inside the pre-G1 band (1e-1), a finding now."""
    v = probe.compare_runs(ref, _cand(tmp_path, hs=_noisy(_hs_requests(), scale=8e-2)),
                           workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert 5e-2 < v["measured"]["hs"]["worst_rel"] < 1e-1


def test_a_ten_percent_steer_scale_error_now_fails(ref, tmp_path):
    """A steer 10% too strong on TP=N (delta rel 0.1) passed the pre-G1 band (2.5e-1); G1's
    worst measured delta rel is 4.3e-3."""
    s = _steer()
    for st, un in zip(s["steered"], s["unsteered"]):
        st["prompt_logprobs"] = un["prompt_logprobs"] + 1.1 * (st["prompt_logprobs"]
                                                               - un["prompt_logprobs"])
    v = probe.compare_runs(ref, _cand(tmp_path, steer=s), workloads=("steer",))
    assert v["verdict"] == "FAIL"
    assert v["measured"]["steer"]["delta_rel"] == pytest.approx(0.1)


def test_the_rederived_bands_are_what_tolerances_md_records():
    assert probe.DEFAULT_BANDS == {"hs_rel": 5e-2, "hs_row": 1.5e-1, "qk_rel": 1e-1,
                                   "qk_head_rel": 3e-1, "steer_delta_rel": 5e-2,
                                   "steer_liveness_floor": 5e-1}


def test_requested_fusion_without_worker_evidence_fails_and_says_why(ref, tmp_path):
    """If the per-rank evidence RPC failed, 'on' cannot be proven: FAIL, naming the missing
    evidence rather than claiming a disarmed pass."""
    m = {"fusion_pass": [{"error": "collective_rpc failed: RuntimeError: x"}],
         "fusion_active": None}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=m), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert "no per-rank evidence" in _failure_text(v, "hs")
    assert "collective_rpc failed" in _failure_text(v, "hs")


# ---- review follow-ups: stale scratch, header-only rank dirs, recorded env, writer mode ----

def test_the_parent_wipes_a_stale_aperture_before_each_child(tmp_path, monkeypatch):
    """With --out/--scratch reused, an earlier run's tp_rank_* dirs used to be listed as this
    run's: a false 'stray', or a 'missing' dir hidden by a stale one."""
    out = tmp_path / "leg"
    stale = out / "_scratch" / "qk" / "aperture" / "tp_rank_3"
    stale.mkdir(parents=True)
    (stale / "qk_q_layer_0.raw").write_bytes(b"old" * 10)

    def fake_run(argv, env, **kw):
        d = Path(env["MIA_APERTURE_DIR"]) / "tp_rank_0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "qk_q_layer_0.raw").write_bytes(b"new")
        return type("R", (), {"returncode": 1})()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    args = probe._parser().parse_args(["--tp", "4", "--out", str(out), "--workloads", "qk"])
    args.fuse = "default"                          # what main() resolves when --fuse is absent
    probe.run_parent(args)
    listing = json.loads((out / "qk" / probe.APERTURE_LISTING).read_text())
    assert sorted(listing["rank_dirs"]) == ["tp_rank_0"]


def test_an_expected_rank_dir_holding_only_a_sidecar_header_fails(ref, tmp_path):
    """'Holding data' used to mean total_bytes > 0, which a sidecar with only its header line
    satisfies. Data is *.raw bytes."""
    lst = _listing("qk", 4)
    lst["rank_dirs"]["tp_rank_1"] = {"files": [{"name": "qk_aperture_meta.jsonl", "bytes": 300}],
                                     "total_bytes": 300}
    v = probe.compare_runs(ref, _cand(tmp_path, listings={"qk": lst}), workloads=("qk",))
    assert v["verdict"] == "FAIL"
    assert "expected dirs holding no data ['tp_rank_1']" in _failure_text(v, "qk")


def test_the_child_records_every_mia_and_vllm_var_and_redacts_credentials():
    env = probe._recorded_env({"MIA_HS_CAPTURE_ALL_RANKS": "1", "MIA_APERTURE_GPU_BYTES": "9",
                               "VLLM_API_KEY": "s3cret", "HF_TOKEN": "x", "PATH": "/bin",
                               "VLLM_PLUGINS": ""})
    assert env["MIA_HS_CAPTURE_ALL_RANKS"] == "1" and env["MIA_APERTURE_GPU_BYTES"] == "9"
    assert env["VLLM_API_KEY"] == "<redacted>"
    assert env["VLLM_PLUGINS"] == "" and env["MIA_APERTURE_PER_REQUEST"] is None
    assert "PATH" not in env and "HF_TOKEN" not in env


def test_an_inherited_capture_env_is_reported_in_the_verdict(ref, tmp_path):
    extra = {"env": {"VLLM_PLUGINS": None, "MIA_APERTURE_PER_REQUEST": None,
                     "MIA_HS_CAPTURE_ALL_RANKS": "1", "MIA_WORKER": "hidden_states"}}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=extra), workloads=("hs",))
    info = [c for c in v["checks"] if c["name"] == "cand inherited MIA_* env"]
    assert info and info[0]["status"] == "INFO"
    assert info[0]["detail"] == "MIA_HS_CAPTURE_ALL_RANKS=1"


def test_the_contract_writer_modes_pass(ref, tmp_path):
    v = probe.compare_runs(ref, _cand(tmp_path), workloads=("hs", "qk", "steer"))
    got = {(c["workload"], c["name"]): c["status"] for c in v["checks"]
           if c["name"].endswith("writer process")}
    assert got == {("hs", "ref writer process"): "PASS", ("hs", "cand writer process"): "PASS",
                   ("qk", "ref writer process"): "PASS", ("qk", "cand writer process"): "PASS",
                   ("steer", "ref writer process"): "INFO",
                   ("steer", "cand writer process"): "INFO"}


START_FAILED = ("in-process (start failed: AssertionError('daemonic processes are not allowed to "
                "have children'))")


@pytest.mark.parametrize("wl,extra,want", [
    ("qk", {"writer": _writers("qk", 4, {1: {"writer_mode": START_FAILED}})},
     "rank 1: \"in-process (start failed"),
    ("hs", {"writer": _writers("hs", 4, {0: {"writer_mode": START_FAILED}})},
     "rank 0: \"in-process (start failed"),
    ("qk", {"writer": _writers("qk", 4, {2: {"alive": False}})}, "rank 2: writer child not alive"),
    ("hs", {"writer": _writers("hs", 4, {3: {"writer_mode": None}})}, "rank 3: None"),
    ("qk", {"writer": _writers("qk", 2)}, "2 rank record(s) for tp=4"),
    ("qk", {"writer": None}, "not recorded"),
    ("hs", {"writer": [{"error": "collective_rpc failed: RuntimeError: x"}] * 4}, "no evidence"),
])
def test_a_capturing_rank_without_its_writer_fails(ref, tmp_path, wl, extra, want):
    """G1's TP>1 state ("failed to start, falling back to in-process save" on every capturing rank)
    used to show only in child.log. Now the verdict FAILS it, and a dead or unrecorded writer."""
    v = probe.compare_runs(ref, _cand(tmp_path, **{f"{wl}_extra": extra}), workloads=(wl,))
    assert v["verdict"] == "FAIL"
    text = _failure_text(v, wl)
    assert "cand writer process" in text and want in text, text


def test_writer_off_by_env_expects_the_in_process_mode(ref, tmp_path):
    env = {"VLLM_PLUGINS": None, "MIA_APERTURE_PER_REQUEST": None, "MIA_WRITER_PROCESS": "0"}
    off = _writers("qk", 4, {r: {"writer_mode": probe.WRITER_OFF, "alive": None}
                               for r in range(4)})
    v = probe.compare_runs(ref, _cand(tmp_path, qk_extra={"env": env, "writer": off}),
                           workloads=("qk",))
    assert not any(c["name"] == "cand writer process" and c["status"] == "FAIL"
                   for c in v["checks"])


def test_writer_state_reads_the_worker_mode_and_child_liveness():
    from types import SimpleNamespace

    wp = SimpleNamespace(alive=lambda: True, _procs=[SimpleNamespace(pid=77)], parent_daemonic=True)
    got = probe._writer_state(SimpleNamespace(rank=2, _writer_mode="process", _writer_process=wp))
    assert got == {"rank": 2, "writer_mode": "process", "alive": True, "child_pids": [77],
                   "parent_daemonic": True}
    sink = probe._writer_state(SimpleNamespace(rank=1, _writer_mode=probe.WRITER_HS_SINK,
                                               _writer_process=None))
    assert sink == {"rank": 1, "writer_mode": probe.WRITER_HS_SINK}


def test_the_probe_writer_modes_are_what_mia_writes():
    """The probe's expected strings are MIA's own, so a rename cannot silently fail every leg."""
    from types import SimpleNamespace

    pytest.importorskip("vllm")
    from mia.graph import writer_process as wpm

    w = SimpleNamespace(rank=1, parallel_config=SimpleNamespace(tensor_parallel_size=2))
    wpm.mark_no_writer(w, "HS sink rank: captures nothing")
    assert w._writer_mode == probe.WRITER_HS_SINK
    src = Path(wpm.__file__).read_text()
    assert f'worker._writer_mode = "{probe.WRITER_ON}"' in src
    assert f'worker._writer_mode = "{probe.WRITER_OFF}"' in src


def test_the_fusion_rpc_ships_by_value_with_its_helpers(tmp_path):
    """The child runs the probe as a script and hands ``_fusion_pass_state`` to collective_rpc,
    which cloudpickles it BY VALUE into workers that cannot import the probe. Its locator helpers
    and constants must travel with it: pickle it the same way (a non-importable module), load it
    in a fresh interpreter with nothing of the probe importable, and run it on a fake worker."""
    cloudpickle = pytest.importorskip("cloudpickle")
    import runpy
    import subprocess
    import sys

    ns = runpy.run_path(str(Path(probe.__file__)), run_name="tp_probe_as_script")
    (tmp_path / "fn.pkl").write_bytes(cloudpickle.dumps(ns["_fusion_pass_state"]))
    code = r'''
import json, pickle, sys, types
from types import SimpleNamespace as NS
fake = types.ModuleType("vllm.compilation.passes.fusion.allreduce_rms_fusion")
class AllReduceFusionPass:
    pass
fake.AllReduceFusionPass, fake.flashinfer_comm = AllReduceFusionPass, object()
pkg = types.ModuleType("vllm.compilation.passes.fusion")
pkg.allreduce_rms_fusion = fake
sys.modules[fake.__name__], sys.modules[pkg.__name__] = fake, pkg
p = AllReduceFusionPass()
p.disabled, p.max_token_num, p.tp_size, p.matched_count = False, 256, 4, 2
pm = NS(passes=[p], pass_config=object())
class LlamaModel:
    def __init__(self):
        self._compiled_callable = lambda *a: None
        self.aot_compiled_fn = NS(_artifacts=NS(compiled_fn=NS(vllm_backend=NS(pass_manager=pm))))
class Top:
    def named_modules(self):
        return [("", self), ("model", LlamaModel())]
worker = NS(rank=3, get_model=Top, vllm_config=NS(compilation_config=NS(
    pass_config=NS(fuse_allreduce_rms=True))))
fn = pickle.loads(open("fn.pkl", "rb").read())
print(json.dumps(fn(worker)))
'''
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert "error" not in got, got
    assert got["rank"] == 3 and got["backends_configured"] == 1
    assert got["passes"] == [{"module": "model", "disabled": False, "max_token_num": 256,
                              "tp_size": 4, "called": True, "matched_count_last_call": 2}]
    assert got["locator"] == probe.FUSION_LOCATOR
    assert probe._fusion_active([got]) is True


# ---- HS layer shard (MIA_HS_TP_SHARD) and the replication leg (hs_replicas) ------------------

def test_a_sharded_hs_leg_is_judged_on_the_merged_capture_and_every_rank_dir(ref, tmp_path):
    """At TP>1 the hs leg's capture is the UNION of every rank's round-robin share (MIA's
    load_hs_aperture_tp): every rank dir must hold data, every header must declare the rule's
    layers, the merge must have succeeded, and the merged capture is judged against TP=1."""
    v = probe.compare_runs(ref, _cand(tmp_path), workloads=("hs",))
    assert v["verdict"] == "PASS", v["failures"]
    by = {c["name"]: c for c in v["checks"]}
    assert by["cand rank dirs"]["status"] == "PASS"
    assert "tp_rank_3" in by["cand rank dirs"]["detail"]
    assert by["cand HS layout headers"]["status"] == "PASS"
    assert by["cand TP merge"]["status"] == "PASS"
    assert by["cand writer process"]["status"] == "PASS"          # every rank captures


_ALL4 = "expected ['tp_rank_0', 'tp_rank_1', 'tp_rank_2', 'tp_rank_3'] (layout round_robin"


@pytest.mark.parametrize("extra,want,detail", [
    ({"flush_dirs": [f"/a/tp_rank_{r}" for r in (0, 1, 3)]}, "cand rank dirs", _ALL4),
    ({"flush_dirs": ["/a/tp_rank_0"]}, "cand rank dirs", _ALL4),     # the old rank-0-only layout
    ({"merge_error": "TPShardError: HS capture is missing tp_rank(s) [2]"}, "cand TP merge",
     "load_hs_aperture_tp raised"),
    ({"writer": _writers("hs", 4, hs_rank0_only=True)}, "cand writer process",
     "expected 'process'"),
])
def test_a_sharded_hs_leg_that_is_not_fully_sharded_fails(ref, tmp_path, extra, want, detail):
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=extra), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    got = [c for c in _failed(v, "hs") if c["name"] == want]
    assert got and detail in got[0]["detail"], v["failures"]


def test_a_sharded_hs_header_that_lies_about_its_layers_fails(ref, tmp_path):
    m = _hs_manifest("hs", 4)
    d1 = sorted(m["rank_headers"])[1]
    m["rank_headers"][d1]["owned_layers"] = [1, 2]
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=m), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert "tp_rank 1: layer_shard='round_robin' owned_layers=[1, 2]" in _failure_text(v, "hs")


def test_the_rank0_only_ab_leg_is_judged_by_its_own_layout(ref, tmp_path):
    """MIA_HS_TP_SHARD=0 recorded in the child env -> the old layout is the contract: tp_rank_0
    only, sink ranks with no writer, headers without shard fields."""
    lst = {"hs": _listing("hs", 4, {"tp_rank_0": 4096})}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra=_rank0_only_hs(), listings=lst),
                           workloads=("hs",))
    assert v["verdict"] == "PASS", v["failures"]
    # ...and a sharded capture under that env is not what the leg asked for.
    sharded = dict(_rank0_only_hs(), **_hs_manifest("hs", 4),
                   writer=_writers("hs", 4))
    v = probe.compare_runs(ref, _cand(tmp_path / "x", hs_extra=sharded), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert {"cand rank dirs", "cand HS layout headers", "cand writer process",
            "cand aperture dirs on disk"} <= {c["name"] for c in _failed(v, "hs")}


def test_the_hs_leg_must_not_run_with_the_replication_diagnostic(ref, tmp_path):
    env = {"VLLM_PLUGINS": None, "MIA_APERTURE_PER_REQUEST": None,
           probe.HS_ALL_RANKS_ENV: "1"}
    v = probe.compare_runs(ref, _cand(tmp_path, hs_extra={"env": env}), workloads=("hs",))
    assert v["verdict"] == "FAIL"
    assert f"cand {probe.HS_ALL_RANKS_ENV}" in {c["name"] for c in _failed(v, "hs")}


def test_the_replication_leg_passes_when_every_rank_holds_the_same_bits(ref, tmp_path):
    v = probe.compare_runs(ref, _cand(tmp_path), workloads=("hs_replicas",))
    assert v["verdict"] == "PASS", v["failures"]
    by = {c["name"]: c for c in v["checks"]}
    assert by["replicas bitwise equal across ranks"]["status"] == "PASS"
    assert by["ref data"]["status"] == "INFO"                 # the TP=1 ref's hs leg
    assert by["cand rank dirs"]["status"] == "PASS"
    assert by["cand HS layout headers"]["status"] == "PASS"
    assert v["measured"]["hs_replicas"]["worst_rel"] < probe.TP_HS_REL_BAND


def test_a_replica_that_differs_by_one_bit_fails(ref, tmp_path):
    data = _noisy(_hs_requests())
    ranks = {r: [{k: t.clone() for k, t in rec["layers"].items()} for rec in data]
             for r in range(4)}
    t = ranks[2][1][3]
    t.view(-1)[0] = torch.nextafter(t.view(-1)[0], torch.tensor(float("inf")))
    v = probe.compare_runs(ref, _cand(tmp_path, replicas=data, replicas_ranks=ranks),
                           workloads=("hs_replicas",))
    assert v["verdict"] == "FAIL"
    assert "rank 2 req1 layer 3: max|diff|" in _failure_text(v, "hs_replicas")


@pytest.mark.parametrize("change,want", [
    ("drop_rank", "cand replica ranks"),
    ("drop_layer", "replicas bitwise equal across ranks"),
    ("no_file", "cand replicas present"),
    ("mia_check", "cand MIA replica check (load_hs_aperture_tp check_replicas)"),
    ("no_env", f"cand {probe.HS_ALL_RANKS_ENV}"),
])
def test_an_incomplete_replication_leg_fails(ref, tmp_path, change, want):
    data = _noisy(_hs_requests())
    ranks = {r: [{k: t.clone() for k, t in rec["layers"].items()} for rec in data]
             for r in range(4)}
    extra = {}
    if change == "drop_rank":
        del ranks[3]
    elif change == "drop_layer":
        del ranks[1][0][2]
    elif change == "no_file":
        ranks = None
    elif change == "mia_check":
        extra["replica_check"] = "TPShardError: all-ranks HS replicas differ"
    else:
        extra["env"] = {"VLLM_PLUGINS": None, "MIA_APERTURE_PER_REQUEST": None}
    v = probe.compare_runs(ref, _cand(tmp_path, replicas=data, replicas_ranks=ranks,
                                      replicas_extra=extra), workloads=("hs_replicas",))
    assert v["verdict"] == "FAIL"
    assert want in {c["name"] for c in _failed(v, "hs_replicas")}, v["failures"]


def test_the_replication_leg_is_required_at_tp_gt_1_and_skipped_at_tp1(ref, tmp_path):
    run = tmp_path / "tp4"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"tp": 4}))
    _write(run, "hs", 4, _noisy(_hs_requests()))
    v = probe.compare_runs(ref, run, workloads=("hs_replicas",))
    assert v["verdict"] == "FAIL"
    assert any(c["name"] == "both runs present" for c in _failed(v, "hs_replicas"))
    v = probe.compare_runs(ref, ref)                          # TP=1 vs TP=1: INFO, not FAIL
    assert v["verdict"] == "PASS", v["failures"]
    skip = [c for c in v["checks"] if c["name"] == "replication leg (TP=1: skipped)"]
    assert skip and skip[0]["status"] == "INFO"


def test_repeat_mode_compares_two_replication_legs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    tp2 = _cand(tmp_path / "a", tp=2)
    rep = _cand(tmp_path / "b", tp=2, replicas=_noisy(_hs_requests(), seed=5))
    v = probe.compare_legs([tp2], rep, workloads=("hs_replicas",), repeat=True)
    assert v["verdict"] == "PASS", v["failures"]
    assert not any(c["name"] == "ref data" for c in v["checks"])   # its own hs_replicas leg


def test_the_parent_runs_the_replication_leg_only_at_tp_gt_1_with_the_diagnostic_on(
        tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, env, **kw):
        wl = argv[argv.index("--child-workload") + 1]
        seen[wl] = env.get(probe.HS_ALL_RANKS_ENV)
        return type("R", (), {"returncode": 1})()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setenv(probe.HS_ALL_RANKS_ENV, "1")          # an inherited diagnostic
    for tp in (1, 2):
        seen.clear()
        out = tmp_path / f"leg{tp}"
        args = probe._parser().parse_args(["--tp", str(tp), "--out", str(out),
                                           "--workloads", "hs,hs_replicas"])
        args.fuse = "default"
        probe.run_parent(args)
        top = json.loads((out / "manifest.json").read_text())
        assert seen.get("hs", "absent") is None              # never inherited by the hs leg
        if tp == 1:
            assert "hs_replicas" not in seen
            assert top["workloads"]["hs_replicas"]["status"] == "skipped"
        else:
            assert seen["hs_replicas"] == "1"
    assert probe.WORKLOADS == ("hs", "qk", "steer", "hs_replicas")
    assert probe._parser().parse_args([]).workloads == "hs,qk,steer,hs_replicas"
