"""Ground truth that shares nothing with MIA except the model weights.

Vanilla vLLM 0.29 + the V2 runner, ``VLLM_PLUGINS=""`` so MIA never loads in this
process, capture via torch's own ``register_forward_hook``. Same weights, same kernels,
same version, same runner as the MIA arm, so a single-request eager greedy comparison
should be BIT-EXACT -- not merely close. Nothing here imports ``mia``; the only module
shared with the MIA arm is ``capture_workload``, whose top half is arm-neutral by
construction (prompts, engine kwargs, sampling params, safetensors serialization), so
the two arms differ in the plugin under test and in nothing else.

What "independent" means concretely, per subsystem:

* **hidden states** -- MIA hooks the decoder block and reads the block's output. So does
  this. The one piece of vLLM-specific knowledge both sides must share is that vLLM's
  decoder blocks return ``(hidden_states, residual)`` with the residual NOT yet added
  (``LlamaDecoderLayer.forward``), so the post-block hidden state is the SUM. That is a
  property of the model definition, not of MIA.
* **q / k** -- MIA hooks ``model.layers.<i>.self_attn.attn`` (vLLM's ``Attention``, called
  positionally as ``self.attn(q, k, v)``) and reads its INPUTS. So does this. But the two
  sides build the growing key history ``k_all`` by completely different routes: MIA reads
  the missing prefix back out of vLLM's **paged KV cache** on every decode step, while
  this reference simply **accumulates** the key rows it saw at each step. If those agree
  bit-for-bit, MIA's KV-cache reconstruction is independently confirmed; if they diverge,
  the KV read is the first suspect.

Batching. The single-request legs run ``max_num_seqs=1``: with one request resident at a
time every forward pass contains exactly one request's rows, so this reference needs **no
batch-row arithmetic at all** -- no ``query_start_loc``, no ``idx_mapping``, no padding
slice, and it therefore shares none of the index logic it is meant to check.

That was also this oracle's blind spot, and it was the deepest finding of the structural
review: every OTHER gate in the suite is MIA-vs-MIA (both arms import ``StepView`` from
``mia/runner.py``, which IS the V1->V2 port surface), so BATCHING -- the thing the port
actually changed -- was never checked against anything outside MIA. ``reference_capture_
batched`` (task D5 item 8) closes that: 33 distinct-length requests in ONE batch, with each
request's rows located from the TOKEN IDS the model was fed rather than from anyone's
routing table. See that section for what it covers and for the limit below.

**The irreducible limit.** A Python forward hook CANNOT OBSERVE A CUDA-GRAPH REPLAY -- a
replay is a graph launch on the device and no Python runs during it, so
``register_forward_hook`` never fires. An independent oracle for GRAPH mode is therefore
not constructible this way, by any hook on any module. This file is eager-only by nature,
and graph-mode correctness continues to rest on MIA-vs-MIA plus the replay band. Recorded
here, in ``tests/mia/parity/TOLERANCES.md``, and beside the batched oracle itself, because the
honest statement of what a suite does NOT cover is part of what its PASS means.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Set before vLLM is imported anywhere in this process.
#
# VLLM_PLUGINS="" parses to the allowlist [""], which matches no entry point, so every
# installed plugin -- MIA included -- is skipped. assert_no_plugin() below proves it
# rather than trusting it.
os.environ["VLLM_PLUGINS"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The engine core must live in THIS process: a forward hook has to be attached to the
# live nn.Module, and the default offline path forks the engine core into a child where
# the model (and anything the hook captured) would be unreachable.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# The MIA arm's package sets this at import time whenever cuda graphs are not armed
# (mia/llm.py). Match it so the two arms cannot differ on compilation -- with
# enforce_eager there is nothing to compile either way, but "cannot differ" beats
# "should not differ" when the gate is bit-exactness.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402

from tests.mia.parity.capture_workload import (  # noqa: E402
    ENGINE_KWARGS,
    MODEL,
    STEER_LIVENESS_ATOL,
    WORKLOADS,
    Workload,
    _ascii,
    _save,
    generation_tensors,
    logprob_delta,
    sampling_params_for,
    steer_config,
    steer_fingerprint,
)

# ---------------------------------------------------------------------------
# INDEX BASE -- the single fact that decides whether this oracle means anything
# ---------------------------------------------------------------------------
#
# The two capture paths do NOT share a layer-numbering convention, and getting it wrong
# is silent in BOTH directions: comparing MIA layer 1 against block 1 compares two
# different layers (phantom failure), and asking the HS path for layer 0 matches no
# module at all, yielding an empty tree that a naive A/B reads as a trivial PASS.
#
#   HS  -- MIA reports layer_num = pytorch_index + 1 (1-based, HuggingFace/Eagle
#          convention: "layer N" = the output after the Nth transformer block).
#          MIA layer N  <->  model.layers[N - 1]
#   QK  -- MIA reports layer_num = the pytorch index itself (0-based), taken straight off
#          the module name `model.layers.<i>.self_attn.attn`.
#          MIA layer N  <->  model.layers[N]
#
# So hs_small's (1, 16, 32) and qk_small's (0, 15, 31) name the SAME three physical
# blocks. assert_layer_mapping() below enforces the legal range of each convention, which
# is what makes the 1-vs-0 confusion fail loudly instead of quietly.
# STEER is a THIRD convention and it matches NEITHER capture path by assumption -- it was
# read off the code (see task C5's report for the trace):
#
#   mia/workers/steer_worker.py::_install_hooks registers each hook with
#   `ln=layer_num` straight from `iter_matched_modules(model, match_layer)`, i.e. the RAW
#   0-based PyTorch index parsed out of `model.layers.<i>` -- with NO +1, unlike
#   hs_capture_worker.py, which registers `ln=layer_num+1`. The hook then keeps a request
#   iff `_steer_targets_layer(cfg["optimal_layer"], this_layer)` matches that raw index,
#   and `_parse_steer_layers` (graph path) accepts exactly `0 <= L < num_layers`. So:
#          steer optimal_layer N  <->  model.layers[N]        (0-based, like QK)
#   The same is true of the graph path, which reuses `match_layer` via install_steer.py.
# The ONE runner this oracle may run on besides vLLM 0.29's V2 runner, and the ONE vLLM
# version it is allowed on: T3's cross-version control arm (see `_model_of`). Pinned as
# constants so the allowance cannot drift into "any runner that is not V2".
_LEGACY_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_LEGACY_VLLM = "0.21."

HS_BLOCK_OFFSET = -1      # block index = MIA HS layer + HS_BLOCK_OFFSET
QK_BLOCK_OFFSET = 0       # block index = MIA QK layer + QK_BLOCK_OFFSET
STEER_BLOCK_OFFSET = 0    # block index = MIA steer optimal_layer + STEER_BLOCK_OFFSET

KINDS = ("capture_hs", "capture_qk")

# Kinds that hook the decoder BLOCK itself rather than the attention op inside it. Steering
# belongs here because it rewrites the block's residual output.
BLOCK_KINDS = ("capture_hs", "steer")
ALL_KINDS = KINDS + ("steer",)

_OFFSETS = {"capture_hs": HS_BLOCK_OFFSET,
            "capture_qk": QK_BLOCK_OFFSET,
            "steer": STEER_BLOCK_OFFSET}


def block_index(kind: str, mia_layer: int) -> int:
    try:
        return int(mia_layer) + _OFFSETS[kind]
    except KeyError:
        raise KeyError(f"unknown kind {kind!r}; expected one of {ALL_KINDS}") from None


def module_name_for(kind: str, mia_layer: int) -> str:
    """The torch module name this reference hooks for MIA layer ``mia_layer``."""
    index = block_index(kind, mia_layer)
    if kind in BLOCK_KINDS:
        return f"model.layers.{index}"
    return f"model.layers.{index}.self_attn.attn"


def assert_layer_mapping(model, kind: str, layers) -> dict:
    """Resolve, print and enforce the layer mapping. Returns {mia_layer: module}.

    Enforced, not assumed:
      * the HS numbering is 1-based, so layer 0 is REFUSED here -- it is the exact input
        that would otherwise produce an empty artifact tree and a false PASS;
      * the QK and STEER numbering is 0-based, so the last legal layer is num_layers - 1;
      * the resolved module is structurally the right kind of thing (a decoder block for
        HS and STEER -- it owns `self_attn` and `mlp`; vLLM's `Attention` op for QK).
    """
    named = dict(model.named_modules())
    blocks = [n for n in named if n.startswith("model.layers.") and n.count(".") == 2]
    num_layers = len(blocks)
    if num_layers == 0:
        raise RuntimeError("no `model.layers.<i>` modules found; the model layout changed")

    low, high = (1, num_layers) if kind == "capture_hs" else (0, num_layers - 1)
    resolved = {}
    for mia_layer in layers:
        if not low <= int(mia_layer) <= high:
            raise RuntimeError(
                f"{kind}: MIA layer {mia_layer} is outside the legal range [{low}, {high}] "
                f"for a {num_layers}-layer model. HS layer numbers are 1-BASED while QK and "
                f"STEER layer numbers are 0-BASED; mixing them touches the wrong block or "
                f"nothing at all.")
        name = module_name_for(kind, mia_layer)
        module = named.get(name)
        if module is None:
            raise RuntimeError(f"{kind}: MIA layer {mia_layer} -> {name}, which does not exist")
        if kind in BLOCK_KINDS:
            if not (hasattr(module, "self_attn") and hasattr(module, "mlp")):
                raise RuntimeError(
                    f"{name} is not a decoder block (no self_attn/mlp): "
                    f"{type(module).__name__}")
        elif type(module).__name__ != "Attention":
            raise RuntimeError(
                f"{name} is {type(module).__name__}, not vLLM's Attention op -- its "
                f"positional inputs would not be (q, k, v)")
        print(f"[T1] layer map {kind}: MIA layer {mia_layer} -> block "
              f"{block_index(kind, mia_layer)} -> {name} ({type(module).__name__})")
        resolved[int(mia_layer)] = module
    return resolved


# ---------------------------------------------------------------------------
# Proof that MIA is not in this process
# ---------------------------------------------------------------------------

def assert_no_plugin() -> None:
    """Print, then enforce, that no plugin loaded here -- MIA least of all.

    The whole value of this oracle is that it shares no machinery with the code under
    test. A reference process that quietly loaded MIA would be MIA agreeing with itself,
    and would still report a clean PASS, so the check is structural rather than a grep of
    the log.
    """
    from importlib.metadata import entry_points

    import vllm.envs as envs

    allowed = envs.VLLM_PLUGINS
    print(f"[T1] VLLM_PLUGINS allowlist = {allowed}")
    if allowed is None:
        raise RuntimeError("VLLM_PLUGINS is unset: every installed plugin would load")
    active = [name for name in allowed if name]
    if active:
        raise RuntimeError(f"reference process has a non-empty plugin allowlist: {active}")

    for ep in sorted(entry_points(group="vllm.general_plugins"), key=lambda e: e.name):
        print(f"[T1] plugin entry-point {ep.name} -> {ep.value} : skipped")

    leaked = sorted(m for m in sys.modules if m == "mia" or m.startswith("mia."))
    if leaked:
        raise RuntimeError(f"MIA is imported in the reference process: {leaked}")
    print("[T1] no MIA module imported in this process: reference is independent")


# ---------------------------------------------------------------------------
# Runner access
# ---------------------------------------------------------------------------

def _model_of(llm):
    """The live nn.Module behind an offline `LLM`, via the in-process engine core.

    Single accessor on purpose: if 0.29.x nests this differently, fix it here and nowhere
    else. `WorkerWrapperBase.__getattr__` forwards to the real worker, which is why
    `driver_worker.model_runner` resolves.
    """
    engine = llm.llm_engine
    executor = getattr(engine, "model_executor", None)
    if executor is None:  # engine core was forked; hooks would be unreachable
        raise RuntimeError(
            "the engine core is not in this process (VLLM_ENABLE_V1_MULTIPROCESSING), so "
            "no forward hook can be attached to the model")
    runner = executor.driver_worker.model_runner
    module = type(runner).__module__
    print(f"[T1] runner = {module}.{type(runner).__name__}")
    if module.startswith("vllm.v1.worker.gpu."):
        return runner.model
    # T3 (tests/mia/parity/t3_crossbranch.py) runs this SAME oracle on vLLM 0.21 to measure what
    # vLLM's own 0.21->0.29 drift costs, with MIA absent on both sides. That arm is on 0.21's
    # V1 runner by definition, so the V2 assertion above cannot hold for it. The escape hatch
    # is deliberately unusable as a way to skip the check on 0.29: it demands an explicit
    # opt-in AND pins the exact legacy module AND pins the vLLM version it is allowed for, so
    # a 0.29 run that somehow landed on a non-V2 runner still fails here.
    if os.environ.get("MIA_PARITY_ALLOW_LEGACY_RUNNER") == "1":
        import vllm

        if module == _LEGACY_RUNNER_MODULE and vllm.__version__.startswith(_LEGACY_VLLM):
            print(f"[T1] LEGACY RUNNER ALLOWED: vLLM {vllm.__version__} on {module} "
                  f"(MIA_PARITY_ALLOW_LEGACY_RUNNER=1). This is the T3 cross-version "
                  f"control arm; MIA is not in this process either way.")
            return runner.model
        raise RuntimeError(
            f"MIA_PARITY_ALLOW_LEGACY_RUNNER=1 permits ONLY {_LEGACY_RUNNER_MODULE} on "
            f"vLLM {_LEGACY_VLLM}x, got {module} on vLLM {vllm.__version__}")
    raise RuntimeError(
        f"the reference must run on the SAME V2 runner as MIA, got "
        f"{module}.{type(runner).__name__}")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _post_block_hidden(output):
    """The hidden state after a decoder block.

    vLLM's blocks use a fused-residual pattern and return ``(hidden_states, residual)``
    with the residual not yet added, so the post-block value is the sum. Property of
    ``LlamaDecoderLayer.forward``, not of MIA.
    """
    if isinstance(output, tuple) and len(output) == 2 and torch.is_tensor(output[1]):
        return output[0] + output[1]
    if isinstance(output, tuple):
        return output[0]
    return output


def _row_plan(output, fires: int) -> list[int]:
    """How many rows of each forward pass belong to this request.

    Derived from the request's own result, never from vLLM's batch metadata: one prefill
    pass covering the whole prompt, then one row per decode step. With max_num_seqs=1 the
    request is alone in every pass, so its rows are the leading rows.
    """
    n_prompt = len(output.prompt_token_ids)
    n_gen = len(output.outputs[0].token_ids)
    if fires != n_gen:
        raise RuntimeError(
            f"expected {n_gen} forward passes for a {n_gen}-token generation "
            f"(1 prefill + {n_gen - 1} decode), but the hook fired {fires} times")
    return [n_prompt] + [1] * (n_gen - 1)


def _slice_rows(tensor: torch.Tensor, rows: int, what: str, step: int) -> torch.Tensor:
    if tensor.shape[0] < rows:
        raise RuntimeError(
            f"{what}: step {step} produced {tensor.shape[0]} rows, fewer than the "
            f"{rows} this request owns")
    if tensor.shape[0] > rows:
        # Worth saying out loud: it would mean V2 pads the token dimension even in eager.
        print(f"[T1] NOTE {what}: step {step} tensor has {tensor.shape[0]} rows for a "
              f"{rows}-row request; slicing to the leading {rows}")
    return tensor[:rows].cpu()


def reference_capture(kind: str, layers, prompts, out_dir: Path, model: str = MODEL,
                      max_tokens: int = 16, granularity: str = "all_tokens") -> Path:
    """Capture ``kind`` at ``layers`` for each prompt, one request at a time.

    Writes ``<out_dir>/req<N>/layer<MIA layer>.safetensors`` plus ``generation.safetensors``
    -- the same tree shape, the same file names and the same tensor keys the MIA arm
    produces, so the two are comparable key-for-key by ``compare_artifacts.py``.

    Consumed by task C5 (steer non-interference) and E1.
    """
    from vllm import LLM

    if kind not in KINDS:
        raise KeyError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if granularity != "all_tokens":
        raise NotImplementedError("the reference implements all_tokens granularity only")

    assert_no_plugin()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_kwargs = dict(ENGINE_KWARGS)
    # Both arms read this from the same dict, driven by MIA_PARITY_MAX_NUM_SEQS. Refuse
    # to run otherwise: at a wider running batch this reference's "the request owns the
    # leading rows" assumption is wrong, and it would compare the wrong rows in silence.
    if engine_kwargs.get("max_num_seqs") != 1:
        raise RuntimeError(
            "T1 requires max_num_seqs=1 in BOTH arms (set MIA_PARITY_MAX_NUM_SEQS=1); "
            f"got {engine_kwargs.get('max_num_seqs')!r}")
    print(f"[T1] engine kwargs = {engine_kwargs}")
    llm = LLM(model=model, enforce_eager=True, **engine_kwargs)
    nn_model = _model_of(llm)
    modules = assert_layer_mapping(nn_model, kind, layers)

    # Sampling params are built by the shared, arm-neutral factory, so both arms decode
    # greedily for the same fixed number of steps with the same logprob channel. `extra`
    # is None here: extra_args is exactly the request-level API that arms MIA, and this
    # process has no MIA to arm.
    sampling = sampling_params_for(
        Workload(subsystem="hs" if kind == "capture_hs" else "qk",
                 prompts=tuple(prompts), layers=tuple(layers),
                 max_tokens=int(max_tokens), hooks_on="both", granularity=granularity),
        None)

    written = 0
    for index, prompt in enumerate(prompts):
        fires: dict[int, list] = {int(n): [] for n in modules}

        def _hs_hook(_module, _args, output, n=None):
            fires[n].append(_post_block_hidden(output).detach().clone())

        def _qk_hook(_module, args, _output, n=None):
            fires[n].append((args[0].detach().clone(), args[1].detach().clone()))

        handles = []
        for mia_layer, module in modules.items():
            hook = (_hs_hook if kind == "capture_hs" else _qk_hook)
            handles.append(module.register_forward_hook(
                lambda m, a, o, _h=hook, _n=mia_layer: _h(m, a, o, n=_n)))
        try:
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
        finally:
            for handle in handles:
                handle.remove()

        output = outputs[0]
        req_dir = out_dir / f"req{index}"
        _save(req_dir / "generation.safetensors", generation_tensors(output))
        written += 1

        for mia_layer, steps in sorted(fires.items()):
            if not steps:
                raise RuntimeError(
                    f"{kind}: MIA layer {mia_layer} ({module_name_for(kind, mia_layer)}) "
                    f"captured NOTHING for request {index}")
            rows = _row_plan(output, len(steps))
            tensors = (_assemble_hs(steps, rows, mia_layer)
                       if kind == "capture_hs" else _assemble_qk(steps, rows, mia_layer))
            _save(req_dir / f"layer{int(mia_layer):03d}.safetensors", tensors)
            written += 1

    print(f"[T1] wrote {written} reference artifact files under {out_dir}")
    if written == 0:
        raise RuntimeError(f"reference capture produced NOTHING under {out_dir}")
    return out_dir


def _assemble_hs(steps, rows, mia_layer: int) -> dict:
    """One per-step tensor per forward pass, right-padded into (steps, max_len, hidden).

    ``pad_sequence`` is the same torch primitive the MIA arm's retrieval uses, so the
    padded layout matches; the VALUES are the thing under test.
    """
    per_step = [_slice_rows(t, n, f"hs L{mia_layer}", s)
                for s, (t, n) in enumerate(zip(steps, rows))]
    stacked = pad_sequence(per_step, batch_first=True)
    print(f"[T1] hs layer {mia_layer}: {len(per_step)} passes -> "
          f"{tuple(stacked.shape)} {stacked.dtype}")
    return {"hidden_states": stacked,
            "layer_num": torch.tensor(int(mia_layer), dtype=torch.int64),
            "hs_mode": _ascii("all_tokens")}


def _assemble_qk(steps, rows, mia_layer: int) -> dict:
    """q padded per pass; k_all as the GROWING key prefix, one prefix per pass.

    The key history is built by pure accumulation of the key rows this reference itself
    saw -- step 0 contributes the whole prompt, each decode step one more row -- so the
    prefix at pass ``s`` is ``cat(k_0..k_s)``. MIA instead re-reads the earlier keys out
    of vLLM's paged KV cache on every decode step. Bit-exact agreement between the two is
    an independent check of that KV read.
    """
    q_steps = [_slice_rows(q, n, f"qk.q L{mia_layer}", s)
               for s, ((q, _k), n) in enumerate(zip(steps, rows))]
    k_steps = [_slice_rows(k, n, f"qk.k L{mia_layer}", s)
               for s, ((_q, k), n) in enumerate(zip(steps, rows))]

    q_stacked = pad_sequence(q_steps, batch_first=True)
    k_full = torch.cat(k_steps, dim=0)
    ends, total = [], 0
    for k in k_steps:
        total += k.shape[0]
        ends.append(total)
    k_all = pad_sequence([k_full[:end] for end in ends], batch_first=True)
    print(f"[T1] qk layer {mia_layer}: {len(q_steps)} passes -> q {tuple(q_stacked.shape)} "
          f"k_all {tuple(k_all.shape)} {k_all.dtype} (prefix ends {ends[0]}..{ends[-1]})")
    return {"q": q_stacked, "k_all": k_all,
            "layer_num": torch.tensor(int(mia_layer), dtype=torch.int64),
            "hookq_mode": _ascii("all_tokens")}


# ---------------------------------------------------------------------------
# BATCHED oracle (task D5 item 8) -- the first independent check of MULTI-ROW routing
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. The structural review's deepest finding: every T2 gate is MIA-vs-MIA.
# Both arms import ``StepView`` / ``step_view`` from ``mia/runner.py``, which IS the V1->V2
# port surface, so a bug there moves both arms identically and every T2 gate still passes.
# This file was the only oracle outside MIA, and it ran ONE request at a time, eager -- so
# batching, the thing the port actually changed, was never checked against anything but MIA
# itself. This section runs SEVERAL distinct-length requests in ONE batch and compares
# vanilla-vLLM forward hooks against MIA's capture at width 33.
#
# WHAT MAKES IT INDEPENDENT. The reference must decide which rows of a batched forward pass
# belong to which request, and it must do so WITHOUT MIA's routing or the oracle collapses
# into the common mode it exists to break. It decides from the TOKEN IDS the model was
# given: a hook on the embedding sees the flat input of every pass, and each request's
# prompt is matched against it (``segment_batch_rows``). No ``query_start_loc``, no
# ``idx_mapping``, no ``StepView`` -- nothing from ``mia/runner.py``, and nothing from
# vLLM's batch metadata either.
#
# THE IRREDUCIBLE LIMIT, stated rather than papered over. A Python forward hook CANNOT
# OBSERVE A CUDA-GRAPH REPLAY: a replay is a single graph launch on the device, and no
# Python executes during it, so `register_forward_hook` never fires. An independent oracle
# for GRAPH mode is therefore NOT CONSTRUCTIBLE this way -- not by a cleverer hook, not by a
# different module, not at all. This oracle is eager-only by nature, and graph-mode
# correctness continues to rest on MIA-vs-MIA (`op-identity`, `routing-identity-*`) plus the
# replay band. What this section adds is that MIA's MULTI-ROW ROUTING -- the part the V2 port
# rewrote -- is now verified against something outside MIA in eager mode, where before
# nothing was. See tests/mia/parity/TOLERANCES.md, "What the batched oracle cannot cover".
#
# WHAT IT COVERS. The PREFILL wave. A hook sees one flat [num_tokens, ...] tensor per pass
# and nothing that says which rows belong to whom; token-id matching recovers that exactly
# for prefill, where each request contributes a contiguous run equal to its own prompt.
# Decode rows carry one token per running request and are not identifiable that way, so they
# are not claimed -- which is why the batched workloads use ``hooks_on="prefill"``. Prefill
# is also where multi-row-per-request routing actually lives: one request, many rows, an
# offset into a shared flat buffer. Generation still runs to max_tokens, so the
# generation.safetensors comparison covers all 16 decode steps at width 33.

def segment_batch_rows(flat_ids, prompt_ids: list[list[int]]) -> dict[int, tuple[int, int]]:
    """``{request index: (start_row, end_row)}`` for one forward pass, from token ids alone.

    ``flat_ids`` is the pass's flat input token ids; ``prompt_ids`` is each request's own
    prompt token ids. A request's prefill rows are the contiguous run equal to its prompt.

    The walk is: at each row, is this the FIRST token of a request we have not placed yet,
    and does the whole prompt match from here? If so that request owns those rows; otherwise
    advance one row (a decode row, or padding) and try again. That is unambiguous only
    because ``assert_batch_oracle_prompts`` has already enforced DISTINCT FIRST TOKENS -- with
    the ``DISTINCT_PROMPTS`` set, where every prompt is a prefix of the next, it would not be.

    Deliberately makes no use of vLLM's batch metadata and none whatsoever of MIA's: the
    whole value of this oracle is that it computes the row mapping from a different fact
    (what the model was fed) than the code under test computes it from.
    """
    ids = [int(t) for t in flat_ids]
    by_first: dict[int, list[int]] = {}
    for index, prompt in enumerate(prompt_ids):
        by_first.setdefault(int(prompt[0]), []).append(index)
    for first, indexes in by_first.items():
        if len(indexes) > 1:
            raise RuntimeError(
                f"requests {indexes} share first token {first}: row segmentation would be "
                f"ambiguous. assert_batch_oracle_prompts is supposed to have refused this.")

    found: dict[int, tuple[int, int]] = {}
    row = 0
    while row < len(ids):
        candidates = by_first.get(ids[row], [])
        placed = False
        for index in candidates:
            if index in found:
                continue
            prompt = prompt_ids[index]
            end = row + len(prompt)
            if end <= len(ids) and ids[row:end] == [int(t) for t in prompt]:
                found[index] = (row, end)
                row = end
                placed = True
                break
        if not placed:
            row += 1
    return found


def _embedding_module(model):
    """The module whose input is the flat token-id tensor of a forward pass.

    Single accessor, like ``_model_of``: if 0.29.x nests the embedding differently, fix it
    here. Only used to LEARN THE ROW LAYOUT, never to capture anything.
    """
    for path in ("model.embed_tokens", "transformer.wte", "model.model.embed_tokens"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            print(f"[T1] batched oracle reads the row layout from {path}")
            return node
    raise RuntimeError(
        "no token-embedding module found on this model, so the batched oracle cannot learn "
        "which rows of a forward pass belong to which request WITHOUT asking the code under "
        "test. Refusing to fall back on MIA's own routing.")


def reference_capture_batched(kind: str, layers, prompts, out_dir: Path, model: str = MODEL,
                              max_tokens: int = 16) -> Path:
    """Capture ``kind`` at ``layers`` for ALL ``prompts`` in ONE batch, prefill rows only.

    Writes the same tree shape the MIA arm writes -- ``<out_dir>/req<N>/layer<L>.safetensors``
    plus ``generation.safetensors`` -- so ``compare_artifacts.py --require-bit-exact``
    compares them key for key. Both arms are eager, greedy, same weights, same engine, same
    batch composition, so a difference IS a finding.
    """
    from vllm import LLM

    from tests.mia.parity.capture_workload import assert_batch_oracle_prompts

    if kind not in KINDS:
        raise KeyError(f"unknown kind {kind!r}; expected one of {KINDS}")
    assert_no_plugin()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_kwargs = dict(ENGINE_KWARGS)
    # ALL of them resident at once, in BOTH arms: a second prefill wave would put some
    # requests in a differently-shaped step from the MIA arm's and make the comparison a
    # scheduling question rather than a routing one.
    if engine_kwargs.get("max_num_seqs") != len(prompts):
        raise RuntimeError(
            f"the batched oracle requires max_num_seqs == {len(prompts)} in BOTH arms (set "
            f"MIA_PARITY_MAX_NUM_SEQS={len(prompts)}); got "
            f"{engine_kwargs.get('max_num_seqs')!r}")
    print(f"[T1] batched oracle engine kwargs = {engine_kwargs}")
    llm = LLM(model=model, enforce_eager=True, **engine_kwargs)
    nn_model = _model_of(llm)
    modules = assert_layer_mapping(nn_model, kind, layers)
    prompt_ids = assert_batch_oracle_prompts(llm.get_tokenizer(), prompts)
    lengths = [len(p) for p in prompt_ids]
    print(f"[T1] batched oracle: {len(prompts)} requests, token counts {min(lengths)}.."
          f"{max(lengths)}, all different, {sum(lengths)} prefill rows total")

    wl = Workload(subsystem="hs" if kind == "capture_hs" else "qk",
                  prompts=tuple(prompts), layers=tuple(layers),
                  max_tokens=int(max_tokens), hooks_on="prefill")
    sampling = sampling_params_for(wl, None)

    layouts: list = []                       # flat input ids, one per forward pass
    fires: dict[int, list] = {int(n): [] for n in modules}

    def _layout_hook(_module, args, _output):
        layouts.append(args[0].detach().clone().cpu())

    def _hs_hook(_module, _args, output, n=None):
        fires[n].append(_post_block_hidden(output).detach().clone())

    def _qk_hook(_module, args, _output, n=None):
        fires[n].append((args[0].detach().clone(), args[1].detach().clone()))

    handles = [_embedding_module(nn_model).register_forward_hook(_layout_hook)]
    for mia_layer, module in modules.items():
        hook = (_hs_hook if kind == "capture_hs" else _qk_hook)
        handles.append(module.register_forward_hook(
            lambda m, a, o, _h=hook, _n=mia_layer: _h(m, a, o, n=_n)))
    try:
        outputs = llm.generate(list(prompts), sampling, use_tqdm=False)
    finally:
        for handle in handles:
            handle.remove()

    # WHICH PASS HELD WHICH REQUEST'S PREFILL, from the token ids and nothing else.
    placement: dict[int, tuple[int, int, int]] = {}
    for pass_index, flat in enumerate(layouts):
        for index, (start, end) in segment_batch_rows(flat, prompt_ids).items():
            if index in placement:
                continue
            placement[index] = (pass_index, start, end)
    missing = [i for i in range(len(prompts)) if i not in placement]
    if missing:
        raise RuntimeError(
            f"requests {missing} were never located in any forward pass ({len(layouts)} "
            f"passes seen, row counts {[int(t.shape[0]) for t in layouts[:4]]}...). Either "
            f"the prefill was CHUNKED -- which splits a request's prompt across passes and "
            f"breaks the contiguous-run assumption this oracle rests on -- or the prompts "
            f"the engine received are not the ones we matched against. Failing rather than "
            f"comparing a subset in silence.")
    for index in sorted(placement):
        pass_index, start, end = placement[index]
        print(f"[T1] batched oracle req{index}: pass {pass_index} rows [{start}:{end}) "
              f"({end - start} rows)")

    written = 0
    for index, output in enumerate(outputs):
        pass_index, start, end = placement[index]
        req_dir = out_dir / f"req{index}"
        _save(req_dir / "generation.safetensors", generation_tensors(output))
        written += 1
        n_prompt = end - start
        if n_prompt != len(output.prompt_token_ids):
            raise RuntimeError(
                f"req{index}: matched {n_prompt} prefill rows but the request reports "
                f"{len(output.prompt_token_ids)} prompt tokens")
        for mia_layer, steps in sorted(fires.items()):
            if not steps:
                raise RuntimeError(
                    f"{kind}: MIA layer {mia_layer} "
                    f"({module_name_for(kind, mia_layer)}) captured NOTHING")
            # One "step" per request: its own prefill rows, taken out of the shared flat
            # pass by the token-id segmentation above. `_assemble_*` then builds exactly the
            # layout the MIA arm's retrieval produces (pad_sequence over one step), so the
            # two trees are comparable key for key with no tolerance.
            if kind == "capture_hs":
                tensors = _assemble_hs([steps[pass_index][start:end]], [n_prompt], mia_layer)
            else:
                q, k = steps[pass_index]
                tensors = _assemble_qk([(q[start:end], k[start:end])], [n_prompt], mia_layer)
            _save(req_dir / f"layer{int(mia_layer):03d}.safetensors", tensors)
            written += 1

    print(f"[T1] wrote {written} batched-oracle artifact files under {out_dir}")
    if written == 0:
        raise RuntimeError(f"the batched oracle produced NOTHING under {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Steering -- the observable is the EFFECT, because there is no artifact
# ---------------------------------------------------------------------------
#
# Capture can be checked against a reference tensor-for-tensor. Steering cannot: it writes
# nothing, it MUTATES the residual stream, so the only thing an outside observer can see is
# what the mutation did to the output distribution. The reference is therefore "the same
# intervention, applied by torch's own forward hook, in a process with no MIA in it", and
# the comparison is on the generated tokens and their logprobs.
#
# THE TRAP (measured in task A6 on this exact workload): steering moved the next-token
# LOGPROBS on 3/3 requests and the sampled TOKEN IDS on 0/3. A token-identity check -- the
# first thing anyone reaches for -- would read a perfectly working steer path as dead. The
# float channel is the signal; the token ids are the weaker, occasionally silent one.
#
# WHERE the intervention lands, exactly. MIA's eager hook (steer_worker.steering_hook)
# takes the decoder block's ``(hidden_states, residual)`` output and replaces element 1 --
# the RESIDUAL -- leaving ``hidden_states`` untouched; the graph path does the same in
# place (install_steer._wrap_layer_class). That is not a detail this reference may choose
# differently: ``adjust_rs`` computes its coefficient FROM the tensor it is applied to
# (``avg_proj - rows @ dir``), so applying it to the residual and applying it to the
# post-block sum are two DIFFERENT interventions with different deltas. This reference
# reimplements MIA's math independently, but it applies it to the same tensor.


def _steer_vector(cfg: dict) -> dict:
    """Load the steering vector independently of MIA (same file, own reader)."""
    raw = torch.load(cfg["vector_path"], weights_only=False)
    vec = {"dir": torch.as_tensor(raw["dir"])}
    if cfg.get("method", "adjust_rs") == "adjust_rs":
        vec["avg_proj"] = torch.as_tensor(raw["avg_proj"])
    print(f"[T1] steer vector {cfg['vector_path']}: dir {tuple(vec['dir'].shape)} "
          f"{vec['dir'].dtype} |dir|={float(vec['dir'].float().norm()):.6f}"
          + (f" avg_proj={float(vec['avg_proj']):.6f}" if "avg_proj" in vec else ""))
    return vec


def _apply_steer(rows: torch.Tensor, cfg: dict, vec: dict) -> torch.Tensor:
    """MIA's two steering methods, reimplemented here from the config dict.

    Written from the definitions, not imported: importing ``mia`` would put the code under
    test inside the oracle (and would trip ``assert_no_plugin``). Both are row-wise, so a
    row slice is the same computation as the whole tensor.
    """
    direction = vec["dir"].to(rows.device, dtype=rows.dtype)
    method = cfg.get("method", "adjust_rs")
    if method == "add_vector":
        return rows + float(cfg.get("coefficient", 0)) * direction.view(1, -1)
    if method == "adjust_rs":
        # Rotate each row's component along `dir` to the vector's stored average
        # projection: delta = (avg_proj - rows.dir) * dir.
        avg_proj = vec["avg_proj"].to(rows.device, dtype=rows.dtype)
        projections = torch.matmul(rows, direction)
        coeff = (avg_proj - projections).unsqueeze(-1)
        return rows + coeff * direction.view(1, -1)
    raise ValueError(f"unknown steering method {method!r}")


def _residual_of(output):
    """The residual-stream tensor inside a decoder block's output, and how to put it back.

    vLLM's blocks return ``(hidden_states, residual)`` with the residual not yet added;
    MIA steers element 1. A block that returns a bare tensor has no split, so the tensor
    itself is the residual stream.
    """
    if isinstance(output, tuple):
        if len(output) < 2 or not torch.is_tensor(output[1]):
            raise RuntimeError(
                f"decoder block returned a {len(output)}-tuple whose element 1 is not a "
                f"tensor; MIA steers element 1, so this reference cannot mirror it")
        return output[1], (lambda new: (output[0], new) + tuple(output[2:]))
    if torch.is_tensor(output):
        return output, (lambda new: new)
    raise RuntimeError(f"decoder block returned {type(output).__name__}, not a tensor/tuple")


def reference_steer(out_dir: Path, model: str = MODEL,
                    workload_name: str = "steer_small") -> Path:
    """Steer with a naive forward hook and record the EFFECT, plus an unsteered control.

    Writes the same tree the MIA steer arm writes -- ``req<N>/generation.safetensors``
    (steered) and ``req<N>/control.safetensors`` (same prompts, steering disarmed) -- so
    ``compare_artifacts.py`` lines the two arms up key-for-key, and so each arm carries its
    OWN liveness control. Both arms take their intervention from the same
    ``capture_workload.steer_config`` dict: the vector path, layer, method, coefficient and
    phase x positions modes cannot drift between them.
    """
    from vllm import LLM

    assert_no_plugin()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wl = WORKLOADS[workload_name]
    cfg = steer_config(wl)
    print(f"[T1] steer config (the SAME dict the MIA arm sends): {cfg}")
    print(f"[T1] steer fingerprint {steer_fingerprint(cfg)}")

    phase, positions = cfg.get("phase"), cfg.get("positions")
    if (phase, positions) != ("decode", "all_tokens"):
        # Not a limitation of the steer path -- a limitation of THIS oracle. A decode pass
        # holds exactly one row per request, which is what lets the reference index rows
        # without touching any batch metadata (the thing it exists to check). Steering a
        # prefill pass would need the request's row span, i.e. the index logic under test.
        raise NotImplementedError(
            f"the steer reference implements phase='decode' x positions='all_tokens'; "
            f"workload {workload_name!r} asks for phase={phase!r} positions={positions!r}")

    engine_kwargs = dict(ENGINE_KWARGS)
    if engine_kwargs.get("max_num_seqs") != 1:
        raise RuntimeError(
            "T1 requires max_num_seqs=1 in BOTH arms (set MIA_PARITY_MAX_NUM_SEQS=1); "
            f"got {engine_kwargs.get('max_num_seqs')!r}")
    print(f"[T1] engine kwargs = {engine_kwargs}")

    llm = LLM(model=model, enforce_eager=True, **engine_kwargs)
    nn_model = _model_of(llm)
    modules = assert_layer_mapping(nn_model, "steer", cfg["optimal_layer"])
    if len(modules) != 1:
        # The pass counter below assumes one fire per forward pass. With N steered layers
        # it would count N per pass and the row plan would silently describe the wrong
        # passes, so refuse rather than mis-index.
        raise NotImplementedError(
            f"the steer reference handles ONE steered layer; got {sorted(modules)}")
    vec = _steer_vector(cfg)
    sampling = sampling_params_for(wl, None)   # no MIA here, so nothing to arm

    prompts = list(wl.prompts)
    steered_outputs, control_outputs = [], []

    for index, prompt in enumerate(prompts):
        rows_seen: list[int] = []
        applied: list[int] = []

        def _steer_hook(_module, _args, output):
            residual, rebuild = _residual_of(output)
            step = len(rows_seen)
            rows_seen.append(int(residual.shape[0]))
            if step == 0:
                return output      # the prefill pass; phase="decode" steers nothing here
            # A decode pass: this request owns the single leading row. MIA steers exactly
            # [query_start_loc[i], query_start_loc[i+1]) = [0, 1); mirror that rather than
            # steering the whole tensor, so a padded tail (if V2 ever grows one) cannot
            # make the two arms compute different things.
            new = residual.clone()
            new[:1] = _apply_steer(residual[:1], cfg, vec)
            applied.append(step)
            return rebuild(new)

        handles = [module.register_forward_hook(_steer_hook)
                   for module in modules.values()]
        try:
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
        finally:
            for handle in handles:
                handle.remove()

        output = outputs[0]
        steered_outputs.append(output)

        # _row_plan hard-fails unless the hook fired exactly once per generated token, so
        # it also proves that pass 0 is THE prefill and every later pass is a decode --
        # which is what the `step == 0` gate above assumes.
        plan = _row_plan(output, len(rows_seen))
        for step, (seen, owned) in enumerate(zip(rows_seen, plan)):
            if seen < owned:
                raise RuntimeError(
                    f"steer: pass {step} handed the hook {seen} rows, fewer than the "
                    f"{owned} this request owns")
            if seen > owned:
                print(f"[T1] NOTE steer: pass {step} tensor has {seen} rows for a "
                      f"{owned}-row request")
        expected = list(range(1, len(plan)))
        if applied != expected:
            raise RuntimeError(
                f"steer reference applied the vector on passes {applied}, expected "
                f"{expected} (every decode pass, never the prefill)")
        print(f"[T1] steer req{index}: {len(plan)} passes, steered {len(applied)} decode "
              f"passes at layer(s) {cfg['optimal_layer']}")

    # Control: identical engine, identical prompts, hooks removed. Same process, same
    # seed, greedy, prefix caching off -- so an inert steer reproduces this EXACTLY.
    for prompt in prompts:
        control_outputs.append(llm.generate([prompt], sampling, use_tqdm=False)[0])

    written = 0
    worst = 0.0
    flipped = 0
    for index, (steered, plain) in enumerate(zip(steered_outputs, control_outputs)):
        a, b = generation_tensors(steered), generation_tensors(plain)
        req_dir = out_dir / f"req{index}"
        _save(req_dir / "generation.safetensors", a)
        _save(req_dir / "control.safetensors", b)
        written += 2
        worst = max(worst, logprob_delta(a, b))
        if not torch.equal(a["token_ids"], b["token_ids"]):
            flipped += 1

    print(f"[T1] steer liveness: max|d(logprob)| steered-vs-control = {worst:.6e} "
          f"(floor {STEER_LIVENESS_ATOL:.1e}); {flipped}/{len(prompts)} requests also "
          f"changed their token ids")
    if worst <= STEER_LIVENESS_ATOL:
        raise RuntimeError(
            f"the REFERENCE steer changed nothing measurable: max|d(logprob)| = "
            f"{worst:.6e} <= {STEER_LIVENESS_ATOL:.1e}. The oracle itself is inert, so "
            f"agreeing with MIA would prove nothing.")

    print(f"[T1] wrote {written} reference artifact files under {out_dir}")
    if written == 0:
        raise RuntimeError(f"reference steer produced NOTHING under {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# T0 -- generation with nothing attached at all
# ---------------------------------------------------------------------------

def reference_generate(out_dir: Path, model: str = MODEL,
                       workload_name: str = "hs_small", graph: bool = False) -> Path:
    """Generate the workload's prompts with NO plugin and NO hooks: T0's capture-off arm.

    T1's reference arm carries torch forward hooks of its own, so T1 compares capture
    against capture. T0 needs the other control: a run with nothing attached anywhere, to
    show that MIA's capture -- and not merely "some hook" -- leaves generation untouched.
    Writes ``req<N>/generation.safetensors`` only.

    ``graph=True`` runs the SAME control under vLLM's FULL CUDA graphs. T0 was eager-only,
    so "capture is an observer" had never been checked on the path where MIA's work is a
    baked in-graph op replayed by vLLM rather than a Python forward hook (task D4).
    """
    from vllm import LLM

    assert_no_plugin()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wl = WORKLOADS[workload_name]
    engine_kwargs = dict(ENGINE_KWARGS)
    if graph:
        # vLLM 0.29 defaults to FULL_AND_PIECEWISE; name FULL so this control replays the
        # same class of graph the MIA arm does.
        engine_kwargs["compilation_config"] = {"cudagraph_mode": "FULL"}
    print(f"[T0] engine kwargs = {engine_kwargs} (graph={graph})")
    llm = LLM(model=model, enforce_eager=not graph, **engine_kwargs)
    _model_of(llm)     # same V2-runner assertion every other leg makes
    sampling = sampling_params_for(wl, None)

    written = 0
    for index, prompt in enumerate(wl.prompts):
        output = llm.generate([prompt], sampling, use_tqdm=False)[0]
        _save(out_dir / f"req{index}" / "generation.safetensors",
              generation_tensors(output))
        written += 1
        print(f"[T0] off-arm req{index}: {len(output.outputs[0].token_ids)} tokens")

    print(f"[T0] wrote {written} capture-off generation files under {out_dir}")
    if written == 0:
        raise RuntimeError(f"capture-off arm produced NOTHING under {out_dir}")
    return out_dir


def reference_for_workload(kind: str, out_dir: Path) -> Path:
    """Run the reference against the shared workload the MIA arm runs.

    Same prompts, same layers, same max_tokens, same engine kwargs -- the whole point of
    `capture_workload`'s arm-neutral half.
    """
    workload = WORKLOADS["hs_small" if kind == "capture_hs" else "qk_small"]
    return reference_capture(kind, workload.layers, list(workload.prompts), Path(out_dir),
                             model=MODEL, max_tokens=workload.max_tokens,
                             granularity=workload.granularity)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Produce one reference arm: a capture tree, the steer effect, or a "
                    "plugin-free generation (T0's capture-off control).")
    ap.add_argument("kind", choices=(*KINDS, "steer", "plain"))
    ap.add_argument("out_dir")
    ap.add_argument("--batched", action="store_true",
                    help="run the BATCHED oracle: all of the workload's prompts in ONE "
                         "batch, prefill rows located from the token ids (task D5 item 8). "
                         "Eager only -- a Python forward hook cannot observe a CUDA-graph "
                         "replay, so no independent oracle for graph mode is constructible "
                         "this way.")
    args = ap.parse_args(argv)
    if args.kind == "steer":
        reference_steer(Path(args.out_dir))
    elif args.kind == "plain":
        reference_generate(Path(args.out_dir))
    elif args.batched:
        workload = WORKLOADS["hs_batch" if args.kind == "capture_hs" else "qk_batch"]
        reference_capture_batched(args.kind, workload.layers, list(workload.prompts),
                                  Path(args.out_dir), model=MODEL,
                                  max_tokens=workload.max_tokens)
    else:
        reference_for_workload(args.kind, Path(args.out_dir))
    return 0


# vLLM may spawn helper processes that re-import this module; without the guard a child
# would boot a second engine on an exclusive_process GPU.
if __name__ == "__main__":
    raise SystemExit(main())
