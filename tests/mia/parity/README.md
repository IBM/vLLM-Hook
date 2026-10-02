# The parity oracle

The correctness suite behind the CUDA-graph capture and steering paths. It answers one
question — *does arming capture or steering change what the model produces?* — against a
reference that does not involve this plugin at all.

| Tier | File | What it establishes |
|---|---|---|
| T0 | `test_t0_noninterference.py` | capture must not change generation: same prompts, armed vs unarmed |
| T1 | `t1_reference.py` | captured artifacts against an independent fp32 HF oracle |
| T2 | `t2_invariants.py` | invariants that must hold across modes (eager vs FULL graph, fused vs reference steer, host vs GPU routing) |
| T3 | `t3_crossbranch.py` | the same oracle run against an older engine, to separate our drift from vLLM's |

`compare_artifacts.py` is the comparator; `capture_workload.py` is the shared workload;
`tp_parity_probe.py` covers the tensor-parallel layouts. Tolerances, and the measurement
behind each one, are in [`TOLERANCES.md`](TOLERANCES.md).

## A note on the job scripts

Several comments here refer to driver scripts (`run_parity.sh`, `run_crossbranch.sh`,
`run_steer_ab_*.sh`) and to numbered jobs such as "LSF 1731353". **Those scripts are not part
of this repository.** They were the batch-scheduler wrappers for one particular GPU cluster —
queue names, node constraints and absolute paths that would be wrong, or actively misleading,
anywhere else — so they are deliberately not distributed.

The references are kept because they are provenance: they record which run produced a
measured bound, so a number in `TOLERANCES.md` can be traced to the job that justified it.
Read them as citations, not as instructions to run something.

Everything needed to *reproduce* the checks is here. The Python entry points are runnable
directly, and each prints its own verdict:

```bash
python tests/mia/parity/test_t0_noninterference.py {off|hs|qk} <out_dir>
python tests/mia/parity/t2_invariants.py --print-bands     # echo every enforced bound
python tests/mia/parity/t1_reference.py <kind> <out_dir>
```

The tiers that need a GPU say so and skip themselves without one. The band arithmetic, the
comparators and the policy gates are pure Python and run in the hermetic suite:

```bash
pytest tests/ -q -m "not gpu"
```
