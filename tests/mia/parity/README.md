# The parity oracle

Does arming capture change what the model produces? This directory answers that against a
reference that does not involve the plugin at all.

| File | What it establishes |
|---|---|
| `test_t0_noninterference.py` | **T0** — capture is an observer: identical generation with hooks on and off |
| `t1_reference.py` | **T1** — captured artifacts against an independent fp32 HuggingFace oracle |
| `capture_workload.py` | the shared workload both tiers drive |
| `compare_artifacts.py` | the comparator |

T0 is the one that matters most: if capture changed generation, nothing else here would be
worth measuring. It needs a GPU, carries the `gpu` marker, and skips itself without one.

```bash
python tests/mia/parity/test_t0_noninterference.py {off|hs|qk} <out_dir>
python tests/mia/parity/t1_reference.py <kind> <out_dir>
```

## References to things that are not here

Comments in these files cite drivers that are **not distributed** — batch-scheduler job
scripts (`run_parity.sh`, `run_steer_ab_*.sh`), the T2 invariant and T3 cross-version drivers,
and the tolerance notes they were tuned against — along with numbered jobs such as
"LSF 1731353".

Those belonged to one GPU cluster and to the development history of this port: queue names,
node constraints and absolute paths that would be wrong anywhere else, plus tolerance
bookkeeping for comparisons against an engine version this tree no longer supports. They are
left out deliberately.

The citations stay because they are provenance: they record which run justified a particular
number. Read them as references, not as instructions to run something.
