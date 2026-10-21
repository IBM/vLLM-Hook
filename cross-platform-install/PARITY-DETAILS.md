# Same-machine parity: details

Start with [the short checklist](PARITY-README.md). Browser IDE access is affected
by [reported issue #24](https://github.com/googlecolab/google-colab-cli/issues/24).

## One-command Colab workflow

`python3 cross-platform-install/colab_cli.py parity` uploads the current source,
prepares an independent CPU environment, and runs the comparator inside the
existing CLI session. It does not allocate or stop hardware. CPU and CUDA run
sequentially and keep all intermediate results on that machine.

## Installation and durable backups

Dependencies and source revisions remain in `pyproject.toml`. On Linux x86_64,
vLLM comes from the official v0.29.0 CPU release wheel. Its SHA-256 is stored
separately in `[tool.mia.colab]`: the installer verifies the download before
passing a local wheel path to uv. uv source URLs contain no hash fragments.
Other platforms retain their existing source route. The installer explicitly
forbids building vLLM or Triton during environment installation, so a missing
binary does not silently trigger a second source build.

Triton CPU is the exception: its pinned Git revision is built into a CPython 3.12
Linux x86_64 wheel when no backup exists. Build dependencies come from the
`cpu-build` group. There is no vLLM source compilation and no Rust installation
in this Colab path. The first Triton build may still take substantial time.

The local command separates these phases:

1. Upload source and locate a cached Triton wheel for the exact pinned revision.
2. Restore that wheel to Colab, or build it there with `uv build --wheel`.
3. Download the wheel to the Mac in 8 MiB pieces; check piece and whole-file hashes
   and wheel metadata. Commit the backup only after successful validation.
4. Install the CPU profile with `uv sync`, selecting that local wheel and the
   official vLLM binary. The uploaded pyproject is temporarily scoped to Linux
   x86_64/Python 3.12, then restored even if the install fails.
5. Reuse or install the CUDA extra in the CLI environment. Run comparisons only
   after the local wheel backup succeeded.
6. Download the final comparison report, including reports of failed comparisons.

Each remote step writes a unique completion record. The local script checks it
instead of trusting the CLI exit code, which can be zero even after a remote
exception. A failed or unverified setup step stops the workflow before tests.

The CPU environment remains separate at `/content/mlr55-cpu`. Source uploads
repoint editable MIA installations without recompiling the native packages.
The legacy `install` command reads the CUDA extra and test group from pyproject.
The dedicated-MIA removal of unused torchcodec is preserved.

Backups live under `<checkout>/.colab-cache/<triton-revision>/`; reports live under
`<checkout>/.colab-cache/reports/`. This folder is ignored by Git. Keep it outside
throwaway worktrees if you intend to delete the checkout; `--cache-dir /path/to/cache`
selects a different parent folder. Only a completed, checksum-verified backup is
reused. Damaged backups produce an error rather than silently recompiling.

A wheel is tied to its Triton revision, CPython 3.12, and Linux x86_64. The manifest
records its build system's glibc version. Restoring to an older glibc is rejected;
use a compatible runner or a separate cache/build for that platform. The wheel
has not been validated across all CPU models or Linux distributions. It does not
replace the Mac CPU/Metal environments. For GitHub runners, preserve this cache as
a workflow artifact/cache and use a compatible Linux/Python environment; no GitHub
workflow or external publication is added by this script.

When the same Colab session already has the matching wheel, the local script
verifies it rather than re-uploading/downloading the binary on every source edit.
`prepare` performs the backup and CPU installation without running comparisons.
`parity` performs those steps and then runs the comparisons. Neither action
allocates hardware automatically.

## What is compared

The runner resolves `facebook/opt-125m` to one immutable Hugging Face snapshot.
Both backends use its exact local weights and tokenizer. Each scenario starts
in a new process, uses float32, seed 0, eager mode, and one generated token.
CPU hides CUDA devices and must select vLLM CPU; CUDA must select vLLM CUDA.

| Scenario | Compared observations |
| --- | --- |
| Hidden states | Two layers, expected tensor shapes, captured values. |
| Attention tracker | Scores for identical heads and prompt token ranges. |
| Core reranker | Document scores and exact ranking order. Near ties that reorder documents fail. |
| Activation steering | Baseline and nonzero-steered log probabilities for ten fixed token IDs, plus the difference between them. |

The steering vector is `sin(arange(hidden_size))` at layer 1. A zero coefficient
baseline and coefficient 1 run use identical prompts. A missing measured effect
fails instead of passing a no-op steering path. This samples ten token scores;
it does not compare the full vocabulary or establish every steering behavior.

Input token IDs, prompts, model revision/path, source hash, capture configuration,
and sampling settings must match exactly. Layer keys, list lengths, tensor shape
and dtype must match. Missing observations, child failures, and non-finite values
fail. Numeric values use `abs(cuda - cpu) <= atol + rtol * abs(cpu)`.

Default tolerances are `atol=1e-4`, `rtol=1.1e-3` (0.11% relative).
The relative default was increased from 0.1% at the user’s request after the
first completed steering comparison narrowly exceeded that threshold. They have not been calibrated on
repeated CPU–CUDA hardware runs. Changing them changes the claim; record and justify any
change. The runner reports maximum absolute error and comparison counts.
Use explicit flags if needed, for example `--atol 0.0001 --rtol 0.0011`.

## Reports

A fresh `/content/mlr55-parity-...` directory contains the model/source manifest,
per-backend observations and versions, scenario logs, and `report.json`.
The command returns nonzero for failed comparisons or execution failures.
`/content/mlr55-last-parity-path.txt` records the latest report directory.

The pre-existing `test` command still runs only CUDA behavioral checks.
Use **`parity`** for the actual comparisons. The pytest parity cases can also
validate a completed local report when `MIA_PARITY_REPORT` points to it.
Without that report they skip with instructions for running the comparator.

## Standalone use on an already provisioned machine

```bash
python3 cross-platform-install/run_parity.py --cpu-python /absolute/path/to/cpu/bin/python --cuda-python /absolute/path/to/cuda/bin/python --output /absolute/path/to/new-results
```

The Python environment running this command needs no ML dependencies; it delegates
model download to the CUDA interpreter. That interpreter needs huggingface_hub.
Each backend must already have MIA and its runtime installed. Use a new result
folder for every run. Existing observations are rejected rather than reused.

## Limits and troubleshooting

- The previous source-build run passed hidden-state, attention, and reranker
  comparisons. Steering stopped on an invalid log-probability count, now fixed.
  The revised prebuilt-wheel and backup workflow still needs live validation;
  local regression tests cover its control flow and transfer integrity.
- First setup builds only the pinned Triton CPU wheel. It may still take a long
  time. A build failure is a setup failure, not a parity mismatch.
- Triton needs a C++ compiler and Git. Missing native tools are installed with
  apt when running as root. vLLM uses the published binary and needs no compiler.
- If the session is reclaimed before the local backup completes, that unfinished
  work may be lost. Once the local save is confirmed, future sessions can restore it.
- A failed CPU setup has no ready marker. Fix the displayed error, then rerun
  `parity`; do not create another GPU session.
- A CLI timeout does not prove remote work stopped. Check its status before retrying.
- Engine logs are saved in the report folder. Failures print their last 30 lines.
- Do not upload changes while a run is active. Keep model caches and environments
  outside uploaded source folders.
- This covers eager CPU–CUDA cases, not CUDA graphs, speed, quantized model parity,
  all model families, or multi-GPU execution.
- Metal generation works separately, but this checkout lacks equivalent MIA Metal
  hook scenarios. CPU–Metal parity is not enabled by this runner.

## Local verification without a GPU

```bash
python3 -m unittest discover -s tests/cpu_behavioral -p 'test_parity*.py' -v
python3 -m unittest discover -s tests/cpu_behavioral -p 'test_run_parity.py' -v
```

## Sources

- [Profiles and pinned sources](../pyproject.toml)
- [vLLM CPU installation at the pinned release](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/getting_started/installation/cpu.x86.inc.md)
- [Triton CPU pinned build instructions](https://github.com/triton-lang/triton-cpu/tree/c8e7cc932320d05d08f39ffc99bfda32dc4f22fa)
- [Official Colab CLI](https://github.com/googlecolab/google-colab-cli)
