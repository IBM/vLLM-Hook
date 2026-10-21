# GPU tests without Colab

One command installs the project and runs four CUDA scenarios: hidden states,
attention tracking, activation steering, and reranking. Uses a small OPT-125m model.
No Google account, notebook, upload, or GitHub push is needed to run it on a GPU machine.
Dependencies come from `pyproject.toml`.

## On a Linux GPU machine

Requires Linux x86_64, a working NVIDIA driver, git, Python 3.12, and internet access.
Run from the checkout:

```bash
python3 -m pip install uv
python3 cross-platform-install/gpu_ci.py cuda
```

For CPU versus GPU numerical comparisons **on that same machine**:

```bash
python3 cross-platform-install/gpu_ci.py parity
```

The CPU and CUDA packages use separate environments because they provide different
builds of the same libraries. Both use the CPUs already present in the GPU machine.
Results stay in `.gpu-ci/results/`. CUDA checks produce JUnit XML; parity produces
`report.json` plus per-scenario logs and captured values. A failure exits nonzero.
Parity defaults: absolute tolerance `0.0001`, relative tolerance `0.0011`.

## GitHub: set up once

1. Create an **Ubuntu x64 GPU larger runner** in your organization's
   **Settings → Actions → Runners → New GitHub-hosted runner**. Choose the T4 GPU
   size and give the repository access to its runner group.
2. In the repository, open **Settings → Secrets and variables → Actions → Variables**.
3. Add `GPU_RUNNER_LABEL`. Its value is the runner name you just created.
4. Merge `.github/workflows/gpu-tests.yml` and the scripts onto the default branch.

GitHub's hosted GPU offering uses a Tesla T4 with 16 GB VRAM. Larger runners require
an eligible organization plan and incur usage charges. There is no universal
`ubuntu-t4` label; this workflow uses your runner's name.
See [GitHub's runner setup](https://docs.github.com/en/actions/how-tos/manage-runners/larger-runners/manage-larger-runners)
and [GPU specifications](https://docs.github.com/en/actions/reference/runners/larger-runners).

## Choose when it runs

**Manual:** Actions → GPU tests → Run workflow → choose `cuda` or `parity`.
Start with `cuda` to check the new T4 runner.

**Optional per PR:** create repository labels `gpu-tests` and `gpu-parity`.
Apply `gpu-tests` for CUDA checks, or `gpu-parity` for numerical comparisons.
The workflow runs when the label is added and on subsequent commits. A PR opened
with either label already applied also qualifies. Parity takes precedence.

**Automatic on PR opening and updates:** add repository variable
`GPU_TESTS_ON_PR` with value `true`. By default it runs CUDA checks.
To run parity instead, also set `GPU_TEST_SUITE` to `parity`.
Leave `GPU_TESTS_ON_PR` unset to keep runs optional.

GitHub may require approval for workflows from fork contributors. The workflow
uses `pull_request`, read-only repository access, and no secrets.
Do not make the GPU job a required branch-protection check while its execution is optional.

## Builds and caches

CUDA uses published wheels for vLLM, Torch, and Triton. If compatible wheels do not
exist, installation fails instead of silently starting a large source build.

Parity additionally needs Triton CPU, which is built from its pinned source revision.
On Actions, that build runs first on `ubuntu-22.04`, then the wheel is passed to the
GPU job. Both CPU setup and the comparisons happen inside the workflow.
The verified vLLM CPU release wheel is downloaded, not built.

The CPU wheel is cached before the GPU job starts. A failed GPU test therefore does
not discard a successful build. A cold cache still requires a source build; cache
retention is controlled by GitHub. Run `parity` manually on the default branch once
to seed a cache usable by PRs. PR-only caches are scoped to that PR.

Use the Ubuntu 22.04-or-newer NVIDIA image. The CPU wheel cache is Linux x86_64,
Python 3.12, Ubuntu 22.04 specific. Do not seed it from the Colab cache: its newer
system libraries may make that wheel incompatible. Changes to dependency settings
produce a new cache key. No environments or credentials are cached.

Actions saves the run log and results as `gpu-results`. Comparisons happen on the
runner; no separate machine or external comparison service is involved.

## Call from another workflow

```yaml
jobs:
  gpu:
    uses: ./.github/workflows/gpu-tests.yml
    with:
      suite: parity
```

Or add `python3 cross-platform-install/gpu_ci.py cuda` as a step in an existing
Linux GPU job after checkout, Python 3.12 setup, and installing `uv`.

To invoke the existing checks directly in an already prepared CUDA environment:

```bash
MIA_REQUIRED_BACKENDS=cuda python -m pytest \
  tests/cpu_behavioral/test_vllm_cpu_fixtures.py \
  -k 'test_vllm_demo_scenario and cuda' -vv -s
```

These four CUDA checks exercise behavior. Choose `parity` when you need numeric
CPU/GPU comparisons, including nonzero steering. This is not the full large-model
or gated-model test suite.

## Validation status

The orchestration and workflow are checked locally. No hosted T4 runner has been
created or exercised yet. The first manual run must establish compatibility of
this project's pinned vLLM/Torch/Triton versions with the actual driver and T4 image.
The script prints the GPU, device capability, and Torch CUDA version and requires a
real CUDA operation before starting tests. It does not silently skip a missing GPU.
