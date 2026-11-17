# Contributing to vLLM-Hook MIA

Thanks for helping. This page covers how to set up, where things go, how to test, and what a PR
needs. For a security problem, don't open an issue: see [SECURITY.md](SECURITY.md).

## Before you start

- **Bug:** open an issue with your setup (MIA commit, vLLM version, model, `enforce_eager` or
  CUDA graphs), what you saw, a short repro, and the cause if you found it. Issues
  [#50](https://github.com/IBM/vLLM-Hook/issues/50) and
  [#55](https://github.com/IBM/vLLM-Hook/issues/55) are good examples.
- **Feature, or a change to core files:** open an issue first and agree on the approach. The core
  files are `mia/llm.py`, `mia/client.py`, `mia/core/_plugin.py` and anything under `mia/core/`.
- **New analyzer, use case or demo:** a PR is fine; mention it in an issue if it is large.

## Set up

Follow [Installation](README.md#-installation) in the README (a dedicated environment, the pinned
versions in `requirement.txt`, then `pip install -e . --no-deps`). Then add the dev tools:

```bash
pip install pytest pre-commit
pre-commit install        # runs the checks on every commit
```

The CPU tests and the checks run without a GPU. Anything that boots an engine needs one.

## Where things go

| You are adding | Put it in | Also |
|---|---|---|
| An analyzer | `mia/analyzers/` | Register it with `PluginRegistry.register_analyzer` in `mia/__init__.py` ([how](README.md#adding-a-worker-or-analyzer)) |
| A worker | `mia/workers/` | A `worker_extension_cls` class; register it with `PluginRegistry.register_worker` in `mia/__init__.py`. New workers use forward hooks, which run in eager mode only |
| A use case | `examples/demo_<name>.py`, `model_configs/<kind>/` | A row in [`docs/use_cases/README.md`](docs/use_cases/README.md) with your GitHub handle; a writeup in `docs/use_cases/<name>.md` if you have one |
| A test without a GPU | `tests/unit/` | |
| A test that boots an engine | `tests/use_cases/` | Mark it `@pytest.mark.gpu` and `@requires_gpu` (see `tests/conftest.py`) |
| A config key or env var | the code | Document it in [`docs/configs.md`](docs/configs.md) |

Keep to the dependency direction in [`mia/README.md`](mia/README.md#dependency-direction).

## Test

```bash
pre-commit run --all-files            # lint and file checks (CI runs these)
pytest tests -q -m "not gpu"          # CPU tests, about 30 s (CI runs these)
pytest tests/use_cases -m gpu         # engine tests; needs a GPU
```

CI has no GPU, so engine behaviour is checked by hand. If your change touches capture, steering or
delivery, run it on a GPU in both modes the maintainers will ask about:

- **CUDA graphs and eager:** `enforce_eager=False` (the default) and `enforce_eager=True`.
- **Offline and served:** `MiaLLM`, and `vllm serve` with `MiaClient` or the `openai` client.

Put the results in the PR as a small table (mode, `main`, your branch), as in
[#52](https://github.com/IBM/vLLM-Hook/pull/52).

## Code style

- [Ruff](https://docs.astral.sh/ruff/) lints the code (`ruff.toml`); pre-commit runs it.
- Lines up to about 100 characters.
- A broad `except Exception` needs a reason: `except Exception:  # noqa: BLE001 -- why`.
- Don't reformat code you aren't changing. It hides the real change from reviewers.

## Pull requests

- One topic per PR, linked to its issue (`Closes #N`).
- Fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md), including the testing section.
- Commit messages say what changes, in the imperative: "Compute the adjust_rs projection on
  hidden_states + residual".
- CI must pass: lint and CPU tests.
- New behaviour comes with a test; a bug fix comes with a test that fails without it.
- New features come with an example or notebook.

### Bug fix PRs

A bug fix PR should include the following (the template's Bug fix section lists the same):

- A brief explanation of the bug and the fix.
- Code or steps to reproduce the bug.
- Error logs or outputs from the relevant settings: CPU or GPU, eager or CUDA graphs.
- The fix itself, with as few unrelated changes as possible.
- Tests showing the bug is fixed.
- Test results showing other features still work.
- Any performance or latency impact, if there is one.
- Anything else a reviewer should know.
