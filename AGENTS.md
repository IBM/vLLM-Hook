# AGENTS.md

Guidance for coding agents working in this repository. Humans: see
[CONTRIBUTING.md](CONTRIBUTING.md).

## What this is

vLLM-Hook MIA is a vLLM plugin that captures model internals (hidden states, Q/K attention) and
steers activations during inference, under CUDA graphs by default. Entry points: `MiaLLM`
(`mia/llm.py`, offline) and `MiaClient` (`mia/client.py`, for `vllm serve`).
[`mia/README.md`](mia/README.md) maps the package.

## Commands

```bash
pip install -r requirement.txt && pip uninstall -y torchcodec && pip install -e . --no-deps
pip install pytest pre-commit

pre-commit run --all-files        # lint and file checks
pytest tests -q -m "not gpu"      # CPU tests, ~30 s; run these before finishing
pytest tests/use_cases -m gpu     # boots real engines; GPU only
```

Run the CPU tests and pre-commit before you report a task as done.

## Rules

- **Pinned stack.** vLLM 0.32.0 and torch 2.13.0 (`requirement.txt`). Don't change versions in
  passing; a vLLM bump is its own PR.
- **Ask before touching core files:** `mia/llm.py`, `mia/client.py`, `mia/core/_plugin.py` and
  `mia/core/`. `mia/core/runner.py` is the only module that reaches into vLLM's V2 model-runner
  internals; keep it that way.
- **Two execution paths.** Every capture and steering feature runs under CUDA graphs (ops baked
  into the graph, `mia/core/hooks/`) and in eager mode (forward hooks). A change to one usually
  needs the same change in the other. Python forward hooks don't run during CUDA-graph replay.
- **vLLM layer outputs.** Decoder layers return `(hidden_states, residual)`; the residual stream
  is their sum. Some models (Gemma 4) return `(hidden_states, None)`.
- **Engines use `spawn`.** `tests/conftest.py` sets it; scripts that build engines need
  `if __name__ == "__main__":` and `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- **Dependency direction:** `llm` and `client` use `core`; `core` uses `vllm`.
- **Tests:** CPU tests go in `tests/unit/`; tests that boot an engine go in `tests/use_cases/`
  with `@pytest.mark.gpu` and `@requires_gpu`.
- **Docs:** a new config key or env var goes in `docs/configs.md`; a new use case gets a row in
  `docs/use_cases/README.md`.

## Style

- Ruff (`ruff.toml`), lines up to about 100 characters.
- `except Exception` only with a reason: `# noqa: BLE001 -- why`.
- Match the surrounding code; don't reformat lines you aren't changing.
- Commit messages: imperative, one line saying what changes.
