## Summary
<!-- Briefly describe what this PR does and why -->

## Type of contribution
- [ ] New worker
- [ ] New analyzer
- [ ] New use case
- [ ] Bug fix (fill in the Bug fix section below)
- [ ] Other (describe below)

## Files modified
<!-- List the files changed. -->

The following files are **core infrastructure**. Please discuss with maintainers before modifying:
`mia/llm.py`, `mia/client.py`, `mia/core/_plugin.py` and anything under `mia/core/`

- [ ] I have NOT modified core files, OR I have discussed the change with maintainers

## Plugin architecture checklist
- [ ] New workers/analyzers are registered with `PluginRegistry` in `mia/__init__.py`
- [ ] New workers are implemented as `worker_extension_cls` (for in-band artifact retrieval and concurrent request support); they use forward hooks, so they run in eager mode only
- [ ] Examples or notebooks are included for new features
- [ ] If I added a new use case, I appended a row to `docs/use_cases/README.md` with my GitHub handle. If the use case has a writeup, I placed it under `docs/use_cases/<name>.md` and linked it from the table.
- [ ] If I added a config key or env var, I documented it in `docs/configs.md`

## Testing
- [ ] `pre-commit run --all-files` passes
- [ ] `pytest tests -q -m "not gpu"` passes
- [ ] New behaviour has a test; a bug fix has a test that fails without the fix
- [ ] Touched capture, steering or delivery: ran it on a GPU
      (CUDA graphs and eager, offline and served)

What I ran:

## Bug fix
Bug fix PRs only; delete this section otherwise.

- [ ] Brief explanation of the bug and the fix
- [ ] Code or steps to reproduce the bug
- [ ] Error logs or outputs from the relevant settings: CPU or GPU, eager or CUDA graphs
- [ ] The fix has as few unrelated changes as possible
- [ ] Tests showing the bug is fixed
- [ ] Test results showing other features still work
- [ ] Performance or latency impact, if any
- [ ] Anything else a reviewer should know

## Related issue
Closes #

## Contribution acknowledgement
If this contribution is included in a future version of the vLLM-Hook technical report, would you like to be credited as a co-author?

- [ ] Yes, please include me as a contributor
- [ ] No, thanks

If yes, please provide:
- **Name**:
- **Affiliation**:
- **One-sentence description of your contribution**:
