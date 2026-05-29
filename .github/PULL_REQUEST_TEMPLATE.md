## Summary
<!-- Briefly describe what this PR does and why -->

## Type of contribution
- [ ] New worker
- [ ] New analyzer
- [ ] Bug fix
- [ ] Other (describe below)

## Files modified
<!-- List the files changed. -->

The following files are **core infrastructure**. Please discuss with maintainers before modifying:
`hook_llm.py`, `_hook_plugin.py`, `registry.py`, `hook_client.py`, `run_utils.py`, `shm_utils.py`, `workers/_common.py`

- [ ] I have NOT modified core files, OR I have discussed the change with maintainers
- [ ] If I added a new worker/analyzer, I registered it in `__init__.py`

## Plugin architecture checklist
- [ ] New workers/analyzers are registered via `PluginRegistry` in `__init__.py`
- [ ] New workers are implemented as `worker_extension_cls`  (for in-band artifact retrieval and concurrent request support)
- [ ] Examples or notebooks are included for new features

## Testing
<!-- Describe how you tested this. Which demo/example did you run? -->

## Related issue
<!-- Link any related issue: Closes #123 -->

## Contribution acknowledgement
If this contribution is included in a future version of the vLLM-Hook technical report, would you like to be credited as a co-author?

- [ ] Yes, please include me as a contributor
- [ ] No, thanks

If yes, please provide:
- **Name**:
- **Affiliation**:
- **One-sentence description of your contribution**:
