## Summary
<!-- Briefly describe what this PR does and why -->

## Type of contribution
- [ ] New worker
- [ ] New analyzer
- [ ] Bug fix
- [ ] Other (describe below)

## Files modified
<!-- List the files changed. Note: hook_llm.py should not be modified. -->

- [ ] I have NOT modified `hook_llm.py`

## Plugin architecture checklist
- [ ] New workers/analyzers are registered via `PluginRegistry` in `__init__.py`
- [ ] New workers extend `V1Worker` (not `HookLLM`)
- [ ] `hooks_on=(prefill, generate)` flag is set correctly for any new worker registration
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
