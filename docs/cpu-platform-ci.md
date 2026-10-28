# CPU platform validation (MLR-58)

This fork branch starts at IBM/vLLM-Hook PR #47 revision
`d943ce6c5104b81c83a6f3643806353efbf1ebf9`. Scenario scripts and assertions are
unchanged. Every run records the exact tested head SHA, workflow SHA, host,
dependencies, phase exit codes and test output.

`CPU platform validation` runs on pushes to `codex/MLR-58-linux-cpu-actions`
and pull requests in the fork. Linux is blocking. macOS ARM64 is experimental (`continue-on-error`); inspect their job steps and artifacts,
not just the overall green status. Branch protection is not changed by this work.
Windows is omitted. An installation failure is not a passed behavioral test. macOS builds
the pinned vLLM source; Linux uses the PR's checksummed CPU release wheel.

All environments use Python 3.12, vLLM 0.29.0 and Triton CPU source
`c8e7cc932320d05d08f39ffc99bfda32dc4f22fa`. Compilation uses at most two jobs
and one link job, with a 330-minute job limit. No `-march=native` is used.
The wheel cache includes OS/image, architecture, Python, compiler versions,
source revision, build settings and helper contents. No partial-key fallback is
allowed. A miss builds normally; a hit checks the saved wheel's checksum.
The compiled wheel is also uploaded as a downloadable artifact.

## Launch and download

Push a commit to this branch to run both jobs. A pull request also starts
both jobs. For manual dispatch, GitHub requires the workflow on the fork's
default branch first; this branch does not change the default branch.
Once installed there:

```sh
gh workflow run cpu-platforms.yml --repo tburleyinfo/vLLM-Hook \
  --ref codex/MLR-58-linux-cpu-actions \
  -f test_sha=FULL_40_CHARACTER_COMMIT -F fresh_build=true
```

Open the run in the fork's Actions tab. Download `mlr58-PLATFORM-evidence-*`
and `mlr58-PLATFORM-triton-wheel-*` from its artifacts section, or:

```sh
gh run download RUN_ID --repo tburleyinfo/vLLM-Hook --dir results/RUN_ID
```

Evidence upload runs even after setup or test failure. JUnit and per-scenario
child logs exist when pytest has run; an earlier failure has phase logs instead.
Runner loss or forced cancellation can prevent artifact upload.

## Repeat execution

Rerun the same workflow run after the first Linux build completes. The exact
cache key must match, and `wheel-source.txt` must say `reused compiled wheel`.
Check the wheel checksum matches the first run. A successful first run plus a
successful cache-hit run is required for MLR-58 acceptance. Neither is established
by adding this workflow. These tests establish scenario execution only; they do
not prove CPU/GPU numerical parity or a steering effect (coefficient is zero).
