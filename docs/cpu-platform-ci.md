# CPU platform checks

The **CPU platform validation** workflow runs four scenarios: hidden-state
capture, attention tracking, activation steering, and CoRe reranking.

| Platform | Setup | Effect on workflow result |
| --- | --- | --- |
| Linux x86_64 | Prebuilt vLLM CPU package | A failure fails the workflow. |
| macOS ARM64 | vLLM built from source | Experimental; failures are allowed. |

Both jobs use Python 3.12, vLLM 0.29.0, and a pinned Triton CPU version.
A passing workflow does not necessarily mean the experimental macOS job passed.

## Run the checks

Checks start automatically when a pull request opens or its branch changes.
New commits cancel older runs for the same PR.
Branch protection controls whether a failing check prevents merging.

To run manually, the workflow must first be present on the default branch.
Then open **Actions → CPU platform validation → Run workflow** and select a branch
to test its latest commit. Optionally enter a full commit SHA to test a specific
revision. Enable **fresh_build** to bypass the compiled-package cache.

## Find results

Open a run in **Actions** and select a platform job to see its progress.
Download its files from the run's **Artifacts** section:

- **evidence**: host details, tested commit, dependency versions, and setup logs.
  Includes scenario logs and JUnit test results when tests run.
- **triton-wheel**: the compiled Triton CPU package and its checksum, if built.

Files are retained for 30 days. Failed setup still uploads available logs;
a job that never gets a runner cannot produce them.

## Reuse the compiled package

Compatible runs reuse the cached Triton CPU package. A cache miss builds it.
To verify reuse, rerun a completed run and check that `wheel-source.txt` in the
evidence download says `reused compiled wheel`.
