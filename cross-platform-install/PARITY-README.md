# MLR55: Colab parity

For Linux GPU machines and GitHub Actions, see [GPU-README.md](GPU-README.md).

**Paste these commands into your Mac’s Terminal, one box at a time.**

## 1. Enter the checkout

```bash
cd /Users/timothyburley/opensource/vLLM-Hook-MLR-55
```

## 2. Check for a session

```bash
colab --auth oauth2 sessions
```

If `mlr55` exists, reuse it and skip the next box.
If it is missing, start a new GPU session (uses your Colab allowance):

```bash
colab --auth oauth2 new -s mlr55 --gpu A100
```

If the CLI itself is missing, install it with `uv tool install google-colab-cli`.

## 3. Run parity

```bash
python3 cross-platform-install/colab_cli.py parity
```

The script:

1. Uploads your saved source changes.
2. Checks the GPU and installs any missing CUDA software.
3. Builds Triton CPU once, or restores its saved wheel.
4. Saves that wheel to your Mac **before installing CPU vLLM or running tests**.
5. Installs the official prebuilt vLLM CPU wheel.
6. Runs the four CPU–CUDA comparisons inside Colab.
7. Saves the comparison report to your Mac, even if a comparison fails.

**The first Triton build can still take a long time. vLLM is no longer compiled.**
Look for `Triton CPU wheel safely saved on your Mac:`. After that message,
losing the Colab session will not lose that wheel.

Wheel backups and reports are in `.colab-cache/` inside MLR55. Keep that folder.
It is excluded from Git and source uploads. A new session uploads the saved wheel
instead of recompiling it. The cache is for Linux x86_64/Python 3.12, not your Mac’s CPU runtime.

Defaults: absolute tolerance `0.0001`, relative tolerance `0.0011` (0.11%).

The revised wheel workflow has local regression tests. Its full new-session
Colab run still needs validation. If it fails, keep the session and share the
last error lines. A failed setup step now stops the script before tests. Do not create another session while a build is active.

## Build and back up without running tests

```bash
python3 cross-platform-install/colab_cli.py prepare
```

Then use `parity` when ready. The saved wheel is reused.

## After another edit

Run the same `parity` command. It uploads your changes and reuses the environment
and saved wheel. No GitHub push is required.

## Is a quiet build still working?

Leave it running. In a second Terminal window:

```bash
colab --auth oauth2 console -s mlr55
```

Inside that remote console:

```bash
ps -eo pid,etime,pcpu,comm --sort=-pcpu | head -15
```

Compiler processes using CPU mean it is still building.

## Browser IDE issue

Use the CLI for this workflow. The browser can connect to a different machine,
matching [issue #24](https://github.com/googlecolab/google-colab-cli/issues/24).
Browser login when the CLI asks you to sign in is separate and still works.

## Login problems

**Wrong account:** clear the saved login, then sign in again:

```bash
rm -f ~/.config/colab-cli/token.json
```

```bash
colab --auth oauth2 sessions
```

**`redirect_uri_mismatch`:** an older custom configuration may conflict with the
new CLI. If `~/.colab-cli-oauth-config.json` exists, back it up:

```bash
mv ~/.colab-cli-oauth-config.json ~/.colab-cli-oauth-config.json.backup-$(date +%s)
```

Then retry `colab --auth oauth2 sessions`.

## Finish

Once the script has saved the wheel and report locally, stop the session:

```bash
colab --auth oauth2 stop -s mlr55
```

[Comparison details and cache behavior](PARITY-DETAILS.md)
