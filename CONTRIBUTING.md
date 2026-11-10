# Contributing

[fork]: https://github.com/IBM/vLLM-Hook/fork
[pr]: https://github.com/IBM/vLLM-Hook/compare
[released]: https://help.github.com/articles/github-terms-of-service/

We are pleased that you would like to contribute to vLLM-Hook. We welcome both reporting issues and submitting pull requests.

## Reporting issues

Please include this information in the issue:

- What commit or release of vLLM-Hook are you running (`pip show vllm-hook`, or `git rev-parse HEAD` for a checkout)? If it is not the latest on `main`, check whether the problem persists with the latest version.
- Python, vLLM and torch versions. MIA is validated on vLLM 0.32.0 with torch 2.13.0 (see [`requirement.txt`](requirement.txt)).
- GPU model, CUDA version and the number of GPUs (tensor-parallel size).
- The worker and analyzer involved, and a minimal script or the demo from `examples/` that reproduces the problem.
- The full traceback or log output.

## Contributing a change

Contributions to this project are [released][released] to the public under the project's [open source license](LICENSE.txt).

Contributors must _sign off_ that they adhere to these requirements by adding a `Signed-off-by` line to all commit messages, with an email address that matches the commit author. `git commit -s` adds it for you:

```
feat: this is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```

### Development setup

```bash
conda create -n vllm-hook python=3.12 pip
conda activate vllm-hook
pip install -r requirement.txt    # vLLM 0.32.0, torch 2.13.0 and the other validated versions
pip install -e . --no-deps        # the plugin itself, from the repo root
pip install pytest
```

vLLM 0.32.0 ships wheels for Linux only, so running the engine needs Linux with an NVIDIA GPU. The README has the full setup notes.  MIA is registered as a vLLM plugin that is then loaded by all engines in the environment.

### Where to put new additions

- **New worker or analyzer:** implement it under `mia/workers/` or `mia/analyzers/` and register it in `register_plugins()` in [`mia/__init__.py`](mia/__init__.py). Workers are implemented as `worker_extension_cls`.
- **Examples:** add a demo under `examples/` and, where useful, a notebook under `notebooks/`.
- **Tests:** add a test under `tests/use_cases/` alongside your demo. See [`tests/README.md`](tests/README.md).
- **Docs:** add a row to `docs/use_cases/README.md` with your GitHub handle, and a writeup under `docs/use_cases/<name>.md` if you have one.
- **Example configs and steering vectors:** `model_configs/` and `steering_vectors/` at the repo root. They are shipped in the wheel as `mia/data/`.

The core infrastructure files listed in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) should be discussed with maintainers before you change them.

### Tests

Most tests require a GPU:

```bash
pytest tests -q                    # includes the checks that need no GPU
pytest tests/use_cases -m gpu      # boots a real engine for each test model
```

Please run the tests relevant to your change and say in the pull request which ones you ran and on what hardware.


## AI-assisted and agentic development

AI-assisted and agentic development are allowed. PR owners remain responsible for understanding, testing, and explaining every change they submit. Please do not submit bulk generated code that you cannot explain or validate.

## Package name and versions

The package is published as **`vllm-hook`**. The import name is **`mia`** (`from mia import MiaLLM`).

The name is set in [`pyproject.toml`](pyproject.toml), under `[project]`:

```toml
[project]
name = "vllm-hook"
dynamic = ["version"]
```

If you rename the package, also update every `pip install` line, `importlib.metadata.version("vllm-hook")` call and README reference. For example, [`examples/check_testpypi_install.py`](examples/check_testpypi_install.py) looks the package up by name.


The version is derived from the latest git tag at build time by [hatch-vcs](https://github.com/ofek/hatch-vcs) (which uses setuptools-scm):

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]

[tool.hatch.version]
source = "vcs"
```

To change the version, create a git tag. 

| Tag state of the commit you build | Version produced |
| --- | --- |
| Exactly on tag `v1.0.2`, clean tree | `1.0.2` |
| Exactly on tag `v1.0.2rc1`, clean tree | `1.0.2rc1` (a pre-release) |
| Exactly on tag `v1.0.2a1`, clean tree | `1.0.2a1` (an alpha of 1.0.2) |
| Commits after the latest tag `v1.0.1` | `1.0.2.dev3+g<hash>` (3 = commits since the tag; not uploadable) |
| On a tag, but with uncommitted changes | `1.0.3.dev0+g<hash>.d<date>` (not uploadable) |

Rules for tags:

- Put one version tag on a commit. With two tags on the same commit git picks one of them, and the version may not be the one you meant.
- Use `v` plus a [PEP 440](https://peps.python.org/pep-0440/) version: `v1.2.3`, `v1.2.3rc1`, `v1.2.3a1`, `v1.2.3.post1`.
- A leading word is stripped (`mia-v1.2.3` also works), but the tag must contain a numeric version. A tag such as `mia-test2` with no number **breaks every build** when it is the latest tag on the commit.
- Follow [semantic versioning](https://semver.org/): patch for fixes, minor for backward-compatible features, major for breaking changes.
- Pre-release suffixes sort below the release (`1.0.2rc1` < `1.0.2`).
- PyPI and TestPyPI refuse a version that was ever uploaded, even if the release was deleted. A failed release needs a new tag, not a re-upload.

### Releasing

1. Merge your work to `main` and make sure the working tree is clean.
2. Tag the commit and push the tag:
   ```bash
   git tag v1.0.2
   git push origin v1.0.2
   ```
3. Build from a clean checkout of the tag, with the full git history (a shallow clone gives a wrong version):
   ```bash
   rm -rf dist build
   python -m build
   python -m twine check dist/*
   ```
   Check that the file names in `dist/` show the plain version, such as `vllm_hook-1.0.2-py3-none-any.whl`. A `.dev` or `+g` suffix means the tree was dirty or the tag was not on the commit.
4. Try it on TestPyPI first:
   ```bash
   python -m twine upload -r testpypi dist/*
   ```
   Then run [`examples/check_testpypi_install.py`](examples/check_testpypi_install.py) on a Linux GPU machine to check the install. Install the dependencies from real PyPI and the package alone from TestPyPI: `pip install --no-deps --index-url https://test.pypi.org/simple/ vllm-hook==1.0.2`. Using `--extra-index-url` can pull in unrelated broken packages hosted on TestPyPI.
5. Upload to PyPI: `python -m twine upload dist/*`.

### Adding files or subpackages

The wheel contains everything under `mia/` automatically. The two repo-root data directories are mapped into the wheel by `force-include` in `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"model_configs" = "mia/data/model_configs"
"steering_vectors" = "mia/data/steering_vectors"
```

After a packaging change, build the wheel and list its contents (`unzip -l dist/*.whl`) to confirm that what you expect is in it.

### Troubleshooting builds

- **`setuptools-scm was unable to detect version`:** the build cannot read git. Check that `.git` is present, the clone is not shallow (`git fetch --tags --unshallow`), and `git describe --tags` works. On an Intel (x86_64) Python running under Rosetta on an Apple Silicon Mac, the `/usr/bin/git` shim fails because `xcrun` is arm64 only. Put a real git first on `PATH` (for example `PATH=/opt/homebrew/bin:$PATH python -m build`), or recreate the environment as arm64.
- **`Can't parse version from tag`:** the latest tag on the commit has no numeric version. Delete it (`git tag -d <tag>` and `git push origin --delete <tag>`) and tag again.
- **`'2.5' is not a valid metadata version` from `twine check`:** the `twine` on your `PATH` is too old. Run `python -m twine` from an environment with a current `twine` and `packaging`.
- **Pinning the version by hand:** `SETUPTOOLS_SCM_PRETEND_VERSION=1.2.3 python -m build` skips git detection. To be used for diagnosis only.

## Submitting a pull request

1. [Fork][fork] and clone the repository.
2. Create a new branch: `git checkout -b my-branch-name`.
3. Make your change with signed-off commits, push to your fork and [submit a pull request][pr]. Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
4. Wait for your pull request to be reviewed and merged.
