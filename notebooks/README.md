# Notebook Setup on Apple Silicon

These notebooks are intended to be run from a local checkout of `vLLM-Hook`.
On Apple Silicon, the local notebook path uses `vllm-metal`.

## Prerequisites

- macOS on Apple Silicon
- Jupyter Lab or Notebook
- `vllm-metal` installed and importable

## Installation

Using the install script, the following will be installed under the
`~/.venv-vllm-metal` directory by default:

- `vllm-metal` plugin
- `vllm` core
- related libraries

Project link:

- <https://github.com/vllm-project/vllm-metal>

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Then activate the environment:

```bash
source ~/.venv-vllm-metal/bin/activate
```

After activation, the `vllm` CLI is available immediately.

For CLI usage, refer to the official vLLM guide:

- <https://docs.vllm.ai/en/latest/cli/>

From there, install the notebook-specific pieces from the `vLLM-Hook` repo root:

```bash
cd <path-to-directory>/vLLM-Hook
pip install -r requirement.txt
pip install -e vllm_hook_plugins
pip install jupyter ipykernel
```

## Reinstallation and Update

Re-run the install script if you need to recreate or refresh the default
`vllm-metal` environment:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Register a Kernel

```bash
python -m ipykernel install --user --name vllm_hook_env --display-name "vllm_hook_env"
```

Then select `vllm_hook_env` inside Jupyter.

## Launch Jupyter

From the repo root:

```bash
cd <path-to-directory>/vLLM-Hook
jupyter lab
```

Open one of:

- [demo_attntracker.ipynb](<path-to-directory>/vLLM-Hook/notebooks/demo_attntracker.ipynb)
- [demo_corer.ipynb](<path-to-directory>/vLLM-Hook/notebooks/demo_corer.ipynb)
- [demo_actsteer.ipynb](<path-to-directory>/vLLM-Hook/notebooks/demo_actsteer.ipynb)

## How the Local Notebooks Work

The local notebooks assume:

- the working directory is `<path-to-directory>/vLLM-Hook/notebooks`
- the repo root is the parent directory
- `vllm_hook_plugins` is installed editable from the local checkout

Their install cells use the local package path:

```python
PKG_DIR = REPO_ROOT / "vllm_hook_plugins"
REQ_FILE = REPO_ROOT / "requirement.txt"
```

## Notes

- The notebooks currently default to `ibm-granite/granite-4.0-micro`.
- If you change the model, keep the notebook config file aligned with that model.
- When you edit helper cells in a notebook, re-run that cell or restart the kernel before re-running later cells.
- Metal-specific worker behavior lives under:
  - [`<path-to-directory>/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal`](<path-to-directory>/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/metal)

## Troubleshooting

- `ModuleNotFoundError: vllm_hook_plugins`
  - Re-run `pip install -e vllm_hook_plugins` from the repo root in the active environment.

- Kernel is using stale notebook code
  - Restart the kernel and run the notebook from the top.

- Metal path is not being used
  - Confirm your environment can import `vllm_metal` and that you are running the local notebook, not the Colab copy.

- Notebook config/model mismatch
  - Check the selected `model` and the `json_path` or config mapping cell in the notebook.
