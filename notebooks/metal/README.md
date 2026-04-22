# Metal Notebook Setup on Apple Silicon

These notebooks are Apple Silicon copies of the standard demos. They use the
isolated Metal entrypoint, `HookLLMMetal`, and the Metal worker modules under
`vllm_hook_plugins/vllm_hook_plugins/workers/metal`.

## Prerequisites

- macOS on Apple Silicon
- Jupyter Lab or Notebook
- `vllm-metal`, `vllm`, `mlx`, and their dependencies installed in the
  `vllm-metal` virtual environment

## Install vllm-metal

The upstream `vllm-metal` install script creates `~/.venv-vllm-metal` by
default:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Activate that environment before installing this repo:

```bash
source ~/.venv-vllm-metal/bin/activate
```

## Install vLLM-Hook Into the vllm-metal Environment

From the repository root:

```bash
cd <path-to-directory>/vLLM-Hook
python -m pip install -r requirement.txt
python -m pip install -e vllm_hook_plugins
python -m pip install jupyter ipykernel
```

Register a Jupyter kernel backed by the same environment:

```bash
python -m ipykernel install --user --name vllm-metal-local --display-name "Python (vllm-metal)"
```

## Launch Jupyter

From the repository root:

```bash
jupyter lab
```

Open one of the Metal notebooks:

- `notebooks/metal/demo_attntracker_metal.ipynb`
- `notebooks/metal/demo_corer_metal.ipynb`
- `notebooks/metal/demo_actsteer_metal.ipynb`

Select the `Python (vllm-metal)` kernel before running cells.

## What Is Different From the Standard Notebooks

- imports `HookLLMMetal` from `vllm_hook_plugins.metal`
- uses Metal-specific workers under `workers/metal`
- uses Metal analyzer wrappers under `analyzers/metal`
- keeps the standard `HookLLM` path untouched
- resolves model config paths from `notebooks/metal` with `../../model_configs/...`

## Troubleshooting

- `ModuleNotFoundError: vllm_metal` or `mlx`
  - Confirm `source ~/.venv-vllm-metal/bin/activate` was run before launching
    Jupyter or registering the kernel.

- `ModuleNotFoundError: vllm_hook_plugins`
  - Re-run `pip install -e vllm_hook_plugins` from the repo root in the active
    `vllm-metal` environment.

- The generic hook path is being used
  - Confirm the notebook imports `HookLLMMetal` and that you opened a notebook
    from `notebooks/metal`.

- Kernel state seems stale
  - Restart the kernel and run the notebook from the top.
