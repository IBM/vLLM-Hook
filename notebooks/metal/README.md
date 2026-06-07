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
```

Register a Jupyter kernel backed by the same environment:

```bash
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name vllm-metal-local --display-name "Python (vllm-metal)"
```

Or run the helper script from this repo, which activates `~/.venv-vllm-metal`
and registers the kernel in one step:

```bash
./scripts/metal/register_vllm_metal_kernel.sh
```

## Launch Jupyter

From the repository root:

```bash
jupyter lab
```

(Or open in VSCode)

Open one of the Metal notebooks:

- `notebooks/metal/demo_attntracker_metal.ipynb`
- `notebooks/metal/demo_corer_metal.ipynb`
- `notebooks/metal/demo_actsteer_metal.ipynb`

The act-steer notebook defaults to
`mlx-community/Phi-3.5-mini-instruct-4bit` to reduce memory pressure on Apple
Silicon. The notebook sets `hf_config_path` and tokenizer to the same repo so
vLLM-Metal resolves the bundled config and tokenizer files from the quantized
MLX-community model.

The attention-tracker and core-reranker Metal notebooks also pin
`max_model_len=2048`. Their base models advertise very large context windows,
and leaving the default max length enabled can make vLLM reserve far more
KV-cache memory than Apple Silicon can spare.

The Metal wrapper defaults to `VLLM_METAL_USE_PAGED_ATTENTION=0` and
`VLLM_METAL_MEMORY_FRACTION=auto` unless those variables are already set. For
Q/K capture runs it also releases the base engine before building the hooked
engine, which keeps peak unified-memory usage lower on Apple Silicon.

The Metal analyzer path uses backend-specific wrappers under
`vllm_hook_plugins/vllm_hook_plugins/analyzers/metal` for attention tracking,
core reranking, and hidden states. The hidden-state wrapper preserves the shared
analyzer behavior while routing disk artifacts through the Metal loader
boundary.

Select the `Python (vllm-metal)` kernel before running cells.

## Using VS Code

VS Code can use the same environment without any repo-specific editor config:

1. Open the repository folder in VS Code.
2. Run `Python: Select Interpreter`.
3. Choose `~/.venv-vllm-metal/bin/python`.
4. Open a notebook and use `Jupyter: Select Notebook Kernel`.
5. Pick `Python (vllm-metal)`.

If the kernel does not appear, run
`./scripts/metal/register_vllm_metal_kernel.sh` once in a terminal after
activating the env. That only registers a Jupyter kernel for your user account;
it does not write VS Code settings.

## What Is Different From the Standard Notebooks

- imports `HookLLMMetal` from `vllm_hook_plugins.metal`
- uses Metal-specific workers under `workers/metal`
- uses Metal analyzer wrappers under `analyzers/metal`, including
  `hidden_states_analyzer_metal.py`
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
