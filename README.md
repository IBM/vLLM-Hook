# 🪝 vLLM.hook Apple Silicon
*A modular plugin library for vLLM on Apple Silicon via vLLM-Metal.*

📄 [Preprint] [**vLLM Hook** v0: A Plug-in for Programming Model Internals on vLLM](https://arxiv.org/abs/2603.06588v1)

🔗 Metal backend: [**vllm-project/vllm-metal**](https://github.com/vllm-project/vllm-metal)

**vLLM-Hook Apple Silicon (Proof of Concept)**

This proof-of-concept adapts vLLM-Hook for **Apple Silicon / MLX** by running on top of the **vLLM-Metal** execution path.

Use the Metal entrypoint when you want to inspect, analyze, or steer model internals locally on a Mac:

```python
from vllm_hook_plugins.metal import HookLLMMetal
```

This includes dynamic analysis of:
- attention patterns
- attention heads
- hidden states
- activation steering behaviors

---

## 🚀 Features

- **Apple Silicon support** through `vllm-metal`
- **MLX-backed worker/analyzer implementations**
  - Metal-specific hooks live beside the standard vLLM-Hook path
- **Introspection** of model internals
- **Interventions** through activation steering
- **Example applications**:
  - Safety guardrails
  - Reranking
  - Hidden state extraction
  - Enhanced instruction following

---

## 🍎 Apple Silicon Notes

The Metal implementation keeps the same high-level vLLM-Hook workflow while swapping in Metal-specific components under:

- `vllm_hook_plugins/vllm_hook_plugins/metal`
- `vllm_hook_plugins/vllm_hook_plugins/workers/metal`
- `vllm_hook_plugins/vllm_hook_plugins/analyzers/metal`

The Metal path is designed for macOS on Apple Silicon and expects the `vllm-metal` virtual environment created by the upstream installer.

Apple Silicon uses unified memory, so smaller models, quantized MLX-community models, and shorter context lengths are recommended. The Metal notebooks pin `max_model_len=2048` to avoid reserving excessive KV-cache memory.

---

## 🧩 Supported Metal Configurations

The Metal path currently supports:

- **Attention tracker** via Q/K capture
- **Core reranker** via Q/K capture and Metal analyzer wrappers
- **Hidden states extraction**
- **Activation steering**

`HookLLMMetal` maps standard worker names to Metal-specific implementations:

| Worker name | Metal implementation |
| --- | --- |
| `probe_hook_qk` | `ProbeHookQKWorkerMetal` |
| `probe_hidden_states` | `ProbeHiddenStatesWorkerMetal` |
| `steer_hook_act` | `SteerHookActWorkerMetal` |

---

## 📦 Installation

### 1. Install vllm-metal

The upstream installer creates `~/.venv-vllm-metal` by default:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Activate the environment:

```bash
source ~/.venv-vllm-metal/bin/activate
```

### 2. Clone the repository

```bash
git clone https://github.com/IBM/vLLM-Hook.git
cd vLLM-Hook
```

### 3. Install the plugin and dependencies

Install vLLM-Hook into the active `vllm-metal` environment:

```bash
python -m pip install -r requirement.txt
python -m pip install -e vllm_hook_plugins
```

---

## 📕 Notebook Setup

Register the active vLLM-Metal environment as a Jupyter kernel:

```bash
./scripts/metal/register_vllm_metal_kernel.sh
```

Then launch Jupyter:

```bash
jupyter lab
```

Inside Jupyter Lab:

```
Kernel → Change Kernel → Python (vllm-metal)
```

For more details, see [`notebooks/metal/README.md`](notebooks/metal/README.md).

---

## 👉 Usage Examples (Metal Notebooks)

Use the included **`notebooks/metal/`** demos to explore the Apple Silicon path.

### 1. Attention Tracker (In-Model Safety Guardrail)

Notebook 📓: `notebooks/metal/demo_attntracker_metal.ipynb`

### 2. Core Reranker (In-Model Relevance Ranking)

Notebook 📓: `notebooks/metal/demo_corer_metal.ipynb`

### 3. Activation Steering (Enhanced instruction following via activation steering)

Notebook 📓: `notebooks/metal/demo_actsteer_metal.ipynb`

You can customize model configurations in the `model_configs/` folder, e.g.:

```
model_configs/<example_name>/<model_name>.json
```

---

## 🏠 Plugin Architecture

The Metal package is structured as follows:

```
vllm_hook_plugins/
├── analyzers/
│   ├── metal/
│   │   ├── attention_tracker_analyzer_metal.py
│   │   ├── core_reranker_analyzer_metal.py
│   │   ├── hidden_states_analyzer_metal.py
├── workers/
│   ├── metal/
│   │   ├── probe_hookqk_worker_metal.py
│   │   ├── probe_hidden_states_worker_metal.py
│   │   ├── steer_activation_worker_metal.py
├── metal/
│   ├── hook_llm_metal.py
│   ├── run_utils_metal.py
```

Each component handles a key stage of the plugin lifecycle:

- **Metal entrypoint** — exposes `HookLLMMetal`
- **Workers** — define MLX/vLLM-Metal hook behavior
- **Analyzers** — route Metal artifacts through analyzer-compatible loaders

---

## 🤝 Contributing

We welcome contributions from the community!

### To contribute:
1. **Fork** this repository
2. **Create a branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to your branch (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Guidelines:
- Users are encouraged to define new worker/analyzer, but should not touch hook_llm
- Include examples and documentation for new features
- New use cases must be added to [`docs/use_cases/README.md`](docs/use_cases/README.md) with the contributor's GitHub handle

---

## 🌟 Feeling Inspired
```
@article{ko2026vllm,
  title={vLLM Hook v0: A Plug-in for Programming Model Internals on vLLM},
  author={Ko, Ching-Yun and Chen, Pin-Yu},
  journal={arXiv preprint arXiv:2603.06588},
  year={2026}
}
```
---


## IBM ❤️ Open Source AI

vLLM.hook has been started by IBM Research.
- Built for the **vLLM** ecosystem
- Extended here for **Apple Silicon / vLLM-Metal**
