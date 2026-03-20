# 🪝 vLLM.hook
*A modular plugin library for vLLM.*
> 📄 [Preprint] [**vLLM Hook** v0: A Plug-in for Programming Model Internals on vLLM](./vLLM_Hook_v0.pdf)

vLLM.hook is a plugin library designed to let developers and researchers **inspect**, **analyze**, and **steer** the internal operations of large language models running under the **vLLM** inference engine.  

This includes dynamic analysis of:  
- attention patterns  
- attention heads  
- activations  
- custom intervention behaviors  

---

## 🚀 Features

- **Model-agnostic plugin system** for vLLM engines  
- **Extensible worker/analyzer abstraction**  
  - Easy to define new hooks, analyzers, and behaviors  
- **Introspection** of model internals  
- **Interventions** (activation steering, attention control, etc.)  
- **Example applications**:  
  - Safety guardrails  
  - Reranking  
  - Enhanced instruction following  

---

## 📦 Installation
### 1. Clone the repository

```bash
git clone https://github.com/IBM/vLLM-Hook.git
cd vLLm.hook
```

### 2. (Optional) Create an environment 

```bash
conda create -n vllm_hook_env
conda activate vllm_hook_env
```

### 3. Install the plugin and dependencies

```bash
pip install -r requirement.txt
pip install -e vllm_hook_plugins
```

---

## 📕 Notebook Setup 

If you plan to use the notebooks under `notebooks/`, you may need to register your environment as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name vllm_hook_env --display-name "vllm_hook_env"
```

Then inside Jupyter Lab:

```
Kernel → Change Kernel → vllm_hook_env
```

---

## 👉 Usage Examples (Notebook / CLI)

You can also use the included **`examples/`** and/or **`notebooks/`** directories to explore different functionalities.

### 1. Attention Tracker (In-Model Safety Guardrail)

Notebook 📓: `notebooks/demo_attntracker.ipynb` <br />
CLI 🧰 : 
```bash
python examples/demo_attntracker.py
```

### 2. Core Reranker (In-Model Relevance Ranking)

Notebook 📓: `notebooks/demo_corer.ipynb` <br />
CLI 🧰 : 
```bash
python examples/demo_actsteer.py
```

### 3. Activation Steering (Enhanced instruction following via activation steering)

Notebook 📓: `notebooks/demo_actsteer.ipynb` <br />
CLI 🧰 : 
```bash
python examples/demo_corer.py
```

You can customize model configurations in the `model_configs/` folder, e.g.:

```
model_configs/<example_name>/<model_name>.json
```
For example `model_configs/attention_tracker/granite-4.0-micro.json`.

---

## 🏠 Plugin Architecture

The main package is structured as follows:

```
vllm_hook_plugins/
├── analyzers/
│   ├── attention_tracker_analyzer.py
│   ├── core_reranker_analyzer.py
├── workers/
│   ├── probe_hookqk_worker.py
│   ├── steer_activation_worker.py
├── hook_llm.py
├── registry.py
```

Each component handles a key stage of the plugin lifecycle:

- **Registry** — manages available hooks and extensions  
- **Workers** — define execution behavior and orchestration  
- **Analyzers** — optionally conduct analysis based on the saved statistics  


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
- The registry will be updated by the admin

---


## IBM ❤️ Open Source AI

vLLM.hook has been started by IBM Research.
- Built for the **vLLM** ecosystem  
- Inspired by community efforts to make LLMs more interpretable and controllable  
