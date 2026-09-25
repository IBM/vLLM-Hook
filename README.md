# 🪝 MIA
*A modular plugin library for vLLM.*

📄 [Preprint] [**vLLM Hook** v0: A Plug-in for Programming Model Internals on vLLM](https://arxiv.org/abs/2603.06588v1)

MIA is a plugin library designed to let developers and researchers **inspect**, **analyze**, and **steer** the internal operations of large language models running under the **vLLM** inference engine.  

This includes dynamic analysis of:  
- attention patterns  
- attention heads  
- activations  
- custom intervention behaviors  

---

## 📰 News & Events

- **July 24, 2026** — Featured in IBM Think: [*A new way of debugging open-weight models*](https://www.ibm.com/think/news/new-way-debugging-open-weight-models).

- **July 6, 2026** — Presented at ICML 2026: [*MIA: Live Programming of Model Internals on vLLM*](https://icml.cc/virtual/2026/75729). (ICML registration and login are required to view the presentation.)

---

## 🚀 Features

- **Model-agnostic plugin system** for vLLM engines  
- **Extensible worker/analyzer abstraction**  
  - Easy to define new hooks, analyzers, and behaviors  
- **Introspection** of model internals  
- **Interventions** (activation steering, attention control, etc.)  
- **FULL CUDA-graph support** — capture and steering stay graph-safe, no fallback to eager
  ([one measured caveat on `logprobs`](#-known-limitations))  
- **Example applications**:  
  - Safety guardrails  
  - Reranking  
  - Enhanced instruction following  

---

## 📊 Performance Analysis

For a detailed benchmark comparing **MIA** against **Native vLLM Eagle** (`ExampleHiddenStatesConnector`) for hidden state extraction, see [`docs/numerical_analysis/`](docs/numerical_analysis/README.md).

Key takeaways:
- MIA (`last_token`) offers significantly lower and prompt-length-invariant latency when only the final-position representation is needed
- MIA (`all_tokens`) is numerically equivalent to Native Eagle while avoiding its GPU memory overhead
- Native Eagle requires loading a speculative decoding drafter model, reducing available KV cache

---

## ⚠️ Known Limitations

- **Runtime envelope.** MIA requires vLLM 0.29.0 with its V2 model runner and `cudagraph_mode`
  `NONE` (`enforce_eager=True`) or `FULL`; `PIECEWISE` and `FULL_AND_PIECEWISE` are rejected at
  engine start. Spotlight and Token Highlighter are not supported on the V2 runner.
- **Logprobs under FULL CUDA graphs.** Arming capture usually keeps generated token ids identical but can
  move per-token logprobs (up to ~1.5e-2 for hidden-state capture, ~5e-7 for Q/K capture),
  because the capture op changes what the compiler fuses. A near-tie greedy step can therefore
  occasionally flip and change the rest of the text. Eager mode is bit-exact; use
  `enforce_eager=True` when you need bit-reproducible logprobs.
- **Fused steering kernel.** The default `MIA_STEER_FUSED=1` is not bit-identical to the
  reference steering path; set `MIA_STEER_FUSED=0` for bit-exact steering.
- **GPU routing.** `MIA_CAPTURE_GPU_ROUTING=1` (off by default) is not bit-reproducible against
  host routing under FULL CUDA graphs.

---

## 🧩 Supported Configurations

Each use case (e.g. attention tracker, activation steering, hidden states extraction, etc) runs across a Cartesian product of configuration axes — execution path (`offline` / `vllm serve`), storage (`rpc` / `disk` / `shm`), and disk format (`pt` / `safetensors`). See [`docs/configs.md`](docs/configs.md) for code snippets showing how to select each config.

Tensor parallelism (TP > 1) is supported for `capture_hs`, `capture_qk` and `steer`: each capturing
rank writes its own `tp_rank_<r>/` directory and MIA's loaders merge them. Pipeline parallelism is
not supported, and Q/K `score` capture requires TP = 1.

---

## 📦 Installation
### 1. Clone the repository

```bash
git clone https://github.com/IBM/vLLM-Hook.git
cd ./vLLM-Hook
```

### 2. Create an environment and install

The plugin is currently validated on **vLLM 0.29.0 with torch 2.13.0**. To use this pinned environment:

```bash
conda create -n mia_v029 python=3.12 pip
conda activate mia_v029
pip install vllm==0.29.0
pip uninstall -y torchcodec
pip install -e . --no-deps
pip install zstandard
```

For the complete list of pinned dependencies, see [`requirement.txt`](requirement.txt).

---

## 📕 Notebook Setup 

If you plan to use the notebooks under `notebooks/`, you may need to register your environment as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name mia_v029 --display-name "mia_v029"
```

Then inside Jupyter Lab:

```
Kernel → Change Kernel → mia_v029
```

---

## 👉 Usage Examples (Notebook / CLI)

You can also use the included **`examples/`** and/or **`notebooks/`** directories to explore different functionalities. For the full list of use cases, see [`docs/use_cases/`](docs/use_cases/README.md).

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
python examples/demo_corer.py
```

### 3. Activation Steering (Enhanced instruction following via activation steering)

Notebook 📓: `notebooks/demo_actsteer.ipynb` <br />
CLI 🧰 : 
```bash
python examples/demo_actsteer.py
```

You can customize model configurations in the `model_configs/` folder, e.g.:

```
model_configs/<example_name>/<model_name>.json
```
For example `model_configs/attention_tracker/granite-3.1-8b-instruct.json`.

---

## 🏠 Plugin Architecture

The main package is structured as follows:

```
mia/
├── analyzers/
│   ├── attention_tracker_analyzer.py
│   ├── core_reranker_analyzer.py
├── workers/
│   ├── qk_capture_worker.py
│   ├── steer_worker.py
├── graph/
│   ├── install.py
│   ├── capture_aperture.py
├── llm.py
├── optimizations.py
├── registry.py
```

Each component handles a key stage of the plugin lifecycle:

- **Registry** — manages available hooks and extensions  
- **Workers** — define execution behavior and orchestration  
- **Analyzers** — optionally conduct analysis based on the saved statistics  
- **Graph** — installs the capture/steering ops and the GPU capture aperture under CUDA graphs  
- **Optimizations** — the public performance levers (`optimizations.py::PUBLIC_LEVERS`)  


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
- Users are encouraged to define new worker/analyzer, but should not touch llm
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

MIA has been started by IBM Research.
- Built for the **vLLM** ecosystem  
- Inspired by community efforts to make LLMs more interpretable and controllable  
