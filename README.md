# 🪝 vLLM-Hook MIA
*A modular plugin library for vLLM.*

📄 [Preprint] [**vLLM Hook** v0: A Plug-in for Programming Model Internals on vLLM](https://arxiv.org/abs/2603.06588v1)

📄 [Preprint] [**MIA**: Runtime-Reconfigurable Access to Model Internals in LLM Inference Engines]

MIA is a plugin library designed to let developers and researchers **inspect**, **analyze**, and **steer** the internal operations of large language models running under the **vLLM** inference engine.  

This includes dynamic analysis of:  
- attention patterns  
- attention heads  
- activations  
- custom intervention behaviors  

MIA is vLLM-Hook for vLLM 0.29 and its V2 model runner, under CUDA graphs by default. Two ways to run it:
- **`MiaLLM`** (offline): builds the vLLM engine in your own Python process; for scripts, notebooks and batch jobs.
- **`vllm serve` + `MiaClient`** (served): MIA runs inside the vLLM server; for serving, or several clients sharing one engine.

New here? Start with the [Quickstart](#-quickstart).

---

## 📰 News & Events

- **July 24, 2026** — Featured in IBM Think: [*A new way of debugging open-weight models*](https://www.ibm.com/think/news/new-way-debugging-open-weight-models).

- **July 6, 2026** — Presented at ICML 2026: [*vLLM-Hook: Live Programming of Model Internals on vLLM*](https://icml.cc/virtual/2026/75729). (ICML registration and login are required to view the presentation.)

---

## 🚀 Features

- **Model-agnostic plugin system** for vLLM engines — no model files are rewritten
  ([model compatibility](docs/configs.md#model-compatibility))  
- **Extensible worker/analyzer abstraction**  
  - Easy to add analyzers ([adding a worker or analyzer](#adding-a-worker-or-analyzer))  
- **Introspection** of model internals  
- **Interventions** (activation steering)  
- **CUDA graphs by default** — capture and steering keep the engine's CUDA graphs
  ([limits](docs/configs.md#limits))  
- **Example applications**:  
  - Safety guardrails  
  - Reranking  
  - Enhanced instruction following  

---

## 📊 Performance

Capture and steering cost a few percent of decode throughput and keep the engine's CUDA graphs.
Measured on vLLM 0.29.0 + the V2 runner, server path, K=3 median, against the native vLLM tier of
the same launch (TPOT overhead):

| req/s | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| Hidden states | +4.9 % | +6.1 % | +6.4 % | +5.8 % |
| Q/K | +6.7 % | +8.3 % | +8.4 % | +7.9 % |
| Steering | +3.9 % | +4.1 % | +3.7 % | +5.6 % |

Capture saturates at rate 32; steering holds to 64 (+5.2 %).

---

## 🧩 Supported Configurations

MIA runs in your own process (`MiaLLM`) and on the server path (`vllm serve` + `MiaClient`),
under CUDA graphs. Each capture use case (attention tracker, hidden-state extraction, …) runs with
either storage (`rpc` / `disk`) and disk format (`pt` / `safetensors`); CoRe is disk-only, and
steering writes no artifact. See [`docs/configs.md`](docs/configs.md) for code snippets showing
how to select each config.

MIA requires vLLM's **V2 model runner** and runs every worker under CUDA graphs by default:

- default `cudagraph_mode` is `FULL_AND_PIECEWISE`: FULL graphs for decode, piecewise graphs for
  mixed steps;
- `FULL_DECODE_ONLY` is accepted, and is used when the compilation mode is not the default or
  `TORCH_COMPILE_DISABLE=1` is set;
- `FULL` is accepted with a warning: it can compute wrong attention on FlashAttention 3 (vLLM 0.29);
- `enforce_eager=True` (`vllm serve --enforce-eager`) opts out; `PIECEWISE` alone is refused.

Tensor parallelism (TP > 1) is supported for `capture_hs`, `capture_qk` and `steer`: each capturing
rank writes its own `tp_rank_<r>/` directory and MIA's loaders merge them. Pipeline parallelism is
not supported, and Q/K `score` capture requires TP = 1.

---

## 📦 Installation

**Requirements**
- Linux and an NVIDIA GPU with a CUDA 13 driver; validated on an H100 80 GB.
- Python 3.11–3.14; validated on 3.12.
- About 12 GB for the environment, plus the models, downloaded on first use (the hidden-state,
  steering, attention-tracker and capture-aperture demos need about 33 GB).
- GPU memory: capture keeps a 4 GiB buffer outside vLLM's `gpu_memory_utilization` share. The demos
  use 0.7 and models up to 8B (a 40 GB card or larger); on a smaller card lower
  `gpu_memory_utilization` (`--gpu-memory-utilization` on a server) or set `MIA_APERTURE_GPU_BYTES`
  ([sizing](docs/configs.md#sizing-the-capture-aperture-and-the-gpu-memory-it-costs)).
- Gated models (Llama, Mistral) need `hf auth login`.

### 1. Clone the repository

```bash
git clone https://github.com/IBM/vLLM-Hook.git
cd ./vLLM-Hook
```

### 2. Create an environment and install

MIA is validated on **vLLM 0.29.0 with torch 2.13.0**:

```bash
conda create -n vllm-hook-mia python=3.12 pip
conda activate vllm-hook-mia
pip install -r requirement.txt    # vLLM 0.29.0, torch 2.13.0 and the other validated versions
pip uninstall -y torchcodec       # vLLM's audio/video decoder; MIA does not use it
pip install -e . --no-deps        # the plugin itself, from the repo root
pip install pytest                # for the check below
```

Versions match [`requirement.txt`](requirement.txt). `pip check` flags the removed `torchcodec`; that is expected.

MIA registers itself as a vLLM plugin, so every vLLM engine in this environment loads it: without
`MIA_WORKER` it installs the hidden-state capture worker (and its 4 GiB GPU buffer) in every
engine. Keep a dedicated environment; set `VLLM_PLUGINS=''` to run stock vLLM in it.

### 3. Check the install

Needs a GPU — the use-case tests boot a real engine for each test model
([`tests/README.md`](tests/README.md)):

```bash
pytest tests/use_cases -m gpu
```

---

## ⚡ Quickstart

Run these from the repo root (the configs name repo-relative steering vectors).

**Capture hidden states, offline:**

```python
import multiprocessing as mp
import os

from vllm import SamplingParams

from mia import MiaLLM

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = MiaLLM(model="Qwen/Qwen2-1.5B-Instruct", worker_name="capture_hs",
                 analyzer_name="hidden_states",
                 config_file="model_configs/hidden_states/Qwen2-1.5B-Instruct.json",
                 gpu_memory_utilization=0.7, max_model_len=2048)
    out = llm.generate("The capital of France is", SamplingParams(temperature=0.0, max_tokens=10))
    stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out[0].probes)
    for layer, tensors in sorted(stats["hidden_states"].items()):
        print(layer, tuple(tensors[0].shape))
    llm.llm_engine.engine_core.shutdown()
```

**Steer, offline:**

```python
import multiprocessing as mp
import os

from vllm import SamplingParams

from mia import MiaLLM

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = MiaLLM(model="microsoft/Phi-3-mini-4k-instruct", worker_name="steer",
                 config_file="model_configs/activation_steer/Phi-3-mini-4k-instruct.json",
                 gpu_memory_utilization=0.7, max_model_len=4096)
    steer = {"method": "add_vector", "coefficient": 10}   # overrides the config's steering keys
    sp = SamplingParams(temperature=0.0, max_tokens=100, extra_args={"steer": steer})
    print(llm.generate("Write three bullet points about tea.", sp)[0].outputs[0].text)
    llm.llm_engine.engine_core.shutdown()
```

**Served:** start the server in one shell and wait for `Application startup complete.` (about a
minute); run the client in a second shell; stop the server with Ctrl-C.

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
    vllm serve Qwen/Qwen2-1.5B-Instruct \
    --max-model-len 2048 --port 8770 --gpu-memory-utilization 0.8
```

```python
from mia import MiaClient

client = MiaClient(base_url="http://localhost:8770/v1", analyzer_name="hidden_states",
                   config_file="model_configs/hidden_states/Qwen2-1.5B-Instruct.json")
client.generate(messages=[{"role": "user", "content": "The capital of France is"}],
                model="Qwen/Qwen2-1.5B-Instruct", max_tokens=10, temperature=0.0)
stats = client.analyze(analyzer_spec={"reduce": "none"})
for layer, tensors in sorted(stats["hidden_states"].items()):
    print(layer, tuple(tensors[0].shape))
```

Steering over `vllm serve` needs only the `openai` client; the steer config travels JSON-encoded:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=steer \
    vllm serve microsoft/Phi-3-mini-4k-instruct --max-model-len 4096 --port 8770
```

```python
import json

import openai

with open("model_configs/activation_steer/Phi-3-mini-4k-instruct.json") as f:
    steer = {**json.load(f)["steering"], "method": "add_vector", "coefficient": 10}
client = openai.OpenAI(base_url="http://localhost:8770/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="microsoft/Phi-3-mini-4k-instruct", max_tokens=100, temperature=0.0,
    messages=[{"role": "user", "content": "Write three bullet points about tea."}],
    extra_body={"vllm_xargs": {"steer": json.dumps(steer)}})
print(response.choices[0].message.content)
```

Next: [`examples/README.md`](examples/README.md) (where data lands, configs, server mode) and
[`docs/configs.md`](docs/configs.md) (every config key, per-request argument and env var).

---

## 👉 Usage

Every demo runs offline: it builds a `MiaLLM` engine in your process, captures or steers, and
reads the data back. From the repo root:

```bash
python examples/demo_hiddenstate.py
```

Each other demo (bar `demo_capture_aperture.py`) also keeps its `vllm serve` version as a
commented block, with the server command to start; `examples/demo_actsteer_serve.py` is the server example
([server mode](examples/README.md#8-server-mode)). [`examples/README.md`](examples/README.md) is
the walkthrough — getting started,
[where captured data lands](examples/README.md#where-the-captured-data-goes), how to confirm a
capture actually happened, and tensor parallelism.
For the full list of use cases see [`docs/use_cases/`](docs/use_cases/README.md).

### Use cases

| Use case | Demo |
|---|---|
| Attention Tracker (in-model safety guardrail) | `python examples/demo_attntracker.py` |
| Core Reranker (in-model relevance ranking) | `python examples/demo_corer.py` |
| Activation Steering (enhanced instruction following) | `python examples/demo_actsteer.py` |
| Hidden-State Probe | `python examples/demo_hiddenstate.py` |
| Science Hallucination Detector | `python examples/demo_scihal.py` |
| H-Node Hallucination Detector | `python examples/demo_halludetect.py` |
| AttnLink-U (schema linking) | `python examples/demo_attnlink.py` |

More demos (language steering, the server-only steering example, the capture-aperture checks,
long decodes): [examples/README.md](examples/README.md#7-running-the-included-demos).

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
├── runner.py
├── optimizations.py
├── registry.py
```

Each component handles a key stage of the plugin lifecycle:

- **Registry** — manages available hooks and extensions  
- **Workers** — define execution behavior and orchestration  
- **Analyzers** — optionally conduct analysis based on the saved statistics  
- **Graph** — installs the capture/steering ops and the GPU capture aperture under CUDA graphs  
- **Runner** — the one place that touches vLLM's V2 model-runner internals  
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
- New analyzers and workers are welcome; discuss before modifying `mia/llm.py`, `mia/_plugin.py`, `mia/client.py` or `mia/graph/`
- Include examples and documentation for new features  
- New use cases must be added to [`docs/use_cases/README.md`](docs/use_cases/README.md) with the contributor's GitHub handle

### Adding a worker or analyzer

- **Analyzer:** a class with `__init__(hook_dir, layer_to_heads)` and
  `analyze(analyzer_spec, run_id=None, probes=None)`; register it with
  `PluginRegistry.register_analyzer("name", Cls)` and pass `analyzer_name="name"`.
  `mia/analyzers/hidden_states_analyzer.py` is the smallest example.
- **Worker:** MIA's three workers each bake their own op into the CUDA graph. A new worker class
  installs forward hooks, which only run in eager mode: build its engine with
  `enforce_eager=True` (`--enforce-eager`).

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

vLLM-Hook MIA was started by IBM Research.
- Built for the **vLLM** ecosystem  
- Inspired by community efforts to make LLMs more interpretable and controllable
