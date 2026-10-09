# Use Cases

Each row maps a use case to its paper, its demo under `examples/`, the plugin names it uses and the corresponding contributor. What each analyzer takes and returns: [`docs/configs.md`](../configs.md#use-cases).

**Note for contributors:** When opening a PR that adds a new use case, append a row here. If your use case ships with a writeup, place it alongside this file as `<use_case>.md` and link it from the first column.

| Use case | Paper | Demo | `analyzer_name` | `worker_name` | Contributor |
| --- | --- | --- | --- | --- | --- |
| Attention Tracker | [Attention Tracker: Detecting Prompt Injection Attacks in LLMs](https://arxiv.org/abs/2411.00348) | `demo_attntracker.py` | `attn_tracker` | `capture_qk` | [@IRENEKO](https://github.com/IRENEKO); vLLM-Hook v0 Colab notebook: [@tburleyinfo](https://github.com/tburleyinfo) |
| Core Reranker | [Contrastive Retrieval Heads Improve Attention-Based Re-Ranking](https://arxiv.org/abs/2510.02219) | `demo_corer.py` | `core_reranker` | `capture_qk` † | [@IRENEKO](https://github.com/IRENEKO); vLLM-Hook v0 Colab notebook: [@tburleyinfo](https://github.com/tburleyinfo) |
| Activation Steering | [Improving Instruction-Following in Language Models through Activation Steering](https://arxiv.org/abs/2410.12877) | `demo_actsteer.py`, `demo_actsteer_serve.py`, `demo_actsteer_language.py` (language steering: [@lingyue404](https://github.com/lingyue404)) | — | `steer` | [@IRENEKO](https://github.com/IRENEKO); vLLM-Hook v0 Colab notebook: [@tburleyinfo](https://github.com/tburleyinfo) |
| Hidden-State Probe | — | `demo_hiddenstate.py` | `hidden_states` | `capture_hs` | [@IRENEKO](https://github.com/IRENEKO) |
| Science Hallucination Detector | [Detecting Hallucinations in Scientific Claims by Combining Prompting Strategies and Internal State Classification](https://aclanthology.org/2025.sdp-1.30/) | `demo_scihal.py` | `science_hallucination` | `capture_hs` † | [@IRENEKO](https://github.com/IRENEKO) |
| [H-Node Detector](hnode_detector.md) | [H-Node Attack and Defense in Large Language Models](https://arxiv.org/abs/2603.26045) | `demo_halludetect.py` | `hnode_hallucination` | `capture_hs` † | [@Samarpit-bhatia](https://github.com/Samarpit-bhatia) |
| [AttnLink-U](attnlink.md) | [AttnLink: Turning Attention into Schema Links for Text-to-SQL](https://arxiv.org/abs/2608.00693) | `demo_attnlink.py` | `attnlink` | `capture_qk` † | [@Songjw133](https://github.com/Songjw133) |
| [Tool-Call Risk Probe](tool_call_risk.md) | — | `demo_toolcallrisk.py` | `tool_call_risk` | `capture_hs` † | [@rishabhsinha17](https://github.com/rishabhsinha17) |
| [Spotlight](spotlight.md) | [Venkateswaran and Contractor, EACL 2026](https://aclanthology.org/2026.eacl-long.174/) | [`demo_spotlight.py`](https://github.com/IBM/vLLM-Hook/blob/v0.2.0/examples/demo_spotlight.py) | — | `probe_spotlight` ‡ | [@danishcontractor](https://github.com/danishcontractor) |
| [Token Highlighter](TokenHighlighter.md) | [Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for LLMs](https://arxiv.org/abs/2412.18171) | [`demo_token_highlighter.py`](https://github.com/IBM/vLLM-Hook/blob/v0.2.0/examples/demo_token_highlighter.py) | `token_highlighter` | `token_highlighter` ‡ | [@asanth7](https://github.com/asanth7) |

> † Reuses an existing worker.
>
> ‡ Supported only on vLLM's V1 model runner (vLLM-Hook [v0.2.0](https://github.com/IBM/vLLM-Hook/tree/v0.2.0)); sunset in MIA.
