# Use Cases

Each row maps a use case to its plugin code and the corresponding contributor.

> **Note for contributors:** When opening a PR that adds a new use case, append a row here. If your use case ships with a writeup, place it alongside this file as `<use_case>.md` and link it from the first column.

| Use case | Worker / Analyzer / Demo | Contributor |
| --- | --- | --- |
| Attention Tracker (prompt-injection guardrail) | `workers/probe_hookqk_worker.py`, `analyzers/attention_tracker_analyzer.py`, `examples/demo_attntracker.py` (Colab: @tburleyinfo) | @IRENEKO |
| Core Reranker | `analyzers/core_reranker_analyzer.py`, `examples/demo_corer.py` (Colab: @tburleyinfo) | @IRENEKO |
| Activation Steering | `workers/steer_activation_worker.py`, `examples/demo_actsteer.py`, `examples/demo_actsteer_serve.py` (Colab: @tburleyinfo) | @IRENEKO |
| Hidden-State Probe | `workers/probe_hidden_states_worker.py`, `analyzers/hidden_states_analyzer.py`, `examples/demo_hiddenstate.py` | @IRENEKO |
| Science Hallucination Detector | `analyzers/science_hallucination_analyzer.py`, `examples/demo_scihal.py` | @IRENEKO |
