# Use Cases

Each row maps a use case to its plugin code and the corresponding contributor.

**Note for contributors:** When opening a PR that adds a new use case, append a row here. If your use case ships with a writeup, place it alongside this file as `<use_case>.md` and link it from the first column.

| Use case | Worker | Analyzer | Demo | Contributor |
| --- | --- | --- | --- | --- |
| Attention Tracker | `probe_hookqk_worker.py` | `attention_tracker_analyzer.py` | `demo_attntracker.py` (Colab: [@tburleyinfo](https://github.com/tburleyinfo)) | [@IRENEKO](https://github.com/IRENEKO) |
| Core Reranker | `probe_hookqk_worker.py` † | `core_reranker_analyzer.py` | `demo_corer.py` (Colab: [@tburleyinfo](https://github.com/tburleyinfo)) | [@IRENEKO](https://github.com/IRENEKO) |
| Activation Steering | `steer_activation_worker.py` | — | `demo_actsteer.py`, `demo_actsteer_serve.py` (Colab: [@tburleyinfo](https://github.com/tburleyinfo)) | [@IRENEKO](https://github.com/IRENEKO) |
| Hidden-State Probe | `probe_hidden_states_worker.py` | `hidden_states_analyzer.py` | `demo_hiddenstate.py` | [@IRENEKO](https://github.com/IRENEKO) |
| Science Hallucination Detector | `probe_hidden_states_worker.py` † | `science_hallucination_analyzer.py` | `demo_scihal.py` | [@IRENEKO](https://github.com/IRENEKO) |

> † Reuses an existing worker.
