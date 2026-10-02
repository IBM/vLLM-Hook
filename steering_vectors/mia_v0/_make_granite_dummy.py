"""One-shot generator for ``steering_vectors/granite_format.pt``.

The granite steer config (``model_configs/activation_steer/granite-3.1-8b-instruct.json``)
uses the ``adjust_rs`` method and points at ``steering_vectors/granite_format.pt``,
which is not checked in. This writes a dummy in the SAME format as the real
``phi3_format.pt`` (``dir`` = unit direction ndarray, ``avg_proj`` = scalar target
projection) sized to Granite-3.1-8B's residual width (``hidden_size = 4096``).

Like ``qwen2_dummy.pt``, the direction is a fixed random unit vector (seed 0) — it
does NOT correspond to any behavioural direction. It exists so the steering demos
and the graph-vs-eager parity harness can exercise the ``adjust_rs`` math under
CUDA-graph (Hybrid) mode on Granite. ``avg_proj`` is set large so the steer is
non-vacuous (it forces the residual's component along ``dir`` to ~``avg_proj``,
which clearly shifts the output); override via the env var if you want a gentler
or stronger steer.

Run once from the project root:

    python steering_vectors/_make_granite_dummy.py
"""
import os

import numpy as np
import torch

HIDDEN_SIZE = 4096  # ibm-granite/granite-3.1-8b-instruct residual stream width
AVG_PROJ = float(os.environ.get("VLLM_GRANITE_DUMMY_AVGPROJ", "300.0"))
OUT_PATH = "steering_vectors/granite_format.pt"

torch.manual_seed(0)
direction = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
direction = direction / direction.norm()  # unit vector — adjust_rs treats dir as the unit axis

# Match phi3_format.pt's container exactly: dir is a numpy ndarray, avg_proj a 0-d tensor.
payload = {
    "dir": direction.numpy().astype(np.float32),
    "avg_proj": torch.tensor(AVG_PROJ, dtype=torch.float32),
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save(payload, OUT_PATH)
print(f"[steering] wrote {OUT_PATH}  dir={payload['dir'].shape}/{payload['dir'].dtype} "
      f"avg_proj={float(payload['avg_proj']):.1f}  (method=adjust_rs)")
