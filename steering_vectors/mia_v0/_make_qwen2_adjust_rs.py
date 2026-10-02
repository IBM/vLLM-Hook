"""One-shot generator for ``steering_vectors/qwen2_adjust_rs.pt`` (adjust_rs format).

The qwen2_dummy.pt vector has only ``dir`` (add_vector). This writes the same vector
in the ``adjust_rs`` container (``dir`` = unit-direction ndarray, ``avg_proj`` = scalar
target projection) sized to Qwen2-1.5B-Instruct's residual width (hidden_size = 1536),
matching phi3_format.pt / granite_format.pt. It lets the graph-vs-eager parity harness
exercise the ``adjust_rs`` math under FULL CUDA graphs on Qwen2-1.5B.

The direction is a fixed random unit vector (seed 0) — meaningless behaviourally.
``avg_proj`` is set large enough that forcing the residual's component along ``dir`` to
~``avg_proj`` clearly shifts the output (non-vacuous steer); override via the env var.

Run once from the project root:

    python steering_vectors/_make_qwen2_adjust_rs.py
"""
import os

import numpy as np
import torch

HIDDEN_SIZE = 1536  # Qwen2-1.5B-Instruct residual stream width
AVG_PROJ = float(os.environ.get("VLLM_QWEN2_ADJRS_AVGPROJ", "50.0"))
OUT_PATH = "steering_vectors/qwen2_adjust_rs.pt"

torch.manual_seed(0)
direction = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
direction = direction / direction.norm()  # unit vector — adjust_rs treats dir as the unit axis

# Match phi3_format.pt / granite_format.pt: dir is a numpy ndarray, avg_proj a 0-d tensor.
payload = {
    "dir": direction.numpy().astype(np.float32),
    "avg_proj": torch.tensor(AVG_PROJ, dtype=torch.float32),
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save(payload, OUT_PATH)
print(f"[steering] wrote {OUT_PATH}  dir={payload['dir'].shape}/{payload['dir'].dtype} "
      f"avg_proj={float(payload['avg_proj']):.1f}  (method=adjust_rs)")
