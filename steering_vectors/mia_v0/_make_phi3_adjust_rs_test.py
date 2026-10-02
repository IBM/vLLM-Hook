"""One-shot generator for ``steering_vectors/phi3_adjust_rs_test.pt`` (adjust_rs format).

The shipped phi3_format.pt / phi3_korean.pt / phi3_chinese.pt vectors are behavioural
demo vectors. This writes a neutral test vector in the ``adjust_rs`` container sized to
Phi-3-mini-4k-instruct's residual width (hidden_size = 3072), so the graph-vs-eager
parity harness can exercise ``adjust_rs`` x the phase/positions modes on Phi-3 without
depending on a demo vector's semantics.

The direction is a fixed random unit vector (seed 0) — meaningless behaviourally.
``avg_proj`` is a 0-d TENSOR, not a Python float: the eager path calls ``.to(device)``
on it while the graph path coerces either, so a float silently works under CUDA graphs
and crashes eager. Match the shipped containers.

Run once from the project root:

    python steering_vectors/_make_phi3_adjust_rs_test.py
"""
import os

import numpy as np
import torch

HIDDEN_SIZE = 3072  # Phi-3-mini-4k-instruct residual stream width
AVG_PROJ = float(os.environ.get("VLLM_PHI3_ADJRS_AVGPROJ", "20.0"))
OUT_PATH = "steering_vectors/phi3_adjust_rs_test.pt"

torch.manual_seed(0)
direction = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
direction = direction / direction.norm()  # unit vector — adjust_rs treats dir as the unit axis

payload = {
    "dir": direction.numpy().astype(np.float32),
    "avg_proj": torch.tensor(AVG_PROJ, dtype=torch.float32),
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save(payload, OUT_PATH)
print(f"[steering] wrote {OUT_PATH}  dir={payload['dir'].shape}/{payload['dir'].dtype} "
      f"avg_proj={float(payload['avg_proj']):.1f}  (method=adjust_rs)")
