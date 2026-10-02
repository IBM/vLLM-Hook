"""One-shot generator for ``steering_vectors/qwen2_dummy.pt``.

Writes a fixed random fp16 tensor of shape ``(hidden_size,)`` = ``(1536,)``
matching Qwen2-1.5B-Instruct's residual stream width. Used only by the
profiling tool's R5 cell to exercise the applied-steering matmul-and-add
path; the direction is meaningless.

Run once from the project root after pulling:

    python steering_vectors/_make_qwen2_dummy.py
"""
import os
import torch

HIDDEN_SIZE = 1536  # Qwen2-1.5B-Instruct
OUT_PATH    = "steering_vectors/qwen2_dummy.pt"

torch.manual_seed(0)
direction = torch.randn(HIDDEN_SIZE, dtype=torch.float16)
direction = direction / direction.norm()    # unit vector — sane magnitude

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save({"dir": direction}, OUT_PATH)
print(f"[steering] wrote {OUT_PATH}  shape={tuple(direction.shape)} "
      f"dtype={direction.dtype}  norm={float(direction.norm()):.3f}")
