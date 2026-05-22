"""Copy to tests/parity_tests/local_wandb_config.py and paste your W&B key there.

tests/parity_tests/local_wandb_config.py is ignored by git.
"""

WANDB_API_KEY = "paste_your_wandb_key_here"
WANDB_PROJECT = "vllm-hook-mlx-gpu-parity"
WANDB_ENTITY = "tm8ctgzqj8-georgia-institute-of-technology"

# Preferred for Colab runs: store the W&B key in Google Cloud Secret Manager
# and leave WANDB_API_KEY local-only.
WANDB_SECRET_PROJECT = "your-gcp-project-id"
WANDB_SECRET_NAME = "wandb-api-key"
WANDB_SECRET_VERSION = "latest"
