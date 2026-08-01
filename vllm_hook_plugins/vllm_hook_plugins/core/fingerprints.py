"""Tokenizer / chat-template / config fingerprint recipes.

Each recipe is deterministic and simple enough to recompute client-side,
so a client can verify it is talking to the model it prepared artifacts
for. All fingerprints are ``sha256:<hex>`` strings.
"""
from __future__ import annotations

import hashlib

from vllm_hook_plugins.core.canonical import canonical_bytes

# Config fields that vary across checkouts of the same logical model.
_VOLATILE_CONFIG_FIELDS = ("_name_or_path", "transformers_version")


def chat_template_fingerprint(template: str | None) -> str:
    """SHA-256 of the UTF-8 resolved chat-template string ('' when None)."""
    return "sha256:" + hashlib.sha256((template or "").encode("utf-8")).hexdigest()


def tokenizer_fingerprint(file_paths: list[str]) -> str:
    """SHA-256 over concatenated ``(filename, sha256(file_bytes))`` pairs,
    sorted by filename. Use ``tokenizer.json`` when present, else the
    vocab/merges file set.

    The hash ingests, per file in basename order, the UTF-8 basename
    followed by the lowercase-hex SHA-256 of the file's bytes.
    """
    digest = hashlib.sha256()
    for file_path in sorted(file_paths, key=lambda p: p.rsplit("/", 1)[-1]):
        basename = file_path.rsplit("/", 1)[-1]
        with open(file_path, "rb") as f:
            file_sha = hashlib.sha256(f.read()).hexdigest()
        digest.update(basename.encode("utf-8"))
        digest.update(file_sha.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def config_fingerprint(config_dict: dict) -> str:
    """SHA-256 of ``canonical_bytes`` of the config dict with volatile
    fields removed (``_name_or_path``, ``transformers_version``).
    """
    cleaned = {k: v for k, v in config_dict.items() if k not in _VOLATILE_CONFIG_FIELDS}
    return "sha256:" + hashlib.sha256(canonical_bytes(cleaned)).hexdigest()
