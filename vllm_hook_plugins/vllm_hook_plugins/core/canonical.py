"""Canonical JSON bytes, spec hashing, and the reference cache-salt derivation.

Every hash on the declarative surface is computed over ``canonical_bytes``
of the object, so two clients producing the same logical spec always agree
on its hash regardless of key order or whitespace.
"""
from __future__ import annotations

import hashlib
import json


def canonical_bytes(obj) -> bytes:
    """UTF-8 of ``json.dumps(obj, sort_keys=True, separators=(",", ":"))``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def spec_hash(spec_obj: dict) -> str:
    """``sha256:<hex>`` of ``canonical_bytes(spec_obj)``. Used for logging
    and cache-salt derivation.
    """
    return "sha256:" + hashlib.sha256(canonical_bytes(spec_obj)).hexdigest()


def request_salt(spec_obj: dict, artifact_ids: list[str]) -> str:
    """Reference ``cache_salt`` derivation: SHA-256 over
    ``canonical_bytes(spec_obj)`` followed by the sorted artifact ids
    (each UTF-8 encoded). Returns the 64-char lowercase-hex digest.
    """
    h = hashlib.sha256(canonical_bytes(spec_obj))
    for artifact_id in sorted(artifact_ids):
        h.update(artifact_id.encode("utf-8"))
    return h.hexdigest()
