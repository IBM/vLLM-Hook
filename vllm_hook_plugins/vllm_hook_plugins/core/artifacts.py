"""Content-addressed safetensors registry.

Layout: ``{root}/{sha[:2]}/{sha}.safetensors`` plus a ``{sha}.json``
sidecar carrying dtype/shape metadata and ``format_version``. Writes are
atomic (tmp+rename, as in ``workers._common``); reads verify the content
hash once and memoize the verification. No pickle on any path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import safetensors.torch

from vllm_hook_plugins.core.schema import (
    E_ARTIFACT_HASH,
    E_ARTIFACT_MISSING,
    E_BAD_PARAM,
    E_SPEC_TOO_LARGE,
    SpecError,
)

logger = logging.getLogger("vllm_hook.artifacts")

FORMAT_VERSION = 1

# Default root sits beside the hook_dir convention shared with HookClient:
# /dev/shm/vllm_hook is a RAM tmpfs on Linux — fast and ephemeral.
_DEFAULT_REGISTRY_DIR = "/dev/shm/vllm_hook/artifacts"


def default_registry_root() -> str:
    """Root resolution: ``VLLM_HOOK_REGISTRY_DIR`` env var, else the
    ``artifacts/`` directory beside the default hook_dir.
    """
    return os.environ.get("VLLM_HOOK_REGISTRY_DIR", _DEFAULT_REGISTRY_DIR)


def _max_artifact_bytes() -> int:
    return int(os.environ.get("VLLM_HOOK_MAX_ARTIFACT_MB", "512")) * 1024 * 1024


class ArtifactRegistry:

    def __init__(self, root: str | None = None):
        self.root = root or default_registry_root()
        self._verified: set = set()  # artifact ids whose content hash checked out

    def path_for(self, artifact_id: str) -> str:
        sha = self._hex(artifact_id)
        return os.path.join(self.root, sha[:2], f"{sha}.safetensors")

    def write(self, tensors: dict, meta: dict | None = None) -> str:
        """Write atomically; return the ``sha256:<hex>`` id.

        Tensors are serialized with sorted names so the same logical
        content always produces the same bytes, and therefore the same id.
        """
        data = safetensors.torch.save({name: tensors[name].contiguous() for name in sorted(tensors)})
        if len(data) > _max_artifact_bytes():
            raise SpecError(
                E_SPEC_TOO_LARGE,
                "",
                f"artifact is {len(data)} bytes; ceiling is VLLM_HOOK_MAX_ARTIFACT_MB"
                f"={_max_artifact_bytes() // (1024 * 1024)} MB",
            )
        sha = hashlib.sha256(data).hexdigest()
        artifact_id = f"sha256:{sha}"

        out_path = self.path_for(artifact_id)
        if os.path.exists(out_path):
            return artifact_id
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, out_path)

        sidecar = {
            "format_version": FORMAT_VERSION,
            "tensors": {
                name: {"dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape)}
                for name, t in tensors.items()
            },
            "meta": meta or {},
        }
        sidecar_path = self._sidecar_path(artifact_id)
        tmp_sidecar = sidecar_path + ".tmp"
        with open(tmp_sidecar, "w") as f:
            json.dump(sidecar, f)
        os.rename(tmp_sidecar, sidecar_path)

        self._verified.add(artifact_id)
        logger.info("wrote artifact %s (%d bytes)", artifact_id, len(data))
        return artifact_id

    def load(self, artifact_id: str, *, path: str = "") -> dict:
        """Load and verify. Raises ``SpecError(E_ARTIFACT_MISSING, ...)`` /
        ``SpecError(E_ARTIFACT_HASH, ...)``.

        ``path`` is the JSON path of the spec element that referenced the
        artifact, carried into any error. The content hash is checked on
        first load and memoized; the file's mtime is refreshed on every
        load so ``gc`` evicts least-recently-used artifacts first.
        """
        file_path = self.path_for(artifact_id)
        if not os.path.exists(file_path):
            raise SpecError(
                E_ARTIFACT_MISSING,
                path,
                f"artifact {artifact_id} not found in registry {self.root}; upload it first",
            )
        if os.path.getsize(file_path) > _max_artifact_bytes():
            raise SpecError(
                E_SPEC_TOO_LARGE,
                path,
                f"artifact {artifact_id} exceeds VLLM_HOOK_MAX_ARTIFACT_MB",
            )
        with open(file_path, "rb") as f:
            data = f.read()
        if artifact_id not in self._verified:
            sha = hashlib.sha256(data).hexdigest()
            if f"sha256:{sha}" != artifact_id:
                raise SpecError(
                    E_ARTIFACT_HASH,
                    path,
                    f"artifact {artifact_id} content hashes to sha256:{sha}; re-upload it",
                )
            self._verified.add(artifact_id)
        os.utime(file_path)
        return safetensors.torch.load(data)

    def gc(self, budget_bytes: int, pinned: set) -> None:
        """LRU-evict beyond ``budget_bytes``, never touching ``pinned``."""
        entries = []
        total = 0
        if not os.path.isdir(self.root):
            return
        for shard in os.listdir(self.root):
            shard_dir = os.path.join(self.root, shard)
            if not os.path.isdir(shard_dir):
                continue
            for name in os.listdir(shard_dir):
                if not name.endswith(".safetensors"):
                    continue
                artifact_id = f"sha256:{name[: -len('.safetensors')]}"
                file_path = os.path.join(shard_dir, name)
                stat = os.stat(file_path)
                entries.append((stat.st_mtime, stat.st_size, artifact_id, file_path))
                total += stat.st_size

        for _mtime, size, artifact_id, file_path in sorted(entries):
            if total <= budget_bytes:
                return
            if artifact_id in pinned:
                continue
            for victim in (file_path, self._sidecar_path(artifact_id)):
                try:
                    os.remove(victim)
                except FileNotFoundError:
                    pass
            self._verified.discard(artifact_id)
            total -= size
            logger.info("evicted artifact %s (%d bytes)", artifact_id, size)

    def _sidecar_path(self, artifact_id: str) -> str:
        sha = self._hex(artifact_id)
        return os.path.join(self.root, sha[:2], f"{sha}.json")

    @staticmethod
    def _hex(artifact_id: str) -> str:
        prefix, _, sha = artifact_id.partition(":")
        if prefix != "sha256" or len(sha) != 64:
            raise SpecError(
                E_BAD_PARAM,
                "",
                "artifact id must be 'sha256:' followed by 64 lowercase hex chars",
            )
        return sha
