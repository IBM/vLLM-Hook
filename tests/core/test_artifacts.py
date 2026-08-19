# tests/core/test_artifacts.py
import json
import os

import pytest
import torch

from vllm_hook_plugins.core.artifacts import FORMAT_VERSION, ArtifactRegistry
from vllm_hook_plugins.core.schema import SpecError


@pytest.fixture
def registry(tmp_path):
    return ArtifactRegistry(str(tmp_path / "registry"))


def test_write_load_round_trip(registry):
    tensors = {"vector": torch.randn(64), "basis": torch.randn(2, 64)}
    artifact_id = registry.write(tensors, meta={"source": "unit-test"})
    assert artifact_id.startswith("sha256:")
    assert len(artifact_id) == len("sha256:") + 64

    loaded = registry.load(artifact_id)
    assert set(loaded) == {"vector", "basis"}
    assert torch.equal(loaded["vector"], tensors["vector"])
    assert torch.equal(loaded["basis"], tensors["basis"])


def test_write_is_content_addressed_and_idempotent(registry):
    tensors = {"vector": torch.ones(8)}
    first = registry.write(tensors)
    second = registry.write({"vector": torch.ones(8)})
    assert first == second
    # different content, different id
    assert registry.write({"vector": torch.zeros(8)}) != first


def test_layout_and_sidecar(registry):
    artifact_id = registry.write({"vector": torch.ones(4, dtype=torch.float32)})
    sha = artifact_id.split(":", 1)[1]
    path = registry.path_for(artifact_id)
    assert path.endswith(os.path.join(sha[:2], f"{sha}.safetensors"))
    assert os.path.exists(path)

    with open(path.replace(".safetensors", ".json")) as f:
        sidecar = json.load(f)
    assert sidecar["format_version"] == FORMAT_VERSION
    assert sidecar["tensors"]["vector"] == {"dtype": "float32", "shape": [4]}


def test_missing_artifact(registry):
    with pytest.raises(SpecError) as excinfo:
        registry.load("sha256:" + "00" * 32, path="ops[0].transform.artifact")
    assert excinfo.value.code == "E_ARTIFACT_MISSING"
    assert excinfo.value.path == "ops[0].transform.artifact"


def test_hash_tamper_detected(registry, tmp_path):
    artifact_id = registry.write({"vector": torch.randn(16)})
    file_path = registry.path_for(artifact_id)
    with open(file_path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        f.write(b"\x00")
    # a fresh registry has no memoized verification
    fresh = ArtifactRegistry(registry.root)
    with pytest.raises(SpecError) as excinfo:
        fresh.load(artifact_id)
    assert excinfo.value.code == "E_ARTIFACT_HASH"


def test_malformed_id_rejected(registry):
    with pytest.raises(SpecError) as excinfo:
        registry.load("md5:abcd")
    assert excinfo.value.code == "E_BAD_PARAM"


def test_gc_evicts_lru_but_never_pinned(registry):
    ids = [registry.write({"vector": torch.randn(1024)}) for _ in range(3)]
    sizes = [os.path.getsize(registry.path_for(i)) for i in ids]
    # order mtimes oldest -> newest explicitly (write timestamps can collide)
    for age, artifact_id in enumerate(ids):
        os.utime(registry.path_for(artifact_id), (1000 + age, 1000 + age))

    # budget for exactly one artifact; ids[0] is oldest but pinned
    registry.gc(budget_bytes=sizes[0], pinned={ids[0]})
    assert os.path.exists(registry.path_for(ids[0]))
    assert not os.path.exists(registry.path_for(ids[1]))
    assert not os.path.exists(registry.path_for(ids[2]))
    # sidecars evicted alongside
    assert not os.path.exists(registry.path_for(ids[1]).replace(".safetensors", ".json"))


def test_load_refreshes_lru_order(registry):
    ids = [registry.write({"vector": torch.randn(1024)}) for _ in range(2)]
    for age, artifact_id in enumerate(ids):
        os.utime(registry.path_for(artifact_id), (1000 + age, 1000 + age))
    registry.load(ids[0])  # refresh the older one
    size = os.path.getsize(registry.path_for(ids[0]))
    registry.gc(budget_bytes=size, pinned=set())
    assert os.path.exists(registry.path_for(ids[0]))
    assert not os.path.exists(registry.path_for(ids[1]))


def test_size_ceiling(registry, monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_MAX_ARTIFACT_MB", "0")
    with pytest.raises(SpecError) as excinfo:
        registry.write({"vector": torch.randn(64)})
    assert excinfo.value.code == "E_SPEC_TOO_LARGE"


def test_registry_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_HOOK_REGISTRY_DIR", str(tmp_path / "from_env"))
    registry = ArtifactRegistry()
    assert registry.root == str(tmp_path / "from_env")
