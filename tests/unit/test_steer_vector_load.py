"""CPU-only: steering vectors load with weights_only and respect MIA_STEER_VECTOR_DIR."""
import os

import numpy as np
import pytest
import torch

from mia.workers.steer_worker import _load_steering_vector

SHIPPED = os.path.join(os.path.dirname(__file__), "..", "..", "steering_vectors")


def _save(path, **kw):
    torch.save({"dir": torch.randn(8), "avg_proj": torch.tensor(0.5), **kw}, path)
    return str(path)


@pytest.mark.parametrize("name", sorted(n for n in os.listdir(SHIPPED) if n.endswith(".pt")))
def test_shipped_vectors_load(name, monkeypatch):
    monkeypatch.delenv("MIA_STEER_VECTOR_DIR", raising=False)
    out = _load_steering_vector(os.path.join(SHIPPED, name))
    assert torch.as_tensor(out["dir"]).ndim == 1


def test_numpy_vector_loads(tmp_path):
    path = _save(tmp_path / "v.pt", dir=np.ones(8, dtype=np.float32))
    assert np.array_equal(_load_steering_vector(path)["dir"], np.ones(8, dtype=np.float32))


class _Payload:
    ran = False

    def __reduce__(self):
        return (_run, ())


def _run():
    _Payload.ran = True


def test_pickle_payload_is_refused(tmp_path):
    path = _save(tmp_path / "evil.pt", dir=_Payload())
    with pytest.raises(Exception):
        _load_steering_vector(path)
    assert not _Payload.ran


def test_vector_dir_unset_allows_any_path(tmp_path, monkeypatch):
    monkeypatch.delenv("MIA_STEER_VECTOR_DIR", raising=False)
    assert "dir" in _load_steering_vector(_save(tmp_path / "v.pt"))


def test_vector_dir_allows_inside(tmp_path, monkeypatch):
    monkeypatch.setenv("MIA_STEER_VECTOR_DIR", str(tmp_path))
    assert "dir" in _load_steering_vector(_save(tmp_path / "v.pt"))


def test_vector_dir_refuses_outside(tmp_path, monkeypatch):
    (tmp_path / "allowed").mkdir()
    outside = _save(tmp_path / "v.pt")
    monkeypatch.setenv("MIA_STEER_VECTOR_DIR", str(tmp_path / "allowed"))
    with pytest.raises(ValueError):
        _load_steering_vector(outside)
    with pytest.raises(ValueError):
        _load_steering_vector(str(tmp_path / "allowed" / ".." / "v.pt"))
    link = tmp_path / "allowed" / "link.pt"
    os.symlink(outside, link)
    with pytest.raises(ValueError):
        _load_steering_vector(str(link))
