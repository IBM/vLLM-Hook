# tests/serve/test_discovery_registration.py
"""Route registration must dependency-inject the raw request.

Regression test for a PEP 563 pitfall: with ``from __future__ import
annotations`` in ``serve/discovery.py``, the ``raw_request: Request``
annotation becomes the string ``"Request"``, which FastAPI resolves
against the handler's module globals. The fastapi import is local to the
registering function, so the name is unresolvable there, and FastAPI
demotes the parameter to a required query field; every bare
``GET /v1/hook/capabilities`` then fails validation with
``('query', 'raw_request') missing`` before reaching the handler.

Runs against a bare FastAPI app with a stubbed engine client, so no
engine, GPU, or server fixture is needed.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_hook_plugins.serve import discovery


class _EngineStub:
    async def collective_rpc(self, method):
        assert method == "hook_capabilities"
        return [{"active_worker": "unified"}]


def test_bare_get_reaches_handler_with_injected_request(monkeypatch):
    monkeypatch.setattr(discovery, "_capabilities_cache", None)

    app = FastAPI()
    discovery._add_capabilities_route(app)
    app.state.engine_client = _EngineStub()

    client = TestClient(app)
    response = client.get("/v1/hook/capabilities")

    assert response.status_code == 200, response.text
    assert response.json()["active_worker"] == "unified"


def test_head_artifact_absent_returns_404(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_HOOK_REGISTRY_DIR", str(tmp_path))

    app = FastAPI()
    discovery._add_artifacts_route(app)

    client = TestClient(app)
    response = client.head("/v1/hook/artifacts/sha256:" + "ab" * 32)

    assert response.status_code == 404


def test_head_artifact_present_returns_200(monkeypatch, tmp_path):
    import os

    from vllm_hook_plugins.core.artifacts import ArtifactRegistry

    monkeypatch.setenv("VLLM_HOOK_REGISTRY_DIR", str(tmp_path))
    artifact_id = "sha256:" + "cd" * 32
    artifact_path = ArtifactRegistry().path_for(artifact_id)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "wb") as handle:
        handle.write(b"payload")

    app = FastAPI()
    discovery._add_artifacts_route(app)

    client = TestClient(app)
    response = client.head("/v1/hook/artifacts/" + artifact_id)

    assert response.status_code == 200
