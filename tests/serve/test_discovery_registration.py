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