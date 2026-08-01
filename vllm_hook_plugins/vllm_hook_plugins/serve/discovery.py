"""Register GET /v1/hook/capabilities on the OpenAI-compatible app.

Patches the api-server app builder using the same multi-path/version
fallback pattern as the response-builder patches in ``_hook_plugin``.
The route lives on the serving app, so it inherits its auth middleware.
Payload comes from collective_rpc("hook_capabilities") at first request
and is memoized for the process lifetime.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger("vllm_hook.discovery")

_capabilities_cache = None


def register_discovery() -> None:
    """Patch the api-server app builders to serve the capabilities route.

    ``python -m vllm.entrypoints.openai.api_server`` executes the module
    as ``__main__`` — a distinct module object from the canonical import
    — so both are patched when present. Idempotent; silently a no-op when
    no api-server module is importable (offline usage — discovery goes
    through ``collective_rpc("hook_capabilities")`` there).
    """
    modules = []
    for _api_module in (
        "vllm.entrypoints.openai.api_server",
    ):
        try:
            import importlib as _il
            modules.append(_il.import_module(_api_module))
        except Exception:
            pass
    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "build_app") and hasattr(main_module, "run_server"):
        modules.append(main_module)

    for module in modules:
        build_app = getattr(module, "build_app", None)
        if build_app is None or getattr(build_app, "_vllm_hook_discovery", False):
            continue

        def _patched_build_app(args, *extra_args, _original=build_app, **kwargs):
            app = _original(args, *extra_args, **kwargs)
            _add_capabilities_route(app)
            return app

        _patched_build_app._vllm_hook_discovery = True
        module.build_app = _patched_build_app


def _add_capabilities_route(app) -> None:
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.get("/v1/hook/capabilities")
    async def hook_capabilities(raw_request: Request):
        global _capabilities_cache
        if _capabilities_cache is None:
            engine_client = raw_request.app.state.engine_client
            try:
                results = await engine_client.collective_rpc("hook_capabilities")
            except Exception:
                # Legacy workers have no hook_capabilities RPC method.
                results = None
            payload = next((r for r in results or () if r is not None), None)
            if payload is None:
                return JSONResponse(
                    {"error": "hook capabilities unavailable; is VLLM_HOOK_WORKER=unified set?"},
                    status_code=503,
                )
            _capabilities_cache = payload
        return JSONResponse(_capabilities_cache)

    logger.info("registered GET /v1/hook/capabilities")
