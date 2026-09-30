from __future__ import annotations

import hmac
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, RedirectResponse

from syntrive.webui.shared.state import HEALTH_PATH

logger = logging.getLogger(__name__)

TOKEN_COOKIE = "syntrive_token"
_LOOPBACK_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _token_ok(candidate: str | None, token: str) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate, token)


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def access_token(request: Request, call_next):
        token = request.app.state.server.token
        if token is None:
            return await call_next(request)
        client = request.client.host if request.client else ""
        if request.url.path == HEALTH_PATH and client in _LOOPBACK_CLIENTS:
            return await call_next(request)
        if _token_ok(request.cookies.get(TOKEN_COOKIE), token):
            return await call_next(request)
        if _token_ok(request.query_params.get("token"), token):
            clean = request.url.remove_query_params("token")
            response = RedirectResponse(str(clean), status_code=303)
            response.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite="strict")
            logger.info("webui_token_accepted: client=%s", client)
            return response
        logger.warning("webui_token_rejected: client=%s path=%s", client, request.url.path)
        return PlainTextResponse("Access token required: open the URL printed by the server.", status_code=401)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - started) * 1000
        level = logging.DEBUG if request.url.path.startswith("/static/") else logging.INFO
        logger.log(
            level, "webui_request: method=%s path=%s status=%s ms=%.0f request_id=%s",
            request.method, request.url.path, response.status_code, ms, request_id,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers.setdefault("Cache-Control", "no-cache")
        return response
