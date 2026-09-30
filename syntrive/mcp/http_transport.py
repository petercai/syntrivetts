from __future__ import annotations

import hmac
import ipaddress
import logging
import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

from syntrive.io import server_registry
from syntrive.mcp.state import McpState

logger = logging.getLogger(__name__)

RECORD_NAME = "mcp"
HEALTH_PATH = "/api/health"
MCP_PATH = "/mcp"
DEFAULT_PORT = 8765
LOOPBACK = "127.0.0.1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def _is_loopback_client(client: str) -> bool:
    try:
        return ipaddress.ip_address(client).is_loopback
    except ValueError:
        return client == "localhost"


def _token_ok(header: str, token: str) -> bool:
    scheme, _, value = header.partition(" ")
    return scheme.lower() == "bearer" and bool(value) and hmac.compare_digest(value.strip(), token)


class BearerTokenMiddleware:
    def __init__(self, app, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        client = (scope.get("client") or ("",))[0]
        if scope["path"] == HEALTH_PATH and _is_loopback_client(client):
            return await self.app(scope, receive, send)
        header = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        if _token_ok(header, self.token):
            return await self.app(scope, receive, send)
        logger.warning("mcp_http_token_rejected: client=%s path=%s had_header=%s", client, scope["path"], bool(header))
        response = JSONResponse({"error": "unauthorized", "message": "Authorization: Bearer <token> required (printed by the server)."},
                                status_code=401, headers={"WWW-Authenticate": "Bearer"})
        await response(scope, receive, send)


class OnShutdown:
    def __init__(self, app, callback) -> None:
        self.app = app
        self.callback = callback

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "lifespan":
            return await self.app(scope, receive, send)

        async def send_wrapper(message) -> None:
            if message["type"] == "lifespan.shutdown.complete":
                self.callback()
            await send(message)

        await self.app(scope, receive, send_wrapper)


def build_app(server: MCPServer, state: McpState, *, host: str, token: Optional[str]):
    @server.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"name": "syntrive-mcp", "transport": "http", "repo_dir": str(state.current().repo_dir)})

    app = server.streamable_http_app(streamable_http_path=MCP_PATH, host=host)
    return BearerTokenMiddleware(app, token) if token else app


def serve_http(server: MCPServer, state: McpState, *, host: str, port: int, token: Optional[str]) -> int:
    import uvicorn

    repo_dir = state.current().repo_dir
    existing = server_registry.find_live_server(repo_dir, RECORD_NAME)
    if existing is not None:
        print(f"Error: this repo already has an MCP HTTP server: {existing.local_url.rstrip('/')}{MCP_PATH}", file=sys.stderr)
        return 1
    try:
        sock = server_registry.bind_port(host, port)
    except OSError as exc:
        logger.error("mcp_http_port_busy: host=%s port=%s error=%s", host, port, exc)
        print(f"Error: cannot listen on {host}:{port} ({exc}). Use --port <other> or --port 0 for a free port.", file=sys.stderr)
        return 1
    port = sock.getsockname()[1]

    def stopped() -> None:
        server_registry.remove_record(repo_dir, RECORD_NAME)
        logger.info("mcp_http_stopped: repo_dir=%s port=%s", repo_dir, port)

    app = OnShutdown(build_app(server, state, host=host, token=token), stopped)
    server_registry.write_record(repo_dir, RECORD_NAME, port, host=host, health_path=HEALTH_PATH)

    reachable = LOOPBACK if host in ("0.0.0.0", "::") or is_loopback(host) else host
    local_url = f"http://{reachable}:{port}{MCP_PATH}"
    print(f"syntrive-mcp (streamable HTTP): {local_url}", file=sys.stderr)
    if token:
        print(f"LAN mode: http://<this-machine-ip>:{port}{MCP_PATH} with header 'Authorization: Bearer {token}'", file=sys.stderr)
    logger.info("mcp_http_started: repo_dir=%s host=%s port=%s lan=%s token_prefix=%s",
                repo_dir, host, port, bool(token), (token[:4] + "…") if token else "-")
    try:
        uvicorn.Server(uvicorn.Config(app, log_level="warning", log_config=None)).run(sockets=[sock])
    finally:
        server_registry.remove_record(repo_dir, RECORD_NAME)
    return 0
