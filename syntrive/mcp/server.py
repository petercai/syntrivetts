from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import secrets
import sys
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer

from syntrive.mcp import prompts, tools
from syntrive.mcp.state import TRANSPORTS, McpState, open_repo

logger = logging.getLogger(__name__)

SERVER_NAME = "syntrive-mcp"
INSTRUCTIONS = (
    "SyntriveTTS turns EPUB ebooks into chaptered audiobooks with neural TTS. This server works on one repo "
    "(repo_info says which; repo_switch changes it). A book goes through 9 steps (pipeline_status); run the "
    "current one with pipeline_run_step, take the decisions with the pipeline_* / tts_settings_* tools, and queue "
    "synthesis with queue_enqueue + queue_start_runner. Destructive tools preview first (dry_run=true) and need the "
    "preview's confirm_token to execute. Errors start with a code: not_found, invalid, conflict, refused, failed."
)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("syntrivetts")
    except PackageNotFoundError:
        return "0.0.0"


def create_server(repo_dir: Path, transport: str = "stdio") -> tuple[MCPServer, McpState]:
    state = McpState(open_repo(repo_dir), transport=transport)
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, version=_version())
    names = []
    for info in sorted(pkgutil.iter_modules(tools.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{tools.__name__}.{info.name}")
        module.register(server, state)
        names.append(info.name)
    prompts.register(server)
    logger.info("mcp_server_created: repo_dir=%s namespaces=%s", state.repo.repo_dir, names)
    return server, state


def _configure_logging(repo_dir: Path, level: str) -> Path:
    from logging.handlers import RotatingFileHandler

    log_dir = repo_dir / ".syntrive" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mcp.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)7s] %(name)s: %(message)s")
    stderr = logging.StreamHandler(sys.stderr)
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 2**20, backupCount=3, encoding="utf-8")
    for handler in (stderr, file_handler):
        handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [stderr, file_handler]
    root.setLevel(level)
    return log_file


def main(argv: Optional[list[str]] = None) -> int:
    from syntrive.mcp import http_transport

    parser = argparse.ArgumentParser(prog="syntrive-mcp", description="SyntriveTTS MCP server (stdio or streamable HTTP).")
    parser.add_argument("-r", "--repo-dir", type=Path, default=Path("tmp"), help="Repo holding syntrivetts.db (default ./tmp).")
    parser.add_argument("--transport", default="stdio", choices=TRANSPORTS, help="stdio (default) or http (DESIGN §13).")
    parser.add_argument("--host", default=http_transport.LOOPBACK,
                        help="http only. Anything but loopback is LAN mode: a bearer token is required (printed once).")
    parser.add_argument("--port", type=int, default=http_transport.DEFAULT_PORT, help="http only (default 8765; 0 = a free port).")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    repo_dir = args.repo_dir.expanduser().resolve()
    if not (repo_dir / "syntrivetts.db").is_file():
        print(f"Not a SyntriveTTS repo (missing syntrivetts.db): {repo_dir}", file=sys.stderr)
        return 2
    log_file = _configure_logging(repo_dir, args.log_level)
    server, state = create_server(repo_dir, args.transport)
    logger.info("mcp_server_started: repo_dir=%s transport=%s log=%s version=%s", repo_dir, args.transport, log_file, _version())
    if args.transport == "http":
        token = None if http_transport.is_loopback(args.host) else secrets.token_urlsafe(24)
        code = http_transport.serve_http(server, state, host=args.host, port=args.port, token=token)
    else:
        server.run()
        code = 0
    logger.info("mcp_server_stopped: repo_dir=%s code=%s", repo_dir, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
