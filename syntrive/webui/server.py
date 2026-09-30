from __future__ import annotations

import logging
import secrets
import sys
import time
import webbrowser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from syntrive.io import server_registry
from syntrive.webui.app import create_app
from syntrive.webui.shared.state import SERVER_NAME, NotARepoError, ServerInfo, open_web_repo

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logger = logging.getLogger(__name__)

_DEFAULT_REPO = "tmp"
_LOOPBACK = "127.0.0.1"


def _version() -> str:
    try:
        return version("syntrivetts")
    except PackageNotFoundError:
        return "0.0.0"


def main(
    repo_dir: Optional[Path] = typer.Option(None, "-r", "--repo-dir", help="SyntriveTTS repo (contains syntrivetts.db). Defaults to ./tmp."),
    host: str = typer.Option(_LOOPBACK, "--host", help="Bind address. Anything but 127.0.0.1 is LAN mode and requires the printed token."),
    open_browser: bool = typer.Option(True, "--browser/--no-browser", help="Open the page in the default browser."),
) -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, stream=sys.stdout)
    try:
        repo = open_web_repo(repo_dir or Path(_DEFAULT_REPO))
    except NotARepoError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    existing = server_registry.find_live_server(repo.repo_dir, SERVER_NAME)
    if existing is not None:
        typer.echo(f"Error: this repo is already served by a running WebUI: {existing.local_url}", err=True)
        raise typer.Exit(code=1)

    lan = host != _LOOPBACK
    token = secrets.token_urlsafe(18) if lan else None
    sock = server_registry.bind_free_port(host)
    port = sock.getsockname()[1]
    info = ServerInfo(port=port, host=host, started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), version=_version(), token=token)
    app = create_app(repo, info)

    local_url = f"http://{_LOOPBACK}:{port}/"
    open_url = f"{local_url}?token={token}" if token else local_url
    typer.echo(f"SyntriveTTS WebUI: {open_url}")
    if lan:
        typer.echo(f"LAN mode: from another device use http://<this-machine-ip>:{port}/?token={token}")
        logger.info("webui_lan_mode: host=%s token_prefix=%s…", host, token[:4])
    if open_browser:
        try:
            webbrowser.open(open_url)
        except Exception as exc:  # pragma: no cover - platform-dependent
            logger.warning("webui_browser_open_failed: error=%s", exc)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    try:
        server.run(sockets=[sock])
    finally:
        server_registry.remove_record(app.state.web.repo_dir, SERVER_NAME)


if __name__ == "__main__":
    typer.run(main)
