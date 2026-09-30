from __future__ import annotations

import contextlib
import logging
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from fastapi import FastAPI, Request

from audio_review.api.routes import router as api_router
from audio_review.repo.lock import check_existing_server, remove_lock, write_lock
from audio_review.repo.state import open_repo
from syntrive.io.server_registry import bind_free_port

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT, stream=sys.stdout)
logger = logging.getLogger(__name__)

_DEFAULT_REPO = "tmp"
_WEB_DIR = Path(__file__).parent / "web"


def _ensure_repo_dir(repo_dir: Optional[Path]) -> Path:
    path = (repo_dir or Path(_DEFAULT_REPO)).resolve()
    if not path.is_dir():
        typer.echo(f"Error: repo dir does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    return path


def _bind_free_port() -> socket.socket:
    return bind_free_port()


def create_app(repo_dir: Path, port: int, *, open_browser: bool = True) -> FastAPI:
    active_repo = open_repo(repo_dir, port)
    write_lock(active_repo.repo_dir, port)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        url = f"http://127.0.0.1:{port}/"
        logger.info(
            "audio_review_server_started: repo_dir=%s port=%s url=%s",
            app.state.active_repo.repo_dir, port, url,
        )
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception as exc:  # pragma: no cover - platform-dependent
                logger.warning("audio_review_browser_open_failed: url=%s error=%s", url, exc)
        yield
        remove_lock(app.state.active_repo.repo_dir)

    app = FastAPI(title="SyntriveTTS Audio Review", lifespan=lifespan)
    app.state.active_repo = active_repo
    app.include_router(api_router)
    app.frontend("/", directory=_WEB_DIR)

    @app.middleware("http")
    async def revalidate_static_assets(request: Request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response
    return app


def main(
    repo_dir: Optional[Path] = typer.Option(
        None,
        "-r",
        "--repo-dir",
        help="SyntriveTTS repo directory (must already contain syntrivetts.db). Defaults to ./tmp.",
        file_okay=False,
        dir_okay=True,
        resolve_path=False,
    ),
) -> None:
    resolved_repo = _ensure_repo_dir(repo_dir)

    existing_url = check_existing_server(resolved_repo)
    if existing_url is not None:
        typer.echo(
            f"Error: repo already served by a running audio_review instance: {existing_url}",
            err=True,
        )
        raise typer.Exit(code=1)

    sock = _bind_free_port()
    port = sock.getsockname()[1]

    app = create_app(resolved_repo, port)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[sock])
    finally:
        remove_lock(resolved_repo)


if __name__ == "__main__":
    typer.run(main)
