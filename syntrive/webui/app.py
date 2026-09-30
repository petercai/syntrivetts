from __future__ import annotations

import contextlib
import importlib
import logging
import pkgutil
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from syntrive.io import server_registry
from syntrive.services.job_lease import force_release
from syntrive.services.pipeline_service import StepRunRegistry
from syntrive.webui import features
from syntrive.webui.shared import errors, middleware
from syntrive.webui.shared.state import HEALTH_PATH, SERVER_NAME, ServerInfo, WebRepo
from syntrive.webui.shared.templating import build_environment

logger = logging.getLogger(__name__)

_WEBUI_DIR = Path(__file__).resolve().parent


def _mount_features(app: FastAPI) -> list:
    nav = []
    for info in sorted(pkgutil.iter_modules(features.__path__), key=lambda m: m.name):
        package_dir = _WEBUI_DIR / "features" / info.name
        if not (package_dir / "router.py").is_file():
            continue
        module = importlib.import_module(f"{features.__name__}.{info.name}.router")
        app.include_router(module.router)
        if (package_dir / "static").is_dir():
            app.mount(f"/static/{info.name}", StaticFiles(directory=package_dir / "static"), name=f"static-{info.name}")
        if getattr(module, "NAV", None) is not None:
            nav.append(module.NAV)
        logger.info("webui_feature_mounted: feature=%s routes=%d", info.name, len(module.router.routes))
    return sorted(nav, key=lambda item: item.order)


def create_app(repo: WebRepo, server: ServerInfo, *, register: bool = True) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "webui_server_started: repo_dir=%s host=%s port=%s lan_token=%s",
            app.state.web.repo_dir, server.host, server.port, "on" if server.token else "off",
        )
        yield
        if register:
            server_registry.remove_record(app.state.web.repo_dir, SERVER_NAME)

    app = FastAPI(title="SyntriveTTS WebUI", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.web = repo
    app.state.server = server
    app.state.register = register
    app.state.runs = StepRunRegistry()
    app.state.jinja = build_environment()

    middleware.install(app)
    errors.install(app)
    app.mount("/static/shared", StaticFiles(directory=_WEBUI_DIR / "shared" / "static"), name="static-shared")
    app.state.nav = _mount_features(app)

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/books", status_code=303)

    @app.get(HEALTH_PATH)
    def health(request: Request) -> dict:
        return {
            "repo_dir": str(request.app.state.web.repo_dir),
            "version": server.version,
            "started_at": server.started_at,
        }

    @app.post("/api/v1/leases/{job_id}/unlock")
    def unlock(job_id: int, request: Request) -> Response:
        removed = force_release(request.app.state.web.db_path, job_id, reason=f"webui unlock request_id={request.state.request_id}")
        logger.info("webui_action: feature=leases action=unlock job_id=%d result=%s", job_id, "released" if removed else "none")
        return Response(status_code=204, headers={"HX-Refresh": "true"})

    if register:
        server_registry.write_record(repo.repo_dir, SERVER_NAME, server.port, host=server.host, health_path=HEALTH_PATH)
    return app
