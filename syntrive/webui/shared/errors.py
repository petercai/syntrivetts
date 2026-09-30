from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from syntrive.services.job_lease import JobLeaseConflict
from syntrive.webui.shared.templating import is_htmx, render

logger = logging.getLogger(__name__)


class WebError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _respond(request: Request, status_code: int, context: dict, *, unlock_job_id: Optional[int] = None):
    ctx = {**context, "status_code": status_code, "unlock_job_id": unlock_job_id}
    if is_htmx(request):
        return render(
            request, "shared/_flash.html", ctx, status_code=status_code,
            headers={"HX-Retarget": "#flash", "HX-Reswap": "innerHTML"},
        )
    return render(request, "shared/error.html", ctx, status_code=status_code)


def install(app: FastAPI) -> None:
    @app.exception_handler(WebError)
    async def web_error(request: Request, exc: WebError):
        logger.info("webui_refused: path=%s status=%s reason=%s", request.url.path, exc.status_code, exc.message)
        return _respond(request, exc.status_code, {"level": "warning", "message": exc.message})

    @app.exception_handler(JobLeaseConflict)
    async def lease_conflict(request: Request, exc: JobLeaseConflict):
        logger.warning("webui_lease_conflict: path=%s job_id=%s holder=%s", request.url.path, exc.job_id, exc.holder.holder_kind)
        return _respond(request, 409, {"level": "warning", "message": str(exc)}, unlock_job_id=exc.job_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        logger.info("webui_validation_failed: path=%s errors=%s", request.url.path, fields)
        return _respond(request, 422, {"level": "warning", "message": fields})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return _respond(request, exc.status_code, {"level": "warning", "message": str(exc.detail)})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        logger.error(
            "webui_unexpected_error: path=%s request_id=%s", request.url.path,
            getattr(request.state, "request_id", ""), exc_info=exc,
        )
        return _respond(request, 500, {"level": "error", "message": None})
