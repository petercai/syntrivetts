from __future__ import annotations

import logging
import threading
from typing import Annotated, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, Response

from syntrive.adapters.tts import env_provisioner
from syntrive.services import batch_runner
from syntrive.services import model_library as ml
from syntrive.webui.shared.errors import WebError
from syntrive.webui.shared.i18n import translate
from syntrive.webui.shared.templating import NavItem, render, request_lang

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


def _nav_count(_web) -> str:
    downloaded, total = ml.model_counts()
    return f"{downloaded}/{total}"


NAV = NavItem(key="models", label_key="nav.models", href="/models", order=50, count=_nav_count, section="library")

_TASKS_LOCK = threading.Lock()


def _tasks(request: Request) -> ml.ModelTasks:
    with _TASKS_LOCK:
        tasks = getattr(request.app.state, "model_tasks", None)
        if tasks is None:
            tasks = request.app.state.model_tasks = ml.ModelTasks()
        return tasks


def _t(request: Request, key: str, **fields: object) -> str:
    return translate(request_lang(request), key, **fields)


def _offline_locked() -> bool:
    return batch_runner.offline_default().locked


def _back(notice: str) -> Response:
    return Response(status_code=200, headers={"HX-Redirect": "/models?" + urlencode({"notice": notice})})


def _row(request: Request, engine: str, item: str) -> ml.RowModel:
    row = ml.find_row(engine, item)
    if row is None:
        raise WebError(404, _t(request, "models.not_found", item=f"{engine}/{item}"))
    return row


def _require_network(request: Request, action: str) -> None:
    if _offline_locked():
        logger.info("webui_action: feature=models action=%s result=refused reason=offline", action)
        raise WebError(409, _t(request, "models.offline_refused"))


def _queued(request: Request, view: Optional[ml.TaskView], action: str, item: str) -> Response:
    if view is None:
        raise WebError(409, _t(request, "models.busy", item=item))
    logger.info("webui_action: feature=models action=%s item=%s task_id=%d", action, item, view.id)
    return _back(f"queued:{view.name}")


@router.get("/models", response_class=HTMLResponse)
def models_page(request: Request, notice: Optional[str] = None) -> HTMLResponse:
    tasks = _tasks(request)
    busy = tasks.busy_keys()
    groups = ml.read_model_catalog()
    device, detected = ml.compute_device()
    return render(request, "models/page.html", {
        "groups": groups,
        "busy": busy,
        "open": {g.engine for g in groups if g.downloaded or any(r.downloaded for r in g.rows)
                 or any(k in busy for r in g.rows for k in ml.row_keys(r)) or f"env:{g.engine}" in busy},
        "row_keys": ml.row_keys,
        "row_status": ml.row_status,
        "counts": ml.model_counts(),
        "cache_bytes": ml.total_cache_bytes(),
        "device": device, "detected": detected, "devices": env_provisioner.VALID_DEVICES,
        "offline": _offline_locked(),
        "tasks": tasks.views(),
        "notice": notice,
    })


@router.get("/api/v1/models/activity", response_class=HTMLResponse)
def activity(request: Request, busy: int = 0) -> Response:
    tasks = _tasks(request)
    if busy and not tasks.busy_keys():
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return render(request, "models/_activity.html", {"tasks": tasks.views(), "busy": tasks.busy_keys()})


@router.get("/api/v1/models/{engine}/{item}/confirm", response_class=HTMLResponse)
def confirm_download(request: Request, engine: str, item: str) -> HTMLResponse:
    row = _row(request, engine, item)
    _require_network(request, "confirm")
    return render(request, "models/_confirm.html", {"row": row, "check": ml.disk_check(row.approx_gb),
                                                    "low_bytes": ml.LOW_SPACE_BYTES})


@router.get("/api/v1/models/{engine}/{item}/files", response_class=HTMLResponse)
def files(request: Request, engine: str, item: str) -> HTMLResponse:
    row = _row(request, engine, item)
    return render(request, "models/_files.html", {"row": row, "files": ml.major_files(engine, item) if row.kind == "model" else []})


@router.post("/api/v1/models/{engine}/{item}/download")
def download(request: Request, engine: str, item: str) -> Response:
    row = _row(request, engine, item)
    _require_network(request, "download")
    tasks = _tasks(request)
    view = tasks.download_resource(engine, item) if row.kind == "resource" else tasks.download(engine, item)
    return _queued(request, view, "download", f"{engine}/{item}")


@router.post("/api/v1/models/{engine}/{item}/update")
def update(request: Request, engine: str, item: str) -> Response:
    row = _row(request, engine, item)
    _require_network(request, "update")
    tasks = _tasks(request)
    view = tasks.download_resource(engine, item) if row.kind == "resource" else tasks.download(engine, item, update=True)
    return _queued(request, view, "update", f"{engine}/{item}")


@router.post("/api/v1/models/{engine}/{item}/check")
def check(request: Request, engine: str, item: str) -> Response:
    row = _row(request, engine, item)
    if row.kind != "model":
        raise WebError(422, _t(request, "models.check_models_only"))
    _require_network(request, "check")
    return _queued(request, _tasks(request).check_update(engine, item), "check", f"{engine}/{item}")


@router.post("/api/v1/models/{engine}/{item}/remove")
def remove(request: Request, engine: str, item: str) -> Response:
    _row(request, engine, item)
    return _queued(request, _tasks(request).remove(engine, item), "remove", f"{engine}/{item}")


@router.post("/api/v1/models/{engine}/provision")
def provision(request: Request, engine: str) -> Response:
    _require_network(request, "provision")
    device, _ = ml.compute_device()
    try:
        view = _tasks(request).provision(engine, device)
    except ml.UnknownModel as exc:
        raise WebError(404, _t(request, "models.no_env", engine=engine)) from exc
    return _queued(request, view, "provision", f"{engine} ({device})")


@router.post("/api/v1/models/device")
def set_device(request: Request, device: Annotated[str, Form()]) -> Response:
    if device not in env_provisioner.VALID_DEVICES:
        raise WebError(422, _t(request, "models.bad_device", device=device))
    ml.save_compute_device(device)
    logger.info("webui_action: feature=models action=device device=%s", device)
    return _back(f"device:{device}")
