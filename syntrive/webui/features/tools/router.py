from __future__ import annotations

import logging
import sqlite3
from typing import Annotated, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response
from sqlalchemy.exc import IntegrityError

from syntrive.services import db_tools_service as dbt
from syntrive.webui.shared.errors import WebError
from syntrive.webui.shared.i18n import translate
from syntrive.webui.shared.state import HOLDER_KIND, WebRepo
from syntrive.webui.shared.templating import NavItem, render, request_lang

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])

NAV = NavItem(key="tools", label_key="nav.tools", href="/tools", order=60,
              count=lambda web: dbt.repo_file_count(web.repo_dir), section="maintenance")


def _web(request: Request) -> WebRepo:
    return request.app.state.web


def _t(request: Request, key: str, **fields: object) -> str:
    return translate(request_lang(request), key, **fields)


def _back(notice: str) -> Response:
    return Response(status_code=200, headers={"HX-Redirect": "/tools?" + urlencode({"notice": notice})})


def _file(request: Request, name: str):
    try:
        return dbt.repo_file(_web(request).repo_dir, name)
    except dbt.RepoFileError as exc:
        logger.info("webui_refused: feature=tools file=%r reason=%s", name, exc.code)
        raise WebError(404 if exc.code == "not_found" else 422, _t(request, f"tools.file_{exc.code}", name=name)) from exc


@router.get("/tools", response_class=HTMLResponse)
def tools_page(request: Request, notice: Optional[str] = None) -> HTMLResponse:
    return render(request, "tools/page.html", {"files": dbt.list_repo_files(_web(request).repo_dir), "notice": notice})


@router.get("/api/v1/tools/export", response_class=HTMLResponse)
def export_dialog(request: Request) -> HTMLResponse:
    return render(request, "tools/_export.html", {"books": dbt.list_books_with_jobs(_web(request).db_path)})


@router.get("/api/v1/tools/files/{name}/import", response_class=HTMLResponse)
def import_dialog(request: Request, name: str) -> HTMLResponse:
    plan = dbt.plan_import(_file(request, name), _web(request).db_path)
    if plan.export.error:
        raise WebError(422, _t(request, "tools.file_unreadable", name=name, error=plan.export.error))
    return render(request, "tools/_import.html", {
        "plan": plan, "name": name, "conflicts": {c.export_job_id: c for c in plan.conflicts}})


@router.get("/api/v1/tools/files/{name}/restore", response_class=HTMLResponse)
def restore_dialog(request: Request, name: str) -> HTMLResponse:
    path = _file(request, name)
    backup = next((b for b in dbt.list_backup_files(_web(request).repo_dir) if b.path == path), None)
    if backup is None:
        raise WebError(422, _t(request, "tools.file_not_backup", name=name))
    return render(request, "tools/_restore.html", {"backup": backup})


@router.post("/api/v1/tools/export")
def export(
    request: Request, job: Annotated[list[int], Form()] = [],  # noqa: B006
    include_voices: Annotated[Optional[str], Form()] = None,
) -> Response:
    if not job:
        raise WebError(422, _t(request, "tools.export_nothing"))
    try:
        result = dbt.export_to_repo(_web(request).db_path, sorted(set(job)), include_reference_voices=bool(include_voices))
    except ValueError as exc:
        raise WebError(422, str(exc)) from exc
    logger.info("webui_action: feature=tools action=export file=%s jobs=%d books=%d voices=%d",
                result.output_path.name, result.job_count, result.book_count, result.reference_voice_count)
    return _back(f"exported:{result.output_path.name}:{result.job_count}:{result.reference_voice_count}")


@router.post("/api/v1/tools/backup")
def backup(request: Request) -> Response:
    path, voices, skipped, tags = dbt.backup_to_repo(_web(request).db_path)
    logger.info("webui_action: feature=tools action=backup file=%s voices=%d skipped=%d tags=%d", path.name, voices, skipped, tags)
    return _back(f"backup:{path.name}:{voices}:{skipped}")


@router.post("/api/v1/tools/upload")
def upload(request: Request, file: Annotated[UploadFile, File()]) -> Response:
    try:
        saved = dbt.save_upload(_web(request).repo_dir, file.filename or "", file.file)
    except dbt.RepoFileError as exc:
        logger.info("webui_refused: feature=tools action=upload name=%r reason=%s", file.filename, exc.code)
        raise WebError(422, _t(request, f"tools.file_{exc.code}", name=exc.name)) from exc
    logger.info("webui_action: feature=tools action=upload name=%s saved_as=%s", file.filename, saved.name)
    return _back(f"uploaded:{saved.name}")


@router.get("/api/v1/tools/files/{name}")
def download(request: Request, name: str) -> FileResponse:
    path = _file(request, name)
    logger.info("webui_action: feature=tools action=download file=%s", name)
    return FileResponse(path, filename=name, media_type="application/octet-stream")


@router.post("/api/v1/tools/files/{name}/import")
async def import_file(request: Request, name: str) -> Response:
    path = _file(request, name)
    form = await request.form()
    try:
        job = sorted({int(v) for v in form.getlist("job")})
        overwrite = frozenset(int(k[5:]) for k, v in form.multi_items() if k.startswith("mode_") and v == "overwrite")
    except ValueError as exc:
        raise WebError(422, _t(request, "tools.import_bad_form")) from exc
    if not job:
        raise WebError(422, _t(request, "tools.import_nothing"))
    try:
        result = await run_in_threadpool(
            dbt.import_into_repo, path, _web(request).db_path, job,
            overwrite_job_ids=overwrite & frozenset(job), holder_kind=HOLDER_KIND)
    except ValueError as exc:
        raise WebError(422, str(exc)) from exc
    logger.info("webui_action: feature=tools action=import file=%s created=%d overwritten=%d skipped=%d voices_unresolved=%d",
                name, result.created_job_count, result.overwritten_job_count, result.skipped_job_count,
                result.reference_voice_unresolved_count)
    return _back(f"imported:{result.created_job_count}:{result.overwritten_job_count}:{result.skipped_job_count}")


@router.post("/api/v1/tools/files/{name}/restore")
def restore(request: Request, name: str) -> Response:
    path = _file(request, name)
    try:
        voices, tags = dbt.restore_reference_voices(_web(request).db_path, path)
    except (IntegrityError, sqlite3.IntegrityError) as exc:
        logger.info("webui_action: feature=tools action=restore file=%s result=collision", name)
        raise WebError(409, _t(request, "tools.restore_collision", name=name)) from exc
    logger.info("webui_action: feature=tools action=restore file=%s voices=%d tags=%d", name, voices, tags)
    return _back(f"restored:{voices}:{tags}")


@router.post("/api/v1/tools/files/{name}/delete")
def delete(request: Request, name: str) -> Response:
    _file(request, name)
    dbt.delete_repo_file(_web(request).repo_dir, name)
    logger.info("webui_action: feature=tools action=delete file=%s", name)
    return _back(f"deleted:{name}")
