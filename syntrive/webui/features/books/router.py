from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from syntrive.services import book_catalog
from syntrive.services.job_service import JobService
from syntrive.webui.shared.errors import WebError
from syntrive.webui.shared.i18n import translate
from syntrive.webui.shared.state import HOLDER_KIND, WebRepo
from syntrive.webui.shared.templating import NavItem, render, request_lang

logger = logging.getLogger(__name__)

router = APIRouter(tags=["books"])

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
_CHUNK = 1024 * 1024
_UPLOADS_DIR = Path(".syntrive") / "uploads"

Tab = Literal["active", "done", "archived"]


def _active_count(web: WebRepo) -> int:
    return sum(1 for row in book_catalog.list_book_rows(web.db_path) if not row.archived)


NAV = NavItem(key="books", label_key="nav.books", href="/books", order=10, count=_active_count)


def _web(request: Request) -> WebRepo:
    return request.app.state.web


def _t(request: Request, key: str, **fields: object) -> str:
    return translate(request_lang(request), key, **fields)


def _panel_context(web: WebRepo, tab: str, q: str) -> dict:
    rows = book_catalog.list_book_rows(web.db_path)
    return {
        "tab": tab,
        "q": q,
        "counts": book_catalog.tab_counts(rows),
        "rows": book_catalog.filter_rows(rows, tab=tab, query=q),
    }


def _require_row(web: WebRepo, request: Request, job_id: int) -> book_catalog.BookRow:
    row = book_catalog.get_book_row(web.db_path, job_id)
    if row is None:
        raise WebError(404, _t(request, "books.not_found", id=job_id))
    return row


@router.get("/books", response_class=HTMLResponse)
def books_page(request: Request, tab: Tab = "active", q: str = "") -> HTMLResponse:
    return render(request, "books/page.html", _panel_context(_web(request), tab, q))


@router.get("/api/v1/books/rows", response_class=HTMLResponse)
def books_rows(request: Request, tab: Tab = "active", q: str = "") -> HTMLResponse:
    return render(request, "books/_panel.html", _panel_context(_web(request), tab, q))


@router.get("/api/v1/books/{job_id}/cover")
def book_cover(request: Request, job_id: int) -> FileResponse:
    row = _require_row(_web(request), request, job_id)
    if row.cover_path is None or not row.cover_path.is_file():
        raise WebError(404, _t(request, "books.no_cover"))
    return FileResponse(row.cover_path)


def _save_upload(upload: UploadFile, dest: Path, request: Request) -> int:
    written = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(_CHUNK):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise WebError(413, _t(request, "books.upload_too_large", mb=MAX_UPLOAD_BYTES // (1024 * 1024)))
            out.write(chunk)
    return written


@router.post("/api/v1/books")
def add_book(request: Request, file: UploadFile) -> Response:
    from syntrive.bootstrap import bootstrap

    web = _web(request)
    name = Path(file.filename or "").name
    if not name.lower().endswith(".epub"):
        raise WebError(422, _t(request, "books.upload_not_epub"))

    staging = web.repo_dir / _UPLOADS_DIR / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=True)
    try:
        size = _save_upload(file, staging / name, request)
        logger.info("webui_upload_received: name=%s bytes=%d", name, size)
        job = bootstrap(web.repo_dir, staging / name)
    except WebError:
        raise
    except Exception as exc:
        logger.error("webui_action: feature=books action=add name=%s result=failed", name, exc_info=True)
        raise WebError(422, _t(request, "books.upload_failed", error=exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("webui_action: feature=books action=add job_id=%d name=%s result=ok", job.id, name)
    return Response(status_code=200, headers={"HX-Redirect": f"/pipeline/{job.id}"})


def _set_archived(request: Request, job_id: int, *, archived: bool, tab: str, q: str, after: str) -> Response:
    web = _web(request)
    row = _require_row(web, request, job_id)
    service = JobService(web.db_path, holder_kind=HOLDER_KIND)
    result = service.archive_job(job_id) if archived else service.unarchive_job(job_id)
    action = "archive" if archived else "restore"
    logger.info("webui_action: feature=books action=%s job_id=%d result=%s changed=%s", action, job_id, "ok" if result.ok else "failed", result.changed)
    if not result.ok:
        raise WebError(422, result.error or _t(request, "books.archive_failed"))
    if after == "refresh":
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    notice = _t(request, "books.archived_notice" if archived else "books.restored_notice", title=row.title)
    return render(request, "books/_panel.html", {**_panel_context(web, tab, q), "notice": notice})


@router.post("/api/v1/books/{job_id}/archive", response_class=HTMLResponse)
def archive_book(
    request: Request, job_id: int,
    tab: Annotated[Tab, Form()] = "active", q: Annotated[str, Form()] = "", after: Annotated[str, Form()] = "panel",
) -> Response:
    return _set_archived(request, job_id, archived=True, tab=tab, q=q, after=after)


@router.post("/api/v1/books/{job_id}/restore", response_class=HTMLResponse)
def restore_book(
    request: Request, job_id: int,
    tab: Annotated[Tab, Form()] = "archived", q: Annotated[str, Form()] = "", after: Annotated[str, Form()] = "panel",
) -> Response:
    return _set_archived(request, job_id, archived=False, tab=tab, q=q, after=after)


@router.get("/api/v1/books/{job_id}/delete", response_class=HTMLResponse)
def delete_dialog(request: Request, job_id: int) -> HTMLResponse:
    preview = book_catalog.delete_preview(_web(request).db_path, job_id)
    if preview is None:
        raise WebError(404, _t(request, "books.not_found", id=job_id))
    return render(request, "books/_delete_dialog.html", {"preview": preview})


@router.post("/api/v1/books/{job_id}/delete")
def delete_book(request: Request, job_id: int, confirm_title: Annotated[str, Form()] = "") -> Response:
    web = _web(request)
    row = _require_row(web, request, job_id)
    if confirm_title.strip() != row.title:
        raise WebError(422, _t(request, "books.delete_title_mismatch"))
    result = JobService(web.db_path, holder_kind=HOLDER_KIND).delete_book_job(job_id)
    logger.info(
        "webui_action: feature=books action=delete job_id=%d result=ok jobs=%s dirs=%s",
        job_id, result.get("jobs_deleted"), result.get("dirs_deleted"),
    )
    return Response(status_code=200, headers={"HX-Redirect": "/books"})
