from __future__ import annotations

import logging
from typing import Annotated, Literal, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, Response

from syntrive.services import batch_runner, queue_overview, synthesis_service
from syntrive.webui.shared.errors import WebError
from syntrive.webui.shared.i18n import translate
from syntrive.webui.shared.state import WebRepo
from syntrive.webui.shared.templating import NavItem, render, request_lang

logger = logging.getLogger(__name__)

router = APIRouter(tags=["queue"])

PRESELECT = 5

NAV = NavItem(key="queue", label_key="nav.queue", href="/queue", order=20,
              count=lambda web: queue_overview.open_batch_count(web.db_path), accent=True)


def _web(request: Request) -> WebRepo:
    return request.app.state.web


def _t(request: Request, key: str, **fields: object) -> str:
    return translate(request_lang(request), key, **fields)


def _back(book: Optional[int], notice: str, view: Optional[str] = None) -> Response:
    query = urlencode({k: v for k, v in (("book", book), ("view", view), ("notice", notice)) if v})
    return Response(status_code=200, headers={"HX-Redirect": f"/queue?{query}"})


def _now_context(now) -> dict:
    from syntrive.services.step_decisions import split_markers

    text = now.progress.text if now.progress and now.progress.text else ""
    return {"now": now, "now_line": split_markers(text)[0] if text.strip() else ()}


@router.get("/queue", response_class=HTMLResponse)
def queue_page(
    request: Request, book: Optional[int] = None, notice: Optional[str] = None,
    view: Literal["chapters", "history"] = "chapters",
) -> HTMLResponse:
    web = _web(request)
    overview = queue_overview.read_queue(web.db_path)
    books = overview.books
    ids = [b.job_id for b in books]
    selected_id = book if book in ids else next((b.job_id for b in books if b.schedulable), ids[0] if ids else None)
    book_queue = queue_overview.read_book_queue(web.db_path, selected_id) if selected_id is not None else None
    return render(request, "queue/page.html", {
        **_now_context(overview),
        "overview": overview,
        "book_queue": book_queue,
        "selected_id": selected_id,
        "preselect": set(book_queue.ready_ids(PRESELECT)) if book_queue else set(),
        "notice": notice,
        "view": view,
        "history": queue_overview.read_history(web.db_path, selected_id) if selected_id is not None else (),
        "offline": batch_runner.offline_default(),
    })


@router.get("/api/v1/queue/now", response_class=HTMLResponse)
def now_strip(request: Request) -> HTMLResponse:
    return render(request, "queue/_now.html", _now_context(queue_overview.read_now(_web(request).db_path)))


def _selected(request: Request, web: WebRepo, book: int, chapter: list[int], action: str) -> list:
    book_queue = queue_overview.read_book_queue(web.db_path, book)
    if book_queue is None:
        raise WebError(404, _t(request, "queue.book_not_found"))
    wanted = set(chapter)
    rows = [c for c in book_queue.chapters if c.chapter_db_id in wanted and c.action == action]
    if not rows:
        raise WebError(422, _t(request, f"queue.nothing_to_{action}"))
    return rows


@router.post("/api/v1/queue/enqueue")
def enqueue(request: Request, book: Annotated[int, Form()], chapter: Annotated[list[int], Form()] = []) -> Response:  # noqa: B006
    web = _web(request)
    rolled_back = synthesis_service.run_redo_detection(web.db_path)
    rows = _selected(request, web, book, chapter, "enqueue")
    result = synthesis_service.enqueue(web.db_path, [r.chapter_db_id for r in rows])
    if not result.ok:
        logger.info("webui_action: feature=queue action=enqueue book=%d result=refused error=%s", book, result.error)
        raise WebError(409, result.error or _t(request, "queue.enqueue_failed"))
    ids = result.batch_ids
    logger.info("webui_action: feature=queue action=enqueue book=%d chapters=%d batches=%s redo_rolled_back=%d",
                book, len(rows), ids, rolled_back)
    return _back(book, f"queued:{len(ids)}:{ids[0]}:{ids[-1]}")


def _set_paused(request: Request, book: int, chapter: list[int], paused: bool) -> Response:
    web = _web(request)
    action = "pause" if paused else "resume"
    rows = _selected(request, web, book, chapter, action)
    batch_ids = sorted({r.batch_id for r in rows})
    result = synthesis_service.set_pause_state(
        web.db_path, pause_batch_ids=batch_ids if paused else None, resume_batch_ids=None if paused else batch_ids)
    if not result.ok:
        raise WebError(422, result.error or _t(request, "queue.pause_failed"))
    logger.info("webui_action: feature=queue action=%s book=%d batches=%s flipped=%d", action, book, batch_ids, result.flipped)
    return _back(book, f"{action}d:{result.flipped}")


@router.post("/api/v1/queue/pause")
def pause(request: Request, book: Annotated[int, Form()], chapter: Annotated[list[int], Form()] = []) -> Response:  # noqa: B006
    return _set_paused(request, book, chapter, paused=True)


@router.post("/api/v1/queue/resume")
def resume(request: Request, book: Annotated[int, Form()], chapter: Annotated[list[int], Form()] = []) -> Response:  # noqa: B006
    return _set_paused(request, book, chapter, paused=False)


@router.post("/api/v1/queue/start")
def start(
    request: Request, book: Annotated[Optional[int], Form()] = None, offline: Annotated[Optional[str], Form()] = None,
) -> Response:
    offline_on = bool(offline) or batch_runner.offline_default().locked
    result = batch_runner.start_runner(_web(request).db_path, offline=offline_on)
    logger.info("webui_action: feature=queue action=start result=%s pid=%s reason=%s offline=%s",
                "started" if result.started else "refused", result.pid, result.reason, offline_on)
    if not result.started:
        raise WebError(409, _t(request, f"queue.start_refused_{result.reason}", pid=result.pid))
    return _back(book, f"started:{result.pid}")
