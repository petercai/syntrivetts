from __future__ import annotations

import logging
from typing import Optional

from mcp.server.mcpserver import MCPServer

from syntrive.mcp.envelope import Envelope, fail, ok, translating
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import READ, WRITE
from syntrive.services import batch_runner, queue_overview, synthesis_service

logger = logging.getLogger(__name__)

DEFAULT_ENQUEUE = 5


def _book_queue(state: McpState, job_id: int) -> queue_overview.BookQueue:
    book = queue_overview.read_book_queue(state.current().db_path, job_id)
    if book is None:
        raise fail("not_found", f"Job {job_id} does not exist in this repo.")
    return book


def _chapter_view(c: queue_overview.QueueChapter) -> dict:
    return {"chapter_id": c.chapter_db_id, "sequence": c.sequence, "title": c.title, "lines": c.lines,
            "status": c.status, "batch_id": c.batch_id, "paused": c.paused, "ready": c.ready, "action": c.action,
            "reason": c.reason, "audio_seconds": c.audio_seconds, "error": c.error, "failed_line": c.failed_line}


def register(server: MCPServer, state: McpState) -> None:

    @server.tool(name="queue_overview", annotations=READ)
    def queue_overview_tool(recent: int = 10) -> Envelope:
        """The whole queue: is a runner working (and on what, with live progress), queued / paused batches in drain
        order, the latest finished chapters, and per-book progress (done / queued / ready)."""
        repo = state.current()
        with translating():
            view = queue_overview.read_queue(repo.db_path, recent=recent)
        return ok({"running": view.running, "runner": view.runner, "progress": view.progress, "live": view.live,
                   "queue": view.queue, "recent": view.recent, "books": view.books},
                  evidence={"queued": len(view.queue), "runnable": view.runnable_count, "stuck": len(view.stuck)})

    @server.tool(annotations=READ)
    def queue_book(job_id: int) -> Envelope:
        """One book's chapters with their synthesis state: ready (can be queued), queued / synthesizing / done / failed,
        paused, and why a chapter cannot be queued yet (reason)."""
        book = _book_queue(state, job_id)
        return ok({"job_id": book.job_id, "title": book.title, "book_reason": book.book_reason,
                   "chapters": [_chapter_view(c) for c in book.chapters]},
                  evidence={"job_id": job_id, "ready": len(book.ready_ids())})

    @server.tool(annotations=READ)
    def queue_history(job_id: Optional[int] = None, limit: int = 50) -> Envelope:
        """Batches that started, newest first (one book's only with job_id): chapter, start, run time, audio length,
        outcome (done / failed / stopped / running) and the failed line + error."""
        with translating():
            rows = queue_overview.read_history(state.current().db_path, job_id, limit)
        return ok(rows, evidence={"rows": len(rows)})

    @server.tool(annotations=READ)
    def queue_now() -> Envelope:
        """Just the runner and its live progress (cheap; poll this while synthesis runs)."""
        with translating():
            now = queue_overview.read_now(state.current().db_path)
        return ok({"running": now.running, "runner": now.runner, "progress": now.progress, "live": now.live},
                  evidence={"running": now.running})

    @server.tool(annotations=WRITE)
    def queue_enqueue(job_id: int, chapter_ids: Optional[list[int]] = None, count: int = DEFAULT_ENQUEUE) -> Envelope:
        """Queue chapters of a book for synthesis (one batch per chapter). chapter_ids from queue_book (only ready ones
        are taken), or omit them to take the first `count` ready chapters. Runs redo detection first, like tts_task.
        Queued chapters are synthesized when a runner works: queue_start_runner."""
        repo = state.current()
        with translating():
            rolled_back = synthesis_service.run_redo_detection(repo.db_path)
            book = _book_queue(state, job_id)
            ready = set(book.ready_ids())
            wanted = [c for c in chapter_ids if c in ready] if chapter_ids is not None else list(book.ready_ids(count))
            if not wanted:
                raise fail("refused", "No ready chapter to queue in that selection (queue_book shows each chapter's reason).")
            result = synthesis_service.enqueue(repo.db_path, wanted)
        if not result.ok:
            raise fail("conflict", result.error or "Enqueue refused.")
        skipped = sorted(set(chapter_ids or ()) - set(wanted))
        logger.info("mcp_action: tool=queue_enqueue job_id=%d chapters=%d batches=%s redo_rolled_back=%d",
                    job_id, len(wanted), list(result.batch_ids), rolled_back)
        return ok({"queued_chapter_ids": wanted, "batch_ids": result.batch_ids, "skipped_not_ready": skipped},
                  evidence={"job_id": job_id, "batch_ids": result.batch_ids},
                  warnings=[f"{len(skipped)} chapter(s) were not ready and were skipped."] if skipped else [])

    def _pause(job_id: int, chapter_ids: list[int], paused: bool) -> Envelope:
        repo = state.current()
        action = "pause" if paused else "resume"
        with translating():
            book = _book_queue(state, job_id)
            wanted = set(chapter_ids)
            batch_ids = sorted({c.batch_id for c in book.chapters if c.chapter_db_id in wanted and c.action == action})
            if not batch_ids:
                raise fail("refused", f"None of those chapters can be {action}d (queue_book shows each chapter's action).")
            result = synthesis_service.set_pause_state(
                repo.db_path, pause_batch_ids=batch_ids if paused else None, resume_batch_ids=None if paused else batch_ids)
        if not result.ok:
            raise fail("failed", result.error or f"{action} failed")
        logger.info("mcp_action: tool=queue_%s job_id=%d batches=%s flipped=%d", action, job_id, batch_ids, result.flipped)
        return ok({"batch_ids": batch_ids, "flipped": result.flipped}, evidence={"job_id": job_id})

    @server.tool(annotations=WRITE)
    def queue_pause(job_id: int, chapter_ids: list[int]) -> Envelope:
        """Pause queued chapters of a book (the runner skips them until resumed)."""
        return _pause(job_id, chapter_ids, True)

    @server.tool(annotations=WRITE)
    def queue_resume(job_id: int, chapter_ids: list[int]) -> Envelope:
        """Resume paused chapters of a book."""
        return _pause(job_id, chapter_ids, False)

    @server.tool(annotations=WRITE)
    def queue_start_runner(offline: Optional[bool] = None) -> Envelope:
        """Start tts_batch in the background for this repo; it synthesizes queued chapters one at a time and keeps
        running after this call (follow it with queue_now). Refused while a runner is working. offline=true never
        downloads models; omitted = the project default (.env), forced when the server's environment is offline."""
        repo = state.current()
        default = batch_runner.offline_default()
        offline_on = default.locked or (default.checked if offline is None else offline)
        with translating():
            result = batch_runner.start_runner(repo.db_path, offline=offline_on)
        logger.info("mcp_action: tool=queue_start_runner started=%s pid=%s reason=%s offline=%s",
                    result.started, result.pid, result.reason, offline_on)
        if not result.started:
            raise fail("conflict", f"A runner is already working (pid {result.pid}, {result.reason}). Follow it with queue_now.")
        return ok({"started": True, "pid": result.pid, "offline": offline_on}, evidence={"pid": result.pid})
