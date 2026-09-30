from __future__ import annotations

import logging
from typing import Literal, Optional

from mcp.server.mcpserver import MCPServer

from syntrive.mcp.envelope import Envelope, HOLDER_KIND, destructive, fail, ok, resolve_path, translating
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import DESTRUCTIVE, READ, WRITE, STEP_NAMES, book_brief, optional_text, parse_step, require_row
from syntrive.services import book_catalog
from syntrive.services.job_service import JobService

logger = logging.getLogger(__name__)


def register(server: MCPServer, state: McpState) -> None:

    @server.tool(annotations=READ)
    def book_list(tab: Literal["all", "active", "done", "archived"] = "all", query: str = "") -> Envelope:
        """Books in this repo (one row per job): title, author, language, current step (of 9), chapters, archived.
        tab: active = in progress, done = finished, archived = read-only. query filters title / author."""
        repo = state.current()
        with translating():
            rows = book_catalog.list_book_rows(repo.db_path)
        shown = rows if tab == "all" else book_catalog.filter_rows(rows, tab=tab, query=query)
        if tab == "all" and query:
            needle = query.casefold()
            shown = tuple(r for r in rows if needle in r.title.casefold() or needle in (r.author or "").casefold())
        return ok([book_brief(r) for r in shown], evidence={"counts": book_catalog.tab_counts(rows), "shown": len(shown)})

    @server.tool(annotations=READ)
    def book_get(job_id: int, include_chapters: bool = True) -> Envelope:
        """One book: summary, all 9 steps with their state, metadata (cover candidates), chapters and recent activity."""
        repo = state.current()
        row = require_row(state, job_id)
        with translating():
            meta = book_catalog.read_book_metadata(repo.db_path, job_id)
            chapters = book_catalog.list_chapters(repo.db_path, job_id) if include_chapters else ()
            activity = book_catalog.list_activity(repo.db_path, job_id, limit=20)
        steps = [{"step": s.step.value, "number": s.number, "state": s.state.value, "mode": s.mode.value}
                 for s in row.pipeline.steps]
        return ok({
            **book_brief(row), "steps": steps,
            "cover": meta.cover_rel if meta else None,
            "cover_candidates": [i.rel_path for i in meta.images if i.exists] if meta else [],
            "chapters": chapters, "activity": activity,
        }, evidence={"job_id": job_id, "process_dir": row.process_dir})

    @server.tool(annotations=WRITE)
    def book_add(epub_path: str) -> Envelope:
        """Add a book from a local EPUB file: creates PROCESSING-<name>/ in this repo, imports it (step 1) and returns
        the new job. The EPUB is copied; the original is left where it is."""
        from syntrive.bootstrap import bootstrap

        repo = state.current()
        path = resolve_path(epub_path)
        if path.suffix.lower() != ".epub":
            raise fail("invalid", f"Only EPUB files can be added: {path.name}")
        if not path.is_file():
            raise fail("not_found", f"EPUB not found: {path}")
        try:
            job = bootstrap(repo.repo_dir, path)
        except Exception as exc:
            logger.error("mcp_action: tool=book_add path=%s result=failed", path, exc_info=True)
            raise fail("failed", f"Could not import {path.name}: {exc}") from exc
        row = require_row(state, job.id)
        logger.info("mcp_action: tool=book_add job_id=%d path=%s", job.id, path)
        return ok(book_brief(row), evidence={"job_id": job.id, "process_dir": row.process_dir})

    @server.tool(annotations=WRITE)
    def book_set_metadata(job_id: int, title: Optional[str] = None, author: Optional[str] = None,
                          language: Optional[str] = None, cover: Optional[str] = None) -> Envelope:
        """Change title / author / language / cover (omit a field to keep it). cover must be one of book_get's
        cover_candidates (a path like images/cover.jpg). The language change takes the job lease."""
        from syntrive.services.tts_settings import PARAMS_BY_FIELD

        repo = state.current()
        meta = book_catalog.read_book_metadata(repo.db_path, job_id)
        if meta is None:
            raise fail("not_found", f"Job {job_id} does not exist in this repo.")
        title, author, language, cover = (optional_text(v) for v in (title, author, language, cover))
        languages = PARAMS_BY_FIELD["language"].choices or ()
        if language and language not in languages:
            raise fail("invalid", f"language must be one of {', '.join(languages)}")
        if cover and cover not in {img.rel_path for img in meta.images}:
            raise fail("invalid", "cover must be one of book_get's cover_candidates")

        service = JobService(repo.db_path, holder_kind=HOLDER_KIND)
        changed = []
        with translating():
            if language and language != meta.language:
                service.update_book_language(job_id, language)
                changed.append("language")
            if title and title != meta.title:
                service.update_book_title(job_id, title)
                changed.append("title")
            if author is not None and author != meta.author:
                service.update_book_author(job_id, author)
                changed.append("author")
            if cover and cover != meta.cover_rel:
                service.update_book_cover(job_id, cover)
                changed.append("cover")
        logger.info("mcp_action: tool=book_set_metadata job_id=%d changed=%s", job_id, changed)
        return ok({"changed": changed, **book_brief(require_row(state, job_id))}, evidence={"job_id": job_id})

    def _archive(job_id: int, archived: bool) -> Envelope:
        repo = state.current()
        require_row(state, job_id)
        service = JobService(repo.db_path, holder_kind=HOLDER_KIND)
        with translating():
            result = service.archive_job(job_id) if archived else service.unarchive_job(job_id)
        if not result.ok:
            raise fail("failed", result.error or "archive change failed")
        logger.info("mcp_action: tool=book_%sarchive job_id=%d changed=%s", "" if archived else "un", job_id, result.changed)
        return ok({"archived": result.archived, "changed": result.changed}, evidence={"job_id": job_id})

    @server.tool(annotations=WRITE)
    def book_archive(job_id: int) -> Envelope:
        """Archive a book: it becomes read-only and leaves the active list (reversible with book_unarchive)."""
        return _archive(job_id, True)

    @server.tool(annotations=WRITE)
    def book_unarchive(job_id: int) -> Envelope:
        """Restore an archived book to the active list."""
        return _archive(job_id, False)

    @server.tool(annotations=DESTRUCTIVE, description=(
        "Restart a book from a step it has already passed: that step and every later one become 'not started' "
        f"(their outputs are regenerated when run again). Steps: {', '.join(STEP_NAMES)}. "
        "Destructive: the first call (dry_run=true) previews; repeat with dry_run=false + its confirm_token."))
    def book_reset_to_step(job_id: int, step: str, dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        from syntrive.services.job_lease import find_blocking_lease
        from syntrive.services.pipeline_service import StepState

        repo = state.current()
        row = require_row(state, job_id)
        wstep = parse_step(step)
        target = row.pipeline.get(wstep)
        if row.archived or target is None or target.state != StepState.DONE:
            raise fail("refused", f"'{wstep.value}' can be restarted only when it is already done (and the book is not archived).")
        reopened = [s.step.value for s in row.pipeline.steps if s.number >= target.number and s.state != StepState.LATER]
        preview = {"job_id": job_id, "title": row.title, "restart_from": wstep.value, "reopens": reopened,
                   "current_step": row.pipeline.current.value if row.pipeline.current else None}
        gate = destructive(state.tokens, "book_reset_to_step", {"job_id": job_id, "step": wstep.value},
                           dry_run=dry_run, confirm_token=confirm_token, preview=preview,
                           fingerprint=(preview["current_step"], reopened))
        if gate is not None:
            return gate
        running = state.runs.get(repo.db_path, job_id)
        if running is not None and running.running:
            raise fail("conflict", f"'{running.step.value}' is running for this book; wait for it (pipeline_run_status).")
        with translating():
            conflict = find_blocking_lease(repo.db_path, [job_id])
            if conflict is not None:
                raise conflict
            JobService(repo.db_path, holder_kind=HOLDER_KIND).reset_job_to_step(job_id, wstep)
        logger.info("mcp_action: tool=book_reset_to_step job_id=%d step=%s reopened=%s", job_id, wstep.value, reopened)
        return ok({**preview, **book_brief(require_row(state, job_id))}, evidence={"job_id": job_id, "reopened": reopened})
