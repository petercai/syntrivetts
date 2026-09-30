from __future__ import annotations

import logging
import time
from typing import Literal, Optional

import anyio
from mcp.server.mcpserver import Context, MCPServer

from syntrive.mcp.envelope import Envelope, HOLDER_KIND, fail, ok, translating
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import READ, WRITE, STEP_NAMES, pipeline_brief, require_row
from syntrive.services import step_decisions as decisions
from syntrive.services.pipeline_service import (
    OnExisting,
    StepOptions,
    StepRun,
    check_runnable,
    existing_output,
    load_job,
)

logger = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 50.0
_POLL_SECONDS = 0.5


def _run_view(run: Optional[StepRun]) -> Optional[dict]:
    if run is None:
        return None
    result = run.result
    return {
        "job_id": run.job_id, "step": run.step.value, "running": run.running, "seconds": round(run.seconds, 1),
        "disposition": result.disposition.value if result else None,
        "error": result.error if result else None,
        "artifacts": result.artifacts if result else {},
        "notes": result.notes if result else None,
    }


async def _wait(state: McpState, job_id: int, wait_seconds: float, ctx: Optional[Context]) -> Optional[StepRun]:
    repo = state.current()
    limit = max(0.0, min(float(wait_seconds or 0), MAX_WAIT_SECONDS))
    started = time.monotonic()
    run = state.runs.get(repo.db_path, job_id)
    while run is not None and run.running and time.monotonic() - started < limit:
        await anyio.sleep(_POLL_SECONDS)
        run = state.runs.get(repo.db_path, job_id)
        if ctx is not None and run is not None:
            elapsed = time.monotonic() - started
            await ctx.report_progress(elapsed, limit, message=f"{run.step.value}: running {run.seconds:.0f} s")
    return run


def register(server: MCPServer, state: McpState) -> None:

    @server.tool(annotations=READ)
    def pipeline_status(job_id: int) -> Envelope:
        """The 9 steps of one book with their state (done / current / later) and mode (run = any client can run it,
        tui = needs a decision taken elsewhere, queue = synthesis via the queue), plus the step run in this process."""
        row = require_row(state, job_id)
        steps = [{"step": s.step.value, "number": s.number, "state": s.state.value, "mode": s.mode.value}
                 for s in row.pipeline.steps]
        run = state.runs.get(state.current().db_path, job_id)
        return ok({**pipeline_brief(row), "title": row.title, "steps": steps, "run": _run_view(run)},
                  evidence={"job_id": job_id})

    @server.tool(annotations=WRITE, description=(
        "Run a book's current step (or the named one, which must be current). Returns at once with the run, or waits "
        f"up to wait_seconds (max {MAX_WAIT_SECONDS:.0f}) sending progress; poll pipeline_run_status for longer runs. "
        "If the step already has output, on_existing='ask' (default) returns needs_confirm with a description instead "
        "of running: call again with 'overwrite' (run again) or 'keep' (keep it and advance). "
        "transcript_text takes text_formats (subset of raw, tts_script) and text_primary; transcript_review takes "
        f"transcript_path (raw or tts_script) as its decision. Steps: {', '.join(STEP_NAMES)}."))
    async def pipeline_run_step(
        job_id: int, ctx: Context, step: Optional[str] = None,
        on_existing: Literal["ask", "overwrite", "keep"] = "ask",
        text_formats: Optional[list[str]] = None, text_primary: Optional[str] = None,
        transcript_path: Optional[str] = None, wait_seconds: float = 0,
    ) -> Envelope:
        from syntrive.mcp.tools._common import parse_step
        from syntrive.services.job_lease import find_blocking_lease

        repo = state.current()
        row = require_row(state, job_id)
        if step is None and row.pipeline.current is None:
            raise fail("refused", "This book has finished every step.")
        wstep = parse_step(step) if step else row.pipeline.current
        options = StepOptions(
            text_formats=frozenset(text_formats) if text_formats is not None else None,
            text_primary=text_primary or None, transcript_path=transcript_path or None,
        )

        def prepare() -> Optional[str]:
            with translating():
                check_runnable(load_job(repo.db_path, job_id), wstep, options)
                conflict = find_blocking_lease(repo.db_path, [job_id])
                if conflict is not None:
                    raise conflict
            return existing_output(repo.db_path, job_id, wstep) if on_existing == "ask" else None

        existing = await anyio.to_thread.run_sync(prepare)
        if existing is not None:
            logger.info("mcp_action: tool=pipeline_run_step job_id=%d step=%s result=needs_confirm", job_id, wstep.value)
            return ok({"started": False, "needs_confirm": True, "step": wstep.value, "existing": existing},
                      evidence={"job_id": job_id},
                      warnings=["The step already has output. Call again with on_existing='overwrite' or 'keep'."])
        with translating():
            state.runs.start(repo.db_path, job_id, wstep, on_existing=OnExisting(on_existing),
                             holder_kind=HOLDER_KIND, options=options)
        logger.info("mcp_action: tool=pipeline_run_step job_id=%d step=%s on_existing=%s wait=%s",
                    job_id, wstep.value, on_existing, wait_seconds)
        run = await _wait(state, job_id, wait_seconds, ctx)
        data = {"started": True, "run": _run_view(run)}
        if run is not None and not run.running:
            data["pipeline"] = pipeline_brief(require_row(state, job_id))
        return ok(data, evidence={"job_id": job_id, "step": wstep.value})

    @server.tool(annotations=READ)
    async def pipeline_run_status(job_id: int, ctx: Context, wait_seconds: float = 0) -> Envelope:
        """The step run of this book in this process (running / finished with disposition done, kept, needs_confirm,
        refused or failed, its error and artifacts). wait_seconds (max 50) waits for it to finish, with progress."""
        run = await _wait(state, job_id, wait_seconds, ctx)
        if run is None:
            return ok({"run": None, **pipeline_brief(require_row(state, job_id))}, evidence={"job_id": job_id},
                      warnings=["No step has been run for this book since the server started."])
        return ok({"run": _run_view(run), **pipeline_brief(require_row(state, job_id))}, evidence={"job_id": job_id})

    @server.tool(annotations=READ)
    def pipeline_merge_settings_get(job_id: int) -> Envelope:
        """Step 3 decision: merge mode (override or auto-detected), per-volume chapter numbering, and the chapters with
        exclusion state, heading level, size and character count (a chapter far below its neighbours is often junk)."""
        repo = state.current()
        with translating():
            merge = decisions.read_merge_settings(repo.db_path, job_id)
            selection = decisions.read_chapter_selection(repo.db_path, job_id)
        return ok({"merge": merge, "selection": selection},
                  evidence={"job_id": job_id, "chapters": len(selection.chapters), "excluded": selection.excluded_count})

    @server.tool(annotations=WRITE)
    def pipeline_merge_settings_set(job_id: int, mode: Optional[str] = None, reset_per_volume: Optional[bool] = None) -> Envelope:
        """Change the merge decision: mode = one of merge_settings choices, or 'auto' to go back to auto-detection;
        reset_per_volume restarts chapter numbers in each volume. Omit a field to keep it. Takes the job lease."""
        repo = state.current()
        override = decisions.UNCHANGED if mode is None else (None if mode == "auto" else mode)
        per_volume = decisions.UNCHANGED if reset_per_volume is None else reset_per_volume
        with translating():
            result = decisions.save_merge_settings(repo.db_path, job_id, override=override,
                                                   reset_per_volume=per_volume, holder_kind=HOLDER_KIND)
        return _decided("merge_settings", job_id, result)

    @server.tool(annotations=READ)
    def pipeline_chapter_text(job_id: int, chapter_id: str) -> Envelope:
        """The start of one merged chapter's text (to judge whether to exclude it)."""
        repo = state.current()
        with translating():
            text, truncated = decisions.read_chapter_text(repo.db_path, job_id, chapter_id)
        return ok({"chapter_id": chapter_id, "text": text, "truncated": truncated}, evidence={"job_id": job_id})

    @server.tool(annotations=WRITE)
    def pipeline_chapters_set_exclusions(job_id: int, excluded: list[str]) -> Envelope:
        """Set which chapters are excluded from the audiobook: the full list of excluded chapter ids (every other
        chapter is included). Takes the job lease."""
        repo = state.current()
        with translating():
            result = decisions.save_exclusions(repo.db_path, job_id, set(excluded), holder_kind=HOLDER_KIND)
        return _decided("exclusions", job_id, result)

    @server.tool(annotations=READ)
    def pipeline_clean_rules_get(job_id: int) -> Envelope:
        """Step 4 decision: the cleaning rules (enabled, default, order) and, after cleaning, how much each chapter
        lost (a high deletion ratio or threshold_blocked means a rule removed too much)."""
        repo = state.current()
        with translating():
            rules = decisions.read_cleaning_rules(repo.db_path, job_id)
            results = sorted(decisions.read_clean_results(repo.db_path, job_id), key=lambda r: -(r.deletion_ratio or 0))
        return ok({"rules": rules, "results": results},
                  evidence={"job_id": job_id, "raw_chars": sum(r.raw_chars or 0 for r in results),
                            "clean_chars": sum(r.cleaned_chars or 0 for r in results)})

    @server.tool(annotations=WRITE)
    def pipeline_clean_rules_set(job_id: int, enabled: dict[str, bool]) -> Envelope:
        """Turn cleaning rules on / off by name ({"Footnote Paragraph Removal": false}); rules not named keep their
        state. Takes the job lease. Re-run step 4 afterwards (book_reset_to_step + pipeline_run_step) to apply."""
        repo = state.current()
        with translating():
            current = {r.name: r.enabled for r in decisions.read_cleaning_rules(repo.db_path, job_id)}
            unknown = sorted(set(enabled) - set(current))
            if unknown:
                raise fail("invalid", f"Unknown rules {unknown}. Rules: {sorted(current)}")
            result = decisions.save_cleaning_rules(repo.db_path, job_id, {**current, **enabled}, holder_kind=HOLDER_KIND)
        return _decided("cleaning_rules", job_id, result)

    @server.tool(annotations=READ)
    def pipeline_transcripts_get(job_id: int, fmt: Optional[str] = None, file: Optional[str] = None) -> Envelope:
        """Step 6 review: without fmt / file, the transcript formats (raw, tts_script) with their files and line counts;
        with both, the text of that file (TTS markers such as ‡break‡ included)."""
        repo = state.current()
        with translating():
            if fmt and file:
                text, truncated = decisions.read_transcript_text(repo.db_path, job_id, fmt, file)
                return ok({"fmt": fmt, "file": file, "text": text, "truncated": truncated}, evidence={"job_id": job_id})
            review = decisions.read_transcript_review(repo.db_path, job_id)
        return ok(review, evidence={"job_id": job_id, "selected": review.selected})


def _decided(what: str, job_id: int, result: decisions.DecisionResult) -> Envelope:
    logger.info("mcp_action: tool=decision what=%s job_id=%d changed=%s rerun_from=%s",
                what, job_id, result.changed, result.rerun_from.value if result.rerun_from else None)
    warnings = ([f"Output of '{result.rerun_from.value}' is now stale: book_reset_to_step to it, then pipeline_run_step."]
                if result.rerun_from else [])
    return ok({"changed": result.changed, "rerun_from": result.rerun_from}, evidence={"job_id": job_id}, warnings=warnings)
