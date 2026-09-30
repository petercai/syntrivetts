from __future__ import annotations

from typing import Optional

from mcp_types import ToolAnnotations

from syntrive.mcp.envelope import fail
from syntrive.mcp.state import McpState
from syntrive.services import book_catalog
from syntrive.workflow.engine import WorkflowStep

READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)

STEP_NAMES = tuple(s.value for s in WorkflowStep if s != WorkflowStep.DONE)


def parse_step(value: str) -> WorkflowStep:
    try:
        step = WorkflowStep(value)
    except ValueError:
        raise fail("invalid", f"Unknown step '{value}'. Steps: {', '.join(STEP_NAMES)}.") from None
    if step == WorkflowStep.DONE:
        raise fail("invalid", "'done' is a state, not a step.")
    return step


def require_row(state: McpState, job_id: int) -> book_catalog.BookRow:
    row = book_catalog.get_book_row(state.current().db_path, job_id)
    if row is None:
        raise fail("not_found", f"Job {job_id} does not exist in this repo. book_list shows the jobs.")
    return row


def pipeline_brief(row: book_catalog.BookRow) -> dict:
    view = row.pipeline
    current = view.get(view.current) if view.current is not None else None
    return {
        "finished": view.finished,
        "current_step": view.current.value if view.current is not None else None,
        "current_number": current.number if current is not None else None,
        "current_mode": current.mode.value if current is not None else None,
    }


def book_brief(row: book_catalog.BookRow) -> dict:
    return {
        "job_id": row.job_id, "book_id": row.book_id, "title": row.title, "author": row.author,
        "language": row.language, "status": row.status, "archived": row.archived, "tab": row.tab,
        "chapters": row.chapter_count, "audio_chapters": row.audio_chapter_count,
        "updated_at": row.updated_at, "process_dir": row.process_dir, **pipeline_brief(row),
    }


def optional_text(value: Optional[str]) -> Optional[str]:
    return value.strip() if value is not None and value.strip() else None
