from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from syntrive.services.pipeline_service import PipelineView, build_pipeline_view

logger = logging.getLogger(__name__)

TABS = ("active", "done", "archived")


@dataclass(frozen=True)
class BookRow:
    job_id: int
    book_id: int
    title: str
    author: str
    language: str
    process_dir: Path
    cover_path: Optional[Path]
    status: str
    archived: bool
    chapter_count: int
    audio_chapter_count: int
    updated_at: Optional[datetime]
    pipeline: PipelineView

    @property
    def tab(self) -> str:
        if self.archived:
            return "archived"
        return "done" if self.pipeline.finished else "active"


@dataclass(frozen=True)
class ChapterRow:
    sequence: str
    number: str
    title: str
    volume: str
    lines: Optional[int]
    transcript_path: Optional[str]
    synthesis_status: str
    has_audio: bool


@dataclass(frozen=True)
class ActivityRow:
    at: Optional[datetime]
    step: str
    action: str
    notes: str


@dataclass(frozen=True)
class DeletePreview:
    job_id: int
    title: str
    job_count: int
    folders: tuple[Path, ...]
    size_bytes: int


def filter_rows(rows: Iterable[BookRow], *, tab: str = "active", query: str = "") -> tuple[BookRow, ...]:
    needle = query.strip().casefold()
    return tuple(
        r for r in rows
        if r.tab == tab and (not needle or needle in r.title.casefold() or needle in r.author.casefold())
    )


def tab_counts(rows: Iterable[BookRow]) -> dict[str, int]:
    counts = dict.fromkeys(TABS, 0)
    for r in rows:
        counts[r.tab] += 1
    return counts


def _folder_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _cover_abs(process_dir: Path, book_cover: Optional[str], job_cover: Optional[str], db_path: Path) -> Optional[Path]:
    from syntrive.db.path_utils import resolve_abs

    if book_cover:
        return process_dir / book_cover
    if job_cover:
        return resolve_abs(db_path, job_cover)
    return None


def _to_row(job, db_path: Path, chapters: int, audio: int) -> BookRow:
    from syntrive.db.path_utils import resolve_abs

    book = job.book
    process_dir = resolve_abs(db_path, job.process_dir)
    return BookRow(
        job_id=job.id,
        book_id=job.book_id,
        title=(book.title if book else None) or process_dir.name,
        author=(book.author if book else None) or "",
        language=(book.language if book else None) or "",
        process_dir=process_dir,
        cover_path=_cover_abs(process_dir, book.cover if book else None, job.cover_path, db_path),
        status=job.status or "pending",
        archived=job.archived_at is not None,
        chapter_count=chapters,
        audio_chapter_count=audio,
        updated_at=job.updated_at or job.created_at,
        pipeline=build_pipeline_view(job.current_step, job.status),
    )


def _query_rows(db_path: Path, job_id: Optional[int] = None) -> tuple[BookRow, ...]:
    from sqlalchemy import func
    from sqlalchemy.orm import joinedload

    from syntrive.db.models import Job, TranscriptChapter
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        counts_q = db.query(
            TranscriptChapter.job_id,
            func.count(TranscriptChapter.id),
            func.count(TranscriptChapter.sequence_number),
            func.count(TranscriptChapter.chapter_audio_path),
        ).group_by(TranscriptChapter.job_id)
        jobs_q = db.query(Job).options(joinedload(Job.book)).order_by(Job.updated_at.desc(), Job.id.desc())
        if job_id is not None:
            counts_q = counts_q.filter(TranscriptChapter.job_id == job_id)
            jobs_q = jobs_q.filter(Job.id == job_id)
        counts = {jid: (sequenced or total, audio) for jid, total, sequenced, audio in counts_q}
        return tuple(_to_row(job, db_path, *counts.get(job.id, (0, 0))) for job in jobs_q)


def list_book_rows(db_path: Path) -> tuple[BookRow, ...]:
    rows = _query_rows(db_path)
    logger.debug("book_catalog_listed: db=%s rows=%d", db_path, len(rows))
    return rows


def get_book_row(db_path: Path, job_id: int) -> Optional[BookRow]:
    rows = _query_rows(db_path, job_id)
    return rows[0] if rows else None


def list_chapters(db_path: Path, job_id: int) -> tuple[ChapterRow, ...]:
    from syntrive.db.models import TranscriptChapter
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        rows = (
            db.query(TranscriptChapter)
            .filter(TranscriptChapter.job_id == job_id)
            .order_by(
                TranscriptChapter.sequence_number.is_(None),
                TranscriptChapter.sequence_number,
                TranscriptChapter.chapter_id,
            )
            .all()
        )
        return tuple(
            ChapterRow(
                sequence=r.sequence_number or "",
                number=r.chapter_number or "",
                title=r.chapter_name or r.title or r.chapter_id,
                volume=r.volume or "",
                lines=r.transcript_lines,
                transcript_path=r.transcript_path,
                synthesis_status=r.synthesis_status or "pending",
                has_audio=bool(r.chapter_audio_path),
            )
            for r in rows
        )


def list_activity(db_path: Path, job_id: int, limit: int = 50) -> tuple[ActivityRow, ...]:
    from syntrive.db.models import WorkflowStepEvent
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        rows = (
            db.query(WorkflowStepEvent)
            .filter(WorkflowStepEvent.job_id == job_id)
            .order_by(WorkflowStepEvent.created_at.desc(), WorkflowStepEvent.id.desc())
            .limit(limit)
            .all()
        )
        return tuple(ActivityRow(r.created_at, r.step_name, r.action, r.user_notes or "") for r in rows)


@dataclass(frozen=True)
class BookImageChoice:
    name: str
    rel_path: str
    path: Path
    image_type: str
    exists: bool


@dataclass(frozen=True)
class BookMetadata:
    job_id: int
    book_id: int
    title: str
    author: str
    language: str
    cover_rel: Optional[str]
    images: tuple[BookImageChoice, ...]


def read_book_metadata(db_path: Path, job_id: int) -> Optional[BookMetadata]:
    from syntrive.db.models import BookImage, Job
    from syntrive.db.path_utils import resolve_abs
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        job = db.get(Job, job_id)
        if job is None or job.book is None:
            return None
        book = job.book
        process_dir = resolve_abs(db_path, job.process_dir)
        images = tuple(
            BookImageChoice(
                name=img.name or Path(img.path).name, rel_path=img.path, path=process_dir / img.path,
                image_type=img.image_type or "internal", exists=(process_dir / img.path).is_file(),
            )
            for img in db.query(BookImage).filter(BookImage.book_id == book.id).order_by(BookImage.name)
        )
        images = tuple(sorted(images, key=lambda i: i.rel_path != book.cover))
        return BookMetadata(
            job_id=job_id, book_id=book.id, title=book.title or "", author=book.author or "",
            language=book.language or "", cover_rel=book.cover, images=images,
        )


def delete_preview(db_path: Path, job_id: int) -> Optional[DeletePreview]:
    from syntrive.db.models import Job
    from syntrive.db.path_utils import resolve_abs
    from syntrive.db.session import get_db_session

    row = get_book_row(db_path, job_id)
    if row is None:
        return None
    with get_db_session(db_path) as db:
        dirs = tuple(resolve_abs(db_path, j.process_dir) for j in db.query(Job).filter(Job.book_id == row.book_id))
    return DeletePreview(
        job_id=job_id, title=row.title, job_count=len(dirs), folders=dirs,
        size_bytes=sum(_folder_size(d) for d in dirs),
    )
