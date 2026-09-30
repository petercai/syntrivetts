from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from syntrive.db.models import Book, Job, SynthesisBatch, TranscriptChapter

logger = logging.getLogger(__name__)

ACTIVE_CHAPTER_STATES = ("queued", "synthesizing", "combining")


def find_earliest_active_batch(db: Session) -> Optional[SynthesisBatch]:
    batch = (
        db.query(SynthesisBatch)
        .join(TranscriptChapter, TranscriptChapter.synthesis_batch_id == SynthesisBatch.synthesis_batch_id)
        .filter(TranscriptChapter.synthesis_status != "done")
        .filter(SynthesisBatch.paused_at.is_(None))
        .order_by(SynthesisBatch.synthesis_batch_id.asc())
        .first()
    )
    logger.debug(
        "find_earliest_active_batch: found=%s",
        batch.synthesis_batch_id if batch else None,
    )
    return batch


def batch_chapters(db: Session, batch_id: int) -> List[TranscriptChapter]:
    return (
        db.query(TranscriptChapter)
        .filter(TranscriptChapter.synthesis_batch_id == batch_id)
        .order_by(TranscriptChapter.job_id.asc(), TranscriptChapter.sequence_number.asc())
        .all()
    )


@dataclass(frozen=True)
class ChapterProgress:
    book_name: str
    book_seq: int
    book_total: int
    chapter_seq: int
    chapter_total: int


def build_progress_plan(db: Session) -> dict:
    rows = (
        db.query(TranscriptChapter, Job, Book)
        .join(Job, TranscriptChapter.job_id == Job.id)
        .join(Book, Job.book_id == Book.id)
        .filter(TranscriptChapter.synthesis_status.in_(ACTIVE_CHAPTER_STATES))
        .order_by(Job.id.asc(), TranscriptChapter.sequence_number.asc())
        .all()
    )

    book_order: List[int] = []
    chapters_by_book: dict = {}
    book_name_by_id: dict = {}
    for chapter, job, book in rows:
        if job.book_id not in chapters_by_book:
            book_order.append(job.book_id)
            chapters_by_book[job.book_id] = []
            book_name_by_id[job.book_id] = book.title
        chapters_by_book[job.book_id].append((job.id, chapter.chapter_id))

    book_total = len(book_order)
    plan: dict = {}
    for book_seq, book_id in enumerate(book_order, start=1):
        chapters = chapters_by_book[book_id]
        chapter_total = len(chapters)
        for chapter_seq, (job_id, chapter_id) in enumerate(chapters, start=1):
            plan[(job_id, chapter_id)] = ChapterProgress(
                book_name=book_name_by_id[book_id], book_seq=book_seq, book_total=book_total,
                chapter_seq=chapter_seq, chapter_total=chapter_total,
            )
    return plan


@dataclass(frozen=True)
class ChapterStatusRow:
    job_id: int
    book_title: str
    chapter_id: str
    chapter_name: Optional[str]
    sequence_number: str
    synthesis_status: str
    chapter_audio_path: Optional[str]
    chapter_audio_seconds: Optional[float]


@dataclass(frozen=True)
class BatchStatusReport:
    batch_id: int
    synth_error: Optional[str]
    failed_line_index: Optional[int]
    synth_started_at: Optional[str]
    synth_finished_at: Optional[str]
    paused_at: Optional[str] = None
    chapters: List[ChapterStatusRow] = field(default_factory=list)

    @property
    def is_done(self) -> bool:
        return all(c.synthesis_status == "done" for c in self.chapters)

    @property
    def is_stuck(self) -> bool:
        return bool(self.synth_error) and not self.is_done


def _book_title_by_job(db: Session, job_ids: set) -> dict:
    if not job_ids:
        return {}
    rows = (
        db.query(Job.id, Book.title)
        .join(Book, Job.book_id == Book.id)
        .filter(Job.id.in_(job_ids))
        .all()
    )
    return {job_id: title for job_id, title in rows}


def batch_status(db: Session, batch_id: int) -> Optional[BatchStatusReport]:
    batch = db.query(SynthesisBatch).filter_by(synthesis_batch_id=batch_id).first()
    if batch is None:
        return None
    chapters = batch_chapters(db, batch_id)
    title_by_job = _book_title_by_job(db, {c.job_id for c in chapters})
    return BatchStatusReport(
        batch_id=batch_id,
        synth_error=batch.synth_error,
        failed_line_index=batch.failed_line_index,
        synth_started_at=batch.synth_started_at.isoformat() if batch.synth_started_at else None,
        synth_finished_at=batch.synth_finished_at.isoformat() if batch.synth_finished_at else None,
        paused_at=batch.paused_at.isoformat() if batch.paused_at else None,
        chapters=[
            ChapterStatusRow(
                job_id=c.job_id,
                book_title=title_by_job.get(c.job_id, f"(book for job {c.job_id})"),
                chapter_id=c.chapter_id,
                chapter_name=c.chapter_name,
                sequence_number=c.sequence_number or "",
                synthesis_status=c.synthesis_status,
                chapter_audio_path=c.chapter_audio_path,
                chapter_audio_seconds=c.chapter_audio_seconds,
            )
            for c in chapters
        ],
    )


def all_batch_ids(db: Session) -> List[int]:
    rows = db.query(SynthesisBatch.synthesis_batch_id).order_by(SynthesisBatch.synthesis_batch_id.asc()).all()
    return [row[0] for row in rows]


def active_batch_ids(db: Session) -> List[int]:
    rows = (
        db.query(SynthesisBatch.synthesis_batch_id)
        .join(TranscriptChapter, TranscriptChapter.synthesis_batch_id == SynthesisBatch.synthesis_batch_id)
        .filter(TranscriptChapter.synthesis_status != "done")
        .filter(SynthesisBatch.paused_at.is_(None))
        .distinct()
        .order_by(SynthesisBatch.synthesis_batch_id.asc())
        .all()
    )
    return [row[0] for row in rows]


@dataclass(frozen=True)
class SchedulableChapter:
    chapter_db_id: int
    chapter_id: str
    sequence_number: str
    chapter_name: Optional[str]
    book_title: str


@dataclass(frozen=True)
class SchedulableBook:
    job_id: int
    book_title: str
    chapters: List[SchedulableChapter] = field(default_factory=list)


def chapter_readiness(chapter: TranscriptChapter, process_dir: Path) -> Optional[str]:
    if chapter.synthesis_status == "done":
        return "done"
    if chapter.synthesis_status != "pending" and chapter.synthesis_batch_id is not None:
        logger.debug(
            "chapter_not_schedulable: job_id=%s chapter_id=%s reason=already_in_batch status=%s batch_id=%s",
            chapter.job_id, chapter.chapter_id, chapter.synthesis_status, chapter.synthesis_batch_id,
        )
        return "in_batch"
    if not chapter.transcript_path or chapter.transcript_lines is None:
        return "no_transcript"
    abs_path = process_dir / chapter.transcript_path
    if not abs_path.is_file():
        logger.debug(
            "chapter_not_schedulable: job_id=%s chapter_id=%s reason=missing_file path=%s",
            chapter.job_id, chapter.chapter_id, abs_path,
        )
        return "missing_file"
    try:
        actual_lines = sum(1 for ln in abs_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    except OSError as exc:
        logger.warning(
            "chapter_not_schedulable: job_id=%s chapter_id=%s reason=read_failed error=%s",
            chapter.job_id, chapter.chapter_id, exc,
        )
        return "unreadable"
    if actual_lines != chapter.transcript_lines:
        logger.debug(
            "chapter_not_schedulable: job_id=%s chapter_id=%s reason=line_count_mismatch db=%d actual=%d",
            chapter.job_id, chapter.chapter_id, chapter.transcript_lines, actual_lines,
        )
        return "line_mismatch"
    return None


def _chapter_is_schedulable(chapter: TranscriptChapter, process_dir: Path) -> bool:
    return chapter_readiness(chapter, process_dir) is None


def book_readiness(job: Job, chapters: List[TranscriptChapter]) -> Optional[str]:
    if job.archived_at is not None:
        return "archived"
    if not chapters:
        return "no_chapters"
    if not _sequence_numbers_are_gap_free(chapters):
        return "sequence_gap"
    return None


def _sequence_numbers_are_gap_free(chapters: List[TranscriptChapter]) -> bool:
    numbered = sorted(int(c.sequence_number) for c in chapters if c.sequence_number)
    return numbered == list(range(1, len(numbered) + 1))


def schedulable_books(db: Session, repo_dir: Path) -> List[SchedulableBook]:
    result: List[SchedulableBook] = []
    for job in db.query(Job).order_by(Job.id.asc()).all():
        chapters = (
            db.query(TranscriptChapter)
            .filter(TranscriptChapter.job_id == job.id)
            .order_by(TranscriptChapter.sequence_number.asc())
            .all()
        )
        reason = book_readiness(job, chapters)
        if reason is not None:
            logger.debug("book_not_schedulable: job_id=%s reason=%s", job.id, reason)
            continue
        process_dir = repo_dir / job.process_dir
        schedulable = [c for c in chapters if _chapter_is_schedulable(c, process_dir)]
        if not schedulable:
            logger.debug(
                "book_not_schedulable: job_id=%s reason=no_chapter_passes_disk_crosscheck "
                "process_dir=%s total_chapters=%d",
                job.id, process_dir, len(chapters),
            )
            continue
        book = db.query(Book).filter_by(id=job.book_id).first()
        book_title = book.title if book else f"(book {job.book_id})"
        entries = [
            SchedulableChapter(
                chapter_db_id=c.id,
                chapter_id=c.chapter_id,
                sequence_number=c.sequence_number or "",
                chapter_name=c.chapter_name,
                book_title=book_title,
            )
            for c in schedulable
        ]
        logger.debug(
            "schedulable_book: job_id=%s title=%r chapter_db_ids=%s",
            job.id, book_title, [e.chapter_db_id for e in entries],
        )
        result.append(
            SchedulableBook(job_id=job.id, book_title=book_title, chapters=entries)
        )
    logger.info(
        "schedulable_books: books=%d total_chapters=%d",
        len(result), sum(len(b.chapters) for b in result),
    )
    return result


@dataclass(frozen=True)
class PausableChapter:
    chapter_db_id: int
    chapter_id: str
    sequence_number: str
    chapter_name: Optional[str]
    synthesis_status: str
    synthesis_batch_id: int
    paused: bool
    book_title: str


@dataclass(frozen=True)
class PausableBook:
    job_id: int
    book_title: str
    chapters: List[PausableChapter] = field(default_factory=list)

    @property
    def paused_count(self) -> int:
        return sum(1 for c in self.chapters if c.paused)


def pausable_books(db: Session) -> List[PausableBook]:
    rows = (
        db.query(TranscriptChapter, SynthesisBatch, Job, Book)
        .join(SynthesisBatch, TranscriptChapter.synthesis_batch_id == SynthesisBatch.synthesis_batch_id)
        .join(Job, TranscriptChapter.job_id == Job.id)
        .join(Book, Job.book_id == Book.id)
        .filter(TranscriptChapter.synthesis_status != "done")
        .order_by(Job.id.asc(), TranscriptChapter.sequence_number.asc())
        .all()
    )

    by_job: dict = {}
    order: List[int] = []
    for chapter, batch, job, book in rows:
        if job.id not in by_job:
            order.append(job.id)
            by_job[job.id] = (book.title, [])
        by_job[job.id][1].append(
            PausableChapter(
                chapter_db_id=chapter.id,
                chapter_id=chapter.chapter_id,
                sequence_number=chapter.sequence_number or "",
                chapter_name=chapter.chapter_name,
                synthesis_status=chapter.synthesis_status,
                synthesis_batch_id=batch.synthesis_batch_id,
                paused=batch.paused_at is not None,
                book_title=book.title,
            )
        )

    result = [
        PausableBook(job_id=job_id, book_title=by_job[job_id][0], chapters=by_job[job_id][1])
        for job_id in order
    ]
    logger.info(
        "pausable_books: books=%d chapters=%d paused=%d",
        len(result),
        sum(len(b.chapters) for b in result),
        sum(b.paused_count for b in result),
    )
    return result
