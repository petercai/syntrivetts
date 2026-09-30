from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from syntrive.contract.manifest import (
    JOB_MANIFEST_FILENAME,
    REPO_MANIFEST_FILENAME,
    BookFacts,
    ChapterFacts,
    ImageFacts,
    JobFacts,
    JobSnapshot,
    PathIssue,
    RepoJobEntry,
    build_job_manifest,
    build_repo_manifest,
)
from syntrive.contract.writer import WriteOutcome, write_manifest
from syntrive.db.models import Book, BookImage, Job, TranscriptChapter
from syntrive.db.path_utils import repo_dir_from_db
from syntrive.db.session import get_db_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobExportResult:
    job_id: int
    ok: bool
    outcome: Optional[WriteOutcome] = None
    issues: tuple[PathIssue, ...] = ()
    error: Optional[str] = None


@dataclass(frozen=True)
class RepoExportResult:
    ok: bool
    repo_outcome: Optional[WriteOutcome] = None
    repo_issues: tuple[PathIssue, ...] = ()
    jobs: tuple[JobExportResult, ...] = field(default_factory=tuple)
    error: Optional[str] = None


def _generator() -> str:
    try:
        return f"SyntriveTTS {metadata.version('SyntriveTTS')}"
    except metadata.PackageNotFoundError:
        return "SyntriveTTS unknown"


def _job_facts(job: Job) -> JobFacts:
    return JobFacts(
        id=job.id,
        process_dir=job.process_dir,
        status=job.status,
        current_step=job.current_step,
        created_at=job.created_at,
        updated_at=job.updated_at,
        archived_at=job.archived_at,
        source_path=job.epub_path,
        cover_path=job.cover_path,
        audiobooks_dir=job.audiobooks_dir,
        sentence_audio_dir=job.sentence_audio_dir,
        chapter_audio_dir=job.chapter_audio_dir,
        transcript_dir=job.transcript_dir,
    )


def _book_facts(book: Book) -> BookFacts:
    return BookFacts(
        id=book.id,
        title=book.title,
        author=book.author,
        publisher=book.publisher,
        language=book.language,
        publish_date=book.publish_date,
        original_filename=book.original_filename,
        cover=book.cover,
    )


def _snapshot(db: Session, job: Job) -> JobSnapshot:
    book = db.get(Book, job.book_id)
    chapters = db.query(TranscriptChapter).filter_by(job_id=job.id).all()
    images = db.query(BookImage).filter_by(book_id=job.book_id).order_by(BookImage.id).all()
    return JobSnapshot(
        job=_job_facts(job),
        book=_book_facts(book),
        chapters=tuple(
            ChapterFacts(
                chapter_id=c.chapter_id,
                sequence_number=c.sequence_number,
                volume_number=c.volume_number,
                volume=c.volume,
                chapter_number=c.chapter_number,
                chapter_name=c.chapter_name,
                title=c.title,
                transcript_path=c.transcript_path,
                chapter_audio_path=c.chapter_audio_path,
                chapter_audio_seconds=c.chapter_audio_seconds,
                synthesis_status=c.synthesis_status,
            )
            for c in chapters
        ),
        images=tuple(ImageFacts(name=i.name, path=i.path, image_type=i.image_type) for i in images),
    )


def _log_issues(scope: str, issues: Iterable[PathIssue]) -> None:
    for issue in issues:
        logger.warning(
            "contract_path_issue: scope=%s field=%s value=%r reason=%s",
            scope, issue.field, issue.value, issue.reason,
        )


def _export_one_job(db: Session, repo_dir: Path, job: Job, now: datetime, generator: str) -> JobExportResult:
    process_dir = repo_dir / job.process_dir
    if not process_dir.is_dir():
        error = f"process_dir missing on disk: {job.process_dir}"
        logger.warning("contract_job_skipped: job_id=%d reason=%s", job.id, error)
        return JobExportResult(job_id=job.id, ok=False, error=error)

    built = build_job_manifest(_snapshot(db, job), generator=generator, generated_at=now)
    _log_issues(f"job:{job.id}", built.issues)
    outcome = write_manifest(process_dir / JOB_MANIFEST_FILENAME, built.data)
    logger.info(
        "contract_job_exported: job_id=%d status=%s path=%s sha=%s chapters=%d with_audio=%d issues=%d",
        job.id, outcome.status.value, outcome.path, outcome.content_sha256[:12],
        built.data["summary"]["chapter_count"], built.data["summary"]["chapters_with_audio"],
        len(built.issues),
    )
    return JobExportResult(job_id=job.id, ok=True, outcome=outcome, issues=built.issues)


def _export_repo_index(db: Session, db_path: Path, repo_dir: Path, now: datetime, generator: str) -> tuple[WriteOutcome, tuple[PathIssue, ...]]:
    rows = db.query(Job, Book).join(Book, Job.book_id == Book.id).all()
    entries = tuple(RepoJobEntry(job=_job_facts(job), book=_book_facts(book)) for job, book in rows)
    built = build_repo_manifest(entries, db_filename=db_path.name, generator=generator, generated_at=now)
    _log_issues("repo", built.issues)
    outcome = write_manifest(repo_dir / REPO_MANIFEST_FILENAME, built.data)
    logger.info(
        "contract_repo_exported: status=%s path=%s sha=%s jobs=%d issues=%d",
        outcome.status.value, outcome.path, outcome.content_sha256[:12],
        len(built.data["jobs"]), len(built.issues),
    )
    return outcome, built.issues


def export_repo_manifests(
    db_path: Path, *, job_ids: Optional[Iterable[int]] = None, now: Optional[datetime] = None
) -> RepoExportResult:
    if not db_path.is_file():
        return RepoExportResult(ok=False, error=f"database not found: {db_path}")

    stamp = now or datetime.utcnow()
    generator = _generator()
    repo_dir = repo_dir_from_db(db_path)
    wanted = set(job_ids) if job_ids is not None else None

    with get_db_session(db_path) as db:
        query = db.query(Job).order_by(Job.id)
        jobs = [j for j in query.all() if wanted is None or j.id in wanted]
        unknown = sorted(wanted - {j.id for j in jobs}) if wanted is not None else []
        results = [_export_one_job(db, repo_dir, job, stamp, generator) for job in jobs]
        results += [JobExportResult(job_id=jid, ok=False, error="unknown job id") for jid in unknown]
        repo_outcome, repo_issues = _export_repo_index(db, db_path, repo_dir, stamp, generator)

    ok = all(r.ok for r in results)
    return RepoExportResult(ok=ok, repo_outcome=repo_outcome, repo_issues=repo_issues, jobs=tuple(results))


def export_job_manifest(db_path: Path, job_id: int, *, now: Optional[datetime] = None) -> RepoExportResult:
    return export_repo_manifests(db_path, job_ids=[job_id], now=now)


def refresh_manifests_best_effort(db_path: Path, *, reason: str, job_id: Optional[int] = None) -> Optional[RepoExportResult]:
    try:
        job_ids = [job_id] if job_id is not None else None
        result = export_repo_manifests(db_path, job_ids=job_ids)
    except Exception as exc:  # noqa: BLE001 -- derived artifact must never break the pipeline
        logger.warning(
            "contract_refresh_failed: reason=%s job_id=%s error=%s", reason, job_id, exc, exc_info=True,
        )
        return None
    logger.info(
        "contract_refresh_done: reason=%s job_id=%s ok=%s jobs=%d error=%s",
        reason, job_id, result.ok, len(result.jobs), result.error,
    )
    return result
