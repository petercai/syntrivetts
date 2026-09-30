import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from syntrive.db.models import (
    Book,
    BookImage,
    Job,
    OutputConfig,
    StageEvent,
    ThresholdBlockEvent,
    TranscriptChapter,
    TtsConfig,
    WorkflowStepEvent,
)

logger = logging.getLogger(__name__)


class JobRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def find_or_create_book(
        self,
        title: str,
        author: Optional[str] = None,
        publisher: Optional[str] = None,
        language: Optional[str] = None,
        publish_date: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> Book:
        existing = (
            self._db.query(Book)
            .filter(Book.title == title, Book.author == author)
            .first()
        )
        if existing:
            logger.debug("Reusing existing book: id=%d title=%s", existing.id, title)
            return existing

        book = Book(
            title=title,
            author=author,
            publisher=publisher,
            language=language,
            publish_date=publish_date,
            original_filename=original_filename,
        )
        self._db.add(book)
        self._db.flush()
        logger.info("Created book: id=%d title=%s language=%s", book.id, title, language)
        return book

    def add_book_image(
        self, book: Book, name: str, path: str, image_type: str = "internal"
    ) -> BookImage:
        img = BookImage(book_id=book.id, name=name, path=path, image_type=image_type)
        self._db.add(img)
        return img

    def upsert_book_image(
        self, book: Book, name: str, path: str, image_type: str = "internal"
    ) -> BookImage:
        existing = (
            self._db.query(BookImage)
            .filter(BookImage.book_id == book.id, BookImage.name == name)
            .first()
        )
        if existing:
            existing.path = path
            existing.image_type = image_type
            logger.debug(
                "upsert_book_image: updated book_id=%d name=%s type=%s",
                book.id, name, image_type,
            )
            return existing

        img = BookImage(book_id=book.id, name=name, path=path, image_type=image_type)
        self._db.add(img)
        logger.debug(
            "upsert_book_image: inserted book_id=%d name=%s type=%s",
            book.id, name, image_type,
        )
        return img

    def update_book_cover(self, book: Book, cover: Optional[str]) -> None:
        prev = book.cover
        book.cover = cover
        synced = (
            self._db.query(Job)
            .filter(Job.book_id == book.id)
            .update({"cover_path": cover}, synchronize_session=False)
        )
        logger.info(
            "update_book_cover: book_id=%d cover %r -> %r (synced cover_path on %d job(s))",
            book.id, prev, cover, synced,
        )

    def update_book_title(self, book: Book, title: str) -> None:
        prev = book.title
        book.title = title
        logger.info(
            "update_book_title: book_id=%d title %r -> %r", book.id, prev, title
        )

    def create_job(
        self,
        book_id: int,
        process_dir: str,
        epub_path: str,
        chapter_audio_dir: Optional[str] = None,
        sentence_audio_dir: Optional[str] = None,
        audiobooks_dir: Optional[str] = None,
        transcript_dir: Optional[str] = None,
    ) -> Job:
        job = Job(
            book_id=book_id,
            process_dir=process_dir,
            epub_path=epub_path,
            stage="init",
            status="pending",
            chapter_audio_dir=chapter_audio_dir,
            sentence_audio_dir=sentence_audio_dir,
            audiobooks_dir=audiobooks_dir,
            transcript_dir=transcript_dir,
            current_step="bootstrap",
        )
        self._db.add(job)
        self._db.flush()
        logger.info("Created job: id=%d process_dir=%s", job.id, process_dir)
        return job

    def get_job(self, job_id: int) -> Optional[Job]:
        return self._db.get(Job, job_id)

    def update_job_stage(self, job: Job, stage: str, status: str = "running") -> None:
        job.stage = stage
        job.status = status
        logger.info("Job %d: stage=%s status=%s", job.id, stage, status)

    def list_jobs_for_book(self, book_id: int) -> list[Job]:
        return (
            self._db.query(Job)
            .filter(Job.book_id == book_id)
            .order_by(Job.created_at)
            .all()
        )

    def list_blocked_jobs(self) -> list[Job]:
        return self._db.query(Job).filter(Job.status == "blocked").all()

    def record_stage_start(self, job: Job, stage_name: str) -> StageEvent:
        return self._record_event(job, stage_name, "started", started_at=datetime.utcnow())

    def record_stage_complete(
        self, job: Job, stage_name: str, artifact_paths: Optional[dict] = None
    ) -> StageEvent:
        return self._record_event(
            job,
            stage_name,
            "completed",
            completed_at=datetime.utcnow(),
            artifact_paths=artifact_paths,
        )

    def record_stage_failed(
        self, job: Job, stage_name: str, error_message: str
    ) -> StageEvent:
        return self._record_event(
            job,
            stage_name,
            "failed",
            completed_at=datetime.utcnow(),
            error_message=error_message,
        )

    def _record_event(
        self,
        job: Job,
        stage_name: str,
        status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        artifact_paths: Optional[dict] = None,
    ) -> StageEvent:
        event = StageEvent(
            job_id=job.id,
            stage_name=stage_name,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            error_message=error_message,
            artifact_paths=artifact_paths,
        )
        self._db.add(event)
        return event

    def upsert_transcript_chapter(
        self,
        job: Job,
        chapter_id: str,
        group_id: Optional[str] = None,
        title: Optional[str] = None,
        source_href: Optional[str] = None,
    ) -> TranscriptChapter:
        existing = (
            self._db.query(TranscriptChapter)
            .filter(
                TranscriptChapter.job_id == job.id,
                TranscriptChapter.chapter_id == chapter_id,
            )
            .first()
        )
        if existing:
            return existing

        chapter = TranscriptChapter(
            job_id=job.id,
            chapter_id=chapter_id,
            group_id=group_id,
            title=title,
            source_href=source_href,
        )
        self._db.add(chapter)
        self._db.flush()
        return chapter

    def record_workflow_action(
        self,
        job: Job,
        step_name: str,
        action: str,
        artifacts_summary: Optional[dict] = None,
        user_notes: Optional[str] = None,
    ) -> WorkflowStepEvent:
        event = WorkflowStepEvent(
            job_id=job.id,
            step_name=step_name,
            action=action,
            artifacts_summary=artifacts_summary,
            user_notes=user_notes,
        )
        self._db.add(event)
        logger.debug(
            "Workflow event: job=%d step=%s action=%s", job.id, step_name, action
        )
        return event

    def get_last_workflow_step(self, job: Job) -> Optional[WorkflowStepEvent]:
        return (
            self._db.query(WorkflowStepEvent)
            .filter(WorkflowStepEvent.job_id == job.id)
            .order_by(WorkflowStepEvent.created_at.desc())
            .first()
        )

    def list_incomplete_jobs(self) -> list[Job]:
        return (
            self._db.query(Job)
            .filter(Job.status.notin_(["completed", "done"]))
            .order_by(Job.updated_at.desc())
            .all()
        )

    def record_threshold_block(
        self,
        job: Job,
        chapter_id: str,
        deletion_ratio: float,
        rule_distribution: Optional[dict] = None,
    ) -> ThresholdBlockEvent:
        event = ThresholdBlockEvent(
            job_id=job.id,
            chapter_id=chapter_id,
            deletion_ratio=deletion_ratio,
            rule_distribution=rule_distribution,
        )
        self._db.add(event)
        logger.warning(
            "Threshold blocked: job=%d chapter=%s ratio=%.2f",
            job.id,
            chapter_id,
            deletion_ratio,
        )
        return event
