import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from syntrive.workflow.engine import WorkflowStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveResult:
    ok: bool
    job_id: int
    archived: bool = False
    changed: bool = False
    archived_at: Optional[datetime] = None
    error: Optional[str] = None


class JobService:
    def __init__(self, db_path: Path, holder_kind: str = "cli") -> None:
        self._db_path = db_path
        self._holder_kind = holder_kind

    def _book_job_ids(self, job_id: int) -> list[int]:
        from syntrive.db.models import Job
        from syntrive.db.session import get_db_session

        with get_db_session(self._db_path) as db:
            job = db.get(Job, job_id)
            if job is None:
                return []
            return [row.id for row in db.query(Job.id).filter(Job.book_id == job.book_id)]

    def _refresh_contract(self, reason: str, job_id: Optional[int] = None) -> None:
        from syntrive.services.contract_export_service import refresh_manifests_best_effort

        refresh_manifests_best_effort(self._db_path, reason=reason, job_id=job_id)

    def reset_job_to_start(self, job_id: int) -> None:
        self.reset_job_to_step(job_id, WorkflowStep.BOOTSTRAP)

    def reset_job_to_step(self, job_id: int, step: WorkflowStep) -> None:
        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository

        with get_db_session(self._db_path) as db:
            repo = JobRepository(db)
            job = repo.get_job(job_id)
            if job is None:
                logger.warning("reset_job_to_step: job %d not found", job_id)
                return

            prev_step = job.current_step or "unknown"
            job.current_step = step.value
            job.stage = "init"
            job.status = "pending"

            repo.record_workflow_action(
                job=job,
                step_name=step.value,
                action="reset",
                user_notes=f"User reset from step={prev_step} to step={step.value}",
            )
            logger.info(
                "Job %d reset to step=%s: prev_step=%s status=pending",
                job_id, step.value, prev_step,
            )

        self._refresh_contract(f"job:reset:{step.value}", job_id=job_id)

    def archive_job(self, job_id: int, note: Optional[str] = None) -> "ArchiveResult":
        return self._set_archived(job_id, archived=True, note=note)

    def unarchive_job(self, job_id: int, note: Optional[str] = None) -> "ArchiveResult":
        return self._set_archived(job_id, archived=False, note=note)

    def _set_archived(self, job_id: int, *, archived: bool, note: Optional[str]) -> "ArchiveResult":
        from syntrive.db.models import Job
        from syntrive.db.repository import JobRepository
        from syntrive.db.session import get_db_session
        from syntrive.services.contract_export_service import refresh_manifests_best_effort

        action = "archived" if archived else "unarchived"
        with get_db_session(self._db_path) as db:
            job = db.get(Job, job_id)
            if job is None:
                logger.warning("job_archive_rejected: job_id=%d reason=not_found", job_id)
                return ArchiveResult(ok=False, job_id=job_id, error="job not found")
            if (job.archived_at is not None) == archived:
                logger.info("job_archive_noop: job_id=%d already=%s", job_id, action)
                return ArchiveResult(ok=True, job_id=job_id, archived=archived, changed=False, archived_at=job.archived_at)

            job.archived_at = datetime.utcnow() if archived else None
            JobRepository(db).record_workflow_action(
                job=job, step_name="archive", action=action, user_notes=note,
            )
            archived_at = job.archived_at
            logger.info("job_archive_changed: job_id=%d action=%s archived_at=%s", job_id, action, archived_at)

        refresh_manifests_best_effort(self._db_path, reason=f"job:{action}", job_id=job_id)
        return ArchiveResult(ok=True, job_id=job_id, archived=archived, changed=True, archived_at=archived_at)

    def update_book_language(self, job_id: int, language: str) -> None:
        from syntrive.services.job_lease import hold_job_leases

        if self._current_book_language(job_id) == language:
            logger.debug("update_book_language: job=%d language %r unchanged -- noop", job_id, language)
            return

        with hold_job_leases(
            self._db_path, self._book_job_ids(job_id),
            holder_kind=self._holder_kind, operation="set_language",
        ):
            changed = self._write_book_language(job_id, language)
        if changed:
            self._refresh_contract("book:language")

    def _current_book_language(self, job_id: int) -> Optional[str]:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book

        with get_db_session(self._db_path) as db:
            job = db.get(Job, job_id)
            book = db.get(Book, job.book_id) if job is not None else None
            return book.language if book is not None else None

    def _write_book_language(self, job_id: int, language: str) -> bool:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book

        with get_db_session(self._db_path) as db:
            job = db.query(Job).filter_by(id=job_id).first()
            if job is None:
                logger.warning("update_book_language: job %d not found", job_id)
                return False
            book = db.query(Book).filter_by(id=job.book_id).first()
            if book is None:
                logger.warning(
                    "update_book_language: book not found for job %d", job_id
                )
                return False
            prev = book.language
            book.language = language
            logger.info(
                "update_book_language: job=%d book=%d language %r -> %r",
                job_id, book.id, prev, language,
            )
            return True

    def ensure_book_images_synced(self, job_id: int) -> int:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book
        from syntrive.db.repository import JobRepository

        from syntrive.db.path_utils import resolve_abs

        with get_db_session(self._db_path) as db:
            job = db.query(Job).filter_by(id=job_id).first()
            if job is None:
                logger.warning("ensure_book_images_synced: job %d not found", job_id)
                return 0
            epub_path = resolve_abs(self._db_path, job.epub_path)
            process_dir = resolve_abs(self._db_path, job.process_dir)

        images_dir = process_dir / "images"

        if images_dir.exists() and not any(images_dir.iterdir()) and epub_path.exists():
            logger.info(
                "ensure_book_images_synced: images dir empty for job %d, extracting from EPUB",
                job_id,
            )
            from syntrive.bootstrap import extract_epub_images

            image_records = extract_epub_images(epub_path, images_dir, process_dir)
            if image_records:
                with get_db_session(self._db_path) as db:
                    repo = JobRepository(db)
                    job_row = db.query(Job).filter_by(id=job_id).first()
                    if job_row is not None:
                        book = db.query(Book).filter_by(id=job_row.book_id).first()
                        if book is not None:
                            for rec in image_records:
                                repo.upsert_book_image(
                                    book=book,
                                    name=rec.name,
                                    path=rec.rel_path,
                                    image_type=rec.image_type,
                                )
                            if book.cover is None:
                                detected = next(
                                    (r for r in image_records if r.image_type == "cover"),
                                    None,
                                )
                                if detected is not None:
                                    repo.update_book_cover(book, detected.rel_path)
                            logger.info(
                                "ensure_book_images_synced: persisted %d image(s) for job %d",
                                len(image_records),
                                job_id,
                            )

        new_count = self.sync_book_images_from_dir(job_id)

        self._backfill_cover(job_id, epub_path)

        return new_count

    def _backfill_cover(self, job_id: int, epub_path: Path) -> None:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book, BookImage
        from syntrive.db.repository import JobRepository

        with get_db_session(self._db_path) as db:
            job = db.query(Job).filter_by(id=job_id).first()
            if job is None:
                return
            book = db.query(Book).filter_by(id=job.book_id).first()
            if book is None or book.cover is not None:
                return
            images = db.query(BookImage).filter_by(book_id=book.id).all()
            if not images:
                return

            cover_name: Optional[str] = None
            if epub_path.exists():
                try:
                    from ebooklib import epub as ebooklib_epub
                    from syntrive.bootstrap import _detect_cover_name
                    epub_book = ebooklib_epub.read_epub(str(epub_path), {"ignore_ncx": True})
                    cover_name = _detect_cover_name(epub_book)
                except Exception as exc:
                    logger.debug("_backfill_cover: EPUB parse failed: %s", exc)

            image_names = sorted(img.name for img in images)
            if cover_name is None:
                cover_name = next((n for n in image_names if "cover" in n.lower()), None)
            if cover_name is None and image_names:
                cover_name = image_names[0]
            if cover_name is None:
                return

            cover_img = next((img for img in images if img.name == cover_name), None)
            if cover_img is None:
                return

            repo = JobRepository(db)
            repo.update_book_cover(book, cover_img.path)
            cover_img.image_type = "cover"
            logger.info(
                "_backfill_cover: set cover=%s for job %d",
                cover_img.path,
                job_id,
            )

    def sync_book_images_from_dir(self, job_id: int) -> int:
        from syntrive.db.path_utils import resolve_abs
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book, BookImage

        _IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"})

        with get_db_session(self._db_path) as db:
            job = db.query(Job).filter_by(id=job_id).first()
            if job is None:
                logger.warning("sync_book_images_from_dir: job %d not found", job_id)
                return 0
            book = db.query(Book).filter_by(id=job.book_id).first()
            if book is None:
                logger.warning("sync_book_images_from_dir: book not found for job %d", job_id)
                return 0

            process_dir = resolve_abs(self._db_path, job.process_dir)
            images_dir = process_dir / "images"

            if not images_dir.exists():
                logger.debug("sync_book_images_from_dir: images_dir does not exist: %s", images_dir)
                return 0

            existing_names = {
                row.name
                for row in db.query(BookImage.name).filter_by(book_id=book.id).all()
            }

            inserted = 0
            for img_file in sorted(images_dir.iterdir()):
                if not img_file.is_file():
                    continue
                if img_file.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                name = img_file.name
                try:
                    rel_path = img_file.relative_to(process_dir).as_posix()
                except ValueError:
                    rel_path = f"images/{name}"

                if name not in existing_names:
                    img = BookImage(
                        book_id=book.id,
                        name=name,
                        path=rel_path,
                        image_type="internal",
                    )
                    db.add(img)
                    inserted += 1
                    logger.debug(
                        "sync_book_images_from_dir: new image book_id=%d name=%s rel=%s",
                        book.id, name, rel_path,
                    )

            logger.info(
                "sync_book_images_from_dir: job=%d inserted=%d",
                job_id, inserted,
            )
            return inserted

    def update_book_title(self, job_id: int, title: str) -> None:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book

        if not title or not title.strip():
            logger.warning("update_book_title: empty title rejected for job %d", job_id)
            return

        with get_db_session(self._db_path) as db:
            job = db.query(Job).filter_by(id=job_id).first()
            if job is None:
                logger.warning("update_book_title: job %d not found", job_id)
                return
            book = db.query(Book).filter_by(id=job.book_id).first()
            if book is None:
                logger.warning("update_book_title: book not found for job %d", job_id)
                return
            prev = book.title
            book.title = title.strip()
            logger.info(
                "update_book_title: job=%d book=%d title %r -> %r",
                job_id, book.id, prev, book.title,
            )

        self._refresh_contract("book:title")

    def update_book_author(self, job_id: int, author: str) -> None:
        from syntrive.db.models import Book, Job
        from syntrive.db.session import get_db_session

        with get_db_session(self._db_path) as db:
            job = db.get(Job, job_id)
            book = db.get(Book, job.book_id) if job is not None else None
            if book is None:
                logger.warning("update_book_author: job %d / its book not found", job_id)
                return
            prev, book.author = book.author, (author or "").strip() or None
            logger.info("update_book_author: job=%d book=%d author %r -> %r", job_id, book.id, prev, book.author)
        self._refresh_contract("book:author")

    def update_book_cover(self, job_id: int, cover_rel_path: Optional[str]) -> None:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book
        from syntrive.db.repository import JobRepository

        with get_db_session(self._db_path) as db:
            job = db.query(Job).filter_by(id=job_id).first()
            if job is None:
                logger.warning("update_book_cover: job %d not found", job_id)
                return
            book = db.query(Book).filter_by(id=job.book_id).first()
            if book is None:
                logger.warning("update_book_cover: book not found for job %d", job_id)
                return
            JobRepository(db).update_book_cover(book, cover_rel_path)

        self._refresh_contract("book:cover")

    def delete_book_job(self, job_id: int) -> dict:
        from syntrive.services.job_lease import hold_job_leases

        with hold_job_leases(
            self._db_path, self._book_job_ids(job_id),
            holder_kind=self._holder_kind, operation="delete_book",
        ):
            result = self._delete_book_job_unleased(job_id)
        if result["jobs_deleted"]:
            self._refresh_contract("book:deleted")
        return result

    def _delete_book_job_unleased(self, job_id: int) -> dict:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book
        from syntrive.db.path_utils import resolve_abs

        process_dirs_to_remove: list[Path] = []
        book_title: Optional[str] = None
        jobs_deleted = 0

        with get_db_session(self._db_path) as db:
            job = db.query(Job).filter_by(id=job_id).first()
            if job is None:
                logger.warning("delete_book_job: job %d not found", job_id)
                return {"book_title": None, "jobs_deleted": 0, "dirs_deleted": 0, "dirs_missing": 0}

            book = db.query(Book).filter_by(id=job.book_id).first()
            if book is None:
                logger.warning("delete_book_job: book not found for job %d", job_id)
                return {"book_title": None, "jobs_deleted": 0, "dirs_deleted": 0, "dirs_missing": 0}

            book_title = book.title
            book_id = book.id

            all_jobs = db.query(Job).filter_by(book_id=book_id).all()
            for j in all_jobs:
                if j.process_dir:
                    process_dirs_to_remove.append(resolve_abs(self._db_path, j.process_dir))
                logger.info(
                    "delete_book_job: scheduling DB delete for job %d process_dir=%s",
                    j.id, j.process_dir,
                )
                db.delete(j)
                jobs_deleted += 1

            db.delete(book)

        logger.info(
            "delete_book_job: DB commit OK book=%r jobs=%d",
            book_title, jobs_deleted,
        )

        dirs_deleted = 0
        dirs_missing = 0
        for abs_dir in process_dirs_to_remove:
            if abs_dir.exists():
                try:
                    shutil.rmtree(abs_dir)
                    dirs_deleted += 1
                    logger.info("delete_book_job: removed dir %s", abs_dir)
                except OSError as exc:
                    logger.warning(
                        "delete_book_job: could not remove %s: %s", abs_dir, exc
                    )
            else:
                dirs_missing += 1
                logger.warning(
                    "delete_book_job: process_dir already absent: %s", abs_dir
                )

        logger.info(
            "delete_book_job: complete book=%r jobs=%d dirs_deleted=%d dirs_missing=%d",
            book_title, jobs_deleted, dirs_deleted, dirs_missing,
        )
        return {
            "book_title": book_title,
            "jobs_deleted": jobs_deleted,
            "dirs_deleted": dirs_deleted,
            "dirs_missing": dirs_missing,
        }

    def get_artifact_status(self, job_id: int) -> dict[WorkflowStep, bool]:
        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository
        from syntrive.workflow.engine import WorkflowEngine

        with get_db_session(self._db_path) as db:
            repo = JobRepository(db)
            job = repo.get_job(job_id)
            if job is None:
                logger.warning("get_artifact_status: job %d not found", job_id)
                return {}
            db.expunge_all()

        engine = WorkflowEngine(job=job, db_path=self._db_path)
        return {
            step: engine.check_artifacts_exist(step)
            for step in WorkflowStep
            if step not in (WorkflowStep.DONE,)
        }
