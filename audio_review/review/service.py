from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from audio_review.review.repository import SentenceReviewRepository
from syntrive.adapters.tts.audio_writer import sentence_audio_path
from syntrive.db.models import Job, SentenceReviewEvent, TranscriptChapter
from syntrive.db.path_utils import resolve_abs, to_repo_relative
from syntrive.orchestration.workflow import chapter_basename as _chapter_basename
from syntrive.orchestration.workflow import chapter_m4a_filename

logger = logging.getLogger(__name__)

TRASH_DIRNAME = ".trash"


def _read_description_tag(flac_path: Path) -> Optional[str]:
    try:
        from mutagen.flac import FLAC

        values = FLAC(str(flac_path)).get("DESCRIPTION")
        return values[0] if values else None
    except Exception as exc:  # noqa: BLE001 -- a read failure must not block the caller
        logger.warning("sentence_review_description_tag_read_failed: path=%s error=%s", flac_path, exc)
        return None


def _sentence_flac_path(
    db_path: Path, job: Job, transcript_chapter: TranscriptChapter, sentence_index: int
) -> Path:
    sentence_audio_dir = resolve_abs(db_path, job.sentence_audio_dir)
    return sentence_audio_path(str(sentence_audio_dir), _chapter_basename(transcript_chapter), sentence_index)


def _trash_path_for(flac_path: Path) -> Path:
    return flac_path.parent / TRASH_DIRNAME / f"{flac_path.name}.{int(time.time())}.flac"


_ROLLBACK_ON_DELETE_STATUSES = frozenset({"done", "combining"})


def _chapter_m4a_candidates(db_path: Path, job: Job, transcript_chapter: TranscriptChapter) -> tuple[Path, ...]:
    paths = []
    if transcript_chapter.chapter_audio_path:
        paths.append(resolve_abs(db_path, transcript_chapter.chapter_audio_path))
    if job.book is not None and transcript_chapter.sequence_number:
        audiobooks_dir = resolve_abs(db_path, job.process_dir) / "audiobooks"
        paths.append(audiobooks_dir / chapter_m4a_filename(job.book.title, transcript_chapter))
    return tuple(dict.fromkeys(p.resolve() for p in paths))


def invalidate_chapter_audio(db_path: Path, job: Job, transcript_chapter: TranscriptChapter) -> bool:
    removed_m4as = [p for p in _chapter_m4a_candidates(db_path, job, transcript_chapter) if p.is_file()]
    for m4a_path in removed_m4as:
        m4a_path.unlink()
    m4a_removed = bool(removed_m4as)

    old_status = transcript_chapter.synthesis_status
    rollback = old_status in _ROLLBACK_ON_DELETE_STATUSES
    if not (m4a_removed or rollback or transcript_chapter.chapter_audio_path):
        return False

    if rollback:
        transcript_chapter.synthesis_status = "synthesizing"
    transcript_chapter.chapter_audio_path = None
    transcript_chapter.chapter_audio_seconds = None
    logger.info(
        "sentence_review_chapter_audio_invalidated: job_id=%s chapter_id=%s status=%s->%s m4a_removed=%s m4a_paths=%s",
        transcript_chapter.job_id, transcript_chapter.chapter_id, old_status,
        transcript_chapter.synthesis_status, m4a_removed, [str(p) for p in removed_m4as],
    )
    return True


def delete_sentence(
    db: Session,
    db_path: Path,
    job: Job,
    transcript_chapter: TranscriptChapter,
    sentence_index: int,
    *,
    issue_category: Optional[str] = None,
    note: Optional[str] = None,
) -> SentenceReviewEvent:
    repo = SentenceReviewRepository(db)
    flac_path = _sentence_flac_path(db_path, job, transcript_chapter, sentence_index)

    if not flac_path.exists():
        existing = repo.get_open_delete(
            job_id=job.id, transcript_chapter_id=transcript_chapter.id, sentence_index=sentence_index
        )
        if existing is not None:
            logger.info(
                "sentence_review_delete_already_deleted: job_id=%s transcript_chapter_id=%s sentence_index=%s",
                job.id, transcript_chapter.id, sentence_index,
            )
            return existing
        raise FileNotFoundError(f"Sentence FLAC not found: {flac_path}")

    description_tag = _read_description_tag(flac_path)
    trash_path = _trash_path_for(flac_path)
    trash_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(flac_path), str(trash_path))
    invalidate_chapter_audio(db_path, job, transcript_chapter)

    event = repo.record_delete(
        job_id=job.id,
        transcript_chapter_id=transcript_chapter.id,
        sentence_index=sentence_index,
        trash_path=to_repo_relative(db_path, trash_path),
        description_tag_snapshot=description_tag,
        issue_category=issue_category,
        note=note,
    )
    logger.info(
        "sentence_review_delete: job_id=%s transcript_chapter_id=%s sentence_index=%s "
        "issue_category=%s trash_path=%s",
        job.id, transcript_chapter.id, sentence_index, issue_category, event.trash_path,
    )
    return event


def undo_delete(db: Session, db_path: Path, event_id: int) -> None:
    repo = SentenceReviewRepository(db)
    event = repo.get(event_id)
    if event is None:
        raise ValueError(f"No SentenceReviewEvent with id={event_id}")
    if event.action != "deleted" or event.trash_path is None:
        raise ValueError(f"SentenceReviewEvent {event_id} is not an open delete (action={event.action!r})")

    trash_abs = resolve_abs(db_path, event.trash_path)
    if not trash_abs.exists():
        raise ValueError(f"Trashed file no longer exists: {trash_abs}")

    job = db.query(Job).filter(Job.id == event.job_id).first()
    transcript_chapter = (
        db.query(TranscriptChapter).filter(TranscriptChapter.id == event.transcript_chapter_id).first()
    )
    if job is None or transcript_chapter is None:
        raise ValueError(f"SentenceReviewEvent {event_id} references a missing job/chapter")

    original_path = _sentence_flac_path(db_path, job, transcript_chapter, event.sentence_index)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(trash_abs), str(original_path))

    repo.delete_event(event_id)
    logger.info(
        "sentence_review_undo: event_id=%s job_id=%s transcript_chapter_id=%s sentence_index=%s restored_path=%s",
        event_id, event.job_id, event.transcript_chapter_id, event.sentence_index, original_path,
    )


def rescan_chapter(
    db: Session, db_path: Path, job: Job, transcript_chapter: TranscriptChapter
) -> List[SentenceReviewEvent]:
    repo = SentenceReviewRepository(db)
    resolved: List[SentenceReviewEvent] = []
    for event in repo.list_open_deletes_for_chapter(job_id=job.id, transcript_chapter_id=transcript_chapter.id):
        flac_path = _sentence_flac_path(db_path, job, transcript_chapter, event.sentence_index)
        if not flac_path.exists():
            continue
        new_tag = _read_description_tag(flac_path)
        repo.record_resolved(event.id)
        resolved.append(event)
        logger.info(
            "sentence_review_resolved: event_id=%s sentence_index=%s regenerated_description_tag=%s",
            event.id, event.sentence_index, new_tag,
        )
    return resolved
