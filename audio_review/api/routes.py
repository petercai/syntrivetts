from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path as FsPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from audio_review.player.playlist import (
    build_chapter_playlist,
    list_reviewable_chapters,
    reference_audio_abs_path,
    sentence_flac_abs_path,
)
from audio_review.repo.state import ActiveRepo
from audio_review.repo.switch import RepoAlreadyServedError, switch_active_repo
from audio_review.review import service as review_service
from syntrive.db.models import Book, Job, SentenceReviewEvent, TranscriptChapter
from syntrive.orchestration.workflow import chapter_basename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["audio-review"])


class HealthResponse(BaseModel):
    status: str
    repo_dir: str


class JobSummary(BaseModel):
    job_id: int
    book_title: str
    process_dir: str
    status: str
    created_at: datetime | None = None


class JobListResponse(BaseModel):
    repo_dir: str
    jobs: list[JobSummary]


class SwitchRepoRequest(BaseModel):
    repo_dir: str = Field(min_length=1)


class DeleteSentenceRequest(BaseModel):
    job_id: int = Field(ge=1)
    transcript_chapter_id: int = Field(ge=1)
    sentence_index: int = Field(ge=1)
    issue_category: str | None = None
    note: str | None = None


class SentenceReviewEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, validate_by_name=True, validate_by_alias=True)

    event_id: int = Field(validation_alias="id")
    job_id: int
    transcript_chapter_id: int
    sentence_index: int
    action: str
    issue_category: str | None = None
    trash_path: str | None = None
    created_at: datetime | None = None
    resolved_at: datetime | None = None


class DeleteSentenceResponse(SentenceReviewEventResponse):
    chapter_synthesis_status: str


class RescanChapterResponse(BaseModel):
    resolved: list[SentenceReviewEventResponse]


class ChapterSummary(BaseModel):
    transcript_chapter_id: int
    chapter_id: str
    chapter_basename: str
    chapter_name: str | None = None
    sequence_number: str | None = None
    synthesis_status: str


class ChapterListResponse(BaseModel):
    job_id: int
    chapters: list[ChapterSummary]


class SentenceEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sentence_index: int
    file_name: str
    text: str
    voice_id: int
    exists: bool
    status: str
    description_tag: str | None = None
    reference_audio_available: bool
    silence_after_seconds: float
    review_event_id: int | None = None


class ChapterPlaylistResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transcript_chapter_id: int
    chapter_basename: str
    sentences: list[SentenceEntryResponse]


def get_active_repo(request: Request) -> ActiveRepo:
    return request.app.state.active_repo


ActiveRepoDep = Annotated[ActiveRepo, Depends(get_active_repo)]


def get_db(active: ActiveRepoDep) -> Iterator[Session]:
    with Session(active.engine) as db:
        yield db


DbDep = Annotated[Session, Depends(get_db)]

JobIdPath = Annotated[int, Path(ge=1)]
ChapterIdPath = Annotated[int, Path(ge=1)]
SentenceIndexPath = Annotated[int, Path(ge=1)]


def _load_job(db: Session, job_id: int) -> Job:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id={job_id}")
    return job


def _load_job_and_chapter(db: Session, job_id: int, transcript_chapter_id: int) -> tuple[Job, TranscriptChapter]:
    job = _load_job(db, job_id)
    chapter = db.query(TranscriptChapter).filter(TranscriptChapter.id == transcript_chapter_id).first()
    if chapter is None:
        raise HTTPException(status_code=404, detail=f"No chapter with id={transcript_chapter_id}")
    if chapter.job_id != job_id:
        raise HTTPException(
            status_code=400,
            detail=f"Chapter {transcript_chapter_id} does not belong to job {job_id}",
        )
    return job, chapter


def get_job(job_id: JobIdPath, db: DbDep) -> Job:
    return _load_job(db, job_id)


def get_job_and_chapter(
    job_id: JobIdPath, transcript_chapter_id: ChapterIdPath, db: DbDep
) -> tuple[Job, TranscriptChapter]:
    return _load_job_and_chapter(db, job_id, transcript_chapter_id)


JobDep = Annotated[Job, Depends(get_job)]
JobChapterDep = Annotated[tuple[Job, TranscriptChapter], Depends(get_job_and_chapter)]


@router.get("/health")
def health(active: ActiveRepoDep) -> HealthResponse:
    return HealthResponse(status="ok", repo_dir=str(active.repo_dir))


def _list_jobs(db: Session) -> list[JobSummary]:
    rows = db.query(Job, Book).join(Book, Job.book_id == Book.id).all()
    return [
        JobSummary(
            job_id=job.id,
            book_title=book.title,
            process_dir=job.process_dir,
            status=job.status,
            created_at=job.created_at,
        )
        for job, book in rows
    ]


@router.get("/jobs")
def list_jobs(active: ActiveRepoDep, db: DbDep) -> JobListResponse:
    jobs = _list_jobs(db)
    logger.info("audio_review_jobs_listed: repo_dir=%s count=%d", active.repo_dir, len(jobs))
    return JobListResponse(repo_dir=str(active.repo_dir), jobs=jobs)


@router.post("/repo/switch")
def switch_repo(body: SwitchRepoRequest, request: Request, active: ActiveRepoDep) -> JobListResponse:
    try:
        new_active = switch_active_repo(active, FsPath(body.repo_dir))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RepoAlreadyServedError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc), "url": exc.url}) from exc

    request.app.state.active_repo = new_active
    with Session(new_active.engine) as db:
        jobs = _list_jobs(db)
    return JobListResponse(repo_dir=str(new_active.repo_dir), jobs=jobs)


@router.post("/sentences/delete")
def delete_sentence(body: DeleteSentenceRequest, active: ActiveRepoDep, db: DbDep) -> DeleteSentenceResponse:
    job, chapter = _load_job_and_chapter(db, body.job_id, body.transcript_chapter_id)
    try:
        event = review_service.delete_sentence(
            db, active.db_path, job, chapter, body.sentence_index,
            issue_category=body.issue_category, note=body.note,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.commit()
    return DeleteSentenceResponse(
        **SentenceReviewEventResponse.model_validate(event).model_dump(),
        chapter_synthesis_status=chapter.synthesis_status,
    )


@router.post("/sentences/{event_id}/undo")
def undo_sentence_delete(
    event_id: Annotated[int, Path(ge=1)], active: ActiveRepoDep, db: DbDep
) -> SentenceReviewEventResponse:
    event = db.query(SentenceReviewEvent).filter(SentenceReviewEvent.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail=f"No SentenceReviewEvent with id={event_id}")
    snapshot = SentenceReviewEventResponse.model_validate(event)
    try:
        review_service.undo_delete(db, active.db_path, event_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return snapshot


@router.post("/jobs/{job_id}/chapters/{transcript_chapter_id}/rescan")
def rescan_chapter(job_chapter: JobChapterDep, active: ActiveRepoDep, db: DbDep) -> RescanChapterResponse:
    job, chapter = job_chapter
    resolved = review_service.rescan_chapter(db, active.db_path, job, chapter)
    db.commit()
    return RescanChapterResponse(resolved=[SentenceReviewEventResponse.model_validate(e) for e in resolved])


@router.get("/jobs/{job_id}/chapters")
def list_chapters(job: JobDep, active: ActiveRepoDep, db: DbDep) -> ChapterListResponse:
    chapters = list_reviewable_chapters(db, active.db_path, job)
    return ChapterListResponse(
        job_id=job.id,
        chapters=[
            ChapterSummary(
                transcript_chapter_id=c.id,
                chapter_id=c.chapter_id,
                chapter_basename=chapter_basename(c),
                chapter_name=c.chapter_name,
                sequence_number=c.sequence_number,
                synthesis_status=c.synthesis_status,
            )
            for c in chapters
        ],
    )


@router.get("/jobs/{job_id}/chapters/{transcript_chapter_id}")
def get_chapter_playlist(job_chapter: JobChapterDep, active: ActiveRepoDep, db: DbDep) -> ChapterPlaylistResponse:
    job, chapter = job_chapter
    try:
        playlist = build_chapter_playlist(db, active.db_path, job, chapter)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChapterPlaylistResponse.model_validate(playlist)


@router.get("/jobs/{job_id}/chapters/{transcript_chapter_id}/sentences/{sentence_index}/audio")
def get_sentence_audio(
    sentence_index: SentenceIndexPath, job_chapter: JobChapterDep, active: ActiveRepoDep
) -> FileResponse:
    job, chapter = job_chapter
    flac_path = sentence_flac_abs_path(active.db_path, job, chapter, sentence_index)
    if not flac_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sentence FLAC not found: {flac_path}")
    return FileResponse(flac_path, media_type="audio/flac")


@router.get("/jobs/{job_id}/chapters/{transcript_chapter_id}/sentences/{sentence_index}/reference-audio")
def get_sentence_reference_audio(
    sentence_index: SentenceIndexPath,
    voice_id: Annotated[int, Query(ge=0)],
    job_chapter: JobChapterDep,
    active: ActiveRepoDep,
    db: DbDep,
) -> FileResponse:
    job, _chapter = job_chapter
    ref_path = reference_audio_abs_path(db, active.db_path, job, voice_id)
    if ref_path is None or not ref_path.is_file():
        raise HTTPException(status_code=404, detail=f"No reference audio available for voice_id={voice_id}")
    return FileResponse(ref_path, media_type="audio/wav")
