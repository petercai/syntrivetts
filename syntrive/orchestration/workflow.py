from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from sqlalchemy.orm import Session

from syntrive.adapters.tts import audio_writer, manager
from syntrive.adapters.tts.ipc import _looks_like_oom_error
from syntrive.db.models import Book, Job, OutputConfig, SynthesisBatch, TranscriptChapter, TtsConfig
from syntrive.orchestration import jobs as jobs_module
from syntrive.orchestration.chapter_pipeline import (
    STAGE_SYNTHESIZE,
    ChapterAudioRequest,
    synthesize_and_assemble_chapter,
)
from syntrive.orchestration.jobs import batch_chapters

logger = logging.getLogger(__name__)

ORCHESTRATION_RETRY_ATTEMPTS = 2

DEFAULT_ORCHESTRATION_OOM_RETRY_ATTEMPTS = 10
_OOM_RETRY_ATTEMPTS_ENV = "SYNTRIVE_TTS_OOM_ORCHESTRATION_RETRY_ATTEMPTS"


def _oom_retry_attempts() -> int:
    raw = os.environ.get(_OOM_RETRY_ATTEMPTS_ENV)
    if not raw:
        return DEFAULT_ORCHESTRATION_OOM_RETRY_ATTEMPTS
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("synthesis_config_invalid: var=%s value=%r", _OOM_RETRY_ATTEMPTS_ENV, raw)
        return DEFAULT_ORCHESTRATION_OOM_RETRY_ATTEMPTS

_FILENAME_UNSAFE_CHARS = '<>:"/\\|?*'


def _sanitize_filename_component(value: str) -> str:
    return "".join("_" if c in _FILENAME_UNSAFE_CHARS else c for c in value).strip()


def chapter_basename(chapter: TranscriptChapter) -> str:
    return f"ch_{chapter.sequence_number}" if chapter.sequence_number else chapter.chapter_id


def chapter_m4a_filename(book_title: str, chapter: TranscriptChapter) -> str:
    return f"{_sanitize_filename_component(book_title)}_ch{chapter.sequence_number}.m4a"


@dataclass(frozen=True)
class EnqueueResult:
    ok: bool
    batch_ids: tuple = ()
    error: str = ""


@dataclass(frozen=True)
class PauseResult:
    ok: bool
    flipped: int = 0
    already_in_state: tuple = ()
    skipped_done: tuple = ()
    error: str = ""


def set_batches_paused(
    db: Session, batch_ids: List[int], *, paused: bool,
) -> PauseResult:
    if not batch_ids:
        return PauseResult(ok=False, error="set_batches_paused: no batch ids given")

    unique_ids = sorted(set(batch_ids))
    batches = (
        db.query(SynthesisBatch)
        .filter(SynthesisBatch.synthesis_batch_id.in_(unique_ids))
        .all()
    )
    found = {b.synthesis_batch_id for b in batches}
    missing = [bid for bid in unique_ids if bid not in found]
    if missing:
        logger.warning("set_batches_paused: unknown batch_id(s)=%s", missing)
        return PauseResult(ok=False, error=f"set_batches_paused: unknown batch id(s): {missing}")

    now = datetime.utcnow()
    flipped = 0
    already: List[int] = []
    skipped_done: List[int] = []
    for batch in batches:
        member_states = [c.synthesis_status for c in batch_chapters(db, batch.synthesis_batch_id)]
        if member_states and all(state == "done" for state in member_states):
            skipped_done.append(batch.synthesis_batch_id)
            continue
        currently_paused = batch.paused_at is not None
        if currently_paused == paused:
            already.append(batch.synthesis_batch_id)
            continue
        batch.paused_at = now if paused else None
        flipped += 1

    logger.info(
        "set_batches_paused: paused=%s batch_ids=%s flipped=%d already_in_state=%s skipped_done=%s",
        paused, unique_ids, flipped, already, skipped_done,
    )
    return PauseResult(
        ok=True, flipped=flipped,
        already_in_state=tuple(already), skipped_done=tuple(skipped_done),
    )


def _reading_order_key(chapter: TranscriptChapter) -> tuple:
    return (chapter.job_id, int(chapter.sequence_number) if chapter.sequence_number else -1)


def enqueue_chapters(db: Session, chapter_db_ids: List[int], note: Optional[str] = None) -> EnqueueResult:
    if not chapter_db_ids:
        return EnqueueResult(ok=False, error="enqueue_chapters: no chapters given")

    chapters = db.query(TranscriptChapter).filter(TranscriptChapter.id.in_(chapter_db_ids)).all()
    found_ids = {c.id for c in chapters}
    missing = set(chapter_db_ids) - found_ids
    if missing:
        return EnqueueResult(ok=False, error=f"enqueue_chapters: unknown chapter id(s): {sorted(missing)}")

    not_pending = [c for c in chapters if c.synthesis_status != "pending"]
    if not_pending:
        names = [f"{c.job_id}/{c.chapter_id} ({c.synthesis_status})" for c in not_pending]
        return EnqueueResult(
            ok=False,
            error=f"enqueue_chapters: {len(not_pending)} chapter(s) are not pending, refusing to enqueue: {names}",
        )

    batch_ids: List[int] = []
    for chapter in sorted(chapters, key=_reading_order_key):
        batch = SynthesisBatch(note=note)
        db.add(batch)
        db.flush()
        chapter.synthesis_batch_id = batch.synthesis_batch_id
        chapter.synthesis_status = "queued"
        batch_ids.append(batch.synthesis_batch_id)
        logger.debug(
            "enqueue_chapter: chapter_db_id=%s job_id=%s seq=%s -> batch_id=%d",
            chapter.id, chapter.job_id, chapter.sequence_number or "", batch.synthesis_batch_id,
        )

    logger.info(
        "enqueue_chapters_ok: batches=%d batch_ids=%s chapters=%d note=%s",
        len(batch_ids), batch_ids, len(batch_ids), note or "",
    )
    return EnqueueResult(ok=True, batch_ids=tuple(batch_ids))


def detect_and_rollback_redo(db: Session, chapter: TranscriptChapter, repo_dir: Path) -> bool:
    if chapter.synthesis_status != "done":
        return False

    job = db.query(Job).filter_by(id=chapter.job_id).first()
    process_dir = repo_dir / job.process_dir

    m4a_path = repo_dir / chapter.chapter_audio_path if chapter.chapter_audio_path else None
    if m4a_path is not None and m4a_path.is_file():
        return False

    sentence_dir = process_dir / "sentence_audio"
    expected = chapter.transcript_lines or 0
    existing = sum(
        1 for i in range(1, expected + 1)
        if audio_writer.sentence_audio_path(str(sentence_dir), chapter_basename(chapter), i).is_file()
    )

    new_status = "synthesizing" if existing < expected else "combining"
    logger.info(
        "detect_and_rollback_redo: job_id=%s chapter_id=%s %s -> %s "
        "(m4a_missing=%s sentence_flacs=%d/%d)",
        chapter.job_id, chapter.chapter_id, chapter.synthesis_status, new_status,
        m4a_path is None or not m4a_path.is_file(), existing, expected,
    )
    chapter.synthesis_status = new_status
    chapter.chapter_audio_path = None
    chapter.chapter_audio_seconds = None
    return True


def _tts_config_to_options(tts_config: TtsConfig) -> dict:
    model_id = tts_config.model
    fine_tuned = tts_config.fine_tuned_model
    if tts_config.engine == "xtts" and model_id and model_id != "internal":
        fine_tuned = model_id
    return {
        "device": tts_config.device,
        "language": tts_config.language,
        "model": model_id,
        "fine_tuned_model": fine_tuned,
        "temperature": tts_config.temperature,
        "length_penalty": tts_config.length_penalty,
        "num_beams": tts_config.num_beams,
        "repetition_penalty": tts_config.repetition_penalty,
        "top_k": tts_config.top_k,
        "top_p": tts_config.top_p,
        "speed": tts_config.speed,
        "enable_text_splitting": tts_config.enable_text_splitting,
        "text_temp": tts_config.text_temp,
        "waveform_temp": tts_config.waveform_temp,
        "cfg_value": tts_config.cfg_value,
        "inference_timesteps": tts_config.inference_timesteps,
        "normalize": tts_config.normalize,
        "denoise": tts_config.denoise,
        "retry_badcase": tts_config.retry_badcase,
        "retry_badcase_max_times": tts_config.retry_badcase_max_times,
        "retry_badcase_ratio_threshold": tts_config.retry_badcase_ratio_threshold,
    }


@dataclass(frozen=True)
class BatchRunResult:
    ok: bool
    chapters_completed: int = 0
    error: str = ""
    failed_chapter_id: str = ""
    failed_line_index: Optional[int] = None
    stopped: bool = False


def run_batch(
    db: Session,
    batch_id: int,
    repo_dir: Path,
    *,
    on_event: Optional[Callable[[str, dict], None]] = None,
    progress_plan: Optional[dict] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> BatchRunResult:
    def emit(event_name: str, **fields) -> None:
        if on_event is not None:
            on_event(event_name, {"batch_id": batch_id, **fields})

    batch = db.query(SynthesisBatch).filter_by(synthesis_batch_id=batch_id).first()
    if batch is None:
        return BatchRunResult(ok=False, error=f"run_batch: unknown batch_id {batch_id}")

    chapters = batch_chapters(db, batch_id)
    emit("synth_batch_start", chapter_count=len(chapters))
    completed = 0
    voice_cache_by_config: dict = {}

    def stopped_result(chapter_id: str, reason: str) -> BatchRunResult:
        logger.warning(
            "run_batch_stopped: batch_id=%d chapter=%s chapters_completed=%d reason=%s",
            batch_id, chapter_id, completed, reason,
        )
        emit("batch_stopped", chapter_id=chapter_id, error=reason)
        return BatchRunResult(
            ok=False, chapters_completed=completed, error=reason,
            failed_chapter_id=chapter_id, stopped=True,
        )

    for chapter in chapters:
        if chapter.synthesis_status == "done":
            continue
        if should_stop is not None and should_stop():
            return stopped_result(chapter.chapter_id, "stopped before chapter (caller requested stop)")

        job = db.query(Job).filter_by(id=chapter.job_id).first()
        tts_config = db.query(TtsConfig).filter_by(job_id=job.id).first()
        output_config = db.query(OutputConfig).filter_by(job_id=job.id).first()
        book = db.query(Book).filter_by(id=job.book_id).first()
        process_dir = repo_dir / job.process_dir
        sentence_dir = process_dir / "sentence_audio"
        audiobooks_dir = process_dir / "audiobooks"

        emit("chapter_start", job_id=job.id, chapter_id=chapter.chapter_id, chapter_seq=chapter.sequence_number)

        transcript_abs = process_dir / chapter.transcript_path
        try:
            transcript_content = transcript_abs.read_text(encoding="utf-8")
        except OSError as exc:
            error = f"cannot open transcript for {chapter.chapter_id}: {exc}"
            logger.error("run_batch_transcript_open_failed: chapter=%s error=%s", chapter.chapter_id, exc)
            batch.synth_error = error
            batch.failed_line_index = None
            emit("batch_aborted", job_id=job.id, chapter_id=chapter.chapter_id, error=error)
            return BatchRunResult(
                ok=False, chapters_completed=completed, error=error, failed_chapter_id=chapter.chapter_id,
            )

        if chapter.synthesis_status == "queued":
            chapter.synthesis_status = "synthesizing"
            if batch.synth_started_at is None:
                batch.synth_started_at = datetime.utcnow()
            db.commit()

        if chapter.synthesis_status not in ("synthesizing", "combining"):
            error = f"run_batch: chapter {chapter.chapter_id} in unexpected state {chapter.synthesis_status!r}"
            logger.error("run_batch_unexpected_chapter_state: %s", error)
            batch.synth_error = error
            batch.failed_line_index = None
            emit("batch_aborted", job_id=job.id, chapter_id=chapter.chapter_id, error=error)
            return BatchRunResult(
                ok=False, chapters_completed=completed, error=error, failed_chapter_id=chapter.chapter_id,
            )

        engine = manager.get_engine(tts_config.engine, options=_tts_config_to_options(tts_config))
        cache = voice_cache_by_config.setdefault(tts_config.id, {})

        progress = (progress_plan or {}).get((job.id, chapter.chapter_id))
        if progress is None:
            progress = jobs_module.ChapterProgress(
                book_name=book.title, book_seq=1, book_total=1,
                chapter_seq=1, chapter_total=1,
            )

        def on_line(index: int, outcome: str, elapsed_ms: float, _job=job, _chapter=chapter) -> None:
            event = {"ok": "line_ok", "skipped": "line_ok", "failed": "line_failed"}[outcome]
            emit(
                event, job_id=_job.id, chapter_id=_chapter.chapter_id,
                chapter_seq=_chapter.sequence_number, line_idx=index,
                engine=tts_config.engine, ms=round(elapsed_ms),
            )

        def on_line_start(index: int, total: int, text: str, _job=job, _chapter=chapter, _progress=progress) -> None:
            emit(
                "line_start", job_id=_job.id, chapter_id=_chapter.chapter_id,
                book=_progress.book_name, book_seq=_progress.book_seq, book_total=_progress.book_total,
                chapter_seq=_progress.chapter_seq, chapter_total=_progress.chapter_total,
                line_idx=index, line_total=total, text=text,
            )

        def resolve_voice(voice_id: int, _engine=engine, _cache=cache) -> Optional[str]:
            return _engine.resolve_voice_for_tts_script(tts_config.id, voice_id, db, _cache)

        metadata = {
            "title": chapter.chapter_name or chapter.chapter_id,
            "album": book.title,
        }
        if book.author:
            metadata["artist"] = book.author
        if chapter.chapter_number:
            metadata["track"] = str(int(chapter.chapter_number))
        if chapter.volume_number:
            metadata["disc"] = str(int(chapter.volume_number))

        request = ChapterAudioRequest(
            transcript_content=transcript_content,
            chapter_basename=chapter_basename(chapter),
            sentence_audio_dir=str(sentence_dir),
            audiobooks_dir=str(audiobooks_dir),
            output_filename=chapter_m4a_filename(book.title, chapter),
            metadata=metadata,
            cover_path=str(process_dir / book.cover) if book.cover else None,
        )

        result = None
        attempt = 0
        oom_retry_budget = _oom_retry_attempts()
        while True:
            attempt += 1
            result = synthesize_and_assemble_chapter(
                request, engine,
                resolve_voice=resolve_voice,
                on_line=on_line,
                on_line_start=on_line_start,
                should_stop=should_stop,
            )
            if result.ok or result.stopped or result.stage != STAGE_SYNTHESIZE:
                break
            is_oom = _looks_like_oom_error(result.error)
            max_attempts = (oom_retry_budget if is_oom else ORCHESTRATION_RETRY_ATTEMPTS) + 1
            if attempt >= max_attempts:
                break
            emit(
                "line_retry", job_id=job.id, chapter_id=chapter.chapter_id,
                chapter_seq=chapter.sequence_number,
                line_idx=result.failed_line_index, attempt=attempt, oom=is_oom,
            )

        if result.stopped:
            return stopped_result(chapter.chapter_id, result.error)

        if not result.ok:
            batch.synth_error = result.error
            batch.failed_line_index = result.failed_line_index
            if result.stage != STAGE_SYNTHESIZE:
                chapter.synthesis_status = "combining"
            emit(
                "batch_aborted", job_id=job.id, chapter_id=chapter.chapter_id,
                error=result.error, line_idx=result.failed_line_index,
            )
            return BatchRunResult(
                ok=False, chapters_completed=completed, error=result.error,
                failed_chapter_id=chapter.chapter_id, failed_line_index=result.failed_line_index,
            )

        chapter.chapter_audio_path = f"{job.process_dir}/audiobooks/{Path(result.m4a_path).name}"
        chapter.chapter_audio_seconds = result.duration_ms / 1000
        chapter.synthesis_status = "done"
        completed += 1
        emit(
            "chapter_merged", job_id=job.id, chapter_id=chapter.chapter_id,
            chapter_seq=chapter.sequence_number, chapter_audio_seconds=chapter.chapter_audio_seconds,
        )
        db.commit()

    batch.synth_finished_at = datetime.utcnow()
    batch.synth_error = None
    batch.failed_line_index = None
    logger.info("run_batch_ok: batch_id=%d chapters_completed=%d", batch_id, completed)
    return BatchRunResult(ok=True, chapters_completed=completed)
