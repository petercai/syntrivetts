from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from audio_review.player.silence_gap import plan_chapter_silence
from audio_review.review.repository import SentenceReviewRepository
from syntrive.adapters.text.tts_script import TextSegment, parse_tts_script
from syntrive.adapters.tts.audio_writer import sentence_audio_chapter_dir, sentence_audio_path
from syntrive.db.models import Job, ReferenceVoice, TranscriptChapter, TtsConfig, TtsVoice
from syntrive.db.path_utils import resolve_abs
from syntrive.orchestration.workflow import chapter_basename as _chapter_basename

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentenceEntry:
    sentence_index: int
    file_name: str
    text: str
    voice_id: int
    exists: bool
    status: str
    description_tag: Optional[str]
    reference_audio_available: bool
    silence_after_seconds: float
    review_event_id: Optional[int] = None


@dataclass(frozen=True)
class ChapterPlaylist:
    transcript_chapter_id: int
    chapter_basename: str
    sentences: List[SentenceEntry]


def _read_description_tag(flac_path: Path) -> Optional[str]:
    try:
        from mutagen.flac import FLAC

        values = FLAC(str(flac_path)).get("DESCRIPTION")
        return values[0] if values else None
    except Exception as exc:  # noqa: BLE001 -- a read failure must not block the scan
        logger.warning("playlist_description_tag_read_failed: path=%s error=%s", flac_path, exc)
        return None


def _resolve_transcript_content(db_path: Path, job: Job, transcript_chapter: TranscriptChapter) -> str:
    if not transcript_chapter.transcript_path:
        raise ValueError(
            f"Chapter {transcript_chapter.chapter_id} has no transcript_path yet "
            "(TRANSCRIPT_REVIEW not confirmed) -- nothing to review"
        )
    process_dir_abs = resolve_abs(db_path, job.process_dir)
    transcript_abs = process_dir_abs / transcript_chapter.transcript_path
    return transcript_abs.read_text(encoding="utf-8")


def sentence_flac_abs_path(db_path: Path, job: Job, transcript_chapter: TranscriptChapter, sentence_index: int) -> Path:
    chapter_basename = _chapter_basename(transcript_chapter)
    sentence_audio_dir = resolve_abs(db_path, job.sentence_audio_dir)
    return sentence_audio_path(str(sentence_audio_dir), chapter_basename, sentence_index)


def list_reviewable_chapters(db: Session, db_path: Path, job: Job) -> List[TranscriptChapter]:
    sentence_audio_dir = str(resolve_abs(db_path, job.sentence_audio_dir))
    candidates = (
        db.query(TranscriptChapter)
        .filter(TranscriptChapter.job_id == job.id, TranscriptChapter.sequence_number.is_not(None))
        .order_by(TranscriptChapter.sequence_number)
        .all()
    )
    reviewable = [
        ch for ch in candidates
        if sentence_audio_chapter_dir(sentence_audio_dir, _chapter_basename(ch)).is_dir()
    ]
    logger.info(
        "audio_review_chapters_listed: job_id=%s sentence_audio_dir=%s total=%d with_sequence=%d reviewable=%d",
        job.id, sentence_audio_dir,
        db.query(TranscriptChapter).filter(TranscriptChapter.job_id == job.id).count(),
        len(candidates), len(reviewable),
    )
    return reviewable


def reference_audio_abs_path(db: Session, db_path: Path, job: Job, voice_id: int) -> Optional[Path]:
    tts_config = db.query(TtsConfig).filter(TtsConfig.job_id == job.id).first()
    if tts_config is None:
        return None
    voice = (
        db.query(TtsVoice)
        .filter(
            TtsVoice.tts_config_id == tts_config.id,
            TtsVoice.voice_id == voice_id,
            TtsVoice.excluded.is_(False),
        )
        .first()
    )
    if voice is None or voice.reference_voice_id is None:
        return None
    ref = db.query(ReferenceVoice).filter(ReferenceVoice.id == voice.reference_voice_id).first()
    if ref is None:
        return None
    return resolve_abs(db_path, ref.path)


def build_chapter_playlist(db: Session, db_path: Path, job: Job, transcript_chapter: TranscriptChapter) -> ChapterPlaylist:
    transcript_content = _resolve_transcript_content(db_path, job, transcript_chapter)
    tokens = parse_tts_script(transcript_content)
    text_segments = [t for t in tokens if isinstance(t, TextSegment)]

    chapter_basename = _chapter_basename(transcript_chapter)
    sentence_audio_dir = resolve_abs(db_path, job.sentence_audio_dir)
    expected_paths = [
        str(sentence_flac_abs_path(db_path, job, transcript_chapter, i))
        for i in range(1, len(text_segments) + 1)
    ]
    _segments, silence_between = plan_chapter_silence(transcript_content, expected_paths)
    silence_after = silence_between + [0.0]

    review_repo = SentenceReviewRepository(db)
    open_deletes = {
        event.sentence_index: event
        for event in review_repo.list_open_deletes_for_chapter(
            job_id=job.id, transcript_chapter_id=transcript_chapter.id
        )
    }

    reference_audio_cache: dict[int, bool] = {}
    sentences: List[SentenceEntry] = []
    for idx, segment in enumerate(text_segments, start=1):
        flac_path = Path(expected_paths[idx - 1])
        exists = flac_path.is_file()
        if segment.voice_id not in reference_audio_cache:
            reference_audio_cache[segment.voice_id] = (
                reference_audio_abs_path(db, db_path, job, segment.voice_id) is not None
            )
        open_event = open_deletes.get(idx)
        status = "ok" if exists else ("pending_regen" if open_event is not None else "missing")
        sentences.append(
            SentenceEntry(
                sentence_index=idx,
                file_name=flac_path.relative_to(sentence_audio_dir).as_posix(),
                text=segment.text,
                voice_id=segment.voice_id,
                exists=exists,
                status=status,
                description_tag=_read_description_tag(flac_path) if exists else None,
                reference_audio_available=reference_audio_cache[segment.voice_id],
                silence_after_seconds=silence_after[idx - 1],
                review_event_id=open_event.id if open_event is not None else None,
            )
        )

    logger.info(
        "audio_review_playlist_built: job_id=%s transcript_chapter_id=%s sentences=%d missing=%d pending_regen=%d",
        job.id, transcript_chapter.id, len(sentences),
        sum(1 for s in sentences if s.status == "missing"),
        sum(1 for s in sentences if s.status == "pending_regen"),
    )
    return ChapterPlaylist(
        transcript_chapter_id=transcript_chapter.id,
        chapter_basename=chapter_basename,
        sentences=sentences,
    )
