from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from syntrive.adapters import ffmpeg as ffmpeg_adapter
from syntrive.adapters.muxing import chapter_assembler
from syntrive.adapters.tts import synthesizer
from syntrive.adapters.tts.engine_capabilities import format_provenance_tag
from syntrive.services.cover_watermark import watermark_m4a_cover

logger = logging.getLogger(__name__)

STAGE_SYNTHESIZE = "synthesize"
STAGE_ASSEMBLE = "assemble"
STAGE_ENCODE = "encode"
STAGE_PROBE = "probe"
STAGE_DONE = "done"


@dataclass(frozen=True)
class ChapterAudioRequest:
    transcript_content: str
    chapter_basename: str
    sentence_audio_dir: str
    audiobooks_dir: str
    output_filename: str
    metadata: dict
    cover_path: Optional[str] = None
    sml_durations_ms: Optional[dict] = None
    loudnorm_kwargs: Optional[dict] = None


@dataclass(frozen=True)
class ChapterAudioResult:
    ok: bool
    stage: str = STAGE_DONE
    m4a_path: Optional[str] = None
    duration_ms: Optional[int] = None
    sentence_flac_paths: Tuple[str, ...] = ()
    synthesized_count: int = 0
    skipped_count: int = 0
    error: str = ""
    failed_line_index: Optional[int] = None
    stopped: bool = False


def synthesize_and_assemble_chapter(
    request: ChapterAudioRequest,
    engine,
    *,
    resolve_voice: Callable[[int], Optional[str]],
    on_line: Optional[Callable[[int, str, float], None]] = None,
    on_line_start: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ChapterAudioResult:
    logger.info(
        "chapter_pipeline_start chapter=%s engine=%s out=%s",
        request.chapter_basename, type(engine).__name__, request.output_filename,
    )

    synth_result = synthesizer.synthesize_chapter(
        request.transcript_content,
        request.chapter_basename,
        request.sentence_audio_dir,
        engine,
        resolve_voice=resolve_voice,
        on_line=on_line,
        on_line_start=on_line_start,
        should_stop=should_stop,
    )
    sentence_paths = tuple(synth_result.sentence_flac_paths)
    if not synth_result.ok:
        logger.error(
            "chapter_pipeline_synthesize_failed chapter=%s line_idx=%s error=%s",
            request.chapter_basename, synth_result.failed_line_index, synth_result.error,
        )
        return ChapterAudioResult(
            ok=False,
            stage=STAGE_SYNTHESIZE,
            sentence_flac_paths=sentence_paths,
            synthesized_count=synth_result.synthesized_count,
            skipped_count=synth_result.skipped_count,
            error=synth_result.error,
            failed_line_index=synth_result.failed_line_index,
            stopped=synth_result.stopped,
        )

    audiobooks_dir = Path(request.audiobooks_dir)
    audiobooks_dir.mkdir(parents=True, exist_ok=True)
    final_m4a_path = audiobooks_dir / request.output_filename

    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch_flac = str(Path(scratch_dir) / f"{request.chapter_basename}_merged.flac")
        assemble_result = chapter_assembler.assemble_chapter(
            request.transcript_content,
            list(sentence_paths),
            scratch_flac,
            sml_durations_ms=request.sml_durations_ms,
            loudnorm_kwargs=request.loudnorm_kwargs,
        )
        if not assemble_result.ok:
            logger.error(
                "chapter_pipeline_assemble_failed chapter=%s segments=%d error=%s",
                request.chapter_basename, len(sentence_paths), assemble_result.error,
            )
            return ChapterAudioResult(
                ok=False,
                stage=STAGE_ASSEMBLE,
                sentence_flac_paths=sentence_paths,
                synthesized_count=synth_result.synthesized_count,
                skipped_count=synth_result.skipped_count,
                error=assemble_result.error,
            )

        description_tag = format_provenance_tag(engine.engine_name, engine.options.get("speed"))
        metadata = {**request.metadata, "description": description_tag}
        logger.info(
            "chapter_pipeline_description_tag chapter=%s engine=%s description=%s",
            request.chapter_basename, engine.engine_name, description_tag,
        )
        m4a_result = ffmpeg_adapter.to_m4a(
            scratch_flac,
            str(final_m4a_path),
            metadata=metadata,
            cover_path=request.cover_path,
        )
        if not m4a_result.ok:
            logger.error(
                "chapter_pipeline_encode_failed chapter=%s out=%s error=%s",
                request.chapter_basename, final_m4a_path, m4a_result.error,
            )
            return ChapterAudioResult(
                ok=False,
                stage=STAGE_ENCODE,
                sentence_flac_paths=sentence_paths,
                synthesized_count=synth_result.synthesized_count,
                skipped_count=synth_result.skipped_count,
                error=m4a_result.error,
            )

    if request.cover_path:
        mark = watermark_m4a_cover(final_m4a_path)
        logger.info("chapter_pipeline_cover_watermark chapter=%s status=%s error=%s",
                    request.chapter_basename, mark.status, mark.error or "-")

    duration_ms = ffmpeg_adapter.probe_duration_ms(str(final_m4a_path))
    if not duration_ms:
        error = f"chapter m4a failed validation (unreadable/zero duration): {final_m4a_path}"
        logger.error("chapter_pipeline_probe_failed chapter=%s error=%s", request.chapter_basename, error)
        return ChapterAudioResult(
            ok=False,
            stage=STAGE_PROBE,
            m4a_path=str(final_m4a_path),
            sentence_flac_paths=sentence_paths,
            synthesized_count=synth_result.synthesized_count,
            skipped_count=synth_result.skipped_count,
            error=error,
        )

    logger.info(
        "chapter_pipeline_ok chapter=%s m4a=%s duration_ms=%d synthesized=%d skipped=%d",
        request.chapter_basename, final_m4a_path, duration_ms,
        synth_result.synthesized_count, synth_result.skipped_count,
    )
    return ChapterAudioResult(
        ok=True,
        stage=STAGE_DONE,
        m4a_path=str(final_m4a_path),
        duration_ms=duration_ms,
        sentence_flac_paths=sentence_paths,
        synthesized_count=synth_result.synthesized_count,
        skipped_count=synth_result.skipped_count,
    )
