from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from syntrive.adapters.text.tts_script import TextSegment, parse_tts_script
from syntrive.adapters.tts import audio_writer
from syntrive.adapters.tts.engine_capabilities import NATIVE_SPEED_ENGINES, format_provenance_tag

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesizeChapterResult:
    ok: bool
    sentence_flac_paths: List[str] = field(default_factory=list)
    error: str = ""
    synthesized_count: int = 0
    skipped_count: int = 0
    failed_line_index: Optional[int] = None
    stopped: bool = False


def synthesize_chapter(
    transcript_content: str,
    chapter_basename: str,
    sentence_audio_dir: str,
    engine,
    *,
    resolve_voice: Callable[[int], Optional[str]],
    on_line: Optional[Callable[[int, str, float], None]] = None,
    on_line_start: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> SynthesizeChapterResult:
    tokens = parse_tts_script(transcript_content)
    text_segments = [token for token in tokens if isinstance(token, TextSegment)]
    if not text_segments:
        return SynthesizeChapterResult(ok=False, error="chapter has no synthesizable text segments")

    description_tag = format_provenance_tag(engine.engine_name, engine.options.get("speed"))
    logger.info(
        "synthesize_chapter_description_tag: chapter=%s engine=%s description=%s",
        chapter_basename, engine.engine_name, description_tag,
    )

    sentence_paths: List[str] = []
    synthesized_count = 0
    skipped_count = 0

    for index, segment in enumerate(text_segments, start=1):
        line_started = time.monotonic()
        final_path = audio_writer.sentence_audio_path(sentence_audio_dir, chapter_basename, index)
        if final_path.exists():
            sentence_paths.append(str(final_path))
            skipped_count += 1
            if on_line is not None:
                on_line(index, "skipped", (time.monotonic() - line_started) * 1000)
            continue

        if should_stop is not None and should_stop():
            logger.warning(
                "synthesize_chapter_stopped: chapter=%s before_sentence=%d synthesized=%d skipped=%d",
                chapter_basename, index, synthesized_count, skipped_count,
            )
            return SynthesizeChapterResult(
                ok=False, sentence_flac_paths=sentence_paths,
                error=f"stopped before sentence {index} (caller requested stop)",
                synthesized_count=synthesized_count, skipped_count=skipped_count, stopped=True,
            )

        voice_path = resolve_voice(segment.voice_id)

        if on_line_start is not None:
            on_line_start(index, len(text_segments), segment.text)

        with tempfile.TemporaryDirectory() as scratch_dir:
            raw_path = str(Path(scratch_dir) / "raw.wav")
            result_path = engine.synthesize(segment.text, voice=voice_path, output_path=raw_path)
            if result_path is None:
                reason = getattr(engine, "last_error", "") or "no reason reported"
                error = f"synthesis failed at sentence {index}: {segment.text[:60]!r} ({reason})"
                logger.error(
                    "synthesize_chapter_line_failed: chapter=%s sentence_index=%d text_preview=%s reason=%s",
                    chapter_basename, index, segment.text[:60], reason,
                )
                if on_line is not None:
                    on_line(index, "failed", (time.monotonic() - line_started) * 1000)
                return SynthesizeChapterResult(
                    ok=False, sentence_flac_paths=sentence_paths, error=error,
                    synthesized_count=synthesized_count, skipped_count=skipped_count,
                    failed_line_index=index,
                )

            needs_post_hoc_speed = engine.engine_name not in NATIVE_SPEED_ENGINES
            speed_for_write = engine.options.get("speed") if needs_post_hoc_speed else None
            logger.info(
                "tts_speed_applied: engine=%s speed_requested=%s speed_mechanism=%s",
                engine.engine_name,
                engine.options.get("speed", 1.0),
                "native" if not needs_post_hoc_speed else "post_hoc",
            )

            written_path = audio_writer.write_sentence_audio(
                raw_path, sentence_audio_dir, chapter_basename, index,
                speed=speed_for_write, description_tag=description_tag,
            )
            if written_path is None:
                error = f"failed to write sentence audio for sentence {index}"
                if on_line is not None:
                    on_line(index, "failed", (time.monotonic() - line_started) * 1000)
                return SynthesizeChapterResult(
                    ok=False, sentence_flac_paths=sentence_paths, error=error,
                    synthesized_count=synthesized_count, skipped_count=skipped_count,
                    failed_line_index=index,
                )

        sentence_paths.append(written_path)
        synthesized_count += 1
        if on_line is not None:
            on_line(index, "ok", (time.monotonic() - line_started) * 1000)

    logger.info(
        "synthesize_chapter_ok: chapter=%s total=%d synthesized=%d skipped=%d",
        chapter_basename, len(text_segments), synthesized_count, skipped_count,
    )
    return SynthesizeChapterResult(
        ok=True, sentence_flac_paths=sentence_paths,
        synthesized_count=synthesized_count, skipped_count=skipped_count,
    )
