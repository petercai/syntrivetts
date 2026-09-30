from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import ffmpeg as ffmpeg_lib

from syntrive.adapters import ffmpeg as ffmpeg_adapter
from syntrive.adapters.text.tts_script import SilenceMarker, TextSegment, parse_tts_script

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEGMENTS_PER_GRAPH = 64
_MAX_SEGMENTS_ENV = "SYNTRIVE_ASSEMBLE_MAX_SEGMENTS_PER_GRAPH"


@dataclass(frozen=True)
class AssembleResult:
    ok: bool
    error: str = ""


def plan_segments_and_silence(
    transcript_content: str,
    sentence_flac_paths: List[str],
    sml_durations_ms: Optional[dict] = None,
):
    tokens = parse_tts_script(transcript_content, sml_durations_ms)
    text_token_count = sum(1 for token in tokens if isinstance(token, TextSegment))
    if text_token_count != len(sentence_flac_paths):
        raise ValueError(
            f"plan_segments_and_silence: transcript has {text_token_count} text segments "
            f"but {len(sentence_flac_paths)} sentence FLAC paths were given"
        )

    segments: List[str] = []
    silence_between: List[float] = []
    pending_silence_ms = 0
    flac_iter = iter(sentence_flac_paths)

    for token in tokens:
        if isinstance(token, SilenceMarker):
            pending_silence_ms = max(pending_silence_ms, token.duration_ms)
            continue
        if segments:
            silence_between.append(pending_silence_ms / 1000)
        segments.append(next(flac_iter))
        pending_silence_ms = 0

    if pending_silence_ms:
        logger.warning(
            "plan_segments_and_silence_trailing_silence_dropped: duration_ms=%d", pending_silence_ms
        )

    return segments, silence_between


def _max_segments_per_graph() -> int:
    raw = os.environ.get(_MAX_SEGMENTS_ENV)
    try:
        value = int(raw) if raw else DEFAULT_MAX_SEGMENTS_PER_GRAPH
    except ValueError:
        logger.warning("assemble_config_invalid: var=%s value=%r", _MAX_SEGMENTS_ENV, raw)
        return DEFAULT_MAX_SEGMENTS_PER_GRAPH
    return max(2, value)


def _reduce_to_limit(
    segments: List[str], silence_between: List[float], scratch_dir: Path, limit: int
) -> Tuple[List[str], List[float], str]:
    level = 0
    while len(segments) > limit:
        next_segments: List[str] = []
        next_silence: List[float] = []
        for start in range(0, len(segments), limit):
            group = segments[start:start + limit]
            group_gaps = silence_between[start:start + len(group) - 1]
            chunk_path = str(scratch_dir / f"chunk_{level}_{start // limit:05d}.flac")
            result = ffmpeg_adapter.concat_audio(group, chunk_path, silence_between=group_gaps)
            if not result.ok:
                logger.error(
                    "assemble_chunk_failed: level=%d start=%d size=%d error=%s",
                    level, start, len(group), result.error,
                )
                return [], [], result.error
            next_segments.append(chunk_path)
            if start + len(group) < len(segments):
                next_silence.append(silence_between[start + len(group) - 1])
        logger.info("assemble_chunk_ok: level=%d in=%d out=%d", level, len(segments), len(next_segments))
        segments, silence_between = next_segments, next_silence
        level += 1
    return segments, silence_between, ""


def assemble_chapter(
    transcript_content: str,
    sentence_flac_paths: List[str],
    out_path: str,
    *,
    sml_durations_ms: Optional[dict] = None,
    loudnorm_kwargs: Optional[dict] = None,
) -> AssembleResult:
    try:
        segments, silence_between = plan_segments_and_silence(
            transcript_content, sentence_flac_paths, sml_durations_ms
        )
    except ValueError as exc:
        logger.error("assemble_chapter_segment_mismatch: error=%s", exc)
        return AssembleResult(ok=False, error=str(exc))

    if not segments:
        return AssembleResult(ok=False, error="assemble_chapter: chapter has no synthesizable text segments")

    loudnorm_kwargs = loudnorm_kwargs or {}
    i = loudnorm_kwargs.get("i", -16.0)
    lra = loudnorm_kwargs.get("lra", 11.0)
    tp = loudnorm_kwargs.get("tp", -1.5)
    denoise_nf = loudnorm_kwargs.get("denoise_nf", -70)
    limit = _max_segments_per_graph()

    with tempfile.TemporaryDirectory() as chunk_dir:
        graph_segments, graph_silence = segments, silence_between
        if len(segments) > limit:
            logger.info("assemble_chapter_chunked: segments=%d max_per_graph=%d", len(segments), limit)
            graph_segments, graph_silence, chunk_error = _reduce_to_limit(
                segments, silence_between, Path(chunk_dir), limit
            )
            if chunk_error:
                return AssembleResult(ok=False, error=chunk_error)

        joined, error = ffmpeg_adapter.build_joined_stream(graph_segments, graph_silence)
        if error is not None:
            logger.error("assemble_chapter_concat_failed: out_path=%s error=%s", out_path, error)
            return AssembleResult(ok=False, error=error)

        try:
            (
                joined.filter("loudnorm", i=i, lra=lra, tp=tp)
                .filter("afftdn", nf=denoise_nf)
                .output(out_path)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except ffmpeg_lib.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if getattr(exc, "stderr", None) else str(exc)
            error_text = stderr.strip()[-2000:]
            logger.error(
                "assemble_chapter_filtergraph_failed: out_path=%s segments=%d error=%s",
                out_path, len(segments), error_text,
            )
            return AssembleResult(ok=False, error=error_text)

    logger.info(
        "assemble_chapter_ok: out_path=%s segments=%d silence_gaps=%d",
        out_path, len(segments), len(silence_between),
    )
    return AssembleResult(ok=True)
