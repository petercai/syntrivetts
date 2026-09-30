from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Union

import ffmpeg

logger = logging.getLogger(__name__)

_STDERR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class Result:
    ok: bool
    error: str = ""


def _decode_stderr(exc: "ffmpeg.Error") -> str:
    raw = exc.stderr if getattr(exc, "stderr", None) else str(exc).encode("utf-8", errors="replace")
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    return text.strip()[-_STDERR_TAIL_CHARS:]


def probe_duration_ms(path: str) -> Optional[int]:
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.Error as exc:
        logger.warning("probe_duration_ms_failed: path=%s error=%s", path, _decode_stderr(exc))
        return None

    duration_s = info.get("format", {}).get("duration")
    if duration_s is None:
        logger.warning("probe_duration_ms_missing_duration: path=%s", path)
        return None
    return round(float(duration_s) * 1000)


def _probe_sample_rate(path: str) -> Optional[int]:
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.Error:
        return None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio" and stream.get("sample_rate"):
            return int(stream["sample_rate"])
    return None


def build_joined_stream(segments: list, silence_between: list):
    if not segments:
        return None, "segments must not be empty"
    if len(silence_between) != len(segments) - 1:
        return None, (
            f"silence_between must have {len(segments) - 1} entries "
            f"for {len(segments)} segments, got {len(silence_between)}"
        )

    first_path = segments[0][0] if isinstance(segments[0], tuple) else segments[0]
    sample_rate = _probe_sample_rate(first_path) or 24000

    streams = []
    for index, segment in enumerate(segments):
        if isinstance(segment, tuple):
            path, trim_head_s, trim_tail_s = segment
            duration_ms = probe_duration_ms(path)
            if not duration_ms:
                return None, f"segment has zero or unreadable duration: {path}"
            end_s = max(duration_ms / 1000 - trim_tail_s, trim_head_s)
            stream = (
                ffmpeg.input(path)
                .audio.filter("atrim", start=trim_head_s, end=end_s)
                .filter("asetpts", "PTS-STARTPTS")
            )
        else:
            duration_ms = probe_duration_ms(segment)
            if not duration_ms:
                return None, f"segment has zero or unreadable duration: {segment}"
            stream = ffmpeg.input(segment).audio
        streams.append(stream)

        if index < len(silence_between) and silence_between[index] > 0:
            silence_stream = ffmpeg.input(
                f"anullsrc=r={sample_rate}:cl=mono", f="lavfi", t=silence_between[index]
            ).audio
            streams.append(silence_stream)

    return ffmpeg.concat(*streams, v=0, a=1), None


def concat_audio(
    segments: list,
    out_path: str,
    *,
    silence_between: list,
) -> Result:
    joined, error = build_joined_stream(segments, silence_between)
    if error is not None:
        return Result(ok=False, error=f"concat_audio: {error}")

    try:
        ffmpeg.output(joined, out_path).overwrite_output().run(
            capture_stdout=True, capture_stderr=True, quiet=True
        )
    except ffmpeg.Error as exc:
        error = _decode_stderr(exc)
        logger.warning("concat_audio_failed: out_path=%s segments=%d error=%s", out_path, len(segments), error)
        return Result(ok=False, error=error)

    logger.info("concat_audio_ok: out_path=%s segments=%d", out_path, len(segments))
    return Result(ok=True)


def loudnorm(
    in_path: str,
    out_path: str,
    *,
    i: float = -16.0,
    lra: float = 11.0,
    tp: float = -1.5,
    denoise_nf: float = -70,
) -> Result:
    try:
        (
            ffmpeg.input(in_path)
            .filter("loudnorm", i=i, lra=lra, tp=tp)
            .filter("afftdn", nf=denoise_nf)
            .output(out_path)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as exc:
        error = _decode_stderr(exc)
        logger.warning("loudnorm_failed: in_path=%s out_path=%s error=%s", in_path, out_path, error)
        return Result(ok=False, error=error)

    logger.info("loudnorm_ok: in_path=%s out_path=%s i=%s lra=%s tp=%s", in_path, out_path, i, lra, tp)
    return Result(ok=True)


def stretch_tempo(in_path: str, out_path: str, *, speed: float) -> Result:
    if speed == 1.0:
        try:
            shutil.copyfile(in_path, out_path)
        except OSError as exc:
            error = str(exc)
            logger.warning(
                "stretch_tempo_failed: in_path=%s out_path=%s error=%s", in_path, out_path, error
            )
            return Result(ok=False, error=error)
        logger.info("stretch_tempo_ok: in_path=%s out_path=%s speed=%s", in_path, out_path, speed)
        return Result(ok=True)

    try:
        (
            ffmpeg.input(in_path)
            .filter("atempo", speed)
            .output(out_path)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as exc:
        error = _decode_stderr(exc)
        logger.warning("stretch_tempo_failed: in_path=%s speed=%s error=%s", in_path, speed, error)
        return Result(ok=False, error=error)

    logger.info("stretch_tempo_ok: in_path=%s out_path=%s speed=%s", in_path, out_path, speed)
    return Result(ok=True)


def to_m4a(
    in_path: str,
    out_path: str,
    *,
    metadata: dict,
    cover_path: Optional[str] = None,
) -> Result:
    args = ["ffmpeg", "-y", "-i", in_path]
    if cover_path:
        args += ["-i", cover_path]
    args += ["-map", "0:a"]
    if cover_path:
        args += ["-map", "1:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    args += ["-c:a", "aac"]
    for key, value in metadata.items():
        args += ["-metadata", f"{key}={value}"]
    args += [out_path]

    try:
        subprocess.run(args, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or "").strip()[-_STDERR_TAIL_CHARS:]
        logger.warning("to_m4a_failed: in_path=%s out_path=%s error=%s", in_path, out_path, error)
        return Result(ok=False, error=error)
    except OSError as exc:
        logger.error("to_m4a_ffmpeg_not_found: error=%s", exc)
        return Result(ok=False, error=str(exc))

    logger.info(
        "to_m4a_ok: in_path=%s out_path=%s metadata_keys=%s has_cover=%s",
        in_path, out_path, list(metadata.keys()), cover_path is not None,
    )
    return Result(ok=True)
