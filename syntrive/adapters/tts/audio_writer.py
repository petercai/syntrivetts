from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from syntrive.adapters import ffmpeg

logger = logging.getLogger(__name__)

_FLAC_SUBTYPE = "PCM_16"

DEFAULT_SILENCE_THRESHOLD = 0.003
DEFAULT_TRIM_BUFFER_SECONDS = 0.004

_ORPHAN_FRAME_SECONDS = 0.025
_ORPHAN_FLOOR_DB = -60.0
_ORPHAN_MIN_GAP_SECONDS = 0.4
_ORPHAN_MAX_SECONDS = 0.6
_ORPHAN_MIN_REL_DB = -4.0


def _frame_levels_db(waveform: np.ndarray, frame_samples: int) -> np.ndarray:
    frame_count = waveform.shape[0] // frame_samples
    if frame_count == 0:
        return np.empty(0, dtype=np.float64)
    frames = waveform[: frame_count * frame_samples].astype(np.float64).reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def drop_tail_orphan(waveform: np.ndarray, sample_rate: int) -> tuple:
    frame_samples = max(int(_ORPHAN_FRAME_SECONDS * sample_rate), 1)
    levels = _frame_levels_db(waveform, frame_samples)
    active = levels > _ORPHAN_FLOOR_DB
    active_idx = np.flatnonzero(active)
    if active_idx.size == 0:
        return waveform, None

    run_end = int(active_idx[-1])
    run_start = run_end
    while run_start > 0 and active[run_start - 1]:
        run_start -= 1

    earlier = np.flatnonzero(active[:run_start])
    if earlier.size == 0:
        return waveform, None

    gap_s = (run_start - int(earlier[-1]) - 1) * _ORPHAN_FRAME_SECONDS
    orphan_s = (run_end - run_start + 1) * _ORPHAN_FRAME_SECONDS
    rel_db = float(levels[run_start : run_end + 1].mean() - np.median(levels[active]))
    if gap_s < _ORPHAN_MIN_GAP_SECONDS or orphan_s > _ORPHAN_MAX_SECONDS or rel_db > _ORPHAN_MIN_REL_DB:
        return waveform, None

    keep = (int(earlier[-1]) + 1) * frame_samples
    info = {
        "gap_s": gap_s,
        "orphan_s": orphan_s,
        "rel_db": rel_db,
        "dropped_s": (waveform.shape[0] - keep) / sample_rate,
        "kept_s": keep / sample_rate,
    }
    return waveform[:keep], info


def sentence_audio_filename(chapter_basename: str, sentence_index: int) -> str:
    return f"{chapter_basename}_s{sentence_index:04d}.flac"


def sentence_audio_chapter_dir(sentence_audio_dir: str, chapter_basename: str) -> Path:
    return Path(sentence_audio_dir) / chapter_basename


def sentence_audio_path(sentence_audio_dir: str, chapter_basename: str, sentence_index: int) -> Path:
    return sentence_audio_chapter_dir(sentence_audio_dir, chapter_basename) / sentence_audio_filename(
        chapter_basename, sentence_index
    )


def trim_tail_silence(
    waveform: np.ndarray,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    buffer_seconds: float = DEFAULT_TRIM_BUFFER_SECONDS,
    sample_rate: int = 24000,
) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError(f"trim_tail_silence expects a 1D (mono) array, got shape {tuple(waveform.shape)}")

    non_silent = np.flatnonzero(np.abs(waveform) > silence_threshold)
    if non_silent.size == 0:
        return waveform[:0]

    buffer_samples = int(buffer_seconds * sample_rate)
    start = max(int(non_silent[0]) - buffer_samples, 0)
    end = min(int(non_silent[-1]) + buffer_samples, waveform.shape[0])
    return waveform[start:end]


def write_sentence_audio(
    raw_audio_path: str,
    sentence_audio_dir: str,
    chapter_basename: str,
    sentence_index: int,
    *,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    buffer_seconds: float = DEFAULT_TRIM_BUFFER_SECONDS,
    speed: Optional[float] = None,
    description_tag: Optional[str] = None,
) -> Optional[str]:
    with contextlib.ExitStack() as stack:
        load_path = raw_audio_path
        if speed is not None and speed != 1.0:
            scratch_dir = stack.enter_context(tempfile.TemporaryDirectory())
            stretched_path = str(Path(scratch_dir) / "stretched.wav")
            stretch_result = ffmpeg.stretch_tempo(raw_audio_path, stretched_path, speed=speed)
            if stretch_result.ok:
                load_path = stretched_path
            else:
                logger.warning(
                    "write_sentence_audio_speed_stretch_failed: raw_audio_path=%s speed=%s error=%s "
                    "-- falling back to unstretched audio",
                    raw_audio_path, speed, stretch_result.error,
                )

        try:
            data, sample_rate = sf.read(load_path, dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001 -- any load failure is a recoverable-line failure, not a crash
            logger.error("write_sentence_audio_load_failed: raw_audio_path=%s error=%s", load_path, exc)
            return None

        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        mono_clean, orphan_info = drop_tail_orphan(mono, sample_rate)
        if orphan_info is not None:
            logger.info(
                "write_sentence_audio_tail_orphan_dropped: raw_audio_path=%s gap_s=%.2f orphan_s=%.2f "
                "rel_db=%.1f dropped_s=%.2f kept_s=%.2f",
                raw_audio_path, orphan_info["gap_s"], orphan_info["orphan_s"], orphan_info["rel_db"],
                orphan_info["dropped_s"], orphan_info["kept_s"],
            )
        trimmed = trim_tail_silence(mono_clean, silence_threshold, buffer_seconds, sample_rate)

        if trimmed.size == 0 and mono.size > 0:
            logger.error(
                "write_sentence_audio_no_audible_content: raw_audio_path=%s samples=%d "
                "silence_threshold=%.4f -- entire clip is below threshold, refusing to write it",
                raw_audio_path, mono.size, silence_threshold,
            )
            return None

        out_dir = sentence_audio_chapter_dir(sentence_audio_dir, chapter_basename)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / sentence_audio_filename(chapter_basename, sentence_index)

        try:
            sf.write(str(out_path), np.clip(trimmed, -1.0, 1.0), sample_rate, format="FLAC", subtype=_FLAC_SUBTYPE)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as a failed line, not a crash
            logger.error("write_sentence_audio_save_failed: out_path=%s error=%s", out_path, exc)
            return None

        if description_tag is not None:
            try:
                from mutagen.flac import FLAC

                tags = FLAC(str(out_path))
                tags["DESCRIPTION"] = description_tag
                tags.save()
            except Exception as exc:  # noqa: BLE001 -- a tagging failure must not lose an otherwise-valid FLAC
                logger.warning(
                    "write_sentence_audio_description_tag_failed: out_path=%s description_tag=%s error=%s",
                    out_path, description_tag, exc,
                )

        logger.info(
            "write_sentence_audio_ok: out_path=%s sample_rate=%d duration_seconds=%.3f",
            out_path, sample_rate, trimmed.shape[0] / sample_rate,
        )
        return str(out_path)
