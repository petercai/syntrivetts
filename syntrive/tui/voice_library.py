from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def scan_voice_library(
    voices_root: str,
    lang_filter: Optional[str] = None,
) -> list[dict]:
    voices_path = Path(voices_root)
    if not voices_path.is_dir():
        logger.warning("scan_voice_library: voices_root not found: %s", voices_root)
        return []

    results = []
    for wav in sorted(voices_path.rglob("*.wav")):
        parts = wav.relative_to(voices_path).parts
        if len(parts) < 2:
            logger.debug("scan_voice_library: skip_no_lan: %s", wav)
            continue
        lan = parts[0]
        if lang_filter and lan != lang_filter:
            continue
        desc_parts = list(parts[1:-1]) + [wav.stem]
        description = "-".join(desc_parts)
        results.append({
            "path":        str(wav),
            "filename":    wav.name,
            "lan":         lan,
            "description": description,
        })

    logger.debug(
        "scan_voice_library: root=%s lang_filter=%s found=%d",
        voices_root, lang_filter, len(results),
    )
    return results


def play_audio(wav_path: str) -> None:
    if not wav_path.lower().endswith(".wav"):
        logger.warning(
            "play_audio_format: only WAV supported, got path=%s", wav_path
        )
        print("[WARN] Only WAV files are supported for in-TUI playback.")
        return

    if not Path(wav_path).is_file():
        logger.warning("play_audio_missing: file not found path=%s", wav_path)
        print(f"[WARN] Audio file not found: {wav_path}")
        return

    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play()
        logger.debug("play_audio_pygame: path=%s", wav_path)
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        return
    except ImportError:
        pass

    try:
        import sounddevice as sd
        import soundfile as sf
        data, samplerate = sf.read(wav_path)
        logger.debug(
            "play_audio_sounddevice: path=%s samplerate=%d", wav_path, samplerate
        )
        sd.play(data, samplerate)
        sd.wait()
        return
    except ImportError:
        pass

    logger.warning(
        "play_audio_unavailable: no in-TUI audio library found path=%s", wav_path
    )
    print("[WARN] No audio library available.")
    print("       Install one of:  pip install pygame   OR   pip install sounddevice soundfile")
