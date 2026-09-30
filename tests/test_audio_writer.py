from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from syntrive.adapters.tts import audio_writer as aw

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class TestSentenceAudioFilename:
    def test_zero_pads_to_four_digits_starting_at_one(self):
        assert aw.sentence_audio_filename("ch_0007", 1) == "ch_0007_s0001.flac"
        assert aw.sentence_audio_filename("ch_0007", 42) == "ch_0007_s0042.flac"


class TestSentenceAudioChapterDir:
    def test_appends_chapter_basename_as_a_subdirectory(self, tmp_path):
        sentence_dir = tmp_path / "sentence_audio"

        result = aw.sentence_audio_chapter_dir(str(sentence_dir), "ch_0007")

        assert result == sentence_dir / "ch_0007"

    def test_sentence_audio_path_nests_the_file_under_the_chapter_dir(self, tmp_path):
        sentence_dir = tmp_path / "sentence_audio"

        result = aw.sentence_audio_path(str(sentence_dir), "ch_0007", 42)

        assert result == sentence_dir / "ch_0007" / "ch_0007_s0042.flac"


class TestTrimTailSilence:
    def _make_signal(self, sample_rate=24000, silence_s=0.2, speech_s=0.5):
        silence = np.zeros(int(silence_s * sample_rate), dtype=np.float32)
        speech = np.full(int(speech_s * sample_rate), 0.5, dtype=np.float32)
        return np.concatenate([silence, speech, silence])

    def test_trims_leading_and_trailing_silence(self):
        sample_rate = 24000
        waveform = self._make_signal(sample_rate)

        trimmed = aw.trim_tail_silence(waveform, sample_rate=sample_rate)

        assert trimmed.shape[0] < waveform.shape[0]
        expected_min = int(0.5 * sample_rate)
        assert trimmed.shape[0] >= expected_min

    def test_keeps_a_small_buffer_around_the_non_silent_region(self):
        sample_rate = 24000
        buffer_seconds = 0.01
        waveform = self._make_signal(sample_rate, silence_s=0.5, speech_s=0.2)

        trimmed = aw.trim_tail_silence(waveform, buffer_seconds=buffer_seconds, sample_rate=sample_rate)

        expected = int(0.22 * sample_rate)
        assert abs(trimmed.shape[0] - expected) < sample_rate * 0.02

    def test_entirely_silent_input_returns_empty_array(self):
        waveform = np.zeros(24000, dtype=np.float32)

        trimmed = aw.trim_tail_silence(waveform, sample_rate=24000)

        assert trimmed.size == 0

    def test_rejects_non_1d_input(self):
        with pytest.raises(ValueError, match="1D"):
            aw.trim_tail_silence(np.zeros((1, 100), dtype=np.float32))


class TestDropTailOrphan:
    SR = 22050

    @staticmethod
    def _tone(seconds, amp, sr=22050):
        t = np.arange(int(seconds * sr)) / sr
        return (amp * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def _clip(self, tail_amp):
        speech = self._tone(0.8, 0.3)
        gap = np.zeros(int(0.7 * self.SR), dtype=np.float32)
        tail = self._tone(0.25, tail_amp)
        return np.concatenate([speech, gap, tail]), speech.shape[0]

    def test_quiet_burst_after_long_silence_is_dropped(self):
        clip, speech_len = self._clip(0.05)

        result, info = aw.drop_tail_orphan(clip, self.SR)

        assert info is not None
        assert info["gap_s"] >= 0.4
        assert abs(result.shape[0] - speech_len) <= int(0.025 * self.SR)

    def test_full_level_last_word_after_long_pause_is_kept(self):
        clip, _ = self._clip(0.3)

        result, info = aw.drop_tail_orphan(clip, self.SR)

        assert info is None
        assert result.shape[0] == clip.shape[0]

    def test_clip_without_internal_gap_is_unchanged(self):
        clip = self._tone(1.5, 0.3)

        result, info = aw.drop_tail_orphan(clip, self.SR)

        assert info is None
        assert result is clip

    def test_short_natural_gap_is_not_treated_as_orphan(self):
        gap = np.zeros(int(0.2 * self.SR), dtype=np.float32)
        clip = np.concatenate([self._tone(0.8, 0.3), gap, self._tone(0.25, 0.05)])

        _, info = aw.drop_tail_orphan(clip, self.SR)

        assert info is None

    def test_all_silent_input_is_unchanged(self):
        clip = np.zeros(self.SR, dtype=np.float32)

        result, info = aw.drop_tail_orphan(clip, self.SR)

        assert info is None
        assert result.shape[0] == clip.shape[0]

    def test_write_sentence_audio_drops_orphan_and_logs(self, tmp_path, caplog):
        clip, speech_len = self._clip(0.05)
        raw = tmp_path / "raw.wav"
        sf.write(str(raw), clip, self.SR)

        with caplog.at_level("INFO"):
            out = aw.write_sentence_audio(str(raw), str(tmp_path / "sa"), "ch_0001", 1)

        info, sr = sf.info(out).duration, self.SR
        assert info < (speech_len / sr) + 0.1
        assert "write_sentence_audio_tail_orphan_dropped" in caplog.text


class TestWriteSentenceAudio:
    def _write_raw_wav(self, tmp_path, sample_rate=24000, silence_s=0.1, speech_s=0.3):
        silence = np.zeros(int(silence_s * sample_rate), dtype=np.float32)
        speech = np.full(int(speech_s * sample_rate), 0.4, dtype=np.float32)
        waveform = np.concatenate([silence, speech, silence])
        raw_path = tmp_path / "raw.wav"
        sf.write(str(raw_path), waveform, sample_rate, subtype="FLOAT")
        return str(raw_path), sample_rate

    def test_writes_a_trimmed_flac_with_correct_filename(self, tmp_path):
        raw_path, sample_rate = self._write_raw_wav(tmp_path)
        sentence_dir = tmp_path / "sentence_audio"

        result = aw.write_sentence_audio(raw_path, str(sentence_dir), "ch_0001", 3)

        assert result == str(sentence_dir / "ch_0001" / "ch_0001_s0003.flac")
        assert (sentence_dir / "ch_0001" / "ch_0001_s0003.flac").exists()

        written, written_sr = sf.read(result, dtype="float32")
        assert written_sr == sample_rate
        raw, _ = sf.read(raw_path, dtype="float32")
        assert written.shape[0] < raw.shape[0]
        info = sf.info(result)
        assert (info.format, info.subtype) == ("FLAC", "PCM_16")

    def test_creates_the_sentence_audio_dir_if_missing(self, tmp_path):
        raw_path, _ = self._write_raw_wav(tmp_path)
        sentence_dir = tmp_path / "does" / "not" / "exist" / "yet"

        result = aw.write_sentence_audio(raw_path, str(sentence_dir), "ch_0001", 1)

        assert result is not None
        assert (sentence_dir / "ch_0001").exists()

    def test_returns_none_for_a_missing_raw_file(self, tmp_path):
        result = aw.write_sentence_audio(
            str(tmp_path / "does_not_exist.wav"), str(tmp_path / "out"), "ch_0001", 1,
        )

        assert result is None

    def test_returns_none_and_logs_error_when_the_whole_clip_has_no_audible_content(self, tmp_path, caplog):
        sample_rate = 24000
        near_silent = np.full(int(0.5 * sample_rate), 0.001, dtype=np.float32)
        raw_path = tmp_path / "raw.wav"
        sf.write(str(raw_path), near_silent, sample_rate, subtype="FLOAT")
        sentence_dir = tmp_path / "sentence_audio"

        with caplog.at_level("ERROR"):
            result = aw.write_sentence_audio(str(raw_path), str(sentence_dir), "ch_0001", 2)

        assert result is None
        assert "write_sentence_audio_no_audible_content" in caplog.text
        assert not (sentence_dir / "ch_0001" / "ch_0001_s0002.flac").exists()

    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not found on PATH")
    def test_speed_stretch_produces_a_shorter_output_than_unstretched(self, tmp_path):
        raw_path, _ = self._write_raw_wav(tmp_path, silence_s=0.05, speech_s=1.0)

        unstretched_path = aw.write_sentence_audio(raw_path, str(tmp_path / "unstretched"), "ch_0001", 1, speed=None)
        stretched_path = aw.write_sentence_audio(raw_path, str(tmp_path / "stretched"), "ch_0001", 1, speed=1.5)

        assert unstretched_path is not None
        assert stretched_path is not None

        unstretched_wave, _ = sf.read(unstretched_path, dtype="float32")
        stretched_wave, _ = sf.read(stretched_path, dtype="float32")
        assert stretched_wave.shape[0] < unstretched_wave.shape[0]

    def test_speed_stretch_failure_falls_back_to_unstretched_audio(self, tmp_path, monkeypatch, caplog):
        raw_path, _ = self._write_raw_wav(tmp_path)
        sentence_dir = tmp_path / "sentence_audio"

        def fake_stretch_tempo(in_path, out_path, *, speed):
            return aw.ffmpeg.Result(ok=False, error="boom")

        monkeypatch.setattr(aw.ffmpeg, "stretch_tempo", fake_stretch_tempo)

        with caplog.at_level("WARNING"):
            result = aw.write_sentence_audio(raw_path, str(sentence_dir), "ch_0001", 1, speed=1.5)

        assert result is not None
        assert Path(result).exists()
        assert "write_sentence_audio_speed_stretch_failed" in caplog.text

    def test_description_tag_is_written_as_a_flac_vorbis_comment(self, tmp_path):
        from mutagen.flac import FLAC

        raw_path, _ = self._write_raw_wav(tmp_path)
        sentence_dir = tmp_path / "sentence_audio"

        result = aw.write_sentence_audio(
            raw_path, str(sentence_dir), "ch_0001", 1, description_tag="syntrive (cosy-1.0)"
        )

        assert result is not None
        assert FLAC(result)["DESCRIPTION"] == ["syntrive (cosy-1.0)"]

    def test_no_description_tag_given_writes_no_description_comment(self, tmp_path):
        from mutagen.flac import FLAC

        raw_path, _ = self._write_raw_wav(tmp_path)
        sentence_dir = tmp_path / "sentence_audio"

        result = aw.write_sentence_audio(raw_path, str(sentence_dir), "ch_0001", 1)

        assert result is not None
        assert "DESCRIPTION" not in FLAC(result)

    def test_description_tag_failure_still_returns_the_written_path(self, tmp_path, monkeypatch, caplog):
        raw_path, _ = self._write_raw_wav(tmp_path)
        sentence_dir = tmp_path / "sentence_audio"

        class _BoomFLAC:
            def __init__(self, path):
                raise OSError("boom")

        monkeypatch.setattr("mutagen.flac.FLAC", _BoomFLAC)

        with caplog.at_level("WARNING"):
            result = aw.write_sentence_audio(
                raw_path, str(sentence_dir), "ch_0001", 1, description_tag="syntrive (cosy-1.0)"
            )

        assert result is not None
        assert Path(result).exists()
        assert "write_sentence_audio_description_tag_failed" in caplog.text

    def test_stereo_input_is_downmixed_to_mono(self, tmp_path):
        sample_rate = 24000
        left = np.full(int(0.3 * sample_rate), 0.6, dtype=np.float32)
        right = np.full(int(0.3 * sample_rate), 0.2, dtype=np.float32)
        raw_path = tmp_path / "stereo.wav"
        sf.write(str(raw_path), np.stack([left, right], axis=1), sample_rate, subtype="FLOAT")

        result = aw.write_sentence_audio(str(raw_path), str(tmp_path / "out"), "ch_0001", 1)

        assert result is not None
        written, _ = sf.read(result, dtype="float32", always_2d=True)
        assert written.shape[1] == 1
        assert abs(float(written.max()) - 0.4) < 0.01

    def test_over_range_samples_saturate_instead_of_wrapping(self, tmp_path):
        sample_rate = 24000
        loud = np.full(int(0.2 * sample_rate), 1.5, dtype=np.float32)
        raw_path = tmp_path / "loud.wav"
        sf.write(str(raw_path), loud, sample_rate, subtype="FLOAT")

        result = aw.write_sentence_audio(str(raw_path), str(tmp_path / "out"), "ch_0001", 1)

        assert result is not None
        written, _ = sf.read(result, dtype="float32")
        assert float(written.min()) > 0.99
