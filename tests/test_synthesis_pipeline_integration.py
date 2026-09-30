from __future__ import annotations

import shutil

import pytest
import torch
import torchaudio

import ffmpeg as ffmpeg_lib

from syntrive.adapters import ffmpeg as ffmpeg_adapter
from syntrive.adapters.muxing.chapter_assembler import assemble_chapter
from syntrive.adapters.tts.engine_capabilities import format_provenance_tag
from syntrive.adapters.tts.synthesizer import synthesize_chapter
from syntrive.orchestration.chapter_pipeline import ChapterAudioRequest, synthesize_and_assemble_chapter

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)


class _FakeEngine:
    def __init__(self, sample_rate=24000, tone_duration_s=0.3):
        self.sample_rate = sample_rate
        self.tone_duration_s = tone_duration_s
        self.calls = 0
        self.engine_name = "cosyvoice"
        self.options = {}

    def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
        self.calls += 1
        waveform = torch.full((1, int(self.tone_duration_s * self.sample_rate)), 0.4)
        torchaudio.save(output_path, waveform, self.sample_rate)
        return output_path


class TestSynthesisPipelineIntegration:
    def test_full_pipeline_produces_a_valid_tagged_m4a(self, tmp_path):
        transcript = (
            "‡voice:0‡\n"
            "PART one ‡pause‡ Section one\n"
            "‡break‡\n"
            "It was a bright cold day."
        )
        sentence_dir = tmp_path / "sentence_audio"
        engine = _FakeEngine(tone_duration_s=0.3)

        synth_result = synthesize_chapter(
            transcript, "ch_0001", str(sentence_dir), engine, resolve_voice=lambda voice_id: None,
        )
        assert synth_result.ok is True
        assert synth_result.synthesized_count == 3
        assert engine.calls == 3

        merged_flac = tmp_path / "ch_0001_merged.flac"
        assemble_result = assemble_chapter(transcript, synth_result.sentence_flac_paths, str(merged_flac))
        assert assemble_result.ok is True
        assert merged_flac.exists()

        merged_ms = ffmpeg_adapter.probe_duration_ms(str(merged_flac))
        assert merged_ms is not None
        assert merged_ms > 1500

        final_m4a = tmp_path / "ch_0001.m4a"
        m4a_result = ffmpeg_adapter.to_m4a(
            str(merged_flac), str(final_m4a),
            metadata={"title": "Chapter One", "album": "Test Book", "track": "1"},
        )
        assert m4a_result.ok is True
        assert final_m4a.exists()
        assert ffmpeg_adapter.probe_duration_ms(str(final_m4a)) == pytest.approx(merged_ms, abs=200)

    def test_resume_after_partial_synthesis_reuses_existing_flacs_before_assembling(self, tmp_path):
        transcript = "first\n‡break‡\nsecond\n‡break‡\nthird"
        sentence_dir = tmp_path / "sentence_audio"
        engine = _FakeEngine(tone_duration_s=0.2)

        first_pass = synthesize_chapter(
            transcript, "ch_0002", str(sentence_dir), engine, resolve_voice=lambda voice_id: None,
        )
        assert first_pass.ok is True
        assert first_pass.synthesized_count == 3
        calls_after_first_pass = engine.calls

        second_pass = synthesize_chapter(
            transcript, "ch_0002", str(sentence_dir), engine, resolve_voice=lambda voice_id: None,
        )
        assert second_pass.ok is True
        assert second_pass.synthesized_count == 0
        assert second_pass.skipped_count == 3
        assert engine.calls == calls_after_first_pass

        merged_flac = tmp_path / "ch_0002_merged.flac"
        assemble_result = assemble_chapter(transcript, second_pass.sentence_flac_paths, str(merged_flac))
        assert assemble_result.ok is True


class TestChapterPipelineDescriptionTag:
    def test_final_m4a_carries_the_engine_speed_description_tag(self, tmp_path):
        transcript = "‡voice:0‡\nonly sentence here"
        engine = _FakeEngine(tone_duration_s=0.2)
        engine.options = {"speed": 1.25}

        request = ChapterAudioRequest(
            transcript_content=transcript,
            chapter_basename="ch_0001",
            sentence_audio_dir=str(tmp_path / "sentence_audio"),
            audiobooks_dir=str(tmp_path / "audiobooks"),
            output_filename="ch_0001.m4a",
            metadata={"title": "Chapter One", "album": "Test Book"},
        )

        result = synthesize_and_assemble_chapter(request, engine, resolve_voice=lambda voice_id: None)

        assert result.ok is True
        probed = ffmpeg_lib.probe(result.m4a_path)
        expected_tag = format_provenance_tag("cosyvoice", 1.25)
        assert probed["format"]["tags"]["description"] == expected_tag
        assert probed["format"]["tags"]["title"] == "Chapter One"


class TestChapterPipelineCoverWatermark:
    def _request(self, tmp_path, cover_path):
        return ChapterAudioRequest(
            transcript_content="‡voice:0‡\nonly sentence here",
            chapter_basename="ch_0001",
            sentence_audio_dir=str(tmp_path / "sentence_audio"),
            audiobooks_dir=str(tmp_path / "audiobooks"),
            output_filename="ch_0001.m4a",
            metadata={"title": "Chapter One", "album": "Test Book"},
            cover_path=cover_path,
        )

    def test_cover_is_watermarked_after_encoding(self, tmp_path):
        from io import BytesIO

        from mutagen.mp4 import MP4
        from PIL import Image, ImageChops

        from syntrive.services.cover_watermark import MARK_KEY

        cover = tmp_path / "cover.jpg"
        Image.new("RGB", (400, 600), (240, 240, 240)).save(cover, "JPEG")
        result = synthesize_and_assemble_chapter(self._request(tmp_path, str(cover)), _FakeEngine(tone_duration_s=0.2),
                                                 resolve_voice=lambda voice_id: None)
        assert result.ok is True
        tags = MP4(result.m4a_path).tags
        assert MARK_KEY in tags
        embedded = Image.open(BytesIO(bytes(tags["covr"][0]))).convert("RGB")
        diff = ImageChops.difference(Image.open(cover).convert("RGB"), embedded).convert("L")
        assert diff.point(lambda v: 255 if v > 60 else 0).getbbox() is not None

    def test_no_cover_no_watermark(self, tmp_path):
        from mutagen.mp4 import MP4

        from syntrive.services.cover_watermark import MARK_KEY

        result = synthesize_and_assemble_chapter(self._request(tmp_path, None), _FakeEngine(tone_duration_s=0.2),
                                                 resolve_voice=lambda voice_id: None)
        assert result.ok is True
        assert MARK_KEY not in (MP4(result.m4a_path).tags or {})
