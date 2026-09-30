from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from syntrive.adapters.tts import audio_writer
from syntrive.adapters.tts.synthesizer import synthesize_chapter


class _FakeEngine:
    def __init__(self, fail_on_text: str = None, engine_name: str = "cosyvoice", options: dict = None):
        self.calls = []
        self.last_error = ""
        self.fail_on_text = fail_on_text
        self.engine_name = engine_name
        self.options = options or {}

    def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
        self.calls.append({"text": text, "voice": voice, "output_path": output_path})
        if self.fail_on_text is not None and text == self.fail_on_text:
            self.last_error = "worker error: simulated decode failure"
            return None

        sample_rate = 24000
        waveform = torch.full((1, int(0.2 * sample_rate)), 0.4)
        torchaudio.save(output_path, waveform, sample_rate)
        return output_path


class TestSynthesizeChapter:
    def test_synthesizes_every_real_text_segment_in_order(self, tmp_path):
        transcript = "‡voice:0‡\nfirst sentence\n‡break‡\nsecond sentence"
        engine = _FakeEngine()

        result = synthesize_chapter(
            transcript, "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: f"/voices/{voice_id}.wav",
        )

        assert result.ok is True
        assert result.synthesized_count == 2
        assert result.skipped_count == 0
        assert len(result.sentence_flac_paths) == 2
        assert (tmp_path / "ch_0001" / "ch_0001_s0001.flac").exists()
        assert (tmp_path / "ch_0001" / "ch_0001_s0002.flac").exists()
        assert [c["text"] for c in engine.calls] == ["first sentence", "second sentence"]

    def test_should_stop_halts_before_the_next_line_and_the_rerun_resumes(self, tmp_path):
        transcript = "first sentence\nsecond sentence\nthird sentence"
        engine = _FakeEngine()

        stopped = synthesize_chapter(
            transcript, "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
            should_stop=lambda: len(engine.calls) >= 1,
        )

        assert stopped.ok is False and stopped.stopped is True
        assert stopped.synthesized_count == 1 and stopped.failed_line_index is None
        assert [c["text"] for c in engine.calls] == ["first sentence"]
        assert sorted(p.name for p in (tmp_path / "ch_0001").glob("*.flac")) == ["ch_0001_s0001.flac"]

        resumed = synthesize_chapter(transcript, "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None)
        assert resumed.ok is True and resumed.skipped_count == 1 and resumed.synthesized_count == 2

    def test_resolves_voice_per_segment_using_the_active_voice_id(self, tmp_path):
        transcript = "narrator line\n‡voice:5‡\nrole line"
        engine = _FakeEngine()
        resolved_voice_ids = []

        def resolve_voice(voice_id):
            resolved_voice_ids.append(voice_id)
            return f"/voices/{voice_id}.wav"

        synthesize_chapter(transcript, "ch_0001", str(tmp_path), engine, resolve_voice=resolve_voice)

        assert resolved_voice_ids == [0, 5]
        assert engine.calls[0]["voice"] == "/voices/0.wav"
        assert engine.calls[1]["voice"] == "/voices/5.wav"

    def test_resumes_by_skipping_sentences_whose_flac_already_exists(self, tmp_path):
        existing = tmp_path / "ch_0001" / "ch_0001_s0001.flac"
        existing.parent.mkdir(parents=True)
        torchaudio.save(str(existing), torch.zeros(1, 100), 24000)
        transcript = "already done\nnot yet done"
        engine = _FakeEngine()

        result = synthesize_chapter(
            transcript, "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is True
        assert result.skipped_count == 1
        assert result.synthesized_count == 1
        assert [c["text"] for c in engine.calls] == ["not yet done"]

    def test_aborts_on_first_failing_line_and_keeps_prior_flacs(self, tmp_path):
        transcript = "good sentence\nbad sentence\nnever reached"
        engine = _FakeEngine(fail_on_text="bad sentence")

        result = synthesize_chapter(
            transcript, "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is False
        assert "sentence 2" in result.error
        assert result.failed_line_index == 2
        assert result.synthesized_count == 1
        assert (tmp_path / "ch_0001" / "ch_0001_s0001.flac").exists()
        assert not (tmp_path / "ch_0001" / "ch_0001_s0002.flac").exists()
        assert [c["text"] for c in engine.calls] == ["good sentence", "bad sentence"]

    def test_on_line_callback_reports_ok_skipped_and_failed_outcomes(self, tmp_path):
        existing = tmp_path / "ch_0001" / "ch_0001_s0001.flac"
        existing.parent.mkdir(parents=True)
        torchaudio.save(str(existing), torch.zeros(1, 100), 24000)
        transcript = "already done\nnot yet done\nbad sentence"
        engine = _FakeEngine(fail_on_text="bad sentence")
        events = []

        synthesize_chapter(
            transcript, "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
            on_line=lambda index, outcome, ms: events.append((index, outcome)),
        )

        assert events == [(1, "skipped"), (2, "ok"), (3, "failed")]

    def test_on_line_callback_is_optional_and_defaults_to_a_no_op(self, tmp_path):
        result = synthesize_chapter(
            "just one sentence", "ch_0001", str(tmp_path), _FakeEngine(), resolve_voice=lambda voice_id: None,
        )
        assert result.ok is True

    def test_chapter_with_no_text_segments_fails_cleanly(self, tmp_path):
        engine = _FakeEngine()

        result = synthesize_chapter(
            "‡break‡\n‡pause‡", "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is False
        assert "no synthesizable text segments" in result.error
        assert engine.calls == []


class TestSpeedDecisionPoint:
    def _capture_write_sentence_audio_speed(self, monkeypatch):
        captured_speeds = []

        def fake_write_sentence_audio(raw_path, sentence_audio_dir, chapter_basename, index, **kwargs):
            captured_speeds.append(kwargs.get("speed"))
            out_dir = Path(sentence_audio_dir) / chapter_basename
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / audio_writer.sentence_audio_filename(chapter_basename, index)
            out_path.write_bytes(b"fake-flac")
            return str(out_path)

        monkeypatch.setattr(audio_writer, "write_sentence_audio", fake_write_sentence_audio)
        return captured_speeds

    def _capture_write_sentence_audio_description_tag(self, monkeypatch):
        captured_tags = []

        def fake_write_sentence_audio(raw_path, sentence_audio_dir, chapter_basename, index, **kwargs):
            captured_tags.append(kwargs.get("description_tag"))
            out_dir = Path(sentence_audio_dir) / chapter_basename
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / audio_writer.sentence_audio_filename(chapter_basename, index)
            out_path.write_bytes(b"fake-flac")
            return str(out_path)

        monkeypatch.setattr(audio_writer, "write_sentence_audio", fake_write_sentence_audio)
        return captured_tags

    def test_native_engine_gets_no_post_hoc_speed(self, tmp_path, monkeypatch):
        captured_speeds = self._capture_write_sentence_audio_speed(monkeypatch)
        engine = _FakeEngine(engine_name="cosyvoice", options={"speed": 1.5})

        result = synthesize_chapter(
            "only sentence", "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is True
        assert captured_speeds == [None]

    def test_non_native_engine_gets_the_configured_speed(self, tmp_path, monkeypatch):
        captured_speeds = self._capture_write_sentence_audio_speed(monkeypatch)
        engine = _FakeEngine(engine_name="voxcpm", options={"speed": 1.5})

        result = synthesize_chapter(
            "only sentence", "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is True
        assert captured_speeds == [1.5]

    def test_qwen3tts_also_gets_the_configured_speed(self, tmp_path, monkeypatch):
        captured_speeds = self._capture_write_sentence_audio_speed(monkeypatch)
        engine = _FakeEngine(engine_name="qwen3tts", options={"speed": 0.8})

        result = synthesize_chapter(
            "only sentence", "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is True
        assert captured_speeds == [0.8]


class TestDescriptionTag:
    def _capture(self, monkeypatch):
        return TestSpeedDecisionPoint()._capture_write_sentence_audio_description_tag(monkeypatch)

    def test_native_engine_still_gets_the_configured_speed_in_its_tag(self, tmp_path, monkeypatch):
        captured_tags = self._capture(monkeypatch)
        engine = _FakeEngine(engine_name="cosyvoice", options={"speed": 1.5})

        result = synthesize_chapter(
            "only sentence", "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is True
        assert captured_tags == ["syntrive (cosy-1.5)"]

    def test_non_native_engine_gets_its_configured_speed_in_its_tag(self, tmp_path, monkeypatch):
        captured_tags = self._capture(monkeypatch)
        engine = _FakeEngine(engine_name="voxcpm", options={"speed": 0.9})

        result = synthesize_chapter(
            "only sentence", "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is True
        assert captured_tags == ["syntrive (vox-0.9)"]

    def test_no_configured_speed_defaults_the_tag_to_1_0(self, tmp_path, monkeypatch):
        captured_tags = self._capture(monkeypatch)
        engine = _FakeEngine(engine_name="qwen3tts", options={})

        result = synthesize_chapter(
            "only sentence", "ch_0001", str(tmp_path), engine, resolve_voice=lambda voice_id: None,
        )

        assert result.ok is True
        assert captured_tags == ["syntrive (qwen-1.0)"]


def test_failure_error_includes_the_engine_reason(tmp_path):
    engine = _FakeEngine(fail_on_text="3")

    result = synthesize_chapter("3", "ch_0007", str(tmp_path), engine, resolve_voice=lambda voice_id: None)

    assert result.ok is False
    assert result.failed_line_index == 1
    assert "simulated decode failure" in result.error
