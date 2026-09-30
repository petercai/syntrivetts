from __future__ import annotations

from pathlib import Path

from syntrive.adapters.text.tts_script import (
    DEFAULT_SML_DURATIONS_MS,
    SilenceMarker,
    TextSegment,
    parse_tts_script,
)

_FIXTURE = Path(__file__).resolve().parent.parent / (
    "tmp/PROCESSING-1984-Orwell_George/transcript_text/tts_script/ch_0001.txt"
)


class TestParseTtsScript:
    def test_whole_line_voice_marker_sets_active_voice(self):
        tokens = parse_tts_script("‡voice:2‡\nhello there")

        assert tokens == [TextSegment(text="hello there", voice_id=2)]

    def test_whole_line_break_and_pause_markers(self):
        tokens = parse_tts_script("a\n‡break‡\nb\n‡pause‡\nc")

        assert tokens == [
            TextSegment(text="a", voice_id=0),
            SilenceMarker(duration_ms=1000),
            TextSegment(text="b", voice_id=0),
            SilenceMarker(duration_ms=500),
            TextSegment(text="c", voice_id=0),
        ]

    def test_inline_marker_mid_sentence_splits_into_two_segments(self):
        tokens = parse_tts_script("PART one ‡pause‡ Section one")

        assert tokens == [
            TextSegment(text="PART one", voice_id=0),
            SilenceMarker(duration_ms=500),
            TextSegment(text="Section one", voice_id=0),
        ]

    def test_voice_switch_applies_to_subsequent_segments_only(self):
        tokens = parse_tts_script("narrator line\n‡voice:5‡\nrole line\n‡voice:0‡\nback to narrator")

        assert tokens == [
            TextSegment(text="narrator line", voice_id=0),
            TextSegment(text="role line", voice_id=5),
            TextSegment(text="back to narrator", voice_id=0),
        ]

    def test_consecutive_markers_each_produce_their_own_token(self):
        tokens = parse_tts_script("a\n‡break‡\n‡break‡\nb")

        assert tokens == [
            TextSegment(text="a", voice_id=0),
            SilenceMarker(duration_ms=1000),
            SilenceMarker(duration_ms=1000),
            TextSegment(text="b", voice_id=0),
        ]

    def test_blank_lines_are_ignored(self):
        tokens = parse_tts_script("a\n\n\nb")

        assert tokens == [TextSegment(text="a", voice_id=0), TextSegment(text="b", voice_id=0)]

    def test_empty_content_returns_no_tokens(self):
        assert parse_tts_script("") == []

    def test_sml_durations_override(self):
        tokens = parse_tts_script("‡break‡", sml_durations_ms={"break": 250})

        assert tokens == [SilenceMarker(duration_ms=250)]
        assert DEFAULT_SML_DURATIONS_MS["pause"] == 500
        assert DEFAULT_SML_DURATIONS_MS["break"] == 1000


class TestParseTtsScriptAgainstRealFixture:
    def test_real_fixture_parses_without_error_and_balances_voice_narrator(self):
        if not _FIXTURE.exists():
            import pytest

            pytest.skip(f"fixture not present: {_FIXTURE}")

        content = _FIXTURE.read_text(encoding="utf-8")
        tokens = parse_tts_script(content)

        assert len(tokens) > 0
        text_segments = [t for t in tokens if isinstance(t, TextSegment)]
        assert all(seg.text for seg in text_segments)
