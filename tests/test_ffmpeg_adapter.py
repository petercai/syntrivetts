from __future__ import annotations

import shutil
import subprocess

import pytest

from syntrive.adapters import ffmpeg as fa

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)


def _make_tone(path, duration_s: float, sample_rate: int = 24000, frequency: int = 440) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={duration_s}",
            "-ar", str(sample_rate), "-ac", "1", str(path),
        ],
        capture_output=True, check=True,
    )


def _make_solid_color_image(path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1", "-frames:v", "1", str(path)],
        capture_output=True, check=True,
    )


class TestProbeDurationMs:
    def test_returns_duration_for_a_real_file(self, tmp_path):
        seg = tmp_path / "seg.flac"
        _make_tone(seg, 1.0)

        assert fa.probe_duration_ms(str(seg)) == pytest.approx(1000, abs=50)

    def test_returns_none_for_a_missing_file(self):
        assert fa.probe_duration_ms("/does/not/exist.flac") is None


class TestConcatAudio:
    def test_concatenates_two_segments_with_silence_between(self, tmp_path):
        seg1 = tmp_path / "seg1.flac"
        seg2 = tmp_path / "seg2.flac"
        _make_tone(seg1, 1.0, frequency=440)
        _make_tone(seg2, 1.0, frequency=880)
        out = tmp_path / "concat.flac"

        result = fa.concat_audio([str(seg1), str(seg2)], str(out), silence_between=[0.5])

        assert result.ok is True
        assert out.exists()
        assert fa.probe_duration_ms(str(out)) == pytest.approx(2500, abs=100)

    def test_rejects_mismatched_silence_between_length(self, tmp_path):
        seg1 = tmp_path / "seg1.flac"
        _make_tone(seg1, 1.0)

        result = fa.concat_audio([str(seg1)], str(tmp_path / "out.flac"), silence_between=[0.5, 0.5])

        assert result.ok is False
        assert "silence_between" in result.error

    def test_rejects_empty_segments(self, tmp_path):
        result = fa.concat_audio([], str(tmp_path / "out.flac"), silence_between=[])

        assert result.ok is False
        assert "segments" in result.error

    def test_reports_failure_for_an_unreadable_segment(self, tmp_path):
        result = fa.concat_audio(
            ["/does/not/exist.flac"], str(tmp_path / "out.flac"), silence_between=[]
        )

        assert result.ok is False
        assert result.error

    def test_reports_failure_for_a_zero_duration_segment(self, tmp_path):
        empty_seg = tmp_path / "empty.flac"
        empty_seg.write_bytes(b"")

        result = fa.concat_audio(
            [str(empty_seg)], str(tmp_path / "out.flac"), silence_between=[]
        )

        assert result.ok is False
        assert "zero or unreadable duration" in result.error


class TestLoudnorm:
    def test_produces_a_normalized_file_of_matching_duration(self, tmp_path):
        seg = tmp_path / "seg.flac"
        _make_tone(seg, 1.0)
        out = tmp_path / "normalized.flac"

        result = fa.loudnorm(str(seg), str(out))

        assert result.ok is True
        assert out.exists()
        assert fa.probe_duration_ms(str(out)) == pytest.approx(1000, abs=100)

    def test_reports_failure_for_a_missing_input(self, tmp_path):
        result = fa.loudnorm("/does/not/exist.flac", str(tmp_path / "out.flac"))

        assert result.ok is False
        assert result.error


class TestStretchTempo:
    def test_speed_greater_than_one_produces_roughly_half_the_duration(self, tmp_path):
        seg = tmp_path / "seg.flac"
        _make_tone(seg, 2.0)
        out = tmp_path / "stretched.flac"

        result = fa.stretch_tempo(str(seg), str(out), speed=2.0)

        assert result.ok is True
        assert out.exists()
        assert fa.probe_duration_ms(str(out)) == pytest.approx(1000, abs=100)

    def test_speed_less_than_one_produces_roughly_double_the_duration(self, tmp_path):
        seg = tmp_path / "seg.flac"
        _make_tone(seg, 1.0)
        out = tmp_path / "stretched.flac"

        result = fa.stretch_tempo(str(seg), str(out), speed=0.5)

        assert result.ok is True
        assert out.exists()
        assert fa.probe_duration_ms(str(out)) == pytest.approx(2000, abs=150)

    def test_speed_one_is_a_no_op_and_preserves_duration(self, tmp_path):
        seg = tmp_path / "seg.flac"
        _make_tone(seg, 1.0)
        out = tmp_path / "unchanged.flac"

        result = fa.stretch_tempo(str(seg), str(out), speed=1.0)

        assert result.ok is True
        assert out.exists()
        assert fa.probe_duration_ms(str(out)) == pytest.approx(1000, abs=50)

    def test_reports_failure_for_a_missing_input(self, tmp_path):
        result = fa.stretch_tempo("/does/not/exist.flac", str(tmp_path / "out.flac"), speed=1.5)

        assert result.ok is False
        assert result.error

    def test_speed_one_reports_failure_for_a_missing_input_instead_of_raising(self, tmp_path):
        result = fa.stretch_tempo("/does/not/exist.flac", str(tmp_path / "out.flac"), speed=1.0)

        assert result.ok is False
        assert result.error


class TestToM4a:
    def test_embeds_metadata_and_produces_a_valid_m4a(self, tmp_path):
        seg = tmp_path / "seg.flac"
        _make_tone(seg, 1.0)
        out = tmp_path / "chapter.m4a"

        result = fa.to_m4a(
            str(seg), str(out),
            metadata={"title": "Test Chapter", "artist": "Test Author", "album": "Test Book", "track": "1"},
        )

        assert result.ok is True
        assert out.exists()
        assert fa.probe_duration_ms(str(out)) == pytest.approx(1000, abs=100)

        probed = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format_tags", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout
        assert "title=Test Chapter" in probed
        assert "artist=Test Author" in probed
        assert "album=Test Book" in probed

    def test_embeds_cover_art_as_an_attached_video_stream(self, tmp_path):
        seg = tmp_path / "seg.flac"
        cover = tmp_path / "cover.jpg"
        _make_tone(seg, 1.0)
        _make_solid_color_image(cover)
        out = tmp_path / "chapter_with_cover.m4a"

        result = fa.to_m4a(str(seg), str(out), metadata={"title": "Has Cover"}, cover_path=str(cover))

        assert result.ok is True
        probed = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_type", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout
        assert "audio" in probed
        assert "video" in probed

    def test_reports_failure_for_a_missing_input(self, tmp_path):
        result = fa.to_m4a("/does/not/exist.flac", str(tmp_path / "out.m4a"), metadata={})

        assert result.ok is False
        assert result.error
