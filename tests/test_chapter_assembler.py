from __future__ import annotations

import shutil
import subprocess

import pytest

from syntrive.adapters import ffmpeg as ffmpeg_adapter
from syntrive.adapters.muxing.chapter_assembler import assemble_chapter, plan_segments_and_silence


class TestPlanSegmentsAndSilence:
    def test_maps_flac_paths_onto_text_segments_and_computes_gaps(self):
        transcript = "a\n‡break‡\nb\n‡pause‡\nc"
        flacs = ["/s1.flac", "/s2.flac", "/s3.flac"]

        segments, silence_between = plan_segments_and_silence(transcript, flacs)

        assert segments == flacs
        assert silence_between == [1.0, 0.5]

    def test_no_gap_between_adjacent_text_segments_with_no_marker(self):
        transcript = "a\nb"
        flacs = ["/s1.flac", "/s2.flac"]

        segments, silence_between = plan_segments_and_silence(transcript, flacs)

        assert silence_between == [0.0]

    def test_consecutive_markers_collapse_to_the_strongest(self):
        flacs = ["/s1.flac", "/s2.flac"]

        for transcript in ("a\n‡break‡\n‡break‡\nb", "a\n‡break‡\n‡pause‡\nb", "a\n‡pause‡\n‡break‡\nb"):
            _, silence_between = plan_segments_and_silence(transcript, flacs)
            assert silence_between == [1.0], transcript

    def test_trailing_marker_with_no_following_text_is_dropped(self, caplog):
        transcript = "a\n‡break‡"
        flacs = ["/s1.flac"]

        segments, silence_between = plan_segments_and_silence(transcript, flacs)

        assert segments == flacs
        assert silence_between == []

    def test_raises_on_segment_count_mismatch(self):
        transcript = "a\nb\nc"

        with pytest.raises(ValueError, match="3 text segments"):
            plan_segments_and_silence(transcript, ["/only_one.flac"])

    def test_sml_durations_override_is_threaded_through(self):
        transcript = "a\n‡pause‡\nb"

        _, silence_between = plan_segments_and_silence(
            transcript, ["/s1.flac", "/s2.flac"], sml_durations_ms={"pause": 250}
        )

        assert silence_between == [0.25]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)
class TestAssembleChapter:
    def _make_tone(self, path, duration_s: float, sample_rate: int = 24000, frequency: int = 440) -> None:
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={duration_s}",
                "-ar", str(sample_rate), "-ac", "1", str(path),
            ],
            capture_output=True, check=True,
        )

    def test_assembles_a_two_sentence_chapter_with_a_gap(self, tmp_path):
        seg1 = tmp_path / "s1.flac"
        seg2 = tmp_path / "s2.flac"
        self._make_tone(seg1, 1.0, frequency=440)
        self._make_tone(seg2, 1.0, frequency=880)
        out = tmp_path / "chapter.flac"
        transcript = "first sentence\n‡break‡\nsecond sentence"

        result = assemble_chapter(transcript, [str(seg1), str(seg2)], str(out))

        assert result.ok is True
        assert out.exists()
        assert ffmpeg_adapter.probe_duration_ms(str(out)) == pytest.approx(3000, abs=150)

    def test_no_synthesizable_segments_fails_cleanly(self, tmp_path):
        result = assemble_chapter("‡break‡", [], str(tmp_path / "out.flac"))

        assert result.ok is False
        assert "no synthesizable text segments" in result.error

    def test_segment_mismatch_fails_before_touching_ffmpeg(self, tmp_path):
        result = assemble_chapter("a\nb", ["/only_one.flac"], str(tmp_path / "out.flac"))

        assert result.ok is False
        assert "text segments" in result.error

    def test_missing_segment_file_reports_failure(self, tmp_path):
        result = assemble_chapter("a", ["/does/not/exist.flac"], str(tmp_path / "out.flac"))

        assert result.ok is False
        assert result.error


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)
class TestChunkedAssembly:
    def _tone(self, path, duration_s: float) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
             "-ar", "24000", "-ac", "1", str(path)],
            capture_output=True, check=True,
        )

    @pytest.mark.parametrize("limit", ["3", "2"])
    def test_chunked_result_keeps_total_duration_including_boundary_silence(self, tmp_path, monkeypatch, limit):
        monkeypatch.setenv("SYNTRIVE_ASSEMBLE_MAX_SEGMENTS_PER_GRAPH", limit)
        segs = []
        for n in range(8):
            p = tmp_path / f"s{n}.flac"
            self._tone(p, 0.5)
            segs.append(str(p))
        transcript = "\n".join(
            f"line {n}" + ("\n‡pause‡" if n in (2, 3) else "") for n in range(8)
        )
        out = tmp_path / "chapter.flac"

        result = assemble_chapter(transcript, segs, str(out))

        assert result.ok is True
        assert ffmpeg_adapter.probe_duration_ms(str(out)) == pytest.approx(5000, abs=250)

    def test_chunk_failure_is_reported_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNTRIVE_ASSEMBLE_MAX_SEGMENTS_PER_GRAPH", "2")
        good = tmp_path / "s0.flac"
        self._tone(good, 0.5)

        result = assemble_chapter(
            "a\nb\nc", [str(good), str(tmp_path / "missing1.flac"), str(tmp_path / "missing2.flac")],
            str(tmp_path / "out.flac"),
        )

        assert result.ok is False
