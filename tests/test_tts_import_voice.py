from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import tts_import_voice as irv  # noqa: E402
from syntrive.io import paths  # noqa: E402


def _write_wav(path: Path, *, sample_rate: int = 22050, channels: int = 1, sampwidth: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * 100)


class TestNormalizeLanguage:
    def test_passthrough_iso1(self):
        assert irv.normalize_language("en") == "en"
        assert irv.normalize_language("zh") == "zh"

    def test_iso2_3_mapped(self):
        assert irv.normalize_language("eng") == "en"
        assert irv.normalize_language("zho") == "zh"
        assert irv.normalize_language("chi") == "zh"

    def test_unknown_passes_through_lowercased(self):
        assert irv.normalize_language("FR") == "fr"


class TestParseTagsFromParts:
    def test_gender_and_generation_matched(self):
        result = irv.parse_tags_from_parts(("adult", "male"))
        assert result == {"gender": "male", "generation": "adult"}

    def test_no_match_returns_none(self):
        result = irv.parse_tags_from_parts(("misc",))
        assert result == {"gender": None, "generation": None}

    def test_case_insensitive(self):
        result = irv.parse_tags_from_parts(("Elder", "Female"))
        assert result == {"gender": "female", "generation": "elder"}


class TestReadWavParams:
    def test_reads_real_wav_header(self, tmp_path: Path):
        wav = tmp_path / "sample.wav"
        _write_wav(wav, sample_rate=16000, channels=2, sampwidth=2)
        params = irv.read_wav_params(wav)
        assert params == {"sample_rate": 16000, "bit_depth": 16, "channels": 2, "duration_seconds": 0.0}

    def test_duration_seconds_computed_from_frame_count_and_rate(self, tmp_path: Path):
        wav = tmp_path / "one_second.wav"
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        params = irv.read_wav_params(wav)
        assert params["duration_seconds"] == 1.0

    def test_missing_file_returns_none_triplet(self, tmp_path: Path):
        params = irv.read_wav_params(tmp_path / "missing.wav")
        assert params == {
            "sample_rate": None, "bit_depth": None, "channels": None, "duration_seconds": None,
        }

    def test_wave_format_extensible_falls_back_to_soundfile(self, tmp_path: Path):
        import numpy as np
        import soundfile as sf

        wav = tmp_path / "extensible.wav"
        sf.write(str(wav), np.zeros(1600, dtype="int16"), 16000, subtype="PCM_16", format="WAVEX")

        if sys.version_info < (3, 12):
            with pytest.raises(Exception):
                wave.open(str(wav), "rb")

        params = irv.read_wav_params(wav)
        assert params == {"sample_rate": 16000, "bit_depth": 16, "channels": 1, "duration_seconds": 0.1}


class TestPathRoundTrip:
    def test_under_repo_root_is_relative(self):
        abs_path = irv._REPO_ROOT / "voices" / "en" / "adult" / "foo.wav"
        stored = irv.to_repo_relative_or_abs(abs_path)
        assert not Path(stored).is_absolute()
        assert stored == "voices/en/adult/foo.wav"
        assert irv.resolve_stored_path(stored) == abs_path.resolve()

    def test_outside_repo_root_is_absolute(self, tmp_path: Path):
        abs_path = tmp_path / "external" / "bar.wav"
        stored = irv.to_repo_relative_or_abs(abs_path)
        assert Path(stored).is_absolute()
        assert irv.resolve_stored_path(stored) == abs_path.resolve()

    def test_stored_path_is_always_posix_style_on_every_os(self, tmp_path: Path):
        under_root = irv.to_repo_relative_or_abs(irv._REPO_ROOT / "voices" / "en" / "foo.wav")
        outside_root = irv.to_repo_relative_or_abs(tmp_path / "external" / "bar.wav")
        assert "\\" not in under_root
        assert "\\" not in outside_root


class TestScanVoicesDir:
    def test_parses_lang_gender_generation_from_directory_convention(self, tmp_path: Path):
        voices_root = tmp_path / "voices"
        _write_wav(voices_root / "zh" / "adult" / "male" / "yunxi.wav")
        _write_wav(voices_root / "en" / "child" / "female" / "amy.wav")

        candidates = irv.scan_voices_dir(voices_root)

        by_name = {c.name: c for c in candidates}
        assert by_name["yunxi"].language == "zh"
        assert by_name["yunxi"].gender == "male"
        assert by_name["yunxi"].extra_tags == ("adult",)
        assert by_name["yunxi"].fine_tuned is False
        assert by_name["amy"].language == "en"
        assert by_name["amy"].gender == "female"
        assert by_name["amy"].extra_tags == ("child",)

    def test_skips_wav_with_no_lang_subdir(self, tmp_path: Path):
        voices_root = tmp_path / "voices"
        _write_wav(voices_root / "orphan.wav")
        assert irv.scan_voices_dir(voices_root) == []

    def test_skips_reserved_double_underscore_topdir(self, tmp_path: Path):
        voices_root = tmp_path / "voices"
        _write_wav(voices_root / "__sessions" / "voice-abc123" / "eng" / "clip.wav")
        _write_wav(voices_root / "__bark" / "en_speaker_4" / "en_speaker_4.wav")
        assert irv.scan_voices_dir(voices_root) == []

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert irv.scan_voices_dir(tmp_path / "does_not_exist") == []


class TestScanFinetunedDir:
    def test_fine_tuned_flag_and_language_parsed_from_path(self, tmp_path: Path):
        base = tmp_path / "fine_tuned_models"
        ckpt_dir = base / "snapshots" / "abc123" / "xtts-v2" / "eng"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"\x00")
        _write_wav(ckpt_dir / "ref.wav")

        candidates = irv.scan_finetuned_dir(base)

        assert len(candidates) == 1
        assert candidates[0].fine_tuned is True
        assert candidates[0].language == "en"
        assert candidates[0].tags == ("fine_tuned",)

    def test_language_left_unset_when_not_in_path(self, tmp_path: Path):
        base = tmp_path / "fine_tuned_models"
        ckpt_dir = base / "some_checkpoint"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model.pt").write_bytes(b"\x00")
        _write_wav(ckpt_dir / "ref.wav")

        candidates = irv.scan_finetuned_dir(base)

        assert len(candidates) == 1
        assert candidates[0].language is None

    def test_non_checkpoint_dir_skipped(self, tmp_path: Path):
        base = tmp_path / "fine_tuned_models"
        (base / "just_some_dir").mkdir(parents=True)
        _write_wav(base / "just_some_dir" / "stray.wav")
        assert irv.scan_finetuned_dir(base) == []


class TestDedupeAgainstExisting:
    def test_skips_already_cataloged_paths(self, tmp_path: Path):
        wav = tmp_path / "voices" / "en" / "adult" / "foo.wav"
        _write_wav(wav)
        candidate = irv._make_candidate(wav, gender="male", language="en", fine_tuned=False, extra_tags=())

        remaining, skipped = irv.dedupe_against_existing([candidate], {candidate.stored_path})
        assert remaining == []
        assert skipped == 1

        remaining2, skipped2 = irv.dedupe_against_existing([candidate], set())
        assert remaining2 == [candidate]
        assert skipped2 == 0


class TestFilterCandidatesByLanguage:
    def _candidates(self, tmp_path: Path):
        wavs = {
            "en": tmp_path / "en.wav", "zh": tmp_path / "zh.wav",
            "fr": tmp_path / "fr.wav", "unknown": tmp_path / "unknown.wav",
        }
        for w in wavs.values():
            _write_wav(w)
        return [
            irv._make_candidate(wavs["en"], gender=None, language="en", fine_tuned=False, extra_tags=()),
            irv._make_candidate(wavs["zh"], gender=None, language="zh", fine_tuned=False, extra_tags=()),
            irv._make_candidate(wavs["fr"], gender=None, language="fr", fine_tuned=False, extra_tags=()),
            irv._make_candidate(wavs["unknown"], gender=None, language=None, fine_tuned=False, extra_tags=()),
        ]

    def test_all_keeps_everything(self, tmp_path: Path):
        candidates = self._candidates(tmp_path)
        assert irv.filter_candidates_by_language(candidates, "all") == candidates

    def test_en_zh_keeps_en_zh_and_undetected_drops_other_languages(self, tmp_path: Path):
        candidates = self._candidates(tmp_path)
        result = irv.filter_candidates_by_language(candidates, "en_zh")
        assert {c.language for c in result} == {"en", "zh", None}
        assert all(c.language != "fr" for c in result)
        assert len(result) == 3


class TestAudioPlayer:
    class _StubStream:
        def __init__(self, samplerate, channels, callback):
            self.callback = callback
            self.started = self.stopped = self.closed = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    def _make_player(self, num_frames: int = 10, channels: int = 1):
        import numpy as np

        data = np.arange(num_frames * channels, dtype="float32").reshape(num_frames, channels)
        captured = {}

        def factory(samplerate, channels, callback):
            captured["stream"] = self._StubStream(samplerate, channels, callback)
            return captured["stream"]

        player = irv._AudioPlayer(data, samplerate=16000, stream_factory=factory)
        return player, captured["stream"], data

    def test_callback_advances_frame_position_and_writes_data(self):
        import numpy as np

        player, stream, data = self._make_player(num_frames=10)
        outdata = np.zeros((4, 1), dtype="float32")

        player._callback(outdata, 4, None, None)
        assert (outdata == data[0:4]).all()
        assert not player.finished

        outdata2 = np.zeros((4, 1), dtype="float32")
        player._callback(outdata2, 4, None, None)
        assert (outdata2 == data[4:8]).all()

    def test_callback_marks_finished_when_data_runs_out(self):
        import numpy as np

        player, stream, data = self._make_player(num_frames=5)
        outdata = np.zeros((4, 1), dtype="float32")
        player._callback(outdata, 4, None, None)
        assert not player.finished

        outdata2 = np.zeros((4, 1), dtype="float32")
        player._callback(outdata2, 4, None, None)
        assert player.finished
        assert (outdata2[0] == data[4]).all()
        assert (outdata2[1:] == 0).all()

    def test_callback_is_contiguous_across_multiple_calls(self):
        import numpy as np

        player, stream, data = self._make_player(num_frames=10)
        outdata = np.zeros((4, 1), dtype="float32")
        player._callback(outdata, 4, None, None)
        assert (outdata == data[0:4]).all()

        outdata2 = np.zeros((4, 1), dtype="float32")
        player._callback(outdata2, 4, None, None)
        assert (outdata2 == data[4:8]).all()

    def test_start_and_close_delegate_to_stream(self):
        player, stream, _ = self._make_player()
        player.start()
        assert stream.started
        player.close()
        assert stream.stopped and stream.closed


class TestGroupCandidatesByFolder:
    def test_groups_by_immediate_parent_directory(self, tmp_path: Path):
        wav1 = tmp_path / "voices" / "en" / "adult" / "male" / "a.wav"
        wav2 = tmp_path / "voices" / "en" / "adult" / "male" / "b.wav"
        wav3 = tmp_path / "voices" / "zh" / "child" / "female" / "c.wav"
        for w in (wav1, wav2, wav3):
            _write_wav(w)
        candidates = [
            irv._make_candidate(w, gender=None, language=None, fine_tuned=False, extra_tags=())
            for w in (wav1, wav2, wav3)
        ]

        groups = irv.group_candidates_by_folder(candidates)

        assert set(groups.keys()) == {wav1.parent, wav3.parent}
        assert {c.name for c in groups[wav1.parent]} == {"a", "b"}
        assert {c.name for c in groups[wav3.parent]} == {"c"}


class TestApplyBatchTags:
    def _candidate(self, tmp_path: Path, **overrides) -> irv.VoiceCandidate:
        wav = tmp_path / "voice.wav"
        _write_wav(wav)
        defaults = dict(gender="male", language="en", fine_tuned=False, extra_tags=("adult",))
        defaults.update(overrides)
        return irv._make_candidate(wav, **defaults)

    def test_blank_fields_leave_existing_values_unchanged(self, tmp_path: Path):
        candidate = self._candidate(tmp_path)
        result = irv.apply_batch_tags(candidate)
        assert result == candidate

    def test_non_blank_fields_overwrite(self, tmp_path: Path):
        candidate = self._candidate(tmp_path)
        result = irv.apply_batch_tags(candidate, gender="female", language="eng", accent="southern", tags="elder, custom")
        assert result.gender == "female"
        assert result.language == "en"
        assert result.accent == "southern"
        assert result.extra_tags == ("elder", "custom")

    def test_partial_fields_only_overwrite_those_given(self, tmp_path: Path):
        candidate = self._candidate(tmp_path, gender=None)
        result = irv.apply_batch_tags(candidate, gender="male")
        assert result.gender == "male"
        assert result.language == "en"
        assert result.extra_tags == ("adult",)


class TestFindReplacementCandidate:
    def test_matches_by_filename_and_audio_params(self, tmp_path: Path):
        wav = tmp_path / "moved" / "voice.wav"
        _write_wav(wav, sample_rate=22050, channels=1, sampwidth=2)
        candidate = irv._make_candidate(wav, gender=None, language=None, fine_tuned=False, extra_tags=())

        match = irv.find_replacement_candidate("voice.wav", (22050, 16, 1), [candidate])
        assert match is candidate

        no_match = irv.find_replacement_candidate("voice.wav", (44100, 16, 1), [candidate])
        assert no_match is None


class TestRefText:
    def test_path_for_swaps_extension(self, tmp_path: Path):
        wav = tmp_path / "yunjian.wav"
        assert irv.ref_text_path_for(wav) == tmp_path / "yunjian.txt"

    def test_read_ref_text_missing_file_returns_empty_string(self, tmp_path: Path):
        wav = tmp_path / "yunjian.wav"
        assert irv.read_ref_text(wav) == ""

    def test_write_then_read_round_trips_stripped_text(self, tmp_path: Path):
        wav = tmp_path / "yunjian.wav"
        irv.write_ref_text(wav, "  你好，欢迎收听。  \n")
        assert irv.read_ref_text(wav) == "你好，欢迎收听。"
        assert (tmp_path / "yunjian.txt").is_file()

    def test_write_ref_text_overwrites_existing_file(self, tmp_path: Path):
        wav = tmp_path / "yunjian.wav"
        irv.write_ref_text(wav, "first draft")
        irv.write_ref_text(wav, "final version")
        assert irv.read_ref_text(wav) == "final version"

    def test_transcribe_reference_audio_joins_segments_with_injected_model(self, tmp_path: Path):
        class _StubSegment:
            def __init__(self, text: str):
                self.text = text

        class _StubModel:
            def transcribe(self, path: str):
                assert path == str(wav)
                return [_StubSegment(" 你好， "), _StubSegment("欢迎收听。 ")], object()

        wav = tmp_path / "yunjian.wav"
        text = irv.transcribe_reference_audio(wav, model=_StubModel())
        assert text == "你好， 欢迎收听。"


class TestUpdateReferenceVoiceDurations:
    def test_backfills_null_audio_params_from_real_wav_files(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        fake_root = tmp_path / "fake_repo"
        wav = fake_root / "voices" / "en" / "adult" / "male" / "alice.wav"
        _write_wav(wav, sample_rate=16000, channels=1, sampwidth=2)

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        stored_path = "voices/en/adult/male/alice.wav"
        with get_db_session(db_path) as db:
            db.add(ReferenceVoice(
                path=stored_path, name="alice", gender="male", language="en", duration_seconds=None,
            ))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(paths, "PROJECT_ROOT", fake_root)
            updated, missing = irv.update_reference_voice_durations(db_path)

        assert (updated, missing) == (1, 0)
        with get_db_session(db_path) as db:
            row = db.query(ReferenceVoice).one()
            assert row.duration_seconds == round(100 / 16000, 2)
            assert (row.sample_rate, row.bit_depth, row.channels) == (16000, 16, 1)

    def test_backfills_wave_format_extensible_rows_too(self, tmp_path: Path):
        import numpy as np
        import soundfile as sf

        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        fake_root = tmp_path / "fake_repo"
        wav = fake_root / "voices" / "en" / "adult" / "male" / "extensible.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav), np.zeros(1600, dtype="int16"), 16000, subtype="PCM_16", format="WAVEX")

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            db.add(ReferenceVoice(
                path="voices/en/adult/male/extensible.wav", name="extensible",
                gender="male", language="en",
            ))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(paths, "PROJECT_ROOT", fake_root)
            updated, missing = irv.update_reference_voice_durations(db_path)

        assert (updated, missing) == (1, 0)
        with get_db_session(db_path) as db:
            row = db.query(ReferenceVoice).one()
            assert (row.sample_rate, row.bit_depth, row.channels, row.duration_seconds) == (16000, 16, 1, 0.1)

    def test_missing_file_is_counted_and_skipped(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            db.add(ReferenceVoice(
                path="voices/en/adult/male/gone.wav", name="gone", gender="male", language="en",
            ))

        updated, missing = irv.update_reference_voice_durations(db_path)
        assert (updated, missing) == (0, 1)

    def test_rerun_is_idempotent_no_op_when_already_correct(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        fake_root = tmp_path / "fake_repo"
        wav = fake_root / "voices" / "en" / "adult" / "male" / "alice.wav"
        _write_wav(wav, sample_rate=16000, channels=1, sampwidth=2)

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            db.add(ReferenceVoice(
                path="voices/en/adult/male/alice.wav", name="alice", gender="male", language="en",
            ))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(paths, "PROJECT_ROOT", fake_root)
            irv.update_reference_voice_durations(db_path)
            updated_second_run, _ = irv.update_reference_voice_durations(db_path)

        assert updated_second_run == 0


class TestCliArgs:
    def test_update_durations_flag_parses(self):
        args = irv._parse_args(["db.sqlite", "--update-durations"])
        assert args.update_durations is True


class TestRunWizardEndToEnd:
    def test_imports_new_voices_and_commits_tags(self, tmp_path: Path, monkeypatch):
        fake_root = tmp_path / "fake_repo"
        _write_wav(fake_root / "voices" / "en" / "adult" / "male" / "sample1.wav")
        _write_wav(fake_root / "voices" / "zh" / "child" / "female" / "sample2.wav")
        monkeypatch.setattr(irv, "_REPO_ROOT", fake_root)
        monkeypatch.setattr(irv, "run_review_gui", lambda candidates: candidates)

        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        exit_code = irv.run_wizard(db_path, external_voice_folder=None)
        assert exit_code == 0

        with get_db_session(db_path) as db:
            rows = db.query(ReferenceVoice).order_by(ReferenceVoice.name).all()
            assert [r.name for r in rows] == ["sample1", "sample2"]
            assert (rows[0].gender, rows[0].language) == ("male", "en")
            assert (rows[1].gender, rows[1].language) == ("female", "zh")
            assert rows[0].duration_seconds == 0.0
            assert rows[1].duration_seconds == 0.0

            tags = {t.tag for t in db.query(ReferenceVoiceTag).all()}
            assert tags == {"adult", "child"}

    def test_review_gui_exclusions_are_not_imported(self, tmp_path: Path, monkeypatch):
        fake_root = tmp_path / "fake_repo"
        _write_wav(fake_root / "voices" / "en" / "adult" / "male" / "sample1.wav")
        _write_wav(fake_root / "voices" / "zh" / "child" / "female" / "sample2.wav")
        monkeypatch.setattr(irv, "_REPO_ROOT", fake_root)
        monkeypatch.setattr(
            irv, "run_review_gui", lambda candidates: [c for c in candidates if c.name == "sample1"]
        )

        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import ReferenceVoice

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        assert irv.run_wizard(db_path, external_voice_folder=None) == 0

        with get_db_session(db_path) as db:
            rows = db.query(ReferenceVoice).all()
            assert [r.name for r in rows] == ["sample1"]

    def test_second_run_is_append_only(self, tmp_path: Path, monkeypatch):
        fake_root = tmp_path / "fake_repo"
        _write_wav(fake_root / "voices" / "en" / "adult" / "male" / "sample1.wav")
        monkeypatch.setattr(irv, "_REPO_ROOT", fake_root)
        monkeypatch.setattr(irv, "run_review_gui", lambda candidates: candidates)

        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import ReferenceVoice

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        assert irv.run_wizard(db_path, external_voice_folder=None) == 0
        assert irv.run_wizard(db_path, external_voice_folder=None) == 0

        with get_db_session(db_path) as db:
            assert db.query(ReferenceVoice).count() == 1

    def test_lan_filter_excludes_other_languages_by_default(self, tmp_path: Path, monkeypatch):
        fake_root = tmp_path / "fake_repo"
        _write_wav(fake_root / "voices" / "en" / "adult" / "male" / "sample1.wav")
        _write_wav(fake_root / "voices" / "fr" / "adult" / "male" / "sample2.wav")
        monkeypatch.setattr(irv, "_REPO_ROOT", fake_root)
        monkeypatch.setattr(irv, "run_review_gui", lambda candidates: candidates)

        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import ReferenceVoice

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        assert irv.run_wizard(db_path, external_voice_folder=None) == 0

        with get_db_session(db_path) as db:
            assert [r.name for r in db.query(ReferenceVoice).all()] == ["sample1"]

    def test_lan_filter_all_keeps_other_languages(self, tmp_path: Path, monkeypatch):
        fake_root = tmp_path / "fake_repo"
        _write_wav(fake_root / "voices" / "en" / "adult" / "male" / "sample1.wav")
        _write_wav(fake_root / "voices" / "fr" / "adult" / "male" / "sample2.wav")
        monkeypatch.setattr(irv, "_REPO_ROOT", fake_root)
        monkeypatch.setattr(irv, "run_review_gui", lambda candidates: candidates)

        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import ReferenceVoice

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        assert irv.run_wizard(db_path, external_voice_folder=None, lan_filter="all") == 0

        with get_db_session(db_path) as db:
            assert {r.name for r in db.query(ReferenceVoice).all()} == {"sample1", "sample2"}


def _seed_db(tmp_path: Path):
    from syntrive.db.session import ensure_schema, make_engine
    db_path = tmp_path / "syntrivetts.db"
    ensure_schema(make_engine(db_path))
    return db_path


def _add_voice(db, *, path: str, name: str, gender="male", language="en", accent=None, tags=()):
    from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag
    rv = ReferenceVoice(path=path, name=name, gender=gender, language=language, accent=accent)
    db.add(rv)
    db.flush()
    for t in tags:
        db.add(ReferenceVoiceTag(reference_voice_id=rv.id, tag=t))
    return rv.id


def _attach_to_a_job(db, reference_voice_id: int):
    from syntrive.db.models import Book, Job, TtsConfig, TtsVoice
    book = Book(title="Ref Book")
    db.add(book)
    db.flush()
    job = Job(book_id=book.id, process_dir="PROCESSING-Ref", epub_path="b.epub")
    db.add(job)
    db.flush()
    cfg = TtsConfig(job_id=job.id, engine="xtts")
    db.add(cfg)
    db.flush()
    db.add(TtsVoice(tts_config_id=cfg.id, name="narrator", voice_id=0,
                    reference_voice_id=reference_voice_id))


class TestLoadReferenceVoices:
    def test_builds_candidates_with_db_id_and_splits_fine_tuned_tag(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            id1 = _add_voice(db, path="voices/zh/adult/male/a.wav", name="a",
                             gender="male", language="zh", tags=("adult", "fine_tuned"))
            id2 = _add_voice(db, path="voices/en/adult/female/b.wav", name="b",
                             gender="female", language="en", tags=())

        with get_db_session(db_path) as db:
            cands = {c.name: c for c in irv.load_reference_voices(db)}

        assert cands["a"].db_id == id1 and cands["b"].db_id == id2
        assert cands["a"].fine_tuned is True
        assert cands["a"].extra_tags == ("adult",)
        assert cands["a"].tags == ("fine_tuned", "adult")
        assert cands["b"].fine_tuned is False and cands["b"].extra_tags == ()
        assert (cands["b"].gender, cands["b"].language) == ("female", "en")

    def test_blank_gender_language_become_none_for_the_combo(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            _add_voice(db, path="voices/x/y.wav", name="y", gender="", language="")
        with get_db_session(db_path) as db:
            c = irv.load_reference_voices(db)[0]
        assert c.gender is None and c.language is None


class TestReferenceVoicesInUse:
    def test_returns_only_ids_referenced_by_a_tts_voice(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            used = _add_voice(db, path="voices/a.wav", name="a")
            _add_voice(db, path="voices/b.wav", name="b")
            _attach_to_a_job(db, used)
        with get_db_session(db_path) as db:
            assert irv.reference_voices_in_use(db) == {used}

    def test_empty_when_no_tts_voices(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            _add_voice(db, path="voices/a.wav", name="a")
        with get_db_session(db_path) as db:
            assert irv.reference_voices_in_use(db) == set()


class TestDeleteReferenceVoice:
    def test_deletes_row_and_its_tags(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            rv_id = _add_voice(db, path="voices/a.wav", name="a", tags=("adult", "fine_tuned"))

        assert irv.delete_reference_voice(db_path, rv_id) is True
        with get_db_session(db_path) as db:
            assert db.get(ReferenceVoice, rv_id) is None
            assert db.query(ReferenceVoiceTag).filter_by(reference_voice_id=rv_id).count() == 0

    def test_refuses_when_referenced_by_a_tts_voice(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            rv_id = _add_voice(db, path="voices/a.wav", name="a")
            _attach_to_a_job(db, rv_id)

        assert irv.delete_reference_voice(db_path, rv_id) is False
        with get_db_session(db_path) as db:
            assert db.get(ReferenceVoice, rv_id) is not None

    def test_returns_false_for_unknown_id(self, tmp_path: Path):
        db_path = _seed_db(tmp_path)
        assert irv.delete_reference_voice(db_path, 9999) is False


class TestUpdateReferenceVoice:
    def test_persists_gender_language_accent_and_tag_diff(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            rv_id = _add_voice(db, path="voices/a.wav", name="a", gender="male",
                               language="en", accent=None, tags=("adult", "old"))

        changed = irv.update_reference_voice(
            db_path, rv_id, gender="Female", language="zho", accent="  beijing ",
            tags=("adult", "new", "fine_tuned"),
        )
        assert changed is True
        with get_db_session(db_path) as db:
            rv = db.get(ReferenceVoice, rv_id)
            assert (rv.gender, rv.language, rv.accent) == ("female", "zh", "beijing")
            assert {t.tag for t in rv.tags} == {"adult", "new", "fine_tuned"}

    def test_no_op_returns_false(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        db_path = _seed_db(tmp_path)
        with get_db_session(db_path) as db:
            rv_id = _add_voice(db, path="voices/a.wav", name="a", gender="male",
                               language="en", tags=("adult",))
        assert irv.update_reference_voice(
            db_path, rv_id, gender="male", language="en", accent="", tags=("adult",)
        ) is False

    def test_unknown_id_returns_false(self, tmp_path: Path):
        db_path = _seed_db(tmp_path)
        assert irv.update_reference_voice(
            db_path, 9999, gender="male", language="en", accent="", tags=()
        ) is False


class TestReviewCliArg:
    def test_review_flag_parses(self):
        assert irv._parse_args(["db.sqlite", "--review"]).review is True

    def test_review_and_update_durations_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            irv._parse_args(["db.sqlite", "--review", "--update-durations"])

    def test_voice_candidate_db_id_defaults_to_none(self, tmp_path: Path):
        _write_wav(tmp_path / "voices" / "en" / "adult" / "male" / "x.wav")
        cands = irv.scan_voices_dir(tmp_path / "voices")
        assert cands and all(c.db_id is None for c in cands)

    def test_run_review_wizard_empty_catalog_returns_zero(self, tmp_path: Path, capsys):
        db_path = _seed_db(tmp_path)
        assert irv.run_review_wizard(db_path) == 0
        assert "No reference voices" in capsys.readouterr().out


class TestResolveAsrModelSource:
    def _snapshot(self, tmp_path: Path, with_model_bin: bool = True) -> Path:
        snap = tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        if with_model_bin:
            (snap / "model.bin").write_bytes(b"x")
        return snap

    def _resolve(self, tmp_path, *, offline, model_dir):
        return irv.resolve_asr_model_source(
            offline=offline, model_dir=model_dir, model_name="small",
            download_root=tmp_path, catalog_id="faster-whisper-small",
        )

    @pytest.mark.parametrize("offline", [True, False])
    def test_local_snapshot_is_loaded_directly_and_local_only(self, tmp_path, offline):
        snap = self._snapshot(tmp_path)

        src = self._resolve(tmp_path, offline=offline, model_dir=snap)

        assert src == irv.AsrModelSource(str(snap), True, None, "models_tts")

    def test_offline_without_local_model_raises_with_actionable_fix(self, tmp_path):
        with pytest.raises(irv.AsrModelUnavailable) as err:
            self._resolve(tmp_path, offline=True, model_dir=None)

        msg = str(err.value)
        assert "faster-whisper-small" in msg and "python tools/model_dl.py asr" in msg

    def test_incomplete_snapshot_without_model_bin_counts_as_missing(self, tmp_path):
        snap = self._snapshot(tmp_path, with_model_bin=False)

        with pytest.raises(irv.AsrModelUnavailable):
            self._resolve(tmp_path, offline=True, model_dir=snap)

    def test_online_without_local_model_downloads_into_models_tts_not_home_cache(self, tmp_path):
        src = self._resolve(tmp_path, offline=False, model_dir=None)

        assert src == irv.AsrModelSource("small", False, str(tmp_path), "download_into_models_tts")


class TestOfflineRequested:
    @pytest.mark.parametrize("var", ["SYNTRIVE_TTS_OFFLINE", "HF_HUB_OFFLINE"])
    def test_either_flag_enables_offline(self, monkeypatch, var):
        monkeypatch.delenv("SYNTRIVE_TTS_OFFLINE", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setenv(var, "1")
        assert irv._offline_requested() is True

    def test_default_is_online(self, monkeypatch):
        monkeypatch.delenv("SYNTRIVE_TTS_OFFLINE", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        assert irv._offline_requested() is False
