from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
import torchaudio

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)


class _FakeEngine:
    def __init__(self, sample_rate=24000, tone_duration_s=0.2, fail_on_text: str = None):
        self.sample_rate = sample_rate
        self.tone_duration_s = tone_duration_s
        self.fail_on_text = fail_on_text
        self.calls = []
        self.engine_name = "cosyvoice"
        self.options = {}

    def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
        self.calls.append(text)
        if self.fail_on_text is not None and text == self.fail_on_text:
            return None
        waveform = torch.full((1, int(self.tone_duration_s * self.sample_rate)), 0.4)
        torchaudio.save(output_path, waveform, self.sample_rate)
        return output_path

    def resolve_voice_for_tts_script(self, tts_config_id, voice_id, db, cache):
        return None


class _FakeEngineOomNTimes:
    engine_name = "indextts"
    options = {}

    def __init__(self, oom_failures: int, sample_rate=24000, tone_duration_s=0.2):
        self.oom_failures = oom_failures
        self.calls_on_oom_line = 0
        self.sample_rate = sample_rate
        self.tone_duration_s = tone_duration_s
        self.last_error = ""

    def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
        if text == "oom line" and self.calls_on_oom_line < self.oom_failures:
            self.calls_on_oom_line += 1
            self.last_error = "worker error: MPS backend out of memory (fake)"
            return None
        self.last_error = ""
        waveform = torch.full((1, int(self.tone_duration_s * self.sample_rate)), 0.4)
        torchaudio.save(output_path, waveform, self.sample_rate)
        return output_path

    def resolve_voice_for_tts_script(self, tts_config_id, voice_id, db, cache):
        return None


@pytest.fixture(autouse=True)
def _patch_get_engine(monkeypatch):
    import syntrive.orchestration.workflow as workflow_module

    state = {"engine": _FakeEngine()}

    def _fake_get_engine(engine_name, options=None):
        return state["engine"]

    monkeypatch.setattr(workflow_module.manager, "get_engine", _fake_get_engine)
    return state


def _seed_job(db, tmp_path: Path, *, job_id: int = 1, book_title: str = "Test Book", engine: str = "xtts"):
    from syntrive.db.models import Book, Job, OutputConfig, TtsConfig

    book = Book(title=book_title, author="Test Author")
    db.add(book)
    db.flush()
    process_dir_name = f"job{job_id}"
    (tmp_path / process_dir_name).mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, book_id=book.id, process_dir=process_dir_name, epub_path=f"{process_dir_name}/book.epub")
    db.add(job)
    db.flush()
    db.add(TtsConfig(job_id=job_id, engine=engine))
    db.add(OutputConfig(job_id=job_id))
    return job, book


def _write_transcript(tmp_path: Path, job_process_dir: str, chapter_id: str, content: str) -> str:
    script_dir = tmp_path / job_process_dir / "transcript_text" / "tts_script"
    script_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"transcript_text/tts_script/{chapter_id}.txt"
    (tmp_path / job_process_dir / rel_path).write_text(content, encoding="utf-8")
    return rel_path


def _seed_chapter(db, job_id: int, chapter_id: str, transcript_path: str, transcript_lines: int, **kwargs):
    from syntrive.db.models import TranscriptChapter

    defaults = dict(
        sequence_number="0001", chapter_number="0001", chapter_name="Chapter One",
        volume="", volume_number="", synthesis_status="pending",
    )
    defaults.update(kwargs)
    chapter = TranscriptChapter(
        job_id=job_id, chapter_id=chapter_id,
        transcript_path=transcript_path, transcript_lines=transcript_lines,
        **defaults,
    )
    db.add(chapter)
    db.flush()
    return chapter


class TestTtsConfigToOptions:
    def _cfg(self, **kw):
        from syntrive.db.models import TtsConfig
        return TtsConfig(job_id=1, **kw)

    def test_model_is_forwarded(self):
        from syntrive.orchestration.workflow import _tts_config_to_options
        opts = _tts_config_to_options(self._cfg(engine="cosyvoice", model="cosyvoice-300m-sft"))
        assert opts["model"] == "cosyvoice-300m-sft"

    def test_null_model_forwarded_as_none(self):
        from syntrive.orchestration.workflow import _tts_config_to_options
        assert _tts_config_to_options(self._cfg(engine="f5tts")).get("model") is None

    def test_xtts_non_internal_model_mirrored_into_fine_tuned_model(self):
        from syntrive.orchestration.workflow import _tts_config_to_options
        opts = _tts_config_to_options(self._cfg(engine="xtts", model="RosamundPike",
                                                fine_tuned_model="internal"))
        assert opts["fine_tuned_model"] == "RosamundPike"

    def test_xtts_internal_model_keeps_fine_tuned_model_as_is(self):
        from syntrive.orchestration.workflow import _tts_config_to_options
        opts = _tts_config_to_options(self._cfg(engine="xtts", model="internal",
                                                fine_tuned_model="internal"))
        assert opts["fine_tuned_model"] == "internal"

    def test_non_xtts_model_does_not_touch_fine_tuned_model(self):
        from syntrive.orchestration.workflow import _tts_config_to_options
        opts = _tts_config_to_options(self._cfg(engine="cosyvoice", model="cosyvoice-300m",
                                                fine_tuned_model="internal"))
        assert opts["fine_tuned_model"] == "internal"


class TestOomRetryAttempts:
    def test_unset_env_falls_back_to_default(self, monkeypatch):
        from syntrive.orchestration.workflow import DEFAULT_ORCHESTRATION_OOM_RETRY_ATTEMPTS, _oom_retry_attempts
        monkeypatch.delenv("SYNTRIVE_TTS_OOM_ORCHESTRATION_RETRY_ATTEMPTS", raising=False)
        assert _oom_retry_attempts() == DEFAULT_ORCHESTRATION_OOM_RETRY_ATTEMPTS

    def test_valid_env_value_is_honored(self, monkeypatch):
        from syntrive.orchestration.workflow import _oom_retry_attempts
        monkeypatch.setenv("SYNTRIVE_TTS_OOM_ORCHESTRATION_RETRY_ATTEMPTS", "5")
        assert _oom_retry_attempts() == 5

    def test_zero_disables_the_larger_oom_budget(self, monkeypatch):
        from syntrive.orchestration.workflow import _oom_retry_attempts
        monkeypatch.setenv("SYNTRIVE_TTS_OOM_ORCHESTRATION_RETRY_ATTEMPTS", "0")
        assert _oom_retry_attempts() == 0

    def test_invalid_env_value_falls_back_to_default_and_warns(self, monkeypatch, caplog):
        from syntrive.orchestration.workflow import DEFAULT_ORCHESTRATION_OOM_RETRY_ATTEMPTS, _oom_retry_attempts
        monkeypatch.setenv("SYNTRIVE_TTS_OOM_ORCHESTRATION_RETRY_ATTEMPTS", "not-a-number")
        assert _oom_retry_attempts() == DEFAULT_ORCHESTRATION_OOM_RETRY_ATTEMPTS


class TestEnqueueChapters:
    def test_enqueues_a_pending_chapter_into_its_own_new_batch(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import enqueue_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\ntwo\n")
            chapter = _seed_chapter(db, job.id, "ch_0001", rel, 2)

            result = enqueue_chapters(db, [chapter.id], note="test batch")
            assert result.ok is True
            assert len(result.batch_ids) == 1
            assert chapter.synthesis_status == "queued"
            assert chapter.synthesis_batch_id == result.batch_ids[0]

    def test_each_selected_chapter_gets_its_own_incrementing_batch_id(self, tmp_path: Path):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import enqueue_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            ids = []
            for seq in ("0001", "0002", "0003"):
                rel = _write_transcript(tmp_path, job.process_dir, f"ch_{seq}", "one\n")
                ids.append(_seed_chapter(
                    db, job.id, f"ch_{seq}", rel, 1, sequence_number=seq, chapter_number=seq,
                ).id)

            result = enqueue_chapters(db, ids)
            assert result.ok is True
            assert len(result.batch_ids) == 3
            assert list(result.batch_ids) == [
                result.batch_ids[0], result.batch_ids[0] + 1, result.batch_ids[0] + 2,
            ]
            rows = db.query(TranscriptChapter).filter(TranscriptChapter.id.in_(ids)).all()
            assert sorted(r.synthesis_batch_id for r in rows) == list(result.batch_ids)
            assert all(r.synthesis_status == "queued" for r in rows)

    def test_batch_ids_follow_reading_order_not_selection_order(self, tmp_path: Path):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import enqueue_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel1 = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\n")
            rel2 = _write_transcript(tmp_path, job.process_dir, "ch_0002", "one\n")
            c1 = _seed_chapter(db, job.id, "ch_0001", rel1, 1, sequence_number="0001", chapter_number="0001")
            c2 = _seed_chapter(db, job.id, "ch_0002", rel2, 1, sequence_number="0002", chapter_number="0002")

            result = enqueue_chapters(db, [c2.id, c1.id])
            assert result.ok is True
            first_batch, second_batch = result.batch_ids
            assert first_batch < second_batch
            assert db.query(TranscriptChapter).filter_by(id=c1.id).first().synthesis_batch_id == first_batch
            assert db.query(TranscriptChapter).filter_by(id=c2.id).first().synthesis_batch_id == second_batch

    def test_refuses_a_chapter_that_is_not_pending(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import enqueue_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\n")
            chapter = _seed_chapter(db, job.id, "ch_0001", rel, 1, synthesis_status="done")

            result = enqueue_chapters(db, [chapter.id])
            assert result.ok is False
            assert "not pending" in result.error
            assert result.batch_ids == ()
            assert chapter.synthesis_batch_id is None

    def test_one_bad_chapter_blocks_the_whole_selection_no_partial_batches(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch, TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import enqueue_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel1 = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\n")
            rel2 = _write_transcript(tmp_path, job.process_dir, "ch_0002", "one\n")
            good = _seed_chapter(db, job.id, "ch_0001", rel1, 1, sequence_number="0001", chapter_number="0001")
            bad = _seed_chapter(
                db, job.id, "ch_0002", rel2, 1, sequence_number="0002", chapter_number="0002",
                synthesis_status="done",
            )

            result = enqueue_chapters(db, [good.id, bad.id])
            assert result.ok is False
            assert db.query(SynthesisBatch).count() == 0
            still_good = db.query(TranscriptChapter).filter_by(id=good.id).first()
            assert still_good.synthesis_batch_id is None
            assert still_good.synthesis_status == "pending"

    def test_empty_selection_fails_cleanly(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import enqueue_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            result = enqueue_chapters(db, [])
            assert result.ok is False
            assert result.batch_ids == ()


class TestSetBatchesPaused:
    def _seed_batch_with_chapter(self, db, tmp_path, *, job_id, chapter_id, status="queued"):
        from syntrive.db.models import SynthesisBatch

        _seed_job(db, tmp_path, job_id=job_id, book_title=f"Book {job_id}")
        rel = _write_transcript(tmp_path, f"job{job_id}", chapter_id, "only line\n")
        batch = SynthesisBatch()
        db.add(batch)
        db.flush()
        _seed_chapter(
            db, job_id, chapter_id, rel, 1,
            synthesis_status=status, synthesis_batch_id=batch.synthesis_batch_id,
        )
        return batch.synthesis_batch_id

    def test_pause_then_resume_flips_paused_at(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import set_batches_paused

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            bid = self._seed_batch_with_chapter(db, tmp_path, job_id=1, chapter_id="ch_0001")

        with get_db_session(db_path) as db:
            r = set_batches_paused(db, [bid], paused=True)
            assert r.ok and r.flipped == 1
        with get_db_session(db_path) as db:
            assert db.query(SynthesisBatch).filter_by(synthesis_batch_id=bid).one().paused_at is not None

        with get_db_session(db_path) as db:
            r = set_batches_paused(db, [bid], paused=False)
            assert r.ok and r.flipped == 1
        with get_db_session(db_path) as db:
            assert db.query(SynthesisBatch).filter_by(synthesis_batch_id=bid).one().paused_at is None

    def test_pausing_an_already_paused_batch_is_a_no_op(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import set_batches_paused

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            bid = self._seed_batch_with_chapter(db, tmp_path, job_id=1, chapter_id="ch_0001")

        with get_db_session(db_path) as db:
            assert set_batches_paused(db, [bid], paused=True).flipped == 1
        with get_db_session(db_path) as db:
            r = set_batches_paused(db, [bid], paused=True)
            assert r.ok and r.flipped == 0 and r.already_in_state == (bid,)

    def test_unknown_batch_id_is_a_hard_error_with_nothing_written(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import set_batches_paused

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            bid = self._seed_batch_with_chapter(db, tmp_path, job_id=1, chapter_id="ch_0001")

        with get_db_session(db_path) as db:
            r = set_batches_paused(db, [bid, 999], paused=True)
            assert r.ok is False and "999" in r.error
        with get_db_session(db_path) as db:
            assert db.query(SynthesisBatch).filter_by(synthesis_batch_id=bid).one().paused_at is None

    def test_a_fully_done_batch_is_skipped_not_paused(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import set_batches_paused

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            bid = self._seed_batch_with_chapter(
                db, tmp_path, job_id=1, chapter_id="ch_0001", status="done",
            )

        with get_db_session(db_path) as db:
            r = set_batches_paused(db, [bid], paused=True)
            assert r.ok and r.flipped == 0 and r.skipped_done == (bid,)
        with get_db_session(db_path) as db:
            assert db.query(SynthesisBatch).filter_by(synthesis_batch_id=bid).one().paused_at is None


class TestDetectAndRollbackRedo:
    def test_a_done_chapter_with_intact_m4a_is_not_rolled_back(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import detect_and_rollback_redo

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\n")
            m4a_rel = f"{job.process_dir}/audiobooks/book_ch0001.m4a"
            (tmp_path / job.process_dir / "audiobooks").mkdir(parents=True)
            (tmp_path / m4a_rel).write_bytes(b"fake m4a bytes")
            chapter = _seed_chapter(
                db, job.id, "ch_0001", rel, 1, synthesis_status="done",
                chapter_audio_path=m4a_rel, chapter_audio_seconds=12.3,
            )

            rolled_back = detect_and_rollback_redo(db, chapter, tmp_path)
            assert rolled_back is False
            assert chapter.synthesis_status == "done"

    def test_missing_m4a_with_all_sentence_flacs_present_rolls_back_to_combining(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import detect_and_rollback_redo

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\ntwo\n")
            sentence_dir = tmp_path / job.process_dir / "sentence_audio" / "ch_0001"
            sentence_dir.mkdir(parents=True)
            (sentence_dir / "ch_0001_s0001.flac").write_bytes(b"x")
            (sentence_dir / "ch_0001_s0002.flac").write_bytes(b"x")
            chapter = _seed_chapter(
                db, job.id, "ch_0001", rel, 2, synthesis_status="done",
                chapter_audio_path=f"{job.process_dir}/audiobooks/missing.m4a",
            )

            rolled_back = detect_and_rollback_redo(db, chapter, tmp_path)
            assert rolled_back is True
            assert chapter.synthesis_status == "combining"
            assert chapter.chapter_audio_path is None

    def test_missing_m4a_and_a_missing_sentence_flac_rolls_back_to_synthesizing(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import detect_and_rollback_redo

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\ntwo\n")
            sentence_dir = tmp_path / job.process_dir / "sentence_audio" / "ch_0001"
            sentence_dir.mkdir(parents=True)
            (sentence_dir / "ch_0001_s0001.flac").write_bytes(b"x")
            chapter = _seed_chapter(
                db, job.id, "ch_0001", rel, 2, synthesis_status="done",
                chapter_audio_path=f"{job.process_dir}/audiobooks/missing.m4a",
            )

            rolled_back = detect_and_rollback_redo(db, chapter, tmp_path)
            assert rolled_back is True
            assert chapter.synthesis_status == "synthesizing"

    def test_a_non_done_chapter_is_never_touched(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import detect_and_rollback_redo

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one\n")
            chapter = _seed_chapter(db, job.id, "ch_0001", rel, 1, synthesis_status="synthesizing")

            assert detect_and_rollback_redo(db, chapter, tmp_path) is False
            assert chapter.synthesis_status == "synthesizing"

    def test_sentence_flac_lookup_uses_sequence_number_not_gappy_chapter_id(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import detect_and_rollback_redo

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0004", "one\ntwo\n")
            sentence_dir = tmp_path / job.process_dir / "sentence_audio" / "ch_0001"
            sentence_dir.mkdir(parents=True)
            (sentence_dir / "ch_0001_s0001.flac").write_bytes(b"x")
            (sentence_dir / "ch_0001_s0002.flac").write_bytes(b"x")
            chapter = _seed_chapter(
                db, job.id, "ch_0004", rel, 2, sequence_number="0001", chapter_number="0001",
                synthesis_status="done",
                chapter_audio_path=f"{job.process_dir}/audiobooks/missing.m4a",
            )

            rolled_back = detect_and_rollback_redo(db, chapter, tmp_path)
            assert rolled_back is True
            assert chapter.synthesis_status == "combining"


class TestRunBatch:
    def test_queued_to_synthesizing_transition_is_committed_before_the_blocking_synthesize_call(
        self, tmp_path: Path, monkeypatch,
    ):
        import sqlite3

        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import run_batch

        db_path = tmp_path / "db.sqlite"
        observed = {}

        def _check_committed_status_from_another_connection(chapter_id: str) -> None:
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute(
                    "SELECT synthesis_status FROM transcript_chapters WHERE chapter_id = ?",
                    (chapter_id,),
                ).fetchone()
            finally:
                conn.close()
            observed["status_seen_by_other_connection"] = row[0] if row else None

        with get_db_session(db_path) as db:
            job, book = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "one line only\n")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _seed_chapter(
                db, job.id, "ch_0001", rel, 1,
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            import syntrive.orchestration.workflow as workflow_module
            fake_engine = _FakeEngine()
            original_synthesize = fake_engine.synthesize

            def _synthesize_and_check(text, voice=None, leading_silence_ms=0, output_path=None):
                if "status_seen_by_other_connection" not in observed:
                    _check_committed_status_from_another_connection("ch_0001")
                return original_synthesize(
                    text, voice=voice, leading_silence_ms=leading_silence_ms, output_path=output_path,
                )

            fake_engine.synthesize = _synthesize_and_check
            monkeypatch.setattr(workflow_module.manager, "get_engine", lambda name, options=None: fake_engine)

            result = run_batch(db, batch.synthesis_batch_id, tmp_path)
            assert result.ok is True

        assert observed["status_seen_by_other_connection"] == "synthesizing"

    def test_unknown_batch_id_fails_cleanly(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import run_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            result = run_batch(db, 999, tmp_path)
            assert result.ok is False

    def test_sentence_flacs_are_named_by_sequence_number_not_gappy_chapter_id(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import run_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, book = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0004", "It was a bright cold day.")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            chapter = _seed_chapter(
                db, job.id, "ch_0004", rel, 1, sequence_number="0001", chapter_number="0001",
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            result = run_batch(db, batch.synthesis_batch_id, tmp_path)

            assert result.ok is True
            assert chapter.synthesis_status == "done"
            sentence_dir = tmp_path / job.process_dir / "sentence_audio" / "ch_0001"
            assert (sentence_dir / "ch_0001_s0001.flac").exists()
            assert not (sentence_dir / "ch_0004_s0001.flac").exists()

    def test_full_happy_path_produces_a_done_chapter_with_a_real_m4a(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import run_batch

        db_path = tmp_path / "db.sqlite"
        events = []
        with get_db_session(db_path) as db:
            job, book = _seed_job(db, tmp_path)
            rel = _write_transcript(
                tmp_path, job.process_dir, "ch_0001",
                "‡voice:0‡\nPART one ‡pause‡ Section one\n‡break‡\nIt was a bright cold day.",
            )
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            chapter = _seed_chapter(
                db, job.id, "ch_0001", rel, 3,
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            result = run_batch(db, batch.synthesis_batch_id, tmp_path, on_event=lambda name, f: events.append((name, f)))

            assert result.ok is True
            assert result.chapters_completed == 1
            assert chapter.synthesis_status == "done"
            assert chapter.chapter_audio_path == f"{job.process_dir}/audiobooks/{book.title}_ch0001.m4a"
            assert chapter.chapter_audio_seconds is not None and chapter.chapter_audio_seconds > 0
            assert (tmp_path / chapter.chapter_audio_path).exists()
            assert batch.synth_finished_at is not None
            assert batch.synth_error is None

            line_starts = [fields for name, fields in events if name == "line_start"]
            assert [f["text"] for f in line_starts] == ["PART one", "Section one", "It was a bright cold day."]
            assert [f["line_idx"] for f in line_starts] == [1, 2, 3]
            assert all(f["line_total"] == 3 for f in line_starts)
            assert all(f["book"] == book.title for f in line_starts)
            assert all((f["book_seq"], f["book_total"]) == (1, 1) for f in line_starts)
            assert all((f["chapter_seq"], f["chapter_total"]) == (1, 1) for f in line_starts)

        event_names = [name for name, _ in events]
        assert event_names[0] == "synth_batch_start"
        assert "chapter_start" in event_names
        assert event_names.count("line_ok") == 3
        assert "chapter_merged" in event_names

    def test_abort_on_failing_line_leaves_chapter_status_unfinished_and_records_the_error(
        self, tmp_path: Path, _patch_get_engine,
    ):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import run_batch

        _patch_get_engine["engine"] = _FakeEngine(fail_on_text="bad line")
        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "good line\nbad line\n")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            chapter = _seed_chapter(
                db, job.id, "ch_0001", rel, 2,
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            result = run_batch(db, batch.synthesis_batch_id, tmp_path)

            assert result.ok is False
            assert result.failed_chapter_id == "ch_0001"
            assert result.failed_line_index == 2
            assert chapter.synthesis_status == "synthesizing"
            assert batch.synth_error is not None
            assert batch.failed_line_index == 2

    def test_resume_after_abort_only_retries_the_failed_line_then_completes(self, tmp_path: Path, _patch_get_engine):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import run_batch

        failing_engine = _FakeEngine(fail_on_text="bad line")
        _patch_get_engine["engine"] = failing_engine
        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path)
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "good line\nbad line\n")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            chapter = _seed_chapter(
                db, job.id, "ch_0001", rel, 2,
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            batch_id = batch.synthesis_batch_id
            first = run_batch(db, batch_id, tmp_path)
            assert first.ok is False
            calls_after_first = list(failing_engine.calls)

        fixed_engine = _FakeEngine()
        _patch_get_engine["engine"] = fixed_engine
        with get_db_session(db_path) as db:
            second = run_batch(db, batch_id, tmp_path)
            assert second.ok is True
            assert second.chapters_completed == 1

        from syntrive.orchestration.workflow import ORCHESTRATION_RETRY_ATTEMPTS

        assert calls_after_first == ["good line"] + ["bad line"] * (ORCHESTRATION_RETRY_ATTEMPTS + 1)
        assert fixed_engine.calls == ["bad line"]

    def test_an_oom_classified_failure_gets_a_much_larger_retry_budget_than_a_plain_failure(
        self, tmp_path: Path, _patch_get_engine,
    ):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import ORCHESTRATION_RETRY_ATTEMPTS, run_batch

        engine = _FakeEngineOomNTimes(oom_failures=ORCHESTRATION_RETRY_ATTEMPTS + 2)
        _patch_get_engine["engine"] = engine
        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path, engine="indextts")
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "good line\noom line\n")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            chapter = _seed_chapter(
                db, job.id, "ch_0001", rel, 2,
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            result = run_batch(db, batch.synthesis_batch_id, tmp_path)

            assert result.ok is True
            assert chapter.synthesis_status == "done"
        assert engine.calls_on_oom_line == ORCHESTRATION_RETRY_ATTEMPTS + 2

    def test_an_oom_classified_failure_still_gives_up_once_its_own_larger_budget_is_exhausted(
        self, tmp_path: Path, _patch_get_engine, monkeypatch,
    ):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration import workflow as workflow_module

        monkeypatch.setenv("SYNTRIVE_TTS_OOM_ORCHESTRATION_RETRY_ATTEMPTS", "2")
        engine = _FakeEngineOomNTimes(oom_failures=999)
        _patch_get_engine["engine"] = engine
        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, _ = _seed_job(db, tmp_path, engine="indextts")
            rel = _write_transcript(tmp_path, job.process_dir, "ch_0001", "good line\noom line\n")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _seed_chapter(
                db, job.id, "ch_0001", rel, 2,
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            result = workflow_module.run_batch(db, batch.synthesis_batch_id, tmp_path)

            assert result.ok is False
        assert engine.calls_on_oom_line == 3

    def test_a_second_chapter_in_the_same_batch_is_skipped_once_already_done(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.workflow import run_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            job, book = _seed_job(db, tmp_path)
            rel1 = _write_transcript(tmp_path, job.process_dir, "ch_0001", "first chapter line\n")
            rel2 = _write_transcript(tmp_path, job.process_dir, "ch_0002", "second chapter line\n")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _seed_chapter(
                db, job.id, "ch_0001", rel1, 1, sequence_number="0001", chapter_number="0001",
                synthesis_status="done", synthesis_batch_id=batch.synthesis_batch_id,
                chapter_audio_path="irrelevant/already_done.m4a", chapter_audio_seconds=99.0,
            )
            chapter2 = _seed_chapter(
                db, job.id, "ch_0002", rel2, 1, sequence_number="0002", chapter_number="0002",
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            result = run_batch(db, batch.synthesis_batch_id, tmp_path)

            assert result.ok is True
            assert result.chapters_completed == 1
            assert chapter2.synthesis_status == "done"
