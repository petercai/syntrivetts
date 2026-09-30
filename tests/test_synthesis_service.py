from __future__ import annotations

from pathlib import Path


def _seed_pending_chapter(db_path: Path, *, job_id: int = 1) -> int:
    from syntrive.db.models import Book, Job, TranscriptChapter, TtsConfig
    from syntrive.db.session import get_db_session

    repo_dir = db_path.parent
    process_dir = f"job{job_id}"
    script_dir = repo_dir / process_dir / "transcript_text" / "tts_script"
    script_dir.mkdir(parents=True, exist_ok=True)
    (script_dir / "ch_0001.txt").write_text("only line\n", encoding="utf-8")

    with get_db_session(db_path) as db:
        book = Book(title="Service Test Book")
        db.add(book)
        db.flush()
        job = Job(id=job_id, book_id=book.id, process_dir=process_dir, epub_path=f"{process_dir}/book.epub")
        db.add(job)
        db.flush()
        db.add(TtsConfig(job_id=job_id, engine="xtts"))
        chapter = TranscriptChapter(
            job_id=job_id, chapter_id="ch_0001",
            sequence_number="0001", chapter_number="0001",
            transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            synthesis_status="pending",
        )
        db.add(chapter)
        db.flush()
        return chapter.id


def _add_pending_chapter(db_path: Path, chapter_id: str, sequence_number: str, *, job_id: int = 1) -> int:
    from syntrive.db.models import TranscriptChapter
    from syntrive.db.session import get_db_session

    process_dir = f"job{job_id}"
    script_dir = db_path.parent / process_dir / "transcript_text" / "tts_script"
    (script_dir / f"{chapter_id}.txt").write_text("only line\n", encoding="utf-8")

    with get_db_session(db_path) as db:
        chapter = TranscriptChapter(
            job_id=job_id, chapter_id=chapter_id,
            sequence_number=sequence_number, chapter_number=sequence_number,
            transcript_path=f"transcript_text/tts_script/{chapter_id}.txt", transcript_lines=1,
            synthesis_status="pending",
        )
        db.add(chapter)
        db.flush()
        return chapter.id


def _fake_engine_get_engine(monkeypatch):
    import syntrive.orchestration.workflow as workflow_module

    class _FakeEngine:
        engine_name = "cosyvoice"
        options = {}

        def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
            import torch
            import torchaudio

            torchaudio.save(output_path, torch.full((1, 4800), 0.4), 24000)
            return output_path

        def resolve_voice_for_tts_script(self, tts_config_id, voice_id, db, cache):
            return None

    monkeypatch.setattr(workflow_module.manager, "get_engine", lambda name, options=None: _FakeEngine())


class TestListSchedulableBooks:
    def test_returns_the_seeded_pending_chapter(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        _seed_pending_chapter(db_path)

        books = synthesis_service.list_schedulable_books(db_path)
        assert len(books) == 1
        assert len(books[0].chapters) == 1

    def test_schedulable_chapter_fields_survive_session_close(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        chapter_db_id = _seed_pending_chapter(db_path)

        books = synthesis_service.list_schedulable_books(db_path)
        entry = books[0].chapters[0]
        assert entry.chapter_db_id == chapter_db_id
        assert entry.chapter_id == "ch_0001"
        assert entry.sequence_number == "0001"
        assert entry.chapter_name is None
        assert entry.book_title == "Service Test Book"


class TestEnqueueAndRunNextBatch:
    def test_no_active_batch_returns_none(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        _seed_pending_chapter(db_path)

        assert synthesis_service.run_next_batch(db_path) is None

    def test_enqueue_then_run_next_batch_round_trips_through_separate_sessions(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        chapter_db_id = _seed_pending_chapter(db_path)

        enqueue_result = synthesis_service.enqueue(db_path, [chapter_db_id], note="svc test")
        assert enqueue_result.ok is True
        assert len(enqueue_result.batch_ids) == 1
        batch_id = enqueue_result.batch_ids[0]

        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).filter_by(id=chapter_db_id).first()
            assert chapter.synthesis_status == "queued"
            assert chapter.synthesis_batch_id == batch_id

        import syntrive.orchestration.workflow as workflow_module

        class _FakeEngine:
            engine_name = "cosyvoice"
            options = {}

            def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
                import torch
                import torchaudio

                torchaudio.save(output_path, torch.full((1, 4800), 0.4), 24000)
                return output_path

            def resolve_voice_for_tts_script(self, tts_config_id, voice_id, db, cache):
                return None

        monkeypatch.setattr(workflow_module.manager, "get_engine", lambda name, options=None: _FakeEngine())

        result = synthesis_service.run_next_batch(db_path)
        assert result is not None
        assert result.ok is True
        assert result.chapters_completed == 1

        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).filter_by(id=chapter_db_id).first()
            assert chapter.synthesis_status == "done"


class TestRunActiveBatches:
    def test_no_active_batch_reports_zero_batches_run(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        _seed_pending_chapter(db_path)

        result = synthesis_service.run_active_batches(db_path)
        assert result.ok is True
        assert result.batches_run == 0
        assert result.chapters_completed == 0

    def test_done_chapter_with_deleted_artifacts_is_reconciled_and_run(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        chapter_db_id = _seed_pending_chapter(db_path)
        batch_id = synthesis_service.enqueue(db_path, [chapter_db_id]).batch_ids[0]
        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).filter_by(id=chapter_db_id).one()
            chapter.synthesis_status = "done"
            chapter.chapter_audio_path = "job1/audiobooks/missing.m4a"

        _fake_engine_get_engine(monkeypatch)
        result = synthesis_service.run_active_batches(db_path, only_batch_id=batch_id)

        assert result.ok is True
        assert result.batches_run == 1
        assert result.chapters_completed == 1
        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).filter_by(id=chapter_db_id).one()
            assert chapter.synthesis_status == "done"
            assert (db_path.parent / chapter.chapter_audio_path).is_file()

    def test_drains_every_per_chapter_batch_in_one_call(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        id1 = _seed_pending_chapter(db_path)
        id2 = _add_pending_chapter(db_path, "ch_0002", "0002")

        enqueue_result = synthesis_service.enqueue(db_path, [id2, id1])
        assert enqueue_result.ok is True
        assert len(enqueue_result.batch_ids) == 2

        _fake_engine_get_engine(monkeypatch)

        seen_batches = []
        result = synthesis_service.run_active_batches(
            db_path,
            on_event=lambda name, f: seen_batches.append(f["batch_id"])
            if name == "synth_batch_start" else None,
        )
        assert result.ok is True
        assert result.batches_run == 2
        assert result.chapters_completed == 2
        assert seen_batches == sorted(set(seen_batches)) == list(enqueue_result.batch_ids)

        with get_db_session(db_path) as db:
            rows = db.query(TranscriptChapter).filter(TranscriptChapter.id.in_([id1, id2])).all()
            assert all(r.synthesis_status == "done" for r in rows)

    def test_only_batch_id_runs_just_that_batch(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        id1 = _seed_pending_chapter(db_path)
        id2 = _add_pending_chapter(db_path, "ch_0002", "0002")
        synthesis_service.enqueue(db_path, [id1, id2])
        with get_db_session(db_path) as db:
            target_batch_id = db.query(TranscriptChapter).filter_by(id=id2).one().synthesis_batch_id
        assert target_batch_id is not None
        _fake_engine_get_engine(monkeypatch)

        seen = []
        result = synthesis_service.run_active_batches(
            db_path, only_batch_id=target_batch_id,
            on_event=lambda name, f: seen.append(f["batch_id"]) if name == "synth_batch_start" else None,
        )
        assert result.ok is True
        assert result.batches_run == 1
        assert result.batch_not_found is False
        assert seen == [target_batch_id]

        with get_db_session(db_path) as db:
            by_id = {r.id: r.synthesis_status for r in
                     db.query(TranscriptChapter).filter(TranscriptChapter.id.in_([id1, id2])).all()}
            assert by_id[id2] == "done"
            assert by_id[id1] == "queued"

    def test_only_batch_id_unknown_reports_batch_not_found(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        _seed_pending_chapter(db_path)

        result = synthesis_service.run_active_batches(db_path, only_batch_id=999)
        assert result.ok is False
        assert result.batch_not_found is True
        assert result.batches_run == 0

    def test_a_failed_batch_does_not_stop_the_rest_of_the_queue(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service
        import syntrive.orchestration.workflow as workflow_module

        db_path = tmp_path / "db.sqlite"
        id1 = _seed_pending_chapter(db_path)
        id2 = _add_pending_chapter(db_path, "ch_0002", "0002")
        id3 = _add_pending_chapter(db_path, "ch_0003", "0003")
        (db_path.parent / "job1" / "transcript_text" / "tts_script" / "ch_0002.txt").write_text(
            "bad line\n", encoding="utf-8",
        )
        enqueue_result = synthesis_service.enqueue(db_path, [id1, id2, id3])
        assert len(enqueue_result.batch_ids) == 3

        class _FlakyEngine:
            engine_name = "cosyvoice"
            options = {}

            def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
                if text == "bad line":
                    return None
                import torch
                import torchaudio

                torchaudio.save(output_path, torch.full((1, 4800), 0.4), 24000)
                return output_path

            def resolve_voice_for_tts_script(self, tts_config_id, voice_id, db, cache):
                return None

        monkeypatch.setattr(workflow_module.manager, "get_engine", lambda name, options=None: _FlakyEngine())

        result = synthesis_service.run_active_batches(db_path)

        assert result.ok is False
        assert result.batches_run == 3
        assert result.chapters_completed == 2
        assert len(result.failed_results) == 1
        assert result.failed_results[0].failed_chapter_id == "ch_0002"
        assert result.last_result is result.failed_results[-1]

        with get_db_session(db_path) as db:
            by_id = {r.id: r.synthesis_status for r in
                     db.query(TranscriptChapter).filter(TranscriptChapter.id.in_([id1, id2, id3])).all()}
            assert by_id[id1] == "done"
            assert by_id[id3] == "done"
            assert by_id[id2] == "synthesizing"

    def test_circuit_breaker_stops_the_queue_after_consecutive_failures(self, tmp_path, monkeypatch):
        from syntrive.services import synthesis_service
        import syntrive.orchestration.workflow as workflow_module

        db_path = tmp_path / "db.sqlite"
        ids = [_seed_pending_chapter(db_path)]
        for n in range(2, 6):
            ids.append(_add_pending_chapter(db_path, f"ch_000{n}", f"000{n}"))
        for n in range(1, 6):
            (db_path.parent / "job1" / "transcript_text" / "tts_script" / f"ch_000{n}.txt").write_text(
                "bad line\n", encoding="utf-8",
            )
        synthesis_service.enqueue(db_path, ids)

        class _AlwaysFailing:
            engine_name = "cosyvoice"
            options = {}
            last_error = "worker error: boom"

            def synthesize(self, text, voice=None, leading_silence_ms=0, output_path=None):
                return None

            def resolve_voice_for_tts_script(self, tts_config_id, voice_id, db, cache):
                return None

        monkeypatch.setattr(workflow_module.manager, "get_engine", lambda name, options=None: _AlwaysFailing())
        monkeypatch.delenv("SYNTRIVE_MAX_CONSECUTIVE_BATCH_FAILURES", raising=False)

        result = synthesis_service.run_active_batches(db_path)

        assert result.ok is False
        assert result.batches_run == 3
        assert len(result.failed_results) == 3

        monkeypatch.setenv("SYNTRIVE_MAX_CONSECUTIVE_BATCH_FAILURES", "0")
        result = synthesis_service.run_active_batches(db_path)
        assert result.batches_run == 5

    def test_an_unexpected_exception_mid_batch_does_not_crash_the_whole_run(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        id1 = _seed_pending_chapter(db_path)
        id2 = _add_pending_chapter(db_path, "ch_0002", "0002")
        id3 = _add_pending_chapter(db_path, "ch_0003", "0003")
        enqueue_result = synthesis_service.enqueue(db_path, [id1, id2, id3])
        assert len(enqueue_result.batch_ids) == 3
        crashing_batch_id = enqueue_result.batch_ids[1]

        real_run_one_batch = synthesis_service.run_one_batch

        def _flaky_run_one_batch(db_path_arg, batch_id, **kwargs):
            if batch_id == crashing_batch_id:
                raise RuntimeError("simulated worker-IPC crash (round 40)")
            return real_run_one_batch(db_path_arg, batch_id, **kwargs)

        monkeypatch.setattr(synthesis_service, "run_one_batch", _flaky_run_one_batch)
        _fake_engine_get_engine(monkeypatch)

        result = synthesis_service.run_active_batches(db_path)

        assert result.ok is False
        assert result.batches_run == 3
        assert result.chapters_completed == 2
        assert len(result.failed_results) == 1
        assert "simulated worker-IPC crash" in result.failed_results[0].error

        with get_db_session(db_path) as db:
            by_id = {r.id: r.synthesis_status for r in
                     db.query(TranscriptChapter).filter(TranscriptChapter.id.in_([id1, id2, id3])).all()}
            assert by_id[id1] == "done"
            assert by_id[id3] == "done"
            assert by_id[id2] == "queued"

        from syntrive.db.models import SynthesisBatch
        from syntrive.services import queue_overview

        with get_db_session(db_path) as db:
            batch = db.query(SynthesisBatch).filter_by(synthesis_batch_id=crashing_batch_id).one()
            assert "simulated worker-IPC crash" in batch.synth_error and batch.failed_line_index is None
        overview = queue_overview.read_queue(db_path)
        assert [r.batch_id for r in overview.stuck] == [crashing_batch_id]


class TestPauseState:
    def test_list_pausable_books_shows_an_enqueued_chapter(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        cid = _seed_pending_chapter(db_path)
        synthesis_service.enqueue(db_path, [cid])

        books = synthesis_service.list_pausable_books(db_path)
        assert [b.book_title for b in books] == ["Service Test Book"]
        assert [c.chapter_id for c in books[0].chapters] == ["ch_0001"]
        assert books[0].chapters[0].paused is False

    def test_pausing_a_batch_makes_run_active_batches_skip_it_then_resume_runs_it(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        cid = _seed_pending_chapter(db_path)
        enq = synthesis_service.enqueue(db_path, [cid])
        batch_id = enq.batch_ids[0]
        _fake_engine_get_engine(monkeypatch)

        pause = synthesis_service.set_pause_state(db_path, pause_batch_ids=[batch_id])
        assert pause.ok and pause.flipped == 1

        result = synthesis_service.run_active_batches(db_path)
        assert result.ok is True and result.batches_run == 0
        with get_db_session(db_path) as db:
            assert db.query(TranscriptChapter).filter_by(id=cid).one().synthesis_status == "queued"

        reports = synthesis_service.get_status(db_path, batch_id=batch_id)
        assert reports[0].paused_at is not None

        resume = synthesis_service.set_pause_state(db_path, resume_batch_ids=[batch_id])
        assert resume.ok and resume.flipped == 1

        result = synthesis_service.run_active_batches(db_path)
        assert result.ok is True and result.batches_run == 1 and result.chapters_completed == 1
        with get_db_session(db_path) as db:
            assert db.query(TranscriptChapter).filter_by(id=cid).one().synthesis_status == "done"

    def test_explicit_batch_id_run_overrides_a_pause(self, tmp_path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        cid = _seed_pending_chapter(db_path)
        batch_id = synthesis_service.enqueue(db_path, [cid]).batch_ids[0]
        _fake_engine_get_engine(monkeypatch)
        synthesis_service.set_pause_state(db_path, pause_batch_ids=[batch_id])

        result = synthesis_service.run_active_batches(db_path, only_batch_id=batch_id)
        assert result.ok is True and result.batches_run == 1
        with get_db_session(db_path) as db:
            assert db.query(TranscriptChapter).filter_by(id=cid).one().synthesis_status == "done"

    def test_set_pause_state_with_an_unknown_id_writes_nothing(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        cid = _seed_pending_chapter(db_path)
        batch_id = synthesis_service.enqueue(db_path, [cid]).batch_ids[0]

        result = synthesis_service.set_pause_state(db_path, pause_batch_ids=[batch_id, 999])
        assert result.ok is False
        assert synthesis_service.get_status(db_path, batch_id=batch_id)[0].paused_at is None


class TestGetStatus:
    def test_unknown_batch_returns_empty_list(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        assert synthesis_service.get_status(db_path, batch_id=999) == []

    def test_no_batch_id_reports_every_batch(self, tmp_path: Path):
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        chapter_db_id = _seed_pending_chapter(db_path)
        enqueue_result = synthesis_service.enqueue(db_path, [chapter_db_id])

        reports = synthesis_service.get_status(db_path)
        assert len(reports) == 1
        assert reports[0].batch_id == enqueue_result.batch_ids[0]


class TestRunRedoDetection:
    def test_rolls_back_a_done_chapter_with_a_missing_m4a(self, tmp_path: Path):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        chapter_db_id = _seed_pending_chapter(db_path)
        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).filter_by(id=chapter_db_id).first()
            chapter.synthesis_status = "done"
            chapter.chapter_audio_path = "job1/audiobooks/missing.m4a"

        rolled_back = synthesis_service.run_redo_detection(db_path)
        assert rolled_back == 1

        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).filter_by(id=chapter_db_id).first()
            assert chapter.synthesis_status in ("combining", "synthesizing")

    def test_a_genuinely_done_chapter_is_left_alone(self, tmp_path: Path):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        db_path = tmp_path / "db.sqlite"
        chapter_db_id = _seed_pending_chapter(db_path)
        m4a_rel = "job1/audiobooks/ok.m4a"
        (db_path.parent / "job1" / "audiobooks").mkdir(parents=True)
        (db_path.parent / m4a_rel).write_bytes(b"x")
        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).filter_by(id=chapter_db_id).first()
            chapter.synthesis_status = "done"
            chapter.chapter_audio_path = m4a_rel

        assert synthesis_service.run_redo_detection(db_path) == 0
