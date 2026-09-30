from __future__ import annotations

from pathlib import Path


def _make_book_and_job(db, job_id: int, process_dir: str, title: str = "Test Book"):
    from syntrive.db.models import Book, Job

    book = Book(title=title)
    db.add(book)
    db.flush()
    job = Job(
        id=job_id,
        book_id=book.id,
        process_dir=process_dir,
        epub_path=f"{process_dir}/book.epub",
    )
    db.add(job)
    db.flush()
    return book, job


def _make_chapter(
    db, job_id: int, chapter_id: str, *, sequence_number: str, chapter_number: str,
    transcript_path: str = None, transcript_lines: int = None,
    synthesis_status: str = "pending", synthesis_batch_id: int = None,
    chapter_audio_path: str = None, chapter_audio_seconds: float = None,
    chapter_name: str = None,
):
    from syntrive.db.models import TranscriptChapter

    chapter = TranscriptChapter(
        job_id=job_id, chapter_id=chapter_id,
        sequence_number=sequence_number, chapter_number=chapter_number,
        chapter_name=chapter_name,
        transcript_path=transcript_path, transcript_lines=transcript_lines,
        synthesis_status=synthesis_status, synthesis_batch_id=synthesis_batch_id,
        chapter_audio_path=chapter_audio_path, chapter_audio_seconds=chapter_audio_seconds,
    )
    db.add(chapter)
    db.flush()
    return chapter


class TestFindEarliestActiveBatch:
    def test_returns_none_when_no_batch_exists(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import find_earliest_active_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            assert find_earliest_active_batch(db) is None

    def test_skips_a_fully_done_batch_and_returns_the_next_active_one(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import find_earliest_active_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            batch1 = SynthesisBatch()
            db.add(batch1)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="done", synthesis_batch_id=batch1.synthesis_batch_id,
            )
            batch2 = SynthesisBatch()
            db.add(batch2)
            db.flush()
            _make_chapter(
                db, 1, "ch_0002", sequence_number="0002", chapter_number="0002",
                synthesis_status="queued", synthesis_batch_id=batch2.synthesis_batch_id,
            )

            found = find_earliest_active_batch(db)
            assert found is not None
            assert found.synthesis_batch_id == batch2.synthesis_batch_id

    def test_a_batch_with_one_active_chapter_among_done_ones_is_still_active(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import find_earliest_active_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="done", synthesis_batch_id=batch.synthesis_batch_id,
            )
            _make_chapter(
                db, 1, "ch_0002", sequence_number="0002", chapter_number="0002",
                synthesis_status="synthesizing", synthesis_batch_id=batch.synthesis_batch_id,
            )

            found = find_earliest_active_batch(db)
            assert found is not None
            assert found.synthesis_batch_id == batch.synthesis_batch_id

    def test_a_paused_batch_is_skipped_and_the_next_active_one_returned(self, tmp_path: Path):
        from datetime import datetime

        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import find_earliest_active_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            batch1 = SynthesisBatch(paused_at=datetime.utcnow())
            db.add(batch1)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="queued", synthesis_batch_id=batch1.synthesis_batch_id,
            )
            batch2 = SynthesisBatch()
            db.add(batch2)
            db.flush()
            _make_chapter(
                db, 1, "ch_0002", sequence_number="0002", chapter_number="0002",
                synthesis_status="queued", synthesis_batch_id=batch2.synthesis_batch_id,
            )

            found = find_earliest_active_batch(db)
            assert found is not None
            assert found.synthesis_batch_id == batch2.synthesis_batch_id

    def test_all_batches_paused_returns_none(self, tmp_path: Path):
        from datetime import datetime

        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import find_earliest_active_batch

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            batch = SynthesisBatch(paused_at=datetime.utcnow())
            db.add(batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )
            assert find_earliest_active_batch(db) is None


class TestActiveBatchIds:
    def test_returns_empty_when_no_batch_exists(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import active_batch_ids

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            assert active_batch_ids(db) == []

    def test_returns_every_active_batch_ascending_skipping_done_and_paused(self, tmp_path: Path):
        from datetime import datetime

        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import active_batch_ids

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))

            done_batch = SynthesisBatch()
            db.add(done_batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="done", synthesis_batch_id=done_batch.synthesis_batch_id,
            )

            paused_batch = SynthesisBatch(paused_at=datetime.utcnow())
            db.add(paused_batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0002", sequence_number="0002", chapter_number="0002",
                synthesis_status="queued", synthesis_batch_id=paused_batch.synthesis_batch_id,
            )

            active_batch_2 = SynthesisBatch()
            db.add(active_batch_2)
            db.flush()
            _make_chapter(
                db, 1, "ch_0003", sequence_number="0003", chapter_number="0003",
                synthesis_status="synthesizing", synthesis_batch_id=active_batch_2.synthesis_batch_id,
            )

            active_batch_1 = SynthesisBatch()
            db.add(active_batch_1)
            db.flush()
            _make_chapter(
                db, 1, "ch_0004", sequence_number="0004", chapter_number="0004",
                synthesis_status="queued", synthesis_batch_id=active_batch_1.synthesis_batch_id,
            )

            assert active_batch_ids(db) == [
                active_batch_2.synthesis_batch_id, active_batch_1.synthesis_batch_id,
            ]


class TestPausableBooks:
    def test_lists_non_done_batch_assigned_chapters_grouped_by_book_with_pause_state(self, tmp_path: Path):
        from datetime import datetime

        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import pausable_books

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path / "job1"), title="Book A")
            _make_book_and_job(db, 2, str(tmp_path / "job2"), title="Book B")

            active = SynthesisBatch()
            paused = SynthesisBatch(paused_at=datetime.utcnow())
            done_batch = SynthesisBatch()
            db.add_all([active, paused, done_batch])
            db.flush()

            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                chapter_name="Alpha", synthesis_status="queued",
                synthesis_batch_id=active.synthesis_batch_id,
            )
            _make_chapter(
                db, 1, "ch_0002", sequence_number="0002", chapter_number="0002",
                synthesis_status="combining", synthesis_batch_id=paused.synthesis_batch_id,
            )
            _make_chapter(
                db, 2, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="done", synthesis_batch_id=done_batch.synthesis_batch_id,
            )
            _make_chapter(
                db, 2, "ch_0002", sequence_number="0002", chapter_number="0002",
                synthesis_status="pending",
            )

        with get_db_session(db_path) as db:
            books = pausable_books(db)

        assert [b.book_title for b in books] == ["Book A"]
        book_a = books[0]
        assert book_a.job_id == 1
        assert [(c.chapter_id, c.paused, c.synthesis_status) for c in book_a.chapters] == [
            ("ch_0001", False, "queued"),
            ("ch_0002", True, "combining"),
        ]
        assert book_a.paused_count == 1
        assert book_a.chapters[0].chapter_name == "Alpha"

    def test_empty_when_nothing_is_batch_assigned(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import pausable_books

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="pending",
            )
        with get_db_session(db_path) as db:
            assert pausable_books(db) == []


class TestBatchChapters:
    def test_orders_by_job_id_then_sequence_number_across_multiple_books(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import batch_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path / "job1"), title="Book A")
            _make_book_and_job(db, 2, str(tmp_path / "job2"), title="Book B")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _make_chapter(db, 2, "ch_0001", sequence_number="0001", chapter_number="0001", synthesis_batch_id=batch.synthesis_batch_id)
            _make_chapter(db, 1, "ch_0002", sequence_number="0002", chapter_number="0002", synthesis_batch_id=batch.synthesis_batch_id)
            _make_chapter(db, 1, "ch_0001", sequence_number="0001", chapter_number="0001", synthesis_batch_id=batch.synthesis_batch_id)

            chapters = batch_chapters(db, batch.synthesis_batch_id)
            assert [(c.job_id, c.sequence_number) for c in chapters] == [
                (1, "0001"), (1, "0002"), (2, "0001"),
            ]

    def test_a_chapter_not_in_this_batch_is_excluded(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import batch_chapters

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            batch1 = SynthesisBatch()
            batch2 = SynthesisBatch()
            db.add_all([batch1, batch2])
            db.flush()
            _make_chapter(db, 1, "ch_0001", sequence_number="0001", chapter_number="0001", synthesis_batch_id=batch1.synthesis_batch_id)
            _make_chapter(db, 1, "ch_0002", sequence_number="0002", chapter_number="0002", synthesis_batch_id=batch2.synthesis_batch_id)

            chapters = batch_chapters(db, batch1.synthesis_batch_id)
            assert [c.chapter_id for c in chapters] == ["ch_0001"]


class TestBuildProgressPlan:
    def test_numbers_books_in_first_seen_order_and_chapters_by_sequence(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import build_progress_plan

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path / "job1"), title="Book A")
            _make_book_and_job(db, 2, str(tmp_path / "job2"), title="Book B")
            _make_chapter(db, 1, "ch_0001", sequence_number="0001", chapter_number="0001", synthesis_status="queued")
            _make_chapter(db, 1, "ch_0002", sequence_number="0002", chapter_number="0002", synthesis_status="synthesizing")
            _make_chapter(db, 2, "ch_0001", sequence_number="0001", chapter_number="0001", synthesis_status="combining")

            plan = build_progress_plan(db)

            assert plan[(1, "ch_0001")].book_name == "Book A"
            assert (plan[(1, "ch_0001")].book_seq, plan[(1, "ch_0001")].book_total) == (1, 2)
            assert (plan[(1, "ch_0001")].chapter_seq, plan[(1, "ch_0001")].chapter_total) == (1, 2)
            assert (plan[(1, "ch_0002")].chapter_seq, plan[(1, "ch_0002")].chapter_total) == (2, 2)
            assert plan[(2, "ch_0001")].book_name == "Book B"
            assert (plan[(2, "ch_0001")].book_seq, plan[(2, "ch_0001")].book_total) == (2, 2)
            assert (plan[(2, "ch_0001")].chapter_seq, plan[(2, "ch_0001")].chapter_total) == (1, 1)

    def test_done_chapters_are_excluded_from_the_snapshot(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import build_progress_plan

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            _make_chapter(db, 1, "ch_0001", sequence_number="0001", chapter_number="0001", synthesis_status="done")
            _make_chapter(db, 1, "ch_0002", sequence_number="0002", chapter_number="0002", synthesis_status="queued")

            plan = build_progress_plan(db)

            assert (1, "ch_0001") not in plan
            assert (plan[(1, "ch_0002")].chapter_seq, plan[(1, "ch_0002")].chapter_total) == (1, 1)


class TestBatchStatus:
    def test_none_for_an_unknown_batch_id(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import batch_status

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            assert batch_status(db, 999) is None

    def test_reports_synth_error_and_chapter_rows(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import batch_status

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path), title="Elon Musk")
            batch = SynthesisBatch(synth_error="engine crashed", failed_line_index=7)
            db.add(batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                chapter_name="The Reckoning",
                synthesis_status="synthesizing", synthesis_batch_id=batch.synthesis_batch_id,
            )

            report = batch_status(db, batch.synthesis_batch_id)
            assert report is not None
            assert report.synth_error == "engine crashed"
            assert report.failed_line_index == 7
            assert report.is_stuck is True
            assert report.is_done is False
            assert len(report.chapters) == 1
            assert report.chapters[0].chapter_id == "ch_0001"
            assert report.chapters[0].book_title == "Elon Musk"
            assert report.chapters[0].chapter_name == "The Reckoning"

    def test_chapter_name_is_none_on_pre_m024_rows(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import batch_status

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path), title="Legacy Book")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )

            report = batch_status(db, batch.synthesis_batch_id)
            assert report.chapters[0].book_title == "Legacy Book"
            assert report.chapters[0].chapter_name is None

    def test_is_done_true_when_every_chapter_is_done(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import batch_status

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, str(tmp_path))
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                synthesis_status="done", synthesis_batch_id=batch.synthesis_batch_id,
                chapter_audio_path="job1/audiobooks/book_ch0001.m4a", chapter_audio_seconds=123.4,
            )

            report = batch_status(db, batch.synthesis_batch_id)
            assert report.is_done is True
            assert report.is_stuck is False
            assert report.chapters[0].chapter_audio_seconds == 123.4


class TestSchedulableBooks:
    def test_a_chapter_with_matching_file_and_line_count_is_schedulable(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        process_dir = tmp_path / "PROCESSING-test"
        script_dir = process_dir / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        (script_dir / "ch_0001.txt").write_text("line one\nline two\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test", title="Schedulable Book")
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=2,
            )

            books = schedulable_books(db, tmp_path)
            assert len(books) == 1
            assert books[0].book_title == "Schedulable Book"
            assert len(books[0].chapters) == 1

    def test_an_archived_job_is_never_offered(self, tmp_path: Path):
        from datetime import datetime

        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        script_dir = tmp_path / "PROCESSING-test" / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        (script_dir / "ch_0001.txt").write_text("line one\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _, job = _make_book_and_job(db, 1, "PROCESSING-test", title="Archived Book")
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            )
            assert len(schedulable_books(db, tmp_path)) == 1

            job.archived_at = datetime.utcnow()
            db.flush()
            assert schedulable_books(db, tmp_path) == []

    def test_a_done_chapter_is_never_schedulable(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        process_dir = tmp_path / "PROCESSING-test"
        script_dir = process_dir / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        (script_dir / "ch_0001.txt").write_text("line one\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test")
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
                synthesis_status="done",
            )

            assert schedulable_books(db, tmp_path) == []

    def test_a_chapter_already_assigned_to_a_batch_is_not_schedulable(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        script_dir = tmp_path / "PROCESSING-test" / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        for name in ("ch_0001", "ch_0002"):
            (script_dir / f"{name}.txt").write_text("line one\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test", title="Half-Enqueued Book")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            )
            _make_chapter(
                db, 1, "ch_0002", sequence_number="0002", chapter_number="0002",
                transcript_path="transcript_text/tts_script/ch_0002.txt", transcript_lines=1,
            )

            books = schedulable_books(db, tmp_path)
            assert len(books) == 1
            assert [leaf_value.chapter_id for leaf_value in books[0].chapters] == ["ch_0002"]

    def test_a_rolled_back_redo_chapter_still_in_its_batch_is_not_schedulable(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        script_dir = tmp_path / "PROCESSING-test" / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        (script_dir / "ch_0001.txt").write_text("line one\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test")
            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
                synthesis_status="combining", synthesis_batch_id=batch.synthesis_batch_id,
            )

            assert schedulable_books(db, tmp_path) == []

    def test_a_missing_transcript_file_is_not_schedulable(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test")
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            )

            assert schedulable_books(db, tmp_path) == []

    def test_a_line_count_mismatch_is_not_schedulable(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        process_dir = tmp_path / "PROCESSING-test"
        script_dir = process_dir / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        (script_dir / "ch_0001.txt").write_text("line one\nline two\nline three\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test")
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=2,
            )

            assert schedulable_books(db, tmp_path) == []

    def test_a_gap_in_sequence_numbers_excludes_the_whole_book(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        process_dir = tmp_path / "PROCESSING-test"
        script_dir = process_dir / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        (script_dir / "ch_0001.txt").write_text("line one\n", encoding="utf-8")
        (script_dir / "ch_0003.txt").write_text("line one\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test")
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            )
            _make_chapter(
                db, 1, "ch_0003", sequence_number="0003", chapter_number="0003",
                transcript_path="transcript_text/tts_script/ch_0003.txt", transcript_lines=1,
            )

            assert schedulable_books(db, tmp_path) == []

    def test_front_matter_at_sequence_one_does_not_exclude_the_book(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        script_dir = tmp_path / "PROCESSING-test" / "transcript_text" / "tts_script"
        script_dir.mkdir(parents=True)
        for name in ("front", "ch_0001", "ch_0002"):
            (script_dir / f"{name}.txt").write_text("only line\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-test", title="Book With Front Matter")
            _make_chapter(
                db, 1, "front", sequence_number="0001", chapter_number="",
                transcript_path="transcript_text/tts_script/front.txt", transcript_lines=1,
            )
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0002", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            )
            _make_chapter(
                db, 1, "ch_0002", sequence_number="0003", chapter_number="0002",
                transcript_path="transcript_text/tts_script/ch_0002.txt", transcript_lines=1,
            )

            books = schedulable_books(db, tmp_path)
            assert len(books) == 1
            assert len(books[0].chapters) == 3

    def test_multiple_books_each_contribute_their_own_schedulable_chapters(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.orchestration.jobs import schedulable_books

        db_path = tmp_path / "db.sqlite"
        for name in ("book1", "book2"):
            script_dir = tmp_path / f"PROCESSING-{name}" / "transcript_text" / "tts_script"
            script_dir.mkdir(parents=True)
            (script_dir / "ch_0001.txt").write_text("only line\n", encoding="utf-8")

        with get_db_session(db_path) as db:
            _make_book_and_job(db, 1, "PROCESSING-book1", title="Book One")
            _make_chapter(
                db, 1, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            )
            _make_book_and_job(db, 2, "PROCESSING-book2", title="Book Two")
            _make_chapter(
                db, 2, "ch_0001", sequence_number="0001", chapter_number="0001",
                transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            )

            books = schedulable_books(db, tmp_path)
            assert {b.book_title for b in books} == {"Book One", "Book Two"}
