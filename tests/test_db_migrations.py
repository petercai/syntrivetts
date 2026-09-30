from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    finally:
        conn.close()


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    finally:
        conn.close()


class TestSentenceAudioPathM029:
    def test_repairs_only_the_exact_legacy_path_and_stays_idempotent(self, tmp_path: Path):
        from syntrive.db.models import Book, Job
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            book = Book(title="Legacy Paths")
            db.add(book)
            db.flush()
            db.add_all([
                Job(
                    book_id=book.id,
                    process_dir="PROCESSING-Legacy",
                    epub_path="PROCESSING-Legacy/book.epub",
                    sentence_audio_dir="PROCESSING-Legacy/chapters/sentences",
                ),
                Job(
                    book_id=book.id,
                    process_dir="PROCESSING-Custom",
                    epub_path="PROCESSING-Custom/book.epub",
                    sentence_audio_dir="custom/audio",
                ),
            ])

        ensure_schema(make_engine(db_path))
        ensure_schema(make_engine(db_path))

        with get_db_session(db_path) as db:
            paths = {
                job.process_dir: job.sentence_audio_dir
                for job in db.query(Job).order_by(Job.id).all()
            }
        assert paths == {
            "PROCESSING-Legacy": "PROCESSING-Legacy/sentence_audio",
            "PROCESSING-Custom": "custom/audio",
        }


class TestTranscriptChapterTextPathRetired:
    def test_fresh_db_has_no_phantom_text_path_column(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        cols = _columns(db_path, "transcript_chapters")
        assert "text_path" not in cols
        assert "transcript_path" in cols

    def test_repeated_ensure_schema_calls_stay_stable(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        for _ in range(3):
            ensure_schema(make_engine(db_path))

        assert "text_path" not in _columns(db_path, "transcript_chapters")


class TestBusyTimeoutConfigured:
    def test_busy_timeout_pragma_is_applied_on_connect(self, tmp_path: Path):
        from syntrive.db.session import SQLITE_BUSY_TIMEOUT_MS, make_engine

        db_path = tmp_path / "syntrivetts.db"
        engine = make_engine(db_path)
        with engine.connect() as conn:
            value = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
        assert value == SQLITE_BUSY_TIMEOUT_MS


class TestM008IdempotentGuard:
    def test_update_only_runs_when_a_row_needs_it(self, tmp_path: Path):
        from syntrive.db.models import Book
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            db.add(Book(title="Already Migrated", extraction_mode="none"))

        conn = sqlite3.connect(str(db_path))
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                started = time.perf_counter()
                ensure_schema(make_engine(db_path))
                elapsed = time.perf_counter() - started
            finally:
                conn.execute("COMMIT")
        finally:
            conn.close()
        assert elapsed < 5.0, (
            f"ensure_schema() took {elapsed:.1f}s while a write lock was held "
            "elsewhere -- M008 must be reading-before-writing, not attempting "
            "an unconditional UPDATE on every call"
        )

    def test_still_migrates_genuinely_unmigrated_rows(self, tmp_path: Path):
        from syntrive.db.models import Book
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            db.add(Book(title="Legacy Auto Mode", extraction_mode="auto"))
            db.add(Book(title="Legacy Null Mode", extraction_mode=None))

        ensure_schema(make_engine(db_path))

        with get_db_session(db_path) as db:
            modes = {book.title: book.extraction_mode for book in db.query(Book).all()}
        assert modes == {"Legacy Auto Mode": "none", "Legacy Null Mode": "none"}

    def test_legacy_db_with_both_columns_gets_text_path_dropped(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        conn = sqlite3.connect(str(db_path))
        conn.execute("ALTER TABLE transcript_chapters ADD COLUMN text_path VARCHAR(1024)")
        conn.commit()
        conn.close()
        assert "text_path" in _columns(db_path, "transcript_chapters")

        ensure_schema(make_engine(db_path))

        cols = _columns(db_path, "transcript_chapters")
        assert "text_path" not in cols
        assert "transcript_path" in cols

    def test_genuinely_unmigrated_legacy_db_still_gets_renamed_not_dropped(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import TranscriptChapter

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            from syntrive.db.models import Book, Job

            book = Book(title="Legacy Book")
            db.add(book)
            db.flush()
            job = Job(book_id=book.id, process_dir="PROCESSING-Legacy", epub_path="book.epub")
            db.add(job)
            db.flush()
            db.add(TranscriptChapter(job_id=job.id, chapter_id="ch_0001", transcript_path="raw/ch_0001.txt"))

        conn = sqlite3.connect(str(db_path))
        conn.execute("ALTER TABLE transcript_chapters RENAME COLUMN transcript_path TO text_path")
        conn.commit()
        conn.close()
        assert _columns(db_path, "transcript_chapters") >= {"text_path"}
        assert "transcript_path" not in _columns(db_path, "transcript_chapters")

        ensure_schema(make_engine(db_path))

        cols = _columns(db_path, "transcript_chapters")
        assert "text_path" not in cols
        assert "transcript_path" in cols
        with get_db_session(db_path) as db:
            chapter = db.query(TranscriptChapter).one()
            assert chapter.transcript_path == "raw/ch_0001.txt"


class TestSynthesisBatchesM025:
    def test_fresh_db_has_synthesis_batches_table_and_new_columns(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        assert "synthesis_batches" in _tables(db_path)
        assert _columns(db_path, "synthesis_batches") == {
            "synthesis_batch_id", "created_at", "note",
            "synth_error", "failed_line_index", "synth_started_at", "synth_finished_at",
            "paused_at",
        }
        tc_cols = _columns(db_path, "transcript_chapters")
        assert {"synthesis_batch_id", "chapter_audio_path", "chapter_audio_seconds"} <= tc_cols

    def test_repeated_ensure_schema_calls_stay_stable(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        for _ in range(3):
            ensure_schema(make_engine(db_path))

        tc_cols = _columns(db_path, "transcript_chapters")
        assert {"synthesis_batch_id", "chapter_audio_path", "chapter_audio_seconds"} <= tc_cols

    def test_legacy_db_missing_m025_surface_gets_backfilled(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        conn = sqlite3.connect(str(db_path))
        conn.execute("ALTER TABLE transcript_chapters DROP COLUMN chapter_audio_path")
        conn.execute("ALTER TABLE transcript_chapters DROP COLUMN chapter_audio_seconds")
        conn.execute("DROP TABLE synthesis_batches")
        conn.commit()
        conn.close()
        assert "synthesis_batches" not in _tables(db_path)

        ensure_schema(make_engine(db_path))

        assert "synthesis_batches" in _tables(db_path)
        tc_cols = _columns(db_path, "transcript_chapters")
        assert {"synthesis_batch_id", "chapter_audio_path", "chapter_audio_seconds"} <= tc_cols

    def test_orm_round_trip_links_chapter_to_batch_and_derives_no_status_column(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import Book, Job, SynthesisBatch, TranscriptChapter

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        with get_db_session(db_path) as db:
            book = Book(title="Batch Book")
            db.add(book)
            db.flush()
            job = Job(book_id=book.id, process_dir="PROCESSING-Batch", epub_path="book.epub")
            db.add(job)
            db.flush()

            batch = SynthesisBatch()
            db.add(batch)
            db.flush()
            assert batch.synthesis_batch_id == 1

            db.add(TranscriptChapter(
                job_id=job.id, chapter_id="ch_0001", sequence_number="0001",
                synthesis_status="synthesizing", synthesis_batch_id=batch.synthesis_batch_id,
            ))
            db.add(TranscriptChapter(
                job_id=job.id, chapter_id="ch_0002", sequence_number="0002",
                synthesis_status="queued", synthesis_batch_id=batch.synthesis_batch_id,
            ))

        with get_db_session(db_path) as db:
            batch = db.query(SynthesisBatch).one()
            batch.synth_error = "engine crashed"
            batch.failed_line_index = 42

        with get_db_session(db_path) as db:
            batch = db.query(SynthesisBatch).one()
            assert len(batch.chapters) == 2

            stuck = (
                db.query(TranscriptChapter)
                .filter(
                    TranscriptChapter.synthesis_batch_id == batch.synthesis_batch_id,
                    TranscriptChapter.synthesis_status.notin_(("done", "queued")),
                )
                .one()
            )
            assert stuck.chapter_id == "ch_0001"
            assert stuck.synthesis_batch.synth_error == "engine crashed"


class TestTtsConfigModelM026:
    def test_fresh_db_has_the_model_column(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        assert "model" in _columns(db_path, "tts_configs")

    def test_repeated_ensure_schema_calls_stay_stable(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        for _ in range(3):
            ensure_schema(make_engine(db_path))
        assert "model" in _columns(db_path, "tts_configs")

    def test_legacy_db_without_model_gets_backfilled_non_destructively(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import Book, Job, TtsConfig

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        with get_db_session(db_path) as db:
            book = Book(title="M026 Book")
            db.add(book)
            db.flush()
            job = Job(book_id=book.id, process_dir="PROCESSING-M026", epub_path="b.epub")
            db.add(job)
            db.flush()
            db.add(TtsConfig(job_id=job.id, engine="cosyvoice"))

        conn = sqlite3.connect(str(db_path))
        conn.execute("ALTER TABLE tts_configs DROP COLUMN model")
        conn.commit()
        conn.close()
        assert "model" not in _columns(db_path, "tts_configs")

        ensure_schema(make_engine(db_path))
        assert "model" in _columns(db_path, "tts_configs")
        with get_db_session(db_path) as db:
            cfg = db.query(TtsConfig).one()
            assert cfg.engine == "cosyvoice"
            assert cfg.model is None

    def test_orm_round_trip_persists_model(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import Book, Job, TtsConfig

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            book = Book(title="Model RT")
            db.add(book)
            db.flush()
            job = Job(book_id=book.id, process_dir="PROCESSING-RT", epub_path="b.epub")
            db.add(job)
            db.flush()
            db.add(TtsConfig(job_id=job.id, engine="cosyvoice", model="cosyvoice-300m-sft"))

        with get_db_session(db_path) as db:
            assert db.query(TtsConfig).one().model == "cosyvoice-300m-sft"


class TestSynthesisBatchPausedAtM027:
    def test_fresh_db_has_the_paused_at_column(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        assert "paused_at" in _columns(db_path, "synthesis_batches")

    def test_repeated_ensure_schema_calls_stay_stable(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        for _ in range(3):
            ensure_schema(make_engine(db_path))
        assert "paused_at" in _columns(db_path, "synthesis_batches")

    def test_legacy_db_without_paused_at_gets_backfilled_non_destructively(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import SynthesisBatch

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            db.add(SynthesisBatch(note="pre-M027 batch"))

        conn = sqlite3.connect(str(db_path))
        conn.execute("ALTER TABLE synthesis_batches DROP COLUMN paused_at")
        conn.commit()
        conn.close()
        assert "paused_at" not in _columns(db_path, "synthesis_batches")

        ensure_schema(make_engine(db_path))
        assert "paused_at" in _columns(db_path, "synthesis_batches")
        with get_db_session(db_path) as db:
            batch = db.query(SynthesisBatch).one()
            assert batch.note == "pre-M027 batch"
            assert batch.paused_at is None

    def test_orm_round_trip_persists_paused_at(self, tmp_path: Path):
        from datetime import datetime

        from syntrive.db.session import ensure_schema, get_db_session, make_engine
        from syntrive.db.models import SynthesisBatch

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))
        stamp = datetime(2026, 9, 9, 12, 0, 0)
        with get_db_session(db_path) as db:
            db.add(SynthesisBatch(paused_at=stamp))

        with get_db_session(db_path) as db:
            assert db.query(SynthesisBatch).one().paused_at == stamp


class TestSpeechEventRecordsDroppedM028:
    def test_fresh_db_has_no_speech_event_records_table(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        assert "speech_event_records" not in _tables(db_path)

    def test_legacy_db_gets_table_dropped_and_other_data_kept(self, tmp_path: Path):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO books (title) VALUES ('Legacy Book')")
        conn.execute(
            "CREATE TABLE speech_event_records (id INTEGER PRIMARY KEY, text TEXT)"
        )
        conn.execute("INSERT INTO speech_event_records (text) VALUES ('hello')")
        conn.commit()
        conn.close()
        assert "speech_event_records" in _tables(db_path)

        for _ in range(2):
            ensure_schema(make_engine(db_path))

        assert "speech_event_records" not in _tables(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            titles = [r[0] for r in conn.execute("SELECT title FROM books").fetchall()]
        finally:
            conn.close()
        assert titles == ["Legacy Book"]
