from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import tts_dbu as td  # noqa: E402


def _seed_full_job(
    db_path: Path, *, title: str = "1984", author: str = "Orwell",
    process_dir: str = "PROCESSING-1984", reference_voice_id: "int | None" = None,
) -> tuple[int, int]:
    from syntrive.db.models import (
        Book, BookImage, Job, OutputConfig, StageEvent, ThresholdBlockEvent,
        TranscriptChapter, TtsConfig, TtsVoice, WorkflowStepEvent,
    )
    from syntrive.db.session import ensure_schema, get_db_session, make_engine

    ensure_schema(make_engine(db_path))
    with get_db_session(db_path) as db:
        book = Book(title=title, author=author, language="en")
        db.add(book)
        db.flush()
        db.add(BookImage(book_id=book.id, name="cover.jpg", path="images/cover.jpg", image_type="cover"))

        job = Job(
            book_id=book.id, process_dir=process_dir, epub_path=f"{process_dir}/book.epub",
            stage="transcript_review", status="running",
        )
        db.add(job)
        db.flush()

        tts_config = TtsConfig(job_id=job.id, engine="xtts", language="en")
        db.add(tts_config)
        db.flush()
        db.add(TtsVoice(
            tts_config_id=tts_config.id, name="narrator", voice_id=0,
            reference_voice_id=reference_voice_id,
        ))

        db.add(OutputConfig(job_id=job.id, output_format="m4a"))
        db.add(StageEvent(job_id=job.id, stage_name="transcript_html", status="completed"))
        db.add(TranscriptChapter(job_id=job.id, chapter_id="ch_0001", title="Chapter 1"))
        db.add(ThresholdBlockEvent(job_id=job.id, deletion_ratio=0.1))
        db.add(WorkflowStepEvent(job_id=job.id, step_name="TRANSCRIPT_REVIEW", action="confirmed"))
        db.flush()
        return book.id, job.id


def _seed_reference_voice(db_path: Path, *, path: str = "voices/en/adult/male/alice.wav", name: str = "alice") -> int:
    from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag
    from syntrive.db.session import ensure_schema, get_db_session, make_engine

    ensure_schema(make_engine(db_path))
    with get_db_session(db_path) as db:
        rv = ReferenceVoice(path=path, name=name, gender="male", language="en")
        db.add(rv)
        db.flush()
        db.add(ReferenceVoiceTag(reference_voice_id=rv.id, tag="adult"))
        return rv.id


class TestPathRoundTrip:
    def test_under_repo_root_is_relative(self):
        abs_path = td._REPO_ROOT / "voices" / "en" / "adult" / "foo.wav"
        stored = td.to_repo_relative_or_abs(abs_path)
        assert not Path(stored).is_absolute()
        assert stored == "voices/en/adult/foo.wav"
        assert td.resolve_stored_path(stored) == abs_path.resolve()

    def test_outside_repo_root_is_absolute(self, tmp_path: Path):
        abs_path = tmp_path / "external" / "bar.wav"
        stored = td.to_repo_relative_or_abs(abs_path)
        assert Path(stored).is_absolute()
        assert td.resolve_stored_path(stored) == abs_path.resolve()

    def test_stored_path_is_always_posix_style_on_every_os(self, tmp_path: Path):
        under_root = td.to_repo_relative_or_abs(td._REPO_ROOT / "voices" / "en" / "foo.wav")
        outside_root = td.to_repo_relative_or_abs(tmp_path / "external" / "bar.wav")
        assert "\\" not in under_root
        assert "\\" not in outside_root


class TestIsRelativeStoredPath:
    def test_relative_path_is_portable(self):
        assert td.is_relative_stored_path("voices/en/adult/male/foo.wav") is True

    def test_absolute_path_is_not_portable(self, tmp_path: Path):
        abs_str = td.to_repo_relative_or_abs(tmp_path / "external" / "bar.wav")
        assert td.is_relative_stored_path(abs_str) is False


class TestSqlLiteral:
    def test_none_becomes_null(self):
        assert td._sql_literal(None) == "NULL"

    def test_bool_becomes_0_or_1(self):
        assert td._sql_literal(True) == "1"
        assert td._sql_literal(False) == "0"

    def test_numbers_pass_through(self):
        assert td._sql_literal(42) == "42"
        assert td._sql_literal(3.5) == "3.5"

    def test_string_is_quoted_and_escaped(self):
        assert td._sql_literal("voices/en/adult/male/o'brien.wav") == "'voices/en/adult/male/o''brien.wav'"


class TestBackupRestoreRoundTrip:
    def _seed_db(self, db_path: Path):
        from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            relative_voice = ReferenceVoice(
                path="voices/en/adult/male/alice.wav", name="alice", gender="male", language="en",
                accent="southern", sample_rate=16000, bit_depth=16, channels=1, duration_seconds=2.5,
            )
            absolute_voice = ReferenceVoice(
                path="/external/drive/bob.wav", name="bob", gender="male", language="en",
                sample_rate=22050, bit_depth=16, channels=1, duration_seconds=1.0,
            )
            db.add(relative_voice)
            db.add(absolute_voice)
            db.flush()
            db.add(ReferenceVoiceTag(reference_voice_id=relative_voice.id, tag="adult"))
            db.add(ReferenceVoiceTag(reference_voice_id=relative_voice.id, tag="narrator"))
            db.add(ReferenceVoiceTag(reference_voice_id=absolute_voice.id, tag="adult"))

    def test_backup_skips_absolute_path_rows(self, tmp_path: Path):
        db_path = tmp_path / "source.db"
        self._seed_db(db_path)
        backup_path = tmp_path / "backup.sql"

        voices, skipped, tags = td.backup_reference_voices(db_path, backup_path)

        assert (voices, skipped, tags) == (1, 1, 2)
        content = backup_path.read_text(encoding="utf-8")
        assert "alice" in content
        assert "bob" not in content
        assert "INSERT INTO reference_voices" in content
        assert "INSERT INTO reference_voice_tags" in content

    def test_restore_reproduces_relative_voice_and_its_tags_only(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        source_db = tmp_path / "source.db"
        self._seed_db(source_db)
        backup_path = tmp_path / "backup.sql"
        td.backup_reference_voices(source_db, backup_path)

        target_db = tmp_path / "target.db"
        ensure_schema(make_engine(target_db))
        voice_inserts, tag_inserts = td.restore_reference_voices(target_db, backup_path)
        assert (voice_inserts, tag_inserts) == (1, 2)

        with get_db_session(target_db) as db:
            rows = db.query(ReferenceVoice).all()
            assert [r.name for r in rows] == ["alice"]
            assert rows[0].duration_seconds == 2.5
            tags = {t.tag for t in db.query(ReferenceVoiceTag).all()}
            assert tags == {"adult", "narrator"}

    def test_restore_missing_file_raises(self, tmp_path: Path):
        db_path = tmp_path / "target.db"
        from syntrive.db.session import ensure_schema, make_engine

        ensure_schema(make_engine(db_path))
        with pytest.raises(FileNotFoundError):
            td.restore_reference_voices(db_path, tmp_path / "does_not_exist.sql")

    def test_restore_onto_conflicting_id_raises_and_does_not_silently_overwrite(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        source_db = tmp_path / "source.db"
        self._seed_db(source_db)
        backup_path = tmp_path / "backup.sql"
        td.backup_reference_voices(source_db, backup_path)

        target_db = tmp_path / "target.db"
        ensure_schema(make_engine(target_db))
        with get_db_session(target_db) as db:
            db.add(ReferenceVoice(
                path="voices/en/adult/female/other.wav", name="other", gender="female", language="en",
            ))

        with pytest.raises(Exception):
            td.restore_reference_voices(target_db, backup_path)

        with get_db_session(target_db) as db:
            rows = db.query(ReferenceVoice).all()
            assert [r.name for r in rows] == ["other"]


class TestExpandBookSelection:
    def _summaries(self):
        return [
            td.BookSummary(
                book_id=1, title="Book A", author=None,
                jobs=(td.JobSummary(job_id=10, process_dir="p10", stage="s", status="running"),
                      td.JobSummary(job_id=11, process_dir="p11", stage="s", status="running")),
            ),
            td.BookSummary(
                book_id=2, title="Book B", author=None,
                jobs=(td.JobSummary(job_id=20, process_dir="p20", stage="s", status="running"),),
            ),
        ]

    def test_book_selection_expands_to_all_its_jobs(self):
        assert td.expand_book_selection(self._summaries(), book_ids=[1]) == [10, 11]

    def test_explicit_job_selection_included_even_without_its_book(self):
        assert td.expand_book_selection(self._summaries(), job_ids=[20]) == [20]

    def test_book_and_job_selections_are_unioned_and_deduped(self):
        result = td.expand_book_selection(self._summaries(), book_ids=[1], job_ids=[10, 20])
        assert result == [10, 11, 20]

    def test_empty_selection_returns_empty(self):
        assert td.expand_book_selection(self._summaries()) == []


class TestListBooksWithJobs:
    def test_returns_book_and_job_summaries(self, tmp_path: Path):
        db_path = tmp_path / "syntrivetts.db"
        book_id, job_id = _seed_full_job(db_path)

        summaries = td.list_books_with_jobs(db_path)

        assert len(summaries) == 1
        assert summaries[0].book_id == book_id
        assert summaries[0].title == "1984"
        assert len(summaries[0].jobs) == 1
        assert summaries[0].jobs[0].job_id == job_id
        assert summaries[0].jobs[0].process_dir == "PROCESSING-1984"


class TestCopyRow:
    def test_copies_every_column_except_id_and_applies_overrides(self, tmp_path: Path):
        from syntrive.db.models import Book

        source = Book(id=999, title="Some Title", author="Some Author", language="en")
        copied = td._copy_row(source, Book, author="Overridden Author")

        assert copied.id is None
        assert copied.title == "Some Title"
        assert copied.author == "Overridden Author"
        assert copied.language == "en"


class TestExportJobs:
    def test_export_creates_standalone_file_with_all_child_rows(self, tmp_path: Path):
        from syntrive.db.models import (
            Book, BookImage, Job, OutputConfig, StageEvent, ThresholdBlockEvent,
            TranscriptChapter, TtsConfig, TtsVoice, WorkflowStepEvent,
        )
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        _, job_id = _seed_full_job(source_db)
        output_path = tmp_path / "export.db"

        result = td.export_jobs(source_db, output_path, [job_id])

        assert output_path.is_file()
        assert (result.book_count, result.job_count, result.reference_voice_count) == (1, 1, 0)
        assert result.process_dir_reminders == ("PROCESSING-1984",)

        with get_db_session(output_path) as db:
            assert db.query(Book).count() == 1
            assert db.query(BookImage).count() == 1
            assert db.query(Job).count() == 1
            assert db.query(TtsConfig).count() == 1
            assert db.query(TtsVoice).count() == 1
            assert db.query(OutputConfig).count() == 1
            assert db.query(StageEvent).count() == 1
            assert db.query(TranscriptChapter).count() == 1
            assert db.query(ThresholdBlockEvent).count() == 1
            assert db.query(WorkflowStepEvent).count() == 1
            exported_job = db.query(Job).one()
            assert not hasattr(Job, "repo_dir")
            assert exported_job.process_dir == "PROCESSING-1984"

    def test_export_missing_job_id_raises(self, tmp_path: Path):
        source_db = tmp_path / "source.db"
        _seed_full_job(source_db)
        with pytest.raises(ValueError):
            td.export_jobs(source_db, tmp_path / "export.db", [9999])

    def test_export_empty_job_ids_raises(self, tmp_path: Path):
        source_db = tmp_path / "source.db"
        _seed_full_job(source_db)
        with pytest.raises(ValueError):
            td.export_jobs(source_db, tmp_path / "export.db", [])

    def test_reference_voice_included_when_opted_in(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag, TtsVoice
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        ref_id = _seed_reference_voice(source_db)
        _, job_id = _seed_full_job(source_db, reference_voice_id=ref_id)
        output_path = tmp_path / "export.db"

        result = td.export_jobs(source_db, output_path, [job_id], include_reference_voices=True)

        assert result.reference_voice_count == 1
        assert result.unbound_tts_voice_warnings == ()
        with get_db_session(output_path) as db:
            rv = db.query(ReferenceVoice).one()
            assert rv.name == "alice"
            assert {t.tag for t in db.query(ReferenceVoiceTag).all()} == {"adult"}
            voice = db.query(TtsVoice).one()
            assert voice.reference_voice_id == rv.id

    def test_reference_voice_excluded_and_warned_when_not_opted_in(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        ref_id = _seed_reference_voice(source_db)
        _, job_id = _seed_full_job(source_db, reference_voice_id=ref_id)
        output_path = tmp_path / "export.db"

        result = td.export_jobs(source_db, output_path, [job_id], include_reference_voices=False)

        assert result.reference_voice_count == 0
        assert len(result.unbound_tts_voice_warnings) == 1
        assert result.unbound_tts_voice_warnings[0] == (job_id, "narrator")
        with get_db_session(output_path) as db:
            assert db.query(ReferenceVoice).count() == 0
            voice = db.query(TtsVoice).one()
            assert voice.reference_voice_id is None

    def test_entire_catalog_included_when_opted_in_even_if_unreferenced(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        ref_id = _seed_reference_voice(source_db, path="voices/en/adult/male/alice.wav", name="alice")
        _seed_reference_voice(source_db, path="voices/en/adult/female/carol.wav", name="carol")
        _, job_id = _seed_full_job(source_db, reference_voice_id=ref_id)
        output_path = tmp_path / "export.db"

        result = td.export_jobs(source_db, output_path, [job_id], include_reference_voices=True)

        assert result.reference_voice_count == 2
        with get_db_session(output_path) as db:
            names = {rv.name for rv in db.query(ReferenceVoice).all()}
            assert names == {"alice", "carol"}

    def test_multiple_jobs_same_book_share_one_exported_book_row(self, tmp_path: Path):
        from syntrive.db.models import Book, Job
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        book_id, job1_id = _seed_full_job(source_db, process_dir="PROCESSING-1984-run1")
        with get_db_session(source_db) as db:
            job2 = Job(book_id=book_id, process_dir="PROCESSING-1984-run2", epub_path="book.epub")
            db.add(job2)
            db.flush()
            job2_id = job2.id

        output_path = tmp_path / "export.db"
        result = td.export_jobs(source_db, output_path, [job1_id, job2_id])

        assert (result.book_count, result.job_count) == (1, 2)
        with get_db_session(output_path) as db:
            assert db.query(Book).count() == 1
            assert db.query(Job).count() == 2

    def test_export_detaches_synthesis_batch_and_resets_inflight_status(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch, TranscriptChapter
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        _, job1_id = _seed_full_job(source_db, title="A", process_dir="PROCESSING-A")
        _, job2_id = _seed_full_job(source_db, title="B", process_dir="PROCESSING-B")
        with get_db_session(source_db) as db:
            db.add(SynthesisBatch(synthesis_batch_id=7))
            db.flush()
            rows = db.query(TranscriptChapter).order_by(TranscriptChapter.job_id).all()
            rows[0].synthesis_batch_id, rows[0].synthesis_status = 7, "queued"
            rows[1].synthesis_batch_id, rows[1].synthesis_status = 7, "done"
            rows[1].chapter_audio_path = "audiobooks/ch_0001.flac"

        output_path = tmp_path / "export.db"
        td.export_jobs(source_db, output_path, [job1_id, job2_id])

        with get_db_session(output_path) as db:
            got = db.query(TranscriptChapter).order_by(TranscriptChapter.job_id).all()
            assert [c.synthesis_batch_id for c in got] == [None, None]
            assert [c.synthesis_status for c in got] == ["pending", "done"]
            assert got[1].chapter_audio_path == "audiobooks/ch_0001.flac"


class TestFindJobConflicts:
    def test_no_conflict_when_book_not_in_target(self, tmp_path: Path):
        source_db = tmp_path / "source.db"
        _, job_id = _seed_full_job(source_db)
        export_path = tmp_path / "export.db"
        td.export_jobs(source_db, export_path, [job_id])

        target_db = tmp_path / "target.db"
        from syntrive.db.session import ensure_schema, make_engine
        ensure_schema(make_engine(target_db))

        assert td.find_job_conflicts(export_path, target_db) == []

    def test_conflict_detected_when_book_and_process_dir_match(self, tmp_path: Path):
        source_db = tmp_path / "source.db"
        _, job_id = _seed_full_job(source_db)
        export_path = tmp_path / "export.db"
        td.export_jobs(source_db, export_path, [job_id])

        target_db = tmp_path / "target.db"
        target_book_id, target_job_id = _seed_full_job(target_db)

        conflicts = td.find_job_conflicts(export_path, target_db)
        assert len(conflicts) == 1
        assert conflicts[0].target_job_id == target_job_id
        assert conflicts[0].process_dir == "PROCESSING-1984"


class TestImportJobs:
    def _export(self, tmp_path: Path, **seed_kwargs) -> Path:
        source_db = tmp_path / f"source-{seed_kwargs.get('process_dir', 'default')}.db"
        _, job_id = _seed_full_job(source_db, **seed_kwargs)
        export_path = tmp_path / f"export-{seed_kwargs.get('process_dir', 'default')}.db"
        td.export_jobs(source_db, export_path, [job_id])
        return export_path

    def test_import_detaches_synthesis_batch_from_legacy_export_file(self, tmp_path: Path):
        from syntrive.db.models import SynthesisBatch, TranscriptChapter
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        export_path = tmp_path / "legacy-export.db"
        _seed_full_job(export_path)
        with get_db_session(export_path) as db:
            db.add(SynthesisBatch(synthesis_batch_id=7))
            db.flush()
            chapter = db.query(TranscriptChapter).one()
            chapter.synthesis_batch_id, chapter.synthesis_status = 7, "synthesizing"
        target_db = tmp_path / "target.db"
        ensure_schema(make_engine(target_db))

        td.import_jobs(export_path, target_db)

        with get_db_session(target_db) as db:
            chapter = db.query(TranscriptChapter).one()
            assert (chapter.synthesis_batch_id, chapter.synthesis_status) == (None, "pending")

    def test_import_creates_new_book_and_job(self, tmp_path: Path):
        from syntrive.db.models import Book, Job
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        export_path = self._export(tmp_path)
        target_db = tmp_path / "target.db"
        ensure_schema(make_engine(target_db))

        result = td.import_jobs(export_path, target_db)

        assert (result.created_job_count, result.overwritten_job_count, result.skipped_job_count) == (1, 0, 0)
        with get_db_session(target_db) as db:
            assert db.query(Book).count() == 1
            assert db.query(Job).count() == 1
            assert db.query(Job).one().process_dir == "PROCESSING-1984"

    def test_import_reuses_existing_book_by_title_author(self, tmp_path: Path):
        from syntrive.db.models import Book
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        export_path = self._export(tmp_path, process_dir="PROCESSING-1984-second-run")
        target_db = tmp_path / "target.db"
        existing_book_id, _ = _seed_full_job(target_db, process_dir="PROCESSING-1984-first-run")

        td.import_jobs(export_path, target_db)

        with get_db_session(target_db) as db:
            assert db.query(Book).count() == 1
            books = db.query(Book).all()
            assert books[0].id == existing_book_id

    def test_import_book_image_add_if_missing_never_overwrites_existing(self, tmp_path: Path):
        from syntrive.db.models import BookImage
        from syntrive.db.session import get_db_session

        export_path = self._export(tmp_path, process_dir="PROCESSING-1984-second-run")
        target_db = tmp_path / "target.db"
        target_book_id, _ = _seed_full_job(target_db, process_dir="PROCESSING-1984-first-run")
        with get_db_session(target_db) as db:
            existing_image = db.query(BookImage).filter(BookImage.book_id == target_book_id).one()
            existing_image.path = "images/manually-changed-cover.jpg"

        td.import_jobs(export_path, target_db)

        with get_db_session(target_db) as db:
            images = db.query(BookImage).filter(BookImage.book_id == target_book_id).all()
            assert len(images) == 1
            assert images[0].path == "images/manually-changed-cover.jpg"

    def test_import_book_cover_fill_if_missing_never_overwrites_existing(self, tmp_path: Path):
        from syntrive.db.models import Book
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        _, job_id = _seed_full_job(source_db)
        with get_db_session(source_db) as db:
            db.query(Book).one().cover = "images/cover.jpg"
        export_path = tmp_path / "export.db"
        td.export_jobs(source_db, export_path, [job_id])

        target_db = tmp_path / "target.db"
        target_book_id, _ = _seed_full_job(target_db, process_dir="PROCESSING-1984-other-run")
        with get_db_session(target_db) as db:
            db.query(Book).filter(Book.id == target_book_id).one().cover = "images/user-chosen-cover.jpg"

        td.import_jobs(export_path, target_db)

        with get_db_session(target_db) as db:
            assert db.query(Book).filter(Book.id == target_book_id).one().cover == "images/user-chosen-cover.jpg"

    def test_import_conflict_without_resolution_raises(self, tmp_path: Path):
        export_path = self._export(tmp_path)
        target_db = tmp_path / "target.db"
        _seed_full_job(target_db)

        with pytest.raises(ValueError):
            td.import_jobs(export_path, target_db)

    def test_import_conflict_skip(self, tmp_path: Path):
        from syntrive.db.models import Job
        from syntrive.db.session import get_db_session

        export_path = self._export(tmp_path)
        target_db = tmp_path / "target.db"
        _, existing_job_id = _seed_full_job(target_db)

        conflicts = td.find_job_conflicts(export_path, target_db)
        result = td.import_jobs(
            export_path, target_db, skip_job_ids=frozenset(c.export_job_id for c in conflicts),
        )

        assert (result.created_job_count, result.skipped_job_count) == (0, 1)
        with get_db_session(target_db) as db:
            assert db.query(Job).count() == 1
            assert db.query(Job).one().id == existing_job_id

    def test_import_conflict_overwrite_replaces_child_rows(self, tmp_path: Path):
        from syntrive.db.models import Job, StageEvent
        from syntrive.db.session import get_db_session

        export_path = self._export(tmp_path)
        target_db = tmp_path / "target.db"
        _, existing_job_id = _seed_full_job(target_db)
        with get_db_session(target_db) as db:
            db.add(StageEvent(job_id=existing_job_id, stage_name="stale_stage", status="completed"))

        conflicts = td.find_job_conflicts(export_path, target_db)
        result = td.import_jobs(
            export_path, target_db, overwrite_job_ids=frozenset(c.export_job_id for c in conflicts),
        )

        assert (result.overwritten_job_count, result.created_job_count) == (1, 0)
        with get_db_session(target_db) as db:
            assert db.query(Job).count() == 1
            new_job = db.query(Job).one()
            stage_names = {se.stage_name for se in db.query(StageEvent).filter(StageEvent.job_id == new_job.id)}
            assert stage_names == {"transcript_html"}

    def test_reference_voice_resolved_by_path_reuses_existing_target_row(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        source_db = tmp_path / "source.db"
        ref_id = _seed_reference_voice(source_db, path="voices/en/adult/male/alice.wav")
        _, job_id = _seed_full_job(source_db, reference_voice_id=ref_id)
        export_path = tmp_path / "export.db"
        td.export_jobs(source_db, export_path, [job_id], include_reference_voices=True)

        target_db = tmp_path / "target.db"
        target_ref_id = _seed_reference_voice(target_db, path="voices/en/adult/male/alice.wav", name="alice-target")

        result = td.import_jobs(export_path, target_db)

        assert result.reference_voice_resolved_count == 1
        with get_db_session(target_db) as db:
            assert db.query(ReferenceVoice).count() == 1
            voice = db.query(TtsVoice).one()
            assert voice.reference_voice_id == target_ref_id

    def test_reference_voice_inserted_when_not_present_in_target(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        source_db = tmp_path / "source.db"
        ref_id = _seed_reference_voice(source_db)
        _, job_id = _seed_full_job(source_db, reference_voice_id=ref_id)
        export_path = tmp_path / "export.db"
        td.export_jobs(source_db, export_path, [job_id], include_reference_voices=True)

        target_db = tmp_path / "target.db"
        ensure_schema(make_engine(target_db))

        result = td.import_jobs(export_path, target_db)

        assert result.reference_voice_resolved_count == 1
        with get_db_session(target_db) as db:
            assert db.query(ReferenceVoice).count() == 1

    def test_unbound_voice_stays_unbound_on_import(self, tmp_path: Path):
        from syntrive.db.models import TtsVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        export_path = self._export(tmp_path)
        target_db = tmp_path / "target.db"
        ensure_schema(make_engine(target_db))

        result = td.import_jobs(export_path, target_db)

        assert (result.reference_voice_resolved_count, result.reference_voice_unresolved_count) == (0, 0)
        with get_db_session(target_db) as db:
            assert db.query(TtsVoice).one().reference_voice_id is None

    def test_import_brings_in_unreferenced_reference_voices_too(self, tmp_path: Path):
        from syntrive.db.models import ReferenceVoice
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        source_db = tmp_path / "source.db"
        ref_id = _seed_reference_voice(source_db, path="voices/en/adult/male/alice.wav", name="alice")
        _seed_reference_voice(source_db, path="voices/en/adult/female/carol.wav", name="carol")
        _, job_id = _seed_full_job(source_db, reference_voice_id=ref_id)
        export_path = tmp_path / "export.db"
        td.export_jobs(source_db, export_path, [job_id], include_reference_voices=True)

        target_db = tmp_path / "target.db"
        ensure_schema(make_engine(target_db))

        result = td.import_jobs(export_path, target_db)

        assert result.reference_voice_resolved_count == 1
        with get_db_session(target_db) as db:
            names = {rv.name for rv in db.query(ReferenceVoice).all()}
            assert names == {"alice", "carol"}


class TestCliArgs:
    def test_export_book_and_job_are_repeatable(self):
        args = td._parse_args(["db.sqlite", "--export-book", "1", "--export-book", "2", "--export-job", "5"])
        assert args.export_book == [1, 2]
        assert args.export_job == [5]

    def test_backup_restore_import_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            td._parse_args(["db.sqlite", "--backup", "--restore"])
        with pytest.raises(SystemExit):
            td._parse_args(["db.sqlite", "--backup", "--import", "export.db"])

    def test_on_conflict_choices_are_restricted(self):
        args = td._parse_args(["db.sqlite", "--import", "export.db", "--on-conflict", "overwrite"])
        assert args.on_conflict == "overwrite"
        with pytest.raises(SystemExit):
            td._parse_args(["db.sqlite", "--import", "export.db", "--on-conflict", "bogus"])

    def test_backup_flag_alone_uses_default_filename(self):
        args = td._parse_args(["db.sqlite", "--backup"])
        assert args.backup == td._DEFAULT_BACKUP_FILENAME

    def test_backup_help_text_clarifies_it_excludes_books_and_jobs(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
            td._parse_args(["db.sqlite", "--help"])
        rendered = buf.getvalue()
        assert "reference-voice catalog ONLY" in rendered
        assert "NOT books/jobs" in rendered


class TestMainRuntimeMessaging:
    def test_backup_prints_scope_clarifying_note(self, tmp_path: Path, capsys):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        exit_code = td.main([str(db_path), "--backup", str(tmp_path / "backup.sql")])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "covers ONLY the reference-voice catalog" in out

    def test_directory_with_explicit_action_flag_is_rejected(self, tmp_path: Path, caplog):
        exit_code = td.main([str(tmp_path), "--backup"])

        assert exit_code == 1
        assert "must be a database FILE" in caplog.text

    def test_directory_with_no_flags_and_no_syntrivetts_db_auto_creates_it(self, tmp_path: Path, capsys):
        exit_code = td.main([str(tmp_path)])

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "creating a new one" in out
        assert "No previously-exported .db files found" in out
        assert (tmp_path / "syntrivetts.db").is_file()


class TestFindExportFiles:
    def test_lists_db_files_except_the_live_syntrivetts_db(self, tmp_path: Path):
        (tmp_path / "syntrivetts.db").write_bytes(b"")
        (tmp_path / "syntrivetts-export-2026-01-01-000000.db").write_bytes(b"")
        (tmp_path / "custom-name.db").write_bytes(b"")
        (tmp_path / "not-a-db.txt").write_bytes(b"")

        found = {p.name for p in td.find_export_files(tmp_path)}

        assert found == {"syntrivetts-export-2026-01-01-000000.db", "custom-name.db"}

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        assert td.find_export_files(tmp_path) == []


class TestRunImportWizardFromRepoDir:
    def test_no_syntrivetts_db_in_repo_dir_is_auto_created(self, tmp_path: Path, capsys):
        exit_code = td.run_import_wizard_from_repo_dir(tmp_path)

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "creating a new one" in out
        assert "No previously-exported .db files found" in out
        assert (tmp_path / "syntrivetts.db").is_file()

    def test_no_export_files_in_repo_dir_returns_early(self, tmp_path: Path, capsys):
        from syntrive.db.session import ensure_schema, make_engine

        ensure_schema(make_engine(tmp_path / "syntrivetts.db"))

        exit_code = td.run_import_wizard_from_repo_dir(tmp_path)

        assert exit_code == 1
        assert "No previously-exported .db files found" in capsys.readouterr().out


class TestRunWizardDispatch:
    def test_directory_dispatches_to_import_wizard(self, tmp_path: Path, capsys):
        exit_code = td.run_wizard(tmp_path)

        assert exit_code == 1
        assert "No previously-exported .db files found" in capsys.readouterr().out

    def test_file_dispatches_to_export_wizard(self, tmp_path: Path, capsys):
        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path))

        exit_code = td.run_wizard(db_path)

        assert exit_code == 0
        assert "No books found" in capsys.readouterr().out
