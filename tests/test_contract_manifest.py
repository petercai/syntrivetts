from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest

from syntrive.contract.manifest import (
    BookFacts,
    ChapterFacts,
    FactsAnchor,
    JobFacts,
    JobSnapshot,
    build_job_manifest,
    content_fingerprint,
    to_process_relative,
)
from syntrive.contract.writer import WriteStatus, write_manifest

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "syntrive" / "contract" / "schemas"
T0 = datetime(2026, 9, 25, 8, 0, 0)
T1 = datetime(2026, 9, 25, 9, 30, 0)


def _validate(data: dict, schema_name: str) -> None:
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(data)


def _snapshot() -> JobSnapshot:
    job = JobFacts(
        id=7,
        process_dir="PROCESSING-Book",
        status="running",
        current_step="tts_config",
        created_at=T0,
        updated_at=T0,
        source_path="PROCESSING-Book/book.epub",
        cover_path="images/cover.jpeg",
        audiobooks_dir="PROCESSING-Book/audiobooks",
        sentence_audio_dir="PROCESSING-Book/sentence_audio",
        chapter_audio_dir="PROCESSING-Book/chapter_audio",
        transcript_dir="PROCESSING-Book/transcript_text/tts_script",
    )
    book = BookFacts(id=3, title="Book", author="Author", language="en", cover="images/cover.jpeg")
    chapters = (
        ChapterFacts(chapter_id="chapter_002", sequence_number="0002", chapter_name="Two",
                     transcript_path="transcript_text/tts_script/ch_0002.txt"),
        ChapterFacts(chapter_id="front", sequence_number=None, chapter_name="Front matter"),
        ChapterFacts(chapter_id="chapter_001", sequence_number="0001", chapter_name="One",
                     transcript_path="transcript_text/tts_script/ch_0001.txt",
                     chapter_audio_path="PROCESSING-Book/audiobooks/Book_ch0001.m4a",
                     chapter_audio_seconds=61.25, synthesis_status="done"),
    )
    return JobSnapshot(job=job, book=book, chapters=chapters)


class TestToProcessRelative:
    def test_repo_anchored_path_is_rebased_onto_process_dir(self):
        assert to_process_relative("PROCESSING-B/audiobooks", FactsAnchor.REPO, "PROCESSING-B", "f") == ("audiobooks", None)

    def test_process_anchored_path_passes_through_and_backslashes_are_normalised(self):
        assert to_process_relative("images\\cover.jpg", FactsAnchor.PROCESS, "PROCESSING-B", "f") == ("images/cover.jpg", None)

    def test_process_dir_itself_maps_to_dot(self):
        assert to_process_relative("PROCESSING-B", FactsAnchor.REPO, "PROCESSING-B", "f") == (".", None)

    @pytest.mark.parametrize("value", ["C:\\repo\\PROCESSING-B\\a.m4a", "/abs/a.m4a", "\\\\nas\\share\\a.m4a"])
    def test_absolute_paths_are_reported_never_emitted(self, value):
        path, issue = to_process_relative(value, FactsAnchor.REPO, "PROCESSING-B", "audio")
        assert path is None
        assert issue is not None and issue.reason == "absolute path stored in DB"

    def test_repo_path_outside_process_dir_is_reported(self):
        path, issue = to_process_relative("OTHER/a.m4a", FactsAnchor.REPO, "PROCESSING-B", "audio")
        assert path is None and "not under process_dir" in issue.reason

    def test_empty_value_is_none_without_issue(self):
        assert to_process_relative(None, FactsAnchor.REPO, "PROCESSING-B", "f") == (None, None)


class TestBuildJobManifest:
    def test_only_sequenced_chapters_in_sequence_order_with_process_relative_paths(self):
        built = build_job_manifest(_snapshot(), generator="test", generated_at=T0)
        data = built.data

        assert built.issues == ()
        assert [c["sequence"] for c in data["chapters"]] == ["0001", "0002"]
        assert data["chapters"][0]["audio_path"] == "audiobooks/Book_ch0001.m4a"
        assert data["chapters"][1]["audio_path"] is None
        assert data["paths"]["source"] == "book.epub"
        assert data["paths"]["transcript_dir"] == "transcript_text/tts_script"
        assert data["paths"]["publish_toc"] == "0_toc_publish.md"
        assert data["book"]["cover"] == "images/cover.jpeg"
        assert data["summary"] == {
            "chapter_count": 2,
            "unsequenced_chapter_count": 1,
            "chapters_with_audio": 1,
            "total_audio_seconds": 61.25,
        }
        _validate(data, "job.schema.json")

    def test_fingerprint_ignores_generated_at_but_tracks_content(self):
        a = build_job_manifest(_snapshot(), generator="test", generated_at=T0).data
        b = build_job_manifest(_snapshot(), generator="test", generated_at=T1).data
        assert a["generated_at"] != b["generated_at"]
        assert a["content_sha256"] == b["content_sha256"] == content_fingerprint(b)

        b["book"]["title"] = "Renamed"
        assert content_fingerprint(b) != a["content_sha256"]


class TestWriter:
    def test_second_identical_write_is_skipped_and_leaves_no_temp_files(self, tmp_path: Path):
        target = tmp_path / "job.json"
        first = write_manifest(target, build_job_manifest(_snapshot(), generator="t", generated_at=T0).data)
        mtime = target.stat().st_mtime_ns
        second = write_manifest(target, build_job_manifest(_snapshot(), generator="t", generated_at=T1).data)

        assert first.status is WriteStatus.WRITTEN and first.previous_sha256 is None
        assert second.status is WriteStatus.UNCHANGED
        assert target.stat().st_mtime_ns == mtime
        assert json.loads(target.read_text(encoding="utf-8"))["generated_at"] == "2026-09-25T08:00:00Z"
        assert [p.name for p in tmp_path.iterdir()] == ["job.json"]

    def test_corrupt_existing_file_is_overwritten(self, tmp_path: Path):
        target = tmp_path / "job.json"
        target.write_text("{not json", encoding="utf-8")
        outcome = write_manifest(target, build_job_manifest(_snapshot(), generator="t", generated_at=T0).data)
        assert outcome.status is WriteStatus.WRITTEN
        _validate(json.loads(target.read_text(encoding="utf-8")), "job.schema.json")


def _seed_repo(repo_dir: Path, *, make_process_dir: bool = True) -> tuple[Path, int]:
    from syntrive.db.models import Book, BookImage, Job, TranscriptChapter
    from syntrive.db.session import get_db_session

    db_path = repo_dir / "syntrivetts.db"
    if make_process_dir:
        (repo_dir / "PROCESSING-新书").mkdir(parents=True)
    with get_db_session(db_path) as db:
        book = Book(title="新书", author="作者", language="zh", cover="images/cover.jpeg")
        db.add(book)
        db.flush()
        db.add(BookImage(book_id=book.id, name="cover.jpeg", path="images/cover.jpeg", image_type="cover"))
        job = Job(
            book_id=book.id,
            process_dir="PROCESSING-新书",
            epub_path="PROCESSING-新书/新书.epub",
            cover_path="images/cover.jpeg",
            audiobooks_dir="PROCESSING-新书/audiobooks",
            sentence_audio_dir="PROCESSING-新书/sentence_audio",
            transcript_dir="PROCESSING-新书/transcript_text/tts_script",
            status="running",
            current_step="tts_config",
        )
        db.add(job)
        db.flush()
        db.add(TranscriptChapter(
            job_id=job.id, chapter_id="ch_0001", sequence_number="0001", chapter_name="第一章",
            transcript_path="transcript_text/tts_script/ch_0001.txt",
            chapter_audio_path="PROCESSING-新书/audiobooks/新书_ch0001.m4a",
            chapter_audio_seconds=12.5, synthesis_status="done",
        ))
        job_id = job.id
    return db_path, job_id


class TestExportService:
    def test_export_writes_schema_valid_repo_and_job_manifests_idempotently(self, tmp_path: Path):
        from syntrive.services.contract_export_service import export_repo_manifests

        db_path, job_id = _seed_repo(tmp_path)
        first = export_repo_manifests(db_path)
        second = export_repo_manifests(db_path)

        assert first.ok and second.ok
        repo_json = json.loads((tmp_path / "repo.json").read_text(encoding="utf-8"))
        job_json = json.loads((tmp_path / "PROCESSING-新书" / "job.json").read_text(encoding="utf-8"))
        _validate(repo_json, "repo.schema.json")
        _validate(job_json, "job.schema.json")

        assert repo_json["jobs"][0]["job_manifest"] == "PROCESSING-新书/job.json"
        assert job_json["job"]["id"] == job_id
        assert job_json["book"]["title"] == "新书"
        assert job_json["chapters"][0]["audio_path"] == "audiobooks/新书_ch0001.m4a"
        assert first.jobs[0].outcome.status is WriteStatus.WRITTEN
        assert second.jobs[0].outcome.status is WriteStatus.UNCHANGED
        assert second.repo_outcome.status is WriteStatus.UNCHANGED

    def test_missing_process_dir_fails_that_job_but_repo_index_is_still_written(self, tmp_path: Path):
        from syntrive.services.contract_export_service import export_repo_manifests

        db_path, job_id = _seed_repo(tmp_path, make_process_dir=False)
        result = export_repo_manifests(db_path)

        assert not result.ok
        assert result.jobs[0].job_id == job_id and "process_dir missing" in result.jobs[0].error
        assert not (tmp_path / "PROCESSING-新书").exists()
        assert (tmp_path / "repo.json").is_file()

    def test_unknown_job_id_is_reported(self, tmp_path: Path):
        from syntrive.services.contract_export_service import export_repo_manifests

        db_path, _ = _seed_repo(tmp_path)
        result = export_repo_manifests(db_path, job_ids=[999])
        assert not result.ok and result.jobs[0].error == "unknown job id"

    def test_best_effort_refresh_never_raises(self, tmp_path: Path, monkeypatch):
        from syntrive.services import contract_export_service as svc

        missing = svc.refresh_manifests_best_effort(tmp_path / "nope.db", reason="test")
        assert missing is not None and missing.error.startswith("database not found")

        def boom(*_a, **_k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(svc, "export_repo_manifests", boom)
        assert svc.refresh_manifests_best_effort(tmp_path / "nope.db", reason="test") is None


class TestWorkflowHook:
    def test_advance_job_step_refreshes_job_manifest(self, tmp_path: Path):
        from syntrive.db.models import Job
        from syntrive.db.session import get_db_session
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        db_path, job_id = _seed_repo(tmp_path)
        with get_db_session(db_path) as db:
            job = db.get(Job, job_id)
            db.expunge(job)

        WorkflowEngine(job, db_path).advance_job_step(WorkflowStep.TTS_CONFIG)

        job_json = json.loads((tmp_path / "PROCESSING-新书" / "job.json").read_text(encoding="utf-8"))
        assert job_json["job"]["current_step"] == WorkflowStep.SYNTHESIS.value
        assert (tmp_path / "repo.json").is_file()


class TestSchemaForwardCompatibility:
    def test_unknown_fields_at_every_level_still_validate(self):
        data = build_job_manifest(_snapshot(), generator="t", generated_at=T0).data
        data["future_top_level"] = {"x": 1}
        data["job"]["future_flag"] = True
        data["book"]["isbn"] = "978-0"
        data["chapters"][0]["future_marker"] = "v2"
        _validate(data, "job.schema.json")


class TestArchivedStatus:
    def test_archive_unarchive_round_trip_updates_db_manifests_and_audit(self, tmp_path: Path):
        from syntrive.db.models import Job, WorkflowStepEvent
        from syntrive.db.session import get_db_session
        from syntrive.services.job_service import JobService

        db_path, job_id = _seed_repo(tmp_path)
        service = JobService(db_path)

        first = service.archive_job(job_id, note="duplicate import")
        again = service.archive_job(job_id)

        repo_json = json.loads((tmp_path / "repo.json").read_text(encoding="utf-8"))
        job_json = json.loads((tmp_path / "PROCESSING-新书" / "job.json").read_text(encoding="utf-8"))
        assert first.ok and first.changed and first.archived_at is not None
        assert again.ok and not again.changed
        assert repo_json["jobs"][0]["archived"] is True
        assert repo_json["jobs"][0]["archived_at"] == job_json["job"]["archived_at"] is not None
        _validate(repo_json, "repo.schema.json")
        _validate(job_json, "job.schema.json")

        restored = service.unarchive_job(job_id)
        repo_json = json.loads((tmp_path / "repo.json").read_text(encoding="utf-8"))
        assert restored.ok and restored.changed
        assert repo_json["jobs"][0]["archived"] is False and repo_json["jobs"][0]["archived_at"] is None

        with get_db_session(db_path) as db:
            assert db.get(Job, job_id).archived_at is None
            actions = [e.action for e in db.query(WorkflowStepEvent).filter_by(job_id=job_id, step_name="archive").order_by(WorkflowStepEvent.id)]
        assert actions == ["archived", "unarchived"]

    def test_archive_unknown_job_is_reported(self, tmp_path: Path):
        from syntrive.services.job_service import JobService

        db_path, _ = _seed_repo(tmp_path)
        result = JobService(db_path).archive_job(999)
        assert not result.ok and result.error == "job not found"

    def test_m030_backfills_archived_at_on_a_legacy_db(self, tmp_path: Path):
        import sqlite3

        from syntrive.db.session import ensure_schema, make_engine

        db_path = tmp_path / "syntrivetts.db"
        ensure_schema(make_engine(db_path), seed=False)
        with sqlite3.connect(db_path) as conn:
            conn.execute("ALTER TABLE jobs DROP COLUMN archived_at")
        ensure_schema(make_engine(db_path), seed=False)
        ensure_schema(make_engine(db_path), seed=False)
        with sqlite3.connect(db_path) as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(jobs)")]
        assert cols.count("archived_at") == 1
