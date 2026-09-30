import shutil
from pathlib import Path

import pytest

EBOOK_DIR = Path("ebooks")
TMP_BASE = Path("tmp/test_bootstrap")


@pytest.fixture(autouse=True)
def isolated_tmp(tmp_path):
    yield tmp_path


def _ebook(filename: str) -> Path:
    path = (EBOOK_DIR / filename).resolve()
    if not path.is_file():
        pytest.skip(f"Test EPUB not found (place in ebooks/): {path}")
    return path


class TestBootstrap:
    def test_bootstrap_creates_process_dir_1984(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap, _REQUIRED_SUBDIRS

        ebook = _ebook("1984.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        process_dir = Path(job.process_dir)
        assert process_dir.exists(), "process_dir must be created"

        for sub in _REQUIRED_SUBDIRS:
            assert (process_dir / sub).exists(), f"Missing subdir: {sub}"

    def test_bootstrap_copies_ebook_to_process_dir(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap

        ebook = _ebook("1984.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        dest = Path(job.epub_path)
        assert dest.is_file(), "Ebook must be copied to process_dir"
        assert dest.parent == Path(job.process_dir), "epub_path must be inside process_dir"
        assert dest.stat().st_size > 0, "Copied ebook must not be empty"

    def test_bootstrap_creates_sqlite_db(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap

        ebook = _ebook("1984.epub")
        bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        db_path = tmp_path / "syntrivetts.db"
        assert db_path.is_file(), "syntrivetts.db must be created in repo_dir"
        assert db_path.stat().st_size > 0, "DB file must not be empty"

    def test_bootstrap_persists_book_metadata(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Book

        ebook = _ebook("1984.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        db_path = tmp_path / "syntrivetts.db"
        with get_db_session(db_path) as db:
            book = db.query(Book).filter(Book.id == job.book_id).first()
            assert book is not None, "Book record must exist"
            assert book.title, "Book title must not be empty"

    def test_bootstrap_persists_job_record(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job

        ebook = _ebook("1984.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        db_path = tmp_path / "syntrivetts.db"
        with get_db_session(db_path) as db:
            db_job = db.query(Job).filter(Job.id == job.id).first()
            assert db_job is not None
            assert db_job.process_dir == job.process_dir
            assert db_job.epub_path == job.epub_path
            assert db_job.stage == "init"
            assert db_job.status == "pending"

    def test_bootstrap_no_duplicate_book_on_rerun(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Book

        ebook = _ebook("1984.epub")
        job1 = bootstrap(repo_dir=tmp_path, ebook_path=ebook)
        job2 = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        assert job1.book_id == job2.book_id, "Both runs must share the same Book record"

        db_path = tmp_path / "syntrivetts.db"
        with get_db_session(db_path) as db:
            count = db.query(Book).count()
            assert count == 1, f"Expected 1 Book record, got {count}"

    def test_bootstrap_no_legacy_aliases_on_job(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap

        ebook = _ebook("1984.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        assert hasattr(job, "process_dir"), "job.process_dir must exist"
        assert hasattr(job, "epub_path"), "job.epub_path must exist"
        assert not hasattr(job, "session_dir"), "session_dir is a dropped legacy alias"
        assert not hasattr(job, "ebook"), "ebook is a dropped legacy alias"

    def test_bootstrap_process_dir_naming_convention(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap, _PROCESS_DIR_PREFIX

        ebook = _ebook("1984.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        process_dir_name = Path(job.process_dir).name
        assert process_dir_name.startswith(
            _PROCESS_DIR_PREFIX
        ), f"process_dir must start with '{_PROCESS_DIR_PREFIX}', got: {process_dir_name}"


class TestJobCoverPathSync:
    def test_ensure_book_images_synced_sets_job_cover_path(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap
        from syntrive.services.job_service import JobService
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book

        ebook = _ebook("1984-Orwell_George.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)
        db_path = tmp_path / "syntrivetts.db"

        JobService(db_path).ensure_book_images_synced(job.id)

        with get_db_session(db_path) as db:
            db_job = db.query(Job).filter(Job.id == job.id).first()
            book = db.query(Book).filter(Book.id == db_job.book_id).first()
            assert book.cover is not None, "books.cover must be detected"
            assert db_job.cover_path == book.cover, (
                f"jobs.cover_path {db_job.cover_path!r} must match "
                f"books.cover {book.cover!r}"
            )

    def test_update_book_cover_syncs_job_cover_path(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap
        from syntrive.services.job_service import JobService
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job, Book

        ebook = _ebook("1984-Orwell_George.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)
        db_path = tmp_path / "syntrivetts.db"
        svc = JobService(db_path)

        svc.update_book_cover(job.id, "images/manual_cover.jpg")

        with get_db_session(db_path) as db:
            db_job = db.query(Job).filter(Job.id == job.id).first()
            book = db.query(Book).filter(Book.id == db_job.book_id).first()
            assert book.cover == "images/manual_cover.jpg"
            assert db_job.cover_path == "images/manual_cover.jpg"

        svc.update_book_cover(job.id, None)
        with get_db_session(db_path) as db:
            db_job = db.query(Job).filter(Job.id == job.id).first()
            assert db_job.cover_path is None

    def test_migration_backfills_stale_cover_path(self, tmp_path: Path):
        from syntrive.bootstrap import bootstrap
        from syntrive.db.session import get_db_session, ensure_schema, make_engine
        from syntrive.db.models import Job, Book

        ebook = _ebook("1984-Orwell_George.epub")
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)
        db_path = tmp_path / "syntrivetts.db"

        with get_db_session(db_path) as db:
            db_job = db.query(Job).filter(Job.id == job.id).first()
            book = db.query(Book).filter(Book.id == db_job.book_id).first()
            book.cover = "images/legacy_cover.jpg"
            db_job.cover_path = None

        ensure_schema(make_engine(db_path), seed=False)

        with get_db_session(db_path) as db:
            db_job = db.query(Job).filter(Job.id == job.id).first()
            assert db_job.cover_path == "images/legacy_cover.jpg"
